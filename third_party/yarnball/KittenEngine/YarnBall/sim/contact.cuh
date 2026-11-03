// Local patch (ropesim): rope vertex vs surface contact (half-space planes and
// posed signed-distance grids), shared by the per-vertex Newton step
// (cosserat.cu cosseratItr), the end-of-substep force readout and broad phase
// (iteration.cu endItr) and the pose kernels (csrc/sdf.cu). Header-only device
// code; include after YarnBall.h.
#pragma once

#include "../YarnBall.h"

namespace YarnBall {
	using Kit::hess3;

	// One surface contact at a rope vertex. f/H follow cosseratItr's convention:
	// f = -grad(incremental potential) at the current iterate, H its
	// (approximate) Hessian; dx = displacement of this substep so far
	// RELATIVE TO THE SURFACE (velocity = dx/h; for a kinematic mesh the
	// material point's own displacement has been subtracted by the caller).
	// Normal: penalty E = 1/2 k pen^2 with k = m (kappa/h)^2, so the contact
	// "frequency" is kappa/h for every vertex (end vertices included) and the
	// numerical bounce is the same at any substep size. No contact damping: a
	// dashpot that switches on at gap = 0 makes the local solve limit-cycle.
	// Friction: regularized Coulomb (PyElastica's slip blend): stick stiffness
	// muS*N/(eps*h) below the slip speed eps, muS -> muK between eps and 2 eps,
	// muK*N above; lagged like YarnBall's friction (stiffness not differentiated).
	// n = unit outward normal, gap = signed distance of the rope SURFACE to the
	// obstacle at the current iterate (< 0 in contact); muS/muK of that surface.
	__device__ inline void surfaceContact(const MetaData* data, const vec3 n, const float gap, const vec3 dx,
		const float k, const float h, const float muS, const float muK, vec3& f, hess3& H) {
		const float eps = data->contactVSlip;
		const float N = -k * gap;
		f += N * n;
		hess3 nn = hess3::outer(n);
		H += nn * k;
		const vec3 ut = dx - dot(n, dx) * n;
		const float ul = length(ut);
		const float vt = ul / h;
		float kt;
		if (vt <= eps) kt = muS * N / (eps * h);
		else {
			const float s = 1.f - fminf(1.f, vt / eps - 1.f);
			kt = (s * muS + (1.f - s) * muK) * N / ul;
		}
		f -= kt * ut;
		nn.s.diag -= vec3(1);	// n n^T - I
		H -= nn * kt;		// += kt (I - n n^T)
	}

	// Float32 trilinear sample of a signed-distance grid (phi and its gradient;
	// the gradient is the exact derivative of the interpolant, so a planar region
	// of the surface gives a constant normal). false = outside the lattice (no
	// contact there). The cell comes from (p - origin) / h, but the fraction
	// inside it from p - node (an exact subtraction when the lattice is
	// float32-aligned, and off by at most a constant ulp(|p - origin|) per cell
	// otherwise), so the gap keeps the resolution of the object-frame coordinate
	// itself (the rest sink is only ~70 ulps of a coordinate near 1 cm) instead
	// of ulp(index) * h.
	__device__ inline bool sdfSample(const MetaData::SdfGrid& g, const vec3 p, float& phi, vec3& grad) {
		const vec3 u = (p - g.origin) * g.invH;
		if (!(u.x >= 0.f && u.y >= 0.f && u.z >= 0.f &&
			u.x < float(g.nx - 1) && u.y < float(g.ny - 1) && u.z < float(g.nz - 1))) return false;
		const int i = (int)u.x, j = (int)u.y, k = (int)u.z;
		const vec3 fr = (p - (g.origin + vec3(float(i), float(j), float(k)) * g.h)) * g.invH;
		const float fx = fminf(fmaxf(fr.x, 0.f), 1.f), fy = fminf(fmaxf(fr.y, 0.f), 1.f), fz = fminf(fmaxf(fr.z, 0.f), 1.f);
		const int sy = g.nz, sx = g.ny * g.nz;
		const float* c = g.v + (size_t)i * sx + (size_t)j * sy + k;
		const float c000 = c[0], c001 = c[1], c010 = c[sy], c011 = c[sy + 1];
		const float c100 = c[sx], c101 = c[sx + 1], c110 = c[sx + sy], c111 = c[sx + sy + 1];
		const float dx00 = c100 - c000, dx01 = c101 - c001, dx10 = c110 - c010, dx11 = c111 - c011;
		const float c00 = c000 + fx * dx00, c01 = c001 + fx * dx01, c10 = c010 + fx * dx10, c11 = c011 + fx * dx11;
		const float dy0 = c10 - c00, dy1 = c11 - c01;
		const float c0 = c00 + fy * dy0, c1 = c01 + fy * dy1;
		phi = c0 + fz * (c1 - c0);
		const float gx0 = dx00 + fy * (dx10 - dx00), gx1 = dx01 + fy * (dx11 - dx01);
		grad = vec3(gx0 + fz * (gx1 - gx0), dy0 + fz * (dy1 - dy0), c1 - c0) * g.invH;
		return true;
	}

	// World AABB of the object-frame box [lo_o, hi_o] under the pose (pos, q).
	__device__ __host__ inline void sdfWorldBox(const vec3 lo_o, const vec3 hi_o, const vec3 pos, const Kit::Rotor q,
		vec3& lo, vec3& hi) {
		lo = vec3(INFINITY); hi = vec3(-INFINITY);
		for (int c = 0; c < 8; c++) {
			const vec3 corner(c & 1 ? hi_o.x : lo_o.x, c & 2 ? hi_o.y : lo_o.y, c & 4 ? hi_o.z : lo_o.z);
			const vec3 w = q * corner + pos;
			lo = min(lo, w); hi = max(hi, w);
		}
	}

	// Contact of one vertex with mesh m of env `env` at the probe point
	// pos + dx (kept separate for precision). On contact returns true and the
	// world normal n, the gap, and the vertex displacement relative to the
	// mesh's material point (the mesh's own motion this substep subtracted when
	// it is kinematic; = dx otherwise, bitwise).
	__device__ inline bool sdfContactAt(const MetaData* data, const MetaData::SdfPose& P, const int m,
		const vec3 pos, const vec3 dx, const float h, vec3& n, float& gap, vec3& dxRel) {
		const Kit::Rotor q(P.q);
		float phi;
		vec3 g;
		// (pos - P.pos) + dx, never pos + dx: the sum would round dx away at
		// ulp(|pos|) (a staircase force far from the world origin, found in review)
		if (!sdfSample(data->sdf[m], q.inverse() * ((pos - P.pos) + dx), phi, g)) return false;
		gap = phi - data->contactRadius;
		if (gap >= 0.f) return false;
		const float gl = length(g);
		if (!(gl > 0.f)) return false;	// medial axis: no direction to push along
		n = q * (g / gl);
		dxRel = dx;
		if (data->sdfMoving) dxRel -= h * (P.vel + cross(P.omega, pos - P.pos));
		return true;
	}

	// Broad phase of vertex tid: n = its stored count (-1 = every mesh of the
	// env); sdfCandCount/sdfCandidate turn that into a loop.
	__device__ inline int sdfCandCount(const MetaData* data, const int n) { return n < 0 ? data->numSdf : n; }
	__device__ inline int sdfCandidate(const MetaData* data, const int tid, const int n, const int c) {
		return n < 0 ? c : data->d_sdfCand[(size_t)tid * MetaData::kMaxSdfCand + c];
	}
	// Rebuild vertex tid's candidates for the next substep: meshes of env `env`
	// whose world AABB, padded by `pad` (the vertex's possible travel) plus the
	// mesh's own travel in one substep (h (|v| + |omega| half-diagonal)), holds p.
	__device__ inline void sdfBuildCandidates(const MetaData* data, const int tid, const int env, const vec3 p, const float pad,
		const float h) {
		const MetaData::SdfPose* poses = data->d_sdfPoses + (size_t)env * MetaData::kMaxSdf;
		int* cand = data->d_sdfCand + (size_t)tid * MetaData::kMaxSdfCand;
		int n = 0;
		for (int m = 0; m < data->numSdf; m++) {
			const MetaData::SdfPose& P = poses[m];
			const float pm = pad + (data->sdfMoving ? h * (length(P.vel) + length(P.omega) * 0.5f * length(P.hi - P.lo)) : 0.f);
			if (p.x >= P.lo.x - pm && p.x <= P.hi.x + pm && p.y >= P.lo.y - pm && p.y <= P.hi.y + pm &&
				p.z >= P.lo.z - pm && p.z <= P.hi.z + pm) {
				if (n < MetaData::kMaxSdfCand) cand[n] = m;
				n++;
			}
		}
		data->d_sdfNumCand[tid] = n <= MetaData::kMaxSdfCand ? n : -1;
	}

	// Every surface (planes, then the candidate SDF grids of env `env`) at one
	// probe point: pos + dx, dx = its displacement this substep, k = its penalty
	// stiffness. Accumulates the point's force/Hessian (the Newton step).
	__device__ inline void surfacesAt(const MetaData* data, const int env, const int tid, const vec3 pos, const vec3 dx,
		const float k, const float h, vec3& f, hess3& H) {
		const float R = data->contactRadius;
		for (int p = 0; p < data->numPlanes; p++) {
			const vec4 pl = data->planes[p];
			const vec3 n(pl.x, pl.y, pl.z);
			const float gap = (dot(n, pos) - pl.w) + dot(n, dx) - R;	// rope surface to plane
			if (gap >= 0) continue;
			surfaceContact(data, n, gap, dx, k, h, data->contactMuS, data->contactMuK, f, H);
		}
		if (data->numSdf > 0) {
			const MetaData::SdfPose* poses = data->d_sdfPoses + (size_t)env * MetaData::kMaxSdf;
			const int nc = data->d_sdfNumCand[tid], cnt = sdfCandCount(data, nc);
			for (int c = 0; c < cnt; c++) {
				const int m = sdfCandidate(data, tid, nc, c);
				const MetaData::SdfPose P = poses[m];
				vec3 n, dxRel;
				float gap;
				if (!sdfContactAt(data, P, m, pos, dx, h, n, gap, dxRel)) continue;
				surfaceContact(data, n, gap, dxRel, k, h, data->sdf[m].muS, data->sdf[m].muK, f, H);
			}
		}
	}

	// Force readout at the END of a substep (dx final): the force every surface
	// applied to vertex `tid` is added to d_contactForce[tid]; the force and
	// torque (about the mesh origin, world frame) the vertex applied to each
	// mesh go to d_sdfWrench[(env * kMaxSdf + m) * 2 + {0, 1}] (atomics: many
	// vertices per mesh, so the sums are not bitwise reproducible run to run).
	// Same arithmetic as surfacesAt at the final iterate; the lever arm is the
	// contact point x - R n (friction acts on the rope's surface).
	__device__ inline void surfaceReadout(const MetaData* data, const int env, const int tid, const vec3 pos,
		const vec3 dx, const float k, const float h) {
		const float R = data->contactRadius;
		vec3 fv(0);
		hess3 H(0);	// surfaceContact's Hessian output, unused here
		for (int p = 0; p < data->numPlanes; p++) {
			const vec4 pl = data->planes[p];
			const vec3 n(pl.x, pl.y, pl.z);
			const float gap = (dot(n, pos) - pl.w) + dot(n, dx) - R;
			if (gap >= 0) continue;
			surfaceContact(data, n, gap, dx, k, h, data->contactMuS, data->contactMuK, fv, H);
		}
		if (data->numSdf > 0) {
			const MetaData::SdfPose* poses = data->d_sdfPoses + (size_t)env * MetaData::kMaxSdf;
			const int nc = data->d_sdfNumCand[tid], cnt = sdfCandCount(data, nc);
			for (int c = 0; c < cnt; c++) {
				const int m = sdfCandidate(data, tid, nc, c);
				const MetaData::SdfPose P = poses[m];
				vec3 n, dxRel;
				float gap;
				if (!sdfContactAt(data, P, m, pos, dx, h, n, gap, dxRel)) continue;
				vec3 fm(0);
				surfaceContact(data, n, gap, dxRel, k, h, data->sdf[m].muS, data->sdf[m].muK, fm, H);
				fv += fm;
				const vec3 r = (pos - P.pos) + dx - R * n;	// lever arm about the mesh origin, at the contact point
				const vec3 tq = cross(r, -fm);
				float* w = (float*)(data->d_sdfWrench + ((size_t)env * MetaData::kMaxSdf + m) * 2);
				atomicAdd(w + 0, -fm.x); atomicAdd(w + 1, -fm.y); atomicAdd(w + 2, -fm.z);
				atomicAdd(w + 3, tq.x); atomicAdd(w + 4, tq.y); atomicAdd(w + 5, tq.z);
			}
		}
		data->d_contactForce[tid] += fv;
	}
}
