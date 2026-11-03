#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "../YarnBall.h"
#include "contact.cuh"

namespace YarnBall {
	// Converts velocity to initial guess. Local patch (ropesim): substep sub of
	// nSub of the current frame; while sdfMoving, threads < sdfEnvs * kMaxSdf
	// also write the kinematic mesh pose of this substep (see MetaData::SdfMotion).
	__global__ void initItr(MetaData* data, int sub, int nSub) {
		const int tid = threadIdx.x + blockIdx.x * blockDim.x;
		if (data->sdfMoving && tid < data->sdfEnvs * MetaData::kMaxSdf) {
			if (tid % MetaData::kMaxSdf < data->numSdf) {
				const MetaData::SdfMotion M = data->d_sdfMotion[tid];
				const int m = tid % MetaData::kMaxSdf;
				const float a = float(sub + 1) / float(nSub);
				const float invDt = 1.f / (float(nSub) * data->h);
				MetaData::SdfPose P;
				const float th = length(M.rot);
				if (th > 0.f || M.dp != vec3(0.f)) {
					P.pos = M.p0 + a * M.dp;
					const Kit::Rotor dq = th > 0.f ? Kit::Rotor::angleAxis(a * th, M.rot / th) : Kit::Rotor();
					P.q = normalize((dq * Kit::Rotor(M.q0)).v);
					P.vel = M.dp * invDt;
					P.omega = M.rot * invDt;
				} else {	// a target equal to the current pose: exactly the static pose (bitwise inert)
					P.pos = M.p0; P.q = M.q0; P.vel = vec3(0.f); P.omega = vec3(0.f);
				}
				const MetaData::SdfGrid& g = data->sdf[m];
				sdfWorldBox(g.origin, g.origin + vec3(float(g.nx - 1), float(g.ny - 1), float(g.nz - 1)) * g.h, P.pos, Kit::Rotor(P.q), P.lo, P.hi);
				data->d_sdfPoses[tid] = P;
			}
		}
		if (tid >= data->numVerts) return;

		const float h = data->h;
		auto verts = data->d_verts;
		auto lastVels = data->d_lastVels;

		const vec3 g = data->gravity;
		const vec3 vel = data->d_vels[tid];

		vec3 dx = h * vel;
		vec3 lastVel = lastVels[tid];
		lastVels[tid] = vel;
		float padding = data->scaledDetectionRadius;
		auto y = dx + (h * h) * g;
		padding += length(y);

		if (verts[tid].invMass != 0) {
			// Compute y (inertial + accel position)
			// Store it in vel (The actual vel is no longer needed)
			data->d_vels[tid] = y;

			// Compute initial guess
			float g2 = length2(g);
			if (g2 > 0) {
				vec3 a = (vel - lastVel) / data->lastH;
				float s = clamp(dot(a, g) / g2, 0.f, 1.f);
				dx += (h * h * s) * g;
			}
		}
		data->d_paddingSize[tid] = padding;
		data->d_dx[tid] = dx;

		// Transfer segment data
		vec3 pos = verts[tid].pos;
		data->d_lastPos[tid] = pos;
	}

	void Sim::startIterate(int sub, int nSub) {
		const int threads = max(meta.numVerts, meta.sdfMoving ? meta.sdfEnvs * (int)MetaData::kMaxSdf : 0);
		initItr << <(threads + 255) / 256, 256, 0, stream >> > (d_meta, sub, nSub);
	}

	// Converts dx back to velocity and advects
	__global__ void endItr(MetaData* data) {
		const int tid = threadIdx.x + blockIdx.x * blockDim.x;
		if (tid >= data->numVerts) return;

		const float h = data->h;
		const float invH = 1 / h;
		auto verts = data->d_verts;

		// Linear velocity
		vec3 dx = data->d_dx[tid];
		if (verts[tid].invMass != 0) {
			vec3 v = dx * invH;
			const float kn = data->dragQuad;
			if (kn > 0) {
				// Local patch (ropesim): quadratic cross-flow air drag
				// (Morison / fly-line form). Split v by the local segment
				// tangent; per component apply the exact frozen-direction
				// substep solution of dv/dt = -k|v|v, i.e. v /= (1 + k|v|h):
				// unconditionally stable, substep-invariant.
				// kn = .5*rho_air*Cd_n*d/lambda [1/m]. Vertex mass and
				// half-segment drag length lump with identical weights
				// (mass = .5*lambda*(l_prev+l_next) in configure()), so a
				// uniform kn is exact at interior, root, and tip vertices.
				// Tip vertex (hasPrev only) uses its one real segment's
				// rotor; pinned verts (invMass==0) never reach here, so the
				// commanded root velocity is never drag-filtered.
				const vec3 t = (verts[tid].flags & (uint32_t)VertexFlags::hasNext) ?
					data->d_qs[tid] * vec3(1, 0, 0) :
					data->d_qs[tid - 1] * vec3(1, 0, 0);
				vec3 vt = t * dot(t, v);
				vec3 vn = v - vt;
				vn /= 1 + kn * length(vn) * h;
				vt /= 1 + data->dragQuadT * length(vt) * h;
				v = vn + vt;
			}
			data->d_vels[tid] = v * (1 - data->drag * h);
			// Local patch (ropesim): contact-force readout at the final configuration
			if (data->sdfForces && (data->numPlanes > 0 || data->numSdf > 0)) {
				const float kap = data->contactKappa / h;
				surfaceReadout(data, tid / data->sdfEnvVerts, tid, verts[tid].pos, dx, kap * kap / verts[tid].invMass, h);
			}
		}
		verts[tid].pos += dx;
		// Local patch (ropesim): broad phase for the next substep (meshes whose
		// padded world AABB holds the new position; pad = twice the travel of a
		// substep at this velocity + 2 cm; pinned vertices never touch)
		if (data->numSdf > 0) {
			if (verts[tid].invMass != 0)
				sdfBuildCandidates(data, tid, tid / data->sdfEnvVerts, verts[tid].pos, 2.f * length(dx) + 0.02f, h);
			else
				data->d_sdfNumCand[tid] = 0;
		}
	}

	void Sim::endIterate() {
		endItr << <(meta.numVerts + 255) / 256, 256, 0, stream >> > (d_meta);
	}
}