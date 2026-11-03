#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <algorithm>
#include <stdexcept>
#include <vector>

#include "batch.h"

namespace YarnBall {
// Rope-rope contact: per-group brute-force broad+narrow
// phase. Same candidate test as collision.cu buildCollisionList (closest
// distance between segment centerlines < sum of the padded detection radii,
// |i-j| > 2, glue exemption) but restricted to segments of the same contact
// group (an env), written to the same d_numCols / d_collisions layout that
// cosseratItr and recomputeStepLimit consume. One thread per segment scans its
// group in ascending j: no atomics (deterministic list order), no host sync,
// fixed launch size (CUDA-graph safe).
__global__ void bruteDetectK(MetaData* data, int groupVerts, int ropeVerts, int* errorReturn) {
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    const int n = data->numVerts;
    if (i >= n) return;
    const uint32_t* flags = data->d_lastFlags;
    if (!(flags[i] & (uint32_t)VertexFlags::hasNext)) { data->d_numCols[i] = 0; return; }
    const int g0 = (i / groupVerts) * groupVerts;
    const int g1 = min(g0 + groupVerts, n);
    const vec3* P = data->d_lastPos;
    const float* pads = data->d_paddingSize;
    const vec3 a0 = P[i];
    const vec3 a1 = P[i + 1] - a0;
    const float ri = fmaxf(pads[i], pads[i + 1]);
    const int c0 = data->d_lastCID[i], c1 = data->d_lastCID[i + 1];
    int cnt = 0;
    for (int j = g0; j < g1; j++) {
        if (!(flags[j] & (uint32_t)VertexFlags::hasNext)) continue;
        if (j / ropeVerts == i / ropeVerts && abs(j - i) <= 2) continue;   // neighbours on the SAME rope only
        if (c0 == j || c1 == j || c0 == j + 1 || c1 == j + 1) continue;
        const vec3 b0 = P[j] - a0, b1 = P[j + 1] - a0;
        const vec2 uv = Kit::segmentClosestPoints(vec3(0), a1, b0, b1);
        const vec3 d = uv.x * a1 - mix(b0, b1, uv.y);
        const float r = ri + fmaxf(pads[j], pads[j + 1]);
        if (Kit::length2(d) < r * r) {
            if (cnt < MAX_COLLISIONS_PER_SEGMENT) data->d_collisions[i + n * cnt] = j;
            cnt++;
        }
    }
    if (cnt > MAX_COLLISIONS_PER_SEGMENT) errorReturn[0] = Sim::ERROR_MAX_COLLISIONS_PER_SEGMENT_EXCEEDED;
    data->d_numCols[i] = min(cnt, MAX_COLLISIONS_PER_SEGMENT);
}

// Befriended by Sim (our YarnBall-Linux patch): exposes exactly what
// Sim::advance() does internally, so RopeBatch can run the substep loop
// itself and inject a root-motion-program kernel between substeps.
class SimAccess {
public:
    // Everything advance() does before its substep loop.
    static void prepare(Sim* s, float h) {
        s->rebuildCUDAGraph();   // no-op unless numItr changed
        s->meta.lastH = s->meta.h;
        s->meta.h = h;
        s->uploadMeta();
    }
    // One substep, identical to advance()'s loop body (collisions off) when
    // ropeContact is false. With ropeContact, rope-rope candidates come from
    // the per-group brute-force detector (bruteDetectK) instead of YarnBall's
    // LBVH, which syncs the host every detection and knows nothing of envs.
    static void substep(Sim* s, bool ropeContact, int groupVerts, int ropeVerts, int sub, int nSub) {
        s->startIterate(sub, nSub);   // substep index: kinematic mesh poses are interpolated on it
        if (ropeContact)
            bruteDetectK<<<(s->meta.numVerts + 127) / 128, 128, 0, s->stream>>>(s->d_meta, groupVerts, ropeVerts, s->d_error);
        cudaGraphLaunch(s->stepGraph, s->stream);
        s->stepCounter++;
    }
    static void forceGraphRebuild(Sim* s) { s->lastItr = -1; }
    static void finish(Sim* s, float dt) {
        s->meta.time += dt;
        s->checkErrors();
    }
};
}  // namespace YarnBall

namespace ropesim {

using YarnBall::Vertex;
using YarnBall::VertexFlags;

namespace {

// One thread per env: write the kinematic root's pos/vel (+ frame).
__global__ void applyRootsK(Vertex* verts, glm::vec3* vels, Kit::Rotor* qs,
                            const float* pos, const float* vel,
                            const float* quat, int N, int V) {
    const int e = threadIdx.x + blockIdx.x * blockDim.x;
    if (e >= N) return;
    const int root = e * V;
    verts[root].pos = glm::vec3(pos[3 * e], pos[3 * e + 1], pos[3 * e + 2]);
    vels[root] = glm::vec3(vel[3 * e], vel[3 * e + 1], vel[3 * e + 2]);
    if (quat) {
        glm::vec4 q(quat[4 * e], quat[4 * e + 1], quat[4 * e + 2], quat[4 * e + 3]);
        float l2 = glm::dot(q, q);
        q = (l2 > 0.f) ? q * rsqrtf(l2) : glm::vec4(0, 0, 0, 1);
        qs[root] = Kit::Rotor(q);  // fixed frames are never renormalized by the solver
    }
}

__device__ glm::vec4 slerp4(glm::vec4 a, glm::vec4 b, float t) {
    float d = glm::dot(a, b);
    if (d < 0.f) { b = -b; d = -d; }                 // shortest arc
    if (d > 0.9995f)                                 // nearly parallel: nlerp
        return glm::normalize(glm::mix(a, b, t));
    const float th = acosf(fminf(d, 1.f));
    const float s = sinf(th);
    return (sinf((1.f - t) * th) / s) * a + (sinf(t * th) / s) * b;
}

// Second-order root motion program, one thread per env, launched once per
// substep s. Position: p(tau) = p0 + v0*tau + 0.5*a*tau^2 with
// a = 2*((p1-p0) - v0*dt)/dt^2 (lands exactly on p1). The position is
// written analytically each substep (no drift) and the velocity is sampled
// at the substep MIDPOINT so the solver's pos += h*vel reproduces the
// quadratic exactly. Orientation: SLERP q0->q1 evaluated at the substep end.
__global__ void applyRootProgramK(Vertex* verts, glm::vec3* vels,
                                  Kit::Rotor* qs, const float* p0,
                                  const float* v0, const float* p1,
                                  const float* q0, const float* q1,
                                  int sIdx, int nSub, float h, float dt,
                                  int N, int V) {
    const int e = threadIdx.x + blockIdx.x * blockDim.x;
    if (e >= N) return;
    const int root = e * V;

    const glm::vec3 P0(p0[3 * e], p0[3 * e + 1], p0[3 * e + 2]);
    const glm::vec3 V0(v0[3 * e], v0[3 * e + 1], v0[3 * e + 2]);
    const glm::vec3 P1(p1[3 * e], p1[3 * e + 1], p1[3 * e + 2]);
    const glm::vec3 A = 2.f / (dt * dt) * ((P1 - P0) - V0 * dt);

    const float tau = sIdx * h;
    verts[root].pos = P0 + V0 * tau + 0.5f * tau * tau * A;
    vels[root] = V0 + (tau + 0.5f * h) * A;

    if (q0) {
        // normalize INPUTS: slerp's dot/angle math assumes unit quaternions
        glm::vec4 qa(q0[4 * e], q0[4 * e + 1], q0[4 * e + 2], q0[4 * e + 3]);
        float la = glm::dot(qa, qa);
        qa = (la > 0.f) ? qa * rsqrtf(la) : glm::vec4(0, 0, 0, 1);
        glm::vec4 q;
        if (q1) {
            glm::vec4 qb(q1[4 * e], q1[4 * e + 1], q1[4 * e + 2], q1[4 * e + 3]);
            float lb = glm::dot(qb, qb);
            qb = (lb > 0.f) ? qb * rsqrtf(lb) : glm::vec4(0, 0, 0, 1);
            q = slerp4(qa, qb, float(sIdx + 1) / float(nSub));
        } else {
            q = qa;
        }
        float l2 = glm::dot(q, q);
        q = (l2 > 0.f) ? q * rsqrtf(l2) : glm::vec4(0, 0, 0, 1);
        qs[root] = Kit::Rotor(q);
    }
}

// One thread per vertex: AoS solver state -> contiguous float arrays
// (copies, so consumers can never mutate live solver state).
__global__ void gatherK(const Vertex* verts, const glm::vec3* vels,
                        const Kit::Rotor* qs, float* outPos, float* outVel,
                        float* outQuat, int total) {
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= total) return;
    const glm::vec3 p = verts[i].pos;
    outPos[3 * i] = p.x; outPos[3 * i + 1] = p.y; outPos[3 * i + 2] = p.z;
    const glm::vec3 v = vels[i];
    outVel[3 * i] = v.x; outVel[3 * i + 1] = v.y; outVel[3 * i + 2] = v.z;
    const glm::vec4 q = qs[i].v;
    outQuat[4 * i] = q.x; outQuat[4 * i + 1] = q.y;
    outQuat[4 * i + 2] = q.z; outQuat[4 * i + 3] = q.w;
}

// One thread per (selected env, vertex): restore construction-time state.
__global__ void resetEnvsK(Vertex* verts, Kit::Rotor* qs, glm::vec3* vels,
                           glm::vec3* lastVels, const glm::vec3* initPos,
                           const glm::vec4* initQ, const int* ids,
                           int count, int V) {
    const int t = threadIdx.x + blockIdx.x * blockDim.x;
    if (t >= count * V) return;
    const int g = ids[t / V] * V + (t % V);
    verts[g].pos = initPos[g];
    qs[g] = Kit::Rotor(initQ[g]);
    vels[g] = glm::vec3(0);
    lastVels[g] = glm::vec3(0);
}

}  // namespace

RopeBatch::RopeBatch(int num_envs, int verts_per_rope, float seg_len,
                     const float* root_pos_host, const float* dir3,
                     float density, float k_stretch, float k_bend,
                     float max_h, int num_itr, bool fix_root_orientation,
                     const float* init_pos_host, const uint8_t* pinned_host, bool straight_rest)
    : sdf(num_envs) {
    N = num_envs;
    V = verts_per_rope;
    fixRootOrientation_ = fix_root_orientation;
    const int total = N * V;

    sim = std::make_unique<YarnBall::Sim>(total);

    glm::vec3 d = glm::normalize(glm::vec3(dir3[0], dir3[1], dir3[2]));
    for (int e = 0; e < N; e++) {
        const glm::vec3 root(root_pos_host[3 * e], root_pos_host[3 * e + 1],
                             root_pos_host[3 * e + 2]);
        for (int v = 0; v < V; v++)
            sim->verts[e * V + v].pos = init_pos_host
                ? glm::vec3(init_pos_host[(e * V + v) * 3], init_pos_host[(e * V + v) * 3 + 1], init_pos_host[(e * V + v) * 3 + 2])
                : root + float(v) * seg_len * d;
        // rope boundary + kinematic root BEFORE configure() (hard requirement)
        sim->verts[e * V + V - 1].flags = 0;
        // pinned vertices are kinematic (invMass 0 before configure(), which
        // then gives every free vertex its lumped mass from the real segments)
        for (int v = 0; v < V; v++)
            if (pinned_host ? pinned_host[e * V + v] != 0 : v == 0)
                sim->verts[e * V + v].invMass = 0;
        if (pinned_host && pinned_host[e * V] == 0) rootsPinned = false;
        if (fix_root_orientation)
            sim->verts[e * V].flags |= (uint32_t)VertexFlags::fixOrientation;
    }

    // de-collisioned config; radius/barrier are dead values here but gate
    // uploadMeta()'s seg-length check — scale them with the discretization
    sim->meta.detectionPeriod = -1;
    sim->meta.useStepSizeLimit = 0;
    float minSeg = seg_len;
    if (init_pos_host) {   // dead collision radii scale with the shortest given segment
        minSeg = 1e30f;
        for (int e = 0; e < N; e++)
            for (int v = 0; v + 1 < V; v++)
                minSeg = std::min(minSeg, glm::length(sim->verts[e * V + v + 1].pos - sim->verts[e * V + v].pos));
    }
    sim->meta.radius = 0.01f * minSeg;
    sim->meta.barrierThickness = 0.08f * minSeg;

    sim->configure(density);
    // configure() makes the initial geometry the rest shape (rest frames = the relative
    // rotation between adjacent segments). With straight_rest the rest shape is a straight,
    // untwisted rope whatever the initial geometry: identity rest rotations, which
    // setKBend() then scales to the bending stiffness.
    if (straight_rest)
        for (int e = 0; e < N; e++)
            for (int v = 0; v + 1 < V; v++) sim->qRests[e * V + v] = glm::vec4(0.f, 0.f, 0.f, 1.f);
    sim->setKStretch(k_stretch);
    sim->setKBend(k_bend);
    sim->maxH = max_h;
    sim->meta.h = max_h;  // seed lastH like a fresh/reset sim (determinism)
    sim->meta.numItr = num_itr;
    sim->printErrors = false;
    sim->upload();

    std::vector<glm::vec3> ip(total);
    std::vector<glm::vec4> iq(total);
    for (int i = 0; i < total; i++) {
        ip[i] = sim->verts[i].pos;
        iq[i] = sim->qs[i].v;
    }
    checkCudaErrors(cudaMalloc(&d_initPos, sizeof(glm::vec3) * total));
    checkCudaErrors(cudaMalloc(&d_initQ, sizeof(glm::vec4) * total));
    checkCudaErrors(cudaMalloc(&d_outPos, sizeof(float) * 3 * total));
    checkCudaErrors(cudaMalloc(&d_outVel, sizeof(float) * 3 * total));
    checkCudaErrors(cudaMalloc(&d_outQuat, sizeof(float) * 4 * total));
    checkCudaErrors(cudaMalloc(&d_ids, sizeof(int) * N));
    checkCudaErrors(cudaMalloc(&d_prog, sizeof(float) * 17 * N));
    checkCudaErrors(cudaMemcpy(d_initPos, ip.data(), sizeof(glm::vec3) * total,
                               cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_initQ, iq.data(), sizeof(glm::vec4) * total,
                               cudaMemcpyHostToDevice));

    gather_nolock();  // observation buffers valid before the first step
    checkCudaErrors(cudaStreamSynchronize(sim->solverStream()));
}

RopeBatch::~RopeBatch() {
    cudaFree(d_initPos);
    cudaFree(d_initQ);
    cudaFree(d_outPos);
    cudaFree(d_outVel);
    cudaFree(d_outQuat);
    cudaFree(d_ids);
    cudaFree(d_prog);
}

void RopeBatch::step(const float* d_root_pos, const float* d_root_vel,
                     const float* d_root_q, float dt, int substeps) {
    std::lock_guard<std::mutex> g(mtx_);
    cudaStream_t s = sim->solverStream();
    applyRootsK<<<(N + 127) / 128, 128, 0, s>>>(
        sim->meta.d_verts, sim->meta.d_vels, sim->meta.d_qs,
        d_root_pos, d_root_vel, d_root_q, N, V);
    sdf.begin_step(sim->meta, s);

    if (ropeContact) {   // same substeps as advance(), plus rope-rope detection
        YarnBall::SimAccess::prepare(sim.get(), dt / substeps);
        for (int i = 0; i < substeps; i++)
            YarnBall::SimAccess::substep(sim.get(), true, groupVerts, V, i, substeps);
        YarnBall::SimAccess::finish(sim.get(), dt);
        sdf.end_step(substeps, sim->solverStream());
        gather_nolock();
        return;
    }
    // exactly `substeps` substeps of h = dt/substeps (max_h restored after)
    const float oldMaxH = sim->maxH;
    sim->maxH = (dt / substeps) * 1.000001f;
    try {
        sim->advance(dt);
    } catch (...) {
        sim->maxH = oldMaxH;
        throw;
    }
    sim->maxH = oldMaxH;
    sdf.end_step(substeps, sim->solverStream());

    gather_nolock();
}

void RopeBatch::step_program(const float* d_p0, const float* d_v0,
                             const float* d_p1, const float* d_q0,
                             const float* d_q1, float dt, int substeps) {
    std::lock_guard<std::mutex> g(mtx_);
    cudaStream_t s = sim->solverStream();

    // Stage caller inputs into our persistent buffer: the program kernel
    // launches `substeps` times and must read stable memory even if the
    // caller's tensors are freed right after this call returns.
    float* P0 = d_prog;
    float* V0 = P0 + 3 * N;
    float* P1 = V0 + 3 * N;
    float* Q0 = P1 + 3 * N;
    float* Q1 = Q0 + 4 * N;
    checkCudaErrors(cudaMemcpyAsync(P0, d_p0, sizeof(float) * 3 * N,
                                    cudaMemcpyDeviceToDevice, s));
    checkCudaErrors(cudaMemcpyAsync(V0, d_v0, sizeof(float) * 3 * N,
                                    cudaMemcpyDeviceToDevice, s));
    checkCudaErrors(cudaMemcpyAsync(P1, d_p1, sizeof(float) * 3 * N,
                                    cudaMemcpyDeviceToDevice, s));
    if (d_q0)
        checkCudaErrors(cudaMemcpyAsync(Q0, d_q0, sizeof(float) * 4 * N,
                                        cudaMemcpyDeviceToDevice, s));
    if (d_q1)
        checkCudaErrors(cudaMemcpyAsync(Q1, d_q1, sizeof(float) * 4 * N,
                                        cudaMemcpyDeviceToDevice, s));

    const float h = dt / substeps;  // exact; we run the loop ourselves
    sdf.begin_step(sim->meta, s);
    YarnBall::SimAccess::prepare(sim.get(), h);
    const int blocks = (N + 127) / 128;
    for (int i = 0; i < substeps; i++) {
        applyRootProgramK<<<blocks, 128, 0, s>>>(
            sim->meta.d_verts, sim->meta.d_vels, sim->meta.d_qs,
            P0, V0, P1, d_q0 ? Q0 : nullptr, d_q1 ? Q1 : nullptr,
            i, substeps, h, dt, N, V);
        YarnBall::SimAccess::substep(sim.get(), ropeContact, groupVerts, V, i, substeps);
    }
    YarnBall::SimAccess::finish(sim.get(), dt);
    sdf.end_step(substeps, sim->solverStream());

    gather_nolock();
}

// Rope-rope contact switch (see bruteDetectK / SimAccess::substep). YarnBall's
// own contact model runs on the candidates: barrier, friction, step limit.
void RopeBatch::set_rope_contact(bool enabled, float radius, float barrier, float kCollision,
                                 float mu, float kFriction, float detectionScaler, int groupRopes) {
    std::lock_guard<std::mutex> g(mtx_);
    auto& m = sim->meta;
    if (enabled) {
        // uploadMeta() enforces this at every step; check it here so a bad call
        // leaves the old state instead of a batch whose every step throws
        if (2 * m.minSegLen < 2 * radius + barrier)
            throw std::invalid_argument("rope contact needs 2 * radius + barrier <= 2 * shortest segment");
        groupVerts = V * (groupRopes > 0 ? groupRopes : 1);
        m.radius = radius; m.barrierThickness = barrier; m.kCollision = kCollision;
        m.frictionCoeff = mu; m.kFriction = kFriction; m.detectionScaler = detectionScaler;
    } else
        checkCudaErrors(cudaMemsetAsync(m.d_numCols, 0, sizeof(int) * m.numVerts, sim->solverStream()));
    ropeContact = enabled;
    m.useStepSizeLimit = enabled ? 1 : 0;
    m.detectionPeriod = -1;   // never YarnBall's LBVH detector
    YarnBall::SimAccess::forceGraphRebuild(sim.get());   // the LIMIT variant is baked into the graph
}

// Substeps with no root program (free ropes, or roots held where they are).
void RopeBatch::step_free(float dt, int substeps) {
    std::lock_guard<std::mutex> g(mtx_);
    const float h = dt / substeps;
    sdf.begin_step(sim->meta, sim->solverStream());
    YarnBall::SimAccess::prepare(sim.get(), h);
    for (int i = 0; i < substeps; i++)
        YarnBall::SimAccess::substep(sim.get(), ropeContact, groupVerts, V, i, substeps);
    YarnBall::SimAccess::finish(sim.get(), dt);
    sdf.end_step(substeps, sim->solverStream());
    gather_nolock();
}

void RopeBatch::reset_envs(const int* ids_host, int count) {
    if (count <= 0) return;
    std::lock_guard<std::mutex> g(mtx_);
    cudaStream_t s = sim->solverStream();
    // pageable HtoD: the driver stages the copy before returning, so the
    // caller's buffer may die immediately after this call
    checkCudaErrors(cudaMemcpyAsync(d_ids, ids_host, sizeof(int) * count,
                                    cudaMemcpyHostToDevice, s));
    const int total = count * V;
    resetEnvsK<<<(total + 255) / 256, 256, 0, s>>>(
        sim->meta.d_verts, sim->meta.d_qs, sim->meta.d_vels,
        sim->meta.d_lastVels, d_initPos, d_initQ, d_ids, count, V);
    gather_nolock();
}

void RopeBatch::gather() {
    std::lock_guard<std::mutex> g(mtx_);
    gather_nolock();
}

void RopeBatch::gather_nolock() {
    const int total = N * V;
    gatherK<<<(total + 255) / 256, 256, 0, sim->solverStream()>>>(
        sim->meta.d_verts, sim->meta.d_vels, sim->meta.d_qs,
        d_outPos, d_outVel, d_outQuat, total);
}

}  // namespace ropesim
