#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <stdexcept>
#include <vector>

#include "batch.h"

namespace YarnBall {
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
    // One substep, identical to advance()'s loop body (collisions off).
    static void substep(Sim* s) {
        s->startIterate();
        cudaGraphLaunch(s->stepGraph, s->stream);
        s->stepCounter++;
    }
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
                     float max_h, int num_itr, bool fix_root_orientation) {
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
            sim->verts[e * V + v].pos = root + float(v) * seg_len * d;
        // rope boundary + kinematic root BEFORE configure() (hard requirement)
        sim->verts[e * V + V - 1].flags = 0;
        sim->verts[e * V].invMass = 0;
        if (fix_root_orientation)
            sim->verts[e * V].flags |= (uint32_t)VertexFlags::fixOrientation;
    }

    // de-collisioned config; radius/barrier are dead values here but gate
    // uploadMeta()'s seg-length check — scale them with the discretization
    sim->meta.detectionPeriod = -1;
    sim->meta.useStepSizeLimit = 0;
    sim->meta.radius = 0.01f * seg_len;
    sim->meta.barrierThickness = 0.08f * seg_len;

    sim->configure(density);
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
    YarnBall::SimAccess::prepare(sim.get(), h);
    const int blocks = (N + 127) / 128;
    for (int i = 0; i < substeps; i++) {
        applyRootProgramK<<<blocks, 128, 0, s>>>(
            sim->meta.d_verts, sim->meta.d_vels, sim->meta.d_qs,
            P0, V0, P1, d_q0 ? Q0 : nullptr, d_q1 ? Q1 : nullptr,
            i, substeps, h, dt, N, V);
        YarnBall::SimAccess::substep(sim.get());
    }
    YarnBall::SimAccess::finish(sim.get(), dt);

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
