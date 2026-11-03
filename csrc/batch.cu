#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <stdexcept>
#include <vector>

#include "batch.h"

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

// One thread per vertex: AoS verts / vels -> contiguous [N*V,3] float arrays.
__global__ void gatherK(const Vertex* verts, const glm::vec3* vels,
                        float* outPos, float* outVel, int total) {
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= total) return;
    const glm::vec3 p = verts[i].pos;
    outPos[3 * i] = p.x; outPos[3 * i + 1] = p.y; outPos[3 * i + 2] = p.z;
    const glm::vec3 v = vels[i];
    outVel[3 * i] = v.x; outVel[3 * i + 1] = v.y; outVel[3 * i + 2] = v.z;
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
    checkCudaErrors(cudaMalloc(&d_ids, sizeof(int) * N));
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
    cudaFree(d_ids);
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
        sim->meta.d_verts, sim->meta.d_vels, d_outPos, d_outVel, total);
}

}  // namespace ropesim
