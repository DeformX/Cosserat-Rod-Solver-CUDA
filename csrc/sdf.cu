// SdfStore: device-side pose state machine for rod-mesh contact (see sdf.h).
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <stdexcept>
#include <string>

#include "sdf.h"
#include "YarnBall/sim/contact.cuh"

namespace {

using Pose7 = SdfStore::Pose7;
using Box = SdfStore::Box;
using Pose = SdfStore::Pose;
using Motion = SdfStore::Motion;
constexpr int MAX = SdfStore::MAX;

void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

// Rotation vector (world frame, shortest arc) taking unit q0 to unit q1:
// q1 = exp(rot) * q0, i.e. rot = log(q1 * conj(q0)).
__device__ glm::vec3 rotvec(const glm::vec4& q0, const glm::vec4& q1) {
    const glm::vec3 a(q1.x, q1.y, q1.z), b(-q0.x, -q0.y, -q0.z);
    glm::vec3 v = q1.w * b + q0.w * a + glm::cross(a, b);
    float w = q1.w * q0.w - glm::dot(a, b);
    if (w < 0.f) { v = -v; w = -w; }
    const float s = glm::length(v);
    if (!(s > 1e-9f)) return glm::vec3(0.f);
    return v * (2.f * atan2f(s, w) / s);
}

// Targets for mesh `index` (rows = 1 or N: one thread per env) or, with index < 0,
// for every mesh from a dense [N, M] block (one thread per (env, mesh)).
__global__ void scatterTargetsK(Pose7* tgt, uint8_t* tele, int index, const float* pos, const float* q, int rows, int M,
                                const uint8_t* teleFlags, int teleAll, int N) {
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    const int per = index < 0 ? M : 1;
    if (i >= N * per) return;
    const int e = i / per, m = index < 0 ? i % per : index;
    const int r = index < 0 ? i : (rows == 1 ? 0 : e);
    glm::vec4 qq(q[4 * r], q[4 * r + 1], q[4 * r + 2], q[4 * r + 3]);
    const float l = glm::length(qq);
    qq = l > 0.f ? qq / l : glm::vec4(0, 0, 0, 1);
    const size_t slot = (size_t)e * MAX + m;
    tgt[slot].pos = glm::vec3(pos[3 * r], pos[3 * r + 1], pos[3 * r + 2]);
    tgt[slot].q = qq;
    tele[slot] = teleAll ? 1 : (teleFlags ? teleFlags[e] : 0);
}

// One thread per (env, mesh): this frame's motion from the current pose to the
// target; a teleported slot becomes the target at once (no motion).
__global__ void stageMotionK(Pose7* cur, const Pose7* tgt, uint8_t* tele, Motion* motion, int total) {
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= total) return;
    if (tele[i]) { cur[i] = tgt[i]; tele[i] = 0; }
    motion[i].p0 = cur[i].pos;
    motion[i].q0 = cur[i].q;
    motion[i].dp = tgt[i].pos - cur[i].pos;
    motion[i].rot = rotvec(cur[i].q, tgt[i].q);
}

// The target of slot i is now its current pose; the static pose (velocity 0)
// and its world AABB go to what the kernels read.
__device__ void commitSlot(Pose7* cur, const Pose7* tgt, uint8_t* tele, Pose* poses, const Box* box, size_t i, int m) {
    cur[i] = tgt[i];
    tele[i] = 0;
    Pose P;
    P.pos = cur[i].pos; P.q = cur[i].q; P.vel = glm::vec3(0.f); P.omega = glm::vec3(0.f);
    YarnBall::sdfWorldBox(box[m].lo, box[m].hi, P.pos, Kit::Rotor(P.q), P.lo, P.hi);
    poses[i] = P;
}
// One thread per (env, mesh): after a moving step.
__global__ void commitK(Pose7* cur, const Pose7* tgt, uint8_t* tele, Pose* poses, const Box* box, int numSdf, int total) {
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= total) return;
    const int m = i % MAX;
    if (m < numSdf) commitSlot(cur, tgt, tele, poses, box, i, m);
}
// One thread per env: a newly added mesh only (other meshes' pending targets stay pending).
__global__ void commitMeshK(Pose7* cur, const Pose7* tgt, uint8_t* tele, Pose* poses, const Box* box, int index, int N) {
    const int e = threadIdx.x + blockIdx.x * blockDim.x;
    if (e >= N) return;
    commitSlot(cur, tgt, tele, poses, box, (size_t)e * MAX + index, index);
}

__global__ void scaleK(float* a, int n, float s) {
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) a[i] *= s;
}

inline int blocks(int n) { return (n + 255) / 256; }

}  // namespace

SdfStore::SdfStore(int num_envs) : N(num_envs) {
    const size_t total = (size_t)N * MAX;
    check(cudaMalloc(&d_cur, sizeof(Pose7) * total), "cudaMalloc(sdf cur)");
    check(cudaMalloc(&d_tgt, sizeof(Pose7) * total), "cudaMalloc(sdf tgt)");
    check(cudaMalloc(&d_tele, total), "cudaMalloc(sdf tele)");
    check(cudaMalloc(&d_motion, sizeof(Motion) * total), "cudaMalloc(sdf motion)");
    check(cudaMalloc(&d_poses, sizeof(Pose) * total), "cudaMalloc(sdf poses)");
    check(cudaMalloc(&d_box, sizeof(Box) * MAX), "cudaMalloc(sdf boxes)");
    check(cudaMalloc(&d_stage, sizeof(float) * 7 * (size_t)N * MAX), "cudaMalloc(sdf stage)");
    check(cudaMalloc(&d_stageTele, N), "cudaMalloc(sdf stage flags)");
    std::vector<Pose7> ident(total, Pose7{glm::vec3(0.f), glm::vec4(0, 0, 0, 1)});
    check(cudaMemcpy(d_cur, ident.data(), sizeof(Pose7) * total, cudaMemcpyHostToDevice), "upload(sdf cur)");
    check(cudaMemcpy(d_tgt, ident.data(), sizeof(Pose7) * total, cudaMemcpyHostToDevice), "upload(sdf tgt)");
    check(cudaMemset(d_tele, 0, total), "memset(sdf tele)");
    check(cudaMemset(d_poses, 0, sizeof(Pose) * total), "memset(sdf poses)");
    check(cudaMemset(d_box, 0, sizeof(Box) * MAX), "memset(sdf boxes)");
}

SdfStore::~SdfStore() {
    clear();
    for (void* p : {(void*)d_cur, (void*)d_tgt, (void*)d_tele, (void*)d_motion, (void*)d_poses, (void*)d_box,
                    (void*)d_stage, (void*)d_stageTele, (void*)d_cand, (void*)d_numCand, (void*)d_force, (void*)d_wrench})
        cudaFree(p);
}

void SdfStore::alloc_verts(int nv) {
    if (nv == numVerts) return;
    cudaFree(d_cand); cudaFree(d_numCand); cudaFree(d_force); cudaFree(d_wrench);
    d_cand = nullptr; d_numCand = nullptr; d_force = nullptr; d_wrench = nullptr;
    numVerts = nv;
    check(cudaMalloc(&d_cand, sizeof(int) * (size_t)nv * CAND), "cudaMalloc(sdf candidates)");
    check(cudaMalloc(&d_numCand, sizeof(int) * nv), "cudaMalloc(sdf candidate counts)");
    check(cudaMemset(d_numCand, 0xFF, sizeof(int) * nv), "memset(sdf candidate counts)");   // -1: every mesh
    check(cudaMalloc(&d_force, sizeof(glm::vec3) * nv), "cudaMalloc(contact forces)");
    check(cudaMalloc(&d_wrench, sizeof(glm::vec3) * 2 * (size_t)N * MAX), "cudaMalloc(mesh wrenches)");
    check(cudaMemset(d_force, 0, sizeof(glm::vec3) * nv), "memset(contact forces)");
    check(cudaMemset(d_wrench, 0, sizeof(glm::vec3) * 2 * (size_t)N * MAX), "memset(mesh wrenches)");
}

int SdfStore::add(const float* values, int nx, int ny, int nz, glm::vec3 origin, float spacing, float muS, float muK,
                  const float* pos, const float* q, int rows, cudaStream_t s) {
    if (count() >= MAX) throw std::invalid_argument("at most " + std::to_string(MAX) + " meshes per sim");
    const size_t n = (size_t)nx * ny * nz;
    float* d = nullptr;
    check(cudaMalloc(&d, sizeof(float) * n), "cudaMalloc(sdf grid)");
    const cudaError_t e = cudaMemcpy(d, values, sizeof(float) * n, cudaMemcpyHostToDevice);
    if (e != cudaSuccess) {
        cudaFree(d);
        check(e, "upload(sdf grid)");
    }
    grids.push_back(Grid{d, nx, ny, nz, origin, spacing, 1.f / spacing, muS, muK});
    boxes.push_back(Box{origin, origin + glm::vec3(float(nx - 1), float(ny - 1), float(nz - 1)) * spacing});
    const int idx = count() - 1;
    check(cudaMemcpyAsync(d_box, boxes.data(), sizeof(Box) * boxes.size(), cudaMemcpyHostToDevice, s), "upload(sdf boxes)");
    const bool pending = pendingSet;                           // other meshes' targets stay pending (review 3)
    set_targets_host(idx, pos, q, rows, nullptr, true, s);     // the new mesh starts at its initial pose
    commitMeshK<<<blocks(N), 256, 0, s>>>(d_cur, d_tgt, d_tele, d_poses, d_box, idx, N);
    check(cudaStreamSynchronize(s), "add_mesh");
    pendingSet = pending;
    return idx;
}

void SdfStore::set_friction(int index, float muS, float muK) {
    if (index < 0 || index >= count()) throw std::invalid_argument("mesh index out of range");
    grids[index].muS = muS; grids[index].muK = muK;
}

void SdfStore::set_targets(int index, const float* d_pos, const float* d_q, int rows, const uint8_t* tele, bool teleAll,
                           cudaStream_t s) {
    if (index >= count() || (index < 0 && count() == 0)) throw std::invalid_argument("mesh index out of range");
    const int threads = index < 0 ? N * count() : N;
    scatterTargetsK<<<blocks(threads), 256, 0, s>>>(d_tgt, d_tele, index, d_pos, d_q, rows, count(), tele, teleAll ? 1 : 0, N);
    check(cudaGetLastError(), "set_mesh_poses");
    pendingSet = true;
}

const uint8_t* SdfStore::stage_flags(const uint8_t* tele, bool teleAll, cudaStream_t s) {
    if (!tele || teleAll) return nullptr;
    check(cudaMemcpyAsync(d_stageTele, tele, N, cudaMemcpyHostToDevice, s), "upload(sdf teleport flags)");
    return d_stageTele;
}

void SdfStore::set_targets_host(int index, const float* pos, const float* q, int rows, const uint8_t* tele, bool teleAll,
                                cudaStream_t s) {
    if (index >= count() || (index < 0 && count() == 0)) throw std::invalid_argument("mesh index out of range");
    const int n = index < 0 ? N * count() : rows;
    float* dq = d_stage + 3 * (size_t)N * MAX;
    // pageable sources: the driver stages them before returning, so the caller's arrays may die right after
    check(cudaMemcpyAsync(d_stage, pos, sizeof(float) * 3 * n, cudaMemcpyHostToDevice, s), "upload(sdf target pos)");
    check(cudaMemcpyAsync(dq, q, sizeof(float) * 4 * n, cudaMemcpyHostToDevice, s), "upload(sdf target quat)");
    set_targets(index, d_stage, dq, rows, stage_flags(tele, teleAll, s), teleAll, s);
}

void SdfStore::clear() {
    for (Grid& g : grids) cudaFree(const_cast<float*>(g.v));
    grids.clear();
    boxes.clear();
    pendingSet = false;
    moving = false;
}

void SdfStore::enable_forces(bool on) { forcesOn = on; }

void SdfStore::write_meta(YarnBall::MetaData& m, int env_verts) {
    alloc_verts(N * (env_verts > 0 ? env_verts : 1));
    m.numSdf = count();
    for (int i = 0; i < MAX; i++)
        m.sdf[i] = i < count() ? grids[i] : Grid{nullptr, 0, 0, 0, glm::vec3(0.f), 0.f, 0.f, 0.f, 0.f};
    m.d_sdfPoses = d_poses;
    m.d_sdfMotion = d_motion;
    m.sdfEnvs = N;
    m.sdfEnvVerts = env_verts > 0 ? env_verts : 1;
    m.d_sdfCand = d_cand;
    m.d_sdfNumCand = d_numCand;
    m.sdfForces = forcesOn ? 1 : 0;
    m.d_contactForce = d_force;
    m.d_sdfWrench = d_wrench;
}

void SdfStore::begin_step(YarnBall::MetaData& m, cudaStream_t s) {
    if (numVerts == 0) return;   // write_meta never called: no meshes, nothing published
    if (pendingSet) {
        stageMotionK<<<blocks(N * MAX), 256, 0, s>>>(d_cur, d_tgt, d_tele, d_motion, N * MAX);
        m.sdfMoving = 1;
        moving = true;
    } else {
        m.sdfMoving = 0;
    }
    // the first substep of every frame visits every mesh (poses may have jumped)
    if (count() > 0) check(cudaMemsetAsync(d_numCand, 0xFF, sizeof(int) * numVerts, s), "reset(sdf candidates)");
    if (forcesOn) {
        check(cudaMemsetAsync(d_force, 0, sizeof(glm::vec3) * numVerts, s), "zero(contact forces)");
        check(cudaMemsetAsync(d_wrench, 0, sizeof(glm::vec3) * 2 * (size_t)N * MAX, s), "zero(mesh wrenches)");
    }
}

void SdfStore::commit(cudaStream_t s) {
    commitK<<<blocks(N * MAX), 256, 0, s>>>(d_cur, d_tgt, d_tele, d_poses, d_box, count(), N * MAX);
    check(cudaGetLastError(), "commit(sdf poses)");
}

void SdfStore::end_step(int substeps, cudaStream_t s) {
    lastSubsteps = substeps > 0 ? substeps : 1;
    if (numVerts == 0) return;
    if (moving) {
        commit(s);
        moving = false;
        pendingSet = false;
    }
    if (forcesOn) {
        const float k = 1.f / (float)lastSubsteps;
        scaleK<<<blocks(3 * numVerts), 256, 0, s>>>((float*)d_force, 3 * numVerts, k);
        scaleK<<<blocks(6 * N * MAX), 256, 0, s>>>((float*)d_wrench, 6 * N * MAX, k);
        check(cudaGetLastError(), "average(contact forces)");
    }
}

void SdfStore::contact_forces(float* out) const {
    if (!forcesOn || !d_force) throw std::invalid_argument("enable_contact_forces(True) first");
    check(cudaMemcpy(out, d_force, sizeof(glm::vec3) * numVerts, cudaMemcpyDeviceToHost), "download(contact forces)");
}

void SdfStore::wrenches(float* out) const {
    if (!forcesOn || !d_wrench) throw std::invalid_argument("enable_contact_forces(True) first");
    std::vector<glm::vec3> w(2 * (size_t)N * MAX);
    check(cudaMemcpy(w.data(), d_wrench, sizeof(glm::vec3) * w.size(), cudaMemcpyDeviceToHost), "download(mesh wrenches)");
    for (int env = 0; env < N; env++)
        for (int m = 0; m < count(); m++) {
            const glm::vec3 F = w[((size_t)env * MAX + m) * 2], T = w[((size_t)env * MAX + m) * 2 + 1];
            float* o = out + ((size_t)env * count() + m) * 6;
            o[0] = F.x; o[1] = F.y; o[2] = F.z; o[3] = T.x; o[4] = T.y; o[5] = T.z;
        }
}
