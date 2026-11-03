// Rod-mesh contact storage: signed-distance grids, per-env object poses and
// their frame-to-frame motion, the broad-phase candidate lists and the optional
// contact-force readout, all on the device, published to the solver through
// MetaData (YarnBall.h; kernels in sim/contact.cuh, sim/iteration.cu, sdf.cu).
// One store per RopeSim/RopeBatch, freed with it.
//
// Pose semantics (the Isaac contract): a target set without teleport is the
// pose the object will have at the END of the next step; the step interpolates
// from the current pose (constant linear + angular velocity over the frame, one
// pose per substep), and friction is relative to the moving surface. After the
// step the target is the current pose. With teleport the current pose is
// replaced at once (resets; a resting rope inside the new volume is ejected).
// Targets arrive from host arrays or from device arrays (torch) alike: both
// land in the device target buffer through one scatter kernel.
#pragma once

#include <cuda_runtime.h>
#include <glm/glm.hpp>

#include <cstdint>
#include <vector>

#include "YarnBall/YarnBall.h"

class SdfStore {
public:
    using Grid = YarnBall::MetaData::SdfGrid;
    using Pose = YarnBall::MetaData::SdfPose;
    using Motion = YarnBall::MetaData::SdfMotion;
    static constexpr int MAX = YarnBall::MetaData::kMaxSdf;
    static constexpr int CAND = YarnBall::MetaData::kMaxSdfCand;
    struct Pose7 { glm::vec3 pos; glm::vec4 q; };
    struct Box { glm::vec3 lo, hi; };

    explicit SdfStore(int num_envs);
    ~SdfStore();
    SdfStore(const SdfStore&) = delete;
    SdfStore& operator=(const SdfStore&) = delete;

    int count() const { return (int)grids.size(); }
    int envs() const { return N; }
    int num_verts() const { return numVerts; }
    bool forces_enabled() const { return forcesOn; }
    int last_substeps() const { return lastSubsteps; }

    // values: nx*ny*nz floats, C order (i slowest); node (i, j, k) sits at
    // origin + (i, j, k) * spacing in the object's frame; muS/muK its friction;
    // pos/q host rows (1 or N) = the initial pose (teleport). Returns the index.
    int add(const float* values, int nx, int ny, int nz, glm::vec3 origin, float spacing, float muS, float muK,
            const float* pos, const float* q, int rows, cudaStream_t s);
    void set_friction(int index, float muS, float muK);
    // Targets from DEVICE arrays for mesh `index`: pos [rows*3], q [rows*4] (rows 1 or N,
    // q = x,y,z,w, normalized here); or, with index = -1, for every mesh from a dense
    // pos [N, count(), 3], q [N, count(), 4]. tele [N] uint8 (device) or null; teleAll overrides.
    void set_targets(int index, const float* d_pos, const float* d_q, int rows, const uint8_t* d_tele, bool teleAll,
                     cudaStream_t s);
    // The same from HOST arrays (validated by the caller), staged through a device buffer.
    void set_targets_host(int index, const float* pos, const float* q, int rows, const uint8_t* tele, bool teleAll,
                          cudaStream_t s);
    // Host teleport flags staged for a device-array call (null / teleAll -> null).
    const uint8_t* stage_flags(const uint8_t* tele, bool teleAll, cudaStream_t s);
    void clear();
    void enable_forces(bool on);

    // Publish to the host-side MetaData (uploaded by the next step); allocates
    // the per-vertex buffers for num_envs * env_verts vertices on first use.
    void write_meta(YarnBall::MetaData& m, int env_verts);
    // Before a step's uploadMeta: stage this frame's motion, reset the broad
    // phase, zero the force accumulators. After the step: commit the targets as
    // the current poses (static poses + AABBs on the device), average the forces.
    void begin_step(YarnBall::MetaData& m, cudaStream_t s);
    void end_step(int substeps, cudaStream_t s);

    // Readouts, averaged over the last step's substeps (host copies), and the
    // device buffers behind them: contact forces [numVerts, 3] (ON the rope),
    // wrenches [N, MAX, 6] (force, torque about the mesh origin, world frame,
    // BY the rope ON the mesh; slots >= count() are zero).
    void contact_forces(float* out) const;
    void wrenches(float* out) const;            // [N, count(), 6]
    uintptr_t contact_forces_ptr() const { return (uintptr_t)d_force; }
    uintptr_t wrenches_ptr() const { return (uintptr_t)d_wrench; }

private:
    void alloc_verts(int nv);
    void commit(cudaStream_t s);

    int N = 0;
    int numVerts = 0;
    std::vector<Grid> grids;
    std::vector<Box> boxes;          // object-frame lattice box per mesh (host mirror of d_box)
    bool pendingSet = false;         // targets set since the last step
    bool moving = false;             // this step interpolates
    bool forcesOn = false;
    int lastSubsteps = 1;
    Pose7* d_cur = nullptr;          // [N * MAX] current pose
    Pose7* d_tgt = nullptr;          // [N * MAX] target pose
    uint8_t* d_tele = nullptr;       // [N * MAX] teleport flags
    Motion* d_motion = nullptr;      // [N * MAX]
    Pose* d_poses = nullptr;         // [N * MAX] what the kernels read
    Box* d_box = nullptr;            // [MAX]
    float* d_stage = nullptr;        // [N * MAX * 7] host-call staging
    uint8_t* d_stageTele = nullptr;  // [N]
    int* d_cand = nullptr;           // [numVerts * CAND]
    int* d_numCand = nullptr;        // [numVerts]
    glm::vec3* d_force = nullptr;    // [numVerts]
    glm::vec3* d_wrench = nullptr;   // [N * MAX * 2]
};
