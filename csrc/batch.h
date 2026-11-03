// RopeBatch — N identical ropes packed into ONE YarnBall::Sim.
//
// Rope e owns vertices [e*V, (e+1)*V). Per the batching requirements
// (INTEGRATION_PLAN.md): each rope's last vertex gets flags=0 and each root
// is pinned (invMass=0, optional fixOrientation) BEFORE configure(), which
// then wires hasPrev/hasNext/masses/frames per rod (cross-rope qRests exist
// but are never read; collisions are off, so rods are fully independent).
//
// Driving and observation are GPU-resident: apply_roots / gather / reset run
// on the solver's own stream (solverStream(), our YarnBall-Linux accessor),
// so they are ordered with the solve at zero extra host syncs. Interop with
// torch relies on legacy-default-stream semantics (torch's default stream is
// the legacy stream, which serializes with our blocking stream); keep torch
// on its default stream or synchronize explicitly.
//
// There is deliberately NO host set_root()/upload() path here: once the GPU
// owns root state, a full host push would clobber it (review finding).

#pragma once

#include <cstdint>
#include <memory>
#include <mutex>

#include "YarnBall/YarnBall.h"

namespace ropesim {

class RopeBatch {
public:
    // root_pos_host: [N*3] world-space root of each rope; dir3: shared unit
    // direction of the initial straight geometry (e.g. {0,0,-1} hanging).
    RopeBatch(int num_envs, int verts_per_rope, float seg_len,
              const float* root_pos_host, const float* dir3,
              float density, float k_stretch, float k_bend,
              float max_h, int num_itr, bool fix_root_orientation,
              const float* init_pos_host = nullptr,    // optional [N*V*3] initial shape
              const uint8_t* pinned_host = nullptr);   // optional [N*V] pins (default: each root)
    ~RopeBatch();
    RopeBatch(const RopeBatch&) = delete;
    RopeBatch& operator=(const RopeBatch&) = delete;

    // One control frame: write root pos/vel (and quat if non-null) from
    // device float32 arrays [N,3]/[N,3]/[N,4], run exactly `substeps`
    // substeps of h = dt/substeps, then refresh the observation buffers.
    // Constant-velocity sweep (the validated legacy path, via Sim::advance).
    void step(const float* d_root_pos, const float* d_root_vel,
              const float* d_root_q, float dt, int substeps);

    // One control frame with a SECOND-ORDER root motion program: the root
    // follows p(tau) = p0 + v0*tau + 0.5*a*tau^2 with a chosen to land
    // exactly on p1 at tau=dt (v0 = the MEASURED root-point velocity at the
    // frame start -> C1-continuous driving up to O(dt^2)), and the root frame
    // is SLERPed q0 -> q1 per substep. q0/q1 may be null (orientation
    // untouched) or q1 null (q0 held). Runs the substep loop itself
    // (YarnBall::SimAccess) to inject the program kernel between substeps.
    void step_program(const float* d_p0, const float* d_v0, const float* d_p1,
                      const float* d_q0, const float* d_q1,
                      float dt, int substeps);

    // Restore the construction-time state (pos/orientation, zero velocities)
    // for the given env indices (host array, pre-validated). Observation
    // buffers are refreshed. Global solver bookkeeping (meta.time, lastH)
    // is shared across envs and intentionally untouched.
    void reset_envs(const int* ids_host, int count);

    void gather();  // refresh observation buffers outside step (rarely needed)

    // Contiguous float32 device SCRATCH buffers, refreshed by
    // step()/reset_envs() — copies, so Python-side writes cannot corrupt
    // solver state:
    //   positions   [N, V, 3]
    //   velocities  [N, V, 3]  (valid between steps; roots show the commanded
    //                           velocity — in program mode the last substep's
    //                           midpoint sample, i.e. v(dt - h/2))
    //   quaternions [N, V, 4]  (x,y,z,w; entry (e, V-1) is padding)
    uintptr_t positions_ptr() const { return (uintptr_t)d_outPos; }
    uintptr_t velocities_ptr() const { return (uintptr_t)d_outVel; }
    uintptr_t quaternions_ptr() const { return (uintptr_t)d_outQuat; }

    int num_envs() const { return N; }
    int verts_per_rope() const { return V; }
    bool fix_root_orientation() const { return fixRootOrientation_; }

    YarnBall::Sim* raw() { return sim.get(); }
    std::mutex& mutex() { return mtx_; }

    // Set by the bindings on the first step()/step_program(). Per the review
    // finding above there is no host upload path AFTER the GPU owns root
    // state; the per-vertex property setters (bindings.cpp) therefore refuse
    // to run once this is true. Between construction and the first step the
    // host mirror is authoritative, so download-mutate-upload is safe there.
    bool stepped = false;

    // False when from_positions() left some rope's vertex 0 free: step() drives
    // each root every substep, so the bindings then allow step_free() only.
    bool rootsPinned = true;

    // Rope-rope contact (per-group brute-force detection + YarnBall's contact
    // model). groupRopes consecutive ropes form one contact group (an env);
    // ropes in different groups never interact.
    void set_rope_contact(bool enabled, float radius, float barrier, float kCollision,
                          float mu, float kFriction, float detectionScaler, int groupRopes);
    // Substeps without a root program (free ropes / roots held in place).
    void step_free(float dt, int substeps);
    bool ropeContact = false;
    int groupVerts = 0;

private:
    void gather_nolock();

    std::unique_ptr<YarnBall::Sim> sim;
    int N = 0, V = 0;
    bool fixRootOrientation_ = false;

    glm::vec3* d_initPos = nullptr;  // [N*V] construction-time positions
    glm::vec4* d_initQ = nullptr;    // [N*V] construction-time frames
    float* d_outPos = nullptr;       // [N*V*3] gathered positions
    float* d_outVel = nullptr;       // [N*V*3] gathered velocities
    float* d_outQuat = nullptr;      // [N*V*4] gathered segment frames
    int* d_ids = nullptr;            // [N] scratch for reset indices
    float* d_prog = nullptr;         // [17*N] staged motion program
                                     // (p0,v0,p1:3N each; q0,q1:4N each) —
                                     // copied from caller tensors at step
                                     // entry so the 50 launches read stable
                                     // memory

    std::mutex mtx_;
};

}  // namespace ropesim
