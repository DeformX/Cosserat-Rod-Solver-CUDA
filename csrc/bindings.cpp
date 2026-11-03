// ropesim — pybind11 wrap of the de-collisioned YarnBall Cosserat solver.
//
// Single rope, root (vertex 0) kinematically pinned. Validation-oriented API:
// full state introspection (positions/velocities/quaternions/masses), solver
// knobs (drag/damping/num_itr/...), reset, per-segment stiffness, and optional
// root-orientation clamping + driving (the EE boundary condition).
//
// Unit mapping (validated by tests/):
//   density   = linear mass density rho  [kg/m]
//   k_stretch = axial stiffness EA       [N]
//   k_bend    = bending stiffness EI     [N*m^2]  (must be > 0: zero rest-frame
//               vectors make the closed-form orientation update normalize(0)=NaN)
// Host round-trips (download/upload) per call are fine at validation scale;
// the batched GPU pin-kernel path replaces them in M1. All methods take an
// object-level mutex so a concurrent Python thread cannot race an advance()
// that runs with the GIL released.

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <mutex>
#include <optional>
#include <vector>

#include "YarnBall/YarnBall.h"
#include "batch.h"
#include "sdf.h"

namespace py = pybind11;

namespace {

using farray = py::array_t<float, py::array::c_style | py::array::forcecast>;

glm::vec3 to_vec3(const farray& a, const char* name) {
    if (a.ndim() != 1 || a.shape(0) != 3)
        throw py::value_error(std::string(name) + " must be a length-3 array");
    auto b = a.unchecked<1>();
    return glm::vec3(b(0), b(1), b(2));
}

// Rod-plane contact (cosserat.cu planeContact). planes: [P<=8, 4] rows
// (nx, ny, nz, w) = half-spaces n.x >= w; P = 0 turns contact off. Every row is
// validated before anything is written, so a bad call leaves the old state.
// Surface-contact material (shared by planes and meshes): rope radius, static /
// kinetic friction, slip speed, stiffness factor kappa (k_i = m_i (kappa/h)^2).
void write_contact(YarnBall::MetaData& m, float radius, float mu_s, float mu_k, float v_slip, float kappa) {
    if (!(radius > 0) || !(kappa > 0) || !(mu_s >= 0) || !(mu_k >= 0) || !(v_slip > 0))
        throw py::value_error("surface contact needs radius > 0, kappa > 0, mu_s >= 0, mu_k >= 0, v_slip > 0");
    m.contactRadius = radius; m.contactMuS = mu_s; m.contactMuK = mu_k;
    m.contactVSlip = v_slip; m.contactKappa = kappa;
}

// set_planes(planes, radius, ...): the planes plus the shared material (sugar for
// set_contact); with P > 0 the material is validated and written, with P == 0 only
// the planes are switched off and the material (meshes may be using it) is kept.
void write_planes(YarnBall::MetaData& m, const farray& planes, float radius, float mu_s,
                  float mu_k, float v_slip, float kappa) {
    if (planes.ndim() != 2 || planes.shape(1) != 4 || planes.shape(0) > 8)
        throw py::value_error("planes must be [P<=8, 4] rows (nx, ny, nz, w): half-spaces n.x >= w");
    const int P = (int)planes.shape(0);
    if (P > 0 && (!(radius > 0) || !(kappa > 0) || !(mu_s >= 0) || !(mu_k >= 0) || !(v_slip > 0)))
        throw py::value_error("plane contact needs radius > 0, kappa > 0, mu_s >= 0, mu_k >= 0, v_slip > 0");
    auto r = planes.unchecked<2>();
    glm::vec4 rows[8];
    for (int i = 0; i < P; i++) {
        glm::vec3 nrm(r(i, 0), r(i, 1), r(i, 2));
        const float l = glm::length(nrm);
        if (!(l > 0) || !std::isfinite(l) || !std::isfinite(r(i, 3)))
            throw py::value_error("plane rows need a finite, nonzero normal and a finite offset");
        rows[i] = glm::vec4(nrm / l, r(i, 3) / l);
    }
    for (int i = 0; i < P; i++) m.planes[i] = rows[i];
    m.numPlanes = P;
    if (P > 0) write_contact(m, radius, mu_s, mu_k, v_slip, kappa);   // an empty list only turns planes off
}

// Rod-mesh contact (sdf.h, cosserat.cu sdfContact). values [nx, ny, nz] float32
// signed distances on a node lattice at origin + (i, j, k) * spacing (object frame).
struct GridArgs { const float* v; int nx, ny, nz; glm::vec3 origin; float spacing; };
GridArgs parse_grid(const farray& values, const farray& origin, float spacing) {
    if (values.ndim() != 3 || values.shape(0) < 2 || values.shape(1) < 2 || values.shape(2) < 2)
        throw py::value_error("sdf values must be [nx, ny, nz] with every dim >= 2");
    if (!(spacing > 0) || !std::isfinite(spacing)) throw py::value_error("sdf spacing must be > 0");
    const float* v = values.data();
    for (py::ssize_t i = 0; i < values.size(); i++)
        if (!std::isfinite(v[i])) throw py::value_error("sdf values must be finite");
    return GridArgs{v, (int)values.shape(0), (int)values.shape(1), (int)values.shape(2), to_vec3(origin, "origin"), spacing};
}
// Host pose rows: finite positions, finite nonzero quaternions (checked before any device state changes).
void validate_rows(const float* p, const float* q, int rows, const char* who) {
    for (int r = 0; r < rows; r++) {
        const float l = std::sqrt(q[4 * r] * q[4 * r] + q[4 * r + 1] * q[4 * r + 1] + q[4 * r + 2] * q[4 * r + 2] + q[4 * r + 3] * q[4 * r + 3]);
        if (!std::isfinite(l) || !(l > 1e-6f)) throw py::value_error(std::string(who) + ": quat must be finite with nonzero norm");
        if (!std::isfinite(p[3 * r]) || !std::isfinite(p[3 * r + 1]) || !std::isfinite(p[3 * r + 2]))
            throw py::value_error(std::string(who) + ": pos must be finite");
    }
}
// pos [3] or [rows, 3]; quat [4] or [rows, 4] (x, y, z, w). Returns the row count (1 or N).
int parse_poses(const farray& pos, const farray& quat, int N, const char* who) {
    const bool p1 = pos.ndim() == 1 && pos.shape(0) == 3, pN = pos.ndim() == 2 && pos.shape(0) == N && pos.shape(1) == 3;
    const bool q1 = quat.ndim() == 1 && quat.shape(0) == 4, qN = quat.ndim() == 2 && quat.shape(0) == N && quat.shape(1) == 4;
    if (!((p1 && q1) || (pN && qN)))
        throw py::value_error(std::string(who) + ": pos must be [3] and quat [4] (x, y, z, w), or [num_envs, 3] and [num_envs, 4]");
    const int rows = p1 ? 1 : N;
    validate_rows(pos.data(), quat.data(), rows, who);
    return rows;
}
// Parsed view of a device array handed to us by torch/cupy/numba.
struct CaiArray {
    uintptr_t ptr = 0;
};
CaiArray parse_cai(const py::object& obj, std::vector<py::ssize_t> want, const char* name);
// Pose rows for set_mesh_poses/add_mesh: host numpy ([3]/[4] or [N,3]/[N,4]) or
// CUDA arrays ([N,3]/[N,4] float32, e.g. torch on cuda). Host arrays are
// validated here; device arrays are normalized on the device.
struct PoseRows {
    bool device = false;
    int rows = 1;
    const float* pos = nullptr;   // host or device pointer
    const float* quat = nullptr;
    farray hp, hq;                // keep host conversions alive
};
PoseRows parse_pose_rows(const py::object& pos, const py::object& quat, int N, const char* who) {
    PoseRows r;
    const bool dp = py::hasattr(pos, "__cuda_array_interface__"), dq = py::hasattr(quat, "__cuda_array_interface__");
    if (dp != dq) throw py::value_error(std::string(who) + ": pos and quat must both be host arrays or both CUDA arrays");
    if (dp) {
        r.device = true; r.rows = N;
        r.pos = reinterpret_cast<const float*>(parse_cai(pos, {N, 3}, "pos").ptr);
        r.quat = reinterpret_cast<const float*>(parse_cai(quat, {N, 4}, "quat").ptr);
        return r;
    }
    r.hp = farray::ensure(pos); r.hq = farray::ensure(quat);
    if (!r.hp || !r.hq) throw py::value_error(std::string(who) + ": pos/quat must be arrays");
    r.rows = parse_poses(r.hp, r.hq, N, who);
    r.pos = r.hp.data(); r.quat = r.hq.data();
    return r;
}
// teleport: False/True for every env, a [num_envs] host bool array, or a [num_envs] CUDA bool array.
struct TeleFlags {
    bool all = false;
    bool device = false;
    const uint8_t* ptr = nullptr;
    std::vector<uint8_t> host;
};
TeleFlags parse_teleport(const py::object& teleport, int N, const char* who) {
    TeleFlags t;
    if (py::isinstance<py::bool_>(teleport)) { t.all = teleport.cast<bool>(); return t; }
    if (py::hasattr(teleport, "__cuda_array_interface__")) {
        py::dict d = teleport.attr("__cuda_array_interface__").cast<py::dict>();
        const auto ts = d["typestr"].cast<std::string>();
        auto shape = d["shape"].cast<std::vector<py::ssize_t>>();
        py::object st = d.attr("get")("strides", py::none());
        const bool contiguous = st.is_none() || N == 1 || st.cast<std::vector<py::ssize_t>>() == std::vector<py::ssize_t>{1};
        if ((ts != "|b1" && ts != "|u1") || shape != std::vector<py::ssize_t>{N} || !contiguous)
            throw py::value_error(std::string(who) + ": teleport must be a bool, a [num_envs] bool array, or a contiguous [num_envs] CUDA bool array");
        t.device = true; t.ptr = reinterpret_cast<const uint8_t*>(d["data"].cast<py::tuple>()[0].cast<uintptr_t>());
        return t;
    }
    auto a = py::array_t<bool, py::array::c_style | py::array::forcecast>::ensure(teleport);
    if (!a || a.ndim() != 1 || a.shape(0) != N)
        throw py::value_error(std::string(who) + ": teleport must be a bool, a [num_envs] bool array, or a [num_envs] CUDA bool array");
    t.host.resize(N);
    for (int i = 0; i < N; i++) t.host[i] = a.data()[i] ? 1 : 0;
    t.ptr = t.host.data();
    return t;
}
// One entry point for both classes: rows + flags -> the store.
void set_mesh_targets(SdfStore& sdf, int index, const PoseRows& r, const TeleFlags& t, cudaStream_t s, const char* who) {
    if (r.device) sdf.set_targets(index, r.pos, r.quat, r.rows, t.device ? t.ptr : sdf.stage_flags(t.ptr, t.all, s), t.all, s);
    else if (t.device) throw py::value_error(std::string(who) + ": host poses need teleport as a bool or a host bool array");
    else sdf.set_targets_host(index, r.pos, r.quat, r.rows, t.all ? nullptr : t.ptr, t.all, s);
}
float mu_or(const py::object& mu, float dflt) { return mu.is_none() ? dflt : mu.cast<float>(); }
void require_contact_radius(const YarnBall::MetaData& m) {
    if (!(m.contactRadius > 0))
        throw py::value_error("set the contact material first: set_contact(radius, ...) or set_planes(..., radius=)");
}

class RopeSim {
    std::unique_ptr<YarnBall::Sim> sim;  // RAII: ctor throws must not leak
    SdfStore sdf_{1};
    int n = 0;
    bool collisions_ = false;
    bool fixRootOrientation_ = false;
    std::vector<glm::vec3> initPos;
    std::vector<Kit::Rotor> initQ;
    mutable std::mutex mtx_;

public:
    RopeSim(int num_verts, float seg_len, glm::vec3 dir, float density,
            float k_stretch, float k_bend, float max_h, int num_itr,
            bool fix_root_orientation, bool collisions) {
        if (num_verts < 3) throw py::value_error("num_verts must be >= 3");
        if (!(seg_len > 0)) throw py::value_error("seg_len must be > 0");
        if (!(density > 0)) throw py::value_error("density must be > 0");
        if (!(k_stretch > 0)) throw py::value_error("k_stretch must be > 0");
        if (!(k_bend > 0))
            throw py::value_error(
                "k_bend must be > 0: k_bend=0 zeroes the rest-frame vectors and "
                "the orientation solve produces NaNs (normalize of a zero vector)");
        if (!(max_h > 0)) throw py::value_error("max_h must be > 0");
        if (num_itr < 1) throw py::value_error("num_itr must be >= 1");
        float dl = glm::length(dir);
        if (!(dl > 0)) throw py::value_error("dir must be nonzero");

        n = num_verts;
        collisions_ = collisions;
        fixRootOrientation_ = fix_root_orientation;
        sim = std::make_unique<YarnBall::Sim>(num_verts);

        glm::vec3 d = dir / dl;
        for (int i = 0; i < num_verts; i++)
            sim->verts[i].pos = float(i) * seg_len * d;

        sim->verts[0].invMass = 0;  // kinematic root
        if (fix_root_orientation)   // clamp segment-0 frame (EE orientation BC)
            sim->verts[0].flags |= (uint32_t)YarnBall::VertexFlags::fixOrientation;

        if (collisions) {           // upstream defaults (for equivalence tests)
            sim->meta.detectionPeriod = 1;
            sim->meta.useStepSizeLimit = 1;
            // uploadMeta() requires 2*seg_len >= 2*radius + barrierThickness.
            // With the upstream yarn-scale defaults that means seg_len >= 5e-4 m;
            // fail early with a clear message instead of upstream's
            // "Use thinner yarn" wording.
            if (2 * seg_len < 2 * sim->meta.radius + sim->meta.barrierThickness)
                throw py::value_error(
                    "collisions=True uses the upstream yarn-scale collision radius "
                    "(1e-4 m): seg_len must be >= 5e-4 m");
        } else {                    // de-collisioned config (the RL path)
            sim->meta.detectionPeriod = -1;
            sim->meta.useStepSizeLimit = 0;
            // radius/barrierThickness feed only the collision pipeline (dead
            // here) plus a validity check in uploadMeta(); scale them with the
            // segment length so arbitrarily fine discretizations construct.
            sim->meta.radius = 0.01f * seg_len;
            sim->meta.barrierThickness = 0.08f * seg_len;
        }

        sim->configure(density);    // density = mass per unit length
        sim->setKStretch(k_stretch);
        sim->setKBend(k_bend);
        sim->maxH = max_h;
        // Sim's ctor seeded meta.h with its default maxH (1e-3); align it so
        // lastH (used by initItr's extrapolation) matches a post-reset sim —
        // keeps fresh and reset trajectories bitwise identical.
        sim->meta.h = max_h;
        sim->meta.numItr = num_itr;
        sim->printErrors = false;

        sim->upload();

        initPos.resize(n);
        initQ.resize(n);
        for (int i = 0; i < n; i++) {
            initPos[i] = sim->verts[i].pos;
            initQ[i] = sim->qs[i];
        }
    }

    RopeSim(const RopeSim&) = delete;
    RopeSim& operator=(const RopeSim&) = delete;

    // Advance by dt. substeps=0: upstream behavior, steps = ceil(dt/max_h)
    // (beware fp: dt=1/60, max_h=1/3000 gives 51, not 50). substeps>0: run
    // EXACTLY that many substeps of h = dt/substeps (max_h restored after) —
    // the contract the Isaac Lab step() will use.
    void advance(float dt, int substeps) {
        if (substeps < 0) throw py::value_error("substeps must be >= 0");
        if (!(dt > 0)) return;                              // no substep runs: a pending mesh target stays pending (review 3)
        std::lock_guard<std::mutex> g(mtx_);
        sdf_.begin_step(sim->meta, sim->solverStream());   // kinematic mesh motion / force readout
        if (substeps > 0) {
            const float oldMaxH = sim->maxH;
            // bias just above dt/substeps so ceil(dt/maxH) == substeps exactly
            sim->maxH = (dt / substeps) * 1.000001f;
            try {
                sim->advance(dt);
            } catch (...) {
                sim->maxH = oldMaxH;
                throw;
            }
            sim->maxH = oldMaxH;
            sdf_.end_step(substeps, sim->solverStream());
        } else {
            sim->advance(dt);
            sdf_.end_step(dt > 0 ? std::max(1, (int)std::ceil(dt / sim->maxH)) : 1, sim->solverStream());
        }
    }

    // Restore initial state (positions/orientations, zero velocities) and
    // clear error/warning codes. Parameters (stiffness, drag, ...) are
    // intentionally NOT restored (RL-reset semantics). NOTE: upstream-private
    // bookkeeping (stepCounter, BVH cadence) is not reachable — irrelevant
    // with collisions off; with collisions=True the detection phase after
    // reset() differs from a fresh sim by at most one detectionPeriod.
    void reset() {
        std::lock_guard<std::mutex> g(mtx_);
        for (int i = 0; i < n; i++) {
            sim->verts[i].pos = initPos[i];
            sim->qs[i] = initQ[i];
            sim->vels[i] = glm::vec3(0);
        }
        sim->upload();
        sim->zeroVelocities();  // also clears device lastVels
        sim->meta.time = 0;
        sim->meta.h = sim->maxH;  // match the freshly-constructed state
        sim->lastErrorCode = YarnBall::Sim::ERROR_NONE;
        sim->lastWarningCode = YarnBall::Sim::ERROR_NONE;
    }

    // Drive the pinned root. The solver applies pos += h*vel per substep, so a
    // constant vel sweeps the root linearly across the frame (boundary velocity
    // injection). q (x,y,z,w) clamps the root-segment frame; it is normalized
    // here and only legal with fix_root_orientation=True — on a free-pivot
    // root the orientation kernel would overwrite it mid-solve, so it could
    // only ever act as a transient, nonphysical impulse.
    void set_root(glm::vec3 pos, glm::vec3 vel, py::object q) {
        bool has_q = !q.is_none();
        Kit::Rotor rq;
        if (has_q) {
            if (!fixRootOrientation_)
                throw py::value_error(
                    "set_root(q=...) requires fix_root_orientation=True");
            farray qa = q.cast<farray>();
            if (qa.ndim() != 1 || qa.shape(0) != 4)
                throw py::value_error("q must be a length-4 array (x,y,z,w)");
            auto b = qa.unchecked<1>();
            glm::vec4 v(b(0), b(1), b(2), b(3));
            float l = std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w);
            if (!std::isfinite(l) || !(l > 1e-6f))
                throw py::value_error("q must be finite with nonzero norm");
            rq = Kit::Rotor(v / l);  // the fixed frame is never renormalized
                                     // by the solver — enforce unit norm here
        }
        std::lock_guard<std::mutex> g(mtx_);
        if (sim->verts[0].invMass != 0)
            throw py::value_error("set_root drives a pinned root; this rope's root was unpinned (unpin_root)");
        sim->download();
        sim->verts[0].pos = pos;
        sim->vels[0] = vel;
        if (has_q) sim->qs[0] = rq;
        sim->upload();
    }

    int num_verts() const { return n; }

    py::array_t<float> positions() {
        std::lock_guard<std::mutex> g(mtx_);
        sim->download();
        py::array_t<float> arr({n, 3});
        auto b = arr.mutable_unchecked<2>();
        for (int i = 0; i < n; i++) {
            const glm::vec3 p = sim->verts[i].pos;
            b(i, 0) = p.x; b(i, 1) = p.y; b(i, 2) = p.z;
        }
        return arr;
    }

    // Valid between advance() calls only (d_vels holds the inertial target
    // mid-step). Pinned verts report their commanded velocity.
    py::array_t<float> velocities() {
        std::lock_guard<std::mutex> g(mtx_);
        sim->download();
        py::array_t<float> arr({n, 3});
        auto b = arr.mutable_unchecked<2>();
        for (int i = 0; i < n; i++) {
            const glm::vec3 v = sim->vels[i];
            b(i, 0) = v.x; b(i, 1) = v.y; b(i, 2) = v.z;
        }
        return arr;
    }

    // Segment frames as quaternions (x,y,z,w); entry i is the frame of segment
    // i -> i+1 (last entry is unused padding kept for shape regularity).
    py::array_t<float> quaternions() {
        std::lock_guard<std::mutex> g(mtx_);
        sim->download();
        py::array_t<float> arr({n, 4});
        auto b = arr.mutable_unchecked<2>();
        for (int i = 0; i < n; i++) {
            const glm::vec4 v = sim->qs[i].v;
            b(i, 0) = v.x; b(i, 1) = v.y; b(i, 2) = v.z; b(i, 3) = v.w;
        }
        return arr;
    }

    py::array_t<float> inv_masses() {
        std::lock_guard<std::mutex> g(mtx_);
        py::array_t<float> arr(n);
        auto b = arr.mutable_unchecked<1>();
        for (int i = 0; i < n; i++) b(i) = sim->verts[i].invMass;
        return arr;
    }

    // Rest length of segment i -> i+1, length n-1.
    py::array_t<float> rest_lengths() {
        std::lock_guard<std::mutex> g(mtx_);
        py::array_t<float> arr(n - 1);
        auto b = arr.mutable_unchecked<1>();
        for (int i = 0; i < n - 1; i++) b(i) = sim->verts[i].lRest;
        return arr;
    }

    // Per-segment axial stiffness EA [N], length n-1, all entries > 0.
    void set_stretch_stiffnesses(const farray& ea) {
        if (ea.ndim() != 1 || ea.shape(0) != n - 1)
            throw py::value_error("expected length n-1 array of EA values");
        auto b = ea.unchecked<1>();
        for (int i = 0; i < n - 1; i++)
            if (!std::isfinite(b(i)) || !(b(i) > 0))
                throw py::value_error("EA values must be finite and > 0");
        std::lock_guard<std::mutex> g(mtx_);
        sim->download();
        for (int i = 0; i < n - 1; i++)
            sim->verts[i].kStretch = b(i) * sim->verts[i].lRest;
        sim->upload();
    }

    // Per-joint bending stiffness EI [N*m^2], length n-1 (last entry unused),
    // all entries > 0 (zero would zero the rest-frame vector -> NaN, see ctor).
    void set_bend_stiffnesses(const farray& ei) {
        if (ei.ndim() != 1 || ei.shape(0) != n - 1)
            throw py::value_error("expected length n-1 array of EI values");
        auto b = ei.unchecked<1>();
        for (int i = 0; i < n - 1; i++)
            if (!std::isfinite(b(i)) || !(b(i) > 0))
                throw py::value_error("EI values must be finite and > 0");
        std::lock_guard<std::mutex> g(mtx_);
        sim->download();
        for (int i = 0; i < n - 1; i++) {
            float k = 4.f * b(i) / sim->verts[i].lRest;  // setKBend's scaling
            sim->qRests[i] = k * glm::normalize((glm::vec4)sim->qRests[i]);
        }
        sim->upload();
    }

    // Overwrite all vertex velocities [n,3] (diagnostic/test API). NOTE: a
    // pinned vertex's row sets its KINEMATIC velocity — the root advects at
    // pos += h*v per substep exactly like set_root() driving (only the
    // drag/velocity WRITE-BACK is invMass-guarded, not advection). Pass
    // zeros for pinned rows to keep them stationary.
    void set_velocities(const farray& vel) {
        if (vel.ndim() != 2 || vel.shape(0) != n || vel.shape(1) != 3)
            throw py::value_error("expected [n,3] array of velocities");
        auto b = vel.unchecked<2>();
        for (int i = 0; i < n; i++)
            for (int j = 0; j < 3; j++)
                if (!std::isfinite(b(i, j)))
                    throw py::value_error("velocities must be finite");
        std::lock_guard<std::mutex> g(mtx_);
        sim->download();
        for (int i = 0; i < n; i++)
            sim->vels[i] = glm::vec3(b(i, 0), b(i, 1), b(i, 2));
        sim->upload();
    }

    // --- rod-plane contact: see write_planes ------------------------------
    void set_planes(const farray& planes, float radius, float mu_s, float mu_k, float v_slip, float kappa) {
        std::lock_guard<std::mutex> g(mtx_);
        write_planes(sim->meta, planes, radius, mu_s, mu_k, v_slip, kappa);
    }
    void set_contact(float radius, float mu_s, float mu_k, float v_slip, float kappa) {
        std::lock_guard<std::mutex> g(mtx_);
        write_contact(sim->meta, radius, mu_s, mu_k, v_slip, kappa);
    }
    // --- rod-mesh contact: see sdf.h ------------------------------------------
    int add_mesh(const farray& values, const farray& origin, float spacing, const farray& pos, const farray& quat,
                 py::object mu_s, py::object mu_k) {
        const GridArgs ga = parse_grid(values, origin, spacing);
        parse_poses(pos, quat, 1, "add_mesh");
        std::lock_guard<std::mutex> g(mtx_);
        require_contact_radius(sim->meta);
        const float ms = mu_or(mu_s, sim->meta.contactMuS), mk = mu_or(mu_k, mu_s.is_none() ? sim->meta.contactMuK : ms);
        if (!(ms >= 0) || !(mk >= 0)) throw py::value_error("mu_s / mu_k must be >= 0");
        const int idx = sdf_.add(ga.v, ga.nx, ga.ny, ga.nz, ga.origin, ga.spacing, ms, mk, pos.data(), quat.data(), 1,
                                 sim->solverStream());
        sdf_.write_meta(sim->meta, n);
        return idx;
    }
    void set_mesh_friction(int index, float mu_s, py::object mu_k) {
        if (!(mu_s >= 0)) throw py::value_error("mu_s must be >= 0");
        const float mk = mu_or(mu_k, mu_s);
        if (!(mk >= 0)) throw py::value_error("mu_k must be >= 0");
        std::lock_guard<std::mutex> g(mtx_);
        sdf_.set_friction(index, mu_s, mk);
        sdf_.write_meta(sim->meta, n);
    }
    // teleport=False: the pose at the END of the next advance() (the mesh moves
    // there over the substeps, friction relative to it); True: replaced now.
    void set_mesh_pose(int index, const farray& pos, const farray& quat, bool teleport) {
        parse_poses(pos, quat, 1, "set_mesh_pose");
        std::lock_guard<std::mutex> g(mtx_);
        sdf_.set_targets_host(index, pos.data(), quat.data(), 1, nullptr, teleport, sim->solverStream());
    }
    void clear_meshes() {
        std::lock_guard<std::mutex> g(mtx_);
        sdf_.clear();
        sdf_.write_meta(sim->meta, n);
    }
    // Contact-force readout (planes + meshes), averaged over the last advance().
    void enable_contact_forces(bool on) {
        std::lock_guard<std::mutex> g(mtx_);
        sdf_.enable_forces(on);
        sdf_.write_meta(sim->meta, n);
    }
    py::array_t<float> contact_forces() const {   // [V, 3]: force ON each vertex
        std::lock_guard<std::mutex> g(mtx_);
        py::array_t<float> out({n, 3});
        sdf_.contact_forces(out.mutable_data());
        return out;
    }
    py::array_t<float> mesh_wrenches() const {    // [M, 6]: (force, torque about the mesh origin) BY the rope ON each mesh
        std::lock_guard<std::mutex> g(mtx_);
        py::array_t<float> out({sdf_.count(), 6});
        sdf_.wrenches(out.mutable_data());
        return out;
    }
    // Make vertex 0 a free mass (uniform segments: m0 = m1/2). Test/scene helper.
    void unpin_root() {
        std::lock_guard<std::mutex> g(mtx_);
        sim->download();
        sim->verts[0].invMass = 2.f * sim->verts[1].invMass;
        sim->vels[0] = glm::vec3(0);
        sim->upload();
    }

    // --- meta knobs (plain field writes; re-uploaded by the next advance) ---
    float get_drag() const { std::lock_guard<std::mutex> g(mtx_); return sim->meta.drag; }
    void set_drag(float v) { std::lock_guard<std::mutex> g(mtx_); sim->meta.drag = v; }
    // Quadratic cross-flow air drag (see YarnBall.h MetaData): drag_quad is
    // the normal coefficient .5*rho_air*Cd_n*d/lambda [1/m] (0 = legacy
    // bit-exact path), drag_quad_t the tangential one (active only while
    // drag_quad > 0).
    float get_drag_quad() const { std::lock_guard<std::mutex> g(mtx_); return sim->meta.dragQuad; }
    void set_drag_quad(float v) {
        if (!std::isfinite(v) || v < 0) throw py::value_error("drag_quad must be finite and >= 0");
        std::lock_guard<std::mutex> g(mtx_);
        sim->meta.dragQuad = v;
    }
    float get_drag_quad_t() const { std::lock_guard<std::mutex> g(mtx_); return sim->meta.dragQuadT; }
    void set_drag_quad_t(float v) {
        if (!std::isfinite(v) || v < 0) throw py::value_error("drag_quad_t must be finite and >= 0");
        std::lock_guard<std::mutex> g(mtx_);
        sim->meta.dragQuadT = v;
    }
    float get_damping() const { std::lock_guard<std::mutex> g(mtx_); return sim->meta.damping; }
    void set_damping(float v) { std::lock_guard<std::mutex> g(mtx_); sim->meta.damping = v; }
    int get_num_itr() const { std::lock_guard<std::mutex> g(mtx_); return sim->meta.numItr; }
    void set_num_itr(int v) {
        if (v < 1) throw py::value_error("num_itr must be >= 1");
        std::lock_guard<std::mutex> g(mtx_);
        sim->meta.numItr = v;  // advance() re-captures the CUDA graph on change
    }
    float get_accel_ratio() const { std::lock_guard<std::mutex> g(mtx_); return sim->meta.accelerationRatio; }
    void set_accel_ratio(float v) { std::lock_guard<std::mutex> g(mtx_); sim->meta.accelerationRatio = v; }
    float get_max_h() const { std::lock_guard<std::mutex> g(mtx_); return sim->maxH; }
    void set_max_h(float v) {
        if (!(v > 0)) throw py::value_error("max_h must be > 0");
        std::lock_guard<std::mutex> g(mtx_);
        sim->maxH = v;
    }
    py::array_t<float> get_gravity() const {
        std::lock_guard<std::mutex> g(mtx_);
        py::array_t<float> arr(3);
        auto b = arr.mutable_unchecked<1>();
        b(0) = sim->meta.gravity.x; b(1) = sim->meta.gravity.y; b(2) = sim->meta.gravity.z;
        return arr;
    }
    void set_gravity(glm::vec3 gvec) { std::lock_guard<std::mutex> g(mtx_); sim->meta.gravity = gvec; }
    float time() const { std::lock_guard<std::mutex> g(mtx_); return sim->meta.time; }
    int last_error() const { std::lock_guard<std::mutex> g(mtx_); return sim->lastErrorCode; }
    int last_warning() const { std::lock_guard<std::mutex> g(mtx_); return sim->lastWarningCode; }
    bool collisions_enabled() const { return collisions_; }
    bool fixed_root_orientation() const { return fixRootOrientation_; }
};

// ---------------- zero-copy CUDA interop (__cuda_array_interface__) --------

// Validate + extract a float32, C-contiguous CUDA array of exactly `want`.
CaiArray parse_cai(const py::object& obj, std::vector<py::ssize_t> want,
                   const char* name) {
    if (!py::hasattr(obj, "__cuda_array_interface__"))
        throw py::value_error(std::string(name) +
                              " must be a CUDA array (e.g. a contiguous float32 "
                              "torch tensor on cuda)");
    py::dict d = obj.attr("__cuda_array_interface__").cast<py::dict>();
    const auto ts = d["typestr"].cast<std::string>();
    if (ts != "<f4")
        throw py::value_error(std::string(name) + " must be float32 (got " + ts + ")");
    auto shape = d["shape"].cast<std::vector<py::ssize_t>>();
    if (shape != want) {
        std::string s = std::string(name) + " has wrong shape; expected (";
        for (auto v : want) s += std::to_string(v) + ",";
        s += ")";
        throw py::value_error(s);
    }
    py::object strides_obj = d.attr("get")("strides", py::none());
    if (!strides_obj.is_none()) {
        auto strides = strides_obj.cast<std::vector<py::ssize_t>>();
        py::ssize_t expect = 4;  // float32, C order
        for (size_t i = shape.size(); i-- > 0;) {
            // stride of a size-1 dim is irrelevant (e.g. expand() gives 0)
            if (shape[i] != 1 && strides[i] != expect)
                throw py::value_error(std::string(name) + " must be C-contiguous");
            expect *= shape[i];
        }
    }
    auto data = d["data"].cast<py::tuple>();
    return {data[0].cast<uintptr_t>()};
}

// Device buffer exposed back to Python: torch.as_tensor(view) shares the
// memory (no copy). Keep the owning RopeBatch alive while views are in use
// (the bindings add keep_alive, and torch holds the view object).
struct CudaView {
    uintptr_t ptr;
    std::vector<py::ssize_t> shape;
};

}  // namespace

PYBIND11_MODULE(_ropesim, m) {
    m.doc() = "De-collisioned YarnBall Cosserat rod solver (single rope, pinned root).";

    py::class_<RopeSim>(m, "RopeSim")
        .def(py::init([](int num_verts, float seg_len, farray dir, float density,
                         float k_stretch, float k_bend, float max_h, int num_itr,
                         bool fix_root_orientation, bool collisions) {
                 return new RopeSim(num_verts, seg_len, to_vec3(dir, "dir"), density,
                                    k_stretch, k_bend, max_h, num_itr,
                                    fix_root_orientation, collisions);
             }),
             py::arg("num_verts"), py::arg("seg_len") = 0.02f,
             py::arg("dir") = farray(py::cast(std::vector<float>{1, 0, 0})),
             py::arg("density") = 0.1f,        // kg/m
             py::arg("k_stretch") = 500.0f,    // EA [N]
             py::arg("k_bend") = 1e-6f,        // EI [N*m^2], must be > 0
             py::arg("max_h") = 1.f / 3000.f,  // ~0.33 ms substep
             py::arg("num_itr") = 8,
             py::arg("fix_root_orientation") = false,
             py::arg("collisions") = false)
        .def("advance", &RopeSim::advance,
             py::arg("dt"), py::arg("substeps") = 0,
             py::call_guard<py::gil_scoped_release>())
        .def("reset", &RopeSim::reset)
        .def("set_root",
             [](RopeSim& s, farray pos, farray vel, py::object q) {
                 s.set_root(to_vec3(pos, "pos"), to_vec3(vel, "vel"), q);
             },
             py::arg("pos"), py::arg("vel"), py::arg("q") = py::none())
        .def("positions", &RopeSim::positions)
        .def("velocities", &RopeSim::velocities)
        .def("quaternions", &RopeSim::quaternions)
        .def("inv_masses", &RopeSim::inv_masses)
        .def("rest_lengths", &RopeSim::rest_lengths)
        .def("set_stretch_stiffnesses", &RopeSim::set_stretch_stiffnesses, py::arg("ea"))
        .def("set_bend_stiffnesses", &RopeSim::set_bend_stiffnesses, py::arg("ei"))
        .def("set_gravity",
             [](RopeSim& s, farray g) { s.set_gravity(to_vec3(g, "gravity")); },
             py::arg("gravity"))
        .def("set_planes",
             [](RopeSim& s, farray planes, float radius, float mu_s, py::object mu_k, float v_slip, float kappa) {
                 s.set_planes(planes, radius, mu_s, mu_k.is_none() ? mu_s : mu_k.cast<float>(), v_slip, kappa);
             },
             py::arg("planes"), py::arg("radius") = 0.f, py::arg("mu_s") = 0.5f, py::arg("mu_k") = py::none(),
             py::arg("v_slip") = 1e-3f, py::arg("kappa") = 2.f)
        .def("unpin_root", &RopeSim::unpin_root)
        .def("set_contact",
             [](RopeSim& s, float radius, float mu_s, py::object mu_k, float v_slip, float kappa) {
                 s.set_contact(radius, mu_s, mu_k.is_none() ? mu_s : mu_k.cast<float>(), v_slip, kappa);
             },
             py::arg("radius"), py::arg("mu_s") = 0.5f, py::arg("mu_k") = py::none(),
             py::arg("v_slip") = 1e-3f, py::arg("kappa") = 2.f,
             "Surface contact material shared by planes and meshes: rope radius [m], static/kinetic friction, slip speed [m/s], stiffness factor kappa (k_i = m_i (kappa/h)^2).")
        .def("add_mesh", &RopeSim::add_mesh, py::arg("values"), py::arg("origin"), py::arg("spacing"),
             py::arg("pos") = farray(py::cast(std::vector<float>{0, 0, 0})),
             py::arg("quat") = farray(py::cast(std::vector<float>{0, 0, 0, 1})),
             py::arg("mu_s") = py::none(), py::arg("mu_k") = py::none(),
             "Add a signed-distance grid (values [nx,ny,nz] float32 at origin + ijk*spacing in the object frame, negative inside; see ropesim.sdf) "
             "posed at pos/quat (x,y,z,w). Friction defaults to the contact material's. Needs set_contact() first; at most 32 meshes. Returns the index.")
        .def("set_mesh_friction", &RopeSim::set_mesh_friction, py::arg("index"), py::arg("mu_s"), py::arg("mu_k") = py::none(),
             "Static / kinetic friction of one mesh (mu_k defaults to mu_s).")
        .def("set_mesh_pose", &RopeSim::set_mesh_pose, py::arg("index"), py::arg("pos"), py::arg("quat"),
             py::arg("teleport") = false,
             "Pose the mesh reaches at the END of the next advance() (it moves there over the substeps; friction is relative to it). "
             "teleport=True replaces the pose at once (resets). The last call before a step wins.")
        .def("clear_meshes", &RopeSim::clear_meshes, "Remove every mesh (the material stays).")
        .def("enable_contact_forces", &RopeSim::enable_contact_forces, py::arg("on") = true,
             "Accumulate contact forces during advance(): contact_forces() and mesh_wrenches() become available.")
        .def("contact_forces", &RopeSim::contact_forces, "[V,3] force ON each vertex from every surface, averaged over the last advance().")
        .def("mesh_wrenches", &RopeSim::mesh_wrenches, "[M,6] force and torque (about the mesh origin, world frame) BY the rope ON each mesh, averaged over the last advance().")
        .def_property("drag", &RopeSim::get_drag, &RopeSim::set_drag)
        .def_property("drag_quad", &RopeSim::get_drag_quad, &RopeSim::set_drag_quad)
        .def_property("drag_quad_t", &RopeSim::get_drag_quad_t, &RopeSim::set_drag_quad_t)
        .def("set_velocities", &RopeSim::set_velocities, py::arg("velocities"))
        .def_property("damping", &RopeSim::get_damping, &RopeSim::set_damping)
        .def_property("num_itr", &RopeSim::get_num_itr, &RopeSim::set_num_itr)
        .def_property("accel_ratio", &RopeSim::get_accel_ratio, &RopeSim::set_accel_ratio)
        .def_property("max_h", &RopeSim::get_max_h, &RopeSim::set_max_h)
        .def_property_readonly("gravity", &RopeSim::get_gravity)
        .def_property_readonly("time", &RopeSim::time)
        .def_property_readonly("last_error", &RopeSim::last_error)
        .def_property_readonly("last_warning", &RopeSim::last_warning)
        .def_property_readonly("collisions_enabled", &RopeSim::collisions_enabled)
        .def_property_readonly("fixed_root_orientation", &RopeSim::fixed_root_orientation)
        .def_property_readonly("num_verts", &RopeSim::num_verts);

    py::class_<CudaView>(m, "CudaView")
        .def_property_readonly("__cuda_array_interface__", [](const CudaView& v) {
            py::dict d;
            py::tuple shape(v.shape.size());
            for (size_t i = 0; i < v.shape.size(); i++) shape[i] = v.shape[i];
            d["shape"] = shape;
            d["typestr"] = "<f4";
            d["data"] = py::make_tuple(v.ptr, false);
            d["strides"] = py::none();
            d["version"] = 2;
            return d;
        })
        .def_property_readonly("ptr", [](const CudaView& v) { return v.ptr; })
        .def_property_readonly("shape", [](const CudaView& v) {
            py::tuple t(v.shape.size());
            for (size_t i = 0; i < v.shape.size(); i++) t[i] = v.shape[i];
            return t;
        });

    using ropesim::RopeBatch;
    py::class_<RopeBatch>(m, "RopeBatch")
        .def(py::init([](int num_envs, int verts_per_rope, farray root_pos,
                         float seg_len, farray dir, float density,
                         float k_stretch, float k_bend, float max_h,
                         int num_itr, bool fix_root_orientation) {
                 if (num_envs < 1) throw py::value_error("num_envs must be >= 1");
                 if (verts_per_rope < 3)
                     throw py::value_error("verts_per_rope must be >= 3");
                 if (!(seg_len > 0)) throw py::value_error("seg_len must be > 0");
                 if (!(density > 0)) throw py::value_error("density must be > 0");
                 if (!(k_stretch > 0)) throw py::value_error("k_stretch must be > 0");
                 if (!(k_bend > 0))
                     throw py::value_error("k_bend must be > 0 (0 -> NaN, see RopeSim)");
                 if (!(max_h > 0)) throw py::value_error("max_h must be > 0");
                 if (num_itr < 1) throw py::value_error("num_itr must be >= 1");
                 if (root_pos.ndim() != 2 || root_pos.shape(0) != num_envs ||
                     root_pos.shape(1) != 3)
                     throw py::value_error("root_pos must be a host [num_envs, 3] array");
                 glm::vec3 d3 = to_vec3(dir, "dir");
                 if (!(glm::length(d3) > 0)) throw py::value_error("dir must be nonzero");
                 return new RopeBatch(num_envs, verts_per_rope, seg_len,
                                      root_pos.data(), &d3.x, density, k_stretch,
                                      k_bend, max_h, num_itr, fix_root_orientation);
             }),
             py::arg("num_envs"), py::arg("verts_per_rope"),
             py::arg("root_pos"),  // host [num_envs,3], required
             py::arg("seg_len") = 0.02f,
             py::arg("dir") = farray(py::cast(std::vector<float>{0, 0, -1})),
             py::arg("density") = 0.1f, py::arg("k_stretch") = 500.0f,
             py::arg("k_bend") = 1e-6f, py::arg("max_h") = 1.f / 3000.f,
             py::arg("num_itr") = 8, py::arg("fix_root_orientation") = true)
        .def("step",
             [](RopeBatch& b, py::object root_pos, py::object root_vel, float dt,
                int substeps, py::object root_q, py::object root_pos_end,
                py::object root_q_end) {
                 if (!(dt > 0)) throw py::value_error("dt must be > 0");
                 if (substeps < 1) throw py::value_error("substeps must be >= 1");
                 if (!b.rootsPinned)
                     throw py::value_error("step() drives every rope's root; ropes built with a free root "
                                           "(from_positions pinned=...) advance with step_free()");
                 auto p = parse_cai(root_pos, {b.num_envs(), 3}, "root_pos");
                 auto v = parse_cai(root_vel, {b.num_envs(), 3}, "root_vel");
                 const float* qp = nullptr;
                 CaiArray q;
                 if (!root_q.is_none()) {
                     if (!b.fix_root_orientation())
                         throw py::value_error(
                             "root_q requires fix_root_orientation=True");
                     q = parse_cai(root_q, {b.num_envs(), 4}, "root_q");
                     qp = reinterpret_cast<const float*>(q.ptr);
                 }
                 if (root_pos_end.is_none()) {
                     // legacy: constant-velocity sweep, root_q held
                     if (!root_q_end.is_none())
                         throw py::value_error(
                             "root_q_end requires root_pos_end (the motion-"
                             "program path)");
                     b.stepped = true;  // per-vertex setters now illegal
                     py::gil_scoped_release rel;
                     b.step(reinterpret_cast<const float*>(p.ptr),
                            reinterpret_cast<const float*>(v.ptr), qp, dt,
                            substeps);
                     return;
                 }
                 // second-order program: root_vel is the MEASURED root-point
                 // velocity at frame start; the quadratic lands on
                 // root_pos_end; orientation SLERPs root_q -> root_q_end.
                 auto p1 = parse_cai(root_pos_end, {b.num_envs(), 3},
                                     "root_pos_end");
                 const float* q1p = nullptr;
                 CaiArray q1;
                 if (!root_q_end.is_none()) {
                     if (qp == nullptr)
                         throw py::value_error("root_q_end requires root_q");
                     q1 = parse_cai(root_q_end, {b.num_envs(), 4}, "root_q_end");
                     q1p = reinterpret_cast<const float*>(q1.ptr);
                 }
                 b.stepped = true;  // per-vertex setters now illegal
                 py::gil_scoped_release rel;
                 b.step_program(reinterpret_cast<const float*>(p.ptr),
                                reinterpret_cast<const float*>(v.ptr),
                                reinterpret_cast<const float*>(p1.ptr),
                                qp, q1p, dt, substeps);
             },
             py::arg("root_pos"), py::arg("root_vel"), py::arg("dt"),
             py::arg("substeps"), py::arg("root_q") = py::none(),
             py::arg("root_pos_end") = py::none(),
             py::arg("root_q_end") = py::none())
        .def("reset_envs",
             [](RopeBatch& b, py::array_t<int, py::array::c_style | py::array::forcecast> ids) {
                 if (ids.ndim() != 1) throw py::value_error("ids must be 1-D");
                 // d_ids is sized for num_envs ints; more (only possible via
                 // duplicates) would overflow the device buffer
                 if (ids.shape(0) > b.num_envs())
                     throw py::value_error("ids longer than num_envs");
                 auto r = ids.unchecked<1>();
                 std::vector<int> v((size_t)r.shape(0));
                 for (py::ssize_t i = 0; i < r.shape(0); i++) {
                     if (r(i) < 0 || r(i) >= b.num_envs())
                         throw py::value_error("env id out of range");
                     v[(size_t)i] = r(i);
                 }
                 py::gil_scoped_release rel;
                 b.reset_envs(v.data(), (int)v.size());
             },
             py::arg("ids"))
        .def("gather", &RopeBatch::gather,
             py::call_guard<py::gil_scoped_release>())
        .def("positions",
             [](RopeBatch& b) {
                 return CudaView{b.positions_ptr(),
                                 {b.num_envs(), b.verts_per_rope(), 3}};
             }, py::keep_alive<0, 1>())
        .def("velocities",
             [](RopeBatch& b) {
                 return CudaView{b.velocities_ptr(),
                                 {b.num_envs(), b.verts_per_rope(), 3}};
             }, py::keep_alive<0, 1>())
        .def("quaternions",
             [](RopeBatch& b) {
                 return CudaView{b.quaternions_ptr(),
                                 {b.num_envs(), b.verts_per_rope(), 4}};
             }, py::keep_alive<0, 1>())
        .def_property("drag",
             [](RopeBatch& b) { std::lock_guard<std::mutex> g(b.mutex()); return b.raw()->meta.drag; },
             [](RopeBatch& b, float v) { std::lock_guard<std::mutex> g(b.mutex()); b.raw()->meta.drag = v; })
        .def_property("drag_quad",
             [](RopeBatch& b) { std::lock_guard<std::mutex> g(b.mutex()); return b.raw()->meta.dragQuad; },
             [](RopeBatch& b, float v) {
                 if (!std::isfinite(v) || v < 0) throw py::value_error("drag_quad must be finite and >= 0");
                 std::lock_guard<std::mutex> g(b.mutex());
                 b.raw()->meta.dragQuad = v;
             })
        .def_property("drag_quad_t",
             [](RopeBatch& b) { std::lock_guard<std::mutex> g(b.mutex()); return b.raw()->meta.dragQuadT; },
             [](RopeBatch& b, float v) {
                 if (!std::isfinite(v) || v < 0) throw py::value_error("drag_quad_t must be finite and >= 0");
                 std::lock_guard<std::mutex> g(b.mutex());
                 b.raw()->meta.dragQuadT = v;
             })
        .def_property("damping",
             [](RopeBatch& b) { std::lock_guard<std::mutex> g(b.mutex()); return b.raw()->meta.damping; },
             [](RopeBatch& b, float v) { std::lock_guard<std::mutex> g(b.mutex()); b.raw()->meta.damping = v; })
        .def_property("num_itr",
             [](RopeBatch& b) { std::lock_guard<std::mutex> g(b.mutex()); return b.raw()->meta.numItr; },
             [](RopeBatch& b, int v) {
                 if (v < 1) throw py::value_error("num_itr must be >= 1");
                 std::lock_guard<std::mutex> g(b.mutex());
                 b.raw()->meta.numItr = v;
             })
        // Build from arbitrary initial shapes [N, V, 3]. pinned: optional bool
        // [N, V] (or [V] for every rope) of kinematic vertices; default = each root.
        .def_static("from_positions",
             [](farray pos, float density, float k_stretch, float k_bend, float max_h, int num_itr,
                bool fix_root_orientation, py::object pinned) {
                 if (pos.ndim() != 3 || pos.shape(2) != 3 || pos.shape(1) < 3)
                     throw py::value_error("positions must be [num_envs, V>=3, 3]");
                 const int N = (int)pos.shape(0), V = (int)pos.shape(1);
                 if (!(density > 0) || !(k_stretch > 0) || !(k_bend > 0) || !(max_h > 0) || num_itr < 1)
                     throw py::value_error("density, k_stretch, k_bend, max_h > 0 and num_itr >= 1 required");
                 std::vector<uint8_t> pin((size_t)N * V, 0);
                 if (pinned.is_none()) {
                     for (int e = 0; e < N; e++) pin[(size_t)e * V] = 1;
                 } else {
                     auto pm = pinned.cast<py::array_t<bool, py::array::c_style | py::array::forcecast>>();
                     const bool per_env = pm.ndim() == 2;
                     if (!((pm.ndim() == 1 && pm.shape(0) == V) || (per_env && pm.shape(0) == N && pm.shape(1) == V)))
                         throw py::value_error("pinned must be [V] or [num_envs, V] booleans");
                     const bool* d = pm.data();
                     for (int e = 0; e < N; e++)
                         for (int v = 0; v < V; v++) pin[(size_t)e * V + v] = d[per_env ? (size_t)e * V + v : (size_t)v];
                 }
                 if (fix_root_orientation)
                     for (int e = 0; e < N; e++)
                         if (!pin[(size_t)e * V]) throw py::value_error("fix_root_orientation=True needs every root pinned");
                 std::vector<float> roots((size_t)N * 3);
                 auto r = pos.unchecked<3>();
                 for (int e = 0; e < N; e++)
                     for (int v = 0; v < V; v++) {
                         const glm::vec3 a(r(e, v, 0), r(e, v, 1), r(e, v, 2));
                         if (!std::isfinite(a.x) || !std::isfinite(a.y) || !std::isfinite(a.z))
                             throw py::value_error("positions must be finite");
                         if (v + 1 < V && !(glm::length(glm::vec3(r(e, v + 1, 0), r(e, v + 1, 1), r(e, v + 1, 2)) - a) > 0))
                             throw py::value_error("consecutive vertices must not coincide");
                     }
                 for (int e = 0; e < N; e++) for (int k = 0; k < 3; k++) roots[(size_t)e * 3 + k] = r(e, 0, k);
                 float dir3[3] = {1.f, 0.f, 0.f};
                 return std::unique_ptr<RopeBatch>(new RopeBatch(N, V, 0.f, roots.data(), dir3, density, k_stretch,
                                                                 k_bend, max_h, num_itr, fix_root_orientation, pos.data(),
                                                                 pin.data()));
             }, py::arg("positions"), py::arg("density"), py::arg("k_stretch"), py::arg("k_bend"),
             py::arg("max_h"), py::arg("num_itr"), py::arg("fix_root_orientation") = false,
             py::arg("pinned") = py::none())
        // Rope-rope contact: YarnBall's barrier/friction/step-limit model, fed by a
        // per-group brute-force detector. Contact starts at centreline distance
        // 2*radius + barrier (use radius 0.9 R, barrier 0.2 R for a rope of radius R).
        // group_ropes consecutive ropes share contact (one env); others never interact.
        .def("set_rope_contact",
             [](RopeBatch& b, bool enabled, float radius, float barrier, float k_collision, float mu,
                float k_friction, float detection_scaler, int group_ropes) {
                 if (enabled && (!(radius > 0) || !(barrier > 0) || !(k_collision > 0) || !(mu >= 0) ||
                                 !(k_friction >= 0) || !(detection_scaler > 1) || group_ropes < 1))
                     throw py::value_error("rope contact needs radius, barrier, k_collision > 0, mu, k_friction >= 0, "
                                           "detection_scaler > 1, group_ropes >= 1");
                 b.set_rope_contact(enabled, radius, barrier, k_collision, mu, k_friction, detection_scaler, group_ropes);
             },
             py::arg("enabled"), py::arg("radius") = 0.f, py::arg("barrier") = 0.f, py::arg("k_collision") = 0.f,
             py::arg("mu") = 0.f, py::arg("k_friction") = 0.f, py::arg("detection_scaler") = 1.2f,
             py::arg("group_ropes") = 1)
        .def("step_free",
             [](RopeBatch& b, float dt, int substeps) {
                 if (!(dt > 0) || substeps < 1) throw py::value_error("dt > 0 and substeps >= 1 required");
                 b.stepped = true;
                 py::gil_scoped_release rel;
                 b.step_free(dt, substeps);
             }, py::arg("dt"), py::arg("substeps"))
        .def("set_velocities",
             [](RopeBatch& b, farray v) {
                 // Overwrite all vertex velocities [N, V, 3]. A pinned vertex's row is its
                 // kinematic velocity (it advects pos += h*v): pass zeros to hold it.
                 const int N = b.num_envs(), V = b.verts_per_rope();
                 if (v.ndim() != 3 || v.shape(0) != N || v.shape(1) != V || v.shape(2) != 3)
                     throw py::value_error("expected [num_envs, V, 3]");
                 auto r = v.unchecked<3>();
                 std::lock_guard<std::mutex> g(b.mutex());
                 auto* s = b.raw(); s->download();
                 for (int e = 0; e < N; e++)
                     for (int i = 0; i < V; i++)
                         s->vels[(size_t)e * V + i] = glm::vec3(r(e, i, 0), r(e, i, 1), r(e, i, 2));
                 s->upload();
             }, py::arg("velocities"))
        .def("set_planes",
             [](RopeBatch& b, farray planes, float radius, float mu_s, py::object mu_k, float v_slip, float kappa) {
                 std::lock_guard<std::mutex> g(b.mutex());
                 write_planes(b.raw()->meta, planes, radius, mu_s, mu_k.is_none() ? mu_s : mu_k.cast<float>(), v_slip, kappa);
             },
             py::arg("planes"), py::arg("radius") = 0.f, py::arg("mu_s") = 0.5f, py::arg("mu_k") = py::none(),
             py::arg("v_slip") = 1e-3f, py::arg("kappa") = 2.f)
        .def("set_contact",
             [](RopeBatch& b, float radius, float mu_s, py::object mu_k, float v_slip, float kappa) {
                 std::lock_guard<std::mutex> g(b.mutex());
                 write_contact(b.raw()->meta, radius, mu_s, mu_k.is_none() ? mu_s : mu_k.cast<float>(), v_slip, kappa);
             },
             py::arg("radius"), py::arg("mu_s") = 0.5f, py::arg("mu_k") = py::none(),
             py::arg("v_slip") = 1e-3f, py::arg("kappa") = 2.f)
        // Rod-mesh contact (sdf.h): one SDF grid per mesh, posed per env.
        .def("add_mesh",
             [](RopeBatch& b, farray values, farray origin, float spacing, farray pos, farray quat, py::object mu_s, py::object mu_k) {
                 const GridArgs ga = parse_grid(values, origin, spacing);
                 const int rows = parse_poses(pos, quat, b.num_envs(), "add_mesh");
                 std::lock_guard<std::mutex> g(b.mutex());
                 auto& m = b.raw()->meta;
                 require_contact_radius(m);
                 const float ms = mu_or(mu_s, m.contactMuS), mk = mu_or(mu_k, mu_s.is_none() ? m.contactMuK : ms);
                 if (!(ms >= 0) || !(mk >= 0)) throw py::value_error("mu_s / mu_k must be >= 0");
                 const int idx = b.sdf.add(ga.v, ga.nx, ga.ny, ga.nz, ga.origin, ga.spacing, ms, mk, pos.data(), quat.data(), rows,
                                           b.raw()->solverStream());
                 b.sdf.write_meta(m, b.verts_per_rope());
                 return idx;
             },
             py::arg("values"), py::arg("origin"), py::arg("spacing"),
             py::arg("pos") = farray(py::cast(std::vector<float>{0, 0, 0})),
             py::arg("quat") = farray(py::cast(std::vector<float>{0, 0, 0, 1})),
             py::arg("mu_s") = py::none(), py::arg("mu_k") = py::none(),
             "Add a signed-distance grid (values [nx,ny,nz] float32 at origin + ijk*spacing in the object frame, negative inside; see ropesim.sdf) "
             "posed at pos/quat (x,y,z,w; [3]/[4] for every env or [N,3]/[N,4] per env). Friction defaults to the contact material's. "
             "Needs set_contact() first; at most 32 meshes. Returns the index.")
        .def("set_mesh_friction",
             [](RopeBatch& b, int index, float mu_s, py::object mu_k) {
                 if (!(mu_s >= 0)) throw py::value_error("mu_s must be >= 0");
                 const float mk = mu_or(mu_k, mu_s);
                 if (!(mk >= 0)) throw py::value_error("mu_k must be >= 0");
                 std::lock_guard<std::mutex> g(b.mutex());
                 b.sdf.set_friction(index, mu_s, mk);
                 b.sdf.write_meta(b.raw()->meta, b.verts_per_rope());
             }, py::arg("index"), py::arg("mu_s"), py::arg("mu_k") = py::none(), "Static / kinetic friction of one mesh (mu_k defaults to mu_s).")
        // teleport=False: the pose each env's mesh will have at the END of the next
        // step (it moves there over the substeps; friction is relative to it);
        // True / [N] bool (host or CUDA): replaced now (resets). pos/quat: host
        // [3]/[4] or [N,3]/[N,4], or CUDA float32 [N,3]/[N,4] (no host round trip).
        .def("set_mesh_poses",
             [](RopeBatch& b, int index, py::object pos, py::object quat, py::object teleport) {
                 const PoseRows r = parse_pose_rows(pos, quat, b.num_envs(), "set_mesh_poses");
                 const TeleFlags t = parse_teleport(teleport, b.num_envs(), "set_mesh_poses");
                 std::lock_guard<std::mutex> g(b.mutex());
                 set_mesh_targets(b.sdf, index, r, t, b.raw()->solverStream(), "set_mesh_poses");
             }, py::arg("index"), py::arg("pos"), py::arg("quat"), py::arg("teleport") = false,
             "Pose of one mesh at the END of the next step, per env or shared (host [3]/[4], [N,3]/[N,4] or CUDA [N,3]/[N,4] float32); "
             "teleport: bool, [N] bool array or CUDA bool array replaces the pose at once (resets). The last call before a step wins.")
        // Every mesh at once: pos [N, M, 3], quat [N, M, 4] (M = the mesh count; host or
        // CUDA float32), teleport as in set_mesh_poses. One call per frame for the Isaac bridge.
        .def("set_mesh_poses_all",
             [](RopeBatch& b, py::object pos, py::object quat, py::object teleport) {
                 const int N = b.num_envs(), M = b.sdf.count();
                 const TeleFlags t = parse_teleport(teleport, N, "set_mesh_poses_all");
                 const bool dp = py::hasattr(pos, "__cuda_array_interface__"), dq = py::hasattr(quat, "__cuda_array_interface__");
                 if (dp != dq) throw py::value_error("set_mesh_poses_all: pos and quat must both be host arrays or both CUDA arrays");
                 PoseRows r;
                 if (dp) {
                     r.device = true; r.rows = N * M;
                     r.pos = reinterpret_cast<const float*>(parse_cai(pos, {N, M, 3}, "pos").ptr);
                     r.quat = reinterpret_cast<const float*>(parse_cai(quat, {N, M, 4}, "quat").ptr);
                 } else {
                     r.hp = farray::ensure(pos); r.hq = farray::ensure(quat);
                     if (!r.hp || !r.hq || r.hp.ndim() != 3 || r.hq.ndim() != 3 || r.hp.shape(0) != N || r.hq.shape(0) != N ||
                         r.hp.shape(1) != M || r.hq.shape(1) != M || r.hp.shape(2) != 3 || r.hq.shape(2) != 4)
                         throw py::value_error("set_mesh_poses_all: pos must be [num_envs, num_meshes, 3] and quat [num_envs, num_meshes, 4]");
                     r.rows = N * M; r.pos = r.hp.data(); r.quat = r.hq.data();
                     validate_rows(r.pos, r.quat, r.rows, "set_mesh_poses_all");
                 }
                 std::lock_guard<std::mutex> g(b.mutex());
                 set_mesh_targets(b.sdf, -1, r, t, b.raw()->solverStream(), "set_mesh_poses_all");
             }, py::arg("pos"), py::arg("quat"), py::arg("teleport") = false,
             "Every mesh's pose at the END of the next step: pos [N,M,3], quat [N,M,4] (host or CUDA float32, M = mesh count), "
             "teleport as in set_mesh_poses. One call per frame is what the Isaac bridge does.")
        .def("clear_meshes",
             [](RopeBatch& b) {
                 std::lock_guard<std::mutex> g(b.mutex());
                 b.sdf.clear();
                 b.sdf.write_meta(b.raw()->meta, b.verts_per_rope());
             }, "Remove every mesh (the material stays).")
        // Contact-force readout (planes + meshes), averaged over the last step.
        .def("enable_contact_forces",
             [](RopeBatch& b, bool on) {
                 std::lock_guard<std::mutex> g(b.mutex());
                 b.sdf.enable_forces(on);
                 b.sdf.write_meta(b.raw()->meta, b.verts_per_rope());
             }, py::arg("on") = true,
             "Accumulate contact forces during steps: contact_forces(_view) and mesh_wrenches(_view) become available (+2-5% cost).")
        .def("contact_forces_view",   // device [N, V, 3] behind contact_forces() (valid until the next step)
             [](RopeBatch& b) {
                 std::lock_guard<std::mutex> g(b.mutex());
                 if (!b.sdf.forces_enabled()) throw py::value_error("enable_contact_forces(True) first");
                 return CudaView{b.sdf.contact_forces_ptr(), {b.num_envs(), b.verts_per_rope(), 3}};
             }, py::keep_alive<0, 1>(), "Device view [N,V,3] of contact_forces() (valid until the next step; torch.as_tensor shares it).")
        .def("mesh_wrenches_view",    // device [N, 32, 6] behind mesh_wrenches() (slots >= the mesh count are zero; valid until the next step)
             [](RopeBatch& b) {
                 std::lock_guard<std::mutex> g(b.mutex());
                 if (!b.sdf.forces_enabled()) throw py::value_error("enable_contact_forces(True) first");
                 return CudaView{b.sdf.wrenches_ptr(), {b.num_envs(), SdfStore::MAX, 6}};
             }, py::keep_alive<0, 1>(), "Device view [N,32,6] of mesh_wrenches() (slots beyond the mesh count are zero; valid until the next step).")
        .def("contact_forces",   // [N, V, 3]: force ON each vertex (world)
             [](RopeBatch& b) {
                 std::lock_guard<std::mutex> g(b.mutex());
                 py::array_t<float> out({b.num_envs(), b.verts_per_rope(), 3});
                 b.sdf.contact_forces(out.mutable_data());
                 return out;
             }, "[N,V,3] force ON each vertex from every surface, averaged over the last step (host copy).")
        .def("mesh_wrenches",    // [N, M, 6]: (force, torque about the mesh origin) BY the rope ON each mesh (world)
             [](RopeBatch& b) {
                 std::lock_guard<std::mutex> g(b.mutex());
                 py::array_t<float> out({b.num_envs(), b.sdf.count(), 6});
                 b.sdf.wrenches(out.mutable_data());
                 return out;
             }, "[N,M,6] force and torque (about the mesh origin, world frame) BY the rope ON each mesh, averaged over the last step (host copy).")
        .def("set_gravity",
             [](RopeBatch& b, farray gv) {
                 glm::vec3 g3 = to_vec3(gv, "gravity");
                 std::lock_guard<std::mutex> g(b.mutex());
                 b.raw()->meta.gravity = g3;
             }, py::arg("gravity"))
        // ------- per-vertex property API (2026-08-24, e6 non-uniform ropes) -------
        // Legal ONLY between construction and the first step (batch.h `stepped`):
        // in that window the host mirror is authoritative, so the RopeSim
        // download-mutate-upload pattern is safe. Arrays broadcast: 1-D applies
        // the same profile to every env; 2-D is per-env. Mirrors the RopeSim
        // setters (kStretch = EA*lRest; qRest = (4*EI/lRest)*normalize(qRest)).
        .def("inv_masses",
             [](RopeBatch& b) {
                 std::lock_guard<std::mutex> g(b.mutex());
                 b.raw()->download();
                 const int N = b.num_envs(), V = b.verts_per_rope();
                 py::array_t<float> arr({N, V});
                 auto o = arr.mutable_unchecked<2>();
                 for (int e = 0; e < N; e++)
                     for (int v = 0; v < V; v++)
                         o(e, v) = b.raw()->verts[(size_t)e * V + v].invMass;
                 return arr;
             })
        .def("set_masses",
             [](RopeBatch& b, farray m) {
                 if (b.stepped)
                     throw py::value_error(
                         "set_masses is only legal before the first step");
                 const int N = b.num_envs(), V = b.verts_per_rope();
                 const bool per_env = m.ndim() == 2;
                 if (!((m.ndim() == 1 && m.shape(0) == V) ||
                       (per_env && m.shape(0) == N && m.shape(1) == V)))
                     throw py::value_error("expected [V] or [num_envs, V] node masses (kg)");
                 auto get = [&](int e, int v) {
                     return per_env ? m.unchecked<2>()(e, v) : m.unchecked<1>()(v);
                 };
                 std::lock_guard<std::mutex> g(b.mutex());
                 b.raw()->download();
                 // pinned vertices (invMass 0: each root, or from_positions' pinned
                 // mask) stay pinned and their column is ignored
                 auto pinned = [&](int e, int v) { return b.raw()->verts[(size_t)e * V + v].invMass == 0; };
                 for (int e = 0; e < N; e++)
                     for (int v = 0; v < V; v++)
                         if (!pinned(e, v) && (!std::isfinite(get(e, v)) || !(get(e, v) > 0)))
                             throw py::value_error("node masses must be finite and > 0 (pinned columns ignored)");
                 for (int e = 0; e < N; e++)
                     for (int v = 0; v < V; v++)
                         if (!pinned(e, v))
                             b.raw()->verts[(size_t)e * V + v].invMass = 1.f / get(e, v);
                 b.raw()->upload();
             }, py::arg("masses"))
        .def("set_stretch_stiffnesses",
             [](RopeBatch& b, farray ea) {
                 if (b.stepped)
                     throw py::value_error(
                         "set_stretch_stiffnesses is only legal before the first step");
                 const int N = b.num_envs(), V = b.verts_per_rope();
                 const bool per_env = ea.ndim() == 2;
                 if (!((ea.ndim() == 1 && ea.shape(0) == V - 1) ||
                       (per_env && ea.shape(0) == N && ea.shape(1) == V - 1)))
                     throw py::value_error("expected [V-1] or [num_envs, V-1] EA values");
                 auto get = [&](int e, int v) {
                     return per_env ? ea.unchecked<2>()(e, v) : ea.unchecked<1>()(v);
                 };
                 for (int e = 0; e < (per_env ? N : 1); e++)
                     for (int v = 0; v < V - 1; v++)
                         if (!std::isfinite(get(e, v)) || !(get(e, v) > 0))
                             throw py::value_error("EA values must be finite and > 0");
                 std::lock_guard<std::mutex> g(b.mutex());
                 b.raw()->download();
                 for (int e = 0; e < N; e++)
                     for (int v = 0; v < V - 1; v++) {
                         const size_t i = (size_t)e * V + v;
                         b.raw()->verts[i].kStretch = get(e, v) * b.raw()->verts[i].lRest;
                     }
                 b.raw()->upload();
             }, py::arg("ea"))
        .def("set_bend_stiffnesses",
             [](RopeBatch& b, farray ei) {
                 if (b.stepped)
                     throw py::value_error(
                         "set_bend_stiffnesses is only legal before the first step");
                 const int N = b.num_envs(), V = b.verts_per_rope();
                 const bool per_env = ei.ndim() == 2;
                 if (!((ei.ndim() == 1 && ei.shape(0) == V - 1) ||
                       (per_env && ei.shape(0) == N && ei.shape(1) == V - 1)))
                     throw py::value_error("expected [V-1] or [num_envs, V-1] EI values");
                 auto get = [&](int e, int v) {
                     return per_env ? ei.unchecked<2>()(e, v) : ei.unchecked<1>()(v);
                 };
                 for (int e = 0; e < (per_env ? N : 1); e++)
                     for (int v = 0; v < V - 1; v++)
                         if (!std::isfinite(get(e, v)) || !(get(e, v) > 0))
                             throw py::value_error("EI values must be finite and > 0 (0 -> NaN, see RopeSim)");
                 std::lock_guard<std::mutex> g(b.mutex());
                 b.raw()->download();
                 for (int e = 0; e < N; e++)
                     for (int v = 0; v < V - 1; v++) {
                         const size_t i = (size_t)e * V + v;
                         const float k = 4.f * get(e, v) / b.raw()->verts[i].lRest;
                         b.raw()->qRests[i] = k * glm::normalize((glm::vec4)b.raw()->qRests[i]);
                     }
                 b.raw()->upload();
             }, py::arg("ei"))
        .def_property_readonly("time",
             [](RopeBatch& b) { std::lock_guard<std::mutex> g(b.mutex()); return b.raw()->meta.time; })
        .def_property_readonly("last_error",
             [](RopeBatch& b) { std::lock_guard<std::mutex> g(b.mutex()); return b.raw()->lastErrorCode; })
        .def_property_readonly("last_warning",
             [](RopeBatch& b) { std::lock_guard<std::mutex> g(b.mutex()); return b.raw()->lastWarningCode; })
        .def_property_readonly("num_envs", &RopeBatch::num_envs)
        .def_property_readonly("verts_per_rope", &RopeBatch::verts_per_rope)
        .def_property_readonly("fixed_root_orientation", &RopeBatch::fix_root_orientation);
}
