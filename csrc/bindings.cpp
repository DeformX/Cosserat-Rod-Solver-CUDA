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

#include <cmath>
#include <memory>
#include <mutex>
#include <vector>

#include "YarnBall/YarnBall.h"

namespace py = pybind11;

namespace {

using farray = py::array_t<float, py::array::c_style | py::array::forcecast>;

glm::vec3 to_vec3(const farray& a, const char* name) {
    if (a.ndim() != 1 || a.shape(0) != 3)
        throw py::value_error(std::string(name) + " must be a length-3 array");
    auto b = a.unchecked<1>();
    return glm::vec3(b(0), b(1), b(2));
}

class RopeSim {
    std::unique_ptr<YarnBall::Sim> sim;  // RAII: ctor throws must not leak
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
        std::lock_guard<std::mutex> g(mtx_);
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
        } else {
            sim->advance(dt);
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

    // --- meta knobs (plain field writes; re-uploaded by the next advance) ---
    float get_drag() const { std::lock_guard<std::mutex> g(mtx_); return sim->meta.drag; }
    void set_drag(float v) { std::lock_guard<std::mutex> g(mtx_); sim->meta.drag = v; }
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
        .def_property("drag", &RopeSim::get_drag, &RopeSim::set_drag)
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
}
