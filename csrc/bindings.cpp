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
#include <optional>
#include <vector>

#include "YarnBall/YarnBall.h"
#include "batch.h"

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

// Parsed view of a device array handed to us by torch/cupy/numba.
struct CaiArray {
    uintptr_t ptr = 0;
};

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
                 for (int e = 0; e < (per_env ? N : 1); e++)
                     for (int v = 1; v < V; v++)
                         if (!std::isfinite(get(e, v)) || !(get(e, v) > 0))
                             throw py::value_error("node masses must be finite and > 0 (root column ignored)");
                 std::lock_guard<std::mutex> g(b.mutex());
                 b.raw()->download();
                 for (int e = 0; e < N; e++)
                     for (int v = 1; v < V; v++)   // v=0 is the pinned root: invMass stays 0
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
