# Cosserat-Rod-Solver-CUDA

A GPU Cosserat rod solver for ropes, cables and wires, packaged as the Python module
`ropesim`. It builds on **Stable Cosserat Rods** (Hsu, Wang, Wu and Yuksel, SIGGRAPH 2025;
the [YarnBall](https://github.com/jerry060599/YarnBall) solver) and adds what a robot-learning
simulator needs: thousands of ropes in one solver, kinematic root driving for a robot end
effector, contact with planes, meshes and other ropes, and a bridge that co-simulates with
PhysX inside NVIDIA Isaac Lab. It is the GPU engine of
[DeformX](https://github.com/DeformX/DeformX).

### [🌐 DeformX project page](https://deformx.github.io/) | [📄 DeformX paper](https://arxiv.org/abs/2606.22116) | [Stable Cosserat Rods](https://graphics.cs.utah.edu/research/projects/stable-cosserat-rods/)

## What is inside

| | |
|---|---|
| `RopeSim` | One rope with a host-driven root: the validation and experimentation API |
| `RopeBatch` | N ropes in one flat solver. Roots are driven from CUDA tensors every control frame by a second-order motion program (start velocity + frame-end pose, SLERP orientation); positions, velocities and frames come back as zero-copy torch views; vectorized resets; per-vertex mass and stiffness |
| Contact (opt-in) | Half-space planes; rope-rope contact; meshes as signed-distance grids (baked with Warp or from closed forms), including kinematic meshes whose pose is set every frame; per-vertex contact forces and per-mesh wrenches; up to 64 meshes with a per-vertex broad phase |
| `ropesim.isaac_bridge` | PhysX rigid bodies and articulation links become rope colliders; the rope's wrench goes back to PhysX as an external force (explicit coupling). Its demos, and the robot-arm RL task, live in [DeformX](https://github.com/DeformX/DeformX) |
| Air drag | Linear decay plus quadratic cross-flow drag split along the segment tangent |
| `tests/` | Analytic checks for stretch, bending, dynamics and contact, plus deterministic replay and example CLI checks. See the measured baselines in [tests/README.md](tests/README.md). |

The solver source is vendored under `third_party/yarnball/` (headless subset, GPL-3.0,
local patches listed in [VENDOR.md](third_party/yarnball/VENDOR.md)); `csrc/` holds the
pybind11 bindings and the batched-driving, contact-detection and signed-distance kernels;
`ropesim/` is the Python package (`sdf.py` grids, `isaac_bridge.py`), `tests/` the
validation suite, `examples/` runnable scripts and `docs/` the solver notes.

## Requirements

- Linux, an NVIDIA GPU, and a CUDA toolkit that matches the torch build you use
  (torch `cu128` ↔ CUDA 12.8). CMake ≥ 3.24, Ninja, a C++17 compiler, OpenMP, and the
  header-only [GLM](https://github.com/g-truc/glm). The release candidate was tested with
  GLM 0.9.9.8. Set `GLM_INCLUDE_DIR` to the directory containing `glm/glm.hpp`.
- Python ≥ 3.10 with `numpy`; `torch` (CUDA build) to use `RopeBatch` tensors; optional
  `warp-lang` to bake triangle meshes (`pip install -e ".[bake]"`); Isaac Lab 2.x for the bridge.

## Install

Set the toolkit path to the CUDA installation that matches PyTorch. Add its runtime
libraries to the loader path before importing `ropesim`. This CUDA 12.8 example installs
the matching PyTorch build, which the test suite requires.

```bash
export CUDA_TOOLKIT_ROOT=/usr/local/cuda-12.8
export CUDACXX="$CUDA_TOOLKIT_ROOT/bin/nvcc"
export LD_LIBRARY_PATH="$CUDA_TOOLKIT_ROOT/lib64:$CUDA_TOOLKIT_ROOT/targets/x86_64-linux/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export GLM_INCLUDE_DIR=/path/to/glm-0.9.9.8
python -m pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e ".[dev,bake]" trimesh scipy
python -c 'import ropesim; import torch; print(ropesim._ropesim.__file__); print(torch.cuda.get_device_name())'
python -m pytest  # full suite takes a few minutes on one GPU
```

The `bake` extra installs Warp. The mesh and showcase checks use Warp, Trimesh and
SciPy. To omit these optional dependencies, install `.[dev]` instead. Tests that
need the optional packages will skip when they are absent.

The GPU architecture defaults to the build machine's GPU. For a binary that runs on several
GPUs, or on a machine without one at build time:

```bash
CMAKE_ARGS="-DCMAKE_CUDA_ARCHITECTURES=89;120" pip install -e .
```

`GLM_INCLUDE_DIR` and `CUDAToolkit_ROOT` can be given the same way (`-D...`) or as
environment variables. Builds go to `build/`.

## A tour of the API

Units everywhere: `density` ρ [kg/m], `k_stretch` EA [N], `k_bend` EI [N·m²] (must be > 0),
`drag` [1/s], `drag_quad` [1/m]. Keep `damping` at 0 (see [docs/SOLVER.md](docs/SOLVER.md)).

```python
import numpy as np, torch, ropesim
from ropesim import sdf

# one rope, host-driven root ----------------------------------------------------------------
r = ropesim.RopeSim(num_verts=33, seg_len=0.03, dir=[0, 0, -1], density=0.1,
                    k_stretch=500.0, k_bend=1e-4, max_h=1/3000, num_itr=8)
r.set_gravity([0, 0, -9.81]); r.drag = 0.2
r.set_root([0, 0, 1.0], [0.1, 0, 0])            # root pose and velocity for the next frame
r.advance(1/60)                                 # 50 substeps of 1/3000 s
P = np.asarray(r.positions())                   # [V, 3]

# N ropes in one solver, roots driven from CUDA tensors ------------------------------------
pos = ...                                       # [N, V, 3] float32 initial vertex positions
pin = np.zeros(pos.shape[:2], bool); pin[:, 0] = True
b = ropesim.RopeBatch.from_positions(pos, 0.117, 11473.0, 4.48e-3, 1/6000, 8, pinned=pin)
b.set_gravity(np.array([0, 0, -9.81]))
b.step(p0, v0, 1/60, 100, root_pos_end=p1)      # p0/v0/p1: [N, 3] CUDA tensors; quadratic from (p0, v0) to p1
X = torch.as_tensor(b.positions())              # [N, V, 3], shares the solver's memory
b.reset_envs(torch.tensor([3, 7]))              # back to the construction state

# contact -------------------------------------------------------------------------------------
b.set_planes(np.array([[0, 0, 1.0, 0.0]], np.float32), radius=0.00635, mu_s=0.5, mu_k=0.4)   # the floor n.x >= w; sets the shared surface material
b.set_contact(radius=0.00635, mu_s=0.5, mu_k=0.4)                       # (or set the material directly, for meshes without planes)
grid = sdf.from_function(sdf.sdf_box((0.2, 0.2, 0.1), round=0.02), (-0.26,) * 3, (0.26,) * 3, 0.01)
i = b.add_mesh(*grid.as_args(), pos=np.array([0, 0, 0.1], np.float32))  # or per env: pos [N, 3], quat [N, 4]
b.set_mesh_poses(i, pos_t, quat_t)              # the pose at the end of the next step (kinematic mesh)
b.set_rope_contact(True, 0.9 * R, 0.2 * R, 0.025, 0.4, 1e7, 1.2, group_ropes=N)   # ropes in a group touch
b.enable_contact_forces(True); W = b.mesh_wrenches()                     # [N, M, 6] by the rope on each mesh
b.step_free(1/60, 100)                          # ropes without a pinned root
# a rope built folded or draped keeps a straight rest shape with straight_rest=True:
# ropesim.RopeBatch.from_positions(pos, ..., pinned=pin, straight_rest=True)

# Isaac Lab: PhysX owns every rigid body, ropesim owns the rope -----------------------------
from ropesim.isaac_bridge import ColliderSpec, RopeColliders
bridge = RopeColliders(rope, scene, [ColliderSpec("robot", bodies=link_names),           # kinematic: the rope feels it
                                     ColliderSpec("box", dynamic=True, mu_s=0.8)])       # dynamic: the wrench goes back
bridge.sync_poses(reset_env_ids); rope.step(...); bridge.apply_wrenches()               # every control frame
```

Use `deterministic=True` when constructing a `RopeBatch` to replay positions, velocities,
and orientations with identical inputs on the same device and build. The mode uses stable
neighbor snapshots during the solver sweeps. Mesh wrench sums use floating-point atomics.

`bake_mesh(vertices, triangles, spacing, margin)` turns any closed triangle mesh into a grid
(Warp).

## Examples

Pure rope simulations, no Isaac Sim needed: four short checks against theory and three
showcase scenes that each write a standalone HTML player. [examples/README.md](examples/README.md)
walks through every one with a figure of what it produces.

```bash
python examples/hanging_rope.py            # one rope settles; elongation vs m g L / 2EA
python examples/batch_swing.py             # 1024 driven ropes and the throughput you get
python examples/mesh_contact.py            # a rope dropped on a box, then carried by it (force readout)
python examples/throughput.py              # solver-only frames/s sweep over the batch size
python examples/twist.py                   # 12 ropes twisted into a bundle, held, unwound
python examples/net.py                     # a rope net catches the Stanford bunny, a free rigid body
python examples/strings.py                 # a Shadow Hand plucks and rakes four strings
```

The Isaac Lab demos (two-way coupling with PhysX bodies, a rope curtain, a rope pile) and the
UR5e rope-swing RL task are in [DeformX](https://github.com/DeformX/DeformX) under `demo/`.

## Performance (measured)

| Setup | Result |
|---|---|
| 1024 ropes × 33 vertices, 50 substeps × 8 iterations, RTX 4090 | 406 frames/s (6.8× real time at 60 Hz), solver only |
| Isaac Lab rope-swing task, headless, RTX 4090 | ~55k env-steps/s at 512 envs, ~153k at 2048 |
| 1024 / 4096 ropes × 21 vertices, 100 × 16, shared GPU | 11.9 / 17.6 ms per frame; one static mesh 14.2 / 21.7; moved every frame 14.2 / 23.3; with force readout 14.5 / 23.4; four moving meshes 17.9 / 29.0 |

Rope-rope contact costs +10–15% at 16 iterations (1.4–2.2× at 8) for the group sizes of
the validation scenes.

## Documentation

- [docs/SOLVER.md](docs/SOLVER.md): the physics model, parameters and units, dissipation,
  root driving, every contact law with its constants, the Isaac Lab bridge and its stability
  rule, caveats and known limits.
- [tests/README.md](tests/README.md): what each test file covers and the measured baselines.
- [third_party/yarnball/VENDOR.md](third_party/yarnball/VENDOR.md): provenance and the local patches.

## License and citation

The vendored solver is GPL-3.0, so this repository is distributed under
[GPL-3.0](LICENSE). If you use it, please cite *Stable Cosserat Rods* (Hsu, Wang, Wu and
Yuksel, SIGGRAPH 2025; BibTeX on its [project page](https://graphics.cs.utah.edu/research/projects/stable-cosserat-rods/))
and DeformX:

```bibtex
@inproceedings{deformx,
  title     = {DeformX: A Versatile Co-Simulation Framework for Deformable Linear Objects},
  author    = {Yang, Yi and Fei, Xiang and Wang, Lehong and Li, Chenhao and Dai, Zilin and Kou, Henry and Li, Lu and Choset, Howie},
  booktitle = {IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  year      = {2026}
}
```
