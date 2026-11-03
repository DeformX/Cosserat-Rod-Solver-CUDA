# ropesim validation suite

Physics-grounded validation of the wrapped, de-collisioned YarnBall Cosserat
solver — run **before** building batching / Isaac Lab layers on top.

```bash
pip install -e .                      # rebuild after any csrc/ change
bash scripts/run_tests.sh             # full suite
bash scripts/run_tests.sh -m 'not slow'
```

## Contact tests (2026-10)

| file | what | needs |
|---|---|---|
| `test_contact.py` | rod–plane and rope–rope contact (AGENTS §5b) | torch |
| `test_mesh_contact.py` | rod–mesh contact on signed-distance grids: box ≡ plane, posed incline, capstan, drape, thin wall, baked vs closed form, per-env poses, API | torch; one test skips without `warp` + `trimesh` |
| `test_mesh_motion.py` | kinematic (moving) meshes: interpolation, moving-surface friction, teleport, force readout | torch |
| `test_mesh_broadphase.py` | 32 meshes, candidate lists, per-mesh friction, CUDA-tensor targets, device views, review-3 regressions | torch |
| `test_isaac_bridge_geometry.py` | the Kit-free half of the Isaac bridge: tessellated primitives, welding, shells, fillets | warp (skips otherwise) |

The Isaac Lab half of the bridge has no pytest (one Kit per process, minutes to start); its
integration checks are the scripts `scripts/isaac_bridge_demo.py` (momentum balance, dragged box,
rolled cylinder) and `scripts/isaac_swing_compare.py` (the swing env with and without contact).

## Why analytic tests instead of an OBJ-diff against the original `Gui`

The integration **wraps the identical compiled kernels** — an output-diff
against the original binary would mostly re-test file parsing, and the JSON
knit scenes can't express our single-rope cases. Instead:

- **Analytic ground truth** validates end-physics (what actually matters for
  sim-to-real): catenary elongation, hanging-chain mode, beam droop.
- `test_equivalence.py` directly checks the *one thing we changed* vs
  upstream — the de-collisioned config (`detectionPeriod=-1`,
  `useStepSizeLimit=0`) — by comparing against the collisions-on config in a
  contact-free scene.

## Unit mapping (established by these tests)

| ropesim arg | physical meaning | validated by |
|---|---|---|
| `density` | linear density ρ [kg/m] | `test_inv_masses` (m_i = ρ·l) |
| `k_stretch` | axial stiffness **EA** [N] | `test_hang_elongation_matches_EA_theory` (ΔL_i = T_i·l_i/EA) |
| `k_bend` | bending stiffness **EI** [N·m²] | `test_cantilever_droop_vs_beam_theory` (δ = wL⁴/8EI) |
| `drag` | velocity decay → e^(−drag·t), **substep-invariant** | convergence tests rely on it |
| `damping` | internal damping scaled by 1/h → **NOT substep-invariant**; keep 0 unless calibrated at a fixed h | (see note) |

⚠ `damping`'s effect changes with substep size (`damping/h` in the kernels).
Tests use `damping=0`; if RL tuning ever uses it, pin `max_h` first.

## What each file covers

- **test_api.py** — construction geometry, mass formula, error paths, knob
  round-trips, pinned root, `reset()` exactness + no-hidden-state.
- **test_statics.py** — vertical-hang equilibrium; **EA** via discrete-exact
  hanging elongation; horizontal→vertical settle; **EI** via clamped
  cantilever droop (absolute + 1/EI proportionality); clamped-root frame
  steers the rod (the EE orientation boundary condition).
- **test_dynamics.py** — hanging-chain pendulum period vs Bessel-J₀ theory
  (T = (4π/z₁)√(L/g)); energy never increases on a free swing (dissipation
  measured & reported); ±θ mirror symmetry; substep convergence (50/frame ≈
  100/frame); iteration-count convergence (numItr 4 vs 16) + constraint
  residuals.
- **test_driving.py** — root sweep exactness (pos += vel·H per frame);
  inter-frame continuity (no teleports); 60 Hz vs 240 Hz control-rate
  convergence; **swept root pumps energy** (the handoff's core coupling
  mechanism); whip-rate stability.
- **test_equivalence.py** — collisions-on ≡ collisions-off when contact-free;
  identical-sim and run/reset/rerun reproducibility (chaotic Gauss-Seidel ⇒
  tolerance-based, spread is printed for RL expectations).
- **test_longrun.py** *(slow)* — 30 s of random sum-of-sines driving: finite,
  bounded strain/reach, zero solver errors, root still exactly on trajectory.

Tolerances encode physics (discretization + fp32 + chaotic-GS noise), not
wishful thinking — when a number surprises, the test prints the measurement
so the bound can be revisited deliberately rather than silently loosened.

## Measured baseline (RTX 4090, 2026-06-10, all 34 tests green)

- **Pendulum period**: +3.6% vs hanging-chain theory (33 verts).
- **EA / EI mappings**: hanging elongation within 2%; cantilever droop in the
  beam-theory band at EI∈{0.4, 0.8}, 1/EI proportionality within 10%.
- **Bitwise deterministic**: identical sims and reset/rerun produce identical
  trajectories (the chaotic-GS races are schedule-stable on this GPU). Needed
  one fix: `meta.h` must be seeded/reset to `maxH` (binding does this).
- **Energy on a free swing (3.5 s, drag=0)**: no gain (-0.02%); retention is a
  CONVERGENCE artifact, not inherent: 50 substeps/8 itr → 16.9%, but
  100 ss/8 it → 59.8%, 50 ss/32 it → 60.2%. The substeps×iterations budget is
  the main fidelity lever for an energetic swing task.
- **Convergence**: substeps apparent order ~1.3 (50ss sits 3.9%L from 100ss on
  a fast whirl); iterations 4→8→16 monotone (7.3%L → 3.4%L vs 16). Constraint
  (strain) residual 0.09% even at numItr=4.
- **De-collisioned ≡ collisions-on** for contact-free scenes (max tip diff
  well under 1%L).
- **⚠ Stiff-bend instability (KNOWN LIMIT)**: at tube-scale EI=10 N·m²
  (h=1/3000, l=0.02, ρ=0.05), root reorientation blows up (strains >300%,
  internal velocities ~300 m/s, masked by drag). Stable at EI=0.4. Encoded as
  `xfail(strict=True)` asserting the *good* physics: today it xfails
  (documents the limit); if an upstream fix moves the limit it XPASSes and
  the suite fails loudly — re-evaluate tube modeling then. Affects only the
  future "tube as stiff segments" option — not the locked rigid-transform plan.
- **Cantilever EI ratio**: measured 0.924/0.928 (EI=0.4/0.8) — mapping good to
  ~8% (consistent discretization bias); test band tightened to [0.80, 1.05].

### Batch (M1, RopeBatch — measured 2026-06-10)
- **Batch ≡ single-rope oracle: bitwise identical** (env 0 vs RopeSim, 1 s of
  driven swing — same kernels over a larger flat array).
- **Cross-talk: exactly zero** (whipped neighbor leaves a static env
  bit-identical).
- **Throughput: 1024 envs × 33 verts → 406 FPS (6.8× realtime)** at 50
  substeps × 8 iters/frame on the RTX 4090, including zero-copy observation
  gather (~416k env-steps/s on the rope side).
- Zero-copy verified: torch tensors share the device buffers (data_ptr match,
  step() visible through existing tensors without re-fetch).
- **Second-order root driving** (quadratic on measured start velocity +
  per-substep SLERP): endpoint-exact (2e-6), degenerates to the legacy sweep
  at a=0 (8e-7), and tracks a same-h 120 Hz reference at **0.001%L vs
  0.068%L** for constant-velocity 60 Hz driving (~68×). Lesson encoded in the
  test: control-rate comparisons must hold substep size h fixed — the
  solver's h-sensitive dissipation otherwise dominates the measurement.
