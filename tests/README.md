# ropesim validation suite

Physics-grounded validation of the wrapped, de-collisioned YarnBall Cosserat
solver — run **before** building batching / Isaac Lab layers on top.

```bash
pip install -e .                      # rebuild after any csrc/ change
bash scripts/run_tests.sh             # full suite
bash scripts/run_tests.sh -m 'not slow'
```

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
