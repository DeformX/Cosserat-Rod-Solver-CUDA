# Solver notes

What the solver computes, which knobs mean what, and the limits that were measured. The
numbers come from the validation suite (`tests/`) and from scratch scenes run while the
features were built; the control frame is 1/60 s unless stated.

## 1. The solver

- **Stable Cosserat Rods** (YarnBall): a split position/orientation scheme. Positions take a
  VBD-style implicit step (per-vertex Newton with a diagonal Hessian), orientations a
  closed-form Gauss–Seidel quasi-static update; ~4–16 iterations per substep, stable at large
  stiffness and time steps, embarrassingly parallel on the GPU. Frames are quasi-static, so there
  is no torsional inertia (negligible for thin ropes).
- **De-collisioned configuration.** YarnBall's own collision pipeline is switched off
  (`detectionPeriod = -1`, `useStepSizeLimit = 0`): the position Hessian becomes
  isotropic-diagonal and the LBVH and per-substep syncs are skipped. Contact is added back
  explicitly (section 5). Verified dynamics-identical to the collisions-on configuration in
  contact-free scenes.
- **Batching.** All ropes share one flat vertex array; each rope is terminated by a flag and
  neighbour guards stop any cross-talk (verified bitwise: a whipped neighbour leaves a static
  rope bit-identical; one rope in a batch matches `RopeSim` bitwise).
- **Substeps.** `max_h` (`RopeSim`) or `h = dt / substeps` (`RopeBatch.step`) is the solver
  substep. Always pass an explicit substep count: the solver's own `advance(dt)` rounds in
  float32 (1/60 ÷ 1/3000 gives 51 substeps, not 50).
- **Determinism.** The Gauss–Seidel sweeps read neighbours racily, yet reset/rerun and
  batch-vs-single comparisons are bitwise reproducible in practice without rope-rope contact
  (one requirement: the substep size is seeded at construction and restored on reset). With
  rope-rope contact reruns differ by ~5e-5 m after 5 s; the per-mesh wrenches use float
  atomics and are reproducible to ~1e-7 relative.
- **Stability limit.** Bending stiffness of the order of a plastic tube (EI ≈ 10 N·m² at
  h = 1/3000, 2 cm segments, 0.05 kg/m) blows up under root reorientation (strains > 300%,
  internal velocities ~300 m/s masked by drag); EI = 0.4 is stable. The suite tripwires this as
  a strict xfail: an unexpected pass means the limit moved. `accel_ratio` (successive
  over-relaxation) above 1 is unstable at rope parameters.

## 2. Parameters and units

| Parameter | Meaning | Note |
|---|---|---|
| `density` | linear density ρ [kg/m] | vertex mass m_i = ρ · l_i |
| `k_stretch` | axial stiffness EA [N] | per-segment stiffness EA / l |
| `k_bend` | bending stiffness EI [N·m²] | must be > 0: zero makes the orientation update normalize a zero vector (NaN) |
| `num_itr` | iterations per substep | 8 default; 16 for stiff or heavily loaded ropes |
| `drag` | velocity decay rate [1/s] | substep-invariant, v ~ e^(−drag·t) |
| `drag_quad`, `drag_quad_t` | quadratic cross-flow drag [1/m] on the segment-normal and tangential velocity | section 3 |
| `damping` | YarnBall's internal damping | keep 0 (section 3) |

Validated mappings (`tests/test_statics.py`): hanging-rope elongation ΔL = m g L / (2 EA) to
0.03%, cantilever droop δ = w L⁴ / (8 EI) within the beam-theory band, pendulum period of a
hanging chain to +0.05% with 1 m / 21 vertices (3.6% with 33 vertices of 2 cm).

Per-vertex properties (`RopeBatch.set_masses`, `set_stretch_stiffnesses`,
`set_bend_stiffnesses`, `[V]` or `[N, V]`) are legal only between construction and the first
step; pinned vertices keep infinite mass.

## 3. Dissipation

Three channels remove energy; two of them are easy to misuse.

1. **`damping`** is applied diagonally in the position update, so it brakes each vertex's
   absolute velocity (rate γ = 2 · damping · EA / (ρ l²)) rather than relative deformation like
   true Rayleigh damping. YarnBall's yarn default of 1e-6 gave γ = 16 /s on a 1 m rope and
   overdamped the swing so strongly that it looked like weak gravity. It is also not
   substep-invariant. Keep it at 0 (the bindings default to 0).
2. **`drag`** is a clean exponential decay and the physical aerodynamic knob when a single
   speed dominates. Linearized cylinder drag F/L = ½ ρ_air C_d d v² (C_d ≈ 1.1) gives an
   equivalent rate γ(v) ≈ 0.112 · v per m/s of speed for a 1 cm rope: `drag` ≈ 0.2–0.25 at
   2 m/s, ~0.1 for slow motion, ~0.4 for whips.
3. **`drag_quad`** is the physical form at every speed: per substep the segment-normal
   velocity component is updated as v_n ← v_n / (1 + k_n |v_n| h) (the exact frozen-direction
   solution, unconditionally stable, substep-invariant), with the tangent taken from the segment
   frame; `drag_quad_t` does the same on the tangential component. k_n = ½ ρ_air C_d d / ρ,
   i.e. 0.094 /m for a 12.7 mm, 0.089 kg/m rope. When `drag_quad` is on, use only a small
   residual `drag` (0.02): both at full strength double-counts. Zero leaves the legacy arithmetic
   bitwise unchanged. Tests: `tests/test_drag_quad.py`.

An **unconverged solve** is a fourth, numerical channel: with 50 substeps and 8 iterations a
1 m, 0.06 kg/m rope at EA = 2e3 N loses swing amplitude at 0.59 /s and at EA ≥ 5e3 does not
oscillate at all; 100 × 8 gives 0.024 /s at EA = 2e3 and 0.087 /s at 3e3; 100 × 16 gives
0.004 /s at 3e3. Raising EA needs more substeps × iterations. Energy retention over a 3.5 s
free swing: 17% at 50 × 8, ~60% at 100 × 8 or 50 × 32. For reference, real nylon rope has
2–5% internal hysteresis.

A calibrated operating point for 10–12.7 mm braided nylon: 21 vertices × 5 cm, ρ from
weighing the rope (the one number that matters; the rest tolerate ±20%), EA = 3e3–1.2e4 N,
EI = 2e-4–4.5e-3 N·m², 100 substeps × 8–16 iterations, `drag` 0.2 or `drag_quad` 0.094 with
`drag` 0.02, `damping` 0.

## 4. Root driving (`RopeBatch.step`)

`step(root_pos, root_vel, dt, substeps, root_q=None, root_pos_end=None, root_q_end=None)`.
Without `root_pos_end` the root sweeps at constant velocity (bitwise the single-rope path).
With it, the root follows p(τ) = p₀ + v₀ τ + ½ a τ² where a lands exactly on p₁ at τ = dt;
velocities are sampled at substep midpoints so the integrated position reproduces the
quadratic exactly (endpoint error 2e-6, no drift), and the root frame SLERPs from q₀ to q₁.
Measured against a same-substep 120 Hz reference: 0.001% of the rope length versus 0.068%
for the constant-velocity sweep. Moving the root across the substeps is the only mechanism
that injects boundary velocity into the rope: holding it still under-energizes the swing.

Inputs are contiguous float32 CUDA tensors (`__cuda_array_interface__`), copied into a
persistent device buffer at step entry; outputs are gathered copies in device views that torch
wraps zero-copy, kept alive by the batch. Ordering relies on torch using its legacy default
stream (true in Isaac Lab). When comparing control rates, hold the substep size fixed: the
solver's dissipation depends on h and otherwise dominates the comparison.

`reset_envs(ids)` restores the construction state and zeroes velocities. A batch built with a
free root (`from_positions(..., pinned=...)` without vertex 0) refuses `step()` and advances
with `step_free(dt, substeps)`.

## 5. Contact

All contact is off by default. `set_contact(radius, mu_s, mu_k, v_slip, kappa)` is the surface
material shared by planes and meshes; `set_planes` also sets it.

### Planes

Up to 8 half-spaces n·x ≥ w, global to all envs. Vertex-local penalty ½ k_i pen² with
k_i = m_i (κ/h)²: every vertex has the same contact frequency κ/h and the same rest sink
g h² / κ², so the bounce does not depend on the substep size; κ sets only bounce and
penetration (κ = 2: restitution 0.09–0.10, 0.31 mm at a 10 m/s impact; κ = 1: 0.25, 0.83 mm;
κ = 3: 0.05, 0.16 mm). There is deliberately no contact damping: a dashpot switching on at
zero gap made the local solve limit-cycle with the iteration count. Friction is regularized
Coulomb with a static→kinetic blend: stick stiffness μ_s N / (ε h) below the slip speed ε,
μ_s → μ_k between ε and 2ε, μ_k N above. Working constants for a rope of radius R: κ = 2,
μ_s 0.5, μ_k 0.4, ε 1e-3 m/s.

### Rope-rope

YarnBall's own model (log barrier at 2·radius + barrier, friction spring capped at μ N,
conservative step limit against tunnelling), fed by a per-group brute-force detector in
`csrc/batch.cu` (`group_ropes` consecutive ropes form one group; groups never interact;
deterministic pair order, no host sync, O(V²) per group). Constants that work: radius 0.9 R,
barrier 0.2 R, k_collision 0.025 J (0.25 J if ropes can be crushed onto each other above
~8 m/s), μ 0.4, k_friction 1e7 N/m. Never reuse YarnBall's yarn constants (kFriction 5,
kCollision 1e-5): at rope mass the friction does nothing.

### Meshes

A mesh is a signed-distance grid in its object frame (`ropesim.sdf`: `bake_mesh` with Warp,
`from_function`, or the closed forms `sdf_box`, `sdf_sphere`, `sdf_cylinder`, `sdf_bowl`), up to
32 per solver, each with a pose per env and its own friction. The vertex is transformed into
the object frame, φ and ∇φ come from a trilinear sample, and the same penalty and friction
law as for planes acts along ∇φ. Grids need a margin beyond the surface (rope radius plus a
few cells); outside the lattice there is no contact.

- **Kinematic meshes.** `set_mesh_poses(index, pos, quat, teleport=False)` gives the pose at
  the end of the next step: the mesh moves there over the substeps (constant velocity and
  angular velocity within the frame, interpolated per substep) and friction acts on the
  displacement relative to the moving surface, so a rope on a conveyor rides along at exactly
  the belt speed. `teleport=True` replaces the pose at once (use it on resets; a rope inside the
  teleported volume is ejected at ~0.8 · penetration / h). Send one pose per mesh per frame;
  the last call before a step wins. The lattice must cover the object's travel within a frame.
- **Contact-force readout.** `enable_contact_forces(True)` evaluates the surface law once
  more at the final displacement of every substep (the force the implicit scheme applied) and
  accumulates per-vertex forces and per-mesh wrenches about the mesh origin, averaged over the
  frame (the impulse a rigid-body engine should receive). Cost +2–5% of a frame; planes
  contribute to per-vertex forces only.
- **Broad phase.** A per-vertex candidate list rebuilt every substep from the world AABBs of
  the (env, mesh) pairs keeps 32 meshes affordable: 1024 / 4096 ropes × 21 vertices at
  100 × 16 cost 11.9 / 17.6 ms per frame bare, 14.2 / 21.7 with one static mesh, 14.2 / 23.3 with
  it moving every frame, 14.5 / 23.4 with the readout, 17.9 / 29.0 with four moving meshes
  (shared GPU).
- **Baked bodies.** Convex edges are best rounded (`round` ≈ half a segment length) when a rope
  is dragged over them, but rounding fattens the body by that amount on every face.
  A rope with bending length (EI / ρ g)^(1/3) = 16 cm conforms to a 25 cm sphere but tips off a
  15 cm one like a plank.

### Caveats

- float32: a rope resting on a plane far from the origin along its normal sits on the float
  grid (60 nm at 0.8 m, 0.5 µm at 5 m) and reports a constant normal velocity ≤ 1e-3 m/s while
  its position does not drift. Prefer env-local coordinates; the signed-distance path subtracts
  the object position before rounding and is exact at any distance.
- Start scenes and resets clear of planes, meshes and other ropes: a vertex that starts a depth
  `pen` inside a surface leaves at up to pen / h.
- Planes are static (friction uses absolute displacement); anything that moves is a mesh.
  Friction acts on the centreline, so there is no rolling, and it is isotropic.
- A perfectly straight, exactly vertical rope lowered end-on onto a floor stays straight to
  9–12% compression and then snaps outward at ~30 m/s; half a degree of tilt removes it.
  Perturb exactly vertical resets if a floor is enabled.
- Slide accuracy follows solver convergence: 10% off at 50 × 8, 0.25% at 100 × 16.

## 6. The Isaac Lab bridge (`ropesim.isaac_bridge`)

`RopeColliders(rope, scene, [ColliderSpec(asset, bodies, dynamic, mu_s, mu_k, spacing, margin,
round, thickness)])` bakes each body's USD collision geometry once into a grid in the PhysX
link frame (the prim's world transform without its scale; a scaled cube that is itself the
rigid body keeps the scale on the shape), with per-env poses from the scene views. Open
meshes are welded first and otherwise baked as a two-sided shell. Each control frame, after
PhysX has stepped: `sync_poses(reset_env_ids)` (link poses → one `set_mesh_poses_all` with CUDA
tensors, teleport for reset envs), `rope.step(...)`, `apply_wrenches()` (the frame-averaged
wrench by the rope on each dynamic body, torque moved to the centre of mass, handed to PhysX
as an external force for its next step). Kinematic bodies get nothing back. Use `round = 0`
for bodies that must match PhysX's surface exactly.

The exchange is explicit with a one-frame lag. That is fine for kinematic links and for
bodies held by the ground (a 2 m rope draped over a 0.5 kg box on a slippery floor rests at
−1.05 N = the weight of the rope off the ground, drifts 0.65 mm/s, and is dragged 29.5 cm by
a 35 cm pull; a 0.3 kg cylinder rolls 8.2 cm without slipping under a pulled rope). It is
unstable for a light body held only by the rope's static friction: the rope acts as a spring
k ≈ 400 N/m with c ≈ 1 N·s/m, and a wrench delayed by τ is negative damping k·τ, stable only
while k·τ < c (τ < ~2.5 ms). Sub-cycle the exchange in that case (`isaac_bridge_demo.py
--exchange K`: PhysX at 60·K Hz, the rope's substeps split K ways); K = 10 quiets a freely
rolling 0.3 kg cylinder, and a stiff axial push on a free body wants an implicit coupling,
which is not built.

## 7. Measured baselines

See [tests/README.md](../tests/README.md) for the per-test numbers (pendulum period, EA/EI
mappings, energy, convergence orders, contact scenes). Throughput: 1024 ropes × 33 vertices
at 50 × 8 run at 406 frames/s on an RTX 4090 (6.8× real time), launch-bound below ~256 ropes
and GPU-bound above; the Isaac Lab rope-swing environment reaches ~55k env-steps/s at 512
envs and ~153k at 2048.
