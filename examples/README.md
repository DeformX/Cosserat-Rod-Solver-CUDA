# Examples

Four standalone rope simulations. They need only `ropesim` and a CUDA GPU, not Isaac Sim, and
each one finishes in a few seconds. Build and install the package first (see the
[top-level README](../README.md)), then run them from the repository root.

| Script | What it shows |
|---|---|
| [`hanging_rope.py`](hanging_rope.py) | One rope (`RopeSim`) settles under gravity; its stretch matches theory |
| [`batch_swing.py`](batch_swing.py) | 1024 ropes in one `RopeBatch`, roots driven from CUDA tensors |
| [`mesh_contact.py`](mesh_contact.py) | Rope on a box and the floor: mesh and plane contact, friction, force readout |
| [`throughput.py`](throughput.py) | Solver frames per second as the batch grows |

Units are SI throughout: density ρ in kg/m, `k_stretch` = EA in N, `k_bend` = EI in N·m²,
`drag` in 1/s. Every example sets `damping = 0`. YarnBall's internal damping brakes each
vertex's absolute velocity, not just deformation, so physical dissipation comes from `drag`
alone ([docs/SOLVER.md](../docs/SOLVER.md), section 3).

## Hanging rope

```bash
python examples/hanging_rope.py            # add --plot to draw the settled rope
```

![Hanging rope: settled shape, strain along the rope against theory, total elongation over time](images/hanging_rope.png)

A 0.96 m rope with 33 vertices hangs from a fixed root. It starts at its rest length, bounces
along its axis, and settles within about two seconds under heavy air drag. Each segment
carries the weight of the rope below it, so its strain falls linearly from root to tip,
ε(s) = ρ g (L − s) / EA. Summed over the rope, the total elongation is m g L / 2EA. The
solver's per-segment strain lies on that line, and its total elongation matches to four digits:

```
rest length 0.960 m, settled length 0.96090 m
elongation 9.041e-04 m, theory m g L / (2 EA) = 9.041e-04 m
```

## Many driven ropes

```bash
python examples/batch_swing.py --num_ropes 1024 --seconds 3
```

![Batch swing: one rope whirling in 3D, the same rope from above, and the tip height of all 1024 ropes](images/batch_swing.png)

1024 ropes, each 1 m long with the properties of a 12.7 mm braided nylon rope, live in one
`RopeBatch`. Each root is pinned and moved around a 0.3 m circle at 1 Hz. This is the pattern
an RL environment uses. Every frame passes the root's position and velocity at the frame start
and its position at the frame end as CUDA tensors. The solver follows a quadratic path between
them across 100 substeps. Positions come back as a zero-copy view of the solver's buffer.

The rope spins up from hanging straight down. Its tip flies out to a circle of about 1.2 m and
climbs toward root height. All 1024 ropes get the same input, and their tip heights agree to
about a millimetre. The script prints the throughput and that spread:

```
1024 ropes x 21 vertices, 100 substeps x 8 iterations per 60 Hz frame
180 frames in 1.53 s: 117.8 frames/s, 120.6k rope-steps/s, 1.96x realtime
tip height mean 1.235 m, spread 0.0009 m (all ropes see the same motion)
```

## Contact with a box and the floor

```bash
python examples/mesh_contact.py
```

![Mesh contact: four snapshots of the rope on the box, and the force the rope exerts on the box over time](images/mesh_contact.png)

A free 1.2 m rope is released 0.4 m above the floor, across a 0.4 × 0.4 × 0.2 m box standing
on it. After two seconds the box slides +x at 0.2 m/s. The box is a signed-distance grid built with `ropesim.sdf`.
The floor is a contact plane. The script reads the force the rope exerts on the box every
half second.

- **Landing:** the rope drapes over the box with its ends on the floor. The box carries
  1.15 N of the rope's 1.38 N weight.
- **Sliding:** friction on the box top carries the rope along, and its ends drag on
  the floor. The steady backward pull on the box is the floor's kinetic friction on the ends
  plus a little air drag. The script checks the two against each other:

```
t= 5.0s  rope z in [0.0063, 0.2146] m  x mean +0.557 m  force on box (-0.104, +0.000, -1.182) N
(steady pull on the box F_x = -0.104 N; floor friction on the dragged ends plus air drag predicts -0.103 N)
```

The box is kinematic here: `set_mesh_poses` moves it every frame and the rope feels the moving
surface. To let the rope push the body back, feed the force readout to a physics engine. The
Isaac Lab bridge (`ropesim.isaac_bridge`) does that.

## Throughput

```bash
python examples/throughput.py              # N = 64 ... 2048
python examples/throughput.py 64 4096      # your own list
```

![Throughput: solver frames per second and rope-steps per second against the number of ropes](images/throughput.png)

The sweep times ropes of 33 vertices at 50 substeps × 8 iterations per 60 Hz frame, and
compares two ways of driving the root:

- **constant velocity:** `step(pos, vel, ...)` sweeps the root linearly across the frame.
- **motion program:** `step(..., root_pos_end=...)` lands the root on the frame-end pose.
  It costs about the same.

Frames per second stay roughly flat up to about 1024 ropes, because the GPU is not yet full.
Past that, total work per second, in rope-steps, keeps climbing. The figure was measured on
an RTX PRO 6000 Blackwell Max-Q shared with other jobs. Expect your numbers to differ with
the GPU and its load.
