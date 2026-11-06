# Demos

Three showcase scenes recorded with the Cosserat rod solver and played back in standalone HTML (three.js):

| script | scene | what is simulated |
|---|---|---|
| `twist.py` | 12 pre-tensioned soft ropes whose ends sit on two fixed, facing circles that counter-rotate in place (nothing else drawn) | both rope ends pinned and driven; rope–rope contact; the ropes stretch (EA 120 N) into a 3-turn bundle until the strain cap, hold, unwind |
| `net.py` | 10 × 10 rope net (14 cm spacing) catches a 0.75 m Stanford bunny | 20 ropes pinned at both ends, rope–rope contact; the net hangs under gravity first; the bunny is a free rigid body integrated from the solver's mesh wrench (sub-cycled exchange); per-frame penetration readout |
| `strings.py` | a Shadow Hand (MuJoCo Menagerie, unmodified) rests three fingertips on a string, sweeps the index finger across four pre-tensioned strings twice (joint motion, fingertip path solved by IK), then straightens all fingers and rakes across with the fingertips below the strings (each string slides down the pads and is plucked off the tips) | 4 strings (361 vertices, one solver batch each so every string collides at its drawn radius) pinned both ends; 23 hand links as kinematic signed-distance grids of their drawn meshes; fingertip paths solved by IK in `hand_export.py`; zero gravity; per-frame string–hand gap readout |

Shared: `viewer.py` + `viewer_core.js` (HTML player: smooth tubes, int16 trajectories, scrub/speed/orbit),
`common.py` (output directory, pin driving, recording), `assets.py` (hole closing, baking), `hand_export.py`
(forward kinematics, per-link meshes and fingertip IK for the hand; runs in a separate venv with `mujoco`).

All three set the solver's gravity explicitly (its default is y-down) and run at h = 1/24000 s (net), 1/48000 s (twist)
or 1/96000 s (strings). Rope-rope friction stiffness above ~1e5 N/m makes a taut twisted bundle chatter (stick-slip);
the twist uses 1e4.

## Running

`ropesim` (built from this repo) must be importable; then e.g. `CUDA_VISIBLE_DEVICES=1 python demos/twist.py`.
Everything else lives in one directory, `$ROPE_DEMOS_DIR` (default `~/rope_demos`):

- outputs `twist.html`, `net.html`, `strings.html` (+ a `.npz` of each recording);
- `assets/bunny.obj` (Stanford bunny) and `assets/shadow_hand/` (the `shadow_hand` folder of
  [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie), untouched);
- `venv_mj/`, the venv that runs `hand_export.py`: `python -m venv venv_mj && venv_mj/bin/pip install mujoco trimesh scipy`.

The strings demo takes ~15 min (four batches stepped sequentially at 96 kHz); the other two a few minutes.
