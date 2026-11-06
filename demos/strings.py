"""Demo 3: four strings of different radius, pinned and pre-tensioned at both ends, played by a Shadow Hand
(MuJoCo Menagerie model, unmodified; kinematic, each link a signed-distance grid of its drawn mesh).

The hand rests in a relaxed posture with the middle, ring and little fingertips on a string; the index finger
sweeps across all four strings twice (joint motion, fingertip path solved by IK); then all fingers straighten and
the hand rakes across the strings with the fingertips below them. One solver batch per string, so each string has
its own contact radius (what is drawn is what collides). Zero gravity, no body: just strings.
Hand kinematics come from hand_export.py run in a venv that has mujoco.
"""
import argparse
import json
import pathlib
import subprocess
import time

import numpy as np
import trimesh
import ropesim
from scipy.ndimage import map_coordinates

from assets import bake_closed
from common import DEMO_DIR, Recorder, device_positions, drive_pinned, smoothstep
from viewer import write_html, to_view, quat_to_view, quat_rotate

p = argparse.ArgumentParser()
p.add_argument("--out", default=f"{DEMO_DIR}/strings.html")
p.add_argument("--hand_xml", default=f"{DEMO_DIR}/assets/shadow_hand/right_hand.xml", help="MuJoCo Menagerie shadow_hand/right_hand.xml")
p.add_argument("--venv_python", default=f"{DEMO_DIR}/venv_mj/bin/python", help="python with mujoco, trimesh, scipy (runs hand_export.py)")
p.add_argument("--work", default=f"{DEMO_DIR}/strings_work", help="where the hand spec and its kinematics export go")
p.add_argument("--fps", type=int, default=240, help="simulation frame rate (hand poses are updated per frame)")
p.add_argument("--record_fps", type=int, default=120)
p.add_argument("--substeps", type=int, default=400, help="per frame: h = 1/(fps*substeps) = 1/96000 s")
p.add_argument("--itr", type=int, default=16)
p.add_argument("--tension", type=float, default=1.6, help="string tension [N] (sets the pitch: 4-8 Hz here)")
p.add_argument("--ea", type=float, default=160.0)
p.add_argument("--tilt", type=float, default=0.35, help="hand pitch [rad]: fingers point across the strings, tilted down by this")
p.add_argument("--yaw", type=float, default=0.0, help="hand yaw [rad] about the vertical")
p.add_argument("--depth", type=float, default=0.004, help="index fingertip sweep height below the resting fingertips [m]")
p.add_argument("--rake_depth", type=float, default=0.007, help="middle fingertip height below the string axes during the final rake [m]; the pad of the "
               "Shadow Hand fingertip reaches 12 mm above the tip, a string above that is caught by the joint step and dragged")
p.add_argument("--rake_lf_above", type=float, default=0.0015, help="little fingertip height above the middle fingertip during the rake [m] "
               "(its knuckle is solved for it: level with the index and ring tips)")
p.add_argument("--strokes", type=int, default=2, help="index-finger sweeps before the final rake")
p.add_argument("--sdf_spacing", type=float, default=0.001)
args = p.parse_args()

L, X0, NS, V = 0.9, -0.45, 4, 361
YS = np.array([-0.045, -0.015, 0.015, 0.045])
RAD = np.array([0.003, 0.0025, 0.002, 0.0015])
RHO = 1100.0 * np.pi * RAD ** 2                     # nylon, kg/m
FPS, SUB, ITR = args.fps, args.substeps, args.itr
DT = 1.0 / FPS
REC = FPS // args.record_fps
EPS = args.tension / args.ea                        # pre-strain
f1 = np.sqrt(args.tension / RHO) / (2 * L)
print("string fundamentals [Hz]:", np.round(f1, 2), " pre-strain %.2f%%" % (100 * EPS))

# ---- hand choreography (anchor = lowest point of the middle fingertip, solver frame z-up) ----
T0 = 1.0                                            # strings are tensioned first
REST = np.array([0.20, 0.013, RAD[2] + 0.0005])     # middle/ring/little fingertips kiss string 3 (y = +15 mm)
LIFT = 0.025
rest = {"rh_FFJ3": 0.2, "rh_FFJ2": 0.2, "rh_FFJ1": 0.12}            # index relaxed, above the strings
for f in ("MF", "RF"):
    rest.update({f"rh_{f}J3": 0.9, f"rh_{f}J2": 1.2, f"rh_{f}J1": 0.7})
rest.update({"rh_THJ5": 0.0, "rh_THJ4": 0.2, "rh_THJ3": 0.0, "rh_THJ2": 0.0, "rh_THJ1": 0.2})
index_rest = {k: rest[k] for k in ("rh_FFJ3", "rh_FFJ2", "rh_FFJ1")}
straight = {f"rh_{f}J{j}": a for f in ("FF", "MF", "RF", "LF") for j, a in ((3, 0.15), (2, 0.15), (1, 0.1))}
SWEEP = (-0.07, 0.05)                               # index fingertip y range relative to the anchor (covers all 4 strings)
RAKE = (-0.10, 0.09)                                # middle fingertip y range during the rake (hand translation)
path, keys = [], []
def at(t, dz=0.0, y=REST[1], z=None): path.append([T0 + t, REST[0], y, (REST[2] + dz) if z is None else z])
def key(t, joints, mode="smooth"): keys.append([T0 + t, dict(joints), mode])
def sweep(t, dur, y0, y1, n=13):
    for i in range(n):
        u = i / (n - 1)
        key(t + dur * u, {"_low": {"FF": [y0 + (y1 - y0) * u, -args.depth]}}, "linear")
at(-T0, LIFT + 0.02); at(0.0, LIFT + 0.02); at(1.0, 0.0)
key(-T0, index_rest); key(1.0, index_rest)
t = 1.5
for s in range(args.strokes):                       # index down-stroke: come down outside string 1, sweep across, lift
    key(t, index_rest); key(t + 0.4, {"_low": {"FF": [SWEEP[0], -args.depth]}})
    sweep(t + 0.45, 0.7, SWEEP[0], SWEEP[1])
    at(t + 1.15, 0.0); at(t + 1.45, LIFT); key(t + 1.5, {"_low": {"FF": [SWEEP[1], -args.depth]}})
    if s + 1 < args.strokes:                        # open the index and settle the hand back onto the strings
        key(t + 1.95, index_rest); at(t + 1.95, LIFT); at(t + 2.3, 0.0)
        t += 2.4
# final rake straight out of the last lift: all fingers straighten (the little finger pivots down at its knuckle to the
# height of the index and ring tips) while the hand moves beyond string 1; then it lowers so the fingertips sit below
# the string axes, translates across all four strings (each string slides down the finger pads and is plucked off the
# tips) and lifts
key(t + 2.3, {**straight, "_low": {"LF": [None, args.rake_lf_above]}})
at(t + 2.4, LIFT, y=RAKE[0]); at(t + 2.8, y=RAKE[0], z=-args.rake_depth)
RAKE_T = (t + 2.9, t + 4.4)
at(RAKE_T[0], y=RAKE[0], z=-args.rake_depth); at(RAKE_T[1], y=RAKE[1], z=-args.rake_depth)
at(t + 4.5, y=RAKE[1], z=-args.rake_depth); at(t + 4.9, LIFT, y=RAKE[1]); at(t + 6.0, LIFT, y=RAKE[1])
DURATION = path[-1][0]
th, ps = args.tilt, args.yaw
c, s_, cp, sp = np.cos(th), np.sin(th), np.cos(ps), np.sin(ps)
spec = {"fps": FPS, "posture": rest, "posture_low": {"RF": [0.004, 0.0], "LF": [0.002, 0.0]}, "anchor": "rh_mfdistal", "path": path, "joint_keys": keys,
        "finger_dir": [sp * c, -cp * c, -s_], "palm_dir": [-sp * s_, cp * s_, -c]}
work = pathlib.Path(args.work); work.mkdir(parents=True, exist_ok=True)
(work / "spec.json").write_text(json.dumps(spec))
t0 = time.time()
subprocess.run([args.venv_python, "-I", str(pathlib.Path(__file__).resolve().parent / "hand_export.py"), args.hand_xml,
                str(work / "spec.json"), str(work / "hand.npz")], check=True, cwd=pathlib.Path(args.hand_xml).parent)
hand = np.load(work / "hand.npz")
col_names = [n for n in hand["names"] if n not in ("rh_forearm", "rh_wrist")]
vis_names = list(hand["vis_names"])
HF = int(hand["frames"])
print(f"hand: {len(col_names)} collision bodies, {len(vis_names)} visual bodies, {HF} frames ({time.time() - t0:.1f}s)")
k_rest, k_rake = int((T0 + 1.2) * FPS), int((T0 + 0.5 * (RAKE_T[0] + RAKE_T[1])) * FPS)
for n in ("rh_ffdistal", "rh_mfdistal", "rh_rfdistal", "rh_lfdistal", "rh_thdistal"):
    lo = hand[f"low_{n}"]
    print(f"  {n:12s} lowest point (y, z) [mm]: rest ({lo[k_rest, 1] * 1000:+.0f}, {lo[k_rest, 2] * 1000:+.1f})  mid-rake ({lo[k_rake, 1] * 1000:+.0f}, {lo[k_rake, 2] * 1000:+.1f})  "
          f"min z over run {lo[:, 2].min() * 1000:+.1f}")

# ---- strings: one batch per string (own contact radius) ----
pos = np.zeros((NS, V, 3), np.float32)
x = np.linspace(X0, X0 + L, V)
for k in range(NS):
    pos[k, :, 0] = x; pos[k, :, 1] = YS[k]
pinned = np.zeros(V, bool); pinned[0] = pinned[-1] = True
pin_mask = np.broadcast_to(pinned, (1, V))
seg = L / (V - 1)
batches = []
for k in range(NS):
    b = ropesim.RopeBatch.from_positions(pos[k:k + 1], density=float(RHO[k]), k_stretch=args.ea, k_bend=1e-6, max_h=DT / SUB, num_itr=ITR, pinned=pinned)
    b.set_masses(np.full((1, V), RHO[k] * seg, np.float32))
    b.set_gravity(np.zeros(3, np.float32))
    b.drag = 0.05; b.damping = 0.0
    b.set_contact(radius=float(RAD[k]), mu_s=0.25, mu_k=0.2, v_slip=1e-3, kappa=2)
    batches.append(b)

# ---- hand links as signed-distance grids of the drawn meshes (shared by the four batches) ----
t0 = time.time()
grids = []
for n in col_names:
    g = bake_closed(hand[f"col_v_{n}"], hand[f"col_f_{n}"], spacing=args.sdf_spacing, margin=float(RAD.max()) + 4 * args.sdf_spacing)
    P0 = hand[f"poses_{n}"][0]
    for b in batches:
        b.add_mesh(g.values, g.origin, g.spacing, pos=P0[:3].astype(np.float32), quat=P0[3:].astype(np.float32))
    grids.append(g)
M = len(grids)
print(f"baked {M} grids ({sum(g.values.size for g in grids) / 1e6:.1f} M nodes) in {time.time() - t0:.1f}s")
hand_pos = np.stack([hand[f"poses_{n}"][:, :3] for n in col_names], 1).astype(np.float32)   # [HF, M, 3]
hand_quat = np.stack([hand[f"poses_{n}"][:, 3:] for n in col_names], 1).astype(np.float32)

def pin_targets(t):
    tg = pos.copy()
    u = smoothstep((t - 0.1) / 0.5)
    tg[:, -1, 0] = X0 + L * (1 + EPS * u)
    return tg

def min_gap(P, k):
    """Per string: smallest signed distance from its vertices to any hand surface minus its own radius (trilinear sample)."""
    best = np.full(NS, np.inf)
    for j, n in enumerate(col_names):
        g = grids[j]; pose = hand[f"poses_{n}"][k]
        q = pose[3:].astype(np.float64); qi = q * np.array([-1, -1, -1, 1.0])
        loc = quat_rotate(qi[None], P.reshape(-1, 3) - pose[:3]).reshape(NS, V, 3)
        ijk = (loc - g.origin) / g.spacing
        inside = ((ijk >= 0) & (ijk <= np.array(g.values.shape) - 1)).all(-1)
        for s in range(NS):
            if inside[s].any():
                phi = map_coordinates(g.values, ijk[s][inside[s]].T, order=1, mode="nearest")
                best[s] = min(best[s], float(phi.min()) - RAD[s])
    return best

rec = Recorder()
frames = int(round(DURATION * FPS))
t0 = time.time()
worst_gap = np.full(NS, np.inf)
for k in range(frames):
    t = k * DT
    tn, tx = pin_targets(t), pin_targets(t + DT)
    kh = min(k + 1, HF - 1)
    for s, b in enumerate(batches):
        drive_pinned(b, pin_mask, tn[s:s + 1], tx[s:s + 1], DT)
        b.set_mesh_poses_all(hand_pos[kh][None].copy(), hand_quat[kh][None].copy())
    for b in batches:                      # the solver does not step several sims concurrently
        b.step_free(DT, SUB)
    if (k + 1) % REC:
        continue
    P = np.concatenate([device_positions(b) for b in batches])
    gap = min_gap(P, kh)
    worst_gap = np.minimum(worst_gap, gap)
    bodies = {n: (hand[f"poses_{n}"][kh, :3], hand[f"poses_{n}"][kh, 3:]) for n in vis_names}
    amp = np.abs(P[:, :, 1] - YS[:, None]).max(1) * 100
    rec.frame(P, bodies, f"string sway [cm] " + " ".join(f"{a:.1f}" for a in amp) + f" · closest string–hand gap {gap.min() * 1000:+.2f} mm")
    if (k + 1) % FPS == 0 or not np.isfinite(P).all():
        strain = np.linalg.norm(np.diff(P, axis=1), axis=-1) / seg - 1
        print(f"t={t:5.2f}s strain mean {100 * strain.mean():.2f}%  max |y-dev| {np.round(amp, 2)} cm  "
              f"max |z-dev| {np.round(np.abs(P[:, :, 2]).max(1) * 100, 2)} cm  gap {np.round(gap * 1000, 2)} mm  finite {np.isfinite(P).all()}  ({time.time() - t0:.1f}s)", flush=True)
    if not np.isfinite(P).all():
        print("DIVERGED"); break

# ---- viewer ----
rp = rec.rope_array()
F = rp.shape[0]
meshes, bodies = {}, []
for n in vis_names:
    meshes[n] = (to_view(hand[f"vis_v_{n}"]), hand[f"vis_f_{n}"])
    pp, qq = rec.body_poses(n)
    rgba = hand[f"vis_rgba_{n}"]
    col = "#%02x%02x%02x" % tuple(int(255 * float(c) ** (1 / 2.2)) for c in rgba[:3])   # the model's sRGB-ish colour
    metal = bool(rgba[0] > 0.5)
    bodies.append({"mesh": n, "color": col, "material": {"metalness": 0.6 if metal else 0.1, "roughness": 0.35 if metal else 0.5},
                   "poses": np.concatenate([to_view(pp), quat_to_view(qq)], 1).astype(np.float32)})
for i, xb in enumerate((X0 - 0.012, X0 + L * (1 + EPS) + 0.012)):
    bar = trimesh.creation.box(extents=(0.016, 0.16, 0.024)); bar.apply_translation([xb, 0, 0])
    meshes[f"bar{i}"] = (to_view(np.asarray(bar.vertices)), np.asarray(bar.faces))
    bodies.append({"mesh": f"bar{i}", "color": "#2b3137", "material": {"metalness": 0.5, "roughness": 0.4}})
COL = ["#c9a35a", "#c9a35a", "#d8dde2", "#d8dde2"]
RF = args.record_fps
scene = {
    "title": "Robot hand on four strings",
    "caption": f"Four nylon strings (radius 3 / 2.5 / 2 / 1.5 mm, {L} m, {V} vertices each) pinned at both ends and tensioned to {args.tension} N "
               f"(fundamentals {', '.join(f'{v:.1f}' for v in f1)} Hz: slow-motion pitch). A Shadow Hand (MuJoCo Menagerie, unmodified) rests three "
               f"fingertips on a string while the index finger sweeps across twice, then rakes across with straight fingers; each link is a kinematic "
               f"signed-distance grid of its drawn mesh and each string collides at its drawn radius. Cosserat rod solver, h = 1/{FPS * SUB} s, {ITR} iterations.",
    "legend": [{"color": "#2a2a2b", "text": f"driven: Shadow Hand, {M} collision links"},
               {"color": "#c9a35a", "text": "simulated: 4 strings, pinned both ends"}],
    "fps": RF, "frames": F, "speed": 0.5,
    "camera": {"pos": [0.72, 0.42, 0.78], "target": [0.12, 0.0, 0.0]},
    "shadow": {"center": [0.1, 0, 0], "extent": 0.7},
    "floor": None, "status": rec.status,
    "ropes": [{"pos": to_view(rp[:, k, ::2]), "radius": float(RAD[k]), "color": COL[k],      # every 2nd vertex (5 mm) is plenty for the tube
               "material": {"metalness": 0.8, "roughness": 0.35}, "radial": 12, "samples": 420} for k in range(NS)],
    "meshes": meshes, "bodies": bodies,
}
out = pathlib.Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
np.savez_compressed(out.with_suffix(".npz"), ropes=rp, status=np.array(rec.status))
nb = write_html(out, scene)
print(f"closest string-hand gap over the run per string: {np.round(worst_gap * 1000, 2)} mm (negative = inside the drawn radius)")
print(f"wrote {out} ({nb / 1e6:.2f} MB), {F} frames, {time.time() - t0:.1f}s total")
