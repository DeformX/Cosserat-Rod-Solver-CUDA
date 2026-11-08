"""Twisted cable bundle: 12 ropes whose ends sit on two fixed, facing circles that counter-rotate in place.

The ropes start straight and slightly pre-tensioned; twisting pulls them into a bundle and stretches them (soft EA,
the twist stops when the mean strain reaches the cap); hold; the ends unwind.
Both rope ends are pinned; the pins are driven kinematically (velocity per frame). Nothing else is drawn.
Writes a standalone HTML player (scene/viewer.py).
"""
import argparse
import pathlib
import time

import numpy as np
import ropesim

from scene.common import (DEMO_DIR, Recorder, check_step, device_positions, device_velocities,
                          drive_pinned, require_finite, segment_strain, smoothstep)
from scene.viewer import write_html, to_view

p = argparse.ArgumentParser()
p.add_argument("--out", default=f"{DEMO_DIR}/twist.html")
p.add_argument("--turns", type=float, default=5.0, help="relative turns between the end circles (the strain cap stops it earlier)")
p.add_argument("--slack", type=float, default=-0.01, help="rest length / end distance - 1 (negative = pre-tensioned straight ropes)")
p.add_argument("--ea", type=float, default=120.0, help="axial stiffness [N]: soft, so the ropes stretch into the bundle")
p.add_argument("--strain_cap", type=float, default=0.25, help="twisting stops as the mean strain approaches this")
p.add_argument("--k_friction", type=float, default=1e4, help="rope-rope friction spring [N/m]; 1e6+ makes a taut bundle chatter (stick-slip)")
p.add_argument("--duration", type=float, default=14.0)
p.add_argument("--substeps", type=int, default=800, help="per frame: h = 1/(fps*substeps) = 1/48000 s")
p.add_argument("--itr", type=int, default=16)
p.add_argument("--fps", type=int, default=60)
p.add_argument("--drag", type=float, default=0.5, help="rope velocity decay rate [1/s] during the recording")
p.add_argument("--settle", type=float, default=4.0, help="seconds of settling under gravity (high drag) before recording")
args = p.parse_args()
if min(args.fps, args.substeps, args.itr) < 1:
    p.error("--fps, --substeps and --itr must be positive")
if not np.isfinite(args.duration) or round(args.duration * args.fps) < 1:
    p.error("--duration must provide at least one recording frame")
if not np.isfinite(args.strain_cap) or args.strain_cap <= 0:
    p.error("--strain_cap must be finite and positive")

N, Rp, D, V, R = 12, 0.10, 0.8, 61, 0.008
RHO, EA, EI = 0.06, args.ea, 2e-4
FPS, SUB, ITR = args.fps, args.substeps, args.itr
DT = 1.0 / FPS
L = D * (1 + args.slack)
THETA = args.turns * 2 * np.pi
T_END = args.duration

T_TWIST, T_HOLD, T_UNWIND = 8.0, 1.5, 3.5

def twist_rate(t):
    """Programmed relative twist rate [rad/s] during the twist phase (smooth ramp in and out)."""
    if t < 1.0 or t > 1.0 + T_TWIST:
        return 0.0
    u = (t - 1.0) / T_TWIST
    return THETA / T_TWIST * 6 * u * (1 - u)   # integrates to THETA over the phase

phi = 2 * np.pi * np.arange(N) / N

def pin_targets(a):
    tg = np.zeros((N, V, 3), np.float32)
    tg[:, 0] = np.stack([np.full(N, -D / 2), Rp * np.cos(phi + a), Rp * np.sin(phi + a)], -1)
    tg[:, -1] = np.stack([np.full(N, D / 2), Rp * np.cos(phi - a), Rp * np.sin(phi - a)], -1)
    return tg

# initial shape: straight ropes of rest length L (the ends are then driven out to +-D/2, stretching them if L < D)
x = np.linspace(-L / 2, L / 2, V)
pos = np.zeros((N, V, 3), np.float32)
for i in range(N):
    pos[i, :, 0] = x
    pos[i, :, 1] = Rp * np.cos(phi[i])
    pos[i, :, 2] = Rp * np.sin(phi[i])
pinned = np.zeros(V, bool); pinned[0] = pinned[-1] = True
pin_mask = np.broadcast_to(pinned, (N, V))
rest = np.linalg.norm(np.diff(pos, axis=1), axis=-1)
print(f"rope rest length {rest.sum(1).mean():.4f} m (D {D}, slack {args.slack}), seg {rest.mean():.4f} m, EA {EA} N")

batch = ropesim.RopeBatch.from_positions(pos, density=RHO, k_stretch=EA, k_bend=EI, max_h=DT / SUB, num_itr=ITR, pinned=pinned)
batch.set_gravity(np.array([0.0, 0.0, -9.81]))        # the solver's default is y-down
batch.damping = 0.0
batch.set_rope_contact(True, radius=0.9 * R, barrier=0.2 * R, k_collision=0.25, mu=0.4, k_friction=args.k_friction, group_ropes=N)

# pull the ends out to +-D/2 over 0.5 s, then settle under gravity (high drag), pins still
t0 = time.time()
batch.drag = 3.0
T_PULL = 0.5
def ends_at(t):
    u = smoothstep(t / T_PULL)
    return pin_targets(0.0) * np.array([(1 - u) * L / D + u, 1, 1], np.float32)
for k in range(int(args.settle * FPS)):
    if k == int((args.settle - 1.0) * FPS):
        batch.drag = args.drag
    drive_pinned(batch, pin_mask, ends_at(k * DT), ends_at((k + 1) * DT), DT)
    batch.step_free(DT, SUB)
    check_step(batch, "settling", (k + 1) * DT)
P = device_positions(batch)
velocities = device_velocities(batch)
require_finite("settling", args.settle, positions=P, velocities=velocities)
print(f"settled {args.settle}s: max |v| {np.abs(velocities).max():.4f} m/s, mid z {P[:, V // 2, 2].min():.3f}..{P[:, V // 2, 2].max():.3f} m ({time.time() - t0:.1f}s)")

rec = Recorder()
frames = int(round(T_END * FPS))
t0 = time.time()
a = 0.0                      # current half-twist
a_hold = None; t_unwind0 = None
strain_mean = 0.0
for k in range(frames):
    t = k * DT
    if t <= 1.0 + T_TWIST:
        gate = float(np.clip(1 - (strain_mean - 0.3 * args.strain_cap) / (0.7 * args.strain_cap), 0.0, 1.0))
        a1 = a + 0.5 * twist_rate(t) * DT * gate
    else:
        if a_hold is None:
            a_hold, t_unwind0 = a, t + T_HOLD
        a1 = a_hold * (1 - smoothstep((t - t_unwind0) / T_UNWIND)) if t >= t_unwind0 else a_hold
    drive_pinned(batch, pin_mask, pin_targets(a), pin_targets(a1), DT)
    batch.step_free(DT, SUB)
    check_step(batch, "recording", t + DT)
    a = a1
    P = device_positions(batch)
    require_finite("recording", t + DT, positions=P)
    strain = segment_strain(P, rest)
    strain_mean = float(strain.mean())
    status = f"twist {2 * a / (2 * np.pi):.2f} turns · mean strain {100 * strain_mean:.1f}%"
    rec.frame(P, None, status)
    if k % FPS == 0:
        print(f"t={t:5.2f}s  twist {2 * a / (2 * np.pi):.2f} turns  strain mean {100 * strain_mean:.2f}% max {100 * strain.max():.2f}%  "
              f"finite {np.isfinite(P).all()}  ({time.time() - t0:.1f}s wall)", flush=True)

# ---- viewer ----
rp = rec.rope_array(frames)  # [F, N, V, 3]
F = rp.shape[0]
COLORS = ["#e4572e", "#17bebb", "#ffc914", "#2e86de"]
scene = {
    "title": "Twisted cable bundle",
    "caption": f"12 soft ropes (ρ {RHO} kg/m, EA {EA:.0f} N, radius {R * 1000:.0f} mm, {-args.slack * 100:.0f}% pre-strain); their ends sit on two fixed circles "
               f"that counter-rotate in place until the ropes have stretched to {args.strain_cap * 100:.0f}% mean strain, hold, and unwind. "
               f"Cosserat rod solver, rope–rope contact, h = 1/{FPS * SUB} s, {ITR} iterations.",
    "legend": [{"color": "#e4572e", "text": "simulated: 12 ropes, rope–rope contact; both ends driven on circles"}],
    "fps": FPS, "frames": F,
    "camera": {"pos": [0.95, 0.45, 1.35], "target": [0.0, -0.04, 0.0]},
    "shadow": {"center": [0, 0, 0], "extent": 0.8},
    "floor": None, "status": rec.status,
    "ropes": [{"pos": to_view(rp[:, i]), "radius": R, "color": COLORS[i % 4]} for i in range(N)],
    "meshes": {},
    "bodies": [],
}
out = pathlib.Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
np.savez_compressed(out.with_suffix(".npz"), ropes=rp, status=np.array(rec.status))
n = write_html(out, scene)
print(f"wrote {out} ({n / 1e6:.2f} MB), {F} frames, {time.time() - t0:.1f}s total")
