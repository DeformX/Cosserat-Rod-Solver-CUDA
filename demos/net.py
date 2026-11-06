"""Demo 2: a 10 x 10 rope net (two crossing layers, every rope pinned at both ends) catches a falling Stanford bunny.

The bunny is a free rigid body integrated here from the wrench the rope solver reports on its
signed-distance mesh; the exchange is sub-cycled K times per frame for stability (AGENTS 5c).
"""
import argparse
import pathlib
import time

import numpy as np
import trimesh
import ropesim
from scipy.ndimage import map_coordinates

from assets import close_holes, bake_closed
from common import DEMO_DIR, Recorder, device_positions, device_velocities, quat_from_axis_angle
from viewer import write_html, to_view, quat_to_view, quat_mul, quat_rotate

p = argparse.ArgumentParser()
p.add_argument("--bunny", default=f"{DEMO_DIR}/assets/bunny.obj")
p.add_argument("--out", default=f"{DEMO_DIR}/net.html")
p.add_argument("--duration", type=float, default=5.0)
p.add_argument("--substeps", type=int, default=400, help="per frame: h = 1/(fps*substeps) = 1/24000 s")
p.add_argument("--itr", type=int, default=16)
p.add_argument("--fps", type=int, default=60)
p.add_argument("--exchange", type=int, default=10, help="pose/wrench exchanges per frame")
p.add_argument("--height", type=float, default=0.75, help="bunny height [m]")
p.add_argument("--density", type=float, default=150.0, help="bunny density [kg/m^3]")
p.add_argument("--drop", type=float, default=0.8, help="height of the bunny's lowest point above the net [m]")
p.add_argument("--span", type=float, default=2.4)
p.add_argument("--spacing", type=float, default=0.14)
p.add_argument("--sdf_spacing", type=float, default=0.015)
p.add_argument("--ea", type=float, default=3e3)
p.add_argument("--body_damping", type=float, default=4.0, help="bunny velocity decay [1/s] while in contact (a soft toy is inelastic)")
p.add_argument("--drag", type=float, default=1.5, help="rope velocity decay rate [1/s] (the dissipation knob; a real net damps through crossing friction)")
p.add_argument("--settle", type=float, default=3.0, help="seconds the net hangs under gravity (high drag) before the bunny is released")
args = p.parse_args()

n, S, SP, R, V = 10, args.span, args.spacing, 0.01, 91
RHO, EA, EI = 0.2, args.ea, 1e-3
FPS, SUB, ITR, K = args.fps, args.substeps, args.itr, args.exchange
DT = 1.0 / FPS; TAU = DT / K; SUBK = SUB // K
GAP = 2 * R + 0.008
G = np.array([0, 0, -9.81])

# ---- net ----
N = 2 * n
pos = np.zeros((N, V, 3), np.float32)
u = np.linspace(-S / 2, S / 2, V)
for j in range(n):
    yj = SP * (j - (n - 1) / 2)
    pos[j, :, 0] = u; pos[j, :, 1] = yj; pos[j, :, 2] = 0.0
    xi = SP * (j - (n - 1) / 2)
    pos[n + j, :, 0] = xi; pos[n + j, :, 1] = u; pos[n + j, :, 2] = -GAP
pinned = np.zeros(V, bool); pinned[0] = pinned[-1] = True

batch = ropesim.RopeBatch.from_positions(pos, density=RHO, k_stretch=EA, k_bend=EI, max_h=TAU / SUBK, num_itr=ITR, pinned=pinned)
batch.set_gravity(np.array([0.0, 0.0, -9.81]))        # the solver's default is y-down
batch.damping = 0.0
batch.set_rope_contact(True, radius=0.9 * R, barrier=0.2 * R, k_collision=0.25, mu=0.5, k_friction=1e7, group_ropes=N)
batch.set_contact(radius=R, mu_s=0.6, mu_k=0.5, v_slip=1e-3, kappa=2)

# ---- bunny: closed mesh, z-up, centred at its centre of mass ----
m0 = trimesh.load(args.bunny, force="mesh", process=True)
mesh = close_holes(m0)
v = np.asarray(mesh.vertices, np.float64)
v = np.stack([v[:, 0], -v[:, 2], v[:, 1]], 1)                  # obj is y-up -> z-up
v *= args.height / (v[:, 2].max() - v[:, 2].min())
mesh = trimesh.Trimesh(vertices=v, faces=mesh.faces, process=False)
com = mesh.center_mass
v_body = v - com
mesh = trimesh.Trimesh(vertices=v_body, faces=mesh.faces, process=False)
mass = args.density * mesh.volume
I_body = args.density * mesh.moment_inertia
print(f"bunny: {len(v_body)} verts, height {args.height} m, volume {mesh.volume:.4f} m^3, mass {mass:.2f} kg, "
      f"I diag {np.round(np.diag(I_body), 3)}, z range [{v_body[:, 2].min():.3f}, {v_body[:, 2].max():.3f}]")
t0 = time.time()
grid = bake_closed(v_body, mesh.faces, spacing=args.sdf_spacing, margin=R + 4 * args.sdf_spacing)
print(f"sdf {grid.values.shape} in {time.time() - t0:.1f}s")

# rigid state
pb = np.array([0.03, 0.02, args.drop - v_body[:, 2].min()])
qb = quat_mul(quat_from_axis_angle([0, 0, 1], 0.4), quat_from_axis_angle([1, 0, 0], 0.1))
vb = np.zeros(3); wb = np.zeros(3)
idx = batch.add_mesh(grid.values, grid.origin, grid.spacing, pos=pb.astype(np.float32), quat=qb.astype(np.float32))
batch.enable_contact_forces(True)

# let the net hang: it starts straight, which is not its shape under gravity
t0 = time.time()
batch.drag = 6.0
for k in range(int(args.settle * FPS)):
    if k == int((args.settle - 0.5) * FPS):
        batch.drag = args.drag
    batch.step_free(DT, SUB)
P = device_positions(batch)
print(f"net settled {args.settle}s: max |v| {np.abs(device_velocities(batch)).max():.4f} m/s, lowest rope point {P[..., 2].min():.3f} m ({time.time() - t0:.1f}s)")

def rot_mat(q):
    return quat_rotate(q[None], np.eye(3)).T   # columns = rotated basis vectors

rec = Recorder()
min_gap = np.inf
frames = int(round(args.duration * FPS))
t0 = time.time()
for k in range(frames):
    Fsum = np.zeros(3)
    for e in range(K):
        p_next = pb + vb * TAU
        wn = np.linalg.norm(wb)
        q_next = quat_mul(quat_from_axis_angle(wb / wn, wn * TAU), qb) if wn > 1e-12 else qb
        q_next = q_next / np.linalg.norm(q_next)
        batch.set_mesh_poses(idx, p_next.astype(np.float32), q_next.astype(np.float32))
        batch.step_free(TAU, SUBK)
        W = np.asarray(batch.mesh_wrenches())[:, idx, :].sum(0)
        Fc, Tc = W[:3], W[3:]
        Rm = rot_mat(q_next)
        Iw = Rm @ I_body @ Rm.T
        vb = vb + (Fc / mass + G) * TAU
        wb = wb + np.linalg.solve(Iw, Tc - np.cross(wb, Iw @ wb)) * TAU
        if np.linalg.norm(Fc) > 1.0:                     # inelastic body: damp while touching the net
            vb = vb * np.exp(-args.body_damping * TAU); wb = wb * np.exp(-args.body_damping * TAU)
        pb, qb = p_next, q_next
        Fsum += Fc
    P = device_positions(batch)
    # penetration check: signed distance of every rope vertex from the bunny surface (trilinear sample, body frame)
    Rm = rot_mat(qb)
    loc = (P.reshape(-1, 3) - pb) @ Rm
    ijk = (loc - grid.origin) / grid.spacing
    phi = map_coordinates(grid.values, ijk.T, order=1, mode="nearest")
    inside = ((ijk >= 0) & (ijk <= np.array(grid.values.shape) - 1)).all(1)
    gap = (phi[inside] - R).min() if inside.any() else np.inf
    min_gap = min(min_gap, gap)
    status = f"bunny z {pb[2]:.2f} m · contact force {np.linalg.norm(Fsum) / K:.0f} N · closest rope {gap * 1000:+.1f} mm"
    rec.frame(P, {"bunny": (pb, qb)}, status)
    if k % (FPS // 4) == 0 or not np.isfinite(P).all():
        print(f"t={k * DT:5.2f}s  bunny z {pb[2]:.3f}  v {np.round(vb, 2)}  |w| {np.linalg.norm(wb):.2f}  F {np.linalg.norm(Fsum) / K:7.1f} N  "
              f"rope z min {P[..., 2].min():.3f}  gap {gap * 1000:+.1f} mm  finite {np.isfinite(P).all()}  ({time.time() - t0:.1f}s)", flush=True)
    if not np.isfinite(P).all() or not np.isfinite(pb).all() or pb[2] < -5:
        print("DIVERGED / fell through"); break

rp = rec.rope_array()
F = rp.shape[0]
pp, qq = rec.body_poses("bunny")
poses = np.concatenate([to_view(pp), quat_to_view(qq)], axis=1).astype(np.float32)
scene = {
    "title": "Bunny on a rope net",
    "caption": f"Two layers of 10 ropes (radius {R * 100:.0f} cm, {S:.1f} m span, {SP * 100:.0f} cm spacing) pinned at both ends, held together only by "
               f"rope–rope contact and friction. The {args.height:.2f} m Stanford bunny ({mass:.0f} kg) is a free rigid body driven by the wrench "
               f"the rope solver reports on its signed-distance mesh ({K} exchanges per frame, h = 1/{FPS * SUB} s, {ITR} iterations).",
    "legend": [{"color": "#2e86de", "text": "simulated: 10 ropes along x"},
               {"color": "#e4572e", "text": "simulated: 10 ropes along y, one diameter below"},
               {"color": "#efe6d8", "text": "rigid body: Stanford bunny (two-way coupled)"}],
    "fps": FPS, "frames": F,
    "camera": {"pos": [2.6, 1.7, 3.3], "target": [0.0, -0.1, 0.0]},
    "shadow": {"center": [0, 0, 0], "extent": 1.8},
    "floor": None, "status": rec.status,
    "ropes": [{"pos": to_view(rp[:, i]), "radius": R, "color": "#2e86de" if i < n else "#e4572e"} for i in range(N)],
    "meshes": {"bunny": (to_view(v_body), np.asarray(mesh.faces))},
    "bodies": [{"mesh": "bunny", "color": "#efe6d8", "material": {"roughness": 0.85}, "poses": poses}],
}
out = pathlib.Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
np.savez_compressed(out.with_suffix(".npz"), ropes=rp, bunny_p=pp, bunny_q=qq, status=np.array(rec.status))
nb = write_html(out, scene)
print(f"closest rope-surface gap over the run: {min_gap * 1000:+.1f} mm (negative = penetration beyond the rope radius)")
print(f"wrote {out} ({nb / 1e6:.2f} MB), {F} frames, {time.time() - t0:.1f}s total")
