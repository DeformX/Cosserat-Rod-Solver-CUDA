"""Rod-mesh contact through signed-distance grids (ropesim.sdf + add_mesh):
the plane contact law evaluated on an SDF. Closed-form shapes give analytic
checks (box floor = plane, posed incline Coulomb, capstan hold/slip, thin
wall); a baked triangle mesh is checked against its closed form.

Rope: 21 vertices x 5 cm, rho 0.117 kg/m, EA 11.5 kN, EI 4.5e-3 (DeformY's
rope_A fit), radius 6.35 mm, h = 1/6000 s, 16 iterations, kappa = 2.
"""
import numpy as np
import pytest
import torch

import ropesim
from ropesim import sdf

NV, SEG, RHO, EA, EI = 21, 0.05, 0.117, 11473.0, 4.48e-3
R = 0.00635
H, ITR, G = 1.0 / 6000.0, 16, 9.81
KAPPA, VSLIP = 2.0, 1e-3
PEN0 = G * H * H / KAPPA ** 2              # rest penetration under full weight


def quat_y(angle):
    """Rotation about +y by `angle` as (x, y, z, w)."""
    return np.array([0.0, np.sin(angle / 2), 0.0, np.cos(angle / 2)], np.float32)


def box_floor_grid(spacing=1 / 128, x_hi=1.3125):
    """Box of half extents (3, 1, 0.125) whose top face is the object plane z = 0 (the object
    origin sits ON the face: the rest gap is only ~70 float32 steps of a coordinate near 0.01 m,
    so the frame origin must be near the contact surface). Lattice nodes are float32-exact. The
    lattice covers only the part of the face the rope can reach plus margin, not the whole box."""
    return sdf.from_function(sdf.sdf_box((3.0, 1.0, 0.125), (0, 0, -0.125)),
                             (-0.3125, -0.3125, -0.1875), (x_hi, 0.3125, 0.1875), spacing), \
        np.array([0.0, 0.0, 0.0], np.float32)


def rope_sim(gravity=(0, 0, -G), mu_s=0.5, mu_k=None, num_itr=ITR, pinned=False):
    r = ropesim.RopeSim(num_verts=NV, seg_len=SEG, dir=np.array([1.0, 0, 0]), density=RHO,
                        k_stretch=EA, k_bend=EI, max_h=H, num_itr=num_itr)
    r.damping = 0.0
    r.drag = 0.0
    r.set_gravity(np.asarray(gravity, float))
    r.set_contact(radius=R, mu_s=mu_s, mu_k=mu_k, v_slip=VSLIP, kappa=KAPPA)
    if not pinned:
        r.unpin_root()
    return r


def free_batch(pos, pinned=None, gravity=(0, 0, -G), mu_s=0.5, mu_k=None, drag=0.0):
    """RopeBatch from explicit shapes [N, V, 3]; all vertices free unless pinned."""
    pos = np.asarray(pos, np.float32)
    N = pos.shape[0]
    pin = np.zeros((N, NV), bool) if pinned is None else pinned
    b = ropesim.RopeBatch.from_positions(pos, RHO, EA, EI, H, ITR, pinned=pin)
    b.damping = 0.0
    b.drag = drag
    b.set_gravity(np.asarray(gravity, float))
    b.set_contact(radius=R, mu_s=mu_s, mu_k=mu_k, v_slip=VSLIP, kappa=KAPPA)
    return b


P_ = lambda b: torch.as_tensor(b.positions()).cpu().numpy()
V_ = lambda b: torch.as_tensor(b.velocities()).cpu().numpy()


# ----------------------------------------------------------------------------- floor


def test_box_floor_matches_plane():
    """On the box's top face the SDF is exactly z, so the mesh path must reproduce the plane path."""
    g, box_pos = box_floor_grid()
    w = -1 / 128                                      # surface at z = w (exact): the rope drops 1.5 mm, then rests
    out = []
    for use_mesh in (False, True):
        r = rope_sim()
        if use_mesh:
            r.add_mesh(*g.as_args(), pos=box_pos + np.array([0, 0, w], np.float32))
        else:
            r.set_planes(np.array([[0.0, 0.0, 1.0, w]]), radius=R, mu_s=0.5, v_slip=VSLIP, kappa=KAPPA)
        for _ in range(60):
            r.advance(1 / 60, substeps=100)
        out.append(r.positions())
        m = 1.0 / r.inv_masses()
        pen = np.maximum(-(out[-1][:, 2] - w - R), 0)
        k = m * (KAPPA / H) ** 2
        assert (k * pen).sum() / (m.sum() * G) == pytest.approx(1.0, abs=5e-3)
        assert pen.max() == pytest.approx(PEN0, rel=0.05)
        assert np.abs(r.velocities()).max() < 1e-3
    assert np.abs(out[0] - out[1]).max() < 1e-6     # same contact term, same answer (rounding only)


@pytest.mark.parametrize("mu,expect_slide", [(0.5, False), (0.2, True)])
def test_posed_box_incline_coulomb(mu, expect_slide):
    """Box rotated 20 deg about y (pose quaternion): its top face is an incline under vertical gravity."""
    th = np.radians(20.0)
    n = np.array([np.sin(th), 0.0, np.cos(th)])          # face normal in the world
    t = np.array([np.cos(th), 0.0, -np.sin(th)])          # downhill tangent
    g, box_pos = box_floor_grid(x_hi=2.6875)                # the sliding rope travels ~1.3 m
    q = quat_y(th)
    # rotate the object offset too, so the top face passes through the world origin
    Rm = np.array([[np.cos(th), 0, np.sin(th)], [0, 1, 0], [-np.sin(th), 0, np.cos(th)]])
    s = np.arange(NV) * SEG                               # inside the lattice's x range [-0.31, 1.31]
    pos = s[:, None] * t[None] + (R - PEN0 * np.cos(th)) * n[None]
    b = free_batch(pos[None], mu_s=mu, mu_k=mu)
    b.add_mesh(*g.as_args(), pos=(Rm @ box_pos).astype(np.float32), quat=q)
    m = 1.0 / b.inv_masses()[0]
    ts, vt = [], []
    for f in range(72):
        b.step_free(1 / 60, 100)
        ts.append((f + 1) / 60)
        vt.append((m[:, None] * V_(b)[0]).sum(0) @ t / m.sum())
    ts, vt = np.array(ts), np.array(vt)
    a = np.polyfit(ts[ts > 0.3], vt[ts > 0.3], 1)[0]
    if expect_slide:
        assert a == pytest.approx(G * (np.sin(th) - mu * np.cos(th)), rel=0.01)
    else:
        assert abs(a) < 1e-3 and abs(vt[-1]) < 2e-3   # creep ~ eps*tan/mu
    # stayed on the face: no sideways drift, no penetration beyond the rest value
    P = P_(b)[0]
    assert np.abs(P[:, 1]).max() < 1e-3
    assert (P @ n).min() > R - 3 * PEN0 - 1e-6


# ----------------------------------------------------------------------------- capstan


def capstan_rope(L1, L2, rc, zc):
    """Vertices along: up side 1 (x = -rc), over the top of a cylinder (axis y at height zc), down side 2."""
    s = np.arange(NV) * SEG
    P = np.zeros((NV, 3))
    arc = np.pi * rc
    for i, si in enumerate(s):
        if si < L1:
            P[i] = (-rc, 0, zc - (L1 - si))
        elif si < L1 + arc:
            phi = np.pi - (si - L1) / rc
            P[i] = (rc * np.cos(phi), 0, zc + rc * np.sin(phi))
        else:
            P[i] = (rc, 0, zc - (si - L1 - arc))
    return P


@pytest.mark.parametrize("mu,holds", [(0.5, True), (0.05, False)])
def test_capstan_hold_slip(mu, holds):
    """Rope over a cylinder with hanging lengths 1:2. Capstan: holds iff 2 <= exp(mu*pi)
    (mu 0.5 -> 4.8 holds; mu 0.05 -> 1.17 slips)."""
    rcyl, zc = 0.10, 0.5
    rc = rcyl + R - PEN0
    arc = np.pi * rc
    L1 = (1.0 - arc) / 3
    L2 = 2 * L1
    g = sdf.from_function(sdf.sdf_cylinder(rcyl, axis=1), (-0.2, -0.1, -0.2), (0.2, 0.1, 0.2), 0.004)
    P0 = capstan_rope(L1, L2, rc, zc)
    b = free_batch(P0[None], mu_s=mu, mu_k=mu)
    b.add_mesh(*g.as_args(), pos=np.array([0, 0, zc], np.float32))
    T = 1.0 if holds else 0.3
    for _ in range(int(T * 60)):
        b.step_free(1 / 60, 100)
    P = P_(b)[0]
    assert np.isfinite(P).all()
    moved = np.linalg.norm(P - P0, axis=1).max()
    d_axis = np.hypot(P[:, 0], P[:, 2] - zc).min()
    assert d_axis > rcyl + R - 1e-3                     # never inside the cylinder
    if holds:
        assert moved < 0.01                              # creep only (regularized friction)
    else:
        assert moved > 0.03 and P[-1, 2] < P0[-1, 2] - 0.03   # the heavy side runs down


# ----------------------------------------------------------------------------- shapes


def test_sphere_drape_settles():
    """A rope dropped over a sphere drapes over it and never enters it. The sphere must be larger
    than rope_A's bending length (EI / rho g)^(1/3) = 16 cm, or the rope tips off like a plank."""
    rs, c = 0.25, np.array([0.5, 0.0, 0.4])
    g = sdf.from_function(sdf.sdf_sphere(rs), (-0.35, -0.35, -0.35), (0.35, 0.35, 0.35), 0.005)
    pos = np.zeros((1, NV, 3)); pos[0, :, 0] = np.arange(NV) * SEG; pos[0, :, 2] = c[2] + rs + R + 0.02
    b = free_batch(pos, drag=2.0)                        # strong air damping: a stiff rope rocks for a while
    b.add_mesh(*g.as_args(), pos=c.astype(np.float32))
    for _ in range(240):
        b.step_free(1 / 60, 100)
    P = P_(b)[0]
    assert np.isfinite(P).all()
    d = np.linalg.norm(P - c, axis=1) - rs
    assert d.min() > R - 1e-3                            # never inside
    assert (d < R + 2e-3).sum() >= 10                    # most of it lies on the sphere
    assert abs(P[:, 0].mean() - c[0]) < 0.05             # still centred
    assert P[[0, -1], 2].max() < c[2]                    # both ends hang below the centre
    assert np.abs(V_(b)).max() < 0.05


def test_bowl_settles_inside():
    """A rope dropped into a spherical bowl stays inside and comes to rest in its lower part
    (a 1 m rope cannot lie flat in a 1.2 m bowl: it curls into a ring a little up the wall)."""
    rb, c = 0.6, np.array([0.5, 0.0, 0.6])
    g = sdf.from_function(sdf.sdf_bowl(rb), (-0.65, -0.65, -0.65), (0.65, 0.65, 0.65), 0.01)
    pos = np.zeros((1, NV, 3)); pos[0, :, 0] = np.arange(NV) * SEG; pos[0, :, 2] = c[2]
    assert (rb - np.linalg.norm(pos[0] - c, axis=1)).min() > R + 0.05   # starts well clear of the wall
    b = free_batch(pos, drag=0.3, mu_s=0.1)              # mu 0.1: rests within atan(0.1) = 6 deg of the bottom
    b.add_mesh(*g.as_args(), pos=c.astype(np.float32))
    for _ in range(300):
        b.step_free(1 / 60, 100)
    P = P_(b)[0]
    assert np.isfinite(P).all()
    assert (rb - np.linalg.norm(P - c, axis=1)).min() > R - 1e-3
    assert P[:, 2].max() < c[2] - rb / 2                   # all of it in the lower part of the bowl
    assert np.linalg.norm(P[:, :2].mean(0) - c[:2]) < 0.05  # centred on the axis
    assert np.abs(V_(b)).max() < 0.05


def test_thin_wall_no_tunnelling():
    """A 5 mm wall stops a rope flying at 8 m/s (no gravity)."""
    g = sdf.from_function(sdf.sdf_box((0.0025, 0.6, 0.6)), (-0.06, -0.65, -0.65), (0.06, 0.65, 0.65), 0.0025)
    pos = np.zeros((1, NV, 3)); pos[0, :, 1] = np.arange(NV) * SEG - 0.5; pos[0, :, 0] = 0.3
    b = free_batch(pos, gravity=(0, 0, 0))
    b.add_mesh(*g.as_args(), pos=np.array([0.5, 0, 0], np.float32))
    v = np.zeros((1, NV, 3), np.float32); v[0, :, 0] = 8.0
    b.set_velocities(v)
    xmax = 0.0
    for _ in range(30):
        b.step_free(1 / 120, 50)
        P = P_(b)[0]
        xmax = max(xmax, P[:, 0].max())
    assert np.isfinite(P).all()
    assert xmax < 0.5 - 0.0025 - R + 1.5e-3            # never past the near face (1.5 mm dynamic penetration allowed)
    assert P[:, 0].max() < 0.49                          # it bounced back


# ----------------------------------------------------------------------------- baked mesh


def test_baked_mesh_matches_closed_form():
    """Warp bake of an icosphere agrees with the analytic sphere, and the rope lands the same on both."""
    pytest.importorskip("warp")
    trimesh = pytest.importorskip("trimesh")
    rs, c = 0.15, np.array([0.5, 0.0, 0.3])
    m = trimesh.creation.icosphere(subdivisions=4, radius=rs)
    baked = sdf.bake_mesh(m.vertices, m.faces, spacing=0.005, margin=0.1)
    ana = np.linalg.norm(baked.nodes().reshape(-1, 3), axis=1) - rs
    assert np.abs(baked.values.reshape(-1) - ana).max() < 5e-4     # tessellation only
    assert (baked.values < 0).sum() > 0 and (baked.values > 0).sum() > 0
    outs = []
    for grid in (sdf.from_function(sdf.sdf_sphere(rs), baked.origin, baked.origin + baked.spacing * (np.array(baked.shape) - 1), baked.spacing), baked):
        pos = np.zeros((1, NV, 3)); pos[0, :, 0] = np.arange(NV) * SEG; pos[0, :, 2] = c[2] + rs + R + 0.02
        b = free_batch(pos, drag=0.3)
        b.add_mesh(*grid.as_args(), pos=c.astype(np.float32))
        for _ in range(120):
            b.step_free(1 / 60, 100)
        outs.append(P_(b)[0])
    assert np.abs(outs[0] - outs[1]).max() < 3e-3


# ----------------------------------------------------------------------------- batch / API


def test_batch_mesh_env_independent():
    """Per-env poses: env 0 is bitwise identical alone or beside envs whose meshes sit elsewhere."""
    g, box_pos = box_floor_grid()
    def run(n):
        pos = np.zeros((n, NV, 3)); pos[:, :, 0] = np.arange(NV) * SEG
        pos[:, :, 2] = 0.2; pos[:, :, 1] = 3.0 * np.arange(n)[:, None]
        b = free_batch(pos, drag=0.1)
        p = np.tile(box_pos, (n, 1)); p[:, 1] += 3.0 * np.arange(n); p[:, 2] -= 0.01 * np.arange(n)
        q = np.tile(np.array([0, 0, 0, 1], np.float32), (n, 1))
        b.add_mesh(*g.as_args(), pos=p.astype(np.float32), quat=q)
        for _ in range(30):
            b.step_free(1 / 60, 100)
        return P_(b)[0]
    one = run(1)
    assert (one[:, 2] < 0.05).all()                      # it reached the floor
    assert np.array_equal(one, run(4))


def test_mesh_api_and_inert_cases():
    g, box_pos = box_floor_grid(spacing=0.05)
    r = ropesim.RopeSim(num_verts=NV, seg_len=SEG, dir=np.array([1.0, 0, 0]), density=RHO,
                        k_stretch=EA, k_bend=EI, max_h=H, num_itr=ITR)
    with pytest.raises(ValueError):
        r.add_mesh(*g.as_args())                          # material not set
    r.set_contact(radius=R)
    with pytest.raises(ValueError):
        r.add_mesh(g.values[0], g.origin, g.spacing)      # not 3-D
    with pytest.raises(ValueError):
        r.add_mesh(g.values, g.origin, 0.0)
    bad = g.values.copy(); bad[0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        r.add_mesh(bad, g.origin, g.spacing)
    with pytest.raises(ValueError):
        r.add_mesh(*g.as_args(), quat=np.zeros(4, np.float32))
    for _ in range(32):
        r.add_mesh(*g.as_args(), pos=np.array([0, 0, -50.0], np.float32))
    with pytest.raises(ValueError):
        r.add_mesh(*g.as_args())                          # 9th
    with pytest.raises(ValueError):
        r.set_mesh_pose(32, np.zeros(3, np.float32), np.array([0, 0, 0, 1], np.float32))

    def run(setup):
        rr = ropesim.RopeSim(num_verts=NV, seg_len=SEG, dir=np.array([1.0, 0, 0]), density=RHO,
                             k_stretch=EA, k_bend=EI, max_h=H, num_itr=8)
        rr.damping = 0.0
        rr.set_contact(radius=R)
        setup(rr)
        for f in range(30):
            t = f / 60
            rr.set_root(np.array([0.1 * np.sin(4 * t), 0.0, 0.0]), np.array([0.4 * np.cos(4 * t), 0.0, 0.0]))
            rr.advance(1 / 60, substeps=100)
        return rr.positions()
    ref = run(lambda rr: None)
    # a mesh whose lattice the rope never enters is bitwise inert ...
    assert np.array_equal(ref, run(lambda rr: rr.add_mesh(*g.as_args(), pos=np.array([0, 0, -50.0], np.float32))))
    # ... and so is a cleared one
    def add_clear(rr):
        rr.add_mesh(*g.as_args(), pos=np.array([0, 0, -0.1 - R + 0.002], np.float32))
        rr.clear_meshes()
    assert np.array_equal(ref, run(add_clear))


# ----------------------------------------------------------------------------- convex edges


def drag_over_edge(round_r, T=3.0, fps=120):
    """Rope on a 60 cm high slab (edges rounded with radius round_r), root driven +x at 0.4 m/s over
    the far edge while the tail hangs over the near edge. Returns (peak strain, min vertex phi)."""
    TH = 0.6
    box = sdf.sdf_box((0.5, 0.4, 0.05), (0, 0, -0.05), round=round_r)
    grid = sdf.from_function(box, (-0.65, -0.55, -0.25), (0.65, 0.55, 0.15), 0.005)
    pos = np.zeros((1, NV, 3), np.float32)
    pos[0, :, 0] = -0.2 - np.arange(NV) * SEG            # root at -0.2, rope trailing past the near edge
    pos[0, :, 2] = TH + R + 0.002
    pin = np.zeros((1, NV), bool); pin[0, 0] = True
    b = free_batch(pos, pinned=pin, mu_s=0.3, mu_k=0.3, drag=0.03)
    b.add_mesh(*grid.as_args(), pos=np.array([0, 0, TH], np.float32))
    strain, phimin = 0.0, np.inf
    for f in range(int(T * fps)):
        x0, x1 = (-0.2 + 0.4 * max(0.0, min(f / fps - 0.5, 3.0))), (-0.2 + 0.4 * max(0.0, min((f + 1) / fps - 0.5, 3.0)))
        p0 = torch.tensor([[x0, 0.0, TH + R + 0.002]], device="cuda"); p1 = torch.tensor([[x1, 0.0, TH + R + 0.002]], device="cuda")
        b.step(p0, (p1 - p0) * fps, 1 / fps, 100 * 60 // fps, root_pos_end=p1)
        P = P_(b)[0]
        strain = max(strain, np.linalg.norm(np.diff(P, axis=0), axis=1).max() / SEG - 1)
        phimin = min(phimin, box(P - np.array([0, 0, TH])).min())
    assert np.isfinite(P).all()
    return strain, phimin


def test_rounded_edge_slides_sharp_edge_snags():
    """Contact acts at vertices, so a sharp 90-deg edge lies below the rope discretization: a rope
    dragged over it from slightly below snags (friction holds the vertex on the side face, tension
    runs away). Rounding the edge to 2 cm (~ half the segment length) makes it slide normally."""
    s_round, phi_round = drag_over_edge(0.02)
    assert s_round < 0.02 and phi_round > R - 1e-3        # ordinary tension, vertices stay outside
    s_sharp, _ = drag_over_edge(0.0)
    assert s_sharp > 0.3                                  # the documented limit (measured ~0.9 strain)


# ----------------------------------------------------------------------------- review probes (2026-10-12)


def test_mesh_far_from_origin_matches_plane():
    """The SDF path keeps pos and dx separate through the pose transform, so a wall at world x = 1 m
    (where one float32 step of the position is 1.75 rest sinks) rests like the plane path instead of
    chattering by one ulp per substep (the review found that with pos + dx rounded first)."""
    g = sdf.from_function(sdf.sdf_box((0.125, 3.0, 2.0), (0.125, 0, 0)), (-0.1875, -0.3125, -0.3125), (0.1875, 0.3125, 1.3125), 1 / 128)   # wall occupies x >= X0
    X0 = 1.0
    out = []
    for use_mesh in (False, True):
        pos = np.zeros((1, NV, 3), np.float32); pos[0, :, 0] = X0 - R - 2e-3; pos[0, :, 2] = 0.1 + np.arange(NV) * SEG
        phi, inside = g.sample(pos[0] - np.array([X0, 0, 0]))   # scene self-check: in the lattice, outside the wall
        assert inside.all() and (phi > R).all()
        b = free_batch(pos, gravity=(G, 0, 0), drag=0.0)         # gravity pushes the rope into the wall x <= X0
        if use_mesh:
            b.add_mesh(*g.as_args(), pos=np.array([X0, 0, 0], np.float32))
        else:
            b.set_planes(np.array([[-1.0, 0.0, 0.0, -X0]], np.float32), radius=R, mu_s=0.5, v_slip=VSLIP, kappa=KAPPA)
        for _ in range(60):
            b.step_free(1 / 60, 100)
        out.append(P_(b)[0])
        assert np.abs(V_(b)).max() < 1e-3                     # at rest (the plane path measures ~8e-5 here)
    assert np.abs(out[0] - out[1]).max() < 1e-6              # same gap arithmetic up to rounding (not bitwise: interpolated vs direct)


def test_per_env_poses_and_pose_update():
    """Four envs with floors at different heights through per-env poses, each rope rests at its own
    floor + R - PEN0; then set_mesh_poses lowers every floor 2 cm and the ropes follow."""
    g, box_pos = box_floor_grid()
    n = 4
    pos = np.zeros((n, NV, 3)); pos[:, :, 0] = np.arange(NV) * SEG; pos[:, :, 2] = 0.05
    b = free_batch(pos, drag=0.3)
    zf = -0.02 * np.arange(n)
    p = np.tile(box_pos, (n, 1)); p[:, 2] += zf
    q = np.tile(np.array([0, 0, 0, 1], np.float32), (n, 1))
    b.add_mesh(*g.as_args(), pos=p.astype(np.float32), quat=q)
    for _ in range(90):
        b.step_free(1 / 60, 100)
    P = P_(b)
    for e in range(n):
        assert P[e, :, 2].min() == pytest.approx(zf[e] + R - PEN0, abs=3e-8)
    p[:, 2] -= 0.02
    b.set_mesh_poses(0, p.astype(np.float32), q)
    for _ in range(90):
        b.step_free(1 / 60, 100)
    P = P_(b)
    for e in range(n):
        assert P[e, :, 2].min() == pytest.approx(zf[e] - 0.02 + R - PEN0, abs=3e-8)


def test_set_planes_empty_keeps_material():
    """set_planes([]) only switches the planes off; the material a mesh is using stays."""
    g, box_pos = box_floor_grid()
    r = rope_sim()
    r.add_mesh(*g.as_args(), pos=box_pos + np.array([0, 0, -1 / 128], np.float32))
    r.set_planes(np.zeros((0, 4), np.float32))
    for _ in range(60):
        r.advance(1 / 60, substeps=100)
    assert r.positions()[:, 2].min() == pytest.approx(-1 / 128 + R - PEN0, abs=3e-8)


def test_grid_sample_matches_closed_form():
    """SdfGrid.sample() reproduces the lattice interpolation (exact for the box's planar faces) and
    reports points outside the lattice as nan/not inside."""
    g, _ = box_floor_grid()
    pts = np.array([[0.5, 0.0, 0.02], [0.5, 0.1, -0.03], [0.5, 0.0, 0.5], [2.0, 0.0, 0.0]])
    phi, inside = g.sample(pts)
    assert inside.tolist() == [True, True, False, False]
    assert np.allclose(phi[:2], sdf.sdf_box((3.0, 1.0, 0.125), (0, 0, -0.125))(pts[:2]), atol=1e-6)
    assert np.isnan(phi[2:]).all()
