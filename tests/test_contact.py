"""Rod-plane contact: analytic checks (support force, Coulomb stick/slip,
static vs kinetic friction, impact) plus the contact-off / batch bitwise
guarantees and the scene helpers (pins, free roots).

Rope: 21 vertices x 5 cm, rho 0.117 kg/m, EA 11.5 kN, EI 4.5e-3 (DeformY's
rope_A fit), radius 6.35 mm, h = 1/6000 s, 16 iterations, contact kappa = 2
(the default).
"""
import numpy as np
import pytest
import torch

import ropesim

NV, SEG, RHO, EA, EI = 21, 0.05, 0.117, 11473.0, 4.48e-3
R = 0.00635
H, ITR, G = 1.0 / 6000.0, 16, 9.81
KAPPA, VSLIP = 2.0, 1e-3                   # contact stiffness k_i = m_i (kappa/h)^2 (defaults)
M_INT = RHO * SEG
PEN0 = G * H * H / KAPPA ** 2              # rest penetration under full weight (any vertex mass)


def free_rope(gravity, plane_w, mu_s, mu_k):
    """Unpinned rope along +x at z=0, one plane z >= plane_w."""
    r = ropesim.RopeSim(num_verts=NV, seg_len=SEG, dir=np.array([1.0, 0, 0]), density=RHO,
                        k_stretch=EA, k_bend=EI, max_h=H, num_itr=ITR)
    r.damping = 0.0
    r.drag = 0.0
    r.set_gravity(np.asarray(gravity, float))
    r.set_planes(np.array([[0.0, 0.0, 1.0, plane_w]]), radius=R, mu_s=mu_s, mu_k=mu_k, v_slip=VSLIP, kappa=KAPPA)
    r.unpin_root()
    return r


def com_vx(r, m):
    return (m[:, None] * r.velocities()).sum(0)[0] / m.sum()


def test_plane_rest_support_equals_weight():
    r = free_rope((0, 0, -G), -(R + 5e-4), 0.5, 0.5)
    m = 1.0 / r.inv_masses()
    for _ in range(60):
        r.advance(1 / 60, substeps=100)
    pen = np.maximum(-(r.positions()[:, 2] - (-(R + 5e-4)) - R), 0)
    k = m * (KAPPA / H) ** 2                       # per-vertex stiffness, end vertices included
    assert (k * pen).sum() / (m.sum() * G) == pytest.approx(1.0, abs=5e-3)
    assert pen.max() == pytest.approx(PEN0, rel=0.05)   # mass-consistent: every vertex sinks the same
    assert np.abs(r.velocities()).max() < 1e-3


@pytest.mark.parametrize("mu,expect_slide", [(0.5, False), (0.2, True)])
def test_plane_incline_coulomb(mu, expect_slide):
    th = np.radians(20.0)                              # tan = 0.364
    r = free_rope((G * np.sin(th), 0, -G * np.cos(th)), -(R - PEN0 * np.cos(th)), mu, mu)
    m = 1.0 / r.inv_masses()
    ts, vx = [], []
    for f in range(72):                                # 1.2 s
        r.advance(1 / 60, substeps=100)
        ts.append((f + 1) / 60)
        vx.append(com_vx(r, m))
    ts, vx = np.array(ts), np.array(vx)
    a = np.polyfit(ts[ts > 0.3], vx[ts > 0.3], 1)[0]
    if expect_slide:
        assert a == pytest.approx(G * (np.sin(th) - mu * np.cos(th)), rel=0.01)
    else:
        assert abs(a) < 1e-3 and abs(vx[-1]) < 2e-3   # creep ~ eps*tan/mu = 0.7 mm/s


def test_plane_static_vs_kinetic():
    """mu_s 0.5 > tan 20 deg > mu_k 0.2: holds from rest, keeps sliding once pushed."""
    th = np.radians(20.0)
    g = (G * np.sin(th), 0, -G * np.cos(th))
    w = -(R - PEN0 * np.cos(th))
    r = free_rope(g, w, 0.5, 0.2)
    m = 1.0 / r.inv_masses()
    for _ in range(60):
        r.advance(1 / 60, substeps=100)
    assert abs(com_vx(r, m)) < 2e-3
    r = free_rope(g, w, 0.5, 0.2)
    v = np.zeros((NV, 3)); v[:, 0] = 0.5
    r.set_velocities(v)
    ts, vs = [], []
    for f in range(60):
        r.advance(1 / 60, substeps=100)
        ts.append((f + 1) / 60)
        vs.append(com_vx(r, m))
    ts, vs = np.array(ts), np.array(vs)
    a = np.polyfit(ts[ts > 0.3], vs[ts > 0.3], 1)[0]
    assert a == pytest.approx(G * (np.sin(th) - 0.2 * np.cos(th)), rel=0.01)


def test_planes_without_contact_are_bitwise_inert():
    """A plane far below the rope changes nothing (contact-off path unchanged)."""
    def run(with_plane):
        r = ropesim.RopeSim(num_verts=NV, seg_len=SEG, dir=np.array([1.0, 0, 0]), density=RHO,
                            k_stretch=EA, k_bend=EI, max_h=H, num_itr=8)
        r.damping = 0.0
        if with_plane:
            r.set_planes(np.array([[0.0, 1.0, 0.0, -50.0]]), radius=R)
        for f in range(40):
            t = f / 60
            r.set_root(np.array([0.1 * np.sin(4 * t), 0.0, 0.0]), np.array([0.4 * np.cos(4 * t), 0.0, 0.0]))
            r.advance(1 / 60, substeps=100)
        return r.positions()
    assert np.array_equal(run(False), run(True))


def batch(roots, dir3=(0.0, 0.0, -1.0)):
    b = ropesim.RopeBatch(num_envs=len(roots), verts_per_rope=NV, root_pos=np.asarray(roots, np.float32),
                          seg_len=SEG, dir=np.asarray(dir3, float), density=RHO, k_stretch=EA, k_bend=EI,
                          max_h=H, num_itr=ITR, fix_root_orientation=False)
    b.damping = 0.0
    b.drag = 0.0
    b.set_gravity(np.array([0.0, 0.0, -G]))
    return b


def test_batch_plane_contact_matches_single_env():
    """Plane contact is vertex-local: env 0 is bitwise identical alone or in a batch."""
    def run(n):
        roots = [[3.0 * e, 0.0, 1.1] for e in range(n)]   # 1 m rope: tip starts 10 cm above the floor
        tilt = np.radians(2.0)                         # perfectly vertical = the end-on column case (AGENTS §5b)
        b = batch(roots, dir3=(np.sin(tilt), 0.0, -np.cos(tilt)))
        b.set_planes(np.array([[0.0, 0.0, 1.0, 0.0]], np.float32), radius=R, mu_s=0.5, mu_k=0.4)
        root = torch.tensor(roots, device="cuda", dtype=torch.float32)
        for f in range(60):                            # lower the root 0.3 m: the rope lands on the floor
            p0 = root + torch.tensor([0.0, 0.0, -0.3 * f / 60], device="cuda")
            p1 = root + torch.tensor([0.0, 0.0, -0.3 * (f + 1) / 60], device="cuda")
            v = torch.tensor([[0.0, 0.0, -0.3]] * n, device="cuda")
            b.step(p0.contiguous(), v.contiguous(), 1 / 60, 100, root_pos_end=p1.contiguous())
        return torch.as_tensor(b.positions()).cpu().numpy()[0]
    one = run(1)
    assert (one[:, 2] < 0.05).any() and (one[:, 2] > R - 1e-4).all()   # on the floor, not through it
    assert np.array_equal(one, run(4))


def test_rope_built_along_minus_x_is_finite():
    """Regression: configure() used to give NaN frames for a segment exactly along -x."""
    pos = np.zeros((1, NV, 3), np.float32)
    pos[0, :, 0] = -np.arange(NV) * SEG
    b = ropesim.RopeBatch.from_positions(pos, RHO, EA, EI, H, ITR)
    assert torch.isfinite(torch.as_tensor(b.quaternions())).all()
    b.set_gravity(np.array([0.0, 0.0, -G]))
    b.step_free(1 / 60, 100)
    assert torch.isfinite(torch.as_tensor(b.positions())).all()


def drop_flat(v0, substeps=100, itr=ITR):
    r = free_rope((0, 0, -G), -(R + 1e-4), 0.5, 0.5) if itr == ITR and substeps == 100 else None
    if r is None:
        r = ropesim.RopeSim(num_verts=NV, seg_len=SEG, dir=np.array([1.0, 0, 0]), density=RHO,
                            k_stretch=EA, k_bend=EI, max_h=1 / (60 * substeps), num_itr=itr)
        r.damping = 0.0
        r.drag = 0.0
        r.set_gravity(np.array([0.0, 0.0, -G]))
        r.set_planes(np.array([[0.0, 0.0, 1.0, -(R + 1e-4)]]), radius=R, kappa=KAPPA)
        r.unpin_root()
    v = np.zeros((NV, 3)); v[:, 2] = -v0
    r.set_velocities(v)
    return r


def test_impact_penetration_and_bounce():
    """Flat 10 m/s impact: penetration stays well under the rope radius; small bounce."""
    r = drop_flat(10.0)
    m = 1.0 / r.inv_masses()
    pen, up = 0.0, 0.0
    for _ in range(300):                              # 50 ms, one substep per call
        r.advance(H, substeps=1)
        pen = max(pen, -(r.positions()[:, 2] + R + 1e-4 - R).min())
        up = max(up, (m * r.velocities()[:, 2]).sum() / m.sum())
    assert pen < 0.6e-3                               # measured 0.31 mm (kappa 1: 0.83, kappa 3: 0.16)
    assert 0.03 < up / 10.0 < 0.2                     # numerical bounce at kappa = 2 (~0.10)


def test_bounce_independent_of_substep_size():
    """k = m (kappa/h)^2 makes the (numerical) restitution the same at any substep count."""
    def bounce(substeps):
        h = 1 / (60 * substeps)
        r = drop_flat(5.0, substeps=substeps)
        m = 1.0 / r.inv_masses()
        up = 0.0
        for _ in range(int(0.06 / h)):
            r.advance(h, substeps=1)
            up = max(up, (m * r.velocities()[:, 2]).sum() / m.sum())
        return up / 5.0
    assert bounce(50) == pytest.approx(bounce(200), abs=0.04)


def test_contact_solve_converges_with_iterations():
    """No limit cycle: a vertex touching down mid-substep ends at the same gap for 15/16/17 iterations."""
    gaps = []
    for itr in (15, 16, 17):
        r = ropesim.RopeSim(num_verts=NV, seg_len=SEG, dir=np.array([1.0, 0, 0]), density=RHO,
                            k_stretch=EA, k_bend=EI, max_h=H, num_itr=itr)
        r.damping = 0.0
        r.drag = 0.0
        r.set_gravity(np.array([0.0, 0.0, -G]))
        r.set_planes(np.array([[0.0, 0.0, 1.0, -(R + 0.9 * 5.0 * H)]]), radius=R, kappa=KAPPA)
        r.unpin_root()
        v = np.zeros((NV, 3)); v[:, 2] = -5.0
        r.set_velocities(v)
        r.advance(H, substeps=1)
        gaps.append(r.positions()[NV // 2, 2])
    assert max(gaps) - min(gaps) < 1e-7


def test_set_planes_validation():
    r = ropesim.RopeSim(num_verts=NV, seg_len=SEG, dir=np.array([1.0, 0, 0]), density=RHO,
                        k_stretch=EA, k_bend=EI, max_h=H, num_itr=ITR)
    with pytest.raises(ValueError):
        r.set_planes(np.array([[0.0, 0.0, 0.0, 0.0]]), radius=R)            # zero normal
    with pytest.raises(ValueError):
        r.set_planes(np.array([[0.0, 0.0, 1.0, 0.0]]))                      # no rope radius
    with pytest.raises(ValueError):
        r.set_planes(np.array([[0.0, 0.0, 1.0, 0.0]]), radius=R, v_slip=0.0)
    with pytest.raises(ValueError):
        r.set_planes(np.zeros((9, 4)), radius=R)                             # more than 8
    r.set_planes(np.zeros((0, 4)), radius=0.0)                               # empty list: contact off


def test_free_root_needs_step_free():
    """step() drives vertex 0 of every rope; a batch built with a free root must use step_free()."""
    pos = np.zeros((1, NV, 3), np.float32)
    pos[0, :, 0] = np.arange(NV) * SEG
    b = ropesim.RopeBatch.from_positions(pos, RHO, EA, EI, H, ITR, pinned=np.zeros(NV, bool))
    p = torch.zeros((1, 3), device="cuda")
    with pytest.raises(ValueError):
        b.step(p, p, 1 / 60, 100)
    with pytest.raises(ValueError):
        ropesim.RopeBatch.from_positions(pos, RHO, EA, EI, H, ITR, fix_root_orientation=True,
                                         pinned=np.zeros(NV, bool))
    bad = pos.copy(); bad[0, 1] = bad[0, 0]
    with pytest.raises(ValueError):
        ropesim.RopeBatch.from_positions(bad, RHO, EA, EI, H, ITR)        # zero-length segment
    b.step_free(1 / 60, 100)
    assert torch.isfinite(torch.as_tensor(b.positions())).all()
    r = free_rope((0, 0, -G), -1.0, 0.5, 0.5)                          # RopeSim after unpin_root()
    with pytest.raises(ValueError):
        r.set_root(np.zeros(3), np.zeros(3))


def test_set_masses_keeps_pins():
    """set_masses() changes free vertices only: mask pins stay pinned, a free root takes its mass."""
    pos = np.zeros((2, NV, 3), np.float32)
    pos[:, :, 0] = np.arange(NV) * SEG
    pos[1, :, 1] = 0.5
    pin = np.zeros((2, NV), bool)
    pin[0, -1] = True                                                  # env 0: tip pinned, root free
    pin[1, 0] = True                                                   # env 1: the usual pinned root
    b = ropesim.RopeBatch.from_positions(pos, RHO, EA, EI, H, ITR, pinned=pin)
    b.set_masses(np.full(NV, 0.01, np.float32))
    im = b.inv_masses()
    assert (im[pin] == 0).all() and np.allclose(im[~pin], 100.0)

