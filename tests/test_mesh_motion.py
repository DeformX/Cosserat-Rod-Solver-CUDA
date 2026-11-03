"""Kinematic (moving) meshes and the contact-force readout (docs/SOLVER.md section 5).

A pose given to set_mesh_pose(s) is the pose at the END of the next step: the
mesh moves there over the substeps at constant linear/angular velocity, and
friction acts on the rope's motion relative to the moving surface. teleport=True
replaces the pose at once (resets). enable_contact_forces() averages, per step,
the force every surface applied to each vertex and the wrench the rope applied
to each mesh. Same rope as test_mesh_contact (rope_A, 21 x 5 cm, h = 1/6000).
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
DT, SUB = 1 / 60, 100                      # one control frame
Q0 = np.array([0, 0, 0, 1], np.float32)


def quat_z(angle):
    return np.array([0.0, 0.0, np.sin(angle / 2), np.cos(angle / 2)], np.float32)


def slab_grid(spacing=1 / 128, lo=(-0.3125, -0.6, -0.1875), hi=(1.3125, 0.6, 0.1875)):
    """Slab whose top face is the object plane z = 0 (origin on the face: see test_mesh_contact)."""
    return sdf.from_function(sdf.sdf_box((3.0, 3.0, 0.125), (0, 0, -0.125)), lo, hi, spacing)


def rope_sim(mu_s=0.5, mu_k=None):
    r = ropesim.RopeSim(num_verts=NV, seg_len=SEG, dir=np.array([1.0, 0, 0]), density=RHO,
                        k_stretch=EA, k_bend=EI, max_h=H, num_itr=ITR)
    r.damping = 0.0
    r.drag = 0.0
    r.set_gravity(np.array([0, 0, -G]))
    r.set_contact(radius=R, mu_s=mu_s, mu_k=mu_k, v_slip=VSLIP, kappa=KAPPA)
    r.unpin_root()
    return r


def batch(pos, pinned=None, mu_s=0.5, mu_k=None, drag=0.0):
    pos = np.asarray(pos, np.float32)
    pin = np.zeros(pos.shape[:2], bool) if pinned is None else pinned
    b = ropesim.RopeBatch.from_positions(pos, RHO, EA, EI, H, ITR, pinned=pin)
    b.damping = 0.0
    b.drag = drag
    b.set_gravity(np.array([0, 0, -G]))
    b.set_contact(radius=R, mu_s=mu_s, mu_k=mu_k, v_slip=VSLIP, kappa=KAPPA)
    return b


def rope_rows(n, z=0.02, y=0.0):
    pos = np.zeros((n, NV, 3)); pos[:, :, 0] = np.arange(NV) * SEG; pos[:, :, 1] = y; pos[:, :, 2] = z
    return pos


P_ = lambda b: torch.as_tensor(b.positions()).cpu().numpy()
V_ = lambda b: torch.as_tensor(b.velocities()).cpu().numpy()
dev = lambda a: torch.as_tensor(np.asarray(a, np.float32)).cuda()


# ----------------------------------------------------------------- interpolation


@pytest.mark.parametrize("w", [0.01, 0.1])
def test_rising_floor_rides_without_kick(w):
    """A floor raised w*dt per frame (RopeSim.advance path). Before per-substep interpolation
    the jump was an impact: 0.67 m/s at w = 1 cm/s, 7.8 m/s at 10 cm/s. Now the rope rides at
    w (sink stays PEN0 under constant velocity: the inertial term vanishes)."""
    r = rope_sim()
    g = slab_grid()
    r.add_mesh(*g.as_args(), pos=np.array([0, 0, -0.02], np.float32))
    for _ in range(60):
        r.advance(DT, substeps=SUB)
    vmax, z = 0.0, -0.02
    for i in range(60):
        z = -0.02 + w * (i + 1) * DT
        r.set_mesh_pose(0, np.array([0, 0, z], np.float32), Q0)
        r.advance(DT, substeps=SUB)
        vmax = max(vmax, np.abs(r.velocities()).max())
    assert vmax < 1.3 * w + 1e-3
    assert np.allclose(r.velocities()[:, 2], w, atol=0.02 * w + 1e-4)
    assert r.positions()[:, 2].min() == pytest.approx(z + R - PEN0, abs=2e-6)
    assert r.positions()[:, 2].max() == pytest.approx(z + R - PEN0, abs=2e-6)


def test_conveyor_carries_rope_with_friction():
    """A slab translating at 0.2 m/s under a resting rope: with friction the rope goes with it
    (belt), without friction it stays. step_free path, pose per frame."""
    v = 0.2
    out = {}
    for mu in (0.5, 0.0):
        b = batch(rope_rows(1), mu_s=mu)
        g = slab_grid()
        b.add_mesh(*g.as_args(), pos=np.array([0, 0, -0.02], np.float32))
        for _ in range(30):
            b.step_free(DT, SUB)
        x0 = P_(b)[0].copy()
        for i in range(60):
            b.set_mesh_poses(0, np.array([v * (i + 1) * DT, 0, -0.02], np.float32), Q0)
            b.step_free(DT, SUB)
        out[mu] = (P_(b)[0] - x0, V_(b)[0])
    d, vel = out[0.5]
    assert np.allclose(vel[:, 0], v, atol=0.01 * v)
    # the belt starts at v in one frame: the rope slips (mu_k = mu_s = 0.5) until it catches up,
    # a lag of v^2 / (2 mu g) = 4.1 mm, then sticks (no creep at constant velocity)
    lag = v * v / (2 * 0.5 * G)
    assert np.allclose(d[:, 0], v * 1.0 - lag, atol=1.5e-3), (d[:, 0], v - lag)
    assert np.abs(d[:, 1:]).max() < 1e-4
    d, vel = out[0.0]
    assert np.abs(vel).max() < 1e-3
    assert np.abs(d).max() < 1e-3


def test_turntable_rotates_rope():
    """A slab spun up about world z to 0.5 rad/s (ramp over 0.5 s, so the rope never has to slip)
    under a resting rope offset from the axis: the rope sticks and follows the rigid rotation
    (angular velocity x lever arm in the friction's surface velocity; the pose quaternion per
    substep)."""
    om = 0.5
    b = batch(rope_rows(1, y=0.3), mu_s=0.5, drag=0.0)
    g = slab_grid(lo=(-0.3125, -0.3125, -0.1875), hi=(1.3125, 0.6, 0.1875))
    b.add_mesh(*g.as_args(), pos=np.array([0, 0, -0.02], np.float32))
    for _ in range(30):
        b.step_free(DT, SUB)
    x0 = P_(b)[0].copy()
    th = 0.0
    for i in range(90):
        t1 = (i + 1) * DT
        th = om * (t1 * t1 / 1.0 if t1 < 0.5 else t1 - 0.25)      # omega ramps linearly to om over 0.5 s
        b.set_mesh_poses(0, np.array([0, 0, -0.02], np.float32), quat_z(th))
        b.step_free(DT, SUB)
    c, s = np.cos(th), np.sin(th)
    expect = x0 @ np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])   # rows rotated by +th about z
    err = np.abs(P_(b)[0] - expect).max()
    assert err < 3e-3, err
    vel = V_(b)[0]
    rvec = P_(b)[0][:, :2]
    assert np.allclose(vel[:, 0], -om * rvec[:, 1], atol=0.02 * om)
    assert np.allclose(vel[:, 1], om * rvec[:, 0], atol=0.02 * om)


def test_rope_sticks_on_slab_moving_up_an_incline():
    """20 deg slab (posed) carrying a rope up the slope at 0.1 m/s with mu 0.5 > tan 20: the rope
    rides along (velocity = the slab's), i.e. friction is relative to the moving surface."""
    th = np.deg2rad(20.0)
    qy = np.array([0.0, np.sin(-th / 2), 0.0, np.cos(-th / 2)], np.float32)   # tilt +x up by th
    up = np.array([np.cos(th), 0.0, np.sin(th)])
    pos = rope_rows(1, z=0.0)[0]
    pos = pos @ np.array([[np.cos(th), 0, np.sin(th)], [0, 1, 0], [-np.sin(th), 0, np.cos(th)]]) + np.array([0, 0, R + 0.002])
    b = batch(pos[None], mu_s=0.5)
    g = slab_grid()
    b.add_mesh(*g.as_args(), pos=np.zeros(3, np.float32), quat=qy)
    for _ in range(60):
        b.step_free(DT, SUB)
    p0 = P_(b)[0].copy()
    v = 0.1
    for i in range(60):
        b.set_mesh_poses(0, (v * (i + 1) * DT * up).astype(np.float32), qy)
        b.step_free(DT, SUB)
    vel = V_(b)[0]
    assert np.allclose(vel, v * up, atol=0.02 * v)
    # abrupt start: slip until caught up at g (mu cos - sin) = 1.25 m/s^2 -> lag v^2 / 2a = 4 mm;
    # then stick with the creep the regularized law needs to hold m g sin(th): v_slip tan / mu
    lag = v * v / (2 * G * (0.5 * np.cos(th) - np.sin(th))) + VSLIP * np.tan(th) / 0.5 * 1.0
    d = P_(b)[0] - p0
    assert np.allclose(d @ up, v * 1.0 - lag, atol=1.5e-3), (d @ up, v - lag)
    n = np.array([-np.sin(th), 0.0, np.cos(th)])
    assert np.abs(d @ n).max() < 2e-4                          # stays on the surface


def test_rope_batch_step_path_interpolates():
    """RopeBatch.step (Sim::advance path, roots driven): a rope lying on a floor with its root
    pinned at the resting height; floor and root rise together at 5 cm/s. The rope rides the
    floor at exactly that speed (no kick, sink PEN0) and translates rigidly."""
    z0 = -0.2
    pos = rope_rows(1, z=z0 + R - PEN0)
    pin = np.zeros((1, NV), bool); pin[:, 0] = True
    b = batch(pos, pinned=pin, drag=0.0)
    g = slab_grid()
    b.add_mesh(*g.as_args(), pos=np.array([0, 0, z0], np.float32))
    root = pos[:, 0].copy(); zero = dev(np.zeros((1, 3)))
    for _ in range(60):
        b.step(dev(root), zero, DT, SUB)
    p0 = P_(b)[0].copy()
    w, vmax = 0.05, 0.0
    wv = dev(np.array([[0, 0, w]]))
    for i in range(60):
        z = z0 + w * (i + 1) * DT
        b.set_mesh_poses(0, np.array([0, 0, z], np.float32), Q0)
        # step(): root placed at the frame-START position, then swept at w over the frame
        b.step(dev(root + np.array([0, 0, w * i * DT])), wv, DT, SUB)
        vmax = max(vmax, np.abs(V_(b)[0, 1:]).max())
    assert vmax < 1.3 * w
    assert P_(b)[0, 1:, 2].min() == pytest.approx(z + R - PEN0, abs=2e-6)
    assert np.allclose(P_(b)[0] - p0, [0, 0, w * 1.0], atol=1e-4)


def test_teleport_replaces_pose_at_once():
    """Both floors go up 1 cm in one frame. env 0 teleports (an impact: the rope is ejected at
    ~0.8 * 1 cm / h), env 1 interpolates (the floor moves at 0.6 m/s for one frame and the rope
    rides it). Per-env teleport mask."""
    b = batch(rope_rows(2), drag=0.0)
    g = slab_grid()
    p = np.array([[0, 0, -0.02], [0, 0, -0.02]], np.float32)
    q = np.tile(Q0, (2, 1))
    b.add_mesh(*g.as_args(), pos=p, quat=q)
    for _ in range(60):
        b.step_free(DT, SUB)
    p[:, 2] += 0.01
    b.set_mesh_poses(0, p, q, teleport=np.array([True, False]))
    b.step_free(DT, SUB)
    v = V_(b)
    assert np.abs(v[0, :, 2]).max() > 10.0
    assert np.abs(v[1, :, 2]).max() < 1.0
    for _ in range(90):
        b.step_free(DT, SUB)
    assert P_(b)[1, :, 2].min() == pytest.approx(-0.01 + R - PEN0, abs=3e-8)
    assert P_(b)[0, :, 2].min() > 1.0          # still flying


def test_static_target_is_inert():
    """set_mesh_poses with the pose the mesh already has leaves the step bitwise unchanged."""
    def run(call):
        b = batch(rope_rows(1), drag=0.1)
        g = slab_grid()
        b.add_mesh(*g.as_args(), pos=np.array([0, 0, -0.02], np.float32))
        for _ in range(30):
            if call:
                b.set_mesh_poses(0, np.array([0, 0, -0.02], np.float32), Q0)
            b.step_free(DT, SUB)
        return P_(b)
    assert np.array_equal(run(False), run(True))


# ------------------------------------------------------------------ force readout


def test_resting_rope_wrench_equals_weight():
    """Rope resting on a slab: the wrench on the mesh is its weight at its centre of mass
    (torque about the mesh origin = r_cm x F), the per-vertex forces sum to +m g; with a plane
    instead, the per-vertex readout still counts it."""
    r = rope_sim()
    g = slab_grid()
    r.add_mesh(*g.as_args(), pos=np.array([0, 0, -0.02], np.float32))
    r.enable_contact_forces()
    for _ in range(90):
        r.advance(DT, substeps=SUB)
    m = 1.0 / np.asarray(r.inv_masses(), np.float64)
    M = m.sum()
    x_cm = (m * r.positions()[:, 0]).sum() / M
    W = r.mesh_wrenches()
    assert W.shape == (1, 6)
    assert np.allclose(W[0, :3], [0, 0, -M * G], atol=2e-3 * M * G)
    assert np.allclose(W[0, 3:], [0, M * G * x_cm, 0], atol=2e-3 * M * G * x_cm)
    F = r.contact_forces()
    assert F.shape == (NV, 3)
    assert np.allclose(F.sum(0), [0, 0, M * G], atol=2e-3 * M * G)
    assert (F[:, 2] > 0).all()
    r2 = rope_sim()
    r2.set_planes(np.array([[0, 0, 1, -0.02]], np.float32), radius=R)
    r2.enable_contact_forces()
    for _ in range(90):
        r2.advance(DT, substeps=SUB)
    assert np.allclose(r2.contact_forces().sum(0), [0, 0, M * G], atol=2e-3 * M * G)
    assert r2.mesh_wrenches().shape == (0, 6)
    r2.enable_contact_forces(False)
    with pytest.raises(ValueError):
        r2.contact_forces()


def test_belt_under_held_rope_sees_kinetic_friction():
    """Root pinned (RopeBatch.step), slab sliding under the rope AWAY from the root at 0.2 m/s
    (the rope is taut and still, slip >> v_slip): the wrench on the slab is the sliding friction
    mu_k * (free mass) * g against the belt's motion plus the weight. The lattice covers the
    rope in the slab's frame over the whole travel."""
    mu_k = 0.3
    pos = rope_rows(1, z=-0.02 + R - PEN0)                # root pinned at the resting height: every free vertex rests on the belt
    pin = np.zeros((1, NV), bool); pin[:, 0] = True
    b = batch(pos, pinned=pin, mu_s=0.5, mu_k=mu_k)
    g = slab_grid(lo=(-0.6, -0.6, -0.1875), hi=(1.3125, 0.6, 0.1875))
    b.add_mesh(*g.as_args(), pos=np.array([-0.25, 0, -0.02], np.float32))
    b.enable_contact_forces()
    root = dev(pos[:, 0]); zero = dev(np.zeros((1, 3)))
    for _ in range(60):
        b.step(root, zero, DT, SUB)
    v = 0.2
    for i in range(60):
        b.set_mesh_poses(0, np.array([-0.25 + v * (i + 1) * DT, 0, -0.02], np.float32), Q0)
        b.step(root, zero, DT, SUB)
    inv = np.asarray(b.inv_masses(), np.float64)[0]
    Mf = (1.0 / inv[inv > 0]).sum()                     # the pinned root carries no contact force
    W = b.mesh_wrenches()
    assert W.shape == (1, 1, 6)
    assert W[0, 0, 2] == pytest.approx(-Mf * G, rel=0.02)
    assert W[0, 0, 0] == pytest.approx(mu_k * W[0, 0, 2], rel=0.01)   # the rope holds the belt back (-x): mu_k N
    assert abs(W[0, 0, 1]) < 1e-3 * Mf * G
    F = b.contact_forces()
    assert F.shape == (1, NV, 3)
    assert F[0, 0].tolist() == [0, 0, 0]
    assert np.allclose(F[0].sum(0), -W[0, 0, :3], rtol=1e-5, atol=1e-6)   # one mesh: action = -reaction
    assert np.abs(V_(b)[0]).max() < 2e-3                 # the held rope stays put


def test_set_mesh_poses_teleport_validation():
    b = batch(rope_rows(2))
    g = slab_grid()
    b.add_mesh(*g.as_args())
    with pytest.raises(ValueError):
        b.set_mesh_poses(0, np.zeros(3, np.float32), Q0, teleport=np.array([True]))
    b.set_mesh_poses(0, np.zeros(3, np.float32), Q0, teleport=np.array([True, False]))
    b.set_mesh_poses(0, np.zeros(3, np.float32), Q0, teleport=True)
