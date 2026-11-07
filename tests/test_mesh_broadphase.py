"""Many meshes per sim (docs/SOLVER.md section 5): the per-vertex broad phase (world AABB candidates rebuilt every
substep), the mesh capacity, per-mesh friction, poses given as CUDA tensors, and the device views of the
force readout. Same rope as test_mesh_contact (rope_A, 21 x 5 cm, h = 1/6000)."""
import numpy as np
import pytest
import torch

import ropesim
from ropesim import sdf

NV, SEG, RHO, EA, EI = 21, 0.05, 0.117, 11473.0, 4.48e-3
R = 0.00635
H, ITR, G = 1.0 / 6000.0, 16, 9.81
KAPPA, VSLIP = 2.0, 1e-3
PEN0 = G * H * H / KAPPA ** 2
DT, SUB = 1 / 60, 100
Q0 = np.array([0, 0, 0, 1], np.float32)


def slab_grid(spacing=1 / 128, lo=(-0.3125, -0.6, -0.1875), hi=(1.3125, 0.6, 0.1875)):
    return sdf.from_function(sdf.sdf_box((3.0, 3.0, 0.125), (0, 0, -0.125)), lo, hi, spacing)


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


def quat_y(angle):
    return np.array([0.0, np.sin(angle / 2), 0.0, np.cos(angle / 2)], np.float32)


# ------------------------------------------------------------------ broad phase / cap


def test_rope_rests_on_mesh_20_of_24_far_meshes_are_inert():
    """24 meshes: the rope's floor is mesh 20; the other 23 sit 5 m away (never candidates). The rest sink
    is exactly the one-mesh value, and the result is bitwise the one-mesh run (far meshes never enter)."""
    g = slab_grid()
    def run(many):
        b = batch(rope_rows(1), drag=0.1)
        if many:
            for m in range(24):
                p = np.array([0, 0, -0.02], np.float32) if m == 20 else np.array([5.0 + m, 5.0, -0.02], np.float32)
                b.add_mesh(*g.as_args(), pos=p)
        else:
            b.add_mesh(*g.as_args(), pos=np.array([0, 0, -0.02], np.float32))
        for _ in range(60):
            b.step_free(DT, SUB)
        return P_(b)
    one, many = run(False), run(True)
    assert one[0, :, 2].min() == pytest.approx(-0.02 + R - PEN0, abs=3e-8)
    assert np.array_equal(one, many)


def test_candidate_overflow_scans_every_mesh():
    """Ten coincident floors (more than the 8 candidate slots: the vertex falls back to scanning all):
    ten penalties share the weight, so the sink is PEN0 / 10, like two coincident floors give PEN0 / 2."""
    g = slab_grid()
    def sink(n_floors):
        b = batch(rope_rows(1), drag=0.1)
        for _ in range(n_floors):
            b.add_mesh(*g.as_args(), pos=np.array([0, 0, -0.02], np.float32))
        for _ in range(60):
            b.step_free(DT, SUB)
        return -0.02 + R - P_(b)[0, :, 2].min()
    assert sink(2) == pytest.approx(PEN0 / 2, rel=0.05)
    assert sink(10) == pytest.approx(PEN0 / 10, rel=0.1)


def test_mesh_63_supports_rope_and_preserves_per_env_wrenches():
    g = slab_grid(spacing=1 / 32)
    pos = rope_rows(2, z=R - PEN0)
    pos[1, :, 1] = 2.0
    b = batch(pos)
    translations = np.zeros((2, 64, 3), np.float32)
    translations[:, :63, 0] = 5.0  # far meshes never enter the candidate list
    translations[1, 63, 1] = 2.0
    quats = np.tile(Q0, (2, 64, 1))
    for m in range(64):
        assert b.add_mesh(*g.as_args(), pos=translations[:, m], quat=quats[:, m]) == m
    with pytest.raises(ValueError):
        b.add_mesh(*g.as_args(), pos=np.array([0, 0, -40.0], np.float32))
    b.enable_contact_forces()
    for _ in range(20):
        b.step_free(DT, SUB)

    wrenches = b.mesh_wrenches()
    np.testing.assert_array_equal(wrenches[:, :63], 0.0)
    mass = (1.0 / np.asarray(b.inv_masses(), np.float64)).sum(axis=1)
    np.testing.assert_allclose(wrenches[:, 63, 2], -mass * G, rtol=2e-3)
    np.testing.assert_array_equal(torch.as_tensor(b.mesh_wrenches_view()).cpu(), wrenches)

    # Exercise packed all-mesh pose updates at the highest slot in env 1.
    for i in range(10):
        translations[1, 63, 2] = (i + 1) * 0.001
        b.set_mesh_poses_all(translations, quats)
        b.step_free(DT, SUB)
    np.testing.assert_allclose(P_(b)[0, :, 2], R - PEN0, atol=1e-6)
    np.testing.assert_allclose(P_(b)[1, :, 2], R - PEN0 + 0.01, atol=5e-5)
    wrenches = b.mesh_wrenches()
    np.testing.assert_array_equal(wrenches[:, :63], 0.0)
    assert np.all(wrenches[:, 63, 2] < 0)


def test_moving_mesh_enters_and_leaves_the_rope():
    """A slab 2.5 m below a hanging rope rises (fast while far, then 0.2 m/s), lifts the rope's tail 5 cm,
    holds, and goes back down: the broad phase must admit it as it arrives (pose per frame) and drop it
    after. Tail rests on the slab while it is there, hangs free before and after."""
    pos = np.stack([np.zeros(NV), np.zeros(NV), 0.55 - np.arange(NV) * SEG], -1)  # hangs from z = 0.55 to -0.45
    pin = np.zeros((1, NV), bool); pin[:, 0] = True
    b = batch(pos[None], pinned=pin, drag=0.5)
    g = slab_grid()
    b.add_mesh(*g.as_args(), pos=np.array([0, 0, -3.0], np.float32))              # slab top at z = -3, far away
    for _ in range(60):
        b.step_free(DT, SUB)
    assert P_(b)[0, -1, 2] == pytest.approx(-0.45, abs=1e-3)                      # hanging straight
    def top(i):                                                                    # -3 -> -0.6 in 30 frames, then 0.2 m/s to -0.4
        return -3.0 + 2.4 * min(1.0, i / 30) if i <= 30 else min(-0.4, -0.6 + 0.2 * (i - 30) * DT)
    for i in range(1, 150):
        b.set_mesh_poses(0, np.array([0, 0, top(i)], np.float32), Q0)
        b.step_free(DT, SUB)
    z = P_(b)[0, :, 2].min()
    assert -0.4 + R - PEN0 - 1e-5 < z < -0.4 + R + 2e-4                           # tail rests on the slab (part of the weight)
    for i in range(1, 150):
        b.set_mesh_poses(0, np.array([0, 0, max(-3.0, -0.4 - 2.6 * i / 60)], np.float32), Q0)
        b.step_free(DT, SUB)
    assert P_(b)[0, -1, 2] == pytest.approx(-0.45, abs=5e-3)                      # hanging again


# ------------------------------------------------------------------ per-mesh friction


def test_per_mesh_friction():
    """Two 20-degree slabs in one sim: mu 0.5 holds the rope (tan 20 = 0.36), mu 0.1 lets it slide; the
    plane material stays the global one."""
    th = np.deg2rad(20.0)
    qy = np.array([0.0, np.sin(-th / 2), 0.0, np.cos(-th / 2)], np.float32)
    rot = np.array([[np.cos(th), 0, np.sin(th)], [0, 1, 0], [-np.sin(th), 0, np.cos(th)]])
    base = rope_rows(1, z=0.0)[0] @ rot + np.array([0, 0, R + 0.002])
    pos = np.stack([base + [0, 0.5, 0], base + [0, -0.5, 0]])
    b = batch(pos, mu_s=0.5)
    g = slab_grid()
    b.add_mesh(*g.as_args(), pos=np.array([[0, 0.5, 0], [0, 0.5, 0]], np.float32), quat=np.tile(qy, (2, 1)))
    b.add_mesh(*g.as_args(), pos=np.array([[0, -0.5, 0], [0, -0.5, 0]], np.float32), quat=np.tile(qy, (2, 1)), mu_s=0.1)
    for _ in range(90):
        b.step_free(DT, SUB)
    P = P_(b)
    assert np.abs(P[0] - pos[0]).max() < 2e-3                       # env 0 rests on the mu 0.5 slab
    assert P[1, :, 0].mean() - pos[1, :, 0].mean() < -0.3           # env 1 slid down the mu 0.1 slab
    b.set_mesh_friction(1, 0.5)                                     # now it holds too
    with pytest.raises(ValueError):
        b.set_mesh_friction(5, 0.5)


# ------------------------------------------------------------------ device poses / views


def test_cuda_pose_targets_match_host_path():
    """set_mesh_poses from torch cuda tensors (pos [N,3], quat [N,4], teleport [N] bool) == numpy path (to the
    solver's own run-to-run noise: YarnBall's orientation update races its neighbours by design, ~1e-7)."""
    v = 0.2
    def run(dev):
        b = batch(rope_rows(2), mu_s=0.5)
        g = slab_grid()
        b.add_mesh(*g.as_args(), pos=np.array([0, 0, -0.02], np.float32))
        for _ in range(30):
            b.step_free(DT, SUB)
        for i in range(60):
            p = np.array([[v * (i + 1) * DT, 0, -0.02]] * 2, np.float32)
            q = np.tile(Q0, (2, 1))
            t = np.array([False, i == 10])                     # env 1 teleports once mid-way
            if dev:
                b.set_mesh_poses(0, torch.as_tensor(p).cuda(), torch.as_tensor(q).cuda(), torch.as_tensor(t).cuda())
            else:
                b.set_mesh_poses(0, p, q, t)
            b.step_free(DT, SUB)
        return P_(b)
    a, c = run(False), run(True)
    assert np.allclose(a, c, atol=1e-5)
    assert np.allclose(a[0, :, 0] - rope_rows(1)[0, :, 0], v * 1.0 - v * v / (2 * 0.5 * G), atol=1.5e-3)


def test_cuda_pose_target_validation():
    b = batch(rope_rows(2))
    b.add_mesh(*slab_grid(spacing=1 / 32).as_args())
    p = torch.zeros((2, 3), device="cuda"); q = torch.tensor([[0, 0, 0, 1.0]] * 2, device="cuda")
    b.set_mesh_poses(0, p, q)                                   # CUDA poses, teleport False
    b.set_mesh_poses(0, p, q, teleport=True)
    with pytest.raises(ValueError):
        b.set_mesh_poses(0, p, np.zeros((2, 4), np.float32))    # mixed host / device
    b.set_mesh_poses(0, p, q, teleport=np.array([True, False]))       # host flags with device poses: staged
    with pytest.raises(ValueError):
        b.set_mesh_poses(0, p.double(), q)                      # float64
    with pytest.raises(ValueError):
        b.set_mesh_poses(0, torch.zeros((3, 3), device="cuda"), q)    # wrong rows


def test_force_views_match_readouts():
    pos = rope_rows(2, z=-0.02 + R - PEN0)
    b = batch(pos)
    b.add_mesh(*slab_grid().as_args(), pos=np.array([0, 0, -0.02], np.float32))
    b.enable_contact_forces()
    for _ in range(30):
        b.step_free(DT, SUB)
    W = b.mesh_wrenches(); F = b.contact_forces()
    Wv = torch.as_tensor(b.mesh_wrenches_view()).cpu().numpy(); Fv = torch.as_tensor(b.contact_forces_view()).cpu().numpy()
    assert Wv.shape == (2, 64, 6) and Fv.shape == (2, NV, 3)
    assert np.array_equal(Wv[:, 0], W[:, 0]) and np.array_equal(Fv, F)
    assert np.abs(Wv[:, 1:]).max() == 0.0
    M = (1.0 / np.asarray(b.inv_masses(), np.float64)[0]).sum()
    assert np.allclose(W[:, 0, 2], -M * G, rtol=2e-3)


def test_set_mesh_poses_all_matches_per_mesh_calls():
    """set_mesh_poses_all(pos [N,M,3], quat [N,M,4], teleport) == M calls of set_mesh_poses (host and CUDA)."""
    v = 0.2
    def run(mode):
        b = batch(rope_rows(2), mu_s=0.5)
        g = slab_grid()
        b.add_mesh(*g.as_args(), pos=np.array([0, 0, -0.02], np.float32))
        b.add_mesh(*g.as_args(), pos=np.array([0, 0, -3.0], np.float32))
        for _ in range(20):
            b.step_free(DT, SUB)
        for i in range(30):
            P = np.zeros((2, 2, 3), np.float32); P[:, 0] = [v * (i + 1) * DT, 0, -0.02]; P[:, 1] = [0, 0, -3.0]
            Qm = np.tile(Q0, (2, 2, 1))
            if mode == "per":
                for m in range(2): b.set_mesh_poses(m, P[:, m], Qm[:, m])
            elif mode == "host": b.set_mesh_poses_all(P, Qm)
            else: b.set_mesh_poses_all(torch.as_tensor(P).cuda(), torch.as_tensor(Qm).cuda(), torch.zeros(2, dtype=torch.bool, device="cuda"))
            b.step_free(DT, SUB)
        return P_(b)
    a, h, d = run("per"), run("host"), run("cuda")
    assert np.allclose(a, h, atol=1e-5) and np.allclose(a, d, atol=1e-5)
    assert np.allclose(a[0, :, 0] - rope_rows(1)[0, :, 0], v * 0.5 - v * v / (2 * 0.5 * G), atol=1.5e-3)
    b = batch(rope_rows(2)); b.add_mesh(*slab_grid(spacing=1 / 32).as_args())
    with pytest.raises(ValueError):
        b.set_mesh_poses_all(np.zeros((2, 2, 3), np.float32), np.zeros((2, 2, 4), np.float32))   # M != mesh count


# ------------------------------------------------------------------ review-3 regressions


def test_add_mesh_keeps_other_meshes_targets_pending():
    """add_mesh after set_mesh_poses must not turn the pending target into a teleport (review 3): the rope
    rides the 1 cm/frame lift either way instead of being ejected."""
    def run(add_between):
        b = batch(rope_rows(1), drag=0.1)
        g = slab_grid()
        b.add_mesh(*g.as_args(), pos=np.array([0, 0, -0.02], np.float32))
        for _ in range(30):
            b.step_free(DT, SUB)
        b.set_mesh_poses(0, np.array([0, 0, -0.01], np.float32), Q0)
        if add_between:
            b.add_mesh(*slab_grid(spacing=1 / 32).as_args(), pos=np.array([0, 0, -5.0], np.float32))
        b.step_free(DT, SUB)
        return np.abs(V_(b)[0, :, 2]).max()
    assert run(False) < 0.7 and run(True) < 0.7


def test_advance_zero_keeps_target_pending():
    """RopeSim.advance(0) runs nothing and must not commit a pending mesh target (review 3)."""
    r = ropesim.RopeSim(num_verts=NV, seg_len=SEG, dir=np.array([1.0, 0, 0]), density=RHO, k_stretch=EA, k_bend=EI, max_h=H, num_itr=ITR)
    r.damping = 0.0; r.drag = 0.1; r.set_gravity(np.array([0, 0, -G])); r.set_contact(radius=R, mu_s=0.5, v_slip=VSLIP, kappa=KAPPA); r.unpin_root()
    r.add_mesh(*slab_grid().as_args(), pos=np.array([0, 0, -0.02], np.float32))
    for _ in range(30):
        r.advance(DT, substeps=SUB)
    r.set_mesh_pose(0, np.array([0, 0, -0.01], np.float32), Q0)
    r.advance(0.0, substeps=SUB)
    r.advance(DT, substeps=SUB)
    assert np.abs(r.velocities()[:, 2]).max() < 0.7


def test_fast_mesh_is_still_caught_by_the_broad_phase():
    """A slab arriving at 6 m/s (10 cm per frame, 1 mm per substep) from far away under a resting rope is
    admitted by the broad phase (the pad includes the mesh's own travel) and lifts the rope."""
    b = batch(rope_rows(1, z=0.3 + R - PEN0), drag=0.3)
    b.set_planes(np.array([[0, 0, 1.0, 0.3]], np.float32), radius=R)          # resting on a plane at z = 0.3
    g = slab_grid()
    b.add_mesh(*g.as_args(), pos=np.array([0, 0, -6.0], np.float32))
    for _ in range(10):
        b.step_free(DT, SUB)
    z = -6.0
    for i in range(80):
        z = min(0.31, -6.0 + 6.0 * (i + 1) * DT)                             # 6 m/s until 1 cm above the plane
        b.set_mesh_poses(0, np.array([0, 0, z], np.float32), Q0)
        b.step_free(DT, SUB)
    for _ in range(60):
        b.step_free(DT, SUB)
    assert P_(b)[0, :, 2].min() == pytest.approx(0.31 + R - PEN0, abs=5e-5)
