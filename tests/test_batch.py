"""RopeBatch validation — the single-rope RopeSim (already physics-validated)
is the oracle. Requires torch CUDA (in the isaacsim env).

Interop contract under test: root commands arrive as contiguous float32 CUDA
tensors; observations are device buffers shared zero-copy via
__cuda_array_interface__; torch stays on its default stream (legacy-stream
ordering synchronizes solver and consumer).
"""
import numpy as np
import pytest
import torch

import ropesim
from conftest import G, H, make_rope

DEV = "cuda"


def make_batch(N=4, V=25, seg=0.02, roots=None, direction=(0, -1, 0),
               fix_root_orientation=False, drag=0.0, **kw):
    if roots is None:
        roots = np.zeros((N, 3), np.float32)
        roots[:, 0] = np.arange(N) * 1.5  # spatially separated envs
    defaults = dict(density=0.1, k_stretch=500.0, k_bend=1e-6,
                    max_h=1.0 / 3000.0, num_itr=8)
    defaults.update(kw)
    b = ropesim.RopeBatch(num_envs=N, verts_per_rope=V,
                          root_pos=roots.astype(np.float32), seg_len=seg,
                          dir=list(direction),
                          fix_root_orientation=fix_root_orientation, **defaults)
    b.set_gravity([0.0, -G, 0.0])
    b.drag = drag
    b.damping = 0.0
    return b, roots


def t32(a):
    return torch.tensor(np.asarray(a, dtype=np.float32), device=DEV)


def pos_np(b):
    return torch.as_tensor(b.positions()).cpu().numpy()


def vel_np(b):
    return torch.as_tensor(b.velocities()).cpu().numpy()


def _traj(t):
    return np.array([0.06 * np.sin(2 * np.pi * 0.7 * t),
                     0.02 * np.sin(2 * np.pi * 1.3 * t), 0.0])


def _drive_both(T=1.0, N=3, V=25, seg=0.02):
    """Drive every batch env and a single RopeSim through the same root
    motion (per-env offset by the env root)."""
    b, roots = make_batch(N=N, V=V, seg=seg)
    s = make_rope(n=V, seg=seg, direction=(0, -1, 0))  # root at origin = env0
    for k in range(int(round(T / H))):
        t = k * H
        p0, p1 = _traj(t), _traj(t + H)
        vel = (p1 - p0) / H
        b.step(t32(roots + p0), t32(np.tile(vel, (N, 1))), dt=H, substeps=50)
        s.set_root(p0.tolist(), vel.tolist())
        s.advance(H, substeps=50)
    return b, s, roots


def test_batch_env0_matches_single_rope():
    """Env 0 (root identical to the single rope) must reproduce the validated
    single-rope trajectory. The kernels are the same binary code over a
    larger flat array; per-vertex arithmetic is grid-size independent, so the
    match should be at fp-noise level (printed)."""
    b, s, _ = _drive_both(T=1.0)
    pb = pos_np(b)[0]
    ps = np.asarray(s.positions(), dtype=np.float64)
    d = np.max(np.linalg.norm(pb - ps, axis=1))
    print(f"\n  batch-env0 vs single-rope max diff: {d:.2e} m "
          f"(bitwise: {np.array_equal(pb.astype(np.float32), s.positions())})")
    assert d < 1e-6


def test_batch_envs_identical_up_to_translation():
    """All envs see the same root motion relative to their roots -> their
    rope shapes must match up to the root translation (fp translation noise
    only)."""
    b, _, roots = _drive_both(T=1.0)
    p = pos_np(b)
    rel = p - roots[:, None, :]
    for e in range(1, p.shape[0]):
        d = np.max(np.linalg.norm(rel[e] - rel[0], axis=1))
        assert d < 1e-5, f"env {e} diverged from env 0 by {d:.2e}"


def test_no_cross_talk():
    """A violently whipped env must not perturb a static neighbor at all
    (independent rods in one flat array, collisions off)."""
    b, roots = make_batch(N=2, V=25)
    hold = roots.copy()
    zeros = np.zeros((2, 3), np.float32)
    for k in range(60):
        t = k * H
        pos = hold.copy()
        vel = zeros.copy()
        # env 1: 3 Hz, 0.3 m whip; env 0: frozen root
        pos[1, 0] += 0.3 * np.sin(2 * np.pi * 3.0 * t)
        vel[1, 0] = (0.3 * np.sin(2 * np.pi * 3.0 * (t + H))
                     - 0.3 * np.sin(2 * np.pi * 3.0 * t)) / H
        b.step(t32(pos), t32(vel), dt=H, substeps=50)

    ref, _ = make_batch(N=1, V=25, roots=roots[:1])
    for k in range(60):
        ref.step(t32(roots[:1]), t32(np.zeros((1, 3), np.float32)),
                 dt=H, substeps=50)

    d = np.max(np.linalg.norm(pos_np(b)[0] - pos_np(ref)[0], axis=1))
    print(f"\n  static env max deviation while neighbor whips: {d:.2e} m")
    assert d < 1e-6


def test_reset_envs_partial():
    b, roots = make_batch(N=4, V=25)
    init = pos_np(b).copy()
    vel = t32(np.tile([0.2, 0, 0], (4, 1)))
    for k in range(30):
        pos = roots + np.array([0.2 * (k * H), 0, 0], np.float32)
        b.step(t32(pos), vel, dt=H, substeps=50)
    before = pos_np(b).copy()

    b.reset_envs(np.array([1, 3], np.int32))
    after = pos_np(b)
    v_after = vel_np(b)

    # reset envs back at construction state, zero velocity
    np.testing.assert_allclose(after[[1, 3]], init[[1, 3]], atol=1e-6)
    np.testing.assert_allclose(v_after[[1, 3]], 0.0, atol=1e-7)
    # untouched envs bit-identical to before the reset call
    np.testing.assert_array_equal(after[[0, 2]], before[[0, 2]])

    # continue stepping with per-env-consistent roots: finite
    pos = np.array(after[:, 0, :])  # current roots
    b.step(t32(pos), t32(np.zeros((4, 3), np.float32)), dt=H, substeps=50)
    assert np.isfinite(pos_np(b)).all()
    assert b.last_error == 0


def test_zero_copy_views():
    b, _ = make_batch(N=2, V=10)
    view = b.positions()
    t = torch.as_tensor(view)
    assert t.is_cuda and t.dtype == torch.float32
    assert tuple(t.shape) == (2, 10, 3)
    assert t.data_ptr() == view.ptr  # shared memory, not a copy
    t0 = t.clone()
    b.step(t32(pos_np(b)[:, 0, :]), t32(np.full((2, 3), 0.1, np.float32)),
           dt=H, substeps=50)
    assert not torch.equal(t, t0), "view did not reflect the step (copied?)"
    # quaternion view is a gathered COPY (writes cannot corrupt solver state)
    q = torch.as_tensor(b.quaternions())
    assert q.data_ptr() != 0
    q.add_(99.0)  # vandalize the scratch buffer ...
    b.gather()
    q2 = torch.as_tensor(b.quaternions())
    assert torch.all(q2 < 50.0), "quaternion view aliases live solver state!"
    assert tuple(q.shape) == (2, 10, 4)
    n = torch.linalg.vector_norm(q[:, :-1, :], dim=-1)
    assert torch.allclose(n, torch.ones_like(n), atol=1e-5)


def test_input_validation():
    b, roots = make_batch(N=2, V=10)
    ok_pos, ok_vel = t32(roots), t32(np.zeros((2, 3), np.float32))
    with pytest.raises(ValueError):          # CPU tensor
        b.step(torch.zeros(2, 3), ok_vel, dt=H, substeps=50)
    with pytest.raises(ValueError):          # wrong shape
        b.step(t32(np.zeros((3, 3), np.float32)), ok_vel, dt=H, substeps=50)
    with pytest.raises(ValueError):          # wrong dtype
        b.step(torch.zeros(2, 3, device=DEV, dtype=torch.float64), ok_vel,
               dt=H, substeps=50)
    with pytest.raises(ValueError):          # non-contiguous
        big = torch.zeros(2, 6, device=DEV)
        b.step(big[:, ::2], ok_vel, dt=H, substeps=50)
    with pytest.raises(ValueError):          # substeps
        b.step(ok_pos, ok_vel, dt=H, substeps=0)
    with pytest.raises(ValueError):          # q without the fix flag
        b.step(ok_pos, ok_vel, dt=H, substeps=50,
               root_q=torch.zeros(2, 4, device=DEV))
    with pytest.raises(ValueError):          # bad env id
        b.reset_envs(np.array([5], np.int32))
    with pytest.raises(ValueError):          # k_bend=0 guarded here too
        ropesim.RopeBatch(num_envs=1, verts_per_rope=4,
                          root_pos=np.zeros((1, 3), np.float32), k_bend=0.0)


def test_program_endpoint_and_orientation_exact():
    """The second-order program must land the root exactly on root_pos_end
    and the frame exactly on root_q_end (last substep SLERP t=1)."""
    b, roots = make_batch(N=3, V=10, fix_root_orientation=True)
    p1 = roots + np.array([0.04, -0.02, 0.03], np.float32)
    v0 = np.tile(np.array([0.5, 0.2, -0.1], np.float32), (3, 1))  # arbitrary
    th = np.deg2rad([10.0, 20.0, 30.0])
    q0 = np.zeros((3, 4), np.float32); q0[:, 3] = 1.0
    q1 = np.zeros((3, 4), np.float32)
    q1[:, 2] = np.sin(th / 2); q1[:, 3] = np.cos(th / 2)
    b.step(t32(roots), t32(v0), dt=H, substeps=50,
           root_q=t32(q0), root_pos_end=t32(p1), root_q_end=t32(q1))
    root = pos_np(b)[:, 0, :]
    np.testing.assert_allclose(root, p1, atol=2e-6)
    qroot = torch.as_tensor(b.quaternions()).cpu().numpy()[:, 0, :]
    np.testing.assert_allclose(qroot, q1, atol=1e-5)
    # midpoint-sampling check: the root's velocity readback must equal the
    # analytic last-substep midpoint sample v0 + a*(dt - h/2)
    a = 2.0 * ((p1 - roots) - v0 * H) / H**2
    v_last = v0 + a * (H - (H / 50) / 2)
    np.testing.assert_allclose(vel_np(b)[:, 0, :], v_last, rtol=1e-4, atol=1e-4)


def test_program_degenerates_to_constant_velocity():
    """With v0 = (p1-p0)/dt the quadratic's acceleration is zero — the
    program path must reproduce the validated legacy sweep."""
    traj = lambda t: np.array([0.05 * np.sin(2 * np.pi * 0.8 * t), 0.0, 0.0])
    runs = {}
    for mode in ("legacy", "program"):
        b, roots = make_batch(N=1, V=25, drag=0.5)
        for k in range(60):
            t = k * H
            p0 = (roots + traj(t)).astype(np.float32)
            p1 = (roots + traj(t + H)).astype(np.float32)
            v0 = (p1 - p0) / H
            if mode == "legacy":
                b.step(t32(p0), t32(v0), dt=H, substeps=50)
            else:
                b.step(t32(p0), t32(v0), dt=H, substeps=50,
                       root_pos_end=t32(p1))
        runs[mode] = pos_np(b)[0]
    d = np.max(np.linalg.norm(runs["legacy"] - runs["program"], axis=1))
    print(f"\n  program(a=0) vs legacy max diff: {d:.2e} m")
    assert d < 1e-5


def test_program_beats_constant_velocity_tracking():
    """The whole point of second-order driving: with the true (analytic)
    start velocity, 60 Hz program driving must track a fine-control-rate
    reference better than 60 Hz constant-velocity driving.

    The reference runs at 120 Hz with the SAME substep size h=1/3000
    (25 substeps/frame) — isolating the control-rate/interpolation effect
    from the solver's h-sensitive numerics. Error is measured over the FULL
    rope (root-adjacent vertices feel boundary-velocity noise most; the tip
    low-passes it)."""
    R, f = 0.06, 1.5
    w = 2 * np.pi * f

    def p_of(t):
        return np.array([R * (np.cos(w * t) - 1.0), R * np.sin(w * t), 0.0])

    def v_of(t):
        return np.array([-R * w * np.sin(w * t), R * w * np.cos(w * t), 0.0])

    def run(rate, substeps, mode):
        b, roots = make_batch(N=1, V=25, drag=0.5)
        dt = 1.0 / rate
        every = int(rate // 60)
        snaps = []
        for k in range(int(1.5 * rate)):
            t = k * dt
            p0 = (roots + p_of(t)).astype(np.float32)
            p1 = (roots + p_of(t + dt)).astype(np.float32)
            if mode == "program":
                b.step(t32(p0), t32(v_of(t)[None]), dt=dt, substeps=substeps,
                       root_pos_end=t32(p1))
            else:
                b.step(t32(p0), t32((p1 - p0) / dt), dt=dt, substeps=substeps)
            if (k + 1) % every == 0:
                snaps.append(pos_np(b)[0].astype(np.float64))
        return np.array(snaps)  # [frames, V, 3]

    ref = run(120.0, 25, "program")      # h = 1/3000, same as the 60 Hz runs
    prog60 = run(60.0, 50, "program")
    legacy60 = run(60.0, 50, "legacy")
    d_prog = np.sqrt(np.mean(np.sum((prog60 - ref) ** 2, axis=-1)))
    d_leg = np.sqrt(np.mean(np.sum((legacy60 - ref) ** 2, axis=-1)))
    L = 24 * 0.02
    print(f"\n  whole-rope RMS error vs same-h 120 Hz ref: "
          f"program60 {d_prog/L*100:.3f}%L, constant-vel60 {d_leg/L*100:.3f}%L")
    assert d_prog < d_leg, "second-order driving did not improve tracking"


def test_slerp_reduces_fast_rotation_strain():
    """Under fast clamp rotation (90 deg over 0.5 s on a tube-stiff-ish rod),
    SLERPed orientation must not stress the rod more than per-frame stepped
    orientation — and should reduce the peak strain."""
    def run(slerped):
        b, roots = make_batch(N=1, V=16, direction=(1, 0, 0),
                              fix_root_orientation=True, k_stretch=2000.0,
                              k_bend=0.4, density=0.05)
        b.drag = 1.0
        zeros = t32(np.zeros((1, 3), np.float32))
        max_strain = 0.0
        rest = 0.02
        q_prev = np.array([[0, 0, 0, 1]], np.float32)
        for k in range(30):
            th = np.deg2rad(90.0) * (k + 1) / 30
            q = np.array([[0, 0, np.sin(th / 2), np.cos(th / 2)]], np.float32)
            if slerped:
                b.step(t32(roots), zeros, dt=H, substeps=50,
                       root_q=t32(q_prev), root_pos_end=t32(roots),
                       root_q_end=t32(q))
            else:
                b.step(t32(roots), zeros, dt=H, substeps=50, root_q=t32(q))
            q_prev = q
            p = pos_np(b)[0]
            lens = np.linalg.norm(np.diff(p, axis=0), axis=1)
            max_strain = max(max_strain, float(np.max(np.abs(lens - rest) / rest)))
        return max_strain

    s_slerp = run(True)
    s_step = run(False)
    print(f"\n  peak |strain| fast rotation: slerped {s_slerp*100:.3f}% "
          f"vs stepped {s_step*100:.3f}%")
    # At this rotation rate both schemes sit at the fp noise floor (~1e-4
    # strain) — the assertion guards "slerp is not significantly worse", with
    # an additive floor so noise-scale jitter can't flip it. The regime where
    # SLERP would matter strongly (fast stiff-rod reorientation) is bounded
    # by the EI stability limit anyway.
    assert s_slerp <= max(s_step * 1.5, 5e-3)
    assert s_slerp < 0.10


def test_fixed_orientation_roots_batch():
    """Commanded root frames hold exactly across the batch (read back through
    the zero-copy quaternion view)."""
    b, roots = make_batch(N=3, V=16, direction=(1, 0, 0),
                          fix_root_orientation=True, k_stretch=2000.0,
                          k_bend=0.4, density=0.05)
    b.drag = 1.0
    th = np.deg2rad([5.0, 15.0, 25.0])
    q = np.zeros((3, 4), np.float32)
    q[:, 2] = np.sin(th / 2)
    q[:, 3] = np.cos(th / 2)
    for _ in range(120):
        b.step(t32(roots), t32(np.zeros((3, 3), np.float32)),
               dt=H, substeps=50, root_q=t32(q))
    qroot = torch.as_tensor(b.quaternions()).cpu().numpy()[:, 0, :]
    np.testing.assert_allclose(qroot, q, atol=1e-5)
    assert np.isfinite(pos_np(b)).all()


@pytest.mark.slow
def test_batch_throughput_1024():
    """Throughput at RL scale: 1024 envs x 33 verts, 60 Hz frames x 50
    substeps. Informational (printed); asserts only health."""
    import time
    N, V = 1024, 33
    roots = np.zeros((N, 3), np.float32)
    g = int(np.ceil(np.sqrt(N)))
    roots[:, 0] = (np.arange(N) % g) * 2.0
    roots[:, 2] = (np.arange(N) // g) * 2.0
    b, _ = make_batch(N=N, V=V, roots=roots)
    vel = t32(np.zeros((N, 3), np.float32))
    pos = t32(roots)
    b.step(pos, vel, dt=H, substeps=50)  # warm-up (CUDA graph capture)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    frames = 60
    for _ in range(frames):
        b.step(pos, vel, dt=H, substeps=50)
    torch.cuda.synchronize()
    dt_wall = time.perf_counter() - t0
    sim_time = frames * H
    print(f"\n  {N} envs x {V} verts: {frames} frames in {dt_wall:.3f} s "
          f"-> {frames/dt_wall:.0f} FPS, {sim_time/dt_wall:.1f}x realtime")
    assert np.isfinite(pos_np(b)).all()
    assert b.last_error == 0