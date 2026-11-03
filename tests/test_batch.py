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
               fix_root_orientation=False, **kw):
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
    b.drag = 0.0
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
    # quaternion view maps straight onto solver state
    q = torch.as_tensor(b.quaternions())
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