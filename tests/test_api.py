"""API contract: construction, state access, parameter plumbing, reset."""
import numpy as np
import pytest

import ropesim
from conftest import G, H, make_rope


def test_constructor_geometry():
    n, seg = 17, 0.03
    d = np.array([1.0, 2.0, -0.5])  # non-unit on purpose: must be normalized
    r = make_rope(n=n, seg=seg, direction=d)
    p = np.asarray(r.positions(), dtype=np.float64)
    assert p.shape == (n, 3)
    np.testing.assert_allclose(p[0], 0.0, atol=1e-7)
    dn = d / np.linalg.norm(d)
    expect = np.outer(np.arange(n) * seg, dn)
    np.testing.assert_allclose(p, expect, atol=1e-6)
    np.testing.assert_allclose(r.rest_lengths(), seg, rtol=1e-5)


def test_invalid_args_raise():
    with pytest.raises(ValueError):
        ropesim.RopeSim(num_verts=2)
    with pytest.raises(ValueError):
        ropesim.RopeSim(num_verts=8, seg_len=0.0)
    with pytest.raises(ValueError):
        ropesim.RopeSim(num_verts=8, dir=[0, 0, 0])
    with pytest.raises(ValueError):
        ropesim.RopeSim(num_verts=8, density=0.0)
    with pytest.raises(ValueError):
        ropesim.RopeSim(num_verts=8, max_h=0.0)
    # k_bend=0 zeroes qRests -> upstream orientation solve normalizes a zero
    # vector -> NaN cascade (reproduced before the guard existed)
    with pytest.raises(ValueError):
        ropesim.RopeSim(num_verts=8, k_bend=0.0)


def test_zero_or_invalid_stiffness_arrays_rejected():
    r = make_rope(n=8)
    with pytest.raises(ValueError):
        r.set_bend_stiffnesses(np.zeros(7, np.float32))
    with pytest.raises(ValueError):
        r.set_stretch_stiffnesses(np.zeros(7, np.float32))
    with pytest.raises(ValueError):
        r.set_bend_stiffnesses(np.full(7, np.nan, np.float32))
    # valid arrays still work after the rejected calls
    r.set_bend_stiffnesses(np.full(7, 1e-4, np.float32))
    r.set_stretch_stiffnesses(np.full(7, 500.0, np.float32))
    for _ in range(10):
        r.advance(H)
    assert np.isfinite(r.positions()).all()


def test_fine_discretization_constructs_and_runs():
    """seg_len below the upstream yarn-scale radius floor (5e-4 m) used to
    fail construction with a misleading error; with collisions off the
    radius/barrier are dead values and are now scaled to seg_len."""
    r = make_rope(n=101, seg=0.0004, drag=1.0)  # 100 x 0.4 mm segments
    for _ in range(30):
        r.advance(H)
    assert np.isfinite(r.positions()).all()


def test_collisions_seg_len_guard_is_clear():
    with pytest.raises(ValueError, match="5e-4"):
        ropesim.RopeSim(num_verts=8, seg_len=4e-4, collisions=True)


def test_root_quaternion_validation():
    # non-unit q is normalized on write (the fixed frame is never
    # renormalized by the solver)
    r = make_rope(n=8, fix_root_orientation=True)
    r.set_root([0, 0, 0], [0, 0, 0], q=np.array([0, 0, 0, 2.0], np.float32))
    np.testing.assert_allclose(np.asarray(r.quaternions())[0], [0, 0, 0, 1.0],
                               atol=1e-6)
    # zero/invalid q rejected
    with pytest.raises(ValueError):
        r.set_root([0, 0, 0], [0, 0, 0], q=np.zeros(4, np.float32))
    # q on a free-pivot root would only act as a transient impulse: rejected
    free = make_rope(n=8)
    with pytest.raises(ValueError):
        free.set_root([0, 0, 0], [0, 0, 0], q=np.array([0, 0, 0, 1], np.float32))


def test_advance_explicit_substeps():
    """advance(dt, substeps=k) runs exactly k substeps of h=dt/k (the M1
    step() contract) — the legacy path's fp ceil gives 51 for 1/60 / (1/3000)."""
    r = make_rope(n=8, direction=(0, -1, 0))
    old_max_h = r.max_h
    v = np.array([0.12, 0.0, 0.0])
    r.set_root([0, 0, 0], v.tolist())
    r.advance(H, substeps=50)
    np.testing.assert_allclose(np.asarray(r.positions(), dtype=np.float64)[0],
                               v * H, atol=1e-6)
    assert r.max_h == pytest.approx(old_max_h)  # restored
    with pytest.raises(ValueError):
        r.advance(H, substeps=-1)


def test_inv_masses():
    n, seg, rho = 9, 0.02, 0.1
    r = make_rope(n=n, seg=seg, density=rho)
    inv = np.asarray(r.inv_masses(), dtype=np.float64)
    assert inv[0] == 0.0                                  # pinned root
    np.testing.assert_allclose(inv[1:-1], 1.0 / (rho * seg), rtol=1e-5)
    np.testing.assert_allclose(inv[-1], 1.0 / (rho * seg / 2), rtol=1e-5)


def test_initial_quaternions_identity_along_x():
    r = make_rope(n=8, direction=(1, 0, 0))
    q = np.asarray(r.quaternions(), dtype=np.float64)[:-1]  # last is padding
    np.testing.assert_allclose(q, np.array([0, 0, 0, 1.0]) * np.ones((7, 1)),
                               atol=1e-6)


def test_param_roundtrip():
    r = make_rope()
    r.drag = 0.25
    r.damping = 1e-5
    r.num_itr = 4
    r.accel_ratio = 0.9
    r.max_h = 1e-3
    r.set_gravity([0.1, -9.0, 0.2])
    assert r.drag == pytest.approx(0.25)
    assert r.damping == pytest.approx(1e-5)
    assert r.num_itr == 4
    assert r.accel_ratio == pytest.approx(0.9)
    assert r.max_h == pytest.approx(1e-3)
    np.testing.assert_allclose(r.gravity, [0.1, -9.0, 0.2], rtol=1e-6)
    assert r.last_error == 0 and r.last_warning == 0


def test_state_shapes_dtypes():
    n = 12
    r = make_rope(n=n)
    assert r.positions().shape == (n, 3) and r.positions().dtype == np.float32
    assert r.velocities().shape == (n, 3)
    assert r.quaternions().shape == (n, 4)
    assert r.inv_masses().shape == (n,)
    assert r.rest_lengths().shape == (n - 1,)
    assert r.num_verts == n


def test_root_stays_pinned_without_driving():
    r = make_rope(direction=(1, 0, 0))
    for _ in range(30):
        r.advance(H)
    p = np.asarray(r.positions(), dtype=np.float64)
    np.testing.assert_allclose(p[0], 0.0, atol=1e-7)
    assert p[-1, 1] < -0.01  # the rest of the rope fell


def test_reset_restores_initial_state():
    r = make_rope(direction=(1, 0, 0))
    p0 = np.array(r.positions())
    q0 = np.array(r.quaternions())
    for _ in range(30):
        r.advance(H)
    assert not np.allclose(np.array(r.positions()), p0, atol=1e-4)
    r.reset()
    np.testing.assert_allclose(np.array(r.positions()), p0, atol=1e-7)
    np.testing.assert_allclose(np.array(r.quaternions()), q0, atol=1e-7)
    np.testing.assert_allclose(np.array(r.velocities()), 0.0, atol=1e-7)
    assert r.time == pytest.approx(0.0)


def _drop30():
    r = make_rope(direction=(1, 0, 0), drag=0.5)
    for _ in range(30):
        r.advance(H)
    return np.asarray(r.positions(), dtype=np.float64)


def test_reset_then_same_dynamics():
    """A reset rope must evolve like a fresh one (no hidden state leaks).

    The bound is calibrated against the divergence of two FRESH sims — the
    chaotic-GS iterations make fast-swing trajectories diverge measurably
    (~mm over 0.5 s) even for identical setups. Reset must not add
    divergence beyond that floor."""
    p_fresh = _drop30()
    floor = np.max(np.linalg.norm(_drop30() - p_fresh, axis=1))

    r = make_rope(direction=(1, 0, 0), drag=0.5)
    for _ in range(60):
        r.advance(H)
    r.reset()
    for _ in range(30):
        r.advance(H)
    p_reset = np.asarray(r.positions(), dtype=np.float64)
    d = np.max(np.linalg.norm(p_reset - p_fresh, axis=1))
    print(f"\n  reset-vs-fresh {d:.2e} m; fresh-vs-fresh chaos floor {floor:.2e} m")
    assert d < max(3.0 * floor, 1.5e-3)


def test_collisions_flag_constructs_and_runs():
    r = make_rope(collisions=True, drag=0.5)
    for _ in range(30):
        r.advance(H)
    assert np.isfinite(r.positions()).all()
    assert r.collisions_enabled
    assert r.last_error == 0
