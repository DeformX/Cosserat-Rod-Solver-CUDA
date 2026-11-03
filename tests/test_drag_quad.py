"""Quadratic cross-flow air drag (drag_quad / drag_quad_t) — analytic checks.

Model under test (sim/iteration.cu endItr, local patch): per substep, each
free vertex's velocity is split by its segment tangent t̂ and each component
gets the exact frozen-direction solution of dv/dt = -k|v|v:

    v_n /= 1 + drag_quad   * |v_n| * h        (cross-flow / normal term)
    v_t /= 1 + drag_quad_t * |v_t| * h        (tangential / skin term)

Closed form for a uniformly translating straight rope in zero gravity:
    damped component:   v(t) = v0 / (1 + k*v0*t)
    undamped component: v(t) = v0

Test design: vertex 0 is always pinned, so we use a LOW-EA rope and measure
only vertices far from the root inside a window shorter than the constraint
wave's arrival (c = sqrt(EA/rho) = sqrt(1/0.1) = 3.2 m/s; in T = 0.05 s the
wave travels ~0.16 m = 8 of 32 segments; we assert from vertex 16). The far
section translates rigidly, so tangents are constant and the per-substep
frozen-direction update is exact — tolerances are tight, not physical.
"""
import numpy as np
import pytest

import ropesim
from conftest import G, H, make_rope, settle
from test_batch import make_batch, pos_np, t32, vel_np

N = 33            # 32 segments x 2 cm = 0.64 m
FAR = 16          # first vertex assumed wave-free over T
KN = 0.5          # exaggerated normal coefficient [1/m] for measurable decay
V0 = 10.0         # initial speed [m/s]
NFRAMES = 3
T = NFRAMES * H   # 0.05 s


def make_free_decay_rope(max_h=1.0 / 3000.0):
    """Straight +x rope, zero gravity, low EA (slow constraint wave)."""
    r = make_rope(n=N, seg=0.02, direction=(1.0, 0.0, 0.0), density=0.1,
                  EA=1.0, EI=1e-6, max_h=max_h)
    r.set_gravity([0.0, 0.0, 0.0])
    return r


def kick_and_run(r, v0_vec, nframes=NFRAMES):
    vel = np.tile(np.asarray(v0_vec, np.float32), (N, 1))
    r.set_velocities(vel)
    for _ in range(nframes):
        r.advance(H)
    return np.asarray(r.velocities(), dtype=np.float64)


def closed_form(v0, k, t):
    return v0 / (1.0 + k * v0 * t) if k > 0 else v0


def test_normal_decay_matches_closed_form():
    """v ⊥ rope axis: |v(t)| = v0/(1+kn*v0*t) on the wave-free section."""
    r = make_free_decay_rope()
    r.drag_quad = KN
    v = kick_and_run(r, (0.0, V0, 0.0))
    expected = closed_form(V0, KN, T)  # 8.0 m/s
    far = v[FAR:]
    assert np.allclose(far[:, 1], expected, rtol=1e-2), (
        f"far vy {far[:, 1]} != {expected}")
    assert np.max(np.abs(far[:, [0, 2]])) < 0.05  # no spurious cross terms


def test_axial_motion_undamped_when_kt_zero():
    """v ∥ rope axis with drag_quad_t = 0: no decay (anisotropy check)."""
    r = make_free_decay_rope()
    r.drag_quad = KN  # normal term on, but v is purely tangential
    v = kick_and_run(r, (V0, 0.0, 0.0))
    far = v[FAR:]
    assert np.allclose(far[:, 0], V0, rtol=2e-3), (
        f"axial speed decayed: {far[:, 0]}")


def test_component_split_isotropic_emulation():
    """kt = kn, v at 45°: each component follows its own closed form."""
    r = make_free_decay_rope()
    r.drag_quad = KN
    r.drag_quad_t = KN
    c = V0 / np.sqrt(2.0)
    v = kick_and_run(r, (c, c, 0.0))
    expected = closed_form(c, KN, T)  # 6.01 m/s
    far = v[FAR:]
    assert np.allclose(far[:, 0], expected, rtol=1e-2)
    assert np.allclose(far[:, 1], expected, rtol=1e-2)


def test_substep_invariance():
    """Same decay at 25 vs 100 substeps per frame (exact per-substep law)."""
    vys = []
    for max_h in (1.0 / 1500.0, 1.0 / 6000.0):
        r = make_free_decay_rope(max_h=max_h)
        r.drag_quad = KN
        v = kick_and_run(r, (0.0, V0, 0.0))
        vys.append(np.mean(v[FAR:, 1]))
    assert abs(vys[0] - vys[1]) / vys[1] < 5e-3, f"substep-dependent: {vys}"


def test_kt_inert_without_kn():
    """drag_quad_t alone (drag_quad == 0) is documented inert — the whole
    quadratic block is gated on drag_quad > 0 and the legacy path runs."""
    r = make_free_decay_rope()
    r.drag_quad_t = KN  # no drag_quad set
    v = kick_and_run(r, (0.0, V0, 0.0))
    far = v[FAR:]
    assert np.allclose(far[:, 1], V0, rtol=2e-3), (
        f"drag_quad_t acted without drag_quad: {far[:, 1]}")


def test_defaults_and_validation():
    """drag_quad defaults to 0 (legacy path); negatives rejected."""
    r = make_rope()
    assert r.drag_quad == 0.0 and r.drag_quad_t == 0.0
    with pytest.raises(ValueError):
        r.drag_quad = -0.1
    with pytest.raises(ValueError):
        r.drag_quad_t = -0.1
    b, _ = make_batch(N=2, V=10)
    assert b.drag_quad == 0.0 and b.drag_quad_t == 0.0
    with pytest.raises(ValueError):
        b.drag_quad = -1.0


def test_batch_envs_identical_with_drag_quad():
    """Two identically-driven batch envs stay identical with the quadratic
    drag on (exercises the batched path incl. per-rope boundary tangents)."""
    b, roots = make_batch(N=2, V=25)
    b.drag_quad = 0.094  # the physical rope value
    for k in range(60):
        t = k * H
        off = np.array([0.15 * np.sin(2 * np.pi * 1.1 * t),
                        0.05 * np.sin(2 * np.pi * 2.3 * t), 0.0], np.float32)
        voff = (np.array([0.15 * np.sin(2 * np.pi * 1.1 * (t + H)),
                          0.05 * np.sin(2 * np.pi * 2.3 * (t + H)), 0.0],
                         np.float32) - off) / H
        b.step(t32(roots + off), t32(np.tile(voff, (2, 1))),
               dt=H, substeps=50)
    v = vel_np(b)
    d = np.max(np.abs(v[1] - v[0]))
    # fp translation noise (roots offset 1.5 m) through 1 s of chaotic
    # dynamics — bitwise equality across translated envs is NOT expected
    # (see conftest header); measured ~1e-4 m/s. Scope: this guards batch
    # allocation/flag-layout asymmetries only — the two envs are
    # value-identical, so a value-symmetric cross-rope indexing mistake is
    # invisible here; the tip-tangent and batch-vs-single tests cover those.
    assert d < 1e-3, f"envs diverged under drag_quad: {d:.2e}"


def test_tip_vertex_uses_prev_segment_tangent():
    """Tip (hasPrev only) must read qs[tid-1], not the padding entry.

    The padding rotor qs[n-1] is a frozen copy of the INITIAL last-segment
    frame (configure() copies lastQ, and the quaternion iteration never
    touches a no-hasNext vertex), so a straight never-rotating rope cannot
    distinguish the two reads. Rotate the rope ~90° from its construction
    pose first (build along +x, settle hanging under gravity), THEN kick
    axially along the real tip tangent: the correct read leaves the tip
    undamped; the padding rotor (still +x) would classify the motion as
    normal and damp it ~45% in one frame at this k."""
    r = make_rope(n=N, seg=0.02, direction=(1.0, 0.0, 0.0), density=0.1,
                  EA=500.0, EI=1e-6, max_h=1.0 / 3000.0)
    r.set_gravity([0.0, -G, 0.0])
    settle(r, t=4.0)  # legacy linear drag; rope swings down and hangs
    p = np.asarray(r.positions(), np.float64)
    t_tip = p[-1] - p[-2]
    t_tip /= np.linalg.norm(t_tip)
    assert t_tip[1] < -0.9, f"rope did not hang: tip tangent {t_tip}"
    r.set_gravity([0.0, 0.0, 0.0])
    r.drag_quad = 5.0  # huge: any misrouting is unmissable
    v = kick_and_run(r, V0 * t_tip, nframes=1)
    tip_speed = np.linalg.norm(v[-1])
    assert abs(tip_speed - V0) / V0 < 0.05, (
        f"tip damped under pure axial motion: |v|={tip_speed}")


def test_batch_matches_single_rope_with_drag_quad():
    """Batch env0 must reproduce the single-rope trajectory with the
    quadratic drag on (same kernel over a larger flat array)."""
    kq, kt = 0.3, 0.01
    b, roots = make_batch(N=2, V=25)
    b.drag_quad, b.drag_quad_t = kq, kt
    s = make_rope(n=25, seg=0.02, direction=(0.0, -1.0, 0.0), density=0.1,
                  EA=500.0, EI=1e-6, max_h=1.0 / 3000.0)
    s.drag_quad, s.drag_quad_t = kq, kt
    for k in range(60):
        t = k * H
        off = np.array([0.10 * np.sin(2 * np.pi * 1.3 * t), 0.0,
                        0.04 * np.sin(2 * np.pi * 2.1 * t)], np.float64)
        voff = (np.array([0.10 * np.sin(2 * np.pi * 1.3 * (t + H)), 0.0,
                          0.04 * np.sin(2 * np.pi * 2.1 * (t + H))],
                         np.float64) - off) / H
        pos = roots + off.astype(np.float32)
        b.step(t32(pos), t32(np.tile(voff.astype(np.float32), (2, 1))),
               dt=H, substeps=50)
        s.set_root(off.tolist(), voff.tolist())
        s.advance(H, substeps=50)
    d = np.max(np.linalg.norm(
        pos_np(b)[0] - np.asarray(s.positions(), np.float64), axis=1))
    assert d < 1e-6, f"batch env0 diverged from single rope: {d:.2e} m"
