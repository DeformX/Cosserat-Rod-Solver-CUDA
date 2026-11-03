"""Root (boundary) driving — the mechanism the Isaac Lab coupling relies on.

Per frame: set_root(pos=traj(t), vel=(traj(t+H)-traj(t))/H); advance(H).
The solver applies pos += h*vel each substep, sweeping the root linearly
across the frame. These tests pin down exactness, continuity, control-rate
adequacy, and the handoff's core worry: that a swept root injects boundary
velocity (a frozen root would under-energize the swing).
"""
import numpy as np
import pytest

from conftest import H, drive, make_rope, masses, rope_length, segment_strains


def test_root_sweep_is_exact():
    r = make_rope(n=25, direction=(0, -1, 0))
    v = np.array([0.10, 0.0, 0.05])
    r.set_root([0, 0, 0], v.tolist())
    r.advance(H)
    root = np.asarray(r.positions(), dtype=np.float64)[0]
    np.testing.assert_allclose(root, v * H, atol=1e-5)

    # and across many frames of a continuous piecewise-linear trajectory
    for k in range(1, 10):
        r.set_root((v * k * H).tolist(), v.tolist())
        r.advance(H)
    root = np.asarray(r.positions(), dtype=np.float64)[0]
    np.testing.assert_allclose(root, v * 10 * H, atol=1e-4)


def test_driving_continuity_no_teleports():
    """Where the sweep ends each frame must equal the next commanded pos."""
    traj = lambda t: np.array([0.1 * np.sin(2 * np.pi * t), 0.0, 0.0])
    r = make_rope(n=25, direction=(0, -1, 0))
    max_jump = 0.0
    for k in range(60):
        t = k * H
        p_cmd = traj(t)
        p_now = np.asarray(r.positions(), dtype=np.float64)[0]
        if k > 0:
            max_jump = max(max_jump, float(np.linalg.norm(p_cmd - p_now)))
        v = (traj(t + H) - p_cmd) / H
        r.set_root(p_cmd.tolist(), v.tolist())
        r.advance(H)
    print(f"\n  max root discontinuity between frames: {max_jump:.2e} m")
    assert max_jump < 1e-4


def test_control_rate_convergence():
    """60 Hz piecewise-constant-velocity driving must track the 240 Hz result:
    validates the planned control rate for the swing task."""
    circle = lambda t: np.array([0.05 * np.cos(2 * np.pi * 0.5 * t) - 0.05,
                                 0.0,
                                 0.05 * np.sin(2 * np.pi * 0.5 * t)])
    tips = {}
    for rate in (60.0, 240.0):
        r = make_rope(n=25, drag=0.5)
        _, tp, _ = drive(r, circle, t_end=2.0, rate=rate,
                         record_every=int(rate // 60))
        tips[rate] = tp
    L = 24 * 0.02
    d = np.sqrt(np.mean(np.sum((tips[60.0] - tips[240.0]) ** 2, axis=1)))
    print(f"\n  RMS tip diff 60 Hz vs 240 Hz control: {d/L*100:.2f}%L")
    assert d < 0.03 * L


def test_swinging_root_injects_energy():
    """The swept root must pump the rope (boundary velocity injection).

    Note: the 'static' reference rope is not perfectly asleep — starting
    unstretched under gravity it rings at the (tiny) stretch frequency. Its
    KE scale is orders below the driven case, so the 50x margin is a sound
    discriminator, not an equilibrium claim."""
    def ke(r):
        v = np.asarray(r.velocities(), dtype=np.float64)
        return 0.5 * float(np.sum(masses(r) * np.sum(v * v, axis=1)))

    # static root: hanging rope stays asleep
    rs = make_rope(n=25, direction=(0, -1, 0))
    for _ in range(int(round(2.0 / H))):
        rs.advance(H)
    ke_static = ke(rs)

    # root oscillating near the chain's natural frequency
    L = 24 * 0.02
    f = 0.9 / ((4 * np.pi / 2.404826) * np.sqrt(L / 9.8))
    sweep = lambda t: np.array([0.15 * np.sin(2 * np.pi * f * t), 0.0, 0.0])
    rd = make_rope(n=25, direction=(0, -1, 0))
    drive(rd, sweep, t_end=2.0)
    ke_driven = ke(rd)

    print(f"\n  KE static {ke_static:.2e} J vs driven {ke_driven:.2e} J")
    assert ke_driven > 50.0 * max(ke_static, 1e-12)
    assert ke_driven > 1e-4  # absolute floor: the swing really moves


def test_whip_driving_stays_stable():
    """Fast aggressive driving: finite state, bounded strain, no solver errors."""
    sweep = lambda t: np.array([0.3 * np.sin(2 * np.pi * 2.0 * t),
                                0.1 * np.sin(2 * np.pi * 4.0 * t), 0.0])
    r = make_rope(n=33, direction=(0, -1, 0), EA=500.0)
    drive(r, sweep, t_end=2.0)
    p = np.asarray(r.positions(), dtype=np.float64)
    assert np.isfinite(p).all()
    strain = np.max(np.abs(segment_strains(r)))
    print(f"\n  max |strain| after whip: {strain*100:.2f}%")
    assert strain < 0.25
    assert np.max(np.linalg.norm(p - p[0], axis=1)) < 2.0 * rope_length(r)
    assert r.last_error == 0
