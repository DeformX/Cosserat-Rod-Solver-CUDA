"""Static equilibria vs analytic ground truth.

These pin down the unit mapping:
  density -> rho [kg/m]   (verified via inv_masses in test_api)
  k_stretch -> EA [N]     (hanging elongation: dL_i = T_i * l_i / EA)
  k_bend -> EI [N*m^2]    (clamped cantilever: delta = w L^4 / (8 EI))
"""
import numpy as np
import pytest

from conftest import (G, H, axis_angle_quat, make_rope, masses, quat_rotate,
                      rope_length, segment_strains, settle)


def test_vertical_hang_equilibrium():
    r = make_rope(n=33, direction=(0, -1, 0))
    settle(r, t=3.0, drag=3.0)
    p = np.asarray(r.positions(), dtype=np.float64)
    L = rope_length(r)
    # straight, vertical, below the pin
    assert np.max(np.abs(p[:, 0])) < 1e-3 * L
    assert np.max(np.abs(p[:, 2])) < 1e-3 * L
    assert np.all(np.diff(p[:, 1]) < 0)
    # static: residual speeds tiny
    v = np.asarray(r.velocities(), dtype=np.float64)
    assert np.max(np.linalg.norm(v, axis=1)) < 5e-3
    # under tension everywhere
    assert np.all(segment_strains(r) > 0)


def test_hang_elongation_matches_EA_theory():
    """Discrete-exact: elongation_i = (g * mass_below_i) * l_i / EA."""
    n, seg, rho, EA = 51, 0.02, 0.1, 500.0
    r = make_rope(n=n, seg=seg, density=rho, EA=EA, direction=(0, -1, 0))
    settle(r, t=4.0, drag=3.0)

    m = masses(r)
    rest = np.asarray(r.rest_lengths(), dtype=np.float64)
    # tension in segment i supports every vertex below it (j > i)
    tension = G * np.array([np.sum(m[i + 1:]) for i in range(n - 1)])
    dL_theory = np.sum(tension * rest / EA)

    p = np.asarray(r.positions(), dtype=np.float64)
    dL_measured = np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1)) - np.sum(rest)

    assert dL_measured == pytest.approx(dL_theory, rel=0.02), \
        f"dL measured {dL_measured:.5f} vs theory {dL_theory:.5f}"


def test_horizontal_release_settles_vertical():
    r = make_rope(n=33, direction=(1, 0, 0))
    settle(r, t=6.0, drag=2.0)
    p = np.asarray(r.positions(), dtype=np.float64)
    L = rope_length(r)
    assert abs(p[-1, 0]) < 0.02 * L
    assert abs(p[-1, 2]) < 0.02 * L
    assert p[-1, 1] == pytest.approx(-L, rel=0.02)  # small stretch correction ok


@pytest.mark.parametrize("EI", [0.4, 0.8])
def test_cantilever_droop_vs_beam_theory(EI):
    """Uniformly-loaded clamped cantilever: delta_tip = w L^4 / (8 EI).

    Validates that k_bend maps to EI. Run at delta/L ~ 2-4% (small-deflection
    beam regime). The discrete rod + quasi-static orientation solve won't be
    exact; assert the ratio is O(1) here, and proportionality (1/EI) tightly
    in test_cantilever_scales_inversely_with_EI.
    """
    n, seg, rho, EA = 26, 0.02, 0.1, 2000.0
    L = (n - 1) * seg
    r = make_rope(n=n, seg=seg, density=rho, EA=EA, EI=EI,
                  direction=(1, 0, 0), fix_root_orientation=True)
    settle(r, t=4.0, drag=2.0)
    p = np.asarray(r.positions(), dtype=np.float64)
    delta = -p[-1, 1]
    w = rho * G
    delta_theory = w * L**4 / (8.0 * EI)
    ratio = delta / delta_theory
    print(f"\n  EI={EI}: delta={delta*1000:.2f} mm, beam theory "
          f"{delta_theory*1000:.2f} mm, ratio={ratio:.3f}")
    assert np.isfinite(delta)
    # Measured 0.924 (EI=0.4) and 0.928 (EI=0.8): the mapping is good to ~8%
    # (consistent low bias ~ discretization). Band tightened around that so a
    # genuine scale error (e.g. a stray factor of 2) cannot pass.
    assert 0.80 < ratio < 1.05, f"k_bend != EI mapping? ratio {ratio:.3f}"


def test_cantilever_scales_inversely_with_EI():
    deltas = {}
    for EI in (0.4, 0.8):
        n, seg = 26, 0.02
        r = make_rope(n=n, seg=seg, density=0.1, EA=2000.0, EI=EI,
                      direction=(1, 0, 0), fix_root_orientation=True)
        settle(r, t=4.0, drag=2.0)
        deltas[EI] = -np.asarray(r.positions(), dtype=np.float64)[-1, 1]
    assert deltas[0.4] / deltas[0.8] == pytest.approx(2.0, rel=0.10)


def _seg_angles_to(r, direction):
    p = np.asarray(r.positions(), dtype=np.float64)
    dirs = np.diff(p, axis=0)
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    return np.rad2deg(np.arccos(np.clip(dirs @ np.asarray(direction), -1.0, 1.0)))


def test_clamped_root_holds_rod_against_gravity():
    """The orientation BC, statically: a rod built at +30 deg with a CLAMPED
    root frame must hold its direction (cantilever physics); the identical rod
    with a free-pivot root must fall and hang. EI=0.4 — within the solver's
    stable stiffness range (see the KNOWN_LIMIT test below)."""
    th = np.deg2rad(30.0)
    d = (np.cos(th), np.sin(th), 0.0)
    n, seg = 16, 0.02
    L = (n - 1) * seg

    r = make_rope(n=n, seg=seg, density=0.05, EA=2000.0, EI=0.4,
                  direction=d, fix_root_orientation=True)
    settle(r, t=4.0, drag=2.0)
    angles = _seg_angles_to(r, d)
    tip_y = float(np.asarray(r.positions(), dtype=np.float64)[-1, 1])
    print(f"\n  clamped: root {angles[0]:.2f} deg off, max {np.max(angles):.2f} "
          f"deg off, tip y {tip_y:.4f} (built {L*np.sin(th):.4f})")
    assert angles[0] < 3.0
    assert np.max(angles) < 8.0          # small beam droop allowed
    assert tip_y > 0.8 * L * np.sin(th)  # held up, not fallen

    pivot = make_rope(n=n, seg=seg, density=0.05, EA=2000.0, EI=0.4,
                      direction=d, fix_root_orientation=False)
    settle(pivot, t=4.0, drag=2.0)
    tip_y_pivot = float(np.asarray(pivot.positions(), dtype=np.float64)[-1, 1])
    print(f"  free pivot: tip y {tip_y_pivot:.4f} (must fall)")
    assert tip_y_pivot < 0.0


def test_clamped_root_steers_rod_smoothly():
    """The orientation BC, dynamically: smooth 60 Hz frame rotation (the way
    the arm drives the EE) must steer the rod. Built at 30 deg, command ramps
    to 40 deg over 1.5 s, hold 2.5 s."""
    th0, th1 = np.deg2rad(30.0), np.deg2rad(40.0)
    d0 = (np.cos(th0), np.sin(th0), 0.0)
    n, seg = 16, 0.02
    r = make_rope(n=n, seg=seg, density=0.05, EA=2000.0, EI=0.4,
                  direction=d0, fix_root_orientation=True)
    r.drag = 1.0
    ramp, hold = 90, 150
    for k in range(ramp + hold):
        th = th0 + (th1 - th0) * min(1.0, (k + 1) / ramp)
        q = axis_angle_quat([0, 0, 1], th)
        r.set_root([0, 0, 0], [0, 0, 0], q=q.astype(np.float32))
        r.advance(H)

    q = axis_angle_quat([0, 0, 1], th1)
    np.testing.assert_allclose(np.asarray(r.quaternions(), dtype=np.float64)[0],
                               q, atol=1e-5)  # binding holds the frame exactly
    angles = _seg_angles_to(r, quat_rotate(q, [1.0, 0, 0]))
    strains = segment_strains(r)
    print(f"\n  steered: root {angles[0]:.2f} deg off, max {np.max(angles):.2f} "
          f"deg off, max |strain| {np.max(np.abs(strains))*100:.2f}%")
    assert angles[0] < 3.0
    assert np.max(angles) < 10.0
    assert np.max(np.abs(strains)) < 0.02


@pytest.mark.xfail(
    strict=True,
    reason="MEASURED solver limit: tube-scale EI=10 N*m^2 (h=1/3000, l=0.02, "
           "rho=0.05) goes unstable under root reorientation — strains blow "
           "past 100% (4.7x stretch measured, internal velocities ~300 m/s, "
           "masked by drag). Stable at EI=0.4; boundary in between. Affects "
           "the handoff's future 'tube as stiff segments' option (needs "
           "longer segments / higher density / smaller h, or solver work); "
           "the locked rigid-transform tube plan is unaffected. strict=True: "
           "if this XPASSes the limit has moved (e.g. upstream fix) — "
           "re-evaluate tube modeling.")
def test_tube_stiffness_regime_KNOWN_LIMIT():
    n, seg = 16, 0.02
    r = make_rope(n=n, seg=seg, density=0.05, EA=2000.0, EI=10.0,
                  direction=(1, 0, 0), fix_root_orientation=True)
    r.drag = 1.0
    theta_end = np.deg2rad(30.0)
    for k in range(120):
        q = axis_angle_quat([0, 0, 1], theta_end * (k + 1) / 120)
        r.set_root([0, 0, 0], [0, 0, 0], q=q.astype(np.float32))
        r.advance(H)
    for _ in range(360):
        r.advance(H)
    s = float(np.max(np.abs(segment_strains(r))))
    print(f"\n  max |strain| at EI=10 after ramp+hold: {s*100:.0f}% "
          f"(physical equilibrium would be ~0.01%)")
    # the GOOD-physics assertion: fails today (xfail documents the limit)
    assert np.isfinite(s) and s < 0.02
