"""Dynamic behavior vs theory: pendulum mode, energy, symmetry, convergence."""
import numpy as np
import pytest

from conftest import G, H, drive, energy_scalar, make_rope, rope_length

# Fundamental swinging mode of a hanging chain (fixed pivot, free end):
# omega = (z1/2) sqrt(g/L) with z1 the first zero of Bessel J0.
Z1 = 2.404825557695773


def _measure_period(ts, xs):
    """Period from zero crossings of a decaying oscillation about 0."""
    s = np.sign(xs)
    idx = np.where(np.diff(s) != 0)[0]
    # linear-interpolate crossing times
    tc = ts[idx] - xs[idx] * (ts[idx + 1] - ts[idx]) / (xs[idx + 1] - xs[idx])
    assert len(tc) >= 5, f"too few zero crossings ({len(tc)}) to measure period"
    return 2.0 * np.mean(np.diff(tc))


def test_hanging_chain_pendulum_period():
    n, seg = 33, 0.02
    L = (n - 1) * seg
    theta = np.deg2rad(2.0)
    r = make_rope(n=n, seg=seg, direction=(np.sin(theta), -np.cos(theta), 0))
    ts, xs = [], []
    nfr = int(round(6.0 / H))
    for k in range(nfr):
        r.advance(H)
        p = r.positions()
        ts.append((k + 1) * H)
        xs.append(float(p[-1, 0]))
    T = _measure_period(np.array(ts), np.array(xs))
    T_theory = (4.0 * np.pi / Z1) * np.sqrt(L / G)
    print(f"\n  period: measured {T:.4f} s vs chain theory {T_theory:.4f} s "
          f"({100*(T/T_theory-1):+.2f}%)")
    assert T == pytest.approx(T_theory, rel=0.05)


def test_energy_no_gain_horizontal_release():
    """Free swing from horizontal with drag=damping=0: total energy must never
    rise above its initial value (no numerical energy injection), and the
    solver's intrinsic dissipation is reported.

    Scope: the energy scalar covers KE + gravity + centerline stretch, which
    captures essentially all energy here (EI=1e-6, free-pivot root, no
    orientation driving). See conftest.total_energy for what is omitted."""
    n, seg, EA = 33, 0.02, 500.0
    r = make_rope(n=n, seg=seg, EA=EA, direction=(1, 0, 0))
    L = rope_length(r)
    E0 = energy_scalar(r, EA)                 # = 0 by construction (y_ref=0)
    # energy scale: |PE| at the vertical hang
    from conftest import masses
    m = masses(r)
    E_scale = G * np.sum(m * np.arange(n) * seg)
    E_hist = []
    for _ in range(int(round(3.5 / H))):
        r.advance(H)
        E_hist.append(energy_scalar(r, EA))
    E_hist = np.array(E_hist)
    gain = (np.max(E_hist) - E0) / E_scale
    retained = 1.0 + (E_hist[-1] - E0) / E_scale  # 1=conserved, 0=fully damped
    print(f"\n  energy gain {100*gain:+.2f}% of scale; retained after 3.5 s: "
          f"{100*retained:.1f}%")
    assert gain < 0.02, "numerical energy injection"
    # Measured baseline (50 substeps, 8 itr): ~17% retained after 3.5 s — the
    # solver dissipates substantially on a free swing. The floor below guards
    # against regressions to near-total damping; the (substeps, itr) tradeoff
    # is quantified in test_energy_dissipation_characterization.
    assert retained > 0.08, "implausibly high dissipation"
    assert np.isfinite(E_hist).all()


def test_gravity_mirror_symmetry():
    """Releasing at +theta vs -theta must mirror the trajectory in x."""
    n, seg, theta = 25, 0.02, np.deg2rad(3.0)
    tip_x = {}
    for s in (+1, -1):
        r = make_rope(n=n, seg=seg,
                      direction=(s * np.sin(theta), -np.cos(theta), 0))
        xs = []
        for _ in range(int(round(2.0 / H))):
            r.advance(H)
            xs.append(float(r.positions()[-1, 0]))
        tip_x[s] = np.array(xs)
    L = (n - 1) * seg
    err = np.max(np.abs(tip_x[+1] + tip_x[-1]))
    print(f"\n  mirror asymmetry: {err/L*100:.3f}% of L")
    assert err < 5e-3 * L


def _circle_traj(radius=0.05, freq=0.5):
    w = 2 * np.pi * freq
    return lambda t: np.array([radius * np.cos(w * t) - radius,
                               0.0,
                               radius * np.sin(w * t)])


def _tip_path(max_h, t_end=1.5, drag=0.5, num_itr=8):
    r = make_rope(n=25, seg=0.02, max_h=max_h, drag=drag)
    r.num_itr = num_itr
    _, tips, _ = drive(r, _circle_traj(), t_end)
    return tips


def test_substep_convergence():
    """Tip path must converge as substeps increase; 50/frame (the planned
    operating point) must already be close to 100/frame."""
    t100 = _tip_path(H / 100)
    t50 = _tip_path(H / 50)
    t25 = _tip_path(H / 25)
    d50 = np.sqrt(np.mean(np.sum((t50 - t100) ** 2, axis=1)))
    d25 = np.sqrt(np.mean(np.sum((t25 - t100) ** 2, axis=1)))
    L = 24 * 0.02
    order = np.log2(d25 / max(d50, 1e-12))
    print(f"\n  RMS tip diff vs 100 substeps: 25ss {d25/L*100:.3f}%L, "
          f"50ss {d50/L*100:.3f}%L (apparent order {order:.2f})")
    assert d50 < 0.7 * d25, "no convergence trend with substep refinement"
    # Measured: 50ss sits ~4%L from 100ss on a fast whirl (order ~1.4).
    # Validation establishes CONVERGENCE; how many substeps the swing task
    # needs is a tuning decision made against task trajectories later.
    assert d50 < 0.06 * L


def test_iteration_count_convergence():
    """Constraints must hold at low iteration counts and trajectories must
    converge monotonically with more iterations. (Measured: 4itr sits ~7%L
    from 16itr on a fast whirl — iterations matter more at rope scale than
    the paper's yarn-scale '~4' suggests. Operating point = tuning decision.)
    """
    t4 = _tip_path(H / 50, num_itr=4)
    t8 = _tip_path(H / 50, num_itr=8)
    t16 = _tip_path(H / 50, num_itr=16)
    L = 24 * 0.02
    d4 = np.sqrt(np.mean(np.sum((t4 - t16) ** 2, axis=1)))
    d8 = np.sqrt(np.mean(np.sum((t8 - t16) ** 2, axis=1)))
    print(f"\n  RMS tip diff vs 16 itr: 4itr {d4/L*100:.2f}%L, "
          f"8itr {d8/L*100:.2f}%L")
    assert d8 < d4, "no convergence with iteration count"
    assert d8 < 0.06 * L

    # the hard constraint (inextensibility) must hold even at numItr=4
    r = make_rope(n=25, seg=0.02, drag=0.5)
    r.num_itr = 4
    drive(r, _circle_traj(), 1.5)
    from conftest import segment_strains
    s = np.max(np.abs(segment_strains(r)))
    print(f"  max |strain| at numItr=4: {s*100:.3f}%")
    assert s < 0.01, "stretch constraint not satisfied at numItr=4"


def test_energy_dissipation_characterization():
    """Quantify intrinsic dissipation vs (substeps, iterations) — informs the
    swing-task accuracy/cost tradeoff. This is a measurement (printed), not a
    bound: the no-gain property is asserted in the test above."""
    from conftest import masses
    n, seg, EA = 33, 0.02, 500.0
    results = {}
    for label, max_h, itr in [("50ss/8it", H / 50, 8),
                              ("100ss/8it", H / 100, 8),
                              ("50ss/32it", H / 50, 32)]:
        r = make_rope(n=n, seg=seg, EA=EA, direction=(1, 0, 0), max_h=max_h)
        r.num_itr = itr
        m = masses(r)
        E_scale = G * np.sum(m * np.arange(n) * seg)
        for _ in range(int(round(3.5 / H))):
            r.advance(H)
        retained = 1.0 + energy_scalar(r, EA) / E_scale
        assert np.isfinite(retained)
        results[label] = retained
    print("\n  energy retained after 3.5 s free swing:")
    for k, v in results.items():
        print(f"    {k:>10s}: {100*v:.1f}%")
