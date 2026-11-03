"""Long-horizon stability under RL-like random driving (slow)."""
import numpy as np
import pytest

from conftest import H, drive, make_rope, rope_length, segment_strains


@pytest.mark.slow
@pytest.mark.parametrize("num_itr,drag_quad", [
    (8, 0.0),      # legacy operating point
    (16, 0.094),   # 2026-08-11 env default + quadratic cross-flow drag
])
def test_long_random_driving_stable(num_itr, drag_quad):
    rng = np.random.default_rng(0)
    amps = rng.uniform(0.03, 0.08, size=(3, 3))
    freqs = rng.uniform(0.3, 1.5, size=(3, 3))
    phases = rng.uniform(0, 2 * np.pi, size=(3, 3))

    def traj(t):
        return np.sum(amps * np.sin(2 * np.pi * freqs * t + phases), axis=1) \
            * np.array([1.0, 0.3, 1.0])

    r = make_rope(n=33, direction=(0, -1, 0), drag=0.05)
    r.num_itr = num_itr
    r.drag_quad = drag_quad
    L = rope_length(r)
    t_end = 30.0
    _, tips, roots = drive(r, traj, t_end)

    p = np.asarray(r.positions(), dtype=np.float64)
    assert np.isfinite(p).all()
    assert np.isfinite(tips).all()
    # rope never exceeds physical reach from the root
    reach = np.max(np.linalg.norm(tips - roots, axis=1))
    print(f"\n  max tip-from-root reach {reach/L:.3f} L over {t_end:.0f} s")
    assert reach < 1.5 * L
    strain = np.max(np.abs(segment_strains(r)))
    print(f"  final max |strain| {strain*100:.2f}%")
    assert strain < 0.10
    assert r.last_error == 0 and r.last_warning == 0
    # root still tracking the commanded trajectory exactly
    expect = traj(t_end)
    np.testing.assert_allclose(roots[-1], expect, atol=1e-3)
