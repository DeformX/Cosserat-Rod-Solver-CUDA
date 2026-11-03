"""Equivalence & reproducibility.

- De-collisioned config (detectionPeriod=-1, useStepSizeLimit=0) must match
  the upstream collisions-on config for a contact-free scene — this is the
  direct check that stripping collisions didn't alter free-space dynamics.
- Determinism: the kernels use chaotic Gauss-Seidel (racy neighbor reads), so
  bitwise reproducibility is not guaranteed; quantify the actual spread that
  RL reproducibility will live with.
"""
import numpy as np
import pytest

from conftest import H, drive, make_rope


def _gentle_sway(t):
    return np.array([0.03 * np.sin(2 * np.pi * 0.5 * t), 0.0, 0.0])


def _run(r, t_end=1.5):
    _, tips, _ = drive(r, _gentle_sway, t_end)
    return tips


def test_collisions_on_off_equivalent_when_contact_free():
    tips = {}
    for col in (False, True):
        r = make_rope(n=25, direction=(0, -1, 0), collisions=col)
        tips[col] = _run(r)
    L = 24 * 0.02
    d = np.max(np.linalg.norm(tips[False] - tips[True], axis=1))
    print(f"\n  max tip divergence collisions on/off: {d/L*100:.3f}%L")
    assert d < 0.01 * L, \
        "de-collisioned config altered contact-free dynamics"


def test_determinism_across_identical_sims():
    a = _run(make_rope(n=25, direction=(0, -1, 0)))
    b = _run(make_rope(n=25, direction=(0, -1, 0)))
    d = np.max(np.linalg.norm(a - b, axis=1))
    bitwise = np.array_equal(a, b)
    print(f"\n  identical-sim divergence: {d:.2e} m (bitwise equal: {bitwise})")
    assert d < 1e-3  # physical reproducibility envelope for RL


def test_reset_reproducibility():
    r = make_rope(n=25, direction=(0, -1, 0))
    a = _run(r)
    r.reset()
    b = _run(r)
    d = np.max(np.linalg.norm(a - b, axis=1))
    print(f"\n  run/reset/rerun divergence: {d:.2e} m")
    assert d < 1e-3
