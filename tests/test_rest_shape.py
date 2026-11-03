"""from_positions(straight_rest=...): whether the initial geometry or a straight rope is the rest shape."""
import numpy as np
import pytest
import ropesim

torch = pytest.importorskip("torch")

V, L = 21, 0.05


def _l_shape():
    pos = np.zeros((2, V, 3), np.float32)
    for v in range(V):           # 10 segments down, then 10 along +x: a right-angle fold
        pos[:, v] = (0.0, 0.0, -v * L) if v <= 10 else ((v - 10) * L, 0.0, -10 * L)
    return pos


def _run(batch, seconds=6.0):
    root = torch.tensor(_l_shape()[:, 0], device="cuda")
    zero = torch.zeros((2, 3), device="cuda")
    for _ in range(int(seconds * 60)):
        batch.step(root, zero, 1 / 60, 50)
    P = torch.as_tensor(batch.positions())
    return (P[:, -1] - P[:, 0]).norm(dim=-1).cpu().numpy()


@pytest.mark.parametrize("straight", [False, True])
def test_from_positions_rest_shape(straight):
    b = ropesim.RopeBatch.from_positions(_l_shape(), 0.1, 500.0, 1e-2, 1 / 3000, 8, straight_rest=straight)
    b.set_gravity(np.zeros(3))
    b.drag = 1.0
    end = _run(b)
    full = (V - 1) * L
    if straight:                 # the fold is strain: without gravity the rope unfolds
        assert np.all(end > 0.9 * full), end
    else:                        # the fold is the rest shape: it stays
        assert np.all(end < 0.8 * full), end
