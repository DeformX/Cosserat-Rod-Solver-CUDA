"""Repeatability across fresh CUDA objects with moving, endpoint-pinned strings."""
import numpy as np
import pytest
import torch

import ropesim


@pytest.mark.parametrize("strings, vertices, iterations", [
    (1, 3, 3), (1, 63, 3), (1, 64, 3), (1, 127, 3), (1, 128, 3),
    (6, 124, 3), (6, 124, 24),
])
def test_deterministic_state_replays_across_cuda_blocks(strings, vertices, iterations):
    dt, substeps = 1 / 480, 32
    u = np.linspace(0, 1, vertices, dtype=np.float32)
    initial = np.zeros((strings, vertices, 3), dtype=np.float32)
    initial[:, :, 0] = 0.64 * u
    initial[:, :, 1] = np.arange(strings)[:, None] * 0.009
    initial[:, :, 2] = (0.002 + 0.001 * np.arange(strings)[:, None]) * np.sin(u * np.pi)
    pins = np.zeros(initial.shape[:2], dtype=bool)
    pins[:, [0, -1]] = True
    roots = initial[:, 0].copy()
    # Precompute the identical commands for each independent batch.
    program = []
    for step in range(16):
        start, end = roots.copy(), roots.copy()
        start[:, 0] -= 0.00001 * step
        end[:, 0] -= 0.00001 * (step + 1)
        program.append(tuple(torch.as_tensor(x, device="cuda") for x in
                             (start, (end - start) / dt, end)))

    def run():
        batch = ropesim.RopeBatch.from_positions(
            initial, 0.005, 7500.0, 1e-7, dt / substeps, iterations,
            pinned=pins, straight_rest=True, deterministic=True,
        )
        batch.set_gravity([0, 0, -9.81])
        batch.damping = 0.0
        batch.drag = 20.0
        states = []
        for start, velocity, end in program:
            batch.step(start, velocity, dt, substeps, root_pos_end=end)
            state = tuple(torch.as_tensor(read()).cpu().numpy() for read in
                          (batch.positions, batch.velocities, batch.quaternions))
            assert all(np.isfinite(array).all() for array in state)
            states.append(state)
        return states

    first, second = run(), run()
    assert not np.array_equal(first[-1][0][:, 1:-1], initial[:, 1:-1])
    for expected, actual in zip(first, second):
        for x, y in zip(expected, actual):
            np.testing.assert_array_equal(x.view("uint32"), y.view("uint32"))
