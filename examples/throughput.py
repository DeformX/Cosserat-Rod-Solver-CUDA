"""RopeBatch throughput sweep (solver only, no Isaac).

    python examples/throughput.py            # N = 64..2048
    python examples/throughput.py 64 4096    # custom list

Reports both driving paths: legacy constant-velocity (Sim::advance) and the
second-order motion program (our substep loop + per-substep kernel).
Config matches the env scaffold: V=33 verts, 50 substeps, 8 iterations.
"""
import sys
import time

import numpy as np
import torch

import ropesim

H = 1.0 / 60.0
V = 33
SUBSTEPS = 50
FRAMES = 120


def bench(N, program):
    g = int(np.ceil(np.sqrt(N)))
    roots = np.zeros((N, 3), np.float32)
    roots[:, 0] = (np.arange(N) % g) * 2.0
    roots[:, 1] = (np.arange(N) // g) * 2.0
    roots[:, 2] = 1.5
    b = ropesim.RopeBatch(num_envs=N, verts_per_rope=V, root_pos=roots,
                          dir=[0.0, 0.0, -1.0], fix_root_orientation=True)
    b.set_gravity([0.0, 0.0, -9.81])
    b.damping = 0.0
    b.drag = 0.05
    pos = torch.tensor(roots, device="cuda")
    vel = torch.zeros(N, 3, device="cuda")
    q = torch.zeros(N, 4, device="cuda")
    q[:, 3] = 1.0

    def step():
        if program:
            b.step(pos, vel, dt=H, substeps=SUBSTEPS, root_q=q,
                   root_pos_end=pos, root_q_end=q)
        else:
            b.step(pos, vel, dt=H, substeps=SUBSTEPS, root_q=q)

    step()  # warm-up: CUDA graph capture
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(FRAMES):
        step()
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    assert np.isfinite(torch.as_tensor(b.positions()).cpu().numpy()).all()
    return FRAMES / wall


def main():
    ns = [int(a) for a in sys.argv[1:]] or [64, 128, 256, 512, 1024, 2048]
    print(f"\nRopeBatch throughput — V={V}, {SUBSTEPS} substeps x 8 iters/frame, "
          f"{FRAMES} frames per point\n")
    print(f"{'N':>6} | {'legacy FPS':>10} | {'program FPS':>11} | "
          f"{'rt x (prog)':>11} | {'env-steps/s':>12}")
    print("-" * 64)
    for n in ns:
        f_leg = bench(n, program=False)
        f_prog = bench(n, program=True)
        print(f"{n:>6} | {f_leg:>10.0f} | {f_prog:>11.0f} | "
              f"{f_prog/60.0:>10.1f}x | {n*f_prog:>12.0f}")


if __name__ == "__main__":
    main()
