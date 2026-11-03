"""Many ropes in one solver, each root driven around a circle, and the throughput you get.

RopeBatch packs every rope into one flat vertex array; the roots are driven from CUDA tensors
with the second-order motion program (start velocity + frame-end pose), and positions are read
back as zero-copy torch tensors. This is the pattern an RL environment uses.

    python examples/batch_swing.py --num_ropes 1024 --seconds 3
"""
import argparse
import time

import numpy as np
import torch

import ropesim

NV, SEG = 21, 0.05                                 # 1 m ropes
RHO, EA, EI = 0.117, 11473.0, 4.48e-3              # a 12.7 mm braided nylon rope
DT, SUB, ITR = 1 / 60, 100, 8


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--num_ropes", type=int, default=1024)
    p.add_argument("--seconds", type=float, default=3.0)
    args = p.parse_args()
    N = args.num_ropes

    # circle centres on a grid 1.5 m up; each rope hangs straight down from its root's start
    # point on the circle, root (vertex 0) pinned
    omega, radius = 2 * np.pi * 1.0, 0.3           # 1 Hz circle of radius 30 cm
    side = int(np.ceil(np.sqrt(N)))
    g = np.stack(np.meshgrid(np.arange(side), np.arange(side), indexing="ij"), -1).reshape(-1, 2)[:N]
    roots = np.c_[g * 2.0, np.full(N, 1.5)].astype(np.float32)
    start = roots + np.array([radius, 0.0, 0.0], np.float32)
    pos = start[:, None, :] + np.arange(NV)[None, :, None] * np.array([0, 0, -SEG], np.float32)
    pin = np.zeros((N, NV), bool)
    pin[:, 0] = True
    rope = ropesim.RopeBatch.from_positions(pos, RHO, EA, EI, DT / SUB, ITR, pinned=pin)
    rope.set_gravity(np.array([0.0, 0.0, -9.81]))
    rope.damping = 0.0                             # keep 0: YarnBall's damping brakes absolute velocity
    rope.drag = 0.2

    roots_t = torch.tensor(roots, device="cuda")

    def circle(t):                                 # root positions and velocities at time t
        c, s = np.cos(omega * t), np.sin(omega * t)
        off = torch.tensor([radius * c, radius * s, 0.0], device="cuda", dtype=torch.float32)
        vel = torch.tensor([-radius * omega * s, radius * omega * c, 0.0], device="cuda", dtype=torch.float32)
        return (roots_t + off).contiguous(), vel.expand(N, 3).contiguous()

    frames = int(args.seconds / DT)
    torch.cuda.synchronize()
    t_start = time.perf_counter()
    for f in range(frames):
        p0, v0 = circle(f * DT)
        p1, _ = circle((f + 1) * DT)
        rope.step(p0, v0, DT, SUB, root_pos_end=p1)       # quadratic from (p0, v0) landing on p1
    torch.cuda.synchronize()
    wall = time.perf_counter() - t_start

    P = torch.as_tensor(rope.positions())                 # [N, NV, 3], zero-copy view of the solver's buffer
    tip = P[:, -1]
    print(f"{N} ropes x {NV} vertices, {SUB} substeps x {ITR} iterations per 60 Hz frame")
    print(f"{frames} frames in {wall:.2f} s: {frames / wall:.1f} frames/s, {N * frames / wall / 1e3:.1f}k rope-steps/s, "
          f"{frames / wall / 60:.2f}x realtime")
    print(f"tip height mean {tip[:, 2].mean().item():.3f} m, spread {tip[:, 2].std().item():.4f} m "
          f"(all ropes see the same motion)")


if __name__ == "__main__":
    main()
