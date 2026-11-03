"""A single rope hangs from a fixed root and settles under gravity.

Checks the solver's stretch stiffness against theory: k_stretch is EA in newtons, so the total
elongation of a hanging rope is m g L / (2 EA). Run time: a second or two on any GPU.

    python examples/hanging_rope.py [--plot]
"""
import argparse

import numpy as np

import ropesim

NV, SEG = 33, 0.03                 # 32 segments x 3 cm = 0.96 m
RHO, EA, EI = 0.1, 500.0, 1e-4     # kg/m, N, N m^2
H, SUB = 1 / 60, 50                # control frame and substeps per frame


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--seconds", type=float, default=4.0)
    p.add_argument("--plot", action="store_true", help="draw the settled rope with matplotlib")
    args = p.parse_args()

    rope = ropesim.RopeSim(num_verts=NV, seg_len=SEG, dir=[0.0, 0.0, -1.0], density=RHO,
                           k_stretch=EA, k_bend=EI, max_h=H / SUB, num_itr=8)
    rope.set_gravity([0.0, 0.0, -9.81])
    rope.damping = 0.0                # keep 0: YarnBall's damping brakes absolute velocity
    rope.drag = 2.0                   # heavy air drag so the swing dies out quickly
    for _ in range(int(args.seconds / H)):
        rope.advance(H)

    P = np.asarray(rope.positions())
    L0 = (NV - 1) * SEG
    L = float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum())
    theory = RHO * L0 * 9.81 * L0 / (2 * EA)
    print(f"rest length {L0:.3f} m, settled length {L:.5f} m")
    print(f"elongation {L - L0:.3e} m, theory m g L / (2 EA) = {theory:.3e} m")
    print(f"tip at ({P[-1, 0]:+.4f}, {P[-1, 1]:+.4f}, {P[-1, 2]:+.4f}) m")
    if args.plot:
        import matplotlib.pyplot as plt
        plt.plot(P[:, 0], P[:, 2], ".-")
        plt.axis("equal"); plt.xlabel("x [m]"); plt.ylabel("z [m]"); plt.title("settled rope")
        plt.show()


if __name__ == "__main__":
    main()
