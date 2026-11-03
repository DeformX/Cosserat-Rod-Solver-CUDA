"""A rope dropped across a box on the floor, then the box slides and carries the rope along.

Rod-mesh contact against a signed-distance grid (ropesim.sdf), rod-plane contact with the
floor, friction, the force the rope exerts on the box (the contact-force readout), and a
kinematic mesh whose pose is set every frame (the rope feels the moving surface).

    python examples/mesh_contact.py
"""
import numpy as np
import torch

import ropesim
from ropesim import sdf

NV, SEG = 25, 0.05                              # 1.2 m rope
RHO, EA, EI, R = 0.117, 11473.0, 4.48e-3, 0.00635
DT, SUB, ITR = 1 / 60, 100, 8
HALF = np.array([0.2, 0.2, 0.1])                # box half extents; it rests on the floor
BOX = np.array([0.0, 0.0, 0.1], np.float32)


def main():
    # the rope starts horizontal, 40 cm up, crossing the box diagonally (free: no pinned vertex)
    s = (np.arange(NV) - (NV - 1) / 2) * SEG
    pos = np.stack([s / np.sqrt(2), s / np.sqrt(2), np.full(NV, 0.4)], -1)[None].astype(np.float32)
    rope = ropesim.RopeBatch.from_positions(pos, RHO, EA, EI, DT / SUB, ITR, pinned=np.zeros((1, NV), bool))
    rope.set_gravity(np.array([0.0, 0.0, -9.81]))
    rope.damping = 0.0                          # keep 0: YarnBall's damping brakes absolute velocity
    rope.drag = 0.2
    rope.set_planes(np.array([[0, 0, 1.0, 0.0]], np.float32), radius=R, mu_s=0.6, mu_k=0.5)   # floor n.x >= 0 + the material
    grid = sdf.from_function(sdf.sdf_box(HALF, round=0.02), -HALF - 0.06, HALF + 0.06, 0.01)
    box = rope.add_mesh(*grid.as_args(), pos=BOX)                              # one box, posed at BOX
    rope.enable_contact_forces(True)

    def report(t):
        P = torch.as_tensor(rope.positions())[0]
        F = torch.as_tensor(rope.mesh_wrenches_view())[0, 0, :3].cpu().numpy()
        print(f"t={t:4.1f}s  rope z in [{P[:, 2].min().item():.4f}, {P[:, 2].max().item():.4f}] m  "
              f"x mean {P[:, 0].mean().item():+.3f} m  force on box ({F[0]:+.3f}, {F[1]:+.3f}, {F[2]:+.3f}) N")

    print("drop: the rope lands on the box and the floor")
    for f in range(int(2.0 / DT)):
        rope.step_free(DT, SUB)
        if f % 30 == 29:
            report((f + 1) * DT)
    weight = RHO * (NV - 1) * SEG * 9.81
    print(f"(rope weight {weight:.3f} N; the box carries the part resting on it, the floor the rest; "
          f"a vertex on the box top sits at z = {2 * HALF[2] + R:.4f} m)")

    print("slide: the box moves +x at 0.2 m/s; friction on its top carries the rope, the ends drag on the floor")
    pos_t = torch.tensor(BOX, device="cuda").reshape(1, 3)
    quat_t = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device="cuda")
    for f in range(int(3.0 / DT)):
        pos_t[0, 0] = 0.2 * (f + 1) * DT
        rope.set_mesh_poses(box, pos_t, quat_t)        # pose at the end of this frame
        rope.step_free(DT, SUB)
        if f % 30 == 29:
            report(2.0 + (f + 1) * DT)
    F = torch.as_tensor(rope.mesh_wrenches_view())[0, 0, :3].cpu().numpy()
    floor = weight + F[2]                                  # F_z < 0 is the share of the weight on the box
    expect = 0.5 * floor + RHO * (NV - 1) * SEG * rope.drag * 0.2   # floor mu_k x load + air drag at 0.2 m/s
    print(f"(steady pull on the box F_x = {F[0]:+.3f} N; floor friction on the dragged ends plus air drag "
          f"predicts {-expect:+.3f} N)")


if __name__ == "__main__":
    main()
