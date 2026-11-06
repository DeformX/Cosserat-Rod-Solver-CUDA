"""Shared helpers for the showcase scenes: output location, geometry, recording, pin driving."""
import os
import time

import numpy as np
import torch

DEMO_DIR = os.environ.get("ROPE_DEMOS_DIR", os.path.expanduser("~/rope_demos"))   # recordings; assets/ and the mujoco venv live here too


def smoothstep(a):
    a = np.clip(a, 0.0, 1.0)
    return a * a * (3 - 2 * a)


def quat_from_axis_angle(axis, angle):
    axis = np.asarray(axis, np.float64)
    axis = axis / np.linalg.norm(axis)
    s = np.sin(angle / 2)
    return np.array([axis[0] * s, axis[1] * s, axis[2] * s, np.cos(angle / 2)])


def segment_strain(pos, rest):
    """pos [N, V, 3], rest [N, V-1] -> strain [N, V-1]."""
    seg = np.linalg.norm(np.diff(pos, axis=1), axis=-1)
    return seg / rest - 1.0


class Recorder:
    def __init__(self):
        self.ropes = []     # list of [R, V, 3] per frame (solver frame)
        self.bodies = {}    # name -> list of (pos[3], quat[4]) per frame
        self.status = []
        self.t0 = time.time()

    def frame(self, pos, bodies=None, status=""):
        self.ropes.append(np.asarray(pos, np.float32).copy())
        for k, v in (bodies or {}).items():
            self.bodies.setdefault(k, []).append((np.asarray(v[0], np.float32).copy(), np.asarray(v[1], np.float32).copy()))
        self.status.append(status)

    @property
    def frames(self):
        return len(self.ropes)

    def rope_array(self):
        return np.stack(self.ropes)   # [F, R, V, 3]

    def body_poses(self, name):
        p = np.stack([x[0] for x in self.bodies[name]]); q = np.stack([x[1] for x in self.bodies[name]])
        return p, q


def device_positions(batch):
    return torch.as_tensor(batch.positions()).cpu().numpy()


def device_velocities(batch):
    return torch.as_tensor(batch.velocities()).cpu().numpy()


def drive_pinned(batch, pinned_mask, targets_now, targets_next, dt):
    """Set the kinematic velocity of pinned vertices so they reach `targets_next` after one frame.

    pinned_mask [N, V] bool; targets [N, V, 3] (only pinned entries are read)."""
    v = device_velocities(batch)
    vel = (targets_next - targets_now) / dt
    v[pinned_mask] = vel[pinned_mask]
    batch.set_velocities(np.ascontiguousarray(v, np.float32))

