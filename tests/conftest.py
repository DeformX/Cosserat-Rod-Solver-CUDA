"""Shared helpers for the ropesim validation suite.

Conventions
-----------
- Units: density = rho [kg/m], k_stretch = EA [N], k_bend = EI [N*m^2].
- Every rope from make_rope() has drag=0, damping=0 unless overridden —
  the solver's *defaults* (drag=0.2) are dissipative and would mask physics.
- `H` is the control-rate frame time (PhysX step in the real integration);
  max_h = H/substeps is the solver substep.
- The position/orientation sweeps use chaotic Gauss-Seidel iterations on GPU
  (cross-thread neighbor reads race benignly), so bitwise reproducibility is
  NOT expected; tests compare with physical tolerances.
"""
import numpy as np
import pytest

import ropesim

G = 9.8
H = 1.0 / 60.0  # control frame


def make_rope(n=33, seg=0.02, direction=(0.0, -1.0, 0.0), density=0.1,
              EA=500.0, EI=1e-6, max_h=1.0 / 3000.0, drag=0.0, damping=0.0,
              **kw):
    r = ropesim.RopeSim(num_verts=n, seg_len=seg, dir=list(direction),
                        density=density, k_stretch=EA, k_bend=EI,
                        max_h=max_h, **kw)
    r.set_gravity([0.0, -G, 0.0])
    r.drag = drag
    r.damping = damping
    return r


def rope_length(r):
    return float(np.sum(r.rest_lengths()))


def masses(r):
    """Vertex masses; pinned verts (invMass==0) get mass 0 (they never move)."""
    inv = np.asarray(r.inv_masses(), dtype=np.float64)
    m = np.zeros_like(inv)
    nz = inv > 0
    m[nz] = 1.0 / inv[nz]
    return m


def segment_strains(r):
    """(len - rest)/rest per segment, from current positions."""
    p = np.asarray(r.positions(), dtype=np.float64)
    lens = np.linalg.norm(np.diff(p, axis=0), axis=1)
    rest = np.asarray(r.rest_lengths(), dtype=np.float64)
    return (lens - rest) / rest


def settle(r, t=3.0, drag=3.0):
    """Run with heavy drag to reach (near) static equilibrium, then restore."""
    old = r.drag
    r.drag = drag
    for _ in range(int(round(t / H))):
        r.advance(H)
    r.drag = old


def drive(r, traj, t_end, rate=60.0, record_every=None):
    """Drive the root along traj(t)->(3,) at the given control rate.

    Each frame sets pos=traj(t), vel=(traj(t+1/rate)-traj(t))*rate, then
    advances 1/rate — the piecewise-constant-velocity sweep the Isaac Lab
    integration will use. Returns (times, tips[,3], roots[,3]).
    """
    h = 1.0 / rate
    nfr = int(round(t_end * rate))
    every = record_every if record_every is not None else 1
    times, tips, roots = [], [], []
    for k in range(nfr):
        t = k * h
        p0 = np.asarray(traj(t), dtype=np.float64)
        p1 = np.asarray(traj(t + h), dtype=np.float64)
        r.set_root(p0.tolist(), ((p1 - p0) * rate).tolist())
        r.advance(h)
        if (k + 1) % every == 0:
            p = np.asarray(r.positions(), dtype=np.float64)
            times.append((k + 1) * h)
            tips.append(p[-1])
            roots.append(p[0])
    return np.array(times), np.array(tips), np.array(roots)


def total_energy(r, y_ref=0.0):
    """KE + gravitational PE + axial elastic PE [J].

    SCOPE: the elastic term uses centerline strain only. The solver's stretch
    energy is the full vector constraint (segment vector vs frame x-axis, i.e.
    includes a shear-like direction term), and bend/orientation energy is
    omitted entirely. Valid for energy accounting when EI is negligible and
    the root frame is not driven; do NOT use to certify no-energy-injection
    in orientation-driven or stiff-bend scenarios."""
    p = np.asarray(r.positions(), dtype=np.float64)
    v = np.asarray(r.velocities(), dtype=np.float64)
    m = masses(r)
    ke = 0.5 * np.sum(m * np.sum(v * v, axis=1))
    pe = G * np.sum(m * (p[:, 1] - y_ref))
    rest = np.asarray(r.rest_lengths(), dtype=np.float64)
    eps = segment_strains(r)
    # E_seg = 0.5*EA*eps^2*l ; recover per-segment EA from kStretch=EA*l is not
    # exposed, so tests that need elastic PE pass uniform EA explicitly.
    return ke, pe, rest, eps


def energy_scalar(r, EA, y_ref=0.0):
    ke, pe, rest, eps = total_energy(r, y_ref)
    elastic = 0.5 * EA * np.sum(eps * eps * rest)
    return ke + pe + elastic


def quat_rotate(q, v):
    """Rotate v by quaternion q=(x,y,z,w) — matches Kit::Rotor layout."""
    q = np.asarray(q, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    u, w = q[:3], q[3]
    return v + 2.0 * np.cross(u, np.cross(u, v) + w * v)


def axis_angle_quat(axis, angle):
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    s = np.sin(angle / 2.0)
    return np.array([axis[0] * s, axis[1] * s, axis[2] * s, np.cos(angle / 2.0)])
