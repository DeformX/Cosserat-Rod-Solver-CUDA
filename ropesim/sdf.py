"""Signed-distance grids for rod-mesh contact (AGENTS §5c).

A grid is phi sampled on a node lattice in the object's own frame: node
(i, j, k) sits at origin + (i, j, k) * spacing, values[i, j, k] = signed
distance (negative inside). Pass it to RopeSim/RopeBatch.add_mesh(values,
origin, spacing). Contact happens where phi < rope radius, so the lattice
must extend at least `margin` >= radius beyond the surface everywhere the rope
can reach; outside the lattice there is no contact at all.

bake_mesh() uses Warp's GPU closest-point query with winding-number signs
(robust to small holes); from_function() samples a closed-form SDF.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SdfGrid:
    values: np.ndarray      # float32 [nx, ny, nz], C order
    origin: np.ndarray      # float32 [3], object frame
    spacing: float

    @property
    def shape(self):
        return self.values.shape

    def nodes(self):
        """Lattice node coordinates [nx, ny, nz, 3] in the object frame."""
        ax = [self.origin[d] + self.spacing * np.arange(self.values.shape[d]) for d in range(3)]
        return np.stack(np.meshgrid(*ax, indexing="ij"), -1).astype(np.float32)

    def as_args(self):
        """(values, origin, spacing) for add_mesh()."""
        return self.values, self.origin, float(self.spacing)

    def sample(self, points):
        """Host-side check of a scene: trilinear phi at object-frame points [M, 3] (the solver's
        own interpolation, in float64) and a mask of the points inside the lattice; outside it
        phi is nan and the solver applies no contact. Use it to verify that a rope starts
        outside every object (phi > rope radius) and inside the lattice wherever it can go."""
        p = np.asarray(points, np.float64).reshape(-1, 3)
        u = (p - self.origin.astype(np.float64)) / self.spacing
        dims = np.array(self.values.shape)
        inside = (u >= 0).all(1) & (u < dims - 1).all(1)
        phi = np.full(len(p), np.nan)
        if inside.any():
            ui = u[inside]
            i = np.floor(ui).astype(int)
            f = ui - i
            c = lambda dx, dy, dz: self.values[i[:, 0] + dx, i[:, 1] + dy, i[:, 2] + dz].astype(np.float64)
            c00 = c(0, 0, 0) + f[:, 0] * (c(1, 0, 0) - c(0, 0, 0)); c01 = c(0, 0, 1) + f[:, 0] * (c(1, 0, 1) - c(0, 0, 1))
            c10 = c(0, 1, 0) + f[:, 0] * (c(1, 1, 0) - c(0, 1, 0)); c11 = c(0, 1, 1) + f[:, 0] * (c(1, 1, 1) - c(0, 1, 1))
            c0 = c00 + f[:, 1] * (c10 - c00); c1 = c01 + f[:, 1] * (c11 - c01)
            phi[inside] = c0 + f[:, 2] * (c1 - c0)
        return phi, inside


def _lattice(lo, hi, spacing):
    lo = np.asarray(lo, np.float64)
    hi = np.asarray(hi, np.float64)
    dims = np.maximum(np.ceil((hi - lo) / spacing).astype(int) + 1, 2)
    return lo.astype(np.float32), dims


def _slabs(g, chunk_nodes=2_000_000):
    """Yield (i0, i1, points [n, 3] float64) over x-slabs of the lattice (bounded memory)."""
    nx, ny, nz = g.values.shape
    ax = [g.origin[d].astype(np.float64) + g.spacing * np.arange(g.values.shape[d]) for d in range(3)]
    step = max(1, chunk_nodes // (ny * nz))
    for i0 in range(0, nx, step):
        i1 = min(nx, i0 + step)
        yield i0, i1, np.stack(np.meshgrid(ax[0][i0:i1], ax[1], ax[2], indexing="ij"), -1).reshape(-1, 3)


def from_function(phi, lo, hi, spacing):
    """Sample a closed-form signed distance phi(points [M, 3]) -> [M] on the box [lo, hi]
    (evaluated in x-slabs of ~2M nodes; the grid itself is 4 bytes per node)."""
    origin, dims = _lattice(lo, hi, spacing)
    g = SdfGrid(np.empty(tuple(dims), np.float32), origin, float(spacing))
    for i0, i1, pts in _slabs(g):
        g.values[i0:i1] = np.asarray(phi(pts), np.float32).reshape(i1 - i0, dims[1], dims[2])
    return g


def is_closed(triangles):
    """True when every edge is shared by exactly two triangles (a closed, manifold surface)."""
    t = np.asarray(triangles, np.int64).reshape(-1, 3)
    e = np.sort(np.concatenate([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]]), axis=1)
    _, counts = np.unique(e, axis=0, return_counts=True)
    return bool((counts == 2).all())


def bake_mesh(vertices, triangles, spacing, margin, thickness=0.0, round=0.0, device=None):
    """Distance grid of a triangle mesh on its padded bounding box.

    vertices [V, 3], triangles [T, 3], any winding. thickness == 0: SIGNED distance
    (sign from the generalized winding number), only for a closed mesh (every edge
    shared by exactly two triangles). thickness > 0: an open mesh (heightfield,
    sheet) becomes a two-sided shell: unsigned distance minus thickness / 2, and
    thickness must be >= 2 * spacing so the interpolated distance has a defined
    gradient at both faces (a bare unsigned distance has a flat, gradient-free
    cell at the sheet, and ropes fall through it). thickness on a CLOSED mesh
    hollows it (only a shell of that thickness is solid). is_closed() is
    topological: an unwelded but watertight asset is refused (weld it first).
    round > 0 offsets the surface outward by that much (phi - round), which rounds
    every convex edge with that radius but also makes the object fatter by `round`
    on every face (a mesh cannot be shrunk first; sdf_box can): round convex edges
    to about half the rope's segment length, or a rope dragged over a sharp edge
    snags (see AGENTS 5c).
    margin [m] pads the box on every side beyond the shell and rounding (use >=
    rope radius + a few cells).
    """
    import warp as wp

    if hasattr(wp, "LOG_WARNING"):      # warp >= 1.5; Kit's bundled warp only has `quiet`
        wp.config.log_level = wp.LOG_WARNING
    else:
        wp.config.quiet = True
    wp.init()
    vertices = np.asarray(vertices, np.float32).reshape(-1, 3)
    triangles = np.asarray(triangles, np.int32).reshape(-1, 3)
    if len(vertices) < 3 or len(triangles) < 1:
        raise ValueError("bake_mesh needs a triangle mesh")
    if triangles.min() < 0 or triangles.max() >= len(vertices):
        raise ValueError("triangle indices out of range")
    if thickness > 0:
        if thickness < 2 * spacing:
            raise ValueError("thickness must be >= 2 * spacing (one cell on each side of the mid-surface)")
    elif not is_closed(triangles):
        raise ValueError("open mesh (some edge is not shared by exactly two triangles): a signed distance "
                         "has no inside. If the asset is watertight but unwelded (duplicate vertices along "
                         "seams, common in exported USD/OBJ), weld it first (trimesh.merge_vertices); "
                         "otherwise pass thickness >= 2 * spacing to bake it as a two-sided shell")
    if round < 0:
        raise ValueError("round must be >= 0")
    pad = margin + 0.5 * thickness + round
    lo, hi = vertices.min(0) - pad, vertices.max(0) + pad
    origin, dims = _lattice(lo, hi, spacing)
    g = SdfGrid(np.empty(tuple(dims), np.float32), origin, float(spacing))
    with wp.ScopedDevice(device or wp.get_device()):
        mesh = wp.Mesh(points=wp.array(vertices, dtype=wp.vec3),
                       indices=wp.array(triangles.ravel(), dtype=wp.int32),
                       support_winding_number=True)
        for i0, i1, pts in _slabs(g):
            out = wp.zeros(len(pts), dtype=wp.float32)
            wp.launch(_sdf_kernel, dim=len(pts), inputs=[mesh.id, wp.array(pts.astype(np.float32), dtype=wp.vec3), out])
            phi = out.numpy()
            if thickness > 0:
                phi = np.abs(phi) - 0.5 * thickness
            g.values[i0:i1] = (phi - round).reshape(i1 - i0, dims[1], dims[2])
    if not np.isfinite(g.values).all():
        raise ValueError("bake_mesh produced non-finite distances")
    return g


try:
    import warp as _wp

    @_wp.kernel
    def _sdf_kernel(mesh_id: _wp.uint64, pts: _wp.array(dtype=_wp.vec3), out: _wp.array(dtype=float)):
        i = _wp.tid()
        q = _wp.mesh_query_point_sign_winding_number(mesh_id, pts[i], 1.0e6, 2.0, 0.5)
        cp = _wp.mesh_eval_position(mesh_id, q.face, q.u, q.v)
        out[i] = q.sign * _wp.length(pts[i] - cp)
except ImportError:          # bake_mesh() raises when called; from_function() still works
    _sdf_kernel = None


# ---- closed-form shapes (object frame), for tests and simple scenes ----------

def sdf_box(half, centre=(0, 0, 0), round=0.0):
    """Axis-aligned box of half extents `half` at `centre` (exact outside and inside), its
    edges rounded with radius `round` (the overall size stays `half`).

    Put the object origin on the face the rope will rest on: the contact gap is resolved
    at float32 steps of the vertex's object-frame coordinate (7 nm at 0.13 m, 60 nm at
    0.8 m) against a rest sink of only g h^2 / kappa^2 (68 nm at h = 1/6000, kappa = 2).
    Round convex edges to about half the rope's segment length (AGENTS 5c)."""
    half = np.asarray(half, np.float64) - round
    if (half < 0).any():
        raise ValueError("round must not exceed the smallest half extent")
    centre = np.asarray(centre, np.float64)

    def phi(p):
        d = np.abs(p - centre) - half
        return np.linalg.norm(np.maximum(d, 0), axis=1) + np.minimum(d.max(1), 0) - round
    return phi


def sdf_sphere(radius, centre=(0, 0, 0)):
    c = np.asarray(centre, np.float64)
    return lambda p: np.linalg.norm(p - c, axis=1) - radius


def sdf_cylinder(radius, axis=2):
    """Infinite cylinder of `radius` along coordinate `axis`."""
    keep = [d for d in range(3) if d != axis]
    return lambda p: np.linalg.norm(p[:, keep], axis=1) - radius


def sdf_bowl(radius, centre=(0, 0, 0)):
    """Spherical cavity: negative outside the sphere, so the rope lives inside it."""
    s = sdf_sphere(radius, centre)
    return lambda p: -s(p)
