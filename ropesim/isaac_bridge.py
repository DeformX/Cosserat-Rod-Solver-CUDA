"""Isaac Lab bridge for rod-mesh contact (AGENTS 5c): PhysX owns every rigid body, ropesim owns the rope.

Once per scene, each collider body is baked into a signed-distance grid in its own link frame (from the
USD collision prims under the body, or its visual meshes). Every control frame, after PhysX stepped:

    bridge.sync_poses(reset_env_ids)   # link poses -> rope.set_mesh_poses_all (teleport for reset envs)
    rope.step(...)                      # the rope solver interpolates the poses over its substeps
    bridge.apply_wrenches()             # rope.mesh_wrenches -> set_external_force_and_torque (dynamic bodies)

The wrench is the frame average of the force the rope applied to each body, re-expressed about the body's
centre of mass, applied in the world frame during PhysX's next step (explicit coupling, one frame lag: how
PhysX consumes any external force). Kinematic bodies (robot links, tables) get no wrench back.

Pure-numpy parts (bake_body_geometry, ColliderSpec) import without Kit; the USD/Isaac parts are imported
lazily inside Kit.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from . import sdf


@dataclass
class ColliderSpec:
    """One rigid body (or a set of articulation links) the rope can touch."""
    asset: str                          # key in the InteractiveScene (e.g. "robot", "box")
    bodies: list[str] | None = None     # link names of an articulation; None = the asset's root body
    dynamic: bool = False               # True: the rope's wrench is applied back to the body
    mu_s: float | None = None           # friction against the rope (None = the rope's contact material)
    mu_k: float | None = None
    spacing: float = 0.005              # grid cell [m]
    margin: float = 0.03                # lattice padding beyond the surface [m] (>= rope radius + a few cells)
    round: float = 0.0                  # fillet radius: rounds convex edges (no snag when a rope is dragged over
                                        # them, AGENTS 5c) but makes the body FATTER by `round` on every face, so
                                        # the rope then rests `round` above the PhysX surface; 0 = exact shape
    thickness: float = 0.0              # shell thickness for a collider that is open even after welding (0 = auto)


# ------------------------------------------------------------------------------------------ geometry


def _triangulate(counts, indices):
    tris, k = [], 0
    for c in counts:
        for j in range(1, c - 1):
            tris.append((indices[k], indices[k + j], indices[k + j + 1]))
        k += c
    return np.asarray(tris, np.int64).reshape(-1, 3)


def weld(vertices, triangles, tol=1e-6):
    """Merge coincident vertices (exported colliders are unwelded along every seam, which makes a watertight
    mesh look open to is_closed) and drop degenerate triangles."""
    key = np.round(np.asarray(vertices, np.float64) / tol).astype(np.int64)
    _, idx, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    F = inv.reshape(-1)[np.asarray(triangles)]
    F = F[(F[:, 0] != F[:, 1]) & (F[:, 1] != F[:, 2]) & (F[:, 0] != F[:, 2])]
    return np.asarray(vertices)[idx], F


def body_geometry(stage, body_path):
    """Triangles of a body's collider (every prim with UsdPhysics.CollisionAPI under the body prim) in the
    PhysX link frame, welded: (vertices [V,3] float32, triangles [T,3]). Meshes keep their points (the
    convex-hull approximation PhysX may use is not applied: the SDF carries the true shape);
    Cube/Sphere/Cylinder/Capsule prims are tessellated. Must run inside Kit (pxr)."""
    from pxr import Usd, UsdGeom, UsdPhysics

    body = stage.GetPrimAtPath(body_path)
    if not body.IsValid():
        raise ValueError(f"no prim at {body_path}")
    xc = UsdGeom.XformCache()
    # PhysX's link frame is the body prim's world transform WITHOUT its scale (a scaled Cube that is
    # itself the rigid body keeps the scale on the shape), so shapes are taken relative to that frame.
    Mb = np.asarray(xc.GetLocalToWorldTransform(body), np.float64)        # row-vector convention
    Rb = Mb[:3, :3] / np.linalg.norm(Mb[:3, :3], axis=1, keepdims=True)
    RT = np.eye(4); RT[:3, :3] = Rb; RT[3, :3] = Mb[3, :3]
    RT_inv = np.linalg.inv(RT)
    verts, tris = [], []
    for prim in Usd.PrimRange(body, Usd.TraverseInstanceProxies()):
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        if prim.IsA(UsdGeom.Mesh):
            m = UsdGeom.Mesh(prim)
            pts = np.asarray(m.GetPointsAttr().Get(), np.float64)
            t = _triangulate(m.GetFaceVertexCountsAttr().Get(), m.GetFaceVertexIndicesAttr().Get())
        elif prim.IsA(UsdGeom.Cube):
            h = 0.5 * UsdGeom.Cube(prim).GetSizeAttr().Get()
            pts, t = _box_mesh(np.array([h, h, h]))
        elif prim.IsA(UsdGeom.Sphere):
            pts, t = _sphere_mesh(UsdGeom.Sphere(prim).GetRadiusAttr().Get())
        elif prim.IsA(UsdGeom.Cylinder) or prim.IsA(UsdGeom.Capsule):
            g = UsdGeom.Cylinder(prim) if prim.IsA(UsdGeom.Cylinder) else UsdGeom.Capsule(prim)
            pts, t = _cylinder_mesh(g.GetRadiusAttr().Get(), g.GetHeightAttr().Get(), "XYZ".index(g.GetAxisAttr().Get()),
                                    capsule=prim.IsA(UsdGeom.Capsule))
        else:
            continue
        M = np.asarray(xc.GetLocalToWorldTransform(prim), np.float64) @ RT_inv   # shape -> link frame, scales kept
        pts = (np.c_[pts, np.ones(len(pts))] @ M)[:, :3]
        tris.append(t + sum(len(v) for v in verts))
        verts.append(pts)
    if not verts:
        raise ValueError(f"no collision geometry under {body_path}")
    V, F = weld(np.concatenate(verts), np.concatenate(tris))
    return V.astype(np.float32), F


def _box_mesh(h):
    v = np.array([[sx * h[0], sy * h[1], sz * h[2]] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], np.float64)
    f = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]])
    return v, f


def _sphere_mesh(r, n=24):
    th = np.linspace(0, np.pi, n + 1); ph = np.linspace(0, 2 * np.pi, 2 * n, endpoint=False)
    T, P = np.meshgrid(th, ph, indexing="ij")
    v = np.stack([r * np.sin(T) * np.cos(P), r * np.sin(T) * np.sin(P), r * np.cos(T)], -1).reshape(-1, 3)
    f = []
    w = len(ph)
    for i in range(n):
        for j in range(w):
            a, b = i * w + j, i * w + (j + 1) % w
            c, d = a + w, b + w
            if i > 0: f.append((a, c, b))
            if i < n - 1: f.append((b, c, d))
    v, idx = np.unique(np.round(v, 9), axis=0, return_inverse=True)
    f = idx.reshape(-1)[np.asarray(f)]
    return v, f[(f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])]


def _cylinder_mesh(r, h, axis, capsule=False, n=48):
    """Closed cylinder along z (a capsule adds a sphere at each end: overlapping closed parts, whose union
    the winding-number bake resolves), then permuted to the prim's axis."""
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ring = np.stack([r * np.cos(ang), r * np.sin(ang)], 1)
    top = np.c_[ring, np.full(n, h / 2)]; bot = np.c_[ring, np.full(n, -h / 2)]
    v = np.concatenate([top, bot, [[0, 0, h / 2]], [[0, 0, -h / 2]]])
    f = []
    for j in range(n):
        k = (j + 1) % n
        f += [(j, n + k, k), (j, n + j, n + k), (2 * n, j, k), (2 * n + 1, n + k, n + j)]   # outward winding
    f = np.asarray(f)
    if capsule:
        s, sf = _sphere_mesh(r, 16)
        v = np.concatenate([v, s + [0, 0, h / 2], s + [0, 0, -h / 2]])
        f = np.concatenate([f, sf + 2 * n + 2, sf + 2 * n + 2 + len(s)])
    if axis == 0: v = v[:, [2, 0, 1]]                          # rotations only (a swap would turn the mesh inside out)
    elif axis == 1: v = v[:, [0, 2, 1]] * np.array([1.0, 1.0, -1.0])
    return v, f


def bake_body(vertices, triangles, spec: ColliderSpec, name=""):
    """Signed-distance grid of a body's (welded) triangles in its link frame (see sdf.bake_mesh). A collider
    that is still open after welding is baked as a two-sided shell of 2 cells (surface one cell outside the
    mesh, hollow inside) and says so: close such assets in the USD if the rope must not pass through."""
    vertices, triangles = weld(vertices, triangles)
    thickness = spec.thickness
    if thickness == 0.0 and not sdf.is_closed(triangles):
        thickness = 2.0 * spec.spacing
        warnings.warn(f"[RopeColliders] {name or 'collider'} is open ({len(triangles)} tris): baked as a {thickness * 1e3:.0f} mm "
                      f"two-sided shell, its surface {spec.spacing * 1e3:.0f} mm outside the mesh; close the asset if the rope "
                      f"must not pass through it", stacklevel=2)
    return sdf.bake_mesh(vertices, triangles, spec.spacing, spec.margin, thickness=thickness, round=spec.round)


# ------------------------------------------------------------------------------------------ bridge


def _xyzw(q_wxyz):
    return q_wxyz[..., [1, 2, 3, 0]]


@dataclass
class _Entry:
    asset: object            # Articulation or RigidObject
    body: int | None         # body index in an articulation, None for a RigidObject root
    dynamic: bool
    path: str


class RopeColliders:
    """Keeps a RopeBatch's meshes in step with Isaac Lab assets (one mesh per body, per-env poses)."""

    def __init__(self, rope, scene, specs, verbose=True):
        import torch
        from isaaclab.assets import Articulation
        from isaacsim.core.utils.stage import get_current_stage

        self.rope, self.scene = rope, scene
        self.N = scene.num_envs
        self.entries: list[_Entry] = []
        self.geometry = []         # welded (vertices, triangles) per entry, link frame: what was baked
        stage = get_current_stage()
        for spec in specs:
            asset = scene[spec.asset]
            is_art = isinstance(asset, Articulation)
            if is_art:
                names = spec.bodies if spec.bodies is not None else list(asset.body_names)
                ids = [asset.body_names.index(n) for n in names]
                paths = [asset.root_physx_view.link_paths[0][i] for i in ids]
            else:
                ids, paths = [None], [asset.root_physx_view.prim_paths[0]]
            for body_idx, path in zip(ids, paths):
                V, F = body_geometry(stage, path)
                g = bake_body(V, F, spec, path)
                pos, quat = self._pose(asset, body_idx)
                m = rope.add_mesh(*g.as_args(), pos=pos.cpu().numpy(), quat=quat.cpu().numpy(), mu_s=spec.mu_s, mu_k=spec.mu_k)
                self.entries.append(_Entry(asset, body_idx, spec.dynamic, path))
                self.geometry.append((V, F))
                if verbose:
                    print(f"[RopeColliders] mesh {m}: {path} ({len(F)} tris -> {'x'.join(map(str, g.shape))} nodes, "
                          f"{g.values.nbytes / 1e6:.1f} MB, {'dynamic' if spec.dynamic else 'kinematic'})", flush=True)
        self.M = len(self.entries)
        self._pos = torch.zeros((self.N, self.M, 3), device="cuda")
        self._quat = torch.zeros((self.N, self.M, 4), device="cuda")
        self._tele = torch.zeros(self.N, dtype=torch.bool, device="cuda")
        self.dynamic_ids = [i for i, e in enumerate(self.entries) if e.dynamic]
        if self.dynamic_ids:
            rope.enable_contact_forces(True)

    @staticmethod
    def _pose(asset, body_idx):
        d = asset.data
        if body_idx is None:
            return d.root_link_pos_w, _xyzw(d.root_link_quat_w)
        return d.body_link_pos_w[:, body_idx], _xyzw(d.body_link_quat_w[:, body_idx])

    @staticmethod
    def _com(asset, body_idx):
        d = asset.data
        return d.root_com_pos_w if body_idx is None else d.body_com_pos_w[:, body_idx]

    def sync_poses(self, reset_env_ids=None):
        """After PhysX stepped (and resets were written): every body's link pose -> the rope's mesh targets.
        reset_env_ids: envs whose bodies were teleported this frame (their meshes jump instead of moving)."""
        for i, e in enumerate(self.entries):
            pos, quat = self._pose(e.asset, e.body)
            self._pos[:, i] = pos
            self._quat[:, i] = quat
        self._tele.zero_()
        if reset_env_ids is not None and len(reset_env_ids):
            self._tele[reset_env_ids] = True
        self.rope.set_mesh_poses_all(self._pos, self._quat, self._tele)

    def wrenches(self):
        """[N, M, 6] frame-averaged (force, torque about the link origin) BY the rope ON each body, world
        frame, for the last rope step (a copy: the solver's buffer is re-used by the next step)."""
        import torch
        return torch.as_tensor(self.rope.mesh_wrenches_view())[:, : self.M].clone()

    def apply_wrenches(self):
        """After rope.step(): the rope's reaction on every dynamic body, as an external wrench at its centre of
        mass for PhysX's next step (the caller's scene.write_data_to_sim() sends it)."""
        if not self.dynamic_ids:
            return
        import torch
        W = self.wrenches()
        for i in self.dynamic_ids:
            e = self.entries[i]
            F, T = W[:, i, :3], W[:, i, 3:]
            pos, _ = self._pose(e.asset, e.body)
            T = T - torch.cross(self._com(e.asset, e.body) - pos, F, dim=-1)     # torque about the COM
            body_ids = [e.body] if e.body is not None else None
            composer = getattr(e.asset, "permanent_wrench_composer", None)        # Isaac Lab >= 2.3 (the old call warns every frame)
            if composer is not None:
                composer.set_forces_and_torques(forces=F[:, None], torques=T[:, None], body_ids=body_ids, is_global=True)
            else:
                e.asset.set_external_force_and_torque(F[:, None], T[:, None], body_ids=body_ids, is_global=True)
