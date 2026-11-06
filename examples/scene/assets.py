"""Mesh asset helpers: load, weld, close holes, bake signed-distance grids."""
import numpy as np


def close_holes(mesh):
    """Fan-triangulate every boundary loop of a trimesh so every edge is shared by two triangles."""
    import trimesh
    mesh = mesh.copy()
    mesh.merge_vertices()
    faces = [np.asarray(mesh.faces)]
    edges = mesh.edges_sorted
    uniq, inv, counts = np.unique(edges, axis=0, return_inverse=True, return_counts=True)
    boundary = uniq[counts == 1]
    if len(boundary) == 0:
        return mesh
    # walk boundary loops
    adj = {}
    for a, b in boundary:
        adj.setdefault(int(a), []).append(int(b)); adj.setdefault(int(b), []).append(int(a))
    seen = set()
    loops = []
    for start in adj:
        if start in seen:
            continue
        loop = [start]; seen.add(start); prev, cur = None, start
        while True:
            nxt = [n for n in adj[cur] if n != prev and n not in seen]
            if not nxt:
                break
            prev, cur = cur, nxt[0]; loop.append(cur); seen.add(cur)
        if len(loop) >= 3:
            loops.append(loop)
    V = np.asarray(mesh.vertices)
    verts, fan = [V], []
    for k, loop in enumerate(loops):            # one centre vertex per loop, fan from it
        verts.append(V[loop].mean(0)[None])
        for i in range(len(loop)):
            fan.append([loop[i], loop[(i + 1) % len(loop)], len(V) + k])
    out = trimesh.Trimesh(vertices=np.concatenate(verts), faces=np.concatenate([faces[0], np.asarray(fan)]), process=False)
    trimesh.repair.fix_normals(out)
    return out


def bake_closed(vertices, triangles, spacing, margin, round=0.0):
    """ropesim.sdf.bake_mesh with the topological check relaxed to 'no edge shared by > 2 triangles'
    (the sign comes from the generalized winding number, which tolerates small holes)."""
    from ropesim import sdf
    t = np.asarray(triangles, np.int64).reshape(-1, 3)
    e = np.sort(np.concatenate([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]]), axis=1)
    _, counts = np.unique(e, axis=0, return_counts=True)
    if sdf.is_closed(t):
        return sdf.bake_mesh(vertices, triangles, spacing, margin, round=round)
    if (counts > 2).any():
        raise ValueError("non-manifold mesh")
    orig = sdf.is_closed                        # bypass the check for this bake only
    sdf.is_closed = lambda tri: True
    try:
        return sdf.bake_mesh(vertices, triangles, spacing, margin, round=round)
    finally:
        sdf.is_closed = orig
