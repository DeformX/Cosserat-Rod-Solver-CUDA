"""Shadow Hand (MuJoCo Menagerie) forward kinematics + mesh export for the string demo.

Runs in a venv with mujoco (not the solver env):  python hand_export.py right_hand.xml spec.json out.npz
spec.json:
  fps            frame rate of the output
  posture        {joint: angle} static joint angles (unlisted joints = 0)
  posture_low    {"LF": [dy, dz], ...} optional: those fingers' J3/J2 (J1 = 0.6 J2) are solved so that the lowest point of
                 their fingertip sits at that (y, z) offset from the anchor point under `posture`; a null dy leaves y
                 free (only J3 is solved, the finger pivots at its knuckle to the height dz); an optional third element
                 [J3, J2] is the solver's first start (the (y, z) map can have two solutions: pick the straighter one)
  joint_keys     [[t, {joint: angle}, mode?], ...] optional joint schedule; unlisted joints keep `posture`; `mode` ("smooth"
                 default, or "linear") interpolates from that key to the next. A key may hold "_low": {"FF": [dy, dz], ...}
                 instead of angles for those fingers (solved as for posture_low), or "_rel": {joint: delta}: the previous
                 key's angles plus these deltas (clamped to the joint range), e.g. a flick of one joint.
  finger_dir, palm_dir   world directions (z-up) of the straight hand's finger axis and of the palm's facing side
  anchor         body whose lowest collision point follows `path` at every frame (under that frame's joint angles,
                 so the base compensates finger motion of the anchor); default rh_ffdistal
  path           [[t, x, y, z, mode?], ...] world path of that point; smoothstep from a waypoint to the next unless its
                 mode is "linear"; the base only translates
Collision shape per body: its visual mesh when watertight, else the visual mesh's convex hull; palm/wrist/forearm use the
model's collision primitives (they never reach the strings). So what touches the strings is what is drawn.
Output npz: names (collision bodies), vis_names, per body col_v_/col_f_/vis_v_/vis_f_ (body frame), vis_rgba_<b>,
            poses_<b> [F, 7] (world xyz + quat xyzw), low_<b> [F, 3] (world position of the lowest collision vertex), frames, fps.
"""
import json
import sys

import mujoco
import numpy as np
import trimesh

PRIMITIVE_BODIES = ("rh_forearm", "rh_wrist", "rh_palm")


def smoothstep(a):
    a = np.clip(a, 0.0, 1.0)
    return a * a * (3 - 2 * a)


def quat_wxyz_to_mat(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def mat_to_quat_xyzw(R):
    q = np.empty(4)
    tr = np.trace(R)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2; q[3] = 0.25 * s; q[0] = (R[2, 1] - R[1, 2]) / s; q[1] = (R[0, 2] - R[2, 0]) / s; q[2] = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2; q[3] = (R[2, 1] - R[1, 2]) / s; q[0] = 0.25 * s; q[1] = (R[0, 1] + R[1, 0]) / s; q[2] = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2; q[3] = (R[0, 2] - R[2, 0]) / s; q[0] = (R[0, 1] + R[1, 0]) / s; q[1] = 0.25 * s; q[2] = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2; q[3] = (R[1, 0] - R[0, 1]) / s; q[0] = (R[0, 2] + R[2, 0]) / s; q[1] = (R[1, 2] + R[2, 1]) / s; q[2] = 0.25 * s
    return q / np.linalg.norm(q)


def geom_mesh(m, g):
    """trimesh of geom g in its body frame (primitives from size; meshes from the model's vertex data)."""
    t, size = int(m.geom_type[g]), m.geom_size[g]
    T = mujoco.mjtGeom
    if t == T.mjGEOM_BOX:
        mesh = trimesh.creation.box(extents=2 * size)
    elif t == T.mjGEOM_CAPSULE:
        mesh = trimesh.creation.capsule(radius=size[0], height=2 * size[1], count=[24, 24])
    elif t == T.mjGEOM_CYLINDER:
        mesh = trimesh.creation.cylinder(radius=size[0], height=2 * size[1], sections=32)
    elif t == T.mjGEOM_SPHERE:
        mesh = trimesh.creation.icosphere(subdivisions=2, radius=size[0])
    elif t == T.mjGEOM_MESH:
        i = int(m.geom_dataid[g])
        v = m.mesh_vert[m.mesh_vertadr[i]:m.mesh_vertadr[i] + m.mesh_vertnum[i]]
        f = m.mesh_face[m.mesh_faceadr[i]:m.mesh_faceadr[i] + m.mesh_facenum[i]]
        mesh = trimesh.Trimesh(vertices=np.asarray(v, np.float64), faces=np.asarray(f, np.int64), process=False)
    else:
        return None
    M = np.eye(4); M[:3, :3] = quat_wxyz_to_mat(m.geom_quat[g]); M[:3, 3] = m.geom_pos[g]
    mesh.apply_transform(M)
    return mesh


def interp_keys(keys, t, default):
    """keys: [[t, {k: v}, mode?], ...] sorted; returns {k: v} at time t (`default` for joints a key does not list)."""
    ks = keys
    val = lambda e, jn: e[1].get(jn, default.get(jn, 0.0))
    if t <= ks[0][0]:
        return {jn: val(ks[0], jn) for jn in ks[0][1]}
    if t >= ks[-1][0]:
        return {jn: val(ks[-1], jn) for jn in ks[-1][1]}
    i = max(j for j in range(len(ks)) if ks[j][0] <= t)
    u = (t - ks[i][0]) / (ks[i + 1][0] - ks[i][0])
    if len(ks[i]) < 3 or ks[i][2] != "linear":
        u = smoothstep(u)
    return {jn: val(ks[i], jn) + (val(ks[i + 1], jn) - val(ks[i], jn)) * u for jn in set(ks[i][1]) | set(ks[i + 1][1])}


J1_COUPLING = 0.6      # J1 = J1_COUPLING * J2 (a natural finger curls its last two joints together)


def solve_finger(m, d, jadr, col_tip, btip, R_base, q, pre, target, x0):
    """Joint angles {name: value} of finger `pre` that put the lowest point of its fingertip (rotated frame) at target
    (y, z), and the residual [m]. J3 and J2 are solved with J1 = J1_COUPLING * J2. A NaN y leaves y free: only J3 is
    solved (J2 and J1 stay as in q), i.e. the finger pivots at its knuckle until the point reaches the height z."""
    from scipy.optimize import least_squares
    free_y = not np.isfinite(target[0])
    jn = {j: f"rh_{pre}J{j}" for j in (3, 2, 1)}

    def angles(x):
        return {jn[3]: x[0]} if free_y else {jn[3]: x[0], jn[2]: x[1], jn[1]: J1_COUPLING * x[1]}

    def residual(x):
        qq = q.copy()
        for name, a in angles(x).items():
            qq[jadr(name)] = a
        d.qpos[:] = qq; mujoco.mj_kinematics(m, d)
        w = (col_tip @ quat_wxyz_to_mat(d.xquat[btip]).T + d.xpos[btip]) @ R_base.T
        lo = w[np.argmin(w[:, 2])]
        return lo[2:] - target[1:] if free_y else lo[1:] - target

    if free_y:
        starts, bounds = [x0[:1], [0.1], [0.5], [1.0]], ([-0.26], [1.57])
    else:                                      # the lowest-point map is piecewise (the lowest vertex changes): several starts
        starts, bounds = [x0, [0.3, 0.3], [0.1, 0.6], [0.8, 0.2], [1.2, 0.7]], ([-0.26, 0.0], [1.57, 1.57])
    best = None
    for start in starts:
        r = least_squares(residual, np.asarray(start, float), bounds=bounds, xtol=1e-7, ftol=1e-10)
        err = float(np.linalg.norm(r.fun))
        if best is None or err < best[1]:
            best = (r.x, err)
        if err < 2e-4:
            break
    return {name: float(a) for name, a in angles(best[0]).items()}, best[1]


def main():
    xml, spec_path, out = sys.argv[1:4]
    spec = json.load(open(spec_path))
    m = mujoco.MjModel.from_xml_path(xml)
    d = mujoco.MjData(m)
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in range(m.nbody)]
    bid = {n: i for i, n in enumerate(names)}
    jadr = lambda jn: m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jn)]
    posture = spec.get("posture", {})
    qpos0 = np.zeros(m.nq)
    for jn, ang in posture.items():
        qpos0[jadr(jn)] = ang

    # meshes per body
    out_d = {}
    col_names, vis_names, col_v = [], [], {}
    for b in range(1, m.nbody):
        gs = [g for g in range(m.ngeom) if m.geom_bodyid[g] == b]
        vis = [geom_mesh(m, g) for g in gs if m.geom_group[g] == 2]
        vis = [v for v in vis if v is not None]
        if vis:
            vm = trimesh.util.concatenate(vis); vis_names.append(names[b])
            out_d[f"vis_v_{names[b]}"] = np.asarray(vm.vertices, np.float32); out_d[f"vis_f_{names[b]}"] = np.asarray(vm.faces, np.int32)
            g0 = [g for g in gs if m.geom_group[g] == 2][0]
            out_d[f"vis_rgba_{names[b]}"] = np.asarray(m.geom_rgba[g0], np.float32)
        if names[b] in PRIMITIVE_BODIES or not vis:
            col = [geom_mesh(m, g) for g in gs if m.geom_group[g] == 3]
            col = [c.convex_hull if int(m.geom_type[g]) == mujoco.mjtGeom.mjGEOM_MESH else c
                   for c, g in zip(col, [g for g in gs if m.geom_group[g] == 3]) if c is not None]
            cm = trimesh.util.concatenate(col) if col else None
        else:
            cm = vm if vm.is_watertight else vm.convex_hull
        if cm is not None:
            col_names.append(names[b]); col_v[names[b]] = np.asarray(cm.vertices, np.float64)
            out_d[f"col_v_{names[b]}"] = np.asarray(cm.vertices, np.float32); out_d[f"col_f_{names[b]}"] = np.asarray(cm.faces, np.int32)

    # reference frame from the STRAIGHT hand (qpos = 0): index finger axis = knuckle -> distal origin; palm normal =
    # the direction the index tip moves when the proximal joint curls. Both in the model frame.
    ff = bid["rh_ffdistal"]
    d.qpos[:] = 0; mujoco.mj_kinematics(m, d)
    axis_model = d.xpos[ff] - d.xpos[bid["rh_ffknuckle"]]; axis_model /= np.linalg.norm(axis_model)
    Rff0 = quat_wxyz_to_mat(d.xquat[ff])
    cv = col_v["rh_ffdistal"]
    tip_local = cv[np.argmax((cv @ Rff0.T) @ axis_model)]
    tip0 = d.xpos[ff] + Rff0 @ tip_local
    q2 = np.zeros(m.nq); q2[jadr("rh_FFJ3")] = 0.5
    d.qpos[:] = q2; mujoco.mj_kinematics(m, d)
    tip1 = d.xpos[ff] + quat_wxyz_to_mat(d.xquat[ff]) @ tip_local
    palm_model = tip1 - tip0; palm_model -= axis_model * (palm_model @ axis_model); palm_model /= np.linalg.norm(palm_model)
    fd = np.asarray(spec.get("finger_dir", [-1, 0, 0]), float); pd = np.asarray(spec.get("palm_dir", [0, 0, -1]), float)
    fd /= np.linalg.norm(fd); pd -= fd * (pd @ fd); pd /= np.linalg.norm(pd)
    Mm = np.stack([axis_model, palm_model, np.cross(axis_model, palm_model)], 1)
    Mw = np.stack([fd, pd, np.cross(fd, pd)], 1)
    R_base = Mw @ Mm.T

    # anchor: the lowest collision vertex of `anchor` (rotated frame, base at the origin) under joint angles q
    anchor = spec.get("anchor", "rh_ffdistal")
    ba = bid[anchor]

    def anchor_point(q):
        d.qpos[:] = q; mujoco.mj_kinematics(m, d)
        wa = (col_v[anchor] @ quat_wxyz_to_mat(d.xquat[ba]).T + d.xpos[ba]) @ R_base.T
        return wa[np.argmin(wa[:, 2])]
    tip_body = lambda pre: f"rh_{pre.lower()}distal"

    def resolve(entry, q_static, x0s):
        """Replace entry["_low"] = {pre: [dy, dz]} by solved joint angles. Targets are offsets from the anchor point
        under the entry's own posture (q_static + the entry's explicit angles); a null dy leaves y free."""
        q = q_static.copy()
        for jn, ang in entry.items():
            if jn != "_low":
                q[jadr(jn)] = ang
        ref = anchor_point(q)
        worst = 0.0
        for pre, off in entry.pop("_low").items():
            tgt = ref[1:] + np.array([np.nan if v is None else v for v in off[:2]], float)
            x0 = np.asarray(off[2], float) if len(off) > 2 else x0s.get(pre, np.array([0.3, 0.3]))
            sol, err = solve_finger(m, d, jadr, col_v[tip_body(pre)], bid[tip_body(pre)], R_base, q, pre, tgt, x0)
            worst = max(worst, err)
            entry.update(sol)
            for jn, ang in sol.items():
                q[jadr(jn)] = ang
            x0s[pre] = np.array([q[jadr(f"rh_{pre}J3")], q[jadr(f"rh_{pre}J2")]])
        return worst
    x0s = {}; worst = 0.0
    if spec.get("posture_low"):
        e = {"_low": spec["posture_low"]}
        worst = max(worst, resolve(e, qpos0, x0s))
        posture.update(e)
        for jn, ang in e.items():
            qpos0[jadr(jn)] = ang

    fps = spec["fps"]; wp = np.asarray([e[:4] for e in spec["path"]], float)
    linear = [len(e) > 4 and e[4] == "linear" for e in spec["path"]]
    frames = int(round(wp[-1, 0] * fps)) + 1
    keys = sorted(spec.get("joint_keys") or [], key=lambda e: e[0])
    prev = {}
    for e in keys:
        if "_low" in e[1]:
            worst = max(worst, resolve(e[1], qpos0, x0s))
        if "_rel" in e[1]:                      # carry the previous key's joints, add the deltas
            rel = e[1].pop("_rel")
            for jn in set(prev) | set(rel):
                j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jn)
                a = prev.get(jn, posture.get(jn, 0.0)) + rel.get(jn, 0.0)
                e[1][jn] = float(np.clip(a, *m.jnt_range[j])) if m.jnt_limited[j] else float(a)
            if rel:
                print(f"  rel key t={e[0]:.2f}: " + " ".join(f"{jn[3:]} {prev.get(jn, 0.0):.2f}->{e[1][jn]:.2f}" for jn in rel))
        prev = e[1]
    if worst:
        print(f"finger IK: worst residual {worst * 1000:.2f} mm")
    poses = {n: np.zeros((frames, 7), np.float32) for n in set(col_names) | set(vis_names)}
    lows = {n: np.zeros((frames, 3), np.float32) for n in col_names}
    for k in range(frames):
        t = k / fps
        if t <= wp[0, 0]:
            target = wp[0, 1:]
        elif t >= wp[-1, 0]:
            target = wp[-1, 1:]
        else:
            i = np.searchsorted(wp[:, 0], t, side="right") - 1
            u = (t - wp[i, 0]) / (wp[i + 1, 0] - wp[i, 0])
            target = wp[i, 1:] + (wp[i + 1, 1:] - wp[i, 1:]) * (u if linear[i] else smoothstep(u))
        q = qpos0.copy()
        if keys:
            for jn, ang in interp_keys(keys, t, posture).items():
                q[jadr(jn)] = ang
        p_base = target - anchor_point(q)        # sets d.qpos = q and runs the kinematics
        for n in poses:
            b = bid[n]
            Rw = R_base @ quat_wxyz_to_mat(d.xquat[b]); pw = p_base + R_base @ d.xpos[b]
            poses[n][k, :3] = pw; poses[n][k, 3:] = mat_to_quat_xyzw(Rw)
            if n in lows:
                w = col_v[n] @ Rw.T + pw
                lows[n][k] = w[np.argmin(w[:, 2])]
    for n, P in poses.items():
        out_d[f"poses_{n}"] = P
    for n, P in lows.items():
        out_d[f"low_{n}"] = P
    out_d["names"] = np.array(col_names); out_d["vis_names"] = np.array(vis_names)
    out_d["frames"] = frames; out_d["fps"] = fps
    np.savez_compressed(out, **out_d)
    print(f"exported {len(col_names)} collision bodies, {len(vis_names)} visual bodies, {frames} frames; "
          f"palm_model {np.round(palm_model, 3)}, axis_model {np.round(axis_model, 3)}, anchor {anchor}")
    for n in ("rh_ffdistal", "rh_mfdistal", "rh_rfdistal", "rh_lfdistal", "rh_thdistal"):
        print(f"  frame0 lowest point of {n}: {np.round(lows[n][0], 4)}")


if __name__ == "__main__":
    main()
