"""Write a standalone HTML player for a recorded rope scene (three.js from a CDN, data embedded).

Coordinates: the solver is z-up; the viewer is y-up. `to_view` converts positions and
`quat_to_view` converts (x, y, z, w) quaternions: viewer = (x, z, -y), a rotation of -90 deg about x.

Scene dict (all positions already in VIEWER coordinates):
  title, caption, legend [{color, text}], fps, frames, camera {pos, target, fov?}, shadow {center, extent},
  floor None | {y}, speed?, status? [str per frame],
  ropes: [{pos: float32 [F, V, 3], radius, color, material?, samples?, radial?}],
  meshes: {name: (vertices [N, 3], faces [M, 3], flat?)},
  bodies: [{mesh, color, material?, poses: float32 [F, 7] (xyz + quat xyzw) | None, pos?, quat?}]
"""
import base64
import json
import pathlib

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
_C = np.array([-np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)])  # rotation of -90 deg about x, (x, y, z, w)


def to_view(p):
    p = np.asarray(p, np.float64)
    return np.stack([p[..., 0], p[..., 2], -p[..., 1]], axis=-1)


def quat_mul(a, b):
    ax, ay, az, aw = np.moveaxis(np.asarray(a, np.float64), -1, 0)
    bx, by, bz, bw = np.moveaxis(np.asarray(b, np.float64), -1, 0)
    return np.stack([aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw,
                     aw * bw - ax * bx - ay * by - az * bz], axis=-1)


def quat_to_view(q):
    """Rotation q (x, y, z, w, solver frame) expressed in viewer axes: c q c^-1."""
    q = np.asarray(q, np.float64)
    c = np.broadcast_to(_C, q.shape)
    cinv = c * np.array([-1, -1, -1, 1.0])
    return quat_mul(quat_mul(c, q), cinv)


def quat_rotate(q, v):
    q = np.asarray(q, np.float64); v = np.asarray(v, np.float64)
    u, w = q[..., :3], q[..., 3:4]
    return v + 2 * np.cross(u, np.cross(u, v) + w * v)


def _b64(b):
    return base64.b64encode(b).decode("ascii")


def pack_mesh(v, f, flat=False):
    v = np.asarray(v, np.float64).reshape(-1, 3)
    lo, hi = v.min(0), v.max(0)
    c = (lo + hi) / 2
    s = np.maximum((hi - lo) / 2 / 32000.0, 1e-9)
    q = np.round((v - c) / s).astype(np.int16)
    f = np.asarray(f).reshape(-1, 3)
    wide = len(v) > 65535
    fb = (f.astype(np.uint32) if wide else f.astype(np.uint16)).tobytes()
    return {"v": _b64(q.tobytes()), "f": _b64(fb), "scale": s.tolist(), "offset": c.tolist(),
            "wide": wide, "flat": bool(flat), "nv": int(len(v)), "nf": int(len(f))}


def write_html(path, scene):
    ropes = scene["ropes"]
    allp = np.concatenate([np.asarray(r["pos"], np.float64).reshape(-1, 3) for r in ropes]) if ropes else np.zeros((1, 3))
    lo, hi = allp.min(0), allp.max(0)
    offset = (lo + hi) / 2
    scale = max(float((hi - lo).max()) / 2 / 32000.0, 1e-6)
    F = int(scene["frames"])
    out_ropes = []
    for r in ropes:
        p = np.asarray(r["pos"], np.float64)
        assert p.shape[0] == F and p.shape[2] == 3, p.shape
        q = np.round((p - offset) / scale).astype(np.int16)
        rec = {"n": int(p.shape[1]), "radius": float(r["radius"]), "color": r["color"], "pos": _b64(q.tobytes()),
               "scale": scale, "offset": offset.tolist()}
        for k in ("material", "samples", "radial"):
            if k in r:
                rec[k] = r[k]
        out_ropes.append(rec)
    meshes = {k: pack_mesh(*v) for k, v in scene.get("meshes", {}).items()}
    bodies = []
    for b in scene.get("bodies", []):
        rec = {"mesh": b["mesh"], "color": b["color"]}
        if b.get("material"):
            rec["material"] = b["material"]
        if b.get("poses") is not None:
            P = np.asarray(b["poses"], np.float32)
            assert P.shape == (F, 7), P.shape
            rec["poses"] = _b64(P.tobytes())
        if b.get("pos") is not None:
            rec["pos"] = [float(x) for x in b["pos"]]
        if b.get("quat") is not None:
            rec["quat"] = [float(x) for x in b["quat"]]
        bodies.append(rec)
    data = {"title": scene["title"], "caption": scene.get("caption", ""), "legend": scene.get("legend", []),
            "fps": scene["fps"], "frames": F, "camera": scene["camera"], "shadow": scene.get("shadow"),
            "floor": scene.get("floor"), "speed": scene.get("speed", 1), "status": scene.get("status"),
            "ropes": out_ropes, "meshes": meshes, "bodies": bodies}
    js = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    core = (HERE / "viewer_core.js").read_text()
    html = TEMPLATE.replace("{{TITLE}}", scene["title"]).replace("{{DATA}}", js).replace("{{CORE_JS}}", core)
    pathlib.Path(path).write_text(html)
    return len(html)


TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{{TITLE}}</title>
<style>
:root {
  --bg: #f2f4f7; --bg2: #d9dee6; --panel: rgba(255,255,255,.84); --fg: #161b22; --muted: #5b6673; --line: #d3d9e0;
  --accent: #0f766e; --accent-fg: #ffffff; --floor: #dde2e8; --grid: #c4ccd5; --hemi-sky: #ffffff; --hemi-ground: #8e98a3;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg: #151b22; --bg2: #090c10; --panel: rgba(21,28,36,.84); --fg: #e4e9ef; --muted: #96a3b3; --line: #273240;
    --accent: #2fc4b0; --accent-fg: #06201c; --floor: #1a2330; --grid: #2b3745; --hemi-sky: #c9d4e0; --hemi-ground: #2a323c; color-scheme: dark;
  }
}
:root[data-theme="dark"] {
  --bg: #151b22; --bg2: #090c10; --panel: rgba(21,28,36,.84); --fg: #e4e9ef; --muted: #96a3b3; --line: #273240;
  --accent: #2fc4b0; --accent-fg: #06201c; --floor: #1a2330; --grid: #2b3745; --hemi-sky: #c9d4e0; --hemi-ground: #2a323c; color-scheme: dark;
}
[hidden] { display: none !important; }
html, body { margin: 0; height: 100%; background: var(--bg); color: var(--fg);
  font: 14px/1.4 system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; overflow: hidden; }
#view { position: fixed; inset: 0; background: radial-gradient(ellipse at 50% 40%, var(--bg) 0%, var(--bg2) 100%); }
#view canvas { display: block; width: 100%; height: 100%; }
.hud { position: fixed; left: 16px; top: 14px; max-width: min(540px, calc(100vw - 32px)); pointer-events: none; }
.hud h1 { margin: 0; font-size: 19px; font-weight: 650; letter-spacing: -.01em; }
.hud p { margin: 4px 0 0; color: var(--muted); font-size: 13px; }
.legend { position: fixed; right: 16px; top: 14px; display: flex; flex-direction: column; gap: 6px; align-items: flex-end; pointer-events: none; }
.chip { display: inline-flex; align-items: center; gap: 8px; background: var(--panel); border: 1px solid var(--line);
  border-radius: 999px; padding: 4px 10px 4px 6px; font-size: 12px; backdrop-filter: blur(6px); }
.chip i { width: 12px; height: 12px; border-radius: 50%; display: inline-block; flex: none; border: 1px solid var(--line); }
.bar { position: fixed; left: 16px; right: 16px; bottom: 14px; display: flex; align-items: center; gap: 10px;
  background: var(--panel); border: 1px solid var(--line); border-radius: 12px; padding: 8px 12px; backdrop-filter: blur(8px); }
button, select { font: inherit; color: var(--fg); background: transparent; border: 1px solid var(--line); border-radius: 8px; padding: 4px 10px; cursor: pointer; }
button.on { background: var(--accent); color: var(--accent-fg); border-color: var(--accent); }
input[type=range] { flex: 1; min-width: 60px; accent-color: var(--accent); }
.time { font-variant-numeric: tabular-nums; min-width: 92px; color: var(--muted); font-size: 12px; }
.status { font-variant-numeric: tabular-nums; color: var(--muted); font-size: 12px; white-space: nowrap; }
.err { position: fixed; left: 16px; bottom: 70px; color: #c0392b; }
@media (max-width: 720px) { .legend, .status, .hud p { display: none; } }
</style>
</head>
<body>
<div id="view"></div>
<div class="hud"><h1 id="title"></h1><p id="caption"></p></div>
<div class="legend" id="legend"></div>
<div class="bar">
  <button id="play" class="on" title="space">Pause</button>
  <span class="time" id="time">0.00 / 0.00 s</span>
  <input id="scrub" type="range" min="0" max="1000" value="0" aria-label="time">
  <select id="speed" aria-label="playback speed">
    <option value="0.1">0.1×</option><option value="0.25">0.25×</option><option value="0.5">0.5×</option><option value="1" selected>1×</option><option value="2">2×</option>
  </select>
  <button id="orbit" title="slowly orbit the camera">Orbit</button>
  <span class="status" id="status"></span>
</div>
<div class="err" id="err" hidden></div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
<script id="scene-data" type="application/json">{{DATA}}</script>
<script>
{{CORE_JS}}
</script>
</body>
</html>
"""
