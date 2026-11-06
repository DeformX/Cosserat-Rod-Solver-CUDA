// Generic player for recorded rope scenes (see viewer.py for the data layout).
(function () {
  "use strict";
  const $ = (id) => document.getElementById(id);
  if (typeof THREE === "undefined") {
    const e = $("err"); e.hidden = false; e.textContent = "three.js did not load: this page needs internet access for the CDN scripts.";
    return;
  }
  const DATA = JSON.parse($("scene-data").textContent);
  const tok = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
  const esc = (x) => String(x).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

  // ---------- decoding ----------
  function b64(b) { const s = atob(b), u = new Uint8Array(s.length); for (let i = 0; i < s.length; i++) u[i] = s.charCodeAt(i); return u; }
  const decI16 = (b) => new Int16Array(b64(b).buffer);
  const decF32 = (b) => new Float32Array(b64(b).buffer);
  function meshGeometry(rec) {
    const q = decI16(rec.v), v = new Float32Array(q.length);
    for (let i = 0; i < q.length; i++) v[i] = q[i] * rec.scale[i % 3] + rec.offset[i % 3];
    const f = rec.wide ? new Uint32Array(b64(rec.f).buffer) : new Uint16Array(b64(rec.f).buffer);
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(v, 3));
    g.setIndex(new THREE.BufferAttribute(f, 1));
    if (rec.flat) { const gg = g.toNonIndexed(); gg.computeVertexNormals(); return gg; }
    g.computeVertexNormals();
    return g;
  }

  // ---------- smooth tube through control points ----------
  class RopeTube {
    constructor(n, radius, color, opts = {}) {
      this.n = n; this.radius = radius;
      this.samples = opts.samples || Math.max(24, Math.round(n * 2.5));
      this.radial = opts.radial || 10;
      this.ctrl = []; for (let i = 0; i < n; i++) this.ctrl.push(new THREE.Vector3());
      this.curve = new THREE.CatmullRomCurve3(this.ctrl, false, "centripetal");
      const S = this.samples + 1, K = this.radial;
      const pos = new Float32Array(S * K * 3), nor = new Float32Array(S * K * 3), idx = [];
      for (let i = 0; i < S - 1; i++) for (let j = 0; j < K; j++) {
        const a = i * K + j, b = i * K + (j + 1) % K, c = (i + 1) * K + j, d = (i + 1) * K + (j + 1) % K;
        idx.push(a, b, c, b, d, c);
      }
      const g = new THREE.BufferGeometry();
      g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
      g.setAttribute("normal", new THREE.BufferAttribute(nor, 3));
      g.setIndex(idx);
      this.geo = g;
      this.mat = new THREE.MeshStandardMaterial(Object.assign({ color, roughness: 0.55, metalness: 0.05 }, opts.material || {}));
      this.mesh = new THREE.Mesh(g, this.mat);
      this.mesh.castShadow = this.mesh.receiveShadow = true; this.mesh.frustumCulled = false;
      const capG = new THREE.SphereGeometry(radius, 12, 8);
      this.caps = [new THREE.Mesh(capG, this.mat), new THREE.Mesh(capG, this.mat)];
      for (const c of this.caps) c.castShadow = true;
      this.group = new THREE.Group(); this.group.add(this.mesh, this.caps[0], this.caps[1]);
      this._P = new THREE.Vector3(); this._T = new THREE.Vector3(); this._N = new THREE.Vector3();
      this._B = new THREE.Vector3(); this._tmp = new THREE.Vector3();
    }
    update() {
      const S = this.samples + 1, K = this.radial, R = this.radius;
      const pos = this.geo.attributes.position.array, nor = this.geo.attributes.normal.array;
      const P = this._P, T = this._T, N = this._N, B = this._B, tmp = this._tmp;
      for (let i = 0; i < S; i++) {
        const u = i / (S - 1);
        this.curve.getPoint(u, P); this.curve.getTangent(u, T);
        if (T.lengthSq() < 1e-14) T.set(0, 1, 0); T.normalize();
        if (i === 0) {
          tmp.set(0, 1, 0); if (Math.abs(T.dot(tmp)) > 0.9) tmp.set(1, 0, 0);
          N.crossVectors(T, tmp).normalize();
        } else {
          N.addScaledVector(T, -T.dot(N));
          if (N.lengthSq() < 1e-12) { tmp.set(0, 1, 0); if (Math.abs(T.dot(tmp)) > 0.9) tmp.set(1, 0, 0); N.crossVectors(T, tmp); }
          N.normalize();
        }
        B.crossVectors(T, N);
        for (let j = 0; j < K; j++) {
          const a = j / K * 2 * Math.PI, ca = Math.cos(a), sa = Math.sin(a), o = (i * K + j) * 3;
          const nx = ca * N.x + sa * B.x, ny = ca * N.y + sa * B.y, nz = ca * N.z + sa * B.z;
          nor[o] = nx; nor[o + 1] = ny; nor[o + 2] = nz;
          pos[o] = P.x + R * nx; pos[o + 1] = P.y + R * ny; pos[o + 2] = P.z + R * nz;
        }
        if (i === 0) this.caps[0].position.copy(P);
        if (i === S - 1) this.caps[1].position.copy(P);
      }
      this.geo.attributes.position.needsUpdate = true; this.geo.attributes.normal.needsUpdate = true;
    }
  }

  // ---------- renderer, camera, lights ----------
  const viewEl = $("view");
  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.shadowMap.enabled = true; renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  renderer.outputEncoding = THREE.sRGBEncoding;
  renderer.toneMapping = THREE.ACESFilmicToneMapping; renderer.toneMappingExposure = 1.05;
  renderer.setClearColor(0x000000, 0);
  viewEl.appendChild(renderer.domElement);
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(DATA.camera.fov || 36, 1, 0.005, 100);
  const controls = new THREE.OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true; controls.dampingFactor = 0.1; controls.autoRotateSpeed = 0.6;
  camera.position.fromArray(DATA.camera.pos); controls.target.fromArray(DATA.camera.target);

  const hemi = new THREE.HemisphereLight(0xffffff, 0x888888, 0.55); scene.add(hemi);
  const key = new THREE.DirectionalLight(0xffffff, 1.15);
  key.castShadow = true; key.shadow.mapSize.set(2048, 2048); key.shadow.bias = -0.0004; key.shadow.normalBias = 0.01;
  const sc = DATA.shadow || { center: DATA.camera.target, extent: 1.5 };
  Object.assign(key.shadow.camera, { left: -sc.extent, right: sc.extent, top: sc.extent, bottom: -sc.extent, near: 0.2, far: 30 });
  key.position.set(sc.center[0] + 2.2 * sc.extent, sc.center[1] + 3.4 * sc.extent, sc.center[2] + 1.6 * sc.extent);
  key.target.position.fromArray(sc.center); scene.add(key, key.target);
  const fill = new THREE.DirectionalLight(0xffffff, 0.35); fill.position.set(-3, 2.5, -1.5); scene.add(fill);
  const rim = new THREE.DirectionalLight(0xffffff, 0.25); rim.position.set(-1, 3, 3); scene.add(rim);

  let floorMat = null, grid = null;
  if (DATA.floor) {
    floorMat = new THREE.MeshStandardMaterial({ roughness: 1, metalness: 0 });
    const floor = new THREE.Mesh(new THREE.CircleGeometry(8, 72), floorMat);
    floor.rotation.x = -Math.PI / 2; floor.position.y = DATA.floor.y; floor.receiveShadow = true; scene.add(floor);
    grid = new THREE.GridHelper(8, 80, 0xffffff, 0xffffff);
    grid.position.y = DATA.floor.y + 0.0005; grid.material.transparent = true; grid.material.opacity = 0.35; grid.material.depthWrite = false; scene.add(grid);
  }
  function applyTheme() {
    if (floorMat) floorMat.color.set(tok("--floor"));
    if (grid) grid.material.color.set(tok("--grid"));
    hemi.color.set(tok("--hemi-sky")); hemi.groundColor.set(tok("--hemi-ground"));
  }
  applyTheme();
  const mq = window.matchMedia("(prefers-color-scheme: dark)");
  if (mq.addEventListener) mq.addEventListener("change", applyTheme); else if (mq.addListener) mq.addListener(applyTheme);

  // ---------- scene content ----------
  const F = DATA.frames, FPS = DATA.fps, DUR = F / FPS;
  const ropes = DATA.ropes.map((r) => {
    const tube = new RopeTube(r.n, r.radius, r.color, { material: r.material, samples: r.samples, radial: r.radial });
    tube.pos = decI16(r.pos); tube.scale = r.scale; tube.offset = r.offset; tube.rec = r;
    scene.add(tube.group); return tube;
  });
  const meshes = {}; for (const k in DATA.meshes) meshes[k] = meshGeometry(DATA.meshes[k]);
  const bodies = DATA.bodies.map((b) => {
    const m = new THREE.Mesh(meshes[b.mesh], new THREE.MeshStandardMaterial(Object.assign({ color: b.color, roughness: 0.6, metalness: 0.05 }, b.material || {})));
    m.castShadow = m.receiveShadow = true; m.frustumCulled = false;
    if (b.poses) m.poses = decF32(b.poses);
    if (b.pos) m.position.fromArray(b.pos);
    if (b.quat) m.quaternion.fromArray(b.quat);
    scene.add(m); return m;
  });
  const qa = new THREE.Quaternion(), qb = new THREE.Quaternion(), va = new THREE.Vector3(), vb = new THREE.Vector3();
  function setTime(t) {
    const ff = Math.min(F - 1, Math.max(0, t * FPS)), f = Math.floor(ff), a = ff - f, f1 = Math.min(F - 1, f + 1);
    for (const r of ropes) {
      const V = r.n, p = r.pos, s = r.scale, o = r.offset, b0 = f * V * 3, b1 = f1 * V * 3;
      for (let i = 0; i < V; i++) {
        const k0 = b0 + i * 3, k1 = b1 + i * 3;
        r.ctrl[i].set((p[k0] + (p[k1] - p[k0]) * a) * s + o[0], (p[k0 + 1] + (p[k1 + 1] - p[k0 + 1]) * a) * s + o[1], (p[k0 + 2] + (p[k1 + 2] - p[k0 + 2]) * a) * s + o[2]);
      }
      r.update();
    }
    for (const m of bodies) {
      if (!m.poses) continue;
      const P = m.poses, k0 = f * 7, k1 = f1 * 7;
      va.set(P[k0], P[k0 + 1], P[k0 + 2]); vb.set(P[k1], P[k1 + 1], P[k1 + 2]); m.position.lerpVectors(va, vb, a);
      qa.set(P[k0 + 3], P[k0 + 4], P[k0 + 5], P[k0 + 6]); qb.set(P[k1 + 3], P[k1 + 4], P[k1 + 5], P[k1 + 6]);
      m.quaternion.copy(qa).slerp(qb, a);
    }
    if (DATA.status) { const i = Math.min(DATA.status.length - 1, f); $("status").textContent = DATA.status[i] || ""; }
  }

  $("title").textContent = DATA.title; document.title = DATA.title;
  $("caption").textContent = DATA.caption || "";
  $("legend").innerHTML = (DATA.legend || []).map((l) => `<span class="chip"><i style="background:${esc(l.color)}"></i>${esc(l.text)}</span>`).join("");

  // ---------- transport ----------
  let t = 0, playing = true, speed = DATA.speed || 1, last = performance.now();
  const scrub = $("scrub"), timeEl = $("time"), playBtn = $("play"), orbitBtn = $("orbit"), speedSel = $("speed");
  speedSel.value = String(speed); if (speedSel.value !== String(speed)) { speedSel.value = "1"; speed = 1; }
  const hash = new URLSearchParams(location.hash.replace(/^#/, ""));
  if (hash.has("t")) t = Math.min(DUR, Math.max(0, parseFloat(hash.get("t")) || 0));
  if (hash.has("speed")) { speed = parseFloat(hash.get("speed")) || 1; speedSel.value = String(speed); }
  function setPlaying(p) { playing = p; playBtn.textContent = p ? "Pause" : "Play"; playBtn.classList.toggle("on", p); }
  if (hash.get("pause") === "1") setPlaying(false);
  if (hash.get("orbit") === "1") { controls.autoRotate = true; orbitBtn.classList.add("on"); }
  playBtn.addEventListener("click", () => setPlaying(!playing));
  speedSel.addEventListener("change", (e) => { speed = parseFloat(e.target.value); });
  orbitBtn.addEventListener("click", () => { controls.autoRotate = !controls.autoRotate; orbitBtn.classList.toggle("on", controls.autoRotate); });
  scrub.addEventListener("input", () => { t = scrub.value / 1000 * DUR; });
  window.addEventListener("keydown", (e) => {
    if (e.target.tagName === "SELECT" || e.target.tagName === "INPUT") return;
    if (e.code === "Space") { e.preventDefault(); setPlaying(!playing); }
    else if (e.code === "ArrowRight") { t = Math.min(DUR, t + 1 / FPS); }
    else if (e.code === "ArrowLeft") { t = Math.max(0, t - 1 / FPS); }
  });
  function resize() {
    const w = viewEl.clientWidth, h = viewEl.clientHeight;
    renderer.setSize(w, h, false); camera.aspect = w / h; camera.updateProjectionMatrix();
  }
  window.addEventListener("resize", resize); resize();
  function frame(now) {
    const dt = Math.min(0.1, (now - last) / 1000); last = now;
    if (playing) { t += dt * speed; if (t >= DUR) t -= DUR; }
    setTime(t);
    controls.update();
    renderer.render(scene, camera);
    scrub.value = Math.round(t / DUR * 1000);
    timeEl.textContent = `${t.toFixed(2)} / ${DUR.toFixed(2)} s`;
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
})();
