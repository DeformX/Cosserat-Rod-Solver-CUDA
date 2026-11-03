# Vendored YarnBall solver (headless subset)

Upstream: https://github.com/Lehong-Wang/YarnBall-Linux @ `fafa4fc8b323eb335679f03857bbf1d3fbefd9ce`
(our Linux CMake port of https://github.com/jerry060599/YarnBall, GPL-3.0 —
see `LICENSE.md` in this directory; it covers everything under it).

Copyright (C) 2025 Jerry Hsu (jerry.hsu.research@gmail.com) — YarnBall solver;
KittenEngine modules Copyright (C) Jerry Hsu 2021–2024. Upstream states the
GPLv3 grant in its README ("Unless otherwise stated in the file header...");
that notice is reproduced here because the README itself is not vendored.

This is the minimal headless subset of the solver that `csrc/` needs — 6
solver TUs + the LBVH TU + 8 engine headers (7 `includes/modules/*.h` +
`lbvh.cuh`) — verified by symbol-closure analysis (`nm`) against the full
static-lib build. The GUI, asset/IO, and rendering
code (KittenEngine src/, opt/, YarnBall io/, glad, imgui, GLFW, assimp,
freetype, jsoncpp, stb) is intentionally NOT vendored: nothing here references
it at link time, which is what drops all GL build/link dependencies.

## Layout

The `KittenEngine/` directory layout mirrors upstream exactly so that include
paths (`-I third_party/yarnball/KittenEngine`) resolve identically and files
diff cleanly against upstream.

## Local patches carried (vs pristine jerry060599/YarnBall)

Committed upstream in YarnBall-Linux (all platform-neutral):

1. `includes/modules/Common.h` — `gpuAssert` throws `std::runtime_error`
   instead of `exit()` so CUDA errors surface as catchable Python exceptions
   (no destructor calls it — safe).
2. `YarnBall/YarnBall.cu` — `~Sim` frees `meta.d_qs` (upstream device leak).
3. `YarnBall/YarnBall.h` — public `solverStream()` accessor.
4. `YarnBall/YarnBall.h` — `friend class SimAccess` so the ropesim batch
   driver can run the substep loop itself.

Added at vendoring time (this repo only):

5. `YARNBALL_HEADLESS` guards in `YarnBall/YarnBall.h` + `YarnBall/YarnBall.cu`:
   the umbrella `KittenEngine.h` include is replaced by the solver-facing
   module headers plus forward declarations of `Kitten::Mesh` /
   `Kitten::ComputeBuffer`, and the four GL-touching blocks (cyl-mesh +
   compute-buffer creation in `configure()`, their deletes in `~Sim`) are
   compiled out. Those blocks were already dead headless at runtime (guarded
   by `if (glGetString)`, NULL without a GL context); the guards make them
   dead at compile/link time too. Builds with `-DYARNBALL_HEADLESS` only;
   without the define the sources are upstream-identical.
6. Quadratic cross-flow air drag (`meta.dragQuad`/`meta.dragQuadT`, added
   2026-08-11): `YarnBall/YarnBall.h` (two MetaData fields),
   `YarnBall/YarnBall.cu` (zero defaults in `Sim()`),
   `YarnBall/sim/iteration.cu` (`endItr`: split v by the segment tangent
   from `d_qs`, apply v /= (1 + k|v|h) per component). All hunks marked
   "Local patch (ropesim)". `dragQuad == 0` (the default) leaves the legacy
   arithmetic DAG unchanged (bitwise-identical output expected), so upstream
   behavior is unchanged unless enabled.

7. Rod-plane contact (2026-10-04): `YarnBall/YarnBall.h` (MetaData: `numPlanes`,
   `planes[8]`, `contactRadius`, `contactKappa`, `contactMuS`, `contactMuK`,
   `contactVSlip`), `YarnBall/YarnBall.cu` (defaults in `Sim()`: no planes),
   `YarnBall/sim/cosserat.cu` (`planeContact()` + its call in `cosseratItr`'s sector-0
   block). Vertex-local penalty with stiffness m_i·(kappa/h)^2 + regularized Coulomb
   friction with a static->kinetic blend, implicit in the per-vertex Newton step; no
   contact damping. `numPlanes == 0` (the default) never calls it, so the legacy
   arithmetic is unchanged (bitwise, tested).
8. Frame-init fix (2026-10-04): `YarnBall/YarnBall.cu` `configure()` gave NaN frames
   when a segment points exactly along -x (`Rotor::fromTo` is degenerate for an
   antiparallel target) or when the frame flips 180 deg between segments (the
   twist projection normalized a zero vector). Now: a 180 deg turn about z, and an
   identity twist when the projection is empty. Upstream bug; output unchanged for
   every geometry that used to be finite.
9. Rope-rope contact guard (2026-10-04): `YarnBall/sim/cosserat.cu` collision loop
   skips a pair whose centrelines coincide exactly (the contact normal was 0/0 =
   NaN). Rope-rope candidates now come from `csrc/batch.cu` (`bruteDetectK`, a
   per-group detector); YarnBall's LBVH detector is unchanged but unused by ropesim.
10. Rod-mesh contact (2026-10-12): `YarnBall/YarnBall.h` (MetaData: `SdfGrid`, `SdfPose`,
   `numSdf`, `sdf[kMaxSdf]` (8 then, 32 since patch 11), `d_sdfPoses`, `sdfEnvVerts`), `YarnBall/YarnBall.cu` (defaults: no
   meshes), `YarnBall/sim/cosserat.cu` (`planeContact()` split into the shared
   `surfaceContact()` + the plane loop, new `sdfSample()` / `sdfContact()` and its call next to
   `planeContact` in `cosseratItr`'s sector-0 block). Grids and poses live in
   `csrc/sdf.h` (ropesim). `numSdf == 0` (default) never calls it.
11. Kinematic meshes + contact-force readout (2026-10-12): the contact device code moved to
   `YarnBall/sim/contact.cuh` (`surfaceContact`, `sdfSample`, `sdfContactAt`, `surfacesAt`,
   `surfaceReadout`); `YarnBall/YarnBall.h` (MetaData: `SdfPose` gains `vel`/`omega`,
   `SdfMotion`, `d_sdfMotion`, `sdfMoving`, `sdfEnvs`, `sdfForces`, `d_contactForce`,
   `d_sdfWrench`, `kMaxSdf = 32`, `kMaxSdfCand`, `d_sdfCand`, `d_sdfNumCand`, per-mesh `muS`/`muK`
   in `SdfGrid`; `startIterate(sub, nSub)`), `YarnBall/YarnBall.cu` (defaults),
   `YarnBall/sim/step.cpp` (`advance()` passes the substep index), `YarnBall/sim/iteration.cu`
   (`initItr(data, sub, nSub)` writes the interpolated pose + world AABB of each (env, mesh) while
   `sdfMoving`; `endItr` runs the force readout while `sdfForces` and rebuilds each vertex's
   broad-phase candidate list whenever `numSdf > 0`). The flags default to 0 and `numSdf` to 0:
   the legacy arithmetic is unchanged (bitwise, checked).

## Updating

Diff against the upstream paths under `KittenEngine/` in YarnBall-Linux,
re-copy changed files, and re-apply patches 5-6: patch 5 = ONE hunk in
`YarnBall.h` and FOUR hunks in `YarnBall.cu`, every one marked "Local patch
(ropesim vendoring)" — missing any of the `YarnBall.cu` hunks breaks the
headless compile (`Kitten::Mesh`/`ComputeBuffer` are forward-declared only);
patch 6 = ONE hunk each in `YarnBall.h`, `YarnBall.cu`, and
`sim/iteration.cu`, marked "Local patch (ropesim)". Patches 7-10 are marked
"Local patch (ropesim)" too: 7 = one hunk each in `YarnBall.h`/`YarnBall.cu` plus
the `planeContact()` function and its call in `sim/cosserat.cu`; 8 = two hunks in
`configure()` (`YarnBall.cu`); 9 = one hunk in `sim/cosserat.cu`; 10 = one hunk each
in `YarnBall.h`/`YarnBall.cu` plus `surfaceContact()`/`sdfSample()`/`sdfContact()`
and one call in `sim/cosserat.cu`; 11 = one hunk each in `YarnBall.h`/`YarnBall.cu`/`sim/step.cpp`,
two in `sim/iteration.cu`, the include + one call in `sim/cosserat.cu`, and the new
`sim/contact.cuh`.
