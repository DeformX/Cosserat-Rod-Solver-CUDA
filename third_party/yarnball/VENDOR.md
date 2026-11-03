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

## Updating

Diff against the upstream paths under `KittenEngine/` in YarnBall-Linux,
re-copy changed files, and re-apply patch 5: ONE hunk in `YarnBall.h` and
FOUR hunks in `YarnBall.cu`, every one marked "Local patch (ropesim
vendoring)". Missing any of the `YarnBall.cu` hunks breaks the headless
compile (`Kitten::Mesh`/`ComputeBuffer` are forward-declared only).
