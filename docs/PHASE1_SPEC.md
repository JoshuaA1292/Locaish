# Phase 1 — Scan: module contracts

Phase 1 turns a LiDAR export from a consumer scanner into a **`Twin`**: metric,
gravity-aligned, yaw-normalised, georeferenced, with a QA report that proves
the geometry is 1:1 rather than merely asserting it.

Everything here is written against `locaish/types.py`. **Read that file first.**
It is frozen — do not edit it, and do not edit `locaish/fixtures.py`. If you
believe a contract there is wrong, say so in your report instead of changing it.

## Canonical frame (non-negotiable)

Z up, right-handed, metres. Floor at `z = 0`. After yaw normalisation the
dominant wall direction is parallel to +X. Origin at the centroid of the floor
footprint. `Plane.normal` points **into the room** (floor +Z, ceiling −Z).

## The pipeline

```
video file                                    export file
  └─ video.frames.extract_frames                  │
      └─ video.backend.reconstruct                │
          └─ video.metric.solve_scale             │
              └─ ScanImport (metric, poses, up)   │
                                    └─────────────┴─ formats.read_scan
                                          ↓
                                    ScanImport (points, mesh?, poses?, up?, software)
      └─ scale.infer_unit_scale → metres
          └─ geom.normals       → per-point normals
              └─ geom.planes    → list[Plane]
                  └─ geom.align → canonical 4×4 (gravity, yaw, origin)
                      ├─ geom.grid    → OccupancyGrid
                      ├─ geom.mesher  → Mesh (only if the import had none)
                      ├─ geom.hull    → CaptureBounds
                      ├─ scan.structure → Structure (floor/ceiling/walls/openings)
                      └─ scan.qa      → QAReport
                          └─ Twin.save(...)
```

## Rules for every module

1. **numpy + scipy + trimesh + pillow + scikit-image only.** No open3d, no
   torch, no network calls. `trimesh` is for file IO only, never for geometry
   analysis — we own the algorithms so we can explain every number.

   **`locaish/video/` is the one exception, and the boundary is exact.** It may
   shell out to COLMAP and use OpenCV, because reconstructing geometry from
   pixels is not something we are going to out-engineer in numpy. (It used to be
   allowed pretrained networks; the Agentic Cinema rules forbid non-Google AI
   models, so the reconstruction is now classical end to end — SIFT, bundle
   adjustment, photometric stereo.) What it may not do is leak: it terminates at
   a `ScanImport`, identical in kind to what a PLY reader returns, and every
   module downstream explains its own numbers. The reconstruction's output
   enters this pipeline as *measurements with error bars* — a cloud, poses, a
   scale factor and its spread — never as a conclusion. Nothing in
   `locaish/video/` may write to a QA report, and no check may be softened
   because the input was video.
2. **Vectorise.** Real inputs are 1–20M points. A per-point Python loop is a
   bug. Use `scipy.spatial.cKDTree` and chunked array ops (`types.chunked`).
3. **Never silently guess.** If a routine is unsure, it returns a confidence or
   appends to a `warnings` list. Downstream must be able to tell a measurement
   from an assumption.
4. **Deterministic.** Seed every RNG. Same input ⇒ same twin, byte for byte.
5. **Docstrings explain the "why".** The non-obvious choice, the failure mode
   you are defending against. Not a restatement of the signature.
6. Type hints everywhere, `from __future__ import annotations` at the top.

## Validating your work

The venv is `.venv/` at the repo root — use `.venv/bin/python`.

`locaish/fixtures.py` builds synthetic rooms with **exact** ground truth,
deliberately mangled the way real exports are (tilted, rotated, in centimetres
or inches, occluded, drifting). `fixtures.catalogue()` lists them and says what
each one is designed to break.

```python
from locaish import fixtures
fx = fixtures.build("tilted")      # fx.points, fx.truth, fx.camera_positions
fx.truth.width, fx.truth.depth, fx.truth.height   # metres, exact
fx.truth.applied_transform         # what the pipeline must undo
```

Write a scratch script, run it, and paste real numbers into your report. A
module that has never been executed is not done.

## Accuracy targets (Phase 1 acceptance)

Measured on `fixtures.build("clean" | "tilted" | "centimetres" | "inches")`:

| quantity | target |
|---|---|
| room width / depth / height | within **±15 mm** of truth |
| gravity alignment (floor normal vs +Z) | within **0.25°** |
| yaw alignment (wall vs +X) | within **0.5°** |
| unit scale | exact factor recovered, or flagged `fail` |
| window/door count | exact on `clean` and `tilted` |
| opening width / height | within **±80 mm** |

`drifty` and `sparse` are expected to *degrade*, and the requirement there is
that `QAReport.verdict` becomes `warn` or `fail` — a wrong number that announces
itself is acceptable, a wrong number that doesn't is not.
