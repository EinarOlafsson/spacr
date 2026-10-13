Native waves and local facet spin, CPU proof — 2026-10-07
======================================================

Functional source commit `7e457e35ab`, parent `bb2d3ddbf5`. The accepted
ambient SHA256 is `a72bae23af4e5a6dce1753685aed5139d9dbd684aa4e95b0c60c60c12b091e8d`;
the earlier renderer is `f36b33c9714b26a49de0e9d362aadaeee5d110a725e022c622d7eb659cf35225`.
Both whole renderer files are compressed here. This individual checkpoint
precedes the separately owned shared Detail/density, production blur and
catalog changes; its whole-module hash is not asserted for later integration.

Waves now change curvature and amplitude while traversing, with analytic
surface normals and a safely widened overscan margin. Pointer gravity repels
waves within the existing finite physical radius; lens attraction is unchanged.
Paper tiles retain the original seeded native resting geometry and palette.
Only nearby facets rotate, faster near the pointer and at higher radius.
Zero radius or leaving the canvas restores the exact original resting image.
Angular history is bounded to one value per cached facet; resize/settings
invalidate it with the existing material cache. No resolution, sample or
density reduction was introduced. Native bilinear tile rotation avoids the
edge aliasing observed with the rejected nearest-neighbor diagnostic.

Acceptance and limits
--------------------

* 59 focused behavior tests passed in 18.31 seconds under capped 4 GiB,
  CUDA-hidden, Qt offscreen execution. These include four complete native
  3840×2160 cropped-vs-uncropped wave pairs, finite-radius repulsion, lens sign,
  local spin/proximity/radius/speed, same-clock idempotence, original resting
  pixels, palette invariance, alpha 255, independently owned frames, painter
  restoration on injected failure, and bounded material lifetime.
* The three changed methods have 143 executable statements with no missing
  statements or branch destinations in this focused run. This is method-level
  coverage, not a whole-module ratchet or whole-suite claim.
* 113 callable/nested documentation guards passed in 80.70 seconds separately.
  Raw logs preserve the observed warnings, including an unrelated concurrent
  write to the real QSettings files while the process itself was sandboxed.
* Actual dark/light native frames show zero changed resting facet pixels versus
  the earlier renderer and zero autonomous facet clock motion. With radius
  0.5, stationary-pointer animation changes 792,282 dark / 771,403 light
  pixels; 354 of 920 facets rotate. Original native tile storage remains
  40,941,680 bytes, plus 920 bounded angle values. The four PNGs are actual
  full-resolution renderer frames, not generated artwork or resized previews.
* Direct native timings exceed the 41.7 ms budget. The initial mixed cold/JIT
  run is deliberately not labelled warm worker or GUI cadence. The separate
  eight-sample alternating facet-only diagnostic, with compilation gated off,
  reports original static lift 48.10 ms versus new native bilinear rotation
  76.04 ms median. Nearest-neighbor rotation 64.27 ms is rejected for aliasing;
  direct native vector repaint 142.40 ms is rejected for cost. The diagnostic
  `no_production_change` flag means it made no additional production edit,
  not that the preceding functional source was unchanged.
* Hard native 24 FPS, human aesthetic approval, installed-app crash acceptance
  and GPU performance are **not accepted by this proof**. Smoothness remains
  open; no faster rejected representation was shipped.

The earlier facet shader was already clock-invariant, and the current Home
tile icon is a static pixmap. No source-backed cause for the reported historical
sequential icon pulse was established. These tests enforce the requested
absence of a global clocked lighting sweep; they do not claim a separate UI
style defect was diagnosed or repaired.

Reproduction
------------

`python reproduce.py` verifies every archived payload and frozen source hash.
`python reproduce.py --git HEAD --repo /path/to/repo` checks committed blobs.
`python reproduce.py --render --repo /path/to/source-7e457e35ab` replays both
bounded renderer probes in a fresh private scratch directory. Invoke the
render replay through `tools/run_capped.sh 4G` with
`CUDA_VISIBLE_DEVICES='' QT_QPA_PLATFORM=offscreen OMP_NUM_THREADS=1
MKL_NUM_THREADS=1`. It refuses another ambient hash. It does not run pytest.
Probe scripts and original result/log files remain unchanged in this archive.
Timings depend on hardware/load; new replay files do not overwrite accepted
receipts. The tests are copied here for evidence; executable source tests live
under `tests/qt/` in the bound commit.
