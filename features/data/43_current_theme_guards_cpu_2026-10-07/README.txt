Current CPU theme CI guard repair, source-bound scoped evidence.

The only app change skips an Aurora curtain whose geometry has fewer than two
columns before texture/index access. Actual empty/single dark/light owned-frame
tests and the original injected malformed-geometry node pass. Both inserted
statements and both guard arcs are directly covered. The test painting helper
now ends its QPainter in finally even when the original exception propagates.

Tests use explicit Density1 only when exercising old geometric baseline, while
default-frame restoration actually restores current0.1. The uncropped atlas
reference now uses the existing bounded high-density pool mapping, preserving
full4K culling/perimeter parity. Removed chromatin runs as a private engine.
Advection uses actual12-by-particle trail matrices and verifies genuine heads.
Facet geometry/rotation remain bounded to current-size/current-density records.

The replacement folded Aurora has seeded irregular native rays, affine fanning
and emission taper at both ends. Retired constant36px comb, stationary-screen
ray columns and asymmetric2.5 slope assertions are replaced by native lattice/
sample population, deterministic retained ray material with changing rendered
folds, and actual emitted texture boundary/interior checks. No threshold was
lowered; existing pixel/cap/cache/ownership checks remain.

Historical protectedb688 QEventLoop139 and later puncta native failures remain
open; this Aurora malformed-sample exception/test-painter GC crash is separate.
57b three full Qt logs show assertion failures and no native-crash markers.
One original-order current fourfile cohort passes50 and bounds a negative
reproduction only. No hosted green/full-suite/whole-memory/GPU/aesthetic or
24FPS acceptance. Phases overlap and must not be summed.

Reproduce from a checkout containing source and tests listed in receipt:
CUDA_VISIBLE_DEVICES='' SPACR_DEVICE=cpu QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg
tools/run_capped.sh 4G python -m pytest -q -p no:randomly
tests/qt/test_data_art_renderer.py tests/qt/test_data_art_gravity_and_geometry.py
tests/qt/test_data_art_waves_and_vortices.py tests/qt/test_data_art_advection_gravity.py
tests/qt/test_data_art_advection_evolution.py tests/qt/test_cov_7_ambient.py
-k 'not widget_reports_back_every_control'

The3 final foldedAurora nodes are named in the frozen test_ambient_motion copy.
The full-module coverage JSON is a focused coverage snapshot, not acceptance
against the complete hosted coverage baseline.
