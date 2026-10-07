Ambient numerical coverage child, 2026-10-07

The hosted numerical ratchet at 392ca4d6a reports 11 uncovered statements and
7 branch destinations for ambient source 08815be8. Current source is b2a19147.
Eight focused behavior/error/lifecycle cases pass in 4.40 seconds under 4 GiB,
hidden CUDA and offscreen Qt. Ruff passes. Renderer source and all ratchet
ceilings are unchanged. The cases exercise equal control values while retaining
actual ready pixels, changed-control invalidation, the retained legacy Resonance
engine's caches, unavailable gravity preferences, unsupported engine mouse
controls and cleanup after an expired application reference.

Run `python features/data/663_ambient_ratchet_cpu_2026-10-07/reproduce.py` from a
repository retaining commit 392ca4d6a and current b2a ambient. The script refuses
other source hashes. Coverage is mapped only through identical source lines;
it deliberately discards all inherited fungal-class coverage. That class instead
uses the previously archived exact-b2a 41-case coverage, whose 154 touched
statements and branch destinations are exercised. The compact input files retain
all executable/missing line and branch data but omit redundant per-function rows.
The conservative union covers all 3,451 current statements and 726 branch
possibilities, leaving zero gaps versus the unchanged baseline allowance 1/0.
This does not constitute a fresh whole-suite run or a successful hosted gate.
It makes no documentation/API, native 24 FPS, appearance or GPU acceptance claim.

Focused reproduction:
CUDA_VISIBLE_DEVICES='' QT_QPA_PLATFORM=offscreen tools/run_capped.sh 4G /home/olafsson/anaconda3/envs/spacr/bin/python -m coverage run --branch --source=spacr.qt.widgets.ambient -m pytest -q -p no:randomly tests/qt/test_ambient_cache_and_cleanup_refusals.py

Historical cache coverage is from the source-bound archive
663_fungal_mature_raster_cpu_2026-10-06 (41 passed, 22.40 seconds under coverage).
Hosted inputs came from ci-392-ratchet-20261007/coverage.json (coverage 7.16.2);
current targeted inputs use coverage 7.15.4. No native parity run was repeated.
