DEFERRED READ-ONLY INTEGRATION PREPARATION, 2026-10-08

Production root is exactly fa95c6677bea19f66b4cfb5e4049ae8858269e5b9cc53e337653fe176301fbf4.
The prepared patch yields the previously measured final scratch candidate
90f3e96a4f4259bead6624c786622970252dbc4aa34ead1799934206c90e82e7.
Only _FungalGrowthEngine._lineage and .geometry bodies change. No helper,
module, signature, docstring, tr literal, UI/catalog or scientific source change.
No patch was applied, no tests/benchmarks run, and no commit made in this review.
Root must wait for first GitHub-green acceptance before deciding integration.

PATCH AND OWNERSHIP
mature-cost-fa95-to-90f3.patch is the exact unified patch against current root.
Existing eight-entry FIFO values become (unchanged immutable lineage, costs).
Only full progress 1.0 uses precomputed original arithmetic, and only when
initial geometry Size/Density plus dimensions/block identify that exact tuple.
Cloned/reordered/foreign lineage tuples and control-mutating overrides fall
back to the original calculation. An authentic tuple returned by a transparent
wrapper can safely reuse. This is NOT a blanket exclusion of subclasses.
_resize/_configure already clear the existing cache; no new state there.
Frame-local costs is moved, not duplicated. Mature raster cache 64/8 MiB unchanged.
Explicit new geometry metadata is bounded by existing eight entries, with
2,671,552 bytes measured at eight functional entries; the formal float-tuple
payload maximum is 3,539,264 bytes plus 448 pair headers. This is not zero RAM.

NEW/CHANGED COVERAGE OBLIGATIONS (see readiness.json line inventory)
1. _lineage existing cache-hit return now returns cached[0]: cold and hit.
2. Positive width/height precompute guard: nonempty, empty and unsupported
   zero/negative private _lineage inputs, preserving original private behavior.
3. Exact identity cache match: authentic tuple; cloned/reordered external
   tuple; missing key after an override changes Size/Density while returning
   base lineage. Initial geometry Size capture must remain (initial draft bug).
4. Mature cached append versus partial-progress or uncached append.
5. costs[index] is None: cached full maturity skips math; partial/foreign
   still uses it, including the original fractional-tip footprint addition.
6. Existing accepted sum/order/budget, selected expensive parent and last
   duplicate-endpoint parent resolution must stay unchanged at IEEE equality.
7. Eight-entry FIFO, Size/resize/configure clear and dimension/Density key
   separation; no second unbounded map or raster-cache memory relaxation.
All 34 changed/reindented AST statement starts listed in readiness.json need
source-current direct execution (not inheritance from differently indented
old statements). Short-circuit branches need explicit matching/refusal cases.
Then reconcile current old eleven-module/pen allowance with strictly mapped
byte-identical old lines plus direct new coverage; never raise ceilings.

AFTER FIRST GITHUB GREEN: FOCUSED ACCEPTANCE PLAN, NOT RUN HERE
Add meaningful cache/override/boundary tests to existing
 tests/qt/test_fungal_growth_engine.py (no new app module/helper).
Preserve the archived reversed-identity mutant and Size-mutating override
counterexamples as failure controls; preserve 17 adjacent IEEE-budget cases.

CUDA_VISIBLE_DEVICES='' QT_QPA_PLATFORM=offscreen PYTHONDONTWRITEBYTECODE=1 \
 tools/run_capped.sh 4G /home/olafsson/anaconda3/envs/spacr/bin/python -m pytest \
 -q -p no:randomly tests/qt/test_fungal_growth_engine.py \
 tests/qt/test_fungal_mature_raster_cache.py \
 tests/qt/test_fungal_scalar_path_overloads.py \
 tests/qt/test_data_art_owned_branch_frames.py \
 --cov=spacr.qt.widgets.ambient --cov-branch \
 --cov-report=json:/mnt/wd4tb/scratch/fungal-mature-cost-final-coverage.json

CUDA_VISIBLE_DEVICES='' QT_QPA_PLATFORM=offscreen \
 tools/run_capped.sh 4G /home/olafsson/anaconda3/envs/spacr/bin/python -m pytest \
 -q -p no:randomly \
 tests/test_every_callable_in_the_package_is_documented.py::test_no_callable_in_the_package_is_undocumented \
 tests/test_nested_functions_are_documented.py::test_no_module_exceeds_its_budget

/home/olafsson/.local/bin/ruff check spacr/qt/widgets/ambient.py --select E9,F63,F7,F82
/home/olafsson/.local/bin/ruff check tests/qt/test_fungal_growth_engine.py

Static signature/docstring/tr literal parity against the frozen baseline must
remain, as already checked in the archived candidate verifier; no generator
run justified by these unchanged source contracts. The current strict Xenon
maintained-target list does not include ambient.py; do not add an exemption
or change that list/ratchet. Official unchanged quality gate still runs in CI.

SOURCE-CURRENT PROOF REUSE
Existing compact archive ec3d4ecfd contains final 90f3 source, 24 geometry
states, 17 equality cases, empty/cycle/duplicate and override counterexamples,
three inclusive path timings, four full native default/Random/custom/light
frame pairs and four actual native worker ABBA logs. If production bytes are
exactly 90f3, bind these rather than repeating huge frame galleries or live
benchmark quartets. If any actual runtime method differs, prove that seam
before using the prior receipt; do not claim an old whole-source hash is new.
No unique-count inflation from initial/final repeated phases. Existing live
same-run gain only 3.9/6.6%; approximately 6.15 -> 6.40/6.55 FPS, not 24.
Cold geometry overhead ~3.81 ms at99.5, cold show mixed. Preserve honest limits.
No new live benchmark or all-theme/whole-Qt/GPU acceptance requested here.
