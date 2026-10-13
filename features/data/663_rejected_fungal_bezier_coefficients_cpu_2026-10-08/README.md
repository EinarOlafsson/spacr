# Rejected scalar Bézier coefficient reuse

Production remains unchanged. Frozen beforefa95c6677bea19f66b4cfb5e4049ae8858269e5b9cc53e337653fe176301fbf4;
scratch aftercandidatesource recorded inrejection.json/receipt.json.gz.
Only two curve-calculation blocks hoist the same three scalar weights for
x/y, preserving each original float product and left-associative sum.
No endpoint simplification, helper, API/prose or persistent cache change.

At95.25/99.5/168.25, warm geometryABBAx5 medians before→after:
55.50→55.11ms;66.94→62.56;45.46→39.85. Inclusive _fungal_paths medians:
78.70→77.97;84.20→84.01;56.77→54.60. The current hot99.5 inclusivebenefit
is~0.2ms, not convincing for an integration. All original order, geometry,
path grouping/maturity/tips match; four native3840×2160 default/Random/
custom/light pairs have zero pixel differences. These are scratch functional
and fixedtiming bounds, not live cadence or hard24FPS acceptance.

One cProfile sample at168.25 attributes55ms geometrybody/79ms inclusive,
with Pythonclamps/lists/dicts dominant and math.hypot~3ms. Profiling adds
overhead; it does not identify allbodycost or establish a nextoptimization.
Peak307480KiB under4GiB/offscreen/CUDAhidden. No liveABBA, productionpatch,
newtests or giantframes. Existing cache64/8MiB and every quality/work control
remain intact. Hard24FPS stays OPEN at previously accepted~5.2–5.5FPS.

Original probe/log, frozen sources and receipts are compressed. Decompress
into scratch, set the probe REPO variable to an isolated nightly worktree,
then run CUDA_VISIBLE_DEVICES='' QT_QPA_PLATFORM=offscreen tools/run_capped.sh
4G python probe.py. Filesystem/Git manifest verification reruns no rendering.
