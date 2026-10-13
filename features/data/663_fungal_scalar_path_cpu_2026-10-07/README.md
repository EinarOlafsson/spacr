Exact scalar mycelium path construction, 2026-10-07

Source/test commit 500aadad8b changes only two Qt overload calls, replacing
temporary QPointF wrappers with scalar coordinates. Before b2a19147, after
ade8062f; source_boundary.json proves every other source byte and callable
prose is unchanged. No pixel, sample, detail, density, palette, alpha, drawing
order, cache bound, output ownership, startup or worker scheduling change.

The earlier scratch-only exact proof is retained rather than repeated:
160 exact QPainterPath/maturity/tip comparisons and 48 byte-identical complete
native frames. Its before and scalar sources are precisely these snapshots.
prior_exact_probe.py expects the snapshots named before.py/scalar.py; to
reproduce independently in private scratch, copy after.py to scalar.py there.
New final-source differential tests compare Qt's previous QPointF overload
against the scalar overload for negative, fractional and large coordinates,
and six complete native 3840x2160 frames across default/Random/custom and
light/dark. Old returned frames stay independent. Nine pass in3.10 seconds.
The final coverage/doc phase has20 passing cases in19.02 seconds (these nine,
eight ambient refusal cases, three whole-package docstring guards); these
phases overlap and are not added into a unique total. The existing fungal and
cache cases also passed on this source (27 cases, recorded in the agent tool
transcript; not claimed as an archived standalone run). Both changed lines
execute under branch coverage. New test Ruff and diff whitespace checks pass;
four inherited ambient I001 import-format findings remain identical.

Fresh isolated actual software-Xvfb 4K widgets/producers request24FPS with
detail2/density3/size2.5/blur0. Default ABBA means18.857->19.461FPS (+3.20%);
Random BAAB21.877->23.048 (+5.35%); one custom pair17.756->18.596 (+4.73%).
Default shade medians51.50/51.56->49.26/50.98ms, Random45.48/45.13->41.79/42.58ms.
Load varies2.54-3.04 in the balanced cases and is retained in each receipt.
First baseline's raw log exists only in the tool transcript; its complete
JSON receipt is archived. All runs stop their producer after hide. Cold costs,
heartbeat, GUI paints, repeated objects, exact source and physical buffers are
retained. Native24FPS, aesthetic approval, whole-suite/hosted, crash repair,
GPU, documentation/API generation and universal smoothness remain OPEN.

Inclusive actual-worker diagnostic wrappers (not additive and slower than
normal execution) attribute after-change medians29.96ms to Qt.drawPath,
8.36ms to sparse reuse,3.33ms to path building (including2.14ms geometry).
GUI paints are about5ms. Publication intervals closely track shade duration;
continuously changing published clocks show no old clock-starvation defect.
Qt rendering remains the largest cost. These measured improvements alone do
not supply the remaining native24FPS budget.

worker_probe.py is the uninstrumented accepted runner. profile_worker.py adds
inclusive diagnostic wrappers; use --profile with a distinct --run-id.
During preparation, four noninstrumented jobs briefly used that diagnostic
wrapper with profiling disabled. Those receipts are excluded and were rerun
with the exact uninstrumented runner; originals remain in scratch explicitly
named nonacceptance-wrapper-transition-* rather than accepted measurements.

Reproduction uses hidden CUDA and capped4G:
CUDA_VISIBLE_DEVICES='' QT_QPA_PLATFORM=xcb QT_OPENGL=software LIBGL_ALWAYS_SOFTWARE=1 xvfb-run -a -s '-screen 0 3840x2160x24 -nolisten tcp' tools/run_capped.sh 4G python features/data/663_fungal_scalar_path_cpu_2026-10-07/worker_probe.py data_art_fungal_growth after --palette spacr --run-id fresh
Set SPACR_PROOF_REPO to your isolated source tree. Runners write receipts beside
themselves, so copy proof files to private scratch before running. Default
balanced order before/after/after/before, Random after/before/before/after.
