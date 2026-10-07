Exact mycelium cache at higher element density, 2026-10-07

Source/test commit d7fa043732 follows the scalar-overload optimization. Before
ambient ade8062f, candidate f36b33c9. Exactly one conservative density eligibility
condition is removed. Every other renderer byte remains identical. Drawing
order, native resolution, geometry, sample count, color/alpha, original work
budget, blur, output ownership, fallback, saturation headroom and all remaining
painter/device/context safeguards are unchanged. Cache remains64entries/8MiB
owned arrays, observed paths64, one warm group per frame. No new module, worker,
dependency, public API, UI string or source comment is introduced.

The distinction matters: existing WORK_BUDGET4 makes detail2/density3 effective
density1, which already permitted caching. This improvement instead targets
detail1/density2 or3, whose effective density is2/3; the growth engine renders
all of them at full physical3840x2160 under an unchanged native pixel budget.
No control, density or raster limit was reduced to obtain these results.

The scratch gate compares12 complete native frames at95/98seconds across
default/Random density2/3 and white/16-bit high-channel custom colors density3.
Every byte matches with actual7-28successful cache hits. Twenty balanced
shader pairs per default/Random density case improve medians:
default density2 70.05->56.74ms, density3 78.24->64.44;
Random density2 58.80->48.75, density3 64.84->54.76.
Native candidate retained about0.67-0.69MiB at the fixed-clock samples.

Eight fresh actual software-Xvfb4K widget/producer probes alternate pair order:
default density2 11.328->12.799FPS (+13.0%), density3 11.692->13.566 (+16.0%);
Random density2 14.014->14.569 (+4.0%), density3 13.642->14.162 (+3.8%).
Each pair has one fresh run per source and recorded shared load3.39-4.02, so
these are measured observations, not guaranteed or load-normalized percentages.
Requested24FPS, detail1/density2or3/size2.5/blur0/nativepixels remain unchanged.
Live cache retains about0.94-1.10MiB in19-29entries; process RSS and cold costs
are retained separately and are not equated with array-cache bytes. All eight
workers stop after hide. These gains still leave native24FPS OPEN.

Final51 focused cases pass in48.37seconds under4GiB, hidden CUDA, offscreen Qt
and branch coverage. Eight added cases compare three full-native frames each
against original Qt rendering for default/Random/actual saved custom16-bit/
white at density2/3. They require real cache hits, alphaFF, retained old-frame
independence, owned arrays and existing memory/cardinality bounds. The previous
test demanding blanket density exclusion was replaced by these meaningful
render/ownership tests. All11remaining unsupported context variants still use
the original paths. Saturation refusal, partial-failure full repaint/painter
restore, weak frame ownership, size invalidation and cache limits remain tested.
Every _FungalGrowthEngine method has no missing statements or reported branch
destinations in this focused cohort. Test Ruff and diff whitespace checks pass.
No whole-suite, hosted-success, GPU, aesthetic, installed-crash repair,
documentation/API-generation or general native24FPS acceptance is claimed.

For reproducibility copy the folder to private scratch; copy candidate.py to
after.py for the worker runner. Set SPACR_PROOF_REPO to an isolated checkout
containing the stated sources. guard_probe.py takes the checkout path as its
single argument; it writes receipt.json and should run under hidden CUDA,
offscreen Qt and tools/run_capped.sh4G. The worker runner uses an actual4K
software Xvfb screen, QT_QPA_PLATFORM=xcb, QT_OPENGL=software,
LIBGL_ALWAYS_SOFTWARE=1, CUDA_VISIBLE_DEVICES='' and the same4G cap. Invoke:
worker_probe.py data_art_fungal_growth before --palette spacr --density 2
and its after counterpart.
Use density3 and Random for the other pairs; custom palette selection is
covered by the native tests and high-channel direct parity, not an extra
live custom worker run. Cold/full-stage work is retained separately from the
five-second warmed interval. No failed parity prototype is credited here.
