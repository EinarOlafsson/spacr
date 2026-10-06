N663 exact Random packed-word CPU review checkpoint, 2026-10-06

Source commit bc170e097f compares parental published45f3cb1ec0 renderer
a324d395c4029fdeb1e307595fcc4d2353cb4fd7f15589f51e456c67f19b417a
with isolated renderer
ac146d3c61bb73def9a08857eeb1fca8e7163de30b1b8892ba3720857244ec20.
Only the optional Random grain material and its compiler warmup change.
No style, geometry, palette RGB, samples, density, detail or native resolution
changes. Constant FF is removed from the private ordering word; rank occupies
its byte until the complete owned image has FF restored before returning.
Intensity ordering and RGB tie ordering remain exactly equivalent. Partial
kernel failure resets the whole working image before exact NumPy fallback.

95 affected cases passed in18.94s. All15 added executable source statements
and9 touching branch arcs are covered. The blocked real producer test proves
its previous owned frame remains published and unchanged until complete alpha
restoration. Startup/error/disabled-JIT and native owned-frame cases pass.
40 existing default/custom full-display frame pairs,24 Random compiled/fallback
native pairs,24 old/new native pairs, plus96 balanced native pairs are exact.
Primitive proof includes all256 ranks,98,304 clipped duplicate points per
dark/light x center/round case, exact packed table ordering and partial recovery.
Fresh closed startup gate imports neither NumBa nor SciPy until release;
all3 rendered frames then match their compiled equivalent. This proof exercises
the private gate; public app readiness remains the root acceptance owner's scope.

Fresh native actual widget runs are deliberately preserved as mixed results:
pair1 before22.60FPS,40.06ms median/53.18p95; after23.39FPS,28.65/45.91ms.
pair2 before23.20FPS,36.87/55.79ms; after22.20FPS,37.20/62.96ms.
Warm ABBA runs: before23.60/23.40FPS,27.18/32.90ms medians;
after23.79/23.60FPS,25.02/27.58ms medians. Publication interval p95 still
54.59/63.16ms for the candidate. Balanced same-process dark shade median
20.53→19.24ms and light23.38→21.56ms (48 pairs each). This is a modest
renderer improvement. Hard native24FPS remains OPEN. Cold cost is separate:
the first candidate fresh widget show takes169.24ms; subsequent cold heartbeat
max45.77ms. Original cold/steady JSON contains both repeats, not merged samples.

Palette tables total98,304bytes (previous196,608). The separate8MiB native rank
plane (8,294,400 bytes) and fallback uint64 full-frame accumulator
(66,355,200 bytes) are removed. Whole-process
RSS is not claimed to improve; JIT and allocator effects vary. Real native
fallback/compiled/compiled/fallback widgets release engine/frame/table/temp-view
weakrefs onhide/delete/GC; every worker stops. Idle RSS336,928–336,972KiB,
peak471,356KiB includes warmed JIT. Geometry caches and compiler lifetime retain
their existing contracts. All actual GUI cases use software Xvfb, CPU only.

Reproduce using the activated spaCR Python environment and an EMPTY scratch
folder. Replace REPO and OUT with caller-owned absolute paths. Modes startup,
primitive, native and stages are offscreen; abba, gc and worker require Xvfb.

CUDA_VISIBLE_DEVICES='' QT_QPA_PLATFORM=offscreen tools/run_capped.sh 4G python features/data/663_random_packed_word_cpu_2026-10-06/reproduce.py REPO OUT startup
CUDA_VISIBLE_DEVICES='' QT_QPA_PLATFORM=xcb QT_OPENGL=software LIBGL_ALWAYS_SOFTWARE=1 tools/run_capped.sh 4G xvfb-run -a -s '-screen 0 3840x2160x24' python features/data/663_random_packed_word_cpu_2026-10-06/reproduce.py REPO OUT abba

Fresh worker mode --variant before|after retains cold as well as warmed scopes.
Snapshots restore only into OUT. No repo app source is changed. JSON records
exact loaded source hashes; manifest.json verifies every archived file. The
coverage run's test cohort is listed in acceptance.json; do not add unrelated or
overlapping phase totals. Two initial probe setup errors were rejected before
measurement and are identified explicitly there. No aesthetic acceptance,
universal frame-budget guarantee or Preferences crash-resolution claim.
