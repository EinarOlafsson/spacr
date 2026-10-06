2026-10-06 strict installed Home compiler-boundary repair

After the first magnifier import repair, hosted209 compatibility still failed
on Linuxx86/ARM and macARM with SciPy loaded before Home readiness. Exact three
job logs are retained compressed; their assertions are not relaxed.

Fresh public-run audit proves the second path: spacr-grain-compile imports
NumBa, whose dependency check imports SciPy at1.952s BEFORE genuine Home
readiness2.257s. App now closes the optional compiler gate before preferences
and Home construction. Existing real post-paint usable-control readiness
releases the gate on the next event-loop turn; both grain and satin kernels
retain exact fallbacks before that point. Profiling-off launches request only
the same readiness callback, without timing reports or import profiling.

After: Home readiness1.952s has no SciPy; first SciPy import1.979s occurs
AFTER readiness. Normal profiling-off public launch has no SciPy at its
post-paint callback2.002s, then first SciPy at2.054s, with zero timing reports.
These local timings describe these processes, not a cross-host speed claim.

Source1b8ac874c8 app/timing/ambient bytes match both freshly built release
artifacts exactly, with SHA256 in artifact-source-identity.json. Each was
installed independently and tested with Python-I against site-packages.
Strict unchanged installed Home probes pass: wheel1.769s, sdist1.887s,
16 actually painted usable controls, heavy_modules_at_ready=[] for both.
The probes quit at readiness, so no later compile is needed to pass Home.

64 integrated renderer/readiness/density/rain/timing tests pass. A separate
83-case cohort covers normal-mode callback readiness, disabled-control and
pre-loop refusal, observer-before-compiler ordering, retirement and unchanged
disabled-timing behavior. All87 nested-function/docstring cases pass.
Renderer compiler/fallback byte parity, native controls and Thore performance
are separately archived under663_theme_controls_thore_startup_cpu_2026-10-06.
Root also inspected the source-bound lightning and native rain crop.

Source/current generated artifacts remain workstation-owned; hosted green
and uninterrupted serial Qt remain OPEN. No import guard, coverage ceiling,
performance budget, serial process order or user-crash acceptance is waived.
