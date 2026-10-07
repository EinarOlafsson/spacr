2026-10-07 — native spinn GUI/producer gap diagnosis, no product change

Individual source254f3dd39f ambient SHA256:
a3d1fed1d18b323fec6859e6ea72bad4ce273c8b60e8e87823ed751d7965eaff.
This is the same source as the previously accepted owned/resting implementation
in data/663_spinn_owned_rest_cpu_2026-10-07, not a new renderer candidate.
Frozen source, actual scripts, JSON and compressed logs are archived here.
No old accepted archive or manifest is replaced.

Question: why does direct active native shading cost about48ms, while an actual
widget/producer delivers only10–12FPS in the accepted short balanced probe?
_FrameProducer._run subtracts shade duration from its remaining interval;
there is no evidence of an extra full-frame sleep/pacing bug.

One bounded instrumented native probe uses3840x2160, Density3/Detail2/Size1,
radius.5, requested24 cap,2652facets/1082spinning. Fixed normalized pointer is
offered on each actual GUI timer tick; viewport pixels/population are asserted.
Original BEFORE/AFTER source parity is not rerun. Normal_A/no-GUI-blit/normal_B
phases preserve full native worker shading/time/input. ONLY the middle diagnostic
suppresses GUI image blitting; this is deliberately unsuitable for displayed
frame acceptance or production. No source, global pool policy or environment
thread count is modified to obtain better FPS. Normal painting is restored.

Normal shade wall medians86.84/77.05ms versus producer threadCPU43.17/38.81ms.
Normal GUI paint wall≈9.37ms/threadCPU≈4ms; image blit wall≈3.8–4.0ms versus
GUI threadCPU≈0.17ms. No-blit diagnostic has shade wall69.00ms/threadCPU40.02ms,
producer14.60FPS versus normal13.37/12.49FPS. Some GUI composition/contention
is implicated, but a large wall/CPU gap remains. It is not valid to attribute
all wall-minus-threadCPU to GIL wait: helper threads, scheduler and synchronization
are not counted by time.thread_time(). Short phase ordering/angles/load and
instrumentation limit causal precision. Calls crossing a phase boundary are
labelled by completion phase; per-phase call counts are not simultaneous CPU
accounting. Distinct published-slot paints are recorded separately from timer
paints, and do not imply a faster physical desktop display.

A separate900ms own-process accounting probe measures processCPU1.838s over
elapsed0.892s (about206% of one CPU). Eight threads named Thread (pooled) each
accrue roughly0.16–0.17CPU seconds, while the ambient producer accrues≈0.57s
and GUI≈0.14s. Their identities are observed; no native stacks are captured,
so exclusive attribution of every pooled operation is not established. There
are no NumPy/compiler tasks launched by this spinn-only harness. /proc thread
reads are sequential and ticks are10ms: do not sum these numbers as an exact
simultaneous accounting identity or subtract them from phase wall time.
The accounting output includes only this owned process's thread data and cgroup
CPU statistics, with no external process attachment or global settings changes.

Each probe retires its actual ambient producer and reaches0Qt widgets after
natural deferred deletion, without forced GC. Qt's pooled threads may remain
idle for process lifetime; that is distinct from a live ambient producer.
Instrumented peakRSS312640KiB for the first probe; all Python runs capped4GiB,
CUDA hidden, Qt offscreen. No GPU, memory reduction, Save crash or aesthetic claim.

Conclusion: no verified quality-preserving source or pool change is justified
by these diagnostics. Hard native24FPS remains OPEN. Do not repeat this same
no-blit/global-pool avenue without a new causal lead. Existing native pixels,
density, detail, spin physics and actual display responsibilities stay unchanged.

Reproduce only when a new diagnostic question warrants it, in an isolated
worktree at254f3dd39f. Copy scripts to private /mnt/wd4tb/scratch first:
  CUDA_VISIBLE_DEVICES='' QT_QPA_PLATFORM=offscreen OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 tools/run_capped.sh 4G /home/olafsson/anaconda3/envs/spacr/bin/python <scratch-script> <isolated-worktree>
Scripts write JSON/log data beside their scratch copy, not into this archive.
Verify filesystem/Git blobs using verify_manifest.py / verify_manifest.py --git HEAD.
