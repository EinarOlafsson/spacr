# Bounded small-reader QThread lifetime probe, 2026-10-07

The native backtrace in this folder proves that a `QThreadWrapper` was being
destroyed when the GUI thread faulted. It does not identify the Python QThread
instance. `_SourceWorker._finished()` is a candidate because it drops its
Python owner and posts `deleteLater()` from the finished signal; this probe
tests its normal small-source lifecycle without changing application code.

The accompanying script used the actual `PrimaryMaskSelector` and twelve real
32×32 TIFF reads over six selector lifetimes. Each QThread native pointer was
recorded before starting its field read, and its `destroyed` signal and Python
weakref were observed after publication. `gc_policy.install(app)` installed
the normal 1-second GUI timer, disabling automatic worker-thread GC. The
script neither calls `gc.collect()` nor flushes posted Qt events manually.

Environment: current selector source blob `41b9620b1643d296e78a0c95c8cba8e48851d759`,
GC policy source blob `61d5fcd88da49d4c446e7316771feb86d6721147`, Python 3.12.13,
PySide6 6.11.2, CUDA hidden, offscreen Qt, 4 GiB cgroup. The run was under
`gdb -batch -x probe-crash-only.gdb` with only a crash-time register dump; no
breakpoints were used. The command was run through `tools/run_capped.sh 4G`
with `PYTHONPATH` pointed at the source checkout and
`CUDA_VISIBLE_DEVICES='' QT_QPA_PLATFORM=offscreen`. GDB exited status 1 only
because the inferior ended normally and crash-only register commands had no
live inferior to inspect.

Result: all twelve snapshots had the expected label, all twelve worker
`destroyed` signals arrived, all twelve weakrefs cleared, and GDB reported
`Inferior 1 ... exited normally`. The GUI GC timer ran naturally: CPython
generation counts changed from `[36472, 0, 10]` after the last read to
`[0, 1, 10]` after the 2.2-second idle event loop. No native crash occurred,
so the earlier faulting `QThreadWrapper` instance remains unidentified. This
negative bounded probe does not clear the contextual three-file or hosted Qt
SIGSEGV, nor the installed Preferences Save report.
