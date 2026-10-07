# Reaped primary-mask reader shutdown, 2026-10-07

The helper `drain_thread` explicitly treats a deleted QThread wrapper as already drained, but `PrimaryMaskSelector.shutdown` previously called `requestInterruption` on that wrapper first. A real `shiboken6.delete(QThread)` regression fails on the old source with `RuntimeError: Internal C++ object already deleted`. The repair catches that request failure and uses the helper’s return value to detach a still-running reader; a stopped reader retains its original parent.

The direct regression plus ten existing selector tests pass 11/11 under CUDA-hidden 4 GiB branch coverage. All changed lines and originating branches are executed; the module’s focused total is 143/146 statements and 29/30 branches. This is a demonstrated close-path defect, not an attribution of the earlier native puncta crash, whose QThreadWrapper instance was never identified. Source and raw result hashes are in `receipt.json`.
