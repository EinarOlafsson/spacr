# Cancelled 86d ordinary and Qt serial runs — 2026-10-08

GitHub reports ordinary run 37735178663 and serial run 37735178905 at `86d98624993ebcc356c70277028a0d3256d3ddcd` as **cancelled** after the corrected required run 37737743925 was dispatched. The newer serial run is 37738253849. Neither cancelled run is a green acceptance verdict.

The serial cleanup artifact 11533325689 contains a **partial** original-order prefix: 178 files completed, the 179th began, and pytest reached 6%. The journal recorded one actual assertion failure in `test_641_module_first_open_timing.py::test_a_module_first_open_stays_inside_its_budget[replication]`: the module-scoped MainWindow had already restored replication from the saved session. The failure detail is complete in the journal (`detail_truncated=false`); `fatal-python.log` is empty for the observed prefix. The maximum recorded RSS/HWM was 3,314,561,024 bytes. The unrun tail has no verdict.

The archived ZIP and gzip job log are the original payloads. JSON snapshots bind both cancellation requests, terminal outcomes, old source and queued replacement sources; `receipt.json` summarizes their scope, and `sha256.json` verifies every other archived payload. The test repair is a separate commit so this archive does not rewrite the failed-source evidence.
