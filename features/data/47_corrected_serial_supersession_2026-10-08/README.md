# Cancelled 4133 Qt serial run — 2026-10-08

The original-order Qt serial run 37728434064 at source `4133beafcd0ae427795617a4a01e295fd40539b7` was intentionally superseded by corrected-source run 37735178905. GitHub reports both the old run and job 113161465305 as **cancelled**. This is not a green or failed full-suite verdict.

The cleanup artifact 11531049419 is preserved losslessly as `partial-artifact.zip`, with the complete job log compressed as `terminal-job.log.gz`. Its journal contains 208 completed file records and 209 begun file records; the last completed file is `tests/qt/test_ambient_home.py`, and `tests/qt/test_ambient_motion.py` had begun. Pytest output reached 10%. The journal contains no `test_failure` event and `fatal-python.log` is empty within this observed prefix. Its highest recorded RSS was 3,289,681,920 bytes and HWM 3,420,073,984 bytes. These observations do not characterize the unrun tail or establish full serial memory acceptance.

`before-*.json`, `request-receipt.json`, `terminal-*.json`, `corrected-pending-run.json`, and `receipt.json` bind the cancellation, source, replacement run, and artifact. `sha256.json` verifies every other archived payload.
