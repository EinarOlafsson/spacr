# Complete 0b8 coverage phase, 2026-10-08

Required tests run `37741615330` used source
`0b8c2c4120a0ede4de568d484ea3fc9d517f1de3`.
All 12 coverage shard jobs completed and uploaded process data. Eleven selected-test
jobs succeeded. Shard 9 failed only the two `test_local_ci_replay.py`
profile-count assertions, which still pinned seven Qt and five coverage
exclusions after the dedicated serial timing file was added. The repaired
count contract belongs to a newer source revision; it is not credited here.

The combine job consumed all 12 shard artifacts. Its combine and unchanged
per-module numerical ratchet steps passed: 664 of 664 shipped modules checked,
zero coverage regressions, zero unconfirmed rises, zero recovered worker
batches, 99 improved modules, and 455 at 100%. The separate step requiring
all coverage shards' selected tests to pass failed, so the combine job is red.
This is **numerical coverage acceptance only**, not a green required workflow
or a conclusion about unrelated tests.

The original `module-coverage-report.zip` is stored byte-for-byte, including
`coverage.json`, `module-coverage-ratchet.json`, and its readable report.
Each `*.log.gz` is a deterministic compressed copy of the full job log;
`receipt.json` records its raw SHA-256. Exact job JSON, a direct jobs snapshot,
and the then-current run snapshot preserve the source and conclusions. The
run snapshot was taken while other required jobs were still active; it is a
coverage-phase snapshot, not a final parent-run verdict.

Run `python verify.py` or `python verify.py --git` after commit to validate
payload hashes, ZIP integrity, source and job identities, selected-test
failure, and the separate numerical and required-shard outcomes.
