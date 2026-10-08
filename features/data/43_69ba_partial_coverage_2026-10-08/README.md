# 69ba superseded coverage, 2026-10-08

Required tests run `37737743925` used source
`69ba4e462f2ea60924a977b192d006ce9b70f42c`. It was intentionally
superseded after a newer source with verified timing-fixture corrections was
dispatched. The run ended **cancelled**, so this archive is partial evidence,
not numerical or required-test acceptance.

Coverage shard 6 completed successfully before cancellation: ten bounded
batches, 6,875 passed and 30 skipped. Its process-data artifact is GitHub
artifact `11534155250`; the verified ZIP SHA is in `receipt.json`. The other
eleven coverage jobs concluded cancelled. Their available logs retain only the
work completed before cancellation. Shard 5 was cancelled before GitHub made
a job log available; no log is invented here. GitHub then started the combine
job, which failed because shards 5 and 11 had no coverage data. Its numerical
ratchet step was skipped, so no 664-module numerical verdict exists for this
source revision. The release gate failed on the cancelled blocking jobs.

The available partial logs record a native SIGSEGV during a Make Masks
parent-source test in shard 0 and two CI replay-profile assertions in shard 9.
These observations are preserved as pre-cancellation failures, without
claiming a terminal shard verdict or cause. A later focused test-only repair
handles the two stale replay-profile counts in a separate source revision.

The `*.log.gz` files are deterministic compressed copies of the 11 available
shard logs and the combine and release logs. Each job has its exact API JSON.
The before/after/final run and jobs snapshots show how GitHub reported the
supersession and its dependent jobs. Run `python verify.py`
or `python verify.py --git` after commit to check every archived payload,
source identity, and the distinction between the one passed shard and the
cancelled partial work.
