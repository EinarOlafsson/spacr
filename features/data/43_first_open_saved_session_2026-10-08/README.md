# First-open timing fixture and persisted session — 2026-10-08

In the cancelled original-order serial run 37735178905, the first-open timing test failed at `[replication]` because its module-scoped `MainWindow()` had already restored replication from a saved session. The untruncated journal record is preserved as `hosted-observed-failure.json`; the full original artifact is archived separately under `features/data/47_obsolete_86_serial_supersession_2026-10-08/`.

The two 641 benchmark fixtures now construct `MainWindow(initial_app="__home__")`, which exercises the actual first navigation without restoring an unrelated saved module. The new test calls the **real structural fixture generator**, not a separately constructed lookalike. With a seeded replication session, a private one-line reversal to `MainWindow()` fails exactly at `assert "replication" not in fresh._screens`; the fixed fixture passes. The timing limit remains 10 seconds and no application code changed.

`negative-old-fixture.log.gz` and `positive-fresh-fixture.log.gz` are the raw bounded one-node runs. `adjacent-two-files.log.gz` shows 103/103 cases passing on the initial fixture fix; it predates the regression follow-up and is not claimed as a full serial-order result. `receipt.json` records exact commands, versions, source identities and limits.
