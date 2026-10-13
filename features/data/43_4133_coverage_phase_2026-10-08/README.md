# 4133 coverage phase, 2026-10-08

This archive records the complete coverage phase of required tests run
`37728397690` on source `4133beafcd0ae427795617a4a01e295fd40539b7`.
It is **not a green acceptance receipt**. All 12 coverage shards finished:
eight passed their selected tests and four failed. The distinct failed nodes
and job IDs are in `receipt.json`.

The combine job downloaded all 12 process-data artifacts and the unchanged
numerical ratchet **passed**: 664/664 shipped modules checked, zero modules
failing the ratchet, zero unconfirmed rises, and zero recovered worker batches.
The combine job itself failed only at its separate requirement that every
coverage shard pass its selected tests. This distinction matters: a complete
coverage measurement does not make the required run green.

`module-report.zip` is the unmodified GitHub artifact `11531244445`, with the
original coverage JSON and ratchet report. The 13 `*.log.gz` files are
deterministic gzip copies of the full 12 shard and combine job logs. Matching
job JSON and run/jobs snapshots preserve source identity and job state. The
parent tests run was still in progress at the coverage-phase snapshot; the
snapshot is not its terminal conclusion. The failed shard 7 process artifact
remains on GitHub as artifact `11530852234`; its verified SHA is in the receipt
without duplicating its 27 MB ZIP here.

Run `python verify.py` for filesystem payloads or `python verify.py --git`
after commit. Both modes check every archived payload hash, the exact source
and job identities, and the separate numerical and selected-test verdicts.
