# 55bad coverage phase, 2026-10-08

This archive records the complete coverage phase of required tests run
`37720484173` on source `55badc57ff25ef6a7e14721561773a412b515318`.
It is a failed run, not a green acceptance receipt. All 12 shard jobs finished:
eight succeeded and four failed. The combine job measured all 664 shipped
modules from all 12 shard artifacts with no missing or recovered inputs, then
failed both the numerical ratchet and the required-shard check.

The only numerical regression was `spacr/qt/widgets/ambient.py`: no uncovered
statements and one uncovered branch, `5957→5960`, against the unchanged
zero-branch allowance. The four failed shards contain five distinct failed
test nodes: the API callable count, the backdrop cache tuple, the archive
case collision, and two guards triggered by one Cellpose mock. Exact nodes,
job IDs, conclusions, source blob and report hash are in `receipt.json`.

`module-report.zip` is the unmodified GitHub artifact `11528168420`; it
contains the original coverage JSON and ratchet JSON/text. The 13 `*.log.gz`
files are deterministic gzip copies of the full 12 shard and combine job
logs. Matching `*-job.json` files and run/jobs snapshots preserve job state.
The parent run was still in progress at the coverage-phase snapshot; these
snapshots must not be read as its terminal verdict. Later fixes belong to
separate source revisions.

Run `python verify.py` for filesystem payloads or `python verify.py --git`
after this archive is committed. Both modes check every payload hash and the
source-bound numerical claims without changing a baseline.
