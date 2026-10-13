# Fast0 OPS two-well resume regression, 2026-10-09

Hosted Fast0 job 113582650451 ran the original source at
`f0b81dacefe089c54b1d22a4aebd363a29d53cfb`. Its batch 114 passed 800
tests and skipped one, then xdist reported worker `gw1` died while executing
`test_two_wells_with_stored_reads_resume_without_rerunning`. The last prior
result was logged at 00:22:12.029 UTC; the worker failure was logged at
00:27:12.329 UTC. The job log does not contain that worker's exit signal or a
native/Python stack. It therefore records a worker failure at the 300-second
test budget, not a proven OOM or native fault.

The original test ran a second complete acquisition with two copies of five
sites. The frozen five-site fixture report records 150.0 seconds in field
reads. `spacr.ops_engine._decode_field` builds a fresh four-thread pool for
each site, and `spacr.resource_log._WorkerStartGate` deliberately spaces
thread starts by 10 seconds. Five sites therefore incur about 150 seconds of
read-pool startup per well; two wells place this test on the 300-second limit.
The unchanged owning file passed locally in 849.99 seconds under a 4 GiB cap;
that local result does not negate the hosted failure.

The test-only repair copies two adjacent real sites (2 and 4) into each of
the two wells. The full five-site fixture and its malformed tile remain in
the sibling tests. Both repaired wells still run real stitch, objects and
decode, each places both sites, stores 232 read rows, and passes the original
per-well count and no-rerun assertions through the real plate driver. Their
archived reports record 60.0 seconds of field reads apiece. The focused test
passed with the unchanged 300-second limit in 275.32 seconds including its
five-site module fixture. No production code, worker pacing or timeout
changed.

`receipt.json` and `MANIFEST.json` bind all payload hashes. Run
`python verify.py --source --git` on a source-identical checkout. The
archive's full-file result, when present, is an adjacent validation of the
modified test file, not a hosted green verdict.
