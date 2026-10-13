# Hosted 2d4 coverage verdict, 2026-10-07

Tests run `37574709966` used exact source
`2d4f8f1c2bae3cf8b3914be3b1c670147a9f3ccf` and ended in failure.
All 29 jobs completed: 15 succeeded and 14 failed. The complete logs for
all 14 failed jobs, terminal run/job API responses, and GitHub's coverage
report ZIP are preserved here. `write_manifest.py` generates `MANIFEST.json`
from those payloads; `VERIFY.py` checks stored and decompressed hashes and
the source-bound verdict. The ZIP SHA-256 matches GitHub artifact
`11471310982`'s upload digest.

The numerical module ratchet **passed**: 664 of 664 shipped modules checked,
all 12 shard coverage artifacts present, zero modules failing the ratchet,
zero unconfirmed rises, and zero stale baseline entries. One below-floor
module has the pre-existing stated exemption. Coverage shard 7 lost one
worker to a native segfault during a Make Masks puncta test; the runner
reran 19 affected files serially and recovered their coverage. The same
puncta test passed in that serial recovery. The aggregate job was red because
selected shard tests failed, so this is not a green CI claim.

The failed selected tests concern a retained queue runner through a captured
cancellation exception, two stale Make Masks loader doubles that returned
`None` instead of a successful drain, the historical Swedish/French runtime
review records, README prose length and checkout-size measurement, and the
missing Divide / Merge label in the README or linked guide. The Fast and
Minimum-dependency shards repeat those source-bound failures; the release
gate is derivative. API symbol-count and Cellpose translation differences
shown as XFAIL diagnostics in some logs are not counted as failed tests.

This receipt describes only the 2d4 source. Local repair commits
`3ef184a7ad` and `5976df4757` are not part of that hosted run, and the
documentation/catalog owner updates require their own source-current run.
No coverage baseline, floor exemption, timeout, or test selection was
changed by this archive.
