# Hosted 032 coverage verdict, 2026-10-07

Tests run `37573419011` used exact source
`0324166b59da0ba4c31e01cf108087f05d3e5992` and ended in failure. All 29
jobs completed: 19 succeeded and 10 failed. The complete logs for the ten
failed jobs, terminal run/job API responses, and GitHub's coverage report ZIP
are archived here. `write_manifest.py` generates `MANIFEST.json` from those
payloads; `VERIFY.py` checks stored and decompressed hashes and the source-bound
verdict. The report ZIP SHA-256 matches GitHub artifact `11464421456`'s upload
digest. The original logs and report were downloaded through GitHub's API.

The numerical module ratchet **passed**: 664 of 664 shipped modules checked,
all 12 shard coverage artifacts present, zero modules failing the ratchet, zero
unconfirmed rises, and zero stale baseline entries. One below-floor module has
the pre-existing stated exemption. The aggregate GitHub job was still red
because its final step requires every shard's selected tests to succeed.
Those test assertions failed; this archive is not a green CI claim.

The failed tests concern the historical Swedish/French reviewed-runtime
retirement list, README prose length and checkout-size measurement, and the
missing Divide / Merge label in the README or linked feature guide. The
Python 3.9 MinDeps log also shows a stale 13182 API-symbol test pin against
13199 source/catalog symbols. The nine `generate_cellpose_masks_sam` translation
differences in MinDeps shard 1 are report-only XFAIL diagnostics and are not
counted as failed tests. The release gate failed because Fast, Minimum
dependencies, and Coverage combine were red. The report ZIP contains both the
full coverage JSON and machine-readable/text ratchet reports.

This receipt describes the 032 source only. A newer run for source
`2d4f8f1c2bae3cf8b3914be3b1c670147a9f3ccf` started after the 032 run
finished; it needs its own verdict. No coverage baseline, floor exemption,
timeout, or test selection was changed by this archive.
