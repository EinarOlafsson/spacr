# Source-bound 7a51 coverage phase (2026-10-08)

This is the complete 12-shard and combine phase of required run `37744767240`,
source `7a51b6c921ea0d9b51ca3f5e6d28a68904ce2cab`.

The numerical ratchet **passes** with all 664 shipped modules measured, all 12
shard data sets present, no coverage regression or unconfirmed rise, and one
recovered SIGABRT worker's coverage. The required test gate **fails**: ten
shards succeeded; shard 0 lost an xdist worker to SIGABRT during a Make Masks
parent/magnifier race test, and shard 5 failed a Qt 6.12.0 metaobject name
assertion. Serial recovery of shard 0's unreported files protects coverage
measurement and does not turn its failed test into a pass. The native artifact
records a rejected ELF identity and **no recovered native backtrace**.

The raw GitHub logs are stored byte-for-byte inside deterministic gzip files;
`receipt.json` records their original SHA-256 values. The original report and
native ZIPs are unchanged. `verify.py` checks all payload hashes, Git source
and workflow blobs, jobs, report integrity and these separate verdicts. The
run snapshot was captured at phase completion while other required jobs could
still be active; this archive does not claim full-run terminal acceptance.
