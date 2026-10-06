2026-10-06 complete failed-job snapshot for obsolete hosted tests run 37494090176

Source: 1f3e4a240339ebc88993d72c49b39ab5716e3904. The saved GitHub
pre-cancel snapshot identifies 21 failed jobs, six successful jobs and one
running Fast shard. All 21 failed-job logs are retained verbatim after gzip
compression in logs/. MANIFEST.json binds each job name and ID to SHA-256 of
both its compressed archive and uncompressed original. The snapshot and a
per-job failed-node index are retained alongside them. Run VERIFY.py from any
working directory to check the archive without GitHub or the original scratch.

These failures belong to the old source, which was cancelled after this
snapshot to free the newer ordinary tests workflow. They cannot establish a
current-source coverage ratchet verdict: the old coverage combine job uploaded
an aggregate but failed because coverage shards had failed. Parent items 43
and 288 remain open pending current-source CI. The protected N47 serial run
was not cancelled.

The logs show a mixture of retired theme/default assertions, historical
Preferences/source failures, and workstation-owned generated API, runtime,
localization, settings-flow, help, notebook and README freshness failures.
No generated files, guards, coverage ceilings or exclusions changed in this
archive. The old source-bound triage is in failure-triage.md; individual logs
and failed-node-index.json.gz are the authoritative detail.

One distinct Python 3.9 minimum-dependencies failure was a native-volume
test assumption about Cellpose conversion. The original spaCR call sends
(Z,Y,X,1) with do_3D=True, z_axis=0 and channel_axis=-1. Official Cellpose
4.0.7 transforms.py lines 524-537 zero-pad that one channel to three after
conversion; current installed Cellpose 4.2.1.1 returns one channel. The
corrected test keeps the raw call/axis/pixel provenance assertions and accepts
only the two supported converted layouts, with copied first channel and zero
extras. The original official 4.0.7 wheel is preserved in proofs/; its wheel
SHA-256 is 1cab108744ab95c6df1928b8624e527d1c1822a85e643771edd336f1bad1b2dd,
and its cellpose/transforms.py SHA-256 is
792c75184158783170480a5fabeab883a01faa223fe4626785cadaf73429b286.
The exact test-only correction is commit 270c7ff4562ebcefb48ab5f0412c094eda257b21
and proofs/native-cellpose-test.patch.gz. Receipts: two parameterized cases pass
under installed 4.2.1.1 and an isolated 4.0.7 wheel overlay; the full native
batch test file passes 47/47 under the installed version. REPRODUCE.sh has
the bounded commands and its actual full exit-zero output is retained as
proofs/archive-reproduce.log.gz. No pretrained inference or GPU claim follows from
these tests.

Current 3ca-source tests run 37511792034, rather than this historical run,
is the relevant mandatory verdict. This snapshot records what failed; it is
not a claim that current CI is green.
