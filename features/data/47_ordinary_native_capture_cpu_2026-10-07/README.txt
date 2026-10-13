Ordinary Qt/Coverage native capture CPU proof, 2026-10-07

Implementation 8e5dead63e plus required leader correction 4c27c26ae5 extends
existing tools only. Protected serial defaults
remain unchanged. No application source, test selection/order, workers, budgets,
failure/recovery policy or ratchets change. Only bounded text/JSON is uploaded.

Validation phases/counts are recorded separately in acceptance.json; the 19-case
ordinary-plugin replay overlaps the 78-case cohort. Final three route and five
systemd-identity cases were added after that cohort. Ruff and diff checks pass.

One controlled owned Python SIGSEGV under gdb yielded a 13,086,152-byte core;
the collector verified NT_PRPSINFO process leader and mapped executable,
obtained its C stack and
deleted the raw core. Native proof git HEAD was 8e5, with final committed
helper bytes already on disk: explicit hashes bind those bytes to this archive.
Final leader phases pass47 plus1 (overlapping earlier extraction cases).
The generated core and its identity journal were not uploaded or archived.
The bounded native report retains the matched identity as provenance.

Reproduce from the repository root with CUDA hidden and a capped CPU Python:
  tools/run_capped.sh 4G python features/data/47_ordinary_native_capture_cpu_2026-10-07/reproduce.py
Requires gdb; creates one private scratch directory and removes its raw core.
This proves the diagnostic mechanism, not a spaCR crash fix. Hosted ordinary
core capture, the original puncta SIGSEGV and protected QEventLoop SIGSEGV are OPEN.
