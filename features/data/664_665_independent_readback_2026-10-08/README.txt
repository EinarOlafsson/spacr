Independent N664/N665 archived CPU evidence readback, 2026-10-08.

Review snapshot: 389ce5376d3f55c8b4f370cde62cd69b7ef793f8.
No spaCR imports, tests, GPU, production edits or private candidate integration.

Verified all 203 initial and 25 supplement payloads against compressed and
uncompressed SHA256/byte lengths, with no unexpected files except each receipt.
All 26 gzip logs (including paired tool logs) are intact. Recorded pytest
outcomes and terminal exit statuses agree with declared cohorts. Failed r2,
r8 and r9 runs are retained; final supplement r10 passed 60 cases. Passing
cohorts overlap and are not a unique aggregate count.

The initial 36 frozen source/test/config/tool bindings and 108 coverage-input
hashes match. All 29 comparison-base snapshots match the available base Git
objects. Original 25 item60 and 76 sequencing test function bodies are
AST-identical. The supplement freezes five final tests; two differ from its
initial archive, three are unchanged. Unlike the initial archive, it has no
source_bindings or coverage_inputs map; this schema difference is explicit.

Both final coverage reports were independently recomputed against frozen
before/current source using exact changed-line mapping and Coverage 7.16.0
static branch normalization. Their actual merged SQLite arcs and executed
lines match every reported hit/miss. All 29 reported production hashes bind
to initial frozen source. Final: 1,137/1,175 changed lines; 328/349 changed arcs.
The remaining 38 lines and 21 arcs stay OPEN. Database changed lines/arcs alone
are complete (297/297 and 83/83). No broader acceptance is implied.

All 29 archived production snapshots differ from current public app bytes;
these are held private candidate sources. Complete private Git-tree assertions
(e.g. documentation/tools unchanged and final patch identical within its
private integrated candidate) are not independently established by this
readback. Do not use this proof as app integration, full-module numerical,
GPU, native-display, hosted, serial, or human aesthetics acceptance.

readback.json contains every payload hash, source comparison, original log
outcome and exact remaining source line/arc inventory. reproduction.json names
the isolated Coverage wheel SHA and bounded command. verify_readback.py reads
Git blobs, so sparse checkouts work. Use Coverage 7.16.0 in an isolated scratch
PYTHONPATH overlay; never install into or change the shared Python environment.
The original archives are referenced, not duplicated.

Verify this compact archive with:
  python verify_manifest.py --repo /path/to/repo --git HEAD
Reproduce the readback with the command in reproduction.json. No pytest run
is required. GIT_NO_LAZY_FETCH prevents fetching private candidate commits.
