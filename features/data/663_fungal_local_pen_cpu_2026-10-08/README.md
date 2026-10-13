# Exact local fungal pen reuse, CPU proof

Before ambient SHA2567c9dcdcfb032a4f9677fcb45d101f5d5540e887d28145b794c5d91625632c9bc;
afterfa95c6677bea19f66b4cfb5e4049ae8858269e5b9cc53e337653fe176301fbf4,
source/test commitaa2c82593. No public signatures, docstrings, tr literals,
geometry, work/quality controls, cache limits, draw order or colors change.
Each original/cached path paint pass owns one local SolidLine/RoundCap/
RoundJoin QPen; color and width still update separately for every group.
Qt keeps its implicitly shared pen value and mutation detaches safely.

Instrumented profile, fixed shader and actual live worker receipts have
separate scopes. The28-frame profile has wrapper overhead and inclusive
stages; caller threadCPU excludesQt raster helperthreads, so wall−CPU
cannot be called solelyGILwaiting. Its per-call rawjournal is archived
compressed with source, script and frame summary. Four fixed setup frames
place4316pen constructions at9.23–9.50ms/frame. Fixed uninstrumentedABBA
medians default153.07→146.46ms, Random155.71→143.84, light147.06→130.75.
That phase is36rendercalls/18paired comparisons of repeated fixedstates,
not36unique fullnativepairs. Fifteen separate pen-value cases pass.

Fresh actual native3840×2160 Density3/Detail2/Size2.5 workerABBA:
* default before4.785/4.757FPS; after5.180/5.353FPS;
* Random before4.777/4.758; after5.377/5.548.
Every process requested24FPS, retained the full native buffer and24cap,
and retired its worker immediately and200ms afterhide. GUI repaintcounts
remainabout24 and do NOT measure distinctframes. Publishedclocks provide
the stated cadence. Cold birth-phase near24FPS is not matureacceptance.
Hard24FPS remains OPEN; no smoothness guarantee or aestheticacceptance.

Nineteen focused tests pass43.20s under4GiB: six realnative dark/light and
default/Random/custom legacy-QPen comparisons with retainedframe ownership,
two partial optional-cache failure recovery cases and eleven unsupported
painter-context cases. All12 new/replaced pen statements are observed
in source-bound branchcoverage. No newcontrolflow or ratchetchanges.
The19 tests and earlier pixel phases overlap; their counts are not summed.
Signature/docstring andtr-call ASTs are unchanged, fulltestRuff and enforced
sourceRuff pass. No appgenerated/API/docs/catalog artifacts were changed.

Live dependency hashes/cold/steady/heartbeat/RSS and retirement are in
each frozenworker receipt. PeakprocessRSS351580–385160KiB; this is separate
from renderer-owned sparse cachearrays. No newpersistentcache exists.
CPUagent quiescence was coordinated, not a universalhostisolation claim.
NoGPU, inference, fullsuite, useraesthetic or hostedCIacceptance is claimed.

`verify_manifest.py` supports filesystem and `--git` committed sparseblobs;
`verify_pen.py` checks frozen sources, tests, changedline execution and live
native/lifetime provenance, including current HEAD production/test bytes. The
agent source commit is provenance only; no agent-only Git object is needed.
These verification commands do not rerun Qt.
