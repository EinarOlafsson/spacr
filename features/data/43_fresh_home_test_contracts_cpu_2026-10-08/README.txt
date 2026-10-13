Fresh Home test contracts, 2026-10-08

Seven constructor substitutions in six existing tests explicitly request Home.
Real persisted target records are written before each constructor by the archived plugin; production session-restore code remains active. Six original assertions fail before and pass after; the seventh closed-category comparison passes in both states and is not claimed as a reproduced failure. Every after constructor starts at Home with an empty module cache. Original assertion bodies, time limits, mode flags and settings/row comparisons are unchanged. Deliberate restore tests and both Qt-owned first-open fixture files are outside this patch.

Actual pytest8.4.2, CUDA-hidden/offscreen CPU, 4 GiB cap: before6 failed/1 passed36.62s; after7 passed28.10s. These durations are correctness-run observations, not performance claims. The initial sparse checkout omitted pytest.ini and emitted marker warnings; after uses normal strict configuration. Both raw logs retain that distinction. Three full-Ruff I001 findings are byte-identical before/after; fatal F/E9 and diff checks pass. No full suite or memory/native-fault acceptance is claimed.

Run verify_manifest.py --git HEAD --bind-current after cherry-picking this proof and implementation. reproduce_after.sh replays the seven saved-session cases in an isolated checkout with the retained pytest8.4.2 overlay; it does not mutate production files.
