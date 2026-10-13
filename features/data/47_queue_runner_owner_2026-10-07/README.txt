QueueScreen parked-reader native ownership repair, 2026-10-07.

A pipeline may be between cooperative cancellation boundaries beyond the
existing five-second close wait. Before, the bridge kept its Python reference
but the closing screen still owned the native QThread. Deleting that owner
causes real SIGABRT -6. The same controlled real-runner probe after the small
conditional drain guard exits zero: owner dies while the runner lives, release
reaches the cancellation checkpoint, both queue items stay runnable, and the
parked reference and native runner are then released. Core dumps are disabled
only for the probe process; no global configuration is changed.

Forty-seven bounded focused checks pass; successful drains retain normal
ownership and already-deleted runner handling remains covered. Both new guard
branches are directly observed. Exact old hosted source is byte-identical to
the before snapshot; unchanged endpoint mapping plus direct new coverage has
zero statement or branch gaps. Old/full hosted 225 statements/40 branches
become 226/42. No coverage allowance, deadline or cancellation boundary changes.
This does not close full Qt acceptance or the separate Shiboken SIGSEGV. A
pipeline that never reaches a safe checkpoint can still outlive the existing
process exit wait; this proof establishes safe owner deletion, not forced
termination. Native logs are raw subprocess capture; focused result provenance
is explicit in focused-tests.txt.
