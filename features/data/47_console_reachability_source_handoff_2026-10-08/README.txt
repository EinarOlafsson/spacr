Incremental Console-only transfer after accepted Home62.

Run verify_archive.py and check the compressed patch before adoption.
Only Console source and its existing regression-test file change. Preserve
workstation-owned source elsewhere. Minimum height retains real output;
hidden Jump-to-end control retains its space so a single normal scroll
leaves the full start/end caret inside the stable outer viewport.

Original strict failure controls are retained: each source change removed
independently causes the actual regression to fail. No pixel allowance,
timeout change or screenshot-only acceptance is used.

Normal API13224 and all runtime source buckets, including UI7263, are
byte-identical to62. Only the Console source hash changes; regenerate
combined source-bound artifacts with normal tools. Private source transfer
is not full original-order serial acceptance or published application code.
