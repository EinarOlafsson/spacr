Two test-only ownership repairs give the Classify family and Regression
analysis-unit SettingsWidgets fixtures a qtbot-owned QWidget parent. The
parent owns their controls, timers and undo stacks through ordinary Qt
teardown. All 18 original test-function ASTs, assertions, parameter cases,
styles, GC behavior and memory guards remain unchanged.

Matched before/after runs execute one real node from each file followed by
the existing field-fade stylesheet sentinel, with no injected GC. Both pass
3 cases. Post-sentinel cached widget/top-level counts change 412/129 to 0/0.
Sampled RSS at that boundary changes 775,503,872 to 568,664,064 bytes. These
are bounded observations, not a whole-suite memory or crash acceptance.

The matched after run precedes import-whitespace formatting only. The final
complete two files plus identical sentinel bind the final source and pass
27 cases in 3.46 seconds. Their final sentinel cached counts are 0/0.
Journal counts use the unchanged test fixture's last safe snapshot; they
can precede actual qtbot/deferred-delete teardown at an earlier file boundary.
No fresh census is inferred from those earlier snapshots.

`receipt.json` records exact baseline/test commits, before/after test hashes,
eight unchanged production/runtime/test-helper hashes, commands, boundary
records and limits. The original logs and JSONL journals are preserved.
Baseline is e2983466b15a875b07d978b15cb19dd7c9576670; test repair is
8e1ef7d23c. This evidence accepts the two fixture ownership children only.
Full Qt serial acceptance and the separate Shiboken native crash remain open.

Verify archive bytes and semantic/source bindings without rerunning:

```
tools/run_capped.sh 2G python features/data/47_family_analysis_fixture_owners_2026-10-07/verify_proof.py
```

Add `--git HEAD` to read committed blobs. Production bindings identify the
measured checkpoint; later unrelated source changes require their own
acceptance. Optional replay uses the exact nodes, environment, unchanged
journal plugin and capped command in the receipt. Create fresh journal paths
under `/mnt/wd4tb/scratch`; no test order/GC/style customization is needed.
