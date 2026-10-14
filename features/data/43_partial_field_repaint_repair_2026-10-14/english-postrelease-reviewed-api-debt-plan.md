# Work / N615: held reviewed-API metadata debt

Execution is held until root releases the canonical publication hold. This is
a plan and raw evidence, not repaired metadata or a green evidence-check claim.

Actual local evidence checker returned EXIT1:18 MOVED and9 STALE records.
The frozen source repair's English audit returned EXIT0 at13418 symbols; fatal
Ruff on final source599043… / owning test51ca9f75… returned EXIT0 with zero findings.

## Configured CI classification

No current file in `.github/workflows/` directly invokes
`tools/check_reviewed_api_evidence.py`. The normal tests workflow invokes marker
partitions through tools/run_pytest_batches.py and its coverage runner.
Reporter tests in tests/test_reviewed_api_reporter_archive_scope.py call
reporter.main() against temporary synthetic records and require correct EXIT1
for moved/rewritten source. These tests remain enforcing; they do not assert
that the live reviewed-record corpus has zero debt.

tests/conftest.py:38 loads tools.pytest_translation_compatibility. Its advisory
allowlist includes non-English API catalog freshness and real-catalog browser
compatibility; `advisory()` excludes the language=en case. Non-English failures
are saved as actual incompatibility reports and surfaced as xfailed, not hidden
or relabelled fresh. English source failures remain blocking.

The docs workflow runs the pinned English sources-only refresh and
tools/report_translation_compatibility.py --english-required, then sphinx -W.
The reporter returns nonzero for incompatible English, and records localized
debt as advisory. API audit loads reviewed records for requested locales; its
English audit requests no locales. Fixture review-binding tests still enforce
their synthetic exact source/context/hash semantics.

Therefore the unchanged27-record localized metadata debt is a postrelease N615
Work item, not a newly configured full-CI required gate. Raw local EXIT1 remains
documented and must never be called PASS. No workflow/checker exception was added.

## Exact held work

Machine-readable postrelease-reviewed-api-debt-plan.json has all27 exact paths,
record indices, source text/hash, current source/context equality, translation
hash, destination and owning file SHA256. All27 recorded source hashes and
contexts are valid. All affected source/review/checker files are byte-identical
to original eb72, proven in reviewed-existing-debt-proof.json. Current English
manifest supplies the exact current block texts; no application import was used.

1. For all nine locales sv,de,es,zh_CN,pt,hi,ko,is,fr: move each existing
   `spacr.qt.widgets.figure_queue.FigureQueue.show_live_canvas#2` to#3, and each
   `spacr.qt.widgets.glass.glass#2` to#3. The source sentence is still exact,
   its source hash/context match, and its existing translation stays valid.
   Rebind the label in place; do NOT retire/retranslate these18 unchanged blocks.
2. In each locale's2026-10-03-nested-batch8.json, preserve the historical
   `spacr.qt.bridge.PipelineWorker.run._capture_show#1` record and add explicit
   retired date/reason after Work reviews the replaced current English block.
   Its source no longer occurs in the current3-block helper document. Do not
   relabel this old translation to new English or silently drop history.
3. Check all source/hash/destination bindings again against the final postrelease
   checkout. If any source differs from this plan, recompute/review the affected
   row instead of applying an obsolete label/index mapping.

## Normal-tool validation after the hold

The current authoring tool has no rebind/retire CLI switch; its writer replaces
same-label rows and would retain the stale old label if used to append#3. Thus
blindly adding a new#3 record is not the in-place relocation above. Apply the
reviewed metadata-only label/retirement edit through Work's existing authoring
lane, preserving unchanged source/hash/context/translation fields, then verify
with the normal tools. Do not hand-edit generated catalogs.

Use the authoring tool's read-only `--show` to verify destinations, for example:

```bash
python tools/write_reviewed_api_record.py --language sv \
  --label spacr.qt.widgets.figure_queue.FigureQueue.show_live_canvas#3 --show
python tools/write_reviewed_api_record.py --language sv \
  --label spacr.qt.widgets.glass.glass#3 --show
```

These exact blocks are shared across locales; the plan binds each locale's
existing translation separately. If a changed current source sentence genuinely
needs a new reviewed translation, use tools/write_reviewed_api_record.py with
its actual current label and reviewed --translation-file; do not reuse the old
capture_show sentence for new wording. Only changed sentences need review.

After metadata edits, in background and under normal caps:

```bash
tools/run_capped.sh 4G python tools/check_reviewed_api_evidence.py
tools/run_capped.sh 4G python tools/build_documentation_i18n.py --audit-english
```

If localized generated output actually needs repair, use the existing
build_documentation_i18n.py --repair-api-blocks --languages path for the affected
locales and inspect the exact diff. No blanket rebuild, README change, ceiling
raise, checker waiver or stale-artifact claim is part of this plan.

No metadata, source, test, generated catalog, workflow or threshold was modified.
