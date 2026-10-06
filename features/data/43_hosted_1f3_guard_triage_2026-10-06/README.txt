2026-10-06 exact hosted CI failure triage

The eight numeric gzip files are unedited GitHub job logs. Seven are failed
coverage/slow jobs from tests run 37494090176 at source 1f3e4a2403; the
eighth is the macOS ARM64 compatibility job 112409761134 from run
37503577086 at b77916cc94. The latter still has the scipy-at-Home failure
because its source predates the installed-Home repair c37b7f7d69. The
manifest records SHA-256 of each uncompressed log and receipt. It also records
the source patch used for the focused local guard repairs. These are
historical run logs, not a green verdict for the newer source.

Home-owned old-run repairs:
  112391877231: retired ripple performance row and obsolete Cells poison.
  112391877369: retired ripple resonance assertion.
  112391877404: obsolete Cells and blobs default theme expectations.
  112391877543: nested pool docstring and watch_normalization_pool category.
  112391877399: obsolete Cells poison.
Focused local receipts: categories-doc-after (82 passed), visual-after
(36 passed), safespacr-after (2 expected-abort cases passed), perf-after
(3 retained measured performance guards passed). The source-and-guard patch
contains the two local commits later integrated by root as d3b7dfc8b7 and
bf7873dec6. No performance budget, coverage ceiling, or translation policy
was relaxed.

Current-source follow-up:
  112391878542 contains two colour-picker guards: preferences.py called
  QColorDialog.getColor directly for custom animation colours. This remains
  a real source issue for the Preferences owner. Its seven-family registry
  failure was already fixed on current source. Its pointer-poll failure did
  not reproduce in one bounded d139ffe70c node replay (1 passed); keep
  order-dependent failure under observation, without an ambient source edit.

Workstation-owned generated artifacts still represented in the old logs:
  112391877324 and 112391877369: reviewed translation source, locale and
  runtime entries stale against the changed UI sources.
  112391877399: Help API index and settings-flow sections stale for the
  watch_normalization_pool source setting.
  112391877543: reviewed translations stale.
  112391878542: any source-current generated catalog drift beyond the
  explicitly listed test failures remains with the generated-artifact owner.
Normal regeneration and review are required; no hand merge or allowance is
recorded here. The exact job logs remain available in logs/ for the owner.
