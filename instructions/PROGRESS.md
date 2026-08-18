# spaCR — instruction progress

Regenerated whenever an item is finished or the maintainer adds one.
Branch `nightly`. Last regenerated 2026-08-18 15:30.

Priority is the order to work them in. Difficulty is what a "left" item costs.
An item a background agent has built but NOT committed is not 100%.

## Done

| # | Summary | % | Difficulty |
|---|---|---|---|
| 131 · 132 · 133 · 134 · 135 · 138 · 143 | earlier work, unchanged | 100% | — |
| 148 | The x and y axes behave as expected, logged or not | 100% | Medium |
| 153 | The model summary survives being loaded from disk | 100% | Low |
| 109 | Image UMAP and the Gate Editor take several databases | 95% | Low |
| 139 | Saved graphs you cannot see (C); box not bar (B) | 95% | High |
| 147 | Both fits visible; a square is a square; menu categories | 97% | Medium |
| 128 | The regression panel: defaults, colouring, the grid, more tabs | 95% | High |
| 129 | Every plot in pyqtgraph, a tab each | 95% | High |
| 141 | The user picks the backend and is told what each one is | 92% | High |
| 121 | Click a gene and everything known about it appears | 90% | Medium |
| 154 | The measurements tab is a workflow (A–E, G) | 85% | Very high |
| 62 | The setting animations show what the setting does | 85% | High |
| 136 | The old matplotlib figures get the house style | 75% | Medium |

## Left

| # | Summary | % | Priority | Difficulty |
|---|---|---|---|---|
| 152 | Two colours -- lines and font -- and they follow the theme | 0% | 1 | Medium |
| 151 | Changing a line width should not take a minute | 0% | 2 | Low |
| 149 | The volcano's y axis is continuous again | 0% | 3 | Medium |
| 150 | A saved figure is for paper, not for the screen | 0% | 4 | Medium |
| 108 | Right-click ANY figure to restyle it | 30% | 5 | High |
| 127 | Modules that overlap or are redundant | 45% | 6 | Medium |
| 139 | Section A -- every generated graph in pyqtgraph | 15% | 7 | High |
| 154 | Section F -- the four steps, and one run per column | 0% | 8 | High |
| 116 | Click a search row; every run has its own volcano | 40% | 9 | Medium |
| 144 | The model box is typeset, not dumped | 0% | 10 | Medium |
| 140 | A long fit says it will be long, and where it has got to | 0% | 11 | Medium |
| 146 | Delete a run from the Runs tab | 0% | 12 | Low |
| 145 | One reader, one writer, one key vocabulary | 0% | 13 | Very high |
| 122 | Regress across screens | 0% | 14 | High |
| 115 | Every regression diagnostic saved and scored | 0% | 15 | Medium |
| 125 | The old volcano, the Runs tab, Qt by default | 0% | 16 | Medium |
| 142 | Force restart, for when stop does not stop | 0% | 17 | Medium |
| 137 | Drag images in, and the regex is worked out for you | 0% | 18 | High |
| 118 | Figure preferences: general, and per graph type | 0% | 19 | Medium |
| 126 | The theme must not lag while a run is going | 0% | 20 | Medium |
| 114 | A parameter search that cannot take the machine down | 0% | 21 | Medium |
| 60 | Every module at 100% test coverage | ON HOLD | — | Very high |

**60 is ON HOLD as a goal.** Tests still ship with every code change.

## What closed this afternoon, and what it cost to find

* **148 — the log axes were LYING.** `setLogMode` relabels the axis and walks
  its items; `ScatterPlotItem` has no `setLogMode` and every point on every
  one of these plots is one. Verified before fixing: axis logMode True,
  scatter data byte-identical, vertical screen movement 0.0 px. The transform
  is ours now across all 18 draw sites, hovers report the real p-value, the
  selection ring moves with the dots, and a non-positive axis REFUSES with the
  count in the reason. 0.37 ms per toggle against pyqtgraph's own 4.7 ms.
* **147 C — the menu categories, which were reverted this morning.** Landed by
  doing the test migration FIRST, green against the flat menu, so the
  restructure became a one-file change. Found on the way: `menu.addMenu(title)`
  returns a QMenu PySide considers Python-owned, so every submenu and its
  actions were destroyed when `build_style_menu` returned -- surfacing later as
  "Internal C++ object (QAction) already deleted". And `addAction(text, cb)`
  connects `triggered`, whose bool is DROPPED for a slot that does not take
  one, so a checkable entry wired that way reports the state it had BEFORE the
  press.
* **154 A — the merge freeze.** On a JobRunner now, measured through the REAL
  button on 4 databases x 40,000 cells: 6.2 s, 100 event-loop passes with the
  window live, progress repainted 16 times, count ending exactly at
  480,000/480,000, and cancellable.
* **154 C — the answer to the file_name/path_name question.** They were
  NEITHER dropped NOR meaned: `aggregation_plan` asks `is_numeric_dtype` first,
  so a text column takes `first`. The MERGE was right and the PRE-merge
  SENTENCE was wrong -- `_table_notes` walked column NAMES and never looked at
  a dtype. But underneath it a real gap: the `first` was silent even where the
  identifier was NOT constant within its group, which is invented provenance.
  Refused and named now.
* **154 D — the join key IS affected**, and the panel is not the one doubling
  it. Measured, with `canonical_plate_id` pinned against `correct_metadata`.
* **153 — `save_summary_to_file` was inside `if settings['verbose']:`**, so
  most runs never wrote a summary at all. The instruction did not know that.
* **62 — 26 of 94 GIFs rendered a near-white border ring** flashing once per
  loop, and `measure_visible_change` counted all 9.75% of it, flattering the
  audit that was supposed to police them.

## Known, not yet filed

* `tests/test_api_i18n_extractor.py` — the documented-API ratchet needs a bump
  for ~21 new public symbols, and `tools/build_documentation_i18n.py
  --sources-only` has not been run for them.
* `tests/test_documentation_i18n.py` — 7 pre-existing failures, stale Korean
  review file.
* `spacr/guide_permutation.py:574` — the last raw `fig.savefig` on the
  regression path: saved-and-invisible, and `dpi=600` applied to PNG only.
* `spacr/sp_stats.py` vs `figures/stats` — 127 finding 2, two engines choosing
  a statistical test and disagreeing on 3 of 5 cases. The correctness item.
