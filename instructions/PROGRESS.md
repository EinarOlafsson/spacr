# spaCR — instruction progress

Regenerated whenever an item is finished or the maintainer adds one.
Branch `nightly`. Last regenerated 2026-08-18 14:25, at commit `e3ecba0f`.

Priority is the order to work them in. Difficulty is what a "left" item costs.
An item a background agent has built but NOT committed is not 100%.

## Done

| # | Summary | % | Difficulty |
|---|---|---|---|
| 132 | Model & Inference states the formula it fits; `mixed` is the default | 100% | High |
| 133 | Every model explained; Toxoplasma annotation on every exported table | 100% | High |
| 134 | `analysis_mode` is a dropdown, not a text box | 100% | Low |
| 135 | The regression settings are one page a user can read | 100% | High |
| 131 | Show the cells behind a dot on the volcano | 100% | High |
| 138 | The explainer text fills the width of its box | 100% | Low |
| 143 | Plate position is opt in (measured); the model box is 2438 -> 892 chars | 100% | Medium |
| 141 | The user picks the backend and is told what each one is | 100% | High |
| 148 | The x and y axes behave as expected, logged or not | 100% | Medium |
| 109 | Image UMAP and the Gate Editor take several databases | 90% | Low |
| 128 | The regression panel: defaults, colouring, the grid, more tabs | 95% | High |
| 129 | Every plot in pyqtgraph, a tab each | 95% | High |
| 121 | Click a gene and everything known about it appears | 90% | Medium |
| 136 | The old matplotlib figures get the house style | 75% | Medium |

**148 closed 2026-08-18.** The log axes were LYING: `setLogMode` relabels the
axis and then walks its items, and `ScatterPlotItem` has no `setLogMode`, so
every one of these plots relabelled the ruler and left the dots where they
were. The transform is ours now, the untransformed values still answer the
tooltips and the hit-testing, a non-positive axis REFUSES log with the count in
the reason, typed limits are in data units, and grid and log are right-click
entries rather than a strip under the plot. `95f56a8e`, `0fae1c3c`, `1c2530d5`.

## Left

| # | Summary | % | Priority | Difficulty |
|---|---|---|---|---|
| 147 | Both fits visible; a square is a square; the menu has categories | 70% | 1 | Medium |
| 139 | Every regression graph in pyqtgraph; saved graphs you cannot see | 65% | 2 | High |
| 127 | Modules that overlap or are redundant | 65% | 3 | Medium |
| 154 | The measurements tab is a workflow, not a report | 0% | 4 | Very high |
| 152 | Two colours -- lines and font -- and they follow the theme | 0% | 5 | Medium |
| 153 | The model summary survives being loaded from disk | 0% | 6 | Low |
| 151 | Changing a line width should not take a minute | 0% | 7 | Low |
| 108 | Right-click ANY figure to restyle it | 10% | 8 | High |
| 62 | The setting animations show what the setting does | 30% | 9 | High |
| 150 | A saved figure is for paper, not for the screen | 0% | 10 | Medium |
| 149 | The volcano's y axis is continuous again | 0% | 11 | Medium |
| 144 | The model box is typeset, not dumped | 0% | 12 | Medium |
| 140 | A long fit says it will be long, and where it has got to | 0% | 13 | Medium |
| 146 | Delete a run from the Runs tab | 0% | 14 | Low |
| 145 | One reader, one writer, one key vocabulary | 0% | 15 | Very high |
| 122 | Regress across screens | 0% | 16 | High |
| 115 | Every regression diagnostic saved and scored | 0% | 17 | Medium |
| 116 | Click a search row to spawn that run's graphs | 40% | 18 | Low |
| 125 | The old volcano, the Runs tab, Qt by default | 0% | 19 | Medium |
| 142 | Force restart, for when stop does not stop | 0% | 20 | Medium |
| 137 | Drag images in, and the regex is worked out for you | 0% | 21 | High |
| 118 | Figure preferences: general, and per graph type | 0% | 22 | Medium |
| 126 | The theme must not lag while a run is going | 0% | 23 | Medium |
| 114 | A parameter search that cannot take the machine down | 0% | 24 | Medium |
| 60 | Every module at 100% test coverage | ON HOLD | — | Very high |

**147 at 70%:** B (a square is the canvas, `aa31a8d8`) and C (the menu has
categories, `d12d56c7` -- with the test migration landed FIRST so it stuck this
time) are done. A is not: the plot still does not say that the other level's
fit exists, which is the half that caused "it only runs once".

**Six items added by the maintainer on 2026-08-18:** 149, 150, 151, 152, 153,
154. Five of the six have a measured or code-located cause recorded in the
file, not just the report.

**60 is ON HOLD as a goal.** Tests still ship with every code change; nobody
hunts uncovered lines.

## Fixed in passing today

* `dc09b890` — HEAD imported `spacr.qt.widgets.database_set`, never committed,
  so a fresh checkout raised ModuleNotFoundError on the whole settings model.
* `482eae6e` — `CellMontageView` was the last screen putting hover help on an
  editable field (113).
* `aac29a84` — pip's editable pointer targeted the stale mirror, so `spacr` on
  PATH launched fifteen-commit-old code and any check run from a scratch path
  verified the mirror. Repointed; trap recorded as HANDOFF 3f.
* `703abbf9` — `regression_qc_report`'s `fmt` both defaulted to 'pdf' over the
  user's preference AND was thrown away before reaching `save_figure`.
* `2f4139f3` — three plates labelled "measurements", "measurements/measurements"
  and "measurements (2)" instead of plate1..3.
* `f0374448` — the mixed model took patsy's kept rows by position, not index.
* `7520f974` — four instruction commits carried only the index, because
  `git commit -- path` does not stage an untracked file. HANDOFF 0c, recurring.

## Known, not yet filed

* `tests/test_documentation_i18n.py` — 7 failures, one cause: a stale Korean
  review file for the `alpha` tooltip. Pre-existing.
* `tests/test_api_i18n_extractor.py` — the documented-API ratchet needs a bump
  naming what the new `DatabaseSetWidget` admitted.
