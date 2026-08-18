# spaCR — instruction progress

Regenerated whenever an item is finished. Branch `nightly`.
Last regenerated 2026-08-18 13:53, at commit `482eae6e`.

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
| 128 | The regression panel: defaults, colouring, the grid, more tabs | 95% | High |
| 129 | Every plot in pyqtgraph, a tab each | 95% | High |
| 121 | Click a gene and everything known about it appears | 90% | Medium |
| 136 | The old matplotlib figures get the house style | 75% | Medium |

**141 closed 2026-08-18** at `84beb1f7`. `describe_backends()` had been written
and tested since the morning and NOTHING in `spacr/qt/` called it, so the box
the maintainer asked for was not on screen -- green tests, unreachable feature.
It is wired now, with a 309-line test that drives the real screen.
HONEST REMAINDER, recorded rather than rounded away: only statsmodels and torch
FIT anything. pymer4, cuML, pyfixest, glum, numpyro and gpytorch are inventory
only -- described, greyed, with their pip command. The acceptance line "a mixed
fit through pymer4 or torch agrees with statsmodels" is met by torch alone, and
no package was installed to meet it.

## Left

| # | Summary | % | Priority | Difficulty |
|---|---|---|---|---|
| 148 | The x and y axes behave as expected, logged or not | 65% | 1 | Medium |
| 109 | Image UMAP and the Gate Editor take several databases | 80% | 2 | Low |
| 139 | Every regression graph in pyqtgraph; saved graphs you cannot see | 60% | 3 | High |
| 127 | Modules that overlap or are redundant | 65% | 4 | Medium |
| 147 | Both fits visible; a square is a square; the menu has categories | 0% | 5 | Medium |
| 108 | Right-click ANY figure to restyle it | 10% | 6 | High |
| 62 | The setting animations show what the setting does | 30% | 7 | High |
| 144 | The model box is typeset, not dumped | 0% | 8 | Medium |
| 149 | The volcano's y axis is continuous again | 0% | 9 | Medium |
| 140 | A long fit says it will be long, and where it has got to | 0% | 10 | Medium |
| 146 | Delete a run from the Runs tab | 0% | 11 | Low |
| 145 | One reader, one writer, one key vocabulary | 0% | 12 | Very high |
| 122 | Regress across screens | 0% | 13 | High |
| 115 | Every regression diagnostic saved and scored | 0% | 14 | Medium |
| 116 | Click a search row to spawn that run's graphs | 40% | 15 | Low |
| 125 | The old volcano, the Runs tab, Qt by default | 0% | 16 | Medium |
| 142 | Force restart, for when stop does not stop | 0% | 17 | Medium |
| 137 | Drag images in, and the regex is worked out for you | 0% | 18 | High |
| 118 | Figure preferences: general, and per graph type | 0% | 19 | Medium |
| 126 | The theme must not lag while a run is going | 0% | 20 | Medium |
| 114 | A parameter search that cannot take the machine down | 0% | 21 | Medium |
| 60 | Every module at 100% test coverage | ON HOLD | — | Very high |

**60 is ON HOLD as a goal**, set 2026-08-18: "it should not be worked towards as
a goal in and of itself, but tests should still be developed by agents building
code and modifying code." So every change still ships with a test that would
fail without it; nobody hunts uncovered lines.

## Fixed in passing today

* `dc09b890` -- HEAD imported `spacr.qt.widgets.database_set`, which was never
  committed, so a fresh checkout raised ModuleNotFoundError on the whole
  settings model. The working tree was fine, which is why it went unnoticed:
  pytest reads the tree and not HEAD.
* `482eae6e` -- `CellMontageView` never called `retarget_field_tooltips`, so it
  was the last screen putting hover help on an editable field (instruction 113).
* `aac29a84` -- pip's editable pointer targeted the stale mirror, so `spacr` on
  PATH launched code fifteen commits old and any check run as
  `python /elsewhere/script.py` verified the mirror. Repointed; trap recorded as
  HANDOFF 3f.
* `f0374448` -- the mixed model took patsy's kept rows by position rather than
  by index.

## Known, not yet filed

* `tests/test_documentation_i18n.py` — 7 failures, all one cause: a stale
  Korean review file for the `alpha` tooltip. Pre-existing, not from this work.
