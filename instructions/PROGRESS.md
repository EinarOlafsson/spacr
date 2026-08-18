# spaCR — instruction progress

Regenerated whenever an item is finished. Branch `nightly`.

Priority is the order to work them in. Difficulty is what a "left" item costs.

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
| 128 | The regression panel: defaults, colouring, the grid, more tabs | 95% | High |
| 129 | Every plot in pyqtgraph, a tab each | 95% | High |
| 121 | Click a gene and everything known about it appears | 90% | Medium |
| 136 | The old matplotlib figures get the house style | 75% | Medium |

## Left

| # | Summary | % | Priority | Difficulty |
|---|---|---|---|---|
| 148 | A log axis moves the dots, and the limits hold | 20% | 1 | Medium |
| 147 | Both fits visible; a square is a square; the menu has categories | 0% | 2 | Medium |
| 144 | The model box is typeset, not dumped | 0% | 3 | Medium |
| 139 | Every regression graph in pyqtgraph; saved graphs you cannot see | 0% | 4 | High |
| 140 | A long fit says it will be long, and where it has got to | 0% | 5 | Medium |
| 146 | Delete a run from the Runs tab | 0% | 6 | Low |
| 141 | The user picks the backend and is told what each one is | 0% | 7 | High |
| 145 | One reader, one writer, one key vocabulary | 0% | 8 | Very high |
| 122 | Regress across screens | 0% | 9 | High |
| 115 | Every regression diagnostic saved and scored | 0% | 10 | Medium |
| 116 | Click a search row to spawn that run's graphs | 40% | 11 | Low |
| 125 | The old volcano, the Runs tab, Qt by default | 0% | 12 | Medium |
| 142 | Force restart, for when stop does not stop | 0% | 13 | Medium |
| 137 | Drag images in, and the regex is worked out for you | 0% | 14 | High |
| 108 | Right-click ANY figure to restyle it | 10% | 15 | High |
| 118 | Figure preferences: general, and per graph type | 0% | 16 | Medium |
| 126 | The theme must not lag while a run is going | 0% | 17 | Medium |
| 62 | The setting animations show what the setting does | 30% | 18 | High |
| 114 | A parameter search that cannot take the machine down | 0% | 19 | Medium |
| 109 | Image UMAP and the Gate Editor take several databases | 60% | 20 | Low |
| 127 | Modules that overlap or are redundant | 40% | 21 | Medium |
| 60 | Every module at 100% test coverage | ongoing | 22 | Very high |

## Known, not yet filed

* `tests/test_documentation_i18n.py` — 7 failures, all one cause: a stale
  Korean review file for the `alpha` tooltip. Pre-existing, not from this work.
