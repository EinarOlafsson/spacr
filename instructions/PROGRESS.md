# spaCR — instruction progress

Regenerated whenever an item is checked off or the maintainer adds one.
Branch `nightly`. Last regenerated 2026-08-18 17:30.

TWO tables, every time. Finished, with the one just closed marked; then
unfinished with the percentage each is at. An item a background agent has built
but NOT committed is not 100%.

## Finished

| # | Summary | % |
|---|---|---|
| **147** | **Both fits visible; a square is a square; the menu has categories** | **100%** |
| **109** | **Image UMAP and the Gate Editor take several databases** | **100%** |
| 148 | The x and y axes behave as expected, logged or not | 100% |
| 153 | The model summary survives being loaded from disk | 100% |
| 131 | Show the cells behind a dot on the volcano | 100% |
| 132 | Model & Inference states the formula it fits | 100% |
| 133 | Every model explained; Toxoplasma annotation on every export | 100% |
| 134 | `analysis_mode` is a dropdown, not a text box | 100% |
| 135 | The regression settings are one page a user can read | 100% |
| 138 | The explainer text fills the width of its box | 100% |
| 143 | Plate position is opt in; the model box is 2438 -> 892 chars | 100% |

**147 closed** on its last open half: the plot now says the OTHER level's fit
exists and keeps saying it after the first click, which is what produced "it
only runs once". B (a square is the canvas, reaching the exported file) and C
(the menu categories, landed by migrating the tests FIRST so they passed either
side of the restructure) closed earlier in the wave.

**109 closed** on its two handoffs: a dropped folder now JOINS the working set
instead of evicting it, and the empty-state banner stopped deciding "is a source
set?" by `isinstance(src, QLineEdit)` -- which had started showing "point this
at some data" over a screen with four databases loaded.

## Unfinished

| # | Summary | % | Difficulty |
|---|---|---|---|
| 154 | The measurements tab is a workflow (A-F done, G in flight) | 95% | Very high |
| 151 | Changing a line width should not take a minute | 95% | Low |
| 127 | Modules that overlap or are redundant | 95% | Medium |
| 149 | The volcano's y axis is continuous again | 90% | Medium |
| 152 | Two colours -- lines and font -- following the theme | 90% | Medium |
| 141 | The user picks the backend and is told what each one is | 85% | High |
| 62 | The setting animations show what the setting does | 85% | High |
| 136 | The old matplotlib figures get the house style | 75% | Medium |
| 150 | A saved figure is for paper, not for the screen | 65% | Medium |
| 139 | Every regression graph in pyqtgraph (C done, A at 15%) | 65% | High |
| 116 | Click a search row; every run has its own volcano | 50% | Medium |
| 108 | Right-click ANY figure to restyle it | 30% | High |
| 155 | The montage knows its own run, and says how it chose | 10% | High |
| 157 | The loaded mark moves and nothing reloads | 0% | Low |
| 156 | Every regression mode gets a summary | 0% | Medium |
| 144 | The model box is typeset, not dumped | 0% | Medium |
| 140 | A long fit says it will be long, and where it has got to | 0% | Medium |
| 146 | Delete a run from the Runs tab | 0% | Low |
| 145 | One reader, one writer, one key vocabulary | 0% | Very high |
| 122 | Regress across screens | 0% | High |
| 115 | Every regression diagnostic saved and scored | 0% | Medium |
| 125 | The old volcano, the Runs tab, Qt by default | 0% | Medium |
| 118 | Figure preferences: general, and per graph type | 0% | Medium |
| 142 | Force restart, for when stop does not stop | 0% | Medium |
| 137 | Drag images in, and the regex is worked out for you | 0% | High |
| 126 | The theme must not lag while a run is going | 0% | Medium |
| 114 | A parameter search that cannot take the machine down | 0% | Medium |
| 60 | Every module at 100% test coverage | ON HOLD | Very high |

In flight right now (wave 4): 157, 155, 156, 141 G.

## Owed, and deliberately last

* THE API RATCHET. 7185 reviewed, 7281 measured at 17:20, 7290 nine minutes
  later -- it moves while agents run, so it is a once-per-batch bump and it is
  the LAST action before the final push. The 96 additions are already
  enumerated by set difference against dad9195d; the patch and the method are
  in the session scratchpad.
* `tools/build_documentation_i18n.py --sources-only`, the same debt: until it
  runs, the localized API pages omit every contract added today.
