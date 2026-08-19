# spaCR — instruction progress

Regenerated whenever an item is checked off or the maintainer adds one.
Branch `nightly`. Last regenerated 2026-08-19 16:05.

## Finished

| # | Summary | % |
|---|---|---|
| **176** | **A panel cannot write an impossible number** | **100%** |
| **177** | **One parse per input file** | **100%** |
| **160** | **Two regressions hung the machine** | **100%** |
| 144 | The model box is typeset, not dumped | 100% |
| 125 | The old volcano, the Runs tab, Qt by default | 100% |
| 157 | The loaded mark moves and nothing reloads | 100% |
| 155 | The montage knows its own run, and says how it chose | 100% |
| 147 | Both fits visible; a square is a square; menu categories | 100% |
| 109 | Image UMAP and the Gate Editor take several databases | 100% |
| 148 | The x and y axes behave as expected, logged or not | 100% |
| 153 | The model summary survives being loaded from disk | 100% |
| 131 · 132 · 133 · 134 · 135 · 138 · 143 | earlier | 100% |

**160 closed.** Reproduced, diagnosed and fixed -- and the cause was NOT the
memory this file spent its length on. `plt.show()` on the JobRunner worker
entered a Qt event loop off the GUI thread and aborted the process, which is
why no traceback ever reached the log. Found by running the maintainer's own
four-plate screen headlessly under `faulthandler.dump_traceback_later`. The
sequence this file was filed about now completes in one process at a peak of
2.6 GB out of 125 GB, with per-stage RSS and GPU written into each run folder.

144 closed earlier: 143 had already cut the box from 2438 to 892 characters, so
the content was settled and this was presentation. 125 closed by AUDIT, not by
code -- the file was wrong about itself.

## Unfinished

| # | Summary | % |
|---|---|---|
| **170** | **Cells tab: two modes + annotator settings** | **75% — PRIORITISED** |
| **175** | **Every gene against every measurement** | **80%** |
| **173** | **A guide and a probability for every cell** | **75%** |
| **172** | **How many cells, and which ones** | **90%** |
| **171** | **One name for loading, one for streaming** | **60%** |
| 167 | The montage uses the scores the run already has | 100% ✓ |
| **168** | **The run summary is readable at a glance** | **0%** |
| **169** | **The figures container resizes and collapses** | **60%** |
| **174** | **Beta as a response transform** | **0%** |
| 116 | Click a search row; every run has its own volcano | 99% |
| 149 | The volcano's y axis is continuous again | 99% |
| 152 | Two colours -- lines and font -- following the theme | 99% |
| 154 | The measurements tab is a workflow | 97% |
| 151 | Changing a line width should not take a minute | 95% |
| 127 | Modules that overlap or are redundant | 95% |
| 156 | Every regression mode gets a summary | 95% |
| 146 | Delete a run from the Runs tab | 90% |
| 140 | A long fit says it will be long, and where it has got to | 90% |
| 118 | Figure preferences: general, and per graph type | 85% |
| 115 | Every regression diagnostic saved and scored | 85% |
| 150 | A saved figure is for paper, not for the screen | 90% |
| 62 | The setting animations show what the setting does | 85% |
| 136 | The old matplotlib figures get the house style | 75% |
| 141 | The user picks the backend and is told what each one is | 90% |
| 108 | Right-click ANY figure to restyle it | 55% |
| 139 | Every regression graph in pyqtgraph | 45% |
| 145 | One reader, one writer, one key vocabulary | 25% |
| 122 | Regress across screens | 0% |
| 142 | Force restart, for when stop does not stop | 0% |
| 137 | Drag images in, and the regex is worked out for you | 0% |
| 126 | The theme must not lag while a run is going | 0% |
| 114 | A parameter search that cannot take the machine down | 0% |
| 60 | Every module at 100% test coverage | ON HOLD |

## 141 went DOWN in scope, and the file was wrong twice

Both errors were in this repository's own instruction file and both were the
same mistake -- a dependency table read as covering more than it tested.

* **cuML is not a safe install.** The table tested six packages and cuML was
  never among them. Re-measured: `cuml-cu12` moves numpy 1.26.4 -> 2.2.6,
  downgrades numba and llvmlite, and moves eight nvidia runtime libraries torch
  is built against. Refused. The penalised families therefore have no GPU path.
* **pymer4 0.9.2 still needs R.** The 0.9.2 wheel declares NO dependencies at
  all, so the dry-run reported one additive package and none of its real ones.
  Installed, it fails on `import polars` and every model module imports rpy2.
  The maintainer's question is answered NO for this version.

WIRED AND PROVEN: pyfixest for ols/wls (1.4x -> 5.9x -> 16.7x as the screen
grows; coefficients agreeing to 3.9e-9) and glum for glm/poisson/logit (2.44x
and 3.08x at n=6000, and 0.70x -- SLOWER -- on a small screen, which the box
says). glum's own `std_errors()` would have shipped every Poisson standard error
1.2934x too large; the agreement test caught it on the first backend it was
applied to, which is the argument for having it.

## 2026-08-19 -- the module driven on the maintainer's own screen

Four 500 MB measurement databases, four score CSVs and four count CSVs, run
headlessly. Seven bugs, none of which synthetic fixtures had ever shown:

* **A count CSV names its plate by WHICH FILE IT IS.** The real tables carry
  `row_name, column_name, grna_name, count` and no plate column, so
  `fractions_from_counts` pooled four plates' `r1/c1` into one well -- 384
  wells instead of 1536, and the fractions still summed to 1, so nothing
  downstream could notice. The same absence broke regression on measurement
  columns outright: 0 of 3 columns fitted. (145, 154)
* **Crop paths do not survive a move.** 0 of 60,816 recorded `png_path`s
  existed; 60,816 of 60,816 existed once rebuilt under the plate folder the
  database was opened from. Now `spacr.portable_paths`, which rewrites only
  onto a file that exists.
* **The default inference returned no `res_folder`,** so every nonparametric
  run was registered by the GUI with `folder=''` -- the reported "No summary:
  this panel was opened from a results table on disk". (156)
* `horseshoe` was refused in Poisson's name; the mixed model's cost is now
  stated before it is spent. (140)

AND THE ONE THAT WAS NOT A BUG. Under `inference='nonparametric'` -- the
default since 2026-08-18 -- the permutation path fits no model, so
`regression_type` is never read. Verified: `ols` and `mixed` produce
BYTE-IDENTICAL results, 1612 rows across all 24 columns. "i ran a mixed model
and an ols model and even if the ols model is marked as loaded i think i still
see the mixed results" was a correct observation, not a display bug. It is now
said before the run rather than only in the summary afterwards.

141 to 90%: all four implemented backends exercised on the real screen.
statsmodels, torch, glum and pyfixest all fit it; pyfixest correctly REFUSES
when `model_plate_position=False`, because there is then nothing to absorb.
`mixed` took 24s on torch against >32 minutes on statsmodels, same data.

## Owed, and last

* THE API RATCHET, once, at the end. It moves while agents run.
* `tools/build_documentation_i18n.py --sources-only`.
