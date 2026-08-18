# spaCR — instruction progress

Regenerated whenever an item is checked off or the maintainer adds one.
Branch `nightly`. Last regenerated 2026-08-18 18:35.

## Finished

| # | Summary | % |
|---|---|---|
| **144** | **The model box is typeset, not dumped** | **100%** |
| **125** | **The old volcano, the Runs tab, Qt by default** | **100%** |
| 157 | The loaded mark moves and nothing reloads | 100% |
| 155 | The montage knows its own run, and says how it chose | 100% |
| 147 | Both fits visible; a square is a square; menu categories | 100% |
| 109 | Image UMAP and the Gate Editor take several databases | 100% |
| 148 | The x and y axes behave as expected, logged or not | 100% |
| 153 | The model summary survives being loaded from disk | 100% |
| 131 · 132 · 133 · 134 · 135 · 138 · 143 | earlier | 100% |

**144 closed.** 143 had already cut the box from 2438 to 892 characters, so the
content was settled and this was presentation. **125 closed by AUDIT, not by
code** -- the file was wrong about itself, which is the ninth time this week, and
the agent was told to audit before building for exactly that reason.

## Unfinished

| # | Summary | % |
|---|---|---|
| 116 | Click a search row; every run has its own volcano | 99% |
| 149 | The volcano's y axis is continuous again | 98% |
| 152 | Two colours -- lines and font -- following the theme | 97% |
| 154 | The measurements tab is a workflow | 95% |
| 151 | Changing a line width should not take a minute | 95% |
| 127 | Modules that overlap or are redundant | 95% |
| 156 | Every regression mode gets a summary | 90% |
| 146 | Delete a run from the Runs tab | 90% |
| 140 | A long fit says it will be long, and where it has got to | 85% |
| 118 | Figure preferences: general, and per graph type | 85% |
| 115 | Every regression diagnostic saved and scored | 85% |
| 150 | A saved figure is for paper, not for the screen | 85% |
| 62 | The setting animations show what the setting does | 85% |
| 136 | The old matplotlib figures get the house style | 75% |
| 141 | The user picks the backend and is told what each one is | 70% |
| 108 | Right-click ANY figure to restyle it | 55% |
| 139 | Every regression graph in pyqtgraph | 45% |
| 145 | One reader, one writer, one key vocabulary | 0% |
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

## Owed, and last

* THE API RATCHET, once, at the end. It moves while agents run.
* `tools/build_documentation_i18n.py --sources-only`.
