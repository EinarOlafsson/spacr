# Notes from `spacr/guide_permutation.py`

Prose lifted out of `spacr/guide_permutation.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [Module level](#module-level) (1 entry)
- [adjusted_value_label](#adjusted_value_label) (1 entry)
- [guide_freedman_lane_test](#guide_freedman_lane_test) (2 entries)
- [plot_guide_permutation_volcano](#plot_guide_permutation_volcano) (6 entries)
- [analyse_long_gene_table](#analyse_long_gene_table) (1 entry)

## Module level

### lines 29-31

```python
from .figures.style import figure_style, theme_target
```

THE HOUSE STYLE (136). `figures.style` imports matplotlib only inside its own functions, so naming it here costs nothing at import time.

## adjusted_value_label

### lines 88-89

```python
return "adjusted P"
```

Every remaining method controls the family-wise error rate, which adjusts the P value rather than producing a q value.

## guide_freedman_lane_test

### lines 398-405

```python
from scipy.stats import rankdata
```

RANKED BEFORE RESIDUALISING, not after. Ranking the residual would rank a quantity the nuisance fit has already shaped; ranking the phenotype first makes the whole statistic a function of order, and the nuisance projection then removes the block from the RANKS, which is what a Spearman-type partial correlation is.

Average ranks for ties, which is what `scipy.stats.rankdata` gives and what every definition of Spearman assumes.

### lines 453-454  _(unsure)_

```python
permuted_outcomes = y_fitted[:, None] + permuted_residuals
```

Freedman--Lane: y* = nuisance fit + permuted reduced-model residual; then remove nuisance effects before evaluating the same statistic.

## plot_guide_permutation_volcano

### lines 711-714

```python
with figure_style(theme_target()):
```

THE STYLE HAS TO BE ON BEFORE THE FIGURE EXISTS:

rcParams reach an artist when it is CREATED, so a context opened after `plt.subplots` would leave the spines, ticks and labels at the caller's globals.

### lines 722-724

```python
edgecolor=_separator(),
```

178 A: the page, not white -- an outline that is brighter than the marker it surrounds is the reverse of what a separator is for, and on the dark theme white is exactly that.

### lines 747-748  _(unsure)_

```python
axis.axvline(
```

Both lines, one legend entry: they are one cut with two sides, and a legend that lists it twice reads as two different rules.

### lines 765-766

```python
if index % 2 == 0:
```

Alternate sides and vertical positions so nearby discoveries do not print on top of one another (as EAF1 g2 and GRA14 g3 otherwise do).

### lines 790-795

```python
from .figures.scene import write_figure
```

THROUGH THE SCENE RENDERER, so the file a run writes is the same picture the screen draws rather than a second rendering of the same numbers. `write_figure` announces it to the gallery itself, which is what `publish` was here for; it falls back to the matplotlib page, with the reason recorded, when an artist is outside the translation's whitelist.

### lines 803-806

```python
plt.close(fig)
```

`publish(close=True)` clears the figure but does not release it: this one comes from `plt.subplots`, so pyplot holds a reference until `plt.close`. A sweep over thresholds and outcomes calls this dozens of times.

## analyse_long_gene_table

### lines 1012-1015

```python
random_state=int(random_state) + index,
```

THE SAME NULL AS THE GUIDE PASS. Same seed, same outcome, same nuisance design, so the permuted residual vectors are the same vectors -- which is what makes the two tables comparable rather than merely similar.
