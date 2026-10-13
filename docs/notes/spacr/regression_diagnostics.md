# Notes from `spacr/regression_diagnostics.py`

Prose lifted out of `spacr/regression_diagnostics.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [Module level](#module-level) (6 entries)
- [design_report](#design_report) (3 entries)
- [collinear_guide_pairs](#collinear_guide_pairs) (1 entry)
- [variance_inflation_factors](#variance_inflation_factors) (1 entry)
- [residual_report](#residual_report) (3 entries)
- [_verdict](#_verdict) (1 entry)
- [plot_design_diagnostics](#plot_design_diagnostics) (5 entries)
- [plot_residual_diagnostics](#plot_residual_diagnostics) (1 entry)
- [plot_inference_diagnostics](#plot_inference_diagnostics) (4 entries)
- [write_diagnostic_suite](#write_diagnostic_suite) (2 entries)
- [write_diagnostic_suite._emit](#write_diagnostic_suite_emit) (1 entry)

## Module level

### lines 39-48

```python
from .figures.style import ROLES, TYPE_SCALE, WEIGHTS, figure_style
```

THE HOUSE PALETTE, BY ROLE. This module drew in seaborn's `deep` -- nine hardcoded hexes with no rule behind which one meant what -- so a run wrote its design and inference panels in a third visual idiom beside the seven house-style panels and the nineteen QC ones.

Mapped rather than renamed: `#4C72B0` was doing the job of "the data" in one panel and "the highlight" in another, so a find-and-replace would have kept the inconsistency and only changed the hues. See `.claude/skills/apicomplexan-figures`: everything is grey except what the sentence is about.

### line 52, trailing  _(unsure)_

```python
_DATA = ROLES["data"]
```

bars, clouds, anything not the claim

### line 53, trailing

```python
_BAD = ROLES["down"]
```

a failed check, a threshold crossed

### line 54, trailing  _(unsure)_

```python
_GOOD = ROLES["up"]
```

a passed check

### line 55, trailing  _(unsure)_

```python
_MARK = ROLES["highlight"]
```

the one series a panel is about

### line 56, trailing  _(unsure)_

```python
_REFERENCE = ROLES["reference"]
```

thresholds and guides

## design_report

### lines 113-115

```python
if isinstance(fractions, pd.DataFrame) and isinstance(block, pd.Series):
```

A named Series normally arrives indexed by well.  Align it rather than trusting incidental row order; the fallback by position keeps the plain-array API working for callers without labels.

### lines 139-142

```python
design = np.column_stack([
```

Rank the matrix whose parameter count is reported.  Previously the report added block terms to ``parameters`` but omitted those columns from ``design``; every multi-plate run could therefore be declared non-identifiable even when the complete design was full rank.

### lines 169-170

```python
"identifiable": bool(rank >= parameters and residual_df > 0),
```

The single verdict. Rank deficiency is not a warning: a coefficient in a rank-deficient fit is one of infinitely many solutions.

## collinear_guide_pairs

### lines 222-224

```python
return pd.DataFrame(columns=columns)
```

A well-designed screen legitimately has no collinear pair. Building the frame from an empty list gives it no columns at all, so sorting raised KeyError('correlation') on exactly the healthy case.

## variance_inflation_factors

### lines 264-266

```python
raise ValueError(
```

The engine would answer `inf` for every guide here, which is true and useless. A rank-deficient design has no VIF to report, and saying so is what sends the caller to the panel that can describe it.

## residual_report

### lines 311-313

```python
if n < 5000:
```

Shapiro-Wilk is exact but degrades above a few thousand points, where D'Agostino's K^2 is the right test -- the same rule the manuscript's own statistics section uses.

### line 324  _(unsure)_

```python
if np.std(yhat) > 0:
```

Breusch-Pagan against the fitted values: is the spread constant?

### line 351  _(unsure)_

```python
report["high_influence_points"] = int(np.sum(cooks > 4.0 / max(n, 1)))
```

4/n is the usual screening rule for "look at this point".

## _verdict

### lines 356-374

```python
def _verdict(level, headline, detail="", score=None, statistic=""):
```

verdicts

EVERY DIAGNOSTIC REACHES A VERDICT, INCLUDING THESE THREE.

`spacr.regression_qc` scores each of its twenty-three panels and stamps the judgement on the panel, so a reader is told whether a number is fine instead of being expected to know. These three sheets did not, and they are the ones a permutation run gets INSTEAD of that suite -- so on the analysis mode where there is no fitted design matrix, nothing in the whole output said whether the design was usable. That is the reading this section exists to prevent.

THE VOCABULARY IS BORROWED, NOT REBUILT. `PanelVerdict`, the four levels and the badge come from `regression_qc`; a second set of words for the same four states is how a run comes to say CHECK on one page and WARN on another about the same fit.

A SHEET GETS ONE VERDICT because a sheet answers one question -- is this design usable, are these residuals behaved, is this null calibrated -- and its panels are the evidence for that one answer.

## plot_design_diagnostics

### lines 700-704

```python
with figure_style():
```

THE STYLE HAS TO BE ON BEFORE THE FIGURE EXISTS. rcParams colour an artist when it is CREATED, so a context opened after plt.subplots would leave the spines, ticks and text at whatever the caller's global style happened to be.

### line 728

```python
axis = axes[0, 2]
```

Identifiability, stated rather than implied.

### line 754  _(unsure)_

```python
axis = axes[1, 0]
```

Cumulative singular-value spectrum: where the rank runs out.

### line 786  _(unsure)_

```python
axis = axes[1, 2]
```

Occupancy map: which wells hold which guides, sorted so structure shows.

### lines 796-799

```python
verdict = score_design(report)
```

SCORED AFTER IT DREW AND STAMPED BEFORE IT IS WRITTEN, exactly as the QC suite does it: the verdict is read off the report the panels were drawn from, so it cannot disagree with the numbers printed beside it.

## plot_residual_diagnostics

### lines 825-829

```python
with figure_style():
```

THE STYLE HAS TO BE ON BEFORE THE FIGURE EXISTS. rcParams colour an artist when it is CREATED, so a context opened after plt.subplots would leave the spines, ticks and text at whatever the caller's global style happened to be.

## plot_inference_diagnostics

### lines 935-939

```python
with figure_style():
```

THE STYLE HAS TO BE ON BEFORE THE FIGURE EXISTS. rcParams colour an artist when it is CREATED, so a context opened after plt.subplots would leave the spines, ticks and text at whatever the caller's global style happened to be.

### line 956  _(unsure)_

```python
axis = axes[1]
```

QQ against the uniform null, on -log10 so the tail is readable.

### line 968  _(unsure)_

```python
inflation = float("nan")
```

Genomic inflation: median observed chi-square over its null median.

### lines 1006-1010

```python
"genomic_inflation": inflation,
```

ON THE REPORT, NOT ONLY ON THE PICTURE. lambda was computed here to colour one annotation and then thrown away, so the one number that says whether the whole family of p-values can be believed was readable by eye and by nothing else -- not by the summary CSV, not by a sweep row, not by a verdict.

## write_diagnostic_suite

### lines 1064-1066

```python
requested = [None] if formats is None else [str(f) for f in formats]
```

None is the sentinel for "the preference decides"; it has to survive as far as `save_figure`, so it is a one-element list rather than a resolved format string.

### lines 1127-1137

```python
levels = [report.get("verdict_level") for report in reports.values()
```

THE SUITE'S VERDICT IS ITS WORST SHEET, and it goes in the SUMMARY rather than in the returned mapping. That mapping's contract is "key -> a file that exists", and a caller iterating it to check its own output is entitled to that; a verdict string in it is a path that is not there. Per-sheet verdicts are already rows here, because each report carries `verdict`, `verdict_level` and `verdict_detail`.

A design that cannot identify its own coefficients beside two clean sheets is not "two out of three": it is a run whose numbers are one of infinitely many answers, and a summary that averages that away loses exactly the thing worth reporting.

## write_diagnostic_suite._emit

### lines 1088-1091

```python
suffix = (os.path.splitext(str(_written))[1].lstrip(".").lower()
```

KEYED BY WHAT WAS WRITTEN, not by what was asked for. The extension `save_figure` chose is the only one that names a file that exists, and a manifest entry pointing at a file that is not there is worse than no entry.
