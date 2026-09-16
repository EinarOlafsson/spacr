# Notes from `spacr/trial_metrics.py`

Prose lifted out of `spacr/trial_metrics.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [residual_diagnostics](#residual_diagnostics) (4 entries)
- [control_recovery](#control_recovery) (7 entries)
- [calibration](#calibration) (1 entry)
- [design_diagnostics](#design_diagnostics) (4 entries)
- [guide_support_summary](#guide_support_summary) (1 entry)
- [qc_verdicts](#qc_verdicts) (4 entries)
- [summarise_trial](#summarise_trial) (3 entries)
- [Module level](#module-level) (7 entries)

## residual_diagnostics

### line 129, trailing  _(unsure)_

```python
except Exception:
```

statsmodels shape varies

### lines 143-145

```python
if exog.shape[1] <= 30:
```

White's test squares every column pair; on a wide screen design that is thousands of terms and minutes of work, so it is only worth attempting on a narrow one.

### line 148, trailing  _(unsure)_

```python
except Exception:
```

singular or too wide

### lines 151-157

```python
if fitted.size == good.size and residuals.size > 2:
```

AGAINST `good`, NOT AGAINST THE MASKED RESIDUALS. `residuals` was already narrowed to its finite entries above, so comparing its length to the FULL fitted vector asks the wrong question twice: a fit with a single non-finite residual silently loses its trend slope, and a fit whose fitted values are a different length entirely reaches `fitted[good]` with a mask longer than the array and raises IndexError -- which every caller swallows, costing the whole residual block rather than this one statistic.

## control_recovery

### lines 205-206  _(unsure)_

```python
frame = frame[~frame[feature].astype(str).str.lower()
```

Rank among real coefficients: the intercept is not a candidate hit and counting it shifts every rank by one.

### lines 221-224

```python
out["n_ranked"] = n_ranked
```

The size of the list the ranks are against. Emitted only once a control has actually been asked for, because a run that named no control must get no control columns at all -- an unexplained count sitting in the table invites someone to read it as a result.

### lines 228-229  _(unsure)_

```python
hit = frame[frame[condition].astype(str).str.strip().str.lower()
```

spaCR's own annotation first. It was computed by the fit itself from these very settings, so it agrees with the volcano.

### lines 233-234  _(unsure)_

```python
hit = frame[frame[feature].astype(str).str.contains(
```

No annotation on this table (a permutation run writes none), so fall back to the substring rule spacr.ml uses to build it.

### lines 240-242

```python
best = hit.iloc[0]
```

`frame` is sorted by p, and boolean masking preserves that order, so row 0 is the BEST-ranked coefficient belonging to this control. A control with several guides is recovered if any one of them is.

### line 247  _(unsure)_

```python
out[f"{label}_control_percentile"] = float(
```

0 is the top of the list, 1 the bottom. This is the sortable one.

### lines 258-259  _(unsure)_

```python
if out.get("positive_control_rank") and out.get("negative_control_rank"):
```

One number for "did the assay work": how far the positive control sits above the negative one in rank.

## calibration

### lines 283-285

```python
counts, _edges = np.histogram(p, bins=20, range=(0.0, 1.0))
```

A screen with signal has a spike in the first bin; a flat histogram with no spike means nothing was found, and a SLOPING one means the model is misspecified regardless of how many hits it reports.

## design_diagnostics

### lines 344-345

```python
try:
```

Not a statsmodels regression model; pay for it once rather than leaving the identifiability question unanswered.

### line 348, trailing  _(unsure)_

```python
except np.linalg.LinAlgError:
```

degenerate exog

### lines 367-385

```python
if out["design_identifiable"] and getattr(inner, "k_constant", 0):
```

VIF, EXACTLY, WITHOUT ONE EXTRA REGRESSION.

regression_diagnostics.variance_inflation_factors regresses each guide on every other guide -- one least-squares solve per column, 0.53 s for twenty-five of four hundred guides, and it truncates at `max_guides` so the largest VIF in a wide design is usually not even among the ones it looked at. For a model with an intercept the same quantity is already implied by the standard errors:

VIF_j = se_j^2 / sigma^2 * (n - 1) * var(x_j)

because se_j^2 = sigma^2 * [(X'X)^-1]_jj and VIF_j = [(X'X)^-1]_jj * S_jj. Checked against that reference implementation to 2e-15 relative error, over ALL columns, in 9 ms on a 1,213-parameter design.

Only when the design is full rank. On a rank-deficient one the standard errors come from a pseudo-inverse, VIF is not defined at all, and the number this identity produces would be meaningless -- so it is omitted rather than reported, and `design_identifiable` already says why.

### lines 400-403

```python
if 2 <= int(varying.sum()) <= _MAX_PREDICTORS_FOR_PAIRWISE:
```

How many predictor pairs are so alike the fit cannot separate them. The COUNT, not the table: regression_diagnostics.collinear_guide_pairs names the offenders but stops at `limit` pairs, so counting its rows would report the cap rather than the truth. 58 ms at 1,213 predictors.

## guide_support_summary

### line 447, trailing  _(unsure)_

```python
except Exception:
```

odd table

## qc_verdicts

### lines 510-515

```python
out: dict[str, Any] = {}
```

THE SCORERS STAY THE ONE JUDGEMENT, and only the key names are translated. They read plain dicts -- `wells`, `wells_per_parameter`, `design_rank`, `genomic_inflation` -- and the row already carries every one of those statistics under spaCR's own column names. Re-deriving the verdicts here with fresh rules would be a second opinion about the same numbers, which is what a sweep table can least afford.

### lines 517-520

```python
scored: list = []
```

THE OBJECTS, not their levels, because `worst_verdict` compares PanelVerdicts -- it asks each one `worse_than`, which a string cannot answer. The Qt side has a string version; importing it here would drag PySide6 into a module that is deliberately headless.

### lines 528-529  _(unsure)_

```python
"identifiable": not (row.get("non_identifiable_directions") or 0),
```

A design is identifiable when it has no null directions. The row counts them, which is the same fact the other way up.

### line 537, trailing

```python
except Exception:
```

one panel must not sink a row

## summarise_trial

### line 592, trailing

```python
except Exception:
```

a metric must not sink a trial

### lines 594-595

```python
try:
```

LAST, because it reads the statistics the blocks above just wrote the verdicts are a judgement ON the row, not another measurement.

### line 598, trailing  _(unsure)_

```python
except Exception:
```

see above

## Module level

### line 610

```python
"qc_design", "qc_inference", "qc_verdict",
```

Diagnostic verdicts are results, not reconstructed regression settings.

### line 612  _(unsure)_

```python
"n_results", "n_significant", "n_primary", "n_below_alpha",
```

hit counts

### line 615  _(unsure)_

```python
"n_rows_fitted", "n_wells", "n_guides", "n_cells", "n_parameters",
```

design size and identifiability

### line 621  _(unsure)_

```python
"r_squared", "r_squared_adj", "aic", "bic", "log_likelihood", "f_pvalue",
```

fit quality

### line 624  _(unsure)_

```python
"durbin_watson", "jarque_bera_p", "residual_skew", "residual_kurtosis",
```

residual behaviour

### line 633  _(unsure)_

```python
"genomic_inflation", "p_first_bin_excess", "n_tests",
```

calibration

### line 635  _(unsure)_

```python
"n_genes_tested", "n_gene_hits", "n_single_guide_hits",
```

guide support
