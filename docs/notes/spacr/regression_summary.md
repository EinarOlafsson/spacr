# Notes from `spacr/regression_summary.py`

Prose lifted out of `spacr/regression_summary.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [Module level](#module-level) (3 entries)
- [_Run](#_run) (1 entry)
- [_fitted_section](#_fitted_section) (4 entries)
- [_formula](#_formula) (1 entry)
- [_equal_variance](#_equal_variance) (1 entry)
- [_multicollinearity](#_multicollinearity) (1 entry)
- [_excluded_section](#_excluded_section) (4 entries)
- [build_run_summary](#build_run_summary) (1 entry)
- [_recommendations](#_recommendations) (2 entries)
- [format_run_summary](#format_run_summary) (5 entries)
- [model_identity_line._Holder](#model_identity_line_holder) (1 entry)
- [_hyperparameter_report](#_hyperparameter_report) (2 entries)
- [write_run_summary](#write_run_summary) (1 entry)

## Module level

### lines 185-188

```python
("excluded", "min_cell_count"): "min_cells_per_well",
```

THE FIELD KEY STAYS AND THE LABEL FOLLOWS THE SETTING. The key is this summary's own vocabulary and is read back by its tests; the label is what a user sees, and it must name the setting they typed `min_cells_per_well` since 364 renamed it.

### lines 218-220

```python
"min_cell_count": "min_cell_count",
```

THE VALUE IS A FIELD NAME, NOT A SETTING NAME, and changing it named a field that does not exist. The label a user reads followed 364's rename; this mapping is the summary's internal wiring and did not.

### lines 660-662

```python
UNIDENTIFIABLE_WARNING = (
```

The identifiability warning, which stays at the top

## _Run

### lines 547-549  _(unsure)_

```python
@dataclass
```

What kind of run is this

## _fitted_section

### lines 739-742

```python
add("regression_type",
```

SAID, because the settings still carry one and a reader will otherwise believe it. The permutation path never reaches `regression_model`: it is a marginal test, so no family is fitted and the setting has no effect on the numbers below.

### lines 756-759

```python
if run.nonparametric:
```

Report only hyperparameters the selected regression family reads. The shared family table also drives the disabled-setting rules and tooltip text, keeping the summary aligned with the interface and preventing an ignored value (such as alpha for OLS) from being presented as fitted.

### lines 798-800

```python
add("backend", value=label if label.startswith(backend) else
```

NOT "statsmodels (statsmodels (CPU))". The spec's label already CONTAINS the backend's name -- it is "pyfixest (CPU)", not "(CPU)" so bracketing it after the name said everything twice.

### lines 827-829

```python
from .ml import BETA_SQUEEZE_NOTE
```

NAMED HERE, not left in the run log. The logit squeeze moves a well sitting at exactly 0 or 1, and a reader comparing the fitted response to their own column has to be told that happened.

## _formula

### lines 880-883

```python
response = (_raw(getattr(inner, "endog_names", None))
```

THE RESPONSE THE ESTIMATOR SAW, not the one the user typed. `transform='log'` fits `log_pred`, and a formula naming `pred` describes a fit nobody ran -- the statsmodels block below this one says `Dep. Variable: log_pred` in the same file.

## _equal_variance

### lines 1442-1444

```python
parts, rejected = [], False
```

ONE VERDICT PER TEST. Reporting the smaller of the two under a single "REJECTED" is how a run where Breusch-Pagan says 0.645 and White says 0.045 comes to read as though both agreed.

## _multicollinearity

### lines 1618-1625

```python
scaled = None
```

THE BANDS ARE FOR THE SCALED NUMBER, AND ONLY FOR IT. `model.condition_number` is what statsmodels prints, and it is UNSCALED: it is dominated by the units of the columns, so a predictor measured in cells rather than thousands of cells moves it by 1000 with no change in the science. Belsley-Kuh-Welsch's 30 / 100 / 1000 apply to the column-scaled one. Reading the bands off the unscaled number reported "severe collinearity" for a full-rank design with a max VIF of 1.38 on the first real run of this module.

## _excluded_section

### lines 1991-1995

```python
requested = _setting(settings, "exclude_grnas")
```

A KNOWN CONTAMINANT MUST LEAVE BEFORE THE FRACTION DENOMINATOR. Merely echoing the setting cannot establish that it matched anything, and a misspelling that removed zero rows is exactly the failure this audit is meant to expose. ``process_reads`` records both the resolved guide names and unmatched requests at the raw-count boundary, before well totals.

### lines 2035-2040

```python
dropped = _exclusion_count(settings, "fraction_threshold")
```

RECORDED SINCE 2026-08-19. `ml.process_reads` takes a `record=` dict and accumulates what it dropped, per plate, into `settings['_regression_exclusions']` -- so this is a number now rather than an admission. The admission is kept for the runs that predate the recorder, because "0 removed" and "nobody counted" are opposite findings and must not be spelled the same way.

### lines 2048-2049

```python
retained = outof - dropped
```

Report both counts and percentage because the percentage makes severe filtering directly comparable across datasets.

### lines 2067-2068  _(unsure)_

```python
paired = _exclusion_count(settings, "wells_paired")
```

Pairing counts are recorded at the score/count join and persist in the settings saved with the run.

## build_run_summary

### lines 2247-2250

```python
present = {one.name for one in section.fields}
```

THE BACKFILL IS THE CONTRACT'S LAST LINE OF DEFENCE. A builder that returns early, or a field added to CONTRACT and not yet to its builder, would otherwise ship a summary with a silent hole -- which is exactly the failure mode this item is about.

## _recommendations

### lines 2286-2288

```python
return []
```

A permutation run assumes none of this, and the sections above say so five times over. Recommending a fix for an assumption that was never made is how the section loses the reader.

### lines 2298-2300

```python
return []
```

A summary is worth more than its last section: a run that reached here has numbers worth reading, and losing them to a failure in the advice would be the wrong trade.

## format_run_summary

### line 2446

```python
top = headline(summary)
```

THE ANSWER FIRST, and every line of it quoted verbatim from below.

### lines 2465-2466

```python
names = ", ".join(one.label for one in deferred)
```

ONE LINE WHERE NINE STOOD, naming them, with every word of the explanation still in the file under its own heading.

### lines 2478-2483

```python
grouped: "OrderedDict[str, List[str]]" = OrderedDict()
```

ONE EXPLANATION PER REASON, NOT PER FIELD. Measured on the maintainer's own run: eleven deferred fields carry SIX distinct explanations, two of them printed three times each -- six paragraphs where two would do, and that repetition is most of what "not very accessable" was about. The fields sharing a reason are named together and the reason is given once.

### lines 2492-2494

```python
lines.extend(textwrap.wrap(joined + ":", width=_WIDTH - 2,
```

A JOINED LABEL LONGER THAN THE COLUMN gets its own line, or the explanation is squeezed into whatever is left and comes out one word wide.

### lines 2512-2518

```python
try:
```

LAST, BECAUSE IT IS WHAT TO DO NEXT. Everything above says what was found; this says what to change, and a reader who stops early has still read the findings.

It is printed even when empty: an absent section reads as a bug, and "every check passed" is a result worth stating rather than implying by silence.

## model_identity_line._Holder

### line 2587, trailing  _(unsure)_

```python
class _Holder:
```

what _hyperparameter_report reads

## _hyperparameter_report

### lines 2621-2622

```python
wanted = [name for name in wanted if name != "cov_type"]
```

`cov_type` is not a hyperparameter of the fit -- it is how the standard errors are computed afterwards -- and it has its own line already.

### lines 2627-2630

```python
chosen = getattr(getattr(run, "model", None), "alpha_", None)
```

THE MODEL KNOWS. `_find_best_alpha` returns the fitted RidgeCV / LassoCV / ElasticNetCV itself, and those carry the alpha they chose as `alpha_` -- so the value that won is on the object the run already holds, and does not need recording separately.

## write_run_summary

### lines 2689-2692

```python
try:
```

THE RUN'S OWN STATSMODELS TEXT, RECOVERED RATHER THAN LOST. It was written into this path minutes ago by `save_summary_to_file`, and a model that cannot render a second time (or was not handed over) would otherwise silently drop it on the way past.
