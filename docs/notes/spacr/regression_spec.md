# Notes from `spacr/regression_spec.py`

Prose lifted out of `spacr/regression_spec.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Module level

### lines 52-66

```python
'spline',
```

THE ONE NONPARAMETRIC FAMILY THAT ANSWERS IN THIS CURRENCY.

Instruction 254 sorts seven methods by what they can honestly report. `spline` belongs here: it fits OLS on a design whose COVARIATES carry a spline basis while every guide column is left alone, so one coefficient and one P value per guide survive and the volcano, the hit list and the attribution all read it unchanged.

`isotonic` IS NOT HERE, and that is the same rule applied honestly. It needs an ORDERED SINGLE PREDICTOR and the guide design is unordered categories, so offering it in this menu would be the "method in the wrong category" that instruction says is worse than not offering it a user picking it would get a coefficient table nobody should read. It is available against a covariate through `spacr.nonparametric_fits.isotonic_fit`, which is where it is true.

### lines 137-138  _(unsure)_

```python
'spline': ('cov_type', 'spline_knots', 'spline_degree'),
```

`spline` fits OLS on a design whose COVARIATES carry a spline basis and whose guide columns are untouched, so it reads what ols reads.

### lines 181-184

```python
'alpha': 1.0,
```

'auto' and None mean "no penalty chosen, cross-validate it", which is not a value an unpenalised model is being asked to honour, so they count as the default rather than as a request. Handled at the call site, which is the only place that knows `alpha` was spelled that way.

### lines 191-194

```python
'spline_knots': 4,
```

Spline OLS changes only the nuisance-covariate design. These defaults must live in the same table as every other policed estimator setting so a re-fit from spline to another family resets them instead of carrying settings the new family cannot read.

### lines 197-203

```python
'group_lasso_lambda': 'auto',
```

Instruction 133's two new backends. Their defaults are the ones `spacr.group_lasso` and `spacr.rra` document for themselves, so a panel that posts the untouched widget posts the value the module would have used anyway and no other backend is refused because of it. 'auto' means "cross-validate it", the same way `alpha`'s 'auto' does, and it is what the panel posts. A number here would make the posted default a request every other backend then had to refuse.

### lines 240-242

```python
DEFAULT_REGRESSION_BACKEND = 'statsmodels'
```

WHO fits it (instruction 141). `regression_type` says WHAT is fitted.

### lines 306-307  _(unsure)_

```python
'package': 'torch',
```

torch is already a hard dependency -- spacr.power_model fits with it -- so this backend adds no package, only a device requirement.

### lines 326-329

```python
'pip': 'pip install pymer4  (plus R, rpy2, lme4)',
```

THE PIP LINE STAYS SHORT because `backend_status` puts it inside a combo entry; what it does not say -- that the package alone is not enough -- is in `cost` below and in the box, which is on screen whether or not the popup is open.

### lines 336-346

```python
'cost': ("Needs R, rpy2 and lme4 installed alongside it: measured "
```

IT STILL NEEDS R, and the earlier note here saying otherwise was read off a metadata gap. pymer4 0.9.2's wheel declares NO dependencies at all -- `importlib.metadata.requires('pymer4')` returns None -- so `pip install --dry-run --report` truthfully said "adds one package, changes nothing" and that was mistaken for "needs no R". Installed and imported on 2026-08-18 it fails at `pymer4/io.py: import polars`, and every model module under it opens with `from rpy2.robjects.packages import importr`; its own README says "This is accomplished using rpy2 to interface between languages". So the maintainer's question -- can lme4 be had without installing or interfacing with R -- is answered NO on this version.

### lines 364-374

```python
'cost': ("It has NO mixed model, so it does not touch that "
```

IT IS NOT ADDITIVE, MEASURED 2026-08-18 and this is the one that matters: `pip install --dry-run --report cuml-cu12` against this environment moves NUMPY 1.26.4 -> 2.2.6, downgrades numba 0.62.1 -> 0.61.2 and llvmlite 0.45.1 -> 0.44.0, and moves eight nvidia-cu12 runtime libraries torch is built against. The dependency table at the top of instruction 141 tested pymer4, gpytorch, numpyro, glum, linearmodels and pyfixest -- cuML was never among them, and "all six are purely additive" was read as covering it. The maintainer's condition was "first test if adding any of those dependencies causes any problems"; for this one the answer is yes, so it is not installed here.

### lines 396-400

```python
'cost': ("Measured on this machine: 1.4x at n=1830/p=736 (the "
```

MEASURED 2026-08-18 on synthetic screens of the shape prepare_formula builds, against sm.OLS on the identical design. See `spacr.ml._fit_absorbed_least_squares` for the table and for why the win is the SOLVER rather than the 5% narrower design pyfixest's own `feols` was measured too and is slower here.

### lines 418-422

```python
'types': ('glm', 'poisson', 'logit'),
```

NOT probit AND NOT quasi_binomial, measured rather than dropped: glum 3.4 ships identity, log, logit, cloglog and Tweedie links and has NO probit, and it has no equivalent of statsmodels' scale='X2', which is the free dispersion that IS quasi-binomial. See `spacr.ml._GLUM_FAMILIES`.
