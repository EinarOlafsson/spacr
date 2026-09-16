# Notes from `spacr/power_model.py`

Prose lifted out of `spacr/power_model.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [prepare_model_data](#prepare_model_data) (8 entries)
- [_module_installed](#_module_installed) (1 entry)
- [_prepare_design](#_prepare_design) (1 entry)
- [_fit_torch_advi](#_fit_torch_advi) (9 entries)
- [_fit_torch_advi._log_joint](#_fit_torch_advi_log_joint) (6 entries)
- [fit_model](#fit_model) (1 entry)
- [gather_model_estimate](#gather_model_estimate) (1 entry)
- [evaluate_model_fit](#evaluate_model_fit) (1 entry)
- [_call_simulator](#_call_simulator) (1 entry)
- [scan_parameters](#scan_parameters) (6 entries)
- [scan_parameters._report](#scan_parameters_report) (1 entry)

## prepare_model_data

### lines 367-369

```python
try:
```

Sort so two runs on the same screen give the same column order and therefore the same beta ordering; pd.unique preserves row order, which depends on how the simulator happened to emit rows.

### line 373, trailing

```python
except TypeError:
```

mixed types that do not compare -- keep first-seen order

### lines 390-393

```python
frame["imaging_n_cells_per_well"] = frame.groupby("well")[
```

This one is a property of the WELL, not of the (well, gene) pair. Zero-filling it would make the well's own total disagree with itself, and the consistency check below would then reject a table that we ourselves corrupted. Fill from the well's observed rows.

### line 397  _(unsure)_

```python
frame = frame.fillna(0)
```

Everything else really is a per-pair count, and "missing" means "none".

### lines 436-438

```python
spread = np.nanmax(well_total_matrix, axis=1) - np.nanmin(
```

Constant within a well by definition; if it is not, the table has been joined wrongly and picking row 0 (which is what the R does) would quietly pick one arbitrary value.

### line 457  _(unsure)_

```python
keep = ntotal > 0
```

drop wells with no imaged cells (R: filter(Ntotal > 0))

### lines 484-489

```python
zero_read_mask = total_reads <= 0
```

log10 read fraction

A well with zero reads gives 0/0. The fraction of a well's reads that belong to a gene, when the well produced no reads, is 0 for every gene not NaN. Such a well still has a valid Npositive/Ntotal and so still informs the intercept; it just carries no gene-level contrast, which is exactly what an all-equal covariate row expresses.

### line 507  _(unsure)_

```python
column_spread = log10expression.max(axis=0) - log10expression.min(axis=0)
```

genes with no contrast are not estimable

## _module_installed

### lines 554-555  _(unsure)_

```python
return False
```

A namespace package shadow or a partially uninstalled distribution can make find_spec itself raise. Treat that as "not usable".

## _prepare_design

### lines 761-763

```python
scales = np.where(scales > 0, scales, 1.0)
```

Guard the constant columns: their sd is 0 and they are about to be zeroed anyway, so a scale of 1 keeps the division finite without inventing a contrast.

## _fit_torch_advi

### line 817, trailing  _(unsure)_

```python
import torch
```

deferred: keeps `import spacr.power_model` off torch's ~4 s import

### line 821, trailing  _(unsure)_

```python
dtype = torch.float64
```

counts times exp() -- float32 loses the tail

### lines 832-836

```python
total_positive = float(model_data.Npositive.sum())
```

Locate the intercept prior at the empirical baseline log-rate. brms does the same thing (it centres the intercept prior on the data), and it matters: without it the optimiser spends its first few hundred steps walking the intercept from 0 down to about -5, and does so by inflating beta, which is exactly the parameter we are trying to keep at zero.

### line 842, trailing  _(unsure)_

```python
n_params = 2 * n_genes + 3
```

z, log_lambda, log_tau, log_c2, intercept

### line 849, trailing  _(unsure)_

```python
loc[2 * n_genes + 2] = mu0
```

intercept

### lines 851-853

```python
log_scale = torch.full(
```

Start the variational scales small: the first draws then sit at the initialisation, which is a sane point, instead of scattered across a region where exp(eta) overflows.

### lines 943-947

```python
torch.nn.utils.clip_grad_norm_([loc, log_scale], max_norm=_GRAD_CLIP_NORM)
```

Clip before stepping. A single outsized gradient -- routine early on, when the intercept is still wrong and exp(eta) is enormous -- would otherwise throw the parameters somewhere exp() overflows, and every subsequent step is NaN. Clipping bounds the step length without moving the optimum.

### lines 954-957

```python
window = max(10, int(0.125 * len(elbo_history)))
```

Convergence for an optimiser is "it stopped improving". Compare the mean ELBO over the last eighth of the run with the eighth before it; a single-point comparison would be dominated by Monte-Carlo noise in the ELBO estimate.

### lines 994-996

```python
beta_scale = "per standard deviation of log10expression"
```

beta was fit per SD of the covariate; report it that way and say so, rather than dividing back and leaving a number whose shrinkage was applied on a scale it is no longer expressed in.

## _fit_torch_advi._log_joint

### line 879  _(unsure)_

```python
log_lambda_tilde = log_lambda - 0.5 * F.softplus(
```

beta, bounded by the slab -- see the docstring.

### line 885, trailing  _(unsure)_

```python
eta = intercept + beta @ X.T + log_offset
```

(..., n_wells)

### lines 886-887

```python
log_lik = (y * eta - torch.exp(eta)).sum(-1)
```

Poisson log-pmf without the constant lgamma(y+1) term, which does not depend on the parameters and so cannot change the optimum.

### lines 891-892  _(unsure)_

```python
log_prior = log_prior + (
```

half-StudentT(df_local, 1) on lambda, plus the log-Jacobian of lambda = exp(log_lambda), which is log_lambda itself.

### line 898  _(unsure)_

```python
log_prior = log_prior + (
```

half-StudentT(df_global, tau0) on tau, same Jacobian trick.

### lines 910-911

```python
log_prior = log_prior + (
```

Student-t(3, mu0, 2.5) on the intercept -- brms's default shape for an intercept, wide enough to be uninformative on the log-rate scale.

## fit_model

### line 1362, trailing  _(unsure)_

```python
else:
```

pymc -- resolve_backend has already rejected anything else

## gather_model_estimate

### lines 1445-1446  _(unsure)_

```python
warnings.simplefilter("ignore", category=RuntimeWarning)
```

All-NaN columns are the unidentified genes and are expected; numpy's "Mean of empty slice" is noise here, not news.

## evaluate_model_fit

### lines 1620-1622

```python
model_auroc = float(roc_auc_score(y_true, y_score))
```

Orientation: y_score = posterior mean of beta, higher = more hit-like. See the docstring for why this is the R's `-mean` scored against event level "no".

## _call_simulator

### line 1813, trailing  _(unsure)_

```python
except (TypeError, ValueError):
```

builtins and C callables have no signature

## scan_parameters

### line 1956, trailing  _(unsure)_

```python
fit_options.pop("backend", None)
```

the sweep's backend wins; one method per sweep

### lines 2010-2012

```python
for column in ("run_key", "backend", "method", "status",
```

Round-tripping through TSV turns empty strings into NaN, which would then compare unequal to the "" that a fresh row carries and would print as 'nan' in the error column of a perfectly fine run.

### lines 2066-2068

```python
point_seed = int(
```

Seed derived from the point's identity, not from iteration order, so re-running a single point reproduces the screen it produced inside the full sweep.

### lines 2136-2139

```python
row["status"] = "not_converged"
```

Metrics stay NaN. A non-converged fit's coefficient ordering is not a posterior ordering, and scoring it would put a number on the plot that the fit does not support.

### lines 2185-2186  _(unsure)_

```python
pd.DataFrame([row], columns=result_columns).to_csv(
```

Append one complete row at a time and close the handle, so a kill -9 between points leaves a file whose last line is whole.

### lines 2194-2195

```python
if not _report(row, point_index, replicate, False):
```

After the progress file, so a cancelled sweep can still be resumed from the point it stopped at rather than redoing it.

## scan_parameters._report

### lines 2048-2050

```python
return verdict is not False
```

`is False`, not falsy: a callback that returns 0, "" or an empty list has almost certainly returned something incidental, and stopping a five-minute sweep on that is not a decision to infer.
