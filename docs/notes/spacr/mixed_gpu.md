# Notes from `spacr/mixed_gpu.py`

Prose lifted out of `spacr/mixed_gpu.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [Module level](#module-level) (1 entry)
- [_codes](#_codes) (1 entry)
- [TorchMixedResults](#torchmixedresults) (1 entry)
- [fit_mixed_reml_torch](#fit_mixed_reml_torch) (9 entries)
- [fit_mixed_reml_torch._solve](#fit_mixed_reml_torch_solve) (2 entries)
- [mixedlm_torch](#mixedlm_torch) (1 entry)

## Module level

### lines 98-101

```python
from .regression_backends import cuda_present_without_importing_torch  # noqa: E402,F401
```

ONE PROBE, NOT TWO. The cheap "is there a driver" question is answered in :mod:`spacr.regression_backends`, which a settings panel may import (it touches nothing heavier than stdlib). Re-exported here so a caller holding this module does not have to know that.

## _codes

### lines 244-246  _(unsure)_

```python
def _codes(labels: Sequence) -> tuple:
```

The design: one integer code per row per random-effects term

## TorchMixedResults

### lines 305-307

```python
@dataclass
```

The result, shaped like MixedLMResults on purpose

## fit_mixed_reml_torch

### line 518, trailing  _(unsure)_

```python
dtype = torch.float64
```

a variance ratio spans decades; float32 loses it

### lines 522-536

```python
_refuse_if_too_large(n, q, dtype=dtype, device=torch_device)
```

THE CROSS-PRODUCTS, FORMED ONCE. Everything the deviance needs is a function of these and of theta, so `n` leaves the optimiser's inner loop entirely and the q x q Cholesky becomes the whole per-iteration cost. That is the operation measured at 204 ms CPU / 7.69 ms GPU. WHAT THIS WILL COST, BEFORE ASKING FOR IT. `Z` is DENSE and n x q, and the shape is known exactly here -- so the bytes are known exactly here. Reported 2026-08-18: running an OLS and then a mixed fit hung the whole machine twice, badly enough to need a restart. An allocation that asks the operating system for more than it has does not fail politely; it takes the session, and everything else the user had open, with it.

So the fit says the number and refuses. A refusal a user can read beats a machine they have to power-cycle, and the alternatives are real ones: `regression_type='ols'` does not build this matrix at all, and fitting at well level rather than cell level is usually what was meant.

### lines 557-558  _(unsure)_

```python
rank = int(np.linalg.matrix_rank(X_values))
```

A rank-deficient fixed part has no identified coefficients and MixedLM reports it three frames deep as a bare LinAlgError. Caught here, named.

### lines 632-640

```python
started = time.perf_counter()
```

RESTARTED UNTIL THE DEVIANCE STOPS MOVING, not run once. L-BFGS stops on its own line-search tolerance, and a single call leaves the variance components differing from statsmodels in the 4th significant figure measured 1.0e-4 relative on the nested fixture. A warm restart from the stopping point is what separates "the line search gave up" from "the gradient is flat"; three of them take the disagreement to 1e-7 and cost about a third of the fit. The loop exits on the deviance, not on a fixed count, so a hard problem gets the passes it needs and an easy one does not pay for them.

### lines 662-664

```python
S_inv = torch.cholesky_inverse(RX)
```

cov(beta) = sigma^2 (X' W^-1 X)^-1, and S IS X' W^-1 X -- the same matrix already factorised as RX, so the standard errors come from the fit rather than from a second, possibly different, solve.

### lines 667-668  _(unsure)_

```python
u = torch.linalg.solve_triangular(
```

u = L^-T (cu - RZX beta), and the random effect on the response scale is Lambda u.

### lines 672-676

```python
Zb_np = (Z_dense @ torch.as_tensor(
```

CONDITIONAL, not marginal. MixedLMResults.fittedvalues adds each group's random effects to X.beta, so `resid` is the conditional residual -- and spacr.ml.fit_mixed_model plots exactly that. A marginal residual here would have looked right and been a histogram of the random effects.

### lines 695-696  _(unsure)_

```python
from scipy import stats as _stats
```

z, not t: MixedLMResults carries use_t=False, so matching it is what makes the p-values in results.csv comparable across backends.

### lines 700-701

```python
random_effects = {}
```

THE BLUPS, keyed and named exactly as MixedLM keys and names them, so `spacr.ml._blup_guide_name` parses this backend's output unchanged.

## fit_mixed_reml_torch._solve

### line 590, trailing  _(unsure)_

```python
lam = torch.sqrt(theta)[expand]
```

q

### line 601  _(unsure)_

```python
pwrss = (yty - (cu * cu).sum()) - (beta * rhs).sum()
```

r^2 = (y - Xb)' W^-1 (y - Xb), through the same factorisation.

## mixedlm_torch

### lines 789-796

```python
kept = X_design.index
```

THE ROWS PATSY KEPT, taken by INDEX and not by position. patsy drops a row whose predictor is NaN, so `groups[:len(X_design)]` would take the first n labels rather than the surviving ones and shift every remaining row into the wrong cluster from the first dropped row onwards. Nothing about the result would look wrong -- the fit completes, the standard errors are simply computed against the wrong grouping. spacr.ml. regression() takes weights, groups and exposure through the same index for the same reason.
