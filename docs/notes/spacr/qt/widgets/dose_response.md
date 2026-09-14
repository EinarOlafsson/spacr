# Notes from `spacr/qt/widgets/dose_response.py`

Prose lifted out of `spacr/qt/widgets/dose_response.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [DoseResponseResult.ec50_fold_uncertainty](#doseresponseresultec50_fold_uncertainty) (1 entry)
- [_profile_bound](#_profile_bound) (1 entry)
- [fit_dose_response](#fit_dose_response) (2 entries)
- [_kinds](#_kinds) (1 entry)
- [selectivity_index._ratio](#selectivity_index_ratio) (1 entry)
- [_effect_curve.effect](#_effect_curveeffect) (1 entry)
- [loewe_surface](#loewe_surface) (1 entry)
- [Module level](#module-level) (2 entries)
- [pool_across_plates](#pool_across_plates) (2 entries)

## DoseResponseResult.ec50_fold_uncertainty

### lines 748-756

```python
return None
```

A bound from `fit_dose_response` IS positive -- both ends are back-transformed out of log space. But this dataclass is public, frozen and validates nothing, so one `dataclasses.replace` away is a bound of zero, and the alternative to declining is a division that yields `inf` and a panel reporting "within a factor of inf".

`<= 0` rather than `== 0`: a negative bound would otherwise take the square root of a negative number.

## _profile_bound

### lines 1499-1505

```python
return None
```

THE WALK NEVER ARRIVED. Sixty doublings covers `step * 2**59`, which is not the same as "any finite limit" -- a step small enough against a large enough limit exhausts the loop, and an infinite limit exhausts it outright.

Either way the answer is the same one the limit case gives:

this experiment does not bound the EC50 on that side.

## fit_dose_response

### lines 1696-1703

```python
reach = (log_min - PROFILE_REACH, log_max + PROFILE_REACH)
```

One reach for both methods. The profile stops walking at

PROFILE_REACH decades past the tested range and calls that side open; the Wald formula has no such stopping rule and will happily return 10 ** 400 as an upper bound on a parameter the data does not identify. Applying the same limit to both is what makes the two methods comparable: past a factor of 10**PROFILE_REACH beyond the highest dose tested, "the interval ends here" and "the interval does not close" are the same statement about the experiment.

### lines 1731-1733

```python
ec50_low = ec50_high = None
```

An interval around a midpoint the data does not locate is a picture of the model, not of the experiment. It is dropped with the point estimate rather than drawn.

## _kinds

### lines 1816-1818  _(unsure)_

```python
def _kinds(frame: pd.DataFrame) -> Mapping[str, str]:
```

Column suggestions — the only seam that reaches into the Qt tree

## selectivity_index._ratio

### lines 2004-2007

```python
def _ratio(numerator, denominator):
```

ONE-SIDED RATHER THAN DROPPED. Interval arithmetic on whichever ends survive: the smallest possible index divides the host's lower bound by the parasite's upper one, and vice versa. Conservative, and it is the only form available when a fit is open on one side.

## _effect_curve.effect

### lines 2168-2174

```python
affected = (1.0 - fraction) if hill < 0 else fraction
```

THE HILL SIGN CARRIES THE DIRECTION, and the two cases are not symmetric. A negative Hill is inhibition: the response FALLS with dose, so at a high dose `fraction` approaches 0 while the affected fraction approaches 1, and the affected fraction is its complement. A positive Hill is activation, where `fraction` already IS the affected fraction and inverting it would report every activator as its own antagonist.

## loewe_surface

### lines 2374-2378

```python
index = np.where((grid_a > 0) | (grid_b > 0), index, np.nan)
```

THE UNTREATED WELL IS NOT INFINITELY SYNERGISTIC. With both doses at zero the index is 0 and the excess reads +1.0, the strongest possible synergy, from the one well where nothing was combined. Measured on a simulated board it was the maximum of the whole surface. Loewe is undefined without a combination, so that cell is NaN.

## Module level

### lines 2408-2410  _(unsure)_

```python
NORMALISE_NONE = "none"
```

The plate: normalisation to its controls, and the Z' it already has

### lines 2804-2806  _(unsure)_

```python
MAX_HETEROGENEITY = 0.9
```

Replicate plates: one EC50, with plate as a random effect

## pool_across_plates

### line 2986  _(unsure)_

```python
fixed_w = 1.0 / variances
```

Stage one: fixed-effect weights, only to measure the disagreement.

### line 2993

```python
c = float(np.sum(fixed_w) - np.sum(fixed_w ** 2) / np.sum(fixed_w))
```

DerSimonian--Laird: the spread the plates' own uncertainty cannot explain.
