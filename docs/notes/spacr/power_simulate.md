# Notes from `spacr/power_simulate.py`

Prose lifted out of `spacr/power_simulate.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [_check_positive](#_check_positive) (1 entry)
- [rbeta_mean_variance](#rbeta_mean_variance) (1 entry)
- [rdirichlet_stable](#rdirichlet_stable) (1 entry)
- [sample_count_mean_variance](#sample_count_mean_variance) (2 entries)
- [simulate_spot_plate](#simulate_spot_plate) (2 entries)
- [_require_columns](#_require_columns) (1 entry)
- [simulate_imaging_plate](#simulate_imaging_plate) (5 entries)
- [simulate_sequencing_plate](#simulate_sequencing_plate) (4 entries)
- [simulate_screen](#simulate_screen) (2 entries)

## _check_positive

### lines 341-343

```python
def _check_positive(name: str, value: float) -> float:
```

Distribution reparameterisations (mean/variance instead of shape/rate)

## rbeta_mean_variance

### lines 532-533

```python
return np.full(n, mean, dtype=float)
```

Degenerate limit: a point mass at the mean. Kept out of the beta call because the shape parameters diverge.

## rdirichlet_stable

### lines 622-623

```python
return weights / weights.sum()
```

exp() of the normalised logs sums to 1 only to within rounding; divide again so callers can assert `sum == 1` rather than `isclose`.

## sample_count_mean_variance

### lines 701-706

```python
n_trials = int(round(mean ** 2 / (mean - var)))
```

Under-dispersed: binomial. n_trials = mean^2 / (mean - var) is generally not an integer. Round it, then recover p from the rounded n as `mean/n` rather than using the closed form `(mean - var)/mean`: that keeps the mean exact and pushes the whole rounding error into the variance. With the naive p the mean is short by up to half a count in *every* well, which a sweep over the variance turns into a drifting baseline that looks like signal.

### lines 708-709

```python
n_trials = max(n_trials, int(np.ceil(mean)), 1)
```

p <= 1 requires n_trials >= mean; the closed form satisfies that for any var >= 0, but rounding down can just cross it when var is tiny.

## simulate_spot_plate

### lines 793-795  _(unsure)_

```python
def simulate_spot_plate(
```

Stage 2 — the spot plate

### line 890  _(unsure)_

```python
gene_index = np.repeat(np.arange(n_genes), n_wells_per_screen)
```

expand_grid(gene, well): gene-major, well varying fastest.

## _require_columns

### lines 932-934  _(unsure)_

```python
def _require_columns(frame: pd.DataFrame, columns, label: str) -> None:
```

Plate-frame plumbing shared by stages 3 and 4

## simulate_imaging_plate

### lines 1051-1053  _(unsure)_

```python
def simulate_imaging_plate(
```

Stage 3 — the imaging plate

### lines 1166-1167

```python
well_totals = sample_count_mean_variance(
```

One well total per well, drawn before the split so that the number of imaged cells does not depend on how many genes happened to land there.

### lines 1180-1181  _(unsure)_

```python
continue
```

R has this branch too: a well with no genes spotted into it still exists, it just has nothing to attribute its cells to.

### lines 1190-1193

```python
prob_pos = rbeta_mean_variance(
```

Both classifier probabilities are drawn for every row and then selected between, exactly as R's `ifelse(hit, rbeta(...), rbeta(...))` does: the per-row draw is what makes the classifier's operating point vary between observations rather than being one number for the whole screen.

### lines 1206-1210

```python
frame['imaging_n_cells_per_well_var'] = (
```

Echo the variance the count model actually used, not the literal argument: `None` means Poisson, and a Poisson's variance is its mean. Writing NaN here instead would put a NaN column into the frame the model half consumes, where "this parameter was left at its default" is indistinguishable from "this number failed to compute".

## simulate_sequencing_plate

### lines 1432-1434  _(unsure)_

```python
def simulate_sequencing_plate(
```

Stage 4 — the sequencing plate

### lines 1582-1584

```python
if read_depth_cv == 0.0 or n_reads_per_well == 0.0:
```

Per-well sequencing depth. cv == 0 pins every well to the target exactly, which is what makes a "no depth variation" baseline reproducible against an arbitrary generator state.

### lines 1614-1616

```python
draw = int(min(urn, well_depth[well_column]))
```

min() against the urn, not against round(sum(cells) * pcr): the two differ because the urn is rounded element-wise, and asking for more balls than the urn holds is a hard numpy error.

### lines 1624-1625

```python
frame['sequencing_n_cells_per_well_var'] = (
```

As in the imaging stage: echo the variance the count model used, so the column is never NaN. `None` means Poisson, whose variance is its mean.

## simulate_screen

### lines 1776-1779

```python
screen = spot_plate.merge(
```

validate='1:1' rather than a positional concat: dplyr::do() returns groups in sorted key order, not input order, so any positional assumption carried over from the R code would misalign genes without changing a single summary statistic.

### lines 1786-1789

```python
screen = drop_low_cell_wells(screen, min_cells_per_well)
```

Last, and after the join, because it is a decision about *wells* taken on the realised imaged cell total -- which is only known once the imaging plate exists, and which the sequencing plate knows nothing about. A well removed here takes its sequencing rows with it.
