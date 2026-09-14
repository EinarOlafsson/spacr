# Notes from `spacr/fraction_calibration.py`

Prose lifted out of `spacr/fraction_calibration.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [well_fractions](#well_fractions) (1 entry)
- [sweep_fraction_threshold](#sweep_fraction_threshold) (3 entries)
- [compare_normalisations](#compare_normalisations) (1 entry)

## well_fractions

### lines 92-93

```python
frame = frame[totals > 0].copy()
```

A well with no reads at all has no shares to compute; it drops out here rather than becoming a column of NaN that reads as a measurement.

## sweep_fraction_threshold

### lines 252-256

```python
row["corrected_slope"] = (row["slope"]
```

AN AFFINE MAP OF THE IMAGING SIDE IS AN AFFINE MAP OF THE

LINE. p_true = (p_obs - (1 - sp)) / (se + sp - 1), so the slope divides by the same denominator and nothing has to be refitted. The correction MOVES the estimate; it does not widen it, and it inflates the variance by the square.

### lines 262-267

```python
wide_enough = [row for row in rows
```

A CANDIDATE IS CHOSEN ON ITS DISAGREEMENT, so one whose disagreement could not be measured cannot be chosen -- and must not reach the comparison below either. NaN loses every ``<=`` it appears in, so a sweep whose fits reported no per-well pairs left ``min()`` an empty iterable and raised ValueError out of a function whose whole contract is to answer "no" in words.

### lines 298-300

```python
best = min(row["median_absolute_disagreement"] for row in usable)
```

THE SMALLEST THRESHOLD THAT MAKES THE TWO MEASUREMENTS AGREE. Once the spurious barcodes are gone a higher threshold changes nothing except how much real data it discards, so ties go to the least destructive answer.

## compare_normalisations

### lines 372-375

```python
if all(value is not None and np.isfinite(value)
```

FINITE, not merely present. A fit that reported no per-well pairs carries a NaN disagreement, and NaN loses every comparison ``min`` makes naming whichever definition happened to be first, as though the two had been measured and one had won.
