# Notes from `spacr/figures/distributions.py`

Prose lifted out of `spacr/figures/distributions.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [gini](#gini) (1 entry)
- [relative_representation](#relative_representation) (1 entry)
- [shape_of](#shape_of) (2 entries)
- [_ratio_ticks](#_ratio_ticks) (3 entries)
- [guide_fraction](#guide_fraction) (3 entries)
- [response](#response) (2 entries)

## gini

### lines 117-119  _(unsure)_

```python
def gini(values) -> float:
```

The statistics, separately so they can be checked without drawing

## relative_representation

### lines 172-180

```python
usable_share = np.where(np.isfinite(shares), shares, np.nan)
```

THE WELL IS COUNTED OVER ITS USABLE SHARES, NOT OVER ITS ROWS. `size` counts a row whose share is NaN or inf; `sum` does not. A well holding one unknown share therefore divided its total by one guide too many, so every OTHER guide in that well came out over-represented -- silently, because the unknown row is dropped and the dropped-row count still looks right. Worse, a well holding one real guide and one unknown passed the two-guide guard and landed at exactly 2.0x: an artefact of pure arithmetic sitting on the "at least twice equal" cut the panel reports, which is the same failure single-guide wells are excluded to avoid.

## shape_of

### line 217, trailing  _(unsure)_

```python
kurtosis = float(stats.kurtosis(array))
```

already excess

### lines 225-227

```python
verdict = "symmetric, heavy-tailed"
```

Symmetric and heavy-tailed is a real and different failure: a normal fitted to it has the right centre and the wrong tail probabilities, which is precisely what a p-value is computed from.

## _ratio_ticks

### lines 286-289

```python
exponents = [e for e in exponents if e % step == 0]
```

THINNED ON e ≡ 0, never by slicing from the low end. Plain `[::step]` starts at whatever the smallest guide happened to be, and on this screen that dropped the tick at 1 -- leaving the reference line standing over an unlabelled position, on the one axis where 1 is the entire point.

### lines 292-297

```python
labels = [f"{p:g}" if p >= 1 else f"{p:.3g}" for p in powers]
```

SIGNIFICANT DIGITS, NOT FOUR DECIMAL PLACES. A fixed `.4f` is only honest down to 2^-6: 2^-9 printed as "0.002" (2.4% out), 2^-12 as "0.0002" (18% out), and 2^-15 rounded to "0.0000", which `.rstrip("0")` then turned into the label "0." -- twice over on a wide enough axis, so two different ticks read the same. Reachable from the raw view of any deeply sequenced library, where a guide's share of a well is 1e-4.

### lines 301-302

```python
ax.xaxis.set_minor_locator(NullLocator())
```

The minor ticks of a log axis are the between-decade marks; on a base-2 axis they land on top of the majors and thicken every tick.

## guide_fraction

### lines 361-364

```python
low, high = float(values.min()), float(values.max())
```

A library where every retained guide holds exactly the same share is degenerate but not impossible (a run that kept one guide per well and a caller who asked for the raw view). geomspace over a zero-width range returns identical edges, which matplotlib bins into nothing.

### lines 380-386

```python
note = (f"n = {values.size:,} guides\nGini = {evenness:.2f}\n"
```

UPPER LEFT, because the reference line stands at 1 with its own rotated label and the distribution's right shoulder is under the upper right. The left is the thin end of a log-ratio histogram and is always free.

The last two lines are only true of the relative view: on the raw axis there is no "equal" for a guide to be twice of, so saying "≥ 2× equal" there would attach a meaning the panel did not measure.

### lines 406-409

```python
built = (f"each as its raw share of its well, on a log2 axis. Wells "
```

No well column: the shares of a 2-guide and a 15-guide well are pooled, so the spread below is evenness AND how many guides landed per well together. Said plainly rather than left for the reader to discover, because the same picture means much less here.

## response

### lines 453-455

```python
normal = bool(values.std(ddof=1) > 0 and family == "gaussian")
```

The reference is the family that was FITTED. Drawing a normal over a Poisson or a beta fit would put a curve on the panel that no part of the model ever assumed, and a reader would take the mismatch for a finding.

### lines 464-467

```python
before = ""
```

A transform is only worth having if it did something, and the pipeline names its transformed response after the raw one. When both are here, say what the transform bought -- it is the one number that tells the maintainer whether to keep it.
