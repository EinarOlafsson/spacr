# Notes from `spacr/ops_register.py`

Prose lifted out of `spacr/ops_register.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [unwrap](#unwrap) (1 entry)
- [_surface_numpy](#_surface_numpy) (2 entries)
- [phase_correlate](#phase_correlate) (5 entries)
- [register_edge](#register_edge) (2 entries)

## unwrap

### lines 182-183  _(unsure)_

```python
base = value + extent * round((about - value) / extent)
```

The two candidates that bracket `about`, since any representative is `value + k * extent`.

## _surface_numpy

### lines 242-246

```python
with np.errstate(invalid="ignore", divide="ignore"):
```

THE NORMALISATION IS THE WHOLE METHOD. Dividing out the magnitude keeps only the phase, which is what makes the answer independent of how bright either field happens to be -- and a zero there is a frequency neither tile carries, not a division to be silently allowed to produce a nan.

### lines 249-255

```python
product[0, 0] = 0
```

THE DC BIN IS MASKED, AND IT IS NOT A DETAIL. Two strips that share no structure still share a BACKGROUND, and after phase normalisation that constant is the only component they agree on so the surface peaks at the origin and the pair reads as no-overlap. On the real plate that cost 27 of 624 TRUE adjacencies, all of them at the well's edge where the overlap band holds few nuclei. With the bin zeroed, every one of the 624 registers.

## phase_correlate

### lines 378-379

```python
LOG.debug("the %s backend could not register this pair", name,
```

A BACKEND THAT RAISES IS NOT A FAILED WELL. An out-of-memory on a shared card, a driver that went away: fall to the next.

### lines 396-412

```python
if expected == (0, 0):
```

THE DANGEROUS DEFAULT, SAID OUT LOUD. `unwrap`'s docstring is emphatic that `about=0` is right only when the true shift is under half the axis, and silent when it is not: a raster pitch of 1,267 px on a 1,480 px tile folds to -213, which is "both wrong and plausible" in that docstring's own words. It cost an afternoon on the first real acquisition -- every measurement taken at -213 said the tiles did not overlap, and the tiles overlap by 14.3%.

Warned rather than refused: a pair genuinely centred near zero is a legitimate call (two cycles of the SAME field), and only the caller knows which it has. The raw peak beyond half the axis is the signal. THE TEST IS THE SIZE OF THE ANSWER, not where the raw peak sat. With about=0 the result always lands in [-extent/2, extent/2), so a SUBSTANTIAL folded shift means the other representative is a plausible raster pitch and the caller has not said which they meant. A genuinely small shift -- two cycles of one field, a few pixels of drift -- is unambiguous and stays quiet.

### lines 425-434

```python
at_origin = (int(peak_y), int(peak_x)) == (0, 0)
```

(0, 0) IS WHAT TWO FIELDS THAT DO NOT TOUCH PRODUCE. Never accepted, however strong the peak: there is no such thing as two adjacent fields of a raster occupying the same place.

JUDGED ON THE RAW PEAK, NOT THE UNWRAPPED SHIFT, and this is not a detail. Once `expected` is non-zero the unwrap moves the origin to whichever representative is nearest the expectation -- a raw (0, 0) against a 222 px strip expecting 116 comes back as 222 -- so a test on the unwrapped pair would never fire again, and the no-overlap signature would silently start reading as a confident placement.

### lines 437-442

```python
rows, columns = ((tolerance, tolerance)
```

THE LAYOUT IS A STRONGER TEST THAN A THRESHOLD. The question is not "was that a confident peak" but "did this pair land where the raster says it should", and the second one is answerable because the layout predicts it. Judged this way on the real well, 624 of 624 edges were accepted and none was guessed; judged on a bare peak ratio, 152 real adjacencies were refused.

### lines 448-458

```python
else:
```

AND THE ORIGIN RULE IS DELIBERATELY NOT APPLIED HERE. A raw peak at (0, 0) unwraps to the representative nearest the expectation, so it is only accepted when that representative lands within tolerance of where the raster says the neighbour is -- which is the same question this branch already asks, answered better. Adding `not at_origin` on top would refuse a pair whose true strip-frame shift genuinely is zero, which is exactly what a caller gets by choosing an overlap fraction equal to the real overlap. Trading a rare aliasing case for a reachable false rejection is the wrong way round, and the DC mask has already removed the reason the origin was suspicious.

## register_edge

### lines 527-528

```python
guess = band - int(expected_overlap) if expected_overlap else 0
```

In the strip frame the expected shift is the strip width minus the overlap: the two bands are that far out of step with each other.

### lines 538-539

```python
lead = extent - band
```

Back to tile coordinates. The strip started `extent - band` into the first tile, so that much of the shift was cropped away.
