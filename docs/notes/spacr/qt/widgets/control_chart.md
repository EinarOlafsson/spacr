# Notes from `spacr/qt/widgets/control_chart.py`

Prose lifted out of `spacr/qt/widgets/control_chart.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [Module level](#module-level) (1 entry)
- [c4](#c4) (2 entries)
- [sd_reference_limits](#sd_reference_limits) (1 entry)
- [ControlChartResult.zone](#controlchartresultzone) (1 entry)
- [control_chart](#control_chart) (2 entries)
- [zprime_frame](#zprime_frame) (1 entry)

## Module level

### lines 222-224  _(unsure)_

```python
D2_MOVING_RANGE = 1.128
```

The constants, and where they come from

## c4

### line 304

```python
return 1.0 - 0.75 / size
```

Γ(171) overflows a float; c4 is within 1e-3 of 1 long before then.

### lines 306-311

```python
from scipy.special import gamma as _gamma_fn
```

Imported here, not at the top: this is the module's only remaining use of scipy, and it is reached when a subgroup chart is actually computed. `math.lgamma` would remove the dependency altogether and is NOT used — it disagrees with this ratio in the last one or two bits, and a published constant that changes in its fifteenth digit because a launch got faster is not a trade worth making.

## sd_reference_limits

### lines 928-930  _(unsure)_

```python
def sd_reference_limits(values: Sequence[float]
```

The reference nobody should use as limits

## ControlChartResult.zone

### lines 1072-1078

```python
return 3
```

``value - centre`` overflows to infinity when a plate sits at the far end of the float range from the centre line — rare, but a raw intensity column reaches it. Such a point is further outside the limits than any finite one, so the outermost band is the honest answer: ``int(inf)`` raises, and calling it zone 0 would paint the worst plate of the campaign the colour of one that never left one sigma, while rule 1 flags it in the same breath.

## control_chart

### lines 1727-1731

```python
notes.append(
```

Not a refusal — a chart of a campaign that is still short is worth drawing. But Phase I and Phase II being the same plates means the limits and the points they judge are the same data, and a rule firing there is a statement about the estimate rather than a test of anything, so it is said rather than left to be noticed.

### line 1795  _(unsure)_

```python
offending = {p for v in result.baseline_violations for p in v.points}
```

One pass, not iterations to convergence — see the module docstring.

## zprime_frame

### lines 1909-1911

```python
record[ZPRIME_ORDER] = index
```

The run order is already resolved, so the chart of this frame must not have to guess it again: a plain 0..k-1 index is the one order column that cannot be mis-sorted.
