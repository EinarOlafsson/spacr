# Notes from `spacr/qt/widgets/trellis_spec.py`

Prose lifted out of `spacr/qt/widgets/trellis_spec.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [TrellisSpec.x](#trellisspecx) (1 entry)
- [trellis](#trellis) (2 entries)

## TrellisSpec.x

### line 163  _(unsure)_

```python
@property
```

the inner spec, reachable without reaching through

## trellis

### lines 680-681  _(unsure)_

```python
shared = _scales_for_group(data.frame, graph, kinds, seats)
```

Scales. Colour and size come from the whole grid, always — see the module docstring. Only the two positional axes take the mode.

### lines 704-707

```python
tops = [_panel_top(data.frame, graph, panel, kind, panel.scales.x_edges)
```

The count axis is the y axis, so it shares along the *y* groups — but each panel's bars are counted with its own x group's edges. Sharing the value axis of an aggregate is the same rule as sharing a data axis; forgetting it is the usual way a faceted histogram lies.
