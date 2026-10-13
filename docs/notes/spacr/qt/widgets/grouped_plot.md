# Notes from `spacr/qt/widgets/grouped_plot.py`

Prose lifted out of `spacr/qt/widgets/grouped_plot.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [GroupedPlot._draw](#groupedplot_draw) (2 entries)
- [GroupedPlot._draw_xy](#groupedplot_draw_xy) (1 entry)
- [GroupedPlot._caption](#groupedplot_caption) (1 entry)

## GroupedPlot._draw

### lines 192-195

```python
if str(spec.kind) in ("scatter", "line") and spec.group \
```

A SCATTER IS NOT A GROUPED MARK. It is two continuous axes, and forcing it through `add_group_mark` would put every point at one categorical position -- a jitter under another name, which is exactly what `graph_types` refuses to offer.

### lines 210-213

```python
self.plot.getAxis("bottom").setTicks(
```

THE n IS ON THE AXIS, not only in the caption. A three-point group and a three-hundred-point group are the same bar, and the label is the only place a reader meets the difference before they have read the sentence underneath.

## GroupedPlot._draw_xy

### lines 256-259

```python
order = np.argsort(x)
```

SORTED BY x, because a line joins points in the order it is given and an unsorted series draws a scribble. `graph_types` only offers a line for an ordered x, and this is the other half of that promise.

## GroupedPlot._caption

### lines 294-296

```python
mark = MARKS.get(str(getattr(spec, "kind", "") or ""), "jitter_bar")
```

THE MARK THAT WAS DRAWN, not the kind that was asked for: an unrecognised kind falls back to the bar, and a caption that read the kind would leave that bar's whisker unnamed.
