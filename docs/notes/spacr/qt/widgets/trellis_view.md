# Notes from `spacr/qt/widgets/trellis_view.py`

Prose lifted out of `spacr/qt/widgets/trellis_view.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [TrellisCanvas.__init__](#trelliscanvas__init__) (1 entry)
- [TrellisCanvas.set_trellis_spec](#trelliscanvasset_trellis_spec) (1 entry)
- [TrellisCanvas.render_now](#trelliscanvasrender_now) (3 entries)
- [TrellisCanvas._label_trellis_panel](#trelliscanvas_label_trellis_panel) (1 entry)
- [TrellisPanelWidget.__init__](#trellispanelwidget__init__) (1 entry)
- [TrellisPanelWidget._sync](#trellispanelwidget_sync) (1 entry)

## TrellisCanvas.__init__

### lines 101-103

```python
self._trellis_spec = TrellisSpec()
```

After the base constructor, which builds the figure and subscribes to the link but does not render — so nothing reads these before they exist.

## TrellisCanvas.set_trellis_spec

### lines 125-126  _(unsure)_

```python
self._spec = spec.graph
```

The inherited helpers read `self._spec`; keeping the two in step is what lets every drawing method be reused unchanged.

## TrellisCanvas.render_now

### lines 190-192

```python
axes = self._figure.subplots(nrows, ncols, squeeze=False,
```

`sharex`/`sharey` are deliberately off: every panel's limits are written from its own scale group below, which is stronger — and under a free or per-row mode, sharing would be wrong.

### lines 201-202

```python
ax.set_visible(False)
```

The remainder of a wrapped division. Not a panel with no data — there is no group here at all — so it is not drawn.

### lines 209-211

```python
previous, self._scales = self._scales, panel.scales
```

The inherited drawing helpers read `self._scales`; pointing it at this panel's group is what makes a free-scale histogram use this panel's bin edges rather than the grid's.

## TrellisCanvas._label_trellis_panel

### lines 272-274

```python
if spec.scale_x == SCALE_SHARED and not is_bottom:
```

Inner ticks are hidden only where the axis really is shared. A panel with its own limits prints its own numbers, or the layout is tidier than it is true.

## TrellisPanelWidget.__init__

### lines 472-474

```python
from ..screens.settings_model import retarget_field_tooltips
```

Hover help belongs on a setting's NAME, not on the field the user is about to type into (instruction 113). One post-pass rather than a convention every hand-built row has to remember.

## TrellisPanelWidget._sync

### lines 558-570

```python
index = self._kind.findData(spec.graph.kind or "")
```

A CONTROL THAT CANNOT SHOW THE SPEC FALLS BACK, it does not keep the last thing it happened to be showing. Leaving the previous value made the picker disagree with the spec, and `_on_controls_changed` reads the PICKER -- so the next touch of any control silently rewrote the spec to whatever the shelf was displaying. Instruction 310 A51..A57, entry A56: a spec restored from a saved layout with kind "empty" left the picker reading "Histogram", and moving the Bins box turned the spec into a histogram without the user choosing one.

Index 0 is the honest answer in both cases: "Automatic" for the plot kind, "shared" for a scale. Neither claims a specific kind the spec did not ask for.
