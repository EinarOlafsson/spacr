# Notes from `spacr/figures/plates.py`

Prose lifted out of `spacr/figures/plates.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [plate_names](#plate_names) (1 entry)
- [well_matrices](#well_matrices) (7 entries)
- [small_multiple_layout](#small_multiple_layout) (1 entry)
- [draw_plate](#draw_plate) (3 entries)
- [build_plates](#build_plates) (2 entries)
- [_colour_bar](#_colour_bar) (1 entry)

## plate_names

### lines 109-111

```python
name = key
```

Not a well key. Kept rather than dropped so a malformed row is visible as its own "plate" instead of silently vanishing into another one.

## well_matrices

### lines 183-186

```python
wanted = ["prc"] + ([variable] if variable in frame.columns else [])
```

Only the columns the aggregation needs. generate_plate_heatmap writes plateID/rowID/columnID onto the frame it is given, and that must not land on the caller's table; a two-column projection is also a great deal cheaper than copying a measurement frame.

### lines 190-200

```python
text = work["prc"].astype(str)
```

ONE PLATE'S ROWS, ONCE. generate_plate_heatmap re-parses every prc in the frame it is handed, on every call, and this asks it for two maps per plate rather than one -- so on a million-row measurement frame the naive loop costs 2n per plate. Handing it only the rows of the plate being drawn makes that 2n across ALL the plates, which is less work than the single-map version was doing.

Only for the plain 3-token identifier. A LONGER prc carries an experiment prefix, and generate_plate_heatmap then treats every row as belonging to the plate it was asked for -- so splitting on the leading token would silently change which wells are drawn.

### lines 206-215

```python
readable = None
```

A ROW IS NOT A MEASUREMENT, and the count map counts rows. generate_plate_heatmap coerces the variable with ``errors='coerce'``, so a well whose every row holds nothing numeric -- an empty cell, an 'n/a', a merge that did not find a match -- aggregates to NaN, is filled with 0, and, having a row count above zero, survives the mask below as a measurement of zero. That is the same defect one step further in, and it sets the bottom of the shared scale in the same way. Which rows carry a number is therefore worked out ONCE, and only when some row does not: on a clean frame this costs one pass and no extra heatmap at all.

### lines 224-227

```python
on_plate = None if head is None else head == str(name)
```

.copy(), because generate_plate_heatmap assigns columns onto the frame it is given and a boolean-mask slice is a view: pandas warns (SettingWithCopyWarning) and, under copy-on-write, the assignment would land somewhere the next call cannot see.

### lines 232-233

```python
if grouping == "count":
```

The count map IS the value map when the caller asked for counts, so that case does not pay for a second pass over the frame.

### lines 264-266

```python
block = grid.to_numpy(dtype="float64", copy=True)
```

Pandas 3 copy-on-write may expose a read-only array here.  The missing-well mask below is an intentional in-place refinement, so request an owned writable buffer explicitly.

### lines 268-270

```python
empty = ~(seen.to_numpy(dtype="float64") > 0)
```

A count of zero -- or a well the reindex invented -- is a well that was not measured. It is not a measurement of zero and must not be painted as one, nor counted when the colour scale is chosen.

## small_multiple_layout

### lines 368-370

```python
penalty = abs(math.log(aspect / target))
```

Compared in log space, so "twice too wide" and "twice too tall" cost the same. Ties go to the wider arrangement, which is what a screen is.

## draw_plate

### lines 443-446

```python
ax.add_patch(Rectangle((0, 0), n_columns, n_rows,
```

The wash goes down as a real artist rather than as the axes facecolor: savefig(transparent=True) -- which the house style asks for -- forces every axes patch to 'none', so a facecolor wash is on screen and gone in the file.

### lines 464-467

```python
ax.tick_params(length=1.6, width=WEIGHTS["spine"], pad=1.4, colors=ink,
```

Colour and size named here rather than left to the rcParams: the figure is built inside the style context but DRAWN outside it, and a tick that resolves its properties at draw time would resolve them against whatever the process happens to hold then.

### line 474  _(unsure)_

```python
ax.set_title(name, fontsize=TYPE_SCALE["annotation"], pad=2.0,
```

A descriptor, not a sentence title: the plate's own name.

## build_plates

### lines 543-546

```python
cell_w = (width - MARGIN["left"] - MARGIN["right"]
```

SIZED FROM THE GRID. The cell is whatever is left after the margins, and the figure is made tall enough for cells of exactly the plate's proportions -- so square wells are what the layout produces, not what it survives.

### lines 573-575

```python
subject = ("objects per well, counted" if grouping == "count" else
```

WHAT WAS ACTUALLY DONE TO THE NUMBERS. A legend that says

"averaged" under a panel drawn with grouping='count' is a legend that misreports its own figure.

## _colour_bar

### lines 635-639

```python
figure.text(0.5, (BAR["bottom"] - 0.022) / height,
```

The name goes on the same line as the two numbers, centred, and is placed in FIGURE INCHES rather than left to `set_xlabel`: a colour bar 0.075 inches tall gives matplotlib almost nothing to measure a label offset from, and where it lands then depends on whether the figure has been drawn yet. Lower case, spelled out -- the axis-label rule.
