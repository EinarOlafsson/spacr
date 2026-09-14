# Notes from `spacr/response_distribution.py`

Prose lifted out of `spacr/response_distribution.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [describe](#describe) (2 entries)
- [compare](#compare) (1 entry)
- [fast_panel](#fast_panel) (1 entry)
- [panel](#panel) (3 entries)

## describe

### lines 61-62  _(unsure)_

```python
out["normality_p"] = float(stats.normaltest(data).pvalue)
```

D'Agostino, which is what `check_distribution` itself uses -- so the number on the panel is the number that chose the family.

### lines 72-74

```python
with contextlib.redirect_stdout(io.StringIO()):
```

`check_distribution` PRINTS its reasoning, which is useful in a run log and is noise when a plot asks it a question. Swallowed here rather than removed there: the printing is somebody's diagnostic.

## compare

### lines 153-154

```python
"changed": bool(changed),
```

A TRANSFORM THAT CHANGED NOTHING IS VISIBLE AS SUCH. An absent panel reads as a missing feature rather than as an answer.

## fast_panel

### lines 242-245

```python
xs = np.repeat(edges, 2)[1:-1]
```

A STEP OUTLINE, NOT FILLED BARS. Two filled histograms on one axis hide each other whichever order they are drawn in; two outlines overlay and stay readable, which is the comparison being asked for.

## panel

### lines 292-295

```python
twin = ax.twiny()
```

SEPARATE AXES, SHARED FIGURE. A log of a proportion and the proportion itself have no common scale, and forcing them onto one puts every point of the smaller into a single bar -- which looks like a finding and is an artefact of the axis.

### line 306

```python
both = np.concatenate([before, after]) if after.size else before
```

ONE AXIS, which is what makes "what changed" readable at a glance.

### lines 318-319

```python
ax.text(0.01, -0.22, caption(result), transform=ax.transAxes,
```

THE NAMES GO ON THE PANEL, which is the substance of the request not left for the reader to judge by eye.
