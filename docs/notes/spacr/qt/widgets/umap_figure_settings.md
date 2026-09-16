# Notes from `spacr/qt/widgets/umap_figure_settings.py`

Prose lifted out of `spacr/qt/widgets/umap_figure_settings.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [Field](#field) (1 entry)
- [Module level](#module-level) (1 entry)
- [_is_fixed_colour](#_is_fixed_colour) (1 entry)
- [redraw_umap_figure](#redraw_umap_figure) (2 entries)
- [UmapFigureSettings._editor](#umapfiguresettings_editor) (1 entry)

## Field

### line 65, trailing  _(unsure)_

```python
kind: str
```

int | float | text | bool | choice | int_or_none

## Module level

### line 79  _(unsure)_

```python
Field("dot_size",        "Dot size",        "int",   1, 4000, TIER_STYLE),
```

applies to the artists already drawn

## _is_fixed_colour

### lines 159-161  _(unsure)_

```python
def _is_fixed_colour(point_color) -> bool:
```

Applying values to a finished figure

## redraw_umap_figure

### lines 299-300

```python
LOG.debug("could not redraw the image overlay", exc_info=True)
```

A montage is decoration; the embedding is the result. Losing the thumbnails must never lose the figure (INVARIANTS 10).

### lines 302-304

```python
fig._spacr_umap_payload = payload
```

`Figure.clear` drops artists, not attributes -- restated rather than relied on, because a figure that loses its payload can never be edited a second time.

## UmapFigureSettings._editor

### lines 509-511

```python
edit = QLineEdit()
```

text and int_or_none. `row_limit` is genuinely nullable -- None means "every row" -- and a spin box has no way to say that, so it is typed rather than clamped.
