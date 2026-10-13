# Notes from `spacr/figures/sheet.py`

Prose lifted out of `spacr/figures/sheet.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [build_sheet](#build_sheet) (2 entries)
- [build_panel](#build_panel) (1 entry)

## build_sheet

### lines 112-114

```python
scratch = plt.figure()
```

Draw once into a throwaway to find out which panels this table can support, so the grid is sized for what will actually appear rather than leaving holes.

### lines 157-158  _(unsure)_

```python
figure.subplots_adjust(left=.09, right=.98, top=.93, bottom=.09,
```

More space between panel groups than within them: the only hierarchy cue the published figures use.

## build_panel

### lines 174-177

```python
with figure_style(target or theme_target(), kind=key):
```

`kind=key` is how the user's PER-GRAPH preference reaches this panel: the volcano's point size is not the heatmap's, which is the whole reason instruction 118 has a per-graph layer. A key that is not a known graph kind simply has no overrides, so this is safe for every panel.
