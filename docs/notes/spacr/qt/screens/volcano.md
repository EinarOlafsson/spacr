# Notes from `spacr/qt/screens/volcano.py`

Prose lifted out of `spacr/qt/screens/volcano.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [find_results_table](#find_results_table) (1 entry)
- [load_results](#load_results) (1 entry)
- [_make_screen.VolcanoScreen.__init__](#_make_screenvolcanoscreen__init__) (1 entry)
- [Module level](#module-level) (1 entry)

## find_results_table

### lines 91-92

```python
for entry in sorted(os.listdir(path)):
```

One level down, so pointing at the run folder rather than its `guide_permutation/list` leaf still works.

## load_results

### lines 116-118

```python
first = frame["outcome"].iloc[0]
```

Several responses were fitted. Show the first; the explorer's own data controls can switch columns, and each response is its own correction family so they must not be pooled into one plot.

## _make_screen.VolcanoScreen.__init__

### lines 141-143

```python
"""Build the screen. ``host`` is the window, NOT a Qt parent.
```

`host` is the main window, passed by the registry for navigation -- NOT a Qt parent. Handing it to QWidget.__init raises, because the registry's host is not always a QWidget.

## Module level

### lines 199-210

NO REGISTRY ROW. The explorer is reached from the module it publishes: it is "Publication figure…" on the Regression volcano's own right-click menu and a button on that screen's masthead, both going through :func:`spacr.qt.screens.regression.publication_opener`, which builds it from the same :func:`_make_screen` factory the tile used and seeds it with the FRAME on screen. That seeding is why the fold is a superset of the tile: a live run, a bare CSV or a frame handed in cannot be found again from a folder, so a standalone tile could only ever publish a re-read of the disk.

The strings above are kept because they are this module's public description the fold button's name and sentence are asserted against them, and the i18n catalogs carry the translations.
