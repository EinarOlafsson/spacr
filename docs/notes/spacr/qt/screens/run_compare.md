# Notes from `spacr/qt/screens/run_compare.py`

Prose lifted out of `spacr/qt/screens/run_compare.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [Module level](#module-level) (2 entries)
- [RunCompareScreen.__init__](#runcomparescreen__init__) (1 entry)
- [RunCompareScreen._fill_combos](#runcomparescreen_fill_combos) (1 entry)
- [RunCompareScreen._set_verdict](#runcomparescreen_set_verdict) (1 entry)
- [_tree](#_tree) (1 entry)
- [_fill_hits](#_fill_hits) (2 entries)

## Module level

### lines 65-69

```python
_ROW = declared_app(APP_KEY)
```

The row this screen puts in the registry is declared in

`spacr.qt.app_catalog`, which is what lets the app be registered without importing this module -- the launch reads the table, not the screen. These read the same row back rather than restating it, so the name, the blurb and the nine translations have one spelling and no second copy to drift from.

### lines 116-119

```python
register_widget_qss("RunCompareBanner", _banner_qss, replace=True)
```

``replace=True`` because this module owns the name: a reimport (a test that reloads it, a plugin that pulls it in twice) must re-register the same block rather than raise on the duplicate and leave the screen unstyled.

## RunCompareScreen.__init__

### lines 161-162  _(unsure)_

```python
from ..dnd import install_for
```

Drop anywhere on this screen: the path is resolved through spaCR's project layout, so the plate folder finds what this screen reads.

## RunCompareScreen._fill_combos

### lines 299-301

```python
self._a_combo.setCurrentIndex(1)
```

Newest as B, the one before it as A: the question is almost always "what did the run I just did do differently?", and that reads better as a change *into* the newest.

## RunCompareScreen._set_verdict

### lines 421-423

```python
widget.style().unpolish(widget)
```

A dynamic property only reaches the stylesheet after the widget is re-polished; without this the error border never appears until something else forces a restyle.

## _tree

### lines 432-434  _(unsure)_

```python
def _tree(columns: Tuple[str, ...]) -> QTreeWidget:
```

Table filling — module functions so they are testable without a screen

## _fill_hits

### lines 527-528  _(unsure)_

```python
header.addChild(tree_item([
```

No status column: the group heading already says what happened, and repeating it in every row is noise.

### lines 534-535  _(unsure)_

```python
header.setExpanded(label != "Held rank")
```

Held ranks are the boring group and the biggest; collapsed so the two that matter are what the tab opens on.
