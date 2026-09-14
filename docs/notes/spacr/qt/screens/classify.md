# Notes from `spacr/qt/screens/classify.py`

Prose lifted out of `spacr/qt/screens/classify.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [Module level](#module-level) (1 entry)
- [LazyFlowViewSection.__init__](#lazyflowviewsection__init__) (2 entries)
- [LazyFlowViewSection._collector_for_open_panel](#lazyflowviewsection_collector_for_open_panel) (1 entry)
- [LazyFlowViewSection._ensure_panel](#lazyflowviewsection_ensure_panel) (1 entry)
- [install_folds](#install_folds) (1 entry)

## Module level

### lines 214-218

```python
"train_compare": _build_train_compare,
```

`train_compare` still holds a registry row; `feature_explorer` never had one and is declared in `app_catalog`. Neither needs a `FOLD_FALLBACK` entry: `fold_description` reads the registry first and the declared catalogue second, so both buttons get their real name, sentence and maturity colour without a third copy here.

## LazyFlowViewSection.__init__

### lines 265-272

```python
self.setAttribute(Qt.WA_StyledBackground, True)
```

WITHOUT THIS THE BOX IS NOT DRAWN AT ALL, and that -- not the colour in it -- is why FlowView read as a black rectangle through two attempts at recolouring it. `CollapsibleSection` is a QWidget, and a plain QWidget ignores a stylesheet background, border and radius unless it is told to style its own background; so the rule registered for this object name was never painted, and what showed was the application ground behind it. `ConsolePanel` carries the same line for the same reason.

### lines 274-276

```python
self._header.setProperty("_spacr_i18n_text", "FlowView")
```

The section is installed after the screen's first translation pass, so render its chrome immediately while retaining the English source properties the next live-language pass needs.

## LazyFlowViewSection._collector_for_open_panel

### line 313, trailing

```python
except Exception:
```

a broken visualisation never reaches Classify

## LazyFlowViewSection._ensure_panel

### lines 374-386

```python
from ..theme import clear_container_surfaces
```

THE SPLITTER INSIDE IT PAINTS AN OPAQUE BLACK RECTANGLE, which is the "black background" reported on 2026-09-04. Measured, the panel rendered over magenta: `FlowViewPanel` itself came back transparent and its `QSplitter` came back `#000000` -- a QSplitter is a plain QWidget, so it takes the blanket `QWidget { background-color: bg }` rule however transparent its parent is, and the section's own QSS never named it.

`clear_container_surfaces` is the helper for exactly this and tags splitters by type. It cannot live in `spacr/flowview/panel.py`: that module is shared and imports no Qt theme, so the tagging belongs here, where the panel is embedded.

## install_folds

### line 476, trailing

```python
except Exception:
```

a broken optional panel must not cost the fold strip
