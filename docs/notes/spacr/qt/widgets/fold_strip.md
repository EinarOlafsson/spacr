# Notes from `spacr/qt/widgets/fold_strip.py`

Prose lifted out of `spacr/qt/widgets/fold_strip.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [Module level](#module-level) (2 entries)
- [folded_modules](#folded_modules) (1 entry)
- [FoldButton.__init__](#foldbutton__init__) (5 entries)
- [FoldButton.set_stage](#foldbuttonset_stage) (1 entry)
- [FoldButton._install_checked_fill](#foldbutton_install_checked_fill) (1 entry)
- [mark_folded_sections](#mark_folded_sections) (1 entry)

## Module level

### lines 95-97

```python
"spacr.qt.screens.foreign",
```

Import is here for a reason the others are not: Import Images never had a registry row, so this tuple is the ONLY route to its name, its sentence and its maturity.

### lines 106-134

```python
"spacr.qt.screens.graph_builder",
```

ALIGN & STITCH WAS HERE and is not any more. OPS was folded onto it stitching, over a plate acquired in sequencing cycles -- and moved to Mask Generation on 2026-09-09 at the user's request, taking the declaration with it. Align declares no `FOLDED_APPS` now, so it would contribute nothing to the walk; the entry is removed rather than left to read as a host that offers something. THE THREE LATE HOSTS, ADDED 2026-09-08. They were deliberately left out on the reasoning that they "fold modules that kept their rows, so the registry still answers for them and adding them here would change which host their folds are attributed to". The registry does still answer but it answers what a KEY SAYS, not WHO HOSTS IT, and only this walk builds the second answer. So their folds were reachable from no host at all: `test_every_folded_module_really_is_folded_and_really_is_reachable` named control_chart, outliers and trellis, and they were offered by hosts nothing was looking at.

The attribution worry does not arise for these keys. `found` is first-host-wins, so it only matters when two hosts declare the same key, and none of plate_view, trellis, layer_viewer, control_chart, outliers, lineage or tabulate is declared anywhere else.

THIS TUPLE HAS NOW FALLEN BEHIND TWICE -- Align & Stitch above, these three here -- so the drift is caught in the parity suite rather than left for the next reader: `test_every_screen_that_declares_folded_apps_is_a_known_host` derives the set from the source and fails when they disagree. It is a test and not a runtime glob on purpose: this walk runs while the menu bar and dock are built, and the comment above is about keeping work off that path.

## folded_modules

### lines 376-381

```python
declared = _host_declarations(module_name)
```

READ, NOT IMPORTED. This runs while the menu bar and the dock are being built, and importing every fold host pulls their dependency trees into the process before Home has painted -- pandas and scipy arrived that way, through `make_masks`, `foreign` and the settings model. The packaged smoke test asserts Home crosses no operation-only import boundary and was failing on exactly this.

## FoldButton.__init__

### lines 491-494

```python
self.setProperty("stage", stage)
```

The stage rides as a Qt property so the stylesheet can select on it -- QPushButton#FoldButton[stage="alpha"]:hover -- exactly as the tiles do. Setting it before the first polish means the first paint already has the right colour.

### lines 505-518

```python
from ..app import _icon_for_app
```

THROUGH `app._icon_for_app`, NOT `iconset.app_icon`. `iconset.app_icon` is told nothing about `_ICON_OVERRIDES`, so it resolves a key by filename alone -- and for every module that BORROWS another module's picture that is the wrong file. Reported 2026-09-02: the Cellpose Workbench button drew a DUMBBELL, because `train_cellpose.png` exists and is the training glyph, while the override sends that key to `cellpose_masks.png`, the white cell outline. The same was true of every other borrower: analyze_plaques, agreement, plate_view, model_compare and model_zoo.

Imported inside the call because `spacr.qt.app` imports this module; at call time the cycle is closed and the lookup is a dict hit.

### lines 526-527

```python
self.setText(name[:1].upper())
```

No icon shipped for this key: fall back to the initial rather than to an empty square the user cannot identify.

### lines 529-530

```python
self.setToolTip(f"{name}\n{description}".strip())
```

The name leads the tooltip because the button has no label; the description follows it as the sentence the tile carried.

### lines 532-536

```python
self.setProperty("moduleNameSource", name)
```

AND THE SAME PROPERTIES THE SIDEBAR CARRIES, so `module_hints` can divert this into the status bar instead of drawing it over the masthead. Canonical English sources, not the rendered tooltip, so a language switch retranslates rather than translating a translation.

## FoldButton.set_stage

### lines 635-637

```python
self.style().unpolish(self)
```

A property the stylesheet selects on is only read at polish, so a button already on screen keeps the old colour until it is polished again.

## FoldButton._install_checked_fill

### lines 661-663

```python
return
```

A maturity the table has never heard of: leave the button with the shipped hover and pressed rules rather than inventing a colour that no tile lights up in.

## mark_folded_sections

### lines 723-731

```python
def mark_folded_sections(key: str, sections: Iterable[QWidget]
```

The other half of the icon: the settings the module left behind

A fold that becomes a BUTTON keeps its picture -- the button is the picture. A fold that becomes SETTINGS CATEGORIES has no button and so nowhere obvious to put it, and a group of settings that arrived from somewhere else says nothing about where. The mark goes on the heading: the same icon, beside the category name, on the host's own form.
