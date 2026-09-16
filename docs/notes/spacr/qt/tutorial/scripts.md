# Notes from `spacr/qt/tutorial/scripts.py`

Prose lifted out of `spacr/qt/tutorial/scripts.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [_find_menu](#_find_menu) (3 entries)
- [_menu_target](#_menu_target) (2 entries)
- [_top_level_menu_containing](#_top_level_menu_containing) (3 entries)
- [_build_classify_steps](#_build_classify_steps) (4 entries)
- [_build_map_barcodes_steps](#_build_map_barcodes_steps) (1 entry)
- [_build_regression_steps](#_build_regression_steps) (1 entry)
- [_build_train_compare_steps](#_build_train_compare_steps) (2 entries)
- [_build_timelapse_steps](#_build_timelapse_steps) (1 entry)

## _find_menu

### lines 173-176

```python
if title == "Demos":
```

Demos is a Help submenu and MainWindow deliberately retains the QMenu wrapper that owns its actions. Qt may reparent that submenu when it is inserted under Help, so it is not reliably returned by the menu bar's child walk on every PySide6 version. Use the retained owner first.

### lines 180-183

```python
if menu is not None:
```

``_demo_menu`` is the semantic owner, independent of the currently selected language. Comparing its rendered title to the English lookup key made the tutorial lose the menu after a retranslation within the same application session.

### line 185, trailing  _(unsure)_

```python
menu.title()
```

prove that the retained Qt wrapper is live

## _menu_target

### lines 218-221

```python
parent = _top_level_menu_containing(window, menu)
```

A SUBMENU HAS NO PLACE ON THE BAR. Demos moved under Help on 2026-08-23, so its own geometry is empty and the cursor would aim at (0, 0). Point at the top-level menu you actually click to reach it, which is where a user's hand goes.

### lines 227-232

```python
actions = list(mb.actions())
```

The target users click is Help, not the submenu itself. During a long tutorial-test session Qt can briefly detach the submenu action while pages are being rebuilt even though ``_demo_menu`` remains live. Target the stable top-level Help action directly in that state. The final-action fallback is language-independent; spaCR and Help are the only top-level menus in the current compact bar.

## _top_level_menu_containing

### lines 251-255

```python
rect = mb.actionGeometry(menu.menuAction())
```

A real rectangle on the bar proves this is already a top-level menu.  Check that first: during long PySide test sessions a released submenu wrapper can be recycled and acquire the same rendered title as an unrelated menu, so title-only relation scans must never be allowed to reclassify a bar menu as its own child.

### lines 266-269

```python
top = _find_menu(window, action.text().replace("&", ""))
```

Retrieve the bar-owned QMenu wrapper through the same stable lookup used elsewhere. Returning ``action.menu()`` directly can leave the caller with a deleted temporary PySide wrapper after ``action`` is released.

### lines 276-279

```python
if (nested is not None and
```

Compare semantic titles only. Temporary PySide wrappers can be recycled after Qt releases them, so Python object identity is not a safe submenu relation across event-loop turns.

## _build_classify_steps

### lines 615-617  _(unsure)_

```python
def _build_classify_steps(window) -> List[Step]:
```

Classify module tutorial — hosted in AnnotateScreen

### lines 682-685

```python
Step(
```

ONE BUTTON NOW, NOT TWO. "Train CV" and "Train XG" were merged into a single `Train...` with a menu, and this step was not moved with them -- so its target resolved to None and the tutorial pointed at nothing. The narration named both old buttons too.

### lines 695-700

```python
Step(
```

AND THEN IT OPENS CLASSIFY. Until 2026-09-04 the tutorial called "classify" stopped here, at Annotate's Train button, having said "both open in the consolidated Classify module" and never opened it. That is the stale module boundary instruction 358 was filed about: a polished lesson teaching a structure the application no longer has.

### lines 712-715

```python
Step(
```

FIVE MODULES WERE FOLDED IN HERE, and instruction 358 asks that each one be named and located rather than left for the reader to find. The specialist lessons that still explain them accurately are kept and pointed at rather than re-narrated.

## _build_map_barcodes_steps

### lines 744-750

```python
def _build_map_barcodes_steps(window) -> List[Step]:
```

Map Barcodes tutorial — the one Core module that reads sequencing, not images

THE ODD ONE OUT, and the tutorial has to say so early. Every other Core module takes microscopy; this one takes FASTQ and produces the table that tells Regression which well got which perturbation. A reader who arrives expecting images needs that said before anything else.

## _build_regression_steps

### lines 817-824

```python
def _build_regression_steps(window) -> List[Step]:
```

Regression tutorial — the module that consumes what everything else produces

NO DEMO OF ITS OWN, and that is honest rather than a gap: regression needs a measured screen AND a barcode mapping, so its live data is the OUTPUT of the two tutorials before it. The script says which ones and in what order, rather than pretending a synthetic single-module dataset would teach the thing that matters here.

## _build_train_compare_steps

### lines 895-908

```python
def _build_train_compare_steps(window) -> List[Step]:
```

The three downstream readers: Training Runs, Prediction Profiler, Investigate Hit

NONE OF THEM HAS A DEMO, and none of them should. Each reads an artefact an earlier module WROTE -- a set of training runs, a fitted model, a regression hit -- so a synthetic single-module dataset would teach a workflow nobody has. Each lesson therefore names the module that produces its input, in the same way the Regression lesson does.

Two of the three are also reachable from Classify's masthead. Instruction 358 asks that a folded action be named and located rather than narrated twice, so these lessons say where the button is instead of the Classify lesson explaining what the module does.

### lines 933-936

```python
Step(
```

NOT IN THE SIDEBAR, and the lesson has to open by saying so. This module has no sidebar row: it is reached from Classify's masthead, because comparing runs is something you do while working on a model rather than a place you navigate to.

## _build_timelapse_steps

### lines 1134-1142

```python
def _build_timelapse_steps(window) -> List[Step]:
```

Timelapse tutorial — the tracking switch on Mask Generation

Timelapse has no destination of its own: it is the mask pipeline with tracking turned on, so what is its own is a couple of settings CATEGORIES and a switch on the Mask masthead that reveals them. The script therefore lands on Mask, loads the timelapse demo -- whose settings file carries `timelapse=True`, which moves the switch as it is applied -- and narrates the switch and the categories it just revealed.
