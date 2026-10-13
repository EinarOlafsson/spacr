# Notes from `spacr/qt/curation_tool.py`

Prose lifted out of `spacr/qt/curation_tool.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [BrushPanel.__init__](#brushpanel__init__) (1 entry)
- [BrushPanel._build](#brushpanel_build) (1 entry)
- [BrushPanel._on_layers_changed](#brushpanel_on_layers_changed) (1 entry)
- [TrackCurationPanel.load](#trackcurationpanelload) (1 entry)
- [TrackCurationPanel._do](#trackcurationpanel_do) (2 entries)

## BrushPanel.__init__

### lines 248-252

```python
self._session.subscribe(self._on_edit_recorded)
```

Two subscriptions, because they fire at different moments. The stack fires per dab (mid-stroke, before anything is recorded); the session fires when a ledger entry lands, which is what this panel is showing. Listening only to the first left the ledger and the undo button one stroke behind for ever.

## BrushPanel._build

### lines 321-325

```python
self.save_mask_button = FlatButton(
```

"Save log" writes the record and not the pixels, and a record on its own asserts corrections to a file nothing edited -- which is the state `spacr.curation.is_curated` then reports as hand-edited. This is the control that makes the claim true, and it sits beside the one that makes it so the two are never separated.

## BrushPanel._on_layers_changed

### lines 381-383

```python
"""Re-bind to the labels layer after the stack changed.
```

Derived, not stored: the ledger and the badge follow the model, so a paint made through the tool and one made from a script look the same here.

## TrackCurationPanel.load

### lines 628-630

```python
self.status.setText(str(exc))
```

After refresh for the same reason as above: refresh() resets this label to "No tracks open", which is true and useless next to the sentence saying WHY nothing opened.

## TrackCurationPanel._do

### lines 678-679

```python
self.status.setText(str(exc))
```

Said, not swallowed: a button that silently declines is indistinguishable from a broken one.

### lines 683-686

```python
self.status.setText(f"{edit.describe()}\n{session.describe()}")
```

After refresh, not before: refresh() writes the session summary into this same label, and setting the line first meant every successful action was immediately overwritten by the summary. Both facts matter, so both are shown.
