# Notes from `spacr/qt/regex_editor.py`

Prose lifted out of `spacr/qt/regex_editor.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [RegexEditorDialog.__init__](#regexeditordialog__init__) (7 entries)
- [RegexEditorDialog._on_regex_changed](#regexeditordialog_on_regex_changed) (2 entries)
- [RegexEditorDialog._on_auto_detect](#regexeditordialog_on_auto_detect) (1 entry)
- [RegexEditorDialog._on_save](#regexeditordialog_on_save) (1 entry)

## RegexEditorDialog.__init__

### lines 119-121

```python
from .preferences import scaled_px
```

SCALED. 760x520 was measured at font scale 1.0; the prose inside it is not, so at 2x the dialog stayed the same size while every line in it doubled.

### lines 140-146

```python
_let_it_have_its_height(intro)
```

(Preferred, Minimum), WHICH IS THE HOUSE RULE `prerun._label` writes down: with Qt's default Preferred height a parent is free to hand a word-wrapped label LESS than its heightForWidth, and the last lines are silently clipped. Measured before this: this label wrapped to 54 px and was given 36 at font scale 1.0, and to 180 px in 88 at 2.0 -- so the sentence naming `chanID`, which is the one thing the dialog exists to explain, was the part cut off.

### lines 166-171

```python
self._workbench_btn = QPushButton("Work it out from the files…")
```

THE WORKBENCH (137 A, C, D). This dialog previews the MATCH; the workbench previews the IMPORT -- one row per file with the name it would get, a dropdown saying what each group means, and the folder tree it would produce, with the unmatched files named. Two windows because they answer two questions, and this one is what a drop opens.

### line 182  _(unsure)_

```python
preset_row = QHBoxLayout()
```

─── Preset dropdown ────────────────────────────────────────

### line 195  _(unsure)_

```python
self._warnings_lbl = QLabel("")
```

─── Warnings + preview ──────────────────────────────────────

### lines 199-202

```python
_let_it_have_its_height(self._warnings_lbl)
```

The same rule, and it matters more here: this label holds the warnings that say WHY a regex will not work, and it grows with however many there are. Clipped, it shows the first and hides the rest.

### line 220  _(unsure)_

```python
if initial_regex:
```

─── Initial state ───────────────────────────────────────────

## RegexEditorDialog._on_regex_changed

### line 228  _(unsure)_

```python
"""Follow the typed pattern with the preset box and the preview.
```

Match the dropdown to the current text if it matches a preset

### line 245  _(unsure)_

```python
self._preset_combo.blockSignals(True)
```

Custom

## RegexEditorDialog._on_auto_detect

### lines 296-300

```python
self._refresh_preview()
```

Rebuild the table explicitly rather than relying on setText to emit textChanged: QLineEdit stays silent when the text is unchanged, so clicking "Auto detect" a second time used to stack another status line onto a stale preview, and the no-pattern branch left the warnings label blank entirely.

## RegexEditorDialog._on_save

### lines 351-358

```python
"""Trim the pattern for ``get_regex`` and accept.
```

TRIMMED FOR `_get_regex`, which appends the extension itself. What this box holds matches WHOLE FILENAMES -- the preview above is matched against them -- and `auto_detect_regex` returns a pattern ending `\.(?:tif|tiff|png|jpg|jpeg)$`. Saved verbatim into `custom_regex` that became `(...$)..tif`: an anchor with characters after it, which can never match. Measured through the real path on eight cellvoyager names: 0 of 8, with no error anywhere and the pattern in the box looking exactly right.
