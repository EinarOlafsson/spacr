# Notes from `spacr/qt/widgets/section.py`

Prose lifted out of `spacr/qt/widgets/section.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [module_mark](#module_mark) (2 entries)
- [Section.__init__](#section__init__) (4 entries)
- [Section.add_prose_row](#sectionadd_prose_row) (2 entries)
- [Section._source_row](#section_source_row) (1 entry)
- [Section._refresh_header_text](#section_refresh_header_text) (1 entry)
- [Section._sync_header_minimum](#section_sync_header_minimum) (1 entry)
- [Section._refresh_tooltip](#section_refresh_tooltip) (1 entry)
- [Section._on_toggle](#section_on_toggle) (2 entries)

## module_mark

### lines 103-105

```python
glyphs = getattr(iconset, "_NAME_TO_GLYPH", {}) or {}
```

The glyph table is the second place a key can have a mark of its own. Read defensively: without it this degrades to "bundled artwork only", which is still a real mark rather than a guess.

### lines 109-112

```python
from ..app import _icon_for_app
```

See the note in `fold_strip.py`: `iconset.app_icon` knows nothing of `_ICON_OVERRIDES`, so a module that borrows another's picture gets the wrong file. This heading marks a folded module's settings and must match its button.

## Section.__init__

### lines 173-183

```python
self._title_source = str(title)
```

A QToolButton reads '&' as a mnemonic marker, so the categories named "Plate Layout & Controls" and "Embedding & Clustering" rendered as "PLATE LAYOUT _CONTROLS" -- the ampersand swallowed and the following letter underlined as an accelerator that goes nowhere. Section headers are not keyboard shortcuts, so the '&' is escaped for display. `title()` still answers with the real text.

THE CATALOG IS KEYED ON THE WRITTEN CATEGORY NAME, so the source is kept as the caller wrote it and uppercased only on the way to the button. Looking up a caption that has already been uppercased finds nothing and leaves the header in English.

### lines 186-190

```python
self._header.setProperty("i18nSkipText", True)
```

The caption is composed -- the category, and for beta or alpha a maturity badge -- so the generic language pass would ask for the finished line as one key and never find it. Keep that pass off the button and rebuild the caption from the translated parts whenever the language changes.

### lines 199-205

```python
self._header.installEventFilter(self)
```

NO POPUP OVER A CATEGORY. The blurb is already shown in the strip under the actions row whenever a header is hovered, so Qt's own tooltip put the same words in a second place -- a tall window that follows the pointer and covers the settings underneath the one being read. The TEXT stays on the widget: assistive technology reads `toolTip()`, and so do the checks that assert a category explains itself. Only the popup is refused.

### lines 216-223

```python
self._form.setFieldGrowthPolicy(
```

SET EXPLICITLY, because the default is the STYLE's answer and not every style answers the same way. Issue 115 reported "field and setting do not expand with container" on macOS. Measured: with Fusion, a 1,178 px section gives its QLineEdit 1,115 px; under a style whose SH_FormLayoutFieldGrowthPolicy is FieldsStayAtSizeHint hostile, but a valid Qt answer, and the shape the reporter's platform style chose -- the same section gives the field 108 px. Naming the policy here takes the decision away from the platform.

## Section.add_prose_row

### lines 306-318

```python
from .eliding import ElidingLabel
```

AN ELIDING LABEL, NOT A BARE ONE, and the difference only shows in a translation. The label column is as wide as its widest label's hint, and every settings label in it elides -- so the column is capped, and a plain QLabel wider than the cap is cut off mid-glyph rather than shortened. Measured on Regression in German: "Herunterladen" wants 83 px, the column grants 58, and English "Download" needs 57 and fits exactly, which is why it was invisible until the sweep of instruction 350 ran in a second locale.

`ElidingLabel.sizeHint` still asks for the FULL width, so where the column can afford it nothing is elided at all; the tooltip carries the whole word for when it cannot.

### lines 326-330

```python
self._form.insertRow(0, form_label, widget)
```

`insertRow(0, ...)`, the same mechanism :meth:`add_prose` uses. Asked for on 2026-09-02 for Regression's Input Tables: "the input tables sould start with download buttons not end wit them" -- a row of buttons that fills the fields below it reads as a footer when it sits under them.

## Section._source_row

### lines 437-439

```python
row.addStretch(1)
```

The heading's own text is painted by the button, under this layout; the stretch keeps the mark off it and against the trailing edge.

## Section._refresh_header_text

### lines 518-523

```python
for badge in dict.fromkeys(
```

Category names historically carried their own ``(BETA)`` or were simply named ``Beta``. Maturity styling then appended a second ``· BETA`` badge, producing ``BETA · BETA``. Keep the original title for configuration lookups, but render one badge. Both spellings are stripped: a translated caption can still carry the English badge if the catalog left that word alone.

## Section._sync_header_minimum

### lines 624-626

```python
if self._header.minimumHeight() != wanted:
```

Guarded: `setMinimumHeight` invalidates the layout, and this runs from inside style and font delivery, where an unconditional write would post a layout request on every polish.

## Section._refresh_tooltip

### lines 631-633

```python
"""Rebuild the header tooltip, adding the caution text off stable.
```

Stable is the normal case, so preserve existing curated tooltips byte-for-byte. Beta/alpha need the caution text because their colour carries information the old tooltip did not.

## Section._on_toggle

### lines 709-710  _(unsure)_

```python
self._body.setVisible(self._expanded)
```

A section outside a scroll area -- a dialog, a test. Nothing to confine, and nothing that could flicker past the section.

### lines 735-738

```python
if saved_policy is not None:
```

Restored while updates are still off, so putting the policy back cannot itself be a visible step. The layout has settled by now, so `AsNeeded` reaches the same answer the pre-resolution did.
