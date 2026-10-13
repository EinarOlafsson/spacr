# Notes from `spacr/resources/home/versions/_generators/render.py`

Prose lifted out of `spacr/resources/home/versions/_generators/render.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [audit](#audit) (1 entry)
- [render_one](#render_one) (1 entry)
- [_scroll_finding._wrapped](#_scroll_finding_wrapped) (1 entry)
- [write_markdown](#write_markdown) (1 entry)
- [self_check](#self_check) (1 entry)
- [measure_sidebar](#measure_sidebar) (3 entries)
- [main](#main) (1 entry)

## audit

### lines 32-34  _(unsure)_

```python
def audit(page) -> Dict[str, list]:
```

Layout audit — a variant with clipped text is a bug, not a variant

## render_one

### lines 111-113

```python
from PySide6.QtCore import QEvent
```

processEvents() does NOT drain DeferredDelete, so without this the thirty pages (and their few thousand child widgets) all stay alive for the whole run and each successive render gets slower.

## _scroll_finding._wrapped

### lines 395-398

```python
return textwrap.fill(sentence, width=72,
```

Re-wrapped to the width the surrounding hand-written findings use, continuation lines under the list item's hanging indent. Substituting one very long line into a numbered list turns the whole findings block into something nobody re-reads.

## write_markdown

### lines 451-454

```python
lines = [_INTRO.format(space_note=space_note, sidebar_h=sidebar_h,
```

n_apps is read, never typed: this document said "29 apps" for long enough that five more were registered without anyone noticing. The scrollbar tally is read too, for the same reason and out of the `reports` this function was already handed.

## self_check

### lines 503-505

```python
packed = ((arr[..., 0].astype(np.int32) << 16)
```

Pack RGB into one int32 before counting: np.unique(axis=0) over 1.3 M rows takes seconds per image, and there are ninety images.

## measure_sidebar

### lines 536-540

```python
for section in list(getattr(bar, "_section_headers", {})):
```

EVERY SECTION OPEN. The dock now starts with Core open and the rest collapsed, so its resting height says nothing about whether the navigation fits -- it fits because most of it is folded away. The height worth measuring is the one a user sees after opening the sections they work in, which is the fully expanded dock.

### lines 547-549

```python
scroll = getattr(bar, "_scroll", None)
```

Private attribute on purpose: there is no public accessor for the scrolled widget, and falling back to the outer layout keeps this working (with a smaller number) if the scroll area is ever removed.

### line 558

```python
return int(need), common.CANVAS_H - 26 - 24
```

900 window - 26 menu strip - 24 status bar

## main

### lines 609-610  _(unsure)_

```python
print("markdown:", write_markdown(specs, themes, reports, need, avail))
```

A partial run keeps the prose for every variant, but only the audit lines for the ones just rendered.
