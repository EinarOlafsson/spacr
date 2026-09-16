# Notes from `spacr/qt/widgets/collapsible_section.py`

Prose lifted out of `spacr/qt/widgets/collapsible_section.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [CollapsibleSection.__init__](#collapsiblesection__init__) (2 entries)
- [CollapsibleSection._apply](#collapsiblesection_apply) (1 entry)

## CollapsibleSection.__init__

### lines 65-90

```python
self._header.setStyleSheet(
```

GRAY, TRANSPARENT, ROUNDED, BLUE ON HOVER. Asked for 2026-08-19: "the black dropdowns should be gray and they should have rounded edges when hovered blue, and they should be transparent". A QToolButton with no rule of its own paints the palette's button colour as an opaque block, which is the "black categories" in both this panel and the measure tab. AND THE RESTING HEADING IS THE THEME'S FOREGROUND (198). Asked for 2026-08-21: "the text in the measurements tab for the sub categories should be white when they are not highlighted (in dark mode oposite in bright mode)."

It was `palette(mid)` at rest and `palette(text)` only when open which is backwards. THE UNHIGHLIGHTED STATE IS THE ONE A USER READS: on a tab with four folded sections at most one is open, and the rest are what they are scanning to decide where to go. Dimming them says "secondary" about the only thing on screen that is not.

`palette(text)`, NEVER A LITERAL. `#FFFFFF` here is the same fault instruction 178 removed from eleven figure call sites: it reads on one theme and vanishes on the other, and the author sees only the one they use. The palette answers "white or near-black" for whichever theme is live.

THE HIGHLIGHT IS STILL VISIBLE, and that is the condition on this change: hover keeps its blue wash and the fold arrow still turns. What no longer distinguishes the states is the text going away.

### lines 111-114

```python
self._open_minimum = max(content.minimumHeight(), 0)
```

The minimum the CONTENT wants, remembered before anything folds it. Restoring the section has to put back the height that made the panel usable, and once folded that number is no longer readable off the widget.

## CollapsibleSection._apply

### lines 176-178

```python
self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
```

BOTH BOUNDS. A minimum alone leaves the splitter free to hand the folded section the space it just gave up, which looks like the fold did nothing.
