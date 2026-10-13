# Notes from `spacr/qt/widgets/folding_summary.py`

Prose lifted out of `spacr/qt/widgets/folding_summary.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [split_sections](#split_sections) (1 entry)
- [split_rows](#split_rows) (2 entries)
- [_RejectionHighlighter.highlightBlock](#_rejectionhighlighterhighlightblock) (1 entry)
- [FoldingSummaryView.__init__](#foldingsummaryview__init__) (1 entry)
- [FoldingSummaryView.copy_to_clipboard](#foldingsummaryviewcopy_to_clipboard) (1 entry)
- [FoldingSummaryView._table](#foldingsummaryview_table) (2 entries)
- [FoldingSummaryView._block](#foldingsummaryview_block) (2 entries)
- [FoldingSummaryView._rebuild](#foldingsummaryview_rebuild) (3 entries)

## split_sections

### lines 51-53

```python
if len(rule) != len(title):
```

THE RULE MUST MATCH THE TITLE'S LENGTH. Without that a row of dashes drawn by statsmodels turns the line above it into a heading, and the summary folds itself into nonsense.

## split_rows

### line 92  _(unsure)_

```python
match = _ROW.match(line)
```

A row is "  label" then at least two spaces then the text.

### lines 104-105  _(unsure)_

```python
rows[-1][1] = (rows[-1][1] + " " + line[lead:].strip()).strip()
```

A continuation of the value above: joined, so the panel can re-wrap it to whatever width it actually has.

## _RejectionHighlighter.highlightBlock

### lines 154-157

```python
self.setFormat(0, len(line), self._format)
```

THE WHOLE LINE, not the matched word. "REJECTED at 0.05" is the verdict on the sentence it sits in, and colouring three words inside a grey line reads as emphasis rather than as a state.

## FoldingSummaryView.__init__

### lines 192-212

```python
self._actions = QWidget(self._body)
```

SAVE AND COPY, because a summary a reader cannot take with them is a summary they retype. Asked for 2026-08-19: "i should be able to click a button to save them and also copy them with the overlapping squares icon". A WIDGET, NOT A BARE LAYOUT, and that is the whole bug.

Reported 2026-08-20: "in the summary section there is a giant save button in the background that can only be pressed on the side of the summay text, presumably because the text is in front and blocking."

`_clear` empties the body layout with takeAt and deletes what it finds -- but it only finds WIDGETS. A bare QHBoxLayout was taken out and dropped, while the buttons inside it stayed children of `_body` with nothing laying them out: still visible, still clickable, stuck at whatever geometry they last had, and painted UNDER the sections added afterwards. Hence a button in the background reachable only where no text covered it.

Held as one widget, it is taken out and put back like everything else, and there is nothing left behind to strand.

## FoldingSummaryView.copy_to_clipboard

### lines 290-292

```python
return False
```

No clipboard on this platform or in this session. Declining is the whole behaviour: a copy button that raises is worse than one that does nothing.

## FoldingSummaryView._table

### lines 409-411

```python
view.viewport().setAutoFillBackground(False)
```

A SURFACE, not a slab and not a window onto the backdrop. See `_reading_surface` -- fully transparent put the animated background directly behind the type.

### lines 416-425

```python
try:
```

RED FOR A REJECTED ASSUMPTION, IN THE TABLE TOO (225). Most of a summary's rows arrive here rather than at `_block` -- anything shaped "label: value" is a row -- so a highlighter on the block path alone colours almost nothing. Found exactly that way: the highlighter worked and the assumptions were still grey, because they were never blocks.

Inline here rather than another highlighter: this is already HTML being built, and a second mechanism for one colour is a second thing to keep in step.

## FoldingSummaryView._block

### lines 474-476

```python
view.viewport().setAutoFillBackground(False)
```

The statsmodels summary is the longest thing in this panel and the hardest to read over a moving picture. Same surface as the tables, for the same reason.

### lines 481-483

```python
try:
```

HELD ON THE VIEW, or it is garbage collected the moment this function returns and highlights nothing -- silently, which is the only way a highlighter ever fails.

## FoldingSummaryView._rebuild

### lines 509-511

```python
self._layout.addWidget(self._block(self._text), 1)
```

NOTHING TO FOLD, so nothing is folded. The statsmodels summary has no spaCR headings and chopping it up by a guess would be worse than leaving it whole.

### lines 523-524

```python
expanded = heading.upper() == ANSWER_HEADING
```

THE VERDICT OPEN, EVERYTHING ELSE FOLDED -- the headings are then the outline the instruction asks for.

### lines 526-528

```python
content = self._table(rows) if rows else self._block(body)
```

A TABLE WHERE THE BODY IS ROWS, the plain block otherwise the statsmodels summary is column-aligned ASCII and re-laying it out would destroy the alignment it carries itself.
