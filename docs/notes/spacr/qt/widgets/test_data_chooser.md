# Notes from `spacr/qt/widgets/test_data_chooser.py`

Prose lifted out of `spacr/qt/widgets/test_data_chooser.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [TestDataChooser.__init__](#testdatachooser__init__) (4 entries)
- [TestDataChooser._size_the_description_pane](#testdatachooser_size_the_description_pane) (1 entry)
- [TestDataChooser.resizeEvent](#testdatachooserresizeevent) (1 entry)

## TestDataChooser.__init__

### lines 90-93

```python
self.setMinimumWidth(self.DIALOG_WIDTH)
```

SET BEFORE THE PANE IS MEASURED, because the measurement asks how tall the longest description is AT A GIVEN WIDTH. Measuring at one width and displaying at another is what left the pane sized for a column it never had.

### lines 108-110

```python
button.setToolTip(description)
```

The tooltip stays as well as the pane. The pane is the better surface, but a tooltip is what a user reaches for by habit and what the accessibility tree reads.

### lines 122-133

```python
self._size_the_description_pane()
```

A HEIGHT THAT CANNOT CLIP, so the dialog does not resize under the pointer as the text changes length -- the buttons would move away from the cursor hovering them -- AND no description is cut off, which a fixed 110 px did to both routes.

MEASURED, NOT CHOSEN. 110 fit whatever the descriptions said when it was written; both routes are two paragraphs now and overflowed it, and a larger font scale or a longer locale overflows any constant. `_size_the_description_pane` asks the font how tall the LONGEST of them is at the pane's own width, so the pane is stable under the pointer and stays right after a translation, a font-scale change, or a new route being added to ROUTES.

### lines 145-148

```python
self.adjustSize()
```

AS SMALL AS IT CAN BE WHILE FITTING THE TEXT. Without this the dialog opened at 509 px tall against a layout that wanted 271: nothing had asked it to be that size, and nothing had asked it not to be. `adjustSize` is the ask.

## TestDataChooser._size_the_description_pane

### lines 200-201  _(unsure)_

```python
tallest += metrics.lineSpacing()
```

A line of slack: boundingRect measures the ink, and a descender on the last line sits below the box it reports.

## TestDataChooser.resizeEvent

### lines 237-238

```python
self._laid_out = True
```

A RESIZE IS THE PROOF a layout pass has happened, so from here the pane's own width is the real one and is what to measure against.
