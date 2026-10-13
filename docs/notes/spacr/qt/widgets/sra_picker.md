# Notes from `spacr/qt/widgets/sra_picker.py`

Prose lifted out of `spacr/qt/widgets/sra_picker.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [_FetchWorker.run](#_fetchworkerrun) (1 entry)
- [SraPicker.__init__](#srapicker__init__) (3 entries)
- [SraPicker._show](#srapicker_show) (1 entry)

## _FetchWorker.run

### lines 92-97

```python
emit_safely(self.finished_all, written, error)
```

`emit_safely`, because this is the LAST line of a QThread::run override. A fetch can outlive the dialog that started it -- that is most of why it is on a thread -- and emitting into a destroyed receiver raises RuntimeError, which leaving `run` turns into an abort of the whole application rather than a traceback. The progress emit above is already inside the try.

## SraPicker.__init__

### lines 124-128

```python
from ..preferences import scaled_px
```

SCALED, NOT RAW. 520 is a width measured at font scale 1.0, and the text inside it is not: at 2x the label "Reads from each file:" needs 253 px and had 177, and the German estimate line needed 522 in 511. A pixel constant that does not follow the font is a constant that is only right for one user.

### lines 151-153

```python
self._reads.setRange(1_000, 100_000_000)
```

A MILLION IS NOT THE CEILING the data has; it is the ceiling this control offers, because past it the download stops being a sample and the "whole file" tick is the honest way to ask for everything.

### lines 194-198

```python
self.adjustSize()
```

AS SMALL AS THE CONTENT NEEDS, once the list has been filled after, because the runs are what decide how tall it wants to be. Asked for on 2026-09-02 about the sibling dialog and applied here for the same reason: a window that opens taller than its contents is a window the user has to fix before reading it.

## SraPicker._show

### lines 216-220

```python
"""Fill the run list, everything ticked.
```

SIGNALS OFF WHILE POPULATING. setCheckState emits itemChanged, which recomputes the estimate -- and it fires while the item being built has no RunFile attached yet, so the estimate reads a None. Set the data first as well: belt and braces, because the order inside a Qt item constructor is not this file's to guarantee.
