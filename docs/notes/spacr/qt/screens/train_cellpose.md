# Notes from `spacr/qt/screens/train_cellpose.py`

Prose lifted out of `spacr/qt/screens/train_cellpose.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [CellposeWorkbenchScreen.__init__](#cellposeworkbenchscreen__init__) (1 entry)
- [CellposeWorkbenchScreen._sync_instruction](#cellposeworkbenchscreen_sync_instruction) (1 entry)

## CellposeWorkbenchScreen.__init__

### lines 193-200

```python
header = getattr(screen, "_header", None)
```

One masthead per page. A module's own is 30px of title under this page's own 30px of title, and the applying half no longer has a registry row for its to read a name or a description out of, so it would render "Cellpose_Masks" over nothing. The tab bar says which half you are on, the line under the title says what that half reads, and the API link on this page's header follows the visible tab (see `_sync_instruction`).

## CellposeWorkbenchScreen._sync_instruction

### lines 339-343

```python
help_label = getattr(self._header, "api_help", None)
```

One masthead serves two modules, so its help has to link to the one on screen. The app key moves rather than the URL alone: a later language change repoints the help from the key the label carries, and would otherwise send the reader to the other tab's documentation.
