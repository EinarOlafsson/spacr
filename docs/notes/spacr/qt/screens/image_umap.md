# Notes from `spacr/qt/screens/image_umap.py`

Prose lifted out of `spacr/qt/screens/image_umap.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Module level

### lines 37-42

```python
DATABASE_CANDIDATES: Tuple[Tuple[str, ...], ...] = (
```

What each of those two said as a TILE -- the name, the sentence and the maturity colour a button has to go on carrying once the row is dropped lives in `spacr.qt.screens.map_barcodes.FOLD_FALLBACK`, because `map_barcodes.fold_description` is what `install_fold_strip` restates these buttons through, and that is the only table it reads. A second copy stood here and nothing consulted it.
