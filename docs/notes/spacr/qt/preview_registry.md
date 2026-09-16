# Notes from `spacr/qt/preview_registry.py`

Prose lifted out of `spacr/qt/preview_registry.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [Module level](#module-level) (1 entry)
- [_attach](#_attach) (1 entry)

## Module level

### lines 88-98

```python
"cellpose_masks": PreviewSpec(
```

attached through this seam

Both of these run Cellpose over one field and are judged entirely by whether the mask came out right, which is exactly the question the Mask panel answers. Their settings even share its names — the panel reads `diameter`, `flow_threshold` and `CP_prob` straight out of the dict, which is what makes reuse honest rather than approximate.

The reverse direction does need translating: the panel speaks Mask's per-compartment names, and `cell_diameter` means nothing to a module that has one object type and calls it `diameter`.

## _attach

### lines 339-340

```python
toggle.setParent(screen)
```

No strip on this screen — put the toggle above the card so the preview is still reachable rather than permanently hidden.
