# Notes from `spacr/napari_bridge.py`

Prose lifted out of `spacr/napari_bridge.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [napari_available](#napari_available) (1 entry)
- [layer_specs](#layer_specs) (1 entry)
- [write_back](#write_back) (1 entry)

## napari_available

### line 198, trailing

```python
except (ImportError, ValueError):
```

a broken meta path

## layer_specs

### lines 324-331

```python
specs.append({"kind": "labels", "data": np.array(handoff.mask, copy=True),
```

A COPY, and this is not defensive tidiness. napari's brush edits the array it was handed IN PLACE, so handing over `handoff.mask` itself would mean the "before" spaCR is holding gets painted on too — and the diff in `write_back` would then be uniformly zero, silently, for every correction ever made through this bridge. The dtype is left alone: napari's labels layer takes any integer dtype, so there is nothing to convert on the way out; the conversion that matters is on the way back, in `to_spacr_mask`.

## write_back

### lines 625-629

```python
artifact = str(save_mask(target, mask))
```

The ledger goes beside the file that was actually written, not beside the path that was asked for. `save_mask` resolves a bare stem to `foo.tif`, and `log_path_for` keys on the full name including the extension -- so writing the ledger for `foo` would leave a second, orphaned history next to the one the brush writes for `foo.tif`.
