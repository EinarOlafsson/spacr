# Notes from `spacr/qt/iconset.py`

Prose lifted out of `spacr/qt/iconset.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [icon](#icon) (1 entry)
- [_blend](#_blend) (1 entry)
- [veil_color](#veil_color) (1 entry)
- [_load_rgba](#_load_rgba) (1 entry)
- [reink](#reink) (4 entries)
- [_write_cached_icon](#_write_cached_icon) (2 entries)
- [themed_qimage](#themed_qimage) (1 entry)
- [Module level](#module-level) (15 entries)

## icon

### lines 141-144

```python
if resolved is not None and not resolved.isNull():
```

qtawesome can fail softly: if its application font was unavailable (or invalidated during a long-lived Qt process), it returns a null QIcon instead of raising. Treat that exactly like an import/render failure so newly-created toolbar buttons never become blank.

## _blend

### lines 164-166  _(unsure)_

```python
def _blend(a: str, b: str, t: float) -> str:
```

Bundled PNGs — re-inked per theme

## veil_color

### line 201, trailing

```python
return ink
```

palette can't do better; use full ink

## _load_rgba

### lines 227-229

```python
im.thumbnail((MAX_WORK_SIZE, MAX_WORK_SIZE),
```

reducing_gap makes PIL box-reduce by an integer factor first and only then resample — ~6x faster than going straight to LANCZOS from 3334 px, same result.

## reink

### lines 346-347  _(unsure)_

```python
new = np.broadcast_to(ink_rgb, rgb.shape).copy()
```

Pure mask: paint it flat in the theme ink and let alpha — including its antialiased edges — do all the shaping.

### lines 353-355

```python
ink_is_bright = mean_lum > 127.5
```

Which end of the tonal range is the drawing? A black glyph on transparent and a white glyph on transparent are the same picture with opposite polarity; guessing wrong inverts it.

### lines 365-366

```python
ink_l = relative_luminance(ink)
```

Keep the hue: scale each pixel toward the target luminance rather than replacing its colour outright.

### lines 372-374

```python
flat = lum < 1.0
```

Where the source was pure black there is no hue to preserve, so lift it to the neutral target instead of multiplying zero.

## _write_cached_icon

### lines 441-442

```python
tmp = path.with_suffix(".part")
```

Write-then-rename: a half-written PNG left by a crash or a full disk would be read as a corrupt icon on every later launch.

### lines 444-447

```python
Image.fromarray(array, "RGBA").save(tmp, format="PNG", optimize=False)
```

`format=` explicitly: PIL infers it from the extension, and the temp name ends in `.part`, which it does not recognise. Without this every write raised and the silent `except` below swallowed it -- a cache that logged nothing and stored nothing.

## themed_qimage

### line 504, trailing  _(unsure)_

```python
return img.copy()
```

detach from the numpy buffer

## Module level

### lines 681-682

```python
_NAME_TO_GLYPH = {
```

Semantic name → Font Awesome glyph. Keep names short + generic so callers don't have to think about the icon library.

### line 684  _(unsure)_

```python
"open":            "fa5s.folder-open",
```

File / source

### line 693  _(unsure)_

```python
"prev":            "fa5s.chevron-left",
```

Navigation

### line 700  _(unsure)_

```python
"brush":           "fa5s.paint-brush",
```

Editing

### lines 704-709

```python
"trash":           "fa5s.trash",
```

Run History's "Clear all" asked for `trash` and this table had no such name, so the one button that throws away recorded runs drew the puzzle piece -- the artwork every unfiled key draws. A solid bin, distinct from `erase_object`'s outlined `trash-alt`; the two never share a screen (one is a Make Masks canvas tool, the other is a Run History toolbar button), so the family resemblance costs nothing.

### lines 714-720

```python
"draw":            "fa5s.draw-polygon",
```

The three Make Masks canvas tools that had no glyph. Each fell through to the shared puzzle piece, so Draw, Divide and Recrop sat in one toolbar row wearing one picture -- three different edits a user cannot tell apart, which is worse than a row with a gap in it. A closed outline traced point by point, scissors through an object, and a crop frame: what each tool does to the field, not what family it belongs to.

### line 733  _(unsure)_

```python
"run":             "fa5s.play",
```

Actions

### line 743  _(unsure)_

```python
"mask":            "fa5s.mask",
```

App keys mirrored from app.py for the sidebar / tiles.

### lines 748-750

```python
"classify_merged": "fa5s.sitemap",
```

Distinct from every other module's glyph on purpose -- two identical icons in the sidebar is a worse affordance than a missing one. Checked by tests/test_classify_merged.py.

### lines 754-757

```python
"embeddings":      "fa5s.vector-square",
```

A vector per object, not a graph of them -- `project-diagram` is already UMAP's and two identical icons in the sidebar is a worse affordance than a missing one. `vector-square` reads as "the thing itself is a vector", which is what this module produces.

### lines 769-772

```python
"align":           "fa5s.border-all",
```

One square divided into four by its own seams: tiles registered into a single canvas. Align & Stitch renders this glyph rather than a bundled PNG (spacr.qt.app._FORCE_GLYPH) because no bundled artwork says "stitched mosaic".

### lines 774-781

```python
"import_images":   "fa5s.images",
```

Stacked photo frames: the module reads a FOLDER OF IMAGES off a microscope, and "images" is the whole thing that separates it from its host Import (`foreign.png`, a net funnelling into a down arrow) and from its two siblings on that fold strip -- Format Converter (`convert.png`, one field split raw/processed) and External Masks (`external_masks.png`, two crops arrowed into a folder). No bundled PNG says "image files", and without a line here the key fell through to the shared puzzle piece on both the dock row and the fold button.

### lines 784-796

```python
"ops":             "fa5s.braille",
```

A FIELD OF DOTS, NOT A BARCODE, and the distinction is the module. `map_barcodes` reads a barcode out of sequencing reads and draws one; OPS reads its code off the IMAGE, as a pattern of spots whose colour is one base per imaging cycle -- eleven cycles on the reference plate. Braille is the honest picture of that: a code carried by where the dots are rather than by a stripe, and it cannot be confused at 16 px with the barcode beside it on the same fold strip.

Not `layer-group`, which is Classify's and says "stacked" without saying what is stacked. Not `border-all`, which is Align & Stitch's one registered canvas -- OPS is that canvas ELEVEN TIMES OVER, and a glyph that says "mosaic" would be the neighbour's story, not this module's.

### lines 799-802

```python
"data_manager":    "fa5s.hdd",
```

Stacked platters: the app is about what a project weighs on disk and what of it can safely go. Without an entry here a new key falls back to the shared puzzle piece, which is artwork every unfiled app draws — indistinguishable tiles on Home.

### lines 804-816

```python
}
```

DELIBERATELY ABSENT: `regression_diagnostics`. It is the one key of the four instruction 355 measured on the fallback that is still there, and it is left there on purpose. Regression Diagnostics is residual-versus-fitted, scale-location, QQ, leverage and Cook's distance (see spacr/regression_diagnostics.py), so the mark that names it is a scatter about a zero line with one point flagged. Nothing bundled draws that and is free -- `outliers.png` is the closest and is the live Outliers QC module's own mark, so taking it would make two modules one picture -- and no FA5 glyph draws it; `stethoscope` and `heartbeat` say "diagnostics" the way a gear says "settings", which is the substitution instruction 355 rules out. A wrong-but-present icon is worse than the fallback, because the fallback at least reads as "nobody has chosen one yet".
