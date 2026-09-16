# Notes from `spacr/qt/imagery.py`

Prose lifted out of `spacr/qt/imagery.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [Module level](#module-level) (3 entries)
- [_srgb_to_linear_lut](#_srgb_to_linear_lut) (1 entry)
- [_open_master](#_open_master) (2 entries)
- [_probe](#_probe) (1 entry)
- [cache_name](#cache_name) (1 entry)
- [background_path](#background_path) (1 entry)
- [master_array](#master_array) (1 entry)
- [solve_image_file](#solve_image_file) (1 entry)
- [build_master](#build_master) (1 entry)

## Module level

### lines 146-159

```python
MASTERS: Dict[str, dict] = {
```

The registry

``source``      original filename, for :func:`build_masters`

``file``        the shipped, cropped, dimmed master ``theme``       which theme's palette this wallpaper is judged against ``source_crop`` (x0, y0, x1, y1) fractions kept from the original — this is where burned-in annotation is removed ``focus``       vertical centre of the aspect crop, as a fraction ``title``       shown in Preferences ``annotation``  measured bounds of burned-in annotation in the original, when it has any. ``source_crop`` must not intersect it; the test suite checks that arithmetic rather than trusting the comment.

### lines 173-175

```python
"source_crop": (0.03, 0.025, 0.97, 0.90),
```

Bottom 10 % dropped: that is where the burned-in "5 um" scale bar sits. The 3 % trim on the other three edges removes the frame's darker border rows.

### lines 214-216  _(unsure)_

```python
_DECODES = 0
```

Decode accounting — the performance claim, made assertable

## _srgb_to_linear_lut

### lines 281-283  _(unsure)_

```python
def _srgb_to_linear_lut() -> np.ndarray:
```

Colour maths — WCAG luminance over numpy arrays

## _open_master

### line 521, trailing  _(unsure)_

```python
Image.MAX_IMAGE_PIXELS = None
```

these are legitimately huge

### line 527, trailing  _(unsure)_

```python
pass
```

PNG has no draft mode; fine

## _probe

### lines 533-535  _(unsure)_

```python
def _probe(image, long_edge: int = 480) -> np.ndarray:
```

Rendering a background for one screen size

## cache_name

### lines 591-593  _(unsure)_

```python
def cache_name(key: str, width: int, height: int) -> str:
```

Disk cache — same directory and contract as the generated sky

## background_path

### lines 659-663

```python
width, height = screen_size()
```

`screen_size` is what applies the MIN_BACKGROUND floor — the stylesheet centres the image without repeating it, so a background narrower than the window would letterbox into bands of flat colour. An explicit size is honoured as given, which is what makes this testable at small sizes.

## master_array

### lines 696-698

```python
def master_array(key: str) -> Optional[np.ndarray]:
```

Measured legibility

## solve_image_file

### line 784, trailing  _(unsure)_

```python
Image.MAX_IMAGE_PIXELS = None
```

NASA masters are huge

## build_master

### lines 834-836

```python
scale = min(1.0, out_w / box_w, out_h / box_h)
```

Never upscale here. ``cell.png`` is only 2048 px wide, and inventing pixels at build time would bake a soft image into the wheel for every user including the ones whose screen is 1920.
