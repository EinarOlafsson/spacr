# Notes from `spacr/qt/widgets/animation_zoom.py`

Prose lifted out of `spacr/qt/widgets/animation_zoom.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [content_mask](#content_mask) (1 entry)
- [ZoomedAnimation.chrome_mask](#zoomedanimationchrome_mask) (1 entry)
- [zoom_frames](#zoom_frames) (2 entries)

## content_mask

### lines 214-217

```python
lit = (
```

Three plane comparisons, not ``frame.max(axis=2) > level``: the reduction runs along the length-3 interleaved axis and measured ten times slower, which across every frame of every animation is the difference between a hover that stutters and one that does not.

## ZoomedAnimation.chrome_mask

### lines 329-330  _(unsure)_

```python
pad = max(3.0, FIELD_PAD * scale + 1.0)
```

The mask has to stay wider than the line after the downscale, or remnants of the well re-enter the measurement.

## zoom_frames

### lines 376-383

```python
erase = None if shows_field else chrome
```

Built once, not per frame: the mask is the same in every frame and rasterising a rounded rectangle twice per frame would dominate the load. It has to be the whole chrome mask — the ring *and* everything outside the well — because that is exactly what the content measurement discounts. Erasing only the ring leaves whatever the generator painted outside the well in the crop, and since a sliced well means the output carries no chrome mask at all, that leftover measures as content at the frame's full extent.

### lines 392-394

```python
cropped = Image.fromarray(source).crop(
```

Pillow's crop pads out-of-bounds regions with black, which is the animations' own background — so scaling content *down* needs no special case.
