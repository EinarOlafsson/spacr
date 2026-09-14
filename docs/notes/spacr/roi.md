# Notes from `spacr/roi.py`

Prose lifted out of `spacr/roi.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [RoiSet.__post_init__](#roiset__post_init__) (1 entry)
- [RoiRegionFilter._raster](#roiregionfilter_raster) (1 entry)
- [RoiRegionFilter._by_overlap](#roiregionfilter_by_overlap) (1 entry)
- [_env_entries](#_env_entries) (1 entry)
- [enable_roi_filter](#enable_roi_filter) (2 entries)

## RoiSet.__post_init__

### lines 293-296

```python
raise RoiError(
```

Refused rather than transposed for the caller: a mask is (Y, X) and the vertices are stored row-first, so this pair means the ROI would be rasterised on its side. It still draws a region, which is exactly why it has to raise.

## RoiRegionFilter._raster

### line 633  _(unsure)_

```python
def _raster(self, context, rois: Sequence[RegionOfInterest]) -> np.ndarray:
```

placing the ROI on the mask

## RoiRegionFilter._by_overlap

### lines 703-704

```python
covered = np.broadcast_to(inside, mask.shape).reshape(-1)
```

A 2-D ROI applies to every z of a 3-D mask: the polygon was drawn looking down the stack, so it names a column through it.

## _env_entries

### lines 719-721  _(unsure)_

```python
def _env_entries(value: str) -> list:
```

Enabling it — including in worker processes

## enable_roi_filter

### line 781, trailing  _(unsure)_

```python
RoiSet.load(roi_path)
```

fail here, not in a worker

### lines 798-802

```python
if HOOK_NAME not in [entry.name for entry in region_filter_hooks()]:
```

Consulting the registry runs the environment installers, which is how this process ends up with a hook tagged 'env' — the same tag a worker gets, and the one measure_crop's start-method warning knows not to shout about. The variable is only read once per process, so if it has already been read this does nothing and we install directly instead.
