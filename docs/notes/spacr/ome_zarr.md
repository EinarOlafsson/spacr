# Notes from `spacr/ome_zarr.py`

Prose lifted out of `spacr/ome_zarr.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [require_codec](#require_codec) (2 entries)
- [spacr_unit_to_ngff](#spacr_unit_to_ngff) (1 entry)
- [_axis_from_ngff](#_axis_from_ngff) (2 entries)
- [spacing_from_axes](#spacing_from_axes) (1 entry)
- [_read_chunk_bytes](#_read_chunk_bytes) (1 entry)
- [_fill_value](#_fill_value) (1 entry)
- [_ZarrArray.read_region](#_zarrarrayread_region) (1 entry)
- [OmeZarrImage.spacing_at](#omezarrimagespacing_at) (1 entry)
- [OmeZarrImage.read](#omezarrimageread) (1 entry)
- [read_ome_zarr](#read_ome_zarr) (3 entries)
- [_downsample_axis](#_downsample_axis) (2 entries)
- [_pyramid](#_pyramid) (1 entry)
- [_write_zarr_v2_array](#_write_zarr_v2_array) (1 entry)
- [write_ome_zarr](#write_ome_zarr) (3 entries)
- [_spacr_version](#_spacr_version) (1 entry)

## require_codec

### line 368, trailing

```python
try:
```

zstd only exists in the stdlib on 3.14+

### line 372, trailing  _(unsure)_

```python
except Exception:
```

an empty buffer is not valid input; fine

## spacr_unit_to_ngff

### line 614

```python
if text.lower() in NGFF_SPACE_UNITS or text.lower() in NGFF_TIME_UNITS:
```

A caller who already holds the NGFF name is not wrong; accept it.

## _axis_from_ngff

### line 783, trailing

```python
if isinstance(entry, str):
```

NGFF 0.3 wrote bare names

### lines 794-795

```python
kind = entry.get("type") or _TYPE_BY_AXIS_NAME.get(str(name).lower(),
```

`type` is SHOULD, not MUST, in 0.4, and files in the wild omit it. The name carries the answer for every axis NGFF actually defines.

## spacing_from_axes

### lines 840-842

```python
raise OmeZarrError(
```

Spacing's own refusals (a zero voxel size, a duplicated axis name) are the right refusals; they just need to say which file they are about, since the caller asked about a path, not about a Spacing.

## _read_chunk_bytes

### lines 993-995  _(unsure)_

```python
def _read_chunk_bytes(path: Path) -> Optional[bytes]:
```

The zarr v2 chunk layer, in pure Python

## _fill_value

### line 1049, trailing  _(unsure)_

```python
try:
```

base64, the spec's escape hatch for raw bits

## _ZarrArray.read_region

### line 1264, trailing  _(unsure)_

```python
return out
```

empty selection, no I/O

## OmeZarrImage.spacing_at

### line 1495, trailing  _(unsure)_

```python
base = self.spacing
```

validates units once

## OmeZarrImage.read

### lines 1703-1706

```python
return _ZarrArray.open(store).read_region(box)
```

Re-reading the level's `.zarray` here rather than caching a handle on the image keeps this dataclass frozen and picklable — it can cross into a worker process — at the cost of one small JSON read per call, against however many chunk decodes follow it.

## read_ome_zarr

### line 1925

```python
arrays = []
```

Levels first: their rank is what an absent `axes` has to be inferred from.

### lines 1939-1940

```python
axes = [Axis(name=n, type=_TYPE_BY_AXIS_NAME.get(n, AXIS_SPACE))
```

0.1-0.3 had no `axes`. The canonical tczyx tail is what those files meant, and inferring it beats refusing to open old data.

### lines 1969-1972

```python
spacing_from_axes(axes)
```

Validate the units HERE, not on first use of `.spacing`. The code path that reads shapes and never asks for a spacing is exactly the one that would carry an untranslatable unit all the way into a measurement without anything raising, so the refusal has to happen at the door.

## _downsample_axis

### lines 2060-2063

```python
dtype = array.dtype
```

Captured before anything else touches the array: np.concatenate below returns NATIVE byte order, so `array.dtype` at the end of this function is not the dtype that came in. A silently byte-swapped pyramid level reads back as different numbers, not as an error.

### lines 2067-2069

```python
edge = array[tuple(slice(None) if i != axis else slice(n - 1, n)
```

Duplicate the last element so the tail block is a full pair. Its mean is that element, which is exactly the partial-block mean — the padding is arithmetic bookkeeping, not an invented sample.

## _pyramid

### line 2090, trailing  _(unsure)_

```python
break
```

every downsampled axis is already 1; stop early

## _write_zarr_v2_array

### line 2142, trailing  _(unsure)_

```python
continue
```

an unwritten chunk reads back as fill_value

## write_ome_zarr

### lines 2327-2328  _(unsure)_

```python
shift = 0.0 if method == "stride" else 1.0
```

Block mean: element 0 of level k covers level-0 elements [0, 2^k), whose centre is (2^k - 1)/2. Stride: element 0 IS level-0 element 0.

### lines 2330-2332

```python
translation = [a.translate + a.scale * ((f - 1.0) / 2.0 * shift)
```

scale * ((f - 1) / 2) rather than scale * (f - 1) / 2: (f - 1) / 2 is exact in binary for a power-of-two f, so this is one rounding instead of two.

### lines 2349-2351

```python
"type": "local mean" if method == "mean" else "nearest (stride)",
```

0.4 defines `type` and `metadata` on a multiscale for exactly this: saying how the pyramid was built, so a reader knows whether the coarse levels can be trusted for a measurement (they cannot).

## _spacr_version

### lines 2389-2392

```python
try:
```

A checkout on PYTHONPATH may have no installed distribution metadata, but it still carries the release helper's synchronized version literal. Metadata remains authoritative for installed packages; this is only the source-tree fallback used when that lookup explicitly found none.
