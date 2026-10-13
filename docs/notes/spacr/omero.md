# Notes from `spacr/omero.py`

Prose lifted out of `spacr/omero.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [_missing_omero_extra](#_missing_omero_extra) (2 entries)
- [require_omero](#require_omero) (1 entry)
- [connect](#connect) (1 entry)
- [parse_object_ref](#parse_object_ref) (1 entry)
- [WellPosition](#wellposition) (1 entry)
- [_plate_index](#_plate_index) (1 entry)
- [omero_indices](#omero_indices) (1 entry)
- [well_from_image_name](#well_from_image_name) (1 entry)
- [PixelSize](#pixelsize) (1 entry)
- [pixel_size_from](#pixel_size_from) (2 entries)
- [_call](#_call) (2 entries)
- [ImageInfo](#imageinfo) (1 entry)
- [_write_planes](#_write_planes) (1 entry)
- [_write_sidecars](#_write_sidecars) (1 entry)
- [format_map_value](#format_map_value) (2 entries)
- [summarise_rows](#summarise_rows) (1 entry)

## _missing_omero_extra

### lines 382-383

```python
root = (getattr(exc, "name", None) or "").split(".", 1)[0]
```

ModuleNotFoundError sets `.name` to the module that was not found; a failed `from omero.gateway import BlitzGateway` sets it to the submodule.

### lines 387-388  _(unsure)_

```python
text = str(exc)
```

Import hooks and hand-raised ImportErrors leave `.name` unset, so fall back to the message text before giving up on the friendly path.

## require_omero

### lines 415-418

```python
origin = getattr(gateway, "__file__", "") or ""
```

A silent self-import would look exactly like a broken OMERO install, so it is checked rather than argued about. It cannot happen through normal absolute-import resolution; it can happen if something has put this package's own directory on sys.path ahead of site-packages.

## connect

### lines 786-787

```python
LOGGER.info("connecting to OMERO at %s", settings.describe())
```

Everything logged about a connection goes through `describe()`, which cannot contain a credential.

## parse_object_ref

### line 865

```python
raise OmeroIdError(f"{value!r} is not an OMERO object id.")
```

bool is an int; `True` is not an object id and must not become 1.

## WellPosition

### lines 939-941  _(unsure)_

```python
@dataclass(frozen=True)
```

The well mapping — OMERO (0-based row, column) -> spaCR keys

## _plate_index

### line 1017

```python
try:
```

A float row index is not a rounding problem, it is a wrong object.

## omero_indices

### line 1047, trailing  _(unsure)_

```python
except Exception as exc:
```

schema raises WellParseError

## well_from_image_name

### line 1131, trailing  _(unsure)_

```python
except Exception:
```

not a well; try the next token

## PixelSize

### lines 1138-1140  _(unsure)_

```python
@dataclass(frozen=True)
```

Pixel size — a length, not a number of micrometres

## pixel_size_from

### lines 1190-1192

```python
return PixelSize(float(length), "MICROMETER")
```

omero-py's `getPixelSizeX()` with no `units=` argument returns a float already converted to micrometres. Recording the unit it used is the whole point of this function.

### line 1201, trailing  _(unsure)_

```python
except Exception:
```

a stub without the call

## _call

### lines 1231-1233  _(unsure)_

```python
def _call(obj: Any, *names: str, default: Any = None) -> Any:
```

Walking containers (the adapter, kept as thin as it can be)

### line 1242, trailing  _(unsure)_

```python
except Exception:
```

the server said no

## ImageInfo

### lines 1313-1315  _(unsure)_

```python
@dataclass(frozen=True)
```

Listing / inspecting, without fetching pixels

## _write_planes

### line 1681  _(unsure)_

```python
plane = pixels.getPlane(plan.z - 1, plan.channel - 1, plan.t - 1)
```

OMERO's getPlane is (theZ, theC, theT), all 0-based.

## _write_sidecars

### lines 1700-1701

```python
payload = {
```

`settings.redacted()` rather than `settings`: this file is written into a project folder that gets copied, zipped and shared.

## format_map_value

### lines 2076-2077

```python
return "True" if bool(value) else "False"
```

np.bool_ is not a bool and does answer __index__, so without this it would render as '1'. The dtype kind is the exact question to ask.

### lines 2086-2089

```python
try:
```

numpy scalars and Decimal reach here: they behave like numbers but are instances of neither builtin type. `__index__` is the exact question "is this an integer?" and is what distinguishes np.int64 from np.float64 without a dtype lookup.

## summarise_rows

### lines 2263-2264

```python
summary[key] = None
```

Mixed text and numbers in one column: there is no honest summary, so say nothing rather than average half of it.
