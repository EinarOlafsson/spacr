# Notes from `spacr/convert.py`

Prose lifted out of `spacr/convert.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [Module level](#module-level) (1 entry)
- [normalise_well](#normalise_well) (1 entry)
- [plate_format_for_names](#plate_format_for_names) (1 entry)
- [assign_wells](#assign_wells) (2 entries)
- [Mapping.to_row](#mappingto_row) (1 entry)
- [_keys_for](#_keys_for) (1 entry)
- [_axes_dims](#_axes_dims) (2 entries)
- [plan](#plan) (3 entries)
- [_to_5d](#_to_5d) (3 entries)
- [_valid_converted_tiff](#_valid_converted_tiff) (1 entry)
- [convert](#convert) (4 entries)

## Module level

### lines 255-256

```python
_CHANNEL_TOKEN = re.compile(r'(?i)(?<=[_\-. ])(?:ch|channel|c|w)[_\-]?(\d{1,3})(?=$|[_\-. ])')
```

Filename token patterns. Each requires a separator in front so that a stem like ``BC1`` is not read as channel 1.

## normalise_well

### lines 457-458  _(unsure)_

```python
return None
```

Reads as a position (``ZZ99`` -> r702/c99) but no standard plate has it. See :func:`off_plate_reason`, which is what the plan says.

## plate_format_for_names

### lines 508-509

```python
needed = max(needed, schema.plate_format_for(row, column))
```

normalise_well is what produced these, so plate_format_for cannot come back None here.

## assign_wells

### lines 556-557  _(unsure)_

```python
limit = len(well_sequence(n_wells))
```

well_sequence validates the format, so a non-standard n_wells is a ConfigurationError naming the known formats, not a KeyError.

### lines 572-574

```python
assigned[name] = next(free)
```

plate_format_for_names sized the plate to len(unique), and every claimed id is one of the names being counted, so the sequence cannot run dry before the names do.

## Mapping.to_row

### lines 743-745

```python
prc = schema.KEY_SEPARATOR.join([str(self.plate), row_id, column_id])
```

Joined here rather than through schema.compose_prc: the plate token is a sanitised source folder name and must be allowed to be anything a conversion can produce, including a name schema would refuse.

## _keys_for

### line 1021, trailing  _(unsure)_

```python
else:
```

plate_well

## _axes_dims

### lines 1047-1049

```python
n_c = int(sizes['S'])
```

'S' is tifffile's "samples per pixel" — RGB-style interleaving. Treating those samples as separate spaCR channels is a decision, not a fact recorded in the file, so it gets said out loud.

### lines 1058-1060

```python
if len(unknown) == 1 and not n_c and shape[unknown[0][0]] <= 4:
```

Left-to-right: T, then Z. A single unknown axis of 4 or fewer planes reads as channels — the same heuristic io.py uses, but said out loud instead of applied in silence.

## plan

### lines 1377-1382

```python
synthetic = [(key, assigned[key])
```

A synthetic address is never handed out silently. A name that is a well keeps it (including Q01 and A25, which a 1536 plate has and a 384 does not); everything else is listed here by name, and a name that *looks* like a well but sits on no plate at all is a warning of its own — that is the case where a typo turns into a well id nobody can trace back.

### lines 1436-1439

```python
if z_index is not None and source.z > 1:
```

A filename z/t token means the stack is spread over files: the token is the output index. A file that ALSO holds planes internally is ambiguous, and guessing which one wins is how planes silently overwrite each other, so it is an error.

### line 1479, trailing  _(unsure)_

```python
else:
```

Z_MAX

## _to_5d

### lines 1592-1593  _(unsure)_

```python
for index in range(array.ndim - 1, -1, -1):
```

Collapse any axis this module does not model (mosaic, block, view) to its first element.

### line 1601  _(unsure)_

```python
array = np.take(array, 0, axis=index)
```

Both present: 'S' is interleaved samples of a channel.

### line 1612  _(unsure)_

```python
total = int(np.prod(array.shape[:-2])) if array.ndim > 2 else 1
```

Unknown axes: fall back to the counts the describer resolved.

## _valid_converted_tiff

### lines 1809-1814

```python
return False
```

This is a validity predicate used to decide whether a field may be resumed. tifffile has changed the public base class of ``TiffFileError`` across releases, so enumerating its exception hierarchy let truncated files escape in some supported environments. Any reader failure means the artifact is not valid enough to trust and must be rebuilt.

## convert

### lines 1891-1892

```python
completed_fields = set()
```

A JSON claim never outranks the artifact. Re-open TIFF headers before accepting a field, and re-queue it when one target is absent or corrupt.

### lines 1903-1904  _(unsure)_

```python
by_source: Dict[Tuple[str, int], List[Mapping]] = {}
```

One read per (file, series): a six-scene CZI is opened once, not six times, and its scenes still land in six different fields.

### lines 1956-1957  _(unsure)_

```python
for field_id in {_conversion_field(mapping) for mapping in group}:
```

Mark only whole fields. A field spanning several source files is not accepted until every planned channel/z/t target validates.

### lines 1970-1972

```python
run.finalize(artifact=result.map_path)
```

Stamp the map itself: a conversion_map.csv that lists 380 of 384 wells looks exactly like a 380-well experiment until the sidecar says otherwise.
