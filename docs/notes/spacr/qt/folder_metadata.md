# Notes from `spacr/qt/folder_metadata.py`

Prose lifted out of `spacr/qt/folder_metadata.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [Module level](#module-level) (2 entries)
- [_classify](#_classify) (1 entry)
- [detect_folder_metadata](#detect_folder_metadata) (3 entries)
- [iter_image_files](#iter_image_files) (1 entry)
- [NameMapping](#namemapping) (1 entry)
- [assign_missing_fields](#assign_missing_fields) (1 entry)

## Module level

### line 45, trailing  _(unsure)_

```python
WELL_COLS = list(range(1, 25))
```

1..24

### lines 67-75

```python
_WELL_RX     = re.compile(r"^[A-Z]{1,2}\d{1,3}$", re.I)
```

Heuristic recognisers for common folder tokens.

The well pattern is one or two letters then the column digits — the same *shape* as ``plate_qc._WELL_RE`` and ``schema._WELL``, and deliberately so: those two modules already agree that a row runs ``A``…``Z``, ``AA``…``AF`` (a 1536 plate has 32 of them), and a third opinion here is how "is this a well?" comes to have two answers. It used to be ``[A-P]``, which is 16 rows — so every folder from ``Q01`` up, the whole bottom half of a 1536 plate, was not recognised as a well at all.

## _classify

### lines 83-91

```python
"""Name the metadata axis a filename token belongs to.
```

Order matters, and it matters more now that the well pattern reaches past P: the FIELD / CHANNEL / PLATE recognisers name their axis explicitly, so they are strictly more specific and are checked first — `F01`, `C01`, `ch1` and `plate2` still classify as themselves. What is left over (`Z01`, `S01`, `T01`) is genuinely ambiguous from a single token, and is read as a well, because on a plate that is what it is: row Z, row S, row T. A dataset that means a z slice, a site or a timepoint should spell it with a prefix the specific recognisers know.

## detect_folder_metadata

### line 137, trailing  _(unsure)_

```python
parts = list(rel.parts[:-1])
```

drop filename

### line 147  _(unsure)_

```python
sequences: Dict[Tuple[str, ...], List[Path]] = {}
```

Take the modal label sequence

### line 153  _(unsure)_

```python
chan_from_filename = any(
```

Is the leaf filename encoding the channel?

## iter_image_files

### lines 182-185

```python
if p.suffix.lower() in IMAGE_EXTS and p.is_file():
```

Suffix before ``is_file()``, deliberately: the suffix test is a string compare, ``is_file()`` is a stat syscall. On a plate folder with 100 000 entries the stats are most of the cost of the walk, and directories almost never carry an image extension.

## NameMapping

### lines 193-195  _(unsure)_

```python
@dataclass
```

Auto-generate missing well / field ids + write filename_map.csv

## assign_missing_fields

### line 265  _(unsure)_

```python
if not have_field:
```

Advance
