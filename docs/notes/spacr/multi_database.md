# Notes from `spacr/multi_database.py`

Prose lifted out of `spacr/multi_database.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [column_kinds](#column_kinds) (1 entry)
- [_plates](#_plates) (1 entry)
- [describe_merge](#describe_merge) (2 entries)
- [MergeDecision](#mergedecision) (1 entry)
- [read_merged](#read_merged) (6 entries)
- [_collision_message](#_collision_message) (1 entry)

## column_kinds

### lines 494-495  _(unsure)_

```python
if declared.strip():
```

SQLite's fifth rule: anything else declared is NUMERIC. Nothing declared at all stays unknown.

## _plates

### lines 554-556

```python
return sorted({canonical_plate_id(row[0])
```

Normalised, so the PLAN names a plate the same way the DATA will after `normalise_plate_ids`. A plan that says `pplate1` while the frame says `plate1` would make the collision check compare two vocabularies.

## describe_merge

### lines 630-631  _(unsure)_

```python
labels = list(source_labels(paths))
```

Named for the whole set at once -- see :func:`source_labels` for why a per-path rule gave every plate of a screen the same useless name.

### lines 664-667

```python
seen: Dict[Tuple[str, str], List[str]] = {}
```

THE CHECK THAT CHANGED. Keyed on (screen, plate), not on plate: two screens each owning a plate1 are two identities, and calling that a clash makes stacking two screens impossible. A plate repeated inside one screen is the original error and is untouched.

## MergeDecision

### lines 691-703

```python
@dataclass(frozen=True)
```

What the user decided, written down

Instruction 109: "Two databases that both contain a plate called plate1 do NOT silently merge those plates. The user is TOLD, and what they choose is RECORDED."

Telling them is the refusal and the plan. RECORDING is this: a merge that a user resolved by hand -- by dropping one of two colliding databases, say leaves no trace in the result, and six months later the frame cannot say which of the two plate1s it is. One appended JSON line per merge, in one place, is the smallest thing that answers that question afterwards.

## read_merged

### lines 928-931

```python
if cancelled is not None and cancelled():
```

BETWEEN SOURCES, NOT INSIDE ONE. A source is read by a single `read_sql_query`; interrupting that would leave a partial frame this function has no honest way to return, and the point of a cancel is that nothing half-made survives it.

### lines 938-947

```python
frame = tabular.read_database(
```

ONE READER. `tabular.read_database` is the door every spaCR read goes through, and it is what applies the vocabulary: canonical names, ONE column per metadata key (a `well` beside a `wellID` is collapsed and the disagreement counted), and the `pplate1` plate repair before anything keys on it. Case-folded, so it cannot produce a frame SQLite will refuse.

read_only, because a merge reads the user's measurement databases and must not be able to write to one; migrate=False follows from that and is what this call has always done.

### lines 954-956

```python
frame = schema.add_screen_column(
```

BEFORE the column filter, so a screen stored in only one source is not intersected away, and so an explicitly named screen reaches every row whether the database had the column or not.

### lines 977-979

```python
merged.attrs["rows_done"] = done
```

HOW FAR THIS CALL GOT, carried on the frame so a caller stacking several tables can continue the count without re-deriving it from row lengths that the column filter may already have changed.

### lines 981-982

```python
merged.attrs["dropped_columns"] = dropped
```

The set a caller is about to analyse, and the set they are not. Carried on the frame so it cannot be separated from the data it describes.

### lines 985-988

```python
merged.attrs["source_rows"] = dict(source_rows)
```

THE ANTI-POOLING EVIDENCE, carried with the data. Pooling two plates that share a name is the one failure here with no symptom, so the counts that would expose it travel on the frame rather than being recomputable only by going back to the files.

## _collision_message

### lines 1022-1028

```python
return (
```

NO 'qualify' IN THIS SENTENCE. It is still available to a caller who wants the plate id rewritten, and it is still the wrong thing to put in front of a user: `plate1` becoming `runA-plate1` makes the keys unique and hides which experiment a plate belongs to INSIDE its own id, where it can no longer be blocked on, tested for or coloured by. This message is what the Gate Editor and the Image UMAP show, so it names the resolutions that keep the experiment analysable.
