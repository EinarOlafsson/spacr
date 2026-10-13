# Notes from `spacr/lineage.py`

Prose lifted out of `spacr/lineage.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [build_forest](#build_forest) (2 entries)
- [orphans](#orphans) (1 entry)
- [read_object_tables](#read_object_tables) (1 entry)

## build_forest

### line 320  _(unsure)_

```python
by_parent: Dict[Tuple[str, str], List[LineageNode]] = {}
```

(field, label) -> the children hanging off it, table by table.

### lines 330-331

```python
rows.sort(key=lambda r: _sort_label(r[schema.OBJECT_LABEL_KEY]))
```

Sorted by label so the tree is stable; `_object_label` normalises '7', 7 and 7.0 to one thing, and 'onone'/'omulti' to nothing.

## orphans

### lines 475-478

```python
loose.append({**row, "table": table, "parent_id": ""})
```

No link at all is a different fact from a broken link: the row never claimed a parent. Reported too, with an empty parent_id, because a pathogen with no cell_id is also a pathogen nothing will ever show.

## read_object_tables

### lines 563-565  _(unsure)_

```python
def read_object_tables(db_path: str,
```

Reading — sqlite, and no Qt
