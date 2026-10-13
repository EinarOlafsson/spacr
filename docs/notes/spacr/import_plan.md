# Notes from `spacr/import_plan.py`

Prose lifted out of `spacr/import_plan.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [plan](#plan) (1 entry)
- [for_get_regex](#for_get_regex) (1 entry)

## plan

### lines 229-230

```python
values: Dict[str, str] = {}
```

THE ROLE WINS OVER THE GROUP NAME, because the dropdown is what the user actually said and the group name may be `g1`.

## for_get_regex

### lines 287-290

```python
match = _TRAILING_EXTENSION.search(text)
```

ANY ALTERNATION OF IMAGE EXTENSIONS, not one exact spelling of it. `auto_detect_regex` returns the full five -- `(?:tif|tiff|png|jpg|jpeg)` while the bundled YOKOGAWA pattern carries `(?:tif|tiff)`, and a literal comparison against one of them silently left the other on.
