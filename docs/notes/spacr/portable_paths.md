# Notes from `spacr/portable_paths.py`

Prose lifted out of `spacr/portable_paths.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [candidate_roots](#candidate_roots) (1 entry)
- [_suffixes](#_suffixes) (1 entry)
- [reroot_column](#reroot_column) (1 entry)

## candidate_roots

### lines 62-64

```python
here = parent
```

Climb unconditionally for the first step when the folder is a known sibling of `data/`; otherwise still climb, because a caller may hand us a screen root whose plates are one level down.

## _suffixes

### lines 80-81  _(unsure)_

```python
for start in range(len(parts) - 1):
```

Longest first: the more of the recorded structure that matches, the less chance the match is a coincidence.

## reroot_column

### lines 228-231

```python
unresolvable: set = set()
```

Folders already searched and NOT found. Every crop of a well shares a folder, so without this a root that resolves nothing costs one full search per ROW -- measured at 8.2s over 60,816 rows against 0.6s when a prefix is found. With it, the same case costs one search per folder.
