# Notes from `spacr/qt/widgets/model_share.py`

Prose lifted out of `spacr/qt/widgets/model_share.py` by `tools/extract_source_notes.py`.
Ordinary comments move here; tool directives and published attribute documentation stay in the module. The path mirrors the source path, which is how its reasons are found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [target_repo](#target_repo) (1 entry)
- [share](#share) (1 entry)

## target_repo

### lines 317-318

```python
return SHARE_REPO, True
```

A collaborator's write access cannot be read back reliably; try it and let the upload itself be the test.

## share

### lines 345-346  _(unsure)_

```python
folder = "staging/" + slugify(fields.get("display_name") or filename)
```

Into staging/ as well: unvetted is unvetted however it arrived, and the community listing looks in exactly one place.
