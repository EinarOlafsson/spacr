# Notes from `spacr/curation_queue.py`

Prose lifted out of `spacr/curation_queue.py` by `tools/extract_source_notes.py`.
Ordinary comments move here; tool directives and published attribute documentation stay in the module. The path mirrors the source path, which is how its reasons are found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## _read_draft

### lines 938-940  _(unsure)_

```python
def _read_draft(item: QueueItem):
```

Reading drafts — the only place numpy is needed
