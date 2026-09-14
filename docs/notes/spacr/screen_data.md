# Notes from `spacr/screen_data.py`

Prose lifted out of `spacr/screen_data.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## published_archives

### line 142  _(unsure)_

```python
try:
```

Older huggingface_hub has no timeout on this call.
