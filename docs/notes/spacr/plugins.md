# Notes from `spacr/plugins.py`

Prose lifted out of `spacr/plugins.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## _installed_sources

### lines 440-444

```python
from importlib import metadata
```

Importing the metadata machinery costs more than the rest of this dependency-light SDK. Keep the documented SPACR_DISABLE_PLUGINS path a true opt-out: _build_registry() returns before reaching this generator, so a headless CLI that disables plugins never imports or scans package metadata at all.
