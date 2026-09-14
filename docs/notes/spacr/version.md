# Notes from `spacr/version.py`

Prose lifted out of `spacr/version.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Module level

### lines 15-16  _(unsure)_

```python
_PACKAGE_CANDIDATES = ("spacr", "spacr-nightly")
```

Prefer the canonical `spacr` distribution. `spacr-nightly` stays as a fallback in case a very old install still uses that name.
