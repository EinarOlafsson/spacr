# Notes from `spacr/qt/ai/github_auth.py`

Prose lifted out of `spacr/qt/ai/github_auth.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [Module level](#module-level) (2 entries)
- [resolve_token](#resolve_token) (1 entry)

## Module level

### lines 35-39

```python
_REAL_HTTP_OPEN = urllib.request.urlopen
```

The HTTP seam used by every GitHub API request.  Production leaves this at the real stdlib transport.  Offline tests replace the seam itself, which is materially different from setting an environment variable that says a real write is allowed: the replacement is process-local, cannot be inherited by a subprocess, and any late teardown call still lands in the fake transport.

### lines 129-131

```python
_BEARER_RE = re.compile(r"(?i)(bearer\s+)[^\s'\"]{8,}")
```

Issue creation

## resolve_token

### lines 105-106

```python
tok = get_stored_token()
```

Calling this also erases a token left by an older spaCR build. The process-only value exists for API injection, never installer/UI login.
