# Notes from `spacr/qt/safespacr.py`

Prose lifted out of `spacr/qt/safespacr.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## main

### lines 27-30

```python
from .preferences import enable_safe_mode
```

BEFORE THE FIRST PREFERENCE IS READ, and before Qt is imported: the palette, the backdrop and the preloader all read preferences while the application is being built, so a flag set after that reaches none of them.

### lines 34-36

```python
os.environ.pop("SPACR_TIMING", None)
```

Not a preference, so `enable_safe_mode` cannot reach it: the timing instrumentation is chosen by the environment, and it patches the import machinery for the life of the process.

### lines 38-39  _(unsure)_

```python
os.environ["SPACR_NO_GL"] = "1"
```

A GL context is created before any Python of ours runs on the crashing path, so refusing it has to happen in the environment too.

### line 32

```python
os.environ["SPACR_NO_BACKDROP"] = "1"
```

The switch `spacr.qt.crash_recovery` already uses, for this process only and never saved. Safe mode's forced "ambient off" already gave the same answer, but only after `get_ambient_enabled` had read the stored animation to see whether it was None, and that read imports `spacr.qt.widgets.ambient` -- so safe mode still imported the backdrop module it never builds (measured 2026-09-15, 296). The environment variable is checked before anything else in that getter, so the module is never reached.

### lines 47-51

```python
argv = list(sys.argv[1:] if argv is None else argv)
```

AND NO FIRST-RUN SETUP. Reading preferences as defaults means "has this profile been set up" reads as "no", so safe mode greeted a long-standing user with the setup wizard -- in front of the settings they opened it to repair. The flag is the same one `spacr-server` uses.
