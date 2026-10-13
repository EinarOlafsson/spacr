# Notes from `spacr/macro.py`

Prose lifted out of `spacr/macro.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [entry_for](#entry_for) (1 entry)
- [_registered_entry_text](#_registered_entry_text) (1 entry)
- [_RunIdCapture](#_runidcapture) (1 entry)
- [finish_recording](#finish_recording) (1 entry)
- [_link](#_link) (1 entry)
- [_Threader.__init__](#_threader__init__) (1 entry)
- [render](#render) (1 entry)
- [MacroError](#macroerror) (1 entry)

## entry_for

### lines 208-210  _(unsure)_

```python
def entry_for(module: str) -> Tuple[str, str]:
```

What a module key actually runs

## _registered_entry_text

### lines 265-266  _(unsure)_

```python
return ""
```

No Qt in this process. A headless recorder is a supported install, and the shipped table above already answered for every built-in.

## _RunIdCapture

### lines 604-606  _(unsure)_

```python
class _RunIdCapture(logging.Handler):
```

Recording — the two calls run_journal.open_run makes

## finish_recording

### lines 761-762  _(unsure)_

```python
LOG.exception("could not record the macro for %s", recording.module)
```

The script is a record of the run, not the run. Losing it is worth a log line and nothing else.

## _link

### lines 879-881  _(unsure)_

```python
def _link(previous: MacroStep, step: MacroStep) -> str:
```

Chaining — when two runs belong in one script

## _Threader.__init__

### line 1036  _(unsure)_

```python
self.roots = sorted(
```

Longest first, so a nested project does not lose to its parent.

## render

### lines 1239-1244

```python
blocks = [_render_settings(step, name, threader)
```

Everything that *uses* the project constants is rendered before the import block is built, though both are emitted later: rendering is what decides whether os.path.join appears, and therefore whether the script needs `import os`. Get that order wrong and a macro whose only joined path is in the MACRO record raises NameError on line one — the exact failure mode this recorder exists to avoid.

## MacroError

### lines 1321-1323  _(unsure)_

```python
class MacroError(ValueError):
```

Reading one back — the seam the methods exporter uses
