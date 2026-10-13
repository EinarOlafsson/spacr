# Notes from `spacr/install_cleanup.py`

Prose lifted out of `spacr/install_cleanup.py` by `tools/extract_source_notes.py`.
Ordinary comments move here; tool directives and published attribute documentation stay in the module. The path mirrors the source path, which is how its reasons are found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [Module level](#module-level) (2 entries)
- [_find_windows_online](#_find_windows_online) (1 entry)
- [_find_macos_apps](#_find_macos_apps) (1 entry)
- [_find_unix_online](#_find_unix_online) (2 entries)
- [_delete._writable_then_retry](#_delete_writable_then_retry) (1 entry)
- [run_update_sequence](#run_update_sequence) (1 entry)
- [_record_from_json](#_record_from_json) (1 entry)
- [_main](#_main) (2 entries)

## Module level

### lines 48-50

```python
_OLD_NAME = "Spa" + "CR"
```

The 1.5.0.1 to 1.5.0.4 installers capitalised the first letter of the folder, app bundle and Start Menu names. Built rather than written, so a scan for the mis-cased project name does not mistake it for prose.

### line 68

```python
_RMTREE_TAKES_ONEXC = sys.version_info >= (3, 12)
```

Python 3.12 renamed shutil.rmtree's error hook from onerror to onexc.

## _find_windows_online

### lines 698-699  _(unsure)_

```python
gone = next((p for p in named if p), defaults[0] if defaults else "")
```

Registered, but the folder is already gone: the Apps list entry and shortcuts of a half-removed copy still have to go.

## _find_macos_apps

### line 797  _(unsure)_

```python
support = supports[0] if supports and index == 0 else ""
```

The package's shared support folder belongs to the first bundle.

## _find_unix_online

### line 859  _(unsure)_

```python
named_by_launchers = []
```

A launcher names its root, which finds a copy put somewhere unusual.

### lines 874-876

```python
installer_only = markers[1:]
```

Any project can have a venv. A folder known only because a launcher names it must also hold what only the installer writes, or a launcher a user made for a project's own venv would make the project an old copy.

## _delete._writable_then_retry

### lines 1143-1145

```python
errors.append((failed_path, exc_info if isinstance(
```

os.open and os.close cannot be called again with a path alone; keep the error rmtree reported (onexc passes the exception, the older onerror an exc_info tuple).

## run_update_sequence

### lines 1328-1330  _(unsure)_

```python
def run_update_sequence(install: Callable[[], object], *,
```

Step 3: install, only after step 2

## _record_from_json

### lines 1400-1402  _(unsure)_

```python
def _record_from_json(data: Dict) -> InstallRecord:
```

Steps 2 and 3 after spaCR has closed

## _main

### lines 1789-1790  _(unsure)_

```python
machine = _Machine(environ=dict(os.environ), fs_root=args.root,
```

A sandboxed file system, for testing an installer: no real registry, package manager or running environment is consulted.

### line 1795  _(unsure)_

```python
machine.running_prefix = None
```

The interpreter running this is a tool, not the spaCR being replaced.
