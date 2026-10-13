# Notes from `spacr/chaining.py`

Prose lifted out of `spacr/chaining.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [Module level](#module-level) (3 entries)
- [PinStore._load](#pinstore_load) (1 entry)
- [PinStore._save](#pinstore_save) (1 entry)
- [_artifact_root](#_artifact_root) (1 entry)
- [chained_inputs](#chained_inputs) (1 entry)
- [resolve_settings](#resolve_settings) (2 entries)
- [next_steps](#next_steps) (2 entries)
- [resolve_drop](#resolve_drop) (5 entries)

## Module level

### lines 188-190  _(unsure)_

```python
PIN_STATE_ENV = "SPACR_CHAINING_PINS"
```

The pin store — which paths the user typed by hand

### lines 374-376  _(unsure)_

```python
ROOT = "root"
```

Bindings — which settings key an input port fills

### lines 1184-1198

```python
DB_SUFFIXES: Tuple[str, ...] = (".db", ".sqlite", ".sqlite3")
```

Layout-aware drops

A dropped folder and an auto-chained one have to arrive at the same answer. Two answers to "where is the database" is how a screen and the run it launches come to disagree, so the drop path does not re-derive anything: it asks the registry through :func:`chained_inputs` exactly as auto-chaining does, and only when the registry has nothing does it fall back to the declared layout in :data:`spacr.ports.PORTS`.

The fallback is the difference between the two, and it is additive: where auto-chaining leaves a field empty because no run was ever registered, a drop still fills it from the folder the user just pointed at. Where the registry *does* have a row, both produce the same string.

## PinStore._load

### lines 248-250

```python
raw = {}
```

No file, an unreadable one, or one someone hand-edited into invalid JSON. A lost pin costs one re-typed path; refusing to open the screen would cost the whole session.

## PinStore._save

### lines 275-276  _(unsure)_

```python
pass
```

A read-only home, a full disk, a locked file on Windows. The pin still holds for this session; it simply will not outlive it.

## _artifact_root

### lines 478-480  _(unsure)_

```python
def _artifact_root(artifact: Artifact, port: Port) -> str:
```

Resolving an artifact into a settings value

## chained_inputs

### lines 657-659

```python
stores: Dict[str, Optional[Registry]] = {}
```

One registry per root for the whole call: opening one runs the schema DDL, and a module with three ports across three candidate roots would otherwise pay for nine of them on every keystroke.

## resolve_settings

### lines 785-792

```python
bound = {binding_for(spec.key, port).setting for port in spec.consumes}
```

A pin is restored BEFORE the lookup, so the candidate roots include the plate the user pinned. Chaining Measure's crops off a pinned src is the whole point of pinning it.

Only the keys an input port binds to are consulted: a pin exists because auto-chaining offered to fill that key, and restoring one for a key nothing chains would let a stale state file quietly override a setting the user is looking at.

### lines 825-826

```python
for key in pinned_values:
```

A pin with nothing to chain against is still held: the interface should say the value is the user's, not that it came from a run.

## next_steps

### lines 1152-1153  _(unsure)_

```python
store = _registry_for(resolved_root, registry)
```

Opened once for the whole answer: every successor is checked against the same project, and each open runs the registry's schema DDL.

### lines 1169-1171

```python
seed[source_key(candidate)] = resolved_root
```

No registry row yet — the successor still runs in the project the finished module ran in, and saying so is better than handing over an empty screen.

## resolve_drop

### lines 1681-1683

```python
targets.append(DropTarget(
```

A screen that takes the project itself. There is no port to resolve and nothing to look up: the answer is the folder the layout walk arrived at, which is the point of having walked it.

### lines 1689-1691

```python
binding = (binding_for(key, port) if declared
```

A port that is not declared by a module has no settings key of its own; its role stands in, so a screen asking for two kinds gets two answers rather than two ports fighting over ``src``.

### lines 1695-1701

```python
continue
```

The key already has its answer. Classify declares both a measurements database and an optional ``data/**/*_png`` crop folder, and *both* bind to ``src`` — so resolving the second would recursively glob a folder of a hundred thousand crops to arrive at the string already in hand. A drop happens with the mouse button down; this is the difference between one millisecond and forty.

### lines 1721-1725

```python
location = resolved.target
```

``target`` and not ``paths[0]``: the port's declared location is the artifact, whether that is one file (``measurements/measurements.db``) or the folder a pattern selects inside (``merged/*.npy``). Naming the first matching file would make a re-drop of the same folder resolve differently as soon as another field was written.

### lines 1737-1739

```python
if any(t.kind == _ports.MEASUREMENTS_DB for t in targets):
```

A database is the one artifact a project can legitimately hold two of. Picking the first would be exactly the silent wrong answer this is here to avoid.
