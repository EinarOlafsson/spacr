# Notes from `spacr/data_manager.py`

Prose lifted out of `spacr/data_manager.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [Module level](#module-level) (1 entry)
- [_real](#_real) (1 entry)
- [scan_project](#scan_project) (3 entries)
- [plan_prune](#plan_prune) (2 entries)
- [_mark_artifacts](#_mark_artifacts) (2 entries)
- [prune](#prune) (4 entries)
- [plan_archive](#plan_archive) (1 entry)
- [archive](#archive) (3 entries)

## Module level

### lines 235-236

```python
("artifacts.db", ports.SETTINGS_CSV),
```

The bookkeeping, first: `artifacts.db` matches `*.db` further down and would otherwise be reported as somebody's measurements.

## _real

### line 350, trailing  _(unsure)_

```python
except OSError:
```

exotic filesystems that will not resolve

## scan_project

### line 628  _(unsure)_

```python
by_path: Dict[str, List[Artifact]] = {}
```

Registered paths, grouped. Several artifacts may name one path.

### line 640  _(unsure)_

```python
owned_bytes: Dict[str, int] = {path: 0 for path in by_path}
```

Attribute every walked file to at most one registered path.

### lines 665-666  _(unsure)_

```python
stats: Dict[str, Dict[str, int]] = {}
```

Per-kind totals: registered bytes attributed to the path's ranked kind, unregistered bytes to whatever the layout suggests.

## plan_prune

### lines 1046-1051

```python
kept.append(PruneSkip(
```

Two registered artifacts, one inside the other. The bytes were attributed to the inner one (longest prefix wins), so deleting the outer would free more than the plan says and take an artifact nobody judged with it. Nothing declares such a pair today; the guard is here because a plan that under-reports what it deletes is the failure this whole module is about.

### lines 1071-1073

```python
newest = next((a for a in entry.artifacts if a.kind == entry.kind),
```

The row this candidate is *reported* as: the newest of the kind the path is filed under. It is only the label and the "how do I get it back" module — every artifact at the path still has to pass.

## _mark_artifacts

### lines 1263-1265

```python
_verified_write(connection, "artifact_inputs",
```

Edges first: `artifacts` is the parent of a foreign key, and the two are one write either way. Both go through the same count-write-compare, on the same predicate.

### lines 1274-1278

```python
connection.execute(
```

The merged JSON is computed per row and staged in a temp table, so the UPDATE below can be ONE statement over ONE predicate the same predicate the count is taken with. A loop of `WHERE artifact_id = ?` updates would be a write on a predicate nothing counted, which is the whole failure mode being avoided.

## prune

### line 1382  _(unsure)_

```python
paths = [c.path for c in plan.candidates]
```

2. Nothing is deleted until the whole plan still describes the disk.

### line 1410  _(unsure)_

```python
rows = 0
```

3. The registry write, counted and verified, before any file goes.

### line 1426  _(unsure)_

```python
removed_files: List[str] = []
```

4. Now delete.

### line 1440  _(unsure)_

```python
left = [c.path for c in plan.candidates if os.path.exists(c.path)]
```

5. And check.

## plan_archive

### lines 1607-1610

```python
inside = [entry for path, entry in by_path.items()
```

Every artifact this entry carries, not only one registered at exactly this path: a whole-project archive moves `measurements/`, and the database's provenance is inside it. Missing that is how the destination ends up describing four of a project's seven artifacts.

## archive

### line 1693  _(unsure)_

```python
provenance: List[Artifact] = []
```

Everything the artifacts know, read before the registry can move.

### lines 1706-1708

```python
ledger_path = os.path.join(plan.root, ARCHIVE_LEDGER_NAME)
```

Read before the move: a whole-project archive moves the ledger too, and the origin's earlier archives must not be forgotten because the file that recorded them went with the data.

### lines 1726-1727  _(unsure)_

```python
registered = 0
```

The destination is made self-describing: same module, kind, role, settings hash and inputs, at the new path.
