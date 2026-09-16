# Notes from `spacr/run_compare.py`

Prose lifted out of `spacr/run_compare.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [_run_from_artifacts](#_run_from_artifacts) (1 entry)
- [count_database](#count_database) (1 entry)
- [read_hits](#read_hits) (1 entry)
- [RunComparison.headline](#runcomparisonheadline) (1 entry)
- [compare_runs](#compare_runs) (1 entry)

## _run_from_artifacts

### lines 309-310

```python
version = "mixed"
```

Two modules of one run produced by different spaCR versions is itself a finding — reporting the newest would hide it.

## count_database

### lines 706-710

```python
present = _tables(connection)
```

``connect`` on a file that is not SQLite succeeds; the failure arrives on the first statement. So the whole read is guarded, not just the open — a truncated or half-written database is a thing the comparison has to *report*, and an exception here would take the screen down instead.

## read_hits

### lines 1074-1075

```python
continue
```

A duplicated key would be ranked twice and then reported as having "moved" against itself.

## RunComparison.headline

### lines 1180-1181

```python
lead.append(self.settings.summary())
```

`comparable` IS `settings is not None`, so re-testing it here would be a branch that cannot go the other way.

## compare_runs

### lines 1210-1215

```python
a_counts = count_database(_database_of(a))
```

Counted first, and deliberately: the plate identity that decides comparability is in the database, not in the settings — ``src`` is cosmetic as far as the settings hash goes and the registry does not keep it. Counting is a handful of read-only COUNT queries, so paying for it before the verdict costs nothing and is the only way the "different plates" blocker fires on a registry-loaded run.
