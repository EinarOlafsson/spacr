# Notes from `spacr/train_compare.py`

Prose lifted out of `spacr/train_compare.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [Module level](#module-level) (2 entries)
- [_read_curve_csv](#_read_curve_csv) (2 entries)
- [_is_outside_any_project](#_is_outside_any_project) (2 entries)
- [_load_settings](#_load_settings) (2 entries)
- [_run_shape_from_path](#_run_shape_from_path) (1 entry)
- [_pick_settings_file](#_pick_settings_file) (1 entry)
- [find_runs](#find_runs) (1 entry)
- [_unique_ids](#_unique_ids) (1 entry)
- [_series_from_run](#_series_from_run) (1 entry)
- [is_env_key](#is_env_key) (1 entry)
- [plot_curves](#plot_curves) (1 entry)

## Module level

### lines 117-121

```python
from .run_journal import (
```

Reused from the provenance diff rather than re-derived. ``values_equal`` is the whole reason the diff is usable (structural comparison across the JSON / CSV / live-dict round-trips a settings dict takes), and the renderers keep the console output of the two features identical. They are private in run_journal because they are not a public API; this is the same feature.

### lines 130-132

```python
from .figures.style import figure_style, theme_target
```

THE HOUSE STYLE (136). `figures.style` imports matplotlib only inside its own functions, so naming it here costs nothing at import time.

## _read_curve_csv

### line 575, trailing  _(unsure)_

```python
except Exception as e:
```

unreadable / malformed — say so, keep going

### lines 581-582

```python
df.insert(0, "epoch", np.arange(1, len(df) + 1))
```

A log without an epoch column can still be ordered by row; say that the x axis is a row index rather than pretending it is an epoch.

## _is_outside_any_project

### line 629, trailing  _(unsure)_

```python
if resolved == resolved.parent:
```

filesystem root

### line 635, trailing  _(unsure)_

```python
stops.add(home.parent)
```

/home, /Users

## _load_settings

### lines 708-709  _(unsure)_

```python
break
```

Backstop: the climb has left anything that could be a spaCR project at all (root / home / the temp directory).

### lines 712-714

```python
continue
```

An ancestor that cannot have written this run's snapshot. Keep climbing — <src> is still four or five steps up a training tree — but do not read what is in this one.

## _run_shape_from_path

### line 749  _(unsure)_

```python
return path.parts[-3], m.group(1)
```

<src>/model/<model_type>/<channels>/epochs_<N>

## _pick_settings_file

### line 779  _(unsure)_

```python
return max(pool, key=lambda p: (p.stat().st_mtime, p.name))
```

Deterministic and defensible: the most recently written one.

## find_runs

### line 985, trailing  _(unsure)_

```python
continue
```

folds are loaded with their parent

## _unique_ids

### lines 1018-1019  _(unsure)_

```python
used.add(rid)
```

The loop bottoms out at the full path, which no two runs share, so there is no further fallback to write.

## _series_from_run

### lines 1052-1054

```python
fold_names = sorted(set(block["fold"]), key=_fold_sort_key)
```

Derived from the rows actually present rather than from run.folds: a fold whose validation.csv is missing must not produce an empty validation series, and ``fold_10`` must sort after ``fold_2``.

## is_env_key

### lines 1141-1143  _(unsure)_

```python
def is_env_key(key: Any, env_keys: Sequence[str] = ()) -> bool:
```

Settings diff — the run_journal bucketing, generalised to N runs

## plot_curves

### lines 1350-1353

```python
with figure_style(theme_target()):
```

THE STYLE HAS TO BE ON BEFORE THE FIGURE EXISTS:

rcParams reach an artist when it is CREATED, so a context opened after `plt.subplots` would leave the spines, ticks and labels at the caller's globals.
