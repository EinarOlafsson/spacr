# Notes from `spacr/refit.py`

Prose lifted out of `spacr/refit.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [prune_for_type](#prune_for_type) (1 entry)
- [refit_settings](#refit_settings) (6 entries)

## prune_for_type

### lines 128-130

```python
if name == "alpha" and (value is None or value == "auto"):
```

The same 'auto'/None spelling regression_model treats as "no penalty chosen": not a request, so not something to report having dropped.

## refit_settings

### lines 201-203

```python
correction_method = canonical_method(correction_method)
```

Canonicalised here rather than passed through: the run raises on an unknown spelling, and it should do that while the dialog is open rather than twenty minutes into a fit.

### lines 210-214

```python
old = settings.get(CORRECTION_ALPHA_KEY, DEFAULT_FDR_ALPHA)
```

Compared against the run's own default when the settings did not record one, so a dialog whose spin box always holds a number does not report "significance level None -> 0.05" as a change on every single re-fit. A note that fires every time is a note nobody reads, and the notes that matter are in the same sentence.

### lines 224-229

```python
if settings.get("random_row_column_effects") and chosen not in (
```

RANDOM EFFECTS WIN OVER A NAMED MODEL, and the run refuses the combination rather than choosing. Turning the flag off when the user has just asked for a specific backend is what they meant by asking leaving it on would fit a MixedLM and file it under the name they picked, which is the exact bug _reconcile_random_row_column_effects was written for.

### lines 236-240

```python
if (str(settings.get("level", "")).lower() == "grna"
```

A GUIDE-LEVEL MIXED FIT STILL HAS NO GUIDE P VALUES, and saying so here is the difference between a re-fit that answers the question and one that returns the same empty volcano under a new folder name. `mixed` makes the guide a RANDOM effect at every level, so asking for level='grna' does not turn its BLUPs into estimates.

### lines 255-257

```python
if settings.get("plot") is False:
```

THE FIGURES COME BACK ON. save_settings writes plot=False so a reload reproduces the run headlessly; a re-fit asked for FROM a figure that then drew no figures would look like it had failed.

### lines 262-265

```python
settings.pop("src", None)
```

`src` is rebuilt by the run from the count data unless it is set, and the previous run left it pointing at its own output root. Dropping it is what puts the re-fit in a sibling folder rather than nested inside the run it is being compared with.
