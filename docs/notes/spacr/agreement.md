# Notes from `spacr/agreement.py`

Prose lifted out of `spacr/agreement.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [Module level](#module-level) (6 entries)
- [kappa_detail](#kappa_detail) (1 entry)
- [_connect](#_connect) (1 entry)
- [load_annotations](#load_annotations) (1 entry)
- [agreement_report](#agreement_report) (3 entries)

## Module level

### lines 107-112

```python
"timeID", "time_id", "prcft",
```

Both spellings of the timepoint. 'timeID' is what filepaths_to_database writes now, and what spacr.utils.rename_columns_in_db migrates an old database to on first read; 'time_id' is what a database written before that still carries until then. Either one is metadata, never an annotation -- and a timelapse database whose time column counted as a candidate annotation column would have been scored for agreement.

### lines 117-120

```python
"crop_format",
```

spacr.crops.CROP_FORMAT_DB_COLUMN, the channel-order version marker stamp_crop_format_in_db adds to png_list. One or two distinct small integers over every row -- the exact shape of an annotation pass, and not one.

### line 150  _(unsure)_

```python
"pred", "cv_predictions",
```

spacr.predictions: the convolutional classifier

### line 152  _(unsure)_

```python
"ml_pred", "predictions",
```

spacr.predictions: the classical-ML classifier

### line 154  _(unsure)_

```python
"prediction", "score",
```

spacr.active_learning.PRED_COLUMN_CANDIDATES, the two not already above

### lines 156-160

```python
"XGboost_annotation", "XGboost_score",
```

The removed Tk Annotate app's built-in XGBoost pass wrote these. The columns are still in databases it produced, so the names stay listed even though nothing writes them any more. The name says "annotation" and it is not one -- it is a model's call, derived from a score in the very next column.

## kappa_detail

### lines 520-521  _(unsure)_

```python
kappa = (p_o - p_e) / (1.0 - p_e)
```

Past this point both annotators used >= 2 classes, so their marginals are not unit vectors and pₑ < 1 strictly: the denominator is safe.

## _connect

### lines 647-651

```python
from .database_concurrency import connect as _connect_database
```

Through the shared helper, which sets a 30s busy timeout and query_only. A bare sqlite3.connect takes sqlite's 5s default, and Measure writes from many worker processes at once -- 5s is routinely exceeded there, so the reader fails with "database is locked" instead of waiting (issue #15).

## load_annotations

### lines 772-774

```python
data: Dict[str, pd.Series] = {
```

dtype=object throughout: a column of ints with NULLs would otherwise come back as float64 with NaN, and "did this annotator abstain?" would stop being an ``is None`` question.

## agreement_report

### lines 978-981

```python
values = df[cols].to_numpy(dtype=object)
```

One pass over the table classifies every row as complete / partial / untouched and counts the real disagreements. Partial rows are abstentions: they never count towards n_disagreements unless the annotators who *did* commit chose differently.

### lines 1023-1024  _(unsure)_

```python
only = universe[int(np.argmax(arr.sum(axis=0)))]
```

n_complete > 0 means every one of those rows carries a label from every annotator, so the universe is non-empty.

### lines 1060-1062

```python
model_cols = [c for c in cols if _is_model_column(c, table_columns(db_path, table))]
```

A caller can always name columns explicitly, and the Qt screen lets one be ticked. Saying so is the difference between a deliberate model validation and a κ quoted as inter-annotator agreement that is not one.
