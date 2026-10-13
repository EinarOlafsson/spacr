# Notes from `spacr/predictions.py`

Prose lifted out of `spacr/predictions.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [_sql_value](#_sql_value) (1 entry)
- [_prcfo_from_metadata](#_prcfo_from_metadata) (2 entries)
- [crop_name_metadata.convert](#crop_name_metadataconvert) (1 entry)
- [crop_name_metadata.bare_object_label](#crop_name_metadatabare_object_label) (2 entries)
- [_result_keys](#_result_keys) (1 entry)
- [merge_prediction_results](#merge_prediction_results) (1 entry)
- [_merge_locked](#_merge_locked) (4 entries)
- [attach_predictions](#attach_predictions) (1 entry)

## _sql_value

### lines 283-284  _(unsure)_

```python
pass
```

pd.isna of a list/array is elementwise, so `if` on it raises. Not a missing value; fall through and let the type handling below decide.

## _prcfo_from_metadata

### lines 391-392

```python
pieces = []
```

A row missing any one component has no key at all -- an empty slot would make 'plate1_r1__f1_o3' collide with a genuinely different object.

### lines 403-412

```python
return pd.Series(
```

THROUGH THE SAME NORMALISER THE STORED KEY GETS. A key rebuilt here and a key read from the `prcfo` column must be the same string or the join silently matches nothing, and the halves disagree exactly where `_clean_prcfo` says they do: an older run stamps the plate `pplate1` and everything computed since stamps it `plate1`. Normalising in one place is what stops the two builders drifting apart again. Construct the result as explicit object data.  Under pandas 3 string inference, ``where(..., other=None).map(...)`` promotes the Series to StringDtype and exposes a missing key as float ``nan``; callers use identity with ``None`` to distinguish an absent key from a real one.

## crop_name_metadata.convert

### line 467

```python
cache[base] = empty if any(v is None for v in parsed) else parsed
```

'error' in any position means the whole name failed to parse.

## crop_name_metadata.bare_object_label

### line 473  _(unsure)_

```python
def bare_object_label(value):
```

object_label without the 'o': that is the spelling the object tables use.

### lines 476-479

```python
text = _clean_key(value)
```

pandas 3 may infer these parsed text columns as StringDtype and materialise a tuple's ``None`` as float ``nan``.  Normalize through the same missing-key boundary used everywhere else before asking a value for string methods.

## _result_keys

### lines 529-536

```python
from .schema import canonicalise_columns
```

SAME FALLBACK `_db_keys` ALREADY HAD. A score table with no path column can still carry the metadata `prcfo` is built from, and an `ml_analysis` score CSV is exactly that: plate/row/column/field and an object id, under the plainer spellings that `schema.canonicalise_columns` resolves. Without this the two sides of the join were asymmetric -- the database could rebuild the key and the results frame could not -- so an XGBoost score file matched zero rows and read as "no per-object score".

## merge_prediction_results

### lines 841-842

```python
con.isolation_level = None
```

Explicit transaction control: ALTER TABLE would otherwise autocommit and an interrupted merge would leave a column added but no rows scored.

## _merge_locked

### lines 869-870

```python
cur.execute(f"SELECT * FROM {quoted_table} LIMIT 0")
```

Raises sqlite3.OperationalError('no such table: ...') -- the loud, correct failure for a database that was never measured.

### line 898  _(unsure)_

```python
value_frames = {db_col: results[src] for db_col, (src, _t) in spec.items()}
```

collapse the results into key -> values, refusing collisions

### lines 906-910

```python
key_list = [_clean_key(value) for value in result_keys]
```

``Series.map`` preserves ``None`` on pandas 2 object columns, while pandas 3's inferred StringDtype materialises the same missing key as float ``nan``.  Identity checks therefore changed the report from one unparsed row to one unmatched row.  Normalize both sides by value before counting or joining so the public merge report is version-independent.

### line 930  _(unsure)_

```python
added = []
```

add the columns we are about to write

## attach_predictions

### lines 1017-1021

```python
score_name = (str(score_source)
```

WHICHEVER NAME THE SCORE ARRIVED UNDER. `score_source` stays the first choice, so an explicit argument still wins; when the table does not carry it, the other names a score goes by are tried before giving up. Refusing here on the name alone is what made an XGBoost score CSV read as "no per-object score" while holding one.
