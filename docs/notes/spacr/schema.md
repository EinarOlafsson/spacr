# Notes from `spacr/schema.py`

Prose lifted out of `spacr/schema.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [Module level](#module-level) (9 entries)
- [canonical_column_name](#canonical_column_name) (1 entry)
- [parse_int_token](#parse_int_token) (2 entries)
- [row_index_from_letters](#row_index_from_letters) (1 entry)
- [_prefixed_id](#_prefixed_id) (2 entries)
- [row_id](#row_id) (1 entry)
- [screen_id](#screen_id) (1 entry)
- [split_object_id](#split_object_id) (1 entry)
- [object_id](#object_id) (1 entry)
- [is_row_column_pair](#is_row_column_pair) (1 entry)
- [parse_well](#parse_well) (2 entries)
- [parse_prcf](#parse_prcf) (2 entries)
- [ObjectTableSchema.identifier_columns](#objecttableschemaidentifier_columns) (1 entry)
- [is_provenance_column](#is_provenance_column) (1 entry)
- [_non_numeric_feature_error.diagnostic_dtype](#_non_numeric_feature_errordiagnostic_dtype) (1 entry)
- [coerce_model_feature_types](#coerce_model_feature_types) (2 entries)
- [model_feature_columns](#model_feature_columns) (1 entry)
- [normalise_plate_columns](#normalise_plate_columns) (1 entry)
- [validate_object_table_frame](#validate_object_table_frame) (1 entry)
- [legacy_safe_int_convert](#legacy_safe_int_convert) (1 entry)
- [correct_metadata_column_names](#correct_metadata_column_names) (1 entry)

## Module level

### line 29  _(unsure)_

```python
'PLATE_KEY', 'ROW_KEY', 'COLUMN_KEY', 'FIELD_KEY', 'OBJECT_KEY',
```

key names

### line 44  _(unsure)_

```python
'parse_int_token', 'row_index_from_letters', 'letters_from_row_index',
```

scalars

### line 52  _(unsure)_

```python
'FieldID', 'ObjectID',
```

identities

### line 71  _(unsure)_

```python
'add_identity_columns', 'canonicalise_columns',
```

pandas

### line 75

```python
'legacy_well_ids', 'legacy_map_wells', 'legacy_safe_int_convert',
```

legacy

### lines 419-424

```python
'object_id':     OBJECT_KEY,
```

Only unambiguous object-identifier spellings are accepted. A bare `object` column is not included because in a measurement table it as often means the object TYPE -- cell, nucleus, pathogen as the object's number, and renaming that into an identifier would corrupt the join it lands in rather than merely mislabel a column. That is the same trap the docstring above records for `c`.

### lines 464-467

```python
(
```

``..._periphery_25_percentile`` -> ``..._periphery_percentile_25``. ``head`` is non-greedy and ``ring`` therefore binds to the *last* periphery/outside token, so a feature whose prefix happens to contain 'outside' does not capture the rewrite.

### lines 475-477

```python
(
```

``organelle_summary_organelle_ch0_...`` -> ``..._channel_0_...``. The anchor is the full ``organelle_summary_organelle_ch`` prefix so a user feature that merely contains ``ch`` followed by a digit is untouched.

### lines 2649-2651  _(unsure)_

```python
_DOUBLED_PLATE_PREFIX = re.compile(r'^pp')
```

One vocabulary: the collision rule, and the plate-value repair

## canonical_column_name

### lines 591-595

```python
alias = _FOLDED_ALIASES.get(fold_column_name(text))
```

Metadata before features: the alias table is a closed vocabulary of short names, none of which can also match a feature pattern (both patterns are anchored and require a trailing '_<digits>_percentile' or a leading 'organelle_summary_'), so the order is a cost decision, not a precedence one -- a metadata column is answered without touching a regex.

## parse_int_token

### line 665  _(unsure)_

```python
return None
```

bool is an int subclass; a True field id is a caller bug, not a 1.

### lines 670-671

```python
if token != token or token in (float('inf'), float('-inf')):
```

NaN and inf hold no integer. int(nan) raises, int(2.7) truncates silently, so neither is acceptable without a check.

## row_index_from_letters

### lines 712-714

```python
return None
```

str(None) is 'None', which is four perfectly good row letters and would come back as row 256573. Anything that is not already text is not a row label.

## _prefixed_id

### lines 816-818

```python
raise KeyParseError(
```

Tier 3: nothing to key on. This is the one case that must raise — an empty id is not an identity, and every row carrying it would merge with every other row that also failed to parse.

### lines 829-830

```python
return f'{prefix}{_sanitise_token(text)}'
```

Tier 2: keep the token. Distinct per input, visibly not a number, and still a usable join key — the run continues without inventing a 0.

## row_id

### line 849  _(unsure)_

```python
return f'r{row_index_from_letters(letters)}'
```

_ROW_ONLY admits precisely the strings the decoder accepts.

## screen_id

### line 924  _(unsure)_

```python
return DEFAULT_SCREEN
```

NaN — what pandas puts in a column a source did not fill in.

## split_object_id

### lines 1034-1037

```python
return (None, text)
```

A bare label: '7'. Untyped, and the only reading that does not invent a type it was never given. Guarded on a leading digit so an unrecognised token ('x7') is still not an object id — see :func:`object_type_prefix` on why the vocabulary is closed.

## object_id

### lines 1094-1097

```python
read_type, read_label = split_object_id(composed)
```

Prove the round trip rather than trusting the vocabulary. The one case this catches in practice: an untyped id whose preserved label starts with a declared type's remainder, e.g. label 'rganelle7' composing to 'organelle7', which reads back as an organelle.

## is_row_column_pair

### lines 1251-1253

```python
return True
```

parse_well puts an unrecognisable well into both slots verbatim, so an equal unprefixed pair is that passthrough and not a deeper key's tail (a field never equals the column it sits in).

## parse_well

### lines 1302-1304

```python
return (f'r{row_index_from_letters(match.group(1))}',
```

_WELL's first group is [A-Za-z]{1,3}, so row_index_from_letters cannot fail here; no defensive branch, because an unreachable one could never be tested and would only ever be wrong.

### lines 1319-1320

```python
return text, text
```

Legacy passthrough: every existing implementation does this, and databases on disk carry rowID == columnID == the raw well.

## parse_prcf

### lines 1708-1713

```python
if (parts[-2][:1].lower() == 'f'
```

The composer deliberately preserves a non-numeric time token (``xy`` becomes ``txy``) so an imperfect instrument export still has a stable join key. Detect the optional time component by grammar, not by whether its payload happens to be numeric: a trailing t-component immediately after an f-component can only be the timepoint in plate_row_column_field[_time].

### lines 1727-1729

```python
raise KeyParseError(
```

An empty component is not a missing token, it is a token every field of the plate shares: join on it and they all merge. This is the same refusal ``ml._split_prc`` makes, for the same reason.

## ObjectTableSchema.identifier_columns

### lines 2055-2059

```python
return (
```

measure._morphological_measurements prefixes the child mapping before _check_integrity runs. These look like ordinary feature names but are identifiers: feature_dict._LINK_COLUMNS documents the same distinction. They may use object dtype after DataFrame.explode() even though every non-null value denotes an integer label.

## is_provenance_column

### lines 2150-2151  _(unsure)_

```python
suffixes = tuple(f'_{obj}' for obj in OBJECT_TYPES) + ('_x', '_y')
```

Joined object tables suffix overlapping columns. The suffix does not turn measurement_ndim_nucleus or object_label_pathogen into features.

## _non_numeric_feature_error.diagnostic_dtype

### lines 2212-2216

```python
def diagnostic_dtype(dtype) -> str:
```

pandas 3 infers ordinary Python text as StringDtype (displayed as ``str``) where earlier versions inferred ``object``.  The diagnostic is a user-facing description of the same text-storage problem, so keep its established wording stable without flattening categorical or other extension dtypes that carry materially different information.

## coerce_model_feature_types

### lines 2322-2323  _(unsure)_

```python
numeric = series.astype('float64')
```

No value was read, so no value can be misread. float64 NaN is the same fact in a dtype the model boundary accepts.

### lines 2326-2328

```python
normalized = series.replace(r'^\s*$', pd.NA, regex=True)
```

Empty strings in SQLite measurement tables represent the same missing value as NULL. Preserve that distinction before conversion.

## model_feature_columns

### line 2426  _(unsure)_

```python
continue
```

pandas/numpy numeric selectors historically omitted bools.

## normalise_plate_columns

### lines 2704-2708

```python
dtype = getattr(values, 'dtype', None)
```

A plate id stored as a number cannot carry a "pp" prefix, so there is nothing to do. pandas 3 infers ordinary Python text as StringDtype rather than object; both are text-bearing inputs, while categorical and other extension dtypes retain their old no-op behaviour.

## validate_object_table_frame

### lines 2968-2972

```python
unresolved = [
```

Every object-table writer crosses this boundary.  Resolve unfamiliar metadata here so modules do not each grow a slightly different rename prompt.  The import stays lazy to preserve schema's lightweight import contract and the headless default raises immediately rather than opening a dialog.

## legacy_safe_int_convert

### lines 3223-3229

```python
def legacy_safe_int_convert(value: Any, default: Any = 0) -> Any:
```

Bug-compatible copies of what is on disk today

These exist so a migration can be done one call site at a time with a test pinning exactly what changed, and so that a reader of an old database can reproduce the key it was written with. They are not for new code.

## correct_metadata_column_names

### lines 3296-3309

```python
def correct_metadata_column_names(df):
```

Moved here from spacr.utils on 2026-08-19.

THE FUNCTION NEVER NEEDED ANYTHING utils IMPORTS. It delegates to `canonicalise_frame` below and touches pandas, and that is all -- but `spacr/utils.py` imports torch, torchvision and cv2 on its line 3, so `from .utils import correct_metadata_column_names` cost 4,336 modules and 6.7 SECONDS. The Cells tab paid it to show nine PNGs, which is why the montage felt slow beside the annotation app: "in the annotation app images load almost instintaniously while in the regression cell montage it takes way longer".

`spacr.utils` re-exports it, so every existing caller is unchanged.
