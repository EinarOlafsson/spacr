# Notes from `spacr/hit_investigation.py`

Prose lifted out of `spacr/hit_investigation.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [_read_cells](#_read_cells) (1 entry)
- [_read_fractions](#_read_fractions) (1 entry)
- [evaluate_blinded_reviews](#evaluate_blinded_reviews) (1 entry)
- [register_settings](#register_settings) (1 entry)

## _read_cells

### lines 81-85

```python
scores = png.merge(predictions[["_crop", score_column]], on="_crop",
```

Only the crop key and the score are taken across. A prediction file is free to spell its crop-path column "png_path" -- an export out of spaCR naturally does -- and merging the whole frame then collided with png_list's own png_path, suffixed both to _x/_y, and left the join below asking for a column that no longer existed.

## _read_fractions

### lines 95-101

```python
from .tabular import read_table
```

THROUGH THE FUNNEL (145). Read raw, this required plateID/rowID/columnID and fell through to the `prc` split when the file spelled them row_name / column_name -- which the count CSVs on the maintainer's own screen do. With no `prc` either, it returned a frame MISSING THE KEYS and every join downstream matched nothing while raising nothing. That is 145's whole finding: a reader that does not canonicalise returns a number rather than an error.

## evaluate_blinded_reviews

### lines 241-242  _(unsure)_

```python
"brier_score": float(np.mean((consensus - probabilities) ** 2)),
```

Squared calibration error supports a soft human consensus (for example, one of two blinded reviewers calling the object positive).

## register_settings

### lines 433-434

```python
tips = {key: value for key, value in tips.items()
```

Shared settings keep the canonical cross-module help. Import order must not decide whether this app or Barcode QC owns ``dst``/``db_path``.
