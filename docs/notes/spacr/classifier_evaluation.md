# Notes from `spacr/classifier_evaluation.py`

Prose lifted out of `spacr/classifier_evaluation.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [Module level](#module-level) (2 entries)
- [split_group_values](#split_group_values) (1 entry)
- [grouped_split](#grouped_split) (6 entries)
- [sample_identity](#sample_identity) (1 entry)
- [cross_calibrate_probabilities](#cross_calibrate_probabilities) (1 entry)
- [calibration_table](#calibration_table) (2 entries)
- [_write_confusion_figure](#_write_confusion_figure) (2 entries)
- [_write_calibration_figure](#_write_calibration_figure) (2 entries)

## Module level

### lines 30-32

```python
from .figures.style import figure_style, theme_target
```

THE HOUSE STYLE (136). `figures.style` imports matplotlib only inside its own functions, so naming it here costs nothing at import time.

### line 51  _(unsure)_

```python
SPLIT_LEVELS: Tuple[str, ...] = ("cell", "field", "well", "plate")
```

Train/test grouping ladder, from least to most independent.

## split_group_values

### lines 227-229

```python
encoded = augmentation_family(path).split("_")
```

A crop needs plate, well, field, and object tokens. Accepting a two-token arbitrary filename as ``plate_well`` invents a well identity and turns a random split into a grouped-looking one.

## grouped_split

### lines 288-303

```python
if len(y) == 0:
```

NOTHING TO SPLIT, SAID HERE RATHER THAN BY SKLEARN (issue #110).

An empty label array falls all the way through to `train_test_split` and surfaces as

ValueError: With n_samples=0, test_size=0.2 the train set will be empty

which names neither the setting that is wrong nor what to do about it, and is filed against spaCR rather than read as a data problem. Every other degenerate shape below is already refused in words; this was the one that was not.

The named-holdout path divides by `len(y)` to report the cell fraction, so an empty array is a ZeroDivisionError there instead. Both are answered by refusing before either can happen.

### lines 314-315

```python
raise ValueError(
```

A group per label is what every split below assumes; a mismatch silently misaligns the two and produces a split that looks valid.

### lines 323-324

```python
if hold_out_groups:
```

A NAMED HOLDOUT SHORT-CIRCUITS THE SAMPLING. There is nothing to stratify: the caller has said which groups are the test side.

### lines 395-398

```python
split_counts = sorted({
```

The nearest whole-group fraction may not contain every class. Try a small ladder down to a half holdout before declaring the design impossible; 25% over four class-confounded wells, for example, has no two-class one-well test set but does have an honest two-well one.

### line 413  _(unsure)_

```python
splitter = GroupShuffleSplit(
```

More candidates make uneven group sizes land closer to test_size.

### line 452, trailing  _(unsure)_

```python
if train_groups & test_groups:
```

defensive: sklearn promises this

## sample_identity

### lines 557-563

```python
family = augmentation_family(path)
```

Exported crop names use either ``plate_well_field_object`` (for example ``plate1_A01_f2_o7``), separate row/column exports such as ``plate1_A_01_1_7``, or canonical PRCFO tokens (``plate1_r1_c1_f2_o7``). Parse from the field token toward the left so plate identifiers may themselves contain underscores. Including a field token in the well identity would split one biological well into multiple leakage groups.

## cross_calibrate_probabilities

### lines 1234-1238

```python
if len(y) and probs.shape[1]:
```

REFUSE a label the head has no column for. Without this every fold's fit raised inside its own try/except, each fell back to temperature 1.0, and the function returned UNCALIBRATED probabilities while reporting that calibration ran -- visible only as a printed warning a caller has no reason to be watching.

## calibration_table

### lines 1329-1331

```python
if len(y) and n_classes:
```

A label with no column matches nothing, so every observed_frequency reads 0.0 and the reliability curve looks catastrophically miscalibrated rather than wrong. Refuse it.

### lines 1372-1375

```python
return pd.DataFrame(rows, columns=CALIBRATION_COLUMNS)
```

Name the columns even with no rows. `pd.DataFrame([])` is shape (0, 0), so a caller indexing 'class_name' or 'bin' on an empty result got a KeyError rather than an empty column -- and an empty table is a legitimate state (every bin can be empty), not a broken frame.

## _write_confusion_figure

### lines 1755-1758

```python
with figure_style(theme_target()):
```

THE STYLE HAS TO BE ON BEFORE THE FIGURE EXISTS:

rcParams reach an artist when it is CREATED, so a context opened after `plt.subplots` would leave the spines, ticks and labels at the caller's globals.

### lines 1776-1787

```python
from .plot import save_figure
```

THE RESOLUTION IS THE USER'S; THE FORMAT IS NOT (108 point 6). This wrote a PNG at a fixed DPI whatever the preferences said, so "Resolution" reached everything except the files a pipeline leaves behind -- and it gains `print_ready`, so a bundle written from a dark session is not white ink on a white page.

BUT `fmt` STAYS PNG, and that is not an oversight. These two files are named in `EVALUATION_FILES`, which is the bundle's CONTRACT: `read_evaluation_bundle` opens `confusion_matrix.png` by that exact name. Letting a format preference rename it makes the bundle unreadable by the function that wrote it -- a preference must not rename a file another part of the code opens by name.

## _write_calibration_figure

### lines 1797-1800

```python
with figure_style(theme_target()):
```

THE STYLE HAS TO BE ON BEFORE THE FIGURE EXISTS:

rcParams reach an artist when it is CREATED, so a context opened after `plt.subplots` would leave the spines, ticks and labels at the caller's globals.

### lines 1819-1830

```python
from .plot import save_figure
```

THE RESOLUTION IS THE USER'S; THE FORMAT IS NOT (108 point 6). This wrote a PNG at a fixed DPI whatever the preferences said, so "Resolution" reached everything except the files a pipeline leaves behind -- and it gains `print_ready`, so a bundle written from a dark session is not white ink on a white page.

BUT `fmt` STAYS PNG, and that is not an oversight. These two files are named in `EVALUATION_FILES`, which is the bundle's CONTRACT: `read_evaluation_bundle` opens `confusion_matrix.png` by that exact name. Letting a format preference rename it makes the bundle unreadable by the function that wrote it -- a preference must not rename a file another part of the code opens by name.
