# Notes from `spacr/surrogate.py`

Prose lifted out of `spacr/surrogate.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [Module level](#module-level) (1 entry)
- [build_surrogate_frame](#build_surrogate_frame) (2 entries)
- [fit_surrogate](#fit_surrogate) (4 entries)
- [_shap_importance](#_shap_importance) (2 entries)
- [explain_classifier](#explain_classifier) (1 entry)
- [write_surrogate_result](#write_surrogate_result) (4 entries)

## Module level

### lines 54-56

```python
from .figures.style import figure_style, theme_target
```

THE HOUSE STYLE (136). `figures.style` imports matplotlib only inside its own functions, so naming it here costs nothing at import time.

## build_surrogate_frame

### lines 333-335

```python
preds["_key"] = preds["png_path"].astype(str).map(os.path.basename)
```

Match on basename as well as full path: a model scored on one machine and a database written on another agree about the file name and not about the mount point.

### lines 353-363

```python
if "prcfo" in features.columns and "prcfo" in bridged.columns:
```

THE CROP PATH IS AN OBJECT IDENTITY TOO, and on a real spaCR database it is the only one both sides have. `prcfo` is written into `png_list`; the measurement tables carry `plateID`, `rowID`, `columnID`, `fieldID` and `object_label` and compose it on demand, so the joined feature frame has no `prcfo` column at all -- and the surrogate refused every database spaCR has ever written (236 B6, driven on plate1 of the tsg101 screen).

The basename is the same key `preds` was matched on ten lines above, for the same reason: two machines agree about a file name and not about a mount point.

## fit_surrogate

### lines 658-659  _(unsure)_

```python
permutation_labels = pd.Series(
```

The estimator was fitted on encoded classes, so its scorer must see those same labels. Displayed predictions are decoded above.

### lines 665-671

```python
permutation_jobs = guarded_n_jobs(
```

joblib's default Loky backend keeps its reusable worker processes alive after ``permutation_importance`` returns.  A completed explanation then owns an ExecutorManagerThread and one process per CPU until interpreter shutdown (32 workers on the CI host).  Permutations share one fitted, read-only estimator, so threads avoid both that lifecycle leak and the cost of serialising the model into every process while retaining the feature-level parallelism.

### line 690, trailing  _(unsure)_

```python
else:
```

compatibility with callers replacing the helper

### line 692, trailing  _(unsure)_

```python
else:
```

compatibility with callers/tests replacing this helper

## _shap_importance

### lines 784-785

```python
warnings.append(
```

Said out loud: a silently truncated sample reads as "explained everything" when it did not.

### lines 800-804

```python
if isinstance(values, list):
```

Older SHAP returns a list of (rows, features), one per class. Newer releases return (rows, features, classes). Normalise both before asking which axis holds features; averaging the old list-shaped array over the wrong axes produces one importance per ROW and used to be silently accepted whenever rows happened to equal features.

## explain_classifier

### lines 847-849  _(unsure)_

```python
def explain_classifier(db_path: str, predictions: pd.DataFrame, *,
```

The one call most people want

## write_surrogate_result

### lines 953-956

```python
with figure_style(theme_target()):
```

THE STYLE HAS TO BE ON BEFORE THE FIGURE EXISTS:

rcParams reach an artist when it is CREATED, so a context opened after `plt.subplots` would leave the spines, ticks and labels at the caller's globals.

### lines 972-977

```python
from .plot import save_figure
```

108 point 6, and the explicit `fmt` is the point: this writes BOTH a pdf and a png of the same figure deliberately, so the format is the loop's and not the preference's. What it gains is the rest of `save_figure` -- the DPI rule, the TrueType embedding, and the repaint for paper that a figure saved from a dark session needs.

### lines 997-1000

```python
with figure_style(theme_target()):
```

THE STYLE HAS TO BE ON BEFORE THE FIGURE EXISTS:

rcParams reach an artist when it is CREATED, so a context opened after `plt.subplots` would leave the spines, ticks and labels at the caller's globals.

### line 1018

```python
from .plot import save_figure
```

108 point 6; both formats on purpose, see above.
