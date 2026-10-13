# Notes from `spacr/classifier_quality.py`

Prose lifted out of `spacr/classifier_quality.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## discover_test_splits

### lines 343-360

```python
def discover_test_splits(root: str, *,
```

Reading a real test split

The maintainer supplied the tsg101 screen's test splits on 2026-08-21, in TWO shapes, and both are read here because both are what the training code writes:

PER-CELL (`*_test_acc.csv`): one row per test crop with `true_label`, `predicted_label` and `class_1_probability`. Everything can be computed from this -- any threshold, the whole ROC. SUMMARY (`*_test_result.csv`): one row with `accuracy`, `neg_accuracy`, `pos_accuracy`, `prauc` and `optimal_threshold`.

`pos_accuracy` IS THE SENSITIVITY and `neg_accuracy` IS THE SPECIFICITY. That is the number instruction 214 asked for and it was already being written to disk on every plate -- which is worth saying plainly, because the request was answered by a file that already existed.

### lines 362-378

```python
def discover_test_splits(root: str, *,
```

NO SCREEN'S NUMBERS LIVE IN THIS FILE. A table of the tsg101 plates' measured sensitivities was here for one commit and was removed on request:

"wahtever information you use from my screen any calculated coefficients need to be recalculable for users whou do their own screens"

WHICH IS RIGHT, AND NOT ONLY ON PRINCIPLE. A constant in a library becomes a default the moment somebody is in a hurry, and a sensitivity measured on one model, one stain and one microscope is wrong for every other screen in a way that produces plausible numbers rather than an error. Every function here takes `sensitivity` and `specificity` as REQUIRED arguments for that reason -- there is nothing to fall back to.

`discover_test_splits` finds a user's own files; `from_test_split` reads them. The tsg101 figures are recorded in `features/new/`, which is a log and cannot be imported.
