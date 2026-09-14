# Notes from `spacr/scorecard.py`

Prose lifted out of `spacr/scorecard.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [dice](#dice) (1 entry)
- [compare_against_baseline](#compare_against_baseline) (1 entry)
- [_auroc](#_auroc) (2 entries)
- [score_classifier](#score_classifier) (4 entries)
- [HoldoutScore](#holdoutscore) (1 entry)
- [score_holdout](#score_holdout) (1 entry)
- [scorecard_rows](#scorecard_rows) (1 entry)
- [HoldoutField](#holdoutfield) (1 entry)
- [headline](#headline) (2 entries)
- [scorecard_figure](#scorecard_figure) (4 entries)

## dice

### line 255, trailing  _(unsure)_

```python
per_object = [2 * i / (1 + i) for i in matched.ious]
```

Dice from IoU

## compare_against_baseline

### lines 423-425

```python
delta.pop("match_iou", None)
```

`match_iou` is a SETTING, not a score. Subtracting it gives 0.0 and reads in a table as "the threshold did not change", which is true and is noise; leaving it in the delta invites someone to plot it.

## _auroc

### lines 430-444

```python
def _auroc(labels: np.ndarray, scores: np.ndarray) -> float:
```

Classifiers

A screen is unbalanced -- that is what a screen IS -- so several of these exist only because accuracy is uninformative when 98% of objects are negative. AUPRC comes first when positives are rare, which in a screen they are, and balanced accuracy and MCC are here for the same reason.

BUILT ON `spacr.classifier_quality.Confusion` RATHER THAN BESIDE IT. That module already owns the confusion matrix, sensitivity, specificity and accuracy, and already refuses to report a correction its inputs cannot identify. A second implementation of the same four numbers is a second thing to keep in step, and they would disagree first in the place nobody is looking.

### lines 460-461

```python
values = np.asarray(scores)[order]
```

Average the ranks of tied scores, or a model that outputs one constant scores 1.0 or 0.0 depending on sort order rather than the 0.5 it earns.

## score_classifier

### line 542, trailing  _(unsure)_

```python
recall = _ratio(tp, tp + fn)
```

sensitivity

### lines 547-548

```python
denominator = float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
```

MCC survives imbalance where accuracy and F1 do not: it is the only one of these that uses all four cells of the matrix symmetrically.

### lines 560-561

```python
"balanced_accuracy": (recall + specificity) / 2.0,
```

THE SECOND ONE BECAUSE SCREENS ARE UNBALANCED. With 98% negatives, calling everything negative scores 0.98 accuracy and 0.5 balanced.

### lines 576-577

```python
"brier": float(np.mean((scores - labels) ** 2)),
```

Brier is the mean squared error of the probability itself, so it penalises a confident wrong answer more than a hesitant one.

## HoldoutScore

### lines 604-614

```python
@dataclass(frozen=True)
```

A held-out SET, not a field

Instruction 370: "A number computed on 'three fields' chosen at call time is not comparable between two models, between two versions of one model, or between two days. What makes the table worth publishing is that every model is scored on the SAME named, versioned, labelled set."

So the unit of a published number is the SET. Everything below aggregates per-field scorecards into one, and the aggregation is not a mean of means.

## score_holdout

### lines 697-698  _(unsure)_

```python
for key in scored[0]:
```

The pixel-wise and boundary measures are field-level by nature, so they are averaged over FIELDS and weighted by the objects each holds.

## scorecard_rows

### lines 734-736

```python
"delta": (a - b) if key not in _NOT_A_SCORE else None,
```

`None` rather than 0.0 for a setting or a count: a "difference" in `match_iou` or `n_truth` is not a result, and a zero in that column reads as one.

## HoldoutField

### lines 768-777

```python
@dataclass(frozen=True)
```

The set itself: named, versioned, checksummed

"What makes the table worth publishing is that every model is scored on the SAME named, versioned, labelled set, and that the set is published beside the models so the number can be checked by somebody else." -- 370.

A manifest is what makes that checkable. Without one, "scored on the hold-out set" is a claim about a folder on somebody's laptop.

## headline

### lines 965-967

```python
text += f" ({delta:+.3f} vs stock)"
```

SIGNED AND EXPLICIT. "F1 0.867" alone answers "what is it"; the request is "is it better", which only the difference answers.

### line 977

```python
counted = []
```

EVERY NUMBER CARRIES ITS N, in the tooltip too.

## scorecard_figure

### lines 1132-1143

```python
from .figures.style import figure_style
```

INSIDE `figure_style`, and the context opens BEFORE `subplots`: rcParams reach an artist when it is CREATED, so a context entered afterwards leaves the spines, ticks and labels at whatever the caller's globals happened to be. A chart published beside a model is the last place to ship a figure in a second visual system.

`dpi` DEFAULTS TO None, NOT 150, and that is load-bearing rather than tidy. `save_figure` reads the user's Resolution preference only when it is handed None; passing a number -- even the old default -- wins over the preference and the setting silently never reaches this figure. An explicit dpi from a caller still overrides, which is the behaviour a caller asking for one expects.

### lines 1163-1165

```python
axes.set_ylim(0.0, 1.08)
```

0 TO 1 ALWAYS, never autoscaled. Every metric here is a fraction, and a y-axis that started at 0.8 would make a 0.02 gain look like a landslide which is exactly the misreading a published chart must not invite.

### lines 1168-1170

```python
axes.legend(frameon=False, loc="upper center", ncol=2,
```

OUTSIDE THE AXES. `lower right` sat on top of the recall bars, which is the corner a high-scoring model fills -- the legend would hide exactly the result the chart is published to show.

### lines 1190-1203

```python
from .plot import save_figure
```

THROUGH `spacr.plot.save_figure`, the one writer. A scorecard is a figure the user KEEPS -- it is published beside the model -- so the figure-format and resolution preferences have to reach it like any other kept figure. A bare `savefig` here would hard-code PNG at whatever dpi this function was called with and ignore both.

THE COST, SAID PLAINLY: `spacr.plot` imports torch, cv2, seaborn and scipy AT MODULE SCOPE, so DRAWING a scorecard now pulls the whole plotting stack. Reading one still does not -- the import is inside this function, which is what `test_importing_the_module_still_needs_no_plotting_stack` pins but a caller who only wanted a picture pays for more than matplotlib. That is the price of one writer, and the alternative was a figure that ignores the user's format and resolution.
