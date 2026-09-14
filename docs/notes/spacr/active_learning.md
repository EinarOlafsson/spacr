# Notes from `spacr/active_learning.py`

Prose lifted out of `spacr/active_learning.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [least_confidence](#least_confidence) (1 entry)
- [disagreement](#disagreement) (2 entries)
- [rank_by_uncertainty](#rank_by_uncertainty) (1 entry)
- [predict_probabilities](#predict_probabilities) (1 entry)
- [_connect](#_connect) (1 entry)
- [_resolve_pred_columns](#_resolve_pred_columns) (1 entry)
- [build_queue](#build_queue) (1 entry)
- [_well_key](#_well_key) (1 entry)
- [_concentration](#_concentration) (1 entry)
- [annotation_coverage](#annotation_coverage) (1 entry)
- [crops_for_object_keys](#crops_for_object_keys) (5 entries)
- [crops_for_object_keys._register](#crops_for_object_keys_register) (1 entry)
- [_utc_now](#_utc_now) (1 entry)
- [should_stop](#should_stop) (1 entry)
- [holdout_report](#holdout_report) (4 entries)
- [retrain_round](#retrain_round) (6 entries)
- [_build_round_model](#_build_round_model) (4 entries)
- [_write_round_card](#_write_round_card) (1 entry)

## least_confidence

### lines 400-407

```python
def least_confidence(probs: Any, normalize: bool = False) -> np.ndarray:
```

Uncertainty measures

Every measure is oriented the SAME way: larger means less certain, minimum at a one-hot row, maximum at the uniform row. That is what lets rank_by_uncertainty treat them interchangeably, and it is why margin() returns 1 − (p₁ − p₂) rather than the margin itself.

## disagreement

### line 560, trailing  _(unsure)_

```python
stack = np.stack(coerced, axis=0)
```

(M, N, C)

### line 572, trailing  _(unsure)_

```python
score = np.maximum(score, 0.0)
```

MI is non-negative

## rank_by_uncertainty

### lines 664-666

```python
primary = np.where(finite, -values, np.inf)
```

Ascending sort on -score = descending sort on score; NaN -> +inf, so unusable rows land at the back instead of wherever NaN happens to compare.

## predict_probabilities

### lines 674-676  _(unsure)_

```python
def predict_probabilities(model: Callable[[Any], Any], batches: Iterable[Any],
```

Live-model bridge — the only torch in this file, imported lazily

## _connect

### lines 773-774

```python
from .database_concurrency import connect as _connect_database
```

Shared helper: 30s busy timeout, not sqlite's 5s default, which Measure's concurrent writers routinely exceed (#15).

## _resolve_pred_columns

### lines 821-827

```python
for prefix in (ROUND_PRED_PREFIX, "pred_", "prob_", "score_"):
```

Multi-class columns first. ``al_prob_`` comes FIRST on purpose: it is written by :func:`retrain_round`, i.e. by a model trained on the labels made in this very annotation session, and it is therefore always fresher than whatever the last full Classify run left in ``pred``. Without this, round 2 of an active-learning loop re-ranks against the round-0 model and serves back the same crops — the loop looks closed and is not.

## build_queue

### lines 1029-1031

```python
spread_cols = div_cols or [c for c in DIVERSITY_GROUPS["well"]
```

How spread the queue is gets reported whatever the strategy — it is most informative when diversity is OFF, because that is where the collapse onto two wells shows up as a number.

## _well_key

### lines 1315-1317  _(unsure)_

```python
def _well_key(frame: pd.DataFrame, columns: Sequence[str]) -> pd.Series:
```

Annotation coverage — how many labels, of which class, from where

## _concentration

### line 1353  _(unsure)_

```python
"effective_groups": (1.0 / hhi) if hhi else 0.0,
```

1/HHI: how many equally-sized groups the labels are "worth".

## annotation_coverage

### lines 1418-1421

```python
n_unfiltered = len(frame)
```

The denominator has to describe the SAME population as the numerator. Counting the rows before the filter printed "12 of 20 crops annotated" for a filtered population of 12 in which nothing was left to annotate. build_queue already gets this right; this says it the same way.

## crops_for_object_keys

### lines 1676-1682

```python
by_escaped_key: Dict[str, Tuple[str, Optional[int]]] = {}
```

Keys in the spelling `selection.object_keys` actually emits, kept apart from the raw ones and consulted FIRST. Both are needed — a key composed before the escape existed is raw, a routed selection's is escaped — but they can collide across rows, since one field literally named 'f%5F1' spells its raw key the way a field named 'f_1' spells its escaped one. The escaped spelling is the authoritative one, so it wins; a single extra dict does that without a second pass over the crop table.

### lines 1707-1711

```python
stated = [(column, _object_label(row[index[column]]))
```

WHICH id column holds the label is the crop's object type:

`filepaths_to_database` writes exactly one per row, the one for the crop mode it was called with. That is what lets a nucleus crop and a pathogen crop of the same label in the same field be told apart — they used to resolve to one key and the first row in the table won.

### lines 1721-1723

```python
label = stated[0][1]
```

Two id columns filled is a row that does not say what it is. The old first-wins precedence still gives a label to key on; claiming a type from it would be a guess.

### lines 1733-1739

```python
if any(c in p for p in parts for c in KEY_ESCAPED_CHARACTERS):
```

`selection._compose` percent-escapes a component that would otherwise smuggle the separator into the key, so a fieldID of 'f_1' reaches us as 'f%5F1' and a raw join misses it entirely. The guard keeps the common path — every plate whose ids are the ordinary ones — at one scan per component and no second key at all: `png_list` has a row per crop, so this runs millions of times on a real screen.

### lines 1761-1763

```python
reduced = untyped_object_key(wanted_key)
```

A typed key against a crop table that cannot say what its rows are. Dropping the type is the honest fallback: the row has not contradicted the key, it has said nothing.

## crops_for_object_keys._register

### lines 1689-1691

```python
target.setdefault("_".join(composed + [label]), entry)
```

Both spellings, so a caller working from either side resolves. The untyped one is first-wins on purpose: it is an under-specified name, and it named one of these crops before the type existed.

## _utc_now

### lines 1853-1855  _(unsure)_

```python
def _utc_now() -> str:
```

Round bookkeeping — the loop's memory

## should_stop

### line 2248  _(unsure)_

```python
accumulated = 0
```

Walk back until the window covers enough NEW labels.

## holdout_report

### lines 2341-2343  _(unsure)_

```python
def holdout_report(y_true: Any, probs: Any,
```

Retraining a round: fit on the labels so far, re-score, re-rank

### lines 2406-2409

```python
truth_classes = (int(y_true[scorable].max()) + 1) if scorable.any() else 0
```

The matrix covers every class the TRUTH names, not only the ones the head can emit. A class outside the head's columns can never be predicted, so its row is all-error — which is the honest score, and the reason its column stays empty.

### lines 2428-2430

```python
col_sums = matrix.sum(axis=0)
```

Macro F1 straight off the matrix — no sklearn import on this path, and the number is then provably the same function of the matrix as the accuracy beside it.

### lines 2446-2447  _(unsure)_

```python
"n": total,
```

The matrix total, not len(y_true): every figure below is a function of the matrix, so n has to describe the same rows they do.

## retrain_round

### lines 2714-2717

```python
crops = crops.loc[~crops.index.duplicated(keep="first")]
```

A database measured with more than one crop_mode holds several rows per prcfo. A duplicated key here fans the feature join out, so the label vector and the feature matrix stop lining up row for row and the model is fitted against the wrong labels — silently, with a plausible score.

### lines 2733-2740

```python
labelled_mask = (
```

HUMAN ANSWERS ONLY, NOT THE MACHINE'S OWN GUESSES. `spacr.suggest` marks a proposal by adding `SUGGESTION_OFFSET` to the class it proposes, so a suggested 1 is stored as 11 in this very column. A bare `notna()` reads those as classes 11 and 12 and fits them as if a person had written them -- a model trained on its own previous output, whose card would not say so. The GUI guards both buttons; anything calling this function directly was not guarded at all, which is why the filter belongs HERE and not at each caller.

### lines 2752-2754

```python
train_index = crops.index[labelled_mask.to_numpy()]
```

INDEX-BASED FROM HERE, not mask-based, because `synthetic_negatives` ADDS rows that carry no label in the column and `balance` DROPS rows that do. A boolean mask over `crops` can express neither.

### lines 2761-2767

```python
present = class_values[0]
```

THE DELIBERATE LIE, ASKED FOR IN SO MANY WORDS: "if only one class randomly choose the same number of images as is annotated for the other class". A random draw from the unannotated pool is MOSTLY-negative, not negative, so what comes back is a RANKING and not a verdict -- and every caller is told so, in `notes` and on the model card, because a card that does not say the negatives were invented describes a model that does not exist.

### lines 2805-2812

```python
train_index, raw_labels, dropped = _downsample_to_smallest(
```

DOWNSAMPLED, NOT WEIGHTED, AND THAT CHANGES THE ANSWER RATHER THAN THE COST. The default estimators already pass `class_weight="balanced"`, so "balanced" alone describes two different models -- which is why `notes` and the card say WHICH. The maintainer asked for the smaller class ("use the class with fewer"), and downsampling is also what leaves the returned probability directly readable as the confidence a suggestion sort depends on: a reweighted fit's probability is not.

### lines 2839-2842

```python
from .classifier_evaluation import (
```

Keep sklearn/scipy off the import-only queue-ranking path. Recent SciPy probes optional array backends while importing sklearn; that path must not run merely to rank an existing score column (and breaks a legitimate no-torch process where ``sys.modules['torch']`` is explicitly blocked).

## _build_round_model

### lines 3035-3037

```python
return Pipeline([
```

Scaled, because measurement features span areas in the thousands and intensities in the fractions, and an unscaled linear model on that is a model of whichever column has the biggest units.

### lines 3051-3064

```python
try:
```

OFFERED, NOT DEFAULTED, and it is an OPTIONAL dependency.

The Suggest request named XGBoost, and `gradient_boosting` above is sklearn's implementation of the same gradient-boosted-trees method -- already installed, already understood by the round machinery. The maintainer asked for both: this one for anyone who wants XGBoost's own implementation or its hyperparameters, that one so the path still runs on a machine that has not installed a second boosting library.

THE REFUSAL NAMES THE ALTERNATIVE, because "no module named xgboost" from inside a retrain round tells a user nothing about what to do next, and the honest answer is that they already have a gradient booster.

### lines 3077-3078

```python
objective=("binary:logistic" if n_classes <= 2
```

The round encodes classes to 0..n-1 before fitting, so the objective follows the class count rather than being guessed.

### lines 3082-3083  _(unsure)_

```python
)
```

NOT `use_label_encoder`, which xgboost removed; passing it warns on 1.x and raises on 2.x.

## _write_round_card

### lines 3177-3186

```python
"balancing": dict(balancing),
```

WHICH BALANCING, NOT MERELY THAT THERE WAS SOME. The default estimators already pass `class_weight="balanced"`, so a card saying "balanced" without saying how describes two different models -- a reweighted fit and a downsampled one do not have the same probabilities, and `spacr.suggest` sorts on those probabilities.

`synthetic_negatives` is the more serious of the two: a card that does not say the negatives were invented describes a model that does not exist.
