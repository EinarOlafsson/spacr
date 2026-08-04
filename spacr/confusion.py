"""``C8`` — the confusion matrix as a set of live queries rather than a picture.

A confusion matrix is the most-looked-at and least-acted-on artefact a
classifier run produces. It says "43 uninfected objects were called infected"
and then stops: the 43 are anonymous, so the only thing anyone can do with the
number is feel bad about it. This module is the part that turns each cell back
into the objects it counted, so the number becomes a question you can open.

Three ideas, and they are separate on purpose
---------------------------------------------

**A cell is a set of objects.** :func:`cell_rows` is the whole primitive: the
rows of the out-of-fold prediction table whose true class is *this* and whose
predicted class is *that*. Everything else here is a way of ordering,
splitting or counting that set.

**A confident error and an unsure error have different causes.** They are
therefore two lists, never one sorted list with a gradient in it:

* **high confidence, and wrong** — the model was sure. When a model that is
  right 95% of the time is *certain* about an object and disagrees with the
  annotation, the likeliest explanation is that the annotation is wrong.
  These are the crops to re-label.
* **low confidence, and wrong** — the model was unsure, and fell the wrong
  side. The label is probably fine; the *boundary* is where the work is —
  more examples near it, a better feature, or an admission that the two
  classes are not separable on this stain.

Handing back one list sorted by confidence buries that distinction in the
middle of a scroll. :func:`split_by_confidence` returns the two lists and
guarantees they partition the cell, so nothing is silently dropped between
them.

**A cell is not a homogeneous population.** 43 errors spread evenly over 20
wells is a model problem. 43 errors *all from well A01* is a staining problem,
and re-labelling any of them is wasted work — the fix is upstream, at the
bench. :func:`breakdown_by` and :func:`describe_breakdown` are that check,
made before anyone opens a single crop.

Where the frame comes from
--------------------------

:func:`spacr.classifier_evaluation.evaluate_predictions` writes the table this
module reads: one row per held-out object, with ``true_class``,
``predicted_class``, ``confidence`` (the calibrated probability of the class
the model chose) and the identity columns :func:`~spacr.classifier_evaluation.
sample_identity` parsed out of the crop path. Nothing here imports Qt, so the
same analysis runs in a notebook, and the expensive half never touches the
event loop.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

__all__ = [
    "ConfusionError",
    "Confusion",
    "ConfusionCell",
    "TRUE_COLUMN",
    "PREDICTED_COLUMN",
    "CONFIDENCE_COLUMN",
    "KEY_COLUMNS",
    "BREAKDOWN_LEVELS",
    "confidence_threshold",
    "object_key_column",
    "object_keys_for",
    "key_collisions",
    "cell_rows",
    "split_by_confidence",
    "confusion_counts",
    "rank_confusions",
    "describe_confusions",
    "breakdown_by",
    "describe_breakdown",
]


class ConfusionError(ValueError):
    """A prediction table that cannot answer the question being asked of it.

    Raised rather than returning an empty result. "No rows in that cell" and
    "this frame has no ``true_class`` column" look identical to a caller that
    only sees a count, and the second one has produced a confusion matrix of
    zeros that everybody read as a perfect classifier.
    """


#: The columns :func:`spacr.classifier_evaluation.evaluate_predictions` writes
#: that this module needs. Named here rather than inlined so a rename shows up
#: as one edit and one failing import rather than six string literals.
TRUE_COLUMN = "true_class"
PREDICTED_COLUMN = "predicted_class"
CONFIDENCE_COLUMN = "confidence"

#: Candidate columns that name an object, best first.
#:
#: ``sample`` is the crop path the model was actually given, and
#: :func:`spacr.active_learning.crops_for_object_keys` resolves a ``png_path``
#: directly — so it is the key that needs no translation step and cannot be
#: confused with a *different* object that happens to share a stem. ``object``
#: (the augmentation-normalised stem) and ``basename`` are the fallbacks for a
#: bundle written from a dataset that was moved after training.
KEY_COLUMNS: Tuple[str, ...] = ("object_key", "sample", "object", "basename")

#: The levels a cell can be broken down by, coarsest last. These are the
#: identity levels :func:`spacr.classifier_evaluation.sample_identity` parses
#: out of a crop name, so they are available without touching the database.
BREAKDOWN_LEVELS: Tuple[str, ...] = ("field", "well", "plate")


def confidence_threshold(n_classes: int) -> float:
    """Where "the model was sure" starts, for a ``n_classes``-way problem.

    ``confidence`` is the probability of the class the model *chose*, so it
    can never fall below ``1 / n_classes`` — a two-class model is at 0.5 when
    it is maximally undecided, and a ten-class model is at 0.1. A fixed 0.5
    would therefore call every ten-class error "high confidence" and every
    two-class error nothing at all, which is the sort of default that makes a
    feature look broken on somebody else's data.

    The midpoint between chance and certainty is the honest default: 0.75 for
    two classes, 0.55 for ten. It is a *default*, not a law — every function
    here takes an explicit ``threshold``, and the screen exposes it, because
    where "sure" starts is a property of the assay and not of arithmetic.

    :raises ConfusionError: for fewer than two classes.
    """
    n = int(n_classes)
    if n < 2:
        raise ConfusionError(
            f"a confusion matrix needs at least two classes, got {n}")
    return (1.0 + 1.0 / n) / 2.0


def _require(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise ConfusionError(
            f"this prediction table is missing {missing}; it has "
            f"{sorted(frame.columns)[:8]}... Bundles written by "
            f"spacr.classifier_evaluation.evaluate_predictions carry them.")


def object_key_column(frame: pd.DataFrame) -> str:
    """Which column of ``frame`` names the objects, best first.

    :raises ConfusionError: when none of :data:`KEY_COLUMNS` is present, which
        means the rows cannot be routed anywhere and a "show me these crops"
        button would be a button that raises on click.
    """
    for column in KEY_COLUMNS:
        if column in frame.columns:
            return column
    raise ConfusionError(
        f"no column in this prediction table names an object (looked for "
        f"{list(KEY_COLUMNS)}), so its rows cannot be opened as crops.")


def _key_values(frame: pd.DataFrame,
                column: Optional[str] = None) -> Tuple[List[str], List[str]]:
    """``(every key, the unique ones in first-seen order)``."""
    column = column or object_key_column(frame)
    if column not in frame.columns:
        raise ConfusionError(
            f"prediction table has no {column!r} column to name objects with")
    values = [str(v) for v in frame[column].tolist()]
    seen = set()
    unique: List[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            unique.append(value)
    return values, unique


def object_keys_for(frame: pd.DataFrame,
                    *, column: Optional[str] = None) -> pd.Index:
    """The object keys of ``frame``'s rows, in frame order, de-duplicated.

    Order is load-bearing — it is what carries "worst error first" through
    :func:`spacr.qt.linked_selection.open_objects` — so this preserves it
    rather than sorting or using a set.

    Duplicates are dropped. They are not assumed away: two rows can
    legitimately share a key — augmented copies of one crop collapse onto the
    same ``object`` stem, and :func:`spacr.selection.object_keys` is not
    injective when an identity component contains the key separator — so
    :func:`key_collisions` says how many rows the grid will be short of the
    count the matrix showed, rather than leaving that to be discovered by
    counting tiles.
    """
    _values, unique = _key_values(frame, column)
    return pd.Index(unique, dtype=object)


def key_collisions(frame: pd.DataFrame,
                   *, column: Optional[str] = None) -> int:
    """How many rows of ``frame`` share a key with an earlier row.

    Zero for a healthy prediction table. Non-zero means the crop grid will
    hold fewer objects than the confusion cell counted, and the difference is
    this number — worth saying on screen rather than leaving as an unexplained
    discrepancy between a matrix and a grid.
    """
    values, unique = _key_values(frame, column)
    return len(values) - len(unique)


def cell_rows(predictions: pd.DataFrame, true_class: Any,
              predicted_class: Any) -> pd.DataFrame:
    """The rows one confusion-matrix cell counted, in the table's own order.

    Compared as text, deliberately. A confusion matrix read back from CSV has
    string class names in its index and header, while the prediction table may
    have kept an integer or a categorical — and a cell that matched nothing
    because ``1 != "1"`` renders as an empty grid with no error, which reads
    as "no mistakes here".

    :returns: a copy, so a caller sorting it cannot reorder the bundle.
    """
    _require(predictions, (TRUE_COLUMN, PREDICTED_COLUMN))
    if predictions.empty:
        return predictions.iloc[:0].copy()
    mask = (
        predictions[TRUE_COLUMN].astype(str).to_numpy() == str(true_class)
    ) & (
        predictions[PREDICTED_COLUMN].astype(str).to_numpy()
        == str(predicted_class)
    )
    return predictions.loc[mask].copy()


def split_by_confidence(rows: pd.DataFrame, threshold: float
                        ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split one cell into *suspect the label* and *suspect the boundary*.

    :param rows: a cell, from :func:`cell_rows`.
    :param threshold: confidence at or above which the model counts as sure.
    :returns: ``(high, low)``.

        ``high`` is confidence ``>= threshold``, **most confident first** —
        the model's flattest contradictions of the annotator, which is the
        order to re-label in.

        ``low`` is confidence ``< threshold``, **least confident first** — the
        objects nearest the decision boundary, which is the order to look at
        when asking whether the boundary is in the right place.

    The two always partition ``rows`` exactly: every row is in one and no row
    is in both, including rows whose confidence is missing. A NaN confidence
    goes to ``low``, because "we do not know how sure the model was" is not
    evidence that the annotation is wrong, and quietly dropping those rows
    would make the two lists sum to less than the cell they came from — a
    discrepancy nobody notices until they try to reconcile the totals.

    Sorting is stable, so rows tied on confidence keep the table's order and
    the split is reproducible run to run.
    """
    if rows.empty:
        empty = rows.iloc[:0].copy()
        return empty, empty.copy()
    if CONFIDENCE_COLUMN not in rows.columns:
        raise ConfusionError(
            f"cannot split a cell by confidence: no {CONFIDENCE_COLUMN!r} "
            f"column. An evaluation bundle written by spaCR always has one.")
    confidence = pd.to_numeric(rows[CONFIDENCE_COLUMN], errors="coerce")
    sure = (confidence >= float(threshold)).fillna(False).to_numpy()
    high = rows.loc[sure].copy()
    low = rows.loc[~sure].copy()
    if not high.empty:
        high = high.iloc[np.argsort(
            -pd.to_numeric(high[CONFIDENCE_COLUMN], errors="coerce")
            .to_numpy(dtype=float), kind="stable")]
    if not low.empty:
        # NaN sorts last under argsort, which is what we want: a row with no
        # confidence is not "the least confident", it is unknown, and it
        # belongs after the rows that actually sat on the boundary.
        low = low.iloc[np.argsort(
            pd.to_numeric(low[CONFIDENCE_COLUMN], errors="coerce")
            .to_numpy(dtype=float), kind="stable")]
    return high, low


def confusion_counts(predictions: pd.DataFrame,
                     classes: Optional[Sequence[Any]] = None) -> pd.DataFrame:
    """Counts per (true, predicted) pair, as a square frame.

    Recomputed from the prediction table rather than read from the bundle's
    ``confusion_counts.csv``, so that the matrix on screen and the objects a
    cell opens can never disagree — the two used to be separate artefacts and
    a filtered prediction table silently produced a matrix nobody could
    reproduce.

    :param classes: the class order. Defaults to every class appearing as a
        true label or a prediction, sorted, so a class the model never chose
        still gets its column rather than vanishing from the matrix.
    """
    _require(predictions, (TRUE_COLUMN, PREDICTED_COLUMN))
    if classes is None:
        names = sorted({str(v) for v in predictions[TRUE_COLUMN]}
                       | {str(v) for v in predictions[PREDICTED_COLUMN]})
    else:
        names = [str(c) for c in classes]
    matrix = pd.DataFrame(0, index=pd.Index(names, name=TRUE_COLUMN),
                          columns=names, dtype=int)
    if predictions.empty:
        return matrix
    pairs = zip(predictions[TRUE_COLUMN].astype(str),
                predictions[PREDICTED_COLUMN].astype(str))
    for true_value, predicted_value in pairs:
        if true_value in matrix.index and predicted_value in matrix.columns:
            matrix.at[true_value, predicted_value] += 1
    return matrix


@dataclass(frozen=True)
class Confusion:
    """One off-diagonal cell, ranked against the others.

    :ivar true_class: what it really was.
    :ivar predicted_class: what the model said.
    :ivar count: how many objects.
    :ivar share_of_errors: this cell's share of *all* mistakes. The number
        that says where to spend the next hour.
    :ivar rate_within_true: this cell as a fraction of the true class's whole
        row. The number that says how bad it is for that class — a cell can
        be 45% of all errors and still only 2% of a huge class, or 8% of the
        errors and 60% of a small one, and those are different problems.
    """

    true_class: str
    predicted_class: str
    count: int
    share_of_errors: float
    rate_within_true: float

    def describe(self) -> str:
        """One line, in the words a person would use."""
        return (f"{self.true_class} → {self.predicted_class}: {self.count} "
                f"object(s), {self.share_of_errors:.0%} of all errors and "
                f"{self.rate_within_true:.0%} of everything truly "
                f"{self.true_class}")


def rank_confusions(counts: pd.DataFrame) -> List[Confusion]:
    """Every off-diagonal cell, worst first.

    "Worst" is share of total errors, because that is what answers "what
    should I fix next". Ties break on the within-class rate and then on the
    class names, so the ranking is stable rather than dependent on dict order.

    :param counts: a square frame from :func:`confusion_counts` or from a
        bundle's ``confusion_counts.csv``.
    """
    if counts.empty:
        return []
    values = counts.to_numpy(dtype=float)
    rows = [str(v) for v in counts.index]
    columns = [str(v) for v in counts.columns]
    total_errors = float(values.sum() - np.trace(
        values[:min(values.shape), :min(values.shape)]
        if values.shape[0] == values.shape[1] else values))
    if values.shape[0] == values.shape[1]:
        total_errors = float(values.sum() - np.trace(values))
    else:
        # A non-square matrix has no diagonal to trace; "off-diagonal" then
        # means every cell whose row and column names differ.
        total_errors = float(sum(
            values[i, j] for i, r in enumerate(rows)
            for j, c in enumerate(columns) if r != c))
    out: List[Confusion] = []
    for i, true_name in enumerate(rows):
        row_total = float(values[i].sum())
        for j, predicted_name in enumerate(columns):
            if true_name == predicted_name:
                continue
            count = int(values[i, j])
            if count <= 0:
                continue
            out.append(Confusion(
                true_class=true_name,
                predicted_class=predicted_name,
                count=count,
                share_of_errors=(count / total_errors) if total_errors else 0.0,
                rate_within_true=(count / row_total) if row_total else 0.0))
    out.sort(key=lambda c: (-c.share_of_errors, -c.rate_within_true,
                            c.true_class, c.predicted_class))
    return out


def describe_confusions(counts: pd.DataFrame, *, limit: int = 3) -> str:
    """The off-diagonal mass in words, worst first.

    A matrix of numbers makes the reader do the ranking, and the ranking is
    the only part of it anybody acts on. This says it: *"Your worst confusion
    is uninfected → infected: 43 object(s), 45% of all errors…"*

    :param limit: how many confusions to name before summarising the rest.
    :returns: one or more lines, no trailing newline. Never empty — a perfect
        classifier gets a sentence saying so, because a blank panel reads as
        a panel that failed to load.
    """
    ranked = rank_confusions(counts)
    if not ranked:
        return ("No off-diagonal mass at all: every object was predicted as "
                "its annotated class. Check the leakage audit before "
                "believing it.")
    lines = [f"Your worst confusion is {ranked[0].describe()}."]
    for confusion in ranked[1:max(1, int(limit))]:
        lines.append(f"Then {confusion.describe()}.")
    rest = ranked[max(1, int(limit)):]
    if rest:
        share = sum(c.share_of_errors for c in rest)
        lines.append(f"The remaining {len(rest)} confusion(s) hold "
                     f"{share:.0%} of the errors between them.")
    return "\n".join(lines)


def breakdown_by(rows: pd.DataFrame, level: str) -> pd.DataFrame:
    """Where one cell's objects came from, most concentrated first.

    :param rows: a cell, from :func:`cell_rows`.
    :param level: an identity column — ``"plate"``, ``"well"`` or
        ``"field"``. Any column of ``rows`` is accepted, so a bundle carrying
        extra metadata can be broken down by it too.
    :returns: a frame of ``level``, ``count`` and ``share``, sorted by count
        descending then by name, so the answer is stable across runs.
    :raises ConfusionError: for a level the frame does not carry — silently
        returning an empty breakdown would read as "the errors are spread
        evenly", which is the opposite of what an absent column means.
    """
    level = str(level)
    if level not in rows.columns:
        raise ConfusionError(
            f"cannot break this cell down by {level!r}: the prediction table "
            f"has no such column. Available identity columns: "
            f"{[c for c in BREAKDOWN_LEVELS if c in rows.columns]}")
    if rows.empty:
        return pd.DataFrame({level: [], "count": [], "share": []})
    counts = rows[level].astype(str).value_counts()
    total = int(counts.sum())
    out = pd.DataFrame({
        level: [str(v) for v in counts.index],
        "count": counts.to_numpy(dtype=int),
        "share": counts.to_numpy(dtype=float) / float(total or 1),
    })
    return out.sort_values(["count", level], ascending=[False, True],
                           kind="stable").reset_index(drop=True)


#: Above this share in one group, a cell is a *source* problem rather than a
#: model problem. Two thirds rather than a half: a binary assay with two
#: plates would trip a half-share threshold by chance.
_CONCENTRATION = 2.0 / 3.0

#: Below this many objects, concentration means nothing — three errors all
#: from one well is what three errors look like.
_CONCENTRATION_FLOOR = 5


def describe_breakdown(rows: pd.DataFrame, level: str) -> str:
    """One cell's origin, in words, with the verdict spelled out.

    The point of this line is to stop wasted work. If all 43 errors come from
    well A01, re-labelling any of them corrects nothing that will recur — the
    fix is a staining or a focus problem at the bench, and the crops are
    evidence for that conversation rather than a re-annotation queue.

    :returns: one or two lines, no trailing newline.
    """
    table = breakdown_by(rows, level)
    if table.empty:
        return f"No objects in this cell, so nothing to break down by {level}."
    total = int(table["count"].sum())
    top = table.iloc[0]
    lines = [
        f"{total} object(s) across {len(table)} {level}(s); worst is "
        f"{top[level]} with {int(top['count'])} ({top['share']:.0%})."
    ]
    if (total >= _CONCENTRATION_FLOOR
            and float(top["share"]) >= _CONCENTRATION):
        lines.append(
            f"{int(top['count'])} of {total} come from a single {level} "
            f"({top[level]}) — that is a {level}-level problem (staining, "
            f"focus, seeding), not a model problem. Fix it upstream rather "
            f"than re-labelling these crops.")
    elif len(table) > 1:
        lines.append(
            f"Spread over {len(table)} {level}(s), so this confusion is the "
            f"model's, not one {level}'s.")
    return "\n".join(lines)


@dataclass(frozen=True)
class ConfusionCell:
    """One clicked cell, already split and already counted.

    Built by :meth:`build` so that the screen does one call and gets
    everything it draws — the two lists, the keys to route, and the sentences
    — rather than orchestrating five functions in a mouse handler.

    :ivar rows: every object in the cell, in table order.
    :ivar high: confidence ``>= threshold``, most confident first. Suspect the
        label.
    :ivar low: confidence ``< threshold``, least confident first. Suspect the
        boundary.
    """

    true_class: str
    predicted_class: str
    threshold: float
    rows: pd.DataFrame
    high: pd.DataFrame
    low: pd.DataFrame

    @classmethod
    def build(cls, predictions: pd.DataFrame, true_class: Any,
              predicted_class: Any, *,
              threshold: Optional[float] = None,
              n_classes: Optional[int] = None) -> "ConfusionCell":
        """Resolve a cell and split it.

        :param threshold: where "sure" starts. Defaults to
            :func:`confidence_threshold` of ``n_classes``, or of the number of
            distinct classes in ``predictions`` when that is not given.
        """
        rows = cell_rows(predictions, true_class, predicted_class)
        if threshold is None:
            if n_classes is None:
                _require(predictions, (TRUE_COLUMN, PREDICTED_COLUMN))
                n_classes = len(
                    {str(v) for v in predictions[TRUE_COLUMN]}
                    | {str(v) for v in predictions[PREDICTED_COLUMN]})
            threshold = confidence_threshold(max(2, int(n_classes)))
        high, low = split_by_confidence(rows, float(threshold))
        return cls(true_class=str(true_class),
                   predicted_class=str(predicted_class),
                   threshold=float(threshold), rows=rows, high=high, low=low)

    @property
    def is_error(self) -> bool:
        """Whether this is an off-diagonal cell."""
        return self.true_class != self.predicted_class

    def __len__(self) -> int:
        return len(self.rows)

    def keys(self, which: str = "all", *,
             column: Optional[str] = None) -> pd.Index:
        """Object keys for ``"high"``, ``"low"`` or ``"all"``, in list order.

        ``"all"`` is high-confidence first and then low-confidence — not table
        order — so a caller that opens the whole cell still gets the objects
        most likely to be mislabelled at the front of the grid.
        """
        which = str(which).lower()
        if which == "high":
            frame = self.high
        elif which == "low":
            frame = self.low
        elif which == "all":
            frame = (pd.concat([self.high, self.low])
                     if len(self.rows) else self.rows)
        else:
            raise ConfusionError(
                f"which must be 'high', 'low' or 'all', got {which!r}")
        if frame.empty:
            return pd.Index([], dtype=object)
        return object_keys_for(frame, column=column)

    def reason(self, which: str = "all") -> str:
        """The line the receiving view puts above the crops.

        Required by :class:`spacr.selection.ObjectRequest`, and load-bearing:
        a grid of twelve crops that does not say why reads as the whole
        dataset. It names the *hypothesis*, not just the cell, so the person
        looking at the crops knows what they are being asked to decide.
        """
        head = f"annotated {self.true_class} · predicted {self.predicted_class}"
        which = str(which).lower()
        if which == "high":
            return (f"{head} · model was sure (≥ {self.threshold:.2f}) — "
                    f"suspect the label")
        if which == "low":
            return (f"{head} · model was unsure (< {self.threshold:.2f}) — "
                    f"suspect the boundary")
        return head

    def describe(self) -> str:
        """The cell in words: the split, and what each half means."""
        if not len(self.rows):
            return (f"No objects were annotated {self.true_class} and "
                    f"predicted {self.predicted_class}.")
        kind = "errors" if self.is_error else "correct predictions"
        return (
            f"{len(self.rows)} {kind}: {len(self.high)} at confidence ≥ "
            f"{self.threshold:.2f} (suspect the label) and {len(self.low)} "
            f"below it (suspect the boundary).")
