"""Active-learning queue — order unannotated crops by model uncertainty.

Annotation is the bottleneck in every screen. A crop the model already
calls with 0.999 teaches the model nothing when a human labels it; the
crops worth a person's afternoon are the ones sitting on the decision
boundary. This module turns an already-scored ``measurements.db`` into a
work queue ordered so the informative crops come first.

Public API
----------
``least_confidence(probs)`` / ``margin(probs)`` / ``entropy(probs)``
    Per-row uncertainty scores. Pure numpy — no torch, no database.
``disagreement(prob_sets)``
    Spread across an ensemble or a set of MC-dropout passes.
``rank_by_uncertainty(probs, measure=…)``
    Row indices, most uncertain first, deterministically tie-broken.
``build_queue(db_path, annotation_column, …)``
    The queue itself, read straight out of ``png_list``.
``queue_rows(queue)``
    The queue as ``[(png_path, None), …]`` — the shape the Annotate
    screen already paginates (see :func:`spacr.qt.annotate_engine.fetch_page`).
``format_queue_summary(queue)``
    The queue's shape, class balance and caveats, as text.
``probabilities_from_logits(logits)`` / ``as_probabilities(scores)``
    Raw-head-output → probability matrix, both head shapes handled.
``predict_probabilities(model, batches)``
    Optional live-model bridge; the only function that touches torch.

Things this module refuses to get wrong
---------------------------------------
**A softmax is not a probability.** Everything here is called an
uncertainty *score*, never a confidence. Modern networks are badly
calibrated — typically over-confident, and more so the deeper they are
(Guo et al., 2017) — so a 0.87 from the head is not "87 % sure". The
scores are used for one thing only: putting crops in an order. Nothing
in this module reports a calibrated probability, and neither should
anything built on it. See :data:`CALIBRATION_NOTE`.

**Already-annotated crops never enter the queue.** ``NULL`` in the
annotation column is the abstention marker — the crop has not been
looked at. ``0`` is a real class, and a queue that re-serves it wastes
exactly the resource this module exists to save. The two are
distinguished by ``IS NULL``, never by falsiness. (Same convention as
:mod:`spacr.agreement`.)

**The two head shapes are handled separately.** The classifier head
emits either a single logit (binary; needs a sigmoid) or C logits
(multiclass; needs a softmax) — see
:func:`spacr.deep_spacr.apply_model_to_tar`. Pushing a single-logit
column through a softmax yields a column of 1.0 and destroys the
ordering; pushing C logits through a sigmoid can *invert* it. Both are
silent failures that produce a confident-looking queue full of nonsense,
so :func:`probabilities_from_logits` branches on the shape and the tests
pin both directions.

**``margin`` and ``least_confidence`` are the same ranking on two
classes.** For C = 2, ``1 − (p₁ − p₂) = 2·min(p, 1−p) = 2·(1 − max p)``:
a linear transform, so identical order and identical ties. They are not
two independent choices on a binary screen — they differ only from three
classes up, where ``margin`` looks at the top *two* classes while
``least_confidence`` looks only at the top one and ``entropy`` looks at
the whole distribution.

**Pure uncertainty ranking collapses onto one region of feature space.**
The 100 most uncertain crops on a real plate are routinely 100 near
copies from the same two wells — one ambiguity, labelled a hundred
times, for almost the information of labelling it once. So the queue is
diversified by default (round-robin across wells) and
:func:`format_queue_summary` prints how many wells the queue actually
covers. What that costs is stated plainly in :func:`build_queue`.

**Uncertainty sampling skews toward the majority class.** On a screen
that is 98 % negative, the decision boundary is mostly populated by
negatives, so the queue will be too. That is not a bug to hide — it is
reported as a class-balance table so the annotator can see it and cap or
rebalance if they want to.

**Determinism.** Same inputs, same seed, same order — including ties.
Ties break on row index by default; ``seed`` swaps in a seeded
permutation, which is still reproducible. Nothing here consults an
unseeded RNG.

Nothing in this module imports torch at module scope; the ranking maths
is numpy only, so the Qt screen can build a queue without waking a
multi-second import chain. :func:`predict_probabilities` imports torch
lazily and is the only entry point that needs it.
"""
from __future__ import annotations

import os
import sqlite3
from collections import OrderedDict
from typing import (Any, Callable, Dict, Iterable, List, Optional, Sequence,
                    Tuple)
from urllib.parse import quote as _urlquote

import numpy as np
import pandas as pd

from . import schema
from .agreement import PNG_KEY, PNG_TABLE

#: ``png_list``'s per-crop id column, and the object type it means.
#:
#: ``filepaths_to_database`` writes exactly one of these per row — the one for
#: the crop mode it was called with — so which column holds a label *is* the
#: crop's object type. Derived from :data:`spacr.schema.OBJECT_TYPES` rather
#: than written out, which keeps it identical to
#: :data:`spacr.utils.PNG_CROP_MODE_BY_ID_COLUMN` without importing
#: :mod:`spacr.utils` and its multi-second chain into this module (see the
#: module docstring). ``tests/test_active_learning_loop.py`` pins the two
#: together.
PNG_ID_COLUMN_TYPES: Dict[str, str] = {
    f"{object_type}_id": object_type for object_type in schema.OBJECT_TYPES
}

__all__ = [
    "CALIBRATION_NOTE",
    "DEFAULT_MEASURE",
    "DIVERSITY_GROUPS",
    "PNG_ID_COLUMN_TYPES",
    "PNG_KEY",
    "PNG_TABLE",
    "PRED_COLUMN_CANDIDATES",
    "ROUND_LOG_TABLE",
    "ROUND_PRED_PREFIX",
    "ROUND_TABLE",
    "UNCERTAINTY_MEASURES",
    "RoundResult",
    "StoppingVerdict",
    "annotation_coverage",
    "as_probabilities",
    "build_queue",
    "crops_for_object_keys",
    "disagreement",
    "ensure_round_tables",
    "entropy",
    "format_coverage_summary",
    "format_learning_curve",
    "format_queue_summary",
    "holdout_report",
    "label_rounds",
    "learning_curve",
    "least_confidence",
    "margin",
    "next_round",
    "predict_probabilities",
    "probabilities_from_logits",
    "queue_rows",
    "rank_by_uncertainty",
    "record_labels",
    "record_round",
    "resolve_measure",
    "retrain_round",
    "round_features",
    "should_stop",
    "uncertainty_scores",
]

#: Printed under every queue summary, and the reason no function here
#: returns anything called a confidence.
CALIBRATION_NOTE = (
    "Uncertainty scores rank crops; they are not calibrated probabilities. "
    "A neural network's sigmoid/softmax output is systematically "
    "over-confident (Guo et al., 2017), so a score of 0.5 does not mean "
    "'a coin flip' and a 0.99 does not mean '99 % sure'. Read the ORDER, "
    "not the numbers."
)

#: Default measure. Entropy uses the whole distribution rather than only
#: the top one or two classes, which is the safer default once C > 2 and
#: is equivalent-in-spirit at C = 2.
DEFAULT_MEASURE = "entropy"

#: Column names :func:`build_queue` looks for when ``pred_column`` is not
#: given, best first. ``pred`` is what the CV classifier writes via
#: :func:`spacr.predictions.merge_cv_predictions` (the positive-class
#: probability, one REAL column); ``ml_pred`` is the same quantity from the
#: classical-ML classifier (:func:`spacr.predictions.merge_ml_predictions`).
#: The CV score comes first because a database carrying both was scored by a
#: model trained on crops, which is what the queue is picking crops for.
PRED_COLUMN_CANDIDATES: Tuple[str, ...] = ("pred", "ml_pred", "prediction", "score")

#: Metadata columns each named diversity strategy stratifies over, in the
#: order they are combined into a group key.
DIVERSITY_GROUPS: Dict[str, Tuple[str, ...]] = {
    "plate": ("plateID",),
    "well": ("plateID", "rowID", "columnID"),
    "row": ("plateID", "rowID"),
    "column": ("plateID", "columnID"),
    "field": ("plateID", "rowID", "columnID", "fieldID"),
}

#: Columns copied into the queue frame when ``png_list`` has them, so the
#: caller can see where each crop came from without a second query.
_METADATA_COLUMNS: Tuple[str, ...] = (
    "plateID", "rowID", "columnID", "fieldID", "prc", "prcfo", "file_name",
    "cell_id", "nucleus_id", "pathogen_id", "cytoplasm_id",
)

#: Prefix of the per-class probability columns :func:`retrain_round` writes
#: back into ``png_list``. One column per class (``al_prob_0``, ``al_prob_1``,
#: …) rather than one positive-class score, so a three-class screen re-ranks
#: on the full distribution rather than on a collapsed binary proxy.
ROUND_PRED_PREFIX = "al_prob_"

#: Per-label provenance: which round each annotation was made in.
ROUND_TABLE = "annotation_rounds"

#: Per-round record: the learning curve, one row per retrain.
ROUND_LOG_TABLE = "annotation_round_log"


# ---------------------------------------------------------------------------
# Array plumbing
# ---------------------------------------------------------------------------

def _to_numpy(values: Any) -> np.ndarray:
    """Return ``values`` as a float array, accepting torch tensors.

    Tensors are converted by duck-typing (``detach``/``cpu``/``numpy``)
    rather than by importing torch, so the ranking maths stays usable in
    a process that has never loaded it.
    """
    obj = values
    if hasattr(obj, "detach"):
        obj = obj.detach()
    if hasattr(obj, "cpu"):
        obj = obj.cpu()
    if not isinstance(obj, np.ndarray) and hasattr(obj, "numpy"):
        obj = obj.numpy()
    return np.asarray(obj, dtype=float)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable logistic function."""
    out = np.empty_like(x, dtype=float)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    exp_x = np.exp(x[~pos])
    out[~pos] = exp_x / (1.0 + exp_x)
    return out


def _softmax(x: np.ndarray) -> np.ndarray:
    """Row-wise softmax, max-subtracted so large logits do not overflow."""
    shifted = x - np.max(x, axis=1, keepdims=True)
    exp = np.exp(shifted)
    total = exp.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return exp / total


def _as_matrix(values: Any) -> np.ndarray:
    """Coerce to a 2-D ``(N, C)`` float array.

    ``(N,)`` becomes ``(N, 1)``. A scalar or a 3-D array is a caller
    error, not something to guess about.
    """
    arr = _to_numpy(values)
    if arr.ndim == 0:
        raise ValueError(
            "Uncertainty needs a per-crop score array, got a scalar. Pass "
            "shape (N,) for a single-logit/binary head or (N, C) for a "
            "C-class head.")
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(
            f"Expected a (N,) or (N, C) array of scores, got shape "
            f"{arr.shape}. For an ensemble of score sets use "
            f"disagreement(), which takes a list of them.")
    if arr.shape[1] == 0:
        raise ValueError(
            f"Got {arr.shape[0]} rows with no columns: there are no class "
            f"scores to be uncertain about.")
    return arr


def probabilities_from_logits(logits: Any) -> np.ndarray:
    """Convert raw classifier-head outputs to an ``(N, C)`` probability matrix.

    The head shape decides the link function, and getting this wrong is
    silent:

    * ``(N,)`` or ``(N, 1)`` — a **single-logit binary head**. Sigmoid,
      then expanded to ``[1 − p, p]`` so every measure sees two classes.
      Pushing this through a softmax instead would return a column of
      1.0 — every crop maximally certain, the whole ordering gone.
    * ``(N, C)`` with C ≥ 2 — a **C-logit head**. Row-wise softmax.
      Sigmoiding these instead reads each logit in isolation and can
      invert the order: ``[5, 5]`` is a perfect 50/50 tie under softmax
      but looks like a confident 0.993 under a sigmoid of column 1.

    Both branches are exactly what :func:`spacr.deep_spacr.apply_model_to_tar`
    and :func:`spacr.deep_spacr.evaluate_model_performance` do at
    inference time.

    :param logits: array-like or torch tensor of raw head outputs.
    :returns: ``(N, C)`` float array whose rows sum to 1 (C ≥ 2).
    :raises ValueError: for a scalar or a 3-D input.

    .. note::
       The output is a probability *vector*, not a calibrated
       probability. See :data:`CALIBRATION_NOTE`.
    """
    arr = _as_matrix(logits)
    if arr.shape[1] == 1:
        p = _sigmoid(arr[:, 0])
        return np.column_stack([1.0 - p, p])
    return _softmax(arr)


def _coerce_probabilities(scores: Any) -> Tuple[np.ndarray, List[str]]:
    """Normalise stored probabilities and report what had to be fixed.

    :returns: ``(probs, notes)``. Rows that cannot be read as a
        distribution come back as all-NaN, which every measure turns
        into a NaN score and :func:`build_queue` drops with a count.
    """
    arr = _as_matrix(scores)
    notes: List[str] = []
    n, c = arr.shape

    if n == 0:
        return np.zeros((0, max(c, 2)), dtype=float), notes

    if c == 1:
        p = arr[:, 0]
        out_of_range = np.isfinite(p) & ((p < 0.0) | (p > 1.0))
        if out_of_range.any():
            lo, hi = float(np.nanmin(p)), float(np.nanmax(p))
            notes.append(
                f"{int(out_of_range.sum())} of {n} values in the prediction "
                f"column fall outside [0, 1] (range {lo:.4g} … {hi:.4g}): "
                f"that column holds logits, not probabilities. Those rows "
                f"are dropped — score them with probabilities_from_logits() "
                f"instead, or re-run the merge step.")
            p = np.where(out_of_range, np.nan, p)
        probs = np.column_stack([1.0 - p, p])
        return probs, notes

    probs = arr.astype(float, copy=True)
    negative = np.isfinite(probs) & (probs < 0.0)
    if negative.any():
        rows = np.unique(np.nonzero(negative)[0])
        notes.append(
            f"{len(rows)} of {n} rows contain negative values, which no "
            f"probability vector can: those columns are logits or a broken "
            f"merge. Those rows are dropped.")
        probs[rows, :] = np.nan

    totals = probs.sum(axis=1)
    valid = np.isfinite(totals)
    zero = valid & np.isclose(totals, 0.0)
    if zero.any():
        notes.append(
            f"{int(zero.sum())} of {n} rows sum to 0 and carry no "
            f"distribution at all; they are dropped.")
        probs[zero, :] = np.nan
        totals = probs.sum(axis=1)
        valid = np.isfinite(totals)

    off = valid & ~np.isclose(totals, 1.0, atol=1e-6)
    if off.any():
        worst = float(np.nanmax(np.abs(totals[off] - 1.0)))
        notes.append(
            f"{int(off.sum())} of {n} probability rows did not sum to 1 "
            f"(largest deviation {worst:.4g}); they were renormalised. "
            f"Uncertainty is only meaningful on a normalised distribution, "
            f"so this is a fix, not a warning to ignore — check the column "
            f"really holds per-class probabilities.")
    with np.errstate(invalid="ignore", divide="ignore"):
        probs = np.where(valid[:, None], probs / totals[:, None], np.nan)
    return probs, notes


def as_probabilities(scores: Any) -> np.ndarray:
    """Read stored scores as an ``(N, C)`` probability matrix.

    Unlike :func:`probabilities_from_logits` this assumes the values are
    *already* probabilities — which is what the ``pred`` column of
    ``png_list`` holds, because
    :func:`spacr.deep_spacr.apply_model_to_tar` applied the sigmoid or
    softmax before writing it.

    * a single column is read as the positive-class probability of a
      binary problem and expanded to ``[1 − p, p]``;
    * ``(N, C)`` rows are renormalised if they do not sum to 1;
    * rows that cannot be a distribution (negatives, all-zero, values
      outside [0, 1] in the single-column case) become all-NaN, so they
      score NaN and get excluded rather than silently mis-ranked.

    :param scores: array-like of stored probabilities.
    :returns: ``(N, C)`` float array.
    """
    return _coerce_probabilities(scores)[0]


# ---------------------------------------------------------------------------
# Uncertainty measures
#
# Every measure is oriented the SAME way: larger means less certain,
# minimum at a one-hot row, maximum at the uniform row. That is what lets
# rank_by_uncertainty treat them interchangeably, and it is why margin()
# returns 1 − (p₁ − p₂) rather than the margin itself.
# ---------------------------------------------------------------------------

def least_confidence(probs: Any, normalize: bool = False) -> np.ndarray:
    """``1 − max_c p_c`` — how much probability mass is *not* on the winner.

    Minimum 0 at a one-hot row; maximum ``1 − 1/C`` at the uniform row.
    It looks only at the top class, so on 3+ classes it cannot tell
    ``[0.5, 0.5, 0.0]`` from ``[0.5, 0.25, 0.25]``; :func:`entropy` can.

    :param probs: ``(N,)`` positive-class probabilities or ``(N, C)``
        rows. Coerced with :func:`as_probabilities`.
    :param normalize: rescale by ``C / (C − 1)`` so the maximum is 1.
        A monotone rescaling — it never changes the ranking, only how
        the number reads.
    :returns: ``(N,)`` uncertainty scores; NaN for unusable rows.
    """
    p, _ = _coerce_probabilities(probs)
    if p.shape[0] == 0:
        return np.zeros(0, dtype=float)
    bad = np.all(~np.isfinite(p), axis=1)
    score = 1.0 - np.max(np.where(np.isfinite(p), p, -np.inf), axis=1)
    score = np.where(bad, np.nan, score)
    if normalize and p.shape[1] > 1:
        score = score * (p.shape[1] / (p.shape[1] - 1.0))
    return score


def margin(probs: Any) -> np.ndarray:
    """``1 − (p₁ − p₂)`` — closeness of the top two classes, as *uncertainty*.

    This returns an uncertainty score, **not** the margin: a small margin
    (the two leading classes neck and neck) is a large return value, so
    it is oriented like every other measure here. Minimum 0 at a one-hot
    row, maximum 1 at any row whose top two classes tie.

    On **two classes this is a linear transform of**
    :func:`least_confidence` — ``1 − (p₁ − p₂) = 2·(1 − max p)`` — so the
    two produce the identical order and the identical ties. They are one
    choice, not two, until C ≥ 3, where margin ignores everything below
    the runner-up and least-confidence ignores everything below the
    winner.

    :param probs: ``(N,)`` or ``(N, C)``; coerced with :func:`as_probabilities`.
    :returns: ``(N,)`` uncertainty scores in [0, 1]; NaN for unusable rows.
    """
    p, _ = _coerce_probabilities(probs)
    if p.shape[0] == 0:
        return np.zeros(0, dtype=float)
    bad = np.all(~np.isfinite(p), axis=1)
    filled = np.sort(np.where(np.isfinite(p), p, -np.inf), axis=1)
    gap = np.zeros(p.shape[0], dtype=float)
    np.subtract(
        filled[:, -1],
        filled[:, -2],
        out=gap,
        where=~bad,
    )
    return np.where(bad, np.nan, 1.0 - gap)


def entropy(probs: Any, base: Optional[float] = None,
            normalize: bool = False) -> np.ndarray:
    """Shannon entropy ``−Σ p log p`` of each row.

    The only measure here that uses the whole distribution. Minimum 0 at
    a one-hot row; maximum ``log C`` at the uniform row (``log 2 ≈
    0.6931`` for two classes in nats). ``0 · log 0`` is taken as 0.

    :param probs: ``(N,)`` or ``(N, C)``; coerced with :func:`as_probabilities`.
    :param base: logarithm base. ``None`` (default) means natural log,
        so the units are nats; pass ``2`` for bits.
    :param normalize: divide by ``log C`` so the maximum is 1. Monotone,
        so the ranking is unchanged.
    :returns: ``(N,)`` uncertainty scores; NaN for unusable rows.
    """
    p, _ = _coerce_probabilities(probs)
    if p.shape[0] == 0:
        return np.zeros(0, dtype=float)
    bad = np.all(~np.isfinite(p), axis=1)
    safe = np.where(np.isfinite(p) & (p > 0.0), p, 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(np.isfinite(p) & (p > 0.0), -safe * np.log(safe), 0.0)
    score = terms.sum(axis=1)
    n_classes = p.shape[1]
    if base is not None:
        if base <= 0 or base == 1:
            raise ValueError(f"Log base must be positive and != 1, got {base!r}.")
        score = score / np.log(base)
    if normalize and n_classes > 1:
        denom = np.log(n_classes)
        if base is not None:
            denom = denom / np.log(base)
        score = score / denom
    return np.where(bad, np.nan, score)


def disagreement(prob_sets: Any, method: str = "variance") -> np.ndarray:
    """How much several score sets disagree about each crop.

    The measures above see one model's opinion and call a 50/50 output
    "uncertain" whether the model is genuinely torn or merely
    mis-calibrated. An ensemble — several checkpoints, several folds, or
    several MC-dropout passes of one model — separates those: crops the
    members *disagree* about are where the model class itself is
    undecided (epistemic uncertainty), which is what a new label
    actually fixes.

    :param prob_sets: sequence of M score arrays, each ``(N,)`` or
        ``(N, C)`` over the SAME N crops in the same order (a 3-D
        ``(M, N, C)`` array works too). Each member is coerced with
        :func:`as_probabilities`.
    :param method:

        * ``'variance'`` (default) — mean across classes of the
          across-member variance (population variance, ddof=0). 0 when
          the members agree exactly.
        * ``'bald'`` — mutual information ``H(mean p) − mean H(p)``
          (Houlsby et al., 2011). 0 when the members agree, regardless
          of how uncertain they jointly are — so unlike ``entropy`` it
          does not fire on crops that are simply ambiguous.

    :returns: ``(N,)`` scores; NaN where any member is unusable for that
        crop.
    :raises ValueError: for fewer than one set, or sets of different
        shapes — a length mismatch means the members are not aligned to
        the same crops, and averaging them would be meaningless.

    A single set returns all zeros: one opinion cannot disagree with
    itself. That is a real answer, not a failure, but it means the queue
    would be in row order, so check the count before using it.
    """
    if isinstance(prob_sets, np.ndarray) and prob_sets.ndim == 3:
        members = [prob_sets[i] for i in range(prob_sets.shape[0])]
    elif isinstance(prob_sets, (list, tuple)):
        members = list(prob_sets)
    else:
        arr = _to_numpy(prob_sets)
        if arr.ndim == 3:
            members = [arr[i] for i in range(arr.shape[0])]
        else:
            members = [arr]
    if not members:
        raise ValueError(
            "disagreement() needs at least one set of scores; got none.")

    coerced = [_coerce_probabilities(m)[0] for m in members]
    shapes = {m.shape for m in coerced}
    if len(shapes) != 1:
        raise ValueError(
            f"Every member must score the same crops in the same order; got "
            f"shapes {sorted(shapes)}. Differing row counts mean the members "
            f"are not aligned and the comparison would be meaningless.")

    stack = np.stack(coerced, axis=0)          # (M, N, C)
    bad = ~np.all(np.isfinite(stack), axis=(0, 2))
    if stack.shape[0] == 1:
        return np.where(bad, np.nan, np.zeros(stack.shape[1]))

    if method == "variance":
        score = np.var(stack, axis=0).mean(axis=1)
    elif method == "bald":
        mean = stack.mean(axis=0)
        h_mean = entropy(mean)
        h_each = np.stack([entropy(m) for m in coerced], axis=0)
        score = h_mean - np.nanmean(h_each, axis=0)
        score = np.maximum(score, 0.0)         # MI is non-negative
    else:
        raise ValueError(
            f"Unknown disagreement method {method!r}; use 'variance' or 'bald'.")
    return np.where(bad, np.nan, score)


#: The single-model measures, by name. All are ``f(probs) -> (N,)`` with
#: larger meaning less certain, so they are interchangeable everywhere a
#: ``measure=`` argument is taken.
#:
#: :func:`disagreement` is deliberately NOT in here: it takes a *list* of
#: score sets, not one, so it cannot be swapped in behind the same
#: argument without silently mis-reading its input.
UNCERTAINTY_MEASURES: Dict[str, Callable[..., np.ndarray]] = {
    "least_confidence": least_confidence,
    "margin": margin,
    "entropy": entropy,
}


def resolve_measure(measure: Any) -> Tuple[str, Callable[..., np.ndarray]]:
    """Turn a measure name (or callable) into ``(name, function)``.

    :param measure: a key of :data:`UNCERTAINTY_MEASURES`, or any
        callable ``f(probs) -> (N,)``.
    :raises ValueError: for an unknown name, listing the valid ones.
    """
    if callable(measure):
        return getattr(measure, "__name__", "custom"), measure
    name = str(measure)
    if name not in UNCERTAINTY_MEASURES:
        raise ValueError(
            f"Unknown uncertainty measure {measure!r}. Available: "
            f"{', '.join(sorted(UNCERTAINTY_MEASURES))}. (On two classes "
            f"'margin' and 'least_confidence' give the same order.)")
    return name, UNCERTAINTY_MEASURES[name]


def uncertainty_scores(probs: Any, measure: Any = DEFAULT_MEASURE) -> np.ndarray:
    """Score every row with ``measure``.

    :param probs: ``(N,)`` or ``(N, C)`` probabilities.
    :param measure: name from :data:`UNCERTAINTY_MEASURES` or a callable.
    :returns: ``(N,)`` uncertainty scores, larger = less certain.
    """
    _, fn = resolve_measure(measure)
    return np.asarray(fn(probs), dtype=float).reshape(-1)


def rank_by_uncertainty(probs: Any, measure: Any = DEFAULT_MEASURE,
                        limit: Optional[int] = None,
                        seed: Optional[int] = None,
                        scores: Optional[Any] = None) -> np.ndarray:
    """Row indices ordered most-uncertain-first.

    Ordering is total and reproducible:

    * primary key — the uncertainty score, descending;
    * ties — row index ascending by default, or a permutation seeded
      with ``seed`` when one is given. Both are deterministic: the same
      inputs and the same seed always give the same order. Ties are the
      normal case on a screen where thousands of crops score exactly
      0.5, so an unseeded shuffle there would reshuffle the annotator's
      queue on every refresh.
    * NaN scores sort last, always, and never in front of a real one.
      They are ranked rather than dropped so the returned indices stay a
      permutation of ``range(N)``; :func:`build_queue` drops them and
      says how many.

    :param probs: ``(N,)`` or ``(N, C)`` probabilities.
    :param measure: name from :data:`UNCERTAINTY_MEASURES` or a callable.
    :param limit: keep only the first ``limit`` indices.
    :param seed: seed for tie-breaking; ``None`` breaks ties on index.
    :param scores: pre-computed scores to rank instead of recomputing
        from ``probs`` (used by :func:`build_queue`, and by anything
        ranking a :func:`disagreement` score).
    :returns: ``(N,)`` (or ``(limit,)``) int array of row indices.
    """
    if scores is None:
        values = uncertainty_scores(probs, measure)
    else:
        values = np.asarray(scores, dtype=float).reshape(-1)
    n = values.shape[0]
    if n == 0:
        return np.zeros(0, dtype=int)

    if seed is None:
        tie = np.arange(n)
    else:
        tie = np.random.default_rng(seed).permutation(n)
    finite = np.isfinite(values)
    # Ascending sort on -score = descending sort on score; NaN -> +inf,
    # so unusable rows land at the back instead of wherever NaN happens
    # to compare.
    primary = np.where(finite, -values, np.inf)
    order = np.lexsort((tie, primary))
    if limit is not None:
        order = order[:max(0, int(limit))]
    return order.astype(int)


# ---------------------------------------------------------------------------
# Live-model bridge — the only torch in this file, imported lazily
# ---------------------------------------------------------------------------

def predict_probabilities(model: Callable[[Any], Any], batches: Iterable[Any],
                          device: Any = None,
                          from_logits: bool = True) -> np.ndarray:
    """Run ``model`` over ``batches`` and return an ``(N, C)`` probability matrix.

    A convenience for scoring crops that are not in the database yet.
    The queue does **not** need this — :func:`build_queue` works from the
    ``pred`` column that :func:`spacr.deep_spacr.merge_predictions_into_db`
    already wrote, which is the normal path.

    torch is imported inside this function, and only to get
    ``no_grad``/``device`` handling; if it is not importable the batches
    are simply iterated and the model called directly. Nothing else in
    this module touches torch.

    :param model: any callable mapping a batch to raw head outputs. A
        ``torch.nn.Module`` is put in ``eval()`` mode first if it has one.
    :param batches: iterable of batches. A batch that is a ``(inputs,
        …)`` tuple has its first element passed to the model, matching
        the loaders in :mod:`spacr.deep_spacr`.
    :param device: optional torch device to move inputs/model to.
    :param from_logits: treat outputs as raw logits and apply
        :func:`probabilities_from_logits` (the default — a model head
        emits logits). Set False if the model already outputs
        probabilities, which then go through :func:`as_probabilities`.
    :returns: ``(N, C)`` probability matrix in batch order.
    """
    try:
        import torch
    except Exception:
        torch = None

    if torch is not None and hasattr(model, "eval"):
        model.eval()
        if device is not None and hasattr(model, "to"):
            model = model.to(device)

    outputs: List[np.ndarray] = []

    def _run() -> None:
        for batch in batches:
            inputs = batch[0] if isinstance(batch, (tuple, list)) else batch
            if (torch is not None and device is not None
                    and hasattr(inputs, "to")):
                inputs = inputs.to(device)
            outputs.append(_as_matrix(model(inputs)))

    if torch is not None:
        with torch.no_grad():
            _run()
    else:
        _run()

    if not outputs:
        return np.zeros((0, 2), dtype=float)
    widths = {o.shape[1] for o in outputs}
    if len(widths) != 1:
        raise ValueError(
            f"Batches produced different head widths {sorted(widths)}; the "
            f"model cannot be both a single-logit binary head and a "
            f"multiclass one.")
    raw = np.concatenate(outputs, axis=0)
    return probabilities_from_logits(raw) if from_logits else as_probabilities(raw)


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

def _quote_ident(name: str) -> str:
    """Double-quote a SQL identifier (already schema-validated)."""
    return '"' + str(name).replace('"', '""') + '"'


def _read_only_uri(path: str) -> str:
    """``file:…?mode=ro`` URI — SQLite itself then refuses every write."""
    return "file:" + _urlquote(str(path).replace("\\", "/"), safe="/:") + "?mode=ro"


def _connect(db_path: str) -> sqlite3.Connection:
    """Open ``db_path`` read-only. Building a queue never writes.

    :raises FileNotFoundError: when the file is not there — sqlite's own
        "unable to open database file" says nothing about which file.
    """
    if not db_path or not str(db_path).strip():
        raise ValueError("No database path given.")
    path = os.path.abspath(os.path.expanduser(str(db_path).strip()))
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No such database: {path}")
    con = sqlite3.connect(_read_only_uri(path), uri=True)
    con.execute("PRAGMA query_only = ON")
    return con


def _table_columns(con: sqlite3.Connection, table: str,
                   db_path: str) -> List[str]:
    """Column names of ``table``, in declaration order.

    :raises ValueError: when the database has no such table — sqlite's
        own "no such table" does not say what the file *does* contain.
    """
    names = [r[0] for r in con.execute(
        "SELECT name FROM sqlite_master WHERE type IN ('table','view')"
    ).fetchall()]
    if table not in names:
        raise ValueError(
            f"{os.path.basename(str(db_path))} has no {table!r} table "
            f"(found: {', '.join(names) or 'nothing'}). Crops and their "
            f"predictions live in {PNG_TABLE!r}; run Measure with "
            f"save_png=True first.")
    return [r[1] for r in con.execute(
        f"PRAGMA table_info({_quote_ident(table)})").fetchall()]


def _resolve_pred_columns(pred_column: Any, columns: Sequence[str],
                          table: str, db_path: str) -> List[str]:
    """Work out which column(s) hold the model's scores.

    :raises ValueError: when none can be found, naming what to run.
    """
    available = list(columns)
    if pred_column is not None:
        wanted = ([pred_column] if isinstance(pred_column, str)
                  else list(pred_column))
        missing = [c for c in wanted if c not in available]
        if missing:
            raise ValueError(
                f"{table!r} has no column(s) {', '.join(map(repr, missing))}. "
                f"Available: {', '.join(available)}.")
        if not wanted:
            raise ValueError("pred_column was an empty list.")
        return [str(c) for c in wanted]

    # Multi-class columns first. ``al_prob_`` comes FIRST on purpose: it is
    # written by :func:`retrain_round`, i.e. by a model trained on the labels
    # made in this very annotation session, and it is therefore always
    # fresher than whatever the last full Classify run left in ``pred``.
    # Without this, round 2 of an active-learning loop re-ranks against the
    # round-0 model and serves back the same crops — the loop looks closed
    # and is not.
    for prefix in (ROUND_PRED_PREFIX, "pred_", "prob_", "score_"):
        numbered = sorted(
            (c for c in available
             if c.startswith(prefix) and c[len(prefix):].isdigit()),
            key=lambda c, p=prefix: int(c[len(p):]))
        if len(numbered) >= 2:
            return numbered
    for candidate in PRED_COLUMN_CANDIDATES:
        if candidate in available:
            return [candidate]

    raise ValueError(
        f"{os.path.basename(str(db_path))} [{table}] has no prediction "
        f"column — there is nothing to be uncertain about, so no queue can "
        f"be built. Columns present: {', '.join(available) or 'none'}. Run a "
        f"model over the crops first (spacr.deep_spacr.deep_spacr with "
        f"apply_model_to_dataset=True, which writes 'pred' via "
        f"merge_predictions_into_db), or pass pred_column= if the scores "
        f"live under another name.")


def _group_columns_for(diversity: Any, columns: Sequence[str],
                       group_columns: Optional[Sequence[str]],
                       notes: List[str]) -> Tuple[str, List[str]]:
    """Resolve the diversity strategy to concrete, present columns.

    Falls back to ``'none'`` with a note when the metadata is not there —
    a missing plate map is a reason to stop diversifying, not to crash.
    """
    if group_columns:
        wanted = [str(c) for c in group_columns]
        name = "custom"
    elif diversity in (None, False, "none", "off"):
        return "none", []
    else:
        key = "well" if diversity in (True, "auto") else str(diversity)
        if key not in DIVERSITY_GROUPS:
            raise ValueError(
                f"Unknown diversity strategy {diversity!r}; use one of "
                f"{', '.join(sorted(DIVERSITY_GROUPS))}, 'none', or pass "
                f"group_columns=[…] explicitly.")
        wanted, name = list(DIVERSITY_GROUPS[key]), key

    present = [c for c in wanted if c in columns]
    if not present:
        notes.append(
            f"Diversity was requested over {', '.join(wanted)}, but "
            f"png_list has none of those columns, so the queue is ordered by "
            f"pure uncertainty. Expect it to cluster: the most uncertain "
            f"crops usually come from a handful of wells.")
        return "none", []
    if len(present) < len(wanted):
        notes.append(
            f"Diversity over {', '.join(wanted)} fell back to "
            f"{', '.join(present)} — the rest are not in png_list.")
    return name, present


def _group_key(frame: pd.DataFrame, cols: Sequence[str]) -> np.ndarray:
    """One string key per row, joining ``cols``. Empty-frame safe."""
    if not len(frame):
        return np.zeros(0, dtype=object)
    return frame[list(cols)].astype(str).apply("_".join, axis=1).to_numpy()


def _round_robin(order: np.ndarray, group_keys: np.ndarray) -> np.ndarray:
    """Spread a ranked index array across groups, one per group per pass.

    Groups are visited in order of their most uncertain member, so
    position 1 of the queue is still the single most uncertain crop
    overall; position 2 is the most uncertain crop in the *next* group,
    and so on. Deterministic: no RNG, and the group order is fixed by
    the (already deterministic) input ranking.
    """
    buckets: "OrderedDict[Any, List[int]]" = OrderedDict()
    for idx in order:
        buckets.setdefault(group_keys[idx], []).append(int(idx))
    out: List[int] = []
    while buckets:
        for key in list(buckets):
            out.append(buckets[key].pop(0))
            if not buckets[key]:
                del buckets[key]
    return np.asarray(out, dtype=int)


def _class_balance(labels: Sequence[Any]) -> Dict[Any, int]:
    """Counts per class as a plain, JSON-friendly dict, sorted by class.

    A class read out of a REAL column arrives as ``1.0``; it is reported
    as ``1``, because "class 1.0" is not a thing anyone annotated.
    """
    series = pd.Series(list(labels)).dropna()
    if series.empty:
        return {}
    counts = series.value_counts()
    out: Dict[Any, int] = {}
    for key in sorted(counts.index, key=lambda v: (str(type(v)), v)):
        value = key
        if isinstance(value, float) and float(value).is_integer():
            value = int(value)
        out[value] = int(counts[key])
    return out


def build_queue(db_path: str, annotation_column: str = "annotate",
                pred_column: Any = None, table: str = PNG_TABLE,
                key: str = PNG_KEY, measure: Any = DEFAULT_MEASURE,
                limit: Optional[int] = None, diversity: Any = "well",
                group_columns: Optional[Sequence[str]] = None,
                seed: Optional[int] = None,
                image_type: Optional[str] = None) -> pd.DataFrame:
    """Build the annotation queue: unlabelled crops, most uncertain first.

    Reads ``png_list`` read-only and returns one row per crop still
    waiting for a label, ordered so the crops that would teach the model
    most come first.

    **Already-annotated crops are excluded.** ``NULL`` in
    ``annotation_column`` means "not looked at"; anything else — *including
    0* — means a human committed to a class. If the column does not exist
    at all, nothing has been annotated yet and every crop is queued (with
    a note saying so). This is the same abstention convention as
    :mod:`spacr.agreement`.

    **The queue is diversified by default, and that costs something.**
    With ``diversity='well'`` the ranked crops are dealt round-robin
    across wells: position 1 is still the single most uncertain crop, but
    position 2 is the most uncertain crop in a *different* well, which may
    be materially less uncertain than the runner-up overall. You give up
    some per-item uncertainty to stop the annotator labelling the same
    ambiguity a hundred times — the failure mode of pure uncertainty
    sampling, where the top 100 crops routinely come from two wells. If
    ``limit`` is smaller than the number of wells, the queue will contain
    roughly one crop from each of ``limit`` wells and none from the rest.
    Pass ``diversity='none'`` for the pure order, ``'field'``/``'plate'``
    for other strata, or ``group_columns=[…]`` for your own — including a
    cluster id you computed from features, which is the more thorough
    diversification this trades away for not needing a feature matrix.

    :param db_path: path to ``measurements.db``.
    :param annotation_column: column the Annotate app writes into
        (default ``'annotate'``).
    :param pred_column: name, or list of names, of the model-score
        column(s). ``None`` (default) auto-detects: ``pred_0, pred_1, …``
        style columns for multiclass, else ``pred``. A single column is
        read as the positive-class probability of a binary problem.
    :param table: table holding the crops (default ``png_list``).
    :param key: row key (default ``png_path``).
    :param measure: ``'entropy'`` (default), ``'least_confidence'``,
        ``'margin'``, or a callable. On two classes the last two give the
        same order.
    :param limit: keep at most this many crops.
    :param diversity: ``'well'`` (default), ``'plate'``, ``'row'``,
        ``'column'``, ``'field'``, or ``'none'``.
    :param group_columns: explicit columns to stratify over, overriding
        ``diversity``.
    :param seed: tie-breaking seed; ``None`` breaks ties on row order.
        Either way the result is reproducible.
    :param image_type: substring filter on ``png_path`` — matches the
        Annotate screen's own ``image_type`` filter (e.g. ``'cell'``).
    :returns: DataFrame with ``rank`` (1-based), the key, ``uncertainty``,
        ``predicted_class``, the probability columns and whatever crop
        metadata ``png_list`` has. Empty (with the same columns) when
        there is nothing to annotate. Diagnostics live in
        ``queue.attrs['spacr_active_learning']``; render them with
        :func:`format_queue_summary`.
    :raises ValueError: when the table or the prediction column is
        missing — both mean there is no queue to build, and guessing
        would produce a plausible-looking wrong order.
    :raises FileNotFoundError: when the database is not there.

    .. warning::
       ``uncertainty`` is a ranking score, not a calibrated confidence.
       See :data:`CALIBRATION_NOTE`.
    """
    notes: List[str] = []
    con = _connect(db_path)
    try:
        columns = _table_columns(con, table, db_path)
        if key not in columns:
            raise ValueError(
                f"{table!r} has no {key!r} column — there is no way to say "
                f"which crop a queue entry refers to.")

        pred_cols = _resolve_pred_columns(pred_column, columns, table, db_path)
        measure_name, measure_fn = resolve_measure(measure)

        has_annotation = annotation_column in columns
        if not has_annotation:
            notes.append(
                f"png_list has no {annotation_column!r} column yet, so nothing "
                f"has been annotated: every scored crop is in the queue. The "
                f"Annotate app creates the column the first time it saves.")

        meta_cols = [c for c in _METADATA_COLUMNS if c in columns]
        div_name, div_cols = _group_columns_for(diversity, columns,
                                                group_columns, notes)
        for c in div_cols:
            if c not in meta_cols:
                meta_cols.append(c)
        # How spread the queue is gets reported whatever the strategy — it
        # is most informative when diversity is OFF, because that is where
        # the collapse onto two wells shows up as a number.
        spread_cols = div_cols or [c for c in DIVERSITY_GROUPS["well"]
                                   if c in columns]

        select_cols = [key] + ([annotation_column] if has_annotation else []) \
            + pred_cols + meta_cols
        select_cols = list(dict.fromkeys(select_cols))
        rows = con.execute(
            f"SELECT {', '.join(_quote_ident(c) for c in select_cols)} "
            f"FROM {_quote_ident(table)}").fetchall()
    finally:
        con.close()

    frame = pd.DataFrame(rows, columns=select_cols)
    n_rows = len(frame)

    if image_type:
        keep = frame[key].astype(str).str.contains(str(image_type), regex=False)
        n_filtered = int((~keep).sum())
        frame = frame[keep]
        if n_filtered:
            notes.append(
                f"image_type={image_type!r} excluded {n_filtered} of {n_rows} "
                f"crops before scoring.")

    if has_annotation:
        labelled_mask = frame[annotation_column].notna()
    else:
        labelled_mask = pd.Series(False, index=frame.index)
    labelled_balance = _class_balance(
        frame.loc[labelled_mask, annotation_column].tolist()
        if has_annotation else [])
    n_annotated = int(labelled_mask.sum())
    pool = frame.loc[~labelled_mask].reset_index(drop=True)
    n_unlabelled = len(pool)

    raw = pool[pred_cols].to_numpy(dtype=float) if n_unlabelled else \
        np.zeros((0, len(pred_cols)))
    probs, prob_notes = _coerce_probabilities(raw)
    notes.extend(prob_notes)
    scores = np.asarray(measure_fn(probs), dtype=float).reshape(-1) \
        if n_unlabelled else np.zeros(0)

    usable = np.isfinite(scores)
    n_dropped = int((~usable).sum())
    if n_dropped:
        notes.append(
            f"{n_dropped} of {n_unlabelled} unlabelled crops have no usable "
            f"score (NULL or NaN in {', '.join(pred_cols)}) and were left out "
            f"of the queue. A crop the model never scored cannot be ranked by "
            f"how unsure the model is about it — score them, or annotate them "
            f"in the normal page order.")

    pool = pool.loc[usable].reset_index(drop=True)
    probs = probs[usable]
    scores = scores[usable]
    n_scored = len(pool)

    predicted = (np.argmax(probs, axis=1).astype(int) if n_scored
                 else np.zeros(0, dtype=int))
    pool_balance = _class_balance(predicted.tolist())

    if n_scored > 1 and np.ptp(scores) == 0.0:
        notes.append(
            f"Every unlabelled crop scored exactly {float(scores[0]):.6g}, so "
            f"the {measure_name} ordering carries no information from the "
            f"model — the queue below is just row order. Check that the "
            f"prediction column was written by a model that actually saw "
            f"these crops.")
    if probs.shape[1] == 2 and n_scored:
        distinct = np.unique(np.round(probs[:, 1], 12))
        if distinct.size == 1:
            notes.append(
                "The prediction column holds a single distinct value: the "
                "model put every crop in the same place, which is a "
                "one-class output, not a ranking.")

    order = rank_by_uncertainty(None, measure=measure_fn, seed=seed,
                                scores=scores)
    pool_keys = _group_key(pool, div_cols) if div_cols else np.zeros(0)
    if div_cols and n_scored:
        order = _round_robin(order, pool_keys)
    if limit is not None:
        order = order[:max(0, int(limit))]

    out = pool.iloc[order].reset_index(drop=True)
    out.insert(0, "rank", np.arange(1, len(out) + 1))
    out["uncertainty"] = scores[order]
    out["predicted_class"] = predicted[order]
    if has_annotation and annotation_column in out.columns:
        out = out.drop(columns=[annotation_column])

    queue_balance = _class_balance(out["predicted_class"].tolist())
    spread_keys = pool_keys if div_cols else (
        _group_key(pool, spread_cols) if spread_cols else np.zeros(0))
    n_groups_pool = int(pd.Series(spread_keys).nunique()) if spread_cols else 0
    n_groups_queue = (int(pd.Series(spread_keys[order]).nunique())
                      if spread_cols else 0)

    if not len(out):
        if n_rows and n_annotated == n_rows:
            notes.append(
                f"The queue is empty: all {n_rows} crops in {table} are "
                f"already annotated in {annotation_column!r}. There is "
                f"nothing left to label — start a new annotation column, or "
                f"retrain on what you have.")
        elif not n_rows:
            notes.append(
                f"{table} is empty — no crops have been written to the "
                f"database. Run Measure with save_png=True first.")
        elif not n_scored:
            notes.append(
                "The queue is empty: no unlabelled crop has a usable score.")

    out.attrs["spacr_active_learning"] = {
        "db_path": str(db_path),
        "table": table,
        "key": key,
        "annotation_column": annotation_column,
        "annotation_column_present": bool(has_annotation),
        "pred_columns": list(pred_cols),
        "n_classes": int(probs.shape[1]) if probs.size else 0,
        "measure": measure_name,
        "diversity": div_name,
        "diversity_columns": list(div_cols),
        "spread_columns": list(spread_cols),
        "seed": seed,
        "limit": limit,
        "image_type": image_type,
        "n_rows": int(n_rows),
        "n_annotated": int(n_annotated),
        "n_unlabelled": int(n_unlabelled),
        "n_scored": int(n_scored),
        "n_unscorable": int(n_dropped),
        "n_queued": int(len(out)),
        "n_groups_pool": n_groups_pool,
        "n_groups_queued": n_groups_queue,
        "queue_class_balance": queue_balance,
        "pool_class_balance": pool_balance,
        "labelled_class_balance": labelled_balance,
        "notes": notes,
        "calibration": CALIBRATION_NOTE,
    }
    return out


def queue_rows(queue: pd.DataFrame,
               key: str = PNG_KEY) -> List[Tuple[str, Optional[int]]]:
    """The queue as ``[(png_path, None), …]``.

    Exactly the shape :func:`spacr.qt.annotate_engine.fetch_page` and
    :func:`spacr.qt.annotate_engine.fetch_filtered_paths` return, so the
    Annotate screen can page through a queue with no other change. The
    annotation is always ``None`` — every crop in a queue is unlabelled
    by construction.

    :param queue: frame from :func:`build_queue`.
    :param key: the path column (default ``png_path``).
    :returns: list of ``(path, None)`` tuples in queue order.
    """
    if key not in queue.columns:
        raise ValueError(
            f"Queue has no {key!r} column; got {', '.join(queue.columns)}.")
    return [(str(p), None) for p in queue[key].tolist()]


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _share(count: int, total: int) -> str:
    """``count`` as a percent of ``total``, or ``'—'`` when there is no total."""
    if not total:
        return "—"
    return f"{100.0 * count / total:5.1f}%"


def format_queue_summary(queue: pd.DataFrame) -> str:
    """Render a queue's shape, class balance and caveats as plain text.

    Reports the numbers that decide whether the queue is worth working
    through: how much of the screen is already labelled, how many crops
    could not be scored, the range of the scores, how many wells the
    queue actually spreads over (the diversity check), and the class
    balance of the queue next to the balance of the pool it came from —
    because uncertainty sampling on an imbalanced screen pulls hard
    toward the majority class's boundary and the annotator should see
    that rather than discover it.

    :param queue: frame from :func:`build_queue`. Works on a slice or a
        copy too, falling back to what can be recomputed from the rows
        when ``attrs`` did not survive.
    :returns: multi-line text, no trailing newline.
    """
    meta: Dict[str, Any] = dict(queue.attrs.get("spacr_active_learning", {}))
    n_queued = int(meta.get("n_queued", len(queue)))
    lines: List[str] = []

    db = os.path.basename(str(meta.get("db_path", ""))) or "(unknown database)"
    lines.append(f"Active-learning queue — {db} [{meta.get('table', PNG_TABLE)}]")
    lines.append(
        f"Measure: {meta.get('measure', '?')}   "
        f"diversity: {meta.get('diversity', '?')}"
        + (f" over {', '.join(meta['diversity_columns'])}"
           if meta.get("diversity_columns") else "")
        + f"   seed: {meta.get('seed')}   "
        f"scores from: {', '.join(meta.get('pred_columns', [])) or '?'}")

    n_rows = int(meta.get("n_rows", 0))
    lines.append(
        f"Crops: {n_rows} total · {meta.get('n_annotated', 0)} already "
        f"annotated ({meta.get('annotation_column', '?')}) · "
        f"{meta.get('n_unlabelled', 0)} unlabelled · "
        f"{meta.get('n_unscorable', 0)} unscorable · {n_queued} queued")

    if not n_queued:
        lines.append("")
        lines.append("Queue is EMPTY — nothing to annotate.")
        for note in meta.get("notes", []):
            lines.append(f"  ! {note}")
        lines.append("")
        lines.append(CALIBRATION_NOTE)
        return "\n".join(lines)

    if "uncertainty" in queue.columns and len(queue):
        vals = pd.to_numeric(queue["uncertainty"], errors="coerce").dropna()
        if len(vals):
            lines.append(
                f"Uncertainty score: max {vals.max():.4f} · median "
                f"{vals.median():.4f} · min {vals.min():.4f} "
                f"(higher = the model is less settled)")

    groups = meta.get("spread_columns") or meta.get("diversity_columns") or []
    if groups and all(c in queue.columns for c in groups):
        counts = pd.Series(_group_key(queue, groups)).value_counts()
        lines.append(
            f"Spread: {counts.size} distinct {'/'.join(groups)} groups in the "
            f"queue of {meta.get('n_groups_pool', counts.size)} in the "
            f"unlabelled pool; the busiest holds {int(counts.iloc[0])} crops "
            f"({_share(int(counts.iloc[0]), n_queued).strip()} of the queue).")
        if not meta.get("diversity_columns") and counts.size > 1:
            lines.append(
                "  Diversity is OFF — this queue is pure uncertainty, so "
                "expect it to concentrate on wherever the model is confused.")

    queue_balance = meta.get("queue_class_balance") or _class_balance(
        queue["predicted_class"].tolist() if "predicted_class" in queue else [])
    pool_balance = meta.get("pool_class_balance") or {}
    if queue_balance:
        pool_total = sum(pool_balance.values())
        lines.append("")
        lines.append("Predicted-class balance (argmax of the stored scores):")
        lines.append(f"  {'class':<8}{'in queue':>10}{'share':>9}"
                     f"{'in pool':>10}{'share':>9}")
        for cls in sorted(set(queue_balance) | set(pool_balance),
                          key=lambda v: str(v)):
            q = int(queue_balance.get(cls, 0))
            p = int(pool_balance.get(cls, 0))
            lines.append(
                f"  {str(cls):<8}{q:>10}{_share(q, n_queued):>9}"
                f"{p:>10}{_share(p, pool_total):>9}")
        lines.append(
            "  A queue skewed against the pool is uncertainty sampling doing "
            "its job on an imbalanced screen — the boundary is where the "
            "majority class lives. Cap it or rebalance if that is not what "
            "you want.")

    labelled = meta.get("labelled_class_balance") or {}
    if labelled:
        lines.append("")
        lines.append(
            "Already annotated: "
            + " · ".join(f"class {k}: {v}" for k, v in labelled.items()))

    if meta.get("notes"):
        lines.append("")
        for note in meta["notes"]:
            lines.append(f"! {note}")

    lines.append("")
    lines.append(CALIBRATION_NOTE)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Annotation coverage — how many labels, of which class, from where
# ---------------------------------------------------------------------------

def _well_key(frame: pd.DataFrame, columns: Sequence[str]) -> pd.Series:
    """Readable ``plate/row/column`` key per row, or ``'(unknown)'``."""
    present = [c for c in columns if c in frame.columns]
    if not present or not len(frame):
        return pd.Series(["(unknown)"] * len(frame), index=frame.index,
                         dtype=object)
    return frame[present].astype(str).apply("/".join, axis=1)


def _concentration(counts: "pd.Series") -> Dict[str, Any]:
    """How lopsided a count distribution is, in three numbers.

    :param counts: labels per group, any order.
    :returns: ``n``, ``n_groups``, ``top``, ``top_n``, ``top_share`` and
        ``hhi`` — the Herfindahl index, the sum of squared shares. ``hhi``
        is 1.0 when every label came from one group and ``1/k`` when they
        are spread evenly over ``k``; it is the single number that says
        "these 200 labels are really 1.06 wells' worth" without needing the
        whole table.
    """
    total = int(counts.sum())
    if not total:
        return {"n": 0, "n_groups": 0, "top": None, "top_n": 0,
                "top_share": 0.0, "hhi": 0.0, "effective_groups": 0.0}
    ordered = counts.sort_values(ascending=False)
    shares = (ordered / total).to_numpy(dtype=float)
    hhi = float(np.sum(shares ** 2))
    return {
        "n": total,
        "n_groups": int((ordered > 0).sum()),
        "top": str(ordered.index[0]),
        "top_n": int(ordered.iloc[0]),
        "top_share": float(ordered.iloc[0]) / total,
        "hhi": hhi,
        # 1/HHI: how many equally-sized groups the labels are "worth".
        "effective_groups": (1.0 / hhi) if hhi else 0.0,
    }


def annotation_coverage(db_path: str, annotation_column: str = "annotate",
                        table: str = PNG_TABLE, key: str = PNG_KEY,
                        image_type: Optional[str] = None) -> pd.DataFrame:
    """Where the annotations actually came from: per class, per well, per plate.

    "I labelled 200 cells" is not a description of a training set. *Which*
    200 decides what the classifier learns, and the failure this function
    exists to make visible is the one that never announces itself: 190 of the
    200 came from a single well, so the model learned that well's staining,
    focus and confluency rather than the biology, and every held-out number
    drawn from a random split of those objects is optimistic.

    Reads ``png_list`` read-only, plus :data:`ROUND_TABLE` when it is there,
    so labels can also be attributed to the active-learning round that
    surfaced them.

    :param db_path: path to ``measurements.db``.
    :param annotation_column: the column the Annotate app writes into.
    :param table: crop table (default ``png_list``).
    :param key: row key (default ``png_path``).
    :param image_type: substring filter on the key, matching the Annotate
        screen's own filter.
    :returns: one row per ``(plateID, rowID, columnID, class)`` that has at
        least one annotation, with ``n`` and ``share`` — plus the whole
        breakdown in ``attrs['spacr_annotation_coverage']``:
        ``by_class``, ``by_plate``, ``by_well``, ``by_class_plate``,
        ``by_class_well``, ``by_round``, ``by_source``, ``concentration``
        (per class) and ``notes``.
    :raises ValueError: when the table has no such column — an empty result
        would read as "nothing annotated yet", which is a different fact.
    :raises FileNotFoundError: when the database is not there.
    """
    notes: List[str] = []
    con = _connect(db_path)
    try:
        columns = _table_columns(con, table, db_path)
        if key not in columns:
            raise ValueError(
                f"{table!r} has no {key!r} column, so an annotation cannot be "
                f"attributed to a crop.")
        if annotation_column not in columns:
            raise ValueError(
                f"{table!r} has no {annotation_column!r} column — nothing has "
                f"been annotated into it yet. The Annotate app creates the "
                f"column the first time it saves. Columns present: "
                f"{', '.join(columns)}.")
        meta = [c for c in ("plateID", "rowID", "columnID", "fieldID", "prc")
                if c in columns]
        select = list(dict.fromkeys([key, annotation_column] + meta))
        rows = con.execute(
            f"SELECT {', '.join(_quote_ident(c) for c in select)} "
            f"FROM {_quote_ident(table)}").fetchall()
        rounds = _read_rounds(con, annotation_column)
    finally:
        con.close()

    frame = pd.DataFrame(rows, columns=select)
    n_rows = len(frame)
    if image_type:
        frame = frame[frame[key].astype(str).str.contains(
            str(image_type), regex=False)]

    well_cols = [c for c in ("plateID", "rowID", "columnID") if c in frame.columns]
    frame = frame.assign(_well=_well_key(frame, well_cols))
    if "plateID" in frame.columns:
        frame = frame.assign(_plate=frame["plateID"].astype(str))
    else:
        frame = frame.assign(_plate="(unknown)")
        notes.append(
            f"{table} has no plateID column, so nothing here can be attributed "
            f"to a plate; every label is reported under '(unknown)'.")

    wells_total = int(frame["_well"].nunique()) if len(frame) else 0
    labelled = frame[frame[annotation_column].notna()].copy()
    n_annotated = len(labelled)
    if n_annotated:
        labelled["_class"] = labelled[annotation_column].map(_class_name)

    if not rounds.empty and n_annotated:
        labelled = labelled.merge(
            rounds, how="left", left_on=key, right_on="png_path")
        labelled["round"] = labelled["round"].fillna(-1).astype(int)
        labelled["source"] = labelled["source"].fillna("unrecorded")
    else:
        labelled["round"] = -1
        labelled["source"] = "unrecorded"
        if n_annotated:
            notes.append(
                f"No {ROUND_TABLE} rows for {annotation_column!r}, so no label "
                f"can be attributed to an annotation round. Rounds are "
                f"recorded from the Annotate screen; labels written before "
                f"that read as 'unrecorded'.")

    if not n_annotated:
        out = pd.DataFrame(columns=["plateID", "rowID", "columnID", "class",
                                    "n", "share"])
        out.attrs["spacr_annotation_coverage"] = {
            "db_path": str(db_path), "table": table,
            "annotation_column": annotation_column,
            "n_rows": n_rows, "n_annotated": 0, "n_classes": 0,
            "wells_total": wells_total, "wells_annotated": 0,
            "plates_total": int(frame["_plate"].nunique()) if len(frame) else 0,
            "plates_annotated": 0,
            "by_class": {}, "by_plate": {}, "by_well": {},
            "by_class_plate": {}, "by_class_well": {},
            "by_round": {}, "by_source": {}, "concentration": {},
            "notes": notes + [
                f"Nothing is annotated in {annotation_column!r} yet "
                f"({n_rows} crops in {table})."],
        }
        return out

    group_cols = [c for c in ("plateID", "rowID", "columnID") if c in labelled.columns]
    grouped = (labelled.groupby(group_cols + ["_class"], dropna=False)
               .size().reset_index(name="n")
               if group_cols else
               labelled.groupby(["_class"]).size().reset_index(name="n"))
    grouped = grouped.rename(columns={"_class": "class"})
    grouped["share"] = grouped["n"] / float(n_annotated)
    grouped = grouped.sort_values(["class", "n"], ascending=[True, False]
                                  ).reset_index(drop=True)

    by_class = {k: int(v) for k, v in
                labelled["_class"].value_counts().sort_index().items()}
    by_plate = {str(k): int(v) for k, v in
                labelled["_plate"].value_counts().sort_index().items()}
    by_well = {str(k): int(v) for k, v in
               labelled["_well"].value_counts().sort_index().items()}
    by_class_plate = {
        str(cls): {str(p): int(n) for p, n in
                   sub["_plate"].value_counts().sort_index().items()}
        for cls, sub in labelled.groupby("_class")}
    by_class_well = {
        str(cls): {str(w): int(n) for w, n in
                   sub["_well"].value_counts().sort_index().items()}
        for cls, sub in labelled.groupby("_class")}
    concentration = {
        str(cls): _concentration(sub["_well"].value_counts())
        for cls, sub in labelled.groupby("_class")}
    concentration["__all__"] = _concentration(labelled["_well"].value_counts())

    for cls, stats in concentration.items():
        if cls == "__all__" or stats["n"] < 10 or stats["n_groups"] < 1:
            continue
        if stats["top_share"] >= 0.5 and stats["n_groups"] > 1:
            notes.append(
                f"Class {cls}: {stats['top_n']} of {stats['n']} labels "
                f"({stats['top_share']:.0%}) come from one well "
                f"({stats['top']}). A classifier trained on this learns that "
                f"well as much as it learns the class.")
        elif stats["n_groups"] == 1:
            notes.append(
                f"Class {cls}: all {stats['n']} labels come from the single "
                f"well {stats['top']}. There is no way to tell the class "
                f"apart from the well, and a random train/test split of these "
                f"objects will report an accuracy that does not transfer.")

    counts = pd.Series(by_class)
    if len(counts) > 1 and int(counts.min()) * 5 < int(counts.max()):
        notes.append(
            f"Class balance is {':'.join(str(int(v)) for v in counts)} "
            f"({', '.join(map(str, counts.index))}). The smallest class has "
            f"{int(counts.min())} labels; per-class accuracy, not the "
            f"aggregate, is the number to read on a model trained here.")

    out = grouped
    out.attrs["spacr_annotation_coverage"] = {
        "db_path": str(db_path), "table": table,
        "annotation_column": annotation_column,
        "n_rows": n_rows, "n_annotated": int(n_annotated),
        "n_classes": len(by_class),
        "wells_total": wells_total,
        "wells_annotated": int(labelled["_well"].nunique()),
        "plates_total": int(frame["_plate"].nunique()),
        "plates_annotated": int(labelled["_plate"].nunique()),
        "by_class": by_class,
        "by_plate": by_plate,
        "by_well": by_well,
        "by_class_plate": by_class_plate,
        "by_class_well": by_class_well,
        "by_round": {int(k): int(v) for k, v in
                     labelled["round"].value_counts().sort_index().items()},
        "by_source": {str(k): int(v) for k, v in
                      labelled["source"].value_counts().sort_index().items()},
        "concentration": concentration,
        "notes": notes,
    }
    return out


def _class_name(value: Any) -> str:
    """``1.0`` and ``1`` are the same class; report it as ``'1'``."""
    if isinstance(value, float) and float(value).is_integer():
        return str(int(value))
    return str(value)


def _object_label(value: Any) -> str:
    """``'o5'``/``5``/``5.0`` → ``'5'``; anything unusable → ``''``.

    ``png_list`` stores the object id as TEXT ``'o5'`` while every object
    table stores an integer, and the sentinels ``'omulti'`` / ``'onone'`` /
    ``'error'`` are real values in that column. Both facts are load-bearing
    here: a naive ``int(value)`` raises on the sentinels, and a naive string
    compare never matches an integer key.
    """
    if value is None:
        return ""
    text = str(value).strip()
    if not text or text.lower() in ("nan", "none", "null"):
        return ""
    if text[:1] in ("o", "O"):
        text = text[1:]
    if not text:
        return ""
    try:
        return str(int(float(text)))
    except (TypeError, ValueError):
        return ""


def crops_for_object_keys(db_path: str, keys: Sequence[str], *,
                          table: str = PNG_TABLE, key: str = PNG_KEY,
                          annotation_column: Optional[str] = None,
                          timelapse: bool = False,
                          image_type: Optional[str] = None
                          ) -> List[Tuple[str, Optional[int]]]:
    """Resolve object keys to crop rows, **in the caller's order**.

    The database half of the object-routing contract in
    :mod:`spacr.qt.linked_selection`: a scatter plot or a confusion-matrix
    cell names objects by :data:`spacr.selection.OBJECT_KEY_COLUMNS` key,
    and the Annotate screen has to turn those into the crops it paginates.
    Kept here rather than in the Qt screen so it is testable without a
    display, and so a second consumer does not have to reimplement the
    ``'o5'``-versus-``5`` trap in :func:`_object_label`.

    Order is preserved because it is the whole point of a routed request:
    "worst errors first" survives the trip only if this does not re-sort into
    table order. Keys with no crop are dropped — a request may name objects a
    crop table does not carry, and that is a shorter result, not an error.

    **A typed key resolves to that object's own crop.** ``png_list`` records
    which object type a crop is by which of its ``<type>_id`` columns holds
    the label (:data:`PNG_ID_COLUMN_TYPES`), so a nucleus 1 and a pathogen 1
    in the same field are two crops. They used to be one: both keyed on
    ``plate_r1_c1_f1_1``, this function kept the first, and which of the two
    you opened depended on the row order of the table. An *untyped* key still
    keeps the first — it names an object without saying which, which is what
    it has always meant — and a typed key against a crop table that cannot say
    what its rows are falls back to the untyped one rather than resolving
    nothing.

    :param db_path: path to ``measurements.db``.
    :param keys: object keys, typed or not. A ``png_path``, a ``prcfo`` or a
        ``file_name`` is also accepted, so a caller working from a crop table
        rather than a measurement table does not need a translation step.
    :param table: crop table.
    :param key: crop key column.
    :param annotation_column: read the existing label too, so an already
        annotated crop renders with its colour rather than blank.
    :param timelapse: the keys carry a timepoint.
    :param image_type: substring filter on the crop key.
    :returns: ``[(png_path, annotation or None), …]`` in the keys' order.
    """
    wanted = [str(k) for k in keys]
    if not wanted:
        return []
    con = _connect(db_path)
    try:
        columns = _table_columns(con, table, db_path)
        if key not in columns:
            raise ValueError(
                f"{table!r} has no {key!r} column, so an object key cannot be "
                f"resolved to a crop.")
        has_annotation = bool(annotation_column
                              and annotation_column in columns)
        select = [key]
        for extra in ("prcfo", "file_name", "plateID", "rowID", "columnID",
                      "fieldID", "timeID", "cell_id", "nucleus_id",
                      "pathogen_id", "cytoplasm_id", "organelle_id"):
            if extra in columns and extra not in select:
                select.append(extra)
        if has_annotation and annotation_column not in select:
            select.append(annotation_column)
        rows = con.execute(
            f"SELECT {', '.join(_quote_ident(c) for c in select)} "
            f"FROM {_quote_ident(table)}").fetchall()
    finally:
        con.close()

    from .selection import untyped_object_key

    index = {c: i for i, c in enumerate(select)}
    id_columns = [c for c in PNG_ID_COLUMN_TYPES if c in index]
    meta_columns = ["plateID", "rowID", "columnID", "fieldID"]
    if timelapse:
        meta_columns.append("timeID")

    by_key: Dict[str, Tuple[str, Optional[int]]] = {}
    for row in rows:
        path = str(row[index[key]])
        if image_type and str(image_type) not in path:
            continue
        annotation = None
        if has_annotation:
            raw = row[index[annotation_column]]
            annotation = None if raw is None else int(raw)
        entry = (path, annotation)

        # WHICH id column holds the label is the crop's object type:
        # `filepaths_to_database` writes exactly one per row, the one for the
        # crop mode it was called with. That is what lets a nucleus crop and a
        # pathogen crop of the same label in the same field be told apart —
        # they used to resolve to one key and the first row in the table won.
        stated = [(column, _object_label(row[index[column]]))
                  for column in id_columns]
        stated = [(column, value) for column, value in stated if value]
        object_type = None
        label = ""
        if len(stated) == 1:
            label = stated[0][1]
            object_type = PNG_ID_COLUMN_TYPES[stated[0][0]]
        elif stated:
            # Two id columns filled is a row that does not say what it is.
            # The old first-wins precedence still gives a label to key on;
            # claiming a type from it would be a guess.
            label = stated[0][1]
        prcfo = (str(row[index["prcfo"]])
                 if "prcfo" in index and row[index["prcfo"]] is not None
                 else "")
        if not label and prcfo:
            label = _object_label(prcfo.rsplit("_", 1)[-1])
        if label and all(c in index for c in meta_columns):
            parts = [str(row[index[c]]) for c in meta_columns]
            # Both spellings, so a caller working from either side resolves.
            # The untyped one is first-wins on purpose: it is an
            # under-specified name, and it named one of these crops before
            # the type existed.
            by_key.setdefault("_".join(parts + [label]), entry)
            if object_type is not None:
                by_key.setdefault(
                    "_".join(parts + [f"{object_type}{label}"]), entry)
        file_name = (str(row[index["file_name"]])
                     if "file_name" in index and
                     row[index["file_name"]] is not None else "")
        for candidate in (path, prcfo, file_name):
            if candidate:
                by_key.setdefault(candidate, entry)

    out: List[Tuple[str, Optional[int]]] = []
    seen = set()
    for wanted_key in wanted:
        entry = by_key.get(wanted_key)
        if entry is None:
            # A typed key against a crop table that cannot say what its rows
            # are. Dropping the type is the honest fallback: the row has not
            # contradicted the key, it has said nothing.
            reduced = untyped_object_key(wanted_key)
            if reduced != wanted_key:
                entry = by_key.get(reduced)
        if entry is None or entry[0] in seen:
            continue
        seen.add(entry[0])
        out.append(entry)
    return out


def format_coverage_summary(coverage: pd.DataFrame) -> str:
    """Render :func:`annotation_coverage` as text, worst concentration first.

    :param coverage: the frame from :func:`annotation_coverage`.
    :returns: multi-line text, no trailing newline.
    """
    meta: Dict[str, Any] = dict(
        coverage.attrs.get("spacr_annotation_coverage", {}))
    lines: List[str] = []
    db = os.path.basename(str(meta.get("db_path", ""))) or "(unknown database)"
    lines.append(f"Annotation coverage — {db} "
                 f"[{meta.get('annotation_column', '?')}]")
    lines.append(
        f"{meta.get('n_annotated', 0)} of {meta.get('n_rows', 0)} crops "
        f"annotated · {meta.get('n_classes', 0)} classes · "
        f"{meta.get('plates_annotated', 0)}/{meta.get('plates_total', 0)} "
        f"plates · {meta.get('wells_annotated', 0)}/"
        f"{meta.get('wells_total', 0)} wells")

    by_class = meta.get("by_class") or {}
    if not by_class:
        lines.append("")
        lines.append("Nothing annotated yet.")
        for note in meta.get("notes", []):
            lines.append(f"  ! {note}")
        return "\n".join(lines)

    total = sum(by_class.values()) or 1
    conc = meta.get("concentration") or {}
    lines.append("")
    lines.append("Per class:")
    lines.append(f"  {'class':<10}{'labels':>8}{'share':>8}{'wells':>8}"
                 f"{'plates':>8}{'busiest well':>22}{'its share':>11}")
    for cls in sorted(by_class, key=str):
        n = int(by_class[cls])
        stats = conc.get(str(cls), {})
        plates = len(meta.get("by_class_plate", {}).get(str(cls), {}) or {})
        lines.append(
            f"  {str(cls):<10}{n:>8}{_share(n, total):>8}"
            f"{stats.get('n_groups', 0):>8}{plates:>8}"
            f"{str(stats.get('top') or '—'):>22}"
            f"{_share(int(stats.get('top_n', 0)), n):>11}")
    overall = conc.get("__all__", {})
    if overall.get("effective_groups"):
        lines.append(
            f"  All {overall['n']} labels are spread over "
            f"{overall['n_groups']} wells, but weighted by size they are "
            f"worth {overall['effective_groups']:.1f} evenly-sampled wells "
            f"(1/HHI).")

    by_plate = meta.get("by_plate") or {}
    if len(by_plate) > 1:
        lines.append("")
        lines.append("Per plate: " + " · ".join(
            f"{p}: {n}" for p, n in sorted(by_plate.items())))

    by_well = meta.get("by_well") or {}
    if by_well:
        busiest = sorted(by_well.items(), key=lambda kv: -kv[1])[:10]
        lines.append("")
        lines.append("Busiest wells (plate/row/column):")
        for well, n in busiest:
            lines.append(f"  {well:<22}{n:>7}{_share(n, total):>9}")

    by_round = meta.get("by_round") or {}
    if by_round:
        lines.append("")
        named = {("before rounds were recorded" if int(k) < 0
                  else f"round {int(k)}"): v for k, v in by_round.items()}
        lines.append("Per round: " + " · ".join(
            f"{k}: {v}" for k, v in named.items()))

    if meta.get("notes"):
        lines.append("")
        for note in meta["notes"]:
            lines.append(f"! {note}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Round bookkeeping — the loop's memory
# ---------------------------------------------------------------------------

def _utc_now() -> str:
    """ISO-8601 UTC, to the second — the stamp every round row carries."""
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _write_connection(db_path: str) -> sqlite3.Connection:
    """A writable connection to ``db_path``, for the round tables only.

    Separate from :func:`_connect`, which is read-only on purpose: building a
    queue must never be able to write, and the two connections being visibly
    different is what keeps that true.
    """
    if not db_path or not str(db_path).strip():
        raise ValueError("No database path given.")
    path = os.path.abspath(os.path.expanduser(str(db_path).strip()))
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No such database: {path}")
    return sqlite3.connect(path, timeout=30)


def ensure_round_tables(db_path: str) -> None:
    """Create :data:`ROUND_TABLE` and :data:`ROUND_LOG_TABLE` if absent.

    Two tables, not one. Per-label provenance and per-round metrics have
    different cardinalities and different lifetimes: a label keeps its round
    forever, a round's held-out accuracy is rewritten if the round is re-fit.
    """
    con = _write_connection(db_path)
    try:
        con.execute(f"""
            CREATE TABLE IF NOT EXISTS {_quote_ident(ROUND_TABLE)} (
                png_path TEXT NOT NULL,
                annotation_column TEXT NOT NULL,
                round INTEGER NOT NULL,
                first_round INTEGER NOT NULL,
                label INTEGER,
                source TEXT NOT NULL DEFAULT 'manual',
                labelled_utc TEXT NOT NULL,
                PRIMARY KEY (png_path, annotation_column)
            )""")
        con.execute(f"""
            CREATE TABLE IF NOT EXISTS {_quote_ident(ROUND_LOG_TABLE)} (
                annotation_column TEXT NOT NULL,
                round INTEGER NOT NULL,
                finished_utc TEXT NOT NULL,
                n_labels INTEGER NOT NULL DEFAULT 0,
                n_new_labels INTEGER NOT NULL DEFAULT 0,
                n_holdout INTEGER NOT NULL DEFAULT 0,
                holdout_accuracy REAL,
                holdout_f1_macro REAL,
                per_class_json TEXT,
                split_rule TEXT,
                model_type TEXT,
                model_path TEXT,
                card_path TEXT,
                measure TEXT,
                diversity TEXT,
                notes_json TEXT,
                PRIMARY KEY (annotation_column, round)
            )""")
        con.commit()
    finally:
        con.close()


def _read_rounds(con: sqlite3.Connection,
                 annotation_column: str) -> pd.DataFrame:
    """Per-label round provenance, or an empty frame when unrecorded."""
    names = {r[0] for r in con.execute(
        "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    if ROUND_TABLE not in names:
        return pd.DataFrame(columns=["png_path", "round", "first_round",
                                     "source", "labelled_utc"])
    rows = con.execute(
        f"SELECT png_path, round, first_round, source, labelled_utc "
        f"FROM {_quote_ident(ROUND_TABLE)} WHERE annotation_column = ?",
        (str(annotation_column),)).fetchall()
    return pd.DataFrame(rows, columns=["png_path", "round", "first_round",
                                       "source", "labelled_utc"])


def record_labels(db_path: str, annotation_column: str,
                  labels: Dict[str, Any], round_index: int,
                  source: str = "manual") -> int:
    """Stamp each label with the round it was made in.

    Called by the Annotate screen every time it flushes a batch. The round a
    label came from is what makes early-round bias auditable: the first
    round's labels are drawn from whatever ordering existed before any model
    had seen this screen, and if 80 % of a class's labels carry round 0 then
    the "active learning" was mostly not active.

    ``first_round`` is preserved across re-labelling while ``round`` follows
    the current value, so both "when was this crop first looked at" and
    "which round set the label it has now" survive a correction.

    :param db_path: path to ``measurements.db``.
    :param annotation_column: the column the labels were written into.
    :param labels: ``{png_path: class or None}``. ``None`` is a *cleared*
        label and is recorded as such rather than dropped — a crop that was
        looked at and deliberately left blank is not the same as one never
        seen.
    :param round_index: the round these labels belong to.
    :param source: how the crop reached the annotator — ``'manual'``,
        ``'queue'``, or a caller's own tag.
    :returns: number of rows written.
    """
    if not labels:
        return 0
    ensure_round_tables(db_path)
    stamp = _utc_now()
    payload = [
        (str(path), str(annotation_column), int(round_index),
         int(round_index),
         (None if value is None else int(value)), str(source), stamp)
        for path, value in labels.items()]
    con = _write_connection(db_path)
    try:
        con.executemany(
            f"INSERT INTO {_quote_ident(ROUND_TABLE)} "
            f"(png_path, annotation_column, round, first_round, label, "
            f"source, labelled_utc) VALUES (?, ?, ?, ?, ?, ?, ?) "
            f"ON CONFLICT(png_path, annotation_column) DO UPDATE SET "
            f"round=excluded.round, label=excluded.label, "
            f"source=excluded.source, labelled_utc=excluded.labelled_utc",
            payload)
        con.commit()
    finally:
        con.close()
    return len(payload)


def label_rounds(db_path: str,
                 annotation_column: str = "annotate") -> pd.DataFrame:
    """Per-label round provenance as a frame (empty when never recorded)."""
    con = _connect(db_path)
    try:
        return _read_rounds(con, annotation_column)
    finally:
        con.close()


def next_round(db_path: str, annotation_column: str = "annotate") -> int:
    """The round number the next batch of labels belongs to.

    Round 0 is "before any model was retrained from inside Annotate" — the
    labels that seeded the loop. The first retrain produces round 1.
    """
    try:
        con = _connect(db_path)
    except (FileNotFoundError, ValueError):
        return 0
    try:
        names = {r[0] for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
        if ROUND_LOG_TABLE not in names:
            return 0
        row = con.execute(
            f"SELECT MAX(round) FROM {_quote_ident(ROUND_LOG_TABLE)} "
            f"WHERE annotation_column = ?",
            (str(annotation_column),)).fetchone()
    finally:
        con.close()
    return int(row[0]) + 1 if row and row[0] is not None else 0


def record_round(db_path: str, annotation_column: str, round_index: int,
                 **fields: Any) -> None:
    """Append (or replace) one row of the learning curve.

    :param db_path: path to ``measurements.db``.
    :param annotation_column: the column this round labelled into.
    :param round_index: the round number.
    :param fields: any of ``n_labels``, ``n_new_labels``, ``n_holdout``,
        ``holdout_accuracy``, ``holdout_f1_macro``, ``per_class`` (dict),
        ``split_rule``, ``model_type``, ``model_path``, ``card_path``,
        ``measure``, ``diversity``, ``notes`` (list).
    """
    import json
    ensure_round_tables(db_path)
    values = {
        "annotation_column": str(annotation_column),
        "round": int(round_index),
        "finished_utc": _utc_now(),
        "n_labels": int(fields.get("n_labels", 0) or 0),
        "n_new_labels": int(fields.get("n_new_labels", 0) or 0),
        "n_holdout": int(fields.get("n_holdout", 0) or 0),
        "holdout_accuracy": _as_float_or_none(fields.get("holdout_accuracy")),
        "holdout_f1_macro": _as_float_or_none(fields.get("holdout_f1_macro")),
        "per_class_json": json.dumps(fields.get("per_class") or {}),
        "split_rule": str(fields.get("split_rule") or ""),
        "model_type": str(fields.get("model_type") or ""),
        "model_path": str(fields.get("model_path") or ""),
        "card_path": str(fields.get("card_path") or ""),
        "measure": str(fields.get("measure") or ""),
        "diversity": str(fields.get("diversity") or ""),
        "notes_json": json.dumps(list(fields.get("notes") or [])),
    }
    con = _write_connection(db_path)
    try:
        con.execute(
            f"INSERT OR REPLACE INTO {_quote_ident(ROUND_LOG_TABLE)} "
            f"({', '.join(_quote_ident(c) for c in values)}) "
            f"VALUES ({', '.join('?' * len(values))})",
            tuple(values.values()))
        con.commit()
    finally:
        con.close()


def _as_float_or_none(value: Any) -> Optional[float]:
    """``float(value)`` unless it is None or non-finite."""
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def learning_curve(db_path: str,
                   annotation_column: str = "annotate") -> pd.DataFrame:
    """Held-out accuracy per round, oldest first — the curve to watch flatten.

    :param db_path: path to ``measurements.db``.
    :param annotation_column: the column the rounds labelled into.
    :returns: a frame with one row per round: ``round``, ``finished_utc``,
        ``n_labels``, ``n_new_labels``, ``n_holdout``, ``holdout_accuracy``,
        ``holdout_f1_macro``, ``per_class`` (dict), ``split_rule``,
        ``model_type``, ``model_path``, ``card_path``, ``notes`` (list), and
        the derived ``gain`` (accuracy change since the previous round).
        Empty (with those columns) when no round has been recorded.
    """
    import json
    columns = ["round", "finished_utc", "n_labels", "n_new_labels",
               "n_holdout", "holdout_accuracy", "holdout_f1_macro",
               "per_class", "split_rule", "model_type", "model_path",
               "card_path", "measure", "diversity", "notes", "gain"]
    try:
        con = _connect(db_path)
    except (FileNotFoundError, ValueError):
        return pd.DataFrame(columns=columns)
    try:
        names = {r[0] for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
        if ROUND_LOG_TABLE not in names:
            return pd.DataFrame(columns=columns)
        rows = con.execute(
            f"SELECT round, finished_utc, n_labels, n_new_labels, n_holdout, "
            f"holdout_accuracy, holdout_f1_macro, per_class_json, split_rule, "
            f"model_type, model_path, card_path, measure, diversity, "
            f"notes_json FROM {_quote_ident(ROUND_LOG_TABLE)} "
            f"WHERE annotation_column = ? ORDER BY round",
            (str(annotation_column),)).fetchall()
    finally:
        con.close()

    if not rows:
        return pd.DataFrame(columns=columns)
    frame = pd.DataFrame(rows, columns=[
        "round", "finished_utc", "n_labels", "n_new_labels", "n_holdout",
        "holdout_accuracy", "holdout_f1_macro", "per_class_json",
        "split_rule", "model_type", "model_path", "card_path", "measure",
        "diversity", "notes_json"])
    frame["per_class"] = [json.loads(v or "{}") for v in frame.pop("per_class_json")]
    frame["notes"] = [json.loads(v or "[]") for v in frame.pop("notes_json")]
    frame["gain"] = frame["holdout_accuracy"].astype(float).diff()
    return frame[columns]


# ---------------------------------------------------------------------------
# The stopping rule
# ---------------------------------------------------------------------------

class StoppingVerdict:
    """Whether the last stretch of annotation bought anything measurable.

    :param stop: the recommendation.
    :param reason: one sentence, in the words the screen shows.
    :param gain: held-out accuracy change over the window examined.
    :param labels_in_window: how many labels that change is attributed to.
    :param window_from: the round the window opened at.
    :param confident: whether ``gain`` is larger than one standard error of
        the held-out accuracy itself. When it is *not*, "flat" and "we
        cannot tell" look identical from the numbers, and this says which
        you have.
    :param noise: one standard error of the latest held-out accuracy,
        ``sqrt(p(1-p)/n)``.
    :param trend: ``'rising'``, ``'flat'``, ``'falling'`` or ``'unknown'``.
    """

    __slots__ = ("stop", "reason", "gain", "labels_in_window", "window_from",
                 "confident", "noise", "trend")

    def __init__(self, stop: bool, reason: str, *, gain: Optional[float] = None,
                 labels_in_window: int = 0, window_from: Optional[int] = None,
                 confident: bool = False, noise: Optional[float] = None,
                 trend: str = "unknown"):
        self.stop = bool(stop)
        self.reason = str(reason)
        self.gain = gain
        self.labels_in_window = int(labels_in_window)
        self.window_from = window_from
        self.confident = bool(confident)
        self.noise = noise
        self.trend = str(trend)

    def __bool__(self) -> bool:
        """True when the recommendation is to stop."""
        return self.stop

    def __repr__(self) -> str:
        return (f"StoppingVerdict(stop={self.stop!r}, trend={self.trend!r}, "
                f"gain={self.gain!r}, labels_in_window="
                f"{self.labels_in_window!r})")

    def to_dict(self) -> Dict[str, Any]:
        """A JSON-friendly copy, for a card or a log."""
        return {name: getattr(self, name) for name in self.__slots__}


def should_stop(curve: pd.DataFrame, *, label_window: int = 50,
                min_gain: float = 0.003,
                min_rounds: int = 2) -> StoppingVerdict:
    """Has the last ``label_window`` labels moved held-out accuracy at all?

    The rule, in one line: **look back over whole rounds until at least
    ``label_window`` new labels have accumulated, and compare held-out
    accuracy at the two ends. If it moved by less than ``min_gain``, stop.**

    Why this rule and not another:

    * **Labels, not rounds, are the unit of cost.** A round is whatever size
      the annotator felt like; "no improvement for 3 rounds" says nothing
      when the rounds were 5, 300 and 8 labels. The thing being spent is
      human attention, one crop at a time, so the window is measured in
      crops.
    * **It looks *back over* rounds, not *at* the last one.** A single round
      that happened to land flat is noise; the question is whether the last
      fifty labels — however they were divided up — bought anything.
    * **It refuses to answer early.** Below ``label_window`` labels since the
      first recorded round there is no window to measure, and a rule that
      fired anyway would tell people to stop after their first twelve labels.
    * **It distinguishes "flat" from "unmeasurable".** Held-out accuracy on
      80 objects has a standard error near 0.05; a 0.3 % change is inside
      the noise, and :attr:`StoppingVerdict.confident` says so instead of
      dressing it up. Flat *is* still the recommendation — if more labels
      are not moving a number you can measure, they are not buying anything
      you can demonstrate — but the reason says the held-out set is too
      small to prove convergence, which is a different piece of work.
    * **A falling curve stops too, and says so.** Accuracy going down is not
      convergence; it usually means the newest labels disagree with the
      earlier ones, or that the held-out split moved. Either way, more of
      the same is the wrong next move.

    :param curve: the frame from :func:`learning_curve`.
    :param label_window: how many labels the window must cover.
    :param min_gain: accuracy change below which the window counts as flat.
    :param min_rounds: rounds required before any verdict is given.
    :returns: a :class:`StoppingVerdict`; ``bool(verdict)`` is the answer.
    """
    if curve is None or not len(curve):
        return StoppingVerdict(
            False, "No round has been recorded yet — retrain once to start "
                   "the curve.")
    usable = curve[curve["holdout_accuracy"].notna()].reset_index(drop=True)
    if len(usable) < max(1, int(min_rounds)):
        return StoppingVerdict(
            False,
            f"Only {len(usable)} round(s) with a held-out score; "
            f"{int(min_rounds)} are needed before a stopping rule means "
            f"anything.",
            trend="unknown")

    latest = usable.iloc[-1]
    accuracy = float(latest["holdout_accuracy"])
    n_holdout = int(latest.get("n_holdout", 0) or 0)
    noise = (float(np.sqrt(max(accuracy * (1.0 - accuracy), 0.0) / n_holdout))
             if n_holdout > 0 else None)

    # Walk back until the window covers enough NEW labels.
    accumulated = 0
    index = len(usable) - 1
    while index > 0 and accumulated < int(label_window):
        accumulated += int(usable.iloc[index].get("n_new_labels", 0) or 0)
        index -= 1
    baseline = usable.iloc[index]
    gain = accuracy - float(baseline["holdout_accuracy"])
    window_from = int(baseline["round"])

    if accumulated < int(label_window):
        return StoppingVerdict(
            False,
            f"Only {accumulated} labels since round {window_from}; the rule "
            f"waits for {int(label_window)} before calling a plateau. "
            f"Held-out accuracy has moved {gain:+.3f} so far.",
            gain=gain, labels_in_window=accumulated, window_from=window_from,
            noise=noise, trend="unknown")

    confident = bool(noise is not None and abs(gain) > noise)
    if gain < -max(float(min_gain), 0.0):
        return StoppingVerdict(
            True,
            f"Held-out accuracy FELL {abs(gain):.1%} over the last "
            f"{accumulated} labels (round {window_from} → "
            f"{int(latest['round'])}). That is not convergence: check "
            f"whether the newest labels disagree with the earlier ones "
            f"before adding more.",
            gain=gain, labels_in_window=accumulated, window_from=window_from,
            confident=confident, noise=noise, trend="falling")

    if gain < float(min_gain):
        reason = (
            f"The last {accumulated} labels moved held-out accuracy by "
            f"{gain:+.1%} (round {window_from} → {int(latest['round'])}, now "
            f"{accuracy:.1%}). That is below the {float(min_gain):.1%} "
            f"threshold — annotating more of the same is not buying "
            f"measurable accuracy.")
        if noise is not None and abs(gain) <= noise:
            reason += (
                f" Note that the held-out set is only {n_holdout} objects, so "
                f"one standard error is {noise:.1%}: this says the gain is "
                f"unmeasurable, not that it is provably zero. A larger "
                f"held-out set is the way to tell those apart.")
        return StoppingVerdict(
            True, reason, gain=gain, labels_in_window=accumulated,
            window_from=window_from, confident=confident, noise=noise,
            trend="flat")

    return StoppingVerdict(
        False,
        f"Still learning: the last {accumulated} labels moved held-out "
        f"accuracy {gain:+.1%} (round {window_from} → "
        f"{int(latest['round'])}, now {accuracy:.1%}). Keep going.",
        gain=gain, labels_in_window=accumulated, window_from=window_from,
        confident=confident, noise=noise, trend="rising")


def format_learning_curve(curve: pd.DataFrame,
                          verdict: Optional[StoppingVerdict] = None) -> str:
    """Render the round-by-round curve and the stopping verdict as text."""
    lines = ["Active-learning rounds"]
    if curve is None or not len(curve):
        lines.append("")
        lines.append("No round recorded yet. Retrain from Annotate to start "
                     "the curve.")
        return "\n".join(lines)
    lines.append(f"  {'round':>5}{'labels':>8}{'new':>6}{'held-out':>10}"
                 f"{'acc':>8}{'gain':>8}  worst class")
    for _, row in curve.iterrows():
        acc = row["holdout_accuracy"]
        gain = row["gain"]
        per_class = row["per_class"] or {}
        worst = ""
        if per_class:
            name = min(per_class, key=lambda k: per_class[k])
            worst = f"{name} {float(per_class[name]):.3f}"
        lines.append(
            f"  {int(row['round']):>5}{int(row['n_labels']):>8}"
            f"{int(row['n_new_labels']):>6}{int(row['n_holdout']):>10}"
            f"{('  —  ' if acc is None or not np.isfinite(float(acc)) else f'{float(acc):8.3f}')}"
            f"{('     —  ' if gain is None or not np.isfinite(float(gain)) else f'{float(gain):+8.3f}')}"
            f"  {worst}")
    if verdict is not None:
        lines.append("")
        lines.append(("STOP — " if verdict.stop else "CONTINUE — ")
                     + verdict.reason)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Retraining a round: fit on the labels so far, re-score, re-rank
# ---------------------------------------------------------------------------

def holdout_report(y_true: Any, probs: Any,
                   classes: Optional[Sequence[Any]] = None) -> Dict[str, Any]:
    """Held-out metrics, with the confusion matrix they were derived from.

    Torch-free, so both the classical-ML round here and
    :func:`spacr.deep_spacr.held_out_report` can use one implementation and
    a model card written by either says the same thing in the same shape.

    Every derived figure is exactly the standard function of the matrix —
    ``accuracy = trace / total``, ``per_class[c] = M[c, c] / M[c, :].sum()``
    — so a reader can recompute the card rather than trust it.

    :param y_true: integer class ids, shape ``(N,)``.
    :param probs: ``(N,)`` positive-class probabilities, or ``(N, C)`` rows.
    :param classes: class names in head order.
    :returns: ``n``, ``num_classes``, ``classes``, ``accuracy``,
        ``f1_macro``, ``per_class_accuracy``, ``class_support``,
        ``predicted_support``, ``confusion_matrix``.
    """
    y_true = np.asarray(y_true, dtype=int).reshape(-1)
    matrix_in = np.asarray(probs, dtype=float)
    if matrix_in.ndim == 1:
        matrix_in = np.column_stack([1.0 - matrix_in, matrix_in])
    elif matrix_in.ndim == 2 and matrix_in.shape[1] == 1:
        col = matrix_in[:, 0]
        matrix_in = np.column_stack([1.0 - col, col])
    n_classes = int(matrix_in.shape[1]) if matrix_in.size else \
        int(max(2, (y_true.max() + 1) if y_true.size else 2))
    preds = (matrix_in.argmax(axis=1).astype(int) if matrix_in.size
             else np.zeros(0, dtype=int))

    matrix = np.zeros((n_classes, n_classes), dtype=np.int64)
    if y_true.size:
        inside = (y_true >= 0) & (y_true < n_classes)
        np.add.at(matrix, (y_true[inside], preds[inside]), 1)
    row_sums = matrix.sum(axis=1)
    per_class = np.where(row_sums > 0,
                         np.diag(matrix) / np.maximum(row_sums, 1), 0.0)
    total = int(matrix.sum())

    # Macro F1 straight off the matrix — no sklearn import on this path, and
    # the number is then provably the same function of the matrix as the
    # accuracy beside it.
    col_sums = matrix.sum(axis=0)
    diag = np.diag(matrix).astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(col_sums > 0, diag / np.maximum(col_sums, 1), 0.0)
        recall = np.where(row_sums > 0, diag / np.maximum(row_sums, 1), 0.0)
        f1 = np.where((precision + recall) > 0,
                      2 * precision * recall / np.maximum(precision + recall, 1e-12),
                      0.0)
    present = row_sums + col_sums > 0
    f1_macro = float(f1[present].mean()) if present.any() else float("nan")

    names = ([str(c) for c in classes] if classes is not None
             and len(classes) == n_classes
             else [f"class_{i}" for i in range(n_classes)])
    return {
        "n": int(len(y_true)),
        "num_classes": n_classes,
        "classes": names,
        "accuracy": (float(np.trace(matrix)) / total) if total else float("nan"),
        "f1_macro": f1_macro,
        "per_class_accuracy": [float(v) for v in per_class],
        "class_support": [int(v) for v in row_sums],
        "predicted_support": [int(v) for v in col_sums],
        "confusion_matrix": [[int(v) for v in row] for row in matrix],
    }


#: What ``group_by`` accepts, and the columns each strategy groups over.
_SPLIT_GROUPS: Dict[str, Tuple[str, ...]] = {
    "well": ("plateID", "rowID", "columnID"),
    "plate": ("plateID",),
    "field": ("plateID", "rowID", "columnID", "fieldID"),
    "none": (),
}


def _grouped_split(groups: np.ndarray, labels: np.ndarray, holdout: float,
                   seed: int, notes: List[str]) -> Tuple[np.ndarray, np.ndarray, str]:
    """Draw a held-out set that does not share a well with the training set.

    The default random split is the reason active-learning accuracy numbers
    are usually too good: with 190 of 200 labels from one well, a random 20 %
    holds out objects whose near neighbours are in the training set, and the
    model is scored on memorising a field of view.

    :returns: ``(train_index, test_index, rule)`` — ``rule`` is the sentence
        that goes into the model card.
    """
    from sklearn.model_selection import (GroupShuffleSplit,
                                         StratifiedGroupKFold,
                                         train_test_split)

    n = len(labels)
    distinct = np.unique(groups) if len(groups) else np.zeros(0)
    frac = min(max(float(holdout), 0.05), 0.5)

    if len(distinct) < 2:
        where = distinct[0] if len(distinct) else "(unknown)"
        notes.append(
            f"Every label comes from one group ({where}), so a grouped "
            f"held-out split is impossible. Falling back to a stratified "
            f"random split of objects, whose accuracy will be optimistic — "
            f"it measures how well the model memorised this one well, not "
            f"whether it transfers.")
        train_idx, test_idx = train_test_split(
            np.arange(n), test_size=frac, random_state=seed,
            stratify=labels if len(np.unique(labels)) > 1 else None)
        return (np.sort(train_idx), np.sort(test_idx),
                f"stratified random {frac:.0%} of objects — NOT grouped, "
                f"because all labels came from {where}")

    n_splits = int(max(2, min(round(1.0 / frac), len(distinct))))
    try:
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                                        random_state=seed)
        for train_idx, test_idx in splitter.split(np.zeros((n, 1)), labels,
                                                  groups):
            if len(np.unique(labels[test_idx])) >= 2:
                return (np.sort(train_idx), np.sort(test_idx),
                        f"StratifiedGroupKFold({n_splits}) over "
                        f"{len(distinct)} groups — no group appears on both "
                        f"sides")
        notes.append(
            "No stratified grouped fold contained more than one class in the "
            "held-out half; falling back to GroupShuffleSplit, so the "
            "held-out class balance is whatever the groups happened to give.")
    except ValueError as exc:
        notes.append(f"StratifiedGroupKFold could not split these labels "
                     f"({exc}); falling back to GroupShuffleSplit.")

    splitter = GroupShuffleSplit(n_splits=1, test_size=frac,
                                 random_state=seed)
    train_idx, test_idx = next(iter(splitter.split(np.zeros((n, 1)), labels,
                                                   groups)))
    return (np.sort(train_idx), np.sort(test_idx),
            f"GroupShuffleSplit({frac:.0%}) over {len(distinct)} groups — no "
            f"group appears on both sides")


def round_features(db_path: str, table: str = PNG_TABLE,
                   key: str = PNG_KEY,
                   tables: Sequence[str] = ("cell", "nucleus", "pathogen",
                                            "cytoplasm"),
                   nuclei_limit: int = 10,
                   pathogen_limit: int = 10) -> pd.DataFrame:
    """The measurement features for every crop, indexed by ``png_path``.

    The feature matrix an in-screen retrain fits on. Measurement features
    rather than pixels on purpose: the point of retraining from inside
    Annotate is to get a fresh ranking in seconds, on the machine the
    annotator is sitting at, without a GPU and without leaving the screen.
    A CNN retrain is the right thing to do at the *end* of the loop, not
    between two pages of crops.

    :param db_path: path to ``measurements.db``.
    :param table: crop table carrying ``png_path`` and ``prcfo``.
    :param key: the crop key column.
    :param tables: object tables to merge features from; missing ones are
        skipped.
    :param nuclei_limit: passed through to :func:`spacr.io._read_and_merge_data`.
    :param pathogen_limit: likewise.
    :returns: numeric features indexed by ``png_path``.
    :raises ValueError: when no object table with features could be read.
    """
    from .io import _read_and_merge_data, _read_db

    con = _connect(db_path)
    try:
        columns = _table_columns(con, table, db_path)
        if "prcfo" not in columns:
            raise ValueError(
                f"{table!r} has no 'prcfo' column, so a crop cannot be matched "
                f"to its measurements. Re-run Measure with save_png=True.")
        rows = con.execute(
            f"SELECT {_quote_ident(key)}, \"prcfo\" FROM {_quote_ident(table)}"
        ).fetchall()
        available = {r[0] for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table','view')"
        ).fetchall()}
    finally:
        con.close()

    wanted = [t for t in tables if t in available]
    if not wanted:
        raise ValueError(
            f"{os.path.basename(str(db_path))} has none of the object tables "
            f"{', '.join(tables)}, so there are no features to fit on. Run "
            f"Measure first, or pass features= explicitly.")

    merged, _ = _read_and_merge_data([str(db_path)], wanted, False,
                                     nuclei_limit=nuclei_limit,
                                     pathogen_limit=pathogen_limit)
    crops = pd.DataFrame(rows, columns=[key, "prcfo"]).dropna(subset=["prcfo"])
    numeric = merged.select_dtypes(include=[np.number])
    joined = crops.join(numeric, on="prcfo", how="inner")
    return joined.drop(columns=["prcfo"]).set_index(key)


class RoundResult:
    """What one retrain round produced.

    :param round_index: the round number recorded.
    :param n_labels: labels the model was fitted on.
    :param n_new_labels: labels added since the previous round.
    :param report: the :func:`holdout_report` for this round.
    :param split_rule: how the held-out set was drawn, in words.
    :param scored: how many crops were re-scored in the database.
    :param score_columns: the columns written back.
    :param model_path: where the fitted model was saved, if it was.
    :param card_path: the model card beside it, if one was written.
    :param verdict: the :class:`StoppingVerdict` after this round.
    :param notes: anything the round wants the annotator to know.
    """

    __slots__ = ("round_index", "n_labels", "n_new_labels", "report",
                 "split_rule", "scored", "score_columns", "model_path",
                 "card_path", "verdict", "notes", "classes", "model_type")

    def __init__(self, **fields: Any):
        for name in self.__slots__:
            setattr(self, name, fields.get(name))
        self.notes = list(self.notes or [])
        self.score_columns = list(self.score_columns or [])
        self.report = dict(self.report or {})

    @property
    def accuracy(self) -> float:
        """Held-out accuracy of this round."""
        return float(self.report.get("accuracy", float("nan")))

    @property
    def per_class(self) -> Dict[str, float]:
        """``{class name: held-out accuracy}`` for this round."""
        names = self.report.get("classes") or []
        accs = self.report.get("per_class_accuracy") or []
        return {str(names[i]) if i < len(names) else f"class_{i}": float(a)
                for i, a in enumerate(accs)}

    def __repr__(self) -> str:
        return (f"RoundResult(round={self.round_index!r}, "
                f"n_labels={self.n_labels!r}, accuracy={self.accuracy:.4f})")

    def summary(self) -> str:
        """One paragraph: the round, its numbers and what to do next."""
        lines = [f"Round {self.round_index}: fitted on {self.n_labels} labels "
                 f"({self.n_new_labels} new), held out "
                 f"{self.report.get('n', 0)}."]
        lines.append(f"Split: {self.split_rule}")
        lines.append(f"Held-out accuracy {self.accuracy:.3f} · macro-F1 "
                     f"{float(self.report.get('f1_macro', float('nan'))):.3f}")
        per_class = self.per_class
        if per_class:
            lines.append("Per class: " + " · ".join(
                f"{k} {v:.3f}" for k, v in per_class.items()))
            worst = min(per_class, key=per_class.get)
            if per_class[worst] < 0.6:
                lines.append(
                    f"Weakest class is {worst} at {per_class[worst]:.3f} — the "
                    f"aggregate above is not describing it.")
        lines.append(f"Re-scored {self.scored} crops into "
                     f"{', '.join(self.score_columns) or 'nothing'}; the queue "
                     f"re-ranks on the next rebuild.")
        if self.verdict is not None:
            lines.append(("STOP — " if self.verdict.stop else "CONTINUE — ")
                         + self.verdict.reason)
        for note in self.notes:
            lines.append(f"! {note}")
        return "\n".join(lines)


def retrain_round(db_path: str, annotation_column: str = "annotate", *,
                  features: Optional[pd.DataFrame] = None,
                  model_type: str = "logistic_regression",
                  group_by: str = "well", holdout: float = 0.25,
                  seed: int = 0, min_labels: int = 8,
                  round_index: Optional[int] = None,
                  table: str = PNG_TABLE, key: str = PNG_KEY,
                  image_type: Optional[str] = None,
                  write_scores: bool = True, save_model: bool = True,
                  model_dir: Optional[str] = None,
                  write_card: bool = True,
                  label_window: int = 50,
                  min_gain: float = 0.003,
                  measure: Any = DEFAULT_MEASURE,
                  diversity: Any = "well") -> RoundResult:
    """Fit a model on the labels so far, score every crop, close the loop.

    This is the half of active learning that has been missing: the queue put
    the informative crops in front of the annotator, and then nothing
    happened. Annotating without retraining is not active learning — it is
    ordinary annotation in a clever order, and the order goes stale after the
    first few dozen labels because it still reflects a model that has not
    seen any of them.

    One call does all five things the loop needs:

    1. fits a model on every label in ``annotation_column``;
    2. scores it on a **grouped** held-out split, so the number is not an
       artefact of 190 labels coming from one well;
    3. writes per-class probabilities back into ``png_list`` as
       :data:`ROUND_PRED_PREFIX` columns, which :func:`build_queue` prefers
       over the older ``pred`` — so the next queue is genuinely re-ranked;
    4. records the round, giving :func:`learning_curve` another point;
    5. returns the :class:`StoppingVerdict` for the curve so far.

    :param db_path: path to ``measurements.db``.
    :param annotation_column: the column holding the labels.
    :param features: feature matrix indexed by the crop key. Omitted, it is
        read from the measurement tables with :func:`round_features`.
    :param model_type: ``'logistic_regression'`` (default — it is the one
        that behaves at 20 labels), ``'random_forest'`` or
        ``'gradient_boosting'``.
    :param group_by: ``'well'`` (default), ``'plate'``, ``'field'`` or
        ``'none'``. What the held-out split refuses to share.
    :param holdout: fraction held out.
    :param seed: makes the split and the fit reproducible.
    :param min_labels: refuse to fit below this many labels.
    :param round_index: override the round number; defaults to
        :func:`next_round`.
    :param table: crop table.
    :param key: crop key column.
    :param image_type: substring filter on the crop key.
    :param write_scores: write the new probabilities back into the database.
    :param save_model: joblib-dump the fitted model beside the database.
    :param model_dir: where to put it; defaults to ``<db dir>/active_learning``.
    :param write_card: write a model card beside the saved model.
    :param label_window: passed to :func:`should_stop`.
    :param min_gain: passed to :func:`should_stop`.
    :param measure: recorded with the round, for the queue that follows.
    :param diversity: likewise.
    :returns: a :class:`RoundResult`.
    :raises ValueError: below ``min_labels`` labels, or with fewer than two
        classes annotated — neither is something to paper over with a model
        that will produce a confident-looking ranking out of nothing.
    """
    notes: List[str] = []

    con = _connect(db_path)
    try:
        columns = _table_columns(con, table, db_path)
        if annotation_column not in columns:
            raise ValueError(
                f"{table!r} has no {annotation_column!r} column — there are no "
                f"labels to retrain on yet.")
        meta = [c for c in ("plateID", "rowID", "columnID", "fieldID")
                if c in columns]
        select = list(dict.fromkeys([key, annotation_column] + meta))
        rows = con.execute(
            f"SELECT {', '.join(_quote_ident(c) for c in select)} "
            f"FROM {_quote_ident(table)}").fetchall()
    finally:
        con.close()

    crops = pd.DataFrame(rows, columns=select)
    if image_type:
        crops = crops[crops[key].astype(str).str.contains(
            str(image_type), regex=False)]
    crops = crops.set_index(key)
    # A database measured with more than one crop_mode holds several rows per
    # prcfo. A duplicated key here fans the feature join out, so the label
    # vector and the feature matrix stop lining up row for row and the model
    # is fitted against the wrong labels — silently, with a plausible score.
    crops = crops.loc[~crops.index.duplicated(keep="first")]

    if features is None:
        features = round_features(db_path, table=table, key=key)
    features = features.select_dtypes(include=[np.number])
    features = features.loc[~features.index.duplicated(keep="first")]
    shared = crops.index.intersection(features.index)
    if not len(shared):
        raise ValueError(
            f"No crop in {table} has a row in the feature matrix, so nothing "
            f"can be fitted. Check that Measure and the crop export ran over "
            f"the same objects.")
    crops = crops.loc[shared]
    matrix = features.loc[shared]

    labelled_mask = crops[annotation_column].notna()
    n_labels = int(labelled_mask.sum())
    if n_labels < int(min_labels):
        raise ValueError(
            f"Only {n_labels} labels in {annotation_column!r} (need at least "
            f"{int(min_labels)}). A model fitted on fewer will still emit a "
            f"confident-looking ranking, and it will be noise.")

    raw_labels = crops.loc[labelled_mask, annotation_column].to_numpy()
    class_values = sorted({_class_value(v) for v in raw_labels})
    if len(class_values) < 2:
        raise ValueError(
            f"Every label in {annotation_column!r} is class "
            f"{class_values[0] if class_values else 'none'}. A classifier "
            f"needs at least two classes; keep annotating until the other "
            f"one appears.")
    class_index = {value: i for i, value in enumerate(class_values)}
    y = np.array([class_index[_class_value(v)] for v in raw_labels], dtype=int)

    train_matrix = matrix.loc[labelled_mask.to_numpy()]
    x = np.nan_to_num(train_matrix.to_numpy(dtype=float), nan=0.0,
                      posinf=0.0, neginf=0.0)

    group_cols = [c for c in _SPLIT_GROUPS.get(str(group_by), ())
                  if c in crops.columns]
    if str(group_by) != "none" and not group_cols:
        notes.append(
            f"{table} carries none of the columns needed to group by "
            f"{group_by!r}, so the held-out split is a plain random one and "
            f"its accuracy is optimistic.")
    groups = (crops.loc[labelled_mask, group_cols].astype(str)
              .apply("/".join, axis=1).to_numpy()
              if group_cols else np.arange(n_labels).astype(str))

    train_idx, test_idx, split_rule = _grouped_split(
        groups, y, holdout, int(seed), notes)

    model = _build_round_model(model_type, int(seed), len(class_values))
    model.fit(x[train_idx], y[train_idx])
    test_probs = _predict_proba(model, x[test_idx], len(class_values))
    report = holdout_report(y[test_idx], test_probs,
                            [str(v) for v in class_values])

    all_x = np.nan_to_num(matrix.to_numpy(dtype=float), nan=0.0,
                          posinf=0.0, neginf=0.0)
    all_probs = _predict_proba(model, all_x, len(class_values))

    if round_index is None:
        round_index = next_round(db_path, annotation_column)
    previous = learning_curve(db_path, annotation_column)
    prior_labels = (int(previous["n_labels"].iloc[-1]) if len(previous)
                    else 0)
    n_new = max(0, n_labels - prior_labels)

    score_columns: List[str] = []
    scored = 0
    if write_scores:
        score_columns = [f"{ROUND_PRED_PREFIX}{i}"
                         for i in range(len(class_values))]
        scored = _write_round_scores(db_path, matrix.index, all_probs,
                                     score_columns, table=table, key=key)

    model_path = ""
    card_path = ""
    if save_model:
        target_dir = model_dir or os.path.join(
            os.path.dirname(os.path.abspath(str(db_path))), "active_learning")
        os.makedirs(target_dir, exist_ok=True)
        model_path = os.path.join(
            target_dir, f"round_{int(round_index):03d}_{model_type}.joblib")
        try:
            import joblib
            joblib.dump({"model": model, "classes": class_values,
                         "features": list(matrix.columns)}, model_path)
        except Exception as exc:
            notes.append(f"Could not save the round model ({exc}).")
            model_path = ""
        if model_path and write_card:
            card_path = _write_round_card(
                model_path, report, split_rule, round_index,
                annotation_column, db_path, class_values, matrix.columns,
                model_type, n_labels, n_new, notes)

    per_class = {str(name): float(acc) for name, acc in
                 zip(report["classes"], report["per_class_accuracy"])}
    record_round(db_path, annotation_column, int(round_index),
                 n_labels=n_labels, n_new_labels=n_new,
                 n_holdout=report["n"],
                 holdout_accuracy=report["accuracy"],
                 holdout_f1_macro=report["f1_macro"],
                 per_class=per_class, split_rule=split_rule,
                 model_type=model_type, model_path=model_path,
                 card_path=card_path,
                 measure=str(measure), diversity=str(diversity),
                 notes=notes)

    verdict = should_stop(learning_curve(db_path, annotation_column),
                          label_window=label_window, min_gain=min_gain)
    return RoundResult(round_index=int(round_index), n_labels=n_labels,
                       n_new_labels=n_new, report=report,
                       split_rule=split_rule, scored=scored,
                       score_columns=score_columns, model_path=model_path,
                       card_path=card_path, verdict=verdict, notes=notes,
                       classes=[str(v) for v in class_values],
                       model_type=model_type)


def _class_value(value: Any) -> Any:
    """``1.0`` and ``1`` are one class; normalise to int where possible."""
    if isinstance(value, float) and float(value).is_integer():
        return int(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    return value


def _build_round_model(model_type: str, seed: int, n_classes: int):
    """The estimator a round fits. Small-data-first, no torch, no GPU."""
    name = str(model_type).lower().replace("-", "_")
    if name in ("logistic_regression", "logistic", "lr"):
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        # Scaled, because measurement features span areas in the thousands
        # and intensities in the fractions, and an unscaled linear model on
        # that is a model of whichever column has the biggest units.
        return Pipeline([
            ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, random_state=seed,
                                         class_weight="balanced")),
        ])
    if name in ("random_forest", "rf"):
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_estimators=300, random_state=seed,
                                      n_jobs=-1, class_weight="balanced")
    if name in ("gradient_boosting", "hist_gradient_boosting", "gb"):
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(random_state=seed)
    raise ValueError(
        f"Unknown model_type {model_type!r}; use 'logistic_regression', "
        f"'random_forest' or 'gradient_boosting'.")


def _predict_proba(model: Any, x: np.ndarray, n_classes: int) -> np.ndarray:
    """``(N, n_classes)`` probabilities, padding classes the fit never saw.

    A fold that happened to contain only two of three classes leaves an
    estimator whose ``predict_proba`` has two columns. Returning that
    unpadded would silently renumber the classes downstream — the queue
    would call class 2 "class 1" — so the missing columns are filled with
    zeros in the right places instead.
    """
    if not len(x):
        return np.zeros((0, n_classes))
    probs = np.asarray(model.predict_proba(x), dtype=float)
    seen = np.asarray(getattr(model, "classes_", np.arange(probs.shape[1])),
                      dtype=int)
    if probs.shape[1] == n_classes and np.array_equal(seen,
                                                      np.arange(n_classes)):
        return probs
    out = np.zeros((probs.shape[0], n_classes), dtype=float)
    for column, class_id in enumerate(seen):
        if 0 <= int(class_id) < n_classes:
            out[:, int(class_id)] = probs[:, column]
    return out


def _write_round_scores(db_path: str, keys: Any, probs: np.ndarray,
                        columns: Sequence[str], table: str = PNG_TABLE,
                        key: str = PNG_KEY) -> int:
    """Write per-class probabilities back into ``table``; rows updated.

    Goes through :func:`spacr.predictions.merge_prediction_results`, which
    already knows that ``png_list`` has a real column called ``rowID`` that
    shadows SQLite's ``rowid`` — the exact trap a hand-rolled UPDATE here
    would fall into.
    """
    from .predictions import merge_prediction_results

    frame = pd.DataFrame(np.asarray(probs, dtype=float),
                         columns=list(columns))
    frame[key] = list(keys)
    report = merge_prediction_results(
        frame, db_path, {c: (c, "REAL") for c in columns},
        table=table, key=key, verbose=False)
    return int(getattr(report, "matched_rows", 0) or 0)


def _write_round_card(model_path: str, report: Dict[str, Any],
                      split_rule: str, round_index: int,
                      annotation_column: str, db_path: str,
                      class_values: Sequence[Any], feature_columns: Any,
                      model_type: str, n_labels: int, n_new: int,
                      notes: List[str]) -> str:
    """Write the model card for one round's model. Never fatal."""
    try:
        from .deep_spacr import model_card
        coverage = annotation_coverage(db_path, annotation_column)
        coverage_meta = dict(
            coverage.attrs.get("spacr_annotation_coverage", {}))
        card, card_path, _artifact = model_card(
            model_path,
            settings={"annotation_column": annotation_column,
                      "model_type": model_type, "round": int(round_index),
                      "db_path": str(db_path)},
            classes=[str(v) for v in class_values],
            split_rule=split_rule,
            held_out=report,
            class_balance={"annotated": coverage_meta.get("by_class", {})},
            dataset_src=os.path.dirname(os.path.abspath(str(db_path))),
            module="active_learning",
            extra={
                "round": int(round_index),
                "n_labels": int(n_labels),
                "n_new_labels": int(n_new),
                "n_features": int(len(list(feature_columns))),
                "annotation_coverage": {
                    k: coverage_meta.get(k) for k in
                    ("by_class", "by_plate", "by_well", "by_round",
                     "concentration", "wells_annotated", "plates_annotated")},
            },
        )
        return card_path
    except Exception as exc:
        notes.append(f"Round model card could not be written ({exc}).")
        return ""
