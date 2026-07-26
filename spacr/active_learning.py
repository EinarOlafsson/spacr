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

from .agreement import PNG_KEY, PNG_TABLE

__all__ = [
    "CALIBRATION_NOTE",
    "DEFAULT_MEASURE",
    "DIVERSITY_GROUPS",
    "PNG_KEY",
    "PNG_TABLE",
    "PRED_COLUMN_CANDIDATES",
    "UNCERTAINTY_MEASURES",
    "as_probabilities",
    "build_queue",
    "disagreement",
    "entropy",
    "format_queue_summary",
    "least_confidence",
    "margin",
    "predict_probabilities",
    "probabilities_from_logits",
    "queue_rows",
    "rank_by_uncertainty",
    "resolve_measure",
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
    gap = filled[:, -1] - filled[:, -2]
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

    # Multi-class columns first: pred_0, pred_1, … / prob_0, prob_1, …
    for prefix in ("pred_", "prob_", "score_"):
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
