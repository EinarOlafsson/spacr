"""Ways of choosing which cells get annotated, and what each one costs.

A screen holds hundreds of thousands of cells and an annotator has an
afternoon, so the choice of WHICH cells to label is the experiment. This
module is the menu of those choices for the Cells tab, and the machinery
that runs one end to end: it picks the cells, fits a classifier on what it
picked, applies that classifier to every remaining cell in the screen and
in the chosen guide wells, and reports how well it did against a hold-out
it was not allowed to choose from.

THE ONE THAT IS ASKED FOR MOST OFTEN. In the chosen guide wells, take the
top-scoring cells as the positive set, draw the same number of cells at
random as the contrast set, fit a gradient-boosted tree on the two, and
apply it to the rest. :data:`TOP_SCORE_RANDOM` is that method.

THE TRAP IN IT. The score already encodes the phenotype. A model trained on
high-score against random can therefore succeed by relearning the score and
nothing else, and its accuracy will look excellent while it has learned no
morphology at all. :func:`score_input_columns` names the columns the score
is a function of, and every fit is reported twice -- once with them and once
without -- so the honest question ("how much of this survives when the
score's own inputs are removed?") is answered on the page rather than left
to the reader. :attr:`LeakageReport.survival` is that number.

TWO RULES HOLD FOR EVERY STRATEGY HERE.

* GROUP BY WELL. Cells from one well are not independent, so a split that
  puts some cells of a well in train and others in test reports a score the
  model will not reach on a new plate. Every split in this module goes
  through :func:`spacr.classifier_evaluation.grouped_split`, which is the
  package's one splitter, at the level :attr:`AnnotationRequest.group_by`
  names.
* THE STRATEGY THAT CHOSE THE LABELS CANNOT ALSO SCORE THEM. Every method
  here selects cells non-randomly, so an accuracy measured on its own
  selection is optimistic by construction. Whole wells are held out first,
  no strategy may select from them, and every reported number is measured
  there -- which is :data:`RANDOM_HOLDOUT`, the plain random draw, doing its
  job for all the others whether or not it is the strategy on screen.

A strategy that is declared but not yet built raises
:class:`StrategyNotImplemented` rather than returning an empty selection,
because a menu entry that silently does nothing is worse than one that is
missing.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import os
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

__all__ = [
    "AnnotationRequest",
    "AnnotationResult",
    "AnnotationStrategyError",
    "FitReport",
    "LeakageReport",
    "NotEnoughLabels",
    "Prepared",
    "Strategy",
    "StrategyNotImplemented",
    "CONTROL_ANCHORS",
    "DIVERSITY",
    "NEIGHBOUR_PROPAGATION",
    "PU_LEARNING",
    "RANDOM_HOLDOUT",
    "SCORE_STRATA",
    "SELF_TRAINING",
    "STRATEGIES",
    "TOP_SCORE_RANDOM",
    "TWO_VIEW_DISAGREEMENT",
    "UNCERTAINTY",
    "candidate_feature_columns",
    "implemented_keys",
    "menu",
    "missing_requirement",
    "prepare",
    "readable_group",
    "run",
    "score_input_columns",
    "strategy",
    "strategy_keys",
    "xgboost_available",
]

#: The per-object classification score a screen writes by default.
DEFAULT_SCORE_COLUMN = "pred"

#: What to do about the score's own inputs when fitting.
#:
#: ``'report'`` fits twice and reports both, which is the only one of the
#: three that answers the question the trap raises; ``'drop'`` fits once
#: without them; ``'keep'`` fits once with them and is the naive method,
#: kept because reproducing it is how its optimism is demonstrated.
LEAKAGE_MODES: Tuple[str, ...] = ("report", "drop", "keep")

#: Name fragments that mark a column as the classifier's OUTPUT rather than
#: a measurement of the cell. A feature matrix containing one of these is
#: fitting the score against itself.
SCORE_NAME_HINTS: Tuple[str, ...] = (
    "pred", "prob", "score", "logit", "class_prob", "cv_", "montage_",
    "recruitment", "annotate",
)

#: Columns that identify a row rather than measure it.
IDENTITY_COLUMNS: Tuple[str, ...] = (
    "plateID", "rowID", "columnID", "fieldID", "prc", "prcfo", "png_path",
    "path_name", "file_name", "object_label", "label", "cell_id",
    "nucleus_id", "pathogen_id", "cytoplasm_id", "object_type", "well",
    "plate", "row_name", "column_name", "field_name", "grna", "gene",
)

#: The most cells a clustering or a nearest-neighbour search runs over.
#: Both build a dense standardised matrix, and a screen of half a million
#: objects by two hundred features is 800 MB before either has started. A
#: larger pool is sampled down to this and the result says so.
MAX_POOL_FOR_DISTANCES = 50_000

#: Feature families :mod:`spacr.column_groups` recognises that describe the
#: object's PIXELS, and the ones that describe its SHAPE. The two-view
#: strategy fits one model on each.
INTENSITY_FAMILIES: Tuple[str, ...] = ("intensity", "texture", "correlation")
SHAPE_FAMILIES: Tuple[str, ...] = ("morphology", "spatial", "moment")


class AnnotationStrategyError(ValueError):
    """A strategy cannot run on the data it was given, and says why."""


class StrategyNotImplemented(AnnotationStrategyError):
    """The strategy is on the menu but has no implementation yet."""


class NotEnoughLabels(AnnotationStrategyError):
    """There is nothing labelled to fit on, so nothing is fitted."""


# ---------------------------------------------------------------------------
# The menu
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Strategy:
    """One entry on the annotation menu.

    :param key: the stored value; stable across releases.
    :param title: what the entry is called on screen.
    :param purpose: what it is for, in one sentence.
    :param cost: what it costs, in one sentence. Every strategy costs
        something and the entry that does not say so is the one that gets
        chosen for the wrong reason.
    :param implemented: whether :func:`run` can execute it. A False entry
        is still offered, and still refuses out loud.
    :param needs: what the data must carry: ``'score'``, ``'features'``,
        ``'labels'``, ``'controls'``.
    """

    key: str
    title: str
    purpose: str
    cost: str
    implemented: bool = True
    needs: Tuple[str, ...] = ()

    def describe(self) -> str:
        """The entry as one paragraph: what it is for, then what it costs."""
        text = f"{self.title} — {self.purpose} Costs: {self.cost}"
        if not self.implemented:
            text += (" NOT YET IMPLEMENTED: this entry is declared so the "
                     "menu is honest about what is coming, and refuses "
                     "rather than selecting nothing.")
        return text


TOP_SCORE_RANDOM = Strategy(
    key="top_score_random",
    title="Top-scoring cells against a matched random draw",
    purpose=(
        "In the chosen guide wells it takes the top-scoring cells as the "
        "positive set, draws the same number of cells at random as the "
        "contrast set, fits a boosted tree on the two and applies it to "
        "every remaining cell in the screen and in those wells."),
    cost=(
        "the score already encodes the phenotype, so the fit can succeed by "
        "relearning the score and learning no morphology at all; the run "
        "therefore reports the same fit with the score's own inputs removed, "
        "and it is the second number that means anything."),
    needs=("score", "features"),
)

UNCERTAINTY = Strategy(
    key="uncertainty",
    title="Uncertainty sampling",
    purpose=(
        "It queues the cells a current model is least sure of — the ones "
        "nearest its decision boundary — because each label bought there "
        "moves the boundary furthest for the money."),
    cost=(
        "it needs a model already, it chases one boundary, and the queue it "
        "returns is not a random sample of anything, so its own cells can "
        "never be used to measure it."),
    needs=("score", "features"),
)

DIVERSITY = Strategy(
    key="diversity",
    title="Diversity sampling over clusters",
    purpose=(
        "It clusters the measured features and queues a representative of "
        "each cluster, which is what stops a whole rare morphology going "
        "unlabelled while uncertainty sampling works one boundary."),
    cost=(
        "clusters are not phenotypes — a cluster can be a plate effect or a "
        "focus artefact — and a budget spread evenly over clusters spends "
        "most of it on the common ones."),
    needs=("features",),
)

CONTROL_ANCHORS = Strategy(
    key="control_anchors",
    title="Control wells as anchors",
    purpose=(
        "Where the plate has known positive and negative control wells it "
        "seeds the labels from them, so there is no manual annotation at "
        "all and the labels carry the experiment's own definition of the "
        "phenotype rather than an annotator's eye."),
    cost=(
        "control wells differ from sample wells in more than the phenotype "
        "— position, seeding density, edge effects — so a model fitted on "
        "them can separate the wells rather than the cells."),
    needs=("controls", "features"),
)

PU_LEARNING = Strategy(
    key="pu_learning",
    title="Positive-unlabelled learning",
    purpose=(
        "The same positives as the named method, but the contrast set is "
        "treated as UNLABELLED rather than negative, and the fitted "
        "probability is rescaled by the labelling rate estimated on held-"
        "out positives — the same idea without the bias of calling every "
        "unlabelled cell a negative."),
    cost=(
        "it assumes the positives were labelled independently of their "
        "features, which top-of-the-score selection breaks; the estimated "
        "labelling rate is reported so the assumption is visible."),
    needs=("score", "features"),
)

SELF_TRAINING = Strategy(
    key="self_training",
    title="Self-training with a fixed audit set",
    purpose=(
        "It fits, adds its own confident predictions as labels, refits and "
        "repeats — the cheapest way to turn a few hundred labels into a "
        "model over the whole screen."),
    cost=(
        "it drifts, confidently: each round trains on the last round's "
        "mistakes. The audit set is fixed before the first round, is held "
        "out of every round, and the run stops when the audit stops "
        "improving rather than when the rounds run out."),
    needs=("score", "features"),
)

TWO_VIEW_DISAGREEMENT = Strategy(
    key="two_view_disagreement",
    title="Disagreement between two feature families",
    purpose=(
        "It fits one model on the intensity and texture columns and another "
        "on the shape and position columns, then queues the cells the two "
        "disagree about — which is where a single model is confidently "
        "wrong."),
    cost=(
        "it needs both families to be present and informative, and it finds "
        "nothing when the two views are really one; the column counts each "
        "view got are reported so a one-sided split is visible."),
    needs=("score", "features"),
)

SCORE_STRATA = Strategy(
    key="score_strata",
    title="Stratified across the score range",
    purpose=(
        "It queues a fixed number of cells per score decile rather than "
        "only the top, because a model shown nothing but extremes has never "
        "seen the middle of the screen, which is most of it."),
    cost=(
        "most deciles hold few positives, so a fixed budget spread across "
        "them buys fewer positive labels than taking the top — it buys "
        "calibration instead of prevalence."),
    needs=("score",),
)

NEIGHBOUR_PROPAGATION = Strategy(
    key="neighbour_propagation",
    title="Neighbour propagation with a shown distance cut",
    purpose=(
        "It carries each seed label to that cell's nearest neighbours in "
        "standardised feature space, within a distance cut, which "
        "multiplies an annotator's effort several times over."),
    cost=(
        "too generous a cut manufactures agreement, so the radius, the "
        "neighbours reached per seed and the share that landed in a "
        "different well are all reported; a large share from other wells "
        "means the cut is measuring the plate, not the phenotype."),
    needs=("score", "features"),
)

RANDOM_HOLDOUT = Strategy(
    key="random_holdout",
    title="An unbiased random sample",
    purpose=(
        "A plain random draw of whole wells, held aside before any strategy "
        "chooses anything. It is the only thing that can measure what a "
        "clever strategy did to the class balance, and every other strategy "
        "here reports against it."),
    cost=(
        "it buys the fewest positives per label of anything on this menu — "
        "which is the point: it is a measurement, not a way to spend an "
        "annotation budget."),
    needs=("score",),
)

#: The menu, in the order it is offered. The named method leads because it
#: is the one asked for; the plain random draw is last because it is the
#: measurement the rest are reported against rather than a way to spend a
#: budget.
STRATEGIES: Tuple[Strategy, ...] = (
    TOP_SCORE_RANDOM,
    UNCERTAINTY,
    DIVERSITY,
    CONTROL_ANCHORS,
    PU_LEARNING,
    SELF_TRAINING,
    TWO_VIEW_DISAGREEMENT,
    SCORE_STRATA,
    NEIGHBOUR_PROPAGATION,
    RANDOM_HOLDOUT,
)


def strategy_keys() -> Tuple[str, ...]:
    """Every strategy key, in menu order."""
    return tuple(entry.key for entry in STRATEGIES)


def implemented_keys() -> Tuple[str, ...]:
    """The keys :func:`run` can actually execute today."""
    return tuple(entry.key for entry in STRATEGIES if entry.implemented)


def strategy(key: Any) -> Strategy:
    """Return the menu entry for ``key``.

    :param key: a strategy key, or a :class:`Strategy`.
    :raises AnnotationStrategyError: no entry has that key; the message
        lists the ones that do.
    """
    if isinstance(key, Strategy):
        return key
    name = str(key or "").strip()
    for entry in STRATEGIES:
        if entry.key == name:
            return entry
    raise AnnotationStrategyError(
        f"Unknown annotation strategy {key!r}. The menu is: "
        f"{', '.join(strategy_keys())}.")


def menu() -> Tuple[str, ...]:
    """Every entry as one line of prose, for a chooser that shows them."""
    return tuple(entry.describe() for entry in STRATEGIES)


def missing_requirement(key: Any, frame: Optional[pd.DataFrame],
                        score_column: str = DEFAULT_SCORE_COLUMN, *,
                        label_column: str = "",
                        positive_control_wells: Sequence[str] = (),
                        negative_control_wells: Sequence[str] = ()) -> str:
    """Why ``key`` cannot run on this table, or ``""`` when it can.

    ASKED BEFORE THE RUN, NOT AFTER IT. Every refusal in this module is
    raised while a strategy is executing, which is the right place for it
    and the wrong time for a user: choosing "Diversity sampling over
    clusters" on a coefficient table with no measurement columns joined to
    it should say so on the control, not a run later. This is the cheap
    pre-flight that lets a chooser grey itself with the reason -- it reads
    dtypes and one column, never the whole matrix, so it can be asked on
    every change of the menu.

    IT IS THE OPTIMISTIC HALF. An empty answer means nothing this can see
    is missing; :func:`prepare` still reads the values and can still
    refuse -- a score whose cells are all in one well, a feature column
    that turns out not to vary. What it will never do is stay silent about
    a table that has no score, no annotations, no measurements, or no
    control wells named for the strategy that needs them.

    :param key: a strategy key, or a :class:`Strategy`.
    :param frame: the object rows on screen, or None.
    :param score_column: the per-object classification score.
    :param label_column: a column of human annotations, when one is named.
    :param positive_control_wells: the positive control wells named.
    :param negative_control_wells: the negative control wells named.
    :returns: one sentence naming what is missing and what to do about it,
        or "" when the strategy can be run.
    :raises AnnotationStrategyError: ``key`` is not on the menu.
    """
    entry = strategy(key)
    if frame is None or not len(frame):
        return ("There are no cells to choose from: the object table is "
                "empty, so nothing would be selected, fitted or measured.")
    needs = tuple(entry.needs or ())

    # THE REFERENCE LABEL, WHICH EVERY STRATEGY NEEDS. It is what the
    # hold-out is scored against, so a strategy that fits nothing still
    # cannot run without one -- `prepare` refuses the same table for the
    # same reason.
    annotated = usable_annotations(frame, label_column)
    scored = scored_cells(frame, score_column)
    if not annotated and scored < 4:
        named = str(label_column or "").strip()
        if named:
            said = (f"the annotation column {named!r} carries fewer than "
                    "four cells over two classes")
        else:
            said = "no annotation column is named"
        if str(score_column) not in frame.columns:
            has = (f"there is no column {str(score_column)!r} in these "
                   f"{len(frame):,} row(s)")
        else:
            has = (f"only {scored} of {len(frame):,} cell(s) carry a finite "
                   f"{str(score_column)!r}")
        return (f"{entry.title} has nothing to label the cells with: {said} "
                f"and {has}. Every strategy is measured against a reference "
                "label on the hold-out, so name an annotation column that "
                "has one, or load objects that carry a classification "
                "score.")

    if "features" in needs and not candidate_feature_columns(
            frame, score_column):
        return (f"{entry.title} fits a model on the cells' own "
                "measurements, and none of these "
                f"{len(frame.columns):,} column(s) is one: every numeric "
                "column either identifies the row or is the classifier's "
                "own output. Join the measurement tables to these objects, "
                "or choose a strategy that fits nothing.")

    if "controls" in needs:
        empty = [name for name, wells in
                 (("positive", positive_control_wells),
                  ("negative", negative_control_wells))
                 if not [w for w in (wells or ()) if str(w).strip()]]
        if empty:
            return (f"{entry.title} takes its labels from the plate's own "
                    f"control wells and the {' and '.join(empty)} control "
                    "well(s) are not named. Name them below — the labels "
                    "come from the wells, so without them there is nothing "
                    "to fit on.")
    return ""


# ---------------------------------------------------------------------------
# What a run is asked for
# ---------------------------------------------------------------------------

@dataclass
class AnnotationRequest:
    """Everything a strategy needs, and every knob it can be run with.

    :param frame: one row per cell, with the score column, the acquisition
        metadata the split level needs, and whatever measurements are to be
        fitted on.
    :param score_column: the per-object classification score.
    :param feature_columns: the columns to fit on. ``None`` means infer
        them -- see :func:`feature_columns`.
    :param score_inputs: the columns the score is a function of. ``None``
        means infer them -- see :func:`score_input_columns`. Naming them
        explicitly is the exact form of the leakage control; inferring them
        is the approximation offered when the classifier's feature list is
        not to hand.
    :param group_by: the independence level a split may not cross;
        ``'well'``, ``'field'``, ``'plate'`` or ``'cell'``.
    :param wells: the chosen guide wells. A well matches when every token
        of the name given is one of the well's own identity tokens, so
        ``'r1_c1'`` matches the well ``plate1/r1/c1``. Empty means the
        whole screen is eligible, which the run says in a note.
    :param label_column: a column of existing human annotations. When it
        holds two or more classes the run uses them and the score's trap
        does not apply to the hold-out; without it the hold-out labels come
        from the score itself and every report says so.
    :param positive_control_wells: wells whose cells are known positives.
    :param negative_control_wells: wells whose cells are known negatives.
    :param n_positive: how many cells the positive set holds, and the size
        of the matched contrast draw and of every queue.
    :param holdout_fraction: the share of wells held aside at random before
        any strategy chooses anything.
    :param seed: the one seed. Same seed, same selection, same split.
    :param leakage: one of :data:`LEAKAGE_MODES`.
    :param model: ``'auto'``, ``'xgboost'`` or ``'hist_gradient_boosting'``.
    :param measure: the uncertainty measure, from
        :data:`spacr.active_learning.UNCERTAINTY_MEASURES`.
    :param n_clusters: clusters for diversity sampling; 0 means one per
        cell in the budget, which is what makes it a representative each.
    :param n_bins: score strata.
    :param confidence: the probability a self-training round needs before
        it accepts its own prediction as a label.
    :param rounds: the most self-training rounds to run.
    :param neighbours: neighbours per seed for propagation.
    :param distance_quantile: the propagation radius, as a quantile of the
        observed nearest-neighbour distances. The radius it resolves to is
        reported in the result.
    :param distance_cut: the propagation radius outright, in standardised
        feature space, when the caller would rather set the number than
        have a quantile resolve one. Shown in the result either way,
        because a cut nobody can see is a cut that can be widened until the
        labels agree.
    :param correlation_cut: a feature whose absolute rank correlation with
        the score reaches this is treated as one of the score's own inputs.
    """

    frame: pd.DataFrame
    score_column: str = DEFAULT_SCORE_COLUMN
    feature_columns: Optional[Sequence[str]] = None
    score_inputs: Optional[Sequence[str]] = None
    group_by: str = "well"
    wells: Sequence[str] = ()
    label_column: str = ""
    positive_control_wells: Sequence[str] = ()
    negative_control_wells: Sequence[str] = ()
    n_positive: int = 100
    holdout_fraction: float = 0.25
    seed: int = 0
    leakage: str = "report"
    model: str = "auto"
    measure: str = "margin"
    n_clusters: int = 0
    n_bins: int = 10
    confidence: float = 0.9
    rounds: int = 5
    neighbours: int = 5
    distance_quantile: float = 0.1
    distance_cut: Optional[float] = None
    correlation_cut: float = 0.5

    def validated(self) -> "AnnotationRequest":
        """Return self after refusing values no strategy could act on."""
        if self.frame is None or not len(self.frame):
            raise AnnotationStrategyError(
                "There are no cells to annotate: the object table handed to "
                "the strategy is empty.")
        if str(self.leakage) not in LEAKAGE_MODES:
            raise AnnotationStrategyError(
                f"leakage={self.leakage!r} is not one of "
                f"{', '.join(LEAKAGE_MODES)}.")
        if int(self.n_positive) < 2:
            raise AnnotationStrategyError(
                "n_positive must be at least 2: one cell cannot make a class.")
        if not 0.0 < float(self.holdout_fraction) < 1.0:
            raise AnnotationStrategyError(
                "holdout_fraction must be a fraction strictly between 0 and "
                "1; it is the share of WELLS held aside before anything is "
                "chosen.")
        return self


# ---------------------------------------------------------------------------
# Columns
# ---------------------------------------------------------------------------

def _numeric_columns(frame: pd.DataFrame) -> List[str]:
    """Numeric columns of ``frame`` that vary, in table order."""
    out: List[str] = []
    for column in frame.columns:
        values = frame[column]
        if not pd.api.types.is_numeric_dtype(values):
            continue
        if bool(pd.api.types.is_bool_dtype(values)):
            values = values.astype(float)
        finite = np.asarray(values, dtype=float)
        finite = finite[np.isfinite(finite)]
        if finite.size < 2 or float(np.nanstd(finite)) <= 0.0:
            continue
        out.append(str(column))
    return out


def _looks_like_the_score(column: str, score_column: str) -> bool:
    """Whether a column name marks a classifier output rather than a pixel."""
    name = str(column).lower()
    if name == str(score_column).lower():
        return True
    return any(hint in name for hint in SCORE_NAME_HINTS)


def feature_columns(frame: pd.DataFrame,
                    score_column: str = DEFAULT_SCORE_COLUMN,
                    explicit: Optional[Sequence[str]] = None
                    ) -> Tuple[str, ...]:
    """The measurement columns a model may be fitted on.

    Columns that identify a row, columns that are a classifier's output, and
    columns that do not vary are all excluded: a model handed ``rowID`` as a
    number learns the plate layout, and one handed the score learns the
    score.

    :param frame: the object table.
    :param score_column: the score, which is never a feature.
    :param explicit: a caller's own list. Names absent from the frame are
        refused rather than silently dropped, because a feature list that
        quietly shrinks is a model fitted on something other than what was
        asked for.
    :returns: the columns, in table order.
    :raises AnnotationStrategyError: an explicit column is absent, or
        nothing at all is left to fit on.
    """
    if explicit is not None:
        wanted = [str(c) for c in explicit]
        missing = [c for c in wanted if c not in frame.columns]
        if missing:
            raise AnnotationStrategyError(
                f"These feature columns are not in the object table: "
                f"{missing}. It has {len(frame.columns)} columns; the first "
                f"few are {list(frame.columns)[:8]}.")
        if not wanted:
            raise AnnotationStrategyError(
                "An empty feature list was given, so there is nothing to fit "
                "on.")
        return tuple(wanted)
    identity = {c.lower() for c in IDENTITY_COLUMNS}
    out = [c for c in _numeric_columns(frame)
           if c.lower() not in identity
           and not _looks_like_the_score(c, score_column)]
    if not out:
        raise AnnotationStrategyError(
            "No measurement column survives: every numeric column in the "
            "object table either identifies the row, is the classifier's own "
            "output, or does not vary. Join the measurement tables to these "
            "rows, or name the feature columns explicitly.")
    return tuple(out)


def candidate_feature_columns(frame: pd.DataFrame,
                              score_column: str = DEFAULT_SCORE_COLUMN
                              ) -> Tuple[str, ...]:
    """The columns that COULD be features, judged on names and dtypes alone.

    :func:`feature_columns` is the authority and it reads every value: a
    numeric column that does not vary is not a feature, and only the values
    say so. This is the cheap half of that question -- one pass over
    ``frame.dtypes`` rather than over the rows -- for a caller that has to
    answer "can this strategy run at all" every time a chooser changes, on
    a table that may hold half a million objects.

    It is therefore the OPTIMISTIC answer: a table this accepts can still
    be refused by :func:`feature_columns`, and the run is what refuses it.
    A table this rejects has no measurement column under any reading.

    :param frame: the object table.
    :param score_column: the score, which is never a feature.
    :returns: the candidate columns, in table order.
    """
    identity = {c.lower() for c in IDENTITY_COLUMNS}
    return tuple(str(name) for name, dtype in frame.dtypes.items()
                 if pd.api.types.is_numeric_dtype(dtype)
                 and str(name).lower() not in identity
                 and not _looks_like_the_score(str(name), score_column))


def usable_annotations(frame: pd.DataFrame, label_column: str) -> int:
    """How many cells carry an annotation in ``label_column``.

    Zero when the column is absent, empty, or holds one class only --
    which is the same answer :func:`_reference_labels` gives, so a chooser
    and a run cannot disagree about whether there are labels to fit on.

    :param frame: the object table.
    :param label_column: the annotation column, or "".
    :returns: the number of annotated cells, or 0 when they are unusable.
    """
    column = str(label_column or "").strip()
    if not column or column not in frame.columns:
        return 0
    raw = frame[column]
    known = raw.notna() & (raw.astype(str).str.strip() != "")
    if int(known.sum()) < 4:
        return 0
    if len(pd.unique(raw[known])) < 2:
        return 0
    return int(known.sum())


def scored_cells(frame: pd.DataFrame,
                 score_column: str = DEFAULT_SCORE_COLUMN) -> int:
    """How many cells carry a finite value in the score column."""
    column = str(score_column or "")
    if column not in frame.columns:
        return 0
    values = pd.to_numeric(frame[column],
                           errors="coerce").to_numpy(dtype=float)
    return int(np.isfinite(values).sum())


def score_input_columns(frame: pd.DataFrame,
                        score_column: str = DEFAULT_SCORE_COLUMN,
                        features: Optional[Sequence[str]] = None,
                        explicit: Optional[Sequence[str]] = None,
                        correlation_cut: float = 0.5) -> Tuple[str, ...]:
    """The columns the score is a function of, as far as this can be known.

    Two rules, and the second is an approximation the caller can replace:

    * anything whose NAME marks it as a classifier output -- the score
      itself, probabilities, logits, other prediction columns;
    * any feature whose absolute Spearman correlation with the score
      reaches ``correlation_cut``. When the classifier's own feature list
      is available, pass it as ``explicit`` instead: that is the exact
      answer and this is the stand-in for it.

    :param frame: the object table.
    :param score_column: the score.
    :param features: the candidate feature columns.
    :param explicit: the classifier's own inputs, when they are known.
    :param correlation_cut: the absolute rank correlation at which a
        feature counts as one of the score's inputs.
    :returns: the columns, in table order, including the score itself.
    """
    if explicit is not None:
        named = [str(c) for c in explicit if str(c) in frame.columns]
        if str(score_column) in frame.columns:
            named.append(str(score_column))
        return tuple(dict.fromkeys(named))
    candidates = list(features) if features is not None \
        else list(frame.columns)
    out = [str(c) for c in candidates
           if _looks_like_the_score(str(c), score_column)]
    if str(score_column) in frame.columns:
        out.append(str(score_column))
    if str(score_column) in frame.columns:
        score = pd.to_numeric(frame[score_column], errors="coerce")
        cut = abs(float(correlation_cut))
        for column in candidates:
            name = str(column)
            if name in out:
                continue
            values = pd.to_numeric(frame[name], errors="coerce")
            usable = score.notna() & values.notna()
            if int(usable.sum()) < 3:
                continue
            correlation = values[usable].corr(score[usable], method="spearman")
            if correlation is not None and np.isfinite(correlation) \
                    and abs(float(correlation)) >= cut:
                out.append(name)
    order = {str(c): i for i, c in enumerate(frame.columns)}
    return tuple(sorted(dict.fromkeys(out),
                        key=lambda c: order.get(c, 1 << 30)))


def feature_views(columns: Sequence[str]) -> Dict[str, Tuple[str, ...]]:
    """Split ``columns`` into the intensity view and the shape view.

    The families come from :func:`spacr.column_groups.classify`, so the
    taxonomy is the package's one taxonomy rather than a second guess at
    what a column name means. Columns in neither family are dealt
    alternately between the two, so a table whose names the taxonomy does
    not recognise still yields two views rather than one empty one -- and
    the two-view strategy reports how many columns each view got, which is
    where a table like that shows up.

    :param columns: the feature columns.
    :returns: ``{'intensity': (...), 'shape': (...)}``.
    """
    names = [str(c) for c in columns]
    try:
        from .column_groups import classify

        families = classify(names).get("family", {})
    except Exception:
        families = {}
    intensity = [c for family in INTENSITY_FAMILIES
                 for c in families.get(family, [])]
    shape = [c for family in SHAPE_FAMILIES
             for c in families.get(family, [])]
    placed = set(intensity) | set(shape)
    for index, column in enumerate(c for c in names if c not in placed):
        (intensity if (index % 2 == 0) else shape).append(column)
    order = {c: i for i, c in enumerate(names)}
    return {
        "intensity": tuple(sorted(set(intensity), key=order.get)),
        "shape": tuple(sorted(set(shape), key=order.get)),
    }


# ---------------------------------------------------------------------------
# Wells, labels, and the hold-out no strategy may choose from
# ---------------------------------------------------------------------------

#: What the splitter joins a group's identity parts with. It is a unit
#: separator, which prints as nothing at all -- so a well written straight
#: out reads as "plate1r1c1".
GROUP_SEPARATOR = "\x1f"


def readable_group(value: Any) -> str:
    """One group id as a person reads it: ``plate1/r1/c1``.

    :param value: a group id from the splitter.
    :returns: the same identity with its parts separated visibly.
    """
    return str(value).replace(GROUP_SEPARATOR, "/")


def _identity_tokens(value: Any) -> frozenset:
    """The lower-case identity tokens of one group id or well name."""
    text = str(value)
    parts = [piece for chunk in text.split(GROUP_SEPARATOR)
             for piece in chunk.split("_")]
    return frozenset(piece.strip().lower() for piece in parts if piece.strip())


def wells_selected(groups: Sequence[Any],
                   wanted: Sequence[str]) -> np.ndarray:
    """A boolean mask over rows whose group is one of ``wanted``.

    A well matches when every token of the name given is one of the group's
    own tokens, so ``'r1_c1'`` matches ``plate1/r1/c1`` and
    ``'plate2_r1_c1'`` does not. Matching on tokens rather than on the
    whole string is what lets a user name a well the way the plate map
    names it without knowing how the split builds its group ids.

    :param groups: one group id per row.
    :param wanted: the well names chosen.
    :returns: the mask; all-True when nothing was named.
    """
    values = np.asarray(groups, dtype=object)
    names = [str(w).strip() for w in wanted if str(w).strip()]
    if not names:
        return np.ones(len(values), dtype=bool)
    tokens = [t for t in (_identity_tokens(name) for name in names) if t]
    if not tokens:
        return np.zeros(len(values), dtype=bool)
    # ONE ANSWER PER DISTINCT GROUP, not per cell: a screen has half a
    # million objects and a few hundred wells.
    as_text = values.astype(str)
    matched = {group for group in set(as_text.tolist())
               if any(t <= _identity_tokens(group) for t in tokens)}
    return np.isin(as_text, list(matched))


@dataclass(frozen=True)
class Prepared:
    """The one setup every strategy shares, built once per run.

    :ivar frame: the object table, unchanged.
    :ivar groups: one independence-group id per row.
    :ivar level: the level those ids are at.
    :ivar features: the columns a model may be fitted on.
    :ivar score_inputs: the columns the score is a function of.
    :ivar honest_features: ``features`` with ``score_inputs`` removed.
    :ivar labels: the reference label per row, 0 or 1. Meaningless where
        ``known`` is False.
    :ivar known: which rows carry a usable reference label. Every row in
        score mode; only the annotated rows when a label column is used,
        because calling an unannotated cell negative is the bias half this
        module exists to avoid.
    :ivar label_source: where those labels came from, in words.
    :ivar threshold: the score at which the reference label turns 1, or
        NaN when the labels are human annotations.
    :ivar holdout: positional indices of the hold-out rows -- every cell of
        the wells drawn at random before anything was chosen.
    :ivar selectable: positional indices no strategy is forbidden.
    :ivar chosen: positional indices inside the chosen guide wells and
        outside the hold-out; the pool the positive set comes from.
    :ivar split: the splitter's own provenance for the hold-out.
    :ivar notes: what the setup had to decide for itself.
    """

    frame: pd.DataFrame
    groups: np.ndarray
    level: str
    features: Tuple[str, ...]
    score_inputs: Tuple[str, ...]
    honest_features: Tuple[str, ...]
    labels: np.ndarray
    known: np.ndarray
    label_source: str
    threshold: float
    holdout: np.ndarray
    selectable: np.ndarray
    chosen: np.ndarray
    split: Any
    notes: Tuple[str, ...] = ()

    @property
    def annotated(self) -> bool:
        """True when the labels are human annotations, not the score."""
        return not np.isfinite(self.threshold)

    def holdout_labels(self) -> np.ndarray:
        """The reference labels of the hold-out rows."""
        return self.labels[self.holdout]

    def labelled(self) -> np.ndarray:
        """Selectable rows that carry a usable reference label."""
        return np.asarray([p for p in self.selectable if self.known[p]],
                          dtype=int)

    def positive_share(self, positions: Sequence[int]) -> float:
        """The share of ``positions`` the reference label calls positive."""
        index = np.asarray(list(positions), dtype=int)
        if index.size == 0:
            return float("nan")
        return float(np.mean(self.labels[index] == 1))


def _reference_labels(frame: pd.DataFrame, request: AnnotationRequest,
                      chosen_mask: np.ndarray
                      ) -> Tuple[np.ndarray, np.ndarray, str, float,
                                 List[str]]:
    """Build one binary reference label per row, and say where it came from.

    Human annotations win when there are any: a label somebody looked at a
    cell to write is not the score, so a model measured against it is not
    measured against its own input. Without them the label is a cut on the
    score -- which is the method as it is usually described, and the reason
    every fit here is also reported with the score's own inputs removed.
    """
    notes: List[str] = []
    column = str(request.label_column or "").strip()
    if column:
        if column not in frame.columns:
            raise AnnotationStrategyError(
                f"label_column={column!r} is not in the object table.")
        raw = frame[column]
        known = raw.notna() & (raw.astype(str).str.strip() != "")
        classes = pd.unique(raw[known])
        if len(classes) >= 2 and int(known.sum()) >= 4:
            positive = sorted(str(c) for c in classes)[-1]
            labels = np.where(
                known.to_numpy() & (raw.astype(str).to_numpy() == positive),
                1, 0).astype(int)
            notes.append(
                f"Labels are the annotations in {column!r}: "
                f"{int(known.sum()):,} of {len(frame):,} cells carry one, and "
                f"{positive!r} is the positive class. The hold-out is scored "
                "against those annotations, so the score's own trap does not "
                "reach it.")
            return (labels, known.to_numpy(dtype=bool),
                    f"annotations in {column!r}", float("nan"), notes)
        notes.append(
            f"{column!r} carries fewer than two usable classes, so the "
            "reference label falls back to a cut on the score.")

    score = str(request.score_column)
    if score not in frame.columns:
        raise AnnotationStrategyError(
            f"The object table has no score column {score!r}, and no usable "
            "annotations, so there is nothing to define a phenotype with.")
    values = pd.to_numeric(frame[score], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(values).any():
        raise AnnotationStrategyError(
            f"Every value of {score!r} is missing, so no cell has a score to "
            "select on.")
    pool = values[chosen_mask & np.isfinite(values)]
    if pool.size < 4:
        raise AnnotationStrategyError(
            f"Only {pool.size} scored cell(s) are in the chosen wells, which "
            "is too few to define a top-scoring set.")
    wanted = min(int(request.n_positive), max(1, pool.size - 1))
    quantile = 1.0 - (wanted / float(pool.size))
    quantile = float(min(max(quantile, 0.0), 1.0))
    threshold = float(np.quantile(pool, quantile))
    labels = (np.isfinite(values) & (values >= threshold)).astype(int)
    if len(np.unique(labels)) < 2:
        raise AnnotationStrategyError(
            f"A cut of the score at {threshold:.6g} puts every cell on one "
            "side, so there is no contrast to fit. The scores in the chosen "
            f"wells run from {float(np.nanmin(pool)):.6g} to "
            f"{float(np.nanmax(pool)):.6g}.")
    notes.append(
        f"Labels are a cut on {score!r} at {threshold:.6g} — the score of "
        f"the {wanted:,}th highest cell in the chosen wells. THE SCORE IS "
        "THEREFORE BOTH THE LABEL AND A FEATURE: read the fit without the "
        "score's own inputs, not the one with them.")
    return (labels, np.ones(len(frame), dtype=bool), f"a cut on {score!r}",
            threshold, notes)


def prepare(request: AnnotationRequest,
            entry: Optional[Strategy] = None) -> Prepared:
    """Resolve columns, group ids, labels and the hold-out, once.

    The hold-out is drawn FIRST, by whole groups, and no strategy may
    select from it. That ordering is the whole of the second rule: a
    strategy that could choose from the cells it is later measured on
    would be marking its own work.

    A TABLE WITH NO MEASUREMENT COLUMNS IS NOT THE END OF THE MENU. Two
    entries fit nothing -- the score strata and the plain random draw --
    and the plain random draw is the one the ninth rule says must be
    available whatever else is chosen. Demanding a feature matrix from
    them would take the only unbiased measurement off a coefficient table
    that has no measurements joined to it, which is the table most runs
    have. So the feature columns are required of the strategies that
    declare they fit on them, and their absence is a note for the ones
    that do not.

    :param request: what to run.
    :param entry: the menu entry about to be run, when it is known. It is
        read only for what the strategy needs; ``None`` requires
        everything, which is the strict answer a caller with no entry to
        hand should get.
    :returns: the shared setup.
    :raises AnnotationStrategyError: the table cannot support a
        leakage-safe split, or has no features, labels or score.
    """
    from .classifier_evaluation import grouped_split, split_group_values

    request = request.validated()
    frame = request.frame
    try:
        level, groups = split_group_values(group_by=request.group_by,
                                           frame=frame,
                                           table="the object table")
    except ValueError as refusal:
        raise AnnotationStrategyError(str(refusal)) from refusal
    groups = np.asarray(groups, dtype=object)
    chosen_mask = wells_selected(groups, request.wells)
    if not chosen_mask.any():
        raise AnnotationStrategyError(
            f"None of the wells {list(request.wells)} appear in the object "
            f"table. It holds {len(set(map(str, groups)))} {level}(s); the "
            f"first few are "
            f"{sorted({readable_group(g) for g in groups})[:6]}.")

    labels, known, source, threshold, notes = _reference_labels(
        frame, request, chosen_mask)
    if not list(request.wells):
        notes.append(
            "No guide wells were named, so every well is eligible and the "
            "positive set is the top-scoring cells of the whole screen.")

    fits_on_features = entry is None or "features" in tuple(entry.needs or ())
    try:
        features = feature_columns(frame, request.score_column,
                                   request.feature_columns)
    except AnnotationStrategyError:
        if fits_on_features:
            raise
        features = ()
        notes.append(
            f"{entry.title} fits no model, and there is no measurement "
            "column in this table, so it runs on the score and the well "
            "layout alone. Nothing is fitted and nothing is predicted; what "
            "it reports is the selection and what it did to the class "
            "balance.")
    inputs = score_input_columns(frame, request.score_column, features,
                                 request.score_inputs,
                                 request.correlation_cut)
    honest = tuple(c for c in features if c not in set(inputs))
    if features and not honest:
        notes.append(
            "EVERY feature column is one of the score's own inputs, so there "
            "is no fit left once they are removed. What is reported is the "
            "leaking fit and nothing else; join measurement columns to these "
            "rows before believing it.")

    # THE SPLIT RUNS OVER THE LABELLED ROWS ONLY. Stratifying over rows that
    # carry no annotation would balance the hold-out on a label nobody
    # wrote, and the wells it drew would be drawn for the wrong reason.
    labelled = np.flatnonzero(np.asarray(known, dtype=bool))
    if labelled.size < 4:
        raise AnnotationStrategyError(
            f"Only {labelled.size} cell(s) carry a reference label, which is "
            "too few to hold any of them aside and still measure anything.")
    # THE SPLITTER REFUSES A DESIGN IT CANNOT MAKE HONEST, and its refusal
    # is the message a user needs; it is re-raised in this module's own type
    # so a caller has one exception to catch.
    try:
        _, held, split = grouped_split(
            groups[labelled], labels[labelled],
            float(request.holdout_fraction), seed=int(request.seed),
            group_by=level)
    except AnnotationStrategyError:
        raise
    except ValueError as refusal:
        raise AnnotationStrategyError(str(refusal)) from refusal
    holdout = np.sort(labelled[np.asarray(held, dtype=int)])
    holdout_groups = {str(g) for g in groups[holdout]}
    # WHOLE WELLS, not whole labels. A strategy that could pick an
    # unannotated cell out of a hold-out well would be choosing inside the
    # group its own score is measured on.
    outside = ~np.isin(groups.astype(str), list(holdout_groups))
    selectable = np.flatnonzero(outside)
    chosen = np.flatnonzero(outside & chosen_mask)
    if chosen.size < 2:
        raise AnnotationStrategyError(
            "The chosen wells have fewer than two cells left once the "
            "hold-out wells are taken out, so nothing can be selected from "
            "them. Name more wells, or lower holdout_fraction.")
    if len(np.unique(labels[holdout])) < 2:
        raise AnnotationStrategyError(
            f"The hold-out drew {len(holdout_groups)} {level}(s) whose "
            "labelled cells are all of one class, so no accuracy measured "
            "there would mean anything. Change the seed, or raise "
            "holdout_fraction so more groups are held aside.")
    notes.append(
        f"Held aside first, and no strategy may select from it. The held-out "
        f"{level}(s) are "
        f"{', '.join(sorted(readable_group(g) for g in holdout_groups)[:6])}"
        f"{', …' if len(holdout_groups) > 6 else ''}. "
        f"{split.summary()}")
    if features:
        notes.append(
            f"{len(honest):,} of {len(features):,} feature column(s) survive "
            f"removing the score's own inputs "
            f"({', '.join(inputs[:6]) or 'none found'}"
            f"{', …' if len(inputs) > 6 else ''}).")
    return Prepared(
        frame=frame, groups=groups, level=level,
        features=tuple(features), score_inputs=tuple(inputs),
        honest_features=honest, labels=np.asarray(labels, dtype=int),
        known=np.asarray(known, dtype=bool),
        label_source=source, threshold=float(threshold), holdout=holdout,
        selectable=selectable, chosen=chosen, split=split,
        notes=tuple(notes))


# ---------------------------------------------------------------------------
# The model, and what a fit is allowed to claim
# ---------------------------------------------------------------------------

def xgboost_available() -> bool:
    """Whether the boosted-tree package the named method calls for is here."""
    from importlib.util import find_spec

    try:
        return find_spec("xgboost") is not None
    except (ImportError, ValueError):
        return False


def _estimator(model: str, seed: int) -> Tuple[Any, str]:
    """Build the classifier, falling back when the first choice is absent.

    ``'auto'`` prefers XGBoost, which is what the named method asks for,
    and falls back to scikit-learn's histogram gradient boosting -- the
    same family of model, already a dependency -- when XGBoost is not
    installed. The name that comes back is written into every report, so a
    run that fell back says which model produced its numbers.

    :param model: ``'auto'``, ``'xgboost'`` or ``'hist_gradient_boosting'``.
    :param seed: the run's seed.
    :returns: ``(estimator, name)``.
    :raises AnnotationStrategyError: ``'xgboost'` was demanded and is absent.
    """
    kind = str(model or "auto").strip().lower()
    if kind in ("auto", "xgboost") and xgboost_available():
        from xgboost import XGBClassifier

        return XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.1,
            subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
            tree_method="hist", eval_metric="logloss", n_jobs=1,
            random_state=int(seed)), "xgboost"
    if kind == "xgboost":
        raise AnnotationStrategyError(
            "model='xgboost' was asked for and the xgboost package is not "
            "installed. Install it, or use model='auto', which falls back to "
            "scikit-learn's hist_gradient_boosting.")
    from sklearn.ensemble import HistGradientBoostingClassifier

    return HistGradientBoostingClassifier(
        max_iter=200, max_depth=4, learning_rate=0.1,
        random_state=int(seed)), "hist_gradient_boosting"


def _matrix(frame: pd.DataFrame, columns: Sequence[str]) -> np.ndarray:
    """The feature matrix, as float, missing values left as NaN.

    Both estimators handle NaN natively and both do it better than a mean
    that invents a cell in the middle of the distribution.
    """
    if not len(columns):
        return np.empty((len(frame), 0), dtype=float)
    return frame.loc[:, list(columns)].apply(
        pd.to_numeric, errors="coerce").to_numpy(dtype=float)


def _standardised(frame: pd.DataFrame,
                  columns: Sequence[str]) -> np.ndarray:
    """The feature matrix centred, scaled, and with missing values filled.

    Distances and clusters need a complete matrix and need every column to
    weigh the same; a column left in its own units dominates the metric for
    no reason but its scale.
    """
    values = _matrix(frame, columns)
    if values.size == 0:
        return values
    centre = np.nanmedian(values, axis=0)
    centre = np.where(np.isfinite(centre), centre, 0.0)
    filled = np.where(np.isfinite(values), values, centre)
    spread = np.nanstd(filled, axis=0)
    spread = np.where(np.isfinite(spread) & (spread > 0), spread, 1.0)
    return (filled - centre) / spread


@dataclass(frozen=True)
class FitReport:
    """One fit, and the hold-out number it is allowed to claim.

    :ivar model: which estimator produced it.
    :ivar features: the columns it was fitted on.
    :ivar n_train: cells fitted on.
    :ivar n_test: hold-out cells scored on.
    :ivar accuracy: hold-out accuracy.
    :ivar balanced_accuracy: hold-out accuracy averaged over the classes,
        which is the number to read when the classes are uneven -- and on
        a screen they always are.
    :ivar roc_auc: hold-out area under the ROC curve, or None when the
        hold-out holds one class.
    :ivar positive_share_train: share of the training rows labelled
        positive.
    :ivar positive_share_test: share of the hold-out labelled positive.
    :ivar label_source: where the labels came from, in words.
    :ivar split_summary: the splitter's own account of the hold-out.
    """

    model: str
    features: Tuple[str, ...]
    n_train: int
    n_test: int
    accuracy: float
    balanced_accuracy: float
    roc_auc: Optional[float]
    positive_share_train: float
    positive_share_test: float
    label_source: str
    split_summary: str

    @property
    def lift(self) -> float:
        """How far above chance the fit is, on the balanced scale.

        Balanced accuracy has chance at 0.5 whatever the class balance, and
        so does the area under the ROC curve, so both give a lift that can
        be compared between two fits on different columns.
        """
        base = self.roc_auc if self.roc_auc is not None \
            else self.balanced_accuracy
        return float(base) - 0.5

    def summary(self) -> str:
        """The fit in one line, hold-out numbers only."""
        auc = ("n/a" if self.roc_auc is None else f"{self.roc_auc:.3f}")
        return (
            f"{self.model} on {len(self.features)} column(s): fitted on "
            f"{self.n_train:,} cell(s), scored on {self.n_test:,} held-out "
            f"cell(s) — balanced accuracy {self.balanced_accuracy:.3f}, "
            f"accuracy {self.accuracy:.3f}, ROC AUC {auc}. Labels: "
            f"{self.label_source}.")


@dataclass(frozen=True)
class LeakageReport:
    """The same selection fitted with and without the score's own inputs.

    :ivar mode: the leakage mode the run was asked for.
    :ivar dropped: the columns removed to make the honest fit.
    :ivar with_score_inputs: the fit that keeps them, or None.
    :ivar without_score_inputs: the fit that removes them, or None.
    :ivar survival: the share of the leaking fit's lift over chance that
        the honest fit keeps. None when either fit is missing or the
        leaking fit was itself at chance, where the ratio means nothing.
    """

    mode: str
    dropped: Tuple[str, ...]
    with_score_inputs: Optional[FitReport] = None
    without_score_inputs: Optional[FitReport] = None
    survival: Optional[float] = None

    def summary(self) -> str:
        """Both numbers and the one sentence that compares them."""
        lines: List[str] = []
        if self.with_score_inputs is not None:
            lines.append("WITH the score's own inputs: "
                         + self.with_score_inputs.summary())
        if self.without_score_inputs is not None:
            lines.append("WITHOUT them (" + (
                ", ".join(self.dropped[:6]) or "nothing to drop")
                + ("…" if len(self.dropped) > 6 else "") + "): "
                + self.without_score_inputs.summary())
        if not lines:
            return (
                f"Nothing was fitted: leakage={self.mode!r} asked for the fit "
                "without the score's own inputs, and every feature column is "
                "one of them. Join measurement columns to these rows, or name "
                "the score's inputs explicitly.")
        if self.survival is not None:
            lines.append(
                f"{self.survival:.0%} of the fit's lift over chance survives "
                "removing the score's own inputs. A number near zero means "
                "the model learned the score and no morphology.")
        elif self.with_score_inputs is not None \
                and self.without_score_inputs is not None:
            lines.append(
                "The fit WITH the score's inputs is already at chance, so "
                "there is no lift for the honest fit to keep a share of.")
        return "\n".join(lines)


def _score_holdout(prepared: Prepared, probabilities: Any, n_train: int,
                   columns: Sequence[str], model: str) -> FitReport:
    """Turn hold-out probabilities into the one report a fit may claim.

    Separate from the fitting so an estimator whose probabilities are
    transformed after the fact -- the positive-unlabelled rescaling is one
    -- is scored by exactly the same code as an ordinary one.

    THE HOLD-OUT LABELS ARE ALWAYS THE REFERENCE LABELS. A strategy chooses
    what it trains on; it does not get to choose what it is measured
    against.

    :param prepared: the shared setup.
    :param probabilities: one positive-class probability per hold-out row.
    :param n_train: how many cells the fit was trained on.
    :param columns: the columns it was trained with.
    :param model: the estimator's name.
    :returns: the hold-out report.
    """
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                                 roc_auc_score)

    values = np.asarray(probabilities, dtype=float).reshape(-1)
    y_test = prepared.labels[prepared.holdout]
    predicted = (values >= 0.5).astype(int)
    try:
        auc = float(roc_auc_score(y_test, values))
    except ValueError:
        auc = None
    return FitReport(
        model=str(model), features=tuple(str(c) for c in columns),
        n_train=int(n_train), n_test=int(len(y_test)),
        accuracy=float(accuracy_score(y_test, predicted)),
        balanced_accuracy=float(balanced_accuracy_score(y_test, predicted)),
        roc_auc=auc,
        positive_share_train=float("nan"),
        positive_share_test=float(np.mean(y_test == 1)),
        label_source=prepared.label_source,
        split_summary=prepared.split.summary())


def _fit_report(prepared: Prepared, train: Sequence[int],
                columns: Sequence[str], request: AnnotationRequest,
                train_labels: Optional[np.ndarray] = None
                ) -> Tuple[FitReport, Any, np.ndarray]:
    """Fit on ``train`` and score on the hold-out. Never the other way.

    :param prepared: the shared setup.
    :param train: positional indices of the cells to fit on.
    :param columns: the feature columns to fit with.
    :param request: the run, for the estimator and the seed.
    :param train_labels: a full-length label array read only at the
        training rows. A strategy that invents its own labels -- top-score
        against a random draw, a control well's identity -- passes them
        here; the hold-out is still scored against the reference labels,
        which is the rule that stops a strategy marking its own work.
    :returns: ``(report, fitted estimator, hold-out probabilities)``.
    :raises NotEnoughLabels: the training rows carry one class, or none.
    """
    y = prepared.labels if train_labels is None \
        else np.asarray(train_labels, dtype=int)
    train = np.asarray(list(train), dtype=int)
    if train.size < 2:
        raise NotEnoughLabels(
            "Fewer than two cells were selected, so there is nothing to fit.")
    y_train = y[train]
    if len(np.unique(y_train)) < 2:
        raise NotEnoughLabels(
            f"Every one of the {train.size:,} selected cells carries the same "
            "label, so a classifier fitted on them would have nothing to "
            "separate.")
    if not len(columns):
        raise NotEnoughLabels(
            "There is no feature column left to fit on.")
    estimator, name = _estimator(request.model, request.seed)
    x_train = _matrix(prepared.frame.iloc[train], columns)
    estimator.fit(x_train, y_train)
    x_test = _matrix(prepared.frame.iloc[prepared.holdout], columns)
    probabilities = np.asarray(estimator.predict_proba(x_test),
                               dtype=float)[:, -1]
    scored = _score_holdout(prepared, probabilities, train.size, columns, name)
    report = FitReport(
        model=scored.model, features=scored.features, n_train=scored.n_train,
        n_test=scored.n_test, accuracy=scored.accuracy,
        balanced_accuracy=scored.balanced_accuracy, roc_auc=scored.roc_auc,
        positive_share_train=float(np.mean(y_train == 1)),
        positive_share_test=scored.positive_share_test,
        label_source=scored.label_source,
        split_summary=scored.split_summary)
    return report, estimator, probabilities


def _leakage_report(prepared: Prepared, train: Sequence[int],
                    request: AnnotationRequest,
                    train_labels: Optional[np.ndarray] = None
                    ) -> Tuple[LeakageReport, Optional[Any], Tuple[str, ...]]:
    """Run the fit the leakage mode asks for, and compare where both ran.

    :param prepared: the shared setup.
    :param train: positional indices of the cells to fit on.
    :param request: the run, for the leakage mode, estimator and seed.
    :param train_labels: the labels the strategy invented, read only at
        the training rows.
    :returns: ``(report, the estimator whose predictions are applied, the
        columns that estimator was fitted on)``. The applied estimator is
        the honest one wherever there is one, because a prediction handed
        to a user should not be the one the score wrote for itself.
    """
    mode = str(request.leakage)
    dropped = tuple(c for c in prepared.features
                    if c in set(prepared.score_inputs))
    leaking = honest = None
    leaking_model = honest_model = None
    if mode in ("report", "keep"):
        leaking, leaking_model, _ = _fit_report(prepared, train,
                                                prepared.features, request,
                                                train_labels)
    if mode in ("report", "drop") and prepared.honest_features:
        honest, honest_model, _ = _fit_report(prepared, train,
                                              prepared.honest_features,
                                              request, train_labels)
    survival = None
    if leaking is not None and honest is not None and leaking.lift > 1e-9:
        survival = float(honest.lift / leaking.lift)
    report = LeakageReport(mode=mode, dropped=dropped,
                           with_score_inputs=leaking,
                           without_score_inputs=honest, survival=survival)
    if honest_model is not None:
        return report, honest_model, prepared.honest_features
    return report, leaking_model, prepared.features


# ---------------------------------------------------------------------------
# What a run produces
# ---------------------------------------------------------------------------

#: The roles a selected cell can carry, and what each one means.
ROLES: Dict[str, str] = {
    "positive": "chosen as a positive example and fitted on",
    "contrast": "drawn at random as the contrast set and fitted on",
    "unlabelled": "the contrast set, treated as UNLABELLED rather than "
                  "negative",
    "anchor_positive": "a cell of a positive control well",
    "anchor_negative": "a cell of a negative control well",
    "queue": "queued for a person to annotate; nothing is fitted on it yet",
    "propagated": "given a neighbour's label inside the distance cut",
    "pseudo": "labelled by the model itself in a self-training round",
    "holdout": "held aside at random before anything was chosen; never "
               "selected from, and every reported number is measured here",
}


@dataclass(frozen=True)
class AnnotationResult:
    """What one strategy chose, what it fitted, and what it may claim.

    :ivar strategy: the key that produced it.
    :ivar title: that strategy's name on screen.
    :ivar selection: one row per chosen cell, indexed as the object table
        is, with ``annotation_role`` naming why it was chosen.
    :ivar holdout: the hold-out rows, same shape.
    :ivar predictions: the fitted model applied to every cell it was not
        fitted on -- the rest of the screen and the rest of the chosen
        wells -- or None when the strategy fitted nothing.
    :ivar fit: the headline hold-out report, or None.
    :ivar leakage: the same selection with and without the score's own
        inputs, or None when nothing was fitted.
    :ivar notes: everything the run had to decide, and everything a reader
        has to know before believing the numbers.
    :ivar counts: the arithmetic, as data.
    """

    strategy: str
    title: str
    selection: pd.DataFrame
    holdout: pd.DataFrame
    predictions: Optional[pd.DataFrame] = None
    fit: Optional[FitReport] = None
    leakage: Optional[LeakageReport] = None
    notes: Tuple[str, ...] = ()
    counts: Mapping[str, Any] = field(default_factory=dict)

    def role_counts(self) -> Dict[str, int]:
        """How many cells each role holds, hold-out included."""
        out: Dict[str, int] = {}
        for frame in (self.selection, self.holdout):
            if frame is None or not len(frame):
                continue
            for role, n in frame["annotation_role"].value_counts().items():
                out[str(role)] = out.get(str(role), 0) + int(n)
        return out

    def summary(self) -> str:
        """The whole run in prose: what was chosen, fitted, and measured."""
        lines = [self.title, ""]
        roles = self.role_counts()
        if roles:
            lines.append("Chosen: " + ", ".join(
                f"{n:,} {role}" for role, n in sorted(roles.items())))
        if self.predictions is not None:
            lines.append(
                f"Applied to {len(self.predictions):,} cell(s) it was not "
                "fitted on — the rest of the screen and the rest of the "
                "chosen wells.")
        if self.leakage is not None:
            lines.append("")
            lines.append(self.leakage.summary())
        elif self.fit is not None:
            lines.append("")
            lines.append(self.fit.summary())
        if self.notes:
            lines.append("")
            lines.extend(f"• {note}" for note in self.notes)
        return "\n".join(lines)

    def write(self, folder: str) -> Dict[str, str]:
        """Write the selection, the hold-out, the predictions and the prose.

        :param folder: the directory to write into; created when absent.
        :returns: ``{what: path}`` for every file written.
        """
        os.makedirs(folder, exist_ok=True)
        written: Dict[str, str] = {}
        pieces = (("selection", self.selection),
                  ("holdout", self.holdout),
                  ("predictions", self.predictions))
        for name, frame in pieces:
            if frame is None or not len(frame):
                continue
            path = os.path.join(folder, f"annotation_{name}.csv")
            frame.to_csv(path, index=True)
            written[name] = path
        path = os.path.join(folder, "annotation_report.txt")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(self.summary() + "\n")
        written["report"] = path
        return written


def _selection_frame(prepared: Prepared,
                     roles: Mapping[str, Sequence[int]],
                     extra: Optional[Mapping[str, Mapping[int, Any]]] = None
                     ) -> pd.DataFrame:
    """One row per chosen cell, carrying why it was chosen.

    :param prepared: the shared setup.
    :param roles: ``{role: positional indices}``.
    :param extra: ``{column: {positional index: value}}`` for a strategy
        that has something more to say about each cell it picked.
    :returns: a frame indexed as the object table, in table order.
    """
    positions: List[int] = []
    names: List[str] = []
    for role, values in roles.items():
        for position in np.asarray(list(values), dtype=int):
            positions.append(int(position))
            names.append(str(role))
    if not positions:
        return pd.DataFrame(
            columns=["annotation_role", "annotation_group",
                     "annotation_reference"])
    order = np.argsort(np.asarray(positions, dtype=int), kind="stable")
    positions = [positions[i] for i in order]
    names = [names[i] for i in order]
    frame = pd.DataFrame(
        {
            "annotation_role": names,
            "annotation_group": [readable_group(prepared.groups[p])
                                 for p in positions],
            "annotation_reference": prepared.labels[positions],
        },
        index=prepared.frame.index[positions])
    for column in ("prcfo", "prc", "png_path"):
        if column in prepared.frame.columns:
            frame[column] = prepared.frame[column].to_numpy()[positions]
    for column, values in (extra or {}).items():
        frame[column] = [values.get(int(p)) for p in positions]
    return frame


def _apply_model(prepared: Prepared, estimator: Any,
                 columns: Sequence[str],
                 fitted_on: Sequence[int]) -> Optional[pd.DataFrame]:
    """The fitted model's answer for every cell it was not fitted on.

    The rest of the screen AND the rest of the chosen wells, which is what
    the named method asks for; the hold-out is in there too and is marked,
    so a reader can tell the measured cells from the predicted ones.
    """
    if estimator is None or not len(columns):
        return None
    keep = np.ones(len(prepared.frame), dtype=bool)
    keep[np.asarray(list(fitted_on), dtype=int)] = False
    rest = np.flatnonzero(keep)
    if rest.size == 0:
        return None
    matrix = _matrix(prepared.frame.iloc[rest], columns)
    probabilities = np.asarray(estimator.predict_proba(matrix),
                               dtype=float)[:, -1]
    in_holdout = np.zeros(len(prepared.frame), dtype=bool)
    in_holdout[prepared.holdout] = True
    in_chosen = np.zeros(len(prepared.frame), dtype=bool)
    in_chosen[prepared.chosen] = True
    return pd.DataFrame(
        {
            "probability": probabilities,
            "predicted": (probabilities >= 0.5).astype(int),
            "annotation_group": [readable_group(g)
                                 for g in prepared.groups[rest]],
            "in_holdout": in_holdout[rest],
            "in_chosen_wells": in_chosen[rest],
        },
        index=prepared.frame.index[rest])


def _sampled_pool(positions: np.ndarray, seed: int
                  ) -> Tuple[np.ndarray, str]:
    """``positions``, cut to :data:`MAX_POOL_FOR_DISTANCES` if it is longer.

    :returns: ``(positions, note)``; the note is ``''`` when nothing was cut.
    """
    if positions.size <= MAX_POOL_FOR_DISTANCES:
        return positions, ""
    rng = np.random.default_rng(int(seed) + 5171)
    sampled = np.sort(rng.choice(positions, size=MAX_POOL_FOR_DISTANCES,
                                 replace=False))
    return sampled, (
        f"{positions.size:,} selectable cells is more than the "
        f"{MAX_POOL_FOR_DISTANCES:,} this search builds a dense matrix over, "
        "so it ran on a random sample of that size. The sample is drawn with "
        "the run's seed, so it is the same sample every time.")


def _require_score(prepared: Prepared, request: AnnotationRequest,
                   what: str) -> np.ndarray:
    """The score column as floats, or the refusal that it is not there.

    A screen can be annotated without a score -- that is what a label
    column is for -- but a strategy that RANKS by the score cannot be.
    """
    column = str(request.score_column)
    if column not in prepared.frame.columns:
        raise AnnotationStrategyError(
            f"{what} ranks cells by {column!r} and the object table has no "
            "such column. Choose a strategy that does not rank by the score "
            "-- diversity sampling and the plain random draw do not -- or "
            "name the score column this table carries.")
    return pd.to_numeric(prepared.frame[column],
                         errors="coerce").to_numpy(dtype=float)


def _score_order(prepared: Prepared, request: AnnotationRequest,
                 positions: Sequence[int]) -> np.ndarray:
    """``positions`` sorted by score, highest first, ties by row order."""
    index = np.asarray(list(positions), dtype=int)
    values = _require_score(prepared, request, "This strategy")[index]
    filled = np.where(np.isfinite(values), values, -np.inf)
    return index[np.lexsort((index, -filled))]


def _top_and_contrast(prepared: Prepared, request: AnnotationRequest
                      ) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                                 List[str]]:
    """The named method's two sets: the top-scoring cells and a random draw.

    :returns: ``(positives, contrast, training labels, notes)``. The labels
        are the strategy's own -- 1 for a positive, 0 for a contrast cell
        -- and are never used to score anything.
    """
    notes: List[str] = []
    pool = prepared.chosen
    wanted = int(min(request.n_positive, pool.size // 2))
    if wanted < 2:
        raise AnnotationStrategyError(
            f"The chosen wells hold {pool.size:,} selectable cell(s), which "
            f"cannot supply {request.n_positive:,} positives and the same "
            "number of contrast cells. Lower n_positive or name more wells.")
    if wanted < int(request.n_positive):
        notes.append(
            f"{request.n_positive:,} positives were asked for and "
            f"{wanted:,} are taken: the positive set and its matched "
            "contrast draw together may not exceed the selectable cells of "
            "the chosen wells.")
    positives = _score_order(prepared, request, pool)[:wanted]
    taken = set(int(p) for p in positives)
    candidates = np.asarray([p for p in prepared.selectable
                             if int(p) not in taken], dtype=int)
    if candidates.size < wanted:
        raise AnnotationStrategyError(
            f"Only {candidates.size:,} cell(s) are left to draw a contrast "
            f"set of {wanted:,} from.")
    rng = np.random.default_rng(int(request.seed))
    contrast = np.sort(rng.choice(candidates, size=wanted, replace=False))
    labels = np.zeros(len(prepared.frame), dtype=int)
    labels[positives] = 1
    notes.append(
        f"The contrast set is {wanted:,} cell(s) drawn at random from every "
        "selectable cell in the screen, not only from the chosen wells, and "
        f"{prepared.positive_share(contrast):.1%} of them are above the "
        "positive cut themselves — a contrast set is not a negative set.")
    return positives, contrast, labels, notes


def _seed_training(prepared: Prepared, request: AnnotationRequest
                   ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """The labelled set a strategy that needs a model starts from.

    Human annotations when there are any, and the named method's own
    top-scoring-against-random pair when there are not. Returned as
    ``(positions, full-length training labels, notes)``.
    """
    if prepared.annotated:
        train = prepared.labelled()
        if train.size < 4 or len(np.unique(prepared.labels[train])) < 2:
            raise NotEnoughLabels(
                f"{train.size} annotated cell(s) are outside the hold-out "
                "wells, and a model needs at least four covering two "
                "classes. Annotate more cells, or run a strategy that needs "
                "no model.")
        return train, prepared.labels, [
            f"The model is seeded from {train.size:,} annotated cell(s), not "
            "from the score."]
    positives, contrast, labels, notes = _top_and_contrast(prepared, request)
    train = np.sort(np.concatenate([positives, contrast]))
    notes.append(
        "There are no annotations to seed a model with, so the seed is the "
        "top-scoring-against-random pair — which carries that method's trap "
        "into this one, and is why the fit is reported without the score's "
        "own inputs as well.")
    return train, labels, notes


def _queue_result(prepared: Prepared, request: AnnotationRequest,
                  entry: Strategy, queue: np.ndarray,
                  notes: Sequence[str],
                  extra: Optional[Mapping[str, Mapping[int, Any]]] = None,
                  fit: Optional[FitReport] = None,
                  leakage: Optional[LeakageReport] = None,
                  predictions: Optional[pd.DataFrame] = None,
                  counts: Optional[Mapping[str, Any]] = None
                  ) -> AnnotationResult:
    """Wrap a queue of cells for a person to annotate as a result.

    Every queue carries the one number that says what it did to the class
    balance: the share of it the reference label calls positive, beside the
    share a plain random draw of the same size gets. That comparison is the
    plain random draw doing its job for a strategy that is not it.
    """
    rng = np.random.default_rng(int(request.seed) + 977)
    pool = prepared.selectable
    size = int(min(queue.size, pool.size))
    baseline = rng.choice(pool, size=max(1, size), replace=False)
    lines = list(notes)
    lines.append(
        f"Class balance: {prepared.positive_share(queue):.1%} of the "
        f"{queue.size:,} queued cell(s) are above the positive cut, against "
        f"{prepared.positive_share(baseline):.1%} of a plain random draw of "
        "the same size. Nothing is fitted on a queue until somebody has "
        "annotated it.")
    return AnnotationResult(
        strategy=entry.key, title=entry.title,
        selection=_selection_frame(prepared, {"queue": queue}, extra),
        holdout=_selection_frame(prepared, {"holdout": prepared.holdout}),
        predictions=predictions, fit=fit, leakage=leakage,
        notes=tuple(lines),
        counts=dict(counts or {}, queued=int(queue.size),
                    positive_share=float(prepared.positive_share(queue)),
                    random_positive_share=float(
                        prepared.positive_share(baseline))))


def _fitted_result(prepared: Prepared, request: AnnotationRequest,
                   entry: Strategy, roles: Mapping[str, Sequence[int]],
                   train: np.ndarray, train_labels: Optional[np.ndarray],
                   notes: Sequence[str],
                   counts: Optional[Mapping[str, Any]] = None,
                   extra: Optional[Mapping[str, Mapping[int, Any]]] = None
                   ) -> AnnotationResult:
    """Fit on ``train``, apply to the rest, and report on the hold-out."""
    leakage, model, columns = _leakage_report(prepared, train, request,
                                              train_labels)
    fit = leakage.without_score_inputs or leakage.with_score_inputs
    predictions = _apply_model(prepared, model, columns, train)
    lines = list(notes)
    if leakage.without_score_inputs is not None:
        lines.append(
            "The predictions applied to the rest of the screen come from the "
            "fit WITHOUT the score's own inputs, because a prediction that "
            "is a copy of the score tells a reader nothing they did not "
            "already have.")
    elif leakage.with_score_inputs is not None:
        lines.append(
            "The predictions applied to the rest of the screen come from the "
            "fit WITH the score's own inputs, so they may be little more "
            "than a copy of the score.")
    else:
        lines.append(
            "Nothing was fitted and nothing was predicted: there was no fit "
            "left to make once the score's own inputs were removed.")
    return AnnotationResult(
        strategy=entry.key, title=entry.title,
        selection=_selection_frame(prepared, roles, extra),
        holdout=_selection_frame(prepared, {"holdout": prepared.holdout}),
        predictions=predictions, fit=fit, leakage=leakage,
        notes=tuple(lines),
        counts=dict(counts or {}, fitted_on=int(np.asarray(train).size),
                    applied_to=int(0 if predictions is None
                                   else len(predictions))))


# ---------------------------------------------------------------------------
# The strategies
# ---------------------------------------------------------------------------

def _run_top_score_random(prepared: Prepared,
                          request: AnnotationRequest) -> AnnotationResult:
    """Top-scoring cells against a matched random draw, fitted and applied."""
    positives, contrast, labels, notes = _top_and_contrast(prepared, request)
    train = np.sort(np.concatenate([positives, contrast]))
    return _fitted_result(
        prepared, request, TOP_SCORE_RANDOM,
        {"positive": positives, "contrast": contrast},
        train, labels, notes,
        counts={"positives": int(positives.size),
                "contrast": int(contrast.size)})


def _run_uncertainty(prepared: Prepared,
                     request: AnnotationRequest) -> AnnotationResult:
    """Queue the cells the seed model is least sure of."""
    from .active_learning import rank_by_uncertainty, resolve_measure

    measure, _ = resolve_measure(request.measure)
    train, labels, notes = _seed_training(prepared, request)
    leakage, model, used = _leakage_report(prepared, train, request, labels)
    fit = leakage.without_score_inputs or leakage.with_score_inputs
    if model is None:
        raise NotEnoughLabels(
            "No model could be fitted, so there is no decision boundary to "
            "be uncertain about.")
    seen = np.zeros(len(prepared.frame), dtype=bool)
    seen[train] = True
    pool = np.asarray([p for p in prepared.selectable if not seen[p]],
                      dtype=int)
    if pool.size == 0:
        raise AnnotationStrategyError(
            "Every selectable cell is already in the seed set, so there is "
            "nothing left to queue.")
    probabilities = np.asarray(
        model.predict_proba(_matrix(prepared.frame.iloc[pool], used)),
        dtype=float)
    order = rank_by_uncertainty(probabilities, measure,
                                limit=int(request.n_positive),
                                seed=int(request.seed))
    queue = pool[np.asarray(order, dtype=int)]
    scores = {int(pool[i]): float(probabilities[i, -1])
              for i in np.asarray(order, dtype=int)}
    notes = list(notes)
    notes.append(
        f"Uncertainty measured with {measure!r} on the seed model's "
        f"probabilities over {pool.size:,} unlabelled selectable cell(s); "
        f"the {queue.size:,} least certain are queued.")
    if fit is not None:
        notes.append("The seed model, measured on the hold-out: "
                     + fit.summary())
    return _queue_result(
        prepared, request, UNCERTAINTY, queue, notes,
        extra={"model_probability": scores}, fit=fit, leakage=leakage,
        counts={"pool": int(pool.size), "measure": measure})


def _run_diversity(prepared: Prepared,
                   request: AnnotationRequest) -> AnnotationResult:
    """Cluster the features and queue one representative per cluster."""
    from sklearn.cluster import KMeans

    columns = prepared.honest_features or prepared.features
    pool, sampled = _sampled_pool(prepared.selectable, request.seed)
    wanted = int(request.n_clusters or request.n_positive)
    k = int(min(max(2, wanted), pool.size))
    matrix = _standardised(prepared.frame.iloc[pool], columns)
    if matrix.shape[1] == 0:
        raise AnnotationStrategyError(
            "There is no feature column to cluster on.")
    model = KMeans(n_clusters=k, n_init=10, random_state=int(request.seed))
    assigned = model.fit_predict(matrix)
    queue: List[int] = []
    sizes: Dict[int, int] = {}
    for cluster in range(k):
        members = np.flatnonzero(assigned == cluster)
        sizes[cluster] = int(members.size)
        if members.size == 0:
            continue
        distance = np.linalg.norm(
            matrix[members] - model.cluster_centers_[cluster], axis=1)
        queue.append(int(pool[members[int(np.argmin(distance))]]))
    order = np.sort(np.asarray(queue, dtype=int))
    counted = [n for n in sizes.values() if n]
    notes = [
        f"{k} cluster(s) over {len(columns)} standardised feature column(s) "
        f"and {pool.size:,} selectable cell(s); the cell nearest each centre "
        "is queued. Cluster sizes run from "
        f"{min(counted):,} to {max(counted):,} cells, so the budget is spread "
        "evenly over clusters that are not evenly sized.",
        "A cluster is not a phenotype: a plate effect or a focus artefact "
        "clusters just as tightly as a morphology does.",
    ]
    if sampled:
        notes.append(sampled)
    return _queue_result(prepared, request, DIVERSITY, order, notes,
                         counts={"clusters": int(k),
                                 "largest_cluster": int(max(counted)),
                                 "smallest_cluster": int(min(counted))})


def _run_control_anchors(prepared: Prepared,
                         request: AnnotationRequest) -> AnnotationResult:
    """Seed the labels from the plate's own control wells."""
    if not list(request.positive_control_wells) or \
            not list(request.negative_control_wells):
        raise AnnotationStrategyError(
            "Control wells as anchors needs BOTH a positive and a negative "
            "control well list; it was given "
            f"{list(request.positive_control_wells)} and "
            f"{list(request.negative_control_wells)}. Name them, or choose a "
            "strategy that does not use them.")
    positive_mask = wells_selected(prepared.groups,
                                   request.positive_control_wells)
    negative_mask = wells_selected(prepared.groups,
                                   request.negative_control_wells)
    outside = np.zeros(len(prepared.frame), dtype=bool)
    outside[prepared.selectable] = True
    positives = np.flatnonzero(positive_mask & outside)
    negatives = np.flatnonzero(negative_mask & outside & ~positive_mask)
    if positives.size < 2 or negatives.size < 2:
        raise AnnotationStrategyError(
            f"The control wells hold {positives.size} positive and "
            f"{negatives.size} negative cell(s) outside the hold-out wells, "
            "which is not enough to fit on. Check the well names against the "
            f"plate: it has "
            f"{sorted({readable_group(g) for g in prepared.groups})[:6]}.")
    labels = np.zeros(len(prepared.frame), dtype=int)
    labels[positives] = 1
    train = np.sort(np.concatenate([positives, negatives]))
    notes = [
        f"Labels come from the plate: {positives.size:,} cell(s) of the "
        f"positive control wells and {negatives.size:,} of the negative "
        "ones, with no manual annotation at all.",
        "Control wells differ from sample wells in more than the phenotype — "
        "position on the plate, seeding density, edge effects — so a model "
        "fitted on them can separate the WELLS rather than the cells. The "
        "hold-out is drawn from every well, which is what makes that visible "
        "rather than invisible.",
    ]
    return _fitted_result(
        prepared, request, CONTROL_ANCHORS,
        {"anchor_positive": positives, "anchor_negative": negatives},
        train, labels, notes,
        counts={"positive_controls": int(positives.size),
                "negative_controls": int(negatives.size)})


def _run_pu_learning(prepared: Prepared,
                     request: AnnotationRequest) -> AnnotationResult:
    """The same positives, with the contrast set treated as unlabelled."""
    from .classifier_evaluation import grouped_split

    positives, contrast, _labels, notes = _top_and_contrast(prepared, request)
    train = np.sort(np.concatenate([positives, contrast]))
    is_positive = np.zeros(len(prepared.frame), dtype=int)
    is_positive[positives] = 1
    columns = prepared.honest_features or prepared.features
    # THE LABELLING RATE IS ESTIMATED WHERE THE MODEL DID NOT FIT. Estimating
    # it on the rows the model was fitted on returns the model's own
    # confidence rather than the rate, and the rescaling it produces is
    # therefore always about 1.
    try:
        inner_train, inner_test, inner_split = grouped_split(
            prepared.groups[train], is_positive[train], 0.3,
            seed=int(request.seed), group_by=prepared.level)
    except AnnotationStrategyError:
        raise
    except ValueError as refusal:
        raise AnnotationStrategyError(
            "The labelling rate cannot be estimated on cells the model did "
            f"not fit: {refusal}") from refusal
    estimator, name = _estimator(request.model, request.seed)
    estimator.fit(_matrix(prepared.frame.iloc[train[inner_train]], columns),
                  is_positive[train[inner_train]])
    validation = train[inner_test]
    labelled = validation[is_positive[validation] == 1]
    if labelled.size == 0:
        raise NotEnoughLabels(
            "The inner split left no labelled positive to estimate the "
            "labelling rate on, so the rescaling cannot be calibrated.")
    g_positive = np.asarray(
        estimator.predict_proba(_matrix(prepared.frame.iloc[labelled],
                                        columns)), dtype=float)[:, -1]
    rate = float(np.mean(g_positive))
    if not np.isfinite(rate) or rate <= 1e-6:
        raise AnnotationStrategyError(
            f"The estimated labelling rate is {rate:.3g}, which the "
            "rescaling cannot divide by. The positives are not separable "
            "from the unlabelled set on these columns.")
    matrix = _matrix(prepared.frame.iloc[prepared.holdout], columns)
    posterior = np.clip(
        np.asarray(estimator.predict_proba(matrix), dtype=float)[:, -1] / rate,
        0.0, 1.0)
    raw = np.asarray(estimator.predict_proba(matrix), dtype=float)[:, -1]
    fit = _score_holdout(prepared, posterior, int(train[inner_train].size),
                         columns, f"{name} + positive-unlabelled rescaling")
    called_raw = float(np.mean(raw >= 0.5))
    called_rescaled = float(np.mean(posterior >= 0.5))
    lines = list(notes)
    lines.append(
        f"The contrast set is UNLABELLED, not negative: the model fits "
        f"P(labelled | cell) and its output is divided by the labelling rate "
        f"c = {rate:.3f}, estimated on {labelled.size:,} held-out positive(s) "
        f"through {inner_split.summary()}")
    lines.append(
        f"Dividing by c moves the positive call and nothing else: it calls "
        f"{called_rescaled:.1%} of the hold-out positive where treating the "
        f"contrast set as negative calls {called_raw:.1%}. The cells are not "
        "reordered, so the ROC AUC is the same either way — the bias this "
        "corrects is in WHERE the line is drawn, not in which cells lead.")
    lines.append(
        "c is estimated under the assumption that a positive was labelled "
        "independently of its features. Taking the TOP of the score breaks "
        "that assumption, so c is a floor on the true labelling rate and the "
        "correction is a partial one.")
    predictions = _apply_model(prepared, estimator, columns, train)
    if predictions is not None:
        predictions["probability"] = np.clip(
            predictions["probability"].to_numpy(dtype=float) / rate, 0.0, 1.0)
        predictions["predicted"] = (
            predictions["probability"].to_numpy(dtype=float) >= 0.5
        ).astype(int)
    return AnnotationResult(
        strategy=PU_LEARNING.key, title=PU_LEARNING.title,
        selection=_selection_frame(prepared, {"positive": positives,
                                              "unlabelled": contrast}),
        holdout=_selection_frame(prepared, {"holdout": prepared.holdout}),
        predictions=predictions, fit=fit, leakage=None, notes=tuple(lines),
        counts={"positives": int(positives.size),
                "unlabelled": int(contrast.size),
                "labelling_rate": rate,
                "called_positive_rescaled": called_rescaled,
                "called_positive_as_negative": called_raw})


def _run_self_training(prepared: Prepared,
                       request: AnnotationRequest) -> AnnotationResult:
    """Fit, accept the confident predictions as labels, refit, and audit."""
    train, labels, notes = _seed_training(prepared, request)
    columns = prepared.honest_features or prepared.features
    labels = np.asarray(labels, dtype=int).copy()
    known = np.zeros(len(prepared.frame), dtype=bool)
    known[train] = True
    pseudo = np.zeros(len(prepared.frame), dtype=bool)
    confidence = float(min(max(request.confidence, 0.5), 0.999))
    curve: List[Tuple[int, int, float]] = []
    best: Optional[FitReport] = None
    best_round = 0
    stopped = ""
    for index in range(max(1, int(request.rounds))):
        fitted = np.flatnonzero(known)
        report, estimator, _ = _fit_report(prepared, fitted, columns, request,
                                           labels)
        curve.append((index, int(fitted.size), report.balanced_accuracy))
        if best is None or report.balanced_accuracy > best.balanced_accuracy:
            best, best_round = report, index
        elif index:
            stopped = (f"stopped after round {index}: the audit set stopped "
                       "improving, and a round that does not improve it is a "
                       "round of the model agreeing with itself")
            break
        pool = np.asarray([p for p in prepared.selectable
                           if not known[p] and not pseudo[p]], dtype=int)
        if pool.size == 0:
            stopped = "stopped: every selectable cell already carries a label"
            break
        probabilities = np.asarray(
            estimator.predict_proba(_matrix(prepared.frame.iloc[pool],
                                            columns)), dtype=float)[:, -1]
        confident = ((probabilities >= confidence) |
                     (probabilities <= 1.0 - confidence))
        accepted = pool[confident]
        if accepted.size == 0:
            stopped = ("stopped: no unlabelled cell reached the "
                       f"{confidence:.2f} confidence the round needs to "
                       "accept its own answer")
            break
        labels[accepted] = (probabilities[confident] >= 0.5).astype(int)
        known[accepted] = True
        pseudo[accepted] = True
    if best is None:
        raise NotEnoughLabels("No self-training round could be fitted.")
    lines = list(notes)
    lines.append(
        "The audit set is the hold-out: fixed before the first round, held "
        "out of every round, and the only thing any round is measured on.")
    lines.append("Audit curve (round, cells fitted on, balanced accuracy): "
                 + "; ".join(f"{r}, {n:,}, {a:.3f}" for r, n, a in curve))
    lines.append(stopped or
                 f"ran the full {len(curve)} round(s) without the audit "
                 "turning down")
    lines.append(f"The reported fit is round {best_round}, the best the audit "
                 "saw; the later rounds are kept in the curve so the drift is "
                 "visible rather than discarded.")
    final = np.flatnonzero(known)
    leakage, model, used = _leakage_report(prepared, final, request, labels)
    predictions = _apply_model(prepared, model, used, final)
    return AnnotationResult(
        strategy=SELF_TRAINING.key, title=SELF_TRAINING.title,
        selection=_selection_frame(
            prepared, {"positive": train[labels[train] == 1],
                       "contrast": train[labels[train] == 0],
                       "pseudo": np.flatnonzero(pseudo)}),
        holdout=_selection_frame(prepared, {"holdout": prepared.holdout}),
        predictions=predictions, fit=best, leakage=leakage,
        notes=tuple(lines),
        counts={"rounds": len(curve), "best_round": best_round,
                "seed_cells": int(train.size),
                "pseudo_labelled": int(pseudo.sum()),
                "confidence": confidence})


def _run_two_view_disagreement(prepared: Prepared,
                               request: AnnotationRequest) -> AnnotationResult:
    """Queue the cells an intensity model and a shape model disagree about."""
    from .active_learning import disagreement

    columns = prepared.honest_features or prepared.features
    views = feature_views(columns)
    if not views["intensity"] or not views["shape"]:
        raise AnnotationStrategyError(
            "Two views need columns in both families and this table has "
            f"{len(views['intensity'])} intensity-like and "
            f"{len(views['shape'])} shape-like column(s). Join the "
            "measurement tables to these rows, or choose another strategy.")
    train, labels, notes = _seed_training(prepared, request)
    seen = np.zeros(len(prepared.frame), dtype=bool)
    seen[train] = True
    pool = np.asarray([p for p in prepared.selectable if not seen[p]],
                      dtype=int)
    if pool.size == 0:
        raise AnnotationStrategyError(
            "Every selectable cell is in the seed set, so there is nothing "
            "for two views to disagree about.")
    fits: Dict[str, FitReport] = {}
    opinions: List[np.ndarray] = []
    for name in ("intensity", "shape"):
        report, estimator, _ = _fit_report(prepared, train, views[name],
                                           request, labels)
        fits[name] = report
        opinions.append(np.asarray(
            estimator.predict_proba(_matrix(prepared.frame.iloc[pool],
                                            views[name])),
            dtype=float)[:, -1])
    scores = np.asarray(disagreement(opinions), dtype=float).reshape(-1)
    ranked = pool[np.lexsort((pool, -scores))][:int(request.n_positive)]
    wanted = {int(q) for q in ranked}
    queued = {int(position): index for index, position in enumerate(pool)
              if int(position) in wanted}
    gap = {position: float(abs(opinions[0][index] - opinions[1][index]))
           for position, index in queued.items()}
    best = max(fits, key=lambda name: fits[name].lift)
    lines = list(notes)
    lines.append(
        f"The intensity view got {len(views['intensity'])} column(s) and the "
        f"shape view {len(views['shape'])}; both were fitted on the same "
        f"{train.size:,} seed cell(s) and asked about the same "
        f"{pool.size:,} unlabelled ones.")
    for name, report in fits.items():
        lines.append(f"The {name} view on the hold-out: {report.summary()}")
    lines.append(
        f"The {best} view carries more of the signal, so a cell the other "
        "view contradicts it on is where a single model would be "
        "confidently wrong.")
    return _queue_result(prepared, request, TWO_VIEW_DISAGREEMENT, ranked,
                         lines, extra={"view_gap": gap}, fit=fits[best],
                         counts={"intensity_columns": len(views["intensity"]),
                                 "shape_columns": len(views["shape"]),
                                 "better_view": best})


def _run_score_strata(prepared: Prepared,
                      request: AnnotationRequest) -> AnnotationResult:
    """Queue a fixed number per score stratum rather than only the top."""
    pool = prepared.selectable
    values = _require_score(prepared, request,
                            "Stratifying across the score range")[pool]
    usable = pool[np.isfinite(values)]
    scores = values[np.isfinite(values)]
    if usable.size < 2:
        raise AnnotationStrategyError(
            "Fewer than two selectable cells carry a finite score, so the "
            "score range cannot be divided.")
    bins = int(max(2, min(request.n_bins, usable.size)))
    edges = np.quantile(scores, np.linspace(0.0, 1.0, bins + 1))
    edges[0] = -np.inf
    edges[-1] = np.inf
    assigned = np.clip(np.digitize(scores, edges[1:-1], right=False),
                       0, bins - 1)
    per_bin = int(max(1, int(request.n_positive) // bins))
    rng = np.random.default_rng(int(request.seed))
    queue: List[int] = []
    counts: Dict[str, int] = {}
    for index in range(bins):
        members = usable[assigned == index]
        counts[f"stratum_{index}"] = int(members.size)
        if members.size == 0:
            continue
        take = int(min(per_bin, members.size))
        queue.extend(int(p) for p in rng.choice(members, size=take,
                                                replace=False))
    order = np.sort(np.asarray(queue, dtype=int))
    empty = sum(1 for value in counts.values() if value == 0)
    notes = [
        f"{bins} equal-count strata of {request.score_column!r} over "
        f"{usable.size:,} selectable cell(s), {per_bin} cell(s) drawn at "
        f"random from each; {order.size:,} queued in all.",
        "A model shown only the extremes has never seen the middle of the "
        "screen, which is most of it. This buys calibration rather than "
        "prevalence: the low strata hold few positives and the budget spent "
        "there buys few positive labels.",
    ]
    if empty:
        notes.append(f"{empty} stratum/strata are empty because ties in the "
                     "score put many cells on one edge.")
    return _queue_result(prepared, request, SCORE_STRATA, order, notes,
                         counts=dict(counts, strata=bins, per_stratum=per_bin))


def _run_neighbour_propagation(prepared: Prepared,
                               request: AnnotationRequest) -> AnnotationResult:
    """Carry each seed label to its neighbours inside a shown distance cut."""
    from sklearn.neighbors import NearestNeighbors

    train, labels, notes = _seed_training(prepared, request)
    columns = prepared.honest_features or prepared.features
    pool, sampled = _sampled_pool(prepared.selectable, request.seed)
    # THE SEEDS STAY IN THE POOL whatever the sample took: a seed that fell
    # out of it could label nothing, which would look like a tight cut.
    pool = np.union1d(pool, np.intersect1d(np.asarray(train, dtype=int),
                                           prepared.selectable))
    matrix = _standardised(prepared.frame.iloc[pool], columns)
    if matrix.shape[1] == 0:
        raise AnnotationStrategyError(
            "There is no feature column to measure a distance in.")
    position_of = {int(p): i for i, p in enumerate(pool)}
    seeds = np.asarray([p for p in train if int(p) in position_of], dtype=int)
    if seeds.size == 0:
        raise AnnotationStrategyError(
            "No seed cell is inside the selectable pool.")
    k = int(min(max(1, request.neighbours) + 1, pool.size))
    finder = NearestNeighbors(n_neighbors=k).fit(matrix)
    rows = np.asarray([position_of[int(p)] for p in seeds], dtype=int)
    distances, neighbours = finder.kneighbors(matrix[rows])
    # THE CUT IS A QUANTILE OF THE DISTANCES ACTUALLY OBSERVED, so it is a
    # number this screen produced rather than one carried in from another.
    observed = distances[:, 1:].reshape(-1)
    observed = observed[np.isfinite(observed)]
    if request.distance_cut is not None:
        radius = float(request.distance_cut)
        how = f"the radius {radius:.4g} this run was given"
    elif observed.size:
        radius = float(np.quantile(observed, float(request.distance_quantile)))
        how = (f"the {request.distance_quantile:.0%} quantile of the "
               f"{observed.size:,} seed-to-neighbour distances this screen "
               "actually produced")
    else:
        radius = 0.0
        how = "no observed distance at all"
    labels = np.asarray(labels, dtype=int).copy()
    votes: Dict[int, List[int]] = {}
    seed_set = set(int(position) for position in seeds)
    reached = 0
    crossed = 0
    for seed_row, row_distances, row_neighbours in zip(
            seeds, distances, neighbours):
        seed_group = str(prepared.groups[int(seed_row)])
        for distance, neighbour in zip(row_distances[1:], row_neighbours[1:]):
            if not np.isfinite(distance) or distance > radius:
                continue
            target = int(pool[int(neighbour)])
            if target in seed_set:
                continue
            votes.setdefault(target, []).append(int(labels[int(seed_row)]))
            reached += 1
            if str(prepared.groups[target]) != seed_group:
                crossed += 1
    propagated: List[int] = []
    ties = 0
    for target, given in votes.items():
        ones = sum(given)
        if ones * 2 == len(given):
            ties += 1
            continue
        labels[target] = 1 if ones * 2 > len(given) else 0
        propagated.append(int(target))
    order = np.sort(np.asarray(propagated, dtype=int))
    if order.size == 0:
        raise AnnotationStrategyError(
            f"No neighbour lies inside the radius {radius:.3g} ({how}), so "
            "nothing propagates. Raise distance_cut or distance_quantile, or "
            "annotate more seeds.")
    # A LABEL THAT CROSSED A WELL BOUNDARY is the one to watch: it says the
    # nearest cell in feature space was in another well, which is either the
    # phenotype repeating or the plate showing through.
    share = crossed / float(max(1, reached))
    train_all = np.sort(np.concatenate([seeds, order]))
    lines = list(notes)
    lines.append(
        f"The distance cut is {radius:.4g} in standardised feature space "
        f"over {len(columns)} column(s) — {how}.")
    lines.append(
        f"{seeds.size:,} seed(s) reached {reached:,} neighbour slot(s) and "
        f"{order.size:,} distinct cell(s) took a label — "
        f"{order.size / max(1, seeds.size):.1f} per seed. {ties} cell(s) "
        "were left alone because their neighbours voted evenly.")
    if sampled:
        lines.append(sampled)
    lines.append(
        f"{share:.0%} of the labels that propagated crossed a "
        f"{prepared.level} boundary — the seed and the cell it labelled were "
        "in different ones. A large share there means the cut is measuring "
        "the plate rather than the phenotype, and the radius should come "
        "down.")
    return _fitted_result(
        prepared, request, NEIGHBOUR_PROPAGATION,
        {"positive": seeds[labels[seeds] == 1],
         "contrast": seeds[labels[seeds] == 0],
         "propagated": order},
        train_all, labels, lines,
        counts={"seeds": int(seeds.size), "propagated": int(order.size),
                "radius": radius, "ties": int(ties),
                "reached": int(reached),
                "crossed_group_share": float(share)})


def _run_random_holdout(prepared: Prepared,
                        request: AnnotationRequest) -> AnnotationResult:
    """The plain random draw, and the baseline number it buys.

    Every other strategy is reported against this hold-out. Run on its own
    it answers the question the others cannot: what a budget of the same
    size buys when it is spent at random.
    """
    rng = np.random.default_rng(int(request.seed))
    pool = prepared.selectable
    size = int(min(request.n_positive * 2, pool.size))
    draw = np.sort(rng.choice(pool, size=size, replace=False))
    notes = [
        f"A plain random draw of {draw.size:,} cell(s) from the "
        f"{pool.size:,} outside the hold-out wells, and the "
        f"{prepared.holdout.size:,} cell(s) of the hold-out itself.",
        f"{prepared.positive_share(draw):.1%} of the draw is above the "
        "positive cut, against "
        f"{prepared.positive_share(prepared.chosen):.1%} of the chosen "
        "wells. That difference is what a clever strategy is buying, and it "
        "is the only measurement that can say so.",
    ]
    counts = {"drawn": int(draw.size),
              "positive_share": float(prepared.positive_share(draw)),
              "holdout": int(prepared.holdout.size)}
    try:
        return _fitted_result(
            prepared, request, RANDOM_HOLDOUT, {"contrast": draw}, draw, None,
            notes, counts=counts)
    except NotEnoughLabels as refusal:
        notes.append(
            f"Nothing was fitted: {str(refusal).rstrip('.')}. That is the "
            "finding — a budget spent at random on a screen this unbalanced "
            "buys too few positives to fit on, or has no measurement column "
            "to fit them with, which is why the other strategies exist.")
        return AnnotationResult(
            strategy=RANDOM_HOLDOUT.key, title=RANDOM_HOLDOUT.title,
            selection=_selection_frame(prepared, {"contrast": draw}),
            holdout=_selection_frame(prepared, {"holdout": prepared.holdout}),
            notes=tuple(notes), counts=counts)


#: Which function runs which menu entry. A key here and no entry in
#: :data:`STRATEGIES` would be unreachable; an entry marked implemented with
#: no function here would raise on being chosen, which
#: :func:`run` turns into the refusal rather than an AttributeError.
_RUNNERS: Dict[str, Any] = {
    TOP_SCORE_RANDOM.key: _run_top_score_random,
    UNCERTAINTY.key: _run_uncertainty,
    DIVERSITY.key: _run_diversity,
    CONTROL_ANCHORS.key: _run_control_anchors,
    PU_LEARNING.key: _run_pu_learning,
    SELF_TRAINING.key: _run_self_training,
    TWO_VIEW_DISAGREEMENT.key: _run_two_view_disagreement,
    SCORE_STRATA.key: _run_score_strata,
    NEIGHBOUR_PROPAGATION.key: _run_neighbour_propagation,
    RANDOM_HOLDOUT.key: _run_random_holdout,
}


def run(key: Any, request: AnnotationRequest,
        prepared: Optional[Prepared] = None) -> AnnotationResult:
    """Run one strategy end to end.

    :param key: the strategy key, or a :class:`Strategy`.
    :param request: what to run it on.
    :param prepared: a setup built earlier by :func:`prepare`, when several
        strategies are being compared on one hold-out. Built here when it
        is not given.
    :returns: what the strategy chose, fitted and measured.
    :raises StrategyNotImplemented: the entry is on the menu and has no
        implementation. It refuses out loud rather than selecting nothing.
    :raises AnnotationStrategyError: the data cannot support the strategy.
    """
    entry = strategy(key)
    runner = _RUNNERS.get(entry.key)
    if not entry.implemented or runner is None:
        raise StrategyNotImplemented(
            f"{entry.title!r} is on the menu but is not implemented yet, so "
            "it selects nothing rather than pretending to. The strategies "
            f"that run today are: {', '.join(implemented_keys())}.")
    setup = prepare(request, entry) if prepared is None else prepared
    result = runner(setup, request)
    return AnnotationResult(
        strategy=result.strategy, title=result.title,
        selection=result.selection, holdout=result.holdout,
        predictions=result.predictions, fit=result.fit,
        leakage=result.leakage,
        notes=tuple(setup.notes) + tuple(result.notes),
        counts=result.counts)
