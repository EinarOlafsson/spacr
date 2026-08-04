"""Which of the four hundred features actually separates the classes.

spaCR measures hundreds of features per object. Plotting them one at a time
until something looks different is not analysis, it is a lottery with a
publication bias, so **the ranking is the feature** and the plotting is the
easy half.

The statistic, and why it is this one
-------------------------------------

The default is **AUC** — the area under the ROC curve of that one feature,
computed from the Mann–Whitney U statistic via ranks and reported as a
*separation* ``|2·AUC − 1|`` in ``[0, 1]``, plus the direction (which class
sits higher).

Three reasons, all of them about this table specifically:

1. **It is rank-based**, so it is invariant under any monotone transform of the
   feature. Half a spaCR measurement table is log-normal-ish — areas,
   integrated intensities, ratios spanning three orders of magnitude — and
   whether ``cell_area`` or ``log(cell_area)`` was measured must not change
   which feature comes top. It does change Cohen's d.
2. **It assumes nothing about the distributions**: not normality, not equal
   variance, not even symmetry. The alternatives assume at least one of those,
   and a segmented-object feature satisfies none of them.
3. **It is bounded and unit-free**, so four hundred features measured in px²,
   in counts and in dimensionless ratios are comparable on one axis. That is
   what ranking *requires*, and it is exactly why an effect size in the
   feature's own units cannot do the job.

And it means something a biologist can check: AUC is the probability that a
randomly chosen object of one class scores above a randomly chosen object of
the other. 0.5 is a coin flip; 0.9 is a feature you could nearly gate on.

**Its failure mode, stated rather than discovered later.** AUC only sees
*stochastic ordering*. A feature where one class is bimodal around the other's
median — same centre, wider spread — scores exactly 0.5 while being obviously
informative. That is not hypothetical in this data: a knockdown that makes some
cells bigger and some smaller is a variance effect, and AUC is blind to it.

So the **KS statistic is computed for every feature, always, whatever the
ranking statistic is**, and a feature with a high KS and an AUC near 0.5 is
flagged :attr:`FeatureScore.is_shape_not_shift`. KS is the largest gap between
the two empirical CDFs, which a variance difference moves and a rank test does
not. The blind spot is not fixed — it is *reported*, on every row it applies to.

The other three, and their failure modes
-----------------------------------------

Offered because different questions want different answers, each with the
sentence a user needs before choosing it (:data:`STATISTIC_FAILURE_MODES`):

* :data:`COHEN_D` — a standardised mean difference. Fails on skew: one object
  three orders of magnitude out moves it, and it assumes the two groups have
  comparable spread, which is precisely the case AUC is blind to.
* :data:`KS` — the largest CDF gap. Sees any difference in distribution
  including variance, but says nothing about *direction*, and is dominated by
  wherever the two curves happen to cross.
* :data:`MUTUAL_INFO` — binned mutual information, normalised by the class
  entropy. Sees non-monotone relationships that AUC cannot, but depends on the
  binning and is **biased upward at small n**: it never reports zero for a
  finite sample, so a table with fifty objects per class produces a tidy
  ranking of pure noise.

The multiple-comparisons problem, out loud
-------------------------------------------

Ranking four hundred features by separation and reading the top one is four
hundred comparisons with one reported. :attr:`ExplorerSpec.n_permutations`
turns on a **label-shuffling null**: the class labels are permuted, the whole
ranking is recomputed, and the *best* score in each shuffle is kept. The 95th
percentile of that is :attr:`ExplorerResult.null_threshold` — the separation
the best of your features reaches by chance alone. A feature below it is not
news, however confidently it is drawn.

It is off by default because it costs a pass per shuffle, and it is computed on
a seeded subsample of at most :data:`NULL_MAX_ROWS` rows, which
:attr:`ExplorerResult.notice` says.

No Qt in here — pure numpy and pandas, like :mod:`spacr.qt.widgets.pca_model`.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .graph_spec import CATEGORICAL, UNPLOTTABLE, column_kinds
from .pivot_spec import LOW_N

__all__ = [
    "ExplorerError",
    "AUC", "COHEN_D", "KS", "MUTUAL_INFO", "STATISTICS",
    "STATISTIC_LABELS", "STATISTIC_FAILURE_MODES",
    "MAX_CLASSES", "MIN_PER_CLASS", "SHAPE_NOT_SHIFT_AUC",
    "SHAPE_NOT_SHIFT_KS", "NULL_MAX_ROWS", "DEFAULT_TOP",
    "ExplorerSpec", "FeatureScore", "ClassSummary", "ExplorerResult",
    "rank_features", "candidate_features", "candidate_labels",
    "auc_of", "cohen_d_of", "ks_of", "mutual_info_of", "distributions",
]


class ExplorerError(ValueError):
    """A ranking that cannot be computed, with the reason in the message."""


AUC = "auc"
COHEN_D = "cohen_d"
KS = "ks"
MUTUAL_INFO = "mutual_info"

#: Every statistic, default first.
STATISTICS: Tuple[str, ...] = (AUC, COHEN_D, KS, MUTUAL_INFO)

STATISTIC_LABELS: Dict[str, str] = {
    AUC: "AUC — rank-based, unit-free, transform-invariant (default)",
    COHEN_D: "Cohen's d — standardised mean difference",
    KS: "KS — largest gap between the two cumulative distributions",
    MUTUAL_INFO: "Mutual information — binned, normalised by class entropy",
}

#: What each one cannot see. Shown beside the picker, because choosing a
#: separation statistic without knowing its blind spot is how a ranking gets
#: believed.
STATISTIC_FAILURE_MODES: Dict[str, str] = {
    AUC: ("blind to a pure difference in spread: a class that is bimodal "
          "around the other's median scores exactly 0.5. The KS column and "
          "the 'shape, not shift' flag are here for that case."),
    COHEN_D: ("assumes comparable spread and is moved by one extreme object; "
              "on a skewed feature it ranks the tail, not the difference."),
    KS: ("says nothing about direction — which class is higher — and is "
         "dominated by wherever the two curves happen to cross."),
    MUTUAL_INFO: ("depends on the bin count, and is biased upward at small n: "
                  "it never reports zero for a finite sample, so a small table "
                  "produces a confident ranking of noise."),
}

#: More classes than this and one-vs-rest stops being readable; the answer is
#: to filter to the comparison you mean.
MAX_CLASSES = 12

#: Below this many objects in a class the feature is scored but flagged. Not
#: a rule about significance — the point at which a distribution is an anecdote.
MIN_PER_CLASS = 3

#: A feature is "shape, not shift" when the rank test is this close to a coin
#: flip while the CDFs are this far apart.
SHAPE_NOT_SHIFT_AUC = 0.10
SHAPE_NOT_SHIFT_KS = 0.20

#: Rows the label-shuffling null is computed over. A null needs many passes
#: and does not need every object.
NULL_MAX_ROWS = 20_000

#: Features the panel lists before the user scrolls.
DEFAULT_TOP = 20


# ---------------------------------------------------------------------------
# The statistics, each over two 1-D arrays of finite values
# ---------------------------------------------------------------------------

def _ranks(values: np.ndarray) -> np.ndarray:
    """Average ranks, 1-based, ties sharing their mean rank.

    Written out rather than taken from scipy so the module has no dependency
    beyond numpy and pandas — the same rule the rest of this package follows.
    """
    order = np.argsort(values, kind="mergesort")
    ranked = np.empty(len(values), dtype=float)
    ranked[order] = np.arange(1, len(values) + 1, dtype=float)
    sorted_values = values[order]
    # Average within each run of equal values. A tied pair contributes exactly
    # half to the U statistic, which is what makes AUC 0.5 for two identical
    # distributions rather than something that depends on the sort order.
    start = 0
    for stop in range(1, len(sorted_values) + 1):
        if stop == len(sorted_values) or sorted_values[stop] != sorted_values[start]:
            if stop - start > 1:
                ranked[order[start:stop]] = ranked[order[start:stop]].mean()
            start = stop
    return ranked


def auc_of(a: np.ndarray, b: np.ndarray) -> float:
    """P(a random ``b`` scores above a random ``a``), ties counted as half.

    Computed from the Mann–Whitney U statistic, so it is exact for ties and
    costs one sort rather than ``len(a) × len(b)`` comparisons.
    """
    n_a, n_b = len(a), len(b)
    if not n_a or not n_b:
        return float("nan")
    ranked = _ranks(np.concatenate([a, b]))
    rank_sum_b = float(ranked[n_a:].sum())
    u_b = rank_sum_b - n_b * (n_b + 1) / 2.0
    return u_b / (n_a * n_b)


def cohen_d_of(a: np.ndarray, b: np.ndarray) -> float:
    """``(mean(b) − mean(a)) / pooled SD``, ddof 1. NaN when either group is
    smaller than two or the pooled SD is zero."""
    n_a, n_b = len(a), len(b)
    if n_a < 2 or n_b < 2:
        return float("nan")
    var_a = float(np.var(a, ddof=1))
    var_b = float(np.var(b, ddof=1))
    pooled = ((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2)
    if not pooled > 0:
        return float("nan")
    return (float(np.mean(b)) - float(np.mean(a))) / float(np.sqrt(pooled))


def ks_of(a: np.ndarray, b: np.ndarray) -> float:
    """The largest gap between the two empirical CDFs, in ``[0, 1]``.

    Evaluated only at the *end* of each run of tied values, so a tie cannot
    produce a gap that neither CDF actually has.
    """
    n_a, n_b = len(a), len(b)
    if not n_a or not n_b:
        return float("nan")
    values = np.concatenate([a, b])
    is_b = np.concatenate([np.zeros(n_a, dtype=bool), np.ones(n_b, dtype=bool)])
    order = np.argsort(values, kind="mergesort")
    values = values[order]
    is_b = is_b[order]
    cdf_b = np.cumsum(is_b) / n_b
    cdf_a = np.cumsum(~is_b) / n_a
    last_of_run = np.empty(len(values), dtype=bool)
    last_of_run[:-1] = values[:-1] != values[1:]
    last_of_run[-1] = True
    return float(np.abs(cdf_a - cdf_b)[last_of_run].max())


def mutual_info_of(a: np.ndarray, b: np.ndarray, bins: int = 16) -> float:
    """Binned mutual information, normalised by the class entropy.

    Equal-*frequency* bins over the pooled values, so the binning adapts to the
    distribution instead of putting 99% of a skewed feature in one bin.

    :returns: ``I(feature; class) / H(class)`` in ``[0, 1]`` — the fraction of
        the class label this one feature explains. Biased upward at small n;
        see :data:`STATISTIC_FAILURE_MODES`.
    """
    n_a, n_b = len(a), len(b)
    if not n_a or not n_b:
        return float("nan")
    pooled = np.concatenate([a, b])
    quantiles = np.linspace(0.0, 1.0, max(2, int(bins)) + 1)
    edges = np.unique(np.quantile(pooled, quantiles))
    if len(edges) < 3:
        # Fewer than two distinct bins: the feature is (almost) constant, and
        # a constant explains nothing.
        return 0.0
    counts = np.stack([np.histogram(a, bins=edges)[0],
                       np.histogram(b, bins=edges)[0]]).astype(float)
    total = counts.sum()
    if not total > 0:
        return float("nan")
    joint = counts / total
    p_class = joint.sum(axis=1, keepdims=True)
    p_bin = joint.sum(axis=0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = joint * np.log2(joint / (p_class * p_bin))
    information = float(np.nansum(np.where(joint > 0, terms, 0.0)))
    with np.errstate(divide="ignore", invalid="ignore"):
        entropy = -float(np.nansum(np.where(p_class > 0,
                                            p_class * np.log2(p_class), 0.0)))
    if not entropy > 0:
        return 0.0
    return max(0.0, min(1.0, information / entropy))


def _separation(statistic: str, a: np.ndarray, b: np.ndarray,
                bins: int) -> float:
    """The ranking score: bigger is more separated, whatever the statistic."""
    if statistic == AUC:
        value = auc_of(a, b)
        return abs(2.0 * value - 1.0) if np.isfinite(value) else float("nan")
    if statistic == COHEN_D:
        return abs(cohen_d_of(a, b))
    if statistic == KS:
        return ks_of(a, b)
    return mutual_info_of(a, b, bins)


# ---------------------------------------------------------------------------
# The spec
# ---------------------------------------------------------------------------

def candidate_labels(frame: pd.DataFrame) -> Tuple[str, ...]:
    """Columns that could say which class an object is in.

    Categorical by :func:`~spacr.qt.widgets.graph_spec.column_kinds`, with
    between two and :data:`MAX_CLASSES` levels, and **not floating point**. A
    class label is text, an integer code or a boolean; a float column with
    eleven distinct values is a measurement that happens to be coarse, and
    offering it as "the thing to separate by" is how someone ends up ranking
    every feature against ``cell_eccentricity``.
    """
    kinds = column_kinds(frame)
    out = []
    for name in frame.columns:
        if kinds.get(name) != CATEGORICAL:
            continue
        if pd.api.types.is_float_dtype(frame[name]):
            continue
        levels = frame[name].dropna().nunique()
        if 2 <= levels <= MAX_CLASSES:
            out.append(str(name))
    return tuple(sorted(out))


def candidate_features(frame: pd.DataFrame,
                       label: Optional[str] = None) -> Tuple[str, ...]:
    """Every numeric column that is a measurement, minus the label.

    **Not** ``column_kinds() == CONTINUOUS``, and the difference matters.
    :func:`~spacr.qt.widgets.data_filter_panel.classify_columns` calls a
    numeric column with twelve or fewer distinct values a *category*, which is
    the right rule for deciding whether to offer a slider or a tick list — and
    the wrong one here, because ``pathogen_count`` runs 0–8 and is exactly the
    kind of feature a ranking exists to surface. A separation statistic is
    perfectly happy on a discrete count.

    What is still excluded is what ``classify_columns`` calls ``skip``: the
    identity columns (``object_label``, ``prcfo``), which identify a row rather
    than describe it, and would each score an AUC of 0.5 or 1.0 depending on
    how the table happened to be sorted.

    Booleans are included as 0/1. A plate or batch column that is numeric is
    included too, deliberately: if ``plateID`` ranks near the top, the classes
    are separated by which plate they were on, and that is the most useful
    thing this screen can tell anyone.
    """
    kinds = column_kinds(frame)
    out = []
    for name in frame.columns:
        if name == label or kinds.get(name) == UNPLOTTABLE:
            continue
        series = frame[name]
        if pd.api.types.is_bool_dtype(series) or \
                pd.api.types.is_numeric_dtype(series):
            out.append(str(name))
    return tuple(sorted(out))


@dataclass(frozen=True)
class ExplorerSpec:
    """What to rank, against what, by which statistic.

    :param label: the column holding the class or condition.
    :param features: the columns to rank. Empty means *every* continuous
        column, which is the point of the screen.
    :param statistic: one of :data:`STATISTICS`.
    :param top: how many to keep in :attr:`ExplorerResult.scores`. The rest are
        counted, not silently dropped.
    :param bins: bins for the mutual information and for the drawn histograms.
    :param n_permutations: label shuffles for the null; ``0`` is off.
    :param seed: for the null and for any subsampling, so a ranking is the
        same ranking twice.
    :raises ExplorerError: on an unknown statistic or a non-positive top.
    """

    label: str = ""
    features: Tuple[str, ...] = ()
    statistic: str = AUC
    top: int = DEFAULT_TOP
    bins: int = 16
    n_permutations: int = 0
    seed: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "label", str(self.label or "").strip())
        object.__setattr__(self, "features",
                           tuple(str(f) for f in self.features if f))
        if self.statistic not in STATISTICS:
            raise ExplorerError(
                f"unknown separation statistic {self.statistic!r}; choose "
                f"from {', '.join(STATISTICS)}")
        if int(self.top) < 1:
            raise ExplorerError(f"top must be at least 1, not {self.top}")
        object.__setattr__(self, "top", int(self.top))
        object.__setattr__(self, "bins", max(2, int(self.bins)))
        object.__setattr__(self, "n_permutations",
                           max(0, int(self.n_permutations)))
        object.__setattr__(self, "seed", int(self.seed))

    def with_statistic(self, statistic: str) -> "ExplorerSpec":
        return replace(self, statistic=statistic)

    def with_label(self, label: str) -> "ExplorerSpec":
        return replace(self, label=label)

    def with_features(self, features: Sequence[str]) -> "ExplorerSpec":
        return replace(self, features=tuple(features))

    def to_dict(self) -> Dict[str, Any]:
        return {"label": self.label, "features": list(self.features),
                "statistic": self.statistic, "top": self.top,
                "bins": self.bins, "n_permutations": self.n_permutations,
                "seed": self.seed}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ExplorerSpec":
        fields = {"label", "features", "statistic", "top", "bins",
                  "n_permutations", "seed"}
        known = {k: v for k, v in dict(payload).items() if k in fields}
        if "features" in known:
            known["features"] = tuple(known["features"] or ())
        return cls(**known)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> "ExplorerSpec":
        return cls.from_dict(json.loads(text))

    def describe(self) -> str:
        what = (f"{len(self.features)} features" if self.features
                else "every continuous column")
        return (f"{what} split by {self.label or '(no class column)'}, "
                f"ranked by {STATISTIC_LABELS[self.statistic].split(' — ')[0]}")


# ---------------------------------------------------------------------------
# The result
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ClassSummary:
    """One class's distribution of one feature — what the panel draws."""

    level: str
    n: int
    median: float
    q25: float
    q75: float
    low: float
    high: float

    @property
    def is_low_n(self) -> bool:
        return 0 < self.n <= LOW_N

    def describe(self) -> str:
        return (f"{self.level}: n={self.n:,}, median {self.median:.4g} "
                f"[{self.q25:.4g}, {self.q75:.4g}]")


@dataclass(frozen=True)
class FeatureScore:
    """One feature's separation, every statistic, and the caveats.

    :param score: the ranking statistic's value, bigger meaning more
        separated. For :data:`AUC` this is ``|2·AUC − 1|``, not the AUC.
    :param auc: the directed AUC, so ``> 0.5`` means :attr:`higher_in` scores
        above the rest.
    :param is_shape_not_shift: the rank test is near a coin flip while the
        CDFs are far apart — a difference in spread, which the default
        statistic cannot see. See the module docstring.
    """

    feature: str
    statistic: str
    score: float
    auc: float
    cohen_d: float
    ks: float
    mutual_info: float
    higher_in: str
    against: str
    n_by_class: Mapping[str, int]
    summaries: Tuple[ClassSummary, ...] = ()

    @property
    def is_shape_not_shift(self) -> bool:
        return (np.isfinite(self.auc) and np.isfinite(self.ks)
                and abs(self.auc - 0.5) <= SHAPE_NOT_SHIFT_AUC
                and self.ks >= SHAPE_NOT_SHIFT_KS)

    @property
    def smallest_class(self) -> int:
        return min(self.n_by_class.values()) if self.n_by_class else 0

    @property
    def is_low_n(self) -> bool:
        return 0 < self.smallest_class <= LOW_N

    def describe(self) -> str:
        parts = [f"{self.feature}: {self.score:.3f}"]
        if np.isfinite(self.auc):
            parts.append(f"AUC {self.auc:.3f}, higher in {self.higher_in}")
        if np.isfinite(self.cohen_d):
            parts.append(f"d {self.cohen_d:+.2f}")
        if np.isfinite(self.ks):
            parts.append(f"KS {self.ks:.3f}")
        if self.is_shape_not_shift:
            parts.append("SHAPE, NOT SHIFT — the classes differ in spread, "
                         "not in level; a rank statistic cannot see it")
        if self.is_low_n:
            parts.append(f"n={self.smallest_class} in the smaller class")
        return " · ".join(parts)


@dataclass(frozen=True)
class ExplorerResult:
    """A ranking, and everything that qualifies it.

    :param scores: the kept features, most separated first.
    :param n_considered: how many features were scored, including the ones
        below :attr:`ExplorerSpec.top`.
    :param skipped: ``{feature: why}`` for columns that could not be scored —
        constant, all missing, or nothing left after dropping NaN. Reported
        rather than dropped: a feature missing from a ranking looks the same
        as a feature that ranked last.
    :param null_threshold: the 95th percentile of the best score under label
        shuffling, or ``None`` when the null was not run.
    """

    spec: ExplorerSpec
    label: str
    classes: Tuple[str, ...]
    scores: Tuple[FeatureScore, ...]
    n_rows: int
    n_considered: int
    skipped: Mapping[str, str] = field(default_factory=dict)
    null_threshold: Optional[float] = None
    notice: str = ""

    def __len__(self) -> int:
        return len(self.scores)

    def top(self, count: Optional[int] = None) -> Tuple[FeatureScore, ...]:
        return self.scores[:count] if count else self.scores

    def score_for(self, feature: str) -> FeatureScore:
        for score in self.scores:
            if score.feature == feature:
                return score
        raise ExplorerError(
            f"{feature!r} is not in this ranking; it has "
            f"{len(self.scores)} of {self.n_considered} features"
            + (f", and {feature!r} was skipped: {self.skipped[feature]}"
               if feature in self.skipped else ""))

    def above_null(self) -> Tuple[FeatureScore, ...]:
        """The features that beat the label-shuffling null.

        The whole ranking when the null was not run — with the caveat that
        "not run" is not the same as "passed", which :meth:`summary` says.
        """
        if self.null_threshold is None:
            return self.scores
        return tuple(s for s in self.scores if s.score > self.null_threshold)

    def summary(self) -> str:
        parts = [f"{self.n_considered:,} features over {self.n_rows:,} objects, "
                 f"split by {self.label} into {len(self.classes)} classes"]
        if self.skipped:
            parts.append(f"{len(self.skipped)} could not be scored")
        if self.null_threshold is not None:
            parts.append(
                f"shuffling the labels reaches {self.null_threshold:.3f} at "
                f"the 95th percentile — {len(self.above_null())} feature(s) "
                f"beat it")
        else:
            parts.append(
                f"ranking {self.n_considered:,} features means "
                f"{self.n_considered:,} comparisons with one read; turn on the "
                f"label-shuffling null to see what the best of them reaches by "
                f"chance")
        if self.notice:
            parts.append(self.notice)
        return " · ".join(parts)


# ---------------------------------------------------------------------------
# Computing it
# ---------------------------------------------------------------------------

def _class_levels(frame: pd.DataFrame, label: str) -> Tuple[np.ndarray,
                                                            Tuple[str, ...]]:
    if label not in frame.columns:
        raise ExplorerError(
            f"there is no column called {label!r} to split by; this table has "
            f"{len(frame.columns)} columns")
    text = frame[label].astype(str).to_numpy()
    known = frame[label].notna().to_numpy()
    levels = tuple(sorted({str(v) for v in frame[label].dropna().unique()}))
    if len(levels) < 2:
        raise ExplorerError(
            f"{label!r} has {len(levels)} class(es); there is nothing to "
            f"separate. Pick a column with at least two values")
    if len(levels) > MAX_CLASSES:
        raise ExplorerError(
            f"{label!r} has {len(levels)} classes, more than the "
            f"{MAX_CLASSES} this screen ranks against. Filter to the "
            f"comparison you mean")
    return np.where(known, text, ""), levels


def _summaries(values: np.ndarray, keys: np.ndarray,
               levels: Sequence[str]) -> Tuple[ClassSummary, ...]:
    out = []
    for level in levels:
        picked = values[keys == level]
        if not picked.size:
            out.append(ClassSummary(level, 0, *[float("nan")] * 5))
            continue
        out.append(ClassSummary(
            level=level, n=int(picked.size),
            median=float(np.median(picked)),
            q25=float(np.quantile(picked, 0.25)),
            q75=float(np.quantile(picked, 0.75)),
            low=float(picked.min()), high=float(picked.max())))
    return tuple(out)


def _score_one(feature: str, values: np.ndarray, keys: np.ndarray,
               levels: Sequence[str], spec: ExplorerSpec) -> FeatureScore:
    """One feature's scores. Two classes directly; more, one-vs-rest.

    With more than two classes the reported separation is the **best**
    one-vs-rest, and :attr:`FeatureScore.higher_in` names which class that was:
    "this feature separates *treated* from everything else" is the answer a
    ranking of a multi-condition plate has to give, and averaging the
    one-vs-rest scores would bury it.
    """
    best: Optional[Tuple[float, str, float, float, float, float]] = None
    # Two classes need one comparison, not two: "a against b" and "b against
    # a" are the same separation with the direction flipped.
    for level in (levels if len(levels) > 2 else levels[1:]):
        this = values[keys == level]
        rest = values[keys != level]
        score = _separation(spec.statistic, rest, this, spec.bins)
        candidate = (score, level, auc_of(rest, this), cohen_d_of(rest, this),
                     ks_of(rest, this), mutual_info_of(rest, this, spec.bins))
        if best is None:
            best = candidate
        elif np.isfinite(score) and (not np.isfinite(best[0])
                                     or score > best[0]):
            best = candidate
    score, level, auc, d, ks, mi = best
    against = ("rest" if len(levels) > 2
               else next(l for l in levels if l != level))
    higher = level if (np.isfinite(auc) and auc >= 0.5) else against
    return FeatureScore(
        feature=feature, statistic=spec.statistic, score=float(score),
        auc=float(auc), cohen_d=float(d), ks=float(ks), mutual_info=float(mi),
        higher_in=higher, against=against,
        n_by_class={lv: int((keys == lv).sum()) for lv in levels},
        summaries=_summaries(values, keys, levels))


def _null_threshold(columns: Dict[str, np.ndarray], keys: np.ndarray,
                    levels: Sequence[str], spec: ExplorerSpec,
                    notices: List[str]) -> Optional[float]:
    """The 95th percentile of the best-of-all-features score under shuffling.

    The *maximum* per shuffle, not the mean: the question is "how big does the
    winner of four hundred features get by chance", and the mean of a null
    answers a question nobody asked.
    """
    if not spec.n_permutations or not columns:
        return None
    rng = np.random.default_rng(spec.seed)
    rows = len(keys)
    take = np.arange(rows)
    if rows > NULL_MAX_ROWS:
        take = np.sort(rng.choice(rows, size=NULL_MAX_ROWS, replace=False))
        notices.append(
            f"the null is computed on a seeded {NULL_MAX_ROWS:,}-row subsample")
    sampled = {name: values[take] for name, values in columns.items()}
    labels = keys[take]
    best: List[float] = []
    for _ in range(spec.n_permutations):
        shuffled = rng.permutation(labels)
        top = 0.0
        for values in sampled.values():
            finite = np.isfinite(values)
            here = values[finite]
            group = shuffled[finite]
            for level in (levels if len(levels) > 2 else levels[1:]):
                score = _separation(spec.statistic, here[group != level],
                                    here[group == level], spec.bins)
                if np.isfinite(score):
                    top = max(top, float(score))
        best.append(top)
    return float(np.quantile(best, 0.95)) if best else None


def rank_features(frame: pd.DataFrame,
                  spec: Optional[ExplorerSpec] = None) -> ExplorerResult:
    """Score every feature against ``spec.label`` and sort by separation.

    :raises ExplorerError: when there is no usable class column, or when none
        of the features can be scored — each with the reason.
    """
    spec = spec or ExplorerSpec()
    if not spec.label:
        raise ExplorerError(
            "pick the column that says which class each object is in; a "
            "separation needs two populations to separate")
    keys, levels = _class_levels(frame, spec.label)
    features = spec.features or candidate_features(frame, spec.label)
    if not features:
        raise ExplorerError(
            "this table has no continuous columns to rank. Every column is "
            "either a category or an identifier")

    notices: List[str] = []
    skipped: Dict[str, str] = {}
    scores: List[FeatureScore] = []
    usable: Dict[str, np.ndarray] = {}
    for feature in features:
        if feature not in frame.columns:
            skipped[feature] = "not a column of this table"
            continue
        values = pd.to_numeric(frame[feature], errors="coerce").to_numpy(float)
        finite = np.isfinite(values)
        if not finite.any():
            skipped[feature] = "no finite values"
            continue
        if np.nanmin(values[finite]) == np.nanmax(values[finite]):
            skipped[feature] = "constant — one value cannot separate anything"
            continue
        present = finite & (keys != "")
        counts = {level: int((keys[present] == level).sum()) for level in levels}
        if min(counts.values()) < 1:
            empty = [lv for lv, n in counts.items() if not n]
            skipped[feature] = (
                f"no object with a value in {', '.join(empty)}")
            continue
        usable[feature] = np.where(present, values, np.nan)
        scores.append(_score_one(feature, values[present], keys[present],
                                 levels, spec))

    if not scores:
        raise ExplorerError(
            f"none of the {len(features)} features could be scored against "
            f"{spec.label!r}. " + "; ".join(
                f"{name}: {why}" for name, why in list(skipped.items())[:5]))

    scores.sort(key=lambda s: (-(s.score if np.isfinite(s.score) else -1.0),
                               s.feature))
    smallest = min(min(s.n_by_class.values()) for s in scores)
    if smallest < MIN_PER_CLASS:
        notices.append(
            f"the smallest class has {smallest} object(s); a separation over "
            f"that is an anecdote with a number on it")
    shape_only = [s.feature for s in scores if s.is_shape_not_shift]
    if shape_only and spec.statistic == AUC:
        notices.append(
            f"{len(shape_only)} feature(s) differ in spread rather than level "
            f"({', '.join(shape_only[:3])}) — AUC scores those near 0.5, so "
            f"they rank low here whatever they are worth")

    null = _null_threshold(usable, keys, levels, spec, notices)

    return ExplorerResult(
        spec=spec, label=spec.label, classes=levels,
        scores=tuple(scores[:spec.top]), n_rows=int(len(frame)),
        n_considered=len(scores), skipped=skipped, null_threshold=null,
        notice="; ".join(dict.fromkeys(notices)))


def distributions(frame: pd.DataFrame, feature: str, label: str, *,
                  bins: int = 16) -> Tuple[np.ndarray,
                                           Dict[str, np.ndarray]]:
    """Shared bin edges and per-class counts, for drawing one feature.

    The edges are computed over **every** class together, so the per-class
    histograms are comparable — the same rule
    :func:`spacr.qt.widgets.graph_spec.scales_for` applies to facets, and for
    the same reason.
    """
    values = pd.to_numeric(frame[feature], errors="coerce").to_numpy(float)
    keys = frame[label].astype(str).to_numpy()
    finite = np.isfinite(values)
    if not finite.any():
        return np.zeros(0), {}
    low, high = float(values[finite].min()), float(values[finite].max())
    if high <= low:
        high = low + (abs(low) * 0.05 or 0.5)
    edges = np.linspace(low, high, max(2, int(bins)) + 1)
    counts = {}
    for level in sorted({str(v) for v in frame[label].dropna().unique()}):
        picked = values[finite & (keys == level)]
        counts[level] = np.histogram(picked, bins=edges)[0]
    return edges, counts
