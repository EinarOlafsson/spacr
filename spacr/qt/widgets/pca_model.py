"""PCA over a measurement table — the statistics, with the decisions written down.

spaCR measures hundreds of features per object and, until this module, had no
principal components anywhere. The reason it is worth having is not
dimensionality reduction for its own sake: it is the one view that answers
*"these two populations separate — on what?"* in a single picture, because the
loadings say which measurements carry the separation.

PCA is also easy to get subtly wrong on exactly this kind of table, in ways
that produce a confident-looking plot of nothing. Three decisions do all the
damage, so all three are made here, in the open, and reported back to the user
with every result.

1. Centring and scaling
-----------------------
**Always centred. Scaled to unit variance by default**
(:data:`SCALE_ZSCORE`), with :data:`SCALE_NONE` available and never implicit.

PCA maximises variance, and variance has units. In a spaCR object table
``cell_area`` is px² and runs to 10^4 with a variance near 10^6;
``eccentricity`` is dimensionless and lives in ``[0, 1]``;
``channel_1_mean_intensity`` is counts. An unscaled (covariance) PCA of those
three returns "PC1 = cell_area" every time — not because size is the
interesting axis but because px² is a big number. Worse, the answer is not
even stable: recording the same areas in µm² instead of px² would rotate every
component.

Standardising each feature to mean 0 and sample SD 1 (a *correlation* PCA)
makes the result invariant to any per-feature linear rescaling, which is the
only defensible default when the columns have no common unit.

It is not free, and the cost is stated rather than hidden: equal weight means
forty near-duplicate Zernike moments outvote one carefully chosen intensity
ratio, and a feature that is pure noise gets the same unit of variance as a
feature that is pure signal. :data:`SCALE_NONE` is there for the case where
the user has already put the features in a common unit and *wants* the big
ones to dominate.

2. NaN
------
**Never imputed by default, and never zero-filled at all.**

A NaN in a spaCR measurement table is usually *structural*, not missing at
random — the same reasoning :mod:`spacr.anndata_export` sets out at length. A
``pathogen_*`` column is NaN for a cell with no pathogen in it; that is a fact
about the cell, not a gap in the record. Mean-imputing it invents a pathogen of
average size and moves an uninfected cell into the middle of the infected
distribution, invisibly.

Dropping rows instead is not neutral either: complete-case analysis over a
feature set containing one ``pathogen_*`` column silently deletes every
uninfected cell, so the result is *the PCA of the infected cells* presented as
the PCA of the cells.

The three failure modes are different in kind, which is what the policies are
built around:

* dropping a **feature** changes the *space* — a smaller question, answered
  exactly;
* dropping a **row** changes the *population* — the same question, answered
  about a self-selected subset;
* **imputing** changes the *data* — the original question, answered about
  numbers nobody measured.

:data:`NAN_AUTO`, the default, picks between the first two by how the
missingness is shaped, because in this table the two shapes are far apart: a
structural NaN affects tens of percent of the rows (every uninfected cell), a
sporadic one affects a handful (one object where mahotas returned nothing). So
a feature missing in more than :data:`DEFAULT_STRUCTURAL_MISSING` of the rows
is treated as structural and dropped by name; rows still carrying a NaN among
the survivors are dropped as sporadic and counted. Nothing is imputed, and both
actions appear in :attr:`PCAResult.notes` and in
:meth:`PCAResult.report`.

:data:`NAN_COMPLETE`, :data:`NAN_DROP_FEATURES` and :data:`NAN_MEAN` are the
explicit versions of the three failure modes above, for a user who knows which
one they want. :data:`NAN_MEAN` is offered and never chosen for them.

**Infinities are missing under every policy.** Ratio features produce ±inf from
a zero denominator, an inf survives ``dropna``, and one of them destroys the
scaling and therefore every component. They become NaN before the policy runs
and are counted separately in :attr:`PCAResult.n_infinite`.

3. Constant and collinear columns
---------------------------------
A **constant** column has SD 0, so standardising it is 0/0. Left in, it
contributes a NaN that spreads through the whole decomposition; "fixed" by
substituting a 1, it becomes a zero-variance direction that the solver is free
to return as a component with an arbitrary loading pattern. It is dropped
before the decomposition and named in :attr:`PCAResult.dropped_features` —
a column with one value cannot explain variation in anything.

**Perfectly collinear** columns are a different problem and get a different
answer: they are *kept*, because the feature list is the user's statement of
what they care about and picking which of ``area`` and ``area_um2`` to discard
is not this module's call. What is refused is inventing components out of the
redundancy. The numerical rank is computed from the singular values and no more
than ``rank`` components are ever returned, so a table of 12 features spanning
9 real directions yields 9 components, not 12 — the last three would have ~0
explained variance and a direction chosen by floating-point noise. Groups of
features that are perfectly correlated are also detected and reported, since a
duplicated feature silently doubles its own weight in a correlation PCA.

Honest explained variance
-------------------------
:attr:`PCAResult.explained_variance_ratio` is over the **total variance of the
analysed matrix**, so it sums to 1 across all ``rank`` components rather than
across the ones that were kept. Alongside it, every result carries how much of
the original table it is actually about (:attr:`PCAResult.n_rows_in` versus
``len(result)``, :attr:`PCAResult.n_features_in` versus its features): "PC1
explains 71%" means something different when the matrix is three features and
8% of the objects.

And the single most common way a PCA of morphology features misleads anyone is
that **PC1 is just size**. When every feature loads with the same sign, PC1 is
a general-magnitude axis — big objects at one end, small at the other — which
is usually a fact about segmentation rather than biology.
:meth:`PCAResult.is_size_like` detects it from the loading signs and
:meth:`PCAResult.headline` says it in words, unprompted.

No Qt in here
-------------
Pure numpy and pandas, like :mod:`spacr.qt.widgets.graph_spec` and
:mod:`spacr.selection`: usable from a notebook, testable without a display, and
free for the later screens (the feature explorer, the gate editor) to reuse
without inheriting a widget.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .graph_spec import CONTINUOUS, column_kinds

__all__ = [
    "PCAError",
    "SCALE_ZSCORE", "SCALE_NONE", "SCALE_MODES",
    "NAN_AUTO", "NAN_COMPLETE", "NAN_DROP_FEATURES", "NAN_MEAN",
    "NAN_POLICIES",
    "DEFAULT_STRUCTURAL_MISSING", "DEFAULT_COMPONENTS", "MAX_COMPONENTS",
    "CONSTANT_TOLERANCE", "COLLINEAR_TOLERANCE", "SIZE_LIKE_AGREEMENT",
    "DEGENERATE_RATIO",
    "PCASpec", "PCAResult", "pca",
    "candidate_features", "component_name", "component_index",
]


class PCAError(ValueError):
    """A PCA that cannot mean anything, with the reason in the message.

    Raised rather than returned as an empty result: every one of these is a
    sentence the user can act on ("every row has a NaN in ``pathogen_area``"),
    and a caller that swallowed it would draw an empty scatter with no
    explanation. Screens catch it and show the message.
    """


# ---------------------------------------------------------------------------
# Policies
# ---------------------------------------------------------------------------

#: Centre and divide by the sample SD — a correlation PCA. The default; see
#: the module docstring for why.
SCALE_ZSCORE = "zscore"
#: Centre only — a covariance PCA. Correct only when every feature is already
#: in the same unit.
SCALE_NONE = "none"
SCALE_MODES: Tuple[str, ...] = (SCALE_ZSCORE, SCALE_NONE)

#: Drop features whose missingness looks structural, then drop the rows still
#: carrying a sporadic NaN. Never imputes. The default.
NAN_AUTO = "auto"
#: Complete cases: keep every feature, drop every row with any NaN.
NAN_COMPLETE = "complete"
#: Keep every row, drop every feature with any NaN.
NAN_DROP_FEATURES = "drop_features"
#: Replace a NaN with its feature's mean over the objects that have one.
#: Explicit only — it fabricates measurements.
NAN_MEAN = "mean"
NAN_POLICIES: Tuple[str, ...] = (NAN_AUTO, NAN_COMPLETE, NAN_DROP_FEATURES,
                                 NAN_MEAN)

#: Above this fraction of missing rows, :data:`NAN_AUTO` calls a feature
#: structurally missing and drops the feature rather than the objects. The two
#: shapes are far apart in a spaCR table — a ``pathogen_*`` column is missing
#: for tens of percent of cells, a failed Zernike for a handful of objects —
#: so anything between about 0.5% and 20% picks the same features. It is a
#: parameter anyway, because "far apart" is an observation and not a law.
DEFAULT_STRUCTURAL_MISSING = 0.02

#: Components computed unless asked otherwise. Past about the tenth, the
#: explained variance is noise and the plot is a scree plot of nothing.
DEFAULT_COMPONENTS = 8
#: Hard ceiling, whatever a caller asks for.
MAX_COMPONENTS = 50

#: A column is constant when its SD is at or below this, relative to its own
#: magnitude. Relative rather than absolute so a column of ``1e-9 ± 0`` is
#: caught and a column of ``1e-9 ± 1e-10`` is not.
CONSTANT_TOLERANCE = 1e-12

#: Two features are reported as collinear at or above this |correlation|.
COLLINEAR_TOLERANCE = 1.0 - 1e-9

#: Share of a component's squared loading that has to sit on one sign before
#: :meth:`PCAResult.is_size_like` calls it a general-magnitude axis.
SIZE_LIKE_AGREEMENT = 0.9

#: Below this share of the total variance a component is reported as
#: unidentified. Near-collinear features leave a numerically full-rank matrix
#: whose trailing directions are floating-point residue rather than structure.
DEGENERATE_RATIO = 0.005

#: Above this many rows the decomposition goes through the p×p Gram matrix
#: instead of a full SVD: ``U`` for a million objects is gigabytes, and the
#: scores are ``X @ V`` either way.
_SVD_ROW_LIMIT = 20_000

#: Beyond this many features the pairwise collinearity scan is skipped — it is
#: O(p²) and a diagnostic, not part of the answer.
_COLLINEAR_SCAN_LIMIT = 1_500


def component_name(index: int) -> str:
    """``0 -> 'PC1'``. The column name a score lands in, in one place."""
    return f"PC{int(index) + 1}"


def component_index(name: str) -> Optional[int]:
    """``'PC1' -> 0``, and ``None`` for anything that is not a PC column."""
    text = str(name).strip()
    if not text.upper().startswith("PC"):
        return None
    try:
        number = int(text[2:])
    except (TypeError, ValueError):
        return None
    return number - 1 if number >= 1 else None


def candidate_features(frame: pd.DataFrame) -> Tuple[str, ...]:
    """The columns worth offering as PCA features, sorted.

    Continuous by :func:`spacr.qt.widgets.graph_spec.column_kinds` — which is
    a re-reading of the Local Data Filter's classifier, the one column
    classifier in this codebase. That rule already excludes object keys (they
    identify rather than describe) and small-cardinality numeric codes like
    ``cell_count`` and a class label, which are counts and labels rather than
    measured quantities and have no business setting the scale of a component.

    A user who disagrees about a particular column puts it in
    :attr:`PCASpec.features` explicitly; nothing here refuses it.
    """
    return tuple(sorted(name for name, kind in column_kinds(frame).items()
                        if kind == CONTINUOUS))


# ---------------------------------------------------------------------------
# The spec
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PCASpec:
    """What to decompose and how. Frozen and JSON round-tripping, like
    :class:`~spacr.qt.widgets.graph_spec.GraphSpec`, so a saved analysis is
    something a settings file or a report can carry.

    :param features: the columns to decompose. Empty means
        :func:`candidate_features` of whatever frame it is run against —
        which is a *default*, not a promise: the result records the features
        it actually used.
    :param n_components: how many to compute. Capped at the numerical rank,
        always: components past the rank have no direction.
    :param scaling: :data:`SCALE_ZSCORE` (default) or :data:`SCALE_NONE`.
    :param nan_policy: one of :data:`NAN_POLICIES`.
    :param structural_missing: the :data:`NAN_AUTO` threshold.
    :raises PCAError: on an unknown scaling or policy, or a non-positive
        component count — at the point the spec is built, not at render time.
    """

    features: Tuple[str, ...] = ()
    n_components: int = DEFAULT_COMPONENTS
    scaling: str = SCALE_ZSCORE
    nan_policy: str = NAN_AUTO
    structural_missing: float = DEFAULT_STRUCTURAL_MISSING

    def __post_init__(self) -> None:
        seen: Dict[str, None] = {}
        for name in self.features or ():
            if name:
                seen.setdefault(str(name), None)
        object.__setattr__(self, "features", tuple(seen))
        if self.scaling not in SCALE_MODES:
            raise PCAError(
                f"unknown scaling {self.scaling!r}; it is "
                f"{SCALE_ZSCORE!r} (unit variance per feature, the default) "
                f"or {SCALE_NONE!r} (centre only)")
        if self.nan_policy not in NAN_POLICIES:
            raise PCAError(
                f"unknown nan_policy {self.nan_policy!r}; choose one of "
                f"{', '.join(NAN_POLICIES)}")
        count = int(self.n_components)
        if count < 1:
            raise PCAError(
                f"n_components must be at least 1, not {self.n_components}")
        object.__setattr__(self, "n_components", min(count, MAX_COMPONENTS))
        fraction = float(self.structural_missing)
        if not 0.0 <= fraction <= 1.0:
            raise PCAError(
                f"structural_missing is a fraction of rows and must be in "
                f"[0, 1], not {self.structural_missing}")
        object.__setattr__(self, "structural_missing", fraction)

    # -- edits ----------------------------------------------------------
    def with_features(self, features: Sequence[str]) -> "PCASpec":
        return replace(self, features=tuple(features))

    def with_scaling(self, scaling: str) -> "PCASpec":
        return replace(self, scaling=scaling)

    def with_nan_policy(self, policy: str) -> "PCASpec":
        return replace(self, nan_policy=policy)

    def with_components(self, n: int) -> "PCASpec":
        return replace(self, n_components=n)

    # -- serialisation ---------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {
            "features": list(self.features),
            "n_components": self.n_components,
            "scaling": self.scaling,
            "nan_policy": self.nan_policy,
            "structural_missing": self.structural_missing,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PCASpec":
        """Rebuild from :meth:`to_dict`; unknown keys ignored, missing keys
        defaulted, so an analysis written by another build still opens."""
        fields = {"features", "n_components", "scaling", "nan_policy",
                  "structural_missing"}
        known = {k: v for k, v in dict(payload).items() if k in fields}
        if "features" in known:
            known["features"] = tuple(known["features"] or ())
        return cls(**known)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> "PCASpec":
        return cls.from_dict(json.loads(text))

    def describe(self) -> str:
        """One line, for a caption."""
        scaling = ("standardised" if self.scaling == SCALE_ZSCORE
                   else "centred only")
        features = (f"{len(self.features)} features" if self.features
                    else "every continuous column")
        return (f"PCA · {features} · {scaling} · "
                f"NaN: {self.nan_policy} · {self.n_components} components")


# ---------------------------------------------------------------------------
# The result
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PCAResult:
    """A decomposition, plus everything needed to read it honestly.

    :param features: the columns actually decomposed, in matrix order. Not
        the ones asked for — see :attr:`dropped_features`.
    :param rows: **positional** indices into the frame :func:`pca` was given,
        naming the objects that survived the NaN policy. Positional because a
        measurement frame carries a duplicated or reset index often enough
        that positions are the only safe currency, the same reason
        :class:`~spacr.qt.widgets.graph_spec.FacetPanel` uses them.
    :param scores: ``(len(rows), k)`` — each object's coordinate on each
        component, in the units of the analysed (post-scaling) matrix.
    :param loadings: ``(len(features), k)``, unit-norm columns. The direction
        of each component in feature space.
    :param correlations: ``(len(features), k)`` Pearson r between each feature
        and each component score. This is what the biplot arrows are: unlike a
        unit-norm loading, an r is a number with a meaning on its own, and for
        a standardised PCA the squared row sums are each feature's communality.
    :param explained_variance: per component, in the analysed matrix's units.
    :param explained_variance_ratio: over the total variance of the analysed
        matrix — so it sums to 1 across all ``rank`` components, not across
        the ones returned.
    :param rank: the numerical rank. No more components than this exist.
    :param dropped_features: ``{column: why}``, for everything that did not
        reach the decomposition.
    :param dropped_rows: objects removed by the NaN policy.
    :param n_infinite: cells that were ±inf and were treated as missing.
    :param collinear_groups: features that are perfectly correlated with each
        other. Kept in the analysis, reported because each such group weighs
        as many times as it has members.
    """

    features: Tuple[str, ...]
    rows: np.ndarray
    scores: np.ndarray
    loadings: np.ndarray
    correlations: np.ndarray
    explained_variance: np.ndarray
    explained_variance_ratio: np.ndarray
    total_variance: float
    centre: np.ndarray
    scale: np.ndarray
    rank: int
    n_rows_in: int
    n_features_in: int
    scaling: str = SCALE_ZSCORE
    nan_policy: str = NAN_AUTO
    dropped_features: Mapping[str, str] = field(default_factory=dict)
    dropped_rows: int = 0
    n_infinite: int = 0
    collinear_groups: Tuple[Tuple[str, ...], ...] = ()
    notes: Tuple[str, ...] = ()

    # -- shape ------------------------------------------------------------
    def __len__(self) -> int:
        """Objects in the analysis."""
        return int(self.scores.shape[0])

    @property
    def n_components(self) -> int:
        return int(self.scores.shape[1])

    @property
    def n_features(self) -> int:
        return len(self.features)

    @property
    def component_names(self) -> Tuple[str, ...]:
        return tuple(component_name(i) for i in range(self.n_components))

    @property
    def cumulative_ratio(self) -> np.ndarray:
        """Running total of :attr:`explained_variance_ratio`."""
        return np.cumsum(self.explained_variance_ratio)

    @property
    def retained_ratio(self) -> float:
        """Variance share of the components actually returned."""
        return float(self.explained_variance_ratio.sum())

    @property
    def row_share(self) -> float:
        """Fraction of the given objects the analysis is about."""
        return (len(self) / self.n_rows_in) if self.n_rows_in else 0.0

    # -- reading one component --------------------------------------------
    def _check(self, k: int) -> int:
        if not 0 <= int(k) < self.n_components:
            raise PCAError(
                f"there is no component {component_name(k)}; this result has "
                f"{self.n_components} "
                f"({', '.join(self.component_names)})")
        return int(k)

    def dominant(self, k: int = 0) -> Tuple[str, float]:
        """``(feature, share)`` — the feature with the largest |loading| on
        component ``k``, and how much of the component's squared loading it
        carries.

        A share near 1 means the component *is* that feature under another
        name, which is worth saying before anyone reads biology into it.
        """
        k = self._check(k)
        column = self.loadings[:, k]
        total = float((column ** 2).sum()) or 1.0
        i = int(np.argmax(np.abs(column)))
        return self.features[i], float(column[i] ** 2 / total)

    def sign_agreement(self, k: int = 0) -> float:
        """How one-sided component ``k``'s loadings are, in ``[0.5, 1]``.

        The larger share of squared loading sitting on a single sign. 1.0 means
        every feature moves the same way along this axis.
        """
        k = self._check(k)
        column = self.loadings[:, k]
        total = float((column ** 2).sum())
        if total <= 0:  # pragma: no cover - a unit-norm column cannot be zero
            return 1.0
        positive = float((column[column > 0] ** 2).sum())
        return max(positive, total - positive) / total

    def is_size_like(self, k: int = 0) -> bool:
        """Whether component ``k`` is a general-magnitude axis.

        Every feature loading with the same sign is the classic "size" factor
        of morphometrics: one end big objects, the other end small ones. It is
        real, it is usually PC1, and it is usually a statement about
        segmentation and focus rather than about biology — so it is detected
        and said out loud rather than left for the reader to notice.

        Two guards keep it from crying wolf. It needs three features, because
        with two "same sign" is a coin flip. And it needs the loading spread
        over more than one of them: a component that is a *single* feature
        trivially agrees with itself about sign, and calling that a size axis
        would be a statement about arithmetic rather than about the objects.
        """
        return (self.n_features >= 3
                and self.dominant(k)[1] < 0.5
                and self.sign_agreement(k) >= SIZE_LIKE_AGREEMENT)

    def is_degenerate(self, k: int = 0) -> bool:
        """Whether component ``k`` holds so little variance that its direction
        is not identified.

        Near-collinear features leave a numerically full-rank matrix whose last
        directions are floating-point residue: the solver returns *a* direction
        because it must, not because the data has one. A component under
        :data:`DEGENERATE_RATIO` of the total variance is reported with that
        said, so nobody reads a loading pattern out of rounding error.
        """
        return float(self.explained_variance_ratio[self._check(k)]) \
            < DEGENERATE_RATIO

    def top_features(self, k: int = 0, count: int = 5
                     ) -> Tuple[Tuple[str, float], ...]:
        """The ``count`` features loading hardest on component ``k``, as
        ``(feature, loading)`` with the sign kept, strongest first."""
        k = self._check(k)
        column = self.loadings[:, k]
        order = np.argsort(np.abs(column))[::-1][:max(0, int(count))]
        return tuple((self.features[i], float(column[i])) for i in order)

    def plane_features(self, kx: int = 0, ky: int = 1, count: int = 8
                       ) -> Tuple[int, ...]:
        """Indices of the features best represented in the ``(kx, ky)`` plane.

        Ranked by ``r_x² + r_y²`` — how much of the feature is visible in this
        plane at all. Drawing the other four hundred arrows would say nothing
        except that the figure is full; drawing the *longest* ones is the
        standard choice because a short arrow means "this feature points
        somewhere you are not looking", not "this feature does not matter".
        """
        kx, ky = self._check(kx), self._check(ky)
        strength = (self.correlations[:, kx] ** 2
                    + self.correlations[:, ky] ** 2)
        order = np.argsort(strength)[::-1][:max(0, int(count))]
        return tuple(int(i) for i in order)

    # -- frames -----------------------------------------------------------
    def scores_frame(self, source: pd.DataFrame, *,
                     components: Optional[int] = None) -> pd.DataFrame:
        """``source``'s surviving rows with ``PC1…PCk`` columns added.

        Every original column is kept, which is the whole point: it is what
        lets the scores plot colour by ``gene``, facet by ``plateID`` and
        publish a real object-key selection when a cluster is brushed. The
        Graph Builder then draws it with no knowledge that it is a PCA.
        """
        k = self.n_components if components is None else \
            max(1, min(int(components), self.n_components))
        frame = source.iloc[self.rows].copy()
        for i in range(k):
            frame[component_name(i)] = self.scores[:, i]
        return frame

    def loadings_frame(self) -> pd.DataFrame:
        """One row per feature: its loading and its correlation on each
        component. What "export the loadings" writes."""
        data: Dict[str, Any] = {"feature": list(self.features)}
        for i in range(self.n_components):
            name = component_name(i)
            data[f"{name}_loading"] = self.loadings[:, i]
            data[f"{name}_r"] = self.correlations[:, i]
        return pd.DataFrame(data)

    def variance_frame(self) -> pd.DataFrame:
        """One row per component — the scree plot's data, and its CSV."""
        return pd.DataFrame({
            "component": list(self.component_names),
            "explained_variance": self.explained_variance,
            "explained_variance_ratio": self.explained_variance_ratio,
            "cumulative_ratio": self.cumulative_ratio,
        })

    # -- saying it in words ------------------------------------------------
    def headline(self, k: int = 0) -> str:
        """One sentence about component ``k``, including the bad news."""
        k = self._check(k)
        name = component_name(k)
        share = self.explained_variance_ratio[k]
        feature, dominance = self.dominant(k)
        parts = [f"{name} takes {share:.2%} of the variance"]
        if self.is_degenerate(k):
            parts.append(
                "and that is close enough to nothing that its direction is "
                "not identified — read no loading pattern out of it")
        elif self.is_size_like(k):
            parts.append(
                f"and it is a general-magnitude axis — every feature loads "
                f"the same way, led by {feature}. An axis like this usually "
                f"says the objects differ in size or in focus more than in "
                f"anything else")
        elif dominance >= 0.5:
            parts.append(
                f"and it is mostly {feature} alone ({dominance:.0%} of the "
                f"component), so it is close to that one measurement under "
                f"another name")
        else:
            leaders = ", ".join(
                f"{f} ({v:+.2f})" for f, v in self.top_features(k, 3))
            parts.append(f"led by {leaders}")
        return "; ".join(parts) + "."

    def caveats(self) -> Tuple[str, ...]:
        """Everything a reader needs before believing the picture."""
        out: List[str] = []
        if self.scaling == SCALE_NONE:
            out.append(
                "Features were centred but not scaled, so a component is "
                "dominated by whichever column has the largest raw variance — "
                "usually an area in px². Meaningful only if every feature is "
                "already in the same unit.")
        if self.dropped_rows:
            out.append(
                f"{self.dropped_rows:,} of {self.n_rows_in:,} objects "
                f"({1 - self.row_share:.1%}) were dropped for missing values, "
                f"so this is a PCA of the remaining {len(self):,}. If the "
                f"missingness is structural — a pathogen feature on "
                f"uninfected cells — those objects are a population, not a "
                f"random sample.")
        if self.dropped_features:
            shown = ", ".join(sorted(self.dropped_features)[:6])
            more = (f" (+{len(self.dropped_features) - 6} more)"
                    if len(self.dropped_features) > 6 else "")
            out.append(f"{len(self.dropped_features)} feature(s) were not "
                       f"used: {shown}{more}.")
        if self.n_infinite:
            out.append(
                f"{self.n_infinite:,} non-finite value(s) were treated as "
                f"missing before anything else — an infinity from a ratio "
                f"feature would otherwise set the scale on its own.")
        if self.collinear_groups:
            groups = "; ".join(" = ".join(g) for g in self.collinear_groups[:3])
            out.append(
                f"Perfectly correlated features: {groups}. They are kept, but "
                f"each such group weighs once per member in a standardised "
                f"PCA.")
        if self.rank < self.n_features:
            out.append(
                f"The {self.n_features} features span only {self.rank} "
                f"independent directions, so there are {self.rank} components "
                f"and no more; the rest would be floating-point noise.")
        degenerate = [component_name(i) for i in range(self.n_components)
                      if self.is_degenerate(i)]
        if degenerate:
            out.append(
                f"{', '.join(degenerate)} hold under "
                f"{DEGENERATE_RATIO:.1%} of the variance each. The matrix is "
                f"numerically full rank but effectively is not: those "
                f"directions are rounding error and their loadings are "
                f"arbitrary.")
        return tuple(out)

    def report(self) -> str:
        """The whole story, as the panel shows it and a report file writes it."""
        scaling = ("standardised to unit variance"
                   if self.scaling == SCALE_ZSCORE else "centred only")
        lines = [
            f"PCA of {len(self):,} objects × {self.n_features} features "
            f"({scaling}, NaN policy {self.nan_policy!r}).",
            f"{self.n_components} component(s) shown, holding "
            f"{self.retained_ratio:.1%} of the total variance; "
            f"{self.rank} exist.",
            "",
        ]
        for i in range(self.n_components):
            lines.append("  " + self.headline(i))
        caveats = self.caveats()
        if caveats:
            lines.append("")
            lines.extend("  ! " + c for c in caveats)
        if self.notes:
            lines.append("")
            lines.extend("  · " + n for n in self.notes)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# The computation
# ---------------------------------------------------------------------------

def _select_features(frame: pd.DataFrame, spec: PCASpec
                     ) -> Tuple[List[str], Dict[str, str]]:
    """The requested features that are actually usable, and why the rest are not."""
    wanted = list(spec.features) or list(candidate_features(frame))
    if not wanted:
        raise PCAError(
            "this table has no continuous columns to decompose. PCA needs "
            "measured quantities; plate/row/well identifiers and small "
            "integer codes are deliberately not offered.")
    dropped: Dict[str, str] = {}
    kept: List[str] = []
    for name in wanted:
        if name not in frame.columns:
            dropped[name] = "not a column of this table"
            continue
        kept.append(name)
    if len(kept) < 2:
        raise PCAError(
            f"PCA needs at least two features and got {len(kept)}. A single "
            f"column has one component, which is the column.")
    return kept, dropped


def _apply_nan_policy(matrix: np.ndarray, features: List[str], spec: PCASpec,
                      dropped: Dict[str, str], notes: List[str]
                      ) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """Resolve the NaN policy. Returns ``(matrix, features, kept_row_positions)``.

    ``matrix`` arrives with every non-finite cell already NaN.
    """
    n_rows = matrix.shape[0]
    missing = np.isnan(matrix)
    fraction = (missing.mean(axis=0) if n_rows
                else np.zeros(matrix.shape[1], dtype=float))

    keep_feature = np.ones(matrix.shape[1], dtype=bool)
    if spec.nan_policy == NAN_DROP_FEATURES:
        keep_feature = fraction <= 0.0
        reason = "at least one object has no value for it"
    elif spec.nan_policy == NAN_AUTO:
        keep_feature = fraction <= spec.structural_missing
        reason = None  # per-feature, below
    elif spec.nan_policy == NAN_MEAN:
        # A feature nobody has a value for has no mean to impute with.
        keep_feature = fraction < 1.0
        reason = "no object has a value for it, so there is nothing to impute"
    else:  # NAN_COMPLETE keeps every feature and pays for it in rows.
        reason = None

    for i, name in enumerate(features):
        if keep_feature[i]:
            continue
        if spec.nan_policy == NAN_AUTO:
            dropped[name] = (
                f"missing for {fraction[i]:.1%} of objects — treated as "
                f"structurally absent (a feature of a part that is not "
                f"there) rather than as a gap, so the feature goes and the "
                f"objects stay")
        else:
            dropped[name] = reason or "missing values"
    kept = [name for i, name in enumerate(features) if keep_feature[i]]
    matrix = matrix[:, keep_feature]

    if len(kept) < 2:
        worst = ", ".join(
            f"{features[i]} ({fraction[i]:.0%} missing)"
            for i in np.argsort(fraction)[::-1][:4])
        raise PCAError(
            f"the NaN policy {spec.nan_policy!r} left {len(kept)} feature(s), "
            f"which is not a PCA. The features missing most often are: "
            f"{worst}. Try nan_policy={NAN_COMPLETE!r} to keep the features "
            f"and drop the objects instead.")

    if spec.nan_policy == NAN_MEAN and matrix.size:
        holes = np.isnan(matrix)
        if holes.any():
            means = np.nanmean(matrix, axis=0)
            matrix = np.where(holes, means[None, :], matrix)
            notes.append(
                f"{int(holes.sum()):,} missing value(s) replaced by their "
                f"feature's mean. Every imputed object now sits at the centre "
                f"of that feature's distribution whether or not it belongs "
                f"there.")

    rows_kept = ~np.isnan(matrix).any(axis=1) if matrix.size else \
        np.ones(n_rows, dtype=bool)
    positions = np.arange(n_rows)[rows_kept]
    if positions.size < 2:
        holes = np.isnan(matrix).mean(axis=0)
        blame = ", ".join(f"{kept[i]} ({holes[i]:.0%} missing)"
                          for i in np.argsort(holes)[::-1][:4])
        if positions.size == 0:
            raise PCAError(
                f"no object has a value for every one of the {len(kept)} "
                f"features, so complete-case PCA has nothing to decompose. "
                f"Worst offenders: {blame}. nan_policy={NAN_AUTO!r} drops "
                f"the structurally-missing features instead of the objects.")
        raise PCAError(
            f"only {positions.size} object has a value for every feature; "
            f"PCA needs at least two. Worst offenders: {blame}.")
    return matrix[rows_kept], kept, positions


def _drop_constant(matrix: np.ndarray, features: List[str],
                   dropped: Dict[str, str]
                   ) -> Tuple[np.ndarray, List[str], np.ndarray, np.ndarray]:
    """Remove columns with no variation. Returns matrix, features, mean, sd."""
    mean = matrix.mean(axis=0)
    sd = matrix.std(axis=0, ddof=1)
    tolerance = np.maximum(np.abs(mean), 1.0) * CONSTANT_TOLERANCE
    constant = ~np.isfinite(sd) | (sd <= tolerance)
    for i, name in enumerate(features):
        if constant[i]:
            dropped[name] = (
                f"constant at {mean[i]:.6g} over the analysed objects — a "
                f"column with one value cannot explain variation, and "
                f"standardising it is a division by zero")
    if constant.all():
        raise PCAError(
            "every selected feature is constant over the analysed objects, "
            "so there is no variance to decompose.")
    kept = [name for i, name in enumerate(features) if not constant[i]]
    if len(kept) < 2:
        raise PCAError(
            f"only {len(kept)} feature varies over the analysed objects "
            f"({', '.join(kept)}); PCA needs at least two.")
    keep = ~constant
    return matrix[:, keep], kept, mean[keep], sd[keep]


def _collinear_groups(standard: np.ndarray, features: List[str],
                      notes: List[str]) -> Tuple[Tuple[str, ...], ...]:
    """Groups of features that are perfectly correlated with one another.

    Reported, never dropped: which of ``area`` and ``area_um2`` to discard is
    the user's call, and silently removing one would make the loadings
    disagree with the feature list the user is reading.
    """
    n, p = standard.shape
    if p < 2 or n < 3:
        return ()
    if p > _COLLINEAR_SCAN_LIMIT:
        notes.append(
            f"the collinearity scan was skipped: it is O(p²) and there are "
            f"{p} features")
        return ()
    sd = standard.std(axis=0, ddof=1)
    safe = np.where(sd > 0, sd, 1.0)
    corr = (standard.T @ standard) / (n - 1) / np.outer(safe, safe)
    groups: List[Tuple[str, ...]] = []
    assigned = np.zeros(p, dtype=bool)
    for i in range(p):
        if assigned[i]:
            continue
        partners = np.flatnonzero(
            (np.abs(corr[i]) >= COLLINEAR_TOLERANCE) & ~assigned)
        if partners.size > 1:
            assigned[partners] = True
            groups.append(tuple(features[j] for j in partners))
        else:
            assigned[i] = True
    return tuple(groups)


def _decompose(standard: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """``(singular_values, right_vectors)`` of the standardised matrix.

    Two routes to the same numbers. A full SVD is the accurate one and is used
    whenever it fits; above :data:`_SVD_ROW_LIMIT` rows its ``U`` alone is
    gigabytes, so the p×p Gram matrix is eigen-decomposed instead. The scores
    are ``X @ V`` either way, so nothing downstream can tell which ran.
    """
    n, p = standard.shape
    if n <= _SVD_ROW_LIMIT or p >= n:
        _u, singular, vt = np.linalg.svd(standard, full_matrices=False)
        return singular, vt.T
    gram = standard.T @ standard
    values, vectors = np.linalg.eigh(gram)
    order = np.argsort(values)[::-1]
    values = np.clip(values[order], 0.0, None)
    return np.sqrt(values), vectors[:, order]


def pca(frame: pd.DataFrame, spec: Optional[PCASpec] = None) -> PCAResult:
    """Decompose ``frame``'s features under ``spec``.

    The whole policy is in the module docstring; the short version is that
    features are standardised unless told otherwise, NaN is never imputed
    unless told to, constant columns are dropped by name, collinear ones are
    kept and reported, and no more components are returned than the data has
    independent directions.

    :raises PCAError: whenever the answer would be meaningless — with the
        reason and the way out in the message.
    """
    spec = spec or PCASpec()
    notes: List[str] = []
    features, dropped = _select_features(frame, spec)
    n_rows_in = int(len(frame))
    n_features_in = len(features)
    if n_rows_in < 2:
        raise PCAError(
            f"PCA needs at least two objects and this table has {n_rows_in}.")

    numeric = frame[features].apply(pd.to_numeric, errors="coerce")
    matrix = numeric.to_numpy(dtype=float, copy=True)
    non_finite = ~np.isfinite(matrix)
    n_infinite = int((non_finite & ~np.isnan(matrix)).sum())
    if n_infinite:
        matrix[non_finite] = np.nan

    matrix, features, positions = _apply_nan_policy(
        matrix, features, spec, dropped, notes)
    dropped_rows = n_rows_in - int(positions.size)

    matrix, features, mean, sd = _drop_constant(matrix, features, dropped)
    scale = sd if spec.scaling == SCALE_ZSCORE else np.ones_like(sd)
    standard = (matrix - mean) / scale

    n, p = standard.shape
    singular, vectors = _decompose(standard)
    largest = float(singular.max()) if singular.size else 0.0
    if largest <= 0:  # pragma: no cover - constants were already removed
        raise PCAError("the analysed matrix has no variance left to decompose.")
    tolerance = largest * max(n, p) * float(np.finfo(float).eps)
    rank = int((singular > tolerance).sum())
    if rank < 1:  # pragma: no cover - implied by largest > 0
        raise PCAError("the analysed matrix has rank 0.")

    k = max(1, min(int(spec.n_components), rank))
    loadings = np.asarray(vectors[:, :k], dtype=float)
    scores = standard @ loadings

    # SVD signs are arbitrary; pin them so the same data draws the same
    # picture every time and a figure in a report matches the screen it came
    # from. Convention: the largest-magnitude loading of each component is
    # positive.
    for i in range(k):
        lead = int(np.argmax(np.abs(loadings[:, i])))
        if loadings[lead, i] < 0:
            loadings[:, i] *= -1.0
            scores[:, i] *= -1.0

    variance = (singular ** 2) / (n - 1)
    total = float(variance.sum())
    ratio = (variance[:k] / total) if total > 0 else np.zeros(k)

    # Feature-component correlations, computed rather than derived, so the
    # arrows in the biplot cannot drift out of step with the scores.
    column_sd = standard.std(axis=0, ddof=1)
    score_sd = scores.std(axis=0, ddof=1)
    denominator = np.outer(column_sd, score_sd) * (n - 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        correlations = np.where(denominator > 0,
                                (standard.T @ scores) / denominator, 0.0)
    correlations = np.clip(np.nan_to_num(correlations), -1.0, 1.0)

    groups = _collinear_groups(standard, features, notes)
    if rank < p:
        notes.append(
            f"rank {rank} out of {p} features: {p - rank} direction(s) are "
            f"redundant, so components past {component_name(rank - 1)} do not "
            f"exist")

    return PCAResult(
        features=tuple(features), rows=positions, scores=scores,
        loadings=loadings, correlations=correlations,
        explained_variance=variance[:k], explained_variance_ratio=ratio,
        total_variance=total, centre=mean, scale=scale, rank=rank,
        n_rows_in=n_rows_in, n_features_in=n_features_in,
        scaling=spec.scaling, nan_policy=spec.nan_policy,
        dropped_features=dict(dropped), dropped_rows=dropped_rows,
        n_infinite=n_infinite, collinear_groups=groups, notes=tuple(notes))
