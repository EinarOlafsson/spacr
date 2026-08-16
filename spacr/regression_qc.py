"""Quality-control figures for the regression step of a pooled CRISPR screen.

Why this exists
---------------
:func:`spacr.ml.regression` fits a model to one row per well and emits a
volcano plot. A volcano plot shows which coefficients are large and small-*p*;
it shows nothing at all about whether the fit those numbers came out of is
trustworthy. Every failure mode that has actually cost this project a screen is
invisible on a volcano:

* a handful of 20-cell wells with leverage 0.4 each, dragging a gene's
  coefficient wherever they like;
* a plate whose column 1 and column 24 are systematically dim, so the "hits"
  are the genes that happened to be plated on the edge;
* a design matrix in which ``gene`` and ``grna`` are nearly aliased, so the
  standard errors are 30x too small and *every* p-value is spuriously tiny;
* a Poisson fit on data whose variance is six times its mean, which makes every
  interval too narrow by a factor of 2.5;
* a logistic fit that is badly calibrated, so the predicted fractions are
  systematically wrong even though the ranking is fine.

Each of those is obvious in one glance at the right panel, and each of them
produces a *believable* volcano plot. That asymmetry — silent, plausible
garbage — is the documented failure mode of this pipeline, so the diagnostics
are not optional decoration; they are the thing that says "do not believe this
run".

What this module does NOT do
----------------------------
It does not duplicate the volcano plot (:func:`spacr.plot.volcano_plot` and
:func:`spacr.toxo.custom_volcano_plot` already draw it, and a second
implementation would drift). The report carries a panel that names where the
volcano was written instead. It also never *changes* a fit: nothing here drops
a well, refits, or reweights. It reports; the user decides.

Degrading by model type
-----------------------
``spacr.ml.regression_model`` can return a statsmodels ``OLS``, ``WLS``,
``RLM``, ``QuantReg``, a ``GLM`` in any of six families, a ``MixedLM``, a
``BetaModel``, a sklearn ``Lasso``/``Ridge``/``ElasticNet`` or a ``LinearSVC``.
Those objects agree on almost nothing: a sklearn ``Lasso`` has no p-values, no
covariance matrix and no ``fittedvalues``; a ``MixedLM`` has no hat matrix; a
Gaussian model has no calibration curve worth drawing. Every panel therefore
either draws or raises :class:`PanelUnavailable` carrying the *reason*, and the
reason is printed on the combined report page and returned in the manifest. A
panel is never silently omitted, and never faked with a substitute statistic
that does not mean what the axis label says.

The trap here is ``model.scale``. Every statsmodels results object has one, and
it means a *different thing* on each of them — see
:func:`resolve_residual_standardisation`:

* ``OLS`` / ``MixedLM``: the error variance, in the metric of ``y - fitted``;
* ``WLS``: the error variance in the metric of ``sqrt(w) * (y - fitted)``,
  which for cell-count weights is hundreds of times larger than the unweighted
  residual variance;
* ``RLM`` (``rlm``/``huber``): a robust estimate of the standard DEVIATION, so
  the variance is its *square*;
* ``QuantReg``: the constant ``1.0``, a placeholder — quantile regression has
  no error-variance parameter at all;
* ``BetaModel``: the constant ``1.0``, which is correct only against the
  *Pearson* residual, never against ``y - fitted``;
* sklearn estimators: absent.

Treating all of those as "the error variance of ``y - fitted``" is how a QC
panel reports the wrong wells as outliers, in a direction that depends on the
units of the response. Standardisation therefore goes through one registry
keyed on the fitted model's class, and where no correct scale exists the six
panels that need one are SKIPPED with that stated on the report.

Reading the output
------------------
``regression_qc_report(...)`` writes into ``<dst>/regression_qc/``:

* one file per panel (``residuals_vs_fitted.pdf`` and friends),
* ``regression_qc_report.pdf`` — every panel on one page, skipped panels shown
  as a grey box stating why,
* ``regression_qc_report.txt`` — the same thing as text, with the numbers,
  so it can be grepped or pasted into a lab notebook,

and returns a manifest dict listing every panel, its status, its path and the
statistics it computed. The statistics are the point: the manifest is what a
caller (or a test) reads to find out that Cook's distance flagged well
``plate1_r3_c11``, not just that a PDF exists.

Figures are built through ``matplotlib.figure.Figure`` directly rather than
through ``pyplot``. That is deliberate: a Figure that pyplot never sees cannot
be leaked into pyplot's global registry, cannot be picked up by a later
``plt.savefig()`` in another module, and needs no ``plt.close()`` discipline to
stay out of the way. This repo has been bitten by figure leaks more than once;
this module cannot leak one.
"""
from __future__ import annotations

import os
import re
import textwrap
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from . import schema
from .figures.style import (ROLES, TRANSPARENT, TYPE_SCALE, WEIGHTS, annotate,
                            descriptor, figure_style, reference_line,
                            resolve_ink, rotate_ticks, text_legend,
                            theme_target)

__all__ = [
    "PANEL_ORDER",
    "PanelUnavailable",
    "QCPanelResult",
    "RegressionQCContext",
    "ResidualStandardisation",
    "build_context",
    "calibration_curve",
    "condition_number",
    "condition_verdict",
    "cooks_distance",
    "dffits",
    "diagnose_p_value_histogram",
    "draw_panel",
    "format_qc_report",
    "leverage_from_design",
    "overdispersion_statistic",
    "panel_names",
    "regression_qc_report",
    "resolve_residual_standardisation",
    "variance_inflation_factors",
]

#: Sub-directory of the run's results folder that the report is written to.
QC_DIRNAME = "regression_qc"

#: Colours. Deliberately not seaborn: seaborn styling is global state and this
#: module runs inside a pipeline that also draws with the caller's rcParams.
_POINT = "#1f6f8b"
_ACCENT = "#d1495b"
_GUIDE = "#8d99ae"
_TREND = "#e07a3f"
_OK = "#2a9d8f"

#: Cook's-distance cut-off drawn on the figure. 4/n is the conventional
#: screening rule (Bollen & Jackman); the stricter D > 1 rule almost never
#: fires on well-level screen data, where n is a few hundred, so 4/n is the one
#: that actually separates "this well is influential" from "this well is not".
_COOKS_RULE = 4.0

#: Leverage guides: 2p/n is the standard "high leverage" rule of thumb, 3p/n
#: the "definitely look at this" one.
_LEVERAGE_RULES = (2.0, 3.0)

#: Condition-number interpretation bands (Belsley, Kuh & Welsch), keyed on the
#: column-scaled condition number of the design matrix.
_CONDITION_BANDS = (
    (10.0, "no collinearity problem"),
    (30.0, "weak dependency between predictors"),
    (100.0, "moderate to strong collinearity — standard errors are inflated"),
    (1000.0, "severe collinearity — coefficients are not separately identified"),
)


class PanelUnavailable(Exception):
    """A panel cannot be computed from this model, for a stated reason.

    Raised by panel drawing functions and caught by
    :func:`regression_qc_report`, which records the reason on the report rather
    than dropping the panel. The message is user-facing prose: it must say what
    was missing and, where there is one, what to do instead.

    :param reason: Why the panel cannot be drawn.
    """


@dataclass
class QCPanelResult:
    """One panel's outcome.

    :param name: Stable machine name (also the file stem).
    :param title: Human-readable panel title.
    :param group: Report section: ``'fit'``, ``'influence'``, ``'design'``,
        ``'response'`` or ``'screen'``.
    :param status: ``'written'`` (drawn, nothing missing), ``'partial'``
        (drawn, but with a stated limitation — e.g. a coefficient plot with no
        confidence intervals), ``'skipped'`` (not computable, see ``reason``)
        or ``'failed'`` (raised unexpectedly, see ``reason``).
    :param path: Absolute path of the per-panel figure, or ``None``.
    :param reason: Why the panel was skipped, or what limits it.
    :param stats: Numbers the panel computed, for callers and tests.
    """

    name: str
    title: str
    group: str
    status: str
    path: Optional[str] = None
    reason: Optional[str] = None
    stats: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RegressionQCContext:
    """Everything the panels need, normalised across model types.

    Built by :func:`build_context`; panels only ever read it. Holding the
    normalisation in one place is what keeps twenty panels from each having
    their own opinion about where the residuals of a ``MixedLM`` live.

    :param model: The fitted model object (statsmodels results or sklearn
        estimator).
    :param X: Design matrix as a DataFrame, one row per well.
    :param y: Response, 1-D, aligned to ``X``.
    :param fitted: Fitted values on the response scale.
    :param resid: ``y - fitted`` (response-scale residuals).
    :param std_resid: Internally studentised residuals, or all-``NaN`` when
        this model class has no error scale (see
        :func:`resolve_residual_standardisation`). Panels must ask
        ``standardisation.available`` rather than test for ``NaN``.
    :param leverage: Diagonal of the hat matrix, one entry per well.
    :param leverage_source: How ``leverage`` was obtained, so a panel can say
        so on the axis.
    :param scale: Error VARIANCE used to standardise the residuals, in the
        metric of ``standardisation.base`` — which is not always ``y -
        fitted``. ``NaN`` when no correct scale exists for this model class.
    :param standardisation: The :class:`ResidualStandardisation` that produced
        ``std_resid``: what was standardised, where the variance came from, or
        the reason there is none.
    :param prediction_note: Set when the fitted values are not a conditional
        mean of ``y`` — a hinge/SVM classifier predicts class labels — so the
        response-scale panels can state that on the report instead of quoting
        an R² that means nothing.
    :param decision_score: The continuous score wells are RANKED by, which is
        not always ``fitted``: a hinge/SVM's ``fitted`` is the hard 0/1 label,
        and an ROC computed on hard labels has exactly two operating points and
        understates the model. ``build_context`` sets this to
        ``decision_function(X)`` for a classifier and to ``fitted`` for
        everything else, oriented so that LARGER means MORE LIKELY POSITIVE for
        both.
    :param labels: Per-well labels (``prc`` where available), used to name
        outliers on the influence panels.
    :param weights: Per-well weights passed to the fit (cell counts, for the
        GLM-binomial path), or ``None``.
    :param metadata: Per-well metadata (``plateID``/``rowID``/``columnID``/
        ``cell_count``/``prc``), or ``None``.
    :param coef_df: The coefficient table built by
        :func:`spacr.ml.process_model_coefficients`, or ``None``.
    :param regression_type: The spaCR regression type string, if known.
    :param family: Name of the GLM family, or ``'Gaussian (least squares)'``.
    :param link: Name of the link function, when there is one.
    :param volcano_path: Where the volcano plot for this run was written.
    :param notes: Free-text notes accumulated while building the context.
    """

    model: Any
    X: pd.DataFrame
    y: np.ndarray
    fitted: np.ndarray
    resid: np.ndarray
    std_resid: np.ndarray
    leverage: np.ndarray
    leverage_source: str
    scale: float
    labels: np.ndarray
    standardisation: Optional["ResidualStandardisation"] = None
    prediction_note: Optional[str] = None
    decision_score: Optional[np.ndarray] = None
    weights: Optional[np.ndarray] = None
    metadata: Optional[pd.DataFrame] = None
    coef_df: Optional[pd.DataFrame] = None
    regression_type: Optional[str] = None
    family: str = "unknown"
    link: Optional[str] = None
    volcano_path: Optional[str] = None
    notes: List[str] = field(default_factory=list)

    @property
    def n(self) -> int:
        """Number of observations (wells) in the fit."""
        return int(self.X.shape[0])

    @property
    def p(self) -> int:
        """Number of columns in the design matrix, intercept included."""
        return int(self.X.shape[1])

    @property
    def standardisation_available(self) -> bool:
        """True when ``std_resid`` means what its name says for this model."""
        return bool(self.standardisation is not None
                    and self.standardisation.available)

    @property
    def is_binomial(self) -> bool:
        """True when the response is a probability/fraction under a binomial family."""
        return (self.family in ("Binomial", "QuasiBinomial")
                or (self.regression_type or "") in ("logit", "probit",
                                                    "quasi_binomial"))

    @property
    def is_count(self) -> bool:
        """True when the response is modelled as a count (Poisson / negative binomial)."""
        return (self.family in ("Poisson", "NegativeBinomial")
                or (self.regression_type or "") == "poisson")

    @property
    def is_classifier(self) -> bool:
        """True when the fit is a discriminative classifier (``hinge``).

        Separate from :attr:`is_binomial`, which asks whether a *binomial
        likelihood* was fitted. A hinge/SVM has no likelihood at all, so it is
        not binomial — but it is the one model spaCR offers whose whole output
        is a discrimination, which is what ROC and precision-recall measure.
        Excluding it from those two panels, which is what asking only
        :attr:`is_binomial` did, left the classifier as the single model type
        with no discrimination QC.
        """
        return ((self.regression_type or "") == "hinge"
                or hasattr(self.model, "decision_function"))

    @property
    def is_binary_response(self) -> bool:
        """True when every response value is exactly 0 or 1.

        spaCR routes ``logit``/``probit`` through GLM-Binomial with a
        *continuous* fraction response weighted by cell count, so a binomial
        family does **not** imply binary labels — and ROC/PR are undefined
        without them. This is the distinction that decides it.
        """
        finite = self.y[np.isfinite(self.y)]
        return finite.size > 0 and bool(np.all((finite == 0) | (finite == 1)))

    @property
    def ranking_score(self) -> np.ndarray:
        """The score ROC and precision-recall rank wells by.

        :attr:`decision_score` when the context carries one, otherwise
        :attr:`fitted`. Larger is more likely positive in both cases; see
        :attr:`decision_score`.
        """
        return self.fitted if self.decision_score is None else self.decision_score


# ---------------------------------------------------------------------------
# Statistics — public because they are what the tests assert on, and because
# each is useful on its own from a notebook.
# ---------------------------------------------------------------------------


def leverage_from_design(X, weights=None):
    """Return the hat-matrix diagonal computed from a design matrix.

    ``h_i = x_i' (X' W X)^+ x_i * w_i``. The pseudo-inverse is used rather than
    the inverse because screen design matrices are routinely rank deficient
    (a gRNA present in exactly one well produces a column that is a multiple of
    another); a ``LinAlgError`` there would take the whole QC report down for a
    property of the data that the report exists to show.

    :param X: Design matrix, ``(n, p)``, array-like.
    :param weights: Optional per-observation weights (IRLS weights, or the
        ``var_weights`` spaCR passes for the cell-count-weighted binomial fit).
    :returns: 1-D array of length ``n``, each entry in ``[0, 1]``.

    Example:
        >>> import numpy as np
        >>> X = np.column_stack([np.ones(4), [0., 0., 0., 10.]])
        >>> h = leverage_from_design(X)
        >>> bool(h[3] > h[0])          # the far-out point has high leverage
        True
        >>> bool(abs(h.sum() - 2) < 1e-9)   # trace(H) == p for a full-rank X
        True
    """
    Xm = np.asarray(X, dtype=float)
    if Xm.ndim != 2:
        raise ValueError(f"design matrix must be 2-D, got shape {Xm.shape}")
    if weights is None:
        gram = Xm.T @ Xm
        hat = np.einsum("ij,jk,ik->i", Xm, np.linalg.pinv(gram), Xm)
    else:
        w = np.asarray(weights, dtype=float).ravel()
        if w.size != Xm.shape[0]:
            raise ValueError(
                f"weights has {w.size} entries but the design matrix has "
                f"{Xm.shape[0]} rows")
        gram = Xm.T @ (Xm * w[:, None])
        hat = w * np.einsum("ij,jk,ik->i", Xm, np.linalg.pinv(gram), Xm)
    # Numerically h can come out a hair outside [0, 1]; clipping keeps
    # sqrt(1 - h) real. Anything materially outside the interval is a bug, so
    # only the float noise is absorbed.
    if np.any(hat < -1e-6) or np.any(hat > 1 + 1e-6):
        raise ValueError(
            "hat-matrix diagonal outside [0, 1]; the design matrix or the "
            "weights are not what they claim to be")
    return np.clip(hat, 0.0, 1.0)


def cooks_distance(std_resid, leverage, n_params):
    """Return Cook's distance per observation.

    Uses the identity ``D_i = r_i^2 / p * h_i / (1 - h_i)`` where ``r_i`` is the
    *internally studentised* residual. Written this way it needs no second pass
    over the data and works for any model for which a studentised residual and
    a leverage exist — including the GLM families, where the textbook
    ``e_i^2`` form would be on the wrong scale.

    :param std_resid: Internally studentised residuals, length ``n``.
    :param leverage: Hat-matrix diagonal, length ``n``.
    :param n_params: Number of estimated parameters ``p`` (intercept included).
    :returns: 1-D array of length ``n``; ``inf`` where ``h_i == 1`` (an
        observation the model fits exactly, which is maximally influential by
        construction).

    Example:
        >>> import numpy as np
        >>> d = cooks_distance(np.array([0.1, 4.0]), np.array([0.2, 0.2]), 2)
        >>> bool(d[1] > 100 * d[0])
        True
    """
    r = np.asarray(std_resid, dtype=float)
    h = np.asarray(leverage, dtype=float)
    if r.shape != h.shape:
        raise ValueError(
            f"std_resid has shape {r.shape} but leverage has shape {h.shape}")
    if n_params <= 0:
        raise ValueError(f"n_params must be positive, got {n_params}")
    with np.errstate(divide="ignore", invalid="ignore"):
        d = (r ** 2 / float(n_params)) * (h / (1.0 - h))
    return np.where(np.isclose(h, 1.0), np.inf, d)


def dffits(std_resid, leverage, n_obs, n_params):
    """Return DFFITS per observation, the change in that observation's own fit.

    ``DFFITS_i = t_i * sqrt(h_i / (1 - h_i))`` with ``t_i`` the *externally*
    studentised residual, obtained from the internally studentised one by
    ``t_i = r_i * sqrt((n - p - 1) / (n - p - r_i^2))``. The conventional
    threshold is ``2 * sqrt(p / n)``.

    :param std_resid: Internally studentised residuals.
    :param leverage: Hat-matrix diagonal.
    :param n_obs: Number of observations.
    :param n_params: Number of estimated parameters.
    :returns: ``(dffits, threshold)`` — the per-observation values (``nan``
        where the external studentisation is undefined) and the threshold.
    """
    r = np.asarray(std_resid, dtype=float)
    h = np.asarray(leverage, dtype=float)
    dof = float(n_obs - n_params - 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        denom = float(n_obs - n_params) - r ** 2
        t = np.where(denom > 0, r * np.sqrt(np.maximum(dof, 0.0) / denom), np.nan)
        value = t * np.sqrt(h / (1.0 - h))
    threshold = 2.0 * np.sqrt(float(n_params) / float(n_obs)) if n_obs else np.nan
    return value, threshold


def variance_inflation_factors(X, tol=1e-10):
    """Return the VIF of every non-constant column of a design matrix.

    Computed from the inverse of the predictor *correlation* matrix — the
    identity ``VIF_j = (R^-1)_jj`` — rather than by running ``p`` auxiliary
    regressions. On a screen design with 1,200 gRNA columns the auxiliary-
    regression route is ``O(p^4)`` and simply does not finish; the correlation
    route is one ``O(p^3)`` decomposition. The two agree exactly when the
    design contains an intercept (see the test that pins this against
    ``statsmodels.stats.outliers_influence.variance_inflation_factor``).

    Constant columns (the intercept, and any predictor that is constant on the
    rows that survived cleaning) have no VIF — a variance of zero cannot be
    inflated — and are reported as ``NaN`` rather than dropped, so the caller
    can see that they were there.

    Exactly collinear columns get ``inf``: they are identified by the near-null
    eigenvectors of the correlation matrix, so *which* columns are aliased is
    reported rather than the whole matrix being declared unusable.

    :param X: Design matrix; DataFrame or array-like.
    :param tol: Relative eigenvalue below which a direction counts as null.
    :returns: ``pandas.Series`` of VIFs indexed by column name.

    Example:
        >>> import numpy as np, pandas as pd
        >>> rng = np.random.default_rng(0)
        >>> a = rng.normal(size=200)
        >>> df = pd.DataFrame({'a': a, 'b': a + 0.01 * rng.normal(size=200),
        ...                    'c': rng.normal(size=200)})
        >>> vif = variance_inflation_factors(df)
        >>> bool(vif['a'] > 100), bool(vif['c'] < 2)
        (True, True)
    """
    frame = X if isinstance(X, pd.DataFrame) else pd.DataFrame(
        np.asarray(X, dtype=float),
        columns=[f"x{i}" for i in range(np.asarray(X).shape[1])])
    frame = frame.astype(float)
    out = pd.Series(np.nan, index=frame.columns, dtype=float)

    std = frame.std(ddof=1)
    varying = [c for c in frame.columns if np.isfinite(std[c]) and std[c] > 0]
    if len(varying) < 2:
        # One varying predictor cannot be collinear with anything.
        for col in varying:
            out[col] = 1.0
        return out

    corr = np.corrcoef(frame[varying].to_numpy(dtype=float), rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0)
    eigvals, eigvecs = np.linalg.eigh(corr)
    largest = float(np.max(eigvals))
    null_mask = eigvals <= tol * max(largest, 1.0)

    values = np.diag(np.linalg.pinv(corr)).astype(float).copy()
    if np.any(null_mask):
        # A direction with (essentially) zero variance is an exact linear
        # dependency; every column with weight in it is unidentified, and its
        # VIF is infinite, not the finite number pinv would hand back.
        loading = np.abs(eigvecs[:, null_mask]).max(axis=1)
        values[loading > 1e-6] = np.inf
    for col, value in zip(varying, values):
        out[col] = float(value)
    return out


def condition_number(X):
    """Return the condition number of a design matrix, scaled and unscaled.

    Two numbers, because they answer different questions. The *unscaled*
    condition number is what ``statsmodels`` prints in a summary, and it is
    dominated by the units of the columns: a predictor measured in cells rather
    than thousands of cells changes it by 1000 with no change in the science.
    The *scaled* one (each column normalised to unit length first, the
    Belsley-Kuh-Welsch definition) is unit-free and is the one whose thresholds
    — 30, 100, 1000 — mean anything.

    :param X: Design matrix, array-like ``(n, p)``.
    :returns: ``(scaled, unscaled, singular_values)`` where ``singular_values``
        are those of the column-scaled matrix, largest first.

    Example:
        >>> import numpy as np
        >>> ortho = np.eye(3)
        >>> round(condition_number(ortho)[0], 6)
        1.0
    """
    Xm = np.asarray(X, dtype=float)
    if Xm.ndim != 2 or Xm.size == 0:
        raise ValueError(f"design matrix must be a non-empty 2-D array, got {Xm.shape}")
    norms = np.linalg.norm(Xm, axis=0)
    # A genuinely all-zero column has no direction; scaling it by 1 leaves it
    # zero, which is exactly what makes the matrix singular, and the singular
    # value of 0 that follows is the honest answer.
    norms = np.where(norms > 0, norms, 1.0)
    scaled_sv = np.linalg.svd(Xm / norms, compute_uv=False)
    raw_sv = np.linalg.svd(Xm, compute_uv=False)

    def _ratio(sv):
        if sv.size == 0 or sv[-1] <= 0:
            return np.inf
        return float(sv[0] / sv[-1])

    return _ratio(scaled_sv), _ratio(raw_sv), scaled_sv


def condition_verdict(scaled_condition_number):
    """Return the plain-English reading of a scaled condition number.

    :param scaled_condition_number: Output of :func:`condition_number`.
    :returns: A short sentence naming the severity band.
    """
    if not np.isfinite(scaled_condition_number):
        return "design matrix is singular — at least one predictor is an exact combination of others"
    for bound, text in _CONDITION_BANDS:
        if scaled_condition_number < bound:
            return text
    return _CONDITION_BANDS[-1][1]


def calibration_curve(y_true, y_pred, n_bins=10, weights=None, strategy="quantile"):
    """Bin predictions and return the observed frequency in each bin.

    Works for a binary response *and* for the continuous per-well fraction that
    spaCR's ``logit``/``probit`` path actually fits: in both cases the question
    is "of the wells where the model said 0.3, what fraction were positive?".
    With ``weights`` (cell counts) the observed value is the weighted mean, so a
    2,000-cell well is not given the same say as a 20-cell one.

    :param y_true: Observed response in ``[0, 1]``.
    :param y_pred: Predicted probability in ``[0, 1]``.
    :param n_bins: Number of bins. Default 10.
    :param weights: Optional per-observation weights.
    :param strategy: ``'quantile'`` (equal counts per bin, the default —
        robust when predictions pile up) or ``'uniform'`` (equal width).
    :returns: dict with ``pred_mean``, ``obs_mean``, ``counts``, ``weight``,
        ``ece`` (weighted mean absolute gap), ``max_gap`` and ``brier``.
    :raises ValueError: if the inputs disagree in length or ``n_bins < 2``.

    Example:
        >>> import numpy as np
        >>> p = np.linspace(0.02, 0.98, 500)
        >>> rng = np.random.default_rng(0)
        >>> y = (rng.uniform(size=500) < p).astype(float)
        >>> out = calibration_curve(y, p, n_bins=5)
        >>> bool(out['ece'] < 0.1)          # a calibrated model hugs y = x
        True
    """
    yt = np.asarray(y_true, dtype=float).ravel()
    yp = np.asarray(y_pred, dtype=float).ravel()
    if yt.size != yp.size:
        raise ValueError(f"y_true has {yt.size} entries, y_pred has {yp.size}")
    if n_bins < 2:
        raise ValueError(f"n_bins must be at least 2, got {n_bins}")
    w = (np.ones_like(yt) if weights is None
         else np.asarray(weights, dtype=float).ravel())
    if w.size != yt.size:
        raise ValueError(f"weights has {w.size} entries, expected {yt.size}")

    keep = np.isfinite(yt) & np.isfinite(yp) & np.isfinite(w) & (w > 0)
    yt, yp, w = yt[keep], yp[keep], w[keep]
    if yt.size == 0:
        raise ValueError("no finite observations left to calibrate")

    if strategy == "quantile":
        edges = np.quantile(yp, np.linspace(0.0, 1.0, n_bins + 1))
        edges = np.unique(edges)
        if edges.size < 3:
            # Predictions are (nearly) constant: quantile edges collapse and
            # every point lands in one bin, which is not a curve. Fall back to
            # equal width over the observed range so the panel still says
            # something true, and let the caller see it in `n_bins`.
            edges = np.linspace(yp.min(), yp.max() + 1e-12, n_bins + 1)
    elif strategy == "uniform":
        edges = np.linspace(0.0, 1.0, n_bins + 1)
    else:
        raise ValueError(f"strategy must be 'quantile' or 'uniform', got {strategy!r}")

    idx = np.clip(np.searchsorted(edges, yp, side="right") - 1, 0, edges.size - 2)
    pred_mean, obs_mean, counts, weight = [], [], [], []
    for b in range(edges.size - 1):
        sel = idx == b
        if not np.any(sel):
            continue
        wb = w[sel]
        pred_mean.append(float(np.average(yp[sel], weights=wb)))
        obs_mean.append(float(np.average(yt[sel], weights=wb)))
        counts.append(int(sel.sum()))
        weight.append(float(wb.sum()))

    pred_mean = np.asarray(pred_mean)
    obs_mean = np.asarray(obs_mean)
    weight = np.asarray(weight)
    gaps = np.abs(obs_mean - pred_mean)
    ece = float(np.average(gaps, weights=weight)) if gaps.size else float("nan")
    return {
        "pred_mean": pred_mean,
        "obs_mean": obs_mean,
        "counts": np.asarray(counts),
        "weight": weight,
        "ece": ece,
        "max_gap": float(gaps.max()) if gaps.size else float("nan"),
        "brier": float(np.average((yp - yt) ** 2, weights=w)),
        "n_bins": int(pred_mean.size),
    }


def overdispersion_statistic(y, mu, df_resid, variance=None, weights=None):
    """Return the Pearson dispersion of a count fit and its verdict.

    ``phi = sum((y - mu)^2 / V(mu)) / df_resid``. Under a correctly specified
    Poisson model ``phi == 1``. ``phi`` well above 1 means the standard errors
    are too small by ``sqrt(phi)`` — at ``phi = 6``, a 2.4-fold inflation, which
    turns noise into a screen full of hits. That is why this number is on the
    report as a number and not as a shape to eyeball.

    :param y: Observed counts.
    :param mu: Fitted means.
    :param df_resid: Residual degrees of freedom (``n - p``).
    :param variance: Callable ``V(mu)``; defaults to the Poisson ``V(mu) = mu``.
    :param weights: Optional per-observation weights.
    :returns: dict with ``dispersion``, ``pearson_chi2``, ``df_resid`` and
        ``verdict``.

    Example:
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> mu = np.full(500, 5.0)
        >>> y = rng.poisson(5.0, size=500).astype(float)
        >>> out = overdispersion_statistic(y, mu, 499)
        >>> bool(0.7 < out['dispersion'] < 1.4)
        True
    """
    yv = np.asarray(y, dtype=float).ravel()
    mv = np.asarray(mu, dtype=float).ravel()
    if yv.size != mv.size:
        raise ValueError(f"y has {yv.size} entries, mu has {mv.size}")
    if df_resid is None or df_resid <= 0:
        raise ValueError(
            f"df_resid must be positive to form a dispersion, got {df_resid!r}")
    var = mv if variance is None else np.asarray(variance(mv), dtype=float).ravel()
    w = (np.ones_like(yv) if weights is None
         else np.asarray(weights, dtype=float).ravel())
    good = np.isfinite(yv) & np.isfinite(mv) & np.isfinite(var) & (var > 0)
    if not np.any(good):
        raise ValueError("no observation has a positive fitted variance")
    chi2 = float(np.sum(w[good] * (yv[good] - mv[good]) ** 2 / var[good]))
    phi = chi2 / float(df_resid)
    if phi > 2.0:
        verdict = ("strongly over-dispersed — refit with negative binomial or "
                   "quasi-Poisson; every interval here is too narrow by "
                   f"{np.sqrt(phi):.1f}x")
    elif phi > 1.5:
        verdict = ("over-dispersed — intervals are too narrow by "
                   f"{np.sqrt(phi):.1f}x")
    elif phi < 0.5:
        verdict = "under-dispersed — intervals are conservative; check for aggregation"
    else:
        verdict = "consistent with the assumed mean-variance relationship"
    return {"dispersion": phi, "pearson_chi2": chi2,
            "df_resid": float(df_resid), "verdict": verdict}


def diagnose_p_value_histogram(p_values, n_bins=20):
    """Classify the shape of a screen's p-value distribution.

    A screen in which most genes do nothing should give p-values that are
    uniform on ``[0, 1]`` with a spike in the first bin (the real hits). Two
    other shapes are diagnostic of a broken fit and are the reason this panel
    exists:

    * a **spike near 1** means the test is conservative — usually a
      variance component soaking up the signal, or duplicated rows inflating n;
    * a **U shape** (both ends enriched) means the null is mis-specified —
      typically unmodelled plate structure, which pushes half the genes one way
      and half the other.

    :param p_values: Iterable of p-values; non-finite entries are dropped.
    :param n_bins: Histogram resolution. Default 20 (bins of width 0.05).
    :returns: dict with ``verdict`` (one of ``'uniform-with-spike'``,
        ``'uniform'``, ``'excess-large'``, ``'u-shaped'``, ``'anti-uniform'``,
        ``'too-few'``), ``message``, ``counts``, ``expected``,
        ``first_bin_ratio``, ``last_bin_ratio`` and ``frac_below_0.05``.

    Example:
        >>> import numpy as np
        >>> rng = np.random.default_rng(1)
        >>> diagnose_p_value_histogram(rng.uniform(size=2000))['verdict']
        'uniform'
        >>> spiky = np.concatenate([rng.uniform(0.9, 1.0, 800),
        ...                         rng.uniform(size=200)])
        >>> diagnose_p_value_histogram(spiky)['verdict']
        'excess-large'
    """
    p = np.asarray(list(p_values), dtype=float).ravel()
    p = p[np.isfinite(p)]
    counts, edges = np.histogram(p, bins=n_bins, range=(0.0, 1.0))
    n = int(counts.sum())
    expected = n / float(n_bins) if n else float("nan")
    out = {
        "counts": counts,
        "edges": edges,
        "n": n,
        "expected": expected,
        "frac_below_0.05": float(np.mean(p <= 0.05)) if n else float("nan"),
    }
    if n < 20:
        out.update(verdict="too-few", message=(
            f"only {n} p-value(s): the shape of this histogram means nothing "
            f"below ~20 coefficients"), first_bin_ratio=float("nan"),
            last_bin_ratio=float("nan"))
        return out

    first = counts[0] / expected
    last = counts[-1] / expected
    # The middle is the reference: it is where the null lives whatever the
    # tails are doing, so comparing the ends to it (rather than to each other)
    # keeps the verdict stable as the number of real hits changes.
    middle = counts[1:-1]
    middle_ratio = float(np.mean(middle) / expected) if middle.size else 1.0
    out["first_bin_ratio"] = float(first)
    out["last_bin_ratio"] = float(last)
    out["middle_ratio"] = middle_ratio

    if last > 1.5 and first > 1.5:
        verdict = "u-shaped"
        message = ("U-shaped: both tails are enriched. The null is "
                   "mis-specified — usually unmodelled plate/row structure. "
                   "Do not read the hit list before fixing the model.")
    elif last > 1.5:
        verdict = "excess-large"
        message = (f"spike near p = 1 ({last:.1f}x uniform): the test is "
                   f"conservative. Check for duplicated wells inflating n, or "
                   f"a random effect absorbing the signal.")
    elif first > 1.5:
        verdict = "uniform-with-spike"
        message = (f"uniform with a spike near 0 ({first:.1f}x uniform): the "
                   f"expected shape for a screen with real hits.")
    elif first < 0.5:
        verdict = "anti-uniform"
        message = ("depleted near p = 0 and flat elsewhere: no signal, and the "
                   "test may be over-conservative.")
    else:
        verdict = "uniform"
        message = ("flat: consistent with no coefficient differing from the "
                   "null. A screen with hits should show a spike in the first "
                   "bin.")
    out["verdict"] = verdict
    out["message"] = message
    return out


# ---------------------------------------------------------------------------
# Residual standardisation — one registry, keyed on the fitted model's class
# ---------------------------------------------------------------------------


@dataclass
class ResidualStandardisation:
    """How a fit's residuals are put on a comparable scale, or why they are not.

    ``std_resid = base / sqrt(variance * (1 - h))``. Both halves of that matter
    and both are model-class-dependent: ``base`` is a Pearson residual for a
    GLM and for beta regression, ``sqrt(w) * (y - fitted)`` for WLS and
    ``y - fitted`` for OLS; ``variance`` is ``model.scale`` for OLS, its
    *square* for RLM, and does not exist at all for quantile regression.

    :param available: True when a correct error scale exists for this fit.
        When it is False, ``base`` and ``variance`` are ``None``/``NaN`` and
        ``reason`` says why — the panels that need a standardised residual
        skip with that reason rather than standardise by a number that
        happens to be there.
    :param metric: What ``base`` is, in words, for the axis and the report.
    :param source: Where ``variance`` came from, in words.
    :param base: The residual that is standardised, length ``n``.
    :param variance: The error variance in the metric of ``base``.
    :param reason: Why no standardisation exists, when ``available`` is False.
    """

    available: bool
    metric: str
    source: str
    base: Optional[np.ndarray] = None
    variance: float = float("nan")
    reason: Optional[str] = None


def _positive_float(value):
    """Return ``value`` as a finite positive float, or ``None``."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) and number > 0 else None


def _residual_variance(resid, n, p):
    """``RSS / (n - p)``, and whether it came out usable.

    Returns ``(variance, exact_fit)``. ``exact_fit`` is True when the residual
    sum of squares is zero (or the degrees of freedom are gone), in which case
    a unit variance is substituted so the influence panels can report the
    saturation instead of dividing by zero.
    """
    dof = max(int(n) - int(p), 1)
    rss = float(np.sum(np.asarray(resid, dtype=float) ** 2))
    variance = _positive_float(rss / dof)
    if variance is None:
        return 1.0, True
    return variance, False


def _fit_weights(model, n):
    """The weights a weighted fit was actually given, or ``None``.

    Taken from the model rather than from the caller: they are what
    ``model.scale`` was formed with, so they are the only weights that make
    ``scale`` and the hat matrix agree with each other.
    """
    inner = getattr(model, "model", None)
    weights = getattr(inner, "weights", None)
    if weights is None:
        return None
    array = np.asarray(weights, dtype=float).ravel()
    if array.size != int(n) or not np.all(np.isfinite(array)) or np.any(array <= 0):
        return None
    return array


def _pearson_base(model, n):
    """``resid_pearson`` as a length-``n`` array, or ``None``."""
    pearson = getattr(model, "resid_pearson", None)
    if pearson is None:
        return None
    array = np.asarray(pearson, dtype=float).ravel()
    return array if array.size == int(n) else None


def _unavailable(reason, metric="none"):
    return ResidualStandardisation(available=False, metric=metric,
                                   source="not available", reason=reason)


def _scale_ols(model, resid, n, p):
    """OLS: ``model.scale`` is ``RSS / (n - p)``, the error variance itself."""
    variance = _positive_float(getattr(model, "scale", None))
    source = "OLS error variance (model.scale = RSS / (n - p))"
    if variance is None:
        variance, exact = _residual_variance(resid, n, p)
        source = ("residual variance RSS / (n - p) recomputed here; the fit "
                  "reports no usable model.scale")
        if exact:
            source = ("unit variance: the fit reproduces every observation "
                      "exactly, so there is no residual variance to divide by")
    return ResidualStandardisation(
        available=True, metric="response-scale residual (y - fitted)",
        source=source, base=np.asarray(resid, dtype=float), variance=variance)


def _scale_wls(model, resid, n, p):
    """WLS: ``model.scale`` is in the metric of ``sqrt(w) * (y - fitted)``.

    ``RegressionResults.scale`` is ``wresid' wresid / df_resid``, and for a WLS
    fit ``wresid`` is ``sqrt(w) * resid``. With spaCR's per-well cell counts
    for ``w`` that is two to three orders of magnitude larger than the
    unweighted residual variance, so the residual has to be weighted to match
    it — not the scale unweighted to match the residual, which would throw
    away the whole point of weighting the fit.
    """
    weights = _fit_weights(model, n)
    if weights is None:
        return _unavailable(
            "this WLS fit does not expose the per-observation weights it was "
            "fitted with, and its model.scale is the error variance of "
            "sqrt(w) * (y - fitted), not of (y - fitted). Standardising "
            "without the weights would rescale every residual by an arbitrary "
            "factor. Refit with spacr.ml.regression_model, which always passes "
            "the cell counts through.")
    variance = _positive_float(getattr(model, "scale", None))
    source = ("WLS error variance (model.scale = sum(w e^2) / (n - p), in the "
              "metric of sqrt(w) * residual)")
    if variance is None:
        dof = max(int(n) - int(p), 1)
        variance = _positive_float(
            float(np.sum(weights * np.asarray(resid, dtype=float) ** 2)) / dof)
        source = "weighted residual variance sum(w e^2) / (n - p) recomputed here"
        if variance is None:
            variance, source = 1.0, (
                "unit variance: the weighted fit reproduces every observation "
                "exactly")
    return ResidualStandardisation(
        available=True,
        metric="weighted residual sqrt(w) * (y - fitted)",
        source=source,
        base=np.sqrt(weights) * np.asarray(resid, dtype=float),
        variance=variance)


def _scale_gls(model, resid, n, p):
    """GLS/GLSAR: ``scale`` lives in a whitened metric this module cannot rebuild.

    ``RegressionResults.scale`` for a GLS fit is the variance of the *whitened*
    residual ``cholsigmainv @ (y - fitted)``, which needs the error covariance
    the caller supplied and which the results object does not hand back in a
    per-observation form. spaCR refuses ``regression_type='gls'`` outright, so
    this branch exists to stop a hand-built GLS fit from being standardised as
    though it were OLS.
    """
    return _unavailable(
        f"{type(getattr(model, 'model', model)).__name__} is a generalised "
        f"least-squares fit: its model.scale is the variance of the WHITENED "
        f"residual, in a metric set by the error covariance passed to the fit, "
        f"and (y - fitted) is not in that metric. spaCR does not fit GLS "
        f"(spacr.ml.UNSUPPORTED_REGRESSION_TYPES says why); use 'ols' with "
        f"cov_type='HC3', 'wls', or 'mixed'.")


def _scale_rlm(model, resid, n, p):
    """RLM / Huber: ``model.scale`` is a standard DEVIATION, not a variance.

    ``RLMResults.scale`` is the robust (MAD by default) estimate of sigma, and
    ``RLMResults.sresid`` is ``resid / scale`` — no square root anywhere.
    Dividing by ``sqrt(scale)`` instead is wrong by a factor of
    ``sqrt(scale)``, which is unit-dependent: it *shrinks* every |z| when the
    response is a fraction and *inflates* every |z| when the response is a
    per-well count.
    """
    sigma = _positive_float(getattr(model, "scale", None))
    if sigma is None:
        return _unavailable(
            "this robust fit reports no usable scale estimate "
            f"(model.scale = {getattr(model, 'scale', None)!r}). RLM's scale "
            "is the robust standard deviation of the residuals; with more "
            "than half the wells fitted exactly the MAD collapses to zero and "
            "there is nothing to standardise by.")
    return ResidualStandardisation(
        available=True, metric="response-scale residual (y - fitted)",
        source=(f"robust scale estimate: RLMResults.scale = {sigma:.6g} is a "
                f"standard deviation, so the variance is its square "
                f"({sigma ** 2:.6g})"),
        base=np.asarray(resid, dtype=float), variance=sigma ** 2)


def _scale_quantreg(model, resid, n, p):
    """Quantile regression: there is no error scale, and ``scale`` is a stub."""
    q = getattr(model, "q", None)
    where = f"the {q:g} quantile" if isinstance(q, float) else "a quantile"
    return _unavailable(
        f"quantile regression estimates {where} of the response, not its "
        f"mean, so it has no error-variance parameter: statsmodels' "
        f"QuantRegResults.scale is hard-coded to 1.0 as a placeholder and is "
        f"not a variance of anything. Its residuals are also asymmetric by "
        f"construction (a fixed fraction of them are negative), so the "
        f"Gaussian-theory studentised residual, Cook's distance and DFFITS "
        f"are undefined here. The residual-vs-fitted, response and design "
        f"panels are unaffected; refit with 'ols' or 'rlm' if you need "
        f"influence diagnostics.")


def _scale_glm(model, resid, n, p):
    """GLM: standardise the Pearson residual by the family's dispersion."""
    base = _pearson_base(model, n)
    if base is None:
        return _unavailable(
            f"{type(model).__name__} reports a GLM family but no per-"
            f"observation Pearson residual, so (y - mu) cannot be put on the "
            f"family's variance scale; standardising on the response scale "
            f"instead would make every well near mu = 0.5 look like an outlier.")
    variance = _positive_float(getattr(model, "scale", None))
    family = type(getattr(model, "family", None)).__name__
    if variance is None:
        return _unavailable(
            f"this {family} GLM reports a non-positive dispersion "
            f"(model.scale = {getattr(model, 'scale', None)!r}), so the "
            f"Pearson residual cannot be standardised.")
    return ResidualStandardisation(
        available=True, metric="Pearson residual (y - mu) / sqrt(V(mu))",
        source=(f"GLM dispersion model.scale = {variance:.6g} "
                f"({family} family; fixed at 1 for Binomial and Poisson, "
                f"estimated otherwise)"),
        base=base, variance=variance)


def _scale_beta(model, resid, n, p):
    """Beta regression: ``scale`` is 1, and it belongs to the Pearson residual.

    ``BetaResults.scale`` is the generic likelihood-model default of ``1.0``.
    That is the right number, but only against
    ``(y - mu) / sqrt(mu (1 - mu) / (1 + phi))`` — never against ``y - mu``,
    whose spread on a fraction response is an order of magnitude smaller.
    """
    base = _pearson_base(model, n)
    if base is None:
        return _unavailable(
            "this statsmodels build's beta-regression results expose no "
            "resid_pearson, and the beta variance mu(1 - mu)/(1 + phi) cannot "
            "be recovered from the results object alone. model.scale is the "
            "generic likelihood default of 1.0 and is not the variance of "
            "(y - mu).")
    return ResidualStandardisation(
        available=True,
        metric="Pearson residual (y - mu) / sqrt(mu(1-mu)/(1+phi))",
        source="beta-regression unit dispersion (the precision phi is already "
               "in the Pearson denominator)",
        base=base, variance=1.0)


def _scale_mixedlm(model, resid, n, p):
    """MixedLM: ``scale`` is the residual variance, and the residuals match it.

    ``MixedLMResults.fittedvalues`` includes the predicted random effects, so
    ``y - fitted`` is the *conditional* residual and ``scale`` is exactly its
    variance. The shrinkage in the BLUPs makes the conditional residual a few
    percent tighter than sigma, which is conservative; the marginal residual
    would be far too wide for this scale, and the report says which one it is.
    """
    variance = _positive_float(getattr(model, "scale", None))
    source = ("MixedLM residual variance (model.scale), matched to residuals "
              "that are CONDITIONAL on the estimated random effects")
    if variance is None:
        variance, exact = _residual_variance(resid, n, p)
        source = "residual variance RSS / (n - p) recomputed here"
        if exact:
            source = "unit variance: the mixed fit reproduces every observation exactly"
    return ResidualStandardisation(
        available=True,
        metric="conditional residual (y - fitted, random effects included)",
        source=source, base=np.asarray(resid, dtype=float), variance=variance)


def _scale_classifier(model, resid, n, p):
    """A classifier predicts labels: there is no error variance to divide by."""
    return _unavailable(
        f"{type(model).__name__} is a classifier: it predicts a class label, "
        f"not a conditional mean, so it has no error variance, no likelihood "
        f"and therefore no standardised residual. Cook's distance and DFFITS "
        f"are least-squares influence measures and do not carry over to a "
        f"hinge loss. spaCR reports the hinge fit's coefficient stability "
        f"through the bootstrap standard errors in the coefficient table "
        f"instead.")


def _scale_estimated(model, resid, n, p):
    """Anything else: estimate the residual variance and say that is what it is."""
    variance, exact = _residual_variance(resid, n, p)
    source = (f"residual variance RSS / (n - p) estimated here; "
              f"{type(model).__name__} reports no dispersion this module "
              f"recognises")
    if exact:
        source = ("unit variance: the fit reproduces every observation "
                  "exactly, so there is no residual variance to divide by")
    return ResidualStandardisation(
        available=True, metric="response-scale residual (y - fitted)",
        source=source, base=np.asarray(resid, dtype=float), variance=variance)


#: Residual standardisation, per fitted-model class. Keyed on a class name
#: looked up along the MRO of the *model* a results object came from (so a
#: subclass resolves to its base, and ``OLS`` — which subclasses ``WLS`` in
#: statsmodels — resolves to ``OLS`` because its own name is found first).
#:
#: A model class that is not in here is not assumed to be least squares: it
#: falls through to :func:`_scale_estimated`, which recomputes the residual
#: variance rather than trusting a ``scale`` attribute whose meaning is unknown.
_SCALE_RESOLVERS: Dict[str, Callable[[Any, np.ndarray, int, int],
                                     ResidualStandardisation]] = {
    "OLS": _scale_ols,
    "WLS": _scale_wls,
    "GLS": _scale_gls,
    "GLSAR": _scale_gls,
    "RLM": _scale_rlm,
    "QuantReg": _scale_quantreg,
    "GLM": _scale_glm,
    "BetaModel": _scale_beta,
    "MixedLM": _scale_mixedlm,
    "ClassifierMixin": _scale_classifier,
}


def _model_kind(model):
    """Return the :data:`_SCALE_RESOLVERS` key for a fitted object.

    statsmodels hands back a *results* object whose class is often shared
    across model types — ``sm.OLS(...).fit()`` and ``sm.WLS(...).fit()`` are
    both a ``RegressionResultsWrapper`` — so the results class cannot tell them
    apart. ``results.model`` can: it is the ``OLS``/``WLS``/``QuantReg``/
    ``RLM``/``GLM``/``MixedLM``/``BetaModel`` instance that was fitted. sklearn
    estimators are their own model, and are matched on their own MRO (which is
    where ``ClassifierMixin`` shows up).

    The MRO is searched against both registries that are keyed this way —
    :data:`_SCALE_RESOLVERS` and :data:`_FAMILY_BY_KIND` — so a class only one
    of them knows about (``Lasso`` has a family name but no scale rule of its
    own) still resolves to its own key rather than to ``None``.

    :param model: Fitted results object or sklearn estimator.
    :returns: ``(key, class_name)`` — the registry key (``None`` when nothing
        matched) and the model class's own name, for messages.

    Example:
        >>> import numpy as np, statsmodels.api as sm
        >>> y = np.array([1.0, 2.0, 3.1, 3.9])
        >>> X = np.column_stack([np.ones(4), [0.0, 1.0, 2.0, 3.0]])
        >>> _model_kind(sm.OLS(y, X).fit())[0]
        'OLS'
        >>> _model_kind(sm.WLS(y, X, weights=np.arange(1, 5.0)).fit())[0]
        'WLS'
    """
    inner = getattr(model, "model", None)
    target = model if inner is None else inner
    for cls in type(target).__mro__:
        if cls.__name__ in _SCALE_RESOLVERS or cls.__name__ in _FAMILY_BY_KIND:
            return cls.__name__, type(target).__name__
    return None, type(target).__name__


def resolve_residual_standardisation(model, resid, n_obs, n_params):
    """Return how this fit's residuals can be standardised, or why they cannot.

    This is the one place that knows what ``model.scale`` means, and it knows
    it per model class rather than per attribute: an attribute that exists is
    not an attribute that means what the caller hoped. See
    :data:`_SCALE_RESOLVERS` for the table and the module docstring for why it
    is not one formula.

    :param model: Fitted statsmodels results object or sklearn estimator.
    :param resid: Response-scale residuals ``y - fitted``, length ``n_obs``.
    :param n_obs: Number of observations.
    :param n_params: Number of columns in the design matrix.
    :returns: :class:`ResidualStandardisation`. Check ``.available`` before
        reading ``.base`` / ``.variance``.

    Example:
        >>> import numpy as np, statsmodels.api as sm
        >>> rng = np.random.default_rng(0)
        >>> X = np.column_stack([np.ones(60), rng.normal(size=60)])
        >>> y = X @ [1.0, 2.0] + rng.normal(size=60)
        >>> fit = sm.RLM(y, X).fit()
        >>> std = resolve_residual_standardisation(fit, y - fit.fittedvalues, 60, 2)
        >>> bool(np.isclose(std.variance, fit.scale ** 2))   # a variance, not an SD
        True
        >>> resolve_residual_standardisation(
        ...     sm.QuantReg(y, X).fit(q=0.5), np.zeros(60), 60, 2).available
        False
    """
    residuals = np.asarray(resid, dtype=float).ravel()
    key, _ = _model_kind(model)
    resolver = _SCALE_RESOLVERS.get(key, _scale_estimated)
    result = resolver(model, residuals, int(n_obs), int(n_params))
    if result.available and result.base is not None:
        base = np.asarray(result.base, dtype=float).ravel()
        if base.size != residuals.size:
            return _unavailable(
                f"the standardisation for {type(model).__name__} produced "
                f"{base.size} residuals for {residuals.size} observations; "
                f"the model and the data handed to the QC report are not the "
                f"same rows.")
        result.base = base
    return result


# ---------------------------------------------------------------------------
# Context construction
# ---------------------------------------------------------------------------


def _as_frame(X):
    """Coerce a design matrix to a DataFrame with usable column names."""
    if isinstance(X, pd.DataFrame):
        return X
    arr = np.asarray(X)
    if arr.ndim == 1:
        arr = arr[:, None]
    return pd.DataFrame(arr, columns=[f"x{i}" for i in range(arr.shape[1])])


def _as_vector(y, name="y"):
    """Coerce a response to a 1-D float array, refusing anything ambiguous."""
    if isinstance(y, pd.DataFrame):
        if y.shape[1] != 1:
            raise ValueError(
                f"{name} has {y.shape[1]} columns; a single response column is "
                f"required")
        y = y.iloc[:, 0]
    arr = np.asarray(y, dtype=float)
    arr = arr.ravel() if arr.ndim > 1 else arr
    return arr


def _model_fitted_values(model, X):
    """Fitted values on the response scale, for statsmodels or sklearn."""
    fitted = getattr(model, "fittedvalues", None)
    if fitted is not None:
        return np.asarray(fitted, dtype=float).ravel()
    predict = getattr(model, "predict", None)
    if predict is None:
        raise ValueError(
            f"{type(model).__name__} exposes neither fittedvalues nor "
            f"predict(); it cannot be QC'd")
    return np.asarray(predict(X), dtype=float).ravel()


def _decision_score(model, X, y):
    """The continuous score a classifier ranks by, oriented positive-is-higher.

    Returns ``None`` for anything that is not a two-class discriminative
    classifier, in which case the panels fall back to the fitted values.

    **The orientation is the entire point of this function.** scikit-learn's
    contract is that ``decision_function(x) > 0`` predicts ``classes_[1]``, and
    ``roc_auc_score`` treats the LARGER label as the event. Those two agree
    whenever ``classes_`` is ascending, which sklearn guarantees for its own
    estimators — but the agreement is an assumption, and an ROC computed on a
    score with the opposite convention returns ``1 - AUC``: 0.94 becomes 0.06,
    0.62 becomes 0.38, and neither reads as an error on the plot. This is the
    same trap R's ``yardstick`` sets by treating the FIRST factor level as the
    event, which this project has already been bitten by.

    So the orientation is derived from ``classes_`` rather than assumed, and
    the score is negated when ``classes_[1]`` is the *smaller* label. It is
    never derived from the data: choosing the sign that makes the AUC exceed
    0.5 would guarantee a flattering answer on noise, which is precisely the
    failure the panel exists to reveal.

    :param model: Fitted estimator.
    :param X: Design matrix the fit saw.
    :param y: Response, used only for the length check.
    :returns: 1-D float array aligned with ``y``, or ``None``.
    """
    decision = getattr(model, "decision_function", None)
    if not callable(decision):
        return None
    try:
        score = np.asarray(decision(X), dtype=float)
    except Exception:                               # noqa: BLE001
        # An estimator that advertises decision_function and cannot run it on
        # its own design is not a reason to lose the other twenty panels.
        return None
    if score.ndim != 1 or score.size != np.size(y):
        # Multi-class one-vs-rest returns (n, k): there is no single ranking to
        # draw one ROC from, so the panel falls back and says so.
        return None

    classes = getattr(model, "classes_", None)
    if classes is not None and len(classes) == 2:
        try:
            low, high = float(classes[0]), float(classes[1])
        except (TypeError, ValueError):
            return score                            # non-numeric labels
        if high < low:
            # classes_[1] is the SMALLER label, so a larger decision value
            # means a more NEGATIVE well. sklearn sorts classes_ and so never
            # lands here; a hand-rolled or wrapped estimator can.
            return -score
    return score


#: Family/link wording per model class, for the axis labels and the report
#: header. Only the GLM branch feeds :attr:`RegressionQCContext.is_binomial`
#: and :attr:`~RegressionQCContext.is_count`; these names are prose. They exist
#: because "Gaussian (least squares)" printed over a robust, quantile or beta
#: fit is a false statement about what was fitted.
_FAMILY_BY_KIND = {
    "OLS": ("Gaussian (least squares)", "Identity"),
    "WLS": ("Gaussian (weighted least squares)", "Identity"),
    "GLS": ("Gaussian (generalised least squares)", "Identity"),
    "GLSAR": ("Gaussian (generalised least squares, AR errors)", "Identity"),
    "RLM": ("Huber M-estimate (robust regression)", "Identity"),
    "QuantReg": ("quantile regression (no error distribution)", None),
    "MixedLM": ("Gaussian (linear mixed effects)", "Identity"),
    "ClassifierMixin": ("hinge loss (linear classifier)", None),
    "Lasso": ("Gaussian (penalised least squares)", "Identity"),
    "Ridge": ("Gaussian (penalised least squares)", "Identity"),
    "ElasticNet": ("Gaussian (penalised least squares)", "Identity"),
}


def _family_and_link(model, regression_type):
    """Name the family and link that were actually fitted.

    A GLM says so itself. Everything else is named from its model class, not
    guessed from the spaCR type string and not defaulted to least squares: a
    ``BetaModel`` reported as "Gaussian (least squares) / Identity" — which is
    what a name check against the *wrapper* class produced — is a caption that
    contradicts the fit it sits under.
    """
    family = getattr(model, "family", None)
    if family is not None:
        link = getattr(family, "link", None)
        return (type(family).__name__,
                None if link is None else type(link).__name__)
    key, _ = _model_kind(model)
    if key == "BetaModel":
        link = getattr(getattr(model, "model", None), "link", None)
        return "Beta", ("Logit" if link is None else type(link).__name__)
    if key in _FAMILY_BY_KIND:
        return _FAMILY_BY_KIND[key]
    if regression_type in ("lasso", "ridge", "elasticnet"):
        return "Gaussian (penalised least squares)", "Identity"
    return "Gaussian (least squares)", "Identity"


def _well_labels(index, metadata, n):
    """Per-well labels: ``prc`` when we have it, otherwise the row index.

    Naming the outliers is the entire point of the influence panels — "well 47"
    sends nobody to a microscope, ``plate1_r3_c11`` does.
    """
    if metadata is not None:
        if schema.PRC_KEY in metadata.columns:
            return metadata[schema.PRC_KEY].astype(str).to_numpy()
        parts = [c for c in schema.WELL_KEY_COLUMNS if c in metadata.columns]
        if len(parts) == len(schema.WELL_KEY_COLUMNS):
            joined = metadata[list(parts)].astype(str).agg(
                schema.KEY_SEPARATOR.join, axis=1)
            return joined.to_numpy()
    if index is not None and len(index) == n:
        return np.asarray([str(v) for v in index])
    return np.asarray([str(i) for i in range(n)])


def _align_metadata(metadata, index, n):
    """Return metadata aligned to the fitted rows, or raise.

    ``spacr.ml.regression`` drops rows in ``check_and_clean_data`` and patsy
    drops more, so a metadata frame handed in whole will be *longer* than the
    fit. Aligning on the index is correct when it survived; a length match is
    the only other thing that can be trusted. Anything else is a silent
    row-misalignment, which would attribute an outlier to the wrong well — the
    single worst thing this report could do.
    """
    if metadata is None:
        return None
    if not isinstance(metadata, pd.DataFrame):
        metadata = pd.DataFrame(metadata)
    covered = 0
    if index is not None and metadata.index.is_unique:
        # `index.isin(metadata.index)`, not the reverse: the question is
        # whether every FITTED row can be found, and a duplicated metadata
        # index would make .loc fan the frame out, so uniqueness is required
        # before the lookup is allowed.
        found = pd.Index(index).isin(metadata.index)
        covered = int(found.sum())
        if covered == len(index):
            return metadata.loc[list(index)].reset_index(drop=True)
    if len(metadata) == n:
        return metadata.reset_index(drop=True)
    raise ValueError(
        f"metadata has {len(metadata)} rows and does not cover the {n} rows "
        f"that were fitted (index overlap {covered}). Pass the metadata for "
        f"exactly the rows the model saw, or pass None; guessing the alignment "
        f"would label the wrong well as an outlier.")


def build_context(model, X, y, *, weights=None, metadata=None, coef_df=None,
                  regression_type=None, volcano_path=None):
    """Normalise a fitted model into the view the QC panels read.

    The residual standardisation is the one piece of real statistics here, and
    it is resolved per model class by
    :func:`resolve_residual_standardisation`::

        std_resid = base / sqrt(variance * (1 - h))

    where ``base`` and ``variance`` come from that registry — the Pearson
    residual and the GLM dispersion for a GLM or a beta fit, ``sqrt(w) *
    (y - fitted)`` and the weighted error variance for WLS, ``y - fitted`` and
    ``scale ** 2`` for a robust fit, and nothing at all for quantile regression
    or a classifier. When the registry reports that no correct scale exists,
    ``std_resid`` is all-``NaN``, ``standardisation.reason`` says why, and the
    six panels built on a standardised residual skip with that reason.

    :param model: Fitted statsmodels results object or sklearn estimator.
    :param X: Design matrix used for the fit (DataFrame preferred).
    :param y: Response used for the fit.
    :param weights: Per-observation weights passed to the fit, if any.
    :param metadata: Per-well metadata frame; aligned by index, else by length.
    :param coef_df: Coefficient table from
        :func:`spacr.ml.process_model_coefficients`.
    :param regression_type: The spaCR regression type string.
    :param volcano_path: Where the run's volcano plot was written.
    :returns: :class:`RegressionQCContext`.
    :raises ValueError: if ``X`` and ``y`` disagree in length, or if
        ``metadata`` cannot be aligned to the fitted rows.
    """
    frame = _as_frame(X)
    response = _as_vector(y)
    if response.size != frame.shape[0]:
        raise ValueError(
            f"y has {response.size} observations but the design matrix has "
            f"{frame.shape[0]} rows; they must be the rows of the same fit")

    fitted = _model_fitted_values(model, frame)
    if fitted.size != response.size:
        raise ValueError(
            f"the model produced {fitted.size} fitted values for "
            f"{response.size} observations")
    resid = response - fitted

    n, p = frame.shape
    w = None if weights is None else np.asarray(weights, dtype=float).ravel()
    if w is not None and w.size != n:
        raise ValueError(
            f"weights has {w.size} entries but the fit has {n} observations")

    # A weighted fit's hat matrix carries its weights. Take them from the model
    # rather than from the caller: they are the weights `model.scale` was
    # formed with, so they are the only ones that make the scale, the residual
    # and the hat diagonal agree. A caller who passes cell counts that are not
    # the fitted weights would otherwise get a leverage for a fit nobody ran.
    kind, model_class = _model_kind(model)
    fitted_weights = _fit_weights(model, n) if kind in ("WLS", "GLS") else None

    # Leverage: prefer whatever the model itself computed, because for a GLM
    # the IRLS weights belong in the hat matrix and the design matrix alone
    # does not know them.
    leverage, source = None, ""
    getter = getattr(model, "get_hat_matrix_diag", None)
    if callable(getter):
        try:
            leverage = np.asarray(getter(), dtype=float).ravel()
            source = "model.get_hat_matrix_diag()"
        except Exception:                       # noqa: BLE001 - see below
            leverage = None
    if leverage is None:
        influence = getattr(model, "get_influence", None)
        if callable(influence):
            try:
                leverage = np.asarray(
                    influence().hat_matrix_diag, dtype=float).ravel()
                source = "model.get_influence().hat_matrix_diag"
            except Exception:                   # noqa: BLE001
                # statsmodels raises a different exception per model class for
                # "influence is not defined here" (MixedLM, regularised fits);
                # the fallback below is exact for the unweighted case and
                # explicitly labelled, so a broad catch costs nothing but the
                # attempt.
                leverage = None
    if leverage is None or leverage.size != n:
        hat_weights = w if fitted_weights is None else fitted_weights
        leverage = leverage_from_design(frame.to_numpy(dtype=float),
                                        weights=hat_weights)
        if hat_weights is None:
            source = "design matrix"
        elif fitted_weights is not None:
            source = f"design matrix ({model_class} fit weights)"
        else:
            source = "design matrix (weighted)"

    family, link = _family_and_link(model, regression_type)

    standardisation = resolve_residual_standardisation(model, resid, n, p)
    if standardisation.available:
        scale = float(standardisation.variance)
        with np.errstate(divide="ignore", invalid="ignore"):
            std_resid = standardisation.base / np.sqrt(
                scale * np.clip(1.0 - leverage, 1e-12, None))
    else:
        # No correct scale exists for this model class. An all-NaN array is
        # not a fallback: it is what forces the panels built on a standardised
        # residual to skip, with `standardisation.reason` printed, instead of
        # naming outlier wells off a number that is not a z-score.
        scale = float("nan")
        std_resid = np.full(n, np.nan)

    index = frame.index if isinstance(X, pd.DataFrame) else None
    aligned_meta = _align_metadata(metadata, index, n)
    labels = _well_labels(index, aligned_meta, n)

    prediction_note = None
    if kind == "ClassifierMixin":
        prediction_note = (
            f"{model_class} predicts a class label, not a conditional mean of "
            f"the response: 'fitted' takes only the values in "
            f"{list(np.unique(fitted))[:4]}, so the residual, its R² and its "
            f"RMSE describe the distance from a decision, not the error of a "
            f"regression")

    decision_score = _decision_score(model, frame, response)

    notes = []
    if source.startswith("design matrix"):
        notes.append(
            f"leverage computed from the design matrix ({type(model).__name__} "
            f"exposes no hat matrix)")
    if fitted_weights is not None and source.endswith("fit weights)"):
        notes.append(
            f"leverage uses the {fitted_weights.size} per-observation weights "
            f"the {model_class} fit was given — the hat matrix of a weighted "
            f"fit carries its weights"
            + (", not the 'weights' argument" if w is not None else ""))
    if standardisation.available:
        notes.append(f"residual scale: {standardisation.source}; standardised "
                     f"quantity is the {standardisation.metric}")
    else:
        notes.append(f"no standardised residual: {standardisation.reason}")
    if prediction_note:
        notes.append(prediction_note)

    return RegressionQCContext(
        model=model, X=frame, y=response, fitted=fitted, resid=resid,
        std_resid=std_resid, leverage=leverage, leverage_source=source,
        scale=scale, labels=labels, standardisation=standardisation,
        prediction_note=prediction_note, weights=w, metadata=aligned_meta,
        coef_df=coef_df, regression_type=regression_type, family=family,
        link=link, volcano_path=volcano_path, notes=notes,
        decision_score=decision_score)


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------


def _finish(ax, title, xlabel, ylabel, n=None, unit="wells"):
    """Label an axes so the panel can be read with no other context."""
    if n is not None:
        title = f"{title}\n(n = {n:,} {unit})"
    ax.set_title(title, fontsize=9)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.tick_params(labelsize=7)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def _note(ax, text, loc="upper left", color="#222222"):
    """Stamp a short statistics block onto an axes."""
    x, y, ha, va = {
        "upper left": (0.02, 0.98, "left", "top"),
        "upper right": (0.98, 0.98, "right", "top"),
        "lower right": (0.98, 0.02, "right", "bottom"),
        "lower left": (0.02, 0.02, "left", "bottom"),
    }[loc]
    ax.text(x, y, text, transform=ax.transAxes, ha=ha, va=va, fontsize=7,
            color=color, linespacing=1.35,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="#cccccc", alpha=0.85))


def _trend(ax, x, y, color=None, label=None):
    """Overlay a smoothed trend, returning its maximum absolute value.

    LOWESS when there is enough data for it to mean anything, a binned median
    otherwise. The returned number is what makes the panel testable: a flat
    residual cloud has a small trend, a curved one does not.

    The curve wears ``ROLES['highlight']`` because on both panels that draw
    one — residuals-vs-fitted and scale-location — the smoother IS the
    sentence; everything else on those axes is the data it was fitted to.
    """
    color = color or ROLES["highlight"]
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    good = np.isfinite(x) & np.isfinite(y)
    x, y = x[good], y[good]
    if x.size < 5 or np.ptp(x) == 0:
        return float("nan")
    if x.size >= 15:
        from statsmodels.nonparametric.smoothers_lowess import lowess
        smoothed = lowess(y, x, frac=min(0.8, max(0.3, 30.0 / x.size)),
                          return_sorted=True)
        sx, sy = smoothed[:, 0], smoothed[:, 1]
    else:
        order = np.argsort(x)
        sx, sy = x[order], y[order]
    ax.plot(sx, sy, color=color, lw=WEIGHTS["data"], zorder=4,
            label=label)
    return float(np.nanmax(np.abs(sy)))


def _natural_key(value):
    """Sort ``r2`` before ``r10`` — plate rows are not lexicographic."""
    text = str(value)
    match = re.match(r"^([A-Za-z]*)0*(\d+)$", text)
    if match:
        return (0, match.group(1).lower(), int(match.group(2)), "")
    return (1, "", 0, text.lower())


def _require_standardisation(ctx, what):
    """Raise unless a correct standardised residual exists for this fit.

    The reason carried by :class:`ResidualStandardisation` is the panel's skip
    reason verbatim: it already names the model class and what is missing, and
    a panel restating it in its own words would drift from the registry that
    decided it.

    :param ctx: The QC context.
    :param what: What needs the standardised residual, e.g. ``"Cook's distance"``.
    :raises PanelUnavailable: with the registry's reason.
    """
    if ctx.standardisation_available:
        return
    reason = (ctx.standardisation.reason if ctx.standardisation is not None
              else "no residual standardisation was resolved for this fit")
    raise PanelUnavailable(f"{what} needs a standardised residual, and {reason}")


def _skip_box(ax, title, reason):
    """Draw the "this panel could not be computed, and here is why" tile."""
    ax.set_axis_off()
    ax.set_facecolor("#ececec")
    # set_axis_off() hides the patch as well, so the grey tile has to be turned
    # back on explicitly — otherwise a skipped panel is an invisible gap, which
    # is precisely the failure mode this box exists to prevent. The dashed
    # border does the same job at a glance: a reader scanning the page must be
    # able to tell "not computed" from "computed and unremarkable".
    ax.patch.set_visible(True)
    ax.add_patch(Rectangle((0.02, 0.02), 0.96, 0.96, transform=ax.transAxes,
                           facecolor="none", edgecolor="#b0b0b0", lw=1.0,
                           ls="--", zorder=2))
    ax.text(0.5, 0.72, title, ha="center", va="center", fontsize=9,
            fontweight="bold", color="#555555", transform=ax.transAxes)
    ax.text(0.5, 0.45, textwrap.fill(f"SKIPPED: {reason}", 46), ha="center",
            va="center", fontsize=7.5, color="#7a2020", transform=ax.transAxes)


# ---------------------------------------------------------------------------
# Panels — model fit
# ---------------------------------------------------------------------------


def _wells(n, unit="wells"):
    """The n, in the words the sentence title used to carry.

    It moved off the title and into the panel's note: the skill forbids a
    sentence title, and a diagnostic panel that does not say how many wells
    it drew cannot be read on its own.
    """
    return f"n = {n:,} {unit}"


def _panel_residuals_vs_fitted(ctx, ax):
    """Residual vs fitted: the single most informative regression diagnostic."""
    with figure_style(_REPORT_TARGET):
        # The wells are the default ink. This panel asks exactly one question
        # — is there structure left in the residuals — and the smoother is
        # the answer, so the smoother is the only artist entitled to colour.
        ax.scatter(ctx.fitted, ctx.resid, s=18, color=ROLES["data"],
                   edgecolors="none", zorder=3)
        reference_line(ax, y=0.0)
        trend = _trend(ax, ctx.fitted, ctx.resid)
        spread = float(np.nanstd(ctx.resid))
        ink = _house_axes(ax, "residuals vs fitted",
                          "fitted value (response scale)",
                          "residual (observed - fitted)")
        text = (f"{_wells(ctx.n)}\nfamily: {ctx.family}\n"
                f"resid SD = {spread:.4g}\n"
                f"mean = {np.nanmean(ctx.resid):+.3g}\n"
                f"|trend| max = {trend:.3g}")
        if ctx.prediction_note:
            text += "\n" + textwrap.fill(ctx.prediction_note, 40)
        annotate(ax, text, colour=ink)
    return {"n_points": int(np.sum(np.isfinite(ctx.resid))),
            "resid_sd": spread, "resid_mean": float(np.nanmean(ctx.resid)),
            "max_abs_trend": trend,
            "max_abs_resid": float(np.nanmax(np.abs(ctx.resid))),
            "limitation": ctx.prediction_note}


def _panel_residual_distribution(ctx, ax):
    """Residual histogram with a KDE and the matching normal density."""
    from scipy import stats as sps

    resid = ctx.resid[np.isfinite(ctx.resid)]
    if resid.size < 3:
        raise PanelUnavailable(
            f"only {resid.size} finite residual(s); a distribution needs at least 3")
    bins = int(np.clip(np.sqrt(resid.size), 10, 60))
    with figure_style(_REPORT_TARGET):
        ax.hist(resid, bins=bins, density=True, color=ROLES["fill"],
                edgecolor="none")
        grid = np.linspace(resid.min(), resid.max(), 256)
        drew_kde = np.ptp(resid) > 0
        if drew_kde:
            # A KDE on constant data raises inside scipy (singular covariance);
            # the range check keeps that failure from reaching the report.
            # The empirical density is what the panel is ABOUT — whether the
            # residuals are normal — so it is the one thing that gets colour.
            ax.plot(grid, sps.gaussian_kde(resid)(grid),
                    color=ROLES["highlight"], lw=WEIGHTS["data"], zorder=4)
        # The normal curve is a reference, not a result: grey, thin and
        # dashed, exactly like style.reference_line, which cannot draw it
        # because it is a curve rather than a horizontal or vertical rule.
        ax.plot(grid, sps.norm.pdf(grid, resid.mean(), resid.std(ddof=1) or 1e-12),
                color=ROLES["reference"], lw=WEIGHTS["reference"],
                ls=(0, (4, 3)), zorder=1)
        entries = [("normal fit", ROLES["reference"])]
        if drew_kde:
            entries.insert(0, ("KDE", ROLES["highlight"]))
        text_legend(ax, entries)

        skew = float(sps.skew(resid))
        kurt = float(sps.kurtosis(resid))
        if resid.size >= 8:
            stat, pval = sps.normaltest(resid)
            test = "D'Agostino K²"
        else:
            stat, pval = float("nan"), float("nan")
            test = "normality test needs n >= 8"
        ink = _house_axes(ax, "residual distribution", "residual", "density")
        # The n moved off the title and into the note: the descriptor says
        # which panel this is, the note carries everything a reader needs to
        # judge the number.
        annotate(ax, f"{_wells(resid.size)}\nskew = {skew:+.2f}\n"
                     f"excess kurtosis = {kurt:+.2f}\n"
                     f"{test}: p = {pval:.3g}" if np.isfinite(pval) else
                 f"{_wells(resid.size)}\nskew = {skew:+.2f}\n"
                 f"excess kurtosis = {kurt:+.2f}\n{test}",
                 x=0.98, ha="right", colour=ink)
    return {"skew": skew, "excess_kurtosis": kurt, "normality_p": float(pval),
            "n_bins": bins, "n_points": int(resid.size),
            "limitation": ctx.prediction_note}


def _panel_scale_location(ctx, ax):
    """sqrt|standardised residual| vs fitted — the heteroscedasticity panel.

    Two statistics, because one is not enough. Spearman's rho catches the
    *monotone* megaphone (spread grows with the fit) and is what most
    implementations stop at — but it is exactly zero for a symmetric funnel,
    where the spread is large at both ends and small in the middle. That shape
    is what a mis-specified link or an unmodelled quadratic produces, it is
    common in this pipeline, and a panel that printed "no trend in spread"
    over it would be handing back a confident wrong answer. A Brown-Forsythe
    test across quartiles of the fitted value sees both.
    """
    from scipy import stats as sps

    _require_standardisation(ctx, "the scale-location panel")
    root = np.sqrt(np.abs(ctx.std_resid))
    good = np.isfinite(root) & np.isfinite(ctx.fitted)
    if good.sum() < 3:
        raise PanelUnavailable("fewer than 3 finite standardised residuals")
    fitted, root = ctx.fitted[good], root[good]
    rho, rho_p = sps.spearmanr(fitted, root)

    # Brown-Forsythe: Levene centred on the median, which is the version that
    # survives the heavy-tailed residuals screen data actually produces.
    levene_p, sd_ratio = float("nan"), float("nan")
    resid = ctx.resid[good]
    if good.sum() >= 20 and np.ptp(fitted) > 0:
        edges = np.unique(np.quantile(fitted, [0.0, 0.25, 0.5, 0.75, 1.0]))
        if edges.size >= 3:
            bucket = np.clip(np.searchsorted(edges, fitted, side="right") - 1,
                             0, edges.size - 2)
            groups = [resid[bucket == b] for b in range(edges.size - 1)]
            groups = [g for g in groups if g.size >= 2]
            if len(groups) >= 2:
                sds = np.array([np.std(g, ddof=1) for g in groups])
                sd_ratio = float(sds.max() / sds.min()) if sds.min() > 0 else np.inf
                try:
                    levene_p = float(sps.levene(*groups, center="median")[1])
                except ValueError:
                    # scipy refuses when a group is constant; that is a real
                    # answer about the data, not a panel failure.
                    levene_p = float("nan")

    unequal = np.isfinite(levene_p) and levene_p < 0.01
    if unequal and rho > 0.3:
        verdict = "variance grows with the fit"
    elif unequal and rho < -0.3:
        verdict = "variance shrinks with the fit"
    elif unequal:
        verdict = "spread differs across the fit, but not monotonically"
    elif abs(rho) > 0.3:
        verdict = ("variance grows with the fit" if rho > 0
                   else "variance shrinks with the fit")
    else:
        verdict = "no detectable trend in spread"
    with figure_style(_REPORT_TARGET):
        ax.scatter(fitted, root, s=18, color=ROLES["data"], edgecolors="none")
        _trend(ax, fitted, root)
        ink = _house_axes(
            ax, "scale-location", "fitted value",
            r"$\sqrt{|\mathrm{standardised\ residual}|}$")
        annotate(ax, f"{_wells(int(good.sum()))}\n"
                     f"Spearman rho = {rho:+.2f} (p = {rho_p:.2g})\n"
                     f"Brown-Forsythe p = {levene_p:.2g}\n"
                     f"max/min quartile SD = {sd_ratio:.2f}", colour=ink)
        # The verdict is the one line a reader acts on, so it is the one line
        # that changes colour: rust when the panel is saying the variance is
        # not constant, plain ink when it is saying nothing is wrong.
        flat = verdict == "no detectable trend in spread"
        annotate(ax, verdict, y=0.78,
                 colour=ink if flat else ROLES["down"])
    return {"spearman_rho": float(rho), "spearman_p": float(rho_p),
            "levene_p": levene_p, "quartile_sd_ratio": sd_ratio,
            "verdict": verdict, "n_points": int(good.sum())}


def _panel_qq_residuals(ctx, ax):
    """Normal Q-Q of the standardised residuals with an R-style reference line."""
    from scipy import stats as sps

    _require_standardisation(ctx, "a Q-Q plot of standardised residuals")
    sample = np.sort(ctx.std_resid[np.isfinite(ctx.std_resid)])
    if sample.size < 5:
        raise PanelUnavailable(
            f"only {sample.size} finite standardised residual(s); a Q-Q plot "
            f"needs at least 5")
    # Blom plotting positions: the standard choice, and unbiased for the
    # normal order statistics that the reference line assumes.
    quantiles = sps.norm.ppf((np.arange(1, sample.size + 1) - 0.375)
                             / (sample.size + 0.25))
    q1_t, q3_t = sps.norm.ppf([0.25, 0.75])
    q1_s, q3_s = np.quantile(sample, [0.25, 0.75])
    slope = (q3_s - q1_s) / (q3_t - q1_t)
    intercept = q1_s - slope * q1_t
    xs = np.array([quantiles[0], quantiles[-1]])
    corr = float(np.corrcoef(quantiles, sample)[0, 1])
    with figure_style(_REPORT_TARGET):
        ax.scatter(quantiles, sample, s=16, color=ROLES["data"],
                   edgecolors="none")
        # The quartile line is a REFERENCE, not a result: the claim of this
        # panel is whether the points follow it, so it is drawn the way every
        # other threshold in the report is — thin, dashed and grey. It used
        # to be the boldest artist on the axes.
        ax.plot(xs, intercept + slope * xs, color=ROLES["reference"],
                lw=WEIGHTS["reference"], ls=(0, (4, 3)), zorder=0)
        ink = _house_axes(ax, "normal q-q", "theoretical normal quantile",
                          "observed quantile")
        text_legend(ax, [("quartile reference", ROLES["reference"])])
        annotate(ax, f"{_wells(int(sample.size))}\n"
                     f"line: slope {slope:.2f}, intercept {intercept:+.2f}\n"
                     f"quantile correlation = {corr:.4f}",
                 x=0.98, y=0.02, ha="right", va="bottom", colour=ink)
    return {"slope": float(slope), "intercept": float(intercept),
            "quantile_correlation": corr, "n_points": int(sample.size)}


def _panel_observed_vs_predicted(ctx, ax):
    """Observed vs predicted with the identity line, R² and RMSE."""
    good = np.isfinite(ctx.y) & np.isfinite(ctx.fitted)
    if good.sum() < 3:
        raise PanelUnavailable("fewer than 3 observations with a finite fit")
    obs, pred = ctx.y[good], ctx.fitted[good]
    lo = float(min(obs.min(), pred.min()))
    hi = float(max(obs.max(), pred.max()))
    pad = 0.02 * (hi - lo or 1.0)

    rss = float(np.sum((obs - pred) ** 2))
    tss = float(np.sum((obs - obs.mean()) ** 2))
    r2 = 1.0 - rss / tss if tss > 0 else float("nan")
    rmse = float(np.sqrt(rss / obs.size))
    mae = float(np.mean(np.abs(obs - pred)))
    pearson = float(np.corrcoef(obs, pred)[0, 1]) if np.ptp(pred) > 0 else float("nan")
    with figure_style(_REPORT_TARGET):
        ax.scatter(pred, obs, s=18, color=ROLES["data"], edgecolors="none")
        # The skill's own words for this plot: "scatter, grey, with a
        # highlighted subset, dotted 1:1 diagonal". There is no subset to
        # highlight here — the whole cloud is the answer — so the diagonal is
        # dotted grey and nothing else is coloured.
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad],
                color=ROLES["reference"], lw=WEIGHTS["reference"],
                ls=(0, (1, 2)), zorder=0)
        ink = _house_axes(ax, "observed vs predicted", "predicted",
                          "observed")
        text_legend(ax, [("1:1", ROLES["reference"])], x=0.74, y=0.86)
        text = (f"{_wells(int(good.sum()))}\n"
                f"R² (response scale) = {r2:.3f}\nRMSE = {rmse:.4g}\n"
                f"MAE = {mae:.4g}\nPearson r = {pearson:.3f}")
        if ctx.prediction_note:
            text += "\n" + textwrap.fill(ctx.prediction_note, 40)
        annotate(ax, text, colour=ink)
    return {"r2": float(r2), "rmse": rmse, "mae": mae,
            "pearson_r": pearson, "n_points": int(good.sum()),
            "limitation": ctx.prediction_note}


# ---------------------------------------------------------------------------
# Panels — influence and leverage
# ---------------------------------------------------------------------------


def _panel_cooks_distance(ctx, ax):
    """Cook's distance per well with the 4/n rule drawn and the top wells named."""
    _require_standardisation(ctx, "Cook's distance")
    d = cooks_distance(ctx.std_resid, ctx.leverage, ctx.p)
    if not np.any(np.isfinite(d)):
        raise PanelUnavailable("Cook's distance is undefined for every well "
                               "(the fit is saturated: leverage == 1)")
    threshold = _COOKS_RULE / ctx.n
    finite = np.where(np.isfinite(d), d, np.nan)
    heights = np.nan_to_num(finite, nan=0.0)
    above = np.where(finite > threshold)[0]
    below = np.setdiff1d(np.arange(ctx.n), above)
    order = above[np.argsort(-np.nan_to_num(finite[above], nan=0.0))][:5]
    worst = int(np.nanargmax(finite)) if np.any(np.isfinite(finite)) else -1
    with figure_style(_REPORT_TARGET):
        # The sentence is "these wells are influential", so the wells above
        # the rule are the only thing on the panel that carries colour, and
        # they are drawn from the same `above` set the stats report. Every
        # other well is the grey it was always entitled to.
        ax.vlines(below, 0, heights[below], color=ROLES["data"],
                  lw=WEIGHTS["reference"])
        ax.vlines(above, 0, heights[above], color=ROLES["down"],
                  lw=WEIGHTS["data"])
        reference_line(ax, y=threshold, label=f"4/n = {threshold:.3g}")
        for i in order:
            ax.annotate(str(ctx.labels[i]), (i, finite[i]),
                        fontsize=TYPE_SCALE["annotation"],
                        textcoords="offset points", xytext=(2, 2),
                        color=ROLES["down"])
        ink = _house_axes(ax, "cook's distance per well", "well (fit order)",
                          "cook's distance")
        # The note goes to whichever top corner the worst well is NOT in: the
        # tallest stem carries a label, and a fixed corner puts the two on top
        # of each other whenever the worst well lands on that side.
        side = "left" if worst >= ctx.n / 2 else "right"
        edge = 0.02 if side == "left" else 0.98
        annotate(ax, f"{above.size} well(s) above 4/n", x=edge, ha=side,
                 colour=ROLES["down"] if above.size else ink)
        annotate(ax, f"{_wells(ctx.n)}\n"
                     f"max = {np.nanmax(finite):.3g} ({ctx.labels[worst]})",
                 x=edge, y=0.91, ha=side, colour=ink)
    return {"threshold": float(threshold), "n_above": int(above.size),
            "max_cooks": float(np.nanmax(finite)),
            "max_label": str(ctx.labels[worst]),
            "max_index": worst,
            "labelled": [str(ctx.labels[i]) for i in order],
            "flagged": [str(ctx.labels[i]) for i in above]}


def _panel_influence(ctx, ax):
    """Leverage vs standardised residual, bubble area proportional to Cook's D."""
    _require_standardisation(ctx, "the leverage-vs-residual panel")
    d = cooks_distance(ctx.std_resid, ctx.leverage, ctx.p)
    finite_d = np.nan_to_num(np.where(np.isfinite(d), d, np.nan), nan=0.0)
    scale = finite_d.max() or 1.0
    sizes = 12.0 + 180.0 * finite_d / scale
    order = np.argsort(-finite_d)[:5]
    named = np.zeros(int(ctx.n), dtype=bool)
    named[order] = True
    guides = [float(mult * ctx.p / ctx.n) for mult in _LEVERAGE_RULES]
    high = int(np.sum(ctx.leverage > guides[0]))
    with figure_style(_REPORT_TARGET):
        # The five wells the panel NAMES are the five it colours. Labelling a
        # well in one colour and drawing its bubble in another was the panel
        # saying two different things about the same well.
        ax.scatter(ctx.leverage[~named], ctx.std_resid[~named],
                   s=sizes[~named], color=ROLES["data"], edgecolors="none")
        ax.scatter(ctx.leverage[named], ctx.std_resid[named], s=sizes[named],
                   color=ROLES["down"], edgecolors="none", zorder=4)
        reference_line(ax, y=0.0)
        for k in (-2.0, 2.0):
            reference_line(ax, y=k)
        for mult, guide in zip(_LEVERAGE_RULES, guides):
            reference_line(ax, x=guide, label=f"{mult:.0f}p/n")
        for i in order:
            ax.annotate(str(ctx.labels[i]),
                        (ctx.leverage[i], ctx.std_resid[i]),
                        fontsize=TYPE_SCALE["annotation"],
                        textcoords="offset points", xytext=(3, 3),
                        color=ROLES["down"])
        ink = _house_axes(ax, "leverage vs residual",
                          "leverage (hat diagonal)",
                          "standardised residual")
        annotate(ax, f"{_wells(ctx.n)}\n"
                     f"{high} well(s) above 2p/n = {guides[0]:.3g}\n"
                     f"max leverage = {ctx.leverage.max():.3g}\n"
                     f"bubble area ∝ cook's D\n"
                     + textwrap.fill(f"hat from {ctx.leverage_source}", 44,
                                     break_long_words=False),
                 x=0.98, ha="right", colour=ink)
    return {"n_high_leverage": high, "leverage_guides": guides,
            "max_leverage": float(ctx.leverage.max()),
            "labelled": [str(ctx.labels[i]) for i in order],
            "n_points": int(ctx.n)}


def _panel_dffits(ctx, ax):
    """|DFFITS| per well against the 2*sqrt(p/n) rule."""
    _require_standardisation(ctx, "DFFITS")
    values, threshold = dffits(ctx.std_resid, ctx.leverage, ctx.n, ctx.p)
    if not np.any(np.isfinite(values)):
        raise PanelUnavailable(
            f"DFFITS needs n > p + 1; this fit has n = {ctx.n}, p = {ctx.p}")
    magnitude = np.abs(values)
    heights = np.nan_to_num(magnitude, nan=0.0)
    above = np.where(magnitude > threshold)[0]
    below = np.setdiff1d(np.arange(ctx.n), above)
    with figure_style(_REPORT_TARGET):
        ax.vlines(below, 0, heights[below], color=ROLES["data"],
                  lw=WEIGHTS["reference"])
        ax.vlines(above, 0, heights[above], color=ROLES["down"],
                  lw=WEIGHTS["data"])
        reference_line(ax, y=threshold,
                       label=f"2·sqrt(p/n) = {threshold:.3g}")
        for i in above[np.argsort(-heights[above])][:5]:
            ax.annotate(str(ctx.labels[i]), (i, magnitude[i]),
                        fontsize=TYPE_SCALE["annotation"],
                        textcoords="offset points", xytext=(2, 2),
                        color=ROLES["down"])
        ink = _house_axes(ax, "dffits per well", "well (fit order)",
                          "|dffits| (fitted-value shift, in standard errors)")
        side = "left" if int(np.argmax(heights)) >= ctx.n / 2 else "right"
        edge = 0.02 if side == "left" else 0.98
        annotate(ax, f"{above.size} well(s) above threshold", x=edge, ha=side,
                 colour=ROLES["down"] if above.size else ink)
        annotate(ax, f"{_wells(ctx.n)}\n"
                     f"max = {np.nanmax(magnitude):.3g}",
                 x=edge, y=0.91, ha=side, colour=ink)
    return {"threshold": float(threshold), "n_above": int(above.size),
            "max_abs_dffits": float(np.nanmax(magnitude)),
            "flagged": [str(ctx.labels[i]) for i in above]}


# ---------------------------------------------------------------------------
# Panels — design and collinearity
# ---------------------------------------------------------------------------


#: Where the panels in this module are going, and therefore which ink they get.
#:
#: ``'print'``, deliberately, and NOT :func:`theme_target`. Every panel here is
#: written to ``<results>/regression_qc/`` as a PDF by
#: :func:`regression_qc_report`; :func:`spacr.ml._write_regression_qc` is its
#: only production caller and no spaCR screen draws these axes. A file is read
#: on a page, and the page this report driver produces is white: the ``Figure``
#: is built before any style context is entered, so it keeps matplotlib's white
#: facecolor and ``_save`` writes that white into the PDF.
#:
#: ``theme_target()`` answers a different question — "what is the GUI theme
#: doing?" — and returns ``'screen'`` for every user who has not explicitly set
#: a white figure background, which resolves to ``INK_SCREEN`` (#E8EDEE).
#: #E8EDEE text on a white PDF page is invisible. Checked, not assumed.
_REPORT_TARGET = "print"


def _house_axes(ax, text, xlabel, ylabel, target=_REPORT_TARGET):
    """Ink, type and L-framing for a panel, and the resolved ink back.

    Used instead of :func:`_finish`, which writes a two-line sentence title
    (the skill forbids sentence titles) and leaves the library's own type
    sizes in place.

    WHY THE INK IS PUSHED ONTO THE AXES RATHER THAN LEFT TO THE CONTEXT
    MANAGER: :func:`~spacr.figures.style.figure_style` is an ``rc_context``,
    and rcParams colour an artist when it is CREATED. The axes a QC panel
    draws into is created by the report driver, outside the ``with`` block and
    before the panel is called at all, so its spines, its tick labels and the
    ``Text`` objects behind ``set_xlabel``/``set_ylabel`` already exist
    carrying matplotlib's defaults. Opening the style does not retro-colour
    them; only what the panel draws inside the block picks the style up.

    :param ax: The axes the panel is drawing into.
    :param text: The descriptor — 2-4 lower-case words, never a sentence.
    :param xlabel: Lower-case x-axis label; ``""`` for none.
    :param ylabel: Lower-case y-axis label; ``""`` for none.
    :param target: ``'print'`` or ``'screen'``; see :data:`_REPORT_TARGET`.
    :returns: The resolved ink, for annotations that are not a warning.

    Two clusters arrived the same day with their own copies of this
    function, ``_distribution_axes`` and ``_scatter_axes``, each differing
    only in that it resolved the ink from ``theme_target()`` — the answer
    this module cannot use, see :data:`_REPORT_TARGET`. Both copies are gone
    and all of their panels call this one.
    """
    ink = resolve_ink(target)
    descriptor(ax, text)
    ax.title.set_color(ink)
    ax.set_xlabel(xlabel, fontsize=TYPE_SCALE["label"], color=ink)
    ax.set_ylabel(ylabel, fontsize=TYPE_SCALE["label"], color=ink)
    ax.tick_params(colors=ink, labelsize=TYPE_SCALE["tick"],
                   width=WEIGHTS["spine"], length=2.6)
    for side, spine in ax.spines.items():
        # Cell-style L framing: left and bottom only.
        spine.set_visible(side in ("left", "bottom"))
        spine.set_color(ink)
        spine.set_linewidth(WEIGHTS["spine"])
    ax.set_facecolor(TRANSPARENT)
    return ink


def _readable_number(value):
    """A large number a panel can actually print.

    ``f"{1.3583e16:,.1f}"`` is ``13,583,837,847,143,152.0`` — nineteen
    characters that overflow the axes and that nobody reads to the end. A
    scaled condition number of 1e16 is a real result on a design carrying the
    dummy-variable trap, so the panel has to be able to say so. Only the
    rendering changes; the manifest still carries the float.
    """
    if not np.isfinite(value):
        return str(value)
    return f"{value:,.1f}" if abs(value) < 1e6 else f"{value:.3g}"


def _panel_vif(ctx, ax):
    """VIF per predictor with the conventional 5 and 10 guides."""
    vif = variance_inflation_factors(ctx.X)
    usable = vif.dropna()
    if usable.empty:
        raise PanelUnavailable(
            "every predictor is constant on the fitted rows, so no variance "
            "can be inflated")
    ordered = usable.sort_values(ascending=False)
    shown = ordered.head(30)
    finite = shown[np.isfinite(shown)]
    ceiling = float(finite.max()) if not finite.empty else 10.0
    plot_values = shown.replace(np.inf, max(ceiling * 1.6, 20.0))
    positions = np.arange(len(shown))
    n_inf = int(np.sum(~np.isfinite(usable)))
    with figure_style(_REPORT_TARGET):
        # Grey and RUST, and nothing else. The traffic light this replaces
        # spent GREEN on a healthy VIF, and GREEN is `up` in the shared
        # vocabulary: a reader who learned that from the coefficient forest
        # would read "no collinearity problem" as "called, upregulated". It
        # also coloured every bar, so on a healthy design the panel shouted in
        # three hues while claiming nothing.
        colors = [ROLES["down"] if v > 10 else ROLES["data"] for v in shown]
        ax.barh(positions, plot_values.to_numpy(), color=colors, zorder=1)
        ax.set_yticks(positions)
        ax.set_yticklabels([str(s)[:28] for s in shown.index],
                           fontsize=TYPE_SCALE["annotation"])
        ax.invert_yaxis()
        for guide in (5.0, 10.0):
            # style.reference_line parks a rule at zorder 0, which is right
            # for a scatter and wrong for bars: the bars would cover the very
            # thresholds they are meant to be read against. Lifted just above
            # them; still thin, still dashed, still grey. Unlabelled, because
            # the x axis already has a tick at 5 and at 10 and the note below
            # counts how many predictors are past each -- the old rotated
            # "VIF=5" tag was ink laid over the top bar to say what two ticks
            # were already saying.
            reference_line(ax, x=guide).set_zorder(2)
        for pos, value in zip(positions, shown):
            if not np.isfinite(value):
                ax.text(plot_values.iloc[pos], pos, "  inf (aliased)",
                        fontsize=TYPE_SCALE["annotation"],
                        color=ROLES["down"], va="center", zorder=3)
        # Room for the "inf (aliased)" tags to the right of the longest bar,
        # and never less than the VIF = 10 guide, which on a healthy design
        # sits far beyond every bar and still has to be on the panel.
        ax.set_xlim(0.0, max(float(np.nanmax(plot_values)) * 1.45, 11.5))
        # Room UNDER the last bar for the note. A ranked-bar panel has no
        # empty corner -- the old note was a white box laid over the bottom
        # five bars and their aliasing tags, which is what a box is for and
        # exactly why the style has none.
        ax.set_ylim(len(shown) + 4.5, -0.5)
        ink = _house_axes(ax, "variance inflation",
                          "variance inflation factor", "")
        annotate(ax, f"{len(shown)} of {len(ordered)} predictors, largest "
                     f"first\n{int(np.sum(usable > 10))} above 10, "
                     f"{int(np.sum(usable > 5))} above 5\n"
                     f"{n_inf} exactly aliased\n"
                     f"{int(vif.isna().sum())} constant (no VIF)",
                 x=0.98, y=0.02, ha="right", va="bottom", colour=ink)
    return {"max_vif": float(np.nanmax(usable.replace(np.inf, np.nan)))
                       if np.any(np.isfinite(usable)) else float("inf"),
            "n_above_10": int(np.sum(usable > 10)),
            "n_above_5": int(np.sum(usable > 5)),
            "n_aliased": n_inf,
            "n_constant": int(vif.isna().sum()),
            "vif": {str(k): float(v) for k, v in ordered.head(30).items()}}


def _panel_condition_number(ctx, ax):
    """The design's condition number, as a number, with its interpretation."""
    scaled, unscaled, singular = condition_number(ctx.X.to_numpy(dtype=float))
    verdict = condition_verdict(scaled)
    positions = np.arange(singular.size)
    rank = int(np.sum(singular > (singular.max() * 1e-12
                                  if singular.size else 0)))
    severe = scaled >= 30
    with figure_style(_REPORT_TARGET):
        # The spectrum has no minority to highlight — its SHAPE is the claim —
        # so the bars stay grey and the colour is spent on the verdict, which
        # is the only thing here that can be a warning.
        ax.bar(positions, np.where(singular > 0, singular, np.nan),
               color=ROLES["data"])
        ax.set_yscale("log")
        positive = singular[singular > 0]
        if positive.size:
            top, bottom = float(positive.max()), float(positive.min())
            # Headroom for the verdict, measured in decades rather than
            # pixels. The old panel wrote the number at 0.88 of the axes and
            # the verdict at 0.75, both straight over the bars, which on any
            # design with a full-height spectrum made both unreadable. A fixed
            # multiplier cannot work either: a healthy spectrum spans a
            # fraction of one decade and a singular one spans sixteen. The
            # floor of 3 is for the orthonormal case, where the spectrum is
            # flat and proportional headroom would be no headroom at all.
            span = max(top / bottom, 1.0)
            ax.set_ylim(bottom / 2.0, top * max(span ** 0.55, 3.0))
        ink = _house_axes(ax, "design conditioning",
                          "singular value, largest first",
                          "singular value of the column-scaled X")
        # One tier up from an annotation, because this number IS the panel —
        # the bars are only the evidence for it. RUST when the design is
        # badly conditioned, plain ink when it is not; never GREEN, which in
        # the shared vocabulary means "called up" and would read as a result.
        ax.text(0.5, 0.97, f"scaled condition number = {_readable_number(scaled)}",
                transform=ax.transAxes, ha="center", va="top",
                fontsize=TYPE_SCALE["label"],
                color=ROLES["down"] if severe else ink)
        annotate(ax, textwrap.fill(verdict, 44), x=0.5, y=0.90, ha="center",
                 colour=ROLES["down"] if severe else ink)
        # Up in the headroom with the verdict, not in the bottom-left corner:
        # a singular-value spectrum starts at the axes floor, so on a design
        # whose bars span the panel there IS no bottom-left corner, and the
        # old note sat on top of the first two hundred bars.
        annotate(ax, f"unscaled = {_readable_number(unscaled)}\n"
                     f"{ctx.p} predictor(s)\nrank = {rank}",
                 x=0.98, y=0.97, ha="right", va="top", colour=ink)
    return {"condition_number": float(scaled),
            "condition_number_unscaled": float(unscaled),
            "verdict": verdict,
            "n_singular_values": int(singular.size),
            "rank": rank}


def _panel_predictor_correlation(ctx, ax):
    """Correlation heatmap of the predictors."""
    varying = ctx.X.loc[:, ctx.X.std(ddof=1) > 0]
    if varying.shape[1] < 2:
        raise PanelUnavailable(
            f"only {varying.shape[1]} non-constant predictor(s); a correlation "
            f"matrix needs at least 2")
    limit = 40
    truncated = varying.shape[1] > limit
    if truncated:
        # Beyond ~40 rows the cells are smaller than the axis labels, so the
        # heatmap stops being readable. Keep the predictors with the largest
        # spread, which are the ones carrying the design.
        keep = varying.std(ddof=1).sort_values(ascending=False).head(limit).index
        varying = varying[keep]
    corr = np.corrcoef(varying.to_numpy(dtype=float), rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0)
    off = corr - np.eye(corr.shape[0])
    flat = np.abs(off)
    worst = np.unravel_index(int(np.argmax(flat)), flat.shape)
    with figure_style(_REPORT_TARGET):
        # THE DIVERGING MAP STAYS. The skill permits one exactly here:
        # "diverging colormaps appear only for genuinely signed quantities",
        # and Pearson r on [-1, 1] is the example. Mapping it to the
        # categorical palette or to Palette.SEQUENTIAL would put r = -0.9 and
        # r = +0.9 at the same lightness and destroy the panel's only
        # encoding. The grey-plus-highlight rule does not apply either: there
        # is no minority to pick out, because the whole matrix is the claim.
        image = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
        bar = ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        # Named on the ticks, so the axes are not labelled "predictor" twice
        # over the names themselves; named on the axes when there are too many
        # predictors to tick, so the reader still knows what the grid is.
        named = varying.shape[1] <= 25
        ink = _house_axes(ax, "predictor correlation",
                          "" if named else "predictor",
                          "" if named else "predictor")
        bar.ax.tick_params(colors=ink, labelsize=TYPE_SCALE["tick"],
                           width=WEIGHTS["spine"], length=2.6)
        bar.outline.set_edgecolor(ink)
        bar.outline.set_linewidth(WEIGHTS["spine"])
        bar.set_label("Pearson r", fontsize=TYPE_SCALE["label"], color=ink)
        if named:
            ax.set_xticks(range(varying.shape[1]))
            ax.set_yticks(range(varying.shape[1]))
            # 20 characters, not 14. Every predictor in a spaCR screen design
            # is named `fraction:grna[<id>]`, and 14 is the length of the
            # prefix -- every tick came out reading `fraction:grna[`.
            ax.set_xticklabels([str(c)[:20] for c in varying.columns],
                               fontsize=TYPE_SCALE["legend"])
            ax.set_yticklabels([str(c)[:20] for c in varying.columns],
                               fontsize=TYPE_SCALE["legend"])
            # 45 degrees right-aligned, not the 90 this drew before: the skill
            # pins the angle, and a vertical label reads a word at a time.
            rotate_ticks(ax, 45)
        else:
            ax.set_xticks([])
            ax.set_yticks([])
        # 28 characters, not 18. Every predictor in a spaCR screen design is
        # named `fraction:grna[<id>]`, and 18 cut every one of them inside the
        # bracket -- "between fraction:grna[2130 and fraction:grna[2276" names
        # no pair at all.
        caption = (f"largest |r| = {flat.max():.2f} between "
                   f"{str(varying.columns[worst[0]])[:28]} and "
                   f"{str(varying.columns[worst[1]])[:28]}")
        if truncated:
            caption += f"\ntop {limit} of {ctx.X.shape[1]} predictors by spread"
        # Below the rotated tick labels when there are any, below the axis
        # label when there are not.
        annotate(ax, caption, x=0.5, y=-0.34 if named else -0.16,
                 ha="center", va="top", colour=ink)
    return {"n_predictors": int(varying.shape[1]),
            "max_abs_offdiagonal": float(flat.max()),
            "max_pair": [str(varying.columns[worst[0]]),
                         str(varying.columns[worst[1]])],
            "truncated": bool(truncated)}


# ---------------------------------------------------------------------------
# Panels — response and coefficients
# ---------------------------------------------------------------------------


def _coefficient_table(ctx):
    """``(DataFrame, note)`` of coefficients with intervals where they exist.

    The note is what the forest panel prints when it has to degrade: a sklearn
    ``Lasso`` genuinely has no covariance matrix, so it genuinely has no
    intervals, and drawing an error bar there would be a fabrication.
    """
    params = getattr(ctx.model, "params", None)
    if params is not None:
        series = pd.Series(np.asarray(params, dtype=float).ravel(),
                           index=getattr(params, "index", ctx.X.columns[:len(params)]))
        conf = getattr(ctx.model, "conf_int", None)
        table = pd.DataFrame({"coefficient": series})
        if callable(conf):
            try:
                intervals = conf()
                table["lower"] = np.asarray(intervals)[:, 0]
                table["upper"] = np.asarray(intervals)[:, 1]
            except Exception as exc:            # noqa: BLE001
                return table, (f"no confidence intervals: conf_int() raised "
                               f"{type(exc).__name__}")
            return table, None
        return table, (f"{type(ctx.model).__name__} exposes no conf_int(); "
                       f"coefficients are shown without intervals")
    coefs = getattr(ctx.model, "coef_", None)
    if coefs is None:
        raise PanelUnavailable(
            f"{type(ctx.model).__name__} exposes neither params nor coef_")
    flat = np.asarray(coefs, dtype=float).ravel()
    names = ctx.X.columns[:flat.size]
    return (pd.DataFrame({"coefficient": flat}, index=names),
            f"{type(ctx.model).__name__} is a penalised point estimator: it has "
            f"no covariance matrix, so no confidence interval exists to draw")


def _panel_coefficient_forest(ctx, ax, top_n=25):
    """Coefficient forest plot, sorted by effect size, with intervals if any."""
    table, note = _coefficient_table(ctx)
    table = table[np.isfinite(table["coefficient"])]
    if table.empty:
        raise PanelUnavailable("every coefficient is non-finite")
    ordered = table.reindex(table["coefficient"].abs()
                            .sort_values(ascending=False).index)
    shown = ordered.head(top_n).iloc[::-1]
    positions = np.arange(len(shown))
    has_ci = {"lower", "upper"}.issubset(shown.columns)
    with figure_style(_REPORT_TARGET):
        if has_ci:
            crosses_zero = ((shown["lower"] <= 0) & (shown["upper"] >= 0))
            # Grey unless the interval excludes zero, and then the sign picks
            # the colour -- the same rule, the same two hues, as
            # spacr.figures.panels.effect_rank, so a reader who has learned
            # green/rust from the volcano sheet reads this panel for free.
            colours = np.where(
                crosses_zero.to_numpy(), ROLES["data"],
                np.where(shown["coefficient"].to_numpy() > 0,
                         ROLES["up"], ROLES["down"]))
            # hlines from lower to upper is exactly what errorbar drew from
            # (coefficient - lower, upper - coefficient); the caps are gone
            # because a cap is ink that carries no number, and the interval
            # can now take one colour per term, which errorbar's single
            # `ecolor` could not.
            ax.hlines(positions, shown["lower"].to_numpy(),
                      shown["upper"].to_numpy(), colors=colours,
                      linewidth=1.0, zorder=1)
        else:
            crosses_zero = pd.Series(False, index=shown.index)
            # No interval means nothing has been called, so nothing is
            # coloured. A penalised fit whose every term came out green or
            # rust would be stating a claim the fit never made.
            colours = np.full(len(shown), ROLES["data"])
        ax.scatter(shown["coefficient"], positions, s=14, c=colours,
                   linewidths=0, zorder=2)
        reference_line(ax, x=0.0)
        ax.set_yticks(positions)
        ax.set_yticklabels([str(s)[:30] for s in shown.index],
                           fontsize=TYPE_SCALE["annotation"])
        ink = _house_axes(ax, "strongest coefficients",
                          f"coefficient ({ctx.family}"
                          + (f" / {ctx.link} link)" if ctx.link else ")"), "")
        text = (f"top {len(shown)} of {len(ordered)} term(s) by |effect|")
        if has_ci:
            text += f"\n{int(crosses_zero.sum())} of {len(shown)} cross zero"
        if note:
            text += "\n" + textwrap.fill(note, 40)
        annotate(ax, text, x=0.98, y=0.02, ha="right", va="bottom", colour=ink)
    return {"n_shown": int(len(shown)), "n_total": int(len(ordered)),
            "has_intervals": bool(has_ci),
            "limitation": note,
            "largest_term": str(ordered.index[0]),
            "largest_coefficient": float(ordered["coefficient"].iloc[0])}


def _p_values(ctx):
    """The p-values to histogram: the screen's coefficient table if we have it."""
    if ctx.coef_df is not None and "p_value" in getattr(ctx.coef_df, "columns", []):
        return np.asarray(ctx.coef_df["p_value"], dtype=float), "coefficient table"
    pvalues = getattr(ctx.model, "pvalues", None)
    if pvalues is not None:
        return np.asarray(pvalues, dtype=float).ravel(), "model.pvalues"
    raise PanelUnavailable(
        f"{type(ctx.model).__name__} produces no p-values and no coefficient "
        f"table was supplied; a penalised fit has no null distribution to test "
        f"against")


def _panel_p_value_histogram(ctx, ax):
    """p-value histogram with the uniform expectation and a stated diagnosis."""
    values, source = _p_values(ctx)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise PanelUnavailable("every p-value is non-finite")
    diag = diagnose_p_value_histogram(finite)
    counts, edges = diag["counts"], diag["edges"]
    bad = diag["verdict"] in ("excess-large", "u-shaped", "anti-uniform")
    with figure_style(_REPORT_TARGET):
        # No bar edge. The old white 0.4pt edge drew white gaps between the
        # bars, which on a transparent or dark ground is a comb of light.
        ax.bar(edges[:-1], counts, width=np.diff(edges), align="edge",
               color=ROLES["fill"], edgecolor="none")
        if np.isfinite(diag["expected"]):
            reference_line(ax, y=diag["expected"], label="uniform")
        ink = _house_axes(ax, "p-value distribution", "p-value",
                          "number of coefficients")
        annotate(ax, f"{_wells(diag['n'], 'coefficients')}\nsource: {source}",
                 x=0.98, ha="right", colour=ink)
        # The verdict is the only thing on this panel that can be a warning,
        # so it is the only thing entitled to a colour. RUST when the shape
        # says the fit is wrong, plain ink when it is not — a verdict drawn
        # red on a healthy histogram teaches the reader to ignore red.
        ax.text(0.5, -0.22, textwrap.fill(diag["message"], 64),
                transform=ax.transAxes, ha="center", va="top",
                fontsize=TYPE_SCALE["annotation"],
                color=ROLES["down"] if bad else ink)
    return {"verdict": diag["verdict"], "message": diag["message"],
            "n": int(diag["n"]), "source": source,
            "frac_below_0.05": diag["frac_below_0.05"],
            "first_bin_ratio": diag["first_bin_ratio"],
            "last_bin_ratio": diag["last_bin_ratio"],
            # Too few coefficients to read a shape is a real limitation of the
            # panel, not a clean pass — it is reported as PARTIAL so nobody
            # quotes the histogram of a five-term model.
            "limitation": (diag["message"] if diag["verdict"] == "too-few"
                           else None)}


def _panel_response_distribution(ctx, ax):
    """The response itself, with the family that was fitted to it named."""
    finite = ctx.y[np.isfinite(ctx.y)]
    if finite.size < 3:
        raise PanelUnavailable("fewer than 3 finite response values")
    bins = int(np.clip(np.sqrt(finite.size), 10, 60))
    family = ctx.family + (f" / {ctx.link} link" if ctx.link else "")
    with figure_style(_REPORT_TARGET):
        # The whole distribution IS the claim here — there is no minority to
        # highlight — so every bar is the house fill and nothing is coloured.
        ax.hist(finite, bins=bins, color=ROLES["fill"], edgecolor="none")
        ink = _house_axes(ax, "response distribution", "response value",
                          "wells")
        annotate(ax, f"{_wells(finite.size)}\nfamily fitted: {family}\n"
                     f"range = [{finite.min():.3g}, {finite.max():.3g}]\n"
                     f"mean = {finite.mean():.3g}, "
                     f"SD = {finite.std(ddof=1):.3g}",
                 x=0.98, ha="right", colour=ink)
    return {"n": int(finite.size), "mean": float(finite.mean()),
            "sd": float(finite.std(ddof=1)), "min": float(finite.min()),
            "max": float(finite.max()), "family": family}


def _panel_calibration(ctx, ax):
    """Calibration curve: does a predicted 0.3 actually happen 30% of the time?"""
    if not ctx.is_binomial:
        raise PanelUnavailable(
            f"calibration is defined for a probability response; this fit uses "
            f"the {ctx.family} family")
    if ctx.n < 15:
        raise PanelUnavailable(
            f"only {ctx.n} wells; a calibration curve needs at least 15 to fill "
            f"three bins")
    n_bins = int(np.clip(ctx.n // 5, 3, 10))
    curve = calibration_curve(ctx.y, ctx.fitted, n_bins=n_bins,
                              weights=ctx.weights)
    ax.plot([0, 1], [0, 1], color=_ACCENT, ls="--", lw=1.2, label="perfect")
    sizes = 20.0 + 120.0 * curve["weight"] / (curve["weight"].max() or 1.0)
    ax.plot(curve["pred_mean"], curve["obs_mean"], color=_POINT, lw=1.4,
            zorder=3)
    ax.scatter(curve["pred_mean"], curve["obs_mean"], s=sizes, color=_POINT,
               zorder=4, label="observed")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    weighted = ctx.weights is not None
    _finish(ax, "Calibration", "mean predicted value in bin",
            "mean observed value in bin", n=ctx.n)
    _note(ax, f"ECE = {curve['ece']:.3f}\nmax gap = {curve['max_gap']:.3f}\n"
              f"Brier = {curve['brier']:.4f}\n"
              f"{curve['n_bins']} bins, "
              f"{'cell-count weighted' if weighted else 'unweighted'}",
          loc="lower right")
    return {"ece": curve["ece"], "max_gap": curve["max_gap"],
            "brier": curve["brier"], "n_bins": int(curve["n_bins"]),
            "weighted": bool(weighted),
            "pred_mean": [float(v) for v in curve["pred_mean"]],
            "obs_mean": [float(v) for v in curve["obs_mean"]]}


def _require_binary(ctx, what):
    """Raise unless the response is binary labels, which ``what`` needs."""
    if not (ctx.is_binomial or ctx.is_classifier):
        raise PanelUnavailable(
            f"{what} is defined for a binary response; this fit uses the "
            f"{ctx.family} family")
    if ctx.is_classifier and not ctx.is_binary_response:
        raise PanelUnavailable(
            f"{what} needs 0/1 labels and this classifier was fitted on a "
            f"response that is not binary; spaCR binarises it before the fit "
            f"(binarise_response), so pass the binarised response to "
            f"build_context if you want this panel")
    if not ctx.is_binary_response:
        raise PanelUnavailable(
            f"{what} needs 0/1 labels, but the response is a continuous "
            f"per-well fraction (spaCR fits logit/probit on fractions weighted "
            f"by cell count). The calibration panel covers this fit instead")
    if ctx.y.min() == ctx.y.max():
        raise PanelUnavailable(
            f"{what} needs both classes; every well has y = {ctx.y[0]:g}")


def _panel_roc(ctx, ax):
    """ROC curve with AUC, for a genuinely binary response.

    Ranked by :attr:`~RegressionQCContext.ranking_score`, whose orientation is
    fixed in :func:`build_context`: larger means more likely positive. That is
    the whole content of this panel and it is the easy thing to get backwards —
    a score built with the opposite convention reports ``1 - AUC``, which for a
    good model lands near 0.05 and for a mediocre one near 0.4, neither of which
    looks obviously wrong on a plot. ``tests/test_regression_orientation.py``
    pins it from both ends.
    """
    _require_binary(ctx, "an ROC curve")
    from sklearn.metrics import roc_auc_score, roc_curve

    score = ctx.ranking_score
    fpr, tpr, _ = roc_curve(ctx.y, score, sample_weight=ctx.weights)
    auc = float(roc_auc_score(ctx.y, score, sample_weight=ctx.weights))
    ax.plot(fpr, tpr, color=_POINT, lw=1.6)
    ax.plot([0, 1], [0, 1], color=_GUIDE, ls="--", lw=1.0, label="chance")
    ax.legend(fontsize=7, frameon=False, loc="lower right")
    n_pos = int(np.sum(ctx.y == 1))
    _finish(ax, "ROC", "false positive rate", "true positive rate", n=ctx.n)
    ranked_by = ("decision function" if ctx.decision_score is not None
                 else "fitted value")
    _note(ax, f"AUC = {auc:.3f}\n{n_pos} positive / {ctx.n - n_pos} negative\n"
              f"ranked by {ranked_by}")
    return {"auc": auc, "n_positive": n_pos, "n_negative": int(ctx.n - n_pos),
            "ranked_by": ranked_by}


def _panel_precision_recall(ctx, ax):
    """Precision-recall curve with average precision and the prevalence baseline.

    Ranked by the same oriented score as :func:`_panel_roc`.
    """
    _require_binary(ctx, "a precision-recall curve")
    from sklearn.metrics import average_precision_score, precision_recall_curve

    score = ctx.ranking_score
    precision, recall, _ = precision_recall_curve(ctx.y, score,
                                                  sample_weight=ctx.weights)
    ap = float(average_precision_score(ctx.y, score,
                                       sample_weight=ctx.weights))
    prevalence = float(np.mean(ctx.y == 1))
    ax.plot(recall, precision, color=_POINT, lw=1.6)
    ax.axhline(prevalence, color=_GUIDE, ls="--", lw=1.0,
               label=f"prevalence = {prevalence:.2f}")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=7, frameon=False, loc="lower left")
    _finish(ax, "Precision-recall", "recall", "precision", n=ctx.n)
    _note(ax, f"average precision = {ap:.3f}\nbaseline = {prevalence:.3f}",
          loc="upper right")
    return {"average_precision": ap, "prevalence": prevalence}


def _panel_count_fit(ctx, ax):
    """Observed vs predicted counts, annotated with the Pearson dispersion."""
    if not ctx.is_count:
        raise PanelUnavailable(
            f"over-dispersion is a property of a count fit; this model uses "
            f"the {ctx.family} family")
    family = getattr(ctx.model, "family", None)
    variance = getattr(family, "variance", None) if family is not None else None
    df_resid = getattr(ctx.model, "df_resid", None)
    if df_resid is None or not np.isfinite(float(df_resid)) or float(df_resid) <= 0:
        df_resid = max(ctx.n - ctx.p, 1)
    stats = overdispersion_statistic(ctx.y, ctx.fitted, float(df_resid),
                                     variance=variance, weights=ctx.weights)
    ax.scatter(ctx.fitted, ctx.y, s=18, alpha=0.6, color=_POINT,
               edgecolors="none")
    hi = float(max(np.nanmax(ctx.fitted), np.nanmax(ctx.y)))
    ax.plot([0, hi], [0, hi], color=_ACCENT, ls="--", lw=1.2, label="identity")
    # Poisson's own +/- 2 SD envelope: outside it the mean-variance assumption
    # is visibly failing, which is the whole point of the panel.
    grid = np.linspace(0, hi, 128)
    ax.fill_between(grid, np.maximum(grid - 2 * np.sqrt(grid), 0),
                    grid + 2 * np.sqrt(grid), color=_GUIDE, alpha=0.18,
                    label="±2 SD under the assumed variance")
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    outside = int(np.sum(np.abs(ctx.y - ctx.fitted)
                         > 2 * np.sqrt(np.maximum(ctx.fitted, 1e-12))))
    _finish(ax, "Count fit and dispersion", "predicted count",
            "observed count", n=ctx.n)
    _note(ax, f"Pearson dispersion = {stats['dispersion']:.2f}\n"
              + textwrap.fill(stats["verdict"], 34)
              + f"\n{outside} well(s) outside ±2 SD", loc="lower right")
    return {"dispersion": stats["dispersion"],
            "pearson_chi2": stats["pearson_chi2"],
            "df_resid": stats["df_resid"], "verdict": stats["verdict"],
            "n_outside_2sd": outside}


# ---------------------------------------------------------------------------
# Panels — screen level
# ---------------------------------------------------------------------------


def _grouped_residuals(ctx, column):
    """``(groups, values)`` of residuals grouped by a metadata column."""
    if ctx.metadata is None:
        raise PanelUnavailable(
            "no per-well metadata was supplied, so plate position is unknown")
    if column not in ctx.metadata.columns:
        raise PanelUnavailable(
            f"metadata has no {column!r} column (present: "
            f"{', '.join(map(str, ctx.metadata.columns[:8]))})")
    keys = ctx.metadata[column].astype(str).to_numpy()
    groups = sorted(pd.unique(keys), key=_natural_key)
    if len(groups) < 2:
        raise PanelUnavailable(
            f"every well has {column} = {groups[0] if groups else 'nothing'}; "
            f"there is no between-{column} effect to see")
    values = [ctx.resid[keys == g] for g in groups]
    return groups, values


#: How strongly the group medians have to disagree before the positional
#: panels spend a colour on one of them. NOT a new statistic: a display
#: threshold on the Kruskal-Wallis p the panel already computes and prints.
_POSITION_ALPHA = 0.05


def _ink_for_ground(ax):
    """``'print'`` or ``'screen'``, decided by the ground this axes sits on.

    :func:`~spacr.figures.style.theme_target` answers "where is this figure
    going" from the user's preference, and it is the right answer for a figure
    that goes to the GUI canvas. Every panel of THIS report goes to a file:
    :func:`regression_qc_report` builds a bare ``Figure`` and saves it, so
    when nothing has put the house style around the save, the ground is
    matplotlib's opaque white — and screen ink on white is an unreadable
    report.

    So the ink follows the ground that is actually there: near-black on a
    light opaque page, and the user's own answer on a transparent or dark one.
    Correct whichever way the driver ends up being wrapped, which is the point.
    """
    from matplotlib.colors import to_rgba

    red, green, blue, opacity = to_rgba(ax.figure.get_facecolor())
    lightness = 0.299 * red + 0.587 * green + 0.114 * blue
    if opacity > 0.5 and lightness > 0.6:
        return "print"
    return theme_target()


def _positional_effect_panel(ctx, ax, column, label, mark_edges):
    """Boxplot of residuals by plate/row/column, with an edge-effect statistic.

    Drawn in the house style of :mod:`spacr.figures.style`, which implements
    the ``apicomplexan-figures`` skill. Its one rule: **everything is grey
    except what the sentence is about.** The sentence here is "does position
    change the residual?", so every mark starts grey and colour is spent only
    where the statistics this panel already computes say there is something to
    look at:

    * BLUE on the group with the largest ``|median|``, and only when
      Kruskal-Wallis rejects at :data:`_POSITION_ALPHA`. With no rejection
      there is no claim to make and the panel comes out entirely grey, which
      is the honest picture of "no positional effect".
    * RUST on the outer groups, and only when the edge statistic fires — the
      same ``|edge - interior| > 0.5 x residual SD`` test that used to append
      ``<-- edge artefact`` to the note. Colouring the wells states it; the
      8%-alpha red ``axvspan`` this replaces was invisible at report size.

    THE STYLE IS APPLIED WITH THE CONTEXT MANAGER, never by writing rcParams.
    spaCR draws from a long-lived GUI process, and a global style change
    restyles every later figure in the session.

    The axes is made by the caller, before that context exists, so its spines,
    ticks and labels are re-inked here explicitly — only artists created inside
    the ``with`` block pick the style up on their own.

    THE STATISTICS ARE UNCHANGED. They are computed before the drawing rather
    than during it, because what gets colour is decided by them.
    """
    from scipy import stats as sps

    groups, values = _grouped_residuals(ctx, column)

    medians = np.array([np.median(v) if v.size else np.nan for v in values])
    worst = int(np.nanargmax(np.abs(medians)))
    try:
        _, kruskal_p = sps.kruskal(*[v for v in values if v.size])
    except ValueError:
        # scipy refuses when every value is identical; that is a legitimate
        # answer ("no difference"), not a failure of the panel.
        kruskal_p = float("nan")

    edge_delta = float("nan")
    if mark_edges and len(groups) >= 4:
        edge = np.concatenate([values[0], values[-1]])
        interior = np.concatenate(values[1:-1])
        if edge.size and interior.size:
            edge_delta = float(np.median(edge) - np.median(interior))
    edge_artefact = bool(np.isfinite(edge_delta)
                         and abs(edge_delta) > 0.5 * np.nanstd(ctx.resid))

    # Who carries colour. Grey by default; the edge claim overrides the
    # largest-median one where the two name the same group, so a fired edge
    # statistic is never quietly recoloured into something else.
    marks = [ROLES["data"]] * len(groups)
    summaries = [ROLES["control_negative"]] * len(groups)
    if np.isfinite(kruskal_p) and kruskal_p < _POSITION_ALPHA:
        marks[worst] = summaries[worst] = ROLES["highlight"]
    if edge_artefact:
        for index in (0, len(groups) - 1):
            marks[index] = summaries[index] = ROLES["down"]
    highlighted = [str(g) for g, colour in zip(groups, marks)
                   if colour != ROLES["data"]]

    target = _ink_for_ground(ax)
    with figure_style(target):
        ink = resolve_ink(target)
        box = ax.boxplot(values, positions=np.arange(len(groups)), widths=0.65,
                         showfliers=False)
        for index, colour in enumerate(summaries):
            box["boxes"][index].set(color=colour, lw=WEIGHTS["spine"])
            # The median bar is the number a reader actually takes off this
            # panel, so it is the one artist drawn at data weight.
            box["medians"][index].set(color=colour, lw=WEIGHTS["data"])
            for part in ("whiskers", "caps"):
                for artist in box[part][2 * index:2 * index + 2]:
                    artist.set(color=colour, lw=WEIGHTS["spine"])
        for i, v in enumerate(values):
            if v.size:
                jitter = (np.random.default_rng(i).uniform(-0.16, 0.16, v.size)
                          if v.size > 1 else np.zeros(1))
                # The skill's superplot exception: the small raw points are
                # the only marks allowed alpha, so the summary reads on top.
                ax.scatter(i + jitter, v, s=4.5, alpha=0.5, color=marks[i],
                           edgecolors="none",
                           zorder=4 if marks[i] != ROLES["data"] else 3)
        reference_line(ax, y=0.0)

        ax.set_xticks(np.arange(len(groups)))
        names = [str(g)[:10] for g in groups]
        ax.set_xticklabels(names)
        for name, spine in ax.spines.items():
            spine.set_visible(name in ("left", "bottom"))
            spine.set_color(ink)
            spine.set_linewidth(WEIGHTS["spine"])
        ax.tick_params(axis="both", colors=ink, labelsize=TYPE_SCALE["tick"],
                       width=WEIGHTS["spine"], length=2.6)
        # 45 degrees is the rule for labels that would not fit flat. `r16` and
        # `c24` do fit, and rotating them costs a third of the panel's height
        # for nothing; `plate1` does not.
        if max(len(name) for name in names) > 3:
            rotate_ticks(ax, 45)
        descriptor(ax, f"residuals by {label}")
        ax.title.set_color(ink)
        ax.set_xlabel(label, fontsize=TYPE_SCALE["label"], color=ink)
        ax.set_ylabel("residual", fontsize=TYPE_SCALE["label"], color=ink)

        # "observations", not "wells": `ctx.n` counts DESIGN ROWS, which is
        # one per well only when the design has one row per well. The
        # guide-level screen designs put several rows in a well -- the real
        # tsg101 fit has 1,945 of them across 610 wells -- and every one is a
        # mark here. Saying "wells" would misstate the unit of replication by
        # a factor of three on the screen this panel was verified against.
        text = (f"{len(groups)} {label}s, n = {ctx.n:,} observations\n"
                f"Kruskal-Wallis p = {kruskal_p:.2g}\n"
                f"largest |median| = {medians[worst]:+.3g} ({groups[worst]})")
        if np.isfinite(edge_delta):
            text += f"\nedge - interior median = {edge_delta:+.3g}"
        annotate(ax, text, x=0.98, y=0.97, ha="right", colour=ink)

        entries = []
        if marks[worst] == ROLES["highlight"]:
            entries.append((f"{groups[worst]}: largest |median|",
                            ROLES["highlight"]))
        if edge_artefact:
            entries.append((f"edge {label}s: edge artefact", ROLES["down"]))
        if entries:
            text_legend(ax, entries)

    return {"n_groups": len(groups), "groups": [str(g) for g in groups],
            "medians": [float(m) for m in medians],
            "kruskal_p": float(kruskal_p),
            "worst_group": str(groups[worst]),
            "worst_median": float(medians[worst]),
            "edge_minus_interior_median": edge_delta,
            "highlighted_groups": highlighted}


def _panel_plate_effects(ctx, ax):
    """Residuals by plate — a plate that fit differently is a batch effect."""
    return _positional_effect_panel(ctx, ax, schema.PLATE_KEY, "plate",
                                    mark_edges=False)


def _panel_row_effects(ctx, ax):
    """Residuals by plate row, with the outer rows shaded (edge artefacts)."""
    return _positional_effect_panel(ctx, ax, schema.ROW_KEY, "row",
                                    mark_edges=True)


def _panel_column_effects(ctx, ax):
    """Residuals by plate column, with the outer columns shaded."""
    return _positional_effect_panel(ctx, ax, schema.COLUMN_KEY, "column",
                                    mark_edges=True)


def _panel_cell_count_vs_effect(ctx, ax):
    """Cell count vs |standardised residual| — do low-n wells drive the tails?"""
    from scipy import stats as sps

    _require_standardisation(ctx, "the cell-count-vs-residual panel")
    counts = None
    if ctx.metadata is not None and "cell_count" in ctx.metadata.columns:
        counts = np.asarray(ctx.metadata["cell_count"], dtype=float)
    elif ctx.weights is not None:
        counts = ctx.weights
    if counts is None:
        raise PanelUnavailable(
            "no per-well cell count available (metadata has no 'cell_count' "
            "column and the fit carried no weights)")
    magnitude = np.abs(ctx.std_resid)
    good = np.isfinite(counts) & np.isfinite(magnitude) & (counts > 0)
    if good.sum() < 5:
        raise PanelUnavailable(
            f"only {int(good.sum())} well(s) have both a positive cell count "
            f"and a finite residual")
    x, mag = counts[good], magnitude[good]
    rho, pval = sps.spearmanr(x, mag)
    low_cut = float(np.quantile(x, 0.1))
    extreme = mag > 2.0
    frac_low = (float(np.mean(x[extreme] <= low_cut)) if np.any(extreme)
                else float("nan"))
    # The panel's whole sentence is "the tails are the small wells", and
    # until now it was stated only in the text block while every point on the
    # axes wore the same colour. These are the wells the sentence is about.
    driving = extreme & (x <= low_cut)
    with figure_style(_REPORT_TARGET):
        ax.scatter(x[~driving], mag[~driving], s=18, color=ROLES["data"],
                   edgecolors="none")
        ax.scatter(x[driving], mag[driving], s=18, color=ROLES["down"],
                   edgecolors="none", zorder=4)
        ax.set_xscale("log")
        reference_line(ax, y=2.0, label="|z| = 2")
        reference_line(ax, x=low_cut, label="10th pct")
        ink = _house_axes(ax, "cell count vs residual",
                          "cells in well (log scale)",
                          "|standardised residual|")
        # A log axis grows minor ticks, which `_house_axes` does not touch
        # because no other panel has any; left alone they stay matplotlib's
        # black while every other mark on the page follows the ink.
        ax.tick_params(which="minor", colors=ink, width=WEIGHTS["spine"],
                       length=1.4)
        # Right-hand corner, not left: the decile guide is the 10th percentile
        # of the counts, so it is always near the left edge and its label runs
        # up the top of the axes -- exactly where a left-hand note sits.
        annotate(ax, f"{_wells(int(good.sum()))}\n"
                     f"Spearman rho = {rho:+.2f} (p = {pval:.2g})\n"
                     f"10th pct count = {low_cut:.0f} cells\n"
                     f"{int(extreme.sum())} well(s) with |z| > 2; "
                     f"{'n/a' if not np.isfinite(frac_low) else f'{100 * frac_low:.0f}%'} "
                     f"of them are in the smallest decile",
                 x=0.98, ha="right", colour=ink)
    return {"spearman_rho": float(rho), "spearman_p": float(pval),
            "low_count_threshold": low_cut,
            "n_extreme": int(extreme.sum()),
            "frac_extreme_in_low_decile": frac_low,
            "min_cell_count": float(x.min()), "n_points": int(good.sum())}


def _panel_volcano_reference(ctx, ax):
    """Point at the volcano plot rather than drawing a second one.

    A signpost, not a panel: no data, no axes, no marks. So the house style
    has almost nothing to say about it beyond the two things that were
    actually wrong — the ink and the type.

    THE INK. ``#333333`` body text is a light-page assumption, and this report
    is written with the ground the caller gives it. On spaCR's dark theme the
    body of this card was near-invisible: the only panel in the suite whose
    entire content is text was also the only one that could vanish completely.
    :func:`_ink_for_ground` resolves it against the page instead.

    THE TYPE. A 10 pt bold heading made this signpost the loudest thing on the
    combined page, louder than any real panel's axis labels at 7 pt. It now
    sits on :data:`~spacr.figures.style.TYPE_SCALE`, so a card that says "the
    figure is elsewhere" is quieter than the figures that are here.

    Nothing else is restyled: giving it a palette, a reference line or a panel
    letter would add ink to a signpost.
    """
    ax.set_axis_off()
    if ctx.volcano_path:
        body = (textwrap.fill("The volcano plot for this run was written by "
                              "spacr.plot.volcano_plot to", 52)
                + f"\n\n{os.path.basename(ctx.volcano_path)}\n\n"
                # Wrapped: the real screen's results folder is 79 characters
                # and ran off both sides of the panel.
                + textwrap.fill(f"in {os.path.dirname(ctx.volcano_path) or '.'}",
                                52))
        state = "referenced"
    else:
        body = (textwrap.fill("The volcano plot is drawn by "
                              "spacr.plot.volcano_plot / "
                              "spacr.toxo.custom_volcano_plot in the "
                              "regression step.", 52)
                + "\n\n"
                + textwrap.fill("No path was passed to this report, so it "
                                "cannot be named here.", 52))
        state = "unlocated"

    target = _ink_for_ground(ax)
    with figure_style(target):
        ink = resolve_ink(target)
        ax.text(0.5, 0.66, "volcano plot", ha="center", va="center",
                fontsize=TYPE_SCALE["label"], fontweight="bold", color=ink,
                transform=ax.transAxes)
        annotate(ax, body, x=0.5, y=0.42, ha="center", va="center", colour=ink)
        ax.text(0.5, 0.06, "not duplicated here on purpose — one implementation",
                ha="center", va="center", fontsize=TYPE_SCALE["legend"],
                style="italic", transform=ax.transAxes,
                color=ROLES["reference"])
    return {"state": state, "volcano_path": ctx.volcano_path}


# ---------------------------------------------------------------------------
# Panel registry
# ---------------------------------------------------------------------------

_PANELS: Tuple[Tuple[str, str, str, Callable[[Any, Any], Dict[str, Any]]], ...] = (
    ("residuals_vs_fitted", "Residuals vs fitted", "fit", _panel_residuals_vs_fitted),
    ("residual_distribution", "Residual distribution", "fit", _panel_residual_distribution),
    ("scale_location", "Scale-location", "fit", _panel_scale_location),
    ("qq_residuals", "Normal Q-Q", "fit", _panel_qq_residuals),
    ("observed_vs_predicted", "Observed vs predicted", "fit", _panel_observed_vs_predicted),
    ("cooks_distance", "Cook's distance", "influence", _panel_cooks_distance),
    ("influence", "Leverage vs standardised residual", "influence", _panel_influence),
    ("dffits", "DFFITS", "influence", _panel_dffits),
    ("vif", "Variance inflation", "design", _panel_vif),
    ("condition_number", "Design conditioning", "design", _panel_condition_number),
    ("predictor_correlation", "Predictor correlation", "design", _panel_predictor_correlation),
    ("response_distribution", "Response distribution", "response", _panel_response_distribution),
    ("coefficient_forest", "Coefficient forest", "response", _panel_coefficient_forest),
    ("p_value_histogram", "p-value distribution", "response", _panel_p_value_histogram),
    ("calibration", "Calibration", "response", _panel_calibration),
    ("roc", "ROC", "response", _panel_roc),
    ("precision_recall", "Precision-recall", "response", _panel_precision_recall),
    ("count_fit", "Count fit and dispersion", "response", _panel_count_fit),
    ("plate_effects", "Residuals by plate", "screen", _panel_plate_effects),
    ("row_effects", "Residuals by row", "screen", _panel_row_effects),
    ("column_effects", "Residuals by column", "screen", _panel_column_effects),
    ("cell_count_vs_effect", "Cell count vs residual", "screen", _panel_cell_count_vs_effect),
    ("volcano_reference", "Volcano plot", "response", _panel_volcano_reference),
)

#: Panel names in report order. Stable: these are file stems on disk.
PANEL_ORDER: Tuple[str, ...] = tuple(name for name, _, _, _ in _PANELS)

_PANEL_BY_NAME = {name: (title, group, fn) for name, title, group, fn in _PANELS}

#: Report section headings, in order.
_GROUP_TITLES = (
    ("fit", "Model fit"),
    ("influence", "Influence and leverage"),
    ("design", "Design and collinearity"),
    ("response", "Response and coefficients"),
    ("screen", "Screen-level structure"),
)


def panel_names(group=None):
    """Return the panel names, optionally restricted to one report section.

    :param group: ``'fit'``, ``'influence'``, ``'design'``, ``'response'``,
        ``'screen'`` or ``None`` for all.
    :returns: Tuple of panel names in report order.
    :raises ValueError: on an unknown group.
    """
    if group is None:
        return PANEL_ORDER
    known = {g for g, _ in _GROUP_TITLES}
    if group not in known:
        raise ValueError(f"unknown panel group {group!r}; expected one of {sorted(known)}")
    return tuple(name for name, _, g, _ in _PANELS if g == group)


def draw_panel(name, ctx, ax):
    """Draw one panel onto an axes and return the statistics it computed.

    This is the unit the tests drive: it takes an axes the caller owns, so the
    contents (how many points the scatter has, what the annotation says) can be
    asserted directly rather than inferred from a PDF's existence.

    :param name: A name from :data:`PANEL_ORDER`.
    :param ctx: :class:`RegressionQCContext` from :func:`build_context`.
    :param ax: A matplotlib ``Axes`` to draw into.
    :returns: dict of statistics; the exact keys are per panel.
    :raises KeyError: if ``name`` is not a known panel.
    :raises PanelUnavailable: if the panel cannot be computed for this model,
        with the reason as the message.
    """
    if name not in _PANEL_BY_NAME:
        raise KeyError(
            f"unknown QC panel {name!r}; known panels: {', '.join(PANEL_ORDER)}")
    _, _, fn = _PANEL_BY_NAME[name]
    return fn(ctx, ax)


# ---------------------------------------------------------------------------
# Report driver
# ---------------------------------------------------------------------------


def _save(fig, path):
    """Write a figure and return the path. Never touches pyplot's registry."""
    fig.savefig(path, bbox_inches="tight")
    # Figures built via matplotlib.figure.Figure are not registered with
    # pyplot, so there is nothing for plt.close() to close; dropping the last
    # reference is the whole clean-up. clf() is belt-and-braces for the case
    # where a panel parked a callback on the figure.
    fig.clf()
    return path


def format_qc_report(manifest):
    """Render a manifest as the plain-text report that is written next to the PDFs.

    :param manifest: The dict returned by :func:`regression_qc_report`.
    :returns: A multi-line string, one block per report section.

    The text form exists because the PDF cannot be grepped and because the
    numbers — dispersion, condition number, the p-value verdict — are the part
    a reviewer quotes.
    """
    lines = ["spaCR regression QC report", "=" * 60,
             f"model            : {manifest.get('model')}",
             f"regression type  : {manifest.get('regression_type')}",
             f"family / link    : {manifest.get('family')}"
             + (f" / {manifest['link']}" if manifest.get("link") else ""),
             f"observations     : {manifest.get('n_observations')} wells",
             f"predictors       : {manifest.get('n_predictors')}",
             f"leverage source  : {manifest.get('leverage_source')}",
             # The residual scale is on the header because it sets every |z|
             # on the influence panels, and getting it wrong is silent: the
             # wells are still named, they are just the wrong wells.
             f"standardised by  : {manifest.get('residual_scale')}",
             f"standardised what: {manifest.get('standardised_quantity')}",
             f"output directory : {manifest.get('directory')}", ""]
    for note in manifest.get("notes", ()):
        lines.append(f"note: {note}")
    if manifest.get("notes"):
        lines.append("")

    by_group: Dict[str, List[QCPanelResult]] = {}
    for panel in manifest["panels"]:
        by_group.setdefault(panel.group, []).append(panel)

    for group, title in _GROUP_TITLES:
        panels = by_group.get(group)
        if not panels:
            continue
        lines.append(title)
        lines.append("-" * len(title))
        for panel in panels:
            head = f"  [{panel.status.upper():<7}] {panel.title}"
            lines.append(head)
            if panel.reason:
                lines.append(textwrap.fill(panel.reason, 72,
                                           initial_indent="      reason: ",
                                           subsequent_indent="              "))
            for key, value in panel.stats.items():
                if value is None:
                    continue
                if isinstance(value, float):
                    rendered = f"{value:.6g}"
                elif isinstance(value, (list, tuple)):
                    if len(value) > 8:
                        rendered = f"[{len(value)} values]"
                    else:
                        rendered = ", ".join(str(v) for v in value)
                elif isinstance(value, dict):
                    rendered = f"[{len(value)} entries]"
                else:
                    rendered = str(value)
                lines.append(f"      {key}: {rendered}")
            if panel.path:
                lines.append(f"      file: {os.path.basename(panel.path)}")
        lines.append("")

    written = sum(1 for p in manifest["panels"] if p.status in ("written", "partial"))
    skipped = [p for p in manifest["panels"] if p.status == "skipped"]
    failed = [p for p in manifest["panels"] if p.status == "failed"]
    lines.append("=" * 60)
    lines.append(f"{written} panel(s) drawn, {len(skipped)} skipped, "
                 f"{len(failed)} failed")
    for panel in skipped:
        lines.append(f"  skipped: {panel.name} — {panel.reason}")
    for panel in failed:
        lines.append(f"  FAILED : {panel.name} — {panel.reason}")
    return "\n".join(lines) + "\n"


def regression_qc_report(model, X, y, dst, *, weights=None, metadata=None,
                         coef_df=None, regression_type=None, volcano_path=None,
                         panels=None, fmt="pdf", combined=True, strict=False,
                         verbose=True):
    """Write the full regression QC suite and return a manifest of what was written.

    Every panel is attempted. A panel that cannot be computed for this model —
    no p-values from a Lasso, no plate column in the metadata, no calibration
    for a Gaussian fit — is recorded as ``skipped`` with the reason, shown as a
    grey tile on the combined page and listed in the text report. It is never
    dropped in silence, because "the panel is not there" and "the panel is
    fine" must not look the same.

    :param model: Fitted model, as returned by
        :func:`spacr.ml.regression_model`.
    :param X: The design matrix that was fitted (DataFrame preferred, so the
        panels can name the predictors).
    :param y: The response that was fitted.
    :param dst: Results folder for the run. The report goes into
        ``<dst>/regression_qc/``.
    :param weights: Per-observation weights passed to the fit (spaCR passes
        cell counts for the GLM-binomial path).
    :param metadata: Per-well frame carrying ``plateID`` / ``rowID`` /
        ``columnID`` / ``prc`` / ``cell_count``. Aligned to the fitted rows by
        index, or by length; anything else raises rather than risk labelling
        the wrong well.
    :param coef_df: The coefficient table from
        :func:`spacr.ml.process_model_coefficients`, used for the p-value
        histogram so it shows the *screen's* p-values.
    :param regression_type: The spaCR regression type string, used to pick the
        family-specific panels when the model object does not say.
    :param volcano_path: Path of the volcano plot for this run, named on the
        report instead of drawing a second volcano.
    :param panels: Optional subset of :data:`PANEL_ORDER`.
    :param fmt: Figure format for the individual panels. Default ``'pdf'``.
    :param combined: Also write the single-page multi-panel report.
    :param strict: Re-raise a panel's unexpected exception instead of recording
        it as ``failed``. Tests use this; the pipeline should not, because a
        broken diagnostic must not take down a fit that already succeeded.
    :param verbose: Print the destination and any failure.
    :returns: dict manifest with ``directory``, ``combined``, ``report``,
        ``panels`` (list of :class:`QCPanelResult`), ``written``, ``skipped``,
        ``failed`` and the model description.
    :raises ValueError: if ``dst`` is falsy — a QC report nobody can find is
        worse than no QC report.

    Example:
        .. code-block:: python

            from spacr.regression_qc import regression_qc_report
            manifest = regression_qc_report(
                model, X, y, dst=res_folder,
                metadata=merged_df.loc[X.index,
                                       ['plateID', 'rowID', 'columnID',
                                        'prc', 'cell_count']],
                coef_df=coef_df, regression_type='ols')
            print(len(manifest['written']), 'panels written')
    """
    if not dst:
        raise ValueError(
            "regression_qc_report needs a destination folder; pass the run's "
            "results directory. Writing diagnostics nowhere is the same as not "
            "computing them.")
    out_dir = os.path.join(str(dst), QC_DIRNAME)
    os.makedirs(out_dir, exist_ok=True)

    ctx = build_context(model, X, y, weights=weights, metadata=metadata,
                        coef_df=coef_df, regression_type=regression_type,
                        volcano_path=volcano_path)
    selected = tuple(panels) if panels is not None else PANEL_ORDER
    unknown = [name for name in selected if name not in _PANEL_BY_NAME]
    if unknown:
        raise ValueError(f"unknown QC panel(s): {unknown}; known: {list(PANEL_ORDER)}")

    results: List[QCPanelResult] = []
    for name in selected:
        title, group, fn = _PANEL_BY_NAME[name]
        fig = Figure(figsize=(5.6, 4.4), dpi=140)
        ax = fig.subplots()
        try:
            stats = fn(ctx, ax)
        except PanelUnavailable as exc:
            fig.clf()
            results.append(QCPanelResult(name=name, title=title, group=group,
                                         status="skipped", reason=str(exc)))
            continue
        except Exception as exc:                # noqa: BLE001 - reported, see below
            fig.clf()
            if strict:
                raise
            # A diagnostic that crashes must be loud (it is printed and it is
            # on the report as FAILED) but must not destroy a fit that already
            # succeeded and cost an hour.
            message = f"{type(exc).__name__}: {exc}"
            if verbose:
                print(f"[regression_qc] panel {name!r} failed: {message}")
            results.append(QCPanelResult(name=name, title=title, group=group,
                                         status="failed", reason=message))
            continue
        limitation = stats.get("limitation") if isinstance(stats, dict) else None
        path = os.path.join(out_dir, f"{name}.{fmt}")
        fig.tight_layout()
        _save(fig, path)
        results.append(QCPanelResult(
            name=name, title=title, group=group,
            status="partial" if limitation else "written",
            path=path, reason=limitation, stats=dict(stats)))

    combined_path = None
    if combined:
        combined_path = _write_combined_page(ctx, results, out_dir, selected)

    manifest = {
        "directory": out_dir,
        "combined": combined_path,
        "report": None,
        "panels": results,
        "written": [r.path for r in results if r.path],
        "skipped": [(r.name, r.reason) for r in results if r.status == "skipped"],
        "failed": [(r.name, r.reason) for r in results if r.status == "failed"],
        "model": type(model).__name__,
        "regression_type": regression_type,
        "family": ctx.family,
        "link": ctx.link,
        "n_observations": ctx.n,
        "n_predictors": ctx.p,
        "leverage_source": ctx.leverage_source,
        "residual_scale": (ctx.standardisation.source
                           if ctx.standardisation is not None else "unknown"),
        "standardised_quantity": (ctx.standardisation.metric
                                  if ctx.standardisation is not None else "unknown"),
        "residual_scale_available": ctx.standardisation_available,
        "residual_scale_reason": (ctx.standardisation.reason
                                  if ctx.standardisation is not None else None),
        "notes": list(ctx.notes),
    }
    report_path = os.path.join(out_dir, "regression_qc_report.txt")
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write(format_qc_report(manifest))
    manifest["report"] = report_path
    if verbose:
        drawn = sum(1 for r in results if r.status in ("written", "partial"))
        print(f"[regression_qc] {drawn}/{len(results)} panel(s) written to "
              f"{out_dir}")
        for panel in results:
            if panel.status == "skipped":
                print(f"[regression_qc]   skipped {panel.name}: {panel.reason}")
    return manifest


def _write_combined_page(ctx, results, out_dir, selected):
    """Draw every panel again onto one page, skipped ones as grey tiles.

    Redrawing rather than re-parenting the individual axes is deliberate:
    matplotlib artists belong to exactly one figure, and moving them is
    unsupported and silently lossy. The panels are cheap (a few hundred wells),
    so a second pass costs nothing worth optimising.
    """
    by_name = {r.name: r for r in results}
    order = [name for name in selected if name in by_name]
    n_cols = 4
    n_rows = int(np.ceil(len(order) / n_cols))
    fig = Figure(figsize=(4.6 * n_cols, 3.7 * n_rows), dpi=110)
    axes = fig.subplots(n_rows, n_cols, squeeze=False)
    for slot, name in enumerate(order):
        ax = axes[slot // n_cols][slot % n_cols]
        result = by_name[name]
        title, _, fn = _PANEL_BY_NAME[name]
        if result.status in ("skipped", "failed"):
            _skip_box(ax, title, result.reason or "no reason recorded")
            continue
        try:
            fn(ctx, ax)
        except Exception as exc:                # noqa: BLE001
            # The panel drew a moment ago on its own figure, so this can only
            # be an axes-specific problem; state it rather than leaving a
            # blank tile.
            ax.clear()
            _skip_box(ax, title, f"redraw failed: {type(exc).__name__}: {exc}")
    for slot in range(len(order), n_rows * n_cols):
        axes[slot // n_cols][slot % n_cols].set_axis_off()

    drawn = sum(1 for r in results if r.status in ("written", "partial"))
    fig.suptitle(
        f"spaCR regression QC — {ctx.family}"
        + (f" / {ctx.link} link" if ctx.link else "")
        + f" — {ctx.n:,} wells, {ctx.p} predictors — "
          f"{drawn}/{len(results)} panels available",
        fontsize=13, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    path = os.path.join(out_dir, "regression_qc_report.pdf")
    return _save(fig, path)
