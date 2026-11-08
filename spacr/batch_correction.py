"""Plate/batch-effect correction for tabular microscopy measurements.

The implementation is dependency-light (pandas/numpy), preserves the original
row/index order, and never changes metadata columns. It is shared by Image
UMAP, Classify (ML), UMAP hyperparameter search, and regression.

Two methods are alpha (feature item 770): ``tvn`` (typical variation
normalisation, a whitening fitted on the negative controls) and ``harmony``
(Harmony integration of a PCA embedding, through the optional ``harmonypy``
package). ``harmony`` returns an *embedding*, not the input features.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

METHODS = (
    "none",
    "center",
    "zscore",
    "robust_zscore",
    "control_center",
    "combat",
    "tvn",
    "harmony",
)
"""Supported correction methods."""

EMBEDDING_METHODS = ("harmony",)
"""Methods that return a corrected embedding instead of the input features.

They are API-only: the GUI screens write corrected values back into named
feature columns, so their batch_correction dropdowns offer
``set(METHODS) - set(EMBEDDING_METHODS)``.
"""

TVN_WHITENINGS = ("ZCA-cor", "ZCA", "PCA-cor", "PCA")
"""Whitening transforms accepted by ``tvn`` (pycytominer ``Spherize`` names)."""

TVN_SCOPES = ("global", "batch")
"""Where ``tvn`` fits its whitening: on all controls, or per batch."""

HARMONY_MISSING_MESSAGE = (
    "batch_correction='harmony' needs the optional `harmonypy` package "
    "(Korsunsky et al. 2019), which spaCR does not install. Install it into "
    "the environment that runs spaCR with `pip install harmonypy`, or choose "
    "another method such as 'tvn' or 'combat'."
)
"""Error text raised when ``harmony`` is requested without ``harmonypy``."""

NO_COVARIATE = "no_covariate"
"""Explicit declaration that no biological signal needs protecting.

ComBat estimates the batch effect from whatever variation is left after the
design matrix has absorbed the biology. If the biology is not in that design,
it is part of "whatever is left" and gets removed along with the plate effect.
That failure is silent: the corrected table looks cleaner, the batch diagnostic
improves, and the treatment effect is gone.

So :func:`correct_batch_effects` refuses to run ``method="combat"`` until the
caller has answered the question. Pass the covariate to keep, or pass this
constant to state on the record that there is nothing to keep -- which is only
true when every batch holds the same mixture of conditions, or when the output
feeds an unsupervised embedding with no contrast to protect.
"""


@dataclass
class BatchCorrectionReport:
    """Diagnostics for one correction operation.

    :param method: Normalized correction method requested for the operation; it
        remains recorded when a one-batch operation becomes a warned no-op.
    :param batch_column: Human-readable metadata-column name used to identify
        the supplied batch labels.
    :param batches: Sorted distinct batch labels after conversion to strings.
    :param features: Numeric feature-column names returned by the operation,
        whether corrected or left unchanged by a no-op.
    :param rows: Number of input feature rows considered.
    :param controls: Total rows matching the reference controls used by
        ``control_center``; zero for other methods.
    :param centroid_spread_before: Mean across-feature standard deviation of
        batch centroids before correction, or ``None`` when unavailable.
    :param centroid_spread_after: The same batch-centroid diagnostic after
        correction or a no-op, or ``None`` when unavailable.
    :param covariate_columns: Source biological-covariate columns supplied to
        ComBat.
    :param covariate_terms: Design-matrix terms expanded from those covariates
        when ComBat was fitted; empty when no fit was performed.
    :param covariate_spread_before: Batch-centroid-style spread across
        categorical covariate groups before correction, or ``None`` for no or
        continuous covariates.
    :param covariate_spread_after: The same categorical-covariate spread after
        correction or a no-op, or ``None`` when unavailable.
    :param warnings: Explicit no-op, fallback, unchanged-batch,
        constant-feature, or ComBat limitation messages.
    """

    method: str
    batch_column: str
    batches: List[str]
    features: List[str]
    rows: int
    controls: int = 0
    centroid_spread_before: Optional[float] = None
    centroid_spread_after: Optional[float] = None
    covariate_columns: List[str] = field(default_factory=list)
    covariate_terms: List[str] = field(default_factory=list)
    covariate_spread_before: Optional[float] = None
    covariate_spread_after: Optional[float] = None
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable report."""
        return asdict(self)


def _as_controls(values: Any) -> List[Any]:
    """Normalize a scalar or iterable control specification.

    ``None`` and blank strings produce an empty list. Comma-separated strings
    are split into trimmed values, including when such a string is nested one
    level inside an iterable. Other scalar values become one-element lists.
    """
    if values is None:
        return []
    if isinstance(values, bytes):
        values = values.decode(errors="replace")
    if isinstance(values, str):
        text = values.strip()
        if not text:
            return []
        if "," in text:
            return [part.strip() for part in text.split(",") if part.strip()]
        return [text]
    if isinstance(values, Iterable):
        out: List[Any] = []
        for value in values:
            if isinstance(value, str) and "," in value:
                out.extend(part.strip() for part in value.split(",")
                           if part.strip())
            else:
                out.append(value)
        return out
    return [values]


def _match_values(series: pd.Series, values: Sequence[Any]) -> pd.Series:
    """Match control values across exact, numeric, and string encodings."""
    mask = pd.Series(False, index=series.index)
    text = series.astype(str).str.strip()
    numeric = pd.to_numeric(series, errors="coerce")
    for value in values:
        try:
            mask |= series == value
        except Exception:
            pass
        mask |= text == str(value).strip()
        value_numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        if pd.notna(value_numeric):
            mask |= numeric == value_numeric
    return mask


def _scale(values: pd.DataFrame, robust: bool) -> pd.Series:
    """Return per-feature standard or robust scale with safe fallbacks."""
    if robust:
        medians = values.median(axis=0)
        scale = (values.subtract(medians).abs().median(axis=0) * 1.4826)
    else:
        scale = values.std(axis=0, ddof=0)
    return scale.replace([np.inf, -np.inf, 0], np.nan)


def _center(values: pd.DataFrame, robust: bool) -> pd.Series:
    """Return per-feature mean or median."""
    return values.median(axis=0) if robust else values.mean(axis=0)


def _centroid_spread(values: pd.DataFrame, batch: pd.Series) -> Optional[float]:
    """Measure how far batch centers differ, averaged across features."""
    try:
        centers = values.groupby(batch, observed=True).mean()
        if len(centers) < 2:
            return 0.0
        return float(centers.std(axis=0, ddof=0).mean())
    except Exception:
        return None



#: Convergence threshold and iteration cap for the empirical-Bayes fixed point.
#: Both match ``sva::ComBat``'s ``it.sol``; the cap is ours, because a fixed
#: point that has not moved in 500 rounds is not going to.
_COMBAT_CONV = 1e-4
_COMBAT_MAX_ITER = 500

#: Floor applied to the posterior scale. ``delta*`` divides the standardized
#: data, so a feature that is exactly constant inside one batch would otherwise
#: produce inf.
_COMBAT_MIN_DELTA = 1e-12

#: A feature whose residual variance is this small *relative to its own
#: magnitude* is treated as carrying no information rather than standardized.
_COMBAT_MIN_VAR = 1e-16


def _is_no_covariate(covariate: Any) -> bool:
    """True when the caller explicitly declared there is no biology to keep."""
    if isinstance(covariate, (pd.Series, pd.DataFrame)):
        return False
    if isinstance(covariate, str):
        return covariate.strip().lower() in {NO_COVARIATE, "none", ""}
    return False


def _covariate_frame(covariate: Any) -> pd.DataFrame:
    """Normalize a covariate specification to a DataFrame of columns."""
    if isinstance(covariate, pd.DataFrame):
        return covariate
    if isinstance(covariate, pd.Series):
        name = covariate.name if covariate.name is not None else "covariate"
        return covariate.to_frame(name=str(name))
    raise ValueError(
        "combat covariate must be a pandas Series or DataFrame of the "
        f"biology to preserve, or batch_correction.NO_COVARIATE; got "
        f"{type(covariate).__name__}."
    )


def _covariate_design(
    covariate: pd.DataFrame,
) -> Tuple[np.ndarray, List[str], List[str]]:
    """Expand covariate columns into a full-column-rank design block.

    Floating-point columns are used as-is, on the assumption that a float is a
    real quantity (a dose, an hours-post-infection). Everything else --
    strings, ints, booleans, categoricals -- is treated as a *factor* and
    dummy-coded with the first level dropped. Integer-coded conditions are
    common in plate metadata and reading ``condition in (0, 1, 2)`` as a
    continuous slope would silently fit the wrong model, so the conservative
    reading is the default; cast a column to float to opt into continuous.

    :returns: ``(design_block, term_names, source_columns)``. ``design_block``
        has one row per sample and may have zero columns.
    """
    blocks: List[np.ndarray] = []
    terms: List[str] = []
    sources: List[str] = []
    for name in covariate.columns:
        series = covariate[name]
        label = str(name)
        sources.append(label)
        if series.isna().any():
            raise ValueError(
                f"combat covariate {label!r} is missing for "
                f"{int(series.isna().sum())} row(s); a row whose biology is "
                "unknown cannot be protected from the correction."
            )
        if pd.api.types.is_float_dtype(series):
            values = series.to_numpy(dtype=float)
            if np.ptp(values) == 0:
                continue
            blocks.append(values.reshape(-1, 1))
            terms.append(label)
            continue
        text = series.astype(str)
        levels = sorted(text.unique().tolist())
        if len(levels) < 2:
            continue
        for level in levels[1:]:
            blocks.append((text == level).to_numpy(dtype=float).reshape(-1, 1))
            terms.append(f"{label}={level}")
    if not blocks:
        return np.zeros((len(covariate), 0), dtype=float), terms, sources
    return np.hstack(blocks), terms, sources


def _covariate_key(covariate: Optional[pd.DataFrame]) -> Optional[pd.Series]:
    """Collapse categorical covariates to one grouping key for diagnostics.

    Returns ``None`` when there is nothing to group by -- no covariate, or an
    entirely continuous one, where "the distance between group centers" is not
    a quantity. The diagnostic it feeds is the one that answers "did the
    biology survive", so silently grouping a dose column into 300 singleton
    levels would produce a reassuring number that means nothing.
    """
    if covariate is None or covariate.empty:
        return None
    categorical = [
        name for name in covariate.columns
        if not pd.api.types.is_float_dtype(covariate[name])
    ]
    if not categorical:
        return None
    key = covariate[categorical[0]].astype(str)
    for name in categorical[1:]:
        key = key.str.cat(covariate[name].astype(str), sep="|")
    return key


def _prior_width(estimates: np.ndarray) -> float:
    """Across-feature variance of one parameter -- the prior's room to move.

    A single feature, or a set of features whose estimates all coincide, gives
    a variance that is undefined or zero: there is no spread to learn a prior
    from. Both answer 0.0, which the posterior reads as "no shrinkage room"
    and resolves to the prior mean. Returning NaN instead would carry
    straight through the fixed point into the corrected table. Estimates that
    agree to within rounding (a spread below ``1e-12`` of their magnitude)
    count as coinciding, so the answer does not depend on the BLAS build.
    """
    if estimates.size < 2:
        return 0.0
    variance = float(np.var(estimates, ddof=1))
    magnitude = float(np.mean(np.abs(estimates)))
    if not np.isfinite(variance) or variance <= (1e-12 * magnitude) ** 2:
        return 0.0
    return variance


def _degenerate_scale_prior(delta_hat: np.ndarray) -> bool:
    """True when the inverse-gamma prior on the scale carries no information.

    ``_a_prior``/``_b_prior`` signal this by answering infinity, which is the
    honest limit but not something the closed-form posterior can be evaluated
    with -- see :func:`_eb_fixed_point`.
    """
    return _prior_width(delta_hat) <= 0.0


def _a_prior(delta_hat: np.ndarray) -> float:
    """Inverse-gamma shape from the method of moments (``sva::aprior``).

    Infinity means the estimates carry no spread, so the prior is infinitely
    precise; :func:`_eb_fixed_point` takes that as its limit rather than
    substituting it into the posterior.
    """
    mean = float(np.mean(delta_hat))
    variance = _prior_width(delta_hat)
    if variance <= 0.0:
        return np.inf
    return (2.0 * variance + mean ** 2) / variance


def _b_prior(delta_hat: np.ndarray) -> float:
    """Inverse-gamma scale from the method of moments (``sva::bprior``).

    Infinity carries the same meaning as in :func:`_a_prior`.
    """
    mean = float(np.mean(delta_hat))
    variance = _prior_width(delta_hat)
    if variance <= 0.0:
        return np.inf
    return (mean * variance + mean ** 3) / variance


def _eb_fixed_point(
    standardized: np.ndarray,
    gamma_hat: np.ndarray,
    delta_hat: np.ndarray,
    gamma_bar: float,
    tau2: float,
    a_prior: float,
    b_prior: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Shrink one batch's location/scale toward the across-feature prior.

    This is ``sva::it.sol``: alternate the conditional posterior mean of the
    additive effect given the scale with the conditional posterior mean of the
    scale given the effect, until neither moves. Both posteriors are closed
    form under the normal/inverse-gamma pair, so each round is two vectorized
    expressions over all features at once.

    A prior with no width is handled as its limit rather than by evaluating
    the closed form. When the per-feature scale estimates all coincide -- two
    features that are linear copies of each other is enough -- the method of
    moments sends both inverse-gamma hyper-parameters to infinity, and
    ``(ss / 2 + inf) / (n / 2 + inf - 1)`` is ``inf / inf``: NaN for every
    feature, and every row of that batch lost from whatever is fitted next.
    The limit is exact and finite: an infinitely precise prior leaves the
    posterior at the prior mean, which for this method-of-moments pair is
    ``mean(delta_hat)`` -- full shrinkage to the pooled scale. The same
    reasoning covers ``tau2``: a zero-width normal prior puts gamma at
    ``gamma_bar``, which the closed form already yields.

    :param standardized: ``(n_features, n_rows_in_batch)`` standardized data.
    :param gamma_hat: per-feature additive batch effect, the fixed-point seed.
    :param delta_hat: per-feature multiplicative batch effect.
    :param gamma_bar: mean of ``gamma_hat`` across features -- the prior mean.
    :param tau2: variance of ``gamma_hat`` across features -- the prior width.
    :returns: ``(gamma_star, delta_star)`` posterior means.
    """
    n = standardized.shape[1]
    tau2 = tau2 if np.isfinite(tau2) and tau2 > 0.0 else 0.0
    pooled_delta = float(np.mean(delta_hat))
    flat_prior = not (np.isfinite(a_prior) and np.isfinite(b_prior))
    gamma_old = gamma_hat.copy()
    delta_old = delta_hat.copy()
    gamma_new = gamma_old
    delta_new = delta_old
    for _ in range(_COMBAT_MAX_ITER):
        gamma_new = (
            (tau2 * n * gamma_hat + delta_old * gamma_bar)
            / (tau2 * n + delta_old)
        )
        if flat_prior:
            delta_new = np.full_like(delta_old, pooled_delta)
        else:
            residual = standardized - gamma_new.reshape(-1, 1)
            sum_squares = np.einsum("ij,ij->i", residual, residual)
            delta_new = ((0.5 * sum_squares + b_prior)
                         / (n / 2.0 + a_prior - 1.0))
        delta_new = np.maximum(delta_new, _COMBAT_MIN_DELTA)
        change = max(
            float(np.max(np.abs(gamma_new - gamma_old)
                         / np.maximum(np.abs(gamma_old), _COMBAT_CONV))),
            float(np.max(np.abs(delta_new - delta_old)
                         / np.maximum(np.abs(delta_old), _COMBAT_CONV))),
        )
        gamma_old = gamma_new
        delta_old = delta_new
        if change < _COMBAT_CONV:
            break
    return gamma_new, delta_new


def _combat(
    numeric: pd.DataFrame,
    labels: pd.Series,
    batches: Sequence[str],
    covariate: Optional[pd.DataFrame],
    report: BatchCorrectionReport,
    *,
    mean_only: bool = False,
    empirical_bayes: bool = True,
) -> pd.DataFrame:
    """Apply parametric empirical-Bayes batch adjustment (Johnson et al. 2007).

    Three steps, on features standardized against a design that already
    contains the biology:

    1. Least squares of every feature on ``[batch indicators | covariates]``.
       The covariate coefficients are *kept*; only the batch coefficients are
       treated as nuisance. This is why the covariate is mandatory -- an
       omitted contrast lands in the batch coefficients and is subtracted.
    2. Per batch and feature, an additive shift ``gamma`` and a multiplicative
       scale ``delta``, each shrunk toward the distribution of that same
       parameter across all features in the batch. The shrinkage is what makes
       ComBat usable on a plate with few wells: a per-feature estimate from
       six wells is noise, but thousands of features constrain the prior.
    3. Undo the standardization, keeping the covariate part of the fit.

    :returns: corrected features, same index and columns as ``numeric``.
    :raises ValueError: when batch and covariate are confounded, when a batch
        has fewer than two rows, or when the design is not identifiable.
    """
    values = numeric.to_numpy(dtype=float).T
    n_features, n_rows = values.shape

    batch_design = np.column_stack(
        [(labels == label).to_numpy(dtype=float) for label in batches]
    )
    batch_sizes = batch_design.sum(axis=0)
    thin = [
        str(label) for label, size in zip(batches, batch_sizes) if size < 2
    ]
    if thin:
        raise ValueError(
            f"combat needs at least 2 rows in every {report.batch_column} "
            f"batch to estimate a within-batch variance; {thin} have fewer."
        )

    if covariate is None:
        covariate_design = np.zeros((n_rows, 0), dtype=float)
    else:
        covariate_design, terms, sources = _covariate_design(covariate)
        report.covariate_columns = sources
        report.covariate_terms = terms
        if covariate_design.shape[1] == 0:
            report.warnings.append(
                "The combat covariate had a single level across every row, so "
                "it constrained nothing; the correction ran as if no biology "
                "had been declared."
            )

    design = np.hstack([batch_design, covariate_design])
    n_batch = batch_design.shape[1]
    if design.shape[1] >= n_rows:
        raise ValueError(
            f"combat design has {design.shape[1]} term(s) for {n_rows} row(s); "
            "there is nothing left to estimate the noise from. Use fewer "
            "covariate levels or aggregate to well level first."
        )
    if np.linalg.matrix_rank(design) < design.shape[1]:
        raise ValueError(
            f"combat cannot separate {report.batch_column} from the declared "
            f"biology {report.covariate_columns or ['(none)']}: the two are "
            "confounded, so any batch effect removed takes the treatment "
            "effect with it. Split conditions across batches, or analyze the "
            "batches separately."
        )

    coefficients, *_ = np.linalg.lstsq(design, values.T, rcond=None)
    grand_mean = (batch_sizes / n_rows) @ coefficients[:n_batch, :]
    standard_mean = np.tile(grand_mean.reshape(-1, 1), (1, n_rows))
    if covariate_design.shape[1]:
        standard_mean = standard_mean + (
            covariate_design @ coefficients[n_batch:, :]
        ).T

    residual = values - (design @ coefficients).T
    var_pooled = np.einsum("ij,ij->i", residual, residual) / n_rows
    scale_floor = _COMBAT_MIN_VAR * np.maximum(
        np.var(values, axis=1), np.mean(values ** 2, axis=1) + 1.0,
    )
    degenerate = ~np.isfinite(var_pooled) | (var_pooled <= scale_floor)
    if degenerate.any():
        report.warnings.append(
            f"{int(degenerate.sum())} feature(s) had no residual variance "
            "after the design was fitted and were left unchanged."
        )
    scale = np.sqrt(np.where(degenerate, 1.0, var_pooled))

    standardized = (values - standard_mean) / scale.reshape(-1, 1)

    gram = batch_design.T @ batch_design
    gamma_hat = np.linalg.solve(gram, batch_design.T @ standardized.T)
    delta_hat = np.vstack([
        np.var(standardized[:, batch_design[:, index] > 0], axis=1, ddof=1)
        for index in range(n_batch)
    ])
    if mean_only:
        delta_hat = np.ones_like(delta_hat)
    delta_hat = np.maximum(delta_hat, _COMBAT_MIN_DELTA)

    adjusted = standardized.copy()
    flat_priors = 0
    for index in range(n_batch):
        rows = batch_design[:, index] > 0
        if empirical_bayes and not mean_only:
            flat_priors += int(_degenerate_scale_prior(delta_hat[index]))
            gamma_star, delta_star = _eb_fixed_point(
                standardized[:, rows],
                gamma_hat[index],
                delta_hat[index],
                float(np.mean(gamma_hat[index])),
                _prior_width(gamma_hat[index]),
                _a_prior(delta_hat[index]),
                _b_prior(delta_hat[index]),
            )
        elif empirical_bayes:
            tau2 = _prior_width(gamma_hat[index])
            n_in_batch = int(rows.sum())
            gamma_star = (
                (tau2 * n_in_batch * gamma_hat[index]
                 + delta_hat[index] * float(np.mean(gamma_hat[index])))
                / (tau2 * n_in_batch + delta_hat[index])
            )
            delta_star = delta_hat[index]
        else:
            gamma_star = gamma_hat[index]
            delta_star = delta_hat[index]
        adjusted[:, rows] = (
            (standardized[:, rows] - gamma_star.reshape(-1, 1))
            / np.sqrt(np.maximum(delta_star, _COMBAT_MIN_DELTA)).reshape(-1, 1)
        )

    if flat_priors:
        report.warnings.append(
            f"{flat_priors} batch(es) gave the same scale estimate for every "
            "feature -- a single feature, or features that are linear copies "
            "of one another -- so the empirical-Bayes prior carried no "
            "information and those batches were shrunk fully to the pooled "
            "scale."
        )

    restored = adjusted * scale.reshape(-1, 1) + standard_mean
    restored[degenerate, :] = values[degenerate, :]
    return pd.DataFrame(
        restored.T, index=numeric.index, columns=numeric.columns,
    )


#: Relative eigenvalue floor below which a whitening direction counts as
#: absent from the reference (``n_components="auto"``).
_TVN_RANK_TOL = 1e-10


def spherize_features(
    features: pd.DataFrame,
    reference: Optional[pd.Series] = None,
    *,
    method: str = "ZCA-cor",
    epsilon: float = 1e-6,
    center: bool = True,
    n_components: Any = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Whiten features with a transform fitted on reference (control) rows.

    This is the sphering step of typical variation normalisation (TVN; Ando,
    McLean & Berndl 2017, bioRxiv 161422) with the semantics of pycytominer's
    ``Spherize`` (Serrano et al. 2025, pycytominer,
    ``pycytominer/operations/transform.py``): the reference rows are centred
    on their mean, their covariance (for the ``-cor`` variants, their
    correlation) is eigendecomposed as ``U diag(s) U^T``, and every row is
    mapped by ``W = diag(1/sqrt(s + epsilon)) U^T`` (PCA) or
    ``U diag(1/sqrt(s + epsilon)) U^T`` (ZCA), the ``-cor`` variants first
    dividing by the reference standard deviation. Afterwards the reference
    rows have identity covariance, so variation that is typical of the
    controls -- plate, position and staining noise -- no longer dominates
    distances.

    The four whitenings differ only by a rotation when every direction is
    kept and ``epsilon`` is negligible, so cosine and Euclidean distances
    between transformed rows agree; they differ once ``n_components``
    truncates.

    The reference needs more rows than features to be full rank. With fewer,
    the directions the controls never vary along have eigenvalue 0 and are
    multiplied by ``1/sqrt(epsilon)`` (1000 at the pycytominer default), so
    noise in those directions can dominate. ``n_components="auto"`` keeps
    only the directions the reference spans; an integer keeps that many.

    :param features: numeric feature frame without missing values.
    :param reference: boolean Series aligned to ``features`` marking the rows
        the transform is fitted on (normally negative-control wells); ``None``
        fits on every row.
    :param method: one of :data:`TVN_WHITENINGS`.
    :param epsilon: regularisation added to every eigenvalue.
    :param center: subtract the reference mean before transforming.
    :param n_components: ``None`` keeps every direction (pycytominer),
        ``"auto"`` keeps those with eigenvalue above ``1e-10`` of the largest,
        an integer keeps the largest that many.
    :returns: ``(whitened, info)``. ZCA variants keep the input columns; PCA
        variants return columns ``tvn_PC1 ...``. ``info`` holds ``method``,
        ``epsilon``, ``rank``, ``n_components``, ``n_reference`` and
        ``warnings``.
    :raises ValueError: for an unknown method, missing values, fewer than two
        reference rows, or a ``-cor`` method with a feature that is constant
        in the reference.
    """
    normalized = str(method or "ZCA-cor").strip()
    lookup = {name.lower(): name for name in TVN_WHITENINGS}
    if normalized.lower() not in lookup:
        raise ValueError(
            f"Unknown tvn whitening {method!r}. Choose one of {TVN_WHITENINGS}."
        )
    normalized = lookup[normalized.lower()]
    if not isinstance(features, pd.DataFrame) or features.empty:
        raise ValueError("features must be a non-empty pandas DataFrame.")
    values = features.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(
            "tvn cannot whiten missing or infinite values; impute or drop "
            f"them first ({int((~np.isfinite(values)).sum())} found)."
        )
    if reference is None:
        mask = np.ones(len(features), dtype=bool)
    else:
        if not reference.index.equals(features.index):
            reference = reference.reindex(features.index)
        mask = reference.fillna(False).to_numpy(dtype=bool)
    fit = values[mask]
    if len(fit) < 2:
        raise ValueError(
            "tvn needs at least 2 reference rows to estimate a covariance; "
            f"got {len(fit)}."
        )
    epsilon = float(epsilon)
    if not np.isfinite(epsilon) or epsilon < 0:
        raise ValueError(f"tvn epsilon must be >= 0, not {epsilon!r}.")
    warnings: List[str] = []
    if normalized.endswith("-cor"):
        sd = (fit - (fit.mean(axis=0) if center else 0.0)).std(axis=0, ddof=1)
        constant = ~(sd > 0)
        if constant.any():
            names = [str(c) for c, k in zip(features.columns, constant) if k]
            raise ValueError(
                f"tvn {normalized!r} divides by each feature's reference "
                f"standard deviation, and {int(constant.sum())} feature(s) "
                f"are constant in the reference rows (e.g. {names[:10]}). "
                "Remove them, or use method='ZCA'."
            )
    # The eigenbasis is spaCR's one TVN core, shared with the OPS TVN
    # (spacr.ops_tvn) and the Measure profiles.
    from .ops_tvn import _control_basis

    basis = _control_basis(fit, center=center, scale=normalized.endswith("-cor"))
    mean, inv_sd = basis["mean"], basis["inv_sd"]
    eigvals, eigvecs = basis["eigvals"], basis["eigvecs"]
    top = float(eigvals[0])
    rank = int((eigvals > _TVN_RANK_TOL * max(top, 1e-300)).sum())
    if n_components is None:
        keep = len(eigvals)
    elif isinstance(n_components, str):
        if n_components.strip().lower() != "auto":
            raise ValueError(
                "tvn n_components must be None, 'auto' or an integer, not "
                f"{n_components!r}."
            )
        keep = max(rank, 1)
    else:
        keep = int(n_components)
        if keep < 1:
            raise ValueError(f"tvn n_components must be >= 1, not {keep}.")
        keep = min(keep, len(eigvals))
    if keep > rank:
        warnings.append(
            f"The {int(mask.sum())} reference row(s) span {rank} of "
            f"{len(eigvals)} feature directions; the other {keep - rank} "
            f"were scaled by 1/sqrt(epsilon={epsilon:g}) and can dominate "
            "distances. Pass n_components='auto' or add controls."
        )
    eigvals = eigvals[:keep]
    eigvecs = eigvecs[:, :keep]
    with np.errstate(divide="ignore"):
        scale = 1.0 / np.sqrt(eigvals + epsilon)
    if not np.isfinite(scale).all():
        raise ValueError(
            "tvn met a direction with zero variance and epsilon=0; use a "
            "positive epsilon or n_components='auto'."
        )
    transformed = (((values - mean) * inv_sd) @ eigvecs) * scale
    if normalized.startswith("ZCA"):
        transformed = transformed @ eigvecs.T
        columns = list(features.columns)
    else:
        columns = [f"tvn_PC{index + 1}" for index in range(keep)]
    info = dict(method=normalized, epsilon=epsilon, rank=rank,
                n_components=keep, n_reference=int(mask.sum()),
                warnings=warnings)
    return (pd.DataFrame(transformed, index=features.index, columns=columns),
            info)


def harmony_embedding(
    features: pd.DataFrame,
    batch: pd.Series,
    *,
    n_components: int = 50,
    theta: float = 2.0,
    standardize: bool = True,
    seed: int = 0,
    max_iter: int = 10,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Batch-integrate a PCA embedding with Harmony (Korsunsky et al. 2019).

    Harmony does **not** correct the features themselves. It clusters a
    low-dimensional embedding with soft k-means, penalises clusters that one
    batch dominates (strength ``theta``) and moves every row by a
    cluster-weighted, batch-specific linear correction until the clusters mix
    the batches. The returned frame is that corrected embedding (columns
    ``harmony_1 ...``): use it for neighbours, clustering, UMAP or a
    classifier, not as per-feature measurements.

    Steps: optional standardisation of every feature over all rows, PCA to
    ``n_components`` (capped by the rank of the data), then
    ``harmonypy.run_harmony`` with the batch as its only covariate.
    ``harmonypy`` is an optional dependency that spaCR does not install.

    :param features: numeric feature frame without missing values.
    :param batch: batch label aligned to ``features``.
    :param n_components: number of principal components to integrate.
    :param theta: Harmony diversity penalty; 0 turns the batch penalty off.
    :param standardize: z-score every feature over all rows before the PCA.
    :param seed: random state passed to Harmony.
    :param max_iter: maximum Harmony rounds.
    :returns: ``(embedding, info)``; ``info`` holds ``n_components``,
        ``theta``, ``explained_variance_ratio`` and ``harmonypy_version``.
    :raises ImportError: with :data:`HARMONY_MISSING_MESSAGE` when
        ``harmonypy`` is not installed.
    :raises ValueError: for missing values or fewer than two batches.
    """
    try:
        import harmonypy
    except ImportError as exc:
        raise ImportError(HARMONY_MISSING_MESSAGE) from exc
    if not isinstance(features, pd.DataFrame) or features.empty:
        raise ValueError("features must be a non-empty pandas DataFrame.")
    values = features.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(
            "harmony cannot embed missing or infinite values; impute or drop "
            "them first."
        )
    if not batch.index.equals(features.index):
        batch = batch.reindex(features.index)
    labels = batch.astype(str)
    if labels.nunique() < 2:
        raise ValueError("harmony needs at least two batches.")
    centered = values - values.mean(axis=0)
    if standardize:
        sd = centered.std(axis=0, ddof=0)
        centered = centered / np.where(sd > 0, sd, 1.0)
    u, singular, _ = np.linalg.svd(centered, full_matrices=False)
    rank = int((singular > 1e-10 * max(float(singular[0]), 1e-300)).sum())
    keep = max(1, min(int(n_components), rank, len(values) - 1))
    scores = u[:, :keep] * singular[:keep]
    total = max(float((singular ** 2).sum()), 1e-300)
    meta = pd.DataFrame({"batch": labels.to_numpy()})
    result = harmonypy.run_harmony(
        scores, meta, ["batch"], theta=float(theta), random_state=int(seed),
        max_iter_harmony=int(max_iter), verbose=False,
    )
    corrected = np.asarray(result.Z_corr, dtype=float)
    if corrected.shape[0] != len(values):
        corrected = corrected.T
    columns = [f"harmony_{index + 1}" for index in range(keep)]
    info = dict(n_components=keep, theta=float(theta),
                explained_variance_ratio=float(
                    (singular[:keep] ** 2).sum() / total),
                harmonypy_version=str(getattr(harmonypy, "__version__", "")))
    return (pd.DataFrame(corrected, index=features.index, columns=columns),
            info)


def _tvn(
    numeric: pd.DataFrame,
    labels: pd.Series,
    batches: Sequence[str],
    control: Optional[pd.Series],
    control_values: Any,
    report: BatchCorrectionReport,
    *,
    method: str,
    epsilon: float,
    scope: str,
    center_batches: bool,
    n_components: Any,
    min_samples: int,
) -> pd.DataFrame:
    """Control-centre each batch, then whiten on the controls (``tvn``)."""
    controls = _as_controls(control_values)
    if control is None or not controls:
        raise ValueError(
            "tvn fits its whitening on reference controls, so it requires "
            "batch_control_column and at least one batch_control_value "
            "(normally the negative control, e.g. DMSO)."
        )
    normalized_scope = str(scope or "global").strip().lower()
    if normalized_scope not in TVN_SCOPES:
        raise ValueError(
            f"Unknown tvn scope {scope!r}. Choose one of {TVN_SCOPES}."
        )
    if not control.index.equals(numeric.index):
        control = control.reindex(numeric.index)
    control_mask = _match_values(control, controls)
    report.controls = int(control_mask.sum())
    if report.controls < max(2, min_samples):
        raise ValueError(
            f"Only {report.controls} reference-control row(s) matched "
            f"{controls!r}; tvn needs at least {max(2, min_samples)}."
        )
    thin = [label for label in batches
            if int((control_mask & (labels == label)).sum()) < min_samples]
    if thin and (center_batches or normalized_scope == "batch"):
        raise ValueError(
            f"tvn needs at least {min_samples} reference-control row(s) in "
            f"every {report.batch_column} batch to centre or whiten it; "
            f"{thin} have fewer."
        )
    work = numeric.copy()
    if center_batches:
        for label in batches:
            rows = labels == label
            shift = numeric.loc[rows & control_mask].median(axis=0)
            work.loc[rows] = numeric.loc[rows].subtract(shift, axis=1)
    if normalized_scope == "global":
        out, info = spherize_features(
            work, control_mask, method=method, epsilon=epsilon,
            n_components=n_components,
        )
        report.warnings.extend(info["warnings"])
        report.features = [str(column) for column in out.columns]
        return out
    pieces = []
    for label in batches:
        rows = labels == label
        part, info = spherize_features(
            work.loc[rows], control_mask.loc[rows], method=method,
            epsilon=epsilon, n_components=n_components,
        )
        if list(part.columns) != list(numeric.columns):
            raise ValueError(
                "tvn_scope='batch' needs a ZCA whitening: per-batch PCA "
                "components are not comparable between batches."
            )
        report.warnings.extend(f"{label}: {note}" for note in info["warnings"])
        pieces.append(part)
    return pd.concat(pieces).reindex(numeric.index)


def correct_batch_effects(
    features: pd.DataFrame,
    batch: pd.Series,
    *,
    method: str = "none",
    batch_column: str = "plateID",
    control: Optional[pd.Series] = None,
    control_values: Any = None,
    covariate: Any = None,
    combat_mean_only: bool = False,
    combat_empirical_bayes: bool = True,
    min_samples: int = 3,
    missing_control: str = "error",
    tvn_method: str = "ZCA-cor",
    tvn_epsilon: float = 1e-6,
    tvn_scope: str = "global",
    tvn_center_batches: bool = True,
    tvn_n_components: Any = "auto",
    harmony_n_components: int = 50,
    harmony_theta: float = 2.0,
    harmony_seed: int = 0,
) -> Tuple[pd.DataFrame, BatchCorrectionReport]:
    """Normalize numeric features within acquisition batches.

    ``center`` removes per-batch mean shifts while preserving the global mean.
    ``zscore`` aligns per-batch means and variances to the global distribution.
    ``robust_zscore`` does the same with median/MAD and is less sensitive to
    heavy-tailed single-cell measurements. ``control_center`` estimates
    only a location shift from negative/reference controls in every batch,
    preserving treatment dispersion.

    ``combat`` is the empirical-Bayes method of Johnson, Li & Rabinovich
    (2007). Unlike the four above it fits a *model*: every feature is regressed
    on batch indicators **and** on the biology named by ``covariate``, and only
    the batch part of that fit is removed. The per-batch location and scale are
    then shrunk toward the distribution of the same parameter across all
    features, which is what makes it usable on a plate with few wells where a
    per-feature estimate would be noise.

    That covariate is not optional and has no default. A batch effect estimated
    without it absorbs any contrast that happens to differ between plates --
    which, in a screen where treatments are laid out plate by plate, is the
    treatment effect. The correction then reports a cleaner batch diagnostic
    and removes the treatment effect. Pass the biology to keep, or pass :data:`NO_COVARIATE`
    to record that there is none.

    ``tvn`` (alpha) is typical variation normalisation (Ando et al. 2017).
    With ``tvn_center_batches`` each batch is first centred on the median of
    its reference controls; :func:`spherize_features` then whitens every row
    with a transform fitted on the controls, once on all of them
    (``tvn_scope="global"``) or per batch (``"batch"``). Like
    ``control_center`` it needs ``control`` and ``control_values``.

    ``harmony`` (alpha) integrates a PCA embedding with Harmony (Korsunsky
    et al. 2019) through the optional ``harmonypy`` package; see
    :func:`harmony_embedding`. It returns the corrected **embedding**
    (columns ``harmony_1 ...``), not the input features, so a caller that
    needs named features must use another method.

    :param features: numeric feature DataFrame; metadata must not be included.
    :param batch: batch/plate label aligned to ``features.index``.
    :param method: one of :data:`METHODS`.
    :param batch_column: human-readable source column for diagnostics.
    :param control: optional aligned series used by ``control_center``.
    :param control_values: scalar/list values selecting reference controls.
    :param covariate: required by ``combat`` and ignored by every other
        method -- a Series or DataFrame of biology to preserve, aligned to
        ``features.index``, or :data:`NO_COVARIATE` to declare there is none.
    :param combat_mean_only: correct only the additive batch shift and leave
        each batch's dispersion alone. The right choice when plates differ in
        offset but the assay's noise model is stable, and when rescaling a
        variance would manufacture significance.
    :param combat_empirical_bayes: ``False`` uses raw per-feature batch
        estimates with no shrinkage. Only sensible with many rows per batch;
        it exists so a test can show the shrinkage is doing something.
    :param min_samples: minimum rows (or controls) required per batch.
    :param missing_control: ``"error"`` or ``"skip"`` for a batch lacking
        enough reference-control rows.
    :param tvn_method: ``tvn`` whitening, one of :data:`TVN_WHITENINGS`; the
        default ``"ZCA-cor"`` is pycytominer ``normalize``'s.
    :param tvn_epsilon: ``tvn`` eigenvalue regularisation.
    :param tvn_scope: ``"global"`` or ``"batch"``, see :data:`TVN_SCOPES`.
    :param tvn_center_batches: centre each batch on its controls before
        whitening.
    :param tvn_n_components: ``"auto"`` (the directions the controls span),
        ``None`` (all of them, as pycytominer) or an integer.
    :param harmony_n_components: principal components Harmony integrates.
    :param harmony_theta: Harmony diversity penalty.
    :param harmony_seed: Harmony random state.
    :returns: ``(corrected_features, report)``.
    :raises ValueError: for unknown methods, misaligned metadata, non-numeric
        features, missing batches, insufficient required controls, a missing
        ``combat`` covariate, or a covariate confounded with batch.
    """
    normalized_method = str(method or "none").strip().lower()
    aliases = {
        "off": "none",
        "false": "none",
        "mean_center": "center",
        "plate_zscore": "zscore",
        "robust": "robust_zscore",
        "negative_control": "control_center",
        "empirical_bayes": "combat",
        "sphere": "tvn",
        "spherize": "tvn",
        "typical_variation": "tvn",
        "harmonypy": "harmony",
    }
    normalized_method = aliases.get(normalized_method, normalized_method)
    if normalized_method not in METHODS:
        raise ValueError(
            f"Unknown batch_correction={method!r}. Choose one of {METHODS}."
        )
    normalized_missing_control = str(missing_control).strip().lower()
    if normalized_missing_control not in {"error", "skip"}:
        raise ValueError(
            "batch_missing_control must be 'error' or 'skip', not "
            f"{missing_control!r}."
        )
    combat_covariate: Optional[pd.DataFrame] = None
    if normalized_method == "combat":
        if _is_no_covariate(covariate):
            pass
        elif covariate is None:
            raise ValueError(
                "batch_correction='combat' needs to know which biology to "
                "keep. ComBat estimates the plate effect from whatever the "
                "design does not explain, so a contrast that is not in the "
                "design is removed as if it were noise -- and the run looks "
                "cleaner for it. Pass covariate= the condition/treatment "
                "column, or covariate=spacr.batch_correction.NO_COVARIATE to "
                "state that there is nothing to preserve."
            )
        else:
            combat_covariate = _covariate_frame(covariate)
    if not isinstance(features, pd.DataFrame) or features.empty:
        raise ValueError("features must be a non-empty pandas DataFrame.")
    if not features.index.equals(batch.index):
        batch = batch.reindex(features.index)
    if batch.isna().any():
        missing = int(batch.isna().sum())
        raise ValueError(
            f"{batch_column} is missing for {missing} feature row(s); batch "
            "correction cannot guess which plate produced them."
        )
    numeric = features.apply(pd.to_numeric, errors="coerce").astype(float)
    newly_invalid = numeric.isna() & features.notna()
    if newly_invalid.any().any():
        columns = newly_invalid.any(axis=0)
        raise ValueError(
            "Batch correction received non-numeric values in: "
            f"{list(columns[columns].index)}."
        )
    if combat_covariate is not None and not (
        features.index.equals(combat_covariate.index)
    ):
        combat_covariate = combat_covariate.reindex(features.index)
    labels = batch.astype(str)
    batches = sorted(labels.unique().tolist())
    report = BatchCorrectionReport(
        method=normalized_method,
        batch_column=str(batch_column),
        batches=batches,
        features=[str(column) for column in numeric.columns],
        rows=len(numeric),
        centroid_spread_before=_centroid_spread(numeric, labels),
    )
    covariate_key = _covariate_key(combat_covariate)
    if covariate_key is not None:
        report.covariate_columns = [
            str(name) for name in combat_covariate.columns
        ]
        report.covariate_spread_before = _centroid_spread(numeric, covariate_key)
    if normalized_method == "none" or len(batches) < 2:
        if normalized_method != "none":
            report.warnings.append(
                f"Only {len(batches)} batch was present; correction was a no-op."
            )
        report.centroid_spread_after = report.centroid_spread_before
        report.covariate_spread_after = report.covariate_spread_before
        return numeric.copy(), report

    min_samples = max(1, int(min_samples))
    counts = labels.value_counts()
    too_small = counts[counts < min_samples]
    if not too_small.empty:
        raise ValueError(
            f"{batch_column} batch(es) have fewer than min_samples="
            f"{min_samples}: {too_small.to_dict()}."
        )

    corrected = numeric.copy()
    robust = normalized_method == "robust_zscore"
    if normalized_method == "harmony":
        corrected, info = harmony_embedding(
            numeric, labels, n_components=harmony_n_components,
            theta=harmony_theta, seed=harmony_seed,
        )
        report.features = [str(column) for column in corrected.columns]
        report.warnings.append(
            f"harmony returned a {info['n_components']}-component corrected "
            "embedding (PCA of the standardised features, "
            f"{info['explained_variance_ratio']:.1%} of their variance), not "
            "the input features; the centroid spread after is measured in "
            "that embedding."
        )
    elif normalized_method == "tvn":
        corrected = _tvn(
            numeric, labels, batches, control, control_values, report,
            method=tvn_method, epsilon=tvn_epsilon, scope=tvn_scope,
            center_batches=bool(tvn_center_batches),
            n_components=tvn_n_components, min_samples=min_samples,
        )
    elif normalized_method == "combat":
        corrected = _combat(
            numeric,
            labels,
            batches,
            combat_covariate,
            report,
            mean_only=bool(combat_mean_only),
            empirical_bayes=bool(combat_empirical_bayes),
        )
    elif normalized_method == "control_center":
        controls = _as_controls(control_values)
        if control is None or not controls:
            raise ValueError(
                "control_center requires batch_control_column and at least "
                "one batch_control_value (normally the negative control)."
            )
        if not control.index.equals(features.index):
            control = control.reindex(features.index)
        control_mask = _match_values(control, controls)
        report.controls = int(control_mask.sum())
        pooled = numeric.loc[control_mask]
        if len(pooled) < min_samples:
            try:
                present = sorted({str(v) for v in control.dropna().unique()})
            except Exception:                                    # noqa: BLE001
                present = []
            seen = (f" The column holds {present[:12]}"
                    + (" and more." if len(present) > 12 else ".")
                    if present else "")
            raise ValueError(
                f"Only {len(pooled)} total reference-control row(s) matched "
                f"{controls!r}; need at least {min_samples}.{seen}"
            )
        pooled_center = pooled.median(axis=0)
        missing_batches = []
        for label in batches:
            rows = labels == label
            reference = numeric.loc[rows & control_mask]
            if len(reference) < min_samples:
                missing_batches.append(label)
                continue
            shift = reference.median(axis=0) - pooled_center
            corrected.loc[rows] = numeric.loc[rows].subtract(shift, axis=1)
        if missing_batches:
            message = (
                f"No usable reference controls in {batch_column} batch(es) "
                f"{missing_batches}; need {min_samples} per batch."
            )
            if normalized_missing_control == "skip":
                report.warnings.append(message + " Those batches were unchanged.")
            else:
                raise ValueError(message)
    else:
        global_center = _center(numeric, robust)
        global_scale = _scale(numeric, robust).fillna(1.0)
        for label in batches:
            rows = labels == label
            values = numeric.loc[rows]
            local_center = _center(values, robust)
            if normalized_method == "center":
                corrected.loc[rows] = values.subtract(
                    local_center, axis=1,
                ).add(global_center, axis=1)
                continue
            local_scale = _scale(values, robust)
            zero_scale = local_scale.isna()
            if zero_scale.any():
                report.warnings.append(
                    f"{label}: {int(zero_scale.sum())} constant feature(s) "
                    "used global scale."
                )
                local_scale = local_scale.fillna(global_scale)
            corrected.loc[rows] = (
                values.subtract(local_center, axis=1)
                .divide(local_scale, axis=1)
                .multiply(global_scale, axis=1)
                .add(global_center, axis=1)
            )

    report.centroid_spread_after = _centroid_spread(corrected, labels)
    if covariate_key is not None:
        report.covariate_spread_after = _centroid_spread(corrected, covariate_key)
    return corrected, report


def correct_from_metadata(
    features: pd.DataFrame,
    metadata: pd.DataFrame,
    *,
    batch_correction: str = "none",
    batch_column: str = "plateID",
    batch_control_column: Optional[str] = None,
    batch_control_values: Any = None,
    batch_covariate_column: Any = None,
    batch_combat_mean_only: bool = False,
    batch_min_samples: int = 3,
    batch_missing_control: str = "error",
) -> Tuple[pd.DataFrame, BatchCorrectionReport]:
    """Correct a feature frame using named columns from an aligned metadata frame.

    This adapter is the shared boundary used by UMAP, ML, and regression. It
    validates column names once and keeps metadata out of the numeric feature
    matrix.

    :param features: numeric feature DataFrame.
    :param metadata: DataFrame containing batch and optional control columns.
    :param batch_correction: correction method accepted by
        :func:`correct_batch_effects`.
    :param batch_column: metadata column identifying batches.
    :param batch_control_column: optional reference-control metadata column.
    :param batch_control_values: values selecting reference controls.
    :param batch_covariate_column: metadata column(s) naming the biology that
        ``combat`` must preserve -- one name, a comma-separated string, or a
        list. The literal ``"none"`` declares that there is none. Required by
        ``combat`` and ignored by every other method; leaving it blank makes
        ``combat`` refuse to run rather than quietly delete the contrast.
    :param batch_combat_mean_only: correct only the additive shift.
    :param batch_min_samples: minimum samples/reference controls per batch.
    :param batch_missing_control: ``"error"`` or ``"skip"``.
    :returns: corrected features and diagnostics report.
    :raises ValueError: when metadata cannot be aligned or required columns
        are absent.
    """
    if not isinstance(metadata, pd.DataFrame):
        raise ValueError("metadata must be a pandas DataFrame.")
    if not features.index.equals(metadata.index):
        try:
            metadata = metadata.reindex(features.index)
        except ValueError as exc:
            raise ValueError(
                "Batch-correction metadata cannot be aligned to feature rows; "
                "ensure their indexes are identical and unique."
            ) from exc
    if batch_column not in metadata.columns:
        raise ValueError(
            f"batch_correction={batch_correction!r} requires "
            f"batch_column={batch_column!r}, but that column is absent."
        )
    control = None
    if batch_control_column:
        if batch_control_column not in metadata.columns:
            raise ValueError(
                f"batch_control_column={batch_control_column!r} is absent "
                "from the input metadata."
            )
        control = metadata[batch_control_column]
    covariate = _resolve_covariate(metadata, batch_covariate_column)
    return correct_batch_effects(
        features,
        metadata[batch_column],
        method=batch_correction,
        batch_column=batch_column,
        control=control,
        control_values=batch_control_values,
        covariate=covariate,
        combat_mean_only=batch_combat_mean_only,
        min_samples=batch_min_samples,
        missing_control=batch_missing_control,
    )


def _resolve_covariate(metadata: pd.DataFrame, spec: Any) -> Any:
    """Turn a covariate-column setting into what ``covariate=`` expects.

    ``None`` and ``""`` pass straight through as "unanswered" so ComBat raises
    its own explanatory error rather than one about a missing column; every
    other method ignores the value entirely.
    """
    if spec is None:
        return None
    if isinstance(spec, str):
        text = spec.strip()
        if not text:
            return None
        if text.lower() in {NO_COVARIATE, "none"}:
            return NO_COVARIATE
        names = [part.strip() for part in text.split(",") if part.strip()]
    elif isinstance(spec, (pd.Series, pd.DataFrame)):
        return spec
    else:
        names = [str(part).strip() for part in spec if str(part).strip()]
    if not names:
        return None
    missing = [name for name in names if name not in metadata.columns]
    if missing:
        raise ValueError(
            f"batch_covariate_column={missing!r} is absent from the input "
            "metadata, so combat cannot protect it. Available columns: "
            f"{sorted(map(str, metadata.columns))[:20]}."
        )
    return metadata.loc[:, names]


def correction_kwargs(
    settings: Mapping[str, Any],
    *,
    default_control_column: Optional[str] = None,
    default_control_values: Any = None,
) -> Dict[str, Any]:
    """Translate shared GUI settings into correction-call keyword arguments.

    Emits exactly six keys -- ``batch_correction``, ``batch_column``,
    ``batch_control_column``, ``batch_control_values``, ``batch_min_samples``
    and ``batch_missing_control`` -- with the same defaults as
    :func:`correct_from_metadata`. The combat-only keys
    ``batch_covariate_column`` and ``batch_combat_mean_only`` are deliberately
    left out so the result stays safe to splat into signatures that do not
    accept them; a caller using ``batch_correction="combat"`` must pass
    ``batch_covariate_column`` alongside this mapping or
    :func:`correct_from_metadata` raises. ``batch_combat_mean_only`` stays
    optional and defaults to ``False``.

    :param settings: settings mapping the batch keys are read from.
    :param default_control_column: control column used when
        ``batch_control_column`` is absent or an empty string; a
        whitespace-only name is kept verbatim.
    :param default_control_values: control values used when
        ``batch_control_values`` is absent or blank.
    :returns: keyword arguments for :func:`correct_from_metadata`.
    """
    control_column = settings.get("batch_control_column")
    if control_column in (None, ""):
        control_column = default_control_column
    control_values = settings.get("batch_control_values")
    if control_values is None or (
        isinstance(control_values, str) and not control_values.strip()
    ):
        control_values = default_control_values
    return {
        "batch_correction": settings.get("batch_correction", "none"),
        "batch_column": settings.get("batch_column", "plateID"),
        "batch_control_column": control_column,
        "batch_control_values": control_values,
        "batch_min_samples": settings.get("batch_min_samples", 3),
        "batch_missing_control": settings.get(
            "batch_missing_control", "error",
        ),
    }


def write_report(report: BatchCorrectionReport, path: Any) -> Path:
    """Write a correction report as stable JSON and return its path.

    :param report: completed batch-correction report to serialize.
    :param path: destination JSON path to replace atomically.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(destination)
    return destination
