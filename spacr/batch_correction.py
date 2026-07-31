"""Plate/batch-effect correction for tabular microscopy measurements.

The implementation is dependency-light (pandas/numpy), preserves the original
row/index order, and never changes metadata columns. It is shared by Image
UMAP, Classify (ML), UMAP hyperparameter search, and regression.
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
)
"""Supported correction methods."""


@dataclass
class BatchCorrectionReport:
    """Diagnostics for one correction operation.

    :ivar method: correction method actually applied.
    :ivar batch_column: metadata column represented by ``batch``.
    :ivar batches: normalized batch labels seen.
    :ivar features: corrected feature names.
    :ivar rows: input row count.
    :ivar controls: number of rows used as reference controls.
    :ivar centroid_spread_before: mean across-feature standard deviation of
        batch centers before correction.
    :ivar centroid_spread_after: same diagnostic after correction.
    :ivar warnings: explicit fallbacks or limitations.
    """

    method: str
    batch_column: str
    batches: List[str]
    features: List[str]
    rows: int
    controls: int = 0
    centroid_spread_before: Optional[float] = None
    centroid_spread_after: Optional[float] = None
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable report."""
        return asdict(self)


def _as_controls(values: Any) -> List[Any]:
    """Normalize a scalar or iterable control specification."""
    if values is None:
        return []
    if isinstance(values, (str, bytes)):
        return [values]
    if isinstance(values, Iterable):
        return list(values)
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


def correct_batch_effects(
    features: pd.DataFrame,
    batch: pd.Series,
    *,
    method: str = "none",
    batch_column: str = "plateID",
    control: Optional[pd.Series] = None,
    control_values: Any = None,
    min_samples: int = 3,
    missing_control: str = "error",
) -> Tuple[pd.DataFrame, BatchCorrectionReport]:
    """Normalize numeric features within acquisition batches.

    ``center`` removes per-batch mean shifts while preserving the global mean.
    ``zscore`` aligns per-batch means and variances to the global distribution.
    ``robust_zscore`` does the same with median/MAD and is the safest default
    for heavy-tailed single-cell measurements. ``control_center`` estimates
    only a location shift from negative/reference controls in every batch,
    preserving treatment dispersion and usually best preserving biology.

    :param features: numeric feature DataFrame; metadata must not be included.
    :param batch: batch/plate label aligned to ``features.index``.
    :param method: one of :data:`METHODS`.
    :param batch_column: human-readable source column for diagnostics.
    :param control: optional aligned series used by ``control_center``.
    :param control_values: scalar/list values selecting reference controls.
    :param min_samples: minimum rows (or controls) required per batch.
    :param missing_control: ``"error"`` or ``"skip"`` for a batch lacking
        enough reference-control rows.
    :returns: ``(corrected_features, report)``.
    :raises ValueError: for unknown methods, misaligned metadata, non-numeric
        features, missing batches, or insufficient required controls.
    """
    normalized_method = str(method or "none").strip().lower()
    aliases = {
        "off": "none",
        "false": "none",
        "mean_center": "center",
        "plate_zscore": "zscore",
        "robust": "robust_zscore",
        "negative_control": "control_center",
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
    if normalized_method == "none" or len(batches) < 2:
        if normalized_method != "none":
            report.warnings.append(
                f"Only {len(batches)} batch was present; correction was a no-op."
            )
        report.centroid_spread_after = report.centroid_spread_before
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
    if normalized_method == "control_center":
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
            raise ValueError(
                f"Only {len(pooled)} total reference-control row(s) matched "
                f"{controls!r}; need at least {min_samples}."
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
    return corrected, report


def correct_from_metadata(
    features: pd.DataFrame,
    metadata: pd.DataFrame,
    *,
    batch_correction: str = "none",
    batch_column: str = "plateID",
    batch_control_column: Optional[str] = None,
    batch_control_values: Any = None,
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
    return correct_batch_effects(
        features,
        metadata[batch_column],
        method=batch_correction,
        batch_column=batch_column,
        control=control,
        control_values=batch_control_values,
        min_samples=batch_min_samples,
        missing_control=batch_missing_control,
    )


def correction_kwargs(
    settings: Mapping[str, Any],
    *,
    default_control_column: Optional[str] = None,
    default_control_values: Any = None,
) -> Dict[str, Any]:
    """Translate shared GUI settings into correction-call keyword arguments."""
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
    """Write a correction report as stable JSON and return its path."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(destination)
    return destination
