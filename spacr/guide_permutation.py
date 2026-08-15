"""Plate-aware marginal guide association tests for pooled screens.

The screen unit is the well.  Each guide is tested separately after both its
well-level read fraction and the well-level phenotype have been residualized
against nuisance covariates.  Empirical two-sided P values are obtained with
Freedman--Lane residual permutations restricted within the experimental block
(normally ``plateID``), followed by a user-selected multiple-testing
correction within each outcome and minimum-support family.

This module deliberately reports *marginal associations*.  It does not claim
to estimate a simultaneous conditional coefficient for every guide when the
number or correlation structure of guides makes that model unidentified.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests


_FDR_ALIASES = {
    "bh": "fdr_bh",
    "benjamini-hochberg": "fdr_bh",
    "benjamini_hochberg": "fdr_bh",
    "by": "fdr_by",
    "benjamini-yekutieli": "fdr_by",
    "benjamini_yekutieli": "fdr_by",
    "bonferroni": "bonferroni",
    "holm": "holm",
    "none": "none",
    "raw": "none",
}


def _normalise_thresholds(min_wells) -> tuple[int, ...]:
    """Return sorted unique positive minimum-well thresholds."""
    if isinstance(min_wells, (int, np.integer)):
        values = [int(min_wells)]
    elif isinstance(min_wells, Iterable) and not isinstance(min_wells, str):
        values = [int(value) for value in min_wells]
    else:
        raise TypeError("min_wells must be a positive integer or iterable of integers")
    if not values or any(value < 1 for value in values):
        raise ValueError("min_wells must contain at least one positive integer")
    return tuple(sorted(set(values)))


def adjust_p_values(p_values, method="fdr_bh", alpha=0.05):
    """Return adjusted P values and rejection calls.

    ``method`` accepts the statsmodels names ``fdr_bh``, ``fdr_by``,
    ``bonferroni`` and ``holm``, plus their common aliases and ``none``.
    Missing P values remain missing and do not enter the correction family.
    """
    values = np.asarray(p_values, dtype=float)
    if not 0 < float(alpha) < 1:
        raise ValueError("alpha must be strictly between 0 and 1")
    key = str(method).strip().lower()
    canonical = _FDR_ALIASES.get(key, key)
    supported = {"fdr_bh", "fdr_by", "bonferroni", "holm", "none"}
    if canonical not in supported:
        raise ValueError(
            f"Unsupported multiple-testing method {method!r}; choose one of "
            f"{sorted(supported)}."
        )

    adjusted = np.full(values.shape, np.nan, dtype=float)
    rejected = np.zeros(values.shape, dtype=bool)
    finite = np.isfinite(values)
    if not finite.any():
        return adjusted, rejected
    if canonical == "none":
        adjusted[finite] = values[finite]
        rejected[finite] = values[finite] < alpha
    else:
        call, corrected, _, _ = multipletests(
            values[finite], alpha=alpha, method=canonical
        )
        adjusted[finite] = corrected
        rejected[finite] = call
    return adjusted, rejected


def prepare_long_guide_data(
    data: pd.DataFrame,
    outcome_columns: str | Sequence[str],
    *,
    well_column: str = "prc",
    guide_column: str = "grna",
    fraction_column: str = "fraction",
    block_column: str = "plateID",
    nuisance_columns: Sequence[str] | None = None,
):
    """Convert spaCR's long regression table into aligned well-level tables.

    The long table must have one fraction per well/guide pair.  Phenotype,
    block and nuisance values may repeat across the guide rows of a well, but
    they must be identical within that well.  Duplicate well/guide rows are
    summed, matching the existing spaCR design-matrix construction.

    Returns ``(fractions, outcomes, guide_metadata)`` where ``fractions`` is a
    zero-filled well-by-guide matrix, ``outcomes`` has one row per well, and
    ``guide_metadata`` reports the number of wells with a positive fraction.
    """
    if isinstance(outcome_columns, str):
        outcome_columns = [outcome_columns]
    else:
        outcome_columns = list(outcome_columns)
    nuisance_columns = list(nuisance_columns or [])
    required = {
        well_column,
        guide_column,
        fraction_column,
        block_column,
        *outcome_columns,
        *nuisance_columns,
    }
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"Long guide table is missing required columns: {missing}")

    frame = data.loc[:, list(required)].copy()
    if frame[[well_column, guide_column]].isna().any().any():
        raise ValueError("well and guide identifiers must not contain missing values")
    frame[fraction_column] = pd.to_numeric(frame[fraction_column], errors="raise")
    if not np.isfinite(frame[fraction_column]).all():
        raise ValueError("guide fractions must be finite")
    if (frame[fraction_column] < 0).any():
        raise ValueError("guide fractions must be non-negative")

    repeated_columns = [block_column, *outcome_columns, *nuisance_columns]
    within_well_unique = frame.groupby(well_column, sort=False)[
        repeated_columns
    ].nunique(dropna=False)
    inconsistent = within_well_unique.gt(1).any(axis=1)
    if inconsistent.any():
        example = inconsistent.index[inconsistent][0]
        raise ValueError(
            f"Phenotype/block/nuisance values are not constant within well {example!r}."
        )

    outcomes = (
        frame.drop_duplicates(well_column)
        .set_index(well_column)[repeated_columns]
        .sort_index()
    )
    for column in outcome_columns:
        outcomes[column] = pd.to_numeric(outcomes[column], errors="raise")
        if not np.isfinite(outcomes[column]).all():
            raise ValueError(f"Outcome {column!r} must be finite for every analyzed well")

    fractions = frame.pivot_table(
        index=well_column,
        columns=guide_column,
        values=fraction_column,
        aggfunc="sum",
        fill_value=0.0,
    ).sort_index()
    fractions.columns = fractions.columns.astype(str)
    fractions = fractions.reindex(outcomes.index, fill_value=0.0)
    support = (fractions > 0).sum(axis=0).astype(int)
    guide_metadata = pd.DataFrame(
        {"guide": fractions.columns, "wells_with_guide": support.to_numpy()}
    ).set_index("guide")
    return fractions, outcomes, guide_metadata


def _nuisance_design(
    outcomes: pd.DataFrame,
    block_column: str,
    nuisance_columns: Sequence[str],
) -> np.ndarray:
    """Build an intercept plus full-rank block/nuisance fixed-effect design."""
    if block_column not in outcomes:
        raise ValueError(f"Block column {block_column!r} is absent from outcomes")
    pieces = [np.ones((len(outcomes), 1), dtype=float)]
    block = pd.get_dummies(
        outcomes[block_column].astype(str), drop_first=True, dtype=float
    )
    if block.shape[1]:
        pieces.append(block.to_numpy(dtype=float))
    for column in nuisance_columns:
        if column == block_column:
            continue
        values = outcomes[column]
        if pd.api.types.is_numeric_dtype(values):
            numeric = pd.to_numeric(values, errors="raise").to_numpy(dtype=float)
            if not np.isfinite(numeric).all():
                raise ValueError(f"Nuisance column {column!r} must be finite")
            pieces.append(numeric[:, None])
        else:
            dummy = pd.get_dummies(
                values.astype(str), prefix=column, drop_first=True, dtype=float
            )
            if dummy.shape[1]:
                pieces.append(dummy.to_numpy(dtype=float))
    design = np.column_stack(pieces)
    if np.linalg.matrix_rank(design) != design.shape[1]:
        raise ValueError("The nuisance fixed-effect design is rank deficient")
    return design


def _residualize(values: np.ndarray, q_basis: np.ndarray) -> np.ndarray:
    return values - q_basis @ (q_basis.T @ values)


def guide_freedman_lane_test(
    fractions: pd.DataFrame,
    outcomes: pd.DataFrame,
    outcome_column: str,
    *,
    min_wells: int | Sequence[int] = 4,
    block_column: str = "plateID",
    nuisance_columns: Sequence[str] | None = None,
    n_permutations: int = 200_000,
    random_state: int | np.random.Generator = 0,
    multiple_testing: str = "fdr_bh",
    alpha: float = 0.05,
    presence_threshold: float = 0.0,
    batch_size: int = 500,
    guide_metadata: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Test marginal guide associations across one or more support families.

    Permutation P values are computed once for every guide satisfying the
    smallest requested support threshold.  For each displayed threshold, the
    eligible subset is then corrected as its own multiple-testing family.
    This keeps the observed statistic and empirical P value for a guide
    identical across thresholds; only the adjusted value changes with family
    size.
    """
    thresholds = _normalise_thresholds(min_wells)
    if int(n_permutations) < 1:
        raise ValueError("n_permutations must be at least 1")
    if int(batch_size) < 1:
        raise ValueError("batch_size must be at least 1")
    if float(presence_threshold) < 0:
        raise ValueError("presence_threshold must be non-negative")
    if outcome_column not in outcomes:
        raise ValueError(f"Outcome column {outcome_column!r} is absent")
    if not fractions.index.equals(outcomes.index):
        try:
            outcomes = outcomes.loc[fractions.index]
        except KeyError as exc:
            raise ValueError("Fraction and outcome well identifiers do not align") from exc

    x_frame = fractions.apply(pd.to_numeric, errors="raise")
    x_values = x_frame.to_numpy(dtype=float)
    if not np.isfinite(x_values).all() or (x_values < 0).any():
        raise ValueError("Guide fractions must be finite and non-negative")
    supports = (x_values > float(presence_threshold)).sum(axis=0).astype(int)
    eligible = supports >= thresholds[0]
    if not eligible.any():
        raise ValueError(
            f"No guide occurs in at least {thresholds[0]} well(s) above "
            f"presence_threshold={presence_threshold}."
        )
    guide_names = x_frame.columns[eligible].astype(str)
    x = x_values[:, eligible]
    y = pd.to_numeric(outcomes[outcome_column], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(y).all():
        raise ValueError(f"Outcome {outcome_column!r} must be finite")

    nuisance_columns = list(nuisance_columns or [])
    design = _nuisance_design(outcomes, block_column, nuisance_columns)
    q_basis, _ = np.linalg.qr(design, mode="reduced")
    x_residual = _residualize(x, q_basis)
    y_fitted = q_basis @ (q_basis.T @ y)
    y_residual = y - y_fitted
    x_norm = np.sqrt(np.sum(x_residual**2, axis=0))
    y_norm = float(np.sqrt(np.sum(y_residual**2)))
    if y_norm <= np.finfo(float).eps:
        raise ValueError("Outcome has no residual variation after nuisance adjustment")
    testable = x_norm > np.finfo(float).eps
    if not testable.all():
        names = guide_names[~testable].tolist()
        raise ValueError(
            "Guide fractions have no residual variation after nuisance adjustment: "
            f"{names[:10]}{' ...' if len(names) > 10 else ''}"
        )
    x_unit = x_residual / x_norm
    observed = (x_unit.T @ y_residual) / y_norm

    blocks = outcomes[block_column].astype(str).to_numpy()
    block_indexes = [
        np.flatnonzero(blocks == level) for level in pd.unique(blocks)
    ]
    rng = (
        random_state
        if isinstance(random_state, np.random.Generator)
        else np.random.default_rng(random_state)
    )
    exceedances = np.zeros(len(guide_names), dtype=np.int64)
    epsilon = np.finfo(float).eps
    n_permutations = int(n_permutations)
    batch_size = int(batch_size)
    for start in range(0, n_permutations, batch_size):
        current = min(batch_size, n_permutations - start)
        permuted_residuals = np.empty((len(y), current), dtype=float)
        for permutation_index in range(current):
            permuted = y_residual.copy()
            for indexes in block_indexes:
                permuted[indexes] = permuted[indexes][
                    rng.permutation(len(indexes))
                ]
            permuted_residuals[:, permutation_index] = permuted
        # Freedman--Lane: y* = nuisance fit + permuted reduced-model residual;
        # then remove nuisance effects before evaluating the same statistic.
        permuted_outcomes = y_fitted[:, None] + permuted_residuals
        permuted_outcomes = _residualize(permuted_outcomes, q_basis)
        permuted_norm = np.sqrt(np.sum(permuted_outcomes**2, axis=0))
        permutation_effects = (
            x_unit.T @ permuted_outcomes
        ) / permuted_norm[None, :]
        exceedances += np.sum(
            np.abs(permutation_effects)
            >= np.abs(observed)[:, None] - epsilon,
            axis=1,
        )

    p_values = (exceedances + 1) / (n_permutations + 1)
    base = pd.DataFrame(
        {
            "outcome": outcome_column,
            "guide": guide_names,
            "wells_with_guide": supports[eligible],
            "standardized_marginal_effect": observed,
            "permutation_exceedances": exceedances,
            "permutations": n_permutations,
            "permutation_p_value": p_values,
            "block_column": block_column,
            "nuisance_columns": ";".join(nuisance_columns),
            "presence_threshold": float(presence_threshold),
        }
    )
    if guide_metadata is not None:
        metadata = guide_metadata.copy()
        if "guide" in metadata.columns:
            metadata = metadata.set_index("guide")
        metadata.index = metadata.index.astype(str)
        extra = metadata.drop(columns=["wells_with_guide"], errors="ignore")
        base = base.join(extra, on="guide", validate="many_to_one")

    frames = []
    for threshold in thresholds:
        family = base.loc[base["wells_with_guide"] >= threshold].copy()
        corrected, rejected = adjust_p_values(
            family["permutation_p_value"].to_numpy(),
            method=multiple_testing,
            alpha=alpha,
        )
        family["minimum_wells_threshold"] = threshold
        family["tested_guides_in_family"] = len(family)
        family["multiple_testing_method"] = _FDR_ALIASES.get(
            str(multiple_testing).strip().lower(),
            str(multiple_testing).strip().lower(),
        )
        family["adjusted_p_value"] = corrected
        family["significant"] = rejected
        family["alpha"] = float(alpha)
        frames.append(family)
    result = pd.concat(frames, ignore_index=True)
    return result.sort_values(
        ["minimum_wells_threshold", "permutation_p_value", "guide"],
        kind="stable",
    ).reset_index(drop=True)


def guide_support_sensitivity(
    fractions: pd.DataFrame,
    outcomes: pd.DataFrame,
    outcome_columns: str | Sequence[str],
    *,
    min_wells: int | Sequence[int] = (1, 2, 3, 4),
    random_state: int = 0,
    **kwargs,
) -> pd.DataFrame:
    """Run the same support-threshold analysis for one or more outcomes."""
    columns = [outcome_columns] if isinstance(outcome_columns, str) else list(outcome_columns)
    frames = []
    for index, outcome in enumerate(columns):
        frames.append(
            guide_freedman_lane_test(
                fractions,
                outcomes,
                outcome,
                min_wells=min_wells,
                random_state=int(random_state) + index,
                **kwargs,
            )
        )
    return pd.concat(frames, ignore_index=True)


def analyse_long_guide_table(
    data: pd.DataFrame,
    outcome_columns: str | Sequence[str],
    *,
    min_wells: int | Sequence[int] = (1, 2, 3, 4),
    well_column: str = "prc",
    guide_column: str = "grna",
    fraction_column: str = "fraction",
    block_column: str = "plateID",
    nuisance_columns: Sequence[str] | None = None,
    **kwargs,
) -> pd.DataFrame:
    """Convenience wrapper for spaCR's saved ``regression_data.csv``."""
    fractions, outcomes, guide_metadata = prepare_long_guide_data(
        data,
        outcome_columns,
        well_column=well_column,
        guide_column=guide_column,
        fraction_column=fraction_column,
        block_column=block_column,
        nuisance_columns=nuisance_columns,
    )
    return guide_support_sensitivity(
        fractions,
        outcomes,
        outcome_columns,
        min_wells=min_wells,
        block_column=block_column,
        nuisance_columns=nuisance_columns,
        guide_metadata=guide_metadata,
        **kwargs,
    )


def save_guide_permutation_results(
    results: pd.DataFrame,
    destination: str | Path,
    *,
    prefix: str = "guide_permutation",
) -> Mapping[str, Path]:
    """Save the long result and one source-data CSV per support threshold."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    long_path = destination / f"{prefix}_results_long.csv"
    results.to_csv(long_path, index=False)
    paths["long"] = long_path
    for threshold in sorted(results["minimum_wells_threshold"].unique()):
        path = destination / f"{prefix}_min_{int(threshold)}_wells.csv"
        results.loc[
            results["minimum_wells_threshold"] == threshold
        ].to_csv(path, index=False)
        paths[f"min_{int(threshold)}"] = path
    return paths


def plot_guide_permutation_volcano(
    results: pd.DataFrame,
    *,
    outcome: str,
    minimum_wells: int,
    save_path: str | Path,
    label_guides: Mapping[str, str] | None = None,
    title: str | None = None,
):
    """Draw a volcano using standardized effect and adjusted P value."""
    import matplotlib.pyplot as plt

    data = results.loc[
        (results["outcome"] == outcome)
        & (results["minimum_wells_threshold"] == int(minimum_wells))
    ].copy()
    if data.empty:
        raise ValueError(
            f"No rows for outcome={outcome!r}, minimum_wells={minimum_wells}."
        )
    data["minus_log10_adjusted_p"] = -np.log10(
        np.clip(data["adjusted_p_value"], np.finfo(float).tiny, 1.0)
    )
    significant = data["significant"].astype(bool)
    method = str(data["multiple_testing_method"].iloc[0]).lower()
    adjusted_label = "BH q" if method == "fdr_bh" else "adjusted P"
    fig, axis = plt.subplots(figsize=(6.2, 4.8))
    axis.scatter(
        data.loc[~significant, "standardized_marginal_effect"],
        data.loc[~significant, "minus_log10_adjusted_p"],
        s=24,
        color="#B8BDC5",
        edgecolor="white",
        linewidth=0.35,
        label=f"{adjusted_label} >= {float(data['alpha'].iloc[0]):g}",
    )
    axis.scatter(
        data.loc[significant, "standardized_marginal_effect"],
        data.loc[significant, "minus_log10_adjusted_p"],
        s=48,
        color="#D55E00",
        edgecolor="#6D2B00",
        linewidth=0.6,
        label=f"{adjusted_label} < {float(data['alpha'].iloc[0]):g}",
    )
    axis.axhline(
        -np.log10(float(data["alpha"].iloc[0])),
        color="#404040",
        linestyle="--",
        linewidth=0.9,
    )
    axis.axvline(0, color="#777777", linewidth=0.7)
    labelled_rows = []
    for guide, label in (label_guides or {}).items():
        row = data.loc[data["guide"] == str(guide)]
        if row.empty:
            continue
        labelled_rows.append((str(label), row.iloc[0]))
    labelled_rows.sort(
        key=lambda item: item[1]["standardized_marginal_effect"]
    )
    for index, (label, row) in enumerate(labelled_rows):
        # Alternate sides and vertical positions so nearby discoveries do not
        # print on top of one another (as EAF1 g2 and GRA14 g3 otherwise do).
        if index % 2 == 0:
            offset, horizontal = (-5, 8), "right"
        else:
            offset, horizontal = (5, -15), "left"
        axis.annotate(
            label,
            (row["standardized_marginal_effect"], row["minus_log10_adjusted_p"]),
            xytext=offset,
            textcoords="offset points",
            fontsize=8,
            fontweight="bold",
            ha=horizontal,
        )
    axis.set_title(title or f"{outcome}: guides in >= {minimum_wells} wells")
    axis.set_xlabel("Block-adjusted standardized marginal effect")
    axis.set_ylabel(f"-log10({adjusted_label})")
    axis.margins(y=0.1)
    axis.grid(axis="y", color="#E6E6E6", linewidth=0.6)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False)
    fig.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=600 if save_path.suffix.lower() == ".png" else None)
    plt.close(fig)
    return save_path
