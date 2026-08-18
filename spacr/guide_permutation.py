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

from .multiple_testing import (
    METHODS as _MULTIPLE_TESTING_METHODS,
    adjust_p_values,
    canonical_method,
)


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


#: Re-exported so existing importers of this module keep working; the
#: inventory itself lives in :mod:`spacr.multiple_testing` so the GUI
#: dropdown, the CLI and this analysis cannot offer different method sets.
MULTIPLE_TESTING_METHODS = _MULTIPLE_TESTING_METHODS


def adjusted_value_label(method) -> str:
    """Axis/legend label for the adjusted value produced by ``method``.

    An FDR method yields a q value, a family-wise method an adjusted P, and
    ``none`` leaves the raw P value. Labelling every correction "BH q" -- as
    this module did while it offered only four methods -- mislabels the axis
    of every plot drawn with any other correction.
    """
    key = canonical_method(method)
    if key == "none":
        return "P"
    short = {
        "fdr_bh": "BH q",
        "fdr_by": "BY q",
        "fdr_tsbh": "TSBH q",
        "fdr_tsbky": "TSBKY q",
        "fdr_gbs": "GBS q",
        "storey": "Storey q",
    }
    if key in short:
        return short[key]
    # Every remaining method controls the family-wise error rate, which
    # adjusts the P value rather than producing a q value.
    return "adjusted P"


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
        family["multiple_testing_method"] = canonical_method(multiple_testing)
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


def _drawable_threshold(value) -> float | None:
    """``float(value)`` when it is a cut worth drawing, else ``None``.

    ``coefficient_threshold`` returns ``None`` for "no cut", and a run that
    has been through a CSV can hand back a NaN instead. Both mean the same
    thing to a plot, and so does zero -- a cut at zero excludes nothing.
    """
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number) or number <= 0:
        return None
    return number


def plot_guide_permutation_volcano(
    results: pd.DataFrame,
    *,
    outcome: str,
    minimum_wells: int,
    save_path: str | Path,
    label_guides: Mapping[str, str] | None = None,
    title: str | None = None,
    effect_threshold: float | None = None,
    effect_threshold_label: str | None = None,
):
    """Draw a volcano using standardized effect and adjusted P value.

    :param effect_threshold: half-width of the effect-size cut, drawn as a
        pair of vertical lines at ``-threshold`` and ``+threshold``. ``None``
        -- and any non-finite or non-positive number -- draws none, because a
        cut at zero excludes nothing and a line at zero is the axis that is
        already there.

        A permutation run drew no cut at all until 2026-08-17, on the reading
        that an empirical P value makes the question moot. It does not: a P
        value says an effect is distinguishable from zero and this line says
        it is big enough to follow up, and the second question is about the
        coefficient however the first was answered. Asked by the maintainer
        as "why cant i see the coefficient threshold if im running
        nonparametric regression?".
    :param effect_threshold_label: the sentence that attributes the cut, e.g.
        ``3x std of 30 controls = 0.84``, for the legend.
        :func:`spacr.thresholds.coefficient_threshold` returns it beside the
        number. Falls back to the number alone.
    """
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
    adjusted_label = adjusted_value_label(data["multiple_testing_method"].iloc[0])
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
    cut = _drawable_threshold(effect_threshold)
    if cut is not None:
        # Both lines, one legend entry: they are one cut with two sides, and
        # a legend that lists it twice reads as two different rules.
        axis.axvline(
            cut, color="#0072B2", linestyle=":", linewidth=1.1,
            label=effect_threshold_label or f"|effect| >= {cut:.3g}",
        )
        axis.axvline(-cut, color="#0072B2", linestyle=":", linewidth=1.1,
                     label="_nolegend_")
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


def gene_fraction_matrix(fractions: pd.DataFrame,
                         gene_of_guide: Mapping[str, str]) -> pd.DataFrame:
    """Sum each gene's guide columns into ONE column per gene.

    This is exactly the quantity the parametric gene fit regresses on:
    :func:`spacr.ml.check_and_clean_data` defines ``gene_fraction`` as the sum
    of the gene's gRNA fractions within a well, and this builds the same
    number from the well-by-guide matrix. The two paths therefore test the
    same regressor, one with a t statistic and one with a permutation null.

    :param fractions: well-by-guide fractions from
        :func:`prepare_long_guide_data`.
    :param gene_of_guide: guide id -> gene id.
    :returns: a well-by-gene frame, columns sorted, one column per gene that
        has at least one guide in ``fractions``.
    :raises ValueError: when a guide column has no gene.
    """
    columns = [str(name) for name in fractions.columns]
    mapping = {str(guide): str(gene) for guide, gene in gene_of_guide.items()}
    missing = sorted({name for name in columns if name not in mapping})
    if missing:
        raise ValueError(
            "every guide column needs a gene to be summed into, and these "
            f"have none: {missing[:10]}{' ...' if len(missing) > 10 else ''}")
    grouped = fractions.T.groupby(
        [mapping[name] for name in columns], sort=True).sum().T
    grouped.columns = grouped.columns.astype(str)
    grouped.index = fractions.index
    return grouped


def prepare_long_gene_data(
    data: pd.DataFrame,
    outcome_columns: str | Sequence[str],
    *,
    well_column: str = "prc",
    guide_column: str = "grna",
    gene_column: str = "gene",
    fraction_column: str = "fraction",
    block_column: str = "plateID",
    nuisance_columns: Sequence[str] | None = None,
):
    """Well-by-GENE fractions, the well table, and how many guides each gene has.

    :returns: ``(gene_fractions, outcomes, gene_metadata)``, the gene-level
        counterpart of :func:`prepare_long_guide_data`.
    :raises ValueError: when ``gene_column`` is absent.
    """
    if gene_column not in data.columns:
        raise ValueError(
            f"the gene-level permutation test needs a {gene_column!r} column "
            f"to sum each gene's guides into one regressor. Columns: "
            f"{sorted(data.columns)[:20]}")
    fractions, outcomes, guide_metadata = prepare_long_guide_data(
        data,
        outcome_columns,
        well_column=well_column,
        guide_column=guide_column,
        fraction_column=fraction_column,
        block_column=block_column,
        nuisance_columns=nuisance_columns,
    )
    pairs = data.loc[:, [guide_column, gene_column]].astype(str)
    pairs = pairs.drop_duplicates()
    ambiguous = pairs[guide_column].duplicated(keep=False)
    if ambiguous.any():
        offenders = sorted(pairs.loc[ambiguous, guide_column].unique())
        raise ValueError(
            "a guide belongs to exactly one gene, and summing its fraction "
            "into two genes would count the same wells twice. These guides "
            f"name more than one gene: {offenders[:10]}"
            f"{' ...' if len(offenders) > 10 else ''}")
    gene_of_guide = dict(zip(pairs[guide_column], pairs[gene_column]))
    gene_fractions = gene_fraction_matrix(fractions, gene_of_guide)

    counts = (
        pd.Series(gene_of_guide)
        .loc[[str(name) for name in fractions.columns]]
        .value_counts()
    )
    gene_metadata = pd.DataFrame({
        "gene": gene_fractions.columns,
        "wells_with_gene": (gene_fractions > 0).sum(axis=0).to_numpy(dtype=int),
        "guides_in_gene": [int(counts.get(name, 0))
                           for name in gene_fractions.columns],
    }).set_index("gene")
    return gene_fractions, outcomes, gene_metadata


def gene_freedman_lane_test(
    gene_fractions: pd.DataFrame,
    outcomes: pd.DataFrame,
    outcome_column: str,
    *,
    gene_metadata: pd.DataFrame | None = None,
    **kwargs,
) -> pd.DataFrame:
    """Test each GENE's guides as a SET, against the same permutation null.

    Instruction 132 asks the nonparametric path for a gene pass beside its
    guide pass. The gene's regressor is the SUM of its guides' fractions --
    one column, one test, one degree of freedom -- residualized against the
    same block/nuisance design and permuted with the same Freedman--Lane
    scheme. Passing the same ``random_state`` as the guide pass makes the
    permutations literally identical, because the permuted object is the
    residual of the same outcome against the same nuisance design.

    IT IS NOT A COMBINATION OF PER-GUIDE P VALUES, and that is deliberate.
    Fisher's and Stouffer's methods both assume the p-values being combined
    are independent, and two guides measured in the same wells are not: they
    share the well's phenotype, the well's plate effect and the well's cells.
    Combining them would report a confidence the design cannot support.

    WHAT A ONE-DEGREE-OF-FREEDOM SET TEST CANNOT SEE, and the reason
    ``guides_in_gene`` travels with every row: a gene whose guides push the
    phenotype in OPPOSITE directions cancels in the sum, exactly as it does in
    the parametric ``y ~ gene_fraction:gene`` fit that this mirrors. Such a
    gene is not evidence of no effect; it is evidence the guides disagree, and
    the guide-level table is where that shows. A gene resting on one guide
    (``guides_in_gene == 1``) is the same test as that guide's own row.

    :param gene_fractions: well-by-gene matrix from
        :func:`prepare_long_gene_data`.
    :param gene_metadata: the per-gene guide counts, joined onto the result.
    :param kwargs: forwarded to :func:`guide_freedman_lane_test`.
    :returns: one row per gene per minimum-wells family, BH-corrected WITHIN
        the gene family and never pooled with the guide family.
    """
    result = guide_freedman_lane_test(
        gene_fractions, outcomes, outcome_column, **kwargs)
    result = result.rename(columns={
        "guide": "gene",
        "wells_with_guide": "wells_with_gene",
        "tested_guides_in_family": "tested_genes_in_family",
    })
    result["level"] = "gene"
    if gene_metadata is not None:
        counts = gene_metadata["guides_in_gene"] if (
            "guides_in_gene" in gene_metadata.columns) else None
        if counts is not None:
            result["guides_in_gene"] = (
                result["gene"].astype(str)
                .map(counts.rename(index=str)).astype("Int64"))
    ordered = ["outcome", "gene", "level"]
    rest = [name for name in result.columns if name not in ordered]
    return result.loc[:, ordered + rest]


def analyse_long_gene_table(
    data: pd.DataFrame,
    outcome_columns: str | Sequence[str],
    *,
    min_wells: int | Sequence[int] = (1, 2, 3, 4),
    well_column: str = "prc",
    guide_column: str = "grna",
    gene_column: str = "gene",
    fraction_column: str = "fraction",
    block_column: str = "plateID",
    nuisance_columns: Sequence[str] | None = None,
    random_state: int = 0,
    **kwargs,
) -> pd.DataFrame:
    """The gene pass over spaCR's saved ``regression_data.csv``.

    The counterpart of :func:`analyse_long_guide_table`. Its BH correction is
    computed over GENES ONLY: two families, never one. Pooling them would be
    wrong twice over -- the same wells produce both, and a gene's regressor is
    literally the sum of its guides' regressors, so they are not independent;
    and doubling the family size costs power for no protection.
    """
    gene_fractions, outcomes, gene_metadata = prepare_long_gene_data(
        data,
        outcome_columns,
        well_column=well_column,
        guide_column=guide_column,
        gene_column=gene_column,
        fraction_column=fraction_column,
        block_column=block_column,
        nuisance_columns=nuisance_columns,
    )
    columns = ([outcome_columns] if isinstance(outcome_columns, str)
               else list(outcome_columns))
    frames = []
    for index, outcome in enumerate(columns):
        frames.append(gene_freedman_lane_test(
            gene_fractions,
            outcomes,
            outcome,
            gene_metadata=gene_metadata,
            min_wells=min_wells,
            block_column=block_column,
            nuisance_columns=nuisance_columns,
            # THE SAME NULL AS THE GUIDE PASS. Same seed, same outcome, same
            # nuisance design, so the permuted residual vectors are the same
            # vectors -- which is what makes the two tables comparable rather
            # than merely similar.
            random_state=int(random_state) + index,
            **kwargs,
        ))
    return pd.concat(frames, ignore_index=True)
