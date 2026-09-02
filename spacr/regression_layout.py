"""Lossless long/wide conversions for regression predictor tables.

The screen pipeline historically receives one row per ``(well, gRNA)`` and
the low-level estimators receive a conventional wide design matrix.  These
helpers make that boundary explicit and also accept count tables that arrive
with one guide per column.  Conversion is deliberately strict: a value is
never silently selected when duplicate rows disagree.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd


REGRESSION_LAYOUTS = ("auto", "long", "wide")

# Columns that describe the observation rather than one independent
# variable.  Callers can add project-specific columns through ``id_columns``.
DEFAULT_ID_COLUMNS = (
    "prc", "plateID", "rowID", "columnID", "screenID", "fieldID",
    "objectID", "timeID", "cell_count", "gene", "condition",
)


def _names(values: Sequence[str] | str | None) -> list[str]:
    if values is None:
        return []
    if isinstance(values, str):
        return [values]
    return [str(value) for value in values]


def infer_regression_layout(
    frame: pd.DataFrame,
    *,
    predictor_column: str = "grna",
    value_column: str = "count",
) -> str:
    """Infer ``long`` only from the paired predictor/value columns.

    A table containing exactly one of the two columns is malformed rather
    than wide.  Treating it as wide would accidentally melt the existing
    value column into a gRNA.
    """
    has_predictor = predictor_column in frame.columns
    has_value = value_column in frame.columns
    if has_predictor and has_value:
        return "long"
    if has_predictor != has_value:
        present = predictor_column if has_predictor else value_column
        missing = value_column if has_predictor else predictor_column
        raise ValueError(
            f"Independent-variable table has {present!r} but not {missing!r}. "
            "A long table needs both; a wide table needs neither because "
            "each predictor is its own column."
        )
    return "wide"


def wide_to_long_regression_data(
    frame: pd.DataFrame,
    *,
    predictor_columns: Sequence[str] | None = None,
    id_columns: Sequence[str] | None = None,
    predictor_name: str = "grna",
    value_name: str = "count",
    drop_zero: bool = False,
) -> pd.DataFrame:
    """Melt one-predictor-per-column data to one row per predictor.

    When ``predictor_columns`` is omitted, every numeric column not named as
    observation metadata is used.  Non-numeric unclassified columns are
    rejected because guessing whether they are metadata or a predictor would
    change the model silently.
    """
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")
    identifiers = list(dict.fromkeys(
        [name for name in _names(id_columns) if name in frame.columns]
    ))
    if not identifiers:
        identifiers = [
            name for name in DEFAULT_ID_COLUMNS if name in frame.columns
        ]
    explicit = _names(predictor_columns)
    if explicit:
        missing = [name for name in explicit if name not in frame.columns]
        if missing:
            raise ValueError(
                "wide predictor columns are absent: " + ", ".join(missing)
            )
        predictors = explicit
    else:
        candidates = [name for name in frame.columns if name not in identifiers]
        predictors = [
            name for name in candidates
            if pd.api.types.is_numeric_dtype(frame[name])
        ]
        unclassified = [name for name in candidates if name not in predictors]
        if unclassified:
            raise ValueError(
                "Could not infer the wide predictor columns because these "
                "non-numeric columns are not declared as metadata: "
                + ", ".join(map(str, unclassified))
                + ". Pass predictor_columns or id_columns explicitly."
            )
    if not predictors:
        raise ValueError("No wide independent-variable columns were found")
    overlap = sorted(set(identifiers) & set(predictors))
    if overlap:
        raise ValueError(
            "Columns cannot be both identifiers and predictors: "
            + ", ".join(overlap)
        )
    numeric = frame[predictors].apply(pd.to_numeric, errors="raise")
    values = numeric.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Wide independent-variable values must be finite")
    prepared = pd.concat(
        [frame[identifiers].reset_index(drop=True), numeric.reset_index(drop=True)],
        axis=1,
    )
    result = prepared.melt(
        id_vars=identifiers,
        value_vars=predictors,
        var_name=predictor_name,
        value_name=value_name,
    )
    if drop_zero:
        result = result.loc[result[value_name].ne(0)].copy()
    return result.reset_index(drop=True)


def long_to_wide_regression_data(
    frame: pd.DataFrame,
    *,
    index_columns: Sequence[str] | str = "prc",
    predictor_column: str = "grna",
    value_column: str = "fraction",
    metadata_columns: Sequence[str] | None = None,
    fill_value: float = 0.0,
    predictor_prefix: str = "",
) -> pd.DataFrame:
    """Pivot one row per observation/predictor to one predictor per column.

    Metadata must be constant inside each observation, and duplicate
    observation/predictor rows must agree.  Repeated identical rows (for
    example after a harmless join) are collapsed once; conflicting values are
    refused.
    """
    indices = _names(index_columns)
    required = indices + [predictor_column, value_column]
    missing = [name for name in required if name not in frame.columns]
    if missing:
        raise ValueError("Long regression table lacks: " + ", ".join(missing))
    if not indices:
        raise ValueError("At least one index column is required")
    values = pd.to_numeric(frame[value_column], errors="raise")
    if not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError("Long independent-variable values must be finite")
    work = frame[required].copy()
    work[value_column] = values
    keys = indices + [predictor_column]
    disagreement = work.groupby(keys, observed=True)[value_column].nunique()
    if disagreement.gt(1).any():
        example = disagreement[disagreement.gt(1)].index[0]
        raise ValueError(
            "A long observation/predictor pair carries conflicting values; "
            f"for example {example!r}. Aggregate or correct it before pivoting."
        )
    work = work.drop_duplicates(keys)
    wide = work.pivot(
        index=indices, columns=predictor_column, values=value_column
    ).fillna(float(fill_value))
    wide.columns = [predictor_prefix + str(name) for name in wide.columns]
    wide = wide.reset_index()

    metadata = [
        name for name in _names(metadata_columns)
        if name not in indices and name in frame.columns
    ]
    if metadata:
        grouped = frame.groupby(indices, observed=True)[metadata].nunique(
            dropna=False
        )
        conflicts = grouped.gt(1)
        if conflicts.any().any():
            row, column = conflicts.stack()[lambda value: value].index[0]
            raise ValueError(
                f"Metadata column {column!r} is not constant within "
                f"observation {row!r}; it cannot be attached to one wide row."
            )
        one = frame[indices + metadata].drop_duplicates(indices)
        wide = one.merge(wide, on=indices, how="right", validate="one_to_one")
    return wide


def normalise_count_table_layout(
    frame: pd.DataFrame,
    *,
    layout: str = "auto",
    guide_column: str = "grna",
    count_column: str = "count",
    wide_predictor_columns: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, str]:
    """Return a canonical long ``grna``/``count`` count table and its input layout."""
    wanted = str(layout or "auto").strip().lower()
    if wanted not in REGRESSION_LAYOUTS:
        raise ValueError(
            f"independent_variable_layout={layout!r}; choose one of "
            f"{REGRESSION_LAYOUTS}."
        )
    # ``process_reads`` has accepted the historical downloadable-data header
    # ``grna_name`` for years.  Canonicalise it before layout inference so an
    # otherwise valid long table is not misdiagnosed as a malformed wide one.
    if (guide_column == "grna" and "grna" not in frame.columns
            and "grna_name" in frame.columns):
        frame = frame.rename(columns={"grna_name": "grna"})
    resolved = (
        infer_regression_layout(
            frame, predictor_column=guide_column, value_column=count_column
        )
        if wanted == "auto" else wanted
    )
    if resolved == "long":
        missing = [
            name for name in (guide_column, count_column)
            if name not in frame.columns
        ]
        if missing:
            raise ValueError(
                "Long independent-variable table lacks: " + ", ".join(missing)
            )
        result = frame.rename(
            columns={guide_column: "grna", count_column: "count"}
        ).copy()
    else:
        result = wide_to_long_regression_data(
            frame,
            predictor_columns=wide_predictor_columns,
            predictor_name="grna",
            value_name="count",
            drop_zero=True,
        )
    return result, resolved


__all__ = [
    "DEFAULT_ID_COLUMNS",
    "REGRESSION_LAYOUTS",
    "infer_regression_layout",
    "long_to_wide_regression_data",
    "normalise_count_table_layout",
    "wide_to_long_regression_data",
]
