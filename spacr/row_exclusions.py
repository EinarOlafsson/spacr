"""General row-exclusion rules shared by UMAP and its parameter search."""

from __future__ import annotations

import ast
import json
from collections.abc import Mapping
from typing import Any


def normalize_row_exclusions(value: Any) -> dict[str, list[Any]]:
    """Return ``{column: [values...]}`` from settings or CSV text.

    ``None`` and an empty value mean no exclusions. A scalar value is accepted
    as a one-item list so hand-written settings remain convenient.
    """
    if value in (None, "", {}, []):
        return {}
    if isinstance(value, str):
        text = value.strip()
        if not text or text.lower() in {"none", "null"}:
            return {}
        try:
            value = json.loads(text)
        except (TypeError, ValueError, json.JSONDecodeError):
            try:
                value = ast.literal_eval(text)
            except (ValueError, SyntaxError) as exc:
                raise ValueError(
                    "exclude_rows must be a mapping such as "
                    "{'columnID': ['c1', 'c2']}."
                ) from exc
    if not isinstance(value, Mapping):
        raise ValueError(
            "exclude_rows must map column names to values, for example "
            "{'columnID': ['c1', 'c2']}."
        )

    normalized: dict[str, list[Any]] = {}
    for raw_column, raw_values in value.items():
        column = str(raw_column).strip()
        if not column:
            continue
        if isinstance(raw_values, (list, tuple, set, frozenset)):
            values = list(raw_values)
        else:
            values = [raw_values]
        deduplicated: list[Any] = []
        seen: set[tuple[str, str]] = set()
        for item in values:
            token = (type(item).__name__, repr(item))
            if token not in seen:
                seen.add(token)
                deduplicated.append(item)
        if deduplicated:
            normalized[column] = deduplicated
    return normalized


def exclude_matching_rows(frame, rules: Any) -> tuple[Any, list[str]]:
    """Drop rows matching any configured column/value rule.

    Values are compared both in their native dtype and as stripped strings.
    This lets a value selected from SQLite text match the equivalent pandas
    numeric value without changing identifiers such as ``"001"``.

    :returns: ``(filtered_frame, notes)``. Each note names the column, values,
        and number of rows removed.
    :raises ValueError: for unknown columns or rules that remove every row.
    """
    import pandas as pd

    normalized = normalize_row_exclusions(rules)
    if not normalized:
        return frame, []

    missing = [column for column in normalized if column not in frame.columns]
    if missing:
        available = ", ".join(str(c) for c in frame.columns[:20])
        if len(frame.columns) > 20:
            available += ", …"
        raise ValueError(
            f"Cannot exclude rows by unknown column(s): {missing}. "
            f"Available columns include: {available}"
        )

    remove = pd.Series(False, index=frame.index)
    notes: list[str] = []
    for column, values in normalized.items():
        series = frame[column]
        native = series.isin(values)
        text_values = {str(value).strip() for value in values}
        as_text = series.astype("string").str.strip().isin(text_values)
        wants_null = any(
            value is None
            or str(value).strip().lower() in {"none", "null", "nan", "<na>"}
            for value in values
        )
        matched = native | as_text
        if wants_null:
            matched |= series.isna()
        newly_removed = matched & ~remove
        remove |= matched
        notes.append(
            f"Excluded {int(newly_removed.sum())} row(s) where "
            f"{column} is one of {values!r}."
        )

    filtered = frame.loc[~remove].copy()
    if filtered.empty:
        raise ValueError(
            "The configured row exclusions removed every UMAP object. "
            "Remove at least one excluded value before running."
        )
    return filtered, notes
