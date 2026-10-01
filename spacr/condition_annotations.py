"""Reproducible condition labels on an unchanged source table.

Assignments use content-bound positional row tokens, never pandas index labels
or the sorted/filtered row number shown by a view. Definitions retain their
source, selected table, schema, full ordered content fingerprint, regex rules
and manually selected tokens. A changed source is refused before any labels
are applied. Source measurements are never overwritten.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


class AnnotationError(ValueError):
    """An invalid condition rule, ambiguous assignment or changed source."""


@dataclass
class ConditionPreview:
    """Validated membership counts and row labels before application.

    :param values: Candidate labels in source row order; overlaps are diagnostic.
    :param counts: Matched row count per condition.
    :param unmatched: Rows assigned to no condition.
    :param overlaps: Source row positions assigned to multiple conditions.
    """
    values: pd.Series
    counts: dict
    unmatched: int
    overlaps: np.ndarray


def source_context(path=None, table=None, merge_definition=None):
    """Identify the source without treating a derived table as a physical one.

    :param path: Source file, or None for an in-memory table.
    :param table: Selected table name; None for a delimited file.
    :param merge_definition: Reproduction configuration for a derived table.
    :returns: JSON-compatible source identity.
    """
    digest = hashlib.sha256(json.dumps(merge_definition, sort_keys=True,
                                      default=str).encode()).hexdigest() if merge_definition else None
    return {"path": str(Path(path).resolve()) if path else None,
            "table": table or None, "merge_sha256": digest}


def table_identity(frame):
    """Hash table values efficiently while ignoring mutable pandas index labels.

    :param frame: Original unannotated source frame.
    :returns: Schema, ordered content digest, and opaque per-row tokens.
    """
    if not frame.columns.is_unique:
        raise AnnotationError("The source has duplicate column names; select a table with unique names.")
    schema = [[str(column), str(dtype)] for column, dtype in frame.dtypes.items()]
    # Vectorized hashing keeps large measurement tables practical. The final
    # digest binds every row in order to its schema and source definition.
    hashes = pd.util.hash_pandas_object(frame, index=False).to_numpy(dtype=np.uint64)
    digest = hashlib.sha256(json.dumps(schema).encode() + hashes.tobytes()).hexdigest()
    occurrences = pd.Series(hashes).groupby(hashes, sort=False).cumcount().to_numpy()
    tokens = [f"{int(value):016x}:{int(occurrence)}"
              for value, occurrence in zip(hashes, occurrences)]
    return schema, digest, tokens


def new_definition(frame, source, *, column="condition"):
    """Create an unassigned condition definition for exactly this source snapshot.

    :param frame: Original source frame.
    :param source: Identity returned by source_context.
    :param column: New output column name.
    :returns: Reproducible JSON-compatible annotation definition.
    """
    schema, digest, _tokens = table_identity(frame)
    return {"version": 1, "source": source, "schema": schema,
            "content_sha256": digest, "row_count": len(frame),
            "column": column, "conditions": []}


def preview(frame, definition, source):
    """Evaluate rules without mutating the frame or accepting conflicting labels.

    An include pattern selects matching nonmissing metadata values. Blank include
    means manual assignments only. Exclude always removes matching rows, including
    manually added rows. Different conditions may overlap in a preview, but Apply
    refuses until those overlaps are resolved.

    :param frame: Original source frame, not a previously annotated copy.
    :param definition: Saved or edited condition rules.
    :param source: Current file/table/merge identity.
    :returns: ConditionPreview with explicit unmatched and overlap diagnostics.
    """
    if definition.get("version") != 1:
        raise AnnotationError("Unsupported annotation version; recreate the conditions.")
    if definition.get("source") != source:
        raise AnnotationError("These conditions belong to another source or table; recreate them here.")
    schema, digest, tokens = table_identity(frame)
    if schema != definition.get("schema") or digest != definition.get("content_sha256"):
        raise AnnotationError("The source rows, order, values or schema changed; review and recreate the conditions.")
    column = str(definition.get("column", "")).strip()
    if not column:
        raise AnnotationError("Give the condition output column a name.")
    if column in frame.columns:
        raise AnnotationError(f"The source already has a {column!r} column; choose a new name to preserve it.")
    names = []
    for condition in definition.get("conditions", []):
        name = str(condition.get("name", "")).strip()
        if not name or name in names:
            raise AnnotationError("Each condition needs a distinct, nonempty name.")
        names.append(name)
    locations = {token: index for index, token in enumerate(tokens)}
    counts = {}
    memberships = np.zeros(len(frame), dtype=np.int32)
    values = pd.Series(pd.NA, index=pd.RangeIndex(len(frame)), dtype="string")
    for name, condition in zip(names, definition.get("conditions", [])):
        column_name = condition.get("metadata_column")
        if column_name not in frame.columns:
            raise AnnotationError(f"{name}: choose an available metadata column.")
        compiled = {}
        for mode in ("include", "exclude"):
            pattern = str(condition.get(mode, ""))
            try:
                compiled[mode] = re.compile(pattern) if pattern else None
            except re.error as exc:
                raise AnnotationError(f"{name}: invalid {mode} regular expression: {exc}") from exc
        text = frame[column_name].reset_index(drop=True).astype("string")
        selected = np.zeros(len(frame), dtype=bool)
        if compiled["include"]:
            selected |= text.str.contains(compiled["include"], na=False).to_numpy(dtype=bool)
        for token in condition.get("manual_rows", []):
            if token not in locations:
                raise AnnotationError(f"{name}: a manually selected row no longer belongs to this source.")
            selected[locations[token]] = True
        if compiled["exclude"]:
            selected &= ~text.str.contains(compiled["exclude"], na=False).to_numpy(dtype=bool)
        counts[name] = int(selected.sum())
        already = selected & (memberships > 0)
        values.loc[selected & ~already] = name
        values.loc[already] = values.loc[already] + " / " + name
        memberships += selected
    return ConditionPreview(values=values, counts=counts, unmatched=int((memberships == 0).sum()),
                            overlaps=np.flatnonzero(memberships > 1))


def apply_conditions(frame, definition, source):
    """Return a labelled copy after source and overlap validation succeeds.

    :param frame: Original source frame.
    :param definition: Complete condition configuration.
    :param source: Current source context.
    :returns: Copy with the requested condition column; original remains unchanged.
    """
    result = preview(frame, definition, source)
    if len(result.overlaps):
        raise AnnotationError(f"{len(result.overlaps):,} rows match multiple conditions. "
                              "Adjust include/exclude patterns or remove manual assignments before applying.")
    output = frame.copy()
    # Assign by position, never Series index alignment: duplicated pandas
    # indices are legal input and have no role in condition identity.
    output[str(definition["column"]).strip()] = result.values.array
    output.attrs["condition_annotation"] = definition
    return output
