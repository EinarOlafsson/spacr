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
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


class AnnotationError(ValueError):
    """An invalid condition rule, ambiguous assignment or changed source."""


@dataclass
class ConditionPreview:
    """Validated membership counts and row labels before application.

    :param values: Final output labels in source row order; conflicts are diagnostic.
    :param counts: Final output row counts per distinct label.
    :param unmatched: Rows missing a value in the final output.
    :param overlaps: Union of positions assigned different labels within any output.
    :param column_values: Candidate values for every ordered output column.
    :param column_previews: Per-column membership and conflict reports.
    :param rule_counts: Individual rule counts before same-label unions.
    """
    values: pd.Series
    counts: dict
    unmatched: int
    overlaps: np.ndarray
    column_values: dict = field(default_factory=dict)
    column_previews: dict = field(default_factory=dict)
    rule_counts: list = field(default_factory=list)


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


def _entries(definition):
    """Normalize legacy definitions without changing their serialized shape."""
    if definition.get("version") == 1:
        return [{"column": definition.get("column", ""), "kind": "rules",
                 "conditions": definition.get("conditions", [])}]
    if definition.get("version") != 2:
        raise AnnotationError("Unsupported annotation version; recreate the conditions.")
    entries = definition.get("columns")
    if not isinstance(entries, list) or not entries:
        raise AnnotationError("Add at least one annotation output column.")
    if any(not isinstance(entry, dict) for entry in entries):
        raise AnnotationError("Each annotation output needs a column definition.")
    return entries


def annotation_columns(definition):
    """Return the ordered output names from a legacy definition or recipe.

    :param definition: Version 1 condition definition or version 2 column recipe.
    :returns: Distinct, nonempty output names in recipe order.
    """
    names = []
    for entry in _entries(definition):
        name = str(entry.get("column", "")).strip()
        if not name or "\x00" in name:
            raise AnnotationError("Give each annotation output column a nonempty name.")
        if name in names:
            raise AnnotationError("Annotation output column names must be distinct.")
        names.append(name)
    return names


def _exact_values(condition, key, name):
    """Validate exact-value selectors without treating a string as characters."""
    values = condition.get(key, [])
    if not isinstance(values, (list, tuple)) or any(
            value is None or not isinstance(value, (str, int, float, bool))
            or (isinstance(value, float) and not np.isfinite(value)) for value in values):
        raise AnnotationError(f"{name}: {key} must be a list of nonmissing metadata values.")
    return {str(value) for value in values}


def _rules_preview(frame, conditions, locations, *, legacy):
    """Evaluate one output, unioning repeated labels before conflict detection."""
    if not isinstance(conditions, list):
        raise AnnotationError("Conditions must be a list of rules.")
    masks = {}
    rule_counts = []
    for condition in conditions:
        if not isinstance(condition, dict):
            raise AnnotationError("Each condition needs a rule definition.")
        name = str(condition.get("name", "")).strip()
        if not name or (legacy and name in masks):
            raise AnnotationError("Each condition needs a distinct, nonempty name.")
        column_name = condition.get("metadata_column")
        if column_name not in frame.columns:
            raise AnnotationError(f"{name}: choose an available metadata column.")
        text = frame[column_name].reset_index(drop=True).astype("string")
        mode = condition.get("match_mode", "regex")
        if mode == "values":
            selected = text.isin(_exact_values(condition, "include_values", name)).to_numpy(dtype=bool)
            excluded = text.isin(_exact_values(condition, "exclude_values", name)).to_numpy(dtype=bool)
        elif mode == "regex":
            compiled = {}
            for selector in ("include", "exclude"):
                pattern = str(condition.get(selector, ""))
                try:
                    compiled[selector] = re.compile(pattern) if pattern else None
                except re.error as exc:
                    raise AnnotationError(f"{name}: invalid {selector} regular expression: {exc}") from exc
            selected = (text.str.contains(compiled["include"], na=False).to_numpy(dtype=bool)
                        if compiled["include"] else np.zeros(len(frame), dtype=bool))
            excluded = (text.str.contains(compiled["exclude"], na=False).to_numpy(dtype=bool)
                        if compiled["exclude"] else np.zeros(len(frame), dtype=bool))
        else:
            raise AnnotationError(f"{name}: choose regex or values matching.")
        for token in condition.get("manual_rows", []):
            if token not in locations:
                raise AnnotationError(f"{name}: a manually selected row no longer belongs to this source.")
            selected[locations[token]] = True
        selected &= ~excluded
        rule_counts.append(int(selected.sum()))
        if name in masks:
            masks[name] |= selected
        else:
            masks[name] = selected
    memberships = np.zeros(len(frame), dtype=np.int32)
    values = pd.Series(pd.NA, index=pd.RangeIndex(len(frame)), dtype="string")
    for name, selected in masks.items():
        already = selected & (memberships > 0)
        values.loc[selected & ~already] = name
        values.loc[already] = values.loc[already] + " / " + name
        memberships += selected
    return ConditionPreview(values, {name: int(mask.sum()) for name, mask in masks.items()},
                            int((memberships == 0).sum()), np.flatnonzero(memberships > 1),
                            rule_counts=rule_counts)


def _combine_preview(frame, entry):
    """Combine available columns positionally, keeping missing components missing."""
    columns = entry.get("columns")
    if not isinstance(columns, list) or not columns or any(
            not isinstance(column, str) or column not in frame.columns for column in columns):
        raise AnnotationError("Combine columns must name source columns or earlier annotation outputs.")
    separator = entry.get("separator", "_")
    if not isinstance(separator, str):
        raise AnnotationError("The combination separator must be text.")
    values = None
    missing = np.zeros(len(frame), dtype=bool)
    for column in columns:
        component = frame[column].reset_index(drop=True).astype("string")
        missing |= (component.isna() | component.eq("").fillna(False)).to_numpy(dtype=bool)
        values = component if values is None else values + separator + component
    values = values.mask(missing, pd.NA)
    return ConditionPreview(values, {str(key): int(value) for key, value in values.value_counts().items()},
                            int(missing.sum()), np.array([], dtype=np.int64))


def preview(frame, definition, source):
    """Evaluate ordered outputs without mutating the source or accepting conflicts.

    Regex or exact-value includes select metadata rows. Manual row assignments
    join those selections, and excludes remove matches. Version 2 rules with the
    same label are unioned; different labels within a column remain conflicts.
    Combinations may reference source columns and earlier recipe outputs only.

    :param frame: Original source frame, not a previously annotated copy.
    :param definition: Saved legacy rules or an ordered version 2 recipe.
    :param source: Current file/table/merge identity.
    :returns: ConditionPreview with all outputs and per-column diagnostics.
    """
    names = annotation_columns(definition)
    if definition.get("source") != source:
        raise AnnotationError("These conditions belong to another source or table; recreate them here.")
    schema, digest, tokens = table_identity(frame)
    if (schema != definition.get("schema") or digest != definition.get("content_sha256")
            or definition.get("row_count", len(frame)) != len(frame)):
        raise AnnotationError("The source rows, order, values or schema changed; review and recreate the conditions.")
    for name in names:
        if name in frame.columns:
            raise AnnotationError(f"The source already has a {name!r} column; choose a new name to preserve it.")
    locations = {token: index for index, token in enumerate(tokens)}
    available = frame.copy(deep=False)
    reports = {}
    outputs = {}
    conflicts = np.zeros(len(frame), dtype=bool)
    for name, entry in zip(names, _entries(definition)):
        kind = entry.get("kind", "rules")
        if kind == "rules":
            report = _rules_preview(available, entry.get("conditions", []), locations,
                                    legacy=definition["version"] == 1)
        elif kind == "combine":
            report = _combine_preview(available, entry)
        else:
            raise AnnotationError(f"{name}: choose rules or combine for the output kind.")
        reports[name] = report
        outputs[name] = report.values
        conflicts[report.overlaps] = True
        available[name] = report.values.array
    last = reports[names[-1]]
    return ConditionPreview(last.values, last.counts, last.unmatched, np.flatnonzero(conflicts),
                            outputs, reports, last.rule_counts)


def apply_conditions(frame, definition, source):
    """Return a labelled copy after source and overlap validation succeeds.

    :param frame: Original source frame.
    :param definition: Complete condition configuration.
    :param source: Current source context.
    :returns: Copy with every requested output column; original remains unchanged.
    """
    result = preview(frame, definition, source)
    if len(result.overlaps):
        raise AnnotationError(f"{len(result.overlaps):,} rows match multiple conditions. "
                              "Adjust include/exclude patterns or remove manual assignments before applying.")
    output = frame.copy()
    # Assign by position, never Series index alignment: duplicated pandas
    # indices are legal input and have no role in condition identity.
    for column, values in result.column_values.items():
        output[column] = values.array
    output.attrs["condition_annotation"] = json.loads(json.dumps(definition))
    return output


PROVENANCE_TABLE = "_spacr_condition_annotations"


def _quote(name):
    """Quote a SQLite identifier without interpreting user-provided SQL."""
    return '"' + name.replace('"', '""') + '"'


def save_annotated_table(path, name, frame, definition, source, *, merge_definition=None):
    """Atomically save a new physical table plus its source and editable rules.

    :param path: Existing SQLite source database, never a delimited input file.
    :param name: New table name; physical, view and saved-derived collisions fail.
    :param frame: Working frame including every annotation output column.
    :param definition: Validated annotation rules used for the working frame.
    :param source: Original annotation source context.
    :param merge_definition: Original merged-source configuration, if applicable.
    :returns: Saved table name. A failure rolls back both table and provenance.
    """
    import sqlite3
    from datetime import date, datetime

    from .derived_tables import load_definitions

    name = str(name).strip()
    if not name or '\x00' in name or name.lower().startswith(('sqlite_', '_spacr_')):
        raise AnnotationError("Choose a nonempty table name outside the reserved sqlite_ and _spacr_ prefixes.")
    if not definition:
        raise AnnotationError("Apply conditions before saving an annotated table.")
    if source.get('path') != str(Path(path).resolve()):
        raise AnnotationError("Save the annotated table in its current source database.")
    if name.casefold() in {item.casefold() for item in load_definitions(path)}:
        raise AnnotationError("That name belongs to a saved merged table; choose a new name.")
    output_columns = annotation_columns(definition)
    if any(column not in frame.columns for column in output_columns):
        raise AnnotationError("Apply every annotation output before saving the table.")
    base = frame.drop(columns=output_columns)
    expected = apply_conditions(base, definition, source)
    if any(not expected[column].equals(frame[column]) for column in output_columns):
        raise AnnotationError("The working annotations changed; apply the conditions again before saving.")
    columns = list(frame.columns)
    types = ['INTEGER' if pd.api.types.is_integer_dtype(dtype) or pd.api.types.is_bool_dtype(dtype)
             else 'REAL' if pd.api.types.is_float_dtype(dtype) else 'TEXT' for dtype in frame.dtypes]

    def scalar(value):
        if value is None or value is pd.NA or value is pd.NaT:
            return None
        if isinstance(value, (datetime, date, pd.Timestamp)):
            return value.isoformat()
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, float) and np.isnan(value):
            return None
        return value

    with sqlite3.connect(f"file:{Path(path).resolve()}?mode=rw", uri=True, timeout=30) as db:
        db.execute('BEGIN IMMEDIATE')
        found = db.execute('SELECT name FROM sqlite_master WHERE lower(name)=lower(?)', (name,)).fetchone()
        if found:
            raise AnnotationError("That table or view already exists; choose a new name to preserve it.")
        # Validate reserved metadata before creating the user table. A conflicting
        # unrelated table is never silently repurposed as our receipt store.
        existing = db.execute('SELECT type FROM sqlite_master WHERE name=?', (PROVENANCE_TABLE,)).fetchone()
        if existing:
            fields = [row[1] for row in db.execute(f'PRAGMA table_info({_quote(PROVENANCE_TABLE)})')]
            if fields != ['table_name', 'payload_json'] or existing[0] != 'table':
                raise AnnotationError("The annotation provenance table has an incompatible schema.")
        else:
            db.execute(f'CREATE TABLE {_quote(PROVENANCE_TABLE)} (table_name TEXT PRIMARY KEY, payload_json TEXT NOT NULL)')
        sql_columns = ', '.join(f'{_quote(str(column))} {dtype}' for column, dtype in zip(columns, types))
        db.execute(f'CREATE TABLE {_quote(name)} ({sql_columns})')
        placeholders = ', '.join('?' for _ in columns)
        db.executemany(f'INSERT INTO {_quote(name)} VALUES ({placeholders})',
                       (tuple(scalar(value) for value in row) for row in frame.itertuples(index=False, name=None)))
        # Bind reopen/edit to what SQLite actually stored, including dtype
        # normalization, rather than assuming a pandas/SQL roundtrip is lossless.
        materialized = pd.read_sql_query(f'SELECT * FROM {_quote(name)}', db)
        editable_base = materialized.drop(columns=output_columns)
        editable = json.loads(json.dumps(definition))
        rebound = new_definition(editable_base, source_context(path, name))
        for key in ('source', 'schema', 'content_sha256', 'row_count'):
            editable[key] = rebound[key]
        old_tokens = table_identity(base)[2]
        new_tokens = table_identity(editable_base)[2]
        token_mapping = dict(zip(old_tokens, new_tokens))
        for entry in _entries(editable):
            for condition in entry.get('conditions', []):
                condition['manual_rows'] = [token_mapping[token] for token in condition.get('manual_rows', [])]
        reproduced = apply_conditions(editable_base, editable, source_context(path, name))
        if any(reproduced[column].fillna('').tolist() != materialized[column].fillna('').tolist()
               for column in output_columns):
            raise AnnotationError("SQLite storage changed a rule's matches; export CSV or adjust the rules before saving.")
        schema, digest, _tokens = table_identity(materialized)
        payload = {'version': 1, 'source': source, 'merge_definition': merge_definition,
                   'original_definition': definition, 'editable_definition': editable,
                   'stored_schema': schema, 'stored_content_sha256': digest}
        db.execute(f'INSERT INTO {_quote(PROVENANCE_TABLE)} VALUES (?, ?)',
                   (name, json.dumps(payload)))
    return name


def saved_table_annotation(path, name, frame):
    """Recover editable rules only when a saved physical table is unchanged.

    :param path: SQLite source database opened read-only.
    :param name: Physical table name.
    :param frame: Actual stored table including every annotation output column.
    :returns: Editable annotation definition or None for an ordinary table.
    """
    import sqlite3

    with sqlite3.connect(f"file:{Path(path).resolve()}?mode=ro", uri=True, timeout=30) as db:
        if not db.execute('SELECT 1 FROM sqlite_master WHERE type=\'table\' AND name=?', (PROVENANCE_TABLE,)).fetchone():
            return None
        fields = [row[1] for row in db.execute(f'PRAGMA table_info({_quote(PROVENANCE_TABLE)})')]
        if fields != ['table_name', 'payload_json']:
            return None
        row = db.execute(f'SELECT payload_json FROM {_quote(PROVENANCE_TABLE)} WHERE table_name=?', (name,)).fetchone()
    if not row:
        return None
    payload = json.loads(row[0])
    schema, digest, _tokens = table_identity(frame)
    if schema != payload.get('stored_schema') or digest != payload.get('stored_content_sha256'):
        raise AnnotationError("The saved annotated table changed; its previous editable rules were not restored.")
    return payload['editable_definition']
