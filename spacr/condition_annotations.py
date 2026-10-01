"""Reproducible condition labels on an unchanged source table.

Assignments use content-bound positional row tokens, never pandas index labels
or the sorted/filtered row number shown by a view. Definitions retain their
source, selected table, schema, full ordered content fingerprint, metadata
rules, and manually selected tokens. A changed source is refused before any
labels are applied. Source measurements are never overwritten.

Version 1 defines one label column. Version 2 defines ordered label columns
and combinations. Version 3 also extracts text with regular expressions,
tests metadata with readable comparison rules, and composes values from
ordered column references and fixed text. Each output may use source columns
and earlier outputs, but cannot replace a source column.
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
    if definition.get("version") not in (2, 3):
        raise AnnotationError("Unsupported annotation version; recreate the conditions.")
    entries = definition.get("columns")
    if not isinstance(entries, list) or not entries:
        raise AnnotationError("Add at least one annotation output column.")
    if any(not isinstance(entry, dict) for entry in entries):
        raise AnnotationError("Each annotation output needs a column definition.")
    return entries


def annotation_columns(definition):
    """Return the ordered output names from a legacy definition or recipe.

    :param definition: Version 1 condition definition, or an ordered column
        recipe using version 2 or 3.
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


def _predicate_selection(frame, criterion, name):
    # Literal operators never interpret regex metacharacters.
    if not isinstance(criterion, dict):
        raise AnnotationError(f"{name}: each criterion needs a column, operator and text value.")
    column = criterion.get("metadata_column")
    if not isinstance(column, str) or column not in frame.columns:
        raise AnnotationError(f"{name}: choose an available metadata column for each criterion.")
    value = criterion.get("value")
    if not isinstance(value, str):
        raise AnnotationError(f"{name}: criterion values must be text.")
    text = frame[column].reset_index(drop=True).astype("string")
    present = text.notna().to_numpy(dtype=bool)
    operator = criterion.get("operator")
    if not isinstance(operator, str):
        raise AnnotationError(f"{name}: choose a criterion operator.")
    if value == "" and operator in {"contains", "not_contains", "starts_with", "ends_with", "regex", "not_regex"}:
        raise AnnotationError(f"{name}: enter nonempty text for {operator}; use equals to match an empty value.")
    negative = operator in {"not_contains", "not_equals", "not_regex"}
    if operator in {"contains", "not_contains"}:
        selected = text.str.contains(value, regex=False, na=False)
    elif operator in {"equals", "not_equals"}:
        selected = text.eq(value).fillna(False)
    elif operator == "starts_with":
        selected = text.str.startswith(value, na=False)
    elif operator == "ends_with":
        selected = text.str.endswith(value, na=False)
    elif operator in {"regex", "not_regex"}:
        try:
            pattern = re.compile(value)
        except re.error as exc:
            raise AnnotationError(f"{name}: invalid criterion regular expression: {exc}") from exc
        selected = text.map(lambda item: bool(pattern.search(item)) if pd.notna(item) else False)
    else:
        raise AnnotationError(f"{name}: unsupported criterion operator {operator!r}.")
    selected = selected.to_numpy(dtype=bool)
    return present & (~selected if negative else selected)


def _criteria_selection(frame, conditions, name, match):
    if not isinstance(conditions, list):
        raise AnnotationError(f"{name}: criteria must be a list.")
    if not isinstance(match, str) or match not in {"all", "any"}:
        raise AnnotationError(f"{name}: criteria matching must be all or any.")
    selections = [_predicate_selection(frame, criterion, name) for criterion in conditions]
    if not selections:
        raise AnnotationError(f"{name}: add at least one criterion, or use a manual-only rule without criteria.")
    return (np.logical_and.reduce(selections) if match == "all"
            else np.logical_or.reduce(selections))


def _rules_preview(frame, conditions, locations, *, legacy, version=2):
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
        criteria = version == 3 and "criteria" in condition
        mode = condition.get("match_mode", "regex")
        # A criteria-only rule needs no unused legacy metadata selector.
        needs_column = (not criteria or bool(condition.get("exclude"))
                        or (mode == "values" and bool(condition.get("exclude_values"))))
        if needs_column and column_name not in frame.columns:
            raise AnnotationError(f"{name}: choose an available metadata column.")
        text = (frame[column_name].reset_index(drop=True).astype("string")
                if column_name in frame.columns else None)
        selected = (_criteria_selection(frame, condition["criteria"], name,
                                        condition.get("match", "all")) if criteria else None)
        excluded = np.zeros(len(frame), dtype=bool)
        if mode == "values":
            if not criteria:
                selected = text.isin(_exact_values(condition, "include_values", name)).to_numpy(dtype=bool)
            exclude_values = _exact_values(condition, "exclude_values", name)
            if exclude_values:
                excluded = text.isin(exclude_values).to_numpy(dtype=bool)
        elif mode == "regex" or (version == 3 and mode in {"contains", "not_contains", "equals"}):
            if not criteria and mode != "regex":
                selected = _predicate_selection(frame, {"metadata_column": column_name,
                    "operator": mode, "value": condition.get("match_text")}, name)
            for selector in ("include", "exclude"):
                if selector == "include" and (criteria or mode != "regex"):
                    continue
                pattern = str(condition.get(selector, ""))
                try:
                    compiled = re.compile(pattern) if pattern else None
                except re.error as exc:
                    raise AnnotationError(f"{name}: invalid {selector} regular expression: {exc}") from exc
                matches = (text.str.contains(compiled, na=False).to_numpy(dtype=bool)
                           if compiled else np.zeros(len(frame), dtype=bool))
                if selector == "include":
                    selected = matches
                else:
                    excluded = matches
        else:
            raise AnnotationError(f"{name}: choose a supported matching mode.")
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


def _value_preview(values):
    values = values.astype("string")
    return ConditionPreview(values, {str(key): int(value) for key, value in values.value_counts().items()},
                            int(values.isna().sum()), np.array([], dtype=np.int64))


def _extract_preview(frame, entry):
    column = entry.get("metadata_column")
    if not isinstance(column, str) or column not in frame.columns:
        raise AnnotationError("Extraction must name a source column or an earlier annotation output.")
    pattern = entry.get("pattern")
    if not isinstance(pattern, str) or not pattern:
        raise AnnotationError("Extraction needs a nonempty regular expression.")
    try:
        compiled = re.compile(pattern)
    except re.error as exc:
        raise AnnotationError(f"Invalid extraction regular expression: {exc}") from exc
    group = entry.get("group", 1)
    if (isinstance(group, bool) or not isinstance(group, (str, int))
            or (isinstance(group, int) and not 0 <= group <= compiled.groups)
            or (isinstance(group, str) and group not in compiled.groupindex)):
        raise AnnotationError("Choose an existing numbered or named capture group.")
    text = frame[column].reset_index(drop=True).astype("string")

    def extract(value):
        if pd.isna(value):
            return pd.NA
        match = compiled.search(value)
        captured = match.group(group) if match is not None else None
        return captured if captured is not None and captured != "" else pd.NA

    return _value_preview(text.map(extract))


def _template_preview(frame, entry):
    parts = entry.get("parts")
    if not isinstance(parts, list) or not parts:
        raise AnnotationError("Add at least one column or fixed-text part to the composition.")
    values = pd.Series("", index=pd.RangeIndex(len(frame)), dtype="string")
    missing = np.zeros(len(frame), dtype=bool)
    for part in parts:
        if not isinstance(part, dict):
            raise AnnotationError("Each composition part must be a column or fixed text.")
        if part.get("kind") == "text":
            text = part.get("text")
            if not isinstance(text, str):
                raise AnnotationError("Fixed composition text must be a string.")
            values = values + text
        elif part.get("kind") == "column":
            column = part.get("column")
            if not isinstance(column, str) or column not in frame.columns:
                raise AnnotationError("Composition columns must name source columns or earlier annotation outputs.")
            component = frame[column].reset_index(drop=True).astype("string")
            missing |= (component.isna() | component.eq("").fillna(False)).to_numpy(dtype=bool)
            values = values + component
        else:
            raise AnnotationError("Each composition part must be a column or fixed text.")
    return _value_preview(values.mask(missing | values.eq("").fillna(False).to_numpy(dtype=bool), pd.NA))


def preview(frame, definition, source):
    """Evaluate ordered outputs and report conflicts without changing source data.

    Regex or exact-value includes select metadata rows. Manual row assignments
    join those selections, and excludes remove matches, including manual rows.
    In versions 2 and 3, combine selections that assign the same label.
    Different labels assigned to one row within a column remain conflicts.

    Version 3 criteria can require all comparisons or any comparison to match.
    Supported comparisons include literal containment, equality, prefixes,
    suffixes, and regular expressions. Containment, equality, and regex
    comparisons also have negative forms. Missing metadata never matches,
    including negative comparisons. Matching is case-sensitive unless a regex
    explicitly changes that behavior. Criteria for containment, prefixes,
    suffixes, and regex matching require nonempty comparison text.

    An empty legacy regex Include field selects no rows. This allows a rule
    that assigns only its manually selected rows; an empty regex criterion
    in version 3 is invalid.

    Version 3 extraction searches each value for the first regex match.
    Return the selected numbered or named capture group; group zero returns
    the entire match. An absent match or empty capture produces a missing value.

    Combinations join columns with a separator. Version 3 templates concatenate
    ordered column references and fixed text. An empty or missing column value
    makes the composed result missing. A template that produces empty text is
    also missing. Outputs may reference source columns and earlier outputs only.

    :param frame: Original source frame, not a previously annotated copy.
    :param definition: Saved version 1 rules, or an ordered version 2 or 3 recipe.
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
                                    legacy=definition["version"] == 1, version=definition["version"])
        elif kind == "combine":
            report = _combine_preview(available, entry)
        elif kind == "extract" and definition["version"] == 3:
            report = _extract_preview(available, entry)
        elif kind == "template" and definition["version"] == 3:
            report = _template_preview(available, entry)
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


_SCHEMA_MAX_BYTES = 8 * 1024 * 1024


def _schema_columns(columns, *, importing, version):
    """Copy only portable recipe fields; never import source-bound memberships."""
    allowed = {
        "rules": {"column", "kind", "conditions"},
        "extract": {"column", "kind", "metadata_column", "pattern", "group"},
        "combine": {"column", "kind", "columns", "separator"},
        "template": {"column", "kind", "parts"},
    }
    rule_fields = {"name", "metadata_column", "include", "exclude", "manual_rows",
                   "match_mode", "match_text", "include_values", "exclude_values",
                   "criteria", "match"}
    if not isinstance(columns, list) or not columns:
        raise AnnotationError("The annotation schema needs at least one output column.")
    try:
        result = json.loads(json.dumps(columns, allow_nan=False))
    except (TypeError, ValueError, RecursionError) as exc:
        raise AnnotationError("The annotation schema must contain finite JSON values.") from exc
    omitted = 0
    for entry in result:
        if not isinstance(entry, dict):
            raise AnnotationError("Each schema output must be a column definition.")
        kind = entry.setdefault("kind", "rules")
        if not isinstance(kind, str) or kind not in allowed or set(entry) - allowed[kind]:
            raise AnnotationError("The annotation schema contains unsupported column fields.")
        if not isinstance(entry.get("column"), str):
            raise AnnotationError("Schema output column names must be text.")
        if kind == "rules":
            conditions = entry.get("conditions", [])
            if not isinstance(conditions, list):
                raise AnnotationError("Schema conditions must be a list.")
            for rule in conditions:
                if not isinstance(rule, dict) or set(rule) - rule_fields:
                    raise AnnotationError("The annotation schema contains unsupported rule fields.")
                if not isinstance(rule.get("name"), str):
                    raise AnnotationError("Schema labels must be text.")
                for key in ("metadata_column", "include", "exclude", "match_mode", "match_text", "match"):
                    if key in rule and not isinstance(rule[key], str):
                        raise AnnotationError(f"Schema rule field {key!r} must be text.")
                for key in ("include_values", "exclude_values"):
                    if key in rule:
                        _exact_values(rule, key, rule["name"])
                if "criteria" in rule and version < 3:
                    raise AnnotationError("Criteria require annotation recipe version 3.")
                manual = rule.get("manual_rows", [])
                if not isinstance(manual, list) or (importing and manual):
                    raise AnnotationError("Reusable schemas cannot contain manual row assignments.")
                omitted += len(manual)
                rule["manual_rows"] = []
                if rule.get("match_mode") in ("contains", "not_contains", "equals"):
                    if "criteria" not in rule:
                        rule["criteria"] = [{"metadata_column": rule.get("metadata_column"),
                                             "operator": rule["match_mode"],
                                             "value": rule.get("match_text")}]
                        rule["match"] = "all"
                    rule["match_mode"] = "regex"
                    rule["include"] = ""
                    rule.pop("match_text", None)
                # Inactive fields must not become active merely because an editor
                # selects a mode after loading. Preserve the evaluator's meaning.
                if rule.get("match_mode", "regex") == "values":
                    rule.pop("include", None)
                    rule.pop("exclude", None)
                else:
                    rule.pop("include_values", None)
                    rule.pop("exclude_values", None)
                if "criteria" in rule:
                    criteria = rule["criteria"]
                    if not isinstance(criteria, list) or any(
                            not isinstance(c, dict)
                            or set(c) != {"metadata_column", "operator", "value"}
                            for c in criteria):
                        raise AnnotationError("Schema criteria need a column, operator and text value.")
        elif kind == "template":
            parts = entry.get("parts")
            if not isinstance(parts, list) or any(
                    not isinstance(part, dict)
                    or (part.get("kind") == "column" and set(part) != {"kind", "column"})
                    or (part.get("kind") == "text" and set(part) != {"kind", "text"})
                    or part.get("kind") not in ("column", "text") for part in parts):
                raise AnnotationError("Schema composition parts must be column references or fixed text.")
    return result, omitted


def _save_schema(path, frame, definition, source):
    """Validate a snapshot and atomically save portable rules, returning omitted rows."""
    from .run_journal import _atomic_write_text

    target = Path(path).expanduser()
    source_path = source.get("path")
    if source_path:
        original = Path(source_path)
        if (target.resolve() == original.resolve()
                or (target.exists() and original.exists() and target.samefile(original))):
            raise AnnotationError("Save the annotation schema separately from the source table.")
    version = definition.get("version")
    if type(version) is not int or version not in (1, 2, 3):
        raise AnnotationError("Unsupported annotation recipe version.")
    # Validate portable fields before the legacy evaluator can coerce malformed
    # values or ignore fields that the schema reader would have to reject.
    columns, omitted = _schema_columns(_entries(definition), importing=False, version=version)
    report = preview(frame, definition, source)
    if len(report.overlaps):
        raise AnnotationError("Resolve overlapping labels before saving the annotation schema.")
    payload = {"format": "spacr.annotation-schema", "version": 1,
               "recipe_version": definition["version"], "manual_rows": "excluded",
               "columns": columns}
    text = json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
    if len(text.encode("utf-8")) > _SCHEMA_MAX_BYTES:
        raise AnnotationError("The annotation schema exceeds the 8 MiB size limit.")
    _atomic_write_text(target, text)
    return omitted


def _schema_object(pairs):
    """Reject ambiguous duplicate JSON keys rather than silently selecting one."""
    result = {}
    for key, value in pairs:
        if key in result:
            raise AnnotationError(f"The annotation schema repeats the JSON key {key!r}.")
        result[key] = value
    return result


def _schema_constant(value):
    raise AnnotationError(f"The annotation schema contains a nonfinite value: {value}.")


def _load_schema(path, frame, source):
    """Read a bounded portable recipe and preview it against this table snapshot."""
    target = Path(path).expanduser()
    if not target.is_file():
        raise AnnotationError("Choose an annotation schema JSON file.")
    with target.open("rb") as handle:
        raw = handle.read(_SCHEMA_MAX_BYTES + 1)
    if len(raw) > _SCHEMA_MAX_BYTES:
        raise AnnotationError("The annotation schema exceeds the 8 MiB size limit.")
    try:
        payload = json.loads(raw.decode("utf-8-sig"), object_pairs_hook=_schema_object,
                             parse_constant=_schema_constant)
    except (UnicodeError, ValueError, RecursionError) as exc:
        raise AnnotationError(f"Could not read the annotation schema: {exc}") from exc
    if (not isinstance(payload, dict)
            or set(payload) != {"format", "version", "recipe_version", "manual_rows", "columns"}
            or payload.get("format") != "spacr.annotation-schema"
            or type(payload.get("version")) is not int or payload["version"] != 1
            or type(payload.get("recipe_version")) is not int
            or payload["recipe_version"] not in (1, 2, 3)
            or payload.get("manual_rows") != "excluded"):
        raise AnnotationError("Unsupported annotation schema format or version.")
    version = payload["recipe_version"]
    columns, _omitted = _schema_columns(payload["columns"], importing=True, version=version)
    if version == 1 and (len(columns) != 1 or columns[0]["kind"] != "rules"):
        raise AnnotationError("A legacy annotation schema must contain one rules column.")
    if version < 3 and any(entry["kind"] in ("extract", "template")
                           or any("criteria" in rule for rule in entry.get("conditions", []))
                           for entry in columns):
        raise AnnotationError("This annotation schema needs recipe version 3.")
    definition = new_definition(frame, source)
    if version == 1:
        definition.update(column=columns[0]["column"], conditions=columns[0].get("conditions", []))
    else:
        definition.pop("column")
        definition.pop("conditions")
        definition.update(version=version, columns=columns)
    return definition, preview(frame, definition, source)


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
