"""Recreate tables from saved definitions without changing input data.

Use the same derived tables for charts and measurement filters.

Default definitions delegate to :mod:`spacr.merge_tables`, the measurement
aggregation used by Regression. Custom definitions describe explicit joins
onto unique base observations; children are aggregated before joining. JSON
sidecars preserve definitions without altering source tables. A definition is
bound to a canonical database path and the selected tables' column schemas.
"""
from __future__ import annotations

import copy
import json
import os
import re
import sqlite3
import tempfile
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from .merge_tables import (
    IDENTITY,
    OBJECT_COLUMN,
    MergeError,
    MergePolicy,
    _align_keys,
    _read,
    aggregation_for,
    merge_tables,
    roll_up,
)
from .object_roles import ANCHOR_COLUMN, anchor_column, is_one_row_per_cell

METHODS = ("mean", "median", "sum", "min", "max", "count", "first", "last",
           "nunique", "any", "all")



def schemas(path):
    """Discover user tables and declared column types without changing them.

    :param path: SQLite database path.
    :returns: Table names mapped to lists of column-name/type pairs.
    """
    with sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True, timeout=30.0) as db:
        names = [r[0] for r in db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name NOT LIKE 'sqlite_%' ORDER BY name")]
        return {name: [[r[1], r[2]] for r in db.execute(
            'PRAGMA table_info("' + name.replace('"', '""') + '")')]
                for name in names}


def column_sample(path, table, limit=200):
    """Read a bounded sample for controls; execution always validates all rows.

    :param path: Database path.
    :param table: Source table name.
    :param limit: Maximum sampled rows.
    :returns: Sample frame preserving SQLite column types when possible.
    """
    with sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True, timeout=30.0) as db:
        return pd.read_sql_query('SELECT * FROM "' + table.replace('"', '""') +
                                 '" LIMIT ?', db, params=(int(limit),))


def identity_columns(frame):
    """Return image and time provenance actually present in a frame.

    :param frame: An object measurement frame.
    :returns: Ordered identity columns, excluding the object label.
    """
    return [c for c in (*IDENTITY, "timeID", "time_id") if c in frame]


def _identifiers(columns):
    """Recognize identifier columns that must never be averaged or summed.

    :param columns: Source column names.
    """
    return [c for c in columns if re.search(
        r"(^|_)(id|label|object_label|prc|prcf|prcfo)$|ID$", str(c))]


def default_definition(path, tables, *, name="Merged measurements", base=None,
                       policy=None):
    """Describe the established spaCR merge, with no inferred external keys.

    :param path: SQLite data source.
    :param tables: Tables to combine, including the output observation table.
    :param name: Display name for this derived table.
    :param base: Output table; defaults to cell, cytoplasm, or the first table.
    :param policy: Optional existing ``MergePolicy`` to retain user choices.
    :returns: JSON-serializable definition; execution validates the schema.
    """
    available = schemas(path)
    tables = list(dict.fromkeys(tables))
    base = base or next((t for t in ("cell", "cytoplasm") if t in tables),
                        tables[0] if tables else "cell")
    if base not in tables:
        tables.insert(0, base)
    missing = set(tables) - set(available)
    if missing:
        raise MergeError("Missing tables: " + ", ".join(sorted(missing)))
    policy = policy or MergePolicy(primary=base)
    base_columns = [c[0] for c in available[base]]
    keys = [c for c in (*IDENTITY, "timeID", "time_id") if c in base_columns]
    joins = []
    for table in tables:
        if table == base:
            continue
        link = anchor_column(table) if table in ANCHOR_COLUMN else ""
        joins.append({"table": table, "left_keys": keys + [OBJECT_COLUMN],
                      "right_keys": keys + ([link] if link else []),
                      "relationship": "one-to-one" if is_one_row_per_cell(table)
                                      else "one-to-many",
                      "how": policy.how_for(table) if table in ANCHOR_COLUMN else "left",
                      "overrides": {c.split(".", 1)[-1]: v for c, v in policy.overrides.items()
                                    if "." not in c or c.startswith(table + ".")},
                      "identifiers": _identifiers(
                          [c[0] for c in available[table]])})
    return {"version": 1, "source": str(Path(path).resolve()), "name": name,
            "schema": {t: available[t] for t in tables}, "mode": "default",
            "base": base, "base_keys": keys + [OBJECT_COLUMN], "joins": joins,
            "policy": asdict(policy), "acknowledged": False}


def validate_source(path, definition):
    """Refuse reuse on another database or a changed selected-table schema.

    :param path: Database being used now.
    :param definition: Saved derived-table definition.
    :raises MergeError: Source or schema no longer matches.
    """
    if definition.get("version") != 1:
        raise MergeError("Unsupported merge definition version; recreate this merge.")
    if str(Path(path).resolve()) != definition.get("source"):
        raise MergeError("This merge belongs to another database; recreate its mapping here.")
    current = schemas(path)
    selected = [definition["base"]] + [j["table"] for j in definition["joins"]]
    if len(set(selected)) != len(selected):
        raise MergeError("Each source table may be included only once.")
    for table in selected:
        if table not in current or current[table] != definition.get("schema", {}).get(table):
            raise MergeError(f"Schema changed for {table}; reopen Customize merging and validate its mapping.")


def allowed_methods(series, *, identifier=False):
    """List aggregations compatible with a column's actual values.

    :param series: Source column values.
    :param identifier: Whether these values identify observations.
    :returns: Supported deterministic method names.
    """
    if identifier:
        return ("first", "last", "count", "nunique")
    if pd.api.types.is_bool_dtype(series) or set(series.dropna().unique()).issubset({0, 1}):
        return METHODS
    if pd.api.types.is_numeric_dtype(series):
        return tuple(m for m in METHODS if m not in ("any", "all"))
    return ("first", "last", "count", "nunique", "min", "max")


def _require_keys(frame, keys, table, *, unique=False):
    """Require explicit, complete keys and optionally unique base observations.

    :param frame: Source table data.
    :param keys: Explicit composite identity columns.
    :param table: Table name for actionable errors.
    :param unique: Require complete, unique base observation identities.
    """
    if not keys or len(set(keys)) != len(keys):
        raise MergeError(f"{table}: choose a nonempty list of distinct join columns.")
    missing = set(keys) - set(frame)
    if missing:
        raise MergeError(f"{table}: missing keys {', '.join(sorted(missing))}. "
                         "Use Customize merging to map the actual columns.")
    if unique and frame[keys].isna().any(axis=None):
        raise MergeError(f"{table}: base observation keys contain missing values.")
    if unique and frame.duplicated(keys).any():
        n = int(frame.duplicated(keys, keep=False).sum())
        raise MergeError(f"{table}: {n} rows have duplicate observation keys {keys}; "
                         "include the image/time columns in the composite key.")


def _align_explicit_keys(left, right, keys):
    """Align explicit external keys without interpreting their values as spaCR IDs.

    :param left: Left frame, whose key types are aligned in place.
    :param right: Right frame, whose key types are aligned in place.
    :param keys: Shared key names after explicit mapping.
    """
    for key in keys:
        if not (pd.api.types.is_numeric_dtype(left[key]) and
                pd.api.types.is_numeric_dtype(right[key])):
            left[key] = left[key].astype("string")
            right[key] = right[key].astype("string")


def _join_diagnostics(base, child, join, *, standard=False):
    """Validate key cardinality and count matched and unmatched source rows.

    :param base: Unique base observation frame.
    :param child: Unaggregated source child frame.
    :param join: Explicit key, relationship and join-type configuration.
    :param standard: Normalize spaCR object IDs only for established default schemas.
    """
    left_keys, right_keys = join["left_keys"], join["right_keys"]
    table = join["table"]
    _require_keys(base, left_keys, "base", unique=True)
    _require_keys(child, right_keys, table)
    if len(left_keys) != len(right_keys):
        raise MergeError(f"{table}: left and right key lists must have the same length.")
    valid = child.dropna(subset=right_keys).copy()
    # A child may carry its own object_label while cell_id maps to the
    # parent's object_label. Retain that separate label with its table prefix.
    collision_names = set(left_keys) - set(right_keys)
    valid = valid.rename(columns={c: table + "_" + c for c in collision_names if c in valid})
    renamed = valid.rename(columns=dict(zip(right_keys, left_keys)))
    if len(set(renamed.columns)) != len(renamed.columns):
        raise MergeError(f"{table}: key mapping collides with another column; use distinct key names.")
    left = base[left_keys].copy()
    if standard:
        _align_keys(left, renamed, left_keys)
    else:
        _align_explicit_keys(left, renamed, left_keys)
    duplicates = int(renamed.duplicated(left_keys, keep=False).sum())
    if join["relationship"] not in ("one-to-one", "one-to-many"):
        raise MergeError(f"{table}: choose one-to-one or one-to-many.")
    if join["relationship"] == "one-to-one" and duplicates:
        raise MergeError(f"{table}: {duplicates} rows violate one-to-one cardinality; "
                         "choose one-to-many aggregation or correct the keys.")
    if join["how"] not in ("left", "inner"):
        raise MergeError(f"{table}: supported join types are left and inner.")
    distinct = renamed[left_keys].drop_duplicates()
    unmatched_base = len(left.merge(distinct, on=left_keys, how="left", indicator=True)
                         .query('_merge == "left_only"'))
    unmatched_child = len(renamed[left_keys].merge(left, on=left_keys, how="left", indicator=True)
                          .query('_merge == "left_only"'))
    return {"table": table, "left_keys": left_keys, "right_keys": right_keys,
            "relationship": join["relationship"], "how": join["how"],
            "input_rows": len(child), "groups": len(distinct),
            "duplicate_key_rows": duplicates, "missing_key_rows": len(child) - len(valid),
            "unmatched_base": unmatched_base, "unmatched_child": unmatched_child}, renamed


def execute(path, definition):
    """Revalidate and materialize a merge without writing source tables.

    :param path: Source SQLite database.
    :param definition: Default or acknowledged custom merge configuration.
    :returns: ``(frame, diagnostics)`` with full counts and actual aggregation rules.
    :raises MergeError: Missing keys, changed schemas, cardinality or type violations.
    """
    validate_source(path, definition)
    definition = copy.deepcopy(definition)
    mode = definition.get("mode")
    if mode not in ("default", "custom", "metadata"):
        raise MergeError("Choose default or custom merging.")
    if mode == "custom" and not definition.get("acknowledged"):
        raise MergeError("Acknowledge the custom merging warning before previewing or applying.")
    base_name = definition["base"]
    base = _read(path, base_name)
    if mode == "metadata":
        if definition["joins"] or not definition.get("original_filenames"):
            raise MergeError("Original filename enrichment requires one source table and a conversion map.")
        output, metadata = _restore_original_filenames(base, definition)
        if "time_id" in output and "timeID" not in output:
            output["timeID"] = output["time_id"]
        keys = list(IDENTITY) + [OBJECT_COLUMN] + (["timeID"] if "timeID" in output else [])
        provenance = (base_name in ANCHOR_COLUMN and set(keys).issubset(output.columns)
                      and not output.duplicated(keys).any()
                      and not output[keys].isna().any(axis=None))
        output.attrs["merge_definition"] = definition
        output.attrs["image_provenance"] = bool(provenance)
        return output, {"base": base_name, "base_rows": len(base), "output_rows": len(output),
                        "joins": [], "image_provenance": bool(provenance),
                        "original_filenames": metadata}
    _require_keys(base, definition["base_keys"], base_name, unique=True)
    if mode == "default" and not is_one_row_per_cell(base_name):
        raise MergeError("Default output is one row per cell or cytoplasm. "
                         "Use Customize merging for another observation level.")
    if mode == "default" and not set(IDENTITY).issubset(base.columns):
        raise MergeError("Default merging requires plateID, rowID, columnID and fieldID. "
                         "Use Customize merging for external data without image provenance.")
    # Keep genuine base provenance. External names are never converted into
    # image identities just because they happen to be unique.
    keep = set(definition["base_keys"]) | set(IDENTITY) | {OBJECT_COLUMN, "prcf", "prcfo", "timeID", "time_id"}
    keep.update(c for join in definition["joins"] for c in join["left_keys"])
    rename = {c: (c if c.startswith(base_name + "_") else base_name + "_" + c)
              for c in base if c not in keep}
    output = base.rename(columns=rename)
    reports = []
    for join in definition["joins"]:
        table = join["table"]
        if mode == "default" and table not in ANCHOR_COLUMN:
            raise MergeError(f"{table}: no standard spaCR relationship; use Customize merging.")
        child = _read(path, table)
        report, aligned = _join_diagnostics(base, child, join, standard=mode == "default")
        keys = join["left_keys"]
        identifiers = set(join.get("identifiers", ())) | set(_identifiers(aligned.columns))
        overrides = join.get("overrides", {})
        if mode == "default":
            policy_overrides = definition.get("policy", {}).get("overrides", {})
            overrides = {c.split(".", 1)[-1]: m for c, m in policy_overrides.items()
                         if "." not in c or c.startswith(table + ".")}
            overrides = {c: m for c, m in overrides.items() if c in aligned and c not in keys}
        plans = {}
        for column in aligned:
            if column in keys:
                continue
            method = overrides.get(column, aggregation_for(
                column, numeric=pd.api.types.is_numeric_dtype(aligned[column])))
            if column in identifiers and column not in overrides:
                method = "first"
            if method not in allowed_methods(aligned[column], identifier=column in identifiers):
                raise MergeError(f"{table}.{column}: {method} is incompatible with this column; "
                                 f"choose {', '.join(allowed_methods(aligned[column], identifier=column in identifiers))}.")
            plans[column] = method
        unknown = set(overrides) - set(plans)
        if unknown:
            raise MergeError(f"{table}: aggregation overrides name absent/key columns: {sorted(unknown)}")
        report["aggregations"] = plans
        reports.append(report)
        if mode == "default":
            continue
        # One-to-one was validated above. Applying the same typed plan to
        # its singleton groups keeps explicit count/nunique overrides real.
        rolled = roll_up(aligned, keys, name=table,
                         policy=MergePolicy(overrides=plans))
        collisions = (set(rolled) & set(output)) - set(keys)
        if collisions:
            raise MergeError(f"{table}: output column collision {sorted(collisions)}; "
                             "choose tables with unambiguous column names.")
        _align_explicit_keys(output, rolled, keys)
        output = output.merge(rolled, on=keys, how=join["how"], validate="one_to_one")
    if mode == "default":
        policy = MergePolicy(**definition["policy"])
        output = merge_tables(path, [j["table"] for j in definition["joins"]], policy=policy)
    if len(output) > len(base):
        raise MergeError("Merge increased the base observation count; check the composite keys.")
    # Gate/filter identities use timeID. Preserve the original spelling too
    # so a chart selecting time_id remains reproducible.
    if "time_id" in output and "timeID" not in output:
        output["timeID"] = output["time_id"]
    provenance = (base_name in ANCHOR_COLUMN and
                  set(IDENTITY + (OBJECT_COLUMN,)).issubset(output.columns) and
                  set(IDENTITY + (OBJECT_COLUMN,)).issubset(definition["base_keys"]))
    if provenance:
        link_keys = list(IDENTITY) + [OBJECT_COLUMN] + (["timeID"] if "timeID" in output else [])
        provenance = not output.duplicated(link_keys).any() and not output[link_keys].isna().any(axis=None)
    metadata = None
    if definition.get("original_filenames"):
        output, metadata = _restore_original_filenames(output, definition)
    output.attrs["merge_definition"] = definition
    output.attrs["image_provenance"] = bool(provenance)
    report = {"base": base_name, "base_rows": len(base), "output_rows": len(output),
              "joins": reports, "image_provenance": output.attrs["image_provenance"]}
    if metadata is not None:
        report["original_filenames"] = metadata
    return output, report


def _restore_original_filenames(frame, definition):
    """Attach read-only conversion metadata and bind its content to the recipe."""
    from .original_filenames import enrich

    options = definition["original_filenames"]
    result, report = enrich(frame, options["map_path"],
                            expected_sha256=options.get("sha256"),
                            output_column=options.get("output_column", "original_filename"))
    definition["original_filenames"] = {
        "map_path": report["map_path"], "sha256": report["sha256"],
        "output_column": report["output_column"]}
    return result, report


def sidecar_path(path):
    """Return the source-adjacent definition file without modifying the database.

    :param path: Database path.
    :returns: Path to its JSON derived-table definitions.
    """
    return Path(str(Path(path).resolve()) + ".spacr-merges.json")


def load_definitions(path):
    """Read saved definitions scoped to one database; execution revalidates schemas.

    :param path: Database path.
    :returns: Name-to-definition mapping, empty when no definitions exist.
    """
    sidecar = sidecar_path(path)
    if not sidecar.exists():
        return {}
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    if payload.get("source") != str(Path(path).resolve()):
        raise MergeError("Saved merge definitions belong to a different database.")
    return payload.get("definitions", {})


def save_definition(path, definition):
    """Validate then atomically save a named definition alongside its source.

    :param path: Database path.
    :param definition: Reproducible configuration, including the display name.
    :returns: Saved display name.
    """
    validate_source(path, definition)
    name = str(definition.get("name", "")).strip()
    if not name or name in schemas(path):
        raise MergeError("Give the result a nonempty name different from every source table.")
    definitions = load_definitions(path)
    definitions[name] = definition
    target = sidecar_path(path)
    fd, temporary = tempfile.mkstemp(prefix=target.name + ".", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump({"source": str(Path(path).resolve()), "definitions": definitions},
                      handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return name
