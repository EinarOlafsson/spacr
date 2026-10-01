"""Recover original image names on measurement rows without changing sources.

Accept spaCR conversion records, older Yokogawa rename logs, and records
produced by sorting image channels. Group records from image channels and
z slices by microscope field. Keep timepoints and plates separate.

Do not read image data. Read an embedded ``conversion_map`` table without
modifying it.
"""
from __future__ import annotations

import hashlib
import io
import re
import sqlite3
from collections import defaultdict
from contextlib import closing
from pathlib import Path

import pandas as pd

from . import schema

MAP_NAMES = ("conversion_map.csv", "rename_log.csv", "channel_sorting_manifest.csv")
_YOKOGAWA = re.compile(
    r"^(?P<plate>.+)_(?P<well>[A-Za-z]{1,2}\d+)_T(?P<time>\d+)"
    r"F(?P<field>\d+)L\d+(?:A\d+)?(?:Z\d+)?C\d+(?:\.tiff?)?$",
    re.IGNORECASE,
)


def discover_maps(db_path) -> list[Path]:
    """Find known map names beside a database and up to three parent folders.

    Discovery is bounded and nonrecursive, including the usual
    ``plate/measurements/measurements.db`` layout. It never scans image trees.
    A SQLite database containing a conversion_map table is also offered.

    :param db_path: measurement database path or dataset directory.
    :returns: existing CSV paths, nearest folder first, followed by the
        database itself when it contains a conversion_map table.
    """
    path = Path(db_path).expanduser().resolve()
    folder = path if path.is_dir() else path.parent
    folders = [folder, *list(folder.parents)[:3]]
    found = [candidate for parent in folders for name in MAP_NAMES
             if (candidate := parent / name).is_file()]
    if path.is_file() and path.suffix.lower() in (".db", ".sqlite", ".sqlite3"):
        try:
            with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as connection:
                exists = connection.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name='conversion_map'"
                ).fetchone()
            if exists:
                found.append(path)
        except sqlite3.Error:
            pass
    return found


def _text(value) -> str:
    """Normalize absent scalar metadata without inventing a filename."""
    return "" if value is None or pd.isna(value) else str(value).strip()


def _identity_token(value) -> str:
    """Retain integer identities promoted to floats by nullable numeric columns."""
    text = _text(value)
    return text.split(".")[0] if re.fullmatch(r"[+-]?\d+\.0+", text) else text


def _basename(value) -> str:
    """Read Windows provenance paths on Linux as well as native paths."""
    return _text(value).replace("\\", "/").rsplit("/", 1)[-1]


def _key(field):
    """Canonical plate/well/field/time tuple, never a plate-free join key."""
    return (field.plateID, schema.row_id(field.rowID),
            schema.column_id(field.columnID), schema.field_id(field.fieldID),
            schema.time_id(field.timeID) if field.timeID else None)


def _filename_key(value):
    """Parse converted Yokogawa or spaCR's actual merged-stack filenames."""
    name = _basename(value)
    match = _YOKOGAWA.fullmatch(name)
    if match:
        return _key(schema.FieldID.build(**match.groupdict(), strict=True))
    for timelapse in (True, False):
        try:
            return _key(schema.parse_field_stem(name, timelapse=timelapse, strict=True))
        except schema.SchemaError:
            pass
    return None


def _with_row_time(key, row, prefix):
    """Refine an untimed identity with either supported time-column spelling."""
    times = {_identity_token(row.get(prefix + alias))
             for alias in schema.TIME_COLUMN_ALIASES
             if _text(row.get(prefix + alias))}
    if not times and prefix:
        times = {_identity_token(row.get(alias)) for alias in schema.TIME_COLUMN_ALIASES
                 if _text(row.get(alias))}
    normalized = {schema.time_id(time, strict=True) for time in times}
    if len(normalized) > 1:
        raise ValueError("Conflicting timeID and time_id identities in measurement row")
    time = next(iter(normalized), key[4])
    if key[4] and time and key[4] != time:
        raise ValueError("Conflicting time identity in measurement row")
    return (*key[:4], time)


def _metadata_keys(row):
    """Read canonical identities, including measurement-role-prefixed columns."""
    keys = []
    for name, value in row.items():
        if name == "prcf" or name.endswith("_prcf"):
            if _text(value):
                try:
                    keys.append(_with_row_time(_key(schema.parse_prcf(value)), row, name[:-4]))
                except schema.SchemaError:
                    pass
        if name == "prcfo" or name.endswith("_prcfo"):
            if _text(value):
                try:
                    keys.append(_with_row_time(_key(schema.parse_prcfo(value)), row, name[:-5]))
                except schema.SchemaError:
                    pass
    prefixes = {name[:-7] for name in row if name.endswith("plateID")}
    for prefix in prefixes:
        values = [(_text if part == "plateID" else _identity_token)(row.get(prefix + part)) for part in
                  ("plateID", "rowID", "columnID", "fieldID")]
        if all(values):
            try:
                key = _key(schema.FieldID.build(
                    values[0], row=values[1], column=values[2], field=values[3], strict=True))
                keys.append(_with_row_time(key, row, prefix))
            except schema.SchemaError:
                pass
    return list(dict.fromkeys(keys))


def _load_map(payload, path):
    """Validate one known CSV shape and retain only successful image records."""
    try:
        frame = pd.read_csv(io.BytesIO(payload), dtype=str, keep_default_na=False)
    except Exception as error:
        raise ValueError(f"Cannot read filename mapping {path}: {error}") from error
    columns = set(frame.columns)
    if {"target", "source", "plate", "well", "field", "channel", "z", "t"} <= columns:
        kind, target, source = "conversion_map", "target", "source"
        if "status" in columns:
            frame = frame[frame.status.isin(("converted", "existing"))]
    elif "Renamed TIFF" in columns and columns.intersection(("Original File", "Original File(s)")):
        kind, target = "rename_log", "Renamed TIFF"
        source = "Original File(s)" if "Original File(s)" in columns else "Original File"
    elif {"kind", "original_path", "new_path", "status", "plate", "well", "field", "time"} <= columns:
        kind, target, source = "channel_sorting_manifest", "new_path", "original_path"
        frame = frame[(frame.kind == "image") & (
            (frame.status == "moved") | frame.status.str.startswith("converted "))]
    else:
        raise ValueError("Not a supported conversion_map.csv, rename_log.csv or channel-sorting manifest")
    entries = []
    for values in frame.itertuples(index=False, name=None):
        row = dict(zip(frame.columns, values))
        destination, original = _text(row[target]), _text(row[source])
        if not destination or not original:
            continue
        key = _filename_key(destination)
        if kind != "rename_log":
            try:
                declared = _key(schema.FieldID.build(
                    row["plate"], well=row["well"], field=_identity_token(row["field"]),
                    time=_identity_token(row["t" if kind == "conversion_map" else "time"]), strict=True))
            except schema.SchemaError as error:
                raise ValueError(f"Invalid field identity in {path}: {error}") from error
            if key is not None and key != declared:
                raise ValueError(f"Conflicting target and plate/field identity for {destination}")
            key = declared
        originals = (original.split(";") if source == "Original File(s)" else [original])
        originals = {_text(item) for item in originals if _text(item)}
        if originals:
            entries.append((destination, key, originals))
    if not entries:
        raise ValueError("The mapping contains no successful image conversions with original names")
    return kind, entries, len(frame)


def enrich(frame, map_path, *, expected_sha256=None, output_column="original_filename"):
    """Return an enriched copy and JSON-safe matching/provenance report.

    Rows, index, order and existing columns are preserved. Multiple original
    images for one field are deduplicated and sorted, joined with ``'; '``.
    Unmatched values are missing. Existing output columns are never replaced.
    Ambiguous plate or timepoint matches raise ValueError, as do changed maps
    when expected_sha256 binds a saved workflow to its reviewed mapping bytes.
    ``original_path`` preserves recorded paths, not a promise they still exist.

    :param frame: current measurement DataFrame; never modified.
    :param map_path: supported CSV or SQLite database with conversion_map.
    :param expected_sha256: optional digest from an earlier preview. CSV
        digests bind the exact parsed bytes; SQLite digests bind sorted
        mapping contents independently of unrelated database writes.
    :param output_column: new filename column name; original_path is also added.
    :returns: (enriched DataFrame, JSON-safe report) with mapping provenance,
        matched/unmatched row counts and up to ten unmatched identity examples.
    :raises ValueError: for an unsupported or changed map, output collisions,
        conflicting/ambiguous identities, or zero matches on nonempty data.
    """
    if not isinstance(frame, pd.DataFrame) or not frame.columns.is_unique:
        raise ValueError("Filename restoration needs a DataFrame with unique columns")
    if not isinstance(output_column, str) or not output_column.strip() or output_column == "original_path":
        raise ValueError("Choose a nonempty original filename column distinct from original_path")
    if output_column in frame.columns or "original_path" in frame.columns:
        raise ValueError("Original filename/path output columns already exist; source columns are preserved")
    path = Path(map_path).expanduser().resolve()
    map_table = None
    if path.suffix.lower() in (".db", ".sqlite", ".sqlite3"):
        try:
            with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as connection:
                stored = pd.read_sql_query('SELECT * FROM "conversion_map"', connection)
            # Order-independent digest: unrelated database writes cannot invalidate it.
            stored = stored.fillna("").astype(str)
            stored = stored.reindex(sorted(stored.columns), axis=1)
            stored = stored.sort_values(list(stored.columns), kind="stable")
            payload = stored.to_csv(index=False, lineterminator="\n").encode("utf-8")
            map_table = "conversion_map"
        except (sqlite3.Error, pd.errors.DatabaseError) as error:
            raise ValueError(f"Cannot read a conversion_map table from {path}: {error}") from error
    else:
        payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError("The filename mapping has changed since this workflow was saved; preview it again")
    kind, entries, valid_rows = _load_map(payload, path)
    by_name, by_field, by_untimed = defaultdict(set), defaultdict(set), defaultdict(set)
    originals_by_field = defaultdict(set)
    for target, key, originals in entries:
        # Unparseable legacy names can still be restored by their exact target.
        group = key if key is not None else ("target", target)
        by_name[_basename(target)].add(group)
        by_name[target.replace("\\", "/")].add(group)
        originals_by_field[group].update(originals)
        if key is not None:
            by_field[key].add(group)
            by_untimed[key[:4]].add(group)
    names, paths, unmatched_examples, cache = [], [], [], {}
    identity_columns = [column for column in frame.columns if isinstance(column, str) and (
        column in ("prcf", "prcfo", "filename", "file_name", "path", "image_path")
        or column.endswith(("_prcf", "_prcfo", "_filename", "_file_name"))
        or column.endswith(("plateID", "rowID", "columnID", "fieldID", "timeID", "time_id")))]
    if not identity_columns:
        raise ValueError("No measurement filenames, prcf or plate/row/column/field identity columns found")
    for values in frame[identity_columns].itertuples(index=False, name=None):
        signature = tuple(_text(value) for value in values)
        if signature not in cache:
            row = dict(zip(identity_columns, signature))
            keys = _metadata_keys(row)
            candidates = []
            for column, value in row.items():
                if column in ("filename", "file_name", "path", "image_path") or column.endswith(("_filename", "_file_name")):
                    if not value:
                        continue
                    parsed = _filename_key(value)
                    if parsed is not None:
                        keys.append(parsed)
                    found = by_name.get(value.replace("\\", "/")) or by_name.get(_basename(value))
                    if found:
                        candidates.append(set(found))
            # A timed filename refines an untimed prcf without merging timepoints.
            if len({key[:4] for key in keys}) > 1 or len({key[4] for key in keys if key[4]}) > 1:
                raise ValueError("Conflicting filename and plate/field identities in measurement row")
            for key in keys:
                found = by_field.get(key) if key[4] is not None else by_untimed.get(key[:4])
                candidates.append(set(found or ()))
            matched = set.intersection(*candidates) if candidates else set()
            if any(candidates) and not matched:
                raise ValueError("Conflicting filename and plate/field identities in measurement row")
            if len(matched) > 1:
                raise ValueError("Ambiguous original filenames across plates or timepoints; retain full field/time identity")
            if matched:
                original_paths = sorted(originals_by_field[next(iter(matched))])
                original_names = sorted({_basename(item) for item in original_paths})
                cache[signature] = ("; ".join(original_names), "; ".join(original_paths))
            else:
                cache[signature] = (pd.NA, pd.NA)
                if len(unmatched_examples) < 10:
                    unmatched_examples.append(dict(row))
        name, original_path = cache[signature]
        names.append(name)
        paths.append(original_path)
    result = frame.copy()
    result[output_column] = pd.array(names, dtype="string")
    result["original_path"] = pd.array(paths, dtype="string")
    matched_rows = int(result[output_column].notna().sum())
    if len(frame) and not matched_rows:
        raise ValueError("No measurement rows matched this filename mapping; check plate and field identities")
    report = {"map_path": str(path), "sha256": digest, "format": kind, "map_table": map_table,
              "matched_rows": matched_rows, "unmatched_rows": len(frame) - matched_rows,
              "mapping_rows": valid_rows, "output_column": output_column,
              "path_column": "original_path", "separator": "; ",
              "unmatched_examples": unmatched_examples}
    return result, report
