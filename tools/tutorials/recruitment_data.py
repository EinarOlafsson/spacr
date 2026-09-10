"""Prepare an isolated Recruitment tutorial project without running spaCR.

Only selected rows are streamed through SQLite; no pandas or image decoding is
used. The source must be quiescent, with no uncheckpointed WAL. Measurements,
including original paths and biological annotations, are never rewritten.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from contextlib import closing
from pathlib import Path
import shutil
import sqlite3
import tempfile


DEFAULT_FIELDS = (
    "plate1_B01_3_1",
    "plate1_N01_3_1",
    "plate1_O02_3_1",
    "plate1_J03_2_1",
)
TABLES = ("cell", "nucleus", "pathogen", "cytoplasm")
LOCATION = ("plateID", "rowID", "columnID", "fieldID")
IDENTITY = LOCATION + ("object_label",)


def _quote(name):
    return '"' + name.replace('"', '""') + '"'


def _positive_integral_id(value):
    """Accept SQLite integer/REAL identities without changing their values."""
    return ((type(value) is int and value > 0)
            or (type(value) is float and math.isfinite(value)
                and value > 0 and value.is_integer()))


def _signature(path):
    stat = path.stat()
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns,
            stat.st_ctime_ns)


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _row_hash(row):
    def encode_blob(value):
        if isinstance(value, bytes):
            return {"sqlite_blob_hex": value.hex()}
        raise TypeError(f"Unsupported SQLite value: {type(value).__name__}")
    payload = json.dumps(row, ensure_ascii=False, separators=(",", ":"),
                         default=encode_blob).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _rows_hash(hashes):
    return hashlib.sha256("".join(hashes[key] for key in sorted(hashes))
                          .encode("ascii")).hexdigest()


def _no_pending_wal(database):
    wal = Path(str(database) + "-wal")
    if wal.exists() and wal.stat().st_size:
        raise ValueError("Source has a nonempty WAL; use a quiescent, "
                         "checkpointed source before preparing a subset")


def _inspect_rows(connection, table, fields, max_rows):
    columns = [row[1] for row in connection.execute(
        f"PRAGMA table_info({_quote(table)})")]
    required = set(IDENTITY + ("file_name",))
    if table in ("nucleus", "pathogen"):
        required.add("cell_id")
    missing = required.difference(columns)
    if missing:
        raise ValueError(f"{table} is missing identity columns: {sorted(missing)}")
    placeholders = ",".join("?" for _ in fields)
    query = (f"SELECT {','.join(map(_quote, columns))} FROM {_quote(table)} "
             f"WHERE file_name IN ({placeholders}) LIMIT ?")
    # The limit rejects oversized selections; it never silently truncates one.
    rows = connection.execute(query, (*fields, max_rows + 1))
    return columns, rows


def _validate_identity(table, record, field_locations, seen):
    location = tuple(record[key] for key in LOCATION)
    if any(not isinstance(value, str) or not value for value in location):
        raise ValueError(f"{table} has an incomplete location identity")
    label = record["object_label"]
    if not _positive_integral_id(label):
        raise ValueError(f"{table} has an invalid object_label")
    key = location + (label,)
    if key in seen:
        raise ValueError(f"Duplicate {table} object identity: {key}")
    stem = record["file_name"]
    if stem in field_locations and field_locations[stem] != location:
        raise ValueError(f"Conflicting field identity for {stem}")
    if any(other != stem and loc == location
           for other, loc in field_locations.items()):
        raise ValueError(f"Multiple filenames identify the same field: {location}")
    field_locations[stem] = location
    if "prcf" in record and record["prcf"] != "_".join(location):
        raise ValueError(f"Inconsistent prcf for {table} object {key}")
    return key


def prepare_subset(source_project, destination, fields=DEFAULT_FIELDS,
                   *, max_rows_per_table=10_000):
    """Create a new private project and return its provenance manifest.

    ``fields`` contains one to four filename stems without ``.npy``. All four
    measurement tables must cover every selected field. Object identities are
    unique within each table; cytoplasm and child links must refer to cells in
    the same plate, row, column and field. Every selected cell must have all
    three associated compartments. Positive integral REAL IDs are accepted
    without coercing their stored values. Numeric assay validity and image/mask
    agreement remain the recorder's responsibility.

    The existing destination, its symlink aliases, and all paths inside the
    source project are refused. Its parent directory must already exist. Work
    is staged beside the destination and published only after validation; an
    exclusive mkdir prevents concurrent preparations from overwriting it.

    A source database hash is computed once. Nonempty WALs are rejected rather
    than ignored, and source file signatures are checked for concurrent edits.
    Original SQL table definitions, indexes, triggers and selected column
    values are preserved. The manifest is also written to provenance.json.
    """
    source = Path(source_project).resolve(strict=True)
    requested_destination = Path(destination).absolute()
    if os.path.lexists(requested_destination):
        raise FileExistsError(f"Destination already exists: {requested_destination}")
    destination = requested_destination.resolve()
    if destination == source or source in destination.parents:
        raise ValueError("Destination must be outside the source project")
    if not destination.parent.is_dir():
        raise FileNotFoundError(f"Destination parent does not exist: {destination.parent}")
    if isinstance(fields, str):
        raise ValueError("fields must be a sequence of filename stems")
    fields = tuple(fields)
    if not 1 <= len(fields) <= 4 or len(set(fields)) != len(fields):
        raise ValueError("Select one to four distinct field stems")
    for field in fields:
        if (not isinstance(field, str) or not field or field in (".", "..")
                or "/" in field or "\\" in field or "\x00" in field
                or field.endswith(".npy")):
            raise ValueError(f"Invalid field stem: {field!r}")
    if not isinstance(max_rows_per_table, int) or max_rows_per_table < 1:
        raise ValueError("max_rows_per_table must be a positive integer")

    database = source / "measurements" / "measurements.db"
    if not database.is_file():
        raise FileNotFoundError(database)
    arrays = {field: source / "merged" / (field + ".npy") for field in fields}
    for array in arrays.values():
        if not array.is_file():
            raise FileNotFoundError(array)
    _no_pending_wal(database)
    original_signature = _signature(database)
    # immutable avoids creating or changing SQLite sidecars in the archive;
    # the WAL and signature checks enforce the required stable source.
    uri = database.as_uri() + "?mode=ro&immutable=1"
    connection = sqlite3.connect(uri, uri=True)
    try:
        connection.execute("PRAGMA query_only=ON")
        connection.execute("PRAGMA cache_size=-2048")
        connection.execute("BEGIN")
        schemas = {}
        schema_objects = []
        for table in TABLES:
            row = connection.execute(
                "SELECT sql FROM sqlite_schema WHERE type='table' AND name=?",
                (table,)).fetchone()
            if row is None or row[0] is None:
                raise ValueError(f"Missing measurement table: {table}")
            schemas[table] = row[0]
            schema_objects.extend(connection.execute(
                "SELECT type,name,sql FROM sqlite_schema WHERE tbl_name=? "
                "AND type IN ('index','trigger') AND sql IS NOT NULL "
                "ORDER BY type,name", (table,)).fetchall())

        field_locations = {}
        identities = {}
        hashes_by_table = {}
        manifest = {
            "kind": "recruitment tutorial subset",
            "source_project": str(source),
            "destination": str(destination),
            "source_database": str(database),
            "fields": list(fields),
            "tables": {},
            "arrays": [],
        }
        with tempfile.TemporaryDirectory(
                prefix=".recruitment-subset-", dir=destination.parent) as temporary:
            stage = Path(temporary)
            (stage / "measurements").mkdir()
            (stage / "merged").mkdir()
            staged_database = stage / "measurements" / "measurements.db"
            with closing(sqlite3.connect(staged_database)) as output, output:
                for table in TABLES:
                    output.execute(schemas[table])
                    columns, rows = _inspect_rows(
                        connection, table, fields, max_rows_per_table)
                    insert = (f"INSERT INTO {_quote(table)} "
                              f"({','.join(map(_quote, columns))}) VALUES "
                              f"({','.join('?' for _ in columns)})")
                    hashes = {}
                    records = []
                    per_field = dict.fromkeys(fields, 0)
                    for row in rows:
                        if len(hashes) >= max_rows_per_table:
                            raise ValueError(f"{table} exceeds max_rows_per_table")
                        record = dict(zip(columns, row))
                        key = _validate_identity(table, record, field_locations, hashes)
                        hashes[key] = _row_hash(row)
                        identity = {name: record[name] for name in IDENTITY}
                        identity["file_name"] = record["file_name"]
                        if table in ("nucleus", "pathogen"):
                            parent = record["cell_id"]
                            if not _positive_integral_id(parent):
                                raise ValueError(f"{table} has an invalid cell_id")
                            identity["cell_id"] = parent
                        records.append(identity)
                        per_field[record["file_name"]] += 1
                        output.execute(insert, row)
                    if not all(per_field.values()):
                        raise ValueError(f"{table} has missing selected fields: {per_field}")
                    identities[table] = records
                    hashes_by_table[table] = hashes
                    manifest["tables"][table] = {
                        "row_count": len(records), "per_field": per_field,
                        "schema_sql": schemas[table],
                        "rows_sha256": _rows_hash(hashes),
                        "identities": sorted(records, key=lambda r: tuple(r[k] for k in IDENTITY)),
                    }

                cell_keys = set(hashes_by_table["cell"])
                if set(hashes_by_table["cytoplasm"]) != cell_keys:
                    raise ValueError("Cytoplasm identities do not match selected cells")
                for table in ("nucleus", "pathogen"):
                    linked = set()
                    for identity in identities[table]:
                        parent_key = tuple(identity[k] for k in LOCATION) + (identity["cell_id"],)
                        if parent_key not in cell_keys:
                            raise ValueError(f"Broken {table} host-cell link: {parent_key}")
                        linked.add(parent_key)
                    if not cell_keys.issubset(linked):
                        raise ValueError(f"Selected cells are missing {table} links")

                for _, _, sql in schema_objects:
                    output.execute(sql)
                for table in TABLES:
                    columns, rows = _inspect_rows(output, table, fields, max_rows_per_table)
                    copied = {}
                    for row in rows:
                        record = dict(zip(columns, row))
                        copied[tuple(record[k] for k in IDENTITY)] = _row_hash(row)
                    if copied != hashes_by_table[table]:
                        raise ValueError(f"Copied {table} rows differ from the source")

            manifest["schema_objects"] = [
                {"type": kind, "name": name, "sql": sql}
                for kind, name, sql in schema_objects]
            manifest["source_database_sha256"] = _sha256(database)
            manifest["subset_database_sha256"] = _sha256(staged_database)
            for field, source_array in arrays.items():
                before = _signature(source_array)
                target = stage / "merged" / source_array.name
                digest = hashlib.sha256()
                with source_array.open("rb") as reader, target.open("xb") as writer:
                    for chunk in iter(lambda: reader.read(1024 * 1024), b""):
                        writer.write(chunk)
                        digest.update(chunk)
                array_hash = digest.hexdigest()
                if _signature(source_array) != before or _sha256(target) != array_hash:
                    raise ValueError(f"Source array changed or copy failed: {source_array}")
                manifest["arrays"].append({
                    "source": str(source_array),
                    "destination": str(destination / "merged" / target.name),
                    "sha256": array_hash, "bytes": before[2],
                    "file_name": field,
                    "identity": dict(zip(LOCATION, field_locations[field])),
                })
            _no_pending_wal(database)
            if _signature(database) != original_signature:
                raise ValueError("Source database changed while preparing the subset")
            with (stage / "provenance.json").open("x", encoding="utf-8") as handle:
                json.dump(manifest, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
            destination.mkdir(exist_ok=False)
            try:
                for child in stage.iterdir():
                    child.rename(destination / child.name)
            except BaseException:
                # Only this call's exclusively created destination is removed.
                shutil.rmtree(destination)
                raise
        return manifest
    finally:
        connection.close()
