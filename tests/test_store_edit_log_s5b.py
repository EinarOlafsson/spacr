"""Unified-store step S5, second part: the remaining in-place mutators.

Hit attribution's promotions and audits, Graph Builder's ``png_list``
labels and their receipt, active learning's round tables and thresholded
column, and the prediction merges of every older store now go through
:mod:`spacr.store`. These tests pin:

1. old vs new: on an older ``measurements.db`` every ported writer leaves
   the database exactly as the SQL it replaced did (the reference SQL is
   kept here, copied from the writers before the port), and a rerun gives
   the same bytes;
2. atomicity: :func:`spacr.store.transaction` lands a check and its writes
   together or not at all, so an interrupted promotion leaves no half
   state on either kind of project;
3. a store project shows the same results through ``<table>_labeled`` and
   the bookkeeping views;
4. :func:`spacr.store.migrate_legacy` turns ``png_list`` label columns into
   ``source="import"`` edits and keeps the ``.db``;
5. the new public API is registered in the S5 arrival contract.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import types
import uuid

import numpy as np
import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("duckdb")
pytest.importorskip("pyarrow")

from spacr import active_learning as al  # noqa: E402
from spacr import condition_annotations as ca  # noqa: E402
from spacr import hit_attribution as ha  # noqa: E402
from spacr import predictions as pr  # noqa: E402
from spacr import store  # noqa: E402
from tests.test_store_edit_log_s5 import (PATHS, PRCFO, _dump,  # noqa: E402
                                          _labels, _legacy, _legacy_labels,
                                          _project, _twins)

NOW = "2026-11-24T00:00:00+00:00"


def _freeze(monkeypatch):
    """Fixed timestamps and promotion ids from 1, so two runs compare."""
    counter = iter(range(10 ** 6))
    monkeypatch.setattr(ha, "_now", lambda: NOW)
    monkeypatch.setattr(ha, "uuid", types.SimpleNamespace(
        uuid4=lambda: uuid.UUID(int=next(counter) + 1)))


@pytest.fixture
def frozen(monkeypatch):
    """:func:`_freeze` for a whole test."""
    _freeze(monkeypatch)
    return monkeypatch


def _result(calls=(True, False, True, False, False, False, False, False)):
    """A minimal attribution result over the eight crops of the fixtures."""
    cells = pd.DataFrame({
        "prcfo": PRCFO, "hit_like_call": list(calls),
        "hit_like_probability": np.linspace(0.1, 0.9, 8),
        "hit_like_uncertainty": np.linspace(0.2, 0.3, 8),
        "attribution_fold": [i % 2 for i in range(8)],
        "target_guide_fraction": np.full(8, 0.5)})
    return types.SimpleNamespace(
        cells=cells, object_columns=["prcfo"], target_gene="GENE1",
        target_guides=["g1"], feature_columns=["area"],
        well_columns=["plate", "well"], score_column="score",
        direction="positive", threshold=0.5, split_level="well",
        random_seed=0, source_regression_run=None, validation={},
        warnings=[])


def _investigation(run="inv-1", calls=(1, 0, 1)):
    """A minimal investigation bundle over the first three crops."""
    cells = pd.DataFrame({"prcfo": PRCFO[:3],
                          "candidate_probability": [0.9, 0.1, 0.8],
                          "candidate_call": [bool(c) for c in calls],
                          "attribution_fold": [0, 1, 0]})
    context = types.SimpleNamespace(
        regression_results_folder="/r", regression_run_sha256="abc",
        gene="GENE1", phenotype="p", effect=1.0, guides=("g1",), fdr=0.05,
        direction="positive")
    return types.SimpleNamespace(
        cells=cells, context=context, feature_columns=["area"],
        split_level="well", warnings=[], enrichment={},
        attribution_run_id=run)


def _sha(path):
    """sha256 of a file."""
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


# reference SQL, copied from spacr.hit_attribution before the port


def _old_write_hit_attribution(db_path: str, result,
                          *, run_id: Optional[str] = None) -> str:
    """Persist a versioned attribution without touching annotation columns.

    :param db_path: path to the existing SQLite measurements database.
    :param result: attribution result whose run metadata and cell scores will
        be stored.
    """
    path = os.path.abspath(os.path.expanduser(db_path))
    if not os.path.isfile(path):
        raise ha.HitAttributionError(f"no database at {path}")
    run = str(run_id or ha.uuid.uuid4())
    created = ha._now()
    manifest = {
        "target_gene": result.target_gene,
        "target_guides": result.target_guides,
        "feature_columns": result.feature_columns,
        "well_columns": result.well_columns,
        "object_columns": result.object_columns,
        "score_column": result.score_column,
        "direction": result.direction,
        "threshold": result.threshold,
        "split_level": result.split_level,
        "random_seed": result.random_seed,
        "source_regression_run": result.source_regression_run,
        "validation": result.validation,
        "warnings": result.warnings,
    }
    with sqlite3.connect(path, timeout=30) as connection:
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("""
            CREATE TABLE IF NOT EXISTS hit_attribution_runs (
                run_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                target_gene TEXT NOT NULL,
                manifest_json TEXT NOT NULL
            )
        """)
        connection.execute("""
            CREATE TABLE IF NOT EXISTS object_hit_attribution (
                run_id TEXT NOT NULL,
                object_key TEXT NOT NULL,
                probability REAL NOT NULL,
                uncertainty REAL NOT NULL,
                threshold REAL NOT NULL,
                hit_like_call INTEGER NOT NULL,
                attribution_fold INTEGER NOT NULL,
                target_guide_fraction REAL NOT NULL,
                PRIMARY KEY (run_id, object_key),
                FOREIGN KEY (run_id) REFERENCES hit_attribution_runs(run_id)
                    ON DELETE CASCADE
            )
        """)
        connection.execute(
            "INSERT INTO hit_attribution_runs VALUES (?, ?, ?, ?)",
            (run, created, result.target_gene,
             json.dumps(manifest, sort_keys=True, default=str)))
        rows = []
        for _, row in result.cells.iterrows():
            rows.append((
                run, ha._object_key(row, result.object_columns),
                float(row["hit_like_probability"]),
                float(row["hit_like_uncertainty"]), result.threshold,
                int(bool(row["hit_like_call"])), int(row["attribution_fold"]),
                float(row["target_guide_fraction"]),
            ))
        connection.executemany(
            "INSERT INTO object_hit_attribution VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            rows)
    return run




def _old_promote_hit_calls(db_path: str, result, *,
                      run_id: str, annotation_column: str,
                      positive_value: Any = 1) -> str:
    """Explicitly promote positive calls to a fresh, reversible annotation.

    :param db_path: path to the SQLite measurements database to update.
    :param result: attribution result containing the positive object calls.
    :param run_id: attribution-run identifier to record in the audit.
    :param annotation_column: fresh ``png_list`` column to receive the calls.
    """
    column = str(annotation_column)
    if not ha._SQL_NAME.fullmatch(column):
        raise ha.HitAttributionError(
            "annotation column must contain only letters, numbers and underscores")
    if result.object_columns != ["prcfo"]:
        raise ha.HitAttributionError(
            "promotion currently requires prcfo as the exact png_list object key")
    promotion_id = str(ha.uuid.uuid4())
    with sqlite3.connect(db_path, timeout=30) as connection:
        columns = {row[1] for row in connection.execute(
            'PRAGMA table_info("png_list")')}
        if "prcfo" not in columns:
            raise ha.HitAttributionError("png_list has no prcfo object key")
        if column in columns:
            raise ha.HitAttributionError(
                f"annotation column {column!r} already exists; choose a fresh "
                "name so hand annotations cannot be overwritten")
        connection.execute(f'ALTER TABLE "png_list" ADD COLUMN "{column}"')
        connection.execute("""
            CREATE TABLE IF NOT EXISTS hit_attribution_promotions (
                promotion_id TEXT NOT NULL,
                run_id TEXT NOT NULL,
                annotation_column TEXT NOT NULL,
                prcfo TEXT NOT NULL,
                previous_value TEXT,
                promoted_value TEXT,
                created_at TEXT NOT NULL,
                undone_at TEXT,
                PRIMARY KEY (promotion_id, prcfo)
            )
        """)
        selected = result.cells[result.cells["hit_like_call"]]
        for value in selected["prcfo"].astype(str):
            connection.execute(
                f'UPDATE "png_list" SET "{column}"=? WHERE prcfo=?',
                (positive_value, value))
            connection.execute(
                "INSERT INTO hit_attribution_promotions VALUES "
                "(?, ?, ?, ?, NULL, ?, ?, NULL)",
                (promotion_id, run_id, column, value,
                 json.dumps(positive_value, default=str), ha._now()))
    return promotion_id


def _old_undo_hit_promotion(db_path: str, promotion_id: str) -> int:
    """Clear values written by one promotion; preserve the audit record.

    :param db_path: path to the SQLite measurements database to update.
    :param promotion_id: identifier returned by :func:`promote_hit_calls`.
    """
    with sqlite3.connect(db_path, timeout=30) as connection:
        rows = connection.execute(
            "SELECT annotation_column, prcfo FROM hit_attribution_promotions "
            "WHERE promotion_id=? AND undone_at IS NULL", (promotion_id,)
        ).fetchall()
        if not rows:
            return 0
        columns = {row[0] for row in rows}
        if len(columns) != 1:
            raise ha.HitAttributionError("promotion audit contains multiple columns")
        column = next(iter(columns))
        if not ha._SQL_NAME.fullmatch(column):
            raise ha.HitAttributionError("stored annotation column is unsafe")
        connection.executemany(
            f'UPDATE "png_list" SET "{column}"=NULL WHERE prcfo=?',
            [(row[1],) for row in rows])
        connection.execute(
            "UPDATE hit_attribution_promotions SET undone_at=? "
            "WHERE promotion_id=? AND undone_at IS NULL", (ha._now(), promotion_id))
        return len(rows)


def _old_store_attribution(db_path: str, result) -> int:
    """Store the GUI investigation bundle under its immutable run context.

    :param db_path: path to the existing SQLite measurements database.
    :param result: investigation bundle to persist.
    """
    path = os.path.abspath(os.path.expanduser(db_path))
    if not os.path.isfile(path):
        raise ha.HitAttributionError(f"no database at {path}")
    ha._require_columns(result.cells, ["prcfo", "candidate_probability"],
                     "investigation cells")
    if result.cells["prcfo"].duplicated().any():
        raise ha.HitAttributionError("investigation contains duplicate prcfo object keys")
    context = {
        "regression_results_folder": result.context.regression_results_folder,
        "regression_run_sha256": result.context.regression_run_sha256,
        "gene": result.context.gene,
        "phenotype": result.context.phenotype,
        "effect": result.context.effect,
        "guides": list(result.context.guides),
        "fdr": result.context.fdr,
        "direction": result.context.direction,
        "feature_columns": result.feature_columns,
        "split_level": result.split_level,
        "warnings": result.warnings,
        "enrichment": result.enrichment,
    }
    with sqlite3.connect(path, timeout=30) as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS hit_investigation_runs (
                attribution_run_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                context_json TEXT NOT NULL
            )
        """)
        connection.execute("""
            CREATE TABLE IF NOT EXISTS hit_investigation_cells (
                attribution_run_id TEXT NOT NULL,
                prcfo TEXT NOT NULL,
                candidate_probability REAL NOT NULL,
                candidate_uncertainty REAL NOT NULL,
                candidate_call INTEGER NOT NULL,
                attribution_fold INTEGER NOT NULL,
                PRIMARY KEY (attribution_run_id, prcfo),
                FOREIGN KEY (attribution_run_id)
                    REFERENCES hit_investigation_runs(attribution_run_id)
                    ON DELETE CASCADE
            )
        """)
        connection.execute(
            "INSERT INTO hit_investigation_runs VALUES (?, ?, ?)",
            (result.attribution_run_id, ha._now(),
             json.dumps(context, sort_keys=True, default=str)))
        rows = []
        for row in result.cells.itertuples(index=False):
            probability = float(getattr(row, "candidate_probability"))
            uncertainty = float(getattr(
                row, "candidate_uncertainty", 1 - abs(2 * probability - 1)))
            call = bool(getattr(row, "candidate_call", probability >= 0.5))
            fold = int(getattr(row, "attribution_fold", -1))
            rows.append((result.attribution_run_id, str(row.prcfo),
                         probability, uncertainty, int(call), fold))
        connection.executemany(
            "INSERT INTO hit_investigation_cells VALUES (?, ?, ?, ?, ?, ?)",
            rows)
    return len(rows)


def _old_promote_calls(db_path: str, attribution_run_id: str,
                  annotation_column: str) -> str:
    """Promote stored calls while recording every previous annotation value.

    :param db_path: path to the SQLite measurements database to update.
    :param attribution_run_id: stored investigation whose calls to promote.
    :param annotation_column: ``png_list`` column to create or update.
    """
    column = str(annotation_column)
    if not ha._SQL_NAME.fullmatch(column):
        raise ha.HitAttributionError(
            "annotation column must contain only letters, numbers and underscores")
    promotion_id = str(ha.uuid.uuid4())
    with sqlite3.connect(db_path, timeout=30) as connection:
        table_columns = {row[1] for row in connection.execute(
            'PRAGMA table_info("png_list")')}
        if "prcfo" not in table_columns:
            raise ha.HitAttributionError("png_list has no prcfo object key")
        if column not in table_columns:
            connection.execute(f'ALTER TABLE "png_list" ADD COLUMN "{column}"')
        calls = connection.execute(
            "SELECT prcfo, candidate_call FROM hit_investigation_cells "
            "WHERE attribution_run_id=?", (attribution_run_id,)).fetchall()
        if not calls:
            raise ha.HitAttributionError(
                f"no stored cells for attribution run {attribution_run_id!r}")
        connection.execute("""
            CREATE TABLE IF NOT EXISTS hit_promotion_audit (
                promotion_id TEXT NOT NULL,
                attribution_run_id TEXT NOT NULL,
                annotation_column TEXT NOT NULL,
                prcfo TEXT NOT NULL,
                previous_json TEXT,
                promoted_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                undone_at TEXT,
                PRIMARY KEY (promotion_id, prcfo)
            )
        """)
        for prcfo, call in calls:
            current = connection.execute(
                f'SELECT "{column}" FROM "png_list" WHERE prcfo=?',
                (prcfo,)).fetchone()
            if current is None:
                raise ha.HitAttributionError(
                    f"attributed object {prcfo!r} is absent from png_list")
            previous = current[0]
            connection.execute(
                f'UPDATE "png_list" SET "{column}"=? WHERE prcfo=?',
                (int(call), prcfo))
            connection.execute(
                "INSERT INTO hit_promotion_audit VALUES "
                "(?, ?, ?, ?, ?, ?, ?, NULL)",
                (promotion_id, attribution_run_id, column, prcfo,
                 json.dumps(previous, default=str), json.dumps(int(call)), ha._now()))
    return promotion_id


def _old_revert_promotion(db_path: str, promotion_id: str) -> int:
    """Restore exactly the values replaced by :func:`promote_calls`.

    Reverting something that was never promoted does nothing and returns
    ``0`` -- including on a database where nothing has ever been promoted.
    The audit table is created by :func:`promote_calls`; when it does not
    exist yet, this function returns ``0`` instead of raising
    ``sqlite3.OperationalError``.

    :param db_path: path to the SQLite measurements database to update.
    :param promotion_id: identifier returned by :func:`promote_calls`.
    """
    with sqlite3.connect(db_path, timeout=30) as connection:
        audited = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' "
            "AND name='hit_promotion_audit'").fetchone()
        if not audited:
            return 0
        rows = connection.execute(
            "SELECT annotation_column, prcfo, previous_json "
            "FROM hit_promotion_audit WHERE promotion_id=? AND undone_at IS NULL",
            (promotion_id,)).fetchall()
        if not rows:
            return 0
        columns = {row[0] for row in rows}
        if len(columns) != 1:
            raise ha.HitAttributionError("promotion audit contains multiple columns")
        column = next(iter(columns))
        if not ha._SQL_NAME.fullmatch(column):
            raise ha.HitAttributionError("stored annotation column is unsafe")
        for _, prcfo, previous_json in rows:
            previous = json.loads(previous_json)
            connection.execute(
                f'UPDATE "png_list" SET "{column}"=? WHERE prcfo=?',
                (previous, prcfo))
        connection.execute(
            "UPDATE hit_promotion_audit SET undone_at=? "
            "WHERE promotion_id=? AND undone_at IS NULL", (ha._now(), promotion_id))
        return len(rows)


# reference SQL, copied from spacr.condition_annotations before the port


def _old_owned_png_list_columns(connection):
    """Columns of ``png_list`` that earlier Graph Builder saves wrote."""
    import sqlite3

    try:
        rows = connection.execute(
            f"SELECT payload_json FROM {ca._quote(ca.PROVENANCE_TABLE)} WHERE table_name LIKE ?",
            (ca._PNG_LIST_RECEIPT + "%",)).fetchall()
    except sqlite3.Error:
        return set()
    owned = set()
    for (payload,) in rows:
        try:
            owned.update(json.loads(payload).get("columns", []))
        except (ValueError, AttributeError):
            continue
    return owned


def _old_write_png_list_labels(path, labels, columns, source, definition, busy_timeout):
    """Write one database's labels into ``png_list`` by ``prcfo``.

    Rows are matched on :func:`_comparable_prcfo` of both sides, so a plate
    stored as ``pplate1`` meets a table that reads it as ``plate1``. When no
    row matches, nothing is written: no column is added and no receipt is
    stored, and a few unmatched keys of each side are returned so the caller
    can show why. Otherwise columns are added with
    :func:`spacr.qt.annotate_engine.ensure_annotation_column`, the call
    Annotate opens a source with, and only rows whose value changes are
    written, in one transaction that also records the receipt.

    :param path: the plate's measurement database.
    :param labels: one row per comparable key (``__key``) with the table's
        ``prcfo`` and every output column.
    :returns: ``objects`` written, ``labelled`` objects, objects with
        ``no_png_list_row``, ``png_list`` rows ``matched`` and ``updated``,
        and ``unmatched``: up to three keys of the ``table`` and of
        ``png_list`` that met nothing, filled only when nothing matched.
    :raises ca.AnnotationError: when ``png_list`` already has an output column
        that no Graph Builder save wrote, so Annotate's labels are never
        overwritten.
    :raises DatabaseBusy: when another program holds the database past
        ``busy_timeout`` seconds.
    """
    import sqlite3

    from spacr.database_concurrency import DatabaseBusy, connect, is_busy_error, transaction
    from spacr.filters import rowid_expression
    from spacr.qt.annotate_engine import ensure_annotation_column

    wanted = set(labels["__key"])
    unmatched = {"table": [], "png_list": []}
    try:
        reader = connect(path, readonly=True, timeout=busy_timeout)
        try:
            existing = [row[1] for row in reader.execute('PRAGMA table_info("png_list")')]
            owned = _old_owned_png_list_columns(reader)
            stored_keys = [(value, ca._comparable_prcfo(value)) for (value,) in
                           reader.execute('SELECT "prcfo" FROM "png_list"').fetchall()]
        finally:
            reader.close()
        present = {comparable for _value, comparable in stored_keys if comparable is not None}
        if not wanted & present:
            unmatched = {
                "table": ca._unmatched_sample(zip(labels["prcfo"].tolist(), labels["__key"].tolist()),
                                           present),
                "png_list": ca._unmatched_sample(
                    ((None if value is None else str(value), comparable)
                     for value, comparable in stored_keys), wanted)}
            return {"objects": 0, "labelled": 0, "no_png_list_row": len(labels),
                    "matched": 0, "updated": 0, "unmatched": unmatched}
        foreign = [column for column in columns if column in existing and column not in owned]
        if foreign:
            raise ca.AnnotationError(
                f"png_list in {path} already has a column named {', '.join(foreign)} that Graph Builder "
                "did not write. Rename the annotation column so those values are not overwritten.")
        for column in columns:
            if column not in existing:
                ensure_annotation_column(path, column)
        connection = connect(path, timeout=busy_timeout)
        try:
            with transaction(connection, busy_timeout=busy_timeout):
                names = [row[1] for row in connection.execute('PRAGMA table_info("png_list")')]
                rowid = rowid_expression(names) or "_rowid_"
                selected = ", ".join(ca._quote(column) for column in columns)
                stored = pd.DataFrame(
                    connection.execute(f'SELECT {rowid}, "prcfo", {selected} FROM "png_list"').fetchall(),
                    columns=["__row", "prcfo", *columns])
                stored["__key"] = [ca._comparable_prcfo(value) for value in stored["prcfo"].tolist()]
                stored = stored[stored["__key"].notna()]
                joined = stored.drop(columns="prcfo").merge(
                    labels.drop(columns="prcfo"), on="__key", how="inner", suffixes=("", "__new"))
                updated = set()
                for column in columns:
                    changed = [(wanted_value, int(row)) for row, current, wanted_value in zip(
                        joined["__row"].tolist(), joined[column].tolist(), joined[column + "__new"].tolist())
                        if not ca._same_label(current, wanted_value)]
                    updated.update(row for _value, row in changed)
                    connection.executemany(
                        f'UPDATE "png_list" SET {ca._quote(column)} = ? WHERE {rowid} = ?', changed)
                found = connection.execute(
                    "SELECT type FROM sqlite_master WHERE name=?", (ca.PROVENANCE_TABLE,)).fetchone()
                if found is None:
                    connection.execute(f"CREATE TABLE {ca._quote(ca.PROVENANCE_TABLE)} "
                                       "(table_name TEXT PRIMARY KEY, payload_json TEXT NOT NULL)")
                elif found[0] != "table" or [row[1] for row in connection.execute(
                        f"PRAGMA table_info({ca._quote(ca.PROVENANCE_TABLE)})")] != ["table_name", "payload_json"]:
                    raise ca.AnnotationError("The annotation provenance table has an incompatible schema.")
                payload = {"version": 1, "kind": "png_list", "source": source,
                           "columns": list(columns), "definition": definition}
                connection.execute(f"INSERT OR REPLACE INTO {ca._quote(ca.PROVENANCE_TABLE)} VALUES (?, ?)",
                                   (ca._receipt_key(source), json.dumps(payload)))
        finally:
            connection.close()
    except sqlite3.OperationalError as exc:
        if isinstance(exc, DatabaseBusy) or is_busy_error(exc):
            raise DatabaseBusy(
                f"{path} is in use by another program, so the annotations were not written "
                f"there. Close what is using it and save again.") from exc
        raise
    matched = set(joined["__key"])
    written = labels[labels["__key"].isin(matched)]
    return {"objects": len(written),
            "labelled": int(written[list(columns)].notna().any(axis=1).sum()),
            "no_png_list_row": len(labels) - len(written),
            "matched": len(joined), "updated": len(updated), "unmatched": unmatched}



# reference SQL, copied from spacr.active_learning before the port


def _old_ensure_round_tables(db_path: str) -> None:
    """Create :data:`ROUND_TABLE` and :data:`ROUND_LOG_TABLE` if absent.

    :param db_path: existing ``measurements.db`` in which to create the tables.

    Two tables, not one. Per-label provenance and per-round metrics have
    different cardinalities and different lifetimes: a label keeps its round
    forever, a round's held-out accuracy is rewritten if the round is re-fit.
    """
    con = sqlite3.connect(db_path, timeout=30)
    try:
        con.execute(f"""
            CREATE TABLE IF NOT EXISTS {al._quote_ident(al.ROUND_TABLE)} (
                png_path TEXT NOT NULL,
                annotation_column TEXT NOT NULL,
                round INTEGER NOT NULL,
                first_round INTEGER NOT NULL,
                label INTEGER,
                source TEXT NOT NULL DEFAULT 'manual',
                labelled_utc TEXT NOT NULL,
                PRIMARY KEY (png_path, annotation_column)
            )""")
        con.execute(f"""
            CREATE TABLE IF NOT EXISTS {al._quote_ident(al.ROUND_LOG_TABLE)} (
                annotation_column TEXT NOT NULL,
                round INTEGER NOT NULL,
                finished_utc TEXT NOT NULL,
                n_labels INTEGER NOT NULL DEFAULT 0,
                n_new_labels INTEGER NOT NULL DEFAULT 0,
                n_holdout INTEGER NOT NULL DEFAULT 0,
                holdout_accuracy REAL,
                holdout_f1_macro REAL,
                per_class_json TEXT,
                split_rule TEXT,
                model_type TEXT,
                model_path TEXT,
                card_path TEXT,
                measure TEXT,
                diversity TEXT,
                notes_json TEXT,
                PRIMARY KEY (annotation_column, round)
            )""")
        con.commit()
    finally:
        con.close()


def _old_record_labels(db_path: str, annotation_column: str,
                  labels: Dict[str, Any], round_index: int,
                  source: str = "manual") -> int:
    """Stamp each label with the round it was made in.

    Called by the Annotate screen every time it flushes a batch. The round a
    label came from is what makes early-round bias auditable: the first
    round's labels are drawn from whatever ordering existed before any model
    had seen this screen, and if 80 % of a class's labels carry round 0 then
    the "active learning" was mostly not active.

    ``first_round`` is preserved across re-labelling while ``round`` follows
    the current value, so both "when was this crop first looked at" and
    "which round set the label it has now" survive a correction.

    :param db_path: path to ``measurements.db``.
    :param annotation_column: the column the labels were written into.
    :param labels: ``{png_path: class or None}``. ``None`` is a *cleared*
        label and is recorded as such rather than dropped — a crop that was
        looked at and deliberately left blank is not the same as one never
        seen.
    :param round_index: the round these labels belong to.
    :param source: how the crop reached the annotator — ``'manual'``,
        ``'queue'``, or a caller's own tag.
    :returns: number of rows written.
    """
    if not labels:
        return 0
    _old_ensure_round_tables(db_path)
    stamp = al._utc_now()
    payload = [
        (str(path), str(annotation_column), int(round_index),
         int(round_index),
         (None if value is None else int(value)), str(source), stamp)
        for path, value in labels.items()]
    con = sqlite3.connect(db_path, timeout=30)
    try:
        con.executemany(
            f"INSERT INTO {al._quote_ident(al.ROUND_TABLE)} "
            f"(png_path, annotation_column, round, first_round, label, "
            f"source, labelled_utc) VALUES (?, ?, ?, ?, ?, ?, ?) "
            f"ON CONFLICT(png_path, annotation_column) DO UPDATE SET "
            f"round=excluded.round, label=excluded.label, "
            f"source=excluded.source, labelled_utc=excluded.labelled_utc",
            payload)
        con.commit()
    finally:
        con.close()
    return len(payload)


def _old_record_round(db_path: str, annotation_column: str, round_index: int,
                 **fields: Any) -> None:
    """Append (or replace) one row of the learning curve.

    :param db_path: path to ``measurements.db``.
    :param annotation_column: the column this round labelled into.
    :param round_index: the round number.
    :param fields: any of ``n_labels``, ``n_new_labels``, ``n_holdout``,
        ``holdout_accuracy``, ``holdout_f1_macro``, ``per_class`` (dict),
        ``split_rule``, ``model_type``, ``model_path``, ``card_path``,
        ``measure``, ``diversity``, ``notes`` (list).
    """
    import json
    _old_ensure_round_tables(db_path)
    values = {
        "annotation_column": str(annotation_column),
        "round": int(round_index),
        "finished_utc": al._utc_now(),
        "n_labels": int(fields.get("n_labels", 0) or 0),
        "n_new_labels": int(fields.get("n_new_labels", 0) or 0),
        "n_holdout": int(fields.get("n_holdout", 0) or 0),
        "holdout_accuracy": al._as_float_or_none(fields.get("holdout_accuracy")),
        "holdout_f1_macro": al._as_float_or_none(fields.get("holdout_f1_macro")),
        "per_class_json": json.dumps(fields.get("per_class") or {}),
        "split_rule": str(fields.get("split_rule") or ""),
        "model_type": str(fields.get("model_type") or ""),
        "model_path": str(fields.get("model_path") or ""),
        "card_path": str(fields.get("card_path") or ""),
        "measure": str(fields.get("measure") or ""),
        "diversity": str(fields.get("diversity") or ""),
        "notes_json": json.dumps(list(fields.get("notes") or [])),
    }
    con = sqlite3.connect(db_path, timeout=30)
    try:
        con.execute(
            f"INSERT OR REPLACE INTO {al._quote_ident(al.ROUND_LOG_TABLE)} "
            f"({', '.join(al._quote_ident(c) for c in values)}) "
            f"VALUES ({', '.join('?' * len(values))})",
            tuple(values.values()))
        con.commit()
    finally:
        con.close()


# reference SQL, copied from spacr.predictions before the port


def _old_merge_locked(cur, results, spec,
                  table, key, timelapse, repaired):
    """Do the merge inside an already-open transaction."""
    quoted_table = pr._quote(table)
    cur.execute(f"SELECT * FROM {quoted_table} LIMIT 0")
    table_columns = [d[0] for d in cur.description]
    key_columns = [c for c in table_columns
                   if c in ("prcfo", "png_path", "file_name")
                   or c in pr._PRCFO_METADATA or c in pr._OBJECT_ID_COLUMNS
                   or c in ("timeID", "time_id")]
    rowid = pr._rowid_alias(table_columns)
    select = ", ".join([rowid] + [pr._quote(c) for c in key_columns])
    rows = cur.execute(f"SELECT {select} FROM {quoted_table}").fetchall()
    rowids = [r[0] for r in rows]
    db_frame = pd.DataFrame([r[1:] for r in rows], columns=key_columns)

    updates, report, order, types = pr._plan_merge(
        results, spec, table, key, timelapse, table_columns,
        db_frame, rowids, repaired)
    for db_col in report.added_columns:
        cur.execute(f"ALTER TABLE {quoted_table} ADD COLUMN "
                    f"{pr._quote(db_col)} {types[db_col]}")
    if updates:
        assignments = ", ".join(f"{pr._quote(c)} = ?" for c in order)
        cur.executemany(
            f"UPDATE {quoted_table} SET {assignments} WHERE {rowid} = ?",
            updates)
    return report


def _old_migrate(db_path, table="png_list"):
    """The SQLite repair as it was before the port."""
    repaired: list = []
    con = sqlite3.connect(str(db_path), timeout=30)
    con.isolation_level = None
    try:
        cur = con.cursor()
        tables = {row[0] for row in
                  cur.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if table not in tables:
            return []
        cols = [row[1] for row in cur.execute(f"PRAGMA table_info({pr._quote(table)})")]
        pending = [c for c in pr.ANNOTATE_ENCODED_COLUMNS if c in cols]
        if not pending:
            return []

        cur.execute("BEGIN")
        try:
            for column in pending:
                quoted = pr._quote(column)
                distinct = {row[0] for row in cur.execute(
                    f"SELECT DISTINCT {quoted} FROM {pr._quote(table)} "
                    f"WHERE {quoted} IS NOT NULL")}
                if not distinct or not distinct.issubset({1, 2}) or 2 not in distinct:
                    continue
                cur.execute(f"UPDATE {pr._quote(table)} SET {quoted} = 0 "
                              f"WHERE {quoted} = 2")
                repaired.append((table, column, cur.rowcount))
            cur.execute("COMMIT")
        except BaseException:
            cur.execute("ROLLBACK")
            raise
    finally:
        con.close()

    return repaired


def _old_merge(results, db_path, columns, table="png_list"):
    """The SQLite merge as it was before the port."""
    spec = {db_col: (src if isinstance(src, (tuple, list)) else (src, "REAL"))
            for db_col, src in columns.items()}
    repaired = _old_migrate(db_path, table=table)
    con = sqlite3.connect(db_path, timeout=30)
    con.isolation_level = None
    try:
        cur = con.cursor()
        cur.execute("BEGIN")
        try:
            report = _old_merge_locked(cur, results, spec, table, "auto",
                                       None, repaired)
            cur.execute("COMMIT")
        except BaseException:
            cur.execute("ROLLBACK")
            raise
    finally:
        con.close()
    return report


# 1. old vs new on a legacy measurements.db


def _hit_flow(path, *, old):
    """Every hit-attribution writer once, old SQL or the store."""
    fn = {name: (globals()[f"_old_{name}"] if old else getattr(ha, name))
          for name in ("write_hit_attribution", "promote_hit_calls",
                       "undo_hit_promotion", "store_attribution",
                       "promote_calls", "revert_promotion")}
    run = fn["write_hit_attribution"](path, _result(), run_id="run-1")
    first = fn["promote_hit_calls"](path, _result(), run_id=run,
                                    annotation_column="hit_like")
    assert fn["undo_hit_promotion"](path, first) == 2
    assert fn["undo_hit_promotion"](path, first) == 0
    assert fn["store_attribution"](path, _investigation()) == 3
    second = fn["promote_calls"](path, "inv-1", "annotate")
    assert fn["revert_promotion"](path, second) == 3
    assert fn["revert_promotion"](path, "never") == 0
    fn["promote_calls"](path, "inv-1", "annotate")
    return first, second


def test_hit_attribution_writers_match_the_old_sql(tmp_path, frozen):
    old, new = _twins(tmp_path, annotate=[1, None, 2, None, None, None, None,
                                          None])
    assert _old_revert_promotion(old, "x") == 0
    assert ha.revert_promotion(new, "x") == 0
    _freeze(frozen)
    first = _hit_flow(old, old=True)
    _freeze(frozen)
    assert _hit_flow(new, old=False) == first
    assert _dump(new) == _dump(old)


def test_a_legacy_hit_rerun_is_byte_identical(tmp_path, monkeypatch):
    def run(path):
        _freeze(monkeypatch)
        _hit_flow(path, old=False)
        return _sha(path)

    first, second = _twins(tmp_path, annotate=[None] * 8)
    assert run(first) == run(second)


def test_png_list_labels_match_the_old_sql(tmp_path):
    old, new = _twins(tmp_path)
    source = {"path": "/src.db", "table": "t"}

    def labels(values):
        frame = pd.DataFrame({"prcfo": ["p" + k for k in PRCFO[:4]]})
        frame["__key"] = [ca._comparable_prcfo(k) for k in frame["prcfo"]]
        frame["condition"] = values
        return frame

    for values in (["a", "b", None, "a"], ["a", None, "c", None]):
        got_old = _old_write_png_list_labels(old, labels(values),
                                             ["condition"], source, {"d": 1},
                                             5.0)
        got_new = ca._write_png_list_labels(new, labels(values),
                                            ["condition"], source, {"d": 1},
                                            5.0)
        assert got_new == got_old
        assert _dump(new) == _dump(old)
    nothing = labels(["a"] * 4).assign(__key=["zz"] * 4)
    assert (ca._write_png_list_labels(new, nothing, ["other"], source, {}, 5.0)
            == _old_write_png_list_labels(old, nothing, ["other"], source, {},
                                          5.0))
    assert _dump(new) == _dump(old)


def test_png_list_labels_refuse_a_column_graph_builder_did_not_write(tmp_path):
    path = _legacy(tmp_path / "m.db", annotate=[None] * 8)
    frame = pd.DataFrame({"prcfo": PRCFO[:2]})
    frame["__key"] = [ca._comparable_prcfo(k) for k in frame["prcfo"]]
    frame["annotate"] = [1, 2]
    before = _dump(path)
    with pytest.raises(ca.AnnotationError):
        ca._write_png_list_labels(path, frame, ["annotate"], {"path": "x"},
                                  {}, 5.0)
    assert _dump(path) == before


def _round_flow(path, *, old):
    """Every round-table writer once, old SQL or the store."""
    ensure = _old_ensure_round_tables if old else al.ensure_round_tables
    labels = _old_record_labels if old else al.record_labels
    rounds = _old_record_round if old else al.record_round
    ensure(path)
    assert labels(path, "annotate", {PATHS[0]: 1, PATHS[1]: None}, 0) == 2
    assert labels(path, "annotate", {PATHS[0]: 2, PATHS[2]: 1}, 1,
                  source="queue") == 2
    rounds(path, "annotate", 0, n_labels=2, holdout_accuracy=0.5,
           per_class={"1": 0.5}, notes=["a"])
    rounds(path, "annotate", 0, n_labels=3, holdout_accuracy=0.75)
    rounds(path, "annotate", 1, n_labels=4)


def test_round_tables_match_the_old_sql(tmp_path, monkeypatch):
    monkeypatch.setattr(al, "_utc_now", lambda: NOW)
    old, new = _twins(tmp_path)
    _round_flow(old, old=True)
    _round_flow(new, old=False)
    assert _dump(new) == _dump(old)


def test_round_tables_on_a_store_project_read_like_legacy(tmp_path,
                                                          monkeypatch):
    monkeypatch.setattr(al, "_utc_now", lambda: NOW)
    project = _project(tmp_path)
    legacy = _legacy(tmp_path / "m.db")
    for target in (project, legacy):
        _round_flow(target, old=False)
    rounds = ('SELECT png_path, annotation_column, "round", first_round, '
              "label, source, labelled_utc FROM annotation_rounds ORDER BY "
              "png_path")
    log = ('SELECT annotation_column, "round", n_labels, holdout_accuracy, '
           'per_class_json FROM annotation_round_log ORDER BY "round"')
    with sqlite3.connect(legacy) as db:
        expected = (db.execute(rounds).fetchall(), db.execute(log).fetchall())
    with store.open(project) as reader:
        got = (reader.sql(rounds), reader.sql(log))
    for frame, rows in zip(got, expected):
        frame = frame.astype(object).where(frame.notna(), None)
        assert [tuple(r) for r in frame.values.tolist()] == rows


def test_the_thresholded_column_and_merges_match_the_old_sql(tmp_path):
    from spacr.predictions import ML_CLASS_COLUMN

    old, new = _twins(tmp_path, extra=(f"{ML_CLASS_COLUMN} INTEGER",))
    for path in (old, new):
        with sqlite3.connect(path) as db:
            db.executemany(f"UPDATE png_list SET {ML_CLASS_COLUMN} = ? WHERE "
                           f"png_path = ?", [(2 - i % 2, p) for i, p in
                                             enumerate(PATHS)])
    results = pd.DataFrame({"prcfo": PRCFO[:6],
                            "pred": np.linspace(0, 1, 6),
                            "cls": [0, 1, 1, 0, None, 1]})
    columns = {"score": ("pred", "REAL"), "label": ("cls", "INTEGER")}
    report_old = _old_merge(results, old, columns)
    report_new = pr.merge_prediction_results(results, new, columns,
                                             verbose=False)
    assert report_new == report_old
    assert report_new.repaired == ((("png_list", ML_CLASS_COLUMN, 4),))
    assert _dump(new) == _dump(old)
    again = results.assign(pred=results["pred"] + 1)
    assert (pr.merge_prediction_results(again, new, columns, verbose=False)
            == _old_merge(again, old, columns))
    assert _dump(new) == _dump(old)
    assert pr.migrate_prediction_columns(new, verbose=False) == []

    table = pd.DataFrame({"png_path": PATHS[:3],
                          "class_thresholded": ["1", "", "2"]})
    assert al._write_thresholded_column(new, table, "annotate_thresholded") == 3
    with sqlite3.connect(new) as db:
        assert db.execute("SELECT annotate_thresholded FROM png_list ORDER BY "
                          "rowid").fetchall()[:4] == [(1,), (None,), (2,),
                                                      (None,)]


def test_a_legacy_merge_rerun_is_byte_identical(tmp_path):
    def run(path):
        results = pd.DataFrame({"prcfo": PRCFO, "pred": np.linspace(0, 1, 8)})
        pr.merge_prediction_results(results, path, {"score": ("pred", "REAL")},
                                    verbose=False)
        table = pd.DataFrame({"png_path": PATHS,
                              "class_thresholded": ["1", "2"] * 4})
        al._write_thresholded_column(path, table, "annotate_thresholded")
        return _sha(path)

    first, second = _twins(tmp_path)
    assert run(first) == run(second)


def test_a_merge_without_png_path_writes_by_prcfo(tmp_path):
    path = str(tmp_path / "m.db")
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE png_list (prcfo TEXT, file_name TEXT)")
        db.executemany("INSERT INTO png_list VALUES (?, ?)",
                       [(PRCFO[0], None), (PRCFO[1], None)])
    results = pd.DataFrame({"prcfo": PRCFO[:2], "pred": [0.1, 0.2]})
    report = pr.merge_prediction_results(results, path,
                                         {"score": ("pred", "REAL")},
                                         verbose=False)
    assert report.matched_rows == 2
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT score FROM png_list").fetchall() == [
            (0.1,), (0.2,)]


# 2. atomicity


def test_an_interrupted_legacy_promotion_leaves_no_half_state(tmp_path,
                                                              monkeypatch,
                                                              frozen):
    path = _legacy(tmp_path / "m.db", annotate=[None] * 8)
    before = _dump(path)

    def fails(self, table, rows, **kwargs):
        if rows:
            raise RuntimeError("interrupted")
        return original(self, table, rows, **kwargs)

    original = store.Transaction.append_rows
    monkeypatch.setattr(store.Transaction, "append_rows", fails)
    with pytest.raises(RuntimeError):
        ha.promote_hit_calls(path, _result(), run_id="r",
                             annotation_column="hit_like")
    assert _dump(path) == before
    monkeypatch.setattr(store.Transaction, "append_rows", original)
    ha.promote_hit_calls(path, _result(), run_id="r",
                         annotation_column="hit_like")
    assert sqlite3.connect(path).execute(
        "SELECT COUNT(*) FROM png_list WHERE hit_like = 1").fetchone() == (2,)


def test_an_interrupted_store_promotion_leaves_no_half_state(tmp_path,
                                                             monkeypatch,
                                                             frozen):
    project = _project(tmp_path)

    def crash(*_args, **_kwargs):
        raise RuntimeError("power cut before the commit marker")

    monkeypatch.setattr(store, "_write_commit_marker", crash)
    with pytest.raises(RuntimeError):
        ha.promote_hit_calls(project, _result(), run_id="r",
                             annotation_column="hit_like")
    assert os.listdir(os.path.join(project, "edits", "promotion"))
    with store.open(project) as reader:
        assert "png_list_labeled" not in reader.tables()
        assert "hit_attribution_promotions" not in reader.tables()
        assert reader.edits(latest=False).empty
    with store.transaction(project) as tx:
        assert "hit_like" not in tx.columns("png_list")
    monkeypatch.undo()
    assert store.compact(project)["rows"] == 0
    assert not os.listdir(os.path.join(project, "edits", "promotion"))
    ha.promote_hit_calls(project, _result(), run_id="r",
                         annotation_column="hit_like")
    labels = _labels(project, "hit_like")
    assert labels == {p: (1 if i in (0, 2) else None)
                      for i, p in enumerate(PATHS)}


def test_a_transaction_error_rolls_back_every_write(tmp_path):
    path = _legacy(tmp_path / "m.db")
    project = _project(tmp_path / "s")
    before = _dump(path)
    ddl = "CREATE TABLE IF NOT EXISTS audit (a TEXT)"
    for target in (path, project):
        with pytest.raises(ValueError):
            with store.transaction(target) as tx:
                tx.ensure_columns("png_list", {"flag": "INTEGER"})
                tx.edit([{"key": PATHS[0], "value": 1}], kind="annotation",
                        source="test", table="png_list", column="flag")
                tx.append_rows("audit", [{"a": "x"}], columns=("a",),
                               ddl=ddl)
                raise ValueError("refused")
    assert _dump(path) == before
    with store.open(project) as reader:
        assert "audit" not in reader.tables()
        assert "png_list_labeled" not in reader.tables()


# 3. a store project shows the same results


def test_hit_promotion_on_a_store_project_reads_like_legacy(tmp_path, frozen):
    project = _project(tmp_path, annotate=[1, None, 2, None, None, None, None,
                                           None])
    legacy = _legacy(tmp_path / "m.db", annotate=[1, None, 2, None, None,
                                                  None, None, None])
    for target in (project, legacy):
        _hit_flow(target, old=False)
    with sqlite3.connect(legacy) as db:
        expected = dict(db.execute("SELECT png_path, annotate FROM png_list"))
        audit = db.execute("SELECT prcfo, previous_json, undone_at FROM "
                           "hit_promotion_audit ORDER BY promotion_id, prcfo"
                           ).fetchall()
        promotions = db.execute(
            "SELECT prcfo, undone_at FROM hit_attribution_promotions "
            "ORDER BY prcfo").fetchall()
    assert _labels(project, "annotate") == expected
    assert _labels(project, "hit_like") == {p: None for p in PATHS}
    with store.open(project) as reader:
        assert reader.sql(
            "SELECT prcfo, previous_json, undone_at FROM hit_promotion_audit "
            "ORDER BY promotion_id, prcfo").values.tolist() == [
                list(r) for r in audit]
        assert reader.sql(
            "SELECT prcfo, undone_at FROM hit_attribution_promotions ORDER BY "
            "prcfo").values.tolist() == [list(r) for r in promotions]
        assert len(reader.sql("SELECT * FROM object_hit_attribution")) == 8
    with pytest.raises(ha.HitAttributionError):
        ha.promote_hit_calls(project, _result(), run_id="r",
                             annotation_column="hit_like")


def test_png_list_labels_on_a_store_project(tmp_path):
    project = _project(tmp_path)
    legacy = _legacy(tmp_path / "m.db")
    source = {"path": "/src.db", "table": "t"}
    frame = pd.DataFrame({"prcfo": PRCFO[:3]})
    frame["__key"] = [ca._comparable_prcfo(k) for k in frame["prcfo"]]
    frame["condition"] = [1, 2, None]
    reports = [ca._write_png_list_labels(target, frame, ["condition"], source,
                                         {"d": 1}, 5.0)
               for target in (project, legacy)]
    assert reports[0] == reports[1]
    assert _labels(project, "condition") == {
        p: v for p, v in sqlite3.connect(legacy).execute(
            "SELECT png_path, condition FROM png_list")}
    again = ca._write_png_list_labels(project, frame, ["condition"], source,
                                      {"d": 2}, 5.0)
    assert again["updated"] == 0
    with store.open(project) as reader:
        receipts = reader.sql(
            f'SELECT payload_json FROM "{ca.PROVENANCE_TABLE}"')
    assert len(receipts) == 1
    assert json.loads(receipts.iloc[0, 0])["definition"] == {"d": 2}



def test_crop_embeddings_on_a_store_project_are_a_derived_table(tmp_path):
    project = _project(tmp_path)
    before = {f for f in os.listdir(os.path.join(project, "tables"))}
    vectors = np.arange(12, dtype=np.float32).reshape(3, 4)
    assert al._store_crop_embeddings(project, PRCFO[:3], vectors) == 3
    stored = al._stored_embedding_frame(project)
    assert stored["prcfo"].tolist() == PRCFO[:3]
    assert stored.drop(columns="prcfo").to_numpy().tolist() == \
        vectors.tolist()
    with store.open(project) as reader:
        active = reader.active_outputs()
    assert active["kind"].tolist() == ["embedding"]
    first = active["table"].iloc[0]
    assert al._store_crop_embeddings(project, PRCFO[:3], vectors) == 3
    with store.open(project) as reader:
        assert reader.active_outputs()["table"].tolist() == [first]
    assert before < set(os.listdir(os.path.join(project, "tables")))
    assert al._stored_embedding_frame(_legacy(tmp_path / "m.db")) is None

# 4. migrate_legacy


def test_migrate_legacy_round_trips_the_png_list_labels(tmp_path):
    legacy = _legacy(tmp_path / "m.db", annotate=[1, None, 2, 1, None, None,
                                                  2, None],
                     extra=("verdict TEXT",))
    with sqlite3.connect(legacy) as db:
        db.execute("UPDATE png_list SET verdict = 'keep' WHERE png_path = ?",
                   (PATHS[0],))
    before = _sha(legacy)
    project = _project(tmp_path)
    shutil.copy(legacy, os.path.join(project, "measurements.db"))
    with pytest.raises(store.StoreError):
        store.migrate_legacy(legacy)
    done = store.migrate_legacy(project, author="importer")
    assert done["columns"] == ["verdict", "annotate"]
    assert done["edits"] == 5
    assert _labels(project, "annotate") == _legacy_labels(legacy, "annotate")
    with store.open(project) as reader:
        frame = reader.query(table="png_list_labeled")
        edits = reader.edits(latest=False)
    assert dict(zip(frame["png_path"], frame["verdict"].where(
        frame["verdict"].notna(), None))) == _legacy_labels(legacy, "verdict")
    assert set(edits["source"]) == {"import"}
    assert set(edits["author"]) == {"importer"}
    assert _sha(legacy) == before
    assert _sha(os.path.join(project, "measurements.db")) == before
    assert store.migrate_legacy(project)["edits"] == 0
    store.edit(project, [{"key": PATHS[0], "value": 2}], kind="annotation",
               source="annotator", table="png_list", column="annotate")
    assert store.migrate_legacy(project, columns=["annotate"])["edits"] == 1
    assert _labels(project, "annotate")[PATHS[0]] == 1


def test_migrate_legacy_needs_a_store_project_and_an_older_file(tmp_path):
    project = _project(tmp_path)
    with pytest.raises(store.StoreError):
        store.migrate_legacy(project)
    legacy = _legacy(tmp_path / "m.db", annotate=[1] * 8)
    with pytest.raises(store.StoreError):
        store.migrate_legacy(project, db=legacy, table="cell")
    assert store.migrate_legacy(project, db=legacy)["edits"] == 8


# 5. the arrival contract of the new public API (2026-11-24)

S5B_BIRTH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data",
                         "release_contracts",
                         "s5_store_edit_log_api_arrivals_2026-11-23.json")
S5B_ARRIVALS = frozenset({
    "spacr.store.transaction", "spacr.store.Transaction",
    "spacr.store.Transaction.tables", "spacr.store.Transaction.columns",
    "spacr.store.Transaction.rows", "spacr.store.Transaction.edit",
    "spacr.store.Transaction.ensure_columns",
    "spacr.store.Transaction.append_rows",
    "spacr.store.Transaction.update_rows", "spacr.store.migrate_legacy"})


def test_the_new_api_is_in_the_s5_arrival_contract():
    from tests.test_store_edit_log_s5 import S5_ARRIVALS, _s5_birth_arrivals

    assert S5B_ARRIVALS <= S5_ARRIVALS
    entries = _s5_birth_arrivals()
    assert S5B_ARRIVALS <= set(entries)
    for name in ("transaction", "Transaction", "migrate_legacy"):
        assert name in store.__all__
