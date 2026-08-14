"""Write per-object classification scores back into a spaCR ``measurements.db``.

Two classifiers in this package score *every object of a whole database*:

* the convolutional one --- :func:`spacr.deep_spacr.apply_model_to_tar`, driven
  by :func:`spacr.deep_spacr.deep_spacr`; and
* the classical-ML one --- :func:`spacr.ml.generate_ml_scores`.

Both produce one row per crop, and both used to leave those rows in a CSV next
to the model. A CSV is not where the rest of spaCR looks: the Annotate app, the
active-learning queue, ``generate_image_umap``, the plate heatmaps and every
GUI table read the ``png_list`` table of ``measurements.db``. So the scores have
to land there, on the row they belong to.

Why this is its own module
--------------------------
The merge used to live in :mod:`spacr.deep_spacr`, which is an odd home for
something :mod:`spacr.ml` needs just as much --- importing ``deep_spacr`` pulls
in torch and torchvision, which the classical-ML path has no use for. The
obvious alternative, :mod:`spacr.utils`, already owns the database *write*
helpers (``_append_to_measurements_db``, ``rename_columns_in_db``), but it is
also 8000 lines and imports most of the package; adding a third database
concern there would make it harder, not easier, to see that CV and ML share one
code path. This module is small, imports nothing from spaCR at module scope
except :mod:`spacr.utils` helpers pulled in lazily, and is what both callers
import.

The join key
------------
``prcfo`` --- ``plate_row_column_field_object`` --- is the canonical per-object
identity in this codebase. :func:`spacr.utils.filepaths_to_database` writes it
onto every ``png_list`` row, :func:`spacr.io._read_and_merge_data` indexes the
merged feature frame by it, and :func:`spacr.ml.generate_ml_scores` already
joins annotations onto features with it. Keying the merge on it means the CV
and the ML stage land on *the same row*, which is the whole point of letting
them coexist.

The previous implementation matched on ``os.path.basename(png_path)``. A
basename is not an identity:

* the tar handed to the model is built with
  ``arcname=os.path.basename(img_path)`` (:func:`spacr.utils.add_images_to_tar`),
  so a run over two source folders whose plates are *both* called ``plate1``
  --- which is what happens whenever the plate name comes from the source
  folder name, see :mod:`spacr.io` ``_rename_and_organize_image_files`` ---
  puts two different crops in the archive under one member name; and
* the old lookup was a plain ``dict`` assignment, so the second of those two
  crops silently overwrote the first and one of the two plates was scored with
  the other plate's predictions.

The second half of that is the real defect, and it is not fixed by changing the
key: two crops that share a basename share a ``prcfo`` too. So the fix is to
*detect* the collision. A key that arrives twice with two different values is
recorded as ambiguous, written nowhere, and counted in the report. A wrong score
is worse than a missing one.

Key selection is measured, not assumed. Candidate keys are built for both sides
(``prcfo``, then the full ``png_path``, then ``file_name`` --- never a basename
computed behind the caller's back when a real column exists) and the one that
actually matches the most rows wins, ties going to the earliest in
:data:`KEY_PRIORITY`. The chosen key and every count are printed, because a
merge that matched 3 of 40000 rows used to look exactly like one that matched
all of them.

Atomicity
---------
Python's :mod:`sqlite3` opens an implicit transaction for DML only, so an
``ALTER TABLE`` runs in autocommit and lands immediately --- the same trap
:func:`spacr.utils.rename_columns_in_db` was fixed for. The transaction here is
opened explicitly and rolled back on any error, so an interrupted merge leaves
the table exactly as it was: no half-added column, no half-scored rows.
"""

from __future__ import annotations

import math
import os
import sqlite3
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

__all__ = [
    "PNG_TABLE",
    "CV_SCORE_COLUMN",
    "CV_CLASS_COLUMN",
    "ML_SCORE_COLUMN",
    "ML_CLASS_COLUMN",
    "ANNOTATE_ENCODED_COLUMNS",
    "KEY_PRIORITY",
    "MergeReport",
    "crop_name_metadata",
    "migrate_prediction_columns",
    "merge_prediction_results",
    "merge_cv_predictions",
    "merge_ml_predictions",
]

#: Table holding one row per object crop.
PNG_TABLE = "png_list"

#: Positive-class probability from the convolutional classifier. Spelled
#: ``pred`` because that is what it has always been called, and because
#: ``spacr.settings`` (``dependent_variable='pred'``), ``spacr.submodules``,
#: ``spacr.plot`` and ``spacr.active_learning.PRED_COLUMN_CANDIDATES`` all read
#: that name; renaming it would break saved settings files for a tidier column.
CV_SCORE_COLUMN = "pred"
#: Thresholded class from the convolutional classifier
#: (:func:`spacr.utils.process_vision_results`).
CV_CLASS_COLUMN = "cv_predictions"

#: Positive-class probability from the classical-ML classifier. This one is
#: new: the ML stage only ever wrote a class, never its confidence. Namespaced
#: rather than reusing ``pred`` precisely so it cannot collide with the CV
#: score -- that collision is what "CV and ML must coexist" is about.
ML_SCORE_COLUMN = "ml_pred"
#: Predicted class from the classical-ML classifier. Kept as ``predictions``,
#: the spelling :func:`spacr.utils.add_column_to_database` has always written,
#: because ``spacr.settings`` names it in two defaults that read ``png_list``
#: --- ``set_analyze_endodyogeny_defaults``' ``class_column`` and
#: ``set_default_analyze_screen``' ``heatmap_feature``. A tidier
#: ``ml_predictions`` would silently break both for anyone who re-ran the ML
#: stage, and the name it would be tidier than is not ambiguous in practice:
#: the CV stage writes ``cv_predictions``, so the two never share a column.
ML_CLASS_COLUMN = "predictions"

#: ``png_list`` class columns an old spaCR wrote in the Annotate app's label
#: encoding. :func:`spacr.utils.add_column_to_database` replaced every ``0``
#: with a ``2`` before storing, because the Annotate app labels classes 1 and
#: 2 -- so the database disagreed with the ``results.csv`` written by the same
#: run. :func:`migrate_prediction_columns` puts them back.
ANNOTATE_ENCODED_COLUMNS: Tuple[str, ...] = (ML_CLASS_COLUMN,)

#: Candidate join keys, best first. Ties in match count are broken by this
#: order, so ``prcfo`` wins whenever it does as well as the alternatives.
KEY_PRIORITY: Tuple[str, ...] = ("prcfo", "png_path", "file_name")

#: Columns a results frame may carry the crop's name in, best first.
_NAME_COLUMNS: Tuple[str, ...] = ("path", "png_path", "file_name")

#: Per-crop-mode object-id columns :func:`spacr.utils.filepaths_to_database`
#: writes ('o<n>' strings). Used to rebuild ``prcfo`` when a table somehow
#: lacks the column but still carries the metadata it is made of.
_OBJECT_ID_COLUMNS: Tuple[str, ...] = (
    "cell_id", "nucleus_id", "pathogen_id", "cytoplasm_id", "object",
)

#: Metadata columns ``prcfo`` is assembled from, in order.
_PRCFO_METADATA: Tuple[str, ...] = ("plateID", "rowID", "columnID", "fieldID")

#: What :func:`spacr.utils._map_wells_png` returns for a name it cannot parse.
_UNPARSED = "error"

#: SQLite's three spellings of the implicit row id, least likely to be shadowed
#: first. **Not paranoia**: ``png_list`` has a column called ``rowID``, SQLite
#: identifiers are case-insensitive, and a table that declares a column named
#: ``rowid`` (in any case) makes the bare name resolve to *that column* rather
#: than to the row id. So on a real measurements database ``SELECT rowid`` from
#: ``png_list`` returns the plate row -- ``'r1'``, ``'r2'`` -- and
#: ``UPDATE ... WHERE rowid = 'r1'`` rewrites **every crop in plate row 1**.
#: The merge this module replaces did exactly that; it went unnoticed because
#: its tests built ``png_list`` by hand without the ``rowID`` column.
_ROWID_ALIASES: Tuple[str, ...] = ("_rowid_", "oid", "rowid")

_MISSING = object()


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def _quote(identifier: str) -> str:
    """Return ``identifier`` quoted for SQLite.

    Column and table names reach this module from settings files and from
    caller keyword arguments, and were previously interpolated into
    ``ALTER TABLE``/``UPDATE`` raw. A name with a space, a reserved word or a
    quote character produced a syntax error at best.

    :param identifier: table or column name.
    :returns: the name wrapped in double quotes, internal quotes doubled.
    :raises ValueError: if ``identifier`` is not a non-empty string.
    """
    if not isinstance(identifier, str) or not identifier:
        raise ValueError(f"Invalid SQL identifier: {identifier!r}")
    return '"' + identifier.replace('"', '""') + '"'


def _rowid_alias(table_columns: Sequence[str]) -> str:
    """Return a spelling of the implicit row id this table does not shadow.

    :param table_columns: the table's declared column names.
    :returns: ``'_rowid_'``, ``'oid'`` or ``'rowid'``.
    :raises ValueError: if the table declares all three, leaving no way to
        address a row by identity.
    """
    taken = {str(name).lower() for name in table_columns}
    for alias in _ROWID_ALIASES:
        if alias not in taken:
            return alias
    raise ValueError(
        "Table declares columns named rowid, oid and _rowid_, so its rows "
        "cannot be addressed by row id; rename one of them before merging.")


def _sql_value(value, sql_type: str):
    """Coerce one DataFrame cell to something SQLite can store.

    NaN becomes NULL rather than the float ``nan`` SQLite would otherwise
    store as a REAL, and numpy scalars are unwrapped so the driver does not
    have to guess.

    A value the declared type cannot hold becomes NULL rather than failing the
    whole merge, and a value of no type SQLite can bind at all is stored as its
    text -- one odd cell in a results frame must not cost 40000 scored rows.

    :param value: the cell.
    :param sql_type: ``'REAL'``, ``'INTEGER'``, or anything else (stored as a
        bindable scalar).
    :returns: ``None``, or something :mod:`sqlite3` can bind.
    """
    if value is None:
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and math.isnan(value):
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        # pd.isna of a list/array is elementwise, so `if` on it raises. Not a
        # missing value; fall through and let the type handling below decide.
        pass
    if sql_type == "INTEGER":
        try:
            return int(value)
        except (TypeError, ValueError):
            return None
    if sql_type == "REAL":
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
    if isinstance(value, (int, float, str, bytes)):
        return value
    return str(value)


def _clean_key(value) -> Optional[str]:
    """Return ``value`` as a usable key string, or ``None`` when it is not one."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    if not text or text == _UNPARSED:
        return None
    return text


def _values_equal(left: Sequence, right: Sequence) -> bool:
    """Whether two rows of already-coerced values are the same.

    A plain comparison is enough because every element has been through
    :func:`_sql_value`, which turns NaN into ``None`` -- so the one case that
    would need special handling (``nan != nan``) cannot reach here.
    """
    return tuple(left) == tuple(right)


def _name_column(frame: pd.DataFrame) -> Optional[str]:
    """Return the column holding the crop's name/path, or ``None``."""
    for name in _NAME_COLUMNS:
        if name in frame.columns:
            return name
    return None


def _prcfo_from_metadata(frame: pd.DataFrame) -> Optional[pd.Series]:
    """Rebuild ``prcfo`` from the metadata columns ``png_list`` carries.

    ``filepaths_to_database`` writes ``plateID``/``rowID``/``columnID``/
    ``fieldID`` (plus ``timeID`` for a timelapse) and one ``<mode>_id`` column
    holding the ``'o<n>'`` object id, which is exactly what ``prcfo`` is joined
    from. A table that has the metadata but not the key can therefore still be
    merged into.

    :param frame: rows of the target table.
    :returns: a Series of keys, or ``None`` if the metadata is not all there.
    """
    from .utils import _time_column

    if not all(col in frame.columns for col in _PRCFO_METADATA):
        return None
    object_col = None
    for candidate in _OBJECT_ID_COLUMNS:
        if candidate in frame.columns:
            object_col = candidate
            break
    if object_col is None:
        return None

    parts: List[str] = list(_PRCFO_METADATA)
    time_col = _time_column(frame.columns)
    if time_col is not None:
        parts.append(time_col)
    parts.append(object_col)

    # A row missing any one component has no key at all -- an empty slot would
    # make 'plate1_r1__f1_o3' collide with a genuinely different object.
    pieces = []
    valid = None
    for col in parts:
        text = frame[col].astype("object").map(_clean_key)
        present = text.notna()
        valid = present if valid is None else (valid & present)
        pieces.append(text.fillna(""))
    key = pieces[0].astype(str)
    for piece in pieces[1:]:
        key = key + "_" + piece.astype(str)
    return key.astype("object").where(valid, other=None)


def crop_name_metadata(names, timelapse: bool = False) -> pd.DataFrame:
    """Parse spaCR crop file names into the metadata ``prcfo`` is built from.

    Every consumer of a classifier's results has to answer the same question --
    which object is this crop? -- and there is exactly one right way to answer
    it: :func:`spacr.utils._map_wells_png`, the parser
    :func:`spacr.utils.filepaths_to_database` used to write these very columns
    onto ``png_list``. Re-deriving them with the writer's own parser is what
    guarantees the two sides cannot drift apart.

    It also recovers what a positional guess gets wrong.
    :func:`spacr.utils.process_vision_results` takes the object id as
    ``path.split('_')[3]``, which on a *timelapse* crop
    (``plate_well_field_time_object``) is the **timepoint**, not the object.

    :param names: crop names or paths; only the basename is parsed.
    :param timelapse: whether the names carry a timepoint component.
    :returns: a DataFrame aligned to ``names`` with ``plateID``, ``rowID``,
        ``columnID``, ``fieldID``, ``timeID`` (only when ``timelapse``),
        ``object_label`` (the bare integer id, no ``'o'`` prefix) and
        ``prcfo``. A name that cannot be parsed gives ``None`` throughout.
    """
    from .utils import _map_wells_png

    if not isinstance(names, pd.Series):
        names = pd.Series(list(names))

    columns = ["plateID", "rowID", "columnID", "fieldID"]
    if timelapse:
        columns.append("timeID")
    columns += ["prcfo", "object_label"]

    cache: Dict[str, Tuple] = {}
    empty = (None,) * len(columns)

    def convert(value):
        name = _clean_key(value)
        if name is None:
            return empty
        base = os.path.basename(name)
        if base not in cache:
            parsed = tuple(_clean_key(v) for v in
                           _map_wells_png(base, timelapse=timelapse))
            # 'error' in any position means the whole name failed to parse.
            cache[base] = empty if any(v is None for v in parsed) else parsed
        return cache[base]

    frame = pd.DataFrame([convert(v) for v in names], columns=columns,
                         index=names.index)
    # object_label without the 'o': that is the spelling the object tables use.
    frame["object_label"] = frame["object_label"].map(
        lambda v: None if v is None else v[1:] if v.startswith("o") else v)
    return frame


def _prcfo_from_names(names: pd.Series, timelapse: bool) -> pd.Series:
    """Derive just the ``prcfo`` key from crop file names."""
    return crop_name_metadata(names, timelapse=timelapse)["prcfo"]


# ---------------------------------------------------------------------------
# key construction
# ---------------------------------------------------------------------------

def _db_keys(kind: str, frame: pd.DataFrame) -> Optional[pd.Series]:
    """Build the ``kind`` key for rows already in the database."""
    if kind == "prcfo":
        if "prcfo" in frame.columns:
            return frame["prcfo"].map(_clean_key)
        return _prcfo_from_metadata(frame)
    if kind == "png_path":
        if "png_path" in frame.columns:
            return frame["png_path"].map(_clean_key)
        return None
    if kind == "file_name":
        if "file_name" in frame.columns:
            return frame["file_name"].map(_clean_key)
        if "png_path" in frame.columns:
            return frame["png_path"].map(
                lambda v: (lambda c: None if c is None else os.path.basename(c))(_clean_key(v)))
        return None
    raise ValueError(f"Unknown join key {kind!r}; expected one of {KEY_PRIORITY}")


def _result_keys(kind: str, results: pd.DataFrame, timelapse: bool) -> Optional[pd.Series]:
    """Build the ``kind`` key for rows of a classifier's results frame."""
    if kind == "prcfo":
        if "prcfo" in results.columns:
            return results["prcfo"].map(_clean_key)
        if results.index.name == "prcfo":
            return pd.Series(results.index, index=results.index).map(_clean_key)
        name_col = _name_column(results)
        if name_col is None:
            return None
        return _prcfo_from_names(results[name_col], timelapse)
    if kind == "png_path":
        for name in ("png_path", "path"):
            if name in results.columns:
                return results[name].map(_clean_key)
        return None
    if kind == "file_name":
        name_col = _name_column(results)
        if name_col is None:
            return None
        return results[name_col].map(
            lambda v: (lambda c: None if c is None else os.path.basename(c))(_clean_key(v)))
    raise ValueError(f"Unknown join key {kind!r}; expected one of {KEY_PRIORITY}")


def _choose_key(results: pd.DataFrame, db_frame: pd.DataFrame,
                timelapse: bool) -> Tuple[str, pd.Series, pd.Series]:
    """Pick the join key that actually matches the most database rows.

    Measured rather than assumed: a key is only better if it lands on more
    rows. Ties go to the earliest entry of :data:`KEY_PRIORITY`, which is how
    ``prcfo`` wins the (normal) case where every candidate matches everything.

    :returns: ``(kind, result_keys, db_keys)``.
    """
    best = None
    for kind in KEY_PRIORITY:
        result_keys = _result_keys(kind, results, timelapse)
        db_keys = _db_keys(kind, db_frame)
        if result_keys is None or db_keys is None:
            continue
        wanted = set(result_keys.dropna().tolist())
        matched = int(db_keys.isin(wanted).sum()) if wanted else 0
        if best is None or matched > best[0]:
            best = (matched, kind, result_keys, db_keys)
    if best is None:
        raise ValueError(
            "No usable join key: the results frame must carry 'prcfo' (or an "
            f"index named 'prcfo'), or one of {_NAME_COLUMNS}, and the target "
            "table must carry 'prcfo', 'png_path' or the plate metadata "
            "'prcfo' is built from.")
    return best[1], best[2], best[3]


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

@dataclass
class MergeReport:
    """What one merge did, in numbers.

    Returned by :func:`merge_prediction_results` and printed by it. Every
    count is here because a merge that matched three rows of forty thousand
    used to be indistinguishable from one that matched all of them.
    """

    table: str
    key: str
    columns: Tuple[str, ...] = ()
    db_rows: int = 0
    result_rows: int = 0
    #: Database rows that got a value written.
    matched_rows: int = 0
    #: Distinct keys that matched something.
    matched_keys: int = 0
    #: Database rows left untouched because no result row had their key.
    unmatched_db_rows: int = 0
    #: Result rows whose key exists nowhere in the table.
    unmatched_result_rows: int = 0
    #: Result rows whose name could not be parsed into a key at all.
    unparsed_result_rows: int = 0
    #: Keys that arrived more than once carrying *different* values. Written
    #: nowhere: a wrong score is worse than a missing one.
    ambiguous_keys: int = 0
    #: Result rows involved in those collisions.
    ambiguous_result_rows: int = 0
    #: Database rows that shared a key with another row and so were given the
    #: same value. Not an error -- one object, several crop modes -- but worth
    #: seeing.
    fanout_rows: int = 0
    #: Legacy columns repaired on the way in, as
    #: ``(table, column, rows_repaired)``.
    repaired: Tuple[Tuple[str, str, int], ...] = ()
    #: Columns created by this merge.
    added_columns: Tuple[str, ...] = ()

    def summary(self) -> str:
        """Return the human-readable multi-line report."""
        cols = ", ".join(self.columns)
        lines = [
            f"Merged {cols} into {self.table} on '{self.key}': "
            f"{self.matched_rows}/{self.db_rows} rows matched"
        ]
        if self.db_rows and not self.matched_rows:
            lines.append(
                f"  !! NOTHING MATCHED. {self.result_rows} result row(s) and "
                f"{self.db_rows} {self.table} row(s) share no '{self.key}' "
                f"value, so no score was written. The results probably come "
                f"from a different experiment than {self.table}.")
        for label, value in (
            ("result row(s) had no matching row in this database",
             self.unmatched_result_rows),
            ("result row(s) had a name no key could be parsed from",
             self.unparsed_result_rows),
            (f"{self.table} row(s) were left unscored", self.unmatched_db_rows),
            (f"{self.table} row(s) shared a key with another row and got the "
             f"same value", self.fanout_rows),
        ):
            if value:
                lines.append(f"  {value} {label}")
        if self.ambiguous_keys:
            lines.append(
                f"  !! {self.ambiguous_keys} key(s) arrived from "
                f"{self.ambiguous_result_rows} result rows with conflicting "
                f"values and were NOT written. Two crops sharing a '{self.key}' "
                f"cannot be told apart -- this happens when two source folders "
                f"give their plates the same name.")
        for _table, column, count in self.repaired:
            lines.append(
                f"  repaired {count} legacy row(s) of '{column}' (the Annotate "
                f"app's 1/2 class encoding back to the model's 0/1)")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()


# ---------------------------------------------------------------------------
# migration
# ---------------------------------------------------------------------------

def _execute(cursor, sql: str) -> None:
    """Run one write statement.

    A named seam, like :func:`_execute_updates`, so a test can make the write
    fail after it has taken effect and prove the transaction rolls it back.
    """
    cursor.execute(sql)


def migrate_prediction_columns(db_path, table: str = PNG_TABLE,
                               verbose: bool = True) -> List[Tuple[str, str, int]]:
    """Put a legacy prediction column back into the encoding it claims to be in.

    :func:`spacr.utils.add_column_to_database` --- the ML stage's old write
    path --- replaced every ``0`` with a ``2`` on the way into the database,
    because the Annotate app labels classes 1 and 2. Nothing else did that, so
    ``png_list.predictions`` said ``2`` where ``results.csv`` from the very same
    run said ``0``: one number, two meanings, depending on which file you
    opened. This repairs it, so a database written before the change reads
    correctly with no manual action --- the same repair-on-read contract
    :func:`spacr.utils.rename_columns_in_db` has, and the same properties:

    * **Idempotent.** After the repair the column holds 0s and 1s, which is not
      the encoding this looks for, so a second pass does nothing.
    * **Never destructive.** The substitution is only reversed when the column
      holds nothing but 1s and 2s. A three-class model's genuine class 2 is
      indistinguishable from a mangled 0, and guessing would be worse than
      leaving it exactly as it is.
    * **All or nothing.** One explicit transaction, rolled back on any error.
      The driver opens one for DML by itself, but not for the ``PRAGMA``/DDL
      this shares a connection with, and a half-repaired column would be a
      column in two encodings at once.

    :param db_path: SQLite file. A missing file is a no-op.
    :param table: table to migrate. Default ``'png_list'``.
    :param verbose: print each repair.
    :returns: list of ``(table, column, rows_repaired)``.
    """
    if not os.path.exists(str(db_path)):
        return []

    repaired: List[Tuple[str, str, int]] = []
    con = sqlite3.connect(str(db_path), timeout=30)
    con.isolation_level = None
    try:
        cur = con.cursor()
        tables = {row[0] for row in
                  cur.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if table not in tables:
            return []
        cols = [row[1] for row in cur.execute(f"PRAGMA table_info({_quote(table)})")]
        pending = [c for c in ANNOTATE_ENCODED_COLUMNS if c in cols]
        if not pending:
            return []

        cur.execute("BEGIN")
        try:
            for column in pending:
                quoted = _quote(column)
                distinct = {row[0] for row in cur.execute(
                    f"SELECT DISTINCT {quoted} FROM {_quote(table)} "
                    f"WHERE {quoted} IS NOT NULL")}
                if not distinct or not distinct.issubset({1, 2}) or 2 not in distinct:
                    continue
                _execute(cur, f"UPDATE {_quote(table)} SET {quoted} = 0 "
                              f"WHERE {quoted} = 2")
                repaired.append((table, column, cur.rowcount))
            cur.execute("COMMIT")
        except BaseException:
            cur.execute("ROLLBACK")
            raise
    finally:
        con.close()

    if verbose:
        for _table, column, count in repaired:
            print(f"Repaired {count} row(s) of `{_table}`.`{column}`: the "
                  f"Annotate app's 1/2 class encoding back to the model's 0/1")
    return repaired


# ---------------------------------------------------------------------------
# the merge
# ---------------------------------------------------------------------------

def _execute_updates(cursor, sql: str, updates: Sequence[Tuple]) -> None:
    """Apply the prepared UPDATE statements.

    A named seam rather than an inline ``executemany`` so a test can make the
    write fail halfway and prove the transaction rolls the whole merge back.
    """
    cursor.executemany(sql, updates)


def merge_prediction_results(results, db_path, columns, table: str = PNG_TABLE,
                             key: str = "auto", timelapse: Optional[bool] = None,
                             verbose: bool = True) -> Optional[MergeReport]:
    """Write a classifier's per-object results onto the rows of ``table``.

    Shared by Classify (CV) and Classify (ML) --- see the module docstring for
    why the key is ``prcfo``, why an ambiguous key is refused rather than
    guessed at, and why the whole thing is one transaction.

    :param results: DataFrame of per-object results. Must carry the source
        columns named in ``columns``, plus something to key on: a ``prcfo``
        column, an index named ``prcfo``, or a ``path``/``png_path``/
        ``file_name`` column holding spaCR crop names.
    :param db_path: SQLite database. A missing file is reported and skipped.
    :param columns: mapping of *database* column name to
        ``(results column, 'REAL' | 'INTEGER')``.
    :param table: target table. Default ``'png_list'``.
    :param key: ``'auto'`` (measure every candidate and use the best) or one of
        :data:`KEY_PRIORITY` to force it.
    :param timelapse: whether crop names carry a timepoint. ``None`` detects it
        from the presence of a time column on the table.
    :param verbose: print the report.
    :returns: a :class:`MergeReport`, or ``None`` if the database is missing.
    :raises KeyError: if ``results`` lacks one of the source columns.
    :raises sqlite3.OperationalError: if ``table`` does not exist.
    """
    db_path = str(db_path)
    if not os.path.exists(db_path):
        print(f"Database not found at {db_path}; skipping merge.")
        return None

    if not isinstance(results, pd.DataFrame):
        results = pd.DataFrame(results)

    spec = {db_col: (src if isinstance(src, (tuple, list)) else (src, "REAL"))
            for db_col, src in columns.items()}
    missing = [src for src, _type in spec.values() if src not in results.columns]
    if missing:
        raise KeyError(
            f"merge_prediction_results: results frame has no column(s) "
            f"{missing}; it carries {list(results.columns)[:12]}"
            + (" ..." if len(results.columns) > 12 else ""))

    repaired = migrate_prediction_columns(db_path, table=table, verbose=False)

    con = sqlite3.connect(db_path, timeout=30)
    # Explicit transaction control: ALTER TABLE would otherwise autocommit and
    # an interrupted merge would leave a column added but no rows scored.
    con.isolation_level = None
    try:
        cur = con.cursor()
        cur.execute("BEGIN")
        try:
            report = _merge_locked(cur, results, spec, table, key, timelapse,
                                   repaired)
            cur.execute("COMMIT")
        except BaseException:
            cur.execute("ROLLBACK")
            raise
    finally:
        con.close()

    if verbose:
        print(report.summary())
    return report


def _merge_locked(cur, results: pd.DataFrame, spec: Mapping[str, Tuple[str, str]],
                  table: str, key: str, timelapse: Optional[bool],
                  repaired: Sequence[Tuple[str, str, int]]) -> MergeReport:
    """Do the merge inside an already-open transaction."""
    from .utils import _time_column

    quoted_table = _quote(table)
    # Raises sqlite3.OperationalError('no such table: ...') -- the loud,
    # correct failure for a database that was never measured.
    cur.execute(f"SELECT * FROM {quoted_table} LIMIT 0")
    table_columns = [d[0] for d in cur.description]

    if timelapse is None:
        timelapse = _time_column(table_columns) is not None

    key_columns = [c for c in table_columns
                   if c in ("prcfo", "png_path", "file_name")
                   or c in _PRCFO_METADATA or c in _OBJECT_ID_COLUMNS
                   or c in ("timeID", "time_id")]
    rowid = _rowid_alias(table_columns)
    select = ", ".join([rowid] + [_quote(c) for c in key_columns])
    rows = cur.execute(f"SELECT {select} FROM {quoted_table}").fetchall()
    rowids = [r[0] for r in rows]
    db_frame = pd.DataFrame([r[1:] for r in rows], columns=key_columns)

    if key == "auto":
        kind, result_keys, db_keys = _choose_key(results, db_frame, timelapse)
    else:
        kind = key
        result_keys = _result_keys(kind, results, timelapse)
        db_keys = _db_keys(kind, db_frame)
        if result_keys is None or db_keys is None:
            raise ValueError(
                f"Join key {kind!r} cannot be built: the results frame or "
                f"{table} does not carry what it is made of.")

    # -- collapse the results into key -> values, refusing collisions --
    value_frames = {db_col: results[src] for db_col, (src, _t) in spec.items()}
    types = {db_col: sql_type for db_col, (_src, sql_type) in spec.items()}
    order = list(spec)

    lookup: Dict[str, Tuple] = {}
    conflicting: Dict[str, int] = {}
    unparsed = 0
    key_list = list(result_keys)
    columns_by_row = [list(value_frames[db_col]) for db_col in order]

    for idx, row_key in enumerate(key_list):
        if row_key is None:
            unparsed += 1
            continue
        values = tuple(_sql_value(columns_by_row[c][idx], types[order[c]])
                       for c in range(len(order)))
        previous = lookup.get(row_key, _MISSING)
        if previous is _MISSING:
            lookup[row_key] = values
        elif not _values_equal(previous, values):
            conflicting[row_key] = conflicting.get(row_key, 1) + 1

    for row_key in conflicting:
        lookup.pop(row_key, None)

    # -- add the columns we are about to write --
    added = []
    for db_col in order:
        if db_col not in table_columns:
            cur.execute(f"ALTER TABLE {quoted_table} ADD COLUMN "
                        f"{_quote(db_col)} {types[db_col]}")
            added.append(db_col)

    # -- match --
    updates = []
    matched_keys = set()
    for position, row_key in enumerate(db_keys):
        values = lookup.get(row_key) if row_key is not None else None
        if values is None:
            continue
        updates.append(tuple(values) + (rowids[position],))
        matched_keys.add(row_key)

    if updates:
        assignments = ", ".join(f"{_quote(c)} = ?" for c in order)
        _execute_updates(
            cur, f"UPDATE {quoted_table} SET {assignments} WHERE {rowid} = ?",
            updates)

    db_key_set = {k for k in db_keys if k is not None}
    unmatched_results = sum(
        1 for k in key_list
        if k is not None and k not in conflicting and k not in db_key_set)

    return MergeReport(
        table=table,
        key=kind,
        columns=tuple(order),
        db_rows=len(rowids),
        result_rows=len(key_list),
        matched_rows=len(updates),
        matched_keys=len(matched_keys),
        unmatched_db_rows=len(rowids) - len(updates),
        unmatched_result_rows=unmatched_results,
        unparsed_result_rows=unparsed,
        ambiguous_keys=len(conflicting),
        ambiguous_result_rows=sum(conflicting.values()),
        fanout_rows=len(updates) - len(matched_keys),
        repaired=tuple(repaired),
        added_columns=tuple(added),
    )


# ---------------------------------------------------------------------------
# the two callers
# ---------------------------------------------------------------------------

def merge_cv_predictions(df, db_path, table: str = PNG_TABLE,
                         score_col: str = CV_SCORE_COLUMN,
                         class_col: str = CV_CLASS_COLUMN,
                         score_source: str = "pred",
                         class_source: str = "cv_predictions",
                         verbose: bool = True) -> Optional[MergeReport]:
    """Merge :func:`spacr.deep_spacr.apply_model_to_tar` results into ``table``.

    :param df: frame from ``apply_model_to_tar`` -> ``process_vision_results``
        (``path``, ``pred``, ``cv_predictions``).
    :param db_path: SQLite database to write into.
    :param table: target table. Default ``'png_list'``.
    :param score_col: database column for the probability.
    :param class_col: database column for the thresholded class.
    :param score_source: column of ``df`` the probability comes from.
    :param class_source: column of ``df`` the class comes from.
    :param verbose: print the report.
    :returns: a :class:`MergeReport`, or ``None`` if the database is missing.
    """
    return merge_prediction_results(
        df, db_path,
        {score_col: (score_source, "REAL"), class_col: (class_source, "INTEGER")},
        table=table, verbose=verbose)


def merge_ml_predictions(df, db_path, table: str = PNG_TABLE,
                         score_col: str = ML_SCORE_COLUMN,
                         class_col: str = ML_CLASS_COLUMN,
                         verbose: bool = True) -> Optional[MergeReport]:
    """Merge :func:`spacr.ml.ml_analysis` results into ``table``.

    ``ml_analysis`` returns the predicted class in ``predictions`` and the
    per-class probabilities in ``prediction_probability_class_<i>``; the
    positive-class probability is taken from class 1 when the model produced
    one, and skipped when it did not (a one-class fit) rather than inventing a
    column.

    :param df: the scored frame -- ``ml_analysis`` output ``[0]``.
    :param db_path: SQLite database to write into.
    :param table: target table. Default ``'png_list'``.
    :param score_col: database column for the probability.
    :param class_col: database column for the predicted class.
    :param verbose: print the report.
    :returns: a :class:`MergeReport`, or ``None`` if the database is missing,
        or if the frame carries no prediction column at all.
    """
    columns: Dict[str, Tuple[str, str]] = {}
    if "predictions" in df.columns:
        columns[class_col] = ("predictions", "INTEGER")
    if "prediction_probability_class_1" in df.columns:
        columns[score_col] = ("prediction_probability_class_1", "REAL")
    if not columns:
        print("No prediction columns on the ML results frame; skipping merge "
              f"into {db_path}.")
        return None
    return merge_prediction_results(df, db_path, columns, table=table,
                                    verbose=verbose)
