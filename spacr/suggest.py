"""Turn a retrained model's scores into labels you can accept or throw away.

Annotating is the slow part of every screen, and the first two hundred crops
already contain most of what separates the classes. Annotate can already fit a
model on those and re-rank the queue by uncertainty -- what it could not do is
WRITE the model's opinion down as a proposed label that a reviewer accepts or
rejects in bulk. That is all this module adds.

IT DELEGATES THE MODEL, DELIBERATELY. :func:`spacr.active_learning.retrain_round`
already fits on every label, scores on a GROUPED held-out split so the number
is not an artefact of 190 labels coming from one well, and writes per-class
probabilities into ``png_list``. Re-fitting here would be a second model with
a second answer, drifting from the one the queue is ranked by -- and a first
draft of this file did exactly that before the existing one was found. What is
new is the three rules below, not the classifier.

THREE RULES THAT PROTECT THE ANNOTATIONS
========================================
1. A SUGGESTION IS NEVER CONFUSABLE WITH A DECISION. Suggestions are stored as
   their own values -- a suggested 1 becomes 11 -- so nothing that reads the
   annotation column can mistake one for a human's answer, and a run that goes
   wrong is undone by deleting a value rather than by remembering which rows
   were touched.
2. A SUGGESTION NEVER OVERWRITES AN ANNOTATION. Writes carry ``IS NULL``. Not
   when the model is confident, not on a re-run. A human's labels are the
   ground truth the model was fitted on; losing one silently would cost hours
   and would not be noticed until a run came out wrong.
3. NO CONFIDENCE FLOOR, SO THE ORDER CARRIES THE DOUBT. Every unannotated crop
   gets a suggestion, sorted most-confident first, and the reviewer stops
   where they stop agreeing. A threshold would make that decision for them,
   with a number nobody chose.
4. A JUDGEMENT IS RECORDED, NOT INFERRED (item 512). Confirming a suggestion
   turns it into an ordinary label and rejecting one clears it -- and both
   are written down beside the column, in ``<column>_verdict``, as the class
   that was confirmed (``+c``) or rejected (``-c``). A rejection is
   information the next round can train on -- "not class 1" is an example
   of class 2 in a two-class column -- where a NULL would have been silence.
   The verdict column is added the first time a source is opened, so tables
   made before it existed gain it without a migration step.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from .selection import _PandasOnFirstUse

pd = _PandasOnFirstUse(globals())


def _connect_writable(db_path):
    """A WRITE connection that waits for Measure rather than failing.

    Same reason as :func:`_connect_read_only`, from the other side: these
    two write annotations back, and a Measure run holding the write lock
    would otherwise turn "accept these suggestions" into "database is
    locked" after five seconds. Accepting is a user action they will
    simply repeat, so failing fast buys nothing and waiting costs nothing.
    """
    from .database_concurrency import connect

    return connect(db_path)


def _connect_read_only(db_path):
    """A read-only connection that WAITS for Measure rather than failing.

    `sqlite3.connect(..., mode=ro)` takes SQLite's five-second default and
    then raises "database is locked" -- which, against a measurements.db
    a Measure run is still writing, is a report that fails for a reason
    that has nothing to do with the data. `database_concurrency.connect`
    sets `busy_timeout` from its own `timeout` and opens with
    `query_only=ON`, so a reader waits out a writer's transaction instead
    of racing it.

    Pinned by `test_no_connection_relies_on_sqlites_five_second_default`.
    """
    from .database_concurrency import connect

    return connect(db_path, readonly=True)


def _has_column(db, table: str, column: str) -> bool:
    """Whether ``table`` really has ``column``, asked before it is used.

    SQLITE READS A DOUBLE-QUOTED NAME THAT IS NOT A COLUMN AS A STRING
    LITERAL rather than failing, so ``WHERE "cell_class" > 10`` against a
    table without that column is ``WHERE 'cell_class' > 10`` -- and a TEXT
    value compares greater than every integer, so it is TRUE for every row.
    :func:`pending_suggestions` therefore reported the WHOLE SCREEN as
    waiting to be accepted before a single suggestion had been made, in the
    Class counts dialog and in the Suggest menu, and its own docstring said
    it returned 0. No quoting avoids this, and no error is raised to catch,
    so the column has to be looked up.

    :param db: an open connection.
    :param table: the table to look in.
    :param column: the column name to look for.
    :returns: True only when the column exists.
    """
    try:
        rows = db.execute(f'PRAGMA table_info("{table}")').fetchall()
    except sqlite3.Error:
        return False
    return any(row[1] == column for row in rows)

#: Added to a class value to mark it a suggestion rather than an answer. Ten,
#: because the annotation column holds small integers: a suggested 1 becomes
#: 11 and cannot collide with a real 2 or 3.
SUGGESTION_OFFSET = 10


@dataclass
class Suggestions:
    """What a suggestion run proposes, and how far to trust it.

    :param frame: one row per unannotated crop with ``png_path``,
        ``suggested``, ``stored`` and ``confidence``, most confident first.
    :param note: what a reader must be told before accepting in bulk.
    :param scored: how many crops carried usable scores.
    :param classes: the class values the scores describe, in column order.
    """

    frame: pd.DataFrame
    note: str = ""
    scored: int = 0
    classes: List[int] = field(default_factory=list)


def _score_columns(columns: Sequence[str]) -> List[str]:
    """The per-class probability columns a retrain wrote, in class order.

    :param columns: the crop table's columns.
    :returns: the score columns, ordered by their class index.
    """
    from .active_learning import ROUND_PRED_PREFIX

    found = [c for c in columns if c.startswith(ROUND_PRED_PREFIX)]

    def index(name):
        """The integer suffix of a score column, for ordering."""
        try:
            return int(name[len(ROUND_PRED_PREFIX):])
        except ValueError:
            return 10 ** 6
    return sorted(found, key=index)


def suggest_from_scores(db_path: str, annotation_column: str, *,
                        png_table: str = "png_list",
                        classes: Optional[Sequence[int]] = None,
                        withhold_rejected: bool = True,
                        ) -> Suggestions:
    """Read the last retrain's probabilities and propose a label for each crop.

    Reads rather than re-fits, so the suggestion a reviewer sees and the
    ranking the queue uses come from ONE model. A second fit here would drift
    from it and there would be no way to tell which was right.

    :param db_path: path to a ``measurements.db``.
    :param annotation_column: the column holding the labels.
    :param png_table: the crop table.
    :param classes: the class value each score column stands for. Defaults to
        the values already present in the annotation column, in order, which
        is what the retrain encoded them from.
    :param withhold_rejected: leave out every crop whose suggestion the
        annotator REJECTED (a negative ``<column>_verdict``), so a rejected
        crop is not suggested again in a later round. It still trains as an example of the other class
        (:func:`rejected_suggestions`); it is only not proposed.
    :returns: a :class:`Suggestions`; its frame is empty when nothing has been
        scored, and ``note`` says why.
    """
    from .active_learning import as_probabilities

    with _connect_read_only(db_path) as db:
        try:
            from .tabular import _read_query
            crops = _read_query(db, f'SELECT * FROM "{png_table}"',
                               report=None)
        except Exception:                                    # noqa: BLE001
            return Suggestions(pd.DataFrame(), note="no crop table")

    if crops.empty:
        return Suggestions(pd.DataFrame(), note="the crop table is empty")

    score_cols = _score_columns(crops.columns)
    if not score_cols:
        return Suggestions(
            pd.DataFrame(),
            note=("no crop has been scored yet -- press Retrain first, which "
                  "fits on the labels so far and writes the probabilities "
                  "this reads"))

    if annotation_column not in crops.columns:
        crops[annotation_column] = np.nan
    labels = pd.to_numeric(crops[annotation_column], errors="coerce")

    if classes is None:
        seen = sorted({int(v) for v in labels.dropna().unique()
                       if int(v) < SUGGESTION_OFFSET})
        classes = seen or list(range(len(score_cols)))
    classes = list(classes)[:len(score_cols)]
    if not classes:
        return Suggestions(pd.DataFrame(),
                           note="nothing has been annotated yet")

    unlabelled = crops[labels.isna()].copy()
    if unlabelled.empty:
        return Suggestions(pd.DataFrame(), classes=classes,
                           note="every crop already carries a value")
    verdict = verdict_column(annotation_column)
    if withhold_rejected and verdict in unlabelled.columns:
        judged = pd.to_numeric(unlabelled[verdict], errors="coerce")
        unlabelled = unlabelled[~(judged < 0)]
        if unlabelled.empty:
            return Suggestions(
                pd.DataFrame(), classes=classes,
                note=("every crop left without a value is one whose "
                      "suggestion was rejected, and a rejected crop is not "
                      "suggested again"))

    raw = unlabelled[score_cols[:len(classes)]].apply(
        pd.to_numeric, errors="coerce")
    usable = raw.notna().all(axis=1)
    unlabelled = unlabelled[usable]
    if unlabelled.empty:
        return Suggestions(pd.DataFrame(), classes=classes,
                           note="no unannotated crop carries a usable score")

    probabilities = as_probabilities(raw[usable].to_numpy(dtype=float))
    best = probabilities.argmax(axis=1)
    out = pd.DataFrame({
        "png_path": unlabelled["png_path"].to_numpy(),
        "suggested": [classes[i] for i in best],
        "confidence": probabilities.max(axis=1),
    })
    out["stored"] = out["suggested"] + SUGGESTION_OFFSET
    out = out.sort_values("confidence", ascending=False, ignore_index=True)
    return Suggestions(out, scored=len(out), classes=classes)


def write_suggestions(db_path: str, annotation_column: str,
                      suggestions: pd.DataFrame, *,
                      png_table: str = "png_list") -> int:
    """Store suggestions, and ONLY where nothing has been annotated.

    The ``IS NULL`` is rule 2 and is not an optimisation.

    :param db_path: path to a ``measurements.db``.
    :param annotation_column: the column to write into.
    :param suggestions: the frame from :func:`suggest_from_scores`.
    :param png_table: the crop table.
    :returns: how many rows were written.
    """
    if suggestions.empty or "png_path" not in suggestions.columns:
        return 0
    if "suggested" in suggestions.columns:
        collides = sorted({int(v) for v in suggestions["suggested"]
                           if pd.notna(v) and int(v) >= SUGGESTION_OFFSET})
        if collides:
            raise ValueError(
                f"class {collides[0]} cannot be suggested: values at or "
                f"above {SUGGESTION_OFFSET} collide with the suggestion "
                f"offset, so a suggested 1 and an answered "
                f"{SUGGESTION_OFFSET + 1} would be the same number. "
                f"Classes 1 to 9 are fine and always have been -- it is the "
                f"VALUE that collides, not the count.")
    with _connect_read_only(db_path) as db:
        if not _has_column(db, png_table, annotation_column):
            existing = []
        else:
            try:
                existing = sorted({
                    int(v[0]) for v in db.execute(
                        f'SELECT DISTINCT "{annotation_column}" FROM '
                        f'"{png_table}" WHERE "{annotation_column}" >= ?',
                        (SUGGESTION_OFFSET,)).fetchall()
                    if v and v[0] is not None})
            except sqlite3.Error:
                existing = []
    if existing:
        raise ValueError(
            f"{annotation_column} already holds {existing[0]}, at or above "
            f"the suggestion offset {SUGGESTION_OFFSET}. A suggested 1 is "
            f"stored as {SUGGESTION_OFFSET + 1}, so the two cannot be told "
            f"apart and a bulk KEEP would rewrite the answer to a 1. "
            f"Resolve any outstanding suggestions first; if these are real "
            f"answers, this column uses class values Suggest cannot mark, "
            f"and the offset would have to move above them.")

    rows = [(int(s), str(p)) for s, p in
            zip(suggestions["stored"], suggestions["png_path"])
            if pd.notna(s)]
    if not rows:
        return 0
    written = 0
    with _connect_writable(db_path) as db:
        for value, path in rows:
            cur = db.execute(
                f'UPDATE "{png_table}" SET "{annotation_column}" = ? '
                f'WHERE png_path = ? AND "{annotation_column}" IS NULL',
                (value, path))
            written += cur.rowcount
        db.commit()
    return written


def resolve_suggestions(db_path: str, annotation_column: str, *,
                        keep: bool, png_table: str = "png_list",
                        paths: Optional[Sequence[str]] = None) -> int:
    """Accept suggestions as annotations, or clear them away.

    Accepting rewrites the offset value to the real class; rejecting sets it
    back to NULL. Both act ONLY on suggestion values, so a human annotation
    caught by the same query is untouched either way.

    :param db_path: path to a ``measurements.db``.
    :param annotation_column: the column holding both.
    :param keep: True to accept, False to discard.
    :param png_table: the crop table.
    :param paths: restrict to these crops; None means every suggestion.
    :returns: how many rows changed.

    A bulk KEEP is a confirmation of every suggestion it keeps, so it is
    recorded in the verdict column too (when the column exists): the crops
    then wear the same mark as ones confirmed one at a time, and the count
    the screen shows agrees with what happened. A bulk THROW is not a
    judgement -- "I do not want to review these" is not "these are wrong"
    -- so it records nothing.
    """
    column = f'"{annotation_column}"'
    where = f"{column} > {SUGGESTION_OFFSET}"
    params: List = []
    if paths is not None:
        if not len(paths):
            return 0
        where += f" AND png_path IN ({','.join('?' * len(paths))})"
        params = list(paths)

    with _connect_writable(db_path) as db:
        if not _has_column(db, png_table, annotation_column):
            return 0
        if keep:
            verdict = verdict_column(annotation_column)
            if _has_column(db, png_table, verdict):
                db.execute(
                    f'UPDATE "{png_table}" SET "{verdict}" = {column} - '
                    f"{SUGGESTION_OFFSET} WHERE {where}", params)
            sql = (f'UPDATE "{png_table}" SET {column} = {column} - '
                   f"{SUGGESTION_OFFSET} WHERE {where}")
        else:
            sql = f'UPDATE "{png_table}" SET {column} = NULL WHERE {where}'
        cur = db.execute(sql, params)
        db.commit()
        return cur.rowcount


def pending_suggestions(db_path: str, annotation_column: str, *,
                        png_table: str = "png_list") -> int:
    """How many suggestions are waiting to be accepted or thrown away.

    :param db_path: path to a ``measurements.db``.
    :param annotation_column: the column holding them.
    :param png_table: the crop table.
    :returns: the count, or 0 when the column does not exist.
    """
    with _connect_read_only(db_path) as db:
        if not _has_column(db, png_table, annotation_column):
            return 0
        try:
            row = db.execute(
                f'SELECT COUNT(*) FROM "{png_table}" '
                f'WHERE "{annotation_column}" > ?', (SUGGESTION_OFFSET,)
            ).fetchone()
        except sqlite3.Error:
            return 0
    return int(row[0]) if row else 0


VERDICT_SUFFIX = "_verdict"


def verdict_column(annotation_column: str) -> str:
    """The column beside ``annotation_column`` that records judgements.

    One value per crop: ``+c`` when a suggested class ``c`` was confirmed,
    ``-c`` when it was rejected, NULL when nothing was judged. It lives in
    the crop table rather than in the screen so a judgement survives a
    restart and reaches the next round of training.

    :param annotation_column: the column the suggestions were written into.
    :returns: the verdict column's name.
    """
    return f"{annotation_column}{VERDICT_SUFFIX}"


def ensure_verdict_column(db_path: str, annotation_column: str, *,
                          png_table: str = "png_list") -> bool:
    """Add the verdict column to ``png_table`` if it is missing.

    Called when a source is opened, which is how a table made before the
    column existed gains it: an ``ALTER TABLE ... ADD COLUMN`` with no
    default is a metadata change in SQLite and rewrites no rows.

    :param db_path: path to a ``measurements.db``.
    :param annotation_column: the column whose judgements it will hold.
    :param png_table: the crop table.
    :returns: True when the column exists afterwards.
    """
    if not annotation_column:
        return False
    verdict = verdict_column(annotation_column)
    with _connect_writable(db_path) as db:
        try:
            rows = db.execute(f'PRAGMA table_info("{png_table}")').fetchall()
        except sqlite3.Error:
            return False
        if not rows:
            return False
        if any(row[1] == verdict for row in rows):
            return True
        safe = verdict.replace('"', '""')
        try:
            db.execute(f'ALTER TABLE "{png_table}" ADD COLUMN "{safe}" INTEGER')
            db.commit()
        except sqlite3.Error:
            return False
    return True


def fetch_verdicts(db_path: str, annotation_column: str,
                   paths: Sequence[str], *,
                   png_table: str = "png_list") -> Dict[str, int]:
    """The recorded judgement of each of ``paths`` that has one.

    :param db_path: path to a ``measurements.db``.
    :param annotation_column: the column the judgements belong to.
    :param paths: the crops on the page.
    :param png_table: the crop table.
    :returns: ``{png_path: verdict}`` for the crops that carry one; empty
        when the column does not exist yet.
    """
    paths = [str(p) for p in paths]
    if not paths:
        return {}
    verdict = verdict_column(annotation_column)
    out: Dict[str, int] = {}
    with _connect_read_only(db_path) as db:
        if not _has_column(db, png_table, verdict):
            return {}
        for start in range(0, len(paths), 500):
            chunk = paths[start:start + 500]
            marks = ",".join("?" * len(chunk))
            try:
                rows = db.execute(
                    f'SELECT png_path, "{verdict}" FROM "{png_table}" '
                    f'WHERE "{verdict}" IS NOT NULL AND png_path IN ({marks})',
                    chunk).fetchall()
            except sqlite3.Error:
                return out
            for path, value in rows:
                try:
                    out[str(path)] = int(value)
                except (TypeError, ValueError):
                    continue
    return out


def judgement_counts(db_path: str, annotation_column: str, *,
                     png_table: str = "png_list") -> Dict[str, int]:
    """How the column's suggestions stand: judged, and still to judge.

    :param db_path: path to a ``measurements.db``.
    :param annotation_column: the column the suggestions were written into.
    :param png_table: the crop table.
    :returns: ``{"left": n, "confirmed": n, "rejected": n}`` -- ``left`` is
        the suggestions nobody has judged yet, the other two are every
        judgement recorded in the column so far.
    """
    counts = {"left": 0, "confirmed": 0, "rejected": 0}
    verdict = verdict_column(annotation_column)
    with _connect_read_only(db_path) as db:
        if _has_column(db, png_table, annotation_column):
            try:
                row = db.execute(
                    f'SELECT COUNT(*) FROM "{png_table}" '
                    f'WHERE "{annotation_column}" > ?',
                    (SUGGESTION_OFFSET,)).fetchone()
                counts["left"] = int(row[0]) if row else 0
            except sqlite3.Error:
                pass
        if _has_column(db, png_table, verdict):
            try:
                row = db.execute(
                    f'SELECT SUM("{verdict}" > 0), SUM("{verdict}" < 0) '
                    f'FROM "{png_table}" WHERE "{verdict}" IS NOT NULL'
                ).fetchone()
            except sqlite3.Error:
                row = None
            if row:
                counts["confirmed"] = int(row[0] or 0)
                counts["rejected"] = int(row[1] or 0)
    return counts


def rejected_suggestions(db_path: str, annotation_column: str, *,
                         png_table: str = "png_list") -> Dict[str, int]:
    """The crops whose suggestion was rejected, and the class that was refused.

    Only crops the annotator has NOT since labelled are returned: a label
    made after a rejection is the stronger statement and is what the fit
    reads from the column itself, so handing the rejection over as well
    would count the crop twice.

    :param db_path: path to a ``measurements.db``.
    :param annotation_column: the column the suggestions were written into.
    :param png_table: the crop table.
    :returns: ``{png_path: rejected class}``; empty when nothing was rejected
        or the verdict column does not exist.
    """
    verdict = verdict_column(annotation_column)
    out: Dict[str, int] = {}
    with _connect_read_only(db_path) as db:
        if not _has_column(db, png_table, verdict):
            return out
        if not _has_column(db, png_table, annotation_column):
            return out
        try:
            rows = db.execute(
                f'SELECT png_path, "{verdict}" FROM "{png_table}" '
                f'WHERE "{verdict}" < 0 AND "{annotation_column}" IS NULL'
            ).fetchall()
        except sqlite3.Error:
            return out
    for path, value in rows:
        try:
            out[str(path)] = -int(value)
        except (TypeError, ValueError):
            continue
    return out
