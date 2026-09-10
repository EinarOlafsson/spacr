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
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd


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
                        classes: Optional[Sequence[int]] = None
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
        # The retrain encoded classes from the sorted annotation values, and
        # a suggestion offset is not one of them.
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
    # 379-C, ANSWERED 2026-09-10 AND NOT THE WAY IT WAS ASKED. The question
    # was "should this extend past two classes"; the answer is that it
    # already does. `suggest_from_scores` argmaxes across N score columns
    # and nothing in it counts to two, so classes 1 to 9 round-trip today --
    # driven in
    # `tests/test_the_suggestion_offset_holds_more_than_two_classes.py`.
    #
    # THE LIMIT IS A CLASS VALUE, NOT A CLASS COUNT, and that is what the
    # refusal below is for. The maintainer confirmed he does not use class
    # values of ten or more, so the offset stays at 10 and this guard stays
    # a refusal rather than becoming a bound. The
    # offset assumes real class values stay below it, and with a class 11 a
    # suggested 1 and an answered 11 are the same integer -- so a bulk KEEP
    # would rewrite somebody's class 11 to a 1 and nothing would ever say
    # so. Refused by name rather than guarded downstream, because by the
    # time the value is in the column the two are indistinguishable.
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
    # 379-C, THE OTHER DIRECTION, and the one the guard above does not
    # reach. That one refuses to SUGGEST a class at or above the offset.
    # This refuses to write suggestions into a column that ALREADY HOLDS
    # one, which is the case that loses a person's work rather than merely
    # confusing a number.
    #
    # `pending_suggestions` counts every value above the offset as
    # outstanding, and `is_suggestion` reads one the same way, so a human
    # who annotated class 11 has a row that the bulk KEEP will rewrite to a
    # 1. PART 2's first rule is that a suggestion must never overwrite a
    # human annotation, and this is the only route by which it still could.
    #
    # SAFE TO REFUSE HERE because the screen clears outstanding suggestions
    # before it asks for new ones -- `resolve_suggestions(keep=False)` in
    # `annotate._on_suggest` -- so anything at or above the offset that
    # survives to this point is a real answer and not a stale proposal. A
    # caller that has not cleared gets told to, which is the same sentence.
    with _connect_read_only(db_path) as db:
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
        if keep:
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
        try:
            row = db.execute(
                f'SELECT COUNT(*) FROM "{png_table}" '
                f'WHERE "{annotation_column}" > ?', (SUGGESTION_OFFSET,)
            ).fetchone()
        except sqlite3.Error:
            return 0
    return int(row[0]) if row else 0
