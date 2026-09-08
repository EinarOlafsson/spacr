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

    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as db:
        try:
            crops = pd.read_sql_query(f'SELECT * FROM "{png_table}"', db)
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
    # 379-C, MADE SAFE WITHOUT ANSWERING IT. Whether this scheme should
    # extend past two classes is the maintainer's decision and is still
    # open; what is NOT open is that it must never silently collide. The
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
                f"{SUGGESTION_OFFSET + 1} would be the same number. See "
                f"379-C -- extending the scheme past two classes is an open "
                f"decision, and this is the collision it has to avoid.")
    rows = [(int(s), str(p)) for s, p in
            zip(suggestions["stored"], suggestions["png_path"])
            if pd.notna(s)]
    if not rows:
        return 0
    written = 0
    with sqlite3.connect(db_path) as db:
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

    with sqlite3.connect(db_path) as db:
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
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as db:
        try:
            row = db.execute(
                f'SELECT COUNT(*) FROM "{png_table}" '
                f'WHERE "{annotation_column}" > ?', (SUGGESTION_OFFSET,)
            ).fetchone()
        except sqlite3.Error:
            return 0
    return int(row[0]) if row else 0
