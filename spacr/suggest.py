"""Train on what you have annotated, and propose the rest.

Annotating is the slow part of every screen, and the first two hundred crops
already contain most of what separates the classes. So a model is trained on
those and asked to label the remainder -- and then the suggestions are made
cheap to reject, because THE REVIEWING IS THE FEATURE and the classifier is
only what makes reviewing possible.

WHAT IT LEARNS FROM. The measurement table spaCR already computes, joined to
each crop on its object id. That was chosen over image embeddings for a reason
worth keeping: the model then explains itself in the same terms the user
already reads, and if the morphology cannot separate the classes, that is a
finding about the experiment rather than a reason to reach for a network.

THREE RULES THAT PROTECT THE ANNOTATIONS
========================================
1. A SUGGESTION IS NEVER CONFUSABLE WITH A DECISION. Suggestions are written
   as their own values -- a suggested 1 is stored as 11 -- so nothing that
   reads the annotation column can mistake one for a human's answer, and a
   run that goes wrong is undone by deleting a value rather than by
   remembering which rows were touched.
2. A SUGGESTION NEVER OVERWRITES AN ANNOTATION. Writes go only where the
   column is NULL. Not when the model is confident, not on a re-run.
3. ONE CLASS IS A DELIBERATE LIE, AND IT IS LABELLED. If only one class has
   been annotated, the negatives are drawn at random from the unannotated
   pool -- which is MOSTLY-negative, not negative. The result is a ranking,
   not a verdict, and :class:`Suggestions` says so in ``was_one_class`` so the
   interface can too.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

#: What is added to a class value to mark it as a suggestion rather than an
#: answer. Ten because the annotation column holds small integers, so a
#: suggested 1 becomes 11 and cannot collide with a real 2 or 3.
SUGGESTION_OFFSET = 10

#: Below this many annotations per class, there is nothing to learn from and
#: the honest answer is to say so rather than to train on noise.
MIN_PER_CLASS = 20


@dataclass
class Suggestions:
    """What a suggestion run proposes, and how much to trust it.

    :param frame: one row per unannotated crop, with ``suggested`` (the class),
        ``stored`` (the offset value written to the column) and ``confidence``,
        SORTED so the doubtful ones sit together at the end.
    :param was_one_class: True when the negatives were drawn from unannotated
        crops rather than annotated ones -- see rule 3. A ranking, not a
        verdict.
    :param trained_on: how many crops of each class the model saw.
    :param features: the measurement columns used, in order.
    :param note: what a reader needs to be told before acting in bulk.
    """

    frame: pd.DataFrame
    was_one_class: bool
    trained_on: Dict[int, int] = field(default_factory=dict)
    features: List[str] = field(default_factory=list)
    note: str = ""


def _object_id_int(value) -> Optional[int]:
    """The integer in a ``png_list`` object id: ``'o12'`` -> ``12``.

    ``'omulti'`` and ``'onone'`` -- a crop overlapping several objects or none
    -- have no single label and come back as None, so they are joined to no
    measurement and simply do not appear.

    :param value: the stored id.
    :returns: the integer label, or None.
    """
    if value is None:
        return None
    text = str(value).strip()
    if text.startswith("o"):
        text = text[1:]
    try:
        return int(float(text))
    except (TypeError, ValueError):
        return None


def training_table(db_path: str, annotation_column: str, *,
                   object_type: str = "cell",
                   png_table: str = "png_list") -> pd.DataFrame:
    """Every crop, its annotation if it has one, and its measurements.

    :param db_path: path to a ``measurements.db``.
    :param annotation_column: the column holding the labels.
    :param object_type: which measurement table the crop is of.
    :param png_table: the crop table.
    :returns: one row per crop, with ``png_path``, ``label`` (NaN when
        unannotated) and every numeric measurement column.
    """
    from .filters import IDENTITY_ALIASES

    id_column = f"{object_type}_id"
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as db:
        crops = pd.read_sql_query(f'SELECT * FROM "{png_table}"', db)
        measured = pd.read_sql_query(f'SELECT * FROM "{object_type}"', db)
    if crops.empty or measured.empty or id_column not in crops.columns:
        return pd.DataFrame()

    crops = crops.copy()
    crops["object_label"] = crops[id_column].map(_object_id_int)
    crops = crops.dropna(subset=["object_label"])
    crops["object_label"] = crops["object_label"].astype(int)

    def spelling(frame, canonical):
        """Which spelling ``frame`` uses for an identity column.

        spaCR has written both ``plateID`` and ``plate`` over the years, and
        a database can carry either. Two tables in the SAME database can also
        disagree, which is why this is asked per frame rather than once.

        :param frame: the table to look in.
        :param canonical: the canonical column name.
        :returns: the spelling present, or None when the frame has none.
        """
        have = {c.lower(): c for c in frame.columns}
        for alias in IDENTITY_ALIASES.get(canonical, (canonical,)):
            if alias.lower() in have:
                return have[alias.lower()]
        return None

    keys = []
    for canonical in ("plateID", "rowID", "columnID", "fieldID"):
        left, right = spelling(crops, canonical), spelling(measured, canonical)
        if left and right:
            crops = crops.rename(columns={left: canonical})
            measured = measured.rename(columns={right: canonical})
            keys.append(canonical)

    label_col = "object_label"
    if label_col not in measured.columns:
        return pd.DataFrame()
    measured[label_col] = pd.to_numeric(measured[label_col], errors="coerce")

    merged = crops.merge(measured, how="inner", on=keys + [label_col],
                         suffixes=("", "_measured"))
    if annotation_column in merged.columns:
        merged["label"] = pd.to_numeric(merged[annotation_column],
                                        errors="coerce")
    else:
        merged["label"] = np.nan
    return merged


def _numeric_features(frame: pd.DataFrame, annotation_column: str
                      ) -> List[str]:
    """The measurement columns worth learning from.

    Identity columns and the annotation itself are excluded by name: a model
    given the label as a feature learns nothing, and one given the plate
    learns the plate.

    :param frame: the joined table.
    :param annotation_column: the label column to exclude.
    :returns: usable column names, sorted for a stable feature order.
    """
    banned = {"label", annotation_column, "object_label", "plateID", "rowID",
              "columnID", "fieldID", "timeID", "prcfo", "prcf", "prc"}
    out = []
    for column in frame.columns:
        if column in banned or column.endswith("_id"):
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.notna().sum() < len(frame) * 0.5:
            continue                       # more than half missing: not a feature
        if values.nunique(dropna=True) <= 1:
            continue                       # constant: carries nothing
        out.append(column)
    return sorted(out)


def suggest(db_path: str, annotation_column: str, *,
            object_type: str = "cell", png_table: str = "png_list",
            min_per_class: int = MIN_PER_CLASS,
            random_state: int = 0) -> Suggestions:
    """Train on the annotated crops and rank every unannotated one.

    NO CONFIDENCE FLOOR: every unannotated crop gets a suggestion, sorted by
    confidence, so the reviewer stops when they stop agreeing rather than
    having a threshold decide for them.

    :param db_path: path to a ``measurements.db``.
    :param annotation_column: the column holding the labels.
    :param object_type: which measurement table the crops are of.
    :param png_table: the crop table.
    :param min_per_class: refuse to train below this many examples.
    :param random_state: seeds the class balancing and the model.
    :returns: a :class:`Suggestions`. Its frame is empty when there was not
        enough to learn from, and ``note`` says why.
    :raises RuntimeError: if xgboost is not installed.
    """
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:                                # pragma: no cover
        raise RuntimeError(
            "Suggest needs xgboost: pip install xgboost"
        ) from exc

    table = training_table(db_path, annotation_column,
                           object_type=object_type, png_table=png_table)
    if table.empty:
        return Suggestions(pd.DataFrame(), False,
                           note="no crops could be joined to measurements")

    features = _numeric_features(table, annotation_column)
    if not features:
        return Suggestions(pd.DataFrame(), False,
                           note="no usable measurement columns")

    annotated = table[table["label"].notna()].copy()
    unannotated = table[table["label"].isna()].copy()
    if unannotated.empty:
        return Suggestions(pd.DataFrame(), False, features=features,
                           note="every crop is already annotated")

    classes = sorted(int(v) for v in annotated["label"].unique())
    rng = np.random.default_rng(random_state)
    was_one_class = len(classes) == 1

    if not classes:
        return Suggestions(pd.DataFrame(), False, features=features,
                           note="nothing has been annotated yet")

    if was_one_class:
        # THE DELIBERATE LIE, rule 3. Draw as many unannotated crops as there
        # are annotated ones and train them as the other class. They are
        # mostly-negative, not negative, so what comes back is a ranking.
        positive = annotated
        n = min(len(positive), len(unannotated))
        drawn = unannotated.sample(n=n, random_state=random_state)
        negative = drawn.copy()
        negative["label"] = -1
        train = pd.concat([positive, negative], ignore_index=True)
        classes = [-1, classes[0]]
    else:
        smallest = min((annotated["label"] == c).sum() for c in classes)
        if smallest < min_per_class:
            return Suggestions(
                pd.DataFrame(), False, features=features,
                trained_on={c: int((annotated["label"] == c).sum())
                            for c in classes},
                note=(f"the smallest class has {smallest} annotations and "
                      f"{min_per_class} are needed; annotate more first"))
        # Downsample the larger class rather than weighting it: with the class
        # sizes equal, the model's own probability is directly readable as
        # confidence, which is what the sort below depends on.
        train = pd.concat(
            [annotated[annotated["label"] == c].sample(
                n=smallest, random_state=random_state) for c in classes],
            ignore_index=True)

    if was_one_class and len(train) < 2 * min_per_class:
        return Suggestions(
            pd.DataFrame(), True, features=features,
            trained_on={int(c): int((train["label"] == c).sum())
                        for c in classes},
            note=(f"only {int((train['label'] != -1).sum())} annotations in "
                  f"one class; {min_per_class} are needed"))

    x_train = train[features].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    codes = {c: i for i, c in enumerate(classes)}
    y_train = train["label"].astype(int).map(codes)

    model = XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.1,
        subsample=0.9, colsample_bytree=0.9,
        random_state=random_state, eval_metric="logloss",
        verbosity=0,
    )
    model.fit(x_train, y_train)

    x_new = unannotated[features].apply(pd.to_numeric,
                                        errors="coerce").fillna(0.0)
    probabilities = model.predict_proba(x_new)
    best = probabilities.argmax(axis=1)
    inverse = {i: c for c, i in codes.items()}

    out = unannotated.copy()
    out["suggested"] = [inverse[i] for i in best]
    out["confidence"] = probabilities.max(axis=1)
    # The drawn negatives are not a class anybody annotated, so a crop the
    # model calls -1 is "unlike the annotated ones" and carries no suggestion.
    out.loc[out["suggested"] == -1, "suggested"] = np.nan
    out["stored"] = out["suggested"] + SUGGESTION_OFFSET

    keep = [c for c in ("png_path", "object_label", "suggested", "stored",
                        "confidence") if c in out.columns]
    ordered = out[keep].sort_values("confidence", ascending=False,
                                    ignore_index=True)

    note = ""
    if was_one_class:
        note = ("Only one class was annotated, so the negatives were drawn at "
                "random from unannotated crops. Those are MOSTLY negative, not "
                "negative, which makes this a ranking rather than a verdict -- "
                "read down it and stop where you stop agreeing.")
    return Suggestions(ordered, was_one_class,
                       trained_on={int(c): int((train["label"] == c).sum())
                                   for c in classes},
                       features=features, note=note)


def write_suggestions(db_path: str, annotation_column: str,
                      suggestions: pd.DataFrame, *,
                      png_table: str = "png_list") -> int:
    """Store suggestions, and ONLY where nothing has been annotated.

    The ``IS NULL`` in the update is rule 2 and is not an optimisation. A
    human's annotation is the ground truth the model was trained on; losing
    one silently would cost hours and would not be noticed until a run came
    out wrong.

    :param db_path: path to a ``measurements.db``.
    :param annotation_column: the column to write into.
    :param suggestions: the frame from :func:`suggest`.
    :param png_table: the crop table.
    :returns: how many rows were written.
    """
    if suggestions.empty or "png_path" not in suggestions.columns:
        return 0
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
    back to NULL. Both act only on SUGGESTION values, so a human annotation
    caught in the same query is untouched either way.

    :param db_path: path to a ``measurements.db``.
    :param annotation_column: the column holding both.
    :param keep: True to accept, False to discard.
    :param png_table: the crop table.
    :param paths: restrict to these crops; None means all suggestions.
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
                   f'{SUGGESTION_OFFSET} WHERE {where}')
        else:
            sql = f'UPDATE "{png_table}" SET {column} = NULL WHERE {where}'
        cur = db.execute(sql, params)
        db.commit()
        return cur.rowcount
