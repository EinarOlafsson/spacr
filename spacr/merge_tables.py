"""Merging object tables, aggregating each measurement by what it MEASURES.

A cell with four pathogens in it has four rows in ``pathogen`` and one in
``cell``. Putting a pathogen measurement on the same axis as a cell
measurement means rolling those four up into one number -- and which number
depends entirely on what is being measured:

    area, perimeter, integrated intensity, counts   -> SUM
    minimum intensity                               -> MIN of the four
    maximum intensity                               -> MAX of the four
    mean, median                                    -> the object mean/median
    shape descriptors, positions                    -> mean
    text                                            -> the first

``spacr.io._read_and_join_tables`` already does this join, and aggregates
every numeric column with ``mean``. That silently answers a different question
per column: four pathogens' total area becomes an average area, a count
becomes an average count, and a MINIMUM becomes a mean of minima, which is not
a minimum of anything. The join is otherwise sound -- it is where the parent
link, the timelapse key and the cardinality checks live -- so this module
changes what the aggregation is, not how the tables find each other.

**Naming.** A merged column carries its table: ``area`` from ``nucleus``
becomes ``nucleus_area``. Prefix rather than suffix so every column of one
object sorts together in the axis picker, which is how anyone looks for them.

**The primary object.** Everything is rolled up onto ONE table -- the cell by
default. That choice decides what a row means, so it is a setting rather than
an assumption: rolling cells onto pathogens is a legitimate thing to want and
gives a different table.
"""
from __future__ import annotations

import logging
import re
import sqlite3
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from .object_roles import (ANCHOR_COLUMN, anchor_column,
                           is_one_row_per_cell)

LOG = logging.getLogger("spacr.merge_tables")

#: How a child's rows combine into one number for the parent.
SUM, MIN, MAX, MEAN, MEDIAN, FIRST = "sum", "min", "max", "mean", "median", "first"

#: The aggregations offered, for a settings dropdown.
AGGREGATIONS: Tuple[str, ...] = (SUM, MIN, MAX, MEAN, MEDIAN, FIRST)

#: Column-name patterns -> aggregation, FIRST MATCH WINS, so the order is the
#: rule. Anything unmatched falls to :data:`DEFAULT_AGGREGATION`.
#:
#: Read this as a statement about measurements, not about strings: each entry
#: is here because combining that quantity any other way answers a different
#: question. `min_intensity` is the clearest -- a mean of four minima is not
#: the minimum of anything.
AGGREGATION_RULES: Tuple[Tuple[str, str], ...] = (
    # Identity first, because it must beat every rule below it. A label is a
    # NAME, not a quantity: averaging the three pathogen labels 1, 2 and 3 in
    # a cell produces `object_label_pathogen` = 2.0, which looks like a
    # measurement, plots like one, and can be handed to a model as a feature
    # -- while naming an object that may not exist. The parent already learns
    # how many children it had from the count column, so the label is carried
    # only so a row can be traced back, and it is carried verbatim.
    (r"(^|_)(object_label|label|id|cell_id|nucleus_id|pathogen_id|"
     r"organelle_id|cytoplasm_id|parent_id|prcfo|prcf|prc)(_|$)", FIRST),
    (r"(^|_)(count|n_objects|number)(_|$)", SUM),
    (r"(^|_)min(imum)?(_|$)", MIN),
    (r"(^|_)max(imum)?(_|$)", MAX),
    (r"(^|_)median(_|$)", MEDIAN),
    (r"(^|_)(mean|average|avg)(_|$)", MEAN),
    # Extent: four objects' AREAS add up. Their LENGTHS do not -- two nuclei
    # each 10 units long are not one nucleus 20 units long, so an axis
    # length, a perimeter and an equivalent diameter are shape descriptors of
    # an individual object and the parent gets the typical one. Volume adds
    # for the same reason area does. (Maintainer's call, 2026-08-11.)
    (r"(^|_)(area|volume|convex_area|filled_area)(_|$)", SUM),
    (r"(^|_)(perimeter|length|width|height|diameter|"
     r"equivalent_diameter|major_axis_length|minor_axis_length)(_|$)", MEAN),
    # Anything already integrated over an object is a total.
    (r"(^|_)(integrated|total|sum|integral)(_|$)", SUM),
    # Spread and shape are properties of each object; the parent gets the
    # typical one.
    (r"(^|_)(std|stdev|var|variance|mad|iqr|percentile|quantile|"
     r"skew|kurtosis|entropy)(_|$)", MEAN),
    (r"(^|_)(eccentricity|solidity|extent|circularity|roundness|"
     r"aspect_ratio|orientation|zernike|moment|hu)(_|$)", MEAN),
    (r"(^|_)(centroid|center|centre|coord|bbox|position|_x|_y)(_|$)", MEAN),
)

#: What an unrecognised numeric measurement gets. MEAN rather than SUM: a
#: measurement nobody thought about is more often an intensity or a ratio than
#: an extent, and a wrong mean is a smaller error than a wrong total.
DEFAULT_AGGREGATION = MEAN

#: Non-numeric columns take the first value: text does not add up.
TEXT_AGGREGATION = FIRST

#: What to do with objects that have no children.
NA_POLICIES: Tuple[str, ...] = ("keep", "zero", "drop")

#: Identity columns a merge joins on, and the child's link to its parent.
IDENTITY = ("plateID", "rowID", "columnID", "fieldID")
PARENT_LINK = "cell_id"
PNG_TABLE = "png_list"
OBJECT_COLUMN = "object_label"

#: The tables that can be merged, and the default primary.
OBJECT_TABLES: Tuple[str, ...] = (
    "cell", "nucleus", "pathogen", "cytoplasm", "organelle",
)
DEFAULT_PRIMARY = "cell"


class MergeError(ValueError):
    """A merge that cannot be done, and why."""


@dataclass(frozen=True)
class MergePolicy:
    """How a merge is performed. Every field is a user-facing setting.

    :param primary: the object everything is rolled up onto. Decides what a
        row of the merged table MEANS.
    :param na: what happens to a primary object with no children --
        ``keep`` leaves NaN, ``zero`` fills with 0, ``drop`` removes the row.
        Not interchangeable: a cell with no pathogens genuinely has a pathogen
        COUNT of zero, and genuinely has no pathogen mean intensity at all.
    :param overrides: column -> aggregation, beating the rules. The rules are
        right most of the time, and a default that is right most of the time
        is a wrong answer nobody can find the rest of it.
    """

    primary: str = DEFAULT_PRIMARY
    na: str = "keep"
    overrides: Mapping[str, str] = None

    def __post_init__(self) -> None:
        if self.na not in NA_POLICIES:
            raise MergeError(
                f"na={self.na!r} is not one of {list(NA_POLICIES)}")
        object.__setattr__(self, "overrides", dict(self.overrides or {}))


def aggregation_for(column: str, *, numeric: bool = True,
                    overrides: Optional[Mapping[str, str]] = None) -> str:
    """How ``column`` combines when several children roll up into one parent.

    :param numeric: text columns take the first value whatever their name.
    :param overrides: explicit choices, which always win.
    :returns: one of :data:`AGGREGATIONS`.
    """
    if overrides and column in overrides:
        chosen = str(overrides[column])
        if chosen not in AGGREGATIONS:
            raise MergeError(
                f"{chosen!r} is not an aggregation; choose from "
                f"{list(AGGREGATIONS)}")
        return chosen
    if not numeric:
        return TEXT_AGGREGATION
    name = str(column).lower()
    for pattern, how in AGGREGATION_RULES:
        if re.search(pattern, name):
            return how
    return DEFAULT_AGGREGATION


def aggregation_plan(frame: pd.DataFrame, *,
                     overrides: Optional[Mapping[str, str]] = None,
                     skip: Sequence[str] = ()) -> Dict[str, str]:
    """The aggregation chosen for every column -- what the user gets shown.

    Returned rather than applied silently so the settings panel can display
    it and the user can override any of it.
    """
    plan: Dict[str, str] = {}
    for column in frame.columns:
        if column in skip:
            continue
        numeric = pd.api.types.is_numeric_dtype(frame[column])
        plan[column] = aggregation_for(column, numeric=numeric,
                                       overrides=overrides)
    return plan


def _connect(db_path: str) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=30)


def table_names(db_path: str) -> Tuple[str, ...]:
    """Every table in the database, excluding SQLite's own internals.

    :param db_path: path to the SQLite database.
    :returns: the table names, sorted.
    """
    with _connect(db_path) as db:
        rows = db.execute(
            "SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    return tuple(str(r[0]) for r in rows)


def mergeable_tables(db_path: str) -> Tuple[str, ...]:
    """The object tables in this database, in preference order."""
    present = set(table_names(db_path))
    return tuple([t for t in OBJECT_TABLES if t in present]
                 + ([PNG_TABLE] if PNG_TABLE in present else []))


def _read(db_path: str, table: str) -> pd.DataFrame:
    with _connect(db_path) as db:
        return pd.read_sql_query(f'SELECT * FROM "{table}"', db)


def _keys_in(frame: pd.DataFrame) -> List[str]:
    return [c for c in IDENTITY if c in frame.columns]


def object_keys(values: pd.Series) -> pd.Series:
    """Object identifiers as integers, whatever spelling they arrived in.

    The object key is an integer in every object table and TEXT in
    ``png_list`` -- ``'o5'`` -- so merging the two raised

        you are trying to merge on int64 and object columns for key
        object_label

    which names the dtypes and not the tables, and stopped the whole merge.
    The ``'o5'`` form is translated by the one function that already knows
    every way it goes wrong (``'omulti'``, ``'onone'``, ``'error'``, NULL);
    plain numeric text is converted directly. Anything left becomes NA, so
    those rows simply do not match rather than taking the merge down.
    """
    if pd.api.types.is_numeric_dtype(values):
        return pd.to_numeric(values, errors="coerce").astype("Int64")
    text = values.astype("string")
    if text.str.match(r"^[a-zA-Z]", na=False).any():
        from .utils import object_label_from_png_id
        return pd.Series(object_label_from_png_id(values),
                         index=values.index).astype("Int64")
    return pd.to_numeric(text, errors="coerce").astype("Int64")


def _align_keys(left: pd.DataFrame, right: pd.DataFrame,
                keys: Sequence[str]) -> None:
    """Make both sides of a merge agree on the TYPE of every key.

    In place, on copies the caller owns. Identity columns are compared as
    text because a plate called ``1`` is read as an integer from one table and
    a string from another depending on what else is in the column -- the same
    class of failure as the object key, and just as fatal to a merge.
    """
    for key in keys:
        if key not in left.columns or key not in right.columns:
            continue
        if key == OBJECT_COLUMN:
            left[key] = object_keys(left[key])
            right[key] = object_keys(right[key])
        elif not (pd.api.types.is_numeric_dtype(left[key])
                  and pd.api.types.is_numeric_dtype(right[key])):
            left[key] = left[key].astype("string")
            right[key] = right[key].astype("string")


def roll_up(child: pd.DataFrame, keys: Sequence[str], *,
            name: str, policy: MergePolicy) -> pd.DataFrame:
    """Aggregate ``child`` onto its parent, one rule per column.

    :param keys: the parent's identity in the child -- the identity columns
        plus the parent link.
    :param name: the child table's name, used to prefix its columns.
    :returns: one row per parent, columns prefixed with ``name``.
    :raises MergeError: the child has none of the keys.
    """
    missing = [k for k in keys if k not in child.columns]
    if missing:
        raise MergeError(
            f"{name} has no {', '.join(missing)}, so its rows cannot be "
            f"matched to a parent; re-run Measure with the parent mask set")

    plan = aggregation_plan(child, overrides=policy.overrides, skip=keys)
    grouped = child.groupby(list(keys), dropna=False)

    out = grouped.agg(plan)
    # The count is the one measurement the child table does not carry and the
    # parent almost always wants: "how many pathogens are in this cell".
    out[f"count"] = grouped.size()
    out = out.reset_index()

    # Measure already writes its columns prefixed -- the nucleus table holds
    # `nucleus_area`, not `area` -- so prefixing unconditionally would hand
    # the user `nucleus_nucleus_area` for a measurement they know by another
    # name. Prefix only what needs it, matching the one-row-per-cell branch
    # in `merge_tables` so a column has ONE name whichever way it was joined.
    renamed = {c: f"{name}_{c}" for c in out.columns
               if c not in keys and not str(c).startswith(f"{name}_")}
    return out.rename(columns=renamed)


#: What happens when the same column arrives from two tables carrying
#: DIFFERENT values for the same object. ``warn`` prints and keeps the
#: left-hand one; ``raise`` stops the analysis.
CONFLICT_POLICIES: Tuple[str, ...] = ("warn", "raise")


#: Columns that name WHICH object a row is, rather than measuring it. Both
#: tables read them off the same image, so they must match -- and when they
#: do not, the two tables are describing different objects under one
#: identity. Everything else is a measurement: cell ``area`` and cytoplasm
#: ``area`` are SUPPOSED to differ, and reporting that as a conflict would
#: bury the real ones.
MUST_AGREE: Tuple[str, ...] = IDENTITY + (
    "prc", "prcf", "prcfo", "object_label", "cell_id", "timeID", "time_id",
    "plate_name", "row_name", "column_name", "field_name",
)


class ColumnConflict(MergeError):
    """One column, two tables, two different values for the same object."""


def _columns_agree(left: pd.Series, right: pd.Series) -> pd.Series:
    """Row-wise equality that treats two missing values as agreement.

    ``NaN != NaN`` is right for arithmetic and wrong here: a column absent
    from both tables for a given object is not a disagreement about it.
    """
    both_missing = left.isna() & right.isna()
    if (pd.api.types.is_numeric_dtype(left)
            and pd.api.types.is_numeric_dtype(right)):
        # Two tables can reach the same number by different arithmetic, so
        # float noise is not a conflict; a real disagreement is never 1e-9.
        same = pd.Series(
            np.isclose(pd.to_numeric(left, errors="coerce"),
                       pd.to_numeric(right, errors="coerce"),
                       rtol=1e-9, atol=1e-12, equal_nan=True),
            index=left.index)
    else:
        same = left.astype(object).eq(right.astype(object))
    return same | both_missing


def reconcile_duplicates(frame: pd.DataFrame, suffix: str, *,
                         key: str = "prcfo",
                         left_name: str = "the primary table",
                         right_name: str = "the joined table",
                         on_conflict: str = "warn") -> pd.DataFrame:
    """Collapse ``col``/``col+suffix`` pairs that agree; report those that do not.

    Joining two measurement tables gives every shared column twice --
    ``plateID`` and ``plateID_cytoplasm`` hold the same plate written by two
    stages of the same run. Carrying both doubles the width of the frame and
    invites a downstream reader to pick the wrong one.

    So the pair is COMPARED, object by object, rather than assumed: identical
    columns collapse to one, and a column that disagrees means the two tables
    describe different objects under the same identity, which is a defect in
    the data no analysis should quietly average over.

    :param frame: the merged frame, modified only by dropping columns.
    :param suffix: what the merge appended to the right-hand duplicates.
    :param key: the identity the comparison is reported against.
    :param on_conflict: ``warn`` keeps the left-hand column and prints;
        ``raise`` stops with :class:`ColumnConflict`.
    :returns: the frame with agreeing duplicates dropped.
    :raises ColumnConflict: a pair disagrees and ``on_conflict='raise'``.
    """
    if on_conflict not in CONFLICT_POLICIES:
        raise MergeError(
            f"on_conflict={on_conflict!r} is not one of "
            f"{list(CONFLICT_POLICIES)}")
    if not suffix:
        return frame

    drop, conflicts = [], []
    for right_col in [c for c in frame.columns if str(c).endswith(suffix)]:
        left_col = str(right_col)[: -len(suffix)]
        if left_col not in frame.columns:
            continue        # a genuinely new column that happens to end so
        agree = _columns_agree(frame[left_col], frame[right_col])
        if bool(agree.all()):
            drop.append(right_col)
            continue
        if left_col not in MUST_AGREE:
            # Two measurements that happen to share a name. They describe
            # different objects, so they differ by design and both are kept.
            continue
        disagreeing = frame.index[~agree]
        where = (frame.loc[disagreeing, key].astype(str).tolist()[:5]
                 if key in frame.columns else
                 [str(i) for i in disagreeing[:5]])
        conflicts.append(
            f"{left_col!r}: {int((~agree).sum())} of {len(agree)} objects "
            f"disagree between {left_name} and {right_name}"
            + (f" (e.g. {key} " + ", ".join(where) + ")" if where else ""))

    if conflicts:
        detail = ("the same column arrived from two tables with different "
                  "values for the same object:\n  "
                  + "\n  ".join(conflicts))
        if on_conflict == "raise":
            raise ColumnConflict(detail)
        LOG.warning(detail)
        print(f"WARNING: {detail}")

    return frame.drop(columns=drop) if drop else frame



def merge_tables(db_path: str, tables: Sequence[str], *,
                 policy: Optional[MergePolicy] = None) -> pd.DataFrame:
    """One table with every chosen object's measurements on it.

    This is what makes "a cell measurement on one axis, nuclear on another and
    pathogen on a third" possible: each table's columns arrive prefixed with
    the object they measure, so they can be told apart and picked separately.

    :param tables: which object tables to include. The primary must be one of
        them, and is added if it is not.
    :param policy: how to aggregate and what to do with childless parents.
    :returns: one row per primary object.
    :raises MergeError: the primary table is not in the database, or a child
        cannot be linked to it.
    """
    policy = policy or MergePolicy()
    available = set(table_names(db_path))
    if policy.primary not in available:
        raise MergeError(
            f"the database has no {policy.primary!r} table to merge onto; it "
            f"has {', '.join(sorted(available & set(OBJECT_TABLES))) or 'none'}")

    wanted = [t for t in dict.fromkeys([policy.primary, *tables])
              if t in available]
    # png_list holds one row per CROP, not per object, and its object key is
    # text. It is merged like any child -- the keys are reconciled below --
    # but it has no measurements to aggregate, so it contributes its paths.
    if PNG_TABLE in tables and PNG_TABLE not in wanted and PNG_TABLE in available:
        wanted.append(PNG_TABLE)

    base = _read(db_path, policy.primary)
    keys = _keys_in(base)
    if OBJECT_COLUMN not in base.columns:
        raise MergeError(
            f"{policy.primary} has no {OBJECT_COLUMN}, so nothing can be "
            f"merged onto it")

    merged = base.rename(
        columns={c: f"{policy.primary}_{c}" for c in base.columns
                 if c not in keys + [OBJECT_COLUMN]})

    for table in wanted:
        if table == policy.primary:
            continue
        child = _read(db_path, table)
        if table == PNG_TABLE:
            merged = _merge_crops(merged, child, keys)
            continue
        # THE ANCHOR HAS TWO NAMES. cell and cytoplasm carry it as
        # `object_label` -- a cytoplasm is the cell minus its interior
        # objects, so its own label IS the cell's -- while nucleus, pathogen,
        # organelle and png_list carry the parent's label in `cell_id`.
        #
        # This assumed `cell_id` for every non-primary table, so CYTOPLASM WAS
        # SILENTLY DROPPED: it logged a line about an unlinkable table and
        # returned a frame with no cytoplasm columns at all.
        #
        # One row per cell also means no roll-up. Aggregating a table that
        # already has one row per cell is not wrong so much as meaningless,
        # and it would put the cytoplasm's own measurements through the
        # sum/mean rules meant for a group of children.
        anchor = anchor_column(table) if table in ANCHOR_COLUMN else PARENT_LINK
        if anchor not in child.columns:
            # Measured without a parent mask: the roll-up is not empty, it is
            # UNDEFINED. Named and skipped, exactly as io.py does -- one
            # unlinkable table must not cost the user the others.
            LOG.info("%s carries no %s, so it cannot be joined onto %s; "
                     "leaving it out", table, anchor, policy.primary)
            continue

        if is_one_row_per_cell(table):
            # Prefixed like the primary table above, with two exceptions that
            # would otherwise produce nonsense: the join keys keep their
            # names, and a column that ALREADY carries the table's name is
            # left alone -- `cytoplasm_area` must not become
            # `cytoplasm_cytoplasm_area`.
            skip = set(_keys_in(child)) | {anchor, "prcf", "prcfo"}
            rolled = child.rename(
                columns={c: (c if c.startswith(f"{table}_") else f"{table}_{c}")
                         for c in child.columns if c not in skip})
        else:
            child_keys = _keys_in(child) + [anchor]
            rolled = roll_up(child, child_keys, name=table, policy=policy)
            rolled = rolled.rename(columns={anchor: OBJECT_COLUMN})
        on = [c for c in keys + [OBJECT_COLUMN] if c in rolled.columns]
        if OBJECT_COLUMN not in on:
            LOG.info("%s cannot be joined to %s on an object key", table,
                     policy.primary)
            continue
        _align_keys(merged, rolled, on)
        merged = merged.merge(rolled, on=on, how="left")

    return _apply_na_policy(merged, policy)


def _merge_crops(merged: pd.DataFrame, png: pd.DataFrame,
                 keys: Sequence[str]) -> pd.DataFrame:
    """Attach crop paths from ``png_list``.

    Not aggregated: a crop is not a measurement, and png_list is one row per
    crop rather than per object. Its object id may be under any of the
    crop-mode columns, so the first one present is used -- a database measured
    in more than one crop mode has several, and the others belong to different
    objects entirely.
    """
    from .utils import PNG_OBJECT_ID_COLUMNS

    id_column = next((c for c in PNG_OBJECT_ID_COLUMNS.values()
                      if c in png.columns), None)
    if id_column is None:
        LOG.info("%s carries no object id column; no crop paths merged",
                 PNG_TABLE)
        return merged

    path_column = next((c for c in ("png_path", "path", "file_path")
                        if c in png.columns), None)
    side = png[[c for c in keys if c in png.columns]].copy()
    side[OBJECT_COLUMN] = object_keys(png[id_column])
    if path_column:
        side[f"{PNG_TABLE}_path"] = png[path_column]
    side = side.dropna(subset=[OBJECT_COLUMN])

    on = [c for c in list(keys) + [OBJECT_COLUMN] if c in side.columns]
    side = side.drop_duplicates(subset=on)
    _align_keys(merged, side, on)
    return merged.merge(side, on=on, how="left")


def _apply_na_policy(frame: pd.DataFrame, policy: MergePolicy) -> pd.DataFrame:
    """What happens to a primary object with no children.

    ``zero`` fills only the COUNTS by default reasoning: a cell with no
    pathogens has a pathogen count of zero, but it does not have a pathogen
    mean intensity of zero -- it has none, and zero would be a measurement
    that was never made. Callers who want the blunter behaviour choose it
    explicitly.
    """
    counts = [c for c in frame.columns if c.endswith("_count")]
    if policy.na == "keep":
        for column in counts:
            frame[column] = frame[column].fillna(0)
        return frame
    if policy.na == "zero":
        return frame.fillna({c: 0 for c in frame.columns
                             if pd.api.types.is_numeric_dtype(frame[c])})
    if policy.na == "drop":
        child_columns = [c for c in frame.columns if "_" in c and c not in counts]
        return frame.dropna(subset=child_columns, how="any").reset_index(drop=True)
    return frame


# ---------------------------------------------------------------------------
# Dimensional reduction -- what xD means
# ---------------------------------------------------------------------------

#: Reductions offered in xD mode.
REDUCTIONS: Tuple[str, ...] = ("pca", "umap", "tsne")


class ReductionError(ValueError):
    """A reduction that cannot be computed, and why."""


def reduce_dimensions(frame: pd.DataFrame, columns: Sequence[str], *,
                      method: str = "pca", components: int = 2,
                      scale: bool = True, min_coverage: float = 0.5,
                      seed: int = 0) -> pd.DataFrame:
    """Reduce many measurements to a few, for gating in xD.

    Gating in more dimensions than can be drawn means drawing something else:
    a projection. The components come back as ORDINARY COLUMNS (``PC1``,
    ``PC2``, ...), so every existing gate tool works on them unchanged -- a
    gate on PC1 vs PC2 is the same kind of object as a gate on area vs
    intensity, and saves, re-applies and exports identically.

    :param columns: the measurements to reduce. At least two.
    :param method: ``pca`` always available; umap and t-SNE if installed.
    :param scale: standardise first. Without it a measurement whose numbers
        are larger dominates every component regardless of what it means.
    :param min_coverage: a column with fewer than this fraction of real values
        is left out. What remains is median-filled rather than row-dropped --
        see the comment in the body, which is the difference between xD
        working on a real table and returning nothing at all.
    :returns: a frame of components, indexed like ``frame``.
    :raises ReductionError: too few columns, too few rows, nothing numeric, or
        a method whose package is not installed.
    """
    if method not in REDUCTIONS:
        raise ReductionError(
            f"{method!r} is not one of {list(REDUCTIONS)}")
    chosen = [c for c in columns if c in frame.columns]
    if len(chosen) < 2:
        raise ReductionError(
            "reducing needs at least two measurements; pick more columns")

    data = frame[chosen].apply(pd.to_numeric, errors="coerce")

    # DROPPING every row with any missing value does not work on a real
    # measurement table. With 60 columns at 2% missing each, two rows in three
    # are lost; with the several hundred columns spaCR actually writes, none
    # survive -- which is why xD looked like it had never been implemented.
    #
    # So: drop the columns that are mostly empty, then fill what is left with
    # the column median. A median fill moves an object to the middle of an
    # axis it had no value on, which is the least it can be moved; discarding
    # the object instead loses every measurement it DID have.
    coverage = data.notna().mean()
    keep = [c for c in chosen if coverage.get(c, 0.0) >= min_coverage]
    dropped = [c for c in chosen if c not in keep]
    if dropped:
        LOG.info("%d column(s) are under %.0f%% complete and were left out of "
                 "the projection: %s", len(dropped), min_coverage * 100,
                 ", ".join(dropped[:6]) + ("…" if len(dropped) > 6 else ""))
    if len(keep) < 2:
        raise ReductionError(
            f"only {len(keep)} of {len(chosen)} measurement(s) are at least "
            f"{min_coverage:.0%} complete, and a projection needs two; lower "
            f"the coverage requirement or pick fuller columns")

    data = data[keep]
    usable = data.fillna(data.median(numeric_only=True))
    # A column that is entirely NaN has no median; it cannot contribute.
    usable = usable.dropna(axis=1, how="any")
    if usable.shape[1] < 2:
        raise ReductionError(
            "no two measurements have enough values in common to project")
    usable = usable.loc[data.notna().any(axis=1)]
    if len(usable) < 3:
        raise ReductionError(
            f"only {len(usable)} object(s) have any of these measurements; "
            f"there is nothing to project")
    chosen = list(usable.columns)

    components = max(2, min(int(components), len(chosen), len(usable)))
    values = usable.to_numpy(dtype=float)
    if scale:
        centre = values.mean(axis=0)
        spread = values.std(axis=0)
        spread[spread == 0] = 1.0
        values = (values - centre) / spread

    if method == "pca":
        from sklearn.decomposition import PCA
        model = PCA(n_components=components, random_state=seed)
        reduced = model.fit_transform(values)
        names = [f"PC{i + 1}" for i in range(reduced.shape[1])]
        out = pd.DataFrame(reduced, index=usable.index, columns=names)
        # Explained variance is the only honest label for a PC axis: "PC1"
        # alone says nothing about whether it is the data or the noise.
        out.attrs["explained_variance"] = list(
            getattr(model, "explained_variance_ratio_", []))
    elif method == "umap":
        try:
            # spacr.utils' lazy loader, not a bare `import umap`: the
            # package's __init__ reaches umap.parametric_umap -> tensorflow,
            # and spaCR's standing rule is that nothing drags TF in. The
            # loader imports umap.umap_ with the TF-backed roots blocked.
            from .utils import umap
            umap.UMAP
        except Exception as exc:
            raise ReductionError(
                "UMAP is not installed in this environment; PCA is always "
                "available") from exc
        reduced = umap.UMAP(n_components=components,
                            random_state=seed).fit_transform(values)
        out = pd.DataFrame(reduced, index=usable.index,
                           columns=[f"UMAP{i + 1}" for i in range(components)])
    else:
        from sklearn.manifold import TSNE
        reduced = TSNE(n_components=min(components, 3),
                       random_state=seed).fit_transform(values)
        out = pd.DataFrame(reduced, index=usable.index,
                           columns=[f"tSNE{i + 1}"
                                    for i in range(reduced.shape[1])])

    # Reindexed to the WHOLE frame: objects that could not be projected keep
    # their row and get NaN, so the components can be added to the table
    # without silently dropping rows out from under every other column.
    return out.reindex(frame.index)
