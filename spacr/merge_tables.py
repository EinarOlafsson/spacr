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
    (r"(^|_)(count|n_objects|number)(_|$)", SUM),
    (r"(^|_)min(imum)?(_|$)", MIN),
    (r"(^|_)max(imum)?(_|$)", MAX),
    (r"(^|_)median(_|$)", MEDIAN),
    (r"(^|_)(mean|average|avg)(_|$)", MEAN),
    # Extent: four objects' areas add up, they do not average.
    (r"(^|_)(area|perimeter|volume|length|width|height|diameter|"
     r"convex_area|filled_area|equivalent_diameter)(_|$)", SUM),
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
    return sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)


def table_names(db_path: str) -> Tuple[str, ...]:
    with _connect(db_path) as db:
        rows = db.execute(
            "SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    return tuple(str(r[0]) for r in rows)


def mergeable_tables(db_path: str) -> Tuple[str, ...]:
    """The object tables in this database, in preference order."""
    present = set(table_names(db_path))
    return tuple(t for t in OBJECT_TABLES if t in present)


def _read(db_path: str, table: str) -> pd.DataFrame:
    with _connect(db_path) as db:
        return pd.read_sql_query(f'SELECT * FROM "{table}"', db)


def _keys_in(frame: pd.DataFrame) -> List[str]:
    return [c for c in IDENTITY if c in frame.columns]


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

    renamed = {c: f"{name}_{c}" for c in out.columns if c not in keys}
    return out.rename(columns=renamed)


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
        if PARENT_LINK not in child.columns:
            # Measured without a parent mask: the roll-up is not empty, it is
            # UNDEFINED. Named and skipped, exactly as io.py does -- one
            # unlinkable table must not cost the user the others.
            LOG.info("%s carries no %s, so it cannot be rolled up onto %s; "
                     "leaving it out", table, PARENT_LINK, policy.primary)
            continue
        child_keys = _keys_in(child) + [PARENT_LINK]
        rolled = roll_up(child, child_keys, name=table, policy=policy)
        rolled = rolled.rename(columns={PARENT_LINK: OBJECT_COLUMN})
        on = [c for c in keys + [OBJECT_COLUMN] if c in rolled.columns]
        if OBJECT_COLUMN not in on:
            LOG.info("%s cannot be joined to %s on an object key", table,
                     policy.primary)
            continue
        merged = merged.merge(rolled, on=on, how="left")

    return _apply_na_policy(merged, policy)


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
                      scale: bool = True,
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
    usable = data.dropna()
    if len(usable) < 3:
        raise ReductionError(
            f"only {len(usable)} object(s) have all of {', '.join(chosen)}; "
            f"there is nothing to project")

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
            import umap
        except ImportError as exc:
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
