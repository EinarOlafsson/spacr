"""Infection metrics, assembled from a finished Measure run.

A REPORT, NOT A PIPELINE, and that was a deliberate choice made after the
derivability was measured: thirteen of the sixteen candidate infection metrics
fall out of tables Measure has already written, with no new image processing.
The three that do not -- intracellular fraction, vacuoles per cell, distance
to the host boundary -- belong to the invasion module and to segmentation, and
are deliberately absent here rather than approximated.

THE DENOMINATOR IS THE WHOLE PROBLEM, and it is why every number this module
produces carries its own denominator beside it. Almost every way of getting an
infection metric wrong is a denominator that does not match its numerator:

* The relationship table has NO ROW for a cell with no parasites.
  ``measure.get_components`` explodes the per-cell child list and drops the
  empty ones, so counting rows there counts INFECTED cells and calls it the
  cell count. The denominator has to come from the ``cell`` table, which
  holds every segmented cell whether or not anything is inside it.
* ``io.py`` drops pathogens with no ``cell_id`` and can filter rows by
  ``pathogen_prcfo_count``. Neither touches the cell table. A numerator built
  from a filtered pathogen table over an unfiltered cell table counts two
  different populations, and the ratio looks entirely plausible.

So :func:`infection_report` returns the denominator, and its size, on every
row. A reader who disagrees with a number can see immediately which of the two
halves they disagree with.

IT DOES NOT INVENT A FIFTH VOCABULARY. ``uninfected``, ``pathogen_count`` and
``pathogen_prcfo_count`` already exist across ``measure``, ``io`` and
``timelapse``. This module reuses them and adds no synonym.
"""
from __future__ import annotations

import sqlite3
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd


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

#: The table holding one row per segmented host cell. THE DENOMINATOR.
CELL_TABLE = "cell"

#: The table holding one row per segmented parasite.
PATHOGEN_TABLE = "pathogen"

#: How a pathogen row names the host cell it sits inside.
HOST_KEY = "cell_id"

#: What a well is, for grouping. Field is deliberately NOT here: a well mean
#: is the unit a screen reports, and the per-field spread is reported
#: separately precisely because a mean hides a settling gradient.
WELL_KEYS: Tuple[str, ...] = ("plateID", "rowID", "columnID")

#: Added to the well keys when the spread across fields is wanted.
FIELD_KEY = "fieldID"


def _present_columns(db: sqlite3.Connection, table: str) -> List[str]:
    """The columns ``table`` actually has, or an empty list if it has none.

    :param db: an open connection.
    :param table: the table to inspect.
    :returns: column names, empty when the table does not exist.
    """
    try:
        rows = db.execute(f'PRAGMA table_info("{table}")').fetchall()
    except sqlite3.Error:
        return []
    return [str(r[1]) for r in rows]


def _canonical(columns: Sequence[str], wanted: str) -> Optional[str]:
    """The spelling ``columns`` uses for the identity column ``wanted``.

    spaCR has written both ``plateID`` and ``plate`` over the years and a
    database can carry either; the aliases live in :mod:`spacr.filters` and
    are reused here rather than duplicated.

    :param columns: the columns a table has.
    :param wanted: the canonical name.
    :returns: the spelling present, or None.
    """
    from .filters import IDENTITY_ALIASES

    have = {c.lower(): c for c in columns}
    for alias in IDENTITY_ALIASES.get(wanted, (wanted,)):
        if alias.lower() in have:
            return have[alias.lower()]
    return None


def _load(db: sqlite3.Connection, table: str,
          columns: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """Read ``table``, or an empty frame when it is absent.

    An absent table is a fact about the run -- a project measured without a
    pathogen channel has no pathogen table -- and not an error to raise into
    a report.

    :param db: an open connection.
    :param table: table name.
    :param columns: restrict to these, when given.
    :returns: the table as a frame.
    """
    have = _present_columns(db, table)
    if not have:
        return pd.DataFrame()
    if columns:
        keep = [c for c in columns if c in have]
        if not keep:
            return pd.DataFrame()
        select = ", ".join(f'"{c}"' for c in keep)
    else:
        select = "*"
    # THROUGH THE FUNNEL. `read_query` is the canonical reader for a
    # connection that is already open -- which is what this helper is
    # handed, because its callers read several tables off one
    # connection and reopening per table would change the transaction
    # each read sees. `report=None` because an absent-or-odd column
    # here is a fact about the run, not something to print into a
    # report the caller is assembling.
    from .tabular import _read_query
    return _read_query(db, f'SELECT {select} FROM "{table}"',
                      report=None)


def parasites_per_cell(db_path: str) -> pd.DataFrame:
    """Every host cell, with how many parasites it contains -- ZERO INCLUDED.

    THE ZEROES ARE THE POINT. The relationship between a cell and its
    parasites is stored on the parasite, so a cell containing none appears
    nowhere in the pathogen table. Any count taken from that table alone is a
    count of infected cells wearing the name of a cell count.

    :param db_path: path to a ``measurements.db``.
    :returns: one row per cell, with the identity columns, ``object_label``
        and ``pathogen_count``. Empty when there is no cell table.
    """
    from .filters import OBJECT_COLUMN

    with _connect_read_only(db_path) as db:
        cells = _load(db, CELL_TABLE)
        pathogens = _load(db, PATHOGEN_TABLE)

    if cells.empty:
        return pd.DataFrame()

    keys = [c for c in (_canonical(cells.columns, k) for k in WELL_KEYS + (FIELD_KEY,))
            if c]
    label = _canonical(cells.columns, OBJECT_COLUMN) or OBJECT_COLUMN
    if label not in cells.columns:
        return pd.DataFrame()

    out = cells[keys + [label]].copy()
    out = out.rename(columns={label: OBJECT_COLUMN})
    out["pathogen_count"] = 0

    if pathogens.empty or HOST_KEY not in pathogens.columns:
        # No pathogen table, or one that never recorded its host: every cell
        # is uninfected as far as this database can say. That is a real
        # answer for an uninfected control plate and a loud one for a plate
        # that should have parasites.
        return out

    p_keys = [c for c in (_canonical(pathogens.columns, k)
                          for k in WELL_KEYS + (FIELD_KEY,)) if c]
    counted = (pathogens.dropna(subset=[HOST_KEY])
               .groupby(p_keys + [HOST_KEY], dropna=False)
               .size().reset_index(name="n"))
    counted = counted.rename(columns={HOST_KEY: OBJECT_COLUMN})
    # The host key is written as a float when it arrives through a merge.
    for frame in (out, counted):
        frame[OBJECT_COLUMN] = pd.to_numeric(frame[OBJECT_COLUMN],
                                             errors="coerce")

    merged = out.merge(counted, how="left", on=keys + [OBJECT_COLUMN])
    merged["pathogen_count"] = merged["n"].fillna(0).astype(int)
    return merged.drop(columns=["n"])


def infection_report(db_path: str, *,
                     by_field: bool = False) -> pd.DataFrame:
    """Infection metrics per well, each with the denominator it was computed over.

    EVERY ROW CARRIES ITS DENOMINATOR because that is where these numbers go
    wrong. ``infection rate`` over "cells" and ``parasites per infected`` over
    "infected cells" are different populations, and a table that reports only
    the ratios cannot be checked.

    :param db_path: path to a ``measurements.db``.
    :param by_field: group by field as well as well, which is what shows a
        settling gradient a well mean hides.
    :returns: tidy frame -- identity columns, then ``metric``, ``value``,
        ``denominator`` and ``n_denominator``. Empty when there is no cell
        table to count.
    """
    per_cell = parasites_per_cell(db_path)
    if per_cell.empty:
        return pd.DataFrame(columns=["metric", "value", "denominator",
                                     "n_denominator"])

    keys = [c for c in per_cell.columns
            if c in {_canonical(per_cell.columns, k) for k in WELL_KEYS}]
    if by_field:
        field = _canonical(per_cell.columns, FIELD_KEY)
        if field:
            keys = keys + [field]

    rows: List[dict] = []
    grouped = per_cell.groupby(keys, dropna=False) if keys else [((), per_cell)]
    for name, block in grouped:
        identity = dict(zip(keys, name if isinstance(name, tuple) else (name,)))
        cells = len(block)
        infected = int((block["pathogen_count"] > 0).sum())
        parasites = int(block["pathogen_count"].sum())

        def add(metric, value, denominator, n):
            """Record one metric together with the population it is over.

            The denominator travels with the value rather than being implied
            by the metric's name, because two of these are ratios over
            different populations and a reader cannot check a ratio whose
            denominator they have to guess.

            :param metric: the metric's name.
            :param value: its value for this group.
            :param denominator: what the value was computed over, in words.
            :param n: how many things that denominator contained.
            """
            rows.append({**identity, "metric": metric, "value": value,
                         "denominator": denominator, "n_denominator": n})

        add("infection_rate", infected / cells if cells else float("nan"),
            "segmented host cells", cells)
        add("infection_index", parasites / cells if cells else float("nan"),
            "segmented host cells", cells)
        add("parasites_per_infected",
            parasites / infected if infected else float("nan"),
            "infected host cells", infected)
        add("cell_count", float(cells), "segmented host cells", cells)
        add("infected_count", float(infected), "segmented host cells", cells)
        add("parasite_count", float(parasites), "parasites with a host cell",
            parasites)
    return pd.DataFrame(rows)


def multiplicity_distribution(db_path: str) -> pd.DataFrame:
    """The histogram behind the means, which is what the means hide.

    Two conditions can share an infection index and differ completely: a few
    heavily infected cells against many lightly infected ones is different
    biology with the same average. The distribution is the number that
    distinguishes them, so it is reported rather than summarised.

    :param db_path: path to a ``measurements.db``.
    :returns: one row per (well, parasite count) with ``cells`` and the
        fraction of the well's cells at that count.
    """
    per_cell = parasites_per_cell(db_path)
    if per_cell.empty:
        return pd.DataFrame(columns=["pathogen_count", "cells", "fraction"])

    keys = [c for c in per_cell.columns
            if c in {_canonical(per_cell.columns, k) for k in WELL_KEYS}]
    counted = (per_cell.groupby(keys + ["pathogen_count"], dropna=False)
               .size().reset_index(name="cells"))
    totals = counted.groupby(keys, dropna=False)["cells"].transform("sum") \
        if keys else counted["cells"].sum()
    counted["fraction"] = counted["cells"] / totals
    return counted


def host_contrast(db_path: str, columns: Sequence[str], *,
                  statistic: str = "mean") -> pd.DataFrame:
    """One host measurement, infected against uninfected, IN THE SAME WELL.

    The comparison is made within a well on purpose: comparing an infected
    well to an uninfected one confounds infection with everything else that
    differs between two wells, and the whole reason this is worth computing
    is that both populations sit in the same one.

    :param db_path: path to a ``measurements.db``.
    :param columns: the host-cell measurements to contrast, e.g. ``("area",
        "eccentricity")``.
    :param statistic: any name ``DataFrameGroupBy.agg`` accepts.
    :returns: one row per (well, measurement) with the infected and
        uninfected values and both group sizes.
    """
    from .filters import OBJECT_COLUMN

    per_cell = parasites_per_cell(db_path)
    if per_cell.empty:
        return pd.DataFrame()

    with _connect_read_only(db_path) as db:
        cells = _load(db, CELL_TABLE)
    if cells.empty:
        return pd.DataFrame()

    label = _canonical(cells.columns, OBJECT_COLUMN) or OBJECT_COLUMN
    keys = [c for c in per_cell.columns
            if c in {_canonical(per_cell.columns, k) for k in WELL_KEYS}]
    join_on = keys + [OBJECT_COLUMN]
    cells = cells.rename(columns={label: OBJECT_COLUMN})
    cells[OBJECT_COLUMN] = pd.to_numeric(cells[OBJECT_COLUMN], errors="coerce")

    wanted = [c for c in columns if c in cells.columns]
    if not wanted:
        return pd.DataFrame()

    merged = per_cell.merge(cells[join_on + wanted], how="left", on=join_on)
    merged["infected"] = merged["pathogen_count"] > 0

    rows: List[dict] = []
    grouped = merged.groupby(keys, dropna=False) if keys else [((), merged)]
    for name, block in grouped:
        identity = dict(zip(keys, name if isinstance(name, tuple) else (name,)))
        yes = block[block["infected"]]
        no = block[~block["infected"]]
        for column in wanted:
            rows.append({
                **identity,
                "measurement": column,
                "infected": (yes[column].agg(statistic) if len(yes) else float("nan")),
                "uninfected": (no[column].agg(statistic) if len(no) else float("nan")),
                "n_infected": len(yes),
                "n_uninfected": len(no),
            })
    return pd.DataFrame(rows)
