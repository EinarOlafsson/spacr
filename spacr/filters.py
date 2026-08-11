"""The ``filters`` table: gates, written back to the database as columns.

A gate drawn in the Gate Editor is a shape on two measurements. What a user
wants out of it is a LABEL -- this object is in my population, that one is not
-- attached to the objects themselves, so it can be merged with anything else
they have measured. That is what this module writes:

    one column per gate, named after the gate, 1 inside and 0 outside.

**Why a separate table.** The columns could be added to ``cell``, but a gate
is not a measurement: it is an interpretation, it is re-drawn often, and it
belongs to whichever object the user was looking at. Writing into the
measurement tables would mix the two, and re-gating would rewrite a table that
the measure step owns. ``filters`` is written only by this module, so it can
be deleted and rebuilt at any time without losing a measurement.

**The bootstrap.** The first gate exported has to create the table, and the
table has to carry enough identity that a filter can be merged back onto ANY
object table or onto ``png_list``. spaCR joins those on
``plate / row / column / field`` plus the object label (and the timepoint,
when the database is a timelapse), so those are exactly the columns
:func:`build_filters_frame` collects -- from every object table present, not
just the anchor, because a gate drawn on nucleus measurements has to merge
onto nuclei.

**A tolerant reader, a strict writer.** Databases in the wild carry both the
current column names and the ones spaCR wrote years ago, so reading accepts
either spelling. Everything written out uses the canonical name, so the
``filters`` table itself never needs the alias machinery.
"""
from __future__ import annotations

import logging
import re
import sqlite3
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

LOG = logging.getLogger("spacr.filters")

#: The table this module owns.
FILTERS_TABLE = "filters"

#: Object tables, in the order one is chosen as the anchor. Preference, not
#: requirement: the whole point is that a database with ONLY nuclei, or only
#: pathogens, or only organelles works exactly as well as the usual one.
OBJECT_TABLES: Tuple[str, ...] = (
    "cell", "nucleus", "pathogen", "cytoplasm", "organelle",
)

#: The crop table, joined on the same keys when it exists.
PNG_TABLE = "png_list"

#: Canonical identity columns, and the spellings accepted for each. spaCR has
#: written both over the years and a database can carry either; a filter that
#: silently failed to merge because a column was called ``row`` instead of
#: ``rowID`` would look like a gate that selected nothing.
IDENTITY_ALIASES: Dict[str, Tuple[str, ...]] = {
    "plateID": ("plateID", "plate", "plate_name", "plateid"),
    "rowID": ("rowID", "row", "row_name", "rowid_", "rowid"),
    "columnID": ("columnID", "column", "col", "column_name", "columnid"),
    "fieldID": ("fieldID", "field", "field_name", "fieldid"),
}

#: Canonical identity columns in join order.
IDENTITY_COLUMNS: Tuple[str, ...] = tuple(IDENTITY_ALIASES)

#: The object key. Integer in every object table.
OBJECT_COLUMN = "object_label"

#: Timepoint spellings. Carried into ``filters`` when the database has one,
#: because on a timelapse the same object label recurs every frame and a join
#: without it is many-to-many -- the bug already documented in
#: :func:`spacr.io._read_and_join_tables`.
TIME_ALIASES: Tuple[str, ...] = ("timeID", "time_id")
TIME_COLUMN = "timeID"

#: Where a crop path lives in ``png_list``.
PNG_PATH_ALIASES: Tuple[str, ...] = ("png_path", "path", "file_path", "filepath")

#: Prefix marking a column as "this object appears in that table". Not a
#: measurement and not a filter, so it is namespaced away from both.
PRESENT_PREFIX = "in_"

#: A gate name has to survive becoming a SQL column name.
_SAFE_NAME = re.compile(r"[^0-9A-Za-z_]+")


class FilterError(ValueError):
    """A filter that cannot be built or written, with the reason."""


# ---------------------------------------------------------------------------
# Reading the database
# ---------------------------------------------------------------------------

def _connect(db_path: str, *, read_only: bool = True) -> sqlite3.Connection:
    if read_only:
        return sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    return sqlite3.connect(db_path)


def table_names(db_path: str) -> Tuple[str, ...]:
    """Every table in the database, in the order SQLite lists them.

    :param db_path: the measurement database, opened read-only through a
        SQLite URI. Read-only mode does not create a file, so a path that is
        not there raises :class:`sqlite3.OperationalError` rather than
        returning an empty tuple -- "no tables" always means an empty
        database, never a wrong path.
    """
    with _connect(db_path) as db:
        rows = db.execute(
            "SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    return tuple(str(r[0]) for r in rows)


def column_names(db_path: str, table: str) -> Tuple[str, ...]:
    """Every column in ``table``, in the order the database declares them.

    :param db_path: path to the SQLite database.
    :param table: table to describe.
    :returns: the column names; empty when the table does not exist.
    """
    with _connect(db_path) as db:
        rows = db.execute(f'PRAGMA table_info("{table}")').fetchall()
    return tuple(str(r[1]) for r in rows)


def object_tables(db_path: str) -> Tuple[str, ...]:
    """The object tables this database actually has, in preference order.

    Step 1 of the bootstrap: *check which tables are in the database*. Only
    tables that exist AND carry an object label count -- a table can be
    present and empty of the identity a filter needs, and discovering that at
    merge time rather than here would produce a filter that quietly matches
    nothing.

    :param db_path: the measurement database. Only the names in
        :data:`OBJECT_TABLES` are looked for, so a table holding some other
        kind of object is invisible here however it is keyed -- add it to that
        tuple rather than expecting discovery.
    """
    present = set(table_names(db_path))
    out: List[str] = []
    for name in OBJECT_TABLES:
        if name not in present:
            continue
        if resolve_column(column_names(db_path, name), (OBJECT_COLUMN,)) is None:
            LOG.info("table %r has no %s; not usable for filters",
                     name, OBJECT_COLUMN)
            continue
        out.append(name)
    return tuple(out)


def choose_anchor(tables: Sequence[str]) -> str:
    """The table the metadata is taken from.

    "usually be cell, but if cell does not exist then another table should be
    used". Preference order, so a database of only nuclei anchors on nuclei
    and a database of only organelles on organelles -- there is no table that
    has to be present.

    :param tables: the object tables actually present, as returned by
        :func:`object_tables`. Only membership is tested -- the preference
        comes from the order of :data:`OBJECT_TABLES`, not from the order
        given here, so putting ``nucleus`` first does not make it the anchor
        while ``cell`` is also in the sequence.
    :raises FilterError: nothing to anchor on. Naming the tables that WERE
        found is the difference between a fixable message and a shrug.
    """
    for name in OBJECT_TABLES:
        if name in tables:
            return name
    raise FilterError(
        "this database has no object table to build filters from; looked for "
        + ", ".join(OBJECT_TABLES))


def resolve_column(columns: Iterable[str],
                   aliases: Sequence[str]) -> Optional[str]:
    """The first alias present in ``columns``, matched case-insensitively.

    Returns the name AS SPELLED in the table, which is what has to go in the
    SQL -- returning the canonical spelling would produce queries for columns
    that are not there.

    :param columns: the column names as the table spells them, typically from
        :func:`column_names`. Where a table carries two spellings that differ
        only in case, the one appearing LAST is the one returned.
    :param aliases: candidate spellings in preference order; the first one
        present wins, which is why the canonical name leads every entry of
        :data:`IDENTITY_ALIASES`. A single-element tuple is the way to ask
        "does this exact column exist, whatever its case".
    """
    lookup = {str(c).lower(): str(c) for c in columns}
    for alias in aliases:
        hit = lookup.get(str(alias).lower())
        if hit is not None:
            return hit
    return None


def identity_columns_of(db_path: str, table: str) -> Dict[str, str]:
    """Map canonical identity name -> the spelling ``table`` uses.

    Missing columns are simply absent from the map. A table without a field
    column still merges on the keys it does have; refusing outright would rule
    out databases that are perfectly usable.

    :param db_path: opened read-only, so unlike a missing table a database
        file that is not there is fatal: it raises
        :class:`sqlite3.OperationalError` instead of an empty map.
    :param table: the table to inspect; it need not be an object table. A name
        that is not in the database is not an error -- ``PRAGMA table_info``
        returns nothing for it, so the result is an empty map, the same answer
        as a table that exists and carries no identity at all.
    """
    columns = column_names(db_path, table)
    found: Dict[str, str] = {}
    for canonical, aliases in IDENTITY_ALIASES.items():
        actual = resolve_column(columns, aliases)
        if actual is not None:
            found[canonical] = actual
    time_column = resolve_column(columns, TIME_ALIASES)
    if time_column is not None:
        found[TIME_COLUMN] = time_column
    object_column = resolve_column(columns, (OBJECT_COLUMN,))
    if object_column is not None:
        found[OBJECT_COLUMN] = object_column
    return found


def read_identity(db_path: str, table: str) -> pd.DataFrame:
    """The identity columns of one table, and nothing else.

    Identity only, because this runs over every object table in the database
    and a measurement table is wide -- hundreds of columns of which four
    matter. ``SELECT *`` here is the difference between a bootstrap that takes
    a moment and one that reads the whole database.

    :param table: must carry an ``object_label`` column -- that name, in any
        case. Every other identity column is optional and is simply absent
        from the returned frame; the object label is not, because without it
        the rows cannot be told apart.
    :raises FilterError: ``table`` has no object label column.
    """
    mapping = identity_columns_of(db_path, table)
    if OBJECT_COLUMN not in mapping:
        raise FilterError(
            f"table {table!r} has no {OBJECT_COLUMN} column, so its rows "
            f"cannot be identified")
    select = ", ".join(f'"{actual}" AS "{canonical}"'
                       for canonical, actual in mapping.items())
    with _connect(db_path) as db:
        frame = pd.read_sql_query(f'SELECT {select} FROM "{table}"', db)
    return frame


# ---------------------------------------------------------------------------
# Building the table
# ---------------------------------------------------------------------------

def key_columns(frame: pd.DataFrame) -> List[str]:
    """The identity columns present, in join order.

    One definition, used by the bootstrap, the merge and the writer, so a
    filter can never be merged on a different key than it was built with.

    :param frame: a frame whose columns use the CANONICAL identity names --
        what :func:`read_identity` returns, having aliased them on the way
        out. Membership is tested exactly and case-sensitively, with no alias
        lookup: a raw ``SELECT *`` from a database that spells them ``plate``
        and ``row`` contributes none of the well keys, and such a frame yields
        ``["object_label"]`` alone. That is why nothing merges on the result
        of this function without putting it through
        :func:`require_full_identity` first -- an object label is unique only
        WITHIN a field, so merging on it alone collides objects across plates,
        wells and fields. Only column names are inspected; no values are
        touched.
    """
    keys = [c for c in IDENTITY_COLUMNS if c in frame.columns]
    if TIME_COLUMN in frame.columns:
        keys.append(TIME_COLUMN)
    if OBJECT_COLUMN in frame.columns:
        keys.append(OBJECT_COLUMN)
    return keys


def require_full_identity(keys: Sequence[str], what: str) -> None:
    """Raise unless ``keys`` names every identity column AND the object label.

    The invariant a write-back depends on: ``object_label`` is unique only
    within one field of one well of one plate, so a merge keyed on anything
    less than the full identity silently joins one plate's object 7 onto
    another's. The partial-identity tolerance elsewhere in this module
    (:func:`identity_columns_of`, :func:`build_filters_frame`) is about
    READING a database that is missing a column; writing a per-object column
    back into it is the one operation where a partial key is a wrong answer
    rather than a reduced one.

    :param keys: what :func:`key_columns` returned for the frame.
    :param what: names the operation in the error message, e.g. ``"a gate"``.
    :raises FilterError: any identity column, or the object label, is absent.
    """
    missing = [c for c in IDENTITY_COLUMNS if c not in keys]
    if OBJECT_COLUMN not in keys:
        missing.append(OBJECT_COLUMN)
    if missing:
        raise FilterError(
            f"this table has no object identity (missing "
            f"{', '.join(missing)}; {what} needs "
            f"{', '.join(IDENTITY_COLUMNS)}, {OBJECT_COLUMN} because an "
            f"object label repeats in every field), so {what} on it cannot be "
            f"written back to the database")


def _png_paths(db_path: str) -> Optional[pd.DataFrame]:
    """``png_list``'s identity and crop path, or None if there is no such table.

    ``png_list`` keys its object as TEXT (``'o5'``) while every object table
    uses an integer, so the two are reconciled through the one function that
    already knows every way that goes wrong -- ``'omulti'``, ``'onone'``,
    ``'error'`` and NULL, all of which real writers produce. Reimplementing
    that translation here would be a second place for it to be wrong.
    """
    if PNG_TABLE not in table_names(db_path):
        return None
    columns = column_names(db_path, PNG_TABLE)
    path_column = resolve_column(columns, PNG_PATH_ALIASES)
    if path_column is None:
        LOG.info("%s has no path column; filters will carry no crop paths",
                 PNG_TABLE)
        return None

    from .utils import PNG_OBJECT_ID_COLUMNS, object_label_from_png_id

    id_column = resolve_column(columns, tuple(PNG_OBJECT_ID_COLUMNS.values()))
    if id_column is None:
        LOG.info("%s carries no object id; filters will carry no crop paths",
                 PNG_TABLE)
        return None

    mapping = {canonical: actual
               for canonical, aliases in IDENTITY_ALIASES.items()
               for actual in [resolve_column(columns, aliases)]
               if actual is not None}
    time_column = resolve_column(columns, TIME_ALIASES)
    if time_column is not None:
        mapping[TIME_COLUMN] = time_column
    select = ", ".join(
        [f'"{actual}" AS "{canonical}"' for canonical, actual in mapping.items()]
        + [f'"{id_column}" AS "_png_object_id"', f'"{path_column}" AS "png_path"'])
    with _connect(db_path) as db:
        frame = pd.read_sql_query(f'SELECT {select} FROM "{PNG_TABLE}"', db)

    labels = object_label_from_png_id(frame["_png_object_id"])
    frame[OBJECT_COLUMN] = labels
    dropped = int(frame[OBJECT_COLUMN].isna().sum())
    if dropped:
        # Not an error: 'omulti'/'onone'/'error'/NULL are states real crops
        # are in. The object keeps its measurements and simply has no path.
        LOG.info("%d %s row(s) have no usable object id; those objects get "
                 "no crop path", dropped, PNG_TABLE)
    frame = frame.dropna(subset=[OBJECT_COLUMN]).copy()
    frame[OBJECT_COLUMN] = frame[OBJECT_COLUMN].astype("int64")
    return frame.drop(columns=["_png_object_id"])


def build_filters_frame(db_path: str) -> pd.DataFrame:
    """The ``filters`` table's identity, before any gate is written to it.

    The three steps asked for, in order:

    1. which tables are in the database (:func:`object_tables`);
    2. the object numbers from EVERY object table, plus the metadata --
       plate, row, column, field, object -- and the crop paths from
       ``png_list`` when it exists;
    3. one table carrying enough to merge a filter onto any of them.

    Every object table contributes rows, not just the anchor. A gate drawn on
    nucleus measurements has to merge onto nuclei, and a filters table built
    only from ``cell`` could not express that. Each object also carries an
    ``in_<table>`` flag saying which tables it appears in, which is what makes
    a merge predictable rather than a thing you discover by counting rows.

    :param db_path: the measurement database.
    :returns: one row per distinct object key.
    :raises FilterError: no object table to build from.
    """
    try:
        return build_filters_from_relationships(db_path)
    except FilterError:
        raise
    except Exception:
        # The relationships route is the intended one; this fallback keeps a
        # database whose relationships cannot be built (an unreadable table,
        # a schema nobody anticipated) gateable rather than blocked.
        LOG.info("could not build %s from %s; falling back to the object "
                 "tables", FILTERS_TABLE, RELATIONSHIPS_TABLE, exc_info=True)

    tables = object_tables(db_path)
    anchor = choose_anchor(tables)
    LOG.info("building %s from %s, anchored on %r",
             FILTERS_TABLE, ", ".join(tables), anchor)

    frames: Dict[str, pd.DataFrame] = {}
    for table in tables:
        try:
            frames[table] = read_identity(db_path, table)
        except FilterError as exc:
            LOG.info("skipping %r: %s", table, exc)

    if not frames:
        raise FilterError(
            f"none of {', '.join(tables)} carries the identity columns a "
            f"filter needs")

    # The anchor decides the key set. A table with fewer identity columns is
    # merged on what it shares, rather than being dropped for lacking a
    # column the others happen to have.
    keys = key_columns(frames[anchor])
    out = frames[anchor][keys].drop_duplicates().copy()
    out[f"{PRESENT_PREFIX}{anchor}"] = 1

    for table, frame in frames.items():
        if table == anchor:
            continue
        shared = [k for k in keys if k in frame.columns]
        if not shared:
            LOG.info("%r shares no identity with the anchor; not merged", table)
            continue
        side = frame[shared].drop_duplicates().copy()
        side[f"{PRESENT_PREFIX}{table}"] = 1
        out = out.merge(side, on=shared, how="outer")

    for table in frames:
        column = f"{PRESENT_PREFIX}{table}"
        if column in out.columns:
            out[column] = out[column].fillna(0).astype("int64")

    paths = _png_paths(db_path)
    if paths is not None:
        shared = [k for k in keys if k in paths.columns]
        if shared:
            paths = paths[shared + ["png_path"]].drop_duplicates(subset=shared)
            out = out.merge(paths, on=shared, how="left")
        else:
            LOG.info("%s shares no identity columns; no crop paths carried",
                     PNG_TABLE)

    return out.reset_index(drop=True)


#: The table holding which object belongs to which. Written after masking,
#: and the base every filters table is copied from.
RELATIONSHIPS_TABLE = "relationships"


def build_relationships_frame(db_path: str) -> pd.DataFrame:
    """Every object relationship in the database, as one flat table.

    One row per object of the FINEST kind present, carrying the label of each
    coarser object it belongs to. Flat rather than a link table per pair
    because every question asked of it -- "cells with more than three
    pathogens", "the mean pathogen intensity per cell" -- is a group-by on
    the parent, and a flat table answers those with no joins at all.

    The parent link is ``cell_id`` on the child, which is what
    :func:`spacr.io._read_and_join_tables` uses and what the measure step
    writes. A child measured without a parent mask has no link, and is
    carried with a null parent rather than dropped: the object exists, and
    saying it has no parent is different from pretending it is not there.

    :param db_path: the measurement database. Read-only: the frame is returned
        and nothing is stored, so a caller that wants it on disk goes through
        :func:`write_relationships` or :func:`ensure_relationships_table`.
    :raises FilterError: no object table to build from.
    """
    tables = object_tables(db_path)
    if not tables:
        raise FilterError(
            "this database has no object table to build relationships from")

    frames: List[pd.DataFrame] = []
    for table in tables:
        columns = column_names(db_path, table)
        mapping = identity_columns_of(db_path, table)
        select = [f'"{actual}" AS "{canonical}"'
                  for canonical, actual in mapping.items()]
        link = resolve_column(columns, ("cell_id",))
        if link is not None:
            select.append(f'"{link}" AS "parent_label"')
        with _connect(db_path) as db:
            frame = pd.read_sql_query(
                f'SELECT {", ".join(select)} FROM "{table}"', db)
        if "parent_label" not in frame.columns:
            frame["parent_label"] = pd.NA
        frame["object_type"] = table
        frames.append(frame)

    out = pd.concat(frames, ignore_index=True, sort=False)
    out["parent_label"] = pd.to_numeric(out["parent_label"],
                                        errors="coerce").astype("Int64")
    out["parent_type"] = np.where(out["parent_label"].notna(), "cell", None)
    return out.reset_index(drop=True)


def ensure_relationships_table(db_path: str, *,
                               rebuild: bool = False) -> pd.DataFrame:
    """The relationships table, built on demand if the mask step never did.

    "if the relationships table does not exist when attempting to make the
    filters table, then the relationships table gets generated first." A user
    who gated before ever running the new mask step must not hit an error
    for it.

    :param db_path: the measurement database. Opened for WRITING whenever the
        table has to be built, so a database that is only readable can be
        gated on if it was bootstrapped already, and not otherwise.
    :param rebuild: derive the table again from the object tables and replace
        whatever is stored. Relationships are never refreshed on their own, so
        a stored table built before the masks changed stays stale until this
        is passed -- which is exactly what :func:`write_relationships` does.
    """
    if not rebuild and RELATIONSHIPS_TABLE in table_names(db_path):
        with _connect(db_path) as db:
            return pd.read_sql_query(
                f'SELECT * FROM "{RELATIONSHIPS_TABLE}"', db)
    frame = build_relationships_frame(db_path)
    with _connect(db_path, read_only=False) as db:
        frame.to_sql(RELATIONSHIPS_TABLE, db, if_exists="replace", index=False)
    return frame


def write_relationships(db_path: str) -> pd.DataFrame:
    """Build and store the relationships table. Called after the mask step.

    Separate from :func:`ensure_relationships_table` so the mask step can say
    "rebuild this, the masks just changed" without a caller having to know
    the flag.

    :param db_path: the measurement database, opened for writing. Any stored
        ``relationships`` table is REPLACED, so this is the call to make after
        re-masking and the wrong one to make merely to read the table.
    """
    return ensure_relationships_table(db_path, rebuild=True)


def ensure_filters_table(db_path: str, *, rebuild: bool = False) -> pd.DataFrame:
    """Return the ``filters`` table, building it the first time.

    :param rebuild: discard and rebuild. The identity is derived entirely from
        the object tables, but any gate columns already written are LOST --
        which is why this is a parameter and not something the export path
        does on its own.
    :returns: the table as it now stands on disk.
    """
    if not rebuild and FILTERS_TABLE in table_names(db_path):
        with _connect(db_path) as db:
            return pd.read_sql_query(f'SELECT * FROM "{FILTERS_TABLE}"', db)

    frame = build_filters_frame(db_path)
    write_filters_table(db_path, frame)
    return frame


def build_filters_from_relationships(db_path: str) -> pd.DataFrame:
    """The filters table as a COPY of the relationships table.

    "to make the filters table this the relationships table should be the
    base (it should be copied) and filters added." A copy rather than a merge
    onto it: a filter then automatically carries every relationship, and
    there is one definition of what an object is rather than two that can
    drift.

    :param db_path: the measurement database. Despite being a build step this
        can WRITE: the relationships table it copies is created on demand when
        the mask step never wrote one. An existing one is used as it stands --
        pass through :func:`write_relationships` first if the masks have
        changed since it was written.
    """
    frame = ensure_relationships_table(db_path).copy()

    # The `in_<table>` flags say the same thing `object_type` does, and are
    # kept because a merge asks "is this object a nucleus" as a column test
    # far more often than as a string comparison. Derived here rather than
    # stored twice in the relationships table itself.
    for table in object_tables(db_path):
        frame[f"{PRESENT_PREFIX}{table}"] = (
            frame["object_type"] == table).astype("int64")

    paths = _png_paths(db_path)
    if paths is not None:
        shared = [k for k in key_columns(frame) if k in paths.columns]
        if shared:
            paths = paths[shared + ["png_path"]].drop_duplicates(subset=shared)
            frame = frame.merge(paths, on=shared, how="left")
    return frame


def write_filters_table(db_path: str, frame: pd.DataFrame) -> None:
    """Replace ``filters`` with ``frame``.

    Whole-table replace rather than ALTER + UPDATE: the table is small (one
    row per object, a handful of columns), it is owned entirely by this
    module, and a partial write that left a gate column half-populated would
    be indistinguishable from a gate that selected those rows.

    :param db_path: the measurement database, opened for writing.
    :param frame: the WHOLE table as it should end up on disk. Since the write
        replaces rather than merges, any gate column missing from ``frame`` is
        dropped from the database -- callers read the current table, add their
        column to it and pass the result back, never the column on its own.
    """
    with _connect(db_path, read_only=False) as db:
        frame.to_sql(FILTERS_TABLE, db, if_exists="replace", index=False)


# ---------------------------------------------------------------------------
# Writing a gate
# ---------------------------------------------------------------------------

def column_name_for(gate_name: str) -> str:
    """The column a gate is written to.

    The gate's own name, with anything that is not a letter, digit or
    underscore collapsed to an underscore. The name is what the user reads in
    both places, so it is kept recognisable rather than hashed.

    :param gate_name: the gate's display name. It is stripped, every run of
        characters outside ``[0-9A-Za-z_]`` collapses to one underscore,
        leading and trailing underscores are dropped, and a result starting
        with a digit is prefixed ``g_`` -- legal in quoted SQLite but not in
        the tools that read the table afterwards. Two gates whose names differ
        only in punctuation therefore land on the SAME column, and
        :func:`export_gate` replaces rather than suffixes.
    :raises FilterError: a name with nothing usable left in it, which would
        otherwise become an anonymous column called ``_``.
    """
    cleaned = _SAFE_NAME.sub("_", str(gate_name).strip()).strip("_")
    if not cleaned:
        raise FilterError(
            f"gate name {gate_name!r} has no letters or digits in it, so it "
            f"cannot become a column name")
    if cleaned[0].isdigit():
        # A leading digit is legal in a quoted SQLite column but trips up
        # every tool that reads the table afterwards, pandas query included.
        cleaned = f"g_{cleaned}"
    return cleaned


def export_gate(db_path: str, frame: pd.DataFrame, inside: np.ndarray,
                gate_name: str, *, rebuild: bool = False) -> Tuple[str, int]:
    """Write one gate to ``filters`` as a 1/0 column.

    :param frame: the objects the gate was evaluated on. Must carry the FULL
        identity (:data:`IDENTITY_COLUMNS` plus ``object_label``, in the
        canonical spellings -- see :func:`require_full_identity`); the
        measurements are not needed and not read.
    :param inside: boolean mask over ``frame``, True for objects in the gate.
    :param gate_name: names the column.
    :param rebuild: rebuild the identity table first, discarding gate columns.
    :returns: ``(column name, objects marked)``.
    :raises FilterError: the frame is missing any part of the object identity,
        or the mask does not match it.

    Objects NOT in ``frame`` get 0, not null. A user who gated on a 20% sample
    and exported it would otherwise get a column that is null for four objects
    in five, and null is not what "outside the gate" means. The GUI re-reads
    the full table before calling this precisely so that the 0s are real --
    see :func:`gate_mask_over_table`.
    """
    inside = np.asarray(inside, dtype=bool)
    if len(inside) != len(frame):
        raise FilterError(
            f"the gate mask has {len(inside):,} value(s) but the table has "
            f"{len(frame):,} row(s)")

    column = column_name_for(gate_name)
    keys = key_columns(frame)
    require_full_identity(keys, "a gate")

    filters = ensure_filters_table(db_path, rebuild=rebuild)
    shared = [k for k in keys if k in filters.columns]
    if OBJECT_COLUMN not in shared:
        raise FilterError(
            f"the {FILTERS_TABLE} table and this measurement table share no "
            f"object key, so the gate cannot be merged onto it")

    marked = frame.loc[inside, shared].drop_duplicates().copy()
    marked[column] = 1

    if column in filters.columns:
        # Re-exporting a gate REPLACES it. The alternative -- refusing, or
        # suffixing -- leaves the user with filters_2, filters_3 and no way to
        # tell which one is the gate currently on screen.
        LOG.info("replacing existing filter column %r", column)
        filters = filters.drop(columns=[column])

    filters = filters.merge(marked, on=shared, how="left")
    filters[column] = filters[column].fillna(0).astype("int64")
    write_filters_table(db_path, filters)
    return column, int(filters[column].sum())


def gate_mask_over_table(db_path: str, table: str, gates, gate_name: str,
                         ) -> Tuple[pd.DataFrame, np.ndarray]:
    """Apply a gate to EVERY object, reading only the columns it needs.

    The point of the sampling setting is that a user gates on a fraction of a
    large table; the point of this is that the export does not. The gate's own
    columns plus the identity columns are read in full -- a handful out of
    hundreds, so this stays cheap even where reading the whole table is not.

    :param gates: a ``GateSet``.
    :param gate_name: which gate in it.
    :returns: ``(identity frame, mask)`` ready for :func:`export_gate`.
    :raises FilterError: a column the gate needs is not in the table.
    """
    needed: List[str] = []
    for gate in gates.path(gate_name):
        needed.extend(gate.columns)

    available = column_names(db_path, table)
    identity = identity_columns_of(db_path, table)
    missing = [c for c in needed if resolve_column(available, (c,)) is None]
    if missing:
        raise FilterError(
            f"table {table!r} does not have {', '.join(sorted(set(missing)))}, "
            f"which gate {gate_name!r} is drawn on")

    select = [f'"{actual}" AS "{canonical}"'
              for canonical, actual in identity.items()]
    select += [f'"{resolve_column(available, (c,))}" AS "{c}"'
               for c in dict.fromkeys(needed)]
    with _connect(db_path) as db:
        frame = pd.read_sql_query(
            f'SELECT {", ".join(select)} FROM "{table}"', db)

    mask = gates.mask(frame, gate_name)
    return frame, np.asarray(mask, dtype=bool)


# ---------------------------------------------------------------------------
# Annotating from several gates at once
# ---------------------------------------------------------------------------

#: How several gates become ONE label.
ANNOTATION_MODES: Tuple[str, ...] = ("binary", "multiclass")


def combination_label(memberships: Sequence[bool], names: Sequence[str]) -> str:
    """The class name for one combination of gate memberships.

    Named after the gates the object IS in, in gate order, so the label reads
    as what it means -- ``live+CD8`` rather than ``class_3``. An object in no
    gate is ``none``, which is a real class: "outside everything" is a
    finding, not a gap.

    :param memberships: one truth value per gate, for ONE object, positionally
        aligned to ``names``.
    :param names: the gate names, in the order the label reads. Order is part
        of the class name -- ``live+CD8`` and ``CD8+live`` are two different
        classes -- so the same order has to be used for every object, which is
        why :func:`annotate_from_gates` fixes it once. The two sequences are
        zipped, so a longer one is truncated silently rather than reported.
    """
    inside = [name for name, is_in in zip(names, memberships) if is_in]
    return "+".join(inside) if inside else "none"


def annotate_from_gates(frame: pd.DataFrame, gates, names: Sequence[str], *,
                        mode: str = "binary") -> pd.Series:
    """Label every object from SEVERAL gates at once.

    ``binary``
        1 when the object is inside EVERY chosen gate, 0 otherwise. The
        intersection, because that is what "annotate based on all the gates"
        means when the answer has to be one column.
    ``multiclass``
        one class per observed combination of memberships. Only combinations
        that actually occur become classes -- enumerating all 2^n would offer
        classes with no objects in them, which no classifier can learn and
        every class-balance report would then have to explain.

    :param gates: a ``GateSet``.
    :param names: which gates to use, in the order the label reads.
    :returns: a Series aligned to ``frame`` -- integers for binary, class
        names for multiclass.
    :raises FilterError: no gates chosen, or a mode that does not exist.
    """
    if mode not in ANNOTATION_MODES:
        raise FilterError(
            f"{mode!r} is not one of {list(ANNOTATION_MODES)}")
    chosen = [n for n in names if n]
    if not chosen:
        raise FilterError("choose at least one gate to annotate from")

    masks = []
    for name in chosen:
        try:
            masks.append(np.asarray(gates.mask(frame, name), dtype=bool))
        except Exception as exc:
            raise FilterError(
                f"gate {name!r} cannot be applied to this table: {exc}") from exc

    if mode == "binary":
        inside = np.ones(len(frame), dtype=bool)
        for mask in masks:
            inside &= mask
        return pd.Series(inside.astype("int64"), index=frame.index)

    stacked = np.column_stack(masks) if masks else np.zeros((len(frame), 0), bool)
    labels = [combination_label(row, chosen) for row in stacked]
    return pd.Series(labels, index=frame.index, dtype="object")


def export_annotation(db_path: str, frame: pd.DataFrame, labels: pd.Series,
                      column: str) -> Tuple[str, int]:
    """Write a gate-derived annotation to ``filters`` as one column.

    Through the same path a single gate takes, so an annotation and a filter
    are the same kind of thing in the database and merge the same way.

    :param db_path: the measurement database. ``filters`` is built first if it
        is not there yet, and rewritten whole afterwards.
    :param frame: the objects the labels describe. Only its identity columns
        are read -- the measurements are not needed -- and it must carry the
        FULL identity (:data:`IDENTITY_COLUMNS` plus ``object_label``, in the
        canonical spellings), because an object label repeats in every field.
        See :func:`require_full_identity`.
    :param labels: one label per row of ``frame``, taken POSITIONALLY: the
        Series index is discarded, so labels that were reindexed or sorted
        away from the frame's row order would be attached to the wrong
        objects. A length mismatch is not checked here and surfaces as a
        pandas ``ValueError``. Where several rows share one object key the
        first label wins.
    :param column: the name to write it under, sanitised by
        :func:`column_name_for`; an existing column of that name is dropped
        and replaced. Objects outside ``frame`` are left NULL rather than
        filled, since a multiclass annotation has no zero.
    :returns: ``(column name, objects labelled)``.
    :raises FilterError: ``frame`` is missing any part of the object identity,
        or shares no object key with the ``filters`` table.
    """
    name = column_name_for(column)
    keys = key_columns(frame)
    require_full_identity(keys, "an annotation")

    filters = ensure_filters_table(db_path)
    shared = [k for k in keys if k in filters.columns]
    if OBJECT_COLUMN not in shared:
        raise FilterError(
            f"the {FILTERS_TABLE} table and this measurement table share no "
            f"object key")

    marked = frame[shared].copy()
    marked[name] = labels.to_numpy()
    marked = marked.drop_duplicates(subset=shared)

    if name in filters.columns:
        filters = filters.drop(columns=[name])
    filters = filters.merge(marked, on=shared, how="left")
    # Unlabelled objects are left blank rather than filled: a multiclass
    # annotation has no zero, and inventing one would create a class.
    write_filters_table(db_path, filters)
    return name, int(filters[name].notna().sum())


# ---------------------------------------------------------------------------
# Sampling -- the reason the module is laggy on a real dataset
# ---------------------------------------------------------------------------

#: SQLite's names for the implicit row id. Any of them can be SHADOWED by a
#: user column of that name, in which case it refers to the user's column
#: instead -- so the right one has to be chosen per table rather than assumed.
ROWID_ALIASES: Tuple[str, ...] = ("_rowid_", "rowid", "oid")


def rowid_expression(columns: Iterable[str]) -> Optional[str]:
    """Which spelling of the implicit row id this table leaves usable.

    **This is not hypothetical.** Every spaCR measurement table has a column
    called ``rowID`` -- the row of the plate -- and SQLite matches column
    names case-insensitively, so ``rowid`` in a query means THAT column. It
    holds 'A'..'P', and ``'A' % 5`` is ``0`` in SQLite, so a sampling clause
    written the obvious way is true for every row and silently samples
    nothing. The symptom is a sampling setting that appears to do nothing,
    which is indistinguishable from the read being slow for another reason.

    :param columns: the table's own column names, typically from
        :func:`column_names`. Compared case-insensitively, which is the whole
        point: it is ``rowID`` that shadows ``rowid``. Pass the columns of the
        table about to be queried -- an alias is only unusable relative to a
        particular table.
    :returns: the first alias not shadowed by a real column, or ``None`` for
        a table that shadows all three (or has no row id at all).
    """
    taken = {str(c).lower() for c in columns}
    for alias in ROWID_ALIASES:
        if alias not in taken:
            return alias
    return None


def sampling_clause(fraction: float, rowid: str = "_rowid_") -> str:
    """A SQL fragment taking roughly ``fraction`` of the rows.

    Systematic on the row id, not ``ORDER BY RANDOM()``: random ordering sorts
    the whole table before discarding most of it, which costs MORE than
    reading everything and is the opposite of the point. Modulo on the row id
    is an index scan and is also reproducible -- the same 20% every time, so a
    gate drawn on Monday sits on the same cloud on Tuesday.

    The bias this trades for is that row id order is insertion order, i.e.
    roughly well by well. For drawing a gate on a cloud of a million objects
    that is not a distinction that matters; for anything where it might, the
    export applies the gate to every row regardless of what was sampled.

    :param fraction: in (0, 1]. 1 means everything, and returns no clause.
    :param rowid: which row id spelling to use -- see
        :func:`rowid_expression`, which is what picks it.
    :raises FilterError: a fraction outside (0, 1].
    """
    value = float(fraction)
    if not 0 < value <= 1:
        raise FilterError(
            f"sample fraction {fraction!r} is not a fraction between 0 and 1")
    if value >= 1:
        return ""
    step = max(2, int(round(1.0 / value)))
    return f'"{rowid}" % {step} = 0'


def read_sampled(db_path: str, table: str, *, fraction: float = 1.0,
                 limit: Optional[int] = None) -> pd.DataFrame:
    """Read ``table``, optionally taking only a fraction of its rows.

    Sampling happens in SQL where it can, because the point is to not read
    the rows. A table that shadows every row id alias is read whole and
    sampled afterwards -- slower, but correct, and it says so in the log
    rather than quietly returning everything.

    :param fraction: how much of the table to read, in (0, 1].
    :param limit: a hard row cap applied after the fraction.
    """
    value = float(fraction)
    if not 0 < value <= 1:
        raise FilterError(
            f"sample fraction {fraction!r} is not a fraction between 0 and 1")

    rowid = rowid_expression(column_names(db_path, table))
    query = f'SELECT * FROM "{table}"'
    fell_back = False
    if value < 1:
        if rowid is not None:
            query += f" WHERE {sampling_clause(value, rowid)}"
        else:
            fell_back = True
            LOG.info("table %r shadows every row id alias; sampling in pandas "
                     "after reading it whole", table)
    if limit and not fell_back:
        query += f" LIMIT {int(limit)}"

    with _connect(db_path) as db:
        frame = pd.read_sql_query(query, db)

    if fell_back:
        step = max(2, int(round(1.0 / value)))
        frame = frame.iloc[::step].reset_index(drop=True)
        if limit:
            frame = frame.iloc[:int(limit)]
    return frame


def row_count(db_path: str, table: str) -> int:
    """How many objects the table has -- what a sample is a fraction OF.

    :param table: goes into the query as a quoted name, so it must be a table
        that exists -- a typo raises :class:`sqlite3.OperationalError` rather
        than counting zero, and a caller sizing a sample should check with
        :func:`table_names` first.
    """
    with _connect(db_path) as db:
        return int(db.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0])
