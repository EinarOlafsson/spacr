"""Build an annotation set by streaming crops, and register it for annotating.

WHAT THIS IS FOR. Annotating means looking at single-object crops, and until
now the only way to get a set of them was to run Measure over a whole plate --
about twenty minutes for 52 fields on a 30-core machine. Deciding afterwards
that the crops should have been cut differently meant running it again.

Every field's objects are already described twice over once Measure has run:
by the label masks inside the merged arrays, and by the coordinate columns in
``measurements.db``. Either is enough to cut crops from, so a second set can be
built in seconds without measuring anything again.

THE TWO ROUTES ARE NOT EQUIVALENT, and the difference is not guessable:

* ``array`` reads the object masks out of the merged stacks, so it can cut to
  the object itself -- masked, or to its bounding box;
* ``database`` reads the coordinate columns, which are all the database
  stores, so it can only ever produce a BOUNDING BOX.

Comparing the two is therefore only meaningful with ``bounding_box=True``.
:func:`spacr.annotation_dataset` says so in the settings it accepts, and the
GUI says it beside the picker.

Instruction 338.
"""
from __future__ import annotations

import logging
import os
import re
import sqlite3
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

from .database_concurrency import connect as connect_database

LOG = logging.getLogger("spacr.annotation_dataset")

__all__ = [
    "PNG_TABLE_BASE",
    "STREAM_SOURCES",
    "next_png_table",
    "filter_selection",
    "png_list_frame",
    "write_png_list",
    "generate_annotation_dataset",
]

#: The table the annotation viewer reads. See `spacr.agreement.PNG_TABLE`.
PNG_TABLE_BASE = "png_list"

#: Where the objects can be read from, as ``(value, label)``.
#:
#: The names match `stream_dataset.STREAM_METHODS` in meaning but not in
#: spelling -- that module says "column" for what a user calls the database.
STREAM_SOURCES: Tuple[Tuple[str, str], ...] = (
    ("array", "the object masks in the merged arrays"),
    ("database", "the coordinate columns in measurements.db"),
)

#: ``png_list`` as `measure_crop` writes it. Matching it exactly is what lets
#: the annotation viewer open a streamed set without knowing how it was made.
PNG_LIST_COLUMNS: Tuple[str, ...] = (
    "png_path", "file_name", "plateID", "rowID", "columnID", "fieldID",
    "prcfo", "cell_id", "annotate",
)

_TABLE_SUFFIX = re.compile(r"^png_list(?:_(\d+))?$")


def next_png_table(connection: sqlite3.Connection) -> str:
    """The name a new annotation set should be written under.

    ``png_list`` when it is free, then ``png_list_2``, ``png_list_3`` and so
    on. An existing set is never overwritten: it may already carry
    annotations, and those are hand-made and unrecoverable.

    THE CALLER MUST HOLD A WRITE TRANSACTION. Choosing a name and creating the
    table are two steps, and two runs started together would otherwise choose
    the same one. :func:`write_png_list` does this correctly; call it rather
    than this.

    :param connection: an open connection to the measurements database.
    :returns: the free table name.
    """
    taken = set()
    for (name,) in connection.execute(
            "select name from sqlite_master where type in ('table', 'view')"):
        match = _TABLE_SUFFIX.match(str(name))
        if match:
            taken.add(int(match.group(1) or 1))
    number = 1
    while number in taken:
        number += 1
    return PNG_TABLE_BASE if number == 1 else f"{PNG_TABLE_BASE}_{number}"


def filter_selection(selection: pd.DataFrame,
                     settings: Mapping[str, Any]) -> pd.DataFrame:
    """Drop objects a run would not have cropped.

    THE SAME PREDICATES `measure_crop` APPLIES, so a streamed set and a
    measured one describe the same population. They are expressed against the
    selection table's columns rather than against a measurement frame, because
    the array route has no measurements -- only labels and geometry.

    Recognised settings, each optional and each skipped when absent:

    ``{object}_min_size`` / ``{object}_max_size``
        Object area in pixels. ``0`` and ``None`` both mean "no bound", which
        is what the Measure panel writes for an unset field -- treating 0 as a
        real minimum would drop nothing and look like it had worked.
    ``wells`` / ``exclude_wells``
        Well ids to keep or drop, as ``rowID``+``columnID`` pairs or as plain
        well names.
    ``max_objects``
        A cap, applied LAST and deterministically (by the sort the selection
        already has), so a capped set is reproducible.

    :param selection: a selection table from :mod:`spacr.stream_dataset`.
    :param settings: the run settings.
    :returns: a new frame; the input is not modified.
    """
    if selection is None or not len(selection):
        return selection
    frame = selection.copy()
    object_type = str(settings.get("object_array") or "cell")

    area = None
    for candidate in ("area", f"{object_type}_area", "object_area"):
        if candidate in frame.columns:
            area = candidate
            break
    if area is not None:
        minimum = settings.get(f"{object_type}_min_size")
        maximum = settings.get(f"{object_type}_max_size")
        # 0 IS NOT A MINIMUM. The Measure panel writes 0 for an unset bound,
        # and honouring it as one would filter nothing while looking filtered.
        if minimum:
            frame = frame[frame[area] >= float(minimum)]
        if maximum:
            frame = frame[frame[area] <= float(maximum)]

    keep = settings.get("wells")
    drop = settings.get("exclude_wells")
    if keep or drop:
        names = _well_names(frame)
        if keep:
            frame = frame[names.isin({str(w) for w in keep})]
            names = _well_names(frame)
        if drop:
            frame = frame[~names.isin({str(w) for w in drop})]

    cap = settings.get("max_objects")
    if cap:
        # Head of the existing order, not a sample: a set that differs between
        # two runs of the same settings cannot be compared with anything.
        frame = frame.head(int(cap))
    return frame.reset_index(drop=True)


def _well_names(frame: pd.DataFrame) -> pd.Series:
    """Well ids as one comparable string per row."""
    if "well" in frame.columns:
        return frame["well"].astype(str)
    row = frame["rowID"].astype(str) if "rowID" in frame.columns else ""
    column = frame["columnID"].astype(str) if "columnID" in frame.columns else ""
    return (row + column) if len(frame) else pd.Series([], dtype=str)


def png_list_frame(selection: pd.DataFrame, paths: Sequence[str]) -> pd.DataFrame:
    """The rows the annotation viewer reads, in `measure_crop`'s own schema.

    :param selection: the filtered selection table.
    :param paths: the written crop path for each of its rows, in order.
    :returns: a frame with exactly :data:`PNG_LIST_COLUMNS`.
    """
    if selection is None or not len(selection):
        return pd.DataFrame(columns=list(PNG_LIST_COLUMNS))
    frame = selection.reset_index(drop=True).copy()
    if len(paths) != len(frame):
        raise ValueError(
            f"{len(paths)} crop paths for {len(frame)} selected objects; the "
            "two must correspond row for row or the table would name the "
            "wrong picture for an object")
    out = pd.DataFrame({
        "png_path": [str(p) for p in paths],
        "file_name": [os.path.basename(str(p)) for p in paths],
        "plateID": frame.get("plateID", ""),
        "rowID": frame.get("rowID", ""),
        "columnID": frame.get("columnID", ""),
        "fieldID": frame.get("fieldID", ""),
        "cell_id": ["o" + str(v) for v in frame.get("objectID", [])],
    })
    # `prcfo` is the join key every other measurement table carries, so a
    # streamed set can be joined to the measurements the same way a measured
    # one can.
    out["fieldID"] = out["fieldID"].astype(str)
    out["prcfo"] = (out["plateID"].astype(str) + "_"
                    + out["rowID"].astype(str) + "_"
                    + out["columnID"].astype(str) + "_f"
                    + out["fieldID"].str.lstrip("f") + "_"
                    + out["cell_id"].astype(str))
    out["annotate"] = None
    return out[list(PNG_LIST_COLUMNS)]


def reserve_png_table(db_path: str) -> str:
    """Claim the next free table name by creating it, empty.

    RESERVED BEFORE THE CROPS ARE CUT, not after, so the folder they are
    written into can be named to match: `png_list_2` gets `data_2`. Deriving
    the folder from the table is what makes a set on disk traceable to the set
    in the database -- with two independent counters they drift the first time
    either is deleted, and then nothing says which folder a table describes.

    Creating the table is what reserves it: two runs started together would
    otherwise choose the same name, and the second would fail on insert after
    it had already written a folder full of crops.

    :param db_path: the measurements database.
    :returns: the reserved table name.
    """
    connection = connect_database(str(db_path))
    try:
        connection.execute("BEGIN IMMEDIATE")
        name = next_png_table(connection)
        columns = ", ".join(f'"{c}"' for c in PNG_LIST_COLUMNS)
        connection.execute(f'CREATE TABLE "{name}" ({columns})')
        connection.execute("COMMIT")
    except Exception:
        try:
            connection.execute("ROLLBACK")
        except sqlite3.Error:
            pass
        raise
    finally:
        connection.close()
    return name


def crops_folder_for(table: str) -> str:
    """The crop folder that belongs to ``table``.

    ``png_list`` -> ``data``; ``png_list_2`` -> ``data_2``. The suffix is
    carried across rather than counted again, so the pair cannot drift.
    """
    suffix = str(table)[len(PNG_TABLE_BASE):]
    return "data" + suffix


def write_png_list(db_path: str, frame: pd.DataFrame, *,
                   table: Optional[str] = None) -> str:
    """Write an annotation set into the measurements database.

    The name is chosen and the table created inside ONE transaction, so two
    runs started together cannot pick the same one. A caller that already
    reserved a name with :func:`reserve_png_table` passes it as ``table``.

    :param db_path: the measurements database.
    :param frame: rows as :func:`png_list_frame` builds them.
    :param table: a name already reserved.
    :returns: the table actually written.
    """
    connection = connect_database(str(db_path))
    try:
        connection.execute("BEGIN IMMEDIATE")
        name = table or next_png_table(connection)
        columns = ", ".join(f'"{c}"' for c in PNG_LIST_COLUMNS)
        if table is None:
            connection.execute(f'CREATE TABLE "{name}" ({columns})')
        placeholders = ", ".join("?" for _ in PNG_LIST_COLUMNS)
        connection.executemany(
            f'INSERT INTO "{name}" ({columns}) VALUES ({placeholders})',
            [tuple(None if pd.isna(v) else v for v in row)
             for row in frame[list(PNG_LIST_COLUMNS)].itertuples(index=False)])
        connection.execute("COMMIT")
    except Exception:
        try:
            connection.execute("ROLLBACK")
        except sqlite3.Error:
            pass
        raise
    finally:
        connection.close()
    LOG.info("wrote %d rows to %s in %s", len(frame), name, db_path)
    return name


def generate_annotation_dataset(settings: Mapping[str, Any]) -> Dict[str, Any]:
    """Stream a set of crops and register it for annotation.

    :param settings: needs ``src`` (the plate folder) and accepts
        ``stream_source`` (:data:`STREAM_SOURCES`), ``object_array``,
        ``channel_arrays``, ``bounding_box``, the filtration keys
        :func:`filter_selection` reads, and ``dst`` for where the crops go.
    :returns: the streaming report, plus ``table`` naming what was written.
    """
    from .stream_dataset import (build_selection, selection_from_objects,
                                 stream)

    source = str(settings.get("stream_source") or "array").lower()
    src = str(settings.get("src") or "")
    merged = str(settings.get("merged_folder") or os.path.join(src, "merged"))
    database = str(settings.get("database")
                   or os.path.join(src, "measurements", "measurements.db"))
    object_type = str(settings.get("object_array") or "cell")

    # THE TABLE NAME IS CLAIMED FIRST, and the crop folder is named after it.
    # `png_list` gets `data`, `png_list_2` gets `data_2` -- so a folder on
    # disk says which table describes it. Two independent counters would
    # drift the first time either was deleted.
    table = str(settings.get("table") or "")
    if not table:
        try:
            table = reserve_png_table(database)
        except Exception:                                    # noqa: BLE001
            LOG.warning("could not reserve a png_list table in %s", database,
                        exc_info=True)
            return {"written": 0, "missing": 0, "fields": 0, "folders": [],
                    "table": "",
                    "trouble": [f"could not open {database} to reserve a "
                                f"table for this set"]}
    destination = str(settings.get("dst")
                      or os.path.join(src, crops_folder_for(table)))

    objects = None
    if source == "database":
        # THE COORDINATE COLUMNS. They are all the database stores, which is
        # why this route can only ever produce a bounding box.
        objects = read_objects_from_database(database, object_type)
        if objects is None or not len(objects):
            return {"written": 0, "missing": 0, "fields": 0, "folders": [],
                    "table": "",
                    "trouble": [f"no {object_type} rows in {database}"]}

    selection, selection_path = build_selection(
        destination, objects=objects, merged_folder=merged,
        object_array=object_type,
        test_split=float(settings.get("test_split") or 0.0),
        seed=int(settings.get("random_seed") or 0))
    selection = filter_selection(selection, settings)
    if not len(selection):
        return {"written": 0, "missing": 0, "fields": 0, "folders": [],
                "table": "", "selection": selection_path,
                "trouble": ["every object was filtered out; the reserved "
                            f"table {table} is empty"]}

    written: List[str] = []

    png_size = list(settings.get("png_size") or (224, 224))
    channels = list(settings.get("channel_arrays") or (0, 1, 2))

    def _write(path, array):
        # `measure_crop`'S OWN WRITER, not a second one.
        #
        # The annotation viewer shows pictures, so a set written as .npy is
        # not an annotation set -- but that is the smaller reason. The bigger
        # one is that instruction 338 asks for a streamed set and a measured
        # set to be the SAME IMAGES, and two writers cannot be relied on to
        # narrow to 8-bit, pad a two-channel crop, or resize identically. One
        # writer makes that a property of the code rather than a coincidence
        # to be re-tested after every change to either.
        from .measure import _save_object_crop

        target = os.path.splitext(str(path))[0] + ".png"
        written.append(_save_object_crop(array, channels, target, png_size))

    report = stream(
        selection, merged, destination,
        channel_arrays=list(settings.get("channel_arrays") or (0, 1, 2)),
        # FORCED for the database route, which has no mask to cut to.
        bounding_box=(True if source == "database"
                      else bool(settings.get("bounding_box", True))),
        crop_mode=object_type, write=_write)
    report["selection"] = selection_path

    if written:
        frame = png_list_frame(selection.head(len(written)), written)
        report["table"] = write_png_list(database, frame, table=table)
    else:
        # The reserved table stays, empty, rather than being dropped: it is
        # the record that this name is spoken for, and dropping it would let
        # a later run reuse a name whose folder is already on disk.
        report["table"] = ""
        report.setdefault("trouble", []).append(
            f"nothing was written; the reserved table {table} is empty")
    return report


def read_objects_from_database(db_path: str, object_type: str
                               ) -> Optional[pd.DataFrame]:
    """The object table a streamed set can be built from.

    :param db_path: the measurements database.
    :param object_type: ``cell``, ``nucleus``, ``pathogen`` …
    :returns: the rows, or ``None`` when the table is not there.
    """
    if not os.path.isfile(str(db_path)):
        return None
    connection = connect_database(str(db_path), readonly=True)
    try:
        names = {r[0] for r in connection.execute(
            "select name from sqlite_master where type='table'")}
        if object_type not in names:
            return None
        return pd.read_sql_query(f'select * from "{object_type}"', connection)
    finally:
        connection.close()
