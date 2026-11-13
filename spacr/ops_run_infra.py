"""Run infrastructure of the OPS phenotype path at plate scale (item 872, alpha).

The large Funk et al. 2022 phenotype run (plates LaC024A-D, about 1,280
fields of 2,960 px per well) found engineering limits that the phase code in
:mod:`spacr.ops_engine` alone does not address. This module holds the parts
that are not phases themselves:

* :class:`FieldStager` reads raw phenotype fields from network storage ahead
  of the run with a few threads, keeps only each field's maximum projection
  on local disk, and deletes it once the field is done, so each raw file is
  read once and local space stays bounded (B2);
* :func:`compare_tables` is the GPU-against-CPU self-check of one measured
  field (B7);
* :func:`archive_well`, :func:`archived_well` and :func:`mark_plate_done`
  copy a finished well to archive storage, verify the copy by size and MD5
  and write the ``WELL_DONE`` / ``PLATE_DONE`` markers a resumed run starts
  from (B9);
* :func:`crop_index` and :func:`stream_crops` read the sharded crop store
  whole shard by whole shard, shuffled within and between shards, as a
  training loader does (WebDataset style; Aizman et al. 2019,
  arXiv:2001.01858) (B16);
* :func:`integer_cell_key` is the integer join key of the measurement
  tables (B17);
* :func:`stress_rows` measures a finished run's store the way the reference
  stress test did and :func:`write_stress_report` writes
  ``results/ops_stress/stress_test.csv`` (B20).

Every function is deterministic for the same inputs except the timings the
stress report records, which depend on the machine.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import random
import shutil
import sqlite3
import tarfile
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np

#: The archive verification modes ``ops_archive_verify`` may name.
ARCHIVE_VERIFY_MODES = ("md5", "size")

#: The marker a verified well copy leaves in its archive folder.
WELL_DONE = "WELL_DONE"

#: The marker a plate's archive folder gets once every well run is copied.
PLATE_DONE = "PLATE_DONE"

#: The columns of the reference stress test (Funk plate A run,
#: ``stress_test.csv``), in its order. Columns the spaCR store has no
#: counterpart for (the one-PNG-per-cell layout) stay empty.
STRESS_COLUMNS = (
    "level", "plate", "well", "fields", "cells", "linked", "wall_s",
    "seg_s_per_field_median", "measure_s_per_field_median", "png_crops",
    "png_s_per_field_median", "png_write_s_per_field_median",
    "npz_sample_crops", "nas_well_bytes", "files_in_well_dir", "png_bytes",
    "nas_size_walk_s_excluding_png", "measure_rows", "measure_write_s",
    "measure_rows_per_s", "png_mover_wait_s", "png_rows", "png_list_write_s",
    "png_list_rows_per_s", "index_build_s", "db_bytes", "query_one_field_s",
    "query_one_field_rows", "query_one_gene_s", "query_one_gene_rows",
    "query_one_gene", "query_join3_s", "query_join3_rows",
    "query_whole_well_s", "query_whole_well_rows", "png_random_read_n",
    "png_random_read_per_s", "png_random_read_MB_per_s", "png_dir_list_s",
    "png_dir_files", "png_dir_bytes_est", "peak_rss_gb_driver",
    "db_copy_to_nas_s", "tar_shards", "tar_crops", "tar_random_read_n",
    "tar_random_read_per_s", "tar_random_read_MB_per_s", "png_deferred",
    "nas_plate_bytes", "nas_plate_files", "nas_walk_s", "aggregate_s",
    "aggregate_peak_rss_gb", "cells_typed", "nas_du_s", "nas_find_s",
    "cells_segmented")


class FieldStager:
    """Stage raw phenotype fields ahead of the run, a bounded number at a time.

    :param folder: the local staging folder; created when missing.
    :param sites: the sites to stage, in the order the run uses them.
    :param read: ``read(site) -> (stack or None, unreadable)``; called once
        per site in a staging thread. ``stack`` is what is kept (the field's
        maximum projection), ``unreadable`` a list of ``(path, reason)``.
    :param ahead: the most fields staged and not yet released at once.
    :param threads: how many fields are read at once.

    A field is read only after a slot is free, and its slot is freed only by
    :meth:`release`, which deletes the local copy, so at most ``ahead``
    copies are ever on disk. :attr:`peak_files` and :attr:`peak_bytes`
    record the most there were.
    """

    def __init__(self, folder: str, sites: Sequence[int],
                 read: Callable[[int], Tuple[Optional[np.ndarray], list]], *,
                 ahead: int = 48, threads: int = 4) -> None:
        """Start staging ``sites`` into ``folder``."""
        self.folder = str(folder)
        os.makedirs(self.folder, exist_ok=True)
        self.ahead = max(1, int(ahead))
        self.peak_files = 0
        self.peak_bytes = 0
        self._read = read
        self._slots = threading.Semaphore(self.ahead)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._sites = [int(s) for s in sites]
        self._ready = {s: threading.Event() for s in self._sites}
        self._futures: Dict[int, Future] = {}
        self._held: Dict[int, int] = {}
        self._released: set = set()
        self._pool = ThreadPoolExecutor(max(1, int(threads)))
        self._feeder = threading.Thread(target=self._feed, daemon=True)
        self._feeder.start()

    def _path(self, site: int) -> str:
        """Where one staged field is kept."""
        return os.path.join(self.folder, f"site-{int(site)}.npy")

    def _feed(self) -> None:
        """Submit each site once a slot is free, in order."""
        for site in self._sites:
            while not self._slots.acquire(timeout=0.05):
                if self._stop.is_set():
                    return
            if self._stop.is_set():
                self._slots.release()
                return
            self._futures[site] = self._pool.submit(self._stage, site)
            self._ready[site].set()

    def _stage(self, site: int) -> Tuple[Optional[str], list]:
        """Read one field and keep its stack locally."""
        try:
            stack, unreadable = self._read(site)
        except BaseException:
            self._free(site, None)
            raise
        if stack is None:
            self._free(site, None)
            return None, list(unreadable)
        path = self._path(site)
        part = os.path.join(self.folder, f"site-{int(site)}.part.npy")
        np.save(part, np.ascontiguousarray(stack))
        os.replace(part, path)
        size = os.path.getsize(path)
        with self._lock:
            self._held[site] = size
            self.peak_files = max(self.peak_files, len(self._held))
            self.peak_bytes = max(self.peak_bytes, sum(self._held.values()))
        return path, list(unreadable)

    def _free(self, site: int, path: Optional[str]) -> None:
        """Delete one staged copy and free its slot, once."""
        with self._lock:
            if site in self._released:
                return
            self._released.add(site)
            self._held.pop(site, None)
        if path and os.path.isfile(path):
            os.remove(path)
        self._slots.release()

    def path(self, site: int) -> Tuple[Optional[str], list]:
        """Wait for one field to be staged.

        :param site: the site.
        :returns: ``(local path or None, unreadable)``; None when the field
            could not be read, in which case its slot is already free.
        """
        self._ready[int(site)].wait()
        return self._futures[int(site)].result()

    def load(self, site: int) -> Tuple[Optional[np.ndarray], list]:
        """One staged field's stack.

        :param site: the site.
        :returns: ``(stack or None, unreadable)``.
        """
        path, unreadable = self.path(site)
        return (np.load(path) if path else None), unreadable

    def release(self, site: int) -> None:
        """Delete one field's local copy; the next field may be staged.

        :param site: the site; releasing it twice does nothing.
        """
        self._free(int(site), self._path(site))

    def close(self) -> None:
        """Stop staging and delete every copy not yet released."""
        self._stop.set()
        self._feeder.join()
        self._pool.shutdown(wait=True)
        for site in list(self._futures):
            self.release(site)


def compare_tables(test, reference, *, rtol: float = 1e-4,
                   keys: Sequence[str] = ("site", "label")) -> Dict[str, Any]:
    """Which numeric columns of two measurement tables of one field differ.

    :param test: the table under test (the GPU measurement).
    :param reference: the reference table (the CPU measurement).
    :param rtol: the relative tolerance; values agree when
        ``abs(a - b) <= rtol * abs(b)`` or both are NaN.
    :param keys: the columns the rows are matched on.
    :returns: ``rows``, ``columns`` (how many numeric columns were
        compared), ``outside`` (one entry per column outside ``rtol``: its
        name, the largest relative difference and how many rows differ),
        ``only_in_test`` and ``only_in_reference``.
    """
    import pandas as pd

    keys = [k for k in keys if k in test.columns and k in reference.columns]
    if keys:
        merged = pd.merge(test, reference, on=keys, how="inner",
                          suffixes=("__test", "__reference"))
    else:
        merged = pd.concat([test.add_suffix("__test"),
                            reference.add_suffix("__reference")], axis=1)
    shared = [c for c in test.columns if c in reference.columns and c not in keys
              and pd.api.types.is_numeric_dtype(test[c])
              and pd.api.types.is_numeric_dtype(reference[c])]
    outside = []
    for column in shared:
        a = merged[f"{column}__test"].to_numpy(float)
        b = merged[f"{column}__reference"].to_numpy(float)
        both_nan = np.isnan(a) & np.isnan(b)
        with np.errstate(divide="ignore", invalid="ignore"):
            relative = np.where(both_nan, 0.0, np.abs(a - b) / np.abs(b))
        relative = np.where((a == b) | both_nan, 0.0, relative)
        relative = np.where(np.isnan(relative), np.inf, relative)
        bad = relative > rtol
        if bad.any():
            outside.append({"column": column,
                            "max_relative": float(np.max(relative)),
                            "rows_outside": int(bad.sum())})
    return {"rows": int(len(merged)), "columns": len(shared), "rtol": rtol,
            "outside": outside,
            "only_in_test": sorted(set(test.columns) - set(reference.columns)),
            "only_in_reference": sorted(set(reference.columns)
                                        - set(test.columns))}


def file_md5(path: str, chunk: int = 1 << 22) -> str:
    """The MD5 of one file, read in chunks.

    :param path: the file.
    :param chunk: bytes read at a time.
    :returns: the hex digest.
    """
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def well_outputs(destination: str, well: str) -> List[str]:
    """The files one well's run wrote, relative to the run folder.

    :param destination: the run folder (``dst_root``).
    :param well: the well name.
    :returns: every file under ``<destination>/<well>/`` and every file in
        ``destination`` itself named ``*_<well>.<ext>``, sorted; partial
        files (``.part``) are left out.
    """
    found: List[str] = []
    folder = os.path.join(destination, str(well))
    if os.path.isdir(folder):
        for here, _folders, files in os.walk(folder):
            for name in files:
                if ".part" in name:
                    continue
                found.append(os.path.relpath(os.path.join(here, name),
                                             destination))
    suffix = f"_{well}."
    if os.path.isdir(destination):
        for name in os.listdir(destination):
            path = os.path.join(destination, name)
            if (os.path.isfile(path) and suffix in name and ".part" not in name
                    and name.rsplit(".", 1)[0].endswith(f"_{well}")):
                found.append(name)
    return sorted(found)


def _copy_verified(source: str, target: str, verify: str,
                   expected: Mapping[str, Any]) -> bool:
    """Copy one file and check the copy against ``expected``."""
    os.makedirs(os.path.dirname(target) or ".", exist_ok=True)
    part = target + ".part"
    shutil.copyfile(source, part)
    os.replace(part, target)
    return _matches(target, verify, expected)


def _matches(path: str, verify: str, expected: Mapping[str, Any]) -> bool:
    """Whether one archived file has the size, and MD5, recorded for it."""
    if not os.path.isfile(path) or os.path.getsize(path) != expected["size"]:
        return False
    return verify != "md5" or file_md5(path) == expected["md5"]


def _marker_path(archive_root: str, plate: str, well: str) -> str:
    """Where one well's ``WELL_DONE`` marker is."""
    return os.path.join(archive_root, str(plate), str(well), WELL_DONE)


def archive_well(destination: str, archive_root: str, plate: str, well: str, *,
                 phases: Sequence[str], verify: str = "md5",
                 delete_local: bool = False, report: Optional[Mapping] = None,
                 copy: Optional[Callable[[str, str, str, Mapping], bool]] = None
                 ) -> Dict[str, Any]:
    """Copy one finished well to archive storage, verify it, mark it done.

    :param destination: the run folder.
    :param archive_root: the archive folder; the well goes to
        ``<archive_root>/<plate>/`` at the same relative paths.
    :param plate: the plate name.
    :param well: the well name.
    :param phases: the per-well phases the run did; recorded in the marker.
    :param verify: ``"md5"`` compares size and MD5 of every copy with the
        local file, ``"size"`` the size only.
    :param delete_local: delete the local files once every copy is verified
        and the marker is written.
    :param report: the well's run report, kept in the marker.
    :param copy: the copy-and-verify function, for tests; the default copies
        through a ``.part`` file.
    :returns: ``files``, ``bytes``, ``recopied`` (files whose first copy
        failed its check), ``marker`` and ``deleted_local``.
    :raises OSError: when a file's copy fails its check twice. Nothing local
        is deleted then, and no marker is written.
    """
    from .qt.i18n import tr

    if verify not in ARCHIVE_VERIFY_MODES:
        raise ValueError(tr("ops_archive_verify must be md5 or size; got "
                            "{value}", value=verify))
    copy = copy or _copy_verified
    target_root = os.path.join(archive_root, str(plate))
    manifest = []
    recopied = []
    for relative in well_outputs(destination, well):
        source = os.path.join(destination, relative)
        expected = {"path": relative, "size": os.path.getsize(source),
                    "md5": file_md5(source) if verify == "md5" else None}
        target = os.path.join(target_root, relative)
        if not copy(source, target, verify, expected):
            recopied.append(relative)
            if not copy(source, target, verify, expected):
                raise OSError(tr(
                    "the archive copy of {path} at {target} does not match "
                    "the local file after two copies", path=source,
                    target=target))
        manifest.append(expected)
    marker = _marker_path(archive_root, plate, well)
    os.makedirs(os.path.dirname(marker), exist_ok=True)
    with open(marker + ".part", "w", encoding="utf-8") as handle:
        json.dump({"plate": str(plate), "well": str(well),
                   "phases": sorted(set(phases)), "verify": verify,
                   "files": manifest, "report": dict(report or {}),
                   "written": time.strftime("%Y-%m-%dT%H:%M:%S")},
                  handle, indent=1, default=str)
    os.replace(marker + ".part", marker)
    deleted = 0
    if delete_local:
        for entry in manifest:
            os.remove(os.path.join(destination, entry["path"]))
            deleted += 1
        folder = os.path.join(destination, str(well))
        for here, _folders, _files in sorted(os.walk(folder), reverse=True):
            if not os.listdir(here):
                os.rmdir(here)
    return {"files": len(manifest),
            "bytes": int(sum(e["size"] for e in manifest)),
            "recopied": recopied, "marker": marker, "deleted_local": deleted}


def archived_well(archive_root: str, plate: str, well: str, *,
                  phases: Sequence[str]) -> Optional[Dict[str, Any]]:
    """Whether one well's archive copy is complete for ``phases``.

    :param archive_root: the archive folder.
    :param plate: the plate name.
    :param well: the well name.
    :param phases: the per-well phases this run asks for.
    :returns: None when there is no marker or it records fewer phases;
        else ``{"ok", "bad", "report", "marker"}``: ``ok`` is True when
        every recorded file is in the archive with its recorded size (and
        MD5, when the marker was written with ``"md5"``), ``bad`` the files
        that are not.
    """
    marker = _marker_path(archive_root, plate, well)
    if not os.path.isfile(marker):
        return None
    with open(marker, encoding="utf-8") as handle:
        record = json.load(handle)
    if not set(phases) <= set(record.get("phases") or ()):
        return None
    verify = record.get("verify") or "md5"
    target_root = os.path.join(archive_root, str(plate))
    bad = [entry["path"] for entry in record.get("files") or ()
           if not _matches(os.path.join(target_root, entry["path"]), verify,
                           entry)]
    return {"ok": not bad, "bad": bad, "report": record.get("report") or {},
            "marker": marker}


def mark_plate_done(archive_root: str, plate: str, wells: Sequence[str]
                    ) -> Optional[str]:
    """Write ``PLATE_DONE`` when every well of ``wells`` has its marker.

    :param archive_root: the archive folder.
    :param plate: the plate name.
    :param wells: the wells of the run.
    :returns: the marker's path, or None when a well is not archived yet.
    """
    if not all(os.path.isfile(_marker_path(archive_root, plate, w))
               for w in wells):
        return None
    path = os.path.join(archive_root, str(plate), PLATE_DONE)
    with open(path + ".part", "w", encoding="utf-8") as handle:
        json.dump({"plate": str(plate), "wells": sorted(map(str, wells)),
                   "written": time.strftime("%Y-%m-%dT%H:%M:%S")}, handle,
                  indent=1)
    os.replace(path + ".part", path)
    return path


def crop_index(root: str, plate: str, wells: Optional[Sequence[str]] = None):
    """Every crop of the sharded store, as one table.

    :param root: the crop store's root (``ops_merged_root`` or ``dst_root``).
    :param plate: the plate name.
    :param wells: the wells; every well in the store when None.
    :returns: the shard indexes concatenated, with a ``shard`` column (the
        tar's path), ordered by shard and position in it.
    """
    import pandas as pd

    base = os.path.join(root, "data", "cell", str(plate))
    frames = []
    if os.path.isdir(base):
        for here, _folders, files in sorted(os.walk(base)):
            well = os.path.relpath(here, base).split(os.sep)[0]
            if wells is not None and well not in set(map(str, wells)):
                continue
            for name in sorted(files):
                if name.endswith(".index.parquet"):
                    frame = pd.read_parquet(os.path.join(here, name))
                    frame.insert(0, "shard", os.path.join(
                        here, name[:-len(".index.parquet")] + ".tar"))
                    frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=["shard", "member", "offset_data", "size"])
    return pd.concat(frames, ignore_index=True)


def _decode_crop(data: bytes):
    """One crop member's bytes as ``(image, mask or None)``."""
    import tifffile

    with tifffile.TiffFile(io.BytesIO(data)) as tiff:
        image = tiff.series[0].asarray()
        mask = tiff.series[1].asarray() if len(tiff.series) > 1 else None
    return image, mask


def stream_crops(root: str, plate: str, wells: Optional[Sequence[str]] = None,
                 *, shuffle: bool = True, seed: int = 0, buffer: int = 1000
                 ) -> Iterator[Tuple[Dict[str, Any], np.ndarray,
                                     Optional[np.ndarray]]]:
    """Every crop of the store, read shard by shard, as a training loader would.

    :param root: the crop store's root.
    :param plate: the plate name.
    :param wells: the wells; every well when None.
    :param shuffle: shuffle the shard order and, through a buffer, the crops
        within and across shards.
    :param seed: the shuffle seed; the same seed gives the same order.
    :param buffer: how many decoded crops the shuffle buffer holds.
    :yields: ``(record, image, mask)``: the crop's index row as a dict, its
        ``(C, H, W)`` image and its mask (None when the store has none).

    Each shard is read once, front to back (``tarfile`` stream mode), which
    is what makes network storage fast: the reference plate's 2 GB shards
    streamed at 842 crops/s against 17 crops/s for a seek per crop.
    """
    index = crop_index(root, plate, wells)
    records = {(row["shard"], row["member"]): row
               for row in index.to_dict("records")}
    shards = list(dict.fromkeys(index["shard"]))
    rng = random.Random(seed)
    if shuffle:
        rng.shuffle(shards)
    pool: List[Tuple[Dict[str, Any], np.ndarray, Optional[np.ndarray]]] = []
    for shard in shards:
        with tarfile.open(shard, "r|") as tar:
            for member in tar:
                if not member.isfile():
                    continue
                data = tar.extractfile(member).read()
                image, mask = _decode_crop(data)
                item = (records.get((shard, member.name),
                                    {"shard": shard, "member": member.name}),
                        image, mask)
                if not shuffle:
                    yield item
                    continue
                pool.append(item)
                if len(pool) >= max(1, int(buffer)):
                    yield pool.pop(rng.randrange(len(pool)))
    while pool:
        yield pool.pop(rng.randrange(len(pool)))


def integer_cell_key(row, column, site, label) -> np.ndarray:
    """An integer key per cell that joins as ``prcfo`` does.

    :param row: the plate row numbers (1 for row A).
    :param column: the plate column numbers.
    :param site: the phenotype field numbers, below 2**24.
    :param label: the cell labels in their field, below 2**20.
    :returns: ``((row * 100 + column) * 2**24 + site) * 2**20 + label`` as
        int64, one per cell: distinct for distinct cells of one plate.
    :raises ValueError: when a site or label is out of range.
    """
    from .qt.i18n import tr

    row = np.asarray(row, dtype=np.int64)
    column = np.asarray(column, dtype=np.int64)
    site = np.asarray(site, dtype=np.int64)
    label = np.asarray(label, dtype=np.int64)
    if ((site < 0) | (site >= 2 ** 24) | (label < 0) | (label >= 2 ** 20)
            | (row < 0) | (row >= 64) | (column < 0) | (column >= 100)).any():
        raise ValueError(tr(
            "an integer cell key needs rows below 64, columns below 100, "
            "sites below 2**24 and labels below 2**20"))
    return ((row * 100 + column) * 2 ** 24 + site) * 2 ** 20 + label


def _timed(conn, sql: str, parameters: Sequence[Any] = ()) -> Tuple[float, int]:
    """Seconds and rows of one query, every row fetched."""
    tick = time.perf_counter()
    rows = conn.execute(sql, tuple(parameters)).fetchall()
    return round(time.perf_counter() - tick, 3), len(rows)


def _folder_size(paths: Sequence[str]) -> Tuple[int, int]:
    """Bytes and files of ``paths``."""
    total = files = 0
    for path in paths:
        if os.path.isfile(path):
            total += os.path.getsize(path)
            files += 1
    return total, files


def stress_rows(destination: str, plate: str, wells: Sequence[str],
                run: Mapping[str, Any], *, crop_root: Optional[str] = None,
                random_reads: int = 1000, seed: int = 0,
                control_prefix: str = "nontargeting") -> List[Dict[str, Any]]:
    """The stress-test rows of one finished run: one per well, one per plate.

    :param destination: the run folder.
    :param plate: the plate name.
    :param wells: the wells run.
    :param run: :func:`spacr.ops_engine.run_ops`'s result so far: the per-well
        reports, ``aggregate``, ``ph_tables`` and ``ph_crops``.
    :param crop_root: the crop store's root; ``destination`` when None.
    :param random_reads: crops read at random per well (prcfo -> shard
        offset -> decode), the training-style read.
    :param seed: the random read's seed.
    :param control_prefix: the control gene prefix, which the one-gene query
        skips.
    :returns: rows keyed by :data:`STRESS_COLUMNS` (missing columns empty).

    Counts (fields, cells, rows, bytes, files, shards, crops) are exact for
    the run folder; the ``*_s`` and ``*_per_s`` values are wall-clock
    timings of this machine.
    """
    import pandas as pd

    from .ops_engine import _peak_rss_gb, _ph_read_crop

    crop_root = crop_root or destination
    reports = run.get("wells") or {}
    tables = run.get("ph_tables") or {}
    database = tables.get("db") or os.path.join(destination, "measurements",
                                                "measurements.db")
    has_db = os.path.isfile(database)
    rows: List[Dict[str, Any]] = []
    index = crop_index(crop_root, plate)
    rng = np.random.default_rng(seed)
    for well in wells:
        report = reports.get(well) or {}
        fields = report.get("ph_fields") or {}
        segment = fields.get("ph_segment") or {}
        measure = fields.get("ph_measure") or {}
        files = [os.path.join(destination, p)
                 for p in well_outputs(destination, well)]
        tick = time.perf_counter()
        nbytes, nfiles = _folder_size(files)
        walk = round(time.perf_counter() - tick, 3)
        row: Dict[str, Any] = {
            "level": "well", "plate": plate, "well": well,
            "fields": fields.get("fields"), "cells": measure.get("cells"),
            "linked": (report.get("merge") or {}).get("merged"),
            "wall_s": report.get("seconds"),
            "seg_s_per_field_median": segment.get("segment_seconds_per_field"),
            "measure_s_per_field_median": measure.get(
                "measure_seconds_per_field"),
            "nas_well_bytes": nbytes, "files_in_well_dir": nfiles,
            "nas_size_walk_s_excluding_png": walk,
            "peak_rss_gb_driver": report.get("peak_rss_gb"),
            "cells_segmented": (segment.get("objects") or {}).get("paired")}
        if has_db:
            from . import schema

            row_id, column_id = schema.parse_well(well)
            conn = sqlite3.connect(database, timeout=30)
            try:
                count = conn.execute(
                    "SELECT COUNT(*) FROM cell WHERE rowID = ? AND columnID = ?",
                    (row_id, column_id)).fetchone()[0]
                measured = count
                for table in ("nucleus", "cytoplasm"):
                    measured += conn.execute(
                        f"SELECT COUNT(*) FROM {table} WHERE rowID = ? AND "
                        "columnID = ?", (row_id, column_id)).fetchone()[0]
                row["measure_rows"] = int(measured)
                row["db_bytes"] = os.path.getsize(database)
                row["index_build_s"] = tables.get("index_seconds")
                first = conn.execute(
                    "SELECT fieldID FROM cell WHERE rowID = ? AND columnID = ? "
                    "ORDER BY fieldID LIMIT 1", (row_id, column_id)).fetchone()
                if first is not None:
                    row["query_one_field_s"], row["query_one_field_rows"] = _timed(
                        conn, "SELECT * FROM cell WHERE plateID = ? AND rowID = ? "
                        "AND columnID = ? AND fieldID = ?",
                        (plate, row_id, column_id, first[0]))
                gene = conn.execute(
                    "SELECT gene, COUNT(*) AS n FROM cell WHERE rowID = ? AND "
                    "columnID = ? AND gene IS NOT NULL AND lower(gene) NOT LIKE ? "
                    "GROUP BY gene ORDER BY n DESC, gene LIMIT 1",
                    (row_id, column_id, control_prefix.lower() + "%")).fetchone()
                if gene is not None:
                    row["query_one_gene"] = gene[0]
                    row["query_one_gene_s"], row["query_one_gene_rows"] = _timed(
                        conn, "SELECT * FROM cell WHERE gene = ? AND rowID = ? "
                        "AND columnID = ?", (gene[0], row_id, column_id))
                row["query_join3_s"], row["query_join3_rows"] = _timed(
                    conn, "SELECT * FROM cell JOIN nucleus USING (prcfo) JOIN "
                    "cytoplasm USING (prcfo) WHERE cell.rowID = ? AND "
                    "cell.columnID = ?", (row_id, column_id))
                row["query_whole_well_s"], row["query_whole_well_rows"] = _timed(
                    conn, "SELECT * FROM cell WHERE rowID = ? AND columnID = ?",
                    (row_id, column_id))
            finally:
                conn.close()
        crops = index[index["shard"].str.contains(
            os.sep + str(well) + os.sep, regex=False)] if len(index) else index
        if len(crops):
            row["tar_shards"] = int(crops["shard"].nunique())
            row["tar_crops"] = int(len(crops))
            take = rng.choice(len(crops), size=min(int(random_reads),
                                                   len(crops)), replace=False)
            tick = time.perf_counter()
            read_bytes = 0
            for position in take:
                record = crops.iloc[int(position)]
                _ph_read_crop(record["shard"], int(record["offset_data"]),
                              int(record["size"]))
                read_bytes += int(record["size"])
            seconds = max(time.perf_counter() - tick, 1e-9)
            row["tar_random_read_n"] = int(len(take))
            row["tar_random_read_per_s"] = round(len(take) / seconds, 1)
            row["tar_random_read_MB_per_s"] = round(read_bytes / seconds / 1e6, 2)
        rows.append(row)
    aggregate = run.get("aggregate") or {}
    tick = time.perf_counter()
    plate_files = []
    for well in wells:
        plate_files.extend(os.path.join(destination, p)
                           for p in well_outputs(destination, well))
    if len(index):
        plate_files.extend(dict.fromkeys(index["shard"]))
    nbytes, nfiles = _folder_size(plate_files)
    rows.append({"level": "plate", "plate": plate, "well": None,
                 "nas_plate_bytes": nbytes, "nas_plate_files": nfiles,
                 "nas_walk_s": round(time.perf_counter() - tick, 3),
                 "aggregate_s": aggregate.get("seconds"),
                 "aggregate_peak_rss_gb": aggregate.get("peak_rss_gb"),
                 "cells_typed": aggregate.get("cells_with_guide"),
                 "peak_rss_gb_driver": _peak_rss_gb(),
                 "cells_segmented": int(sum(r.get("cells_segmented") or 0
                                            for r in rows))})
    return [{column: row.get(column) for column in STRESS_COLUMNS}
            for row in rows]


def write_stress_report(destination: str, rows: Sequence[Mapping[str, Any]]
                        ) -> str:
    """Write the stress-test rows to ``results/ops_stress/stress_test.csv``.

    :param destination: the run folder.
    :param rows: :func:`stress_rows`.
    :returns: the CSV's path; ``stress_rows.jsonl`` is written beside it.
    """
    import pandas as pd

    from .tabular import write_table

    folder = os.path.join(destination, "results", "ops_stress")
    os.makedirs(folder, exist_ok=True)
    target = os.path.join(folder, "stress_test.csv")
    write_table(pd.DataFrame(list(rows), columns=list(STRESS_COLUMNS)), target,
                canonicalise=False)
    with open(os.path.join(folder, "stress_rows.jsonl"), "w",
              encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(
                {k: v for k, v in row.items() if v is not None},
                default=str) + "\n")
    return target
