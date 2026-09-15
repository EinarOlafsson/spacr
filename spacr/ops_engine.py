"""Run an optical pooled screen's sequencing acquisition from tiles to tables.

One call takes each well from its tile files to three tables in
``measurements.db``: ``ops_geometry``, where each nuclear tile sits in the
well frame; ``ops_objects``, one row per nucleus, segmented on the composed
nuclear map and numbered once for the well; and ``ops_barcodes``, one row per
nucleus whose attributed reads agree.

A file that cannot be read costs one cycle of the field it belongs to and is
listed in the report; it does not cost the well. The phenotype acquisition
is not placed by this step.
"""
from __future__ import annotations

import json
import logging
import os
import re
import resource
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

LOG = logging.getLogger("spacr.ops_engine")

__all__ = ["run_ops"]

#: A tile of this acquisition: ``10X_c<cycle>_<well>_<channels>_Site-<n>.tif``,
#: where ``<channels>`` is one channel (``CY3``) or a stack's planes joined by
#: hyphens (``DAPI-CY3-A594-CY5-CY7``). The channel token is what tells a
#: five-plane cycle-1 stack from a cycle's four single-channel files, so the
#: pattern has to capture it -- the stitch-only pattern in
#: :mod:`spacr.spacrops` does not.
_TILE_PATTERN = re.compile(
    r"(?P<mag>\d+X)_c(?P<cycle>\d+)_(?P<well>[A-Z]\d{1,2})_"
    r"(?P<channel>[A-Za-z0-9-]+?)_Site[-_](?P<site>\d+)\.tiff?$",
    re.IGNORECASE)

#: The nuclear stain's channel name, and the base channels in the order the
#: bases are read (372 PART 14-L, from the reference run's
#: ``20200202_6W-LaC024A_0_sbs.smk``: channels CY3, A594, CY5, CY7 and
#: ``BASES = 'GTAC'``).
_NUCLEAR = "DAPI"
_BASE_CHANNELS = ("CY3", "A594", "CY5", "CY7")
_BASES = ("G", "T", "A", "C")

#: The measured raster's overlap and the along-raster acceptance of an edge,
#: in pixels (PART 11-D; `tools/run_ops_a2.py`'s defaults).
_RASTER_OVERLAP = 213
_STITCH_TOLERANCE = 4

#: Segmentation windows and their overlap, as the validated run used them
#: (PART 14-L: 1,024 px windows, 96 px overlap, 1,280,002 objects).
_WINDOW = 1024
_WINDOW_OVERLAP = 96

#: The read detection threshold on the spot score's local contrast -- the
#: reference run's own ``THRESHOLD_READS`` for this plate.
_THRESHOLD_READS = 315.0

#: How far beyond a nucleus a read may lie and still be its read, and how
#: close two boundaries may be before a read is given to neither (PART 14-L:
#: nucleus + 10 px holds 98 % of spots, + 3 px holds 67 %).
_FOOTPRINT = 10.0
_TIE = 1.0

#: PART 9 hole 2's gate: a well whose detected spots land inside the
#: footprint less often than this is reported loudly.
_MIN_CONTAINMENT = 0.8

#: PART 9 hole 1: a field with fewer usable cycles than this is not decoded.
_MIN_CYCLES = 3

#: How many fields decode at once, whatever ``n_workers`` asks for. One field
#: holds eleven cycles of four 1,480 px channels twice over (the aligned
#: stack and its filtered copy), about 1.5 GB at its peak.
_DECODE_WORKERS_CAP = 8

#: The three steps, in the order they run.
_PHASES = ("stitch", "objects", "decode")

#: The guide library a decode worker compares calls against, set once per
#: process by :func:`_init_decode_worker` rather than shipped with every
#: field.
_LIBRARY: frozenset = frozenset()


def _say(message: str) -> None:
    """Print one progress line and log it.

    :param message: the line, without the prefix.
    """
    print(f"OPS: {message}", flush=True)
    LOG.info(message)


def _index_tiles(root: str) -> Dict[str, Dict[int, Dict[int, Dict[str, str]]]]:
    """Every tile under ``root``, as ``well -> cycle -> site -> channel -> path``.

    :param root: the acquisition folder, searched recursively.
    :returns: the index; empty when nothing matched.

    Names that are directories are skipped by the walk itself, which matters
    here: PART 14-D found ``*.tif`` names on this plate that are folders.
    """
    found: Dict[str, Dict[int, Dict[int, Dict[str, str]]]] = {}
    for folder, _folders, files in os.walk(root, followlinks=True):
        for name in files:
            match = _TILE_PATTERN.search(name)
            if not match:
                continue
            well = match["well"].upper()
            cycle, site = int(match["cycle"]), int(match["site"])
            found.setdefault(well, {}).setdefault(cycle, {}).setdefault(
                site, {})[match["channel"].upper()] = os.path.join(folder, name)
    return found


def _plane_sources(files: Mapping[str, str]) -> Dict[str, Tuple[str, Optional[int]]]:
    """``channel -> (path, plane)`` for one site of one cycle.

    :param files: ``channel token -> path`` from :func:`_index_tiles`.
    :returns: where each named channel is: a single-channel file (plane
        None) or one plane of a stack. A channel present both ways is taken
        from its own file.
    """
    out: Dict[str, Tuple[str, Optional[int]]] = {}
    for token, path in sorted(files.items(), key=lambda item: "-" in item[0]):
        names = token.split("-")
        if len(names) == 1:
            out.setdefault(names[0], (path, None))
        else:
            for index, name in enumerate(names):
                out.setdefault(name, (path, index))
    return out


def _read_plane(source: Optional[Tuple[str, Optional[int]]],
                unreadable: Optional[list] = None) -> Optional[np.ndarray]:
    """One 2-D plane, or None when the file cannot give it.

    :param source: ``(path, plane)`` from :func:`_plane_sources`, or None.
    :param unreadable: a list to append ``(path, reason)`` to on failure.
    :returns: the plane as float32, or None.

    THREE OUTCOMES, NOT TWO. A path that is not a file, a file that will not
    read, and a plane the file does not hold are all "no image here", and
    each is recorded with its reason. ``(OSError, ValueError)`` and not
    ``OSError`` alone: 372 PART 14-L found sixteen truncated sequencing files
    on this plate, and tifffile raises ValueError on a short read -- "failed
    to read 4380800 bytes, got 2578" -- which a handler for OSError lets
    through to end the run.
    """
    if source is None:
        return None
    path, plane = source
    if not os.path.isfile(path):
        if unreadable is not None:
            unreadable.append((path, "not a file"))
        return None
    try:
        import tifffile

        array = np.asarray(tifffile.imread(path))
    except (OSError, ValueError) as failure:
        if unreadable is not None:
            unreadable.append((path, f"{type(failure).__name__}: {failure}"[:200]))
        return None
    if plane is not None:
        if array.ndim < 3 or array.shape[0] <= plane:
            if unreadable is not None:
                unreadable.append((path, f"has no plane {plane}"))
            return None
        array = array[plane]
    while array.ndim > 2:
        array = array[0]
    return array.astype(np.float32)


def _largest_component(edges: Mapping[Tuple[int, int], object],
                       sites: Sequence[int]) -> List[int]:
    """The sites of the largest group joined by accepted edges.

    :param edges: ``(a, b) -> registration`` from the stitch.
    :param sites: every site that was offered.
    :returns: the sites of the largest connected component, ascending.

    :func:`spacr.ops_solve.solve_placements` pins every component at its own
    origin, so a tile no edge reached still comes back with a position --
    ``(0, 0)`` -- and composing it would paste it into the well's corner.
    Only the component the well was solved as is kept.
    """
    parent = {site: site for site in sites}

    def find(site: int) -> int:
        """The representative of ``site``'s component.

        :param site: a site.
        :returns: its root.
        """
        while parent[site] != site:
            parent[site] = parent[parent[site]]
            site = parent[site]
        return site

    for (a, b), found in edges.items():
        if getattr(found, "accepted", False) and a in parent and b in parent:
            parent[find(a)] = find(b)
    groups: Dict[int, List[int]] = {}
    for site in sites:
        groups.setdefault(find(site), []).append(site)
    return sorted(max(groups.values(), key=len)) if groups else []


def _replace_well_rows(db: str, table: str, frame, plate: str, well: str) -> int:
    """Write one well's rows into ``table``, replacing that well's old rows.

    :param db: the measurements database.
    :param table: an OPS table.
    :param frame: this well's rows, with ``plate`` and ``well`` columns.
    :param plate: the plate name.
    :param well: the well name.
    :returns: the rows the table holds afterwards.

    The other wells' rows are read back and written with this one's, because
    :func:`spacr.ops_store.write_table` asserts the parquet cache and the
    database hold the same rows -- an append of one well would leave a cache
    that describes a different table.
    """
    import pandas as pd

    from .ops_store import read_table, row_count, write_table

    parts = []
    if row_count(db, table) is not None:
        existing = read_table(db, table)
        if {"plate", "well"} <= set(existing.columns):
            other = ~((existing["plate"].astype(str) == str(plate))
                      & (existing["well"].astype(str) == str(well)))
            if other.any():
                parts.append(existing[other])
    parts.append(frame)
    combined = pd.concat(parts, ignore_index=True) if len(parts) > 1 else frame
    return write_table(db, table, combined, if_exists="replace")


def _well_rows(db: str, table: str, plate: str, well: str):
    """One well's rows of one table.

    :param db: the measurements database.
    :param table: an OPS table.
    :param plate: the plate name.
    :param well: the well name.
    :returns: the rows, or None when the table is absent.
    """
    from .ops_store import read_table, row_count

    if row_count(db, table) is None:
        return None
    frame = read_table(db, table)
    if not {"plate", "well"} <= set(frame.columns):
        return None
    keep = (frame["plate"].astype(str) == str(plate)) & (
        frame["well"].astype(str) == str(well))
    return frame[keep].reset_index(drop=True)


def _stitch(db: str, plate: str, well: str, cycle_files, reference: int,
            gpu: bool) -> Dict[str, Any]:
    """Solve the reference cycle's nuclear tiles and store ``ops_geometry``.

    :param db: the measurements database.
    :param plate: the plate name.
    :param well: the well name.
    :param cycle_files: ``cycle -> site -> channel -> path`` for this well.
    :param reference: the cycle carrying the nuclear stain.
    :param gpu: let the registration use the card.
    :returns: the stitch report.
    """
    import functools

    import pandas as pd

    from .ops_layout import round_well_layout
    from .ops_stitch import stitch_well

    started = time.perf_counter()
    sources = {site: _plane_sources(files).get(_NUCLEAR)
               for site, files in cycle_files[reference].items()}
    sites = sorted(site for site, source in sources.items() if source)
    unreadable: list = []
    read = functools.lru_cache(maxsize=64)(
        lambda site: _read_plane(sources[site], unreadable))
    first = next((site for site in sites if read(site) is not None), None)
    if first is None:
        raise ValueError(f"no nuclear tile of well {well} could be read")
    ordered = [first] + [site for site in sites if site != first]
    layout = round_well_layout(max(cycle_files[reference]) + 1)
    result = stitch_well(read, layout, overlap=_RASTER_OVERLAP,
                         tolerance=_STITCH_TOLERANCE, gpu=gpu, sites=ordered)
    placed = _largest_component(result.edges, sites)
    origin_y = min(result.placements[site][0] for site in placed)
    origin_x = min(result.placements[site][1] for site in placed)
    height, width = result.tile_shape
    frame = pd.DataFrame([{
        "plate": plate, "well": well, "cycle": reference, "site": site,
        "y": result.placements[site][0] - origin_y,
        "x": result.placements[site][1] - origin_x,
        "origin_y": origin_y, "origin_x": origin_x,
        "tile_height": height, "tile_width": width,
    } for site in placed])
    _replace_well_rows(db, "ops_geometry", frame, plate, well)
    residuals = result.residuals
    report = {
        "sites": len(sites), "placed": len(placed),
        "edges_proposed": result.proposed, "edges_accepted": result.accepted,
        "residual_median_px": float(np.median(residuals)) if residuals.size else None,
        "residual_max_px": float(residuals.max()) if residuals.size else None,
        "canvas": list(result.canvas), "expected_canvas": list(result.expected_canvas),
        "canvas_agrees": bool(result.canvas_agrees()),
        "unreadable": unreadable, "seconds": round(time.perf_counter() - started, 1),
    }
    _say(f"{well} stitch: {result.summary()}; {len(placed)} of {len(sites)} "
         f"in the solved component, {report['seconds']} s")
    return report


def _placements(db: str, plate: str, well: str):
    """This well's placements and tile shape, read back from ``ops_geometry``.

    :param db: the measurements database.
    :param plate: the plate name.
    :param well: the well name.
    :returns: ``({site: (y, x)}, (height, width))``.
    :raises ValueError: when the well has not been stitched.
    """
    frame = _well_rows(db, "ops_geometry", plate, well)
    if frame is None or frame.empty:
        raise ValueError(
            f"well {well} has no ops_geometry rows in {db}; run the stitch "
            "phase first")
    placements = {int(row.site): (float(row.y), float(row.x))
                  for row in frame.itertuples()}
    shape = (int(frame["tile_height"].iloc[0]), int(frame["tile_width"].iloc[0]))
    return placements, shape


def _cellpose_model(settings: Mapping[str, Any], gpu: bool):
    """The Cellpose model the objects are segmented with.

    :param settings: read for ``cellpose_model``. Empty or ``cpsam`` builds
        the library's own default model, which is how the well was validated;
        any other name is passed to Cellpose as the model to load.
    :param gpu: put it on this machine's card through
        :func:`spacr.accelerator.cellpose_kwargs`, or keep it on the CPU.
    :returns: a ``cellpose.models.CellposeModel``.
    """
    from cellpose import models

    kwargs: Dict[str, Any] = {"gpu": False}
    if gpu:
        from .accelerator import cellpose_kwargs

        kwargs = cellpose_kwargs()
    # 372 PART 14-L built the model with no ``pretrained_model`` at all, so
    # it ran the library's default -- ``cpsam_v2`` in the installed release,
    # not the ``cpsam`` the OPS settings name. Passing that name explicitly
    # would load different weights from the ones the well was validated with,
    # so the default name means the default model. Another name goes in as a
    # keyword rather than as a dict entry: the settings-flow analyser reads a
    # string subscript as a settings key, and ``kwargs["pretrained_model"]``
    # published a setting nobody can set.
    name = str(settings.get("cellpose_model") or "").strip()
    if name in ("", "cpsam"):
        return models.CellposeModel(**kwargs)
    return models.CellposeModel(pretrained_model=name, **kwargs)


def _objects(db: str, plate: str, well: str, cycle_files, reference: int,
             settings: Mapping[str, Any], gpu: bool) -> Dict[str, Any]:
    """Compose, segment, sew and number the well; store ``ops_objects``.

    :param db: the measurements database.
    :param plate: the plate name.
    :param well: the well name.
    :param cycle_files: ``cycle -> site -> channel -> path`` for this well.
    :param reference: the cycle carrying the nuclear stain.
    :param settings: read for the Cellpose model and diameter.
    :param gpu: segment on the card.
    :returns: the objects report.
    """
    import functools

    from .ops_compose import compose_window, overlap_gain, windows_over
    from .ops_objects import (ObjectsError, number, objects_frame,
                              objects_in_window, sew)
    from .ops_store import objects_ready

    started = time.perf_counter()
    placements, shape = _placements(db, plate, well)
    sources = {site: _plane_sources(cycle_files[reference].get(site, {})).get(
        _NUCLEAR) for site in placements}
    unreadable: list = []
    read = functools.lru_cache(maxsize=48)(
        lambda site: _read_plane(sources[site], unreadable))
    usable = {site: place for site, place in placements.items()
              if sources[site] is not None and read(site) is not None}
    canvas = (int(np.ceil(max(y for y, _ in usable.values()))) + shape[0],
              int(np.ceil(max(x for _, x in usable.values()))) + shape[1])
    model = _cellpose_model(settings, gpu)
    diameter = settings.get("cellpose_diameter")
    extra = {"diameter": float(diameter)} if diameter else {}

    windows = list(windows_over(canvas, size=_WINDOW, overlap=_WINDOW_OVERLAP))
    seconds = Counter()
    observations: list = []
    gains = []
    empty = 0
    for count, window in enumerate(windows, start=1):
        tick = time.perf_counter()
        image, coverage = compose_window(window, usable, read, tile_shape=shape)
        seconds["compose"] += time.perf_counter() - tick
        covered = coverage > 0
        if not covered.any():
            empty += 1
            continue
        gains.append((overlap_gain(coverage), int(covered.sum())))
        fill = image.copy()
        fill[~covered] = float(np.median(image[covered]))
        tick = time.perf_counter()
        masks = np.asarray(model.eval(fill, batch_size=16, **extra)[0], np.int32)
        seconds["segment"] += time.perf_counter() - tick
        masks[~covered] = 0
        tick = time.perf_counter()
        observations.extend(objects_in_window(window, masks, canvas=canvas))
        seconds["objects_in_window"] += time.perf_counter() - tick
        if count % 50 == 0:
            _say(f"{well} objects: {count} of {len(windows)} windows, "
                 f"{len(observations)} observations")

    tick = time.perf_counter()
    groups = sew(observations)
    seconds["sew"] = time.perf_counter() - tick
    tick = time.perf_counter()
    refusal = ""
    try:
        objects = number(groups, strict=True)
    except ObjectsError as failure:
        refusal = str(failure)
        objects = number(groups, strict=False)
    seconds["number"] = time.perf_counter() - tick
    if not objects:
        raise ValueError(f"well {well}: segmentation found no nuclei")
    frame = objects_frame(objects)
    frame.insert(0, "well", well)
    frame.insert(0, "plate", plate)
    tick = time.perf_counter()
    _replace_well_rows(db, "ops_objects", frame, plate, well)
    seconds["store"] = time.perf_counter() - tick
    gate = objects_ready(db)
    report = {
        "canvas": list(canvas), "windows": len(windows), "empty_windows": empty,
        "observations": len(observations), "groups": len(groups),
        "objects": len(objects),
        "all_clipped_groups": sum(1 for g in groups if all(o.clipped for o in g)),
        "strict_refusal": refusal[:500],
        "n_observations": dict(Counter(int(o.n_observations) for o in objects)),
        "area_median_px": float(np.median(frame["area"])),
        "overlap_gain": round(sum(g * a for g, a in gains) / max(1, sum(a for _, a in gains)), 3),
        "objects_ready": [bool(gate), gate.rows, gate.reason],
        "unreadable": unreadable,
        "seconds": {k: round(v, 1) for k, v in seconds.items()},
        "total_seconds": round(time.perf_counter() - started, 1),
    }
    _say(f"{well} objects: {len(objects)} from {len(observations)} observations "
         f"in {report['total_seconds']} s")
    return report


def _init_decode_worker(library: frozenset) -> None:
    """Hand a decode worker process the guide library once.

    :param library: the guide barcodes.
    """
    global _LIBRARY
    _LIBRARY = library


def _decode_field(task: Mapping[str, Any]) -> Dict[str, Any]:
    """Align, detect, call and attribute the reads of one field.

    :param task: ``site``, ``planes`` (``cycle -> [source per base
        channel]``), ``cycles``, ``reference``, and the nearby objects in
        this tile's frame -- ``centroids``, ``areas``, ``ids`` and ``owned``,
        whether each object's nearest tile is this one -- plus ``gpu``.
    :returns: the field's reads attributed to the objects it owns, and its
        counts.

    Reads are attributed to every nearby object, so a read in the overlap
    with the neighbouring tile is not given to the wrong nucleus merely
    because the right one belongs to the neighbour; only the reads whose
    owner this tile owns are returned, so no read is counted twice.
    """
    from scipy import ndimage

    from .ops_cycles import align_field
    from .ops_sbs import (attribute_reads, call_reads, estimate_read_locations,
                          extract_bases, find_peaks)

    ticks = Counter()
    tick = time.perf_counter()
    site, cycles, reference = task["site"], list(task["cycles"]), task["reference"]
    gpu = bool(task.get("gpu", False))
    unreadable: list = []
    planes: Dict[int, Optional[list]] = {}
    with ThreadPoolExecutor(4) as pool:
        pending = {cycle: [None if source is None else
                           pool.submit(_read_plane, source, unreadable)
                           for source in sources]
                   for cycle, sources in task["planes"].items()}
        for cycle, futures in pending.items():
            got = [None if future is None else future.result() for future in futures]
            planes[cycle] = None if any(one is None for one in got) else got
    ticks["read"] = time.perf_counter() - tick
    base = {"site": site, "unreadable": unreadable,
            "missing": sorted(c for c in cycles if planes.get(c) is None)}
    if planes.get(reference) is None:
        return {**base, "skipped": "the reference cycle could not be read"}

    tick = time.perf_counter()
    field = align_field(planes, reference=reference, gpu=gpu)
    ticks["align"] = time.perf_counter() - tick
    base.update(kept=list(field.kept), refused=list(field.refused),
                refused_channels=[list(pair) for pair in field.refused_channels])
    if len(field.kept) < _MIN_CYCLES:
        return {**base, "skipped": f"{len(field.kept)} usable cycles"}

    tick = time.perf_counter()
    stack = field.stack
    filtered = np.empty_like(stack)
    for c in range(stack.shape[0]):
        for k in range(stack.shape[1]):
            filtered[c, k] = np.clip(
                -ndimage.gaussian_laplace(stack[c, k], 1.0), 0, None)
    del stack
    score = estimate_read_locations(filtered)
    peaks = find_peaks(score, min_distance=2, gpu=gpu)
    strength = score - ndimage.minimum_filter(score, size=5)
    shifts = [abs(v) for pair in field.cycle_shifts.values() for v in pair]
    shifts += [abs(v) for per in field.channel_shifts.values()
               for pair in per for v in pair]
    margin = 5 + max(shifts + [0])
    height, width = score.shape
    if peaks.size:
        keep = strength[peaks[:, 0], peaks[:, 1]] > _THRESHOLD_READS
        keep &= (peaks[:, 0] >= margin) & (peaks[:, 0] < height - margin)
        keep &= (peaks[:, 1] >= margin) & (peaks[:, 1] < width - margin)
        peaks = peaks[keep]
    measured = extract_bases(filtered, peaks, window=1)
    del filtered
    values = np.full((len(peaks), len(cycles), measured.shape[-1]), np.nan,
                     np.float32)
    for index, cycle in enumerate(field.kept):
        values[:, cycles.index(cycle)] = measured[:, index]
    ticks["spots"] = time.perf_counter() - tick

    tick = time.perf_counter()
    calls, quality = call_reads(values, bases=_BASES, gpu=gpu)
    owner, ambiguous = attribute_reads(peaks.astype(float), task["centroids"],
                                       task["areas"], footprint=_FOOTPRINT,
                                       tie=_TIE)
    has_owner = owner >= 0
    owned = np.zeros(len(peaks), dtype=bool)
    owned[has_owner] = np.asarray(task["owned"], dtype=bool)[owner[has_owner]]
    rows = np.flatnonzero(owned)
    ids = np.asarray(task["ids"], dtype=np.int64)[owner[rows]]
    ticks["call_attribute"] = time.perf_counter() - tick
    return {
        **base, "n_cycles": len(field.kept), "spots": int(len(peaks)),
        "attributed": int(has_owner.sum()), "ambiguous": int(ambiguous.sum()),
        "exact": int(sum(code in _LIBRARY for code in calls)) if _LIBRARY else None,
        "ids": ids, "calls": [calls[i] for i in rows],
        "quality": quality[rows].astype(np.float32),
        "max_channel_shift": int(max([abs(v) for per in field.channel_shifts.values()
                                      for pair in per for v in pair] + [0])),
        "seconds": {k: round(v, 2) for k, v in ticks.items()},
    }


def _load_library(library) -> List[str]:
    """The guide barcodes, from a sequence or a CSV path.

    :param library: a sequence of barcodes, or a CSV with a ``prefix``,
        ``barcode`` or ``sequence`` column.
    :returns: the barcodes, empty when ``library`` is None.
    :raises ValueError: when a CSV has none of those columns.
    """
    if library is None:
        return []
    if isinstance(library, (str, os.PathLike)):
        # The standard-library reader, not a DataFrame: a guide library is a
        # list of sequences rather than a measurement table, so it has no
        # column the tabular funnel's canonicalisation is there to repair.
        import csv

        with open(library, newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        columns = rows[0].keys() if rows else ()
        for column in ("prefix", "barcode", "sequence"):
            if column in columns:
                return [row[column] for row in rows if row.get(column)]
        raise ValueError(
            f"{library} has no prefix, barcode or sequence column; name the "
            "column holding the guide barcodes one of those")
    return [str(v) for v in library]


def _decode(db: str, plate: str, well: str, cycle_files, reference: int,
            settings: Mapping[str, Any], gpu: bool,
            library: Sequence[str]) -> Dict[str, Any]:
    """Decode every field of the well and store ``ops_barcodes``.

    :param db: the measurements database.
    :param plate: the plate name.
    :param well: the well name.
    :param cycle_files: ``cycle -> site -> channel -> path`` for this well.
    :param reference: the cycle the objects' frame was stitched on.
    :param settings: read for ``n_workers``.
    :param gpu: let the decode use the card when it runs in this process.
    :param library: the guide barcodes, possibly empty.
    :returns: the decode report.
    :raises ValueError: when the objects are not ready.
    """
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    import pandas as pd
    from scipy.spatial import cKDTree

    from .ops_sbs import assign_reads_to_objects, correct_to_library
    from .ops_store import objects_ready

    started = time.perf_counter()
    gate = objects_ready(db)
    if not gate:
        raise ValueError(gate.reason)
    placements, shape = _placements(db, plate, well)
    objects = _well_rows(db, "ops_objects", plate, well)
    if objects is None or objects.empty:
        raise ValueError(f"well {well} has no ops_objects rows; run the "
                         "objects phase first")
    cycles = sorted(cycle_files)
    order = sorted(placements)
    centres = np.array([[placements[s][0] + shape[0] / 2,
                         placements[s][1] + shape[1] / 2] for s in order])
    oy = objects["centroid_y"].to_numpy(float)
    ox = objects["centroid_x"].to_numpy(float)
    areas = objects["area"].to_numpy(float)
    ids = objects["object_id"].to_numpy(np.int64)
    _, nearest = cKDTree(centres).query(np.column_stack([oy, ox]))
    owner_site = np.asarray(order)[nearest]

    tasks = []
    for site in order:
        top, left = placements[site]
        near = np.flatnonzero((oy > top - 30) & (oy < top + shape[0] + 30)
                              & (ox > left - 30) & (ox < left + shape[1] + 30))
        planes = {}
        for cycle in cycles:
            sources = _plane_sources(cycle_files[cycle].get(site, {}))
            planes[cycle] = [sources.get(name) for name in _BASE_CHANNELS]
        tasks.append({
            "site": site, "planes": planes, "cycles": cycles,
            "reference": reference,
            "centroids": np.column_stack([oy[near] - top, ox[near] - left]),
            "areas": areas[near], "ids": ids[near],
            "owned": owner_site[near] == site,
        })

    library_set = frozenset(library)
    workers = max(1, min(int(settings.get("n_workers") or 1),
                         _DECODE_WORKERS_CAP, len(tasks)))
    results = []
    if workers == 1:
        _init_decode_worker(library_set)
        for task in tasks:
            results.append(_decode_field({**task, "gpu": gpu}))
    else:
        # A card shared between processes is a tenant nobody announced, so
        # the fields decoded in worker processes stay on the CPU.
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(workers, mp_context=context,
                                 initializer=_init_decode_worker,
                                 initargs=(library_set,)) as pool:
            for count, result in enumerate(pool.map(
                    _decode_field, [{**t, "gpu": False} for t in tasks]), start=1):
                results.append(result)
                if count % 25 == 0:
                    _say(f"{well} decode: {count} of {len(tasks)} fields")

    decoded = [r for r in results if "skipped" not in r]
    all_ids = np.concatenate([r["ids"] for r in decoded]) if decoded else np.zeros(0, np.int64)
    all_calls = [code for r in decoded for code in r["calls"]]
    all_quality = (np.concatenate([r["quality"] for r in decoded])
                   if decoded else np.zeros(0, np.float32))
    cycles_of = {}
    for r in decoded:
        for object_id in np.unique(r["ids"]):
            cycles_of[int(object_id)] = r["n_cycles"]
    assigned = assign_reads_to_objects(all_ids, all_calls, quality=all_quality)
    distinct = sorted({row["barcode"] for row in assigned.values()})
    mapped = dict(zip(distinct, correct_to_library(distinct, library))) if library else {}
    frame = pd.DataFrame([{
        "plate": plate, "well": well, "object_id": object_id,
        "barcode": row["barcode"], "quality": row["quality"],
        "n_reads": row["reads"], "n_agreeing": row["agreeing"],
        "fraction": row["fraction"], "n_cycles": cycles_of.get(object_id, 0),
        "mapped_guide": mapped.get(row["barcode"]) or "",
    } for object_id, row in sorted(assigned.items())],
        columns=["plate", "well", "object_id", "barcode", "quality", "n_reads",
                 "n_agreeing", "fraction", "n_cycles", "mapped_guide"])
    stored = _replace_well_rows(db, "ops_barcodes", frame, plate, well)

    spots = sum(r["spots"] for r in decoded)
    inside = sum(r["attributed"] + r["ambiguous"] for r in decoded)
    containment = inside / spots if spots else 0.0
    decoded_sites = {r["site"] for r in decoded}
    owned_objects = int(np.isin(owner_site, list(decoded_sites)).sum())
    report = {
        "fields": len(tasks), "fields_decoded": len(decoded),
        "skipped": {str(r["site"]): r["skipped"] for r in results if "skipped" in r},
        "cycles_used": dict(Counter(int(r["n_cycles"]) for r in decoded)),
        "cycles_missing": {str(r["site"]): r["missing"] for r in decoded if r["missing"]},
        "cycles_refused": {str(r["site"]): r["refused"] for r in decoded if r["refused"]},
        "channels_refused": sum(len(r["refused_channels"]) for r in decoded),
        "unreadable": [item for r in results for item in r["unreadable"]],
        "spots": spots,
        "library_exact_rate": (round(sum(r["exact"] for r in decoded) / spots, 4)
                               if library and spots else None),
        "containment": round(containment, 4),
        "containment_below_gate": bool(containment < _MIN_CONTAINMENT),
        "reads_ambiguous": sum(r["ambiguous"] for r in decoded),
        "reads_owned": int(len(all_calls)),
        "objects_owned_by_decoded_fields": owned_objects,
        "objects_with_a_read": int(len(np.unique(all_ids))),
        "objects_assigned": len(assigned),
        "objects_assigned_library_exact": (sum(row["barcode"] in library_set
                                               for row in assigned.values())
                                           if library else None),
        "objects_mapped": int((frame["mapped_guide"] != "").sum()) if library else None,
        "ops_barcodes_rows": stored, "workers": workers,
        "field_seconds": dict(sum((Counter(r["seconds"]) for r in results
                                   if "seconds" in r), Counter())),
        "total_seconds": round(time.perf_counter() - started, 1),
    }
    if report["containment_below_gate"]:
        _say(f"{well} decode: ONLY {containment:.1%} OF SPOTS LIE WITHIN "
             f"{_FOOTPRINT:g} PX OF A NUCLEUS, below the {_MIN_CONTAINMENT:.0%} "
             "gate. The reads are not where the objects are; check the "
             "segmentation and the footprint before trusting these barcodes.")
    _say(f"{well} decode: {len(assigned)} objects assigned from {spots} spots "
         f"in {len(decoded)} fields, {report['total_seconds']} s")
    return report


def run_ops(settings: Mapping[str, Any], *,
            wells: Optional[Sequence[str]] = None,
            phases: Sequence[str] = _PHASES,
            library=None) -> Dict[str, Any]:
    """Stitch, segment and decode the wells of one sequencing acquisition.

    :param settings: the OPS settings. Read here: ``genotype_source``, the
        folder of sequencing tiles, searched recursively; ``dst_root``, where
        ``measurements.db`` and the per-well reports go, the source folder
        when empty; ``plate``, the source folder's name when empty;
        ``ops_gpu``; ``n_workers``, how many fields decode at once;
        ``cellpose_model`` and ``cellpose_diameter``.
    :param wells: which wells to run; every well found when None.
    :param phases: any of ``"stitch"``, ``"objects"`` and ``"decode"``, run
        in that order. A phase left out reads what it needs from the
        database, so a well can be stitched once and decoded again.
    :param library: the guide barcodes, as a sequence or the path of a CSV
        with a ``prefix``, ``barcode`` or ``sequence`` column. When given,
        each barcode is also mapped to its closest guide.
    :returns: ``{"db": path, "wells": {well: {phase: report}}}``.
    :raises ValueError: when the source holds no tiles this can name, an
        unknown phase or well is asked for, or a phase's input is missing.
    """
    settings = dict(settings or {})
    root = settings.get("genotype_source")
    if not isinstance(root, (str, os.PathLike)) or not os.path.isdir(str(root)):
        raise ValueError(
            f"genotype_source must be the folder of sequencing tiles; got {root!r}")
    root = str(root)
    unknown = [phase for phase in phases if phase not in _PHASES]
    if unknown:
        raise ValueError(f"unknown phases {unknown}; choose from {list(_PHASES)}")
    destination = str(settings.get("dst_root") or root)
    os.makedirs(destination, exist_ok=True)
    db = os.path.join(destination, "measurements.db")
    plate = str(settings.get("plate") or os.path.basename(os.path.normpath(root)))
    gpu = bool(settings.get("ops_gpu", True))
    barcodes = _load_library(library)

    index = _index_tiles(root)
    if not index:
        raise ValueError(
            f"no tile under {root} is named like 10X_c1_A1_DAPI-CY3-A594-CY5-"
            "CY7_Site-0.tif: magnification, cycle, well, channels, site")
    chosen = [str(w).upper() for w in wells] if wells else sorted(index)
    missing = [well for well in chosen if well not in index]
    if missing:
        raise ValueError(f"no tiles for wells {missing}; found {sorted(index)}")

    out: Dict[str, Any] = {"db": db, "plate": plate, "wells": {}}
    for well in chosen:
        cycle_files = index[well]
        nuclear = [cycle for cycle in sorted(cycle_files)
                   if any(_NUCLEAR in _plane_sources(files)
                          for files in cycle_files[cycle].values())]
        if not nuclear:
            raise ValueError(f"well {well} has no {_NUCLEAR} plane in any cycle")
        reference = nuclear[0]
        report: Dict[str, Any] = {"reference_cycle": reference,
                                  "cycles": sorted(cycle_files)}
        started = time.perf_counter()
        if "stitch" in phases:
            report["stitch"] = _stitch(db, plate, well, cycle_files, reference, gpu)
        if "objects" in phases:
            report["objects"] = _objects(db, plate, well, cycle_files, reference,
                                         settings, gpu)
        if "decode" in phases:
            report["decode"] = _decode(db, plate, well, cycle_files, reference,
                                       settings, gpu, barcodes)
        report["seconds"] = round(time.perf_counter() - started, 1)
        report["peak_rss_gb"] = round(max(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss) / 1024 / 1024, 2)
        folder = os.path.join(destination, well)
        os.makedirs(folder, exist_ok=True)
        with open(os.path.join(folder, "ops_report.json"), "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=1, default=str)
        out["wells"][well] = report
    return out
