"""Run an optical pooled screen's sequencing acquisition from tiles to tables.

One call takes each well from its tile files to the tables of
``measurements.db``: ``ops_geometry``, where each nuclear tile sits in the
well frame; ``ops_phenotype``, where each field of the high-magnification
phenotype acquisition lands on that frame and which tile covers it;
``ops_objects``, one row per nucleus, segmented on the composed nuclear map
and numbered once for the well; ``ops_barcodes``, one row per nucleus whose
attributed reads agree; and, when asked for, ``ops_reads``, one row per read
per cycle behind those barcodes.

A file that cannot be read costs one cycle of the field it belongs to and is
listed in the report; it does not cost the well.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

try:
    import resource
except ImportError:
    # Windows has no `resource`. The module still has to import there: the
    # only thing it is used for is the peak-memory line in the report, and a
    # missing number is not a reason to refuse to run the pipeline.
    resource = None

LOG = logging.getLogger("spacr.ops_engine")


def _peak_rss_gb() -> float:
    """Peak resident memory of this process and its children, in GB.

    Returns 0.0 where the platform cannot report it, which today means
    Windows. The caller writes the number into the well report, so a
    platform that cannot measure it says zero rather than failing the well.
    """
    if resource is None:
        return 0.0
    return round(max(
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss) / 1024 / 1024, 2)

__all__ = ["run_ops"]

#: A tile of this acquisition: ``10X_c<cycle>_<well>_<channels>_Site-<n>.tif``,
#: where ``<channels>`` is one channel (``CY3``) or a stack's planes joined by
#: hyphens (``DAPI-CY3-A594-CY5-CY7``). The channel token is what tells a
#: five-plane cycle-1 stack from a cycle's four single-channel files, so the
#: pattern has to capture it.
_TILE_PATTERN = re.compile(
    r"(?P<mag>\d+X)_c(?P<cycle>\d+)_(?P<well>[A-Z]\d{1,2})_"
    r"(?P<channel>[A-Za-z0-9-]+?)_Site[-_](?P<site>\d+)\.tiff?$",
    re.IGNORECASE)

#: A tile of an acquisition with NO CYCLE, which is what the phenotype half
#: of the reference plate is: ``20X_DAPI-GFP-A594-AF750_B1_DAPI-GFP_Site-0
#: .tif`` -- magnification, the acquisition's whole channel set, the well
#: AFTER that set, then the channel or channels THIS file holds, then the
#: site.
#:
#: THIS FILE PREDICTED ``20X_c1_B1_DAPI-GFP-A594-AF750_Site-0.tif`` AND WAS
#: WRONG, checked against
#: ``screenA/20200202_6W-LaC024A/phenotype/images/input/`` on 2026-09-19.
#: There is no cycle field, the well and the channel set are the other way
#: round, and the file's own channel is a fifth field. A phenotype
#: acquisition read with :data:`_TILE_PATTERN` alone therefore indexes ZERO
#: tiles and A4 refuses a well whose images are all present. Three channel
#: tokens per site -- ``DAPI-GFP`` as one two-plane stack, ``A594`` and
#: ``AF750`` -- for 1,281 sites in each of the six wells.
_UNCYCLED_TILE_PATTERN = re.compile(
    r"(?P<mag>\d+X)_(?P<acquisition>[A-Za-z0-9-]+)_(?P<well>[A-Z]\d{1,2})_"
    r"(?P<channel>[A-Za-z0-9-]+)_Site[-_](?P<site>\d+)\.tiff?$",
    re.IGNORECASE)

#: The cycle an uncycled acquisition is filed under. It has exactly one, and
#: the rest of the engine addresses a well as ``cycle -> site -> channel``.
_UNCYCLED_CYCLE = 1


def _match_tile(name: str, scheme: Optional[str] = None):
    """A tile name parsed, by one naming scheme or by either.

    :param name: the file's base name.
    :param scheme: ``"cycled"`` or ``"uncycled"`` to accept only that one;
        None to try the cycled pattern and then the uncycled one.
    :returns: ``(well, cycle, site, channel, magnification)``, or None.
    :raises ValueError: when ``scheme`` is neither name.

    THE CYCLED PATTERN IS TRIED FIRST AND THE ORDER MATTERS. A sequencing
    name of one channel -- ``10X_c2_A1_A594_Site-0.tif`` -- also satisfies
    the uncycled pattern, which would read its cycle token ``c2`` as the
    acquisition's channel set and file all eleven cycles as one. Tried in
    this order, a name that carries a cycle is never read as one that does
    not.

    SO THE TWO SCHEMES ARE EXCLUSIVE, NOT ORDERED, and asking for
    ``"uncycled"`` is not "skip the first pattern": it is "a name the first
    pattern does NOT claim". Reading it as the former puts every sequencing
    tile in the phenotype index of a root that holds both, which is the
    same collision the other way round. Their union is what ``None``
    returns, and nothing is in both.

    AND ONE SCHEME AT A TIME IS WHY THIS TAKES A PARAMETER. The two halves
    of this plate sit side by side under one folder, so "either scheme"
    over a root that holds both is not a generous reading -- it is two
    acquisitions in one index. See :func:`_index_tiles`.
    """
    if scheme not in (None, "cycled", "uncycled"):
        raise ValueError(
            f"scheme must be 'cycled', 'uncycled' or None; got {scheme!r}")
    found = _TILE_PATTERN.search(name)
    if found:
        if scheme == "uncycled":
            return None
        return (found["well"].upper(), int(found["cycle"]),
                int(found["site"]), found["channel"].upper(), found["mag"])
    if scheme == "cycled":
        return None
    found = _UNCYCLED_TILE_PATTERN.search(name)
    if not found:
        return None
    return (found["well"].upper(), _UNCYCLED_CYCLE, int(found["site"]),
            found["channel"].upper(), found["mag"])

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

#: How many phenotype fields A4 tries to align before it fits the raster.
#: Six is what PART 14-M measured: six alignments predicted well A1's other
#: 472 field centres to a median 1.2 px and a worst 4.0 px, and A2's 451 to a
#: worst 4.4 px. Three is the arithmetic floor and leaves no residual to read,
#: so a bad anchor cannot be seen; six leaves three degrees of freedom.
_PHENOTYPE_ANCHORS = 6

#: How many candidates A4 may try to reach :data:`_PHENOTYPE_ANCHORS`. An
#: alignment that refuses costs one field's points and nothing else, and a
#: well where more than this many refuse is not a well the raster should be
#: fitted on at all.
_PHENOTYPE_CANDIDATES = 18

#: How far either side of a seeded centre the sequencing window reaches, in
#: well-frame pixels. It has to hold the SEED's error, and no more: every
#: extra pixel of window adds sequencing nuclei that are not under the field,
#: and the alignment has to find the field's own among them. PART 14-B/C
#: used 1,600 px when the seed was a grid index off by up to 1,900 px; the
#: seed is now the fitted sequencing raster, and on the plate (372,
#: 2026-09-22) it landed within 256 px of the aligned centre on wells A1 and
#: A2. At 1,600 px, 4 of A1's 18 candidates aligned and 2 of A2's, so A2
#: refused; at 800 px, 10 and 9 did, and a field found at both sizes landed
#: within 0.8 px of itself. 600 px lost fields whose seed was 240 px out.
_ANCHOR_SEARCH_PX = 800

#: The most sequencing nuclei an anchor alignment is offered. The seed search
#: is a KD-tree query per trial and the trial count is proportional to the
#: target, so an unbounded window would make the cost quadratic in the pad.
_ANCHOR_TARGET_POINTS = 3000

#: The detectors ``ops_spot_detector`` may name. ``native`` is spaCR's own
#: Laplacian-of-Gaussian score; ``spotnet`` is DeepCell's SpotNet, run in its
#: own environment, NON-COMMERCIAL ACADEMIC USE ONLY, and opt-in.
_SPOT_DETECTORS = ("native", "spotnet")

#: SpotNet's detection probability. deepcell-spots' own default.
_SPOTNET_THRESHOLD = 0.95

#: How many fields decode at once, whatever ``n_workers`` asks for. One field
#: holds eleven cycles of four 1,480 px channels twice over (the aligned
#: stack and its filtered copy), about 1.5 GB at its peak.
_DECODE_WORKERS_CAP = 8

#: The four steps, in the order they run. ``phenotype`` sits between the
#: stitch and the segmentation because it needs the well frame and nothing
#: else, so a run that only wants the placement stops after two phases.
_PHASES = ("stitch", "phenotype", "objects", "decode")

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


def _index_tiles(root: str, scheme: Optional[str] = None
                 ) -> Dict[str, Dict[int, Dict[int, Dict[str, str]]]]:
    """Every tile under ``root``, as ``well -> cycle -> site -> channel -> path``.

    :param root: the acquisition folder, searched recursively.
    :param scheme: which naming scheme counts -- ``"cycled"`` for a
        sequencing acquisition, ``"uncycled"`` for a phenotype one, None to
        take whichever the folder turns out to use.
    :returns: the index; empty when nothing matched. An acquisition whose
        names carry no cycle -- the phenotype half -- is filed under cycle
        :data:`_UNCYCLED_CYCLE`.

    ONE INDEX IS ONE ACQUISITION, WHICH IS WHY ``scheme`` EXISTS AND WHY
    None DOES NOT MEAN "BOTH". The two halves of the reference plate are
    siblings -- ``20200202_6W-LaC024A/sequencing`` and
    ``.../phenotype`` -- and both settings that name a folder say the
    subfolders are searched too, so a root holding both is what an operator
    who points either setting at the plate reaches. Filing both under one
    well would put 1,281 phenotype fields and 333 sequencing ones in one
    site range, replace cycle 1's ``A594`` tile with the 20X file of the
    same channel, and hand the scale a 1,480 px tile where a 2,960 px one
    belongs. So a mixed root resolves to ONE half: the scheme asked for, or
    -- when nothing is asked for -- the cycled half if there is one, since
    a cycled name is the more specific of the two.

    Names that are directories are skipped by the walk itself, which matters
    here: PART 14-D found ``*.tif`` names on this plate that are folders.
    Names that are not a readable tile are skipped by the pattern, which
    matters too: the phenotype folder holds thirteen
    ``*.tif.lftp-pget-status`` files, the remains of interrupted downloads,
    and an index that took those for tiles would hand tifffile a status
    file.
    """
    wanted = ("cycled", "uncycled") if scheme is None else (scheme,)
    found: Dict[str, Dict[str, Dict[int, Dict[int, Dict[str, str]]]]] = {
        name: {} for name in wanted}
    for folder, _folders, files in os.walk(root, followlinks=True):
        for name in files:
            for which in wanted:
                parsed = _match_tile(name, which)
                if parsed is None:
                    continue
                well, cycle, site, channel, _magnification = parsed
                found[which].setdefault(well, {}).setdefault(
                    cycle, {}).setdefault(
                        site, {})[channel] = os.path.join(folder, name)
                break
    for which in wanted:
        if found[which]:
            return found[which]
    return {}


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


def _channel_axis(array: np.ndarray, axes: str) -> int:
    """Which axis of a stack the channels lie along.

    :param array: the series as read.
    :param axes: the axis letters tifffile reports for that series, or an
        empty string when it reports none.
    :returns: the axis index. Never one of the last two, which are the
        image.

    THE CHANNEL AXIS IS NOT ALWAYS THE FIRST ONE, and taking it for the
    first is right on one half of this plate and wrong on the other. Read
    off ``screenA/20200202_6W-LaC024A`` on 2026-09-19, one
    ``tifffile.TiffFile`` open per file and no pixels:

    ===================================== ========== ======================
    file                                  axes       shape
    ===================================== ========== ======================
    ``10X_c1_B1_DAPI-CY3-A594-CY5-CY7_..`` ``CYX``   ``(5, 1480, 1480)``
    ``20X_..._B1_DAPI-GFP_Site-0.tif``     ``ZCYX``  ``(4, 2, 2960, 2960)``
    ``20X_..._B1_A594_Site-0.tif``         ``ZYX``   ``(4, 2960, 2960)``
    ===================================== ========== ======================

    So the sequencing stack's plane 1 IS axis 0's element 1, and the
    phenotype stack's plane 1 is not: axis 0 there is four focal planes.
    ``DAPI-GFP`` indexed on axis 0 gives DAPI for plane 0 -- by luck, since
    DAPI is also z 0 -- and for plane 1 gives z 1's DAPI rather than GFP.
    A4 reads only DAPI, so nothing it measured moves; Phase C4 measures the
    phenotype channels at the same object ids and would have read the wrong
    one.

    WHEN THE FILE NAMES NO AXES, AXIS 0 IS TAKEN AND NOTHING IS GUESSED.
    A 4-D file that names nothing is genuinely ambiguous, and the two
    orderings are both in this repository:
    ``tests/test_the_ops_engine_says_why_a_well_cannot_run.py`` pins a
    ``(channel, z, y, x)`` fixture and the plate writes ``(z, channel, y,
    x)``. Nothing in the bytes tells them apart, so the unnamed case keeps
    the reading this engine has always had -- which is also right for every
    3-D stack, the sequencing half included -- and the plate is read
    correctly because its files say ``ZCYX`` rather than because a rule
    guessed it.
    """
    letters = list(axes) if len(axes) == array.ndim else []
    if "C" in letters[:-2]:
        return letters.index("C")
    return 0


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
    through to end the run. ``IndexError`` joins them because a file with no
    series at all answers the same question. ``tifffile.TiffFileError``
    joins them because it derives from ``Exception`` alone, not from
    ValueError: a ZERO-BYTE file raises "not a TIFF file b''". On this plate
    that is ``c4/10X_c4_B3_CY3_Site-59.tif``, an lftp download that never
    finished and left its ``.lftp-pget-status`` beside it, and on 2026-09-26
    that one file failed all of well B3 in the decode phase after 45 minutes.

    THE CHANNEL IS TAKEN ON THE AXIS THE FILE SAYS IT IS ON -- see
    :func:`_channel_axis` -- AND EVERY OTHER NON-IMAGE AXIS AT 0. On this
    plate that last part is the focal plane: the phenotype half has four,
    and the first is taken because nothing here has compared them. The
    queued plate runs record the other three's sharpness so that stays a
    choice somebody made rather than one nobody noticed.
    """
    if source is None:
        return None
    path, plane = source
    if not os.path.isfile(path):
        if unreadable is not None:
            unreadable.append((path, "not a file"))
        return None
    import tifffile

    try:
        with tifffile.TiffFile(path) as handle:
            series = handle.series[0]
            axes = str(getattr(series, "axes", "") or "")
            array = np.asarray(series.asarray())
    except (OSError, ValueError, IndexError,
            getattr(tifffile, "TiffFileError", ValueError)) as failure:
        if unreadable is not None:
            unreadable.append((path, f"{type(failure).__name__}: {failure}"[:200]))
        return None
    if plane is not None:
        axis = _channel_axis(array, axes) if array.ndim >= 3 else -1
        if axis < 0 or array.shape[axis] <= plane:
            if unreadable is not None:
                unreadable.append((path, f"has no plane {plane}"))
            return None
        array = np.take(array, plane, axis=axis)
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
            settings: Mapping[str, Any], gpu: bool) -> Dict[str, Any]:
    """Solve the reference cycle's nuclear tiles and store ``ops_geometry``.

    :param db: the measurements database.
    :param plate: the plate name.
    :param well: the well name.
    :param cycle_files: ``cycle -> site -> channel -> path`` for this well.
    :param reference: the cycle carrying the nuclear stain.
    :param settings: read for ``ops_raster_overlap``.
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
    overlap = _setting_number(settings.get("ops_raster_overlap"),
                              "ops_raster_overlap", _RASTER_OVERLAP)
    result = stitch_well(read, layout, overlap=overlap,
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
        "canvas_agrees": bool(result.canvas_agrees()), "raster_overlap": overlap,
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


def _setting_number(value: Any, key: str, fallback):
    """One numeric setting, or the measured constant when it is not set.

    THE CONSTANT REMAINS THE DEFAULT. Each of these numbers was measured on
    the reference plate and is recorded with its measurement at the top of
    this module; the setting exists so a different acquisition can be run
    without editing the package, not so the measured value moves. An empty
    box therefore means "the measured one", which is also what keeps a
    settings dict that predates the key working.

    THE VALUE IS PASSED IN RATHER THAN THE MAPPING, which is not a style
    choice. Every generator in this repository that answers "where does
    this setting go" -- ``tools/settings_flow.py``,
    ``tools/build_setting_consumer_map.py`` -- reads the package with
    ``ast`` and recognises exactly ``settings[...]`` and
    ``settings.get(...)`` on a literal. A key spelled as an argument to a
    helper is invisible to all of them, so ``ops_raster_overlap``,
    ``ops_window_overlap``, ``ops_read_threshold`` and ``ops_footprint``
    had a tooltip and a type and no flow section, no consumer row, and a
    tooltip link that landed nowhere -- while the four read with a literal
    ``settings.get`` did not. Reading the value at the call site puts all
    eight in one class.

    :param value: what the settings hold for ``key``: pass
        ``settings.get("<key>")`` with the key written out.
    :param key: the setting's name, for the message.
    :param fallback: the module constant, whose type the value is coerced to.
    :returns: the number to use.
    :raises ValueError: when the setting holds something that is not a number.
    """
    if value is None or value == "":
        return fallback
    try:
        return type(fallback)(value)
    except (TypeError, ValueError) as failure:
        raise ValueError(
            f"{key} must be a number; got {value!r}") from failure


def _base_channels(settings: Mapping[str, Any]) -> Tuple[str, ...]:
    """The channels carrying the bases, in the order the bases are read.

    :param settings: read for ``ops_base_channels``, a comma-separated list.
    :returns: the channel tokens, upper-cased as :func:`_index_tiles` stores
        them.
    :raises ValueError: when the list does not name exactly one channel per
        base. A short list would decode a shorter barcode than the library
        holds and a long one would read a base that has no letter, and both
        are quieter as a refusal here than as a low library match later.
    """
    raw = settings.get("ops_base_channels")
    if not raw:
        return _BASE_CHANNELS
    if isinstance(raw, str):
        parts = [piece.strip() for piece in raw.replace(";", ",").split(",")]
    else:
        parts = [str(piece).strip() for piece in raw]
    names = tuple(piece.upper() for piece in parts if piece)
    if len(names) != len(_BASES):
        raise ValueError(
            f"ops_base_channels must name one channel per base "
            f"({', '.join(_BASES)}); got {list(names)}")
    return names


def _tile_magnification(path: str) -> Optional[float]:
    """The objective a tile was taken with, from its name.

    :param path: any tile path :func:`_index_tiles` indexed.
    :returns: the number before the ``X``, or None when the name does not
        carry one.
    """
    parsed = _match_tile(os.path.basename(str(path)))
    if parsed is None:
        return None
    try:
        return float(parsed[4][:-1])
    except ValueError:
        return None


def _one_acquisition(index: Mapping[str, Any], setting: str, root: str) -> None:
    """Refuse an index that holds tiles of more than one objective.

    :param index: a ``well -> cycle -> site -> channel -> path`` index.
    :param setting: the setting that named ``root``, for the message.
    :param root: the folder, for the message.
    :raises ValueError: when two magnifications are indexed together.

    ONE ACQUISITION HAS ONE OBJECTIVE, so two in one index is two
    acquisitions, and everything downstream is keyed on there being one: the
    tile shape the stitch solves at, and the scale A4 divides by. Naming the
    scheme (:func:`_index_tiles`) separates the two halves of THIS plate,
    where the phenotype names carry no cycle. It cannot separate a phenotype
    acquisition that does carry one from the sequencing acquisition beside
    it, because then both are the same scheme -- and that is the case this
    catches, at the cost of one regular expression per indexed path.
    """
    seen = sorted({found for cycles in index.values()
                   for sites in cycles.values()
                   for files in sites.values()
                   for found in [_tile_magnification(path)
                                 for path in files.values()]
                   if found is not None})
    if len(seen) > 1:
        raise ValueError(
            f"{setting} ({root}) holds tiles taken at "
            f"{' and '.join(f'{value:g}X' for value in seen)}, which is more "
            "than one acquisition. Point it at the one acquisition's folder.")


def _any_tile(files) -> Optional[str]:
    """One tile path out of a ``cycle -> site -> channel -> path`` index.

    :param files: the index of one well.
    :returns: a path, or None when the index is empty.
    """
    for _cycle, sites in sorted(files.items()):
        for _site, channels in sorted(sites.items()):
            for _channel, path in sorted(channels.items()):
                return path
    return None


def _fitted_raster(layout, centres: Mapping[int, Tuple[float, float]]):
    """Origin, column step and row step of a measured acquisition raster.

    :param layout: the acquisition's :class:`spacr.ops_layout.WellLayout`.
    :param centres: ``{site: (y, x)}``, measured well-frame centres.
    :returns: a ``(3, 2)`` array -- the grid origin and the two steps -- or
        None when the layout does not hold every site, or the sites lie on
        one grid line.

    This is the sequencing side of what
    :func:`spacr.ops_phenotype.phenotype_centres` does for the phenotype
    acquisition, and it is only ever used to SEED the anchor search: a seed
    that is wrong by a tile still puts the right pixels inside the window,
    and the window is a tile wide on each side.
    """
    sites = sorted(centres)
    try:
        grid = np.array([layout.position(site) for site in sites], float)
    except IndexError:
        return None
    design = np.column_stack([np.ones(len(sites)), grid])
    if len(sites) < 3 or np.linalg.matrix_rank(design) < 3:
        return None
    measured = np.array([centres[site] for site in sites], float)
    solution, *_ = np.linalg.lstsq(design, measured, rcond=None)
    return solution


def _anchor_candidates(layout, wanted: int, limit: int) -> List[int]:
    """Sites to try as anchors, spread over the well, best first.

    :param layout: the phenotype acquisition's layout.
    :param wanted: how many anchors the raster fit wants.
    :param limit: the most candidates to return.
    :returns: site numbers.

    SPREAD, NOT THE FIRST N. The fit is an origin and two steps, so anchors
    from one corner fix the origin well and the steps badly, and anchors
    from one grid line fix nothing across it -- which
    :func:`spacr.ops_phenotype.phenotype_centres` refuses outright. One
    field at the centre and the rest around a circle inside the well give a
    conditioned design whichever ones then fail to align.
    """
    points = np.array(layout.positions(), float)
    centre = np.array(layout.centre, float)
    spokes = max(1, int(wanted) - 1)
    targets = [centre]
    for step in range(spokes):
        angle = 2.0 * np.pi * step / spokes
        targets.append(centre + 0.65 * float(layout.radius)
                       * np.array([np.cos(angle), np.sin(angle)]))
    chosen: List[int] = []
    for target in targets:
        order = np.argsort(np.hypot(*(points - target).T))
        for site in order:
            if int(site) not in chosen:
                chosen.append(int(site))
                break
    rest = [site for site in range(len(points)) if site not in chosen]
    while rest and len(chosen) < limit:
        taken = points[chosen]
        away = [min(np.hypot(*(taken - points[site]).T)) for site in rest]
        pick = rest.pop(int(np.argmax(away)))
        chosen.append(pick)
    return chosen[:limit]


def _anchor_window(seed: Tuple[float, float], extent: Tuple[float, float],
                   canvas: Tuple[int, int]):
    """The sequencing window one phenotype field is searched for in.

    :param seed: the predicted ``(y, x)`` centre in the well frame.
    :param extent: the field's ``(height, width)`` in well-frame pixels.
    :param canvas: the stitched well's ``(height, width)``.
    :returns: the :class:`spacr.ops_compose.Window`, clamped to the canvas,
        or None when it does not meet the canvas at all.
    """
    from .ops_compose import Window

    top = int(np.floor(seed[0] - extent[0] / 2.0 - _ANCHOR_SEARCH_PX))
    left = int(np.floor(seed[1] - extent[1] / 2.0 - _ANCHOR_SEARCH_PX))
    bottom = int(np.ceil(seed[0] + extent[0] / 2.0 + _ANCHOR_SEARCH_PX))
    right = int(np.ceil(seed[1] + extent[1] / 2.0 + _ANCHOR_SEARCH_PX))
    top, left = max(0, top), max(0, left)
    bottom, right = min(int(canvas[0]), bottom), min(int(canvas[1]), right)
    if bottom - top <= 0 or right - left <= 0:
        return None
    return Window(top=top, left=left, height=bottom - top, width=right - left)


def _phenotype(db: str, plate: str, well: str, cycle_files, reference: int,
               phenotype_files, magnifications: Tuple[Optional[float],
                                                      Optional[float]],
               settings: Mapping[str, Any]) -> Dict[str, Any]:
    """Place every phenotype field on the stitched well; store ``ops_phenotype``.

    :param db: the measurements database.
    :param plate: the plate name.
    :param well: the well name.
    :param cycle_files: the SEQUENCING index for this well.
    :param reference: the sequencing cycle carrying the nuclear stain.
    :param phenotype_files: the PHENOTYPE index for this well, same shape.
    :param magnifications: the two acquisitions' objectives, sequencing
        first, from the tile names; either may be None, which only costs the
        alignment its second seed.
    :param settings: unused today; taken so the phase signature matches the
        others and a future knob does not change every call site.
    :returns: the phenotype report.
    :raises ValueError: when the well has no phenotype tile with a nuclear
        plane, when its field count is not a round well's, or when fewer
        than three fields aligned -- a raster cannot be fitted on two.

    THE STEP IN ONE SENTENCE: align a handful of phenotype fields to the
    stitched sequencing map, fit the acquisition raster to those few, and
    predict where every other field's centre lands. PART 14-M measured the
    alternative -- halving a phenotype grid index about the well centre --
    at 0.45 to 0.61 of fields on the right tile, because at twice the tile
    density half the fields sit on or near a sequencing tile boundary and a
    grid index cannot see where the boundary fell.

    THE SCALE IS NOT THE MAGNIFICATION RATIO, and taking it for one is an
    error a fixture will not catch if the fixture was built from the same
    mistake. The objectives give the field of view, the tile gives the
    pixels across it, and the scale is the ratio of the two densities:
    ``(sbs width x sbs magnification) / (phenotype width x phenotype
    magnification)``. On the reference plate that is
    ``1480 x 10 / (2960 x 20) = 0.25`` -- a quarter, not a half, because the
    phenotype tile spends twice as many pixels on half the field. 0.25 is
    the number :mod:`spacr.ops_phenotype` records every one of its
    constants against.
    """
    import functools

    import pandas as pd

    from .ops_compose import compose_window
    from .ops_layout import round_well_layout
    from .ops_phenotype import (align_phenotype_to_sbs, nuclear_points,
                                phenotype_centres, phenotype_site_map)

    started = time.perf_counter()
    placements, shape = _placements(db, plate, well)
    sbs_centres = {site: (top + shape[0] / 2.0, left + shape[1] / 2.0)
                   for site, (top, left) in placements.items()}
    canvas = (int(np.ceil(max(y for y, _ in placements.values()))) + shape[0],
              int(np.ceil(max(x for _, x in placements.values()))) + shape[1])

    cycle = min(phenotype_files)
    sources = {site: _plane_sources(files).get(_NUCLEAR)
               for site, files in phenotype_files[cycle].items()}
    sites = sorted(site for site, source in sources.items() if source)
    if not sites:
        raise ValueError(
            f"no phenotype tile of well {well} carries a {_NUCLEAR} plane; "
            f"A4 aligns on the nuclear stain and has nothing to align")
    offered = max(phenotype_files[cycle]) + 1
    try:
        layout = round_well_layout(offered)
    except ValueError as failure:
        raise ValueError(
            f"well {well}'s phenotype acquisition has {offered} fields, "
            f"which no round well holds: {failure}") from failure

    sbs_magnification, phenotype_magnification = magnifications
    scale: Optional[float] = None
    sbs_layout = None
    try:
        sbs_layout = round_well_layout(max(placements) + 1)
    except ValueError:
        sbs_layout = None
    raster = (_fitted_raster(sbs_layout, sbs_centres)
              if sbs_layout is not None else None)
    middle = (float(np.mean([y for y, _ in sbs_centres.values()])),
              float(np.mean([x for _, x in sbs_centres.values()])))
    ratio = (float(sbs_layout.radius) / float(layout.radius)
             if sbs_layout is not None and layout.radius else 1.0)

    def seed_of(site: int) -> Tuple[float, float]:
        """Where a phenotype field's centre is expected, before aligning it.

        :param site: the phenotype site.
        :returns: the ``(y, x)`` seed in the well frame.
        """
        if raster is None or sbs_layout is None:
            return middle
        column, row = layout.position(site)
        position = (np.array(sbs_layout.centre, float)
                    + (np.array([column, row], float)
                       - np.array(layout.centre, float)) * ratio)
        return tuple(raster[0] + position[0] * raster[1]
                     + position[1] * raster[2])

    unreadable: list = []
    sbs_sources = {site: _plane_sources(cycle_files[reference].get(site, {})
                                        ).get(_NUCLEAR) for site in placements}
    read_sbs = functools.lru_cache(maxsize=48)(
        lambda site: _read_plane(sbs_sources[site], unreadable))

    def usable_under(window) -> Dict[int, Tuple[float, float]]:
        """The placed tiles that touch ``window`` and read.

        :param window: the rectangle about to be composed.
        :returns: ``site -> (y, x)`` for :func:`~spacr.ops_compose.compose_window`.

        A WELL IS NOT OPENED TO COMPOSE A WINDOW OF IT. Six anchors need
        about nine tiles each; testing readability over every placement
        instead pulled all 333 of the reference well's sequencing tiles,
        and ``tifffile`` reads the whole five-plane 1,480 px file for one
        plane of it -- 22 MB each, about 7 GB over NFS per well, before
        ``compose_window`` then re-read the nine it wanted. The overlap
        test is the same rounding ``compose_window`` uses, so a tile is
        offered here exactly when it would have been used there; a tile
        that will not read is dropped so the composer never sees a None.
        """
        near: Dict[int, Tuple[float, float]] = {}
        for site, (top, left) in placements.items():
            if sbs_sources.get(site) is None:
                continue
            t_top, t_left = int(round(float(top))), int(round(float(left)))
            if t_top + shape[0] <= window.top or t_top >= window.bottom:
                continue
            if t_left + shape[1] <= window.left or t_left >= window.right:
                continue
            if read_sbs(site) is None:
                continue
            near[site] = (top, left)
        return near

    anchors: Dict[int, Tuple[float, float]] = {}
    records: List[Dict[str, Any]] = []
    tried = 0
    for site in _anchor_candidates(layout, _PHENOTYPE_ANCHORS,
                                   _PHENOTYPE_CANDIDATES):
        if len(anchors) >= _PHENOTYPE_ANCHORS:
            break
        plane = _read_plane(sources.get(site), unreadable)
        if plane is None:
            continue
        tried += 1
        if scale is None and sbs_magnification and phenotype_magnification:
            scale = float(shape[1] * sbs_magnification) / float(
                plane.shape[1] * phenotype_magnification)
        extent = (plane.shape[0] * (scale or 1.0),
                  plane.shape[1] * (scale or 1.0))
        window = _anchor_window(seed_of(site), extent, canvas)
        if window is None:
            continue
        image, coverage = compose_window(window, usable_under(window), read_sbs,
                                         tile_shape=shape)
        covered = coverage > 0
        if not covered.any():
            continue
        image[~covered] = float(np.median(image[covered]))
        found = align_phenotype_to_sbs(
            nuclear_points(plane), nuclear_points(
                image, top=_ANCHOR_TARGET_POINTS),
            expected_scale=scale)
        if found is None:
            continue
        centre = found.apply(np.array([[plane.shape[0] / 2.0,
                                        plane.shape[1] / 2.0]]))[0]
        anchors[site] = (float(centre[0]) + window.top,
                         float(centre[1]) + window.left)
        records.append({
            "site": site, "inliers": found.inliers,
            "residual_px": round(found.residual_px, 3),
            "scale": round(found.scale, 5),
            "degrees": round(found.degrees, 3),
            "points": found.points,
            "centre_y": round(anchors[site][0], 2),
            "centre_x": round(anchors[site][1], 2),
        })

    if len(anchors) < 3:
        raise ValueError(
            f"well {well}: {len(anchors)} of {tried} phenotype fields "
            f"aligned to the sequencing frame, and an acquisition raster "
            f"takes at least three that are not on one grid line. Check "
            f"that the phenotype folder is the same well and that its "
            f"nuclear channel is named {_NUCLEAR}.")

    predicted = phenotype_centres(layout, anchors)
    mapping = phenotype_site_map(layout, sbs_centres, anchors,
                                 tile_shape=shape)
    by_site = {record["site"]: record for record in records}
    for record in records:
        fitted = predicted[record["site"]]
        record["raster_residual_px"] = round(float(np.hypot(
            fitted[0] - anchors[record["site"]][0],
            fitted[1] - anchors[record["site"]][1])), 3)

    frame = pd.DataFrame([{
        "plate": plate, "well": well, "site": site,
        "centre_y": centre[0], "centre_x": centre[1],
        "sbs_site": int(mapping.get(site, -1)),
        "is_anchor": int(site in anchors),
        "inliers": int(by_site[site]["inliers"]) if site in by_site else 0,
        "alignment_residual_px": (float(by_site[site]["residual_px"])
                                  if site in by_site else float("nan")),
        "raster_residual_px": (float(by_site[site]["raster_residual_px"])
                               if site in by_site else float("nan")),
    } for site, centre in sorted(predicted.items())],
        columns=["plate", "well", "site", "centre_y", "centre_x", "sbs_site",
                 "is_anchor", "inliers", "alignment_residual_px",
                 "raster_residual_px"])
    _replace_well_rows(db, "ops_phenotype", frame, plate, well)

    residuals = [record["raster_residual_px"] for record in records]
    report = {
        "fields": len(predicted), "tiles_found": len(sites),
        "anchors_tried": tried, "anchors_used": len(anchors),
        "anchors": records,
        "anchor_residual_px": {
            "median": round(float(np.median(residuals)), 3),
            "max": round(float(np.max(residuals)), 3),
        },
        "expected_scale": scale,
        "raster": None if raster is None else {
            "origin": [round(float(v), 2) for v in raster[0]],
            "column_step": [round(float(v), 2) for v in raster[1]],
            "row_step": [round(float(v), 2) for v in raster[2]],
        },
        "fields_mapped": len(mapping),
        "fields_off_the_stitch": len(predicted) - len(mapping),
        "sbs_tiles_used": len(set(mapping.values())),
        "ops_phenotype_rows": int(len(frame)),
        "unreadable": unreadable,
        "seconds": round(time.perf_counter() - started, 1),
    }
    _say(f"{well} phenotype: {len(mapping)} of {len(predicted)} fields placed "
         f"on {report['sbs_tiles_used']} tiles from {len(anchors)} anchors, "
         f"{report['seconds']} s")
    return report


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

    kwargs: Dict[str, Any] = {"gpu": False, "use_bfloat16": False}
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
                              objects_in_window, sew, unseen_records)
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

    window_overlap = int(_setting_number(settings.get("ops_window_overlap"),
                                         "ops_window_overlap",
                                         _WINDOW_OVERLAP))
    windows = list(windows_over(canvas, size=_WINDOW,
                                overlap=window_overlap))
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
    refused: Tuple[Dict[str, Any], ...] = ()
    try:
        objects = number(groups, strict=True)
    except ObjectsError as failure:
        refusal = str(failure)
        refused = unseen_records(groups)
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
        "refusals": (_explain_refusals(refused, frame, window_overlap)
                     if refused else None),
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


#: How many refused groups the report describes one by one. Well A1 refused
#: 54 and A2 62; a well that refuses thousands has a different problem and
#: the counts above the list say so without a megabyte of JSON.
_REFUSALS_LISTED = 500


def _explain_refusals(refused: Sequence[Mapping[str, Any]], frame,
                      overlap: int) -> Dict[str, Any]:
    """Say of each refused group whether a numbered object covers it.

    THE QUESTION THIS ANSWERS was left open by well A1's run: 54 groups had
    no complete observation, ``number(strict=True)`` refused the well, and
    the engine numbered it without them. Whether that lost 54 nuclei or
    dropped 54 leftovers of nuclei already counted could not be told
    afterwards, because the observations are not stored and the report kept
    one box and a count. It can be told HERE, while both are in memory, and
    it costs one KD-tree query per refusal.

    A refusal whose box a numbered object overlaps is a sliver of a nucleus
    that another window saw whole: dropping it cost nothing, and the join
    that left it unattached is the thing to look at. A refusal with no
    numbered object over it is the only evidence of something at that spot,
    so dropping it dropped that something -- a Cellpose fragment of debris,
    or a nucleus no window saw whole.

    :param refused: :func:`spacr.ops_objects.unseen_records` output.
    :param frame: the ``ops_objects`` rows that WERE numbered.
    :param overlap: the window overlap the run used, in pixels.
    :returns: the counts, and one record per refusal up to
        :data:`_REFUSALS_LISTED`. A refusal with no numbered object within
        reach carries ``nearest_object_px`` of None rather than infinity,
        because the report is written with :func:`json.dump` and
        ``Infinity`` is Python's spelling of it rather than JSON's.
    """
    from scipy.spatial import cKDTree

    records = [dict(one) for one in refused]
    centroids = np.column_stack([frame["centroid_y"].to_numpy(float),
                                 frame["centroid_x"].to_numpy(float)])
    boxes = np.column_stack([frame[name].to_numpy(float) for name in
                             ("bbox_top", "bbox_left", "bbox_bottom",
                              "bbox_right")])
    ids = frame["object_id"].to_numpy(np.int64)
    tree = cKDTree(centroids) if len(centroids) else None
    for record in records:
        centre = np.array([record["centre_y"], record["centre_x"]], float)
        reach = float(np.hypot(record["height"], record["width"])) / 2.0 + 64.0
        record["covered"] = False
        record["nearest_object_id"] = -1
        record["nearest_object_px"] = None
        if tree is None:
            continue
        near = np.asarray(tree.query_ball_point(centre, r=reach), np.int64)
        if near.size:
            away = np.hypot(*(centroids[near] - centre).T)
            closest = int(near[int(np.argmin(away))])
            record["nearest_object_id"] = int(ids[closest])
            record["nearest_object_px"] = round(float(away.min()), 2)
            over = ((boxes[near, 0] <= record["bottom"])
                    & (boxes[near, 2] >= record["top"])
                    & (boxes[near, 1] <= record["right"])
                    & (boxes[near, 3] >= record["left"]))
            record["covered"] = bool(over.any())
    largest = max(records, key=lambda one: max(one["height"], one["width"]))
    return {
        "groups": len(records),
        "largest_px": [largest["height"], largest["width"]],
        "largest_centre": [round(largest["centre_y"], 1),
                           round(largest["centre_x"], 1)],
        "window_overlap_px": int(overlap),
        "wider_than_the_overlap": sum(
            1 for one in records
            if max(one["height"], one["width"]) > overlap),
        "spanning_a_window_side": sum(1 for one in records
                                      if one["spanned_side"]),
        "covered_by_a_numbered_object": sum(1 for one in records
                                            if one["covered"]),
        "orphaned": sum(1 for one in records if not one["covered"]),
        "area_total_px": sum(int(one["area_total"]) for one in records),
        "records": records[:_REFUSALS_LISTED],
    }


def _spot_detector(settings: Mapping[str, Any]) -> str:
    """The sequencing-spot detector ``settings`` choose, checked it can run.

    :param settings: read for ``ops_spot_detector``; empty means native.
    :returns: ``native`` or ``spotnet``.
    :raises ValueError: for an unknown name, or SpotNet when it cannot run
        here, with the reason -- never a silent fall back to native, which
        would give a run that asked for one detector the other's reads.
    """
    name = str(settings.get("ops_spot_detector") or "native").strip().lower()
    if name not in _SPOT_DETECTORS:
        raise ValueError(f"ops_spot_detector must be one of "
                         f"{list(_SPOT_DETECTORS)}; got {name!r}")
    if name == "spotnet":
        from ._segmentation_backends import _spotnet_readiness

        ready, reason = _spotnet_readiness()
        if not ready:
            raise ValueError(f"ops_spot_detector='spotnet' cannot run: {reason}")
    return name


def _spotnet_peaks(stack: np.ndarray, detect=None) -> np.ndarray:
    """SpotNet's read positions for one aligned field, as whole pixels.

    SpotNet sees one image: each cycle's brightest base channel, scaled by
    its own 99.9th percentile so no cycle outweighs the rest, averaged over
    the cycles. A read is bright in some channel in every cycle, so it is
    bright in that image, which is the same thing the native score looks
    for across the stack.

    :param stack: ``cycles x channels x H x W``, aligned.
    :param detect: :func:`spacr._segmentation_backends._detect_spots`, or a
        stand-in for tests.
    :returns: ``N x 2`` integer ``(y, x)``, unique and inside the field.
    """
    if detect is None:
        from ._segmentation_backends import _detect_spots as detect
    brightest = stack.max(axis=1).astype(np.float32)
    scale = np.percentile(brightest.reshape(len(brightest), -1), 99.9, axis=1)
    scale[~(scale > 0)] = 1.0
    image = (brightest / scale[:, None, None]).mean(axis=0)
    spots = np.asarray(detect(image, threshold=_SPOTNET_THRESHOLD), float)
    spots = spots.reshape(-1, 2)
    if not len(spots):
        return np.zeros((0, 2), dtype=np.int64)
    peaks = np.rint(spots).astype(np.int64)
    peaks[:, 0] = np.clip(peaks[:, 0], 0, image.shape[0] - 1)
    peaks[:, 1] = np.clip(peaks[:, 1], 0, image.shape[1] - 1)
    return np.unique(peaks, axis=0)


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
        whether each object's nearest tile is this one -- plus ``gpu``,
        ``threshold``, ``footprint``, ``store_reads`` and
        ``spot_detector`` (``native`` or ``spotnet``; SpotNet's positions
        replace the native score's peaks and its threshold, and everything
        after them -- the margin, the bases, the calls and the attribution
        -- is the same code either way).
    :returns: the field's reads attributed to the objects it owns, and its
        counts. With ``store_reads`` it also returns each owned read's
        position in this tile's frame, its per-cycle margin and the
        intensities the call was made on, which is what ``ops_reads`` holds.

    Reads are attributed to every nearby object, so a read in the overlap
    with the neighbouring tile is not given to the wrong nucleus merely
    because the right one belongs to the neighbour; only the reads whose
    owner this tile owns are returned, so no read is counted twice.

    THE NUMBERS COME IN THROUGH THE TASK, not off the module. Fields decode
    in SPAWNED processes, which re-import this module and get the shipped
    constants back however the parent was configured; a threshold set in the
    parent and read here would apply on one worker and not on eight.
    """
    from scipy import ndimage

    from .ops_cycles import align_field
    from .ops_sbs import (attribute_reads, call_reads, called_bases,
                          estimate_read_locations, extract_bases, find_peaks)

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
    detector = str(task.get("spot_detector") or "native")
    spotnet = _spotnet_peaks(stack) if detector == "spotnet" else None
    filtered = np.empty_like(stack)
    for c in range(stack.shape[0]):
        for k in range(stack.shape[1]):
            filtered[c, k] = np.clip(
                -ndimage.gaussian_laplace(stack[c, k], 1.0), 0, None)
    del stack
    shifts = [abs(v) for pair in field.cycle_shifts.values() for v in pair]
    shifts += [abs(v) for per in field.channel_shifts.values()
               for pair in per for v in pair]
    margin = 5 + max(shifts + [0])
    height, width = filtered.shape[-2:]
    if spotnet is not None:
        peaks = spotnet
        keep = np.ones(len(peaks), dtype=bool)
    else:
        score = estimate_read_locations(filtered)
        peaks = find_peaks(score, min_distance=2, gpu=gpu)
        strength = score - ndimage.minimum_filter(score, size=5)
        keep = (strength[peaks[:, 0], peaks[:, 1]] > float(
            task.get("threshold", _THRESHOLD_READS))) if peaks.size else None
    if peaks.size:
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
    store_reads = bool(task.get("store_reads", False))
    if store_reads:
        intensities, calls, margins = called_bases(values, bases=_BASES,
                                                   gpu=gpu)
        worst = np.where(np.isnan(margins), np.inf, margins).min(axis=1) \
            if len(calls) else np.zeros(0, np.float32)
        quality = np.where(np.isfinite(worst), worst, 0.0).astype(np.float32)
    else:
        intensities = margins = None
        calls, quality = call_reads(values, bases=_BASES, gpu=gpu)
    owner, ambiguous = attribute_reads(
        peaks.astype(float), task["centroids"], task["areas"],
        footprint=float(task.get("footprint", _FOOTPRINT)), tie=_TIE)
    has_owner = owner >= 0
    owned = np.zeros(len(peaks), dtype=bool)
    owned[has_owner] = np.asarray(task["owned"], dtype=bool)[owner[has_owner]]
    rows = np.flatnonzero(owned)
    ids = np.asarray(task["ids"], dtype=np.int64)[owner[rows]]
    ticks["call_attribute"] = time.perf_counter() - tick
    result = {
        **base, "n_cycles": len(field.kept), "spots": int(len(peaks)),
        "attributed": int(has_owner.sum()), "ambiguous": int(ambiguous.sum()),
        "exact": int(sum(code in _LIBRARY for code in calls)) if _LIBRARY else None,
        "ids": ids, "calls": [calls[i] for i in rows],
        "quality": quality[rows].astype(np.float32),
        "max_channel_shift": int(max([abs(v) for per in field.channel_shifts.values()
                                      for pair in per for v in pair] + [0])),
        "seconds": {k: round(v, 2) for k, v in ticks.items()},
    }
    if store_reads:
        result["peaks"] = peaks[rows].astype(np.float32)
        result["margins"] = (margins[rows].astype(np.float32)
                             if margins is not None and len(margins)
                             else np.zeros((len(rows), len(cycles)), np.float32))
        result["intensities"] = (
            intensities[rows].astype(np.float32)
            if intensities is not None and len(intensities)
            else np.zeros((len(rows), len(cycles), len(_BASES)), np.float32))
    return result


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


#: The ``ops_reads`` columns, in order. Four intensities follow, one per base
#: -- named by BASE and not by channel, because the channel that carries a
#: base is an acquisition's choice (``ops_base_channels``) and the letter is
#: not, so a table written on two plates stays one table.
_READS_COLUMNS = ("plate", "well", "read_id", "object_id", "site", "y", "x",
                  "cycle", "base", "quality")


def _reads_frame(plate: str, well: str, placements, decoded, cycles, bases):
    """The ``ops_reads`` rows of one well: one per read PER CYCLE.

    THE CONTRACT HAD NO READ ID, which is why this table went unwritten
    (372 PART 14-L, V15): ``object_id, cycle, intensity, base, quality``
    repeats for every read of a nucleus, and a nucleus has several -- the
    validated well averaged about six -- so those columns name no row.
    The id added here is the well's reads in raster order on their
    WELL-FRAME position, which is how :func:`spacr.ops_objects.number`
    numbers objects and for the same reason: the id is a join key, so two
    runs over the same pixels have to produce the same numbers whatever
    order the fields came back in.

    :param plate: the plate name.
    :param well: the well name.
    :param placements: ``{site: (y, x)}``, to lift a read out of its tile's
        frame into the well's.
    :param decoded: the fields that decoded, carrying ``peaks``, ``margins``
        and ``intensities`` from :func:`_decode_field`.
    :param cycles: the cycle numbers, in the order the barcode spells them.
    :param bases: the letter each channel carries.
    :returns: the DataFrame, empty of rows when nothing was stored.
    """
    import pandas as pd

    columns = list(_READS_COLUMNS) + [f"intensity_{base}" for base in bases]
    parts = [r for r in decoded
             if r.get("peaks") is not None and len(r["peaks"])]
    if not parts:
        return pd.DataFrame({name: [] for name in columns})

    count = len(cycles)
    y = np.concatenate([r["peaks"][:, 0] + placements[r["site"]][0]
                        for r in parts])
    x = np.concatenate([r["peaks"][:, 1] + placements[r["site"]][1]
                        for r in parts])
    owner = np.concatenate([np.asarray(r["ids"], np.int64) for r in parts])
    site = np.concatenate([np.full(len(r["peaks"]), r["site"], np.int32)
                           for r in parts])
    margins = np.concatenate([r["margins"] for r in parts])
    values = np.concatenate([r["intensities"] for r in parts])
    letters = np.frombuffer(
        "".join(code for r in parts for code in r["calls"]).encode("ascii"),
        dtype=np.uint8).reshape(-1, count)

    order = np.lexsort((site, x, y))
    rows = len(order)
    data = {
        "plate": np.full(rows * count, plate, dtype=object),
        "well": np.full(rows * count, well, dtype=object),
        "read_id": np.repeat(np.arange(1, rows + 1, dtype=np.int64), count),
        "object_id": np.repeat(owner[order], count),
        "site": np.repeat(site[order], count),
        "y": np.repeat(y[order].astype(np.float32), count),
        "x": np.repeat(x[order].astype(np.float32), count),
        "cycle": np.tile(np.asarray(cycles, np.int32), rows),
        "base": np.frombuffer(np.ascontiguousarray(letters[order]).tobytes(),
                              dtype="S1").astype("U1"),
        "quality": margins[order].reshape(-1).astype(np.float32),
    }
    flat = values[order].reshape(-1, values.shape[-1])
    for index, base in enumerate(bases):
        data[f"intensity_{base}"] = flat[:, index].astype(np.float32)
    return pd.DataFrame(data, columns=columns)


def _decode(db: str, plate: str, well: str, cycle_files, reference: int,
            settings: Mapping[str, Any], gpu: bool,
            library: Sequence[str]) -> Dict[str, Any]:
    """Decode every field of the well and store ``ops_barcodes``.

    :param db: the measurements database.
    :param plate: the plate name.
    :param well: the well name.
    :param cycle_files: ``cycle -> site -> channel -> path`` for this well.
    :param reference: the cycle the objects' frame was stitched on.
    :param settings: read for ``n_workers``, ``ops_base_channels``,
        ``ops_read_threshold``, ``ops_footprint``, ``ops_store_reads`` and
        ``ops_spot_detector``.
    :param gpu: let the decode use the card when it runs in this process.
    :param library: the guide barcodes, possibly empty.
    :returns: the decode report. Its ``ops_barcodes_rows`` and
        ``ops_reads_rows`` count this well's rows only, the number a
        ``plate``/``well`` filter on the stored table returns;
        ``ops_reads_rows`` is None when reads were not stored.
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

    channels = _base_channels(settings)
    threshold = float(_setting_number(settings.get("ops_read_threshold"),
                                      "ops_read_threshold",
                                      _THRESHOLD_READS))
    footprint = float(_setting_number(settings.get("ops_footprint"),
                                      "ops_footprint", _FOOTPRINT))
    store_reads = bool(settings.get("ops_store_reads", False))
    detector = _spot_detector(settings)
    tasks = []
    for site in order:
        top, left = placements[site]
        near = np.flatnonzero((oy > top - 30) & (oy < top + shape[0] + 30)
                              & (ox > left - 30) & (ox < left + shape[1] + 30))
        planes = {}
        for cycle in cycles:
            sources = _plane_sources(cycle_files[cycle].get(site, {}))
            planes[cycle] = [sources.get(name) for name in channels]
        tasks.append({
            "site": site, "planes": planes, "cycles": cycles,
            "reference": reference,
            "centroids": np.column_stack([oy[near] - top, ox[near] - left]),
            "areas": areas[near], "ids": ids[near],
            "owned": owner_site[near] == site,
            "threshold": threshold, "footprint": footprint,
            "store_reads": store_reads, "spot_detector": detector,
        })

    library_set = frozenset(library)
    workers = max(1, min(int(settings.get("n_workers") or 1),
                         _DECODE_WORKERS_CAP, len(tasks)))
    if detector == "spotnet" and workers > 1:
        _say(f"{well} decode: SpotNet runs in one worker of its own, so the "
             f"fields decode one at a time rather than {workers} at once")
        workers = 1
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
    _replace_well_rows(db, "ops_barcodes", frame, plate, well)
    reads_rows = None
    if store_reads:
        tick = time.perf_counter()
        reads = _reads_frame(plate, well, placements, decoded, cycles, _BASES)
        table_rows = _replace_well_rows(db, "ops_reads", reads, plate, well)
        reads_rows = int(len(reads))
        _say(f"{well} decode: stored {reads_rows} ops_reads rows for this well, "
             f"{table_rows} in the table "
             f"({round(time.perf_counter() - tick, 1)} s)")

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
        "ops_barcodes_rows": int(len(frame)), "ops_reads_rows": reads_rows,
        "read_threshold": threshold, "footprint": footprint,
        "spot_detector": detector,
        "base_channels": list(channels), "workers": workers,
        "field_seconds": dict(sum((Counter(r["seconds"]) for r in results
                                   if "seconds" in r), Counter())),
        "total_seconds": round(time.perf_counter() - started, 1),
    }
    if report["containment_below_gate"]:
        _say(f"{well} decode: ONLY {containment:.1%} OF SPOTS LIE WITHIN "
             f"{footprint:g} PX OF A NUCLEUS, below the {_MIN_CONTAINMENT:.0%} "
             "gate. The reads are not where the objects are; check the "
             "segmentation and the footprint before trusting these barcodes.")
    if report["unreadable"]:
        first_path, first_reason = report["unreadable"][0]
        _say(f"{well} decode: WARNING {len(report['unreadable'])} source "
             "file(s) could not be read and were left out; each costs its "
             "cycle in that one field, not the well. First: "
             f"{os.path.basename(str(first_path))} ({first_reason}). All are "
             "listed under the report's decode.unreadable.")
    _say(f"{well} decode: {len(assigned)} objects assigned from {spots} spots "
         f"in {len(decoded)} fields, {report['total_seconds']} s")
    return report


def run_ops(settings: Mapping[str, Any], *,
            wells: Optional[Sequence[str]] = None,
            phases: Sequence[str] = _PHASES,
            library=None) -> Dict[str, Any]:
    """Stitch, place, segment and decode the wells of one sequencing acquisition.

    :param settings: the OPS settings. Read here: ``genotype_source``, the
        folder of sequencing tiles, searched recursively; ``phenotype_source``,
        the folder of the high-magnification phenotype acquisition, which the
        ``phenotype`` phase places on the stitched well and which an empty
        value skips; ``dst_root``, where ``measurements.db`` and the per-well
        reports go, the source folder when empty; ``plate``, the source
        folder's name when empty; ``ops_gpu``; ``n_workers``, how many fields
        decode at once; ``cellpose_model`` and ``cellpose_diameter``;
        ``ops_library``, a guide library CSV the ``library`` keyword
        overrides; and the five measured numbers ``ops_base_channels``,
        ``ops_read_threshold``, ``ops_raster_overlap``,
        ``ops_window_overlap`` and ``ops_footprint``, each of which falls
        back to the value this plate was validated at. ``ops_store_reads``
        writes ``ops_reads``.

        EITHER FOLDER MAY BE A PARENT OF BOTH HALVES, and each setting
        still reads its own: ``genotype_source`` indexes only names that
        carry a cycle and ``phenotype_source`` only names that do not,
        falling back to cycled names for a phenotype acquisition that
        carries them. Pointing both at ``20200202_6W-LaC024A`` therefore
        runs the plate, rather than filing 1,281 phenotype fields and 333
        sequencing ones as one acquisition -- see :func:`_index_tiles`.
    :param wells: which wells to run; every well found when None.
    :param phases: any of ``"stitch"``, ``"phenotype"``, ``"objects"`` and
        ``"decode"``, run in that order. A phase left out reads what it needs
        from the database, so a well can be stitched once and decoded again.
    :param library: the guide barcodes, as a sequence or the path of a CSV
        with a ``prefix``, ``barcode`` or ``sequence`` column. When given,
        each barcode is also mapped to its closest guide. Overrides the
        ``ops_library`` setting.
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
    if "decode" in phases:
        _spot_detector(settings)
    barcodes = _load_library(library if library is not None
                             else settings.get("ops_library") or None)

    index = _index_tiles(root, "cycled")
    if not index:
        raise ValueError(
            f"no tile under {root} is named like 10X_c1_A1_DAPI-CY3-A594-CY5-"
            "CY7_Site-0.tif: magnification, cycle, well, channels, site")
    _one_acquisition(index, "genotype_source", root)
    chosen = [str(w).upper() for w in wells] if wells else sorted(index)
    missing = [well for well in chosen if well not in index]
    if missing:
        raise ValueError(f"no tiles for wells {missing}; found {sorted(index)}")

    phenotype_root = settings.get("phenotype_source")
    phenotype_index: Dict[str, Any] = {}
    if "phenotype" in phases and phenotype_root:
        if not os.path.isdir(str(phenotype_root)):
            raise ValueError(
                f"phenotype_source must be the folder of phenotype tiles; "
                f"got {phenotype_root!r}")
        phenotype_index = (_index_tiles(str(phenotype_root), "uncycled")
                           or _index_tiles(str(phenotype_root), "cycled"))
        if not phenotype_index:
            raise ValueError(
                f"no tile under {phenotype_root} is named like "
                "20X_DAPI-GFP-A594-AF750_A1_DAPI-GFP_Site-0.tif "
                "(magnification, channel set, well, this file's channels, "
                "site) or like 20X_c1_A1_DAPI-GFP_Site-0.tif")
        _one_acquisition(phenotype_index, "phenotype_source",
                         str(phenotype_root))

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
            report["stitch"] = _stitch(db, plate, well, cycle_files,
                                       reference, settings, gpu)
        if "phenotype" in phases:
            if not phenotype_root:
                report["phenotype"] = {
                    "skipped": "no phenotype_source, so nothing to place"}
            elif well not in phenotype_index:
                report["phenotype"] = {
                    "skipped": f"the phenotype acquisition has no well {well}; "
                               f"it holds {sorted(phenotype_index)}"}
            else:
                report["phenotype"] = _phenotype(
                    db, plate, well, cycle_files, reference,
                    phenotype_index[well],
                    (_tile_magnification(_any_tile(cycle_files)),
                     _tile_magnification(_any_tile(phenotype_index[well]))),
                    settings)
        if "objects" in phases:
            report["objects"] = _objects(db, plate, well, cycle_files, reference,
                                         settings, gpu)
        if "decode" in phases:
            report["decode"] = _decode(db, plate, well, cycle_files, reference,
                                       settings, gpu, barcodes)
        report["seconds"] = round(time.perf_counter() - started, 1)
        report["peak_rss_gb"] = _peak_rss_gb()
        folder = os.path.join(destination, well)
        os.makedirs(folder, exist_ok=True)
        with open(os.path.join(folder, "ops_report.json"), "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=1, default=str)
        out["wells"][well] = report
    return out


_ST_CONTROL_PREFIXES = ("NegControl", "BLANK", "Blank", "Unassigned",
                        "Deprecated", "Intergenic", "antisense_", "NegPrb")

_ST_WIDE_GENE_LIMIT = 1500

_ST_OBJECT_TYPES = ("cell", "nucleus", "pathogen", "vacuole")

_ST_H5PY_MESSAGE = (
    "Reading a 10x feature-barcode matrix (.h5) needs h5py, which is not "
    "installed in this environment (missing module: {module}).\n\n"
    "Install it with:\n\n    python -m pip install h5py")

_ST_PYARROW_MESSAGE = (
    "Reading Xenium or Visium HD parquet files needs pyarrow, which is not "
    "installed in this environment (missing module: {module}).\n\n"
    "Install it with:\n\n    python -m pip install pyarrow")


def _st_decode(values) -> List[str]:
    """Strings from an HDF5 or Arrow column that may hold bytes.

    :param values: an iterable of ``str`` or ``bytes``.
    :returns: the values as ``str``.
    """
    return [v.decode("utf-8") if isinstance(v, bytes) else str(v)
            for v in values]


def _st_read_10x_h5(path: str):
    """Read a 10x Genomics feature-barcode matrix ``.h5``.

    Only the Gene Expression features are kept, so antibody-capture or
    control rows do not appear as genes.

    :param path: the ``filtered_feature_bc_matrix.h5`` or
        ``cell_feature_matrix.h5`` file.
    :returns: ``(matrix, barcodes, genes)``, where ``matrix`` is a
        barcodes-by-genes :class:`scipy.sparse.csr_matrix`.
    """
    from scipy import sparse

    from .tabular import _require_optional

    h5py = _require_optional("h5py", _ST_H5PY_MESSAGE)
    with h5py.File(path, "r") as handle:
        group = handle["matrix"]
        shape = tuple(int(v) for v in group["shape"][:])
        matrix = sparse.csc_matrix(
            (group["data"][:], group["indices"][:], group["indptr"][:]),
            shape=shape).T.tocsr()
        barcodes = _st_decode(group["barcodes"][:])
        features = group["features"]
        genes = _st_decode(features["name"][:])
        kinds = (_st_decode(features["feature_type"][:])
                 if "feature_type" in features else
                 ["Gene Expression"] * len(genes))
    keep = np.array([kind == "Gene Expression"
                     and not str(gene).startswith(_ST_CONTROL_PREFIXES)
                     for kind, gene in zip(kinds, genes)], dtype=bool)
    return (matrix[:, np.flatnonzero(keep)].tocsr(), np.asarray(barcodes),
            np.asarray(genes, dtype=object)[keep])


def _st_platform(folder: str) -> str:
    """Which 10x platform wrote ``folder``.

    :param folder: a Space Ranger ``outs`` folder or a Xenium output bundle.
    :returns: ``"xenium"``, ``"visium_hd"`` or ``"visium"``.
    :raises FileNotFoundError: when neither layout is present.
    """
    if (os.path.exists(os.path.join(folder, "transcripts.parquet"))
            or os.path.exists(os.path.join(folder, "experiment.xenium"))):
        return "xenium"
    if os.path.isdir(os.path.join(folder, "binned_outputs")):
        return "visium_hd"
    if os.path.isdir(os.path.join(folder, "spatial")):
        return "visium"
    raise FileNotFoundError(
        f"{folder} holds neither a Xenium bundle (transcripts.parquet, "
        "experiment.xenium) nor Space Ranger output (spatial/, "
        "binned_outputs/).")


def _st_first(folder: str, names: Sequence[str]) -> Optional[str]:
    """The first of ``names`` that exists in ``folder``.

    :param folder: the folder searched.
    :param names: file names in order of preference.
    :returns: the path, or None.
    """
    for name in names:
        path = os.path.join(folder, name)
        if os.path.exists(path):
            return path
    return None


def _st_read_visium(folder: str, bin_um: int = 8) -> Dict[str, Any]:
    """Read Space Ranger output for Visium or one Visium HD bin size.

    :param folder: the ``outs`` folder. For Visium HD, the bin folder
        ``binned_outputs/square_<bin>um`` under it is read.
    :param bin_um: the Visium HD bin size in micrometres (2, 8 or 16).
    :returns: the bundle: ``platform``, ``genes``, ``counts`` (spots by
        genes), ``obs`` (one row per spot with its full-resolution pixel
        position ``x_full``, ``y_full``), ``scalefactors``, ``images``,
        ``spot_diameter_full``, ``spot_shape`` and ``microns_per_pixel``.
    """
    import json

    import pandas as pd

    from .tabular import read_table

    platform = "visium"
    if os.path.isdir(os.path.join(folder, "binned_outputs")):
        platform = "visium_hd"
        folder = os.path.join(folder, "binned_outputs",
                              f"square_{int(bin_um):03d}um")
    spatial = os.path.join(folder, "spatial")
    h5 = _st_first(folder, ("filtered_feature_bc_matrix.h5",))
    if h5 is None:
        found = [n for n in sorted(os.listdir(folder))
                 if n.endswith("filtered_feature_bc_matrix.h5")]
        h5 = os.path.join(folder, found[0]) if found else None
    if h5 is None:
        raise FileNotFoundError(
            f"No filtered_feature_bc_matrix.h5 in {folder}.")
    counts, barcodes, genes = _st_read_10x_h5(h5)
    columns = ["barcode", "in_tissue", "array_row", "array_col",
               "pxl_row_in_fullres", "pxl_col_in_fullres"]
    positions = _st_first(spatial, ("tissue_positions.parquet",
                                    "tissue_positions.csv"))
    if positions is not None and positions.endswith(".parquet"):
        from .tabular import _require_optional

        _require_optional("pyarrow", _ST_PYARROW_MESSAGE)
        table = read_table(positions, canonicalise=False)
    elif positions is not None:
        table = read_table(positions, canonicalise=False)
    else:
        legacy = _st_first(spatial, ("tissue_positions_list.csv",))
        if legacy is None:
            raise FileNotFoundError(f"No tissue_positions file in {spatial}.")
        table = read_table(legacy, canonicalise=False, header=None,
                           names=columns)
    table = table.set_index("barcode")
    missing = [b for b in barcodes if b not in table.index]
    if missing:
        raise ValueError(
            f"{len(missing)} barcodes of the matrix have no position, "
            f"for example {missing[0]}.")
    table = table.loc[list(barcodes)]
    with open(os.path.join(spatial, "scalefactors_json.json"),
              encoding="utf-8") as handle:
        scalefactors = json.load(handle)
    obs = pd.DataFrame({
        "barcode": barcodes,
        "in_tissue": table["in_tissue"].to_numpy().astype(int),
        "array_row": table["array_row"].to_numpy(),
        "array_col": table["array_col"].to_numpy(),
        "x_full": table["pxl_col_in_fullres"].to_numpy().astype(float),
        "y_full": table["pxl_row_in_fullres"].to_numpy().astype(float),
    })
    images = {}
    for key, name in (("hires", "tissue_hires_image.png"),
                      ("lowres", "tissue_lowres_image.png")):
        path = os.path.join(spatial, name)
        if os.path.exists(path):
            images[key] = path
    outs = folder if platform == "visium" else os.path.dirname(
        os.path.dirname(folder))
    full = [n for n in sorted(os.listdir(outs))
            if n.lower().endswith((".tif", ".tiff", ".btf"))]
    if full:
        images["full"] = os.path.join(outs, full[0])
    diameter = float(scalefactors.get("spot_diameter_fullres", 0.0))
    microns = scalefactors.get("microns_per_pixel")
    if microns is None and diameter > 0 and platform == "visium":
        microns = 55.0 / diameter
    return {
        "platform": platform, "folder": folder, "genes": genes,
        "counts": counts, "obs": obs, "scalefactors": scalefactors,
        "images": images, "spot_diameter_full": diameter,
        "spot_shape": "square" if platform == "visium_hd" else "circle",
        "microns_per_pixel": float(microns) if microns else None,
    }


def _st_read_xenium(folder: str, min_qv: float = 20.0) -> Dict[str, Any]:
    """Read a Xenium output bundle.

    Transcripts below ``min_qv`` and every control probe or codeword are
    dropped, as Xenium's own cell-feature matrix drops them.

    :param folder: the Xenium output bundle.
    :param min_qv: the lowest Phred-scaled quality value kept.
    :returns: the bundle: ``platform``, ``genes``, ``transcripts`` (one row
        per transcript with ``gene``, ``x_um``, ``y_um``, ``z_um``, ``qv``
        and Xenium's own ``xenium_cell_id``), ``cells``, ``pixel_size``
        (micrometres per pixel of the full-resolution morphology image) and
        ``images``.
    """
    import json

    import pandas as pd

    from .tabular import _require_optional, read_table

    _require_optional("pyarrow", _ST_PYARROW_MESSAGE)
    pixel_size = 0.2125
    meta = os.path.join(folder, "experiment.xenium")
    if os.path.exists(meta):
        with open(meta, encoding="utf-8") as handle:
            pixel_size = float(json.load(handle).get("pixel_size",
                                                     pixel_size))
    raw = read_table(os.path.join(folder, "transcripts.parquet"),
                     canonicalise=False,
                     columns=["cell_id", "feature_name", "x_location",
                              "y_location", "z_location", "qv"])
    names = raw["feature_name"]
    if len(names) and isinstance(names.iloc[0], bytes):
        names = names.str.decode("utf-8")
    names = names.astype(str)
    keep = (raw["qv"].to_numpy() >= float(min_qv)) & ~names.str.startswith(
        _ST_CONTROL_PREFIXES).to_numpy()
    genes = np.array(sorted(names[keep].unique()), dtype=object)
    transcripts = pd.DataFrame({
        "gene": pd.Categorical(names[keep], categories=genes),
        "x_um": raw["x_location"].to_numpy()[keep].astype(float),
        "y_um": raw["y_location"].to_numpy()[keep].astype(float),
        "z_um": raw["z_location"].to_numpy()[keep].astype(float),
        "qv": raw["qv"].to_numpy()[keep].astype(float),
        "xenium_cell_id": raw["cell_id"].astype(str).to_numpy()[keep],
    })
    cells_path = os.path.join(folder, "cells.parquet")
    cells = (read_table(cells_path, canonicalise=False)
             if os.path.exists(cells_path) else None)
    images = {}
    focus = os.path.join(folder, "morphology_focus")
    image = (_st_first(focus, ("morphology_focus_0000.ome.tif",))
             if os.path.isdir(focus) else None)
    image = image or _st_first(folder, ("morphology_focus.ome.tif",
                                        "morphology_mip.ome.tif",
                                        "morphology.ome.tif"))
    if image:
        images["morphology"] = image
    return {"platform": "xenium", "folder": folder, "genes": genes,
            "transcripts": transcripts, "cells": cells,
            "pixel_size": pixel_size, "images": images,
            "dropped": int((~keep).sum())}


def _st_read_bundle(folder: str, platform: str = "auto", *,
                    bin_um: int = 8, min_qv: float = 20.0) -> Dict[str, Any]:
    """Read a Visium, Visium HD or Xenium output folder.

    :param folder: the platform's output folder.
    :param platform: ``"auto"``, ``"visium"``, ``"visium_hd"`` or
        ``"xenium"``.
    :param bin_um: the Visium HD bin size in micrometres.
    :param min_qv: the lowest Xenium transcript quality kept.
    :returns: the bundle described by :func:`_st_read_visium` or
        :func:`_st_read_xenium`.
    """
    folder = os.path.expanduser(str(folder))
    platform = _st_platform(folder) if platform in ("", "auto") else platform
    if platform == "xenium":
        return _st_read_xenium(folder, min_qv=min_qv)
    return _st_read_visium(folder, bin_um=bin_um)


def _st_load_image(path: str, level: int = 0) -> np.ndarray:
    """Read one image plane: a PNG, JPEG or a level of a (pyramidal) TIFF.

    A z-stack is maximum-projected; colour images keep their channels.

    :param path: the image file.
    :param level: the pyramid level of an OME-TIFF, 0 for full resolution.
    :returns: a 2-D or ``(Y, X, 3|4)`` array.
    """
    lower = path.lower()
    if lower.endswith((".tif", ".tiff", ".btf")):
        import tifffile

        with tifffile.TiffFile(path) as handle:
            series = handle.series[0]
            levels = getattr(series, "levels", None) or [series]
            chosen = levels[min(int(level), len(levels) - 1)]
            array = chosen.asarray()
            axes = chosen.axes
        while array.ndim > 2 and axes[0] not in "YX" and not (
                array.ndim == 3 and axes[-1] in "SC" and array.shape[-1] in (3, 4)):
            array = array.max(axis=0)
            axes = axes[1:]
        return array
    from matplotlib import image as mpimage

    return mpimage.imread(path)


def _st_gray(image: np.ndarray) -> np.ndarray:
    """A float grayscale copy of ``image``, stretched to 0..1.

    :param image: a 2-D or colour image.
    :returns: the 2-D float image.
    """
    array = np.asarray(image, dtype=float)
    if array.ndim == 3:
        array = array[..., :3].mean(axis=-1)
    low, high = np.percentile(array, (1, 99.5))
    return np.clip((array - low) / max(high - low, 1e-9), 0, 1)


def _st_platform_xy(bundle: Mapping[str, Any], image: str = "",
                    level: int = 0) -> Tuple[np.ndarray, float, float]:
    """The sequencing coordinates in the pixel frame of a platform image.

    Visium spots are scaled from full resolution by the scale factor of the
    chosen image; Xenium transcripts are divided by the morphology image's
    pixel size at the chosen pyramid level.

    :param bundle: a bundle from :func:`_st_read_bundle`.
    :param image: ``"hires"``, ``"lowres"`` or ``"full"`` for Visium;
        ignored for Xenium. Empty picks ``"hires"``.
    :param level: the OME-TIFF pyramid level for Xenium or a full-resolution
        Visium image.
    :returns: ``(xy, radius, microns_per_pixel)``: an ``(N, 2)`` array of x
        (column) and y (row), the spot radius in pixels (0 for transcripts)
        and the micrometres one pixel spans (0 when unknown).
    """
    if bundle["platform"] == "xenium":
        size = float(bundle["pixel_size"]) * (2 ** int(level))
        frame = bundle["transcripts"]
        xy = np.column_stack([frame["x_um"].to_numpy() / size,
                              frame["y_um"].to_numpy() / size])
        return xy, 0.0, size
    image = image or "hires"
    factors = bundle["scalefactors"]
    scale = {"hires": factors.get("tissue_hires_scalef", 1.0),
             "lowres": factors.get("tissue_lowres_scalef", 1.0)}.get(
                 image, 1.0 / (2 ** int(level)))
    obs = bundle["obs"]
    xy = np.column_stack([obs["x_full"].to_numpy(),
                          obs["y_full"].to_numpy()]) * float(scale)
    radius = 0.5 * float(bundle["spot_diameter_full"]) * float(scale)
    microns = bundle.get("microns_per_pixel") or 0.0
    return xy, radius, (float(microns) / float(scale) if microns else 0.0)


def _st_fit_affine(source, target) -> Tuple[np.ndarray, float]:
    """The least-squares affine transform taking ``source`` onto ``target``.

    :param source: ``(N, 2)`` landmark positions (x, y) in the platform
        image, N of at least 3.
    :param target: the same landmarks in the user's image.
    :returns: ``(matrix, rms)``: a 3x3 homogeneous matrix and the root mean
        square landmark residual in target pixels.
    :raises ValueError: with fewer than three landmark pairs.
    """
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)
    if len(source) < 3 or source.shape != target.shape:
        raise ValueError("An affine fit needs at least three landmark pairs.")
    design = np.column_stack([source, np.ones(len(source))])
    solution, *_ = np.linalg.lstsq(design, target, rcond=None)
    matrix = np.eye(3)
    matrix[:2, :] = solution.T
    residual = _st_apply_affine(matrix, source) - target
    return matrix, float(np.sqrt((residual ** 2).sum(axis=1).mean()))


def _st_apply_affine(matrix, xy) -> np.ndarray:
    """Map ``(N, 2)`` points through a 3x3 homogeneous matrix.

    :param matrix: the affine matrix.
    :param xy: the points, x then y.
    :returns: the mapped points.
    """
    xy = np.asarray(xy, dtype=float)
    return xy @ np.asarray(matrix)[:2, :2].T + np.asarray(matrix)[:2, 2]


def _st_read_landmarks(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Landmark pairs from a table with ``source_x``, ``source_y``,
    ``target_x`` and ``target_y`` columns.

    Source is the platform image, target the user's image, both in pixels.

    :param path: the landmark table (CSV, TSV, Excel or Parquet).
    :returns: ``(source, target)`` as ``(N, 2)`` arrays.
    """
    from .tabular import read_table

    frame = read_table(path, canonicalise=False)
    wanted = ["source_x", "source_y", "target_x", "target_y"]
    absent = [c for c in wanted if c not in frame.columns]
    if absent:
        raise ValueError(f"{path} lacks the landmark columns {absent}.")
    values = frame[wanted].to_numpy(dtype=float)
    return values[:, :2], values[:, 2:]


def _st_register_intensity(moving, fixed, *, max_side: int = 1024
                           ) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Register the platform image onto the user's image by its content.

    ORB keypoints matched in both images and a RANSAC affine fit; when too
    few keypoints agree, phase correlation gives a translation at a common
    scale instead.

    :param moving: the platform image (the frame the coordinates are in).
    :param fixed: the user's image of the same section.
    :param max_side: both images are shrunk so their longer side is at most
        this many pixels before matching.
    :returns: ``(matrix, info)``: the 3x3 matrix taking platform pixels to
        user pixels, and ``method``, ``inliers`` and ``matches``.
    """
    from skimage.feature import ORB, match_descriptors
    from skimage.measure import ransac
    from skimage.transform import AffineTransform, rescale

    def shrink(image):
        gray = _st_gray(image)
        factor = min(1.0, float(max_side) / max(gray.shape))
        return (rescale(gray, factor, anti_aliasing=True)
                if factor < 1 else gray), factor

    small_moving, f_moving = shrink(moving)
    small_fixed, f_fixed = shrink(fixed)
    info: Dict[str, Any] = {"method": "orb_ransac", "inliers": 0,
                            "matches": 0}
    try:
        found = []
        for image in (small_moving, small_fixed):
            orb = ORB(n_keypoints=2000, fast_threshold=0.05)
            orb.detect_and_extract(image)
            found.append((orb.keypoints[:, ::-1], orb.descriptors))
        pairs = match_descriptors(found[0][1], found[1][1], cross_check=True,
                                  max_ratio=0.85)
        info["matches"] = int(len(pairs))
        if len(pairs) >= 6:
            src = found[0][0][pairs[:, 0]]
            dst = found[1][0][pairs[:, 1]]
            model, inliers = ransac((src, dst), AffineTransform,
                                    min_samples=3, residual_threshold=2.0,
                                    max_trials=2000,
                                    rng=np.random.default_rng(0))
            if model is not None and inliers is not None \
                    and int(inliers.sum()) >= 6:
                info["inliers"] = int(inliers.sum())
                down = np.diag([f_moving, f_moving, 1.0])
                up = np.diag([1.0 / f_fixed, 1.0 / f_fixed, 1.0])
                return up @ model.params @ down, info
    except Exception:
        LOG.debug("keypoint registration failed", exc_info=True)
    from skimage.registration import phase_cross_correlation

    factor = f_fixed / f_moving
    resized = rescale(small_moving, factor) if abs(factor - 1) > 1e-6 \
        else small_moving
    rows = max(resized.shape[0], small_fixed.shape[0])
    cols = max(resized.shape[1], small_fixed.shape[1])
    pad_a = np.zeros((rows, cols))
    pad_b = np.zeros((rows, cols))
    pad_a[:resized.shape[0], :resized.shape[1]] = resized
    pad_b[:small_fixed.shape[0], :small_fixed.shape[1]] = small_fixed
    shift, _error, _phase = phase_cross_correlation(pad_b, pad_a)
    matrix = np.eye(3)
    matrix[0, 0] = matrix[1, 1] = f_moving / f_fixed
    matrix[0, 2] = shift[1] / f_fixed
    matrix[1, 2] = shift[0] / f_fixed
    info["method"] = "phase_correlation"
    return matrix, info


def _st_load_mask(path: str) -> np.ndarray:
    """Read a label mask saved by spaCR (``.npy``) or as an image.

    :param path: the mask file.
    :returns: the 2-D integer label image.
    :raises ValueError: when the file does not hold one 2-D mask.
    """
    if path.lower().endswith(".npy"):
        mask = np.load(path)
    elif path.lower().endswith(".npz"):
        with np.load(path) as bundle:
            mask = bundle[list(bundle.keys())[0]]
    else:
        mask = _st_load_image(path)
    mask = np.squeeze(mask)
    if mask.ndim != 2:
        raise ValueError(f"{path} holds a {mask.ndim}-D array, not one mask.")
    return mask.astype(np.int64, copy=False)


def _st_points_in_mask(xy, mask) -> np.ndarray:
    """The mask label under each point, 0 outside every object.

    The point-in-mask rule the OPS reads use: a point belongs to the object
    whose pixel it falls in.

    :param xy: ``(N, 2)`` points, x (column) then y (row), in mask pixels.
    :param mask: the label image.
    :returns: an ``(N,)`` integer array of labels.
    """
    xy = np.asarray(xy, dtype=float)
    cols = np.floor(xy[:, 0]).astype(np.int64)
    rows = np.floor(xy[:, 1]).astype(np.int64)
    inside = ((rows >= 0) & (rows < mask.shape[0])
              & (cols >= 0) & (cols < mask.shape[1]))
    labels = np.zeros(len(xy), dtype=np.int64)
    labels[inside] = mask[rows[inside], cols[inside]]
    return labels


def _st_spot_coverage(xy, radius: float, mask, shape: str = "circle"):
    """Which objects each spot covers, and what fraction of the spot each
    occupies.

    :param xy: ``(N, 2)`` spot centres in mask pixels.
    :param radius: the spot radius (half the side of a square bin) in
        pixels.
    :param mask: the label image.
    :param shape: ``"circle"`` for Visium spots, ``"square"`` for Visium HD
        bins.
    :returns: a frame with ``spot`` (row index of the spot), ``object_label``,
        ``pixels`` and ``fraction`` (of the whole spot's area).
    """
    import pandas as pd

    reach = max(float(radius), 0.5)
    span = int(np.ceil(reach))
    offsets = np.arange(-span, span + 1)
    dy, dx = np.meshgrid(offsets, offsets, indexing="ij")
    if shape == "square":
        footprint = (np.abs(dx) <= reach) & (np.abs(dy) <= reach)
    else:
        footprint = dx ** 2 + dy ** 2 <= reach ** 2
    fy, fx = dy[footprint], dx[footprint]
    area = float(footprint.sum())
    spots, labels, pixels = [], [], []
    height, width = mask.shape
    for index, (x, y) in enumerate(np.asarray(xy, dtype=float)):
        rows = int(round(y)) + fy
        cols = int(round(x)) + fx
        inside = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
        if not inside.any():
            continue
        under = mask[rows[inside], cols[inside]]
        under = under[under > 0]
        if not under.size:
            continue
        found, counts = np.unique(under, return_counts=True)
        spots.extend([index] * len(found))
        labels.extend(found.tolist())
        pixels.extend(counts.tolist())
    frame = pd.DataFrame({"spot": np.asarray(spots, dtype=np.int64),
                          "object_label": np.asarray(labels, dtype=np.int64),
                          "pixels": np.asarray(pixels, dtype=np.int64)})
    frame["fraction"] = frame["pixels"] / area
    return frame


def _st_object_counts(bundle: Mapping[str, Any], xy, mask,
                      radius: float = 0.0):
    """Gene counts per mask object.

    Xenium transcripts are counted in the object they fall in. Visium spot
    counts are shared out by the fraction of the spot each object occupies,
    so an object's count is an estimate and need not be whole.

    :param bundle: a bundle from :func:`_st_read_bundle`.
    :param xy: the transcripts' or spots' positions in mask pixels.
    :param mask: the label image.
    :param radius: the spot radius in mask pixels (Visium only).
    :returns: ``(labels, matrix, coverage)``: the object labels with any
        count, an objects-by-genes :class:`scipy.sparse.csr_matrix`, and the
        spot coverage frame (None for Xenium).
    """
    from scipy import sparse

    genes = bundle["genes"]
    if bundle["platform"] == "xenium":
        labels = _st_points_in_mask(xy, mask)
        hit = labels > 0
        objects, rows = np.unique(labels[hit], return_inverse=True)
        cols = bundle["transcripts"]["gene"].cat.codes.to_numpy()[hit]
        matrix = sparse.coo_matrix(
            (np.ones(int(hit.sum()), dtype=np.float64), (rows, cols)),
            shape=(len(objects), len(genes))).tocsr()
        return objects, matrix, None
    coverage = _st_spot_coverage(xy, radius, mask,
                                 bundle.get("spot_shape", "circle"))
    objects, rows = np.unique(coverage["object_label"].to_numpy(),
                              return_inverse=True)
    weights = sparse.coo_matrix(
        (coverage["fraction"].to_numpy(), (rows, coverage["spot"].to_numpy())),
        shape=(len(objects), bundle["counts"].shape[0])).tocsr()
    matrix = (weights @ bundle["counts"].astype(np.float64)).tocsr()
    return objects, matrix, coverage


def _st_prcfo(prcf: str, labels) -> List[str]:
    """spaCR object keys for ``labels`` of the image ``prcf``.

    :param prcf: the image key, ``plate_row_column_field``.
    :param labels: object labels.
    :returns: the ``prcfo`` keys.
    """
    return [f"{prcf}_o{int(label)}" for label in labels]


def _st_replace_image_rows(db: str, table: str, frame, prcf: str) -> None:
    """Write ``frame`` into ``table``, replacing that image's earlier rows.

    :param db: the measurement database.
    :param table: the table name.
    :param frame: this image's rows, with a ``prcfo`` column.
    :param prcf: the image key whose earlier rows are replaced.
    """
    import pandas as pd

    from .tabular import database_tables, read_table, write_database

    parts = [frame]
    if os.path.exists(db) and table in database_tables(db):
        old = read_table(db, table=table, canonicalise=False, report=None)
        if "prcfo" in old.columns:
            old = old[~old["prcfo"].astype(str).str.startswith(f"{prcf}_o")]
        if len(old):
            parts.insert(0, old)
    combined = pd.concat(parts, ignore_index=True) if len(parts) > 1 \
        else frame
    write_database(combined, db, table, if_exists="replace")


def _st_write_counts(db: str, object_type: str, prcf: str, labels, genes,
                     matrix, platform: str) -> Dict[str, Any]:
    """Write per-object gene counts beside the object's measurements.

    Two tables keyed by ``prcfo``, the key of the measurement tables:
    ``<object>_expression``, one row per object with an ``expr_<gene>``
    column per gene (the most-counted genes when the panel has more than
    SQLite's column limit allows) and ``expr_total``; and, for Xenium,
    ``<object>_expression_long``, one row per object and detected gene.
    Visium counts shared out from spots touch nearly every gene of every
    object, so their long form is left to the spot coverage table and the
    platform's own matrix rather than written row by row.

    :param db: the measurement database; created if absent.
    :param object_type: ``cell``, ``nucleus``, ``pathogen`` or ``vacuole``.
    :param prcf: the image key the mask belongs to.
    :param labels: the object labels, one per matrix row.
    :param genes: the gene names, one per matrix column.
    :param matrix: the objects-by-genes counts.
    :param platform: the platform the counts came from.
    :returns: the table names and the number of objects and genes written.
    """
    import pandas as pd

    keys = _st_prcfo(prcf, labels)
    totals = np.asarray(matrix.sum(axis=0)).ravel()
    order = np.argsort(-totals, kind="stable")
    wide_genes = order[:_ST_WIDE_GENE_LIMIT]
    wide_genes = np.sort(wide_genes)
    dense = matrix[:, wide_genes].toarray()
    wide = pd.DataFrame(dense, columns=[f"expr_{genes[i]}"
                                        for i in wide_genes])
    wide.insert(0, "expr_total", np.asarray(matrix.sum(axis=1)).ravel())
    wide.insert(0, "expr_platform", platform)
    wide.insert(0, "object_label", np.asarray(labels, dtype=np.int64))
    wide.insert(0, "prcfo", keys)
    wide_table = f"{object_type}_expression"
    _st_replace_image_rows(db, wide_table, wide, prcf)
    tables = (wide_table,)
    if platform == "xenium":
        coo = matrix.tocoo()
        long = pd.DataFrame({
            "prcfo": np.asarray(keys, dtype=object)[coo.row],
            "object_label": np.asarray(labels, dtype=np.int64)[coo.row],
            "gene": np.asarray(genes, dtype=object)[coo.col],
            "count": coo.data,
        })
        tables += (f"{object_type}_expression_long",)
        _st_replace_image_rows(db, tables[1], long, prcf)
    return {"tables": tables, "objects": int(len(keys)),
            "genes": int(len(genes)), "wide_genes": int(len(wide_genes))}


def _st_object_geometry(mask, others: Mapping[str, np.ndarray],
                        microns_per_pixel: float = 0.0):
    """Centroid, infection state and distance to the nearest parasite for
    each object of ``mask``.

    :param mask: the label image of the objects described.
    :param others: ``pathogen`` and/or ``vacuole`` label images of the same
        frame; an object overlapping any of their objects is infected.
    :param microns_per_pixel: converts distances to micrometres; 0 keeps
        pixels.
    :returns: a frame indexed by object label with ``centroid_x``,
        ``centroid_y``, ``infected``, ``parasite_objects`` and
        ``distance_to_parasite`` (0 inside, in micrometres when the scale is
        known, else pixels).
    """
    import pandas as pd
    from scipy import ndimage

    labels = np.unique(mask[mask > 0])
    centres = ndimage.center_of_mass(np.ones_like(mask, dtype=np.uint8),
                                     mask, labels)
    centres = np.asarray(centres, dtype=float).reshape(-1, 2)
    frame = pd.DataFrame({"centroid_x": centres[:, 1],
                          "centroid_y": centres[:, 0]},
                         index=pd.Index(labels, name="object_label"))
    parasite = np.zeros(mask.shape, dtype=bool)
    counts = pd.Series(0, index=frame.index)
    for other in others.values():
        if other is None or other.shape != mask.shape:
            continue
        parasite |= other > 0
        both = (other > 0) & (mask > 0)
        pairs = np.unique(np.column_stack([mask[both], other[both]]), axis=0)
        if len(pairs):
            counts = counts.add(pd.Series(pairs[:, 0]).value_counts(),
                                fill_value=0)
    frame["parasite_objects"] = counts.reindex(frame.index).fillna(0).astype(int)
    frame["infected"] = frame["parasite_objects"] > 0
    if parasite.any():
        distance = ndimage.distance_transform_edt(~parasite)
        rows = np.clip(np.round(centres[:, 0]).astype(int), 0, mask.shape[0] - 1)
        cols = np.clip(np.round(centres[:, 1]).astype(int), 0, mask.shape[1] - 1)
        values = distance[rows, cols]
        frame["distance_to_parasite"] = values * (microns_per_pixel or 1.0)
    else:
        frame["distance_to_parasite"] = np.nan
    return frame


def _st_normalise(matrix) -> np.ndarray:
    """Counts scaled to the median object total, then ``log1p``.

    :param matrix: objects-by-genes counts.
    :returns: a dense float array.
    """
    dense = np.asarray(matrix.toarray() if hasattr(matrix, "toarray")
                       else matrix, dtype=float)
    totals = dense.sum(axis=1, keepdims=True)
    target = float(np.median(totals[totals > 0])) if (totals > 0).any() else 1.0
    return np.log1p(np.divide(dense * target, totals,
                              out=np.zeros_like(dense), where=totals > 0))


def _st_testable_genes(matrix, genes, *, min_objects: int = 5,
                       limit: int = 3000) -> np.ndarray:
    """Columns detected in at least ``min_objects`` objects, the most
    counted first, at most ``limit``.

    :param matrix: objects-by-genes counts.
    :param genes: gene names.
    :param min_objects: fewest objects a gene must be seen in.
    :param limit: most genes kept.
    :returns: the kept column indices, ascending.
    """
    detected = np.asarray((matrix > 0).sum(axis=0)).ravel()
    totals = np.asarray(matrix.sum(axis=0)).ravel()
    candidates = np.flatnonzero(detected >= int(min_objects))
    candidates = candidates[np.argsort(-totals[candidates], kind="stable")]
    return np.sort(candidates[:int(limit)])


def _st_compare_groups(matrix, genes, in_group, *, labels=("infected",
                                                          "uninfected"),
                       min_objects: int = 5):
    """Genes enriched in one group of objects against the rest.

    A two-sided Mann-Whitney U test per gene on library-size normalised,
    log-transformed counts, with Benjamini-Hochberg q-values.

    :param matrix: objects-by-genes counts.
    :param genes: gene names.
    :param in_group: a boolean per object, True for the first group.
    :param labels: the names of the two groups, used in the column names.
    :param min_objects: genes seen in fewer objects are not tested.
    :returns: one row per tested gene with both groups' mean normalised
        expression and detection fraction, ``log2_fold_change`` of mean
        counts per object, ``p_value`` and ``q_value``, sorted by
        ``p_value``.
    """
    import pandas as pd
    from scipy.stats import mannwhitneyu
    from statsmodels.stats.multitest import multipletests

    in_group = np.asarray(in_group, dtype=bool)
    if in_group.all() or not in_group.any():
        raise ValueError(
            f"Both groups need objects; found {int(in_group.sum())} "
            f"{labels[0]} and {int((~in_group).sum())} {labels[1]}.")
    columns = _st_testable_genes(matrix, genes, min_objects=min_objects)
    counts = matrix[:, columns].toarray().astype(float)
    normal = _st_normalise(matrix)[:, columns]
    a, b = normal[in_group], normal[~in_group]
    _u, p = mannwhitneyu(a, b, axis=0, alternative="two-sided")
    p = np.nan_to_num(np.asarray(p, dtype=float), nan=1.0)
    q = multipletests(p, method="fdr_bh")[1] if len(p) else p
    mean_a = counts[in_group].mean(axis=0)
    mean_b = counts[~in_group].mean(axis=0)
    first, second = labels
    frame = pd.DataFrame({
        "gene": np.asarray(genes, dtype=object)[columns],
        f"mean_{first}": a.mean(axis=0), f"mean_{second}": b.mean(axis=0),
        f"detected_{first}": (counts[in_group] > 0).mean(axis=0),
        f"detected_{second}": (counts[~in_group] > 0).mean(axis=0),
        "log2_fold_change": np.log2((mean_a + 0.1) / (mean_b + 0.1)),
        "p_value": p, "q_value": q,
    })
    frame = frame.sort_values("p_value", kind="stable").reset_index(drop=True)
    frame.attrs["n"] = {first: int(in_group.sum()),
                        second: int((~in_group).sum())}
    return frame


def _st_distance_trend(matrix, genes, distance, *, min_objects: int = 5):
    """How each gene's expression changes with distance to the nearest
    parasite.

    Spearman's rank correlation per gene between normalised expression and
    distance, over the objects with a finite distance, with
    Benjamini-Hochberg q-values.

    :param matrix: objects-by-genes counts.
    :param genes: gene names.
    :param distance: one distance per object.
    :param min_objects: genes seen in fewer objects are not tested.
    :returns: one row per tested gene with ``spearman_rho``, ``p_value``
        and ``q_value``, sorted by ``p_value``.
    """
    import pandas as pd
    from scipy.stats import rankdata, t as student
    from statsmodels.stats.multitest import multipletests

    distance = np.asarray(distance, dtype=float)
    finite = np.isfinite(distance)
    if finite.sum() < 4:
        raise ValueError("Fewer than four objects have a parasite distance.")
    columns = _st_testable_genes(matrix[finite], genes,
                                 min_objects=min_objects)
    normal = _st_normalise(matrix[finite])[:, columns]
    ranks = np.apply_along_axis(rankdata, 0, normal)
    reference = rankdata(distance[finite])
    ranks = ranks - ranks.mean(axis=0)
    reference = reference - reference.mean()
    denominator = np.sqrt((ranks ** 2).sum(axis=0) * (reference ** 2).sum())
    rho = np.divide(ranks.T @ reference, denominator,
                    out=np.zeros(len(columns)), where=denominator > 0)
    n = int(finite.sum())
    stat = rho * np.sqrt((n - 2) / np.clip(1 - rho ** 2, 1e-12, None))
    p = 2 * student.sf(np.abs(stat), n - 2)
    q = multipletests(p, method="fdr_bh")[1] if len(p) else p
    frame = pd.DataFrame({"gene": np.asarray(genes, dtype=object)[columns],
                          "spearman_rho": rho, "p_value": p, "q_value": q,
                          "objects": n})
    return frame.sort_values("p_value", kind="stable").reset_index(drop=True)


def _st_region_summary(matrix, genes, regions):
    """Per-region expression summary.

    :param matrix: objects-by-genes counts.
    :param genes: gene names.
    :param regions: one region name per object.
    :returns: one row per region and gene with ``objects``,
        ``mean_count`` and ``detected`` (the fraction of the region's
        objects with any count), genes never counted in a region left out.
    """
    import pandas as pd

    regions = pd.Series(np.asarray(regions)).astype(str)
    rows = []
    for region, index in regions.groupby(regions).groups.items():
        part = matrix[np.asarray(list(index))]
        means = np.asarray(part.mean(axis=0)).ravel()
        detected = np.asarray((part > 0).mean(axis=0)).ravel()
        keep = means > 0
        rows.append(pd.DataFrame({
            "region": region, "objects": int(part.shape[0]),
            "gene": np.asarray(genes, dtype=object)[keep],
            "mean_count": means[keep], "detected": detected[keep]}))
    return (pd.concat(rows, ignore_index=True) if rows
            else pd.DataFrame(columns=["region", "objects", "gene",
                                       "mean_count", "detected"]))


def _st_write_anndata(path: str, labels, genes, matrix, obs, spatial):
    """Write objects-by-genes counts as ``.h5ad``.

    :param path: the file written.
    :param labels: object keys, the ``obs`` index.
    :param genes: gene names, the ``var`` index.
    :param matrix: the counts.
    :param obs: per-object columns (measurements, infection state).
    :param spatial: ``(N, 2)`` object centroids, stored as
        ``obsm["spatial"]`` for squidpy.
    :returns: the path written.
    """
    import pandas as pd

    from .anndata_export import require_anndata

    anndata = require_anndata()
    obs = obs.copy()
    obs.index = pd.Index([str(v) for v in labels], name="prcfo")
    for column in obs.columns:
        if obs[column].dtype == object:
            obs[column] = obs[column].astype(str)
    var = pd.DataFrame(index=pd.Index([str(g) for g in genes], name="gene"))
    adata = anndata.AnnData(X=matrix.astype(np.float32), obs=obs, var=var)
    adata.obsm["spatial"] = np.asarray(spatial, dtype=float)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    adata.write_h5ad(path)
    return path


def _st_draw_overlay(figure, image, xy, values=None, *, radius: float = 0.0,
                     mask=None, title: str = "", max_side: int = 1600,
                     max_points: int = 60000) -> None:
    """Draw the sequencing coordinates over the image, coloured by a gene.

    :param figure: a matplotlib figure; cleared first.
    :param image: the image the coordinates are registered to.
    :param xy: ``(N, 2)`` points in that image's pixels.
    :param values: one value per point (a gene's count); None draws every
        point alike.
    :param radius: the spot radius in pixels; 0 draws transcripts as dots.
    :param mask: an optional label image whose object outlines are drawn.
    :param title: the axis title.
    :param max_side: the image is shrunk to at most this many pixels a side.
    :param max_points: at most this many uncoloured points are drawn.
    """
    from skimage.segmentation import find_boundaries

    figure.clear()
    axis = figure.add_subplot(111)
    step = max(1, int(np.ceil(max(image.shape[:2]) / float(max_side))))
    shown = np.asarray(image)[::step, ::step]
    if shown.ndim == 2:
        axis.imshow(_st_gray(shown), cmap="gray", interpolation="nearest")
    else:
        rgb = shown[..., :3].astype(float)
        axis.imshow(rgb / max(float(rgb.max()), 1.0), interpolation="nearest")
    if mask is not None:
        edges = find_boundaries(np.asarray(mask)[::step, ::step], mode="inner")
        overlay = np.zeros(edges.shape + (4,))
        overlay[edges] = (1.0, 1.0, 1.0, 0.6)
        axis.imshow(overlay, interpolation="nearest")
    xy = np.asarray(xy, dtype=float) / step
    size = max(2.0, (2 * radius / step) ** 2 * 0.6) if radius else 1.5
    if values is None:
        chosen = np.arange(len(xy))
        if len(chosen) > max_points:
            chosen = np.random.default_rng(0).choice(len(xy), max_points,
                                                     replace=False)
        axis.scatter(xy[chosen, 0], xy[chosen, 1], s=size, c="#e4572e",
                     linewidths=0, alpha=0.6)
    else:
        values = np.asarray(values, dtype=float)
        if radius:
            from matplotlib.collections import EllipseCollection

            width = 2.0 * radius / step
            points = EllipseCollection(
                width, width, 0.0, units="xy", offsets=xy,
                offset_transform=axis.transData, cmap="viridis",
                linewidths=0, alpha=0.55)
            points.set_array(np.log1p(values))
            axis.add_collection(points)
            figure.colorbar(points, ax=axis, fraction=0.035,
                            label="log1p(count)")
        else:
            hit = values > 0
            axis.scatter(xy[hit, 0], xy[hit, 1], s=3, c="#e4572e",
                         linewidths=0)
    axis.set_title(title, fontsize=9)
    axis.set_axis_off()
    figure.tight_layout()


def _st_register(bundle: Mapping[str, Any], request: Mapping[str, Any]):
    """Put the bundle's coordinates in the pixel frame of the analysis image.

    Without a user image the frame is the platform's own image (Visium
    ``hires``/``lowres``/``full``, Xenium morphology at a pyramid level).
    With one, the platform image is registered onto it through landmarks
    when a landmark table is given, else by image content.

    :param bundle: a bundle from :func:`_st_read_bundle`.
    :param request: ``image`` (a platform image name or a file of the user's
        own), ``level`` and ``landmarks``.
    :returns: a dict with ``xy``, ``radius``, ``microns_per_pixel``,
        ``image`` (the analysis image), ``matrix`` (3x3, identity without a
        user image) and ``registration`` (method and quality).
    """
    choice = str(request.get("image") or "")
    level = int(request.get("level") or 0)
    own = choice if choice and os.path.isfile(os.path.expanduser(choice)) \
        else ""
    if bundle["platform"] == "xenium":
        name = "morphology"
    else:
        name = choice if choice in ("hires", "lowres", "full") else "hires"
        if own:
            name = "hires" if "hires" in bundle["images"] else "lowres"
    if name not in bundle["images"]:
        raise FileNotFoundError(f"The bundle has no {name} image.")
    xy, radius, microns = _st_platform_xy(bundle, name, level)
    platform_image = _st_load_image(bundle["images"][name], level)
    result = {"xy": xy, "radius": radius, "microns_per_pixel": microns,
              "image": platform_image, "matrix": np.eye(3),
              "registration": {"method": "platform", "frame": name,
                               "level": level}}
    if not own:
        return result
    user_image = _st_load_image(os.path.expanduser(own))
    landmarks = str(request.get("landmarks") or "")
    if landmarks:
        source, target = _st_read_landmarks(os.path.expanduser(landmarks))
        matrix, rms = _st_fit_affine(source, target)
        info = {"method": "landmarks", "rms_px": rms,
                "landmarks": int(len(source))}
    else:
        matrix, info = _st_register_intensity(platform_image, user_image)
    scale = float(np.sqrt(abs(np.linalg.det(matrix[:2, :2]))))
    info.update({"frame": "user", "platform_frame": name})
    result.update({
        "xy": _st_apply_affine(matrix, xy), "radius": radius * scale,
        "microns_per_pixel": microns / scale if microns else 0.0,
        "image": user_image, "matrix": matrix, "registration": info})
    return result


def _st_gene_values(bundle: Mapping[str, Any], gene: str):
    """One value per spot or transcript for colouring the overlay.

    :param bundle: a bundle from :func:`_st_read_bundle`.
    :param gene: the gene name; empty or unknown gives None.
    :returns: the spot counts of ``gene`` (Visium) or a 0/1 flag per
        transcript (Xenium), or None.
    """
    genes = list(bundle["genes"])
    if not gene or gene not in genes:
        return None
    if bundle["platform"] == "xenium":
        return (bundle["transcripts"]["gene"].to_numpy() == gene).astype(float)
    column = genes.index(gene)
    return np.asarray(bundle["counts"][:, column].toarray()).ravel()


def _st_run(request: Mapping[str, Any], *, bundle=None,
            registered=None) -> Dict[str, Any]:
    """Read, register, assign to spaCR objects, write and analyse.

    :param request: ``folder`` (the platform output), ``platform``,
        ``bin_um``, ``min_qv``, ``image``, ``level``, ``landmarks``,
        ``masks`` (``{object type: mask file}``), ``region_mask``, ``db``
        (the measurement database), ``prcf`` (the image key; defaults to the
        first mask's file stem), ``output`` (the results folder; defaults to
        ``spatial_transcriptomics`` beside the database), ``gene`` (coloured
        in the overlay) and ``anndata`` (write ``.h5ad`` when anndata is
        installed).
    :param bundle: an already-read bundle, to skip reading it again.
    :param registered: an already-computed :func:`_st_register` result.
    :returns: a summary with the objects written per type, the analyses'
        file paths, the registration and the messages worth showing.
    """
    import pandas as pd

    from .plot import save_figure
    from .tabular import database_tables, read_table, write_table

    bundle = bundle or _st_read_bundle(
        request["folder"], str(request.get("platform") or "auto"),
        bin_um=int(request.get("bin_um") or 8),
        min_qv=float(request.get("min_qv") if request.get("min_qv")
                     is not None else 20.0))
    registered = registered or _st_register(bundle, request)
    masks = {kind: _st_load_mask(os.path.expanduser(path))
             for kind, path in (request.get("masks") or {}).items() if path}
    if not masks:
        raise ValueError("Give at least one object mask to assign reads to.")
    frame_shape = np.asarray(registered["image"]).shape[:2]
    for kind, mask in masks.items():
        if mask.shape != frame_shape:
            raise ValueError(
                f"The {kind} mask is {mask.shape[1]}x{mask.shape[0]} px but "
                f"the image the reads are registered to is "
                f"{frame_shape[1]}x{frame_shape[0]} px; segment that image "
                "or choose the image the mask was made from.")
    first = next(path for path in request["masks"].values() if path)
    prcf = str(request.get("prcf") or os.path.splitext(
        os.path.basename(str(first)))[0])
    db = os.path.expanduser(str(request["db"]))
    output = os.path.expanduser(str(request.get("output") or os.path.join(
        os.path.dirname(os.path.abspath(db)), "spatial_transcriptomics")))
    os.makedirs(output, exist_ok=True)
    genes = bundle["genes"]
    microns = registered["microns_per_pixel"]
    parasites = {k: masks[k] for k in ("pathogen", "vacuole") if k in masks}
    summary: Dict[str, Any] = {
        "platform": bundle["platform"], "prcf": prcf, "db": db,
        "output": output, "registration": registered["registration"],
        "objects": {}, "files": {}, "messages": []}
    per_object = {}
    for kind, mask in masks.items():
        labels, matrix, coverage = _st_object_counts(
            bundle, registered["xy"], mask, registered["radius"])
        written = _st_write_counts(db, kind, prcf, labels, genes, matrix,
                                   bundle["platform"])
        if bundle["platform"] == "xenium":
            assigned = int(matrix.sum())
            written["assigned_fraction"] = assigned / max(
                len(bundle["transcripts"]), 1)
        if coverage is not None:
            coverage = coverage.assign(
                barcode=bundle["obs"]["barcode"].to_numpy()[
                    coverage["spot"].to_numpy()],
                prcfo=_st_prcfo(prcf, coverage["object_label"]))
            _st_replace_image_rows(db, f"{kind}_spot_coverage",
                                   coverage.drop(columns=["spot"]), prcf)
            written["tables"] += (f"{kind}_spot_coverage",)
            written["spots_covering"] = int(coverage["spot"].nunique())
        summary["objects"][kind] = written
        per_object[kind] = (labels, matrix)
    host = "cell" if "cell" in masks else next(iter(masks))
    labels, matrix = per_object[host]
    geometry = _st_object_geometry(masks[host], {
        k: v for k, v in parasites.items() if k != host}, microns)
    geometry = geometry.reindex(labels)
    obs = geometry.reset_index()
    obs.insert(0, "prcfo", _st_prcfo(prcf, labels))
    if os.path.exists(db) and host in database_tables(db):
        measured = read_table(db, table=host, report=None)
        if "prcfo" in measured.columns:
            measured = measured.drop_duplicates("prcfo").set_index("prcfo")
            extra = measured.drop(columns=[c for c in measured.columns
                                           if c in obs.columns])
            obs = obs.join(extra, on="prcfo")
    files = summary["files"]
    if parasites and host not in parasites:
        infected = obs["infected"].to_numpy(dtype=bool)
        try:
            table = _st_compare_groups(matrix, genes, infected)
            files["infected_vs_uninfected"] = write_table(
                table, os.path.join(output, f"{prcf}_{host}_infected_vs_uninfected.csv"))
            summary["infected_vs_uninfected"] = {
                "n": table.attrs.get("n"), "genes": int(len(table)),
                "significant": int((table["q_value"] < 0.05).sum())}
        except ValueError as error:
            summary["messages"].append(str(error))
        try:
            trend = _st_distance_trend(matrix, genes,
                                       obs["distance_to_parasite"])
            files["distance_trend"] = write_table(
                trend, os.path.join(output, f"{prcf}_{host}_distance_trend.csv"))
        except ValueError as error:
            summary["messages"].append(str(error))
    region_path = str(request.get("region_mask") or "")
    if region_path:
        region_mask = _st_load_mask(os.path.expanduser(region_path))
        centres = obs[["centroid_x", "centroid_y"]].to_numpy()
        regions = _st_points_in_mask(centres, region_mask)
        obs["region"] = regions
    elif parasites and host not in parasites:
        regions = np.where(obs["infected"], "infected", "uninfected")
    else:
        regions = np.full(len(obs), "all")
    files["region_summary"] = write_table(
        _st_region_summary(matrix, genes, regions),
        os.path.join(output, f"{prcf}_{host}_region_summary.csv"))
    if request.get("anndata", True):
        try:
            files["anndata"] = _st_write_anndata(
                os.path.join(output, f"{prcf}_{host}.h5ad"), obs["prcfo"],
                genes, matrix, obs.drop(columns=["prcfo"]),
                obs[["centroid_x", "centroid_y"]].to_numpy())
        except ImportError as error:
            summary["messages"].append(str(error).splitlines()[0])
    from matplotlib.figure import Figure

    figure = Figure(figsize=(7.0, 6.0))
    gene = str(request.get("gene") or "")
    _st_draw_overlay(figure, registered["image"], registered["xy"],
                     _st_gene_values(bundle, gene),
                     radius=registered["radius"], mask=masks[host],
                     title=f"{bundle['platform']} {gene}".strip())
    files["overlay"] = save_figure(figure, os.path.join(
        output, f"{prcf}_overlay.png"))
    summary["obs"] = obs
    return summary
