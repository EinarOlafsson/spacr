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
    stored = _replace_well_rows(db, "ops_phenotype", frame, plate, well)

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
        "ops_phenotype_rows": stored,
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
    stored = _replace_well_rows(db, "ops_barcodes", frame, plate, well)
    reads_rows = None
    if store_reads:
        tick = time.perf_counter()
        reads = _reads_frame(plate, well, placements, decoded, cycles, _BASES)
        reads_rows = _replace_well_rows(db, "ops_reads", reads, plate, well)
        _say(f"{well} decode: ops_reads holds {reads_rows} rows "
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
        "ops_barcodes_rows": stored, "ops_reads_rows": reads_rows,
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
