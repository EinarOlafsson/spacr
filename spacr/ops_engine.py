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

import contextvars
import json
import logging
import os
import re
import time
from collections import Counter
from .resource_log import _parallel_thread_executor as ThreadPoolExecutor
from ._gpu_governor import _governed_eval
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

try:
    import resource
except ImportError:
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
#: own environment, NON-COMMERCIAL ACADEMIC USE ONLY, and opt-in;
#: ``spotiflow`` is Spotiflow's general model, run in its own environment.
_SPOT_DETECTORS = ("native", "spotnet", "spotiflow")

#: The detectors that run in a backend worker of their own, one field at a
#: time.
_BACKEND_SPOT_DETECTORS = ("spotnet", "spotiflow")

#: How a nucleus's reads become one barcode (``ops_cell_rule``).
#:
#: ``agree`` is the first validated rule: the reads vote on their RAW called
#: strings, at least two of them, three in five agreeing, and the winning
#: string is then corrected to the library at one mismatch. ``mapped`` matches
#: every read to the library FIRST and votes among the matched reads only.
#:
#: Measured on subset_v1, 116 stratified fields of the Funk et al. 2022
#: plates A-D (2026-11-04, decode-yield study): 83 % of nuclei carry
#: at least one read, the same as brieflow's 81.7 % "any barcode". ``agree``
#: assigned 28 % of nuclei, because a nucleus with one read can never vote and
#: two reads of one guide split the vote whenever either has a wrong base.
#: ``mapped`` with exact matches, one read allowed and the defaults below
#: assigned 62.0 % (field-bootstrap 95 % CI 61.1-62.9) against 58.2 % for
#: brieflow's own rule on the same reads, at 0.060 decoy matches per 100
#: assigned nuclei (CI 0.035-0.095) against brieflow's 0.102 and ``agree``'s
#: 0.43 -- the decoys being a second library of base-shuffled prefixes,
#: matched alongside the real one. ``agree`` is the worst of the three
#: because it corrects its winning string at one mismatch: this library's
#: guides sit two edits apart, so a read one error from a guide is often an
#: error from another guide.
_CELL_RULES = ("agree", "mapped")
_CELL_RULE = "mapped"

#: How many mismatches a single READ may carry and still match its guide
#: under the ``mapped`` rule. Zero: see :data:`_CELL_RULES` for the
#: decoy measurement. A base lost with its cycle is an N, not a mismatch.
_READ_MISMATCHES = 0

#: The share of a nucleus's library-matched reads its guide must hold, with
#: no tie for first. 0.6 is the ``agree`` rule's own bar, kept.
_MIN_FRACTION = 0.6

#: The quality a nucleus resting on ONE matched read needs (``ops_single_read_
#: quality``): that read's worst-cycle margin, after unmixing. The operating
#: point chosen on subset_v1 as the largest yield whose decoy rate, upper 95 %
#: bound included, stays below brieflow's rule's point estimate on the same
#: reads: 0 gave 62.5 % at 0.118 per 100, 0.03 gives 62.0 % at 0.060.
_SINGLE_READ_QUALITY = 0.03

#: How many reads a nucleus needs, per rule, when ``ops_min_reads`` is empty.
_MIN_READS = {"agree": 2, "mapped": 1}

#: How the base caller undoes dye bleed-through (``ops_crosstalk``), after
#: each cycle's channels are divided by their median. ``none`` is the first
#: validated caller; ``argmax_median`` is brieflow's median transform -- see
#: :func:`spacr.ops_sbs.compensate_crosstalk`. On subset_v1 it raised the
#: library-exact read rate on every plate and LOWERED the decoy match rate
#: (0.156 to 0.118 per 100 nuclei before the single-read gate), so it is the
#: default; ``percentile`` is
#: the variant 372 PART 14-L measured lowering the match, kept for that
#: comparison.
_CROSSTALK_METHODS = ("none", "argmax_median", "median", "percentile")

#: How each base channel is registered (``ops_channel_registration``); see
#: :data:`spacr.ops_cycles.CHANNEL_MODES`.
_CHANNEL_REGISTRATIONS = ("background", "spots", "spots_refine")
_CHANNEL_REGISTRATION = "spots"
_CROSSTALK = "argmax_median"

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

#: The phenotype path (alpha): build the plate's flat-field maps, segment
#: and measure every placed phenotype field, pass over its masks, merge its
#: cells onto the sequencing nuclei, and reduce cells to guides and genes.
#: Not in the default run, because each needs the four above and a design
#: table; a run includes them only when ``phases`` names them.
_PH_PHASES = ("ph_flatfield", "ph_segment", "ph_measure", "ph_masks",
              "ph_ownership", "merge", "ph_merged", "aggregate", "ph_tables",
              "ph_crops", "report", "stress")

#: The phenotype phases that run once per plate rather than per well; a
#: well's archive marker (item 872, B9) records only the others.
_PH_PLATE_PHASES = ("ph_flatfield", "aggregate", "ph_tables", "ph_crops",
                    "report", "stress")

#: The profile path (alpha): stored per-well cell tables to gene profiles,
#: then their retrieval metrics, their module atlas, the Leiden resolution
#: sweep (item 916) and the per-gene E-test (item 917). Needs no image step
#: and no database.
_PROFILE_PHASES = ("profile", "evaluate", "modules", "mitocarta",
                   "cluster_sweep", "etest", "cluster_consensus",
                   "cluster_bench_test")

#: The nuclear-median flat-field path (alpha, item 912): per-plate maps from
#: the per-nucleus tables, then their held-out check. Needs no image step
#: beyond the held-out fields' raw images.
_FLAT_PHASES = ("flat_nuclear", "flat_validate")

#: The shared scoring of gene profiles and clusterings (alpha, item 913):
#: copairs activity and consistency and EFAAR recall, with controls.
_SCORING_PHASES = ("score",)

#: The cropped-site layout path (alpha, items 919 and 920): index the Cell
#: Painting Gallery's per-channel site TIFFs, write their painting stacks to
#: ``merged/``, and decode the sequencing stacks against spacer prefixes.
_CPG_PHASES = ("cpg_index", "cpg_merged", "prefix_decode")

#: Every phase :func:`run_ops` accepts, in the order they run.
_ALL_PHASES = (_PHASES + _PH_PHASES + ("decode_qc", "compare_calls")
               + _PROFILE_PHASES + _FLAT_PHASES + _SCORING_PHASES + _CPG_PHASES)


def _run_cpg(settings: Mapping[str, Any], phases: Sequence[str]) -> Dict[str, Any]:
    """Run the cropped-site phases of :func:`run_ops` (items 919 and 920).

    :param settings: the OPS settings with ``ops_layout`` ``'cpg_cropped'``.
    :param phases: the phases asked for. The default tile phases mean
        ``cpg_index`` and ``cpg_merged``, plus ``prefix_decode`` when
        ``ops_library_mode`` is ``'spacer_prefix'``.
    :returns: ``{phase: report}`` and ``"layout": "cpg_cropped"``.
    :raises ValueError: when the layout is not ``'cpg_cropped'``, when a tile
        phase is asked for, or when ``prefix_decode`` is asked for without
        ``ops_library_mode='spacer_prefix'``.
    """
    from .cpg_cropped import run_cpg_phases

    layout = str(settings.get("ops_layout") or "tiles")
    if layout != "cpg_cropped":
        raise ValueError(f"phases {list(_CPG_PHASES)} read the cropped-site layout; "
                         f"set ops_layout to 'cpg_cropped' (got {layout!r})")
    mode = str(settings.get("ops_library_mode") or "barcode")
    if tuple(phases) == _PHASES:
        phases = ("cpg_index", "cpg_merged") + (
            ("prefix_decode",) if mode == "spacer_prefix" else ())
    other = [phase for phase in phases if phase not in _CPG_PHASES]
    if other:
        raise ValueError(f"ops_layout 'cpg_cropped' runs only {list(_CPG_PHASES)}; "
                         f"the crops are already stitched and registered, so {other} "
                         "do not apply")
    if "prefix_decode" in phases and mode != "spacer_prefix":
        raise ValueError("prefix_decode needs ops_library_mode 'spacer_prefix'")
    out: Dict[str, Any] = {"layout": "cpg_cropped", "phases": list(phases)}
    reader = [phase for phase in _CPG_PHASES[:2] if phase in phases]
    if reader:
        out["cpg"] = run_cpg_phases(settings, reader, report=_say)
    if "prefix_decode" in phases:
        from .ops_prefix import run_prefix_decode

        out["prefix_decode"] = run_prefix_decode(settings, report=_say)
    return out

#: The columns of ``ops_phenotype`` that hold each field's similarity
#: transform into the well frame: ``well = scale * R(rotation_deg) @
#: (row, col) + (offset_y, offset_x)``.
_PH_TRANSFORM_COLUMNS = ("scale", "rotation_deg", "offset_y", "offset_x",
                         "field_height", "field_width", "transform_source")

#: The guide library a decode worker compares calls against, set once per
#: process by :func:`_init_decode_worker` rather than shipped with every
#: field.
_LIBRARY: frozenset = frozenset()

#: The decoy barcodes a decode worker matches alongside :data:`_LIBRARY`
#: when a decoy FDR is asked for (item 873).
_DECOYS: frozenset = frozenset()


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

    A CHANNEL TOKEN SEEN TWICE FOR ONE SITE IS REFUSED, not overwritten
    (item 872, B3). Funk plate B's two phenotype acquisitions both hold an
    ``A594`` file per site, and the walk used to keep whichever it reached
    last. Naming each acquisition's folder and channels in
    ``ops_ph_acquisitions`` (:func:`_ph_acquisition_index`) indexes such a
    plate.

    :raises ValueError: when two files carry the same well, cycle, site
        and channel token.
    """
    from .qt.i18n import tr

    wanted = ("cycled", "uncycled") if scheme is None else (scheme,)
    found: Dict[str, Dict[str, Dict[int, Dict[int, Dict[str, str]]]]] = {
        name: {} for name in wanted}
    for folder, _folders, files in os.walk(root, followlinks=True):
        for name in sorted(files):
            for which in wanted:
                parsed = _match_tile(name, which)
                if parsed is None:
                    continue
                well, cycle, site, channel, _magnification = parsed
                held = found[which].setdefault(well, {}).setdefault(
                    cycle, {}).setdefault(site, {})
                path = os.path.join(folder, name)
                if channel in held and held[channel] != path:
                    raise ValueError(tr(
                        "two files under {root} carry channel {channel} for "
                        "well {well}, site {site}: {first} and {second}. Name "
                        "each acquisition's folder and channels in "
                        "ops_ph_acquisitions, or point the setting at one "
                        "acquisition.", root=root, channel=channel, well=well,
                        site=site, first=held[channel], second=path))
                held[channel] = path
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
        from .runctx import _is_overload_failure

        if _is_overload_failure(failure):
            raise
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

    A TABLE WITHOUT A PARQUET CACHE IS QUERIED FOR THE ONE WELL (item 872,
    B8): reading Funk plate A's 8 M-row ``ops_objects`` whole for each well
    took about 5 minutes and 6.6 GB, and the ``(plate, well, object_id)``
    index answers the well's query in seconds. The rows come back in the
    table's order, as the whole-table read filtered gave them. A well the
    query does not find -- a plate or well stored as a number -- is looked
    for in the whole table as before.
    """
    import sqlite3

    from .ops_store import CACHED_TABLES, _resolved, read_table, row_count
    from .tabular import _read_query

    if row_count(db, table) is None:
        return None
    if table not in CACHED_TABLES:
        with sqlite3.connect(_resolved(db), timeout=30) as conn:
            names = {r[1] for r in conn.execute(
                f"PRAGMA table_info({table})")}
            if not {"plate", "well"} <= names:
                return None
            frame = _read_query(
                conn, f"SELECT * FROM {table} WHERE plate = ? AND well = ? "
                      "ORDER BY rowid", params=(str(plate), str(well)),
                canonicalise=False, report=None)
        if len(frame):
            return frame.reset_index(drop=True)
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
    :param settings: read for ``ops_raster_overlap``,
        ``ops_registration_precision``, ``ops_stitch_read_threads``,
        ``ops_stitch_edge_workers``, ``ops_placement_seed`` and
        ``ops_placement_seed_export``.
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
    overlap = _setting_number(settings.get("ops_raster_overlap"),
                              "ops_raster_overlap", _RASTER_OVERLAP)
    precision = _registration_choice(settings.get("ops_registration_precision"),
                                     "ops_registration_precision")
    if settings.get("ops_placement_seed"):
        return _stitch_seeded(db, plate, well, cycle_files, reference,
                              settings, gpu, sources, sites, overlap,
                              precision, started)
    probe = functools.partial(_nuclear_plane, sources, [])
    if next((site for site in sites if probe(site) is not None), None) is None:
        raise ValueError(f"no nuclear tile of well {well} could be read")
    layout = round_well_layout(max(cycle_files[reference]) + 1)
    threads = int(_setting_number(settings.get("ops_stitch_read_threads"),
                                  "ops_stitch_read_threads",
                                  _STITCH_READ_THREADS))
    workers = max(1, int(_setting_number(settings.get("ops_stitch_edge_workers"),
                                         "ops_stitch_edge_workers", 1)))
    plane = functools.partial(_nuclear_plane, sources, unreadable)
    reader = (_TileReadAhead(plane, _first_use_order(layout, sites), threads)
              if threads > 1 else None)
    read = reader.get if reader is not None else functools.lru_cache(
        maxsize=64)(plane)
    try:
        first = next((site for site in sites if read(site) is not None), None)
        if first is None:
            raise ValueError(f"no nuclear tile of well {well} could be read")
        ordered = [first] + [site for site in sites if site != first]
        result = stitch_well(read, layout, overlap=overlap,
                             tolerance=_STITCH_TOLERANCE, gpu=gpu, sites=ordered,
                             edge_workers=workers, precision=precision)
    finally:
        if reader is not None:
            reader.close()
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
        "placement": "solved", "registration_precision": precision,
        "read_threads": threads, "edge_workers": workers,
    }
    if bool(settings.get("ops_placement_seed_export", True)):
        rows = {site: (result.placements[site][0] - origin_y,
                       result.placements[site][1] - origin_x) for site in placed}
        report["placement_seed_file"] = _export_placement_seed(
            db, plate, well, reference, rows, (height, width),
            (origin_y, origin_x), sources, layout.site_count, overlap)
    _say(f"{well} stitch: {result.summary()}; {len(placed)} of {len(sites)} "
         f"in the solved component, {report['seconds']} s")
    return report


#: How many threads read the nuclear tiles of a well for the stitch
#: (``ops_stitch_read_threads``, item 936). On well A1 of Funk et al. 2022
#: plates A and B, four reader threads halved the stitch step's wall time
#: (234-237 s to 114-132 s, tiles on network storage) for about 25-35 extra
#: CPU s per well, with ``ops_geometry`` bit-identical.
_STITCH_READ_THREADS = 4


def _nuclear_plane(sources: Mapping[int, Any], unreadable: list,
                   site: int) -> Optional[np.ndarray]:
    """One site's nuclear plane, or None, recording a failure.

    :param sources: ``site -> (path, plane)``.
    :param unreadable: where a failure is recorded.
    :param site: the site.
    :returns: the plane as float32, or None.
    """
    return _read_plane(sources[site], unreadable)


def _first_use_order(layout, sites: Sequence[int]) -> List[int]:
    """The order the stitch first asks for each site.

    :param layout: the well layout.
    :param sites: the sites with a nuclear plane.
    :returns: the sites in the order the layout's pairs first use them,
        then any site no pair uses.
    """
    known = set(sites)
    order: List[int] = []
    seen = set()
    for a, b, _axis in layout.pairs():
        for site in (a, b):
            if site in known and site not in seen:
                seen.add(site)
                order.append(site)
    order.extend(site for site in sites if site not in seen)
    return order


class _TileReadAhead:
    """Read a well's tiles ahead of the stitch on a few threads (item 936).

    The stitch asks for tiles in the order its pairs use them, and each read
    from network storage mostly waits. This reads the next tiles of that
    order on ``threads`` threads while the stitch registers the current
    ones, keeps the 64 most recently used, and gives back exactly what one
    sequential read would: a tile asked for out of order, or after it was
    dropped, is read on the spot. The reads are I/O, so the threads start
    together rather than ten seconds apart like spaCR's processing pools.

    :param read: ``site -> plane``.
    :param order: the sites in the order they will be asked for.
    :param threads: how many reader threads.
    :param keep: how many read tiles to keep.
    """

    def __init__(self, read, order: Sequence[int], threads: int,
                 keep: int = 64):
        """Start the reader threads.

        :param read: ``site -> plane``.
        :param order: the expected order of requests.
        :param threads: reader threads.
        :param keep: tiles kept after use.
        """
        from collections import OrderedDict
        from concurrent.futures import ThreadPoolExecutor as _ReaderThreads

        self._read = read
        self._order = list(order)
        self._next = 0
        self._ahead = 4 * int(threads)
        self._futures: Dict[int, Any] = {}
        self._kept: "OrderedDict[int, Any]" = OrderedDict()
        self._keep = int(keep)
        self._pool = _ReaderThreads(max_workers=int(threads),
                                    thread_name_prefix="ops-stitch-read")
        self._top_up()

    def _top_up(self) -> None:
        """Queue reads until ``4 x threads`` are waiting to be used."""
        while len(self._futures) < self._ahead and self._next < len(self._order):
            site = self._order[self._next]
            self._next += 1
            if site not in self._kept and site not in self._futures:
                self._futures[site] = self._pool.submit(self._read, site)

    def get(self, site: int):
        """The plane of ``site``, as the sequential read gives it.

        :param site: the site.
        :returns: the plane, or None for an unreadable tile.
        """
        if site in self._kept:
            self._kept.move_to_end(site)
            return self._kept[site]
        future = self._futures.pop(site, None)
        value = future.result() if future is not None else self._read(site)
        self._kept[site] = value
        while len(self._kept) > self._keep:
            self._kept.popitem(last=False)
        self._top_up()
        return value

    def close(self) -> None:
        """Stop the threads and drop every queued read."""
        for future in self._futures.values():
            future.cancel()
        self._futures.clear()
        self._kept.clear()
        self._pool.shutdown(wait=True)


def _compose_canvas(placements: Mapping[int, Tuple[float, float]],
                    shape: Tuple[int, int]) -> Tuple[int, int]:
    """The canvas the objects phase composes a well on.

    :param placements: ``site -> (y, x)`` in well pixels.
    :param shape: the tile ``(height, width)``.
    :returns: ``(height, width)``: the furthest tile corner, rounded up.
    """
    return (int(np.ceil(max(y for y, _ in placements.values()))) + shape[0],
            int(np.ceil(max(x for _, x in placements.values()))) + shape[1])


def _plane_record(source: Optional[Tuple[str, Optional[int]]]):
    """``(file name, byte size)`` of a plane's file, never its pixels.

    :param source: ``(path, plane)``.
    :returns: the pair, or None when the file cannot be listed.
    """
    if not source:
        return None
    try:
        return os.path.basename(source[0]), int(os.path.getsize(source[0]))
    except OSError:
        return None


def _plane_dtype(source: Optional[Tuple[str, Optional[int]]]) -> Optional[str]:
    """The pixel type a plane's file stores, read from its header.

    :param source: ``(path, plane)``.
    :returns: the dtype name, or None when the header cannot be read.
    """
    if not source:
        return None
    try:
        import tifffile

        with tifffile.TiffFile(source[0]) as handle:
            return str(handle.series[0].dtype)
    except Exception:                                    # noqa: BLE001
        return None


def _seed_settings_hash(overlap: int) -> str:
    """The hash of the settings that decide a well's placement.

    :param overlap: the raster overlap the stitch used.
    :returns: a sha256 hex digest.
    """
    import hashlib

    text = json.dumps({"ops_raster_overlap": int(overlap),
                       "tolerance": _STITCH_TOLERANCE}, sort_keys=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _tile_records(sources: Mapping[int, Any],
                  sites) -> Dict[int, Tuple[str, int]]:
    """``site -> (file name, byte size)`` for the sites whose file is listed.

    :param sources: ``site -> (path, plane)``.
    :param sites: the sites.
    :returns: the records.
    """
    found = {}
    for site in sites:
        record = _plane_record(sources.get(site))
        if record is not None:
            found[site] = record
    return found


def _export_placement_seed(db: str, plate: str, well: str, reference: int,
                           placements: Mapping[int, Tuple[float, float]],
                           shape: Tuple[int, int], origin: Tuple[float, float],
                           sources: Mapping[int, Any], n_tiles: int,
                           overlap: int) -> Optional[str]:
    """Write the well's placement file after a solved stitch (item 940).

    ``<plate>_<well>.ops_placement.json`` in the well's folder beside the
    database: the ``ops_geometry`` rows just written, with the provenance a
    later subset run checks (:func:`spacr.stitch_core.validate_placement_seed`).
    A file that cannot be written is logged, not raised: the stitch itself
    succeeded.

    :param db: the measurements database.
    :param plate: the plate name.
    :param well: the well name.
    :param reference: the reference cycle.
    :param placements: ``site -> (y, x)`` as stored.
    :param shape: the tile ``(height, width)``.
    :param origin: ``(origin_y, origin_x)`` as stored.
    :param sources: ``site -> (path, plane)`` of the nuclear planes.
    :param n_tiles: the layout's tile count.
    :param overlap: the raster overlap the stitch used.
    :returns: the path written, or None.
    """
    from .stitch_core import write_placement_seed

    folder = os.path.join(os.path.dirname(os.path.abspath(db)), str(well))
    path = os.path.join(folder, f"{plate}_{well}.ops_placement.json")
    try:
        return write_placement_seed(
            path, plate=plate, well=well, placements=placements,
            tile_shape=shape, origin=origin,
            canvas=_compose_canvas(placements, shape), acquisition="sbs",
            cycle=int(reference),
            dtype=_plane_dtype(sources.get(min(placements))),
            n_tiles_in_well=int(n_tiles), raster_overlap=int(overlap),
            tile_files=_tile_records(sources, placements),
            settings_hash=_seed_settings_hash(overlap))
    except (OSError, ValueError) as failure:
        LOG.warning("the placement file of well %s could not be written: %s",
                    well, failure)
        return None


def _load_seed_setting(value: Any, plate: str, well: str):
    """The placement seed ``ops_placement_seed`` names, read and checked.

    :param value: a ``.ops_placement.json`` file, a folder holding
        ``<plate>_<well>.ops_placement.json``, or a ``measurements.db``
        whose ``ops_geometry`` rows are taken
        (:func:`spacr.stitch_core.ops_seed_from_db`).
    :param plate: the plate name.
    :param well: the well name.
    :returns: the seed from :func:`spacr.stitch_core.load_placement_seed`.
    """
    from .stitch_core import load_placement_seed, ops_seed_from_db

    path = os.fspath(value)
    if os.path.isdir(path):
        path = os.path.join(path, f"{plate}_{well}.ops_placement.json")
    if path.lower().endswith(".db"):
        return load_placement_seed(ops_seed_from_db(path, plate, well))
    return load_placement_seed(path)


def _seed_pixel_check(seed: Mapping[str, Any], positions, sources, sites,
                      layout, overlap: int, precision: str, gpu: bool,
                      well: str, tolerance: float,
                      unreadable: list) -> Dict[str, Any]:
    """Re-register up to six seam pairs and compare them with the seed.

    :param seed: the loaded seed.
    :param positions: ``site -> (y, x)`` from the seed.
    :param sources: ``site -> (path, plane)``.
    :param sites: the sites of this run.
    :param layout: the well layout.
    :param overlap: the raster overlap.
    :param precision: the registration precision.
    :param gpu: let the registration use the card.
    :param well: the well name, for the message.
    :param tolerance: the largest accepted disagreement in pixels.
    :param unreadable: where unreadable planes are recorded.
    :returns: the pairs compared and the outcome.
    :raises ValueError: naming the pair and both offsets, when one
        disagrees by more than ``tolerance``.
    """
    from .ops_register import register_edge
    from .qt.i18n import tr

    known = set(sites)
    pairs = [(a, b, axis) for a, b, axis in layout.pairs()
             if a in known and b in known][:6]
    compared = []
    for a, b, axis in pairs:
        tile_a = _read_plane(sources[a], unreadable)
        tile_b = _read_plane(sources[b], unreadable)
        if tile_a is None or tile_b is None:
            continue
        found = register_edge(tile_a, tile_b, axis, expected_overlap=overlap,
                              tolerance=_STITCH_TOLERANCE, gpu=gpu,
                              precision=precision)
        if not found.accepted:
            continue
        stored = (positions[b][0] - positions[a][0],
                  positions[b][1] - positions[a][1])
        gap = float(np.hypot(stored[0] - found.dy, stored[1] - found.dx))
        compared.append({"pair": [a, b], "file": list(stored),
                         "measured": [found.dy, found.dx], "px": gap})
        if gap > tolerance:
            raise ValueError(tr(
                "placement file {path}: sites {a} and {b} of well {well} are "
                "{stored} apart in the file but {measured} in the tiles, "
                "{gap} px more than ops_placement_seed_tolerance {tolerance}",
                path=seed.get("path"), a=a, b=b, well=well,
                stored=[round(v, 2) for v in stored],
                measured=[found.dy, found.dx], gap=round(gap, 2),
                tolerance=tolerance))
    return {"asked": True, "pairs": compared, "tolerance_px": tolerance,
            "passed": True}


def _stitch_seeded(db: str, plate: str, well: str, cycle_files, reference: int,
                   settings: Mapping[str, Any], gpu: bool, sources, sites,
                   overlap: int, precision: str,
                   started: float) -> Dict[str, Any]:
    """Place the well's tiles from a placement seed instead of solving (item 940).

    The seed's positions are written to ``ops_geometry`` in the seed's
    whole-well frame -- its origin, not one re-zeroed to this run's tiles --
    for the sites of this run, so the objects phase composes each window
    the full run composed from the same tiles, at the same places. No pair
    is registered and nothing is solved, and the largest-component rule is
    not applied: the file defines the placement. With
    ``ops_placement_seed_check_pixels`` up to six seam pairs are registered
    and compared with the file, within ``ops_placement_seed_tolerance``
    pixels; the positions are never changed.

    :param db: the measurements database.
    :param plate: the plate name.
    :param well: the well name.
    :param cycle_files: ``cycle -> site -> channel -> path``.
    :param reference: the reference cycle.
    :param settings: read for the ``ops_placement_seed*`` keys.
    :param gpu: let the optional pixel check use the card.
    :param sources: ``site -> (path, plane)`` of the nuclear planes.
    :param sites: the sites with a nuclear plane.
    :param overlap: the raster overlap.
    :param precision: the registration precision of the pixel check.
    :param started: when the phase started.
    :returns: the stitch report, marked ``placement: seeded``. The well's
        layout is the seed's (``n_tiles_in_well``), not one fitted to this
        run's highest site, which a subset need not hold; under 'subset' the
        run's highest site must lie inside it, under 'strict' the counts
        must agree.
    :raises ValueError: when the seed does not belong to this run, or the
        pixel check disagrees.
    """
    import pandas as pd

    from .stitch_core import seed_to_placements, validate_placement_seed

    from .ops_layout import round_well_layout

    seed = _load_seed_setting(settings.get("ops_placement_seed"), plate, well)
    mode = str(settings.get("ops_placement_seed_mode") or "subset").strip().lower()
    unreadable: list = []
    count = max(cycle_files[reference]) + 1
    first, plane = None, None
    for site in sites:
        plane = _read_plane(sources[site], unreadable)
        if plane is not None:
            first = site
            break
    if first is None:
        raise ValueError(f"no nuclear tile of well {well} could be read")
    shape = (int(plane.shape[0]), int(plane.shape[1]))
    warnings = validate_placement_seed(
        seed, plate=plate, well=well, sites=sites, mode=mode,
        n_tiles_in_well=count, tile_shape=shape,
        dtype=_plane_dtype(sources.get(first)), acquisition="sbs",
        cycle=reference, tile_files=_tile_records(sources, sites))
    positions = seed_to_placements(seed, sites)
    check: Dict[str, Any] = {"asked": False}
    if bool(settings.get("ops_placement_seed_check_pixels", False)):
        layout = round_well_layout(int(seed.get("n_tiles_in_well") or count))
        check = _seed_pixel_check(
            seed, positions, sources, sites, layout, overlap, precision, gpu,
            well, float(_setting_number(
                settings.get("ops_placement_seed_tolerance"),
                "ops_placement_seed_tolerance", 2.0)), unreadable)
    origin_y, origin_x = (float(v) for v in seed["origin"])
    frame = pd.DataFrame([{
        "plate": plate, "well": well, "cycle": reference, "site": site,
        "y": positions[site][0], "x": positions[site][1],
        "origin_y": origin_y, "origin_x": origin_x,
        "tile_height": shape[0], "tile_width": shape[1],
    } for site in sorted(positions)])
    _replace_well_rows(db, "ops_geometry", frame, plate, well)
    listed = {int(tile["site"]) for tile in seed["tiles"]}
    report = {
        "placement": "seeded", "sites": len(sites), "placed": len(positions),
        "edges_proposed": None, "edges_accepted": None,
        "canvas": list(seed["canvas"]), "raster_overlap": overlap,
        "placement_seed": {
            "path": seed.get("path"), "sha256": seed.get("sha256"),
            "schema_version": seed.get("schema_version"), "mode": mode,
            "tiles_used": len(positions), "tiles_in_file": len(listed),
            "provenance": seed.get("provenance"), "check": check,
            "warnings": warnings,
        },
        "unreadable": unreadable,
        "seconds": round(time.perf_counter() - started, 1),
    }
    _say(f"{well} stitch: {len(positions)} of {len(listed)} tiles placed from "
         f"the placement file {seed.get('path')}, nothing solved, "
         f"{report['seconds']} s")
    return report


def _seeded_compose(settings: Mapping[str, Any], plate: str, well: str,
                    usable: Mapping[int, Tuple[float, float]],
                    shape: Tuple[int, int]):
    """The seed's canvas and the tiles it lists that this run does not use.

    :param settings: read for ``ops_placement_seed``.
    :param plate: the plate name.
    :param well: the well name.
    :param usable: ``site -> (y, x)`` of the tiles this run composes.
    :param shape: the tile ``(height, width)``.
    :returns: ``(canvas, unlisted)`` -- the seed's canvas and ``site -> (y,
        x)`` of the seed's other tiles -- or None without a seed.
    """
    from .stitch_core import seed_to_placements

    if not settings.get("ops_placement_seed"):
        return None
    seed = _load_seed_setting(settings.get("ops_placement_seed"), plate, well)
    others = {site: place for site, place in seed_to_placements(seed).items()
              if site not in usable}
    return (int(seed["canvas"][0]), int(seed["canvas"][1])), others


def _partial_windows(windows, usable: Mapping[int, Tuple[float, float]],
                     others: Mapping[int, Tuple[float, float]],
                     shape: Tuple[int, int]) -> List[List[int]]:
    """Windows a seeded run composes that an unlisted tile also covers.

    A window's pixels depend on every tile that covers it, so a window
    covered by this run's tiles and by a tile the seed lists but this run
    does not use differs from the full run where that tile contributed. These
    windows are composed and reported (item 940), and are left out of
    identity checks.

    :param windows: the windows of the run.
    :param usable: ``site -> (y, x)`` of the tiles this run composes.
    :param others: ``site -> (y, x)`` of the seed's other tiles.
    :param shape: the tile ``(height, width)``.
    :returns: ``[top, left]`` of each such window.
    """
    found = []
    for window in windows:
        if _touches(window, others, shape) and _touches(window, usable, shape):
            found.append([int(window.top), int(window.left)])
    return found


def _touches(window, placements: Mapping[int, Tuple[float, float]],
             shape: Tuple[int, int]) -> bool:
    """Whether any tile, at its rounded position, overlaps ``window``.

    :param window: an :class:`spacr.ops_compose.Window`.
    :param placements: ``site -> (y, x)``.
    :param shape: the tile ``(height, width)``.
    :returns: True when one does, by the rounding ``compose_window`` uses.
    """
    for top, left in placements.values():
        t_top, t_left = int(round(float(top))), int(round(float(left)))
        if (t_top < window.bottom and t_top + shape[0] > window.top
                and t_left < window.right and t_left + shape[1] > window.left):
            return True
    return False


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
    :param settings: read for ``ops_compose_blend``, the blend of the
        sequencing windows the anchor fields are aligned to -- the same
        blend the objects phase segments, so both see one mosaic.
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
    blend = _compose_blend(settings)
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

        Only tiles that touch the window are opened, not the whole well. Six
        anchors need about nine tiles each; testing readability over every
        placement instead would read all 333 of the reference well's
        sequencing tiles, and ``tifffile`` reads the whole five-plane
        1,480 px file for one plane of it -- 22 MB each, about 7 GB over NFS
        per well, before ``compose_window`` re-reads the nine it needs. The overlap
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
    field_shape: Optional[Tuple[int, int]] = None
    tried = 0
    for site in _anchor_candidates(layout, _PHENOTYPE_ANCHORS,
                                   _PHENOTYPE_CANDIDATES):
        if len(anchors) >= _PHENOTYPE_ANCHORS:
            break
        plane = _read_plane(sources.get(site), unreadable)
        if plane is None:
            continue
        tried += 1
        field_shape = field_shape or (int(plane.shape[0]), int(plane.shape[1]))
        if scale is None and sbs_magnification and phenotype_magnification:
            scale = float(shape[1] * sbs_magnification) / float(
                plane.shape[1] * phenotype_magnification)
        extent = (plane.shape[0] * (scale or 1.0),
                  plane.shape[1] * (scale or 1.0))
        window = _anchor_window(seed_of(site), extent, canvas)
        if window is None:
            continue
        image, coverage = compose_window(window, usable_under(window), read_sbs,
                                         tile_shape=shape, blend=blend)
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
    frame = _ph_transform_columns(
        frame, float(np.median([r["scale"] for r in records])),
        float(np.median([r["degrees"] for r in records])), field_shape)
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


def _cellpose_model(settings: Mapping[str, Any], gpu: bool,
                    key: str = "cellpose_model"):
    """The Cellpose model the objects are segmented with.

    :param settings: read for ``key``. Empty or ``cpsam`` builds
        the library's own default model, which is how the well was validated;
        any other name is passed to Cellpose as the model to load.
    :param gpu: put it on this machine's card through
        :func:`spacr.accelerator.cellpose_kwargs`, or keep it on the CPU.
    :param key: the setting naming the model; the phenotype phases pass
        ``ops_ph_nucleus_model`` and ``ops_ph_cell_model``.
    :returns: a ``cellpose.models.CellposeModel``.
    """
    from cellpose import models

    kwargs: Dict[str, Any] = {"gpu": False, "use_bfloat16": False}
    if gpu:
        from .accelerator import cellpose_kwargs

        kwargs = cellpose_kwargs()
    from .model_zoo import _resolve_pinned_model_file

    pinned = _resolve_pinned_model_file(settings, key, kinds=("cellpose",))
    name = (str(pinned) if pinned is not None
            else str(settings.get(key) or "").strip())
    if name in ("", "cpsam"):
        return models.CellposeModel(**kwargs)
    return models.CellposeModel(pretrained_model=name, **kwargs)


def _compose_blend(settings: Mapping[str, Any]) -> str:
    """The ``ops_compose_blend`` the run asked for, checked before any read.

    :param settings: read for ``ops_compose_blend``; empty means feather.
    :returns: ``'feather'`` or ``'average'``, the blends
        :func:`spacr.ops_compose.compose_window` takes.
    :raises ValueError: for any other name -- a typo falling back to the
        default would compose a different mosaic than the run recorded.
    """
    from .ops_compose import _BLENDS
    from .qt.i18n import tr

    blend = str(settings.get("ops_compose_blend") or _BLENDS[0]).strip().lower()
    if blend not in _BLENDS:
        raise ValueError(tr(
            "ops_compose_blend must be one of {blends}; got {value}",
            blends=list(_BLENDS), value=blend))
    return blend


def _objects(db: str, plate: str, well: str, cycle_files, reference: int,
             settings: Mapping[str, Any], gpu: bool) -> Dict[str, Any]:
    """Compose, segment, sew and number the well; store ``ops_objects``.

    :param db: the measurements database.
    :param plate: the plate name.
    :param well: the well name.
    :param cycle_files: ``cycle -> site -> channel -> path`` for this well.
    :param reference: the cycle carrying the nuclear stain.
    :param settings: read for the Cellpose model and diameter, and for
        ``ops_compose_blend``.
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
    seeded = _seeded_compose(settings, plate, well, usable, shape)
    canvas = seeded[0] if seeded else _compose_canvas(usable, shape)
    blend = _compose_blend(settings)
    model = _cellpose_model(settings, gpu)
    diameter = settings.get("cellpose_diameter")
    extra = {"diameter": float(diameter)} if diameter else {}

    window_overlap = int(_setting_number(settings.get("ops_window_overlap"),
                                         "ops_window_overlap",
                                         _WINDOW_OVERLAP))
    windows = list(windows_over(canvas, size=_WINDOW,
                                overlap=window_overlap))
    partial = (_partial_windows(windows, usable, seeded[1], shape)
               if seeded else None)
    seconds = Counter()
    observations: list = []
    gains = []
    empty = 0
    for count, window in enumerate(windows, start=1):
        tick = time.perf_counter()
        image, coverage = compose_window(window, usable, read, tile_shape=shape,
                                         blend=blend)
        seconds["compose"] += time.perf_counter() - tick
        covered = coverage > 0
        if not covered.any():
            empty += 1
            continue
        gains.append((overlap_gain(coverage), int(covered.sum())))
        fill = image.copy()
        fill[~covered] = float(np.median(image[covered]))
        tick = time.perf_counter()
        masks = np.asarray(_governed_eval(model, fill, site='ops_stitched',
                                          settings=settings, batch_size=16, **extra)[0], np.int32)
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
        "compose_blend": blend,
        "overlap_gain": round(sum(g * a for g, a in gains) / max(1, sum(a for _, a in gains)), 3),
        "objects_ready": [bool(gate), gate.rows, gate.reason],
        "unreadable": unreadable,
        "seconds": {k: round(v, 1) for k, v in seconds.items()},
        "total_seconds": round(time.perf_counter() - started, 1),
    }
    if seeded:
        report.update({"placement": "seeded", "partial_windows": partial,
                       "n_partial_windows": len(partial)})
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
    :returns: ``native``, ``spotnet`` or ``spotiflow``.
    :raises ValueError: for an unknown name, or a backend when it cannot run
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
    if name == "spotiflow":
        from ._segmentation_backends import _spotiflow_readiness

        ready, reason = _spotiflow_readiness()
        if not ready:
            raise ValueError(
                f"ops_spot_detector='spotiflow' cannot run: {reason}")
    return name


def _spotnet_peaks(stack: np.ndarray, detect=None,
                   threshold: Optional[float] = _SPOTNET_THRESHOLD
                   ) -> np.ndarray:
    """SpotNet's (or Spotiflow's) read positions for one aligned field, as
    whole pixels.

    SpotNet sees one image: each cycle's brightest base channel, scaled by
    its own 99.9th percentile so no cycle outweighs the rest, averaged over
    the cycles. A read is bright in some channel in every cycle, so it is
    bright in that image, which is the same thing the native score looks
    for across the stack.

    :param stack: ``cycles x channels x H x W``, aligned.
    :param detect: :func:`spacr._segmentation_backends._detect_spots`,
        :func:`spacr._segmentation_backends._spotiflow_spots`, or a
        stand-in for tests.
    :param threshold: the detector's own probability threshold; None keeps
        Spotiflow's optimised one.
    :returns: ``N x 2`` integer ``(y, x)``, unique and inside the field.
    """
    if detect is None:
        from ._segmentation_backends import _detect_spots as detect
    brightest = stack.max(axis=1).astype(np.float32)
    scale = np.percentile(brightest.reshape(len(brightest), -1), 99.9, axis=1)
    scale[~(scale > 0)] = 1.0
    image = (brightest / scale[:, None, None]).mean(axis=0)
    spots = np.asarray(detect(image, threshold=threshold), float)
    spots = spots.reshape(-1, 2)
    if not len(spots):
        return np.zeros((0, 2), dtype=np.int64)
    peaks = np.rint(spots).astype(np.int64)
    peaks[:, 0] = np.clip(peaks[:, 0], 0, image.shape[0] - 1)
    peaks[:, 1] = np.clip(peaks[:, 1], 0, image.shape[1] - 1)
    return np.unique(peaks, axis=0)


def _init_decode_worker(library: frozenset,
                        decoys: frozenset = frozenset()) -> None:
    """Hand a decode worker process the guide library once.

    :param library: the guide barcodes.
    :param decoys: the decoy barcodes matched alongside them.
    """
    global _LIBRARY, _DECOYS
    _LIBRARY = library
    _DECOYS = decoys


def _decode_field(task: Mapping[str, Any]) -> Dict[str, Any]:
    """Align, detect, call and attribute the reads of one field.

    :param task: ``site``, ``planes`` (``cycle -> [source per base
        channel]``), ``cycles``, ``reference``, and the nearby objects in
        this tile's frame -- ``centroids``, ``areas``, ``ids`` and ``owned``,
        whether each object's nearest tile is this one -- plus ``gpu``,
        ``threshold``, ``footprint``, ``store_reads`` and
        ``spot_detector`` (``native``, ``spotnet`` or ``spotiflow``; a
        backend's positions replace the native score's peaks and its
        threshold, and everything after them -- the margin, the bases, the
        calls and the attribution -- is the same code either way).
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

    from .ops_cycles import (FFT_BACKEND, FFT_PRECISION, _fft_backend_used,
                             align_field)
    from .ops_sbs import (attribute_reads, call_reads, called_bases,
                          estimate_read_locations, extract_bases, find_peaks)

    ticks = Counter()
    tick = time.perf_counter()
    site, cycles, reference = task["site"], list(task["cycles"]), task["reference"]
    gpu = bool(task.get("gpu", False))
    unreadable: list = []
    planes: Dict[int, Optional[list]] = {}
    slots = [(cycle, channel, source)
             for cycle, sources in task['planes'].items()
             for channel, source in enumerate(sources)]
    loaded = {cycle: [None] * len(sources)
              for cycle, sources in task['planes'].items()}

    def load_slot(slot):
        """Read one original source with the shared deferred overload policy."""
        cycle, channel, source = slot
        return cycle, channel, _read_plane(source, unreadable)

    with ThreadPoolExecutor(max(1, min(4, len(slots)))) as pool:
        for cycle, channel, array in pool.map(load_slot, slots):
            loaded[cycle][channel] = array
    for cycle, got in loaded.items():
        planes[cycle] = None if any(one is None for one in got) else got
    ticks["read"] = time.perf_counter() - tick
    base = {"site": site, "unreadable": unreadable,
            "missing": sorted(c for c in cycles if planes.get(c) is None)}
    if planes.get(reference) is None:
        return {**base, "skipped": "the reference cycle could not be read"}

    tick = time.perf_counter()
    diagnostics = (_field_diagnostics_before(task, planes, unreadable)
                   if task.get("diagnostics_folder") else None)
    field = align_field(planes, reference=reference, gpu=gpu,
                        channel_mode=str(task.get("channel_mode")
                                         or "background"),
                        registrations=(diagnostics["registrations"]
                                       if diagnostics is not None else None),
                        **task.get("registration", {}))
    ticks["align"] = time.perf_counter() - tick
    base.update(kept=list(field.kept), refused=list(field.refused),
                refused_channels=[list(pair) for pair in field.refused_channels],
        spot_replaced=[list(pair) for pair in field.spot_replaced],
        registration_backend=_fft_backend_used(
            task.get("registration", {}).get("backend", FFT_BACKEND), gpu,
            task.get("registration", {}).get("precision", FFT_PRECISION)))
    if len(field.kept) < _MIN_CYCLES:
        return {**base, "skipped": f"{len(field.kept)} usable cycles"}

    tick = time.perf_counter()
    stack = field.stack
    detector = str(task.get("spot_detector") or "native")
    spotnet = None
    if detector == "spotnet":
        spotnet = _spotnet_peaks(stack)
    elif detector == "spotiflow":
        from ._segmentation_backends import _spotiflow_spots

        spotnet = _spotnet_peaks(stack, detect=_spotiflow_spots,
                                 threshold=None)
    filtered = np.empty_like(stack)
    for c in range(stack.shape[0]):
        for k in range(stack.shape[1]):
            filtered[c, k] = np.clip(
                -ndimage.gaussian_laplace(stack[c, k], 1.0), 0, None)
    shifts = [abs(v) for pair in field.cycle_shifts.values() for v in pair]
    shifts += [abs(v) for per in field.channel_shifts.values()
               for pair in per for v in pair]
    margin = 5 + max(shifts + [0])
    if diagnostics is not None:
        _write_field_diagnostics(task, field, filtered, diagnostics, base)
    if task.get("rule_a"):
        result = _decode_field_rule_a(task, field, stack, filtered, spotnet,
                                      margin, gpu)
        ticks["spots_calls"] = time.perf_counter() - tick
        return {**base, **result,
                "seconds": {k: round(v, 2) for k, v in ticks.items()}}
    del stack
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
        intensities, calls, margins = called_bases(
            values, bases=_BASES, gpu=gpu, **_crosstalk_arguments(task))
        worst = np.where(np.isnan(margins), np.inf, margins).min(axis=1) \
            if len(calls) else np.zeros(0, np.float32)
        quality = np.where(np.isfinite(worst), worst, 0.0).astype(np.float32)
    else:
        intensities = margins = None
        calls, quality = call_reads(values, bases=_BASES, gpu=gpu,
                                    **_crosstalk_arguments(task))
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


def _registration_choice(value: Any, key: str) -> str:
    """``ops_registration_precision`` or ``ops_registration_backend``, checked.

    :param value: the setting's value; None or empty takes the default.
    :param key: which of the two keys it is.
    :returns: the value, lower case; the default when the key is unset.
    :raises ValueError: for a value the registration does not offer.
    """
    from .ops_cycles import (FFT_BACKEND, FFT_BACKENDS, FFT_PRECISION,
                             FFT_PRECISIONS)

    precision = key == "ops_registration_precision"
    allowed = FFT_PRECISIONS if precision else FFT_BACKENDS
    value = str(value or (FFT_PRECISION if precision
                          else FFT_BACKEND)).strip().lower()
    if value not in allowed:
        raise ValueError(f"{key} must be one of {list(allowed)}; got {value!r}")
    return value


def _registration_report(tasks, results) -> Dict[str, Any]:
    """The decode report's ``registration`` entry (item 932).

    :param tasks: the field tasks.
    :param results: every field's result.
    :returns: the precision and backend asked for, how many fields ran on
        each backend, and the diagnostics folder or None.
    """
    asked = tasks[0]["registration"] if tasks else {}
    folder = tasks[0].get("diagnostics_folder") if tasks else None
    table = _field_diagnostics_table(folder)
    path = None
    if table is not None and len(table):
        path = os.path.join(folder, "field_diagnostics.csv")
        table.to_csv(path, index=False)
    return {
        "precision": asked.get("precision"), "backend": asked.get("backend"),
        "fields_by_backend": dict(Counter(r["registration_backend"]
                                          for r in results
                                          if "registration_backend" in r)),
        "field_diagnostics": folder, "field_diagnostics_table": path,
    }


def _field_diagnostics_table(folder: Optional[str]):
    """One row per field from the diagnostics JSONs, or None when there are none.

    The decode-yield study's per-field table (its ``field_diag``): the
    weakest channel and cycle registration peaks, the median cycle peak,
    the largest channel shift, the refused cycles, the DAPI focus and each
    cycle's focus and mean 99.9th percentile. The diagnostics are optional
    (``ops_field_diagnostics``, off by default), so a missing folder gives
    None and a field written without a key gives NaN in its columns rather
    than an error.

    :param folder: ``<well>/decode_diagnostics``, or None.
    :returns: a DataFrame sorted by site, or None.
    """
    import glob

    import pandas as pd

    if not folder or not os.path.isdir(folder):
        return None
    rows = []
    for path in sorted(glob.glob(os.path.join(folder, "site*.json"))):
        try:
            with open(path, encoding="utf-8") as handle:
                record = json.load(handle)
        except (OSError, ValueError):
            continue
        registrations = record.get("registrations") or []
        cycles = len(record.get("kept") or []) + len(record.get("refused") or [])
        channels = [r[2] for r in registrations[:3 * cycles]]
        across = [r[2] for r in registrations[3 * cycles:]]
        shifts = [abs(v) for per in (record.get("channel_shifts") or {}).values()
                  for pair in per for v in pair]
        diag = record.get("diag") or {}
        row = {"site": record.get("site"),
               "chan_peak_min": min(channels) if channels else np.nan,
               "cycle_peak_min": min(across) if across else np.nan,
               "cycle_peak_median": float(np.median(across)) if across else np.nan,
               "max_chan_shift": max(shifts) if shifts else np.nan,
               "refused": len(record.get("refused") or []),
               "dapi_focus": diag.get("dapi_focus", np.nan)}
        for cycle, values in sorted(diag.items()):
            if isinstance(values, dict):
                row[f"focus_c{cycle}"] = values.get("focus_lapvar", np.nan)
                row[f"p999_c{cycle}"] = (float(np.mean(values["p999"]))
                                         if values.get("p999") else np.nan)
        rows.append(row)
    if not rows:
        return None
    return pd.DataFrame(rows).sort_values("site", kind="stable").reset_index(drop=True)


def _field_diagnostics_before(task: Mapping[str, Any], planes,
                              unreadable: list) -> Dict[str, Any]:
    """The per-field diagnostics measured on the raw planes (item 932).

    Off unless ``ops_field_diagnostics`` is set: no rule reads them, and on
    a 1,480 px field they cost about 4 CPU s of the decode.

    :param task: the field's task; ``dapi`` is the reference cycle's DAPI
        source, or None.
    :param planes: ``cycle -> raw planes``, None for a cycle not read.
    :param unreadable: the field's unreadable sources, appended to.
    :returns: ``diag`` (per cycle the planes' 50th and 99.9th percentiles
        and the variance of the Laplacian of the cycle's
        :func:`spacr.ops_cycles.sbs_mean_target`; ``dapi_focus``, the
        variance of the Laplacian of the DAPI plane divided by its 99th
        percentile) and an empty ``registrations`` list for
        :func:`spacr.ops_cycles.align_field` to fill.
    """
    from scipy import ndimage

    from .ops_cycles import sbs_mean_target

    diag: Dict[str, Any] = {}
    for cycle in task["cycles"]:
        found = planes.get(cycle)
        if found is None:
            continue
        laplace = ndimage.laplace(np.asarray(sbs_mean_target(found), np.float32))
        diag[str(cycle)] = {
            "p50": [float(np.percentile(plane, 50)) for plane in found],
            "p999": [float(np.percentile(plane, 99.9)) for plane in found],
            "focus_lapvar": float(laplace.var())}
    dapi = _read_plane(task["dapi"], unreadable) if task.get("dapi") else None
    if dapi is not None:
        dapi = np.asarray(dapi, np.float32)
        diag["dapi_focus"] = float(ndimage.laplace(
            dapi / max(np.percentile(dapi, 99), 1)).var())
    return {"diag": diag, "registrations": []}


def _write_field_diagnostics(task: Mapping[str, Any], field, filtered,
                             diagnostics: Mapping[str, Any],
                             base: Mapping[str, Any]) -> str:
    """Write one field's diagnostics as ``site<NNNN>.json`` (item 932).

    The keys are the decode-yield study's field JSON: ``diag``,
    ``log_p999`` (per kept cycle, each channel's 99.9th percentile after the
    -LoG filter), ``registrations`` (every background registration as
    ``[dy, dx, peak_ratio, accepted]``) and the field's alignment.

    :param task: the field's task, with ``diagnostics_folder``.
    :param field: the aligned field.
    :param filtered: the aligned stack after the -LoG filter.
    :param diagnostics: from :func:`_field_diagnostics_before`, its
        ``registrations`` filled by the alignment.
    :param base: the field's result so far.
    :returns: the path written.
    """
    folder = task["diagnostics_folder"]
    os.makedirs(folder, exist_ok=True)
    record = {
        "site": int(task["site"]), "kept": list(field.kept),
        "refused": list(field.refused), "missing": list(field.missing),
        "cycle_shifts": {str(k): list(v) for k, v in field.cycle_shifts.items()},
        "channel_shifts": {str(k): [list(p) for p in v]
                           for k, v in field.channel_shifts.items()},
        "spot_replaced": [list(p) for p in field.spot_replaced],
        "channel_mode": str(task.get("channel_mode") or "background"),
        "registration": dict(task.get("registration", {}),
                             used=base.get("registration_backend")),
        "registrations": [list(r) for r in diagnostics["registrations"]],
        "diag": diagnostics["diag"],
        "log_p999": {str(c): [float(np.percentile(filtered[i, k], 99.9))
                              for k in range(filtered.shape[1])]
                     for i, c in enumerate(field.kept)},
        "unreadable": [list(map(str, item)) for item in base.get("unreadable", [])],
    }
    path = os.path.join(folder, f"site{int(task['site']):04d}.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(record, handle)
    return path


def _decode_field_rule_a(task: Mapping[str, Any], field, stack, filtered,
                         spotnet, margin: int, gpu: bool) -> Dict[str, Any]:
    """The rest of :func:`_decode_field` under the item-873 decode rules.

    :param task: the field's task, carrying ``rule_a`` (the rule from
        :func:`spacr.ops_decode.decode_rule` with ``threshold``,
        ``footprint``, ``max_distance``, ``min_fraction`` and
        ``single_read_quality``), ``crosstalk`` and the objects.
    :param field: the aligned field.
    :param stack: its aligned stack.
    :param filtered: the stack after the -LoG filter.
    :param spotnet: a backend detector's peaks, or None for the native score.
    :param margin: how far from the edge a read must lie.
    :param gpu: let the peak finder use the card.
    :returns: ``n_cycles``, ``spots``, ``attributed``, ``ambiguous``,
        ``exact``, ``max_channel_shift``, ``rule_a`` (from
        :func:`spacr.ops_decode.field_calls`) and, when the field cache is
        stored, ``cache``.
    """
    from scipy import ndimage

    from .ops_decode import (field_calls, field_evidence, matcher_for,
                             save_field_cache, _CACHE_FLOOR)
    from .ops_sbs import estimate_read_locations, find_peaks

    rule = task["rule_a"]
    score = estimate_read_locations(filtered)
    if spotnet is not None:
        peaks = np.asarray(spotnet).reshape(-1, 2)
        strength = np.full(len(peaks), 1e6, np.float32)
    else:
        peaks = find_peaks(score, min_distance=2, gpu=gpu)
        surface = score - ndimage.minimum_filter(score, size=5)
        strength = (surface[peaks[:, 0], peaks[:, 1]] if peaks.size
                    else np.zeros(0, np.float32))
    floor = min(_CACHE_FLOOR, rule["threshold"]) if rule["store_cache"] \
        else rule["threshold"]
    evidence = field_evidence(
        stack, filtered, score, peaks, strength, field.kept, task, edge=margin,
        floor=floor,
        pixel_percentile=rule["pixel_percentile"] if rule["pixels"] else None,
        features=bool(rule.get("features")))
    matcher, decoys = matcher_for(_LIBRARY, _DECOYS)
    calls = field_calls(
        evidence, rule, sorted(_LIBRARY), matcher, decoys,
        threshold=rule["threshold"], footprint=rule["footprint"],
        max_distance=rule["max_distance"], min_fraction=rule["min_fraction"],
        single_read_quality=rule["single_read_quality"],
        crosstalk=str(task.get("crosstalk") or "none"), evidence_rows=True)
    out = {"n_cycles": len(field.kept), "spots": calls["reads"],
           "attributed": calls["reads_near"] - calls["reads_ambiguous"],
           "ambiguous": calls["reads_ambiguous"], "exact": None,
           "rule_a": calls, "ids": np.zeros(0, np.int64), "calls": [],
           "quality": np.zeros(0, np.float32),
           "max_channel_shift": int(max([abs(v) for per in field.channel_shifts.values()
                                         for pair in per for v in pair] + [0]))}
    if rule["store_cache"] and task.get("cache_folder"):
        out["cache"] = save_field_cache(task["cache_folder"], task["site"],
                                        evidence)
    return out


def _crosstalk_arguments(task: Mapping[str, Any]) -> Dict[str, Any]:
    """The base caller's unmixing arguments for one field's task.

    :param task: read for ``crosstalk``, one of :data:`_CROSSTALK_METHODS`.
    :returns: ``compensate`` and ``method`` for
        :func:`spacr.ops_sbs.called_bases`; ``none`` turns unmixing off.
    """
    method = str(task.get("crosstalk") or "none")
    if method == "none":
        return {"compensate": False}
    return {"compensate": True, "method": method}


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


def _cell_rule(settings: Mapping[str, Any], library: Sequence[str]
               ) -> Tuple[str, int, int, float]:
    """The rule that turns a nucleus's reads into one barcode, checked.

    :param settings: read for ``ops_cell_rule``, ``ops_read_mismatches``,
        ``ops_min_reads`` and ``ops_min_fraction``; empty takes the defaults.
    :param library: the guide barcodes. ``mapped`` needs them, so without
        a library the rule is ``agree`` whatever was asked, and said so.
    :returns: ``(rule, mismatches, min_reads, min_fraction)``.
    :raises ValueError: for an unknown rule, a negative mismatch count, a
        minimum below one read, or a fraction outside ``(0, 1]``.
    """
    rule = str(settings.get("ops_cell_rule") or _CELL_RULE).strip().lower()
    if rule not in _CELL_RULES:
        raise ValueError(f"ops_cell_rule must be one of {list(_CELL_RULES)}; "
                         f"got {rule!r}")
    if rule == "mapped" and not library:
        _say("decode: ops_cell_rule 'mapped' needs a guide library and none "
             "was given, so the reads vote on their raw strings ('agree')")
        rule = "agree"
    mismatches = int(_setting_number(settings.get("ops_read_mismatches"),
                                     "ops_read_mismatches", _READ_MISMATCHES))
    asked = settings.get("ops_min_reads")
    min_reads = (_MIN_READS[rule] if asked in (None, "")
                 else int(_setting_number(asked, "ops_min_reads", 1)))
    fraction = float(_setting_number(settings.get("ops_min_fraction"),
                                     "ops_min_fraction", _MIN_FRACTION))
    if mismatches < 0:
        raise ValueError(f"ops_read_mismatches must be 0 or more; got {mismatches}")
    if min_reads < 1:
        raise ValueError(f"ops_min_reads must be 1 or more; got {min_reads}")
    if not 0 < fraction <= 1:
        raise ValueError(f"ops_min_fraction must lie in (0, 1]; got {fraction}")
    return rule, mismatches, min_reads, fraction


def _rule_a_flag(settings: Mapping[str, Any], key: str) -> bool:
    """One boolean decode setting of item 873, its default when empty.

    :param settings: the settings.
    :param key: the key.
    :returns: the value.
    """
    from .ops_decode import _flag

    return _flag(settings, key)


def _decode_rule_a_report(db, plate, well, settings, rule_a, tasks, results,
                          decoded, skipped_sites, placements, owner_site,
                          cycles, channels, store_reads, detector, crosstalk,
                          channel_mode, workers, started) -> Dict[str, Any]:
    """Store a well decoded under the item-873 rules and report it.

    :param db: the measurements database.
    :param plate: the plate.
    :param well: the well.
    :param settings: the resolved settings.
    :param rule_a: the rule the fields were decoded under.
    :param tasks: the field tasks.
    :param results: every field's result, skipped ones included.
    :param decoded: the fields that decoded.
    :param skipped_sites: the unplaced sites.
    :param placements: ``site -> (y, x)``.
    :param owner_site: each object's owning site.
    :param cycles: the cycles.
    :param channels: the base channels.
    :param store_reads: unused under these rules; reported.
    :param detector: the spot detector.
    :param crosstalk: the unmixing method.
    :param channel_mode: the channel registration.
    :param workers: how many fields decoded at once.
    :param started: when the decode started.
    :returns: the decode report, with the FDR report under ``fdr``.
    """
    import pandas as pd

    from .ops_decode import barcode_rows, write_well_calls, write_well_reports

    columns = ["plate", "well", "object_id", "barcode", "quality", "n_reads",
               "n_agreeing", "fraction", "n_cycles", "mapped_guide"]
    frame = pd.DataFrame(barcode_rows(plate, well, decoded), columns=columns)
    _replace_well_rows(db, "ops_barcodes", frame, plate, well)
    fdr = write_well_reports(db, plate, well, decoded, rule_a, skipped_sites)
    calls_path = write_well_calls(db, well, decoded)
    spots = sum(r["spots"] for r in decoded)
    inside = sum(r["attributed"] + r["ambiguous"] for r in decoded)
    containment = inside / spots if spots else 0.0
    decoded_sites = {r["site"] for r in decoded}
    sources = Counter(hit["source"] for r in decoded
                      for hit in r["rule_a"]["calls"].values() if not hit["decoy"])
    report = {
        "fields": len(tasks), "fields_decoded": len(decoded),
        "skipped": {str(r["site"]): r["skipped"] for r in results if "skipped" in r},
        "skipped_unplaced_sites": skipped_sites,
        "cycles_used": dict(Counter(int(r["n_cycles"]) for r in decoded)),
        "cycles_missing": {str(r["site"]): r["missing"] for r in decoded if r["missing"]},
        "cycles_refused": {str(r["site"]): r["refused"] for r in decoded if r["refused"]},
        "channels_refused": sum(len(r["refused_channels"]) for r in decoded),
        "unreadable": [item for r in results for item in r["unreadable"]],
        "spots": spots, "containment": round(containment, 4),
        "containment_below_gate": bool(containment < _MIN_CONTAINMENT),
        "reads_ambiguous": sum(r["ambiguous"] for r in decoded),
        "reads_owned": sum(r["rule_a"]["reads_owned"] for r in decoded),
        "objects_owned_by_decoded_fields": int(np.isin(owner_site, list(decoded_sites)).sum()),
        "objects_assigned": int(len(frame)),
        "objects_mapped": int(len(frame)),
        "objects_assigned_decoy": int(fdr["assigned_decoy"]),
        "assigned_by_source": dict(sources),
        "ops_barcodes_rows": int(len(frame)), "ops_reads_rows": None,
        "cell_rule": "mapped", "decode_rules": {
            k: v for k, v in rule_a.items() if k not in ("active",)},
        "crosstalk": crosstalk, "channel_registration": channel_mode,
        "channels_spot_registered": sum(len(r.get("spot_replaced", ()))
                                        for r in decoded),
        "registration": _registration_report(tasks, results),
        "read_threshold": rule_a["threshold"], "footprint": rule_a["footprint"],
        "spot_detector": detector, "store_reads": bool(store_reads),
        "base_channels": list(channels), "workers": workers,
        "fdr": fdr, "calls": calls_path,
        "field_cache": sorted(r["cache"] for r in decoded if r.get("cache")),
        "field_seconds": dict(sum((Counter(r["seconds"]) for r in results
                                   if "seconds" in r), Counter())),
        "total_seconds": round(time.perf_counter() - started, 1),
    }
    _say(f"{well} decode: {len(frame)} objects assigned "
         f"({fdr['pct']} % of {fdr['nuclei']} nuclei), FDR "
         f"{fdr['fdr_per_100']} per 100, {len(decoded)} fields, "
         f"{report['total_seconds']} s")
    return report


def _decode(db: str, plate: str, well: str, cycle_files, reference: int,
            settings: Mapping[str, Any], gpu: bool,
            library: Sequence[str]) -> Dict[str, Any]:
    """Decode every field of the well and store ``ops_barcodes``.

    A stored ``barcode`` is what was read, not the guide it matched: the
    commonest raw string among the reads behind the guide, so an N from a
    lost cycle still shows. ``n_reads`` counts every read the nucleus owns,
    matched to the library or not.

    :param db: the measurements database.
    :param plate: the plate name.
    :param well: the well name.
    :param cycle_files: ``cycle -> site -> channel -> path`` for this well.
    :param reference: the cycle the objects' frame was stitched on.
    :param settings: read for ``n_workers``, ``ops_base_channels``,
        ``ops_read_threshold``, ``ops_footprint``, ``ops_store_reads``,
        ``ops_spot_detector`` and the cell rule's four keys -- see
        :func:`_cell_rule`.
    :param gpu: let the decode use the card when it runs in this process.
    :param library: the guide barcodes, possibly empty.
    :returns: the decode report. Its ``ops_barcodes_rows`` and
        ``ops_reads_rows`` count this well's rows only, the number a
        ``plate``/``well`` filter on the stored table returns;
        ``ops_reads_rows`` is None when reads were not stored.
    :raises ValueError: when the objects are not ready.
    """
    import multiprocessing
    from .resource_log import _parallel_process_executor as ProcessPoolExecutor

    import pandas as pd
    from scipy.spatial import cKDTree

    from .ops_sbs import assign_reads_to_objects, correct_to_library
    from .ops_store import objects_ready

    from .ops_decode import decode_rule, resolve_settings, unplaced_sites

    started = time.perf_counter()
    settings = resolve_settings(settings)
    gate = objects_ready(db)
    if not gate:
        raise ValueError(gate.reason)
    placements, shape = _placements(db, plate, well)
    skipped_sites = unplaced_sites(cycle_files, placements)
    rule_a = decode_rule(settings)
    if skipped_sites:
        if not _rule_a_flag(settings, "ops_decode_skip_unplaced_sites"):
            raise ValueError(
                f"well {well}: the stitch placed no position for sites "
                f"{skipped_sites}; set ops_decode_skip_unplaced_sites to "
                "decode the placed sites and log these")
        _say(f"{well} decode: {len(skipped_sites)} site(s) the stitch did not "
             f"place are skipped and logged: {skipped_sites}")
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
    crosstalk = str(settings.get("ops_crosstalk") or _CROSSTALK).strip().lower()
    if crosstalk not in _CROSSTALK_METHODS:
        raise ValueError(f"ops_crosstalk must be one of "
                         f"{list(_CROSSTALK_METHODS)}; got {crosstalk!r}")
    channel_mode = str(settings.get("ops_channel_registration")
                       or _CHANNEL_REGISTRATION).strip().lower()
    if channel_mode not in _CHANNEL_REGISTRATIONS:
        raise ValueError(f"ops_channel_registration must be one of "
                         f"{list(_CHANNEL_REGISTRATIONS)}; got {channel_mode!r}")
    registration = {
        "refine_px": int(_setting_number(settings.get("ops_registration_refine_px"),
                                         "ops_registration_refine_px", 3)),
        "spot_min_peak": float(_setting_number(settings.get("ops_spot_min_peak"),
                                               "ops_spot_min_peak", 20.0)),
        "spot_overrule_share": float(_setting_number(
            settings.get("ops_spot_overrule_share"), "ops_spot_overrule_share",
            0.5)),
        "precision": _registration_choice(
            settings.get("ops_registration_precision"), "ops_registration_precision"),
        "backend": _registration_choice(
            settings.get("ops_registration_backend"), "ops_registration_backend"),
    }
    diagnostics_folder = (os.path.join(os.path.dirname(os.path.abspath(db)),
                                       well, "decode_diagnostics")
                          if bool(settings.get("ops_field_diagnostics", False))
                          else None)
    decoys: List[str] = []
    rule, mismatches, min_reads, min_fraction = _cell_rule(settings, library)
    if rule_a["active"] and rule == "mapped":
        from .ops_sbs import decoy_library

        rule_a.update({
            "threshold": threshold, "footprint": footprint,
            "max_distance": mismatches, "min_fraction": min_fraction,
            "single_read_quality": float(_setting_number(
                settings.get("ops_single_read_quality"),
                "ops_single_read_quality", _SINGLE_READ_QUALITY)),
            "features": rule_a["features"],
        })
        if rule_a["decoy_k"] > 0:
            decoys = decoy_library(library, k=rule_a["decoy_k"])
        cache_folder = os.path.join(os.path.dirname(os.path.abspath(db)),
                                    well, "decode_cache")
    else:
        if rule_a["active"]:
            _say(f"{well} decode: the item-873 decode rules need "
                 "ops_cell_rule 'mapped' and a guide library; decoding with "
                 f"the '{rule}' rule instead")
        rule_a = None
        cache_folder = None
    tasks = []
    for site in order:
        top, left = placements[site]
        near = np.flatnonzero((oy > top - 30) & (oy < top + shape[0] + 30)
                              & (ox > left - 30) & (ox < left + shape[1] + 30))
        planes = {}
        dapi = None
        for cycle in cycles:
            sources = _plane_sources(cycle_files[cycle].get(site, {}))
            planes[cycle] = [sources.get(name) for name in channels]
            if cycle == reference:
                dapi = sources.get("DAPI")
        tasks.append({
            "site": site, "planes": planes, "cycles": cycles,
            "reference": reference,
            "centroids": np.column_stack([oy[near] - top, ox[near] - left]),
            "areas": areas[near], "ids": ids[near],
            "owned": owner_site[near] == site,
            "threshold": threshold, "footprint": footprint,
            "store_reads": store_reads, "spot_detector": detector,
            "crosstalk": crosstalk, "channel_mode": channel_mode,
            "registration": registration, "rule_a": rule_a,
            "cache_folder": cache_folder,
            "diagnostics_folder": diagnostics_folder,
            "dapi": dapi if diagnostics_folder else None,
        })

    library_set = frozenset(library)
    decoy_set = frozenset(decoys)
    workers = max(1, min(int(settings.get("n_workers") or 1),
                         _DECODE_WORKERS_CAP, len(tasks)))
    if detector in _BACKEND_SPOT_DETECTORS and workers > 1:
        _say(f"{well} decode: {detector} runs in one worker of its own, so the "
             f"fields decode one at a time rather than {workers} at once")
        workers = 1
    if workers > 1 and tasks:
        from .resource_log import _array_file_nbytes, _guard_workers
        first = [entry[0] for entries in tasks[0]["planes"].values()
                 for entry in entries if entry]
        workers = _guard_workers(
            "ops_decode", workers,
            len(first) * _array_file_nbytes(first[0]) if first else 0,
            settings=settings)
    results = []
    if workers == 1:
        _init_decode_worker(library_set, decoy_set)
        for task in tasks:
            results.append(_decode_field({**task, "gpu": gpu}))
    else:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(workers, mp_context=context,
                                 initializer=_init_decode_worker,
                                 initargs=(library_set, decoy_set)) as pool:
            for count, result in enumerate(pool.map(
                    _decode_field, [{**t, "gpu": False} for t in tasks]), start=1):
                results.append(result)
                if count % 25 == 0:
                    _say(f"{well} decode: {count} of {len(tasks)} fields")

    decoded = [r for r in results if "skipped" not in r]
    if rule_a is not None:
        return _decode_rule_a_report(
            db, plate, well, settings, rule_a, tasks, results, decoded,
            skipped_sites, placements, owner_site, cycles, channels,
            store_reads, detector, crosstalk, channel_mode, workers, started)
    all_ids = np.concatenate([r["ids"] for r in decoded]) if decoded else np.zeros(0, np.int64)
    all_calls = [code for r in decoded for code in r["calls"]]
    all_quality = (np.concatenate([r["quality"] for r in decoded])
                   if decoded else np.zeros(0, np.float32))
    cycles_of = {}
    for r in decoded:
        for object_id in np.unique(r["ids"]):
            cycles_of[int(object_id)] = r["n_cycles"]
    reads_mapped = None
    if rule == "mapped":
        calls_seen = sorted(set(all_calls))
        to_guide = dict(zip(calls_seen, correct_to_library(
            calls_seen, library, max_distance=mismatches)))
        guides = [to_guide[code] or "" for code in all_calls]
        matched = np.fromiter((bool(g) for g in guides), dtype=bool,
                              count=len(guides))
        reads_mapped = int(matched.sum())
        assigned = assign_reads_to_objects(
            np.where(matched, all_ids, 0), guides, quality=all_quality,
            min_reads=min_reads, min_fraction=min_fraction)
        totals = Counter(all_ids.tolist())
        raw: Dict[int, Counter] = {}
        for index in np.flatnonzero(matched):
            object_id = int(all_ids[index])
            row = assigned.get(object_id)
            if row is not None and guides[index] == row["barcode"]:
                raw.setdefault(object_id, Counter())[all_calls[index]] += 1
        mapped = {}
        single_q = float(_setting_number(
            settings.get("ops_single_read_quality"),
            "ops_single_read_quality", _SINGLE_READ_QUALITY))
        assigned = {object_id: row for object_id, row in assigned.items()
                    if row["agreeing"] > 1
                    or (row["quality"] or 0.0) >= single_q}
        for object_id, row in assigned.items():
            row["reads"] = totals[object_id]
            guide = row["barcode"]
            row["barcode"] = raw[object_id].most_common(1)[0][0]
            mapped[object_id] = guide
        objects_with_mapped = int(len(np.unique(all_ids[matched])))
    else:
        assigned = assign_reads_to_objects(
            all_ids, all_calls, quality=all_quality, min_reads=min_reads,
            min_fraction=min_fraction)
        distinct = sorted({row["barcode"] for row in assigned.values()})
        corrected = (dict(zip(distinct, correct_to_library(distinct, library)))
                     if library else {})
        mapped = {object_id: corrected.get(row["barcode"])
                  for object_id, row in assigned.items()}
        objects_with_mapped = None
    frame = pd.DataFrame([{
        "plate": plate, "well": well, "object_id": object_id,
        "barcode": row["barcode"], "quality": row["quality"],
        "n_reads": row["reads"], "n_agreeing": row["agreeing"],
        "fraction": row["fraction"], "n_cycles": cycles_of.get(object_id, 0),
        "mapped_guide": mapped.get(object_id) or "",
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
        "cell_rule": rule, "read_mismatches": mismatches,
        "crosstalk": crosstalk, "channel_registration": channel_mode,
        "channels_spot_registered": sum(len(r.get("spot_replaced", ()))
                                        for r in decoded),
        "registration": _registration_report(tasks, results),
        "min_reads": min_reads, "min_fraction": min_fraction,
        "reads_mapped": reads_mapped,
        "objects_with_a_mapped_read": objects_with_mapped,
        "read_threshold": threshold, "footprint": footprint,
        "spot_detector": detector, "skipped_unplaced_sites": skipped_sites,
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


def _ph_transform_columns(frame, scale: float, degrees: float,
                          field_shape: Optional[Tuple[int, int]]):
    """``ops_phenotype`` rows with each field's similarity transform added.

    :param frame: one well's ``ops_phenotype`` rows.
    :param scale: well pixels per field pixel.
    :param degrees: the fields' rotation in the well frame.
    :param field_shape: ``(height, width)`` of a field in its own pixels.
    :returns: the frame with :data:`_PH_TRANSFORM_COLUMNS` filled.

    ONE SCALE AND ONE ROTATION PER WELL. Every field is the same camera on
    the same objective, so the anchors' median is every field's; what
    differs per field is where its centre lands, which is the raster fit
    the phase already stores. The merge phase re-solves each field on its
    own nuclei and records that separately (``refined_*``).
    """
    from .ops_merge import field_transform

    out = frame.copy()
    height, width = field_shape if field_shape else (np.nan, np.nan)
    offsets = [field_transform(scale, degrees, (cy, cx), (height, width))[2]
               if field_shape else (np.nan, np.nan)
               for cy, cx in zip(out["centre_y"], out["centre_x"])]
    out["scale"] = float(scale)
    out["rotation_deg"] = float(degrees)
    out["offset_y"] = [float(o[0]) for o in offsets]
    out["offset_x"] = [float(o[1]) for o in offsets]
    out["field_height"] = height
    out["field_width"] = width
    out["transform_source"] = np.where(out["is_anchor"].astype(int) > 0,
                                       "anchor", "raster")
    return out


def _ph_transforms(db: str, plate: str, well: str, destination: str,
                   field_shape: Optional[Tuple[int, int]] = None):
    """One well's ``ops_phenotype`` rows with their transforms, migrated.

    :param db: the measurements database.
    :param plate: the plate name.
    :param well: the well name.
    :param destination: the run folder holding ``<well>/ops_report.json``.
    :param field_shape: the field size, needed only when the rows predate
        the transform columns.
    :returns: the rows, with :data:`_PH_TRANSFORM_COLUMNS`.
    :raises ValueError: when the well was never placed, or an old database
        cannot be migrated because neither its report nor the call gives a
        scale or a field size.

    A DATABASE WRITTEN BEFORE THE COLUMNS EXISTED STAYS READABLE: its rows
    are completed from the anchors its own ``ops_report.json`` recorded --
    the median anchor scale and rotation, or the report's
    ``expected_scale`` and no rotation -- and written back, so the
    migration runs once per well.
    """
    from .qt.i18n import tr

    frame = _well_rows(db, "ops_phenotype", plate, well)
    if frame is None or frame.empty:
        raise ValueError(tr(
            "well {well} has no ops_phenotype rows in {db}; run the "
            "phenotype phase first", well=well, db=db))
    have = set(frame.columns)
    if set(_PH_TRANSFORM_COLUMNS) <= have and frame["scale"].notna().all() \
            and frame["offset_y"].notna().all():
        return frame
    report_path = os.path.join(destination, well, "ops_report.json")
    placed: Dict[str, Any] = {}
    if os.path.isfile(report_path):
        with open(report_path, encoding="utf-8") as handle:
            placed = (json.load(handle) or {}).get("phenotype") or {}
    anchors = [a for a in placed.get("anchors") or [] if "scale" in a]
    if anchors:
        scale = float(np.median([float(a["scale"]) for a in anchors]))
        degrees = float(np.median([float(a.get("degrees", 0.0))
                                   for a in anchors]))
    elif placed.get("expected_scale"):
        scale, degrees = float(placed["expected_scale"]), 0.0
    elif "scale" in have and frame["scale"].notna().any():
        scale = float(frame["scale"].dropna().median())
        degrees = float(frame.get("rotation_deg", 0.0).fillna(0.0).median())
    else:
        raise ValueError(tr(
            "well {well}'s ops_phenotype rows carry no transform and "
            "{report} records no scale; rerun the phenotype phase",
            well=well, report=report_path))
    if field_shape is None and "field_height" in have \
            and frame["field_height"].notna().any():
        field_shape = (int(frame["field_height"].dropna().iloc[0]),
                       int(frame["field_width"].dropna().iloc[0]))
    if field_shape is None:
        raise ValueError(tr(
            "well {well}'s ops_phenotype rows carry no field size; read one "
            "phenotype field first", well=well))
    base = frame.drop(columns=[c for c in _PH_TRANSFORM_COLUMNS if c in have])
    migrated = _ph_transform_columns(base, scale, degrees, field_shape)
    _replace_well_rows(db, "ops_phenotype", migrated, plate, well)
    return migrated


def _ph_sites(value) -> Optional[set]:
    """The phenotype sites a setting names, or None for all of them.

    :param value: None, an empty string, or ``"520-525,555"``.
    :returns: the set of sites.
    """
    if value in (None, ""):
        return None
    if isinstance(value, (list, tuple, set)):
        return {int(v) for v in value}
    sites: set = set()
    for part in str(value).replace(" ", "").split(","):
        if not part:
            continue
        if "-" in part:
            low, high = part.split("-", 1)
            sites.update(range(int(low), int(high) + 1))
        else:
            sites.add(int(part))
    return sites


def _ph_channels(settings: Mapping[str, Any]) -> Tuple[str, ...]:
    """The phenotype channels, in the order the cell table names them."""
    raw = settings.get("ops_ph_channels") or "DAPI,GFP,A594,AF750"
    return tuple(c.strip() for c in str(raw).split(",") if c.strip())


def _ph_split_files(files: Mapping[str, str], secondary: Sequence[str]
                    ) -> Tuple[Dict[str, str], Dict[str, str]]:
    """One site's files split into the primary and the second acquisition.

    :param files: ``channel token -> path`` for the site.
    :param secondary: the channels the second acquisition holds.
    :returns: ``(primary, second)``: a file whose channel token names any
        of ``secondary`` belongs to the second acquisition, every other
        file to the primary one.
    """
    wanted = {str(c).upper() for c in secondary}
    primary: Dict[str, str] = {}
    second: Dict[str, str] = {}
    for token, path in files.items():
        names = {name.upper() for name in str(token).split("-")}
        (second if names & wanted or "@" in str(token)
         else primary)[token] = path
    return primary, second


def _ph_acquisitions(settings: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    """``ops_ph_acquisitions`` read: each acquisition's folder and channels.

    :param settings: read for ``ops_ph_acquisitions``, a mapping (or its
        JSON text) of acquisition folder to the channels taken from it,
        comma-separated or a list, plus an optional ``register_on``
        channel; for example ``{"phenotype": "DAPI,GFP,A594", "phenotype_af750":
        "AF750", "register_on": "A594"}``. A relative folder is under
        ``phenotype_source``. The first acquisition is the primary one.
    :returns: ``{"folders": [(folder, [channels])], "register_on": channel
        or None}``, or None when the setting is empty.
    :raises ValueError: when the setting is not a mapping, an acquisition
        names no channel, or a channel is named for two acquisitions.
    """
    from .qt.i18n import tr

    raw = settings.get("ops_ph_acquisitions")
    if not raw:
        return None
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except ValueError as failure:
            raise ValueError(tr(
                "ops_ph_acquisitions must map each acquisition folder to its "
                "channels; {value!r} is not that: {error}", value=raw,
                error=failure)) from failure
    if not isinstance(raw, Mapping):
        raise ValueError(tr(
            "ops_ph_acquisitions must map each acquisition folder to its "
            "channels; got {value!r}", value=raw))
    register_on = raw.get("register_on")
    folders: List[Tuple[str, List[str]]] = []
    owner: Dict[str, str] = {}
    for folder, channels in raw.items():
        if folder == "register_on":
            continue
        names = [c.upper() for c in _ph_names(channels)]
        if not names:
            raise ValueError(tr(
                "ops_ph_acquisitions names no channel for {folder}",
                folder=folder))
        for name in names:
            if name in owner:
                raise ValueError(tr(
                    "ops_ph_acquisitions takes channel {channel} from both "
                    "{first} and {second}; each channel comes from one "
                    "acquisition", channel=name, first=owner[name],
                    second=folder))
            owner[name] = str(folder)
        folders.append((str(folder), names))
    if not folders:
        raise ValueError(tr("ops_ph_acquisitions names no acquisition"))
    return {"folders": folders,
            "register_on": str(register_on).upper() if register_on else None}


def _ph_acquisition_index(root: str, acquisitions: Mapping[str, Any]
                          ) -> Tuple[Dict[str, Any], Dict[str, int]]:
    """One phenotype index built from the acquisitions ``ops_ph_acquisitions`` names.

    :param root: ``phenotype_source``; relative acquisition folders are
        under it.
    :param acquisitions: :func:`_ph_acquisitions`.
    :returns: ``(index, ignored)``: the ``well -> cycle -> site -> channel
        token -> path`` index of the files whose token names a channel
        listed for their acquisition, and how many files each acquisition
        had that were left out. A later acquisition's single-channel file
        of the ``register_on`` channel is kept under the token
        ``<channel>@<n>`` (n the acquisition's position), which
        :func:`_ph_secondary_pair` registers on and no measurement reads.
    :raises ValueError: when a folder is missing or holds no tiles, or two
        files give one site the same token.

    A FOLDER THAT IS NOT LISTED IS NOT READ, which is how Funk plate D's
    single-file GFP re-image stays out of the index: its in-stack GFP is
    the one listed. Each acquisition is checked for one objective on its
    own (:func:`_one_acquisition`).
    """
    from .qt.i18n import tr

    register_on = acquisitions.get("register_on")
    merged: Dict[str, Any] = {}
    ignored: Dict[str, int] = {}
    for number, (folder, channels) in enumerate(acquisitions["folders"]):
        path = folder if os.path.isabs(folder) else os.path.join(root, folder)
        if not os.path.isdir(path):
            raise ValueError(tr(
                "ops_ph_acquisitions names {folder}, which is not a folder "
                "under {root}", folder=folder, root=root))
        index = (_index_tiles(path, "uncycled")
                 or _index_tiles(path, "cycled"))
        if not index:
            raise ValueError(tr(
                "ops_ph_acquisitions names {folder}, which holds no phenotype "
                "tiles", folder=path))
        _one_acquisition(index, "ops_ph_acquisitions", path)
        listed = set(channels)
        ignored[folder] = 0
        for well, cycles in index.items():
            for cycle, sites in cycles.items():
                for site, files in sites.items():
                    target = merged.setdefault(well, {}).setdefault(
                        cycle, {}).setdefault(site, {})
                    for token, file in files.items():
                        names = set(token.split("-"))
                        if names & listed:
                            key = token
                        elif number and names == {register_on}:
                            key = f"{register_on}@{number}"
                        else:
                            ignored[folder] += 1
                            continue
                        if key in target:
                            raise ValueError(tr(
                                "ops_ph_acquisitions gives well {well}, site "
                                "{site} two {token} files: {first} and "
                                "{second}", well=well, site=site, token=key,
                                first=target[key], second=file))
                        target[key] = file
    return merged, ignored


def _ph_shift_plane(plane: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """``plane`` moved by whole pixels, the uncovered border zero.

    :param plane: a 2-D image.
    :param dy: rows down.
    :param dx: columns right.
    :returns: a new array of the same shape and dtype.
    """
    out = np.zeros_like(plane)
    height, width = plane.shape
    dy = int(max(-height, min(height, dy)))
    dx = int(max(-width, min(width, dx)))
    if dy >= 0:
        rows_from, rows_to = slice(0, height - dy), slice(dy, height)
    else:
        rows_from, rows_to = slice(-dy, height), slice(0, height + dy)
    if dx >= 0:
        cols_from, cols_to = slice(0, width - dx), slice(dx, width)
    else:
        cols_from, cols_to = slice(-dx, width), slice(0, width + dx)
    out[rows_to, cols_to] = plane[rows_from, cols_from]
    return out


def _ph_read_field(files: Mapping[str, str], channels: Sequence[str],
                   unreadable: list, secondary: Optional[Mapping[str, Any]] = None
                   ) -> Optional[np.ndarray]:
    """One phenotype field as ``(H, W, C)`` float32, focal planes maxed.

    :param files: ``channel token -> path`` for the site.
    :param channels: which channels, in order.
    :param unreadable: ``(path, reason)`` is appended for a file that fails.
    :param secondary: for a plate imaged in two acquisitions,
        ``{"channels": [...], "dy": int, "dx": int}``: those channels are
        read from the second acquisition's file and moved by the field's
        registration shift (:func:`_ph_secondary_shifts`); every other
        channel only from the primary acquisition's files.
    :returns: the stack, or None when any channel is missing or unreadable.

    THE FOCAL PLANES ARE MAXIMUM-PROJECTED, which is what Funk et al.'s
    ``apply_illumination_correction(zproject=True)`` does before their
    segmentation: the reference plate's phenotype files hold four.
    """
    import tifffile

    moved: set = set()
    shift = (0, 0)
    if secondary and secondary.get("channels"):
        primary_files, second_files = _ph_split_files(
            files, secondary["channels"])
        sources = _plane_sources(primary_files)
        extra = _plane_sources(second_files)
        moved = {str(c).upper() for c in secondary["channels"]}
        for name in moved:
            if name in extra:
                sources[name] = extra[name]
            else:
                sources.pop(name, None)
        shift = (int(secondary.get("dy") or 0), int(secondary.get("dx") or 0))
    else:
        sources = _plane_sources(files)
    cache: Dict[str, Tuple[np.ndarray, str]] = {}
    planes = []
    for channel in channels:
        source = sources.get(channel)
        if source is None:
            unreadable.append((str(files), f"no {channel} plane"))
            return None
        path, plane = source
        if path not in cache:
            try:
                with tifffile.TiffFile(path) as handle:
                    series = handle.series[0]
                    cache[path] = (np.asarray(series.asarray()),
                                   str(getattr(series, "axes", "") or ""))
            except (OSError, ValueError, IndexError,
                    getattr(tifffile, "TiffFileError", ValueError)) as failure:
                unreadable.append((path, f"{type(failure).__name__}: "
                                         f"{failure}"[:200]))
                return None
        array, axes = cache[path]
        if plane is not None:
            axis = _channel_axis(array, axes) if array.ndim >= 3 else -1
            if axis < 0 or array.shape[axis] <= plane:
                unreadable.append((path, f"has no plane {plane}"))
                return None
            array = np.take(array, plane, axis=axis)
        while array.ndim > 2:
            array = array.max(axis=0)
        if channel.upper() in moved and shift != (0, 0):
            array = _ph_shift_plane(array, *shift)
        planes.append(array.astype(np.float32))
    return np.stack(planes, axis=-1)


def _ph_downscale(plane: np.ndarray, factor: int) -> np.ndarray:
    """``plane`` block-averaged by ``factor``, edges padded by reflection."""
    if factor <= 1:
        return plane
    height = -(-plane.shape[0] // factor) * factor
    width = -(-plane.shape[1] // factor) * factor
    padded = np.pad(plane, ((0, height - plane.shape[0]),
                            (0, width - plane.shape[1])), mode="reflect")
    return padded.reshape(height // factor, factor, width // factor,
                          factor).mean(axis=(1, 3))


def _ph_upscale(mask: np.ndarray, factor: int, shape) -> np.ndarray:
    """A label mask repeated back up to ``shape``."""
    mask = np.asarray(mask)
    if factor > 1:
        mask = mask.repeat(factor, axis=0).repeat(factor, axis=1)
    return np.ascontiguousarray(mask[:shape[0], :shape[1]]).astype(np.int32)


def _ph_segment_field(stack: np.ndarray, channels: Sequence[str], models,
                      settings: Mapping[str, Any]):
    """Nuclei and cells of one field, paired so nucleus k is in cell k.

    :param stack: the ``(H, W, C)`` field.
    :param channels: the stack's channel names.
    :param models: ``(nucleus_model, cell_model)``.
    :param settings: read for ``ops_ph_cell_channel``, ``ops_ph_downsample``,
        ``ops_ph_batch_size`` and the two diameters.
    :returns: ``(nuclei, cells, counts)``.

    NUCLEI ON DAPI, CELLS ON THE CELL STAIN WITH DAPI AS THE SECOND
    CHANNEL, as the Cellpose cytoplasm models are trained and as Funk et
    al. segment cells on phalloidin around their nuclei. Both run on the
    field block-averaged by ``ops_ph_downsample`` and the masks are
    repeated back up: at 2 a nucleus is still about 20 px across, which is
    what the models are trained on, and the cost falls fourfold.
    """
    from .ops_sample import pair_nuclei_with_cells
    from .qt.i18n import tr

    factor = max(1, int(_setting_number(settings.get("ops_ph_downsample"),
                                        "ops_ph_downsample", 2)))
    batch = max(1, int(_setting_number(settings.get("ops_ph_batch_size"),
                                       "ops_ph_batch_size", 8)))
    cell_channel = str(settings.get("ops_ph_cell_channel") or "AF750")
    if _NUCLEAR not in channels or cell_channel not in channels:
        raise ValueError(tr(
            "ops_ph_channels {channels} must name {nuclear} and the cell "
            "channel {cell}", channels=list(channels), nuclear=_NUCLEAR,
            cell=cell_channel))
    dna = _ph_downscale(stack[..., channels.index(_NUCLEAR)], factor)
    body = _ph_downscale(stack[..., channels.index(cell_channel)], factor)
    extra = {}
    for key, name in (("ops_ph_nucleus_diameter", "nucleus"),
                      ("ops_ph_cell_diameter", "cell")):
        value = settings.get(key)
        extra[name] = {"diameter": float(value) / factor} if value else {}
    nucleus_model, cell_model = models
    nuclei = _governed_eval(nucleus_model, dna, site='ops_phenotype',
                            settings=settings, batch_size=batch,
                            **extra["nucleus"])[0]
    cells = _governed_eval(cell_model, np.stack([body, dna], axis=-1),
                           site='ops_phenotype', settings=settings,
                           batch_size=batch, channel_axis=-1,
                           **extra["cell"])[0]
    shape = stack.shape[:2]
    return pair_nuclei_with_cells(_ph_upscale(nuclei, factor, shape),
                                  _ph_upscale(cells, factor, shape))


def _ph_cells_path(destination: str, well: str) -> str:
    """Where one well's per-cell phenotype table is written."""
    return os.path.join(destination, f"ops_phenotype_cells_{well}.parquet")


def _ph_missing_path(destination: str, well: str) -> str:
    """Where one well's fields missing from the source are recorded."""
    return os.path.join(destination, f"ops_missing_fields_{well}.parquet")


def _ph_missing_fields(destination: str, well: str) -> Dict[int, str]:
    """``site -> reason`` for the fields a run found missing or unreadable."""
    import pandas as pd

    path = _ph_missing_path(destination, well)
    if not os.path.isfile(path):
        return {}
    table = pd.read_parquet(path, columns=["site", "reason"])
    return {int(s): str(r) for s, r in zip(table["site"], table["reason"])}


def _ph_record_missing(plate: str, well: str, destination: str,
                       found: Mapping[int, str]) -> None:
    """Add ``found`` (``site -> reason``) to the well's missing-field record."""
    import pandas as pd

    if not found:
        return
    known = _ph_missing_fields(destination, well)
    known.update({int(s): str(r) for s, r in found.items()})
    table = pd.DataFrame([{"plate": plate, "well": well, "site": s,
                           "reason": r} for s, r in sorted(known.items())])
    path = _ph_missing_path(destination, well)
    table.to_parquet(path + ".part", index=False)
    os.replace(path + ".part", path)


def _ph_ownership_path(destination: str, well: str) -> str:
    """Where one well's per-cell ownership flags are written."""
    return os.path.join(destination, f"ops_ownership_{well}.parquet")


def _ph_fieldcheck_path(destination: str, well: str) -> str:
    """Where one well's per-field registration check is written."""
    return os.path.join(destination, f"ops_fieldcheck_{well}.parquet")


def _ph_maskpass_path(destination: str, well: str) -> str:
    """Where one well's per-cell mask pass table is written."""
    return os.path.join(destination, f"ops_maskpass_{well}.parquet")


#: The rules ``ops_ownership`` may name, the first being the default.
_PH_OWNERSHIP_RULES = ("closer_match", "a1", "a1b")


def _ph_ownership_rule(settings: Mapping[str, Any]) -> str:
    """The ``ops_ownership`` rule, checked against :data:`_PH_OWNERSHIP_RULES`."""
    from .qt.i18n import tr

    rule = str(settings.get("ops_ownership") or "closer_match").strip().lower()
    if rule not in _PH_OWNERSHIP_RULES:
        raise ValueError(tr(
            "ops_ownership must be one of {rules}; got {value}",
            rules=list(_PH_OWNERSHIP_RULES), value=rule))
    return rule


#: The modes ``ops_flatfield`` may name, the first being the default.
_PH_FLATFIELD_MODES = ("none", "analytic", "exact")


def _ph_flatfield_mode(settings: Mapping[str, Any]) -> str:
    """The ``ops_flatfield`` mode, checked against :data:`_PH_FLATFIELD_MODES`."""
    from .qt.i18n import tr

    mode = str(settings.get("ops_flatfield") or "none").strip().lower()
    if mode not in _PH_FLATFIELD_MODES:
        raise ValueError(tr(
            "ops_flatfield must be one of {modes}; got {value}",
            modes=list(_PH_FLATFIELD_MODES), value=mode))
    return mode


def _ph_flat_folder(settings: Mapping[str, Any], destination: str) -> str:
    """The folder of the plate's flat-field maps: ``ops_flatfield_maps``,
    else ``<dst_root>/flatfield``."""
    return str(settings.get("ops_flatfield_maps")
               or os.path.join(destination, "flatfield"))


def _ph_options_marker(folder: str, options: Mapping[str, Any]) -> None:
    """Refuse to mix per-field parts made with different options.

    :param folder: a folder of per-field parts.
    :param options: what this run makes the parts with; every option's
        default is None or False.
    :raises ValueError: when parts made with other options are already
        there. A field that is not redone keeps its part, so a well would
        otherwise mix corrected and uncorrected fields without notice.

    A folder with parts but no marker was made before the options existed,
    so with every option at its default.
    """
    from .qt.i18n import tr

    path = os.path.join(folder, "part_options.json")
    now = json.loads(json.dumps(dict(options)))
    earlier = None
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as handle:
            earlier = json.load(handle)
        earlier = {**{key: (False if isinstance(value, bool) else None)
                      for key, value in now.items()}, **earlier}
    elif any(name.endswith((".parquet", ".npz")) for name in os.listdir(folder)):
        earlier = {key: (False if isinstance(value, bool) else None)
                   for key, value in now.items()}
    if earlier is not None and earlier != now:
        raise ValueError(tr(
            "{folder} holds parts made with {earlier}; this run asks for "
            "{now}. Move the folder away or rerun with the same options.",
            folder=folder, earlier=earlier, now=now))
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(now, handle, indent=1)


#: Flat-field planes a worker process has loaded, by folder and field shape.
_PH_FLAT_CACHE: Dict[Any, Any] = {}


def _ph_flat_planes(folder: str, channels: Sequence[str], shape):
    """The plate's flat planes at field resolution, loaded once per process.

    :returns: ``(planes, offsets, inverse)``: float32 flats as the exact
        correction divides by them, the offsets, and float64 ``1 / flat``
        as the mask pass averages it.
    """
    from .ops_flatfield import flat_lookups, flat_planes, load_flat_maps

    key = (os.path.abspath(folder), tuple(channels), tuple(shape))
    if key not in _PH_FLAT_CACHE:
        maps, offsets, info = load_flat_maps(folder, channels)
        planes = flat_planes(maps, shape, lookups=flat_lookups(info))
        inverse = {c: 1.0 / planes[c].astype(np.float64) for c in planes}
        _PH_FLAT_CACHE.clear()
        _PH_FLAT_CACHE[key] = (planes, offsets, inverse)
    return _PH_FLAT_CACHE[key]


def _ph_flat_blocks(task: Mapping[str, Any]) -> Dict[str, Any]:
    """Block means of one field; the ``ph_flatfield`` worker."""
    from .ops_flatfield import field_blocks

    unreadable: list = []
    stack = _ph_read_field(task["files"], task["channels"], unreadable)
    return {"well": task["well"], "site": task["site"],
            "blocks": None if stack is None else field_blocks(stack),
            "unreadable": unreadable}


def _ph_flatfield(plate: str, wells: Sequence[str], phenotype_index,
                  settings: Mapping[str, Any], destination: str,
                  db: str) -> Dict[str, Any]:
    """Build the plate's flat-field maps once; the ``ph_flatfield`` phase.

    :param plate: the plate name.
    :param wells: the wells whose fields may be drawn.
    :param phenotype_index: the phenotype tiles, ``{well: {cycle: {site:
        files}}}``.
    :param settings: read for ``ops_flatfield_maps``, ``ops_flatfield_fields``,
        ``ops_flatfield_field_list``, ``ops_ph_channels`` and ``n_workers``.
    :param destination: the run folder.
    :param db: the measurements database, for the placed sites.
    :returns: the report, with the maps' md5 and the fields used.

    Maps that exist are reused and never rebuilt: every analysis records the
    md5 of the maps it used, so a folder's maps never change. The fields are
    ``ops_flatfield_fields`` drawn at random (seed 0) from the placed fields
    of the wells run, or exactly those ``ops_flatfield_field_list`` names,
    as ``[[well, site], ...]`` -- the ``fields.json`` a build writes, so a
    plate's maps can be rebuilt from the same fields.
    """
    import multiprocessing

    from .ops_flatfield import (build_flat_maps, choose_flat_fields,
                                load_flat_maps, write_flat_maps)
    from .qt.i18n import tr
    from .resource_log import _parallel_process_executor as ProcessPoolExecutor

    started = time.perf_counter()
    channels = _ph_channels(settings)
    folder = _ph_flat_folder(settings, destination)
    if os.path.isfile(os.path.join(folder, "flat_info.json")):
        _maps, offsets, info = load_flat_maps(folder, channels)
        return {"reused": folder, "md5": info.get("md5"),
                "offsets": offsets, "n_fields": info.get("n_fields")}
    files = {}
    for well in wells:
        index = phenotype_index.get(well) or {}
        files[well] = index.get(min(index), {}) if index else {}
    listed = settings.get("ops_flatfield_field_list")
    if listed:
        with open(str(listed), encoding="utf-8") as handle:
            chosen = [(str(w), int(site)) for w, site in json.load(handle)]
        missing = [f for f in chosen if f[1] not in files.get(f[0], {})]
        if missing:
            raise ValueError(tr(
                "ops_flatfield_field_list names fields without phenotype "
                "tiles: {missing}", missing=missing[:5]))
    else:
        pool = []
        for well in wells:
            placed = _well_rows(db, "ops_phenotype", plate, well)
            sites = sorted(int(s) for s in placed["site"]) if placed is not \
                None and len(placed) else sorted(files.get(well, {}))
            pool.extend((well, s) for s in sites if s in files.get(well, {}))
        count = int(_setting_number(settings.get("ops_flatfield_fields"),
                                    "ops_flatfield_fields", 204))
        chosen = choose_flat_fields(pool, count, seed=0)
    if not chosen:
        raise ValueError(tr("no phenotype field to build flat-field maps from"))
    tasks = [{"well": w, "site": site, "files": files[w][site],
              "channels": channels} for w, site in chosen]
    workers = max(1, min(int(settings.get("n_workers") or 1), len(tasks)))
    if workers == 1:
        results = [_ph_flat_blocks(task) for task in tasks]
    else:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(workers, mp_context=context) as pool_:
            results = list(pool_.map(_ph_flat_blocks, tasks))
    unreadable = [u for r in results for u in r["unreadable"]]
    if unreadable:
        raise ValueError(tr(
            "{count} flat-field fields could not be read, first {first}",
            count=len(unreadable), first=unreadable[0]))
    blocks = np.stack([r["blocks"] for r in results])
    maps, method = build_flat_maps(blocks, channels,
                                   polynomial_channels=(_NUCLEAR,))
    info = write_flat_maps(folder, plate, maps, method, chosen, extra={
        "wells_available": list(wells),
        "built": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
    _say(tr("{plate} flat-field maps from {fields} fields written to "
            "{folder}", plate=plate, fields=len(chosen), folder=folder))
    return {"built": folder, "md5": info["md5"], "n_fields": len(chosen),
            "fields_by_well": info["fields_by_well"],
            "offsets": {c: m["offset"] for c, m in method.items()},
            "workers": workers,
            "seconds": round(time.perf_counter() - started, 1)}


def _ph_mask_field(task: Mapping[str, Any]) -> Dict[str, Any]:
    """The mask pass of one field; the ``ph_masks`` worker."""
    from .ops_flatfield import mask_pass

    with np.load(task["mask_path"]) as saved:
        nuclei = saved["nuclei"]
        cells = saved["cells"]
    inverse = None
    if task.get("flat_folder"):
        inverse = _ph_flat_planes(task["flat_folder"], task["channels"],
                                  cells.shape)[2]
    table = mask_pass(nuclei, cells, inverse)
    table.insert(0, "site", int(task["site"]))
    table.to_parquet(task["part_path"] + ".part", index=False)
    os.replace(task["part_path"] + ".part", task["part_path"])
    return {"site": task["site"], "cells": int(len(table))}


def _ph_masks(plate: str, well: str, settings: Mapping[str, Any],
              destination: str) -> Dict[str, Any]:
    """Pass over one well's saved masks; the ``ph_masks`` phase.

    :param plate: the plate name.
    :param well: the well name.
    :param settings: read for ``ops_flatfield``, ``ops_flatfield_maps``,
        ``ops_ph_channels`` and ``n_workers``.
    :param destination: the run folder.
    :returns: the report.

    Writes ``ops_maskpass_<well>.parquet``, one row per cell
    (:func:`spacr.ops_flatfield.mask_pass`): whether the cell or its
    nucleus touches the field's edge, its pixel counts and centroids and,
    with ``ops_flatfield='analytic'``, the mean flat-field gain under each
    compartment. Resumable per field, as the measurement is. Reads the
    masks only, never the images, so a measured run gets its edge flags
    and gains in minutes rather than the hours a re-measure takes.
    """
    import multiprocessing

    import pandas as pd

    from .qt.i18n import tr
    from .resource_log import _parallel_process_executor as ProcessPoolExecutor
    from .tabular import write_table

    started = time.perf_counter()
    channels = _ph_channels(settings)
    mask_dir = os.path.join(destination, well, "phenotype_masks")
    if not os.path.isdir(mask_dir):
        raise ValueError(tr(
            "well {well} has no masks in {folder}; run ph_segment first",
            well=well, folder=mask_dir))
    sites = sorted(int(name[5:-4]) for name in os.listdir(mask_dir)
                   if name.startswith("site-") and name.endswith(".npz")
                   and ".part" not in name)
    wanted = _ph_sites(settings.get("ops_ph_sites"))
    if wanted is not None:
        sites = [s for s in sites if s in wanted]
    flat_folder = None
    options: Dict[str, Any] = {"flatfield": None}
    if _ph_flatfield_mode(settings) == "analytic":
        from .ops_flatfield import load_flat_maps, record_flat_md5

        flat_folder = _ph_flat_folder(settings, destination)
        flat_info = load_flat_maps(flat_folder, channels)[2]
        record_flat_md5(destination, flat_info)
        options["flatfield"] = flat_info["md5"]
    part_dir = os.path.join(destination, well, "phenotype_maskpass")
    os.makedirs(part_dir, exist_ok=True)
    _ph_options_marker(part_dir, options)

    parts = {s: os.path.join(part_dir, f"site-{s}.parquet") for s in sites}

    tasks = [{"site": s, "mask_path": os.path.join(mask_dir, f"site-{s}.npz"),
              "part_path": parts[s], "flat_folder": flat_folder,
              "channels": channels}
             for s in sites if not os.path.isfile(parts[s])]
    workers = max(1, min(int(settings.get("n_workers") or 1),
                         len(tasks) or 1))
    if workers == 1:
        for task in tasks:
            _ph_mask_field(task)
    else:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(workers, mp_context=context) as pool:
            for count, _result in enumerate(
                    pool.map(_ph_mask_field, tasks, chunksize=2), start=1):
                if count % 100 == 0:
                    _say(tr("{well} ph_masks: {count} of {total} fields",
                            well=well, count=count, total=len(tasks)))
    frames = [f for f in (pd.read_parquet(parts[s]) for s in sites)
              if len(f)]
    table = (pd.concat(frames, ignore_index=True) if frames
             else pd.DataFrame(columns=["site", "label"]))
    table.insert(0, "well", well)
    table.insert(0, "plate", plate)
    write_table(table, _ph_maskpass_path(destination, well),
                canonicalise=False)
    report = {"fields": len(sites), "passed": len(tasks),
              "resumed": len(sites) - len(tasks), "cells": int(len(table)),
              "touch_field_edge": int(table["touches_field_edge"].sum())
              if "touches_field_edge" in table else 0,
              "flatfield_md5": options["flatfield"], "workers": workers,
              "table": _ph_maskpass_path(destination, well),
              "seconds": round(time.perf_counter() - started, 1)}
    _say(tr("{well} ph_masks: {cells} cells in {fields} fields", well=well,
            cells=report["cells"], fields=report["fields"]))
    return report


#: How many fields this process has measured, so the report can tell the
#: first field of each measurement worker (which pays the GPU kernel
#: compile, item 872 B15) from the rest.
_PH_WORKER_FIELDS = {"count": 0}


def _ph_measure_table(task: Mapping[str, Any]):
    """One segmented phenotype field measured: ``(table or None, unreadable, shape)``.

    :param task: as :func:`_ph_measure_field` takes it; ``staged``, when
        set, is the field's stack staged by :class:`spacr.ops_run_infra.FieldStager`,
        read in place of the raw files.
    """
    from .ops_sample import measure_phenotype_field

    unreadable: list = []
    if task.get("staged"):
        stack = np.load(task["staged"])
    else:
        stack = _ph_read_field(task["files"], task["channels"], unreadable,
                               task.get("secondary"))
    if stack is None:
        return None, unreadable, None
    with np.load(task["mask_path"]) as saved:
        nuclei = saved["nuclei"].astype(np.int32)
        cells = saved["cells"].astype(np.int32)
    if task.get("flat_folder"):
        from .ops_flatfield import correct_field

        planes, offsets, _inverse = _ph_flat_planes(
            task["flat_folder"], task["channels"], stack.shape[:2])
        stack = correct_field(stack, planes, offsets, task["channels"])
    if task.get("measure_gpu"):
        os.environ.setdefault("CUPY_CACHE_DIR", os.path.join(
            os.path.expanduser("~"), ".spacr", "cupy_cache"))
    table = measure_phenotype_field(
        nuclei, cells, stack, task["channels"],
        texture=bool(task["texture"]),
        funk_features=bool(task.get("funk_features", False)),
        zernike=bool(task.get("zernike", False)),
        gpu=bool(task.get("measure_gpu", False)),
        profile_features=task.get("profile_features"),
        granularity_subsample=float(task.get("granularity_subsample") or 1.0))
    table.insert(0, "site", int(task["site"]))
    return table, unreadable, tuple(int(v) for v in stack.shape[:2])


def _ph_measure_field(task: Mapping[str, Any]) -> Dict[str, Any]:
    """Measure one segmented phenotype field; the ``ph_measure`` worker.

    :param task: ``site``, ``files``, ``channels``, ``mask_path``,
        ``cell_path`` and ``texture``; optionally ``flat_folder``, whose
        maps correct the field before it is measured
        (:func:`spacr.ops_flatfield.correct_field`), ``funk_features``,
        ``zernike``, ``profile_features``, ``measure_gpu``, which measures intensity and
        texture on the card, and ``staged``, a staged copy of the field
        read in place of its raw files.
    :returns: ``site``, ``cells``, ``seconds``, ``unreadable`` and
        ``first_in_worker``, True for the first field this process measured.

    Module level so a spawned process can run it. It reads its own field
    and masks and writes its own Parquet part, so nothing large crosses the
    process boundary. The correction is applied to the copy in memory; the
    raw files are never changed.
    """
    tick = time.perf_counter()
    _PH_WORKER_FIELDS["count"] += 1
    first = _PH_WORKER_FIELDS["count"] == 1
    table, unreadable, shape = _ph_measure_table(task)
    if table is None:
        return {"site": task["site"], "cells": None, "seconds": 0.0,
                "unreadable": unreadable, "first_in_worker": first}
    table.to_parquet(task["cell_path"] + ".part", index=False)
    os.replace(task["cell_path"] + ".part", task["cell_path"])
    return {"site": task["site"], "cells": int(len(table)),
            "seconds": time.perf_counter() - tick, "unreadable": unreadable,
            "shape": shape, "first_in_worker": first}


def _ph_gpu_check(task: Mapping[str, Any], rtol: float = 1e-4
                  ) -> Dict[str, Any]:
    """Measure one field on the card and on the CPU and compare (item 872, B7).

    :param task: a ``ph_measure`` task with ``measure_gpu`` set.
    :param rtol: the relative tolerance a column must keep.
    :returns: ``site`` and :func:`spacr.ops_run_infra.compare_tables` of the
        GPU table against the CPU table; ``outside`` lists the columns
        beyond ``rtol``. The run then goes on: the check reports, it does
        not stop the run.
    """
    from .ops_run_infra import compare_tables

    gpu, _unreadable, _shape = _ph_measure_table({**task, "measure_gpu": True})
    cpu, _unreadable, _shape = _ph_measure_table({**task, "measure_gpu": False})
    if gpu is None or cpu is None:
        return {"site": task["site"], "skipped": "the field is unreadable"}
    return {"site": task["site"], **compare_tables(gpu, cpu, rtol=rtol)}


def _ph_profile_features(settings: Mapping[str, Any]) -> Optional[List[str]]:
    """The profiling families ``ops_ph_profile_features`` asks for, or None.

    :param settings: read for ``ops_ph_profile_features``
        (:func:`spacr.measure.profile_feature_families`).
    :returns: the family names, or None when the setting is empty, so a run
        without it measures, and marks its parts, exactly as before.
    :raises ValueError: for a name that is not a family.
    """
    from .measure import profile_feature_families

    families = profile_feature_families(
        settings.get("ops_ph_profile_features"))
    return list(families) or None


def _ph_granularity_subsample(settings: Mapping[str, Any]
                               ) -> Optional[float]:
    """``ops_ph_granularity_subsample`` when granularity is measured, else None.

    :param settings: read for ``ops_ph_profile_features`` and
        ``ops_ph_granularity_subsample`` (default 1.0).
    :returns: the subsample factor, or None when no granularity is
        measured, so the part marker of a run without it is unchanged.
    """
    families = _ph_profile_features(settings) or []
    if "granularity" not in families:
        return None
    return float(settings.get("ops_ph_granularity_subsample") or 1.0)


def _ph_measure_workers(settings: Mapping[str, Any], fields: int,
                        measure_gpu: bool) -> int:
    """How many fields are measured at once.

    :param settings: read for ``ops_ph_measure_workers`` (``n_workers`` when
        empty) and, with ``ops_gpu_admission`` and ``ops_gpu_vram_cap_gb``
        set on a GPU measurement, the card's memory in use
        (:meth:`spacr.gpu_admission.GpuAdmission.workers_under_cap`).
    :param fields: how many fields there are to measure.
    :param measure_gpu: whether the measurement runs on the card.
    :returns: at least 1 and at most ``fields``.
    """
    asked = settings.get("ops_ph_measure_workers") or settings.get("n_workers")
    workers = max(1, min(int(asked or 1), fields or 1))
    if (measure_gpu and settings.get("ops_gpu_admission")
            and settings.get("ops_gpu_vram_cap_gb")):
        from .gpu_admission import admission_from_settings

        workers = admission_from_settings(settings, "").workers_under_cap(
            workers)
    return workers


def _ph_fields(db: str, plate: str, well: str, phenotype_files,
               settings: Mapping[str, Any], gpu: bool, destination: str, *,
               segment: bool, measure: bool) -> Dict[str, Any]:
    """Segment and/or measure every placed phenotype field of one well.

    :param db: the measurements database.
    :param plate: the plate name.
    :param well: the well name.
    :param phenotype_files: the phenotype index for this well.
    :param settings: the OPS settings; ``ops_ph_measure_workers`` (else
        ``n_workers``) sets how many fields are measured at once.
    :param gpu: segment on the card.
    :param destination: the run folder.
    :param segment: run ``ph_segment``.
    :param measure: run ``ph_measure``.
    :returns: the report.

    RESUMABLE PER FIELD. Each field's masks go to
    ``<well>/phenotype_masks/site-<n>.npz`` and its cells to
    ``<well>/phenotype_cells/site-<n>.parquet``; a field whose file exists
    is not redone, so a run stopped at field 900 resumes at 901. The well's
    table, ``ops_phenotype_cells_<well>.parquet``, is the concatenation and
    is rewritten at the end.

    SEGMENTATION RUNS IN THIS PROCESS AND MEASUREMENT IN WORKERS: Cellpose
    already uses every core (or the card), while spaCR's measurement of a
    field is one core for a few minutes on the reference plate's 2,960 px
    fields, so measuring fields side by side is what keeps the machine busy.

    FIELD BY FIELD WHEN ASKED (item 872, B1 and B2). ``ops_ph_pipeline``
    hands each field to the measurement workers as soon as its masks are
    saved, so the card segments field n + 1 while field n is measured, in
    place of segmenting the whole well first. ``ops_ph_stage_dir`` stages
    each field's maximum projection on local disk with
    ``ops_ph_stage_threads`` reader threads, at most ``ops_ph_stage_ahead``
    fields ahead (:class:`spacr.ops_run_infra.FieldStager`): each raw file
    is then read once, not once to segment and once to measure, and the
    local copy is deleted when the field is done. Staging without the
    pipeline measures each field right after segmenting it. Masks and
    tables are the same either way.

    With ``ops_ph_measure_gpu`` and ``ops_ph_measure_gpu_check`` (the
    default), the first field to measure is measured on the card and on
    the CPU first and the columns beyond relative 1e-4 are reported under
    ``ph_measure_gpu_check`` (item 872, B7).
    """
    import multiprocessing
    from concurrent.futures import ThreadPoolExecutor as _OneThread

    import pandas as pd

    from .qt.i18n import tr
    from .resource_log import _parallel_process_executor as ProcessPoolExecutor

    started = time.perf_counter()
    channels = _ph_channels(settings)
    rows = _well_rows(db, "ops_phenotype", plate, well)
    if rows is None or rows.empty:
        raise ValueError(tr(
            "well {well} has no ops_phenotype rows in {db}; run the "
            "phenotype phase first", well=well, db=db))
    wanted = _ph_sites(settings.get("ops_ph_sites"))
    files = phenotype_files.get(min(phenotype_files), {}) if phenotype_files else {}
    sites = [int(s) for s in sorted(rows["site"])
             if (wanted is None or int(s) in wanted) and int(s) in files]
    mask_dir = os.path.join(destination, well, "phenotype_masks")
    cell_dir = os.path.join(destination, well, "phenotype_cells")
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(cell_dir, exist_ok=True)

    def mask_path(site: int) -> str:
        """Where one field's masks are kept."""
        return os.path.join(mask_dir, f"site-{site}.npz")

    def cell_path(site: int) -> str:
        """Where one field's cell table is kept."""
        return os.path.join(cell_dir, f"site-{site}.parquet")

    unreadable: list = []
    report: Dict[str, Any] = {"fields": len(sites)}
    field_shape = None
    secondary = _ph_secondary_by_site(db, plate, well, settings)
    flat_folder = None
    options: Dict[str, Any] = {}
    if measure:
        options = {
            "flatfield": None,
            "funk_features": bool(settings.get("ops_ph_funk_features",
                                               False)),
            "zernike": bool(settings.get("ops_ph_zernike", False)),
            "profile_features": _ph_profile_features(settings),
            "granularity_subsample": _ph_granularity_subsample(settings),
            "measure_gpu": bool(settings.get("ops_ph_measure_gpu", False)),
            "secondary": _ph_secondary(settings) if secondary else None}
        if _ph_flatfield_mode(settings) == "exact":
            from .ops_flatfield import load_flat_maps, record_flat_md5

            flat_folder = _ph_flat_folder(settings, destination)
            flat_info = load_flat_maps(flat_folder, channels)[2]
            record_flat_md5(destination, flat_info)
            options["flatfield"] = flat_info["md5"]
        _ph_options_marker(cell_dir, options)

    def measure_task(site: int) -> Dict[str, Any]:
        """The ``ph_measure`` task of one field."""
        return {"site": site, "files": files[site], "channels": channels,
                "mask_path": mask_path(site), "cell_path": cell_path(site),
                "texture": bool(settings.get("ops_ph_texture", True)),
                "flat_folder": flat_folder,
                "funk_features": options["funk_features"],
                "zernike": options["zernike"],
                "profile_features": options["profile_features"],
                "granularity_subsample": options["granularity_subsample"],
                "measure_gpu": options["measure_gpu"],
                "secondary": secondary.get(site)}

    check_gpu = bool(measure and options["measure_gpu"]
                     and settings.get("ops_ph_measure_gpu_check", True))
    pipelined = bool(segment and measure and settings.get("ops_ph_pipeline"))
    stage_dir = settings.get("ops_ph_stage_dir") or None
    streamed = pipelined or bool(stage_dir)
    todo: List[int] = []
    found: Dict[int, str] = {}
    missing: Dict[int, str] = {}
    if segment:
        missing = ({} if settings.get("ops_retry_missing")
                   else _ph_missing_fields(destination, well))
        todo = [s for s in sites if not os.path.isfile(mask_path(s))
                and s not in missing]
        found = {
            int(s): "no file for the site in phenotype_source"
            for s in sorted(rows["site"])
            if (wanted is None or int(s) in wanted) and int(s) not in files
            and int(s) not in missing}
    to_measure = ({s for s in sites if not os.path.isfile(cell_path(s))
                   and (os.path.isfile(mask_path(s)) or s in todo)}
                  if measure else set())
    stager = None
    if stage_dir:
        from .ops_run_infra import FieldStager

        def stage_read(site: int):
            """One field's projected stack, for the stager."""
            lost: list = []
            return _ph_read_field(files[site], channels, lost,
                                  secondary.get(site)), lost

        stager = FieldStager(
            os.path.join(str(stage_dir), str(plate), str(well)),
            [s for s in sites if s in to_measure or s in todo], stage_read,
            ahead=int(_setting_number(settings.get("ops_ph_stage_ahead"),
                                      "ops_ph_stage_ahead", 48)),
            threads=int(_setting_number(settings.get("ops_ph_stage_threads"),
                                        "ops_ph_stage_threads", 4)))
    pool = None
    futures: List[Any] = []
    results: List[Dict[str, Any]] = []
    workers = 1
    if streamed and measure:
        workers = _ph_measure_workers(settings, len(to_measure),
                                      options["measure_gpu"])
        if pipelined:
            pool = (_OneThread(1) if workers == 1 else ProcessPoolExecutor(
                workers, mp_context=multiprocessing.get_context("spawn")))

    def measure_now(site: int) -> None:
        """Measure one field in the stream: in the pool, or here."""
        task = measure_task(site)
        if stager is not None:
            staged, lost = stager.path(site)
            if staged is None:
                unreadable.extend(lost)
                return
            task["staged"] = staged
        if check_gpu and "ph_measure_gpu_check" not in report:
            report["ph_measure_gpu_check"] = _ph_gpu_check(task)
        if pool is None:
            results.append(_ph_measure_field(task))
            if stager is not None:
                stager.release(site)
            return
        future = pool.submit(_ph_measure_field, task)
        if stager is not None:
            future.add_done_callback(lambda _done, s=site: stager.release(s))
        futures.append(future)

    stream_tick = time.perf_counter()
    try:
        if segment:
            models = None
            seconds = Counter()
            counts = Counter()
            order = ([s for s in sites if s in todo or s in to_measure]
                     if streamed else todo)
            for count, site in enumerate(order, start=1):
                if site not in todo:
                    measure_now(site)
                    continue
                tick = time.perf_counter()
                before = len(unreadable)
                if stager is not None:
                    stack, lost = stager.load(site)
                    unreadable.extend(lost)
                else:
                    stack = _ph_read_field(files[site], channels, unreadable,
                                           secondary.get(site))
                seconds["read"] += time.perf_counter() - tick
                if stack is None:
                    found[site] = "; ".join(str(r) for _p, r in
                                            unreadable[before:]) or "unreadable"
                    continue
                field_shape = field_shape or stack.shape[:2]
                if models is None:
                    nucleus_model = _cellpose_model(settings, gpu,
                                                    "ops_ph_nucleus_model")
                    same = (str(settings.get("ops_ph_cell_model") or "cpsam")
                            == str(settings.get("ops_ph_nucleus_model")
                                   or "cpsam"))
                    models = (nucleus_model, nucleus_model if same else
                              _cellpose_model(settings, gpu,
                                              "ops_ph_cell_model"))
                tick = time.perf_counter()
                nuclei, cells, paired = _ph_segment_field(stack, channels,
                                                          models, settings)
                seconds["segment"] += time.perf_counter() - tick
                counts.update(paired)
                np.savez_compressed(mask_path(site) + ".part.npz",
                                    nuclei=nuclei.astype(np.uint32),
                                    cells=cells.astype(np.uint32))
                os.replace(mask_path(site) + ".part.npz", mask_path(site))
                _say(tr("{well} ph_segment: site {site}, {cells} cells, "
                        "{count} of {total}", well=well, site=site,
                        cells=paired["paired"], count=count,
                        total=len(order)))
                if streamed and site in to_measure:
                    measure_now(site)
                elif stager is not None:
                    stager.release(site)
            fresh = sum(1 for s in todo if os.path.isfile(mask_path(s)))
            _ph_record_missing(plate, well, destination, found)
            report["ph_segment"] = {
                "missing_recorded": sorted(found),
                "missing_skipped": sorted(
                    s for s in missing if (wanted is None or s in wanted)
                    and not os.path.isfile(mask_path(s))),
                "segmented": fresh, "resumed": len(sites) - len(todo),
                "objects": dict(counts),
                "seconds": {k: round(v, 1) for k, v in seconds.items()},
                "segment_seconds_per_field": (
                    round(seconds["segment"] / fresh, 2) if fresh else None),
                "read_seconds_per_field": (round(seconds["read"] / fresh, 2)
                                           if fresh else None),
            }
        elif streamed and measure:
            for site in [s for s in sites if s in to_measure]:
                measure_now(site)
        for future in futures:
            results.append(future.result())
    finally:
        if pool is not None:
            pool.shutdown(wait=True)
        if stager is not None:
            stager.close()
    if streamed:
        report["ph_stream"] = {
            "pipelined": pipelined, "staged": stager is not None,
            "workers": workers,
            "wall_seconds": round(time.perf_counter() - stream_tick, 1),
            "stage_peak_files": stager.peak_files if stager else None,
            "stage_peak_bytes": stager.peak_bytes if stager else None}
    if measure:
        tick = time.perf_counter()
        if not streamed:
            tasks = [measure_task(s) for s in sites
                     if os.path.isfile(mask_path(s))
                     and not os.path.isfile(cell_path(s))]
            if check_gpu and tasks:
                report["ph_measure_gpu_check"] = _ph_gpu_check(tasks[0])
            workers = _ph_measure_workers(settings, len(tasks),
                                          options["measure_gpu"])
            if workers == 1:
                results = [_ph_measure_field(task) for task in tasks]
            else:
                context = multiprocessing.get_context("spawn")
                with ProcessPoolExecutor(workers, mp_context=context) as pool:
                    for count, result in enumerate(
                            pool.map(_ph_measure_field, tasks), start=1):
                        results.append(result)
                        if count % 10 == 0:
                            _say(tr("{well} ph_measure: {count} of {total} "
                                    "fields", well=well, count=count,
                                    total=len(tasks)))
        for result in results:
            unreadable.extend(result["unreadable"])
            field_shape = field_shape or result.get("shape")
        parts = [cell_path(s) for s in sites if os.path.isfile(cell_path(s))]
        frames = [f for f in (pd.read_parquet(p) for p in parts) if len(f)]
        cells_table = (pd.concat(frames, ignore_index=True) if frames
                       else pd.DataFrame(columns=["site", "label"]))
        cells_table.insert(0, "well", well)
        cells_table.insert(0, "plate", plate)
        from .tabular import write_table

        write_table(cells_table, _ph_cells_path(destination, well),
                    canonicalise=False)
        measured = [r for r in results if r["cells"] is not None]
        first = [r["seconds"] for r in measured if r.get("first_in_worker")]
        rest = [r["seconds"] for r in measured if not r.get("first_in_worker")]
        report["ph_measure"] = {
            "measured": len(measured), "resumed": len(parts) - len(measured),
            "workers": workers, "cells": int(len(cells_table)),
            "features": len(_ph_feature_columns(cells_table.columns)),
            "measure_seconds_per_field": (
                round(float(np.mean([r["seconds"] for r in measured])), 2)
                if measured else None),
            "first_field_seconds_per_worker": [round(v, 2) for v in first],
            "later_field_seconds_median": (round(float(np.median(rest)), 2)
                                           if rest else None),
            "wall_seconds": round(time.perf_counter() - tick, 1),
            "table": _ph_cells_path(destination, well),
            "flatfield": ({"mode": "exact", "folder": flat_folder,
                           "md5": options["flatfield"]}
                          if flat_folder else None),
            "funk_features": options["funk_features"],
            "zernike": options["zernike"],
            "profile_features": options["profile_features"],
            "granularity_subsample": options["granularity_subsample"],
            "measure_gpu": options["measure_gpu"],
        }
    if field_shape is not None:
        _ph_transforms(db, plate, well, destination, tuple(field_shape))
    report["unreadable"] = unreadable
    report["total_seconds"] = round(time.perf_counter() - started, 1)
    _say(tr("{well} phenotype fields: {fields} in {seconds} s", well=well,
            fields=len(sites), seconds=report["total_seconds"]))
    return report


def _ph_field_check(plate: str, well: str, cells, candidates, refined_rows,
                    placed, target, by_site, threshold: float):
    """Which fields of a well the aggregate phase must leave out.

    :param plate: the plate name.
    :param well: the well name.
    :param cells: the well's phenotype cells (``site``, ``label``).
    :param candidates: every candidate match before an overlap keeps one
        copy, with its ``site``.
    :param refined_rows: per field, whether the transform re-solve worked.
    :param placed: the stored field placements.
    :param target: the sequencing nuclei, ``(y, x)`` in well pixels.
    :param by_site: the placement of each field.
    :param threshold: the merge distance.
    :returns: one row per field: ``cells``, ``matched``, ``link_rate``,
        ``sbs_nuclei_under_field``, ``chance_rate``, ``link_over_chance``,
        ``refined``, ``dropped`` and ``reason``.

    THE RULE IS THE PLATE A PILOT'S. The chance rate is the share of cells
    an unrelated field would link: the density of sequencing nuclei under
    the field's footprint times the area of the merge circle, ``pi *
    threshold ** 2``, as ``1 - exp(-density * area)``. A field is dropped
    when its transform re-solve failed and it links under three times
    chance, or when it links under twice chance.
    """
    import pandas as pd

    from .ops_merge import field_footprint

    refined = {int(r["site"]): r for r in refined_rows}
    cell_counts = cells.groupby("site").size()
    link_counts = candidates.groupby("site").size()
    rows = []
    for site in sorted(refined):
        record = by_site[site]
        r = refined[site]
        transform = (r["refined_scale"], np.array(
            [[np.cos(np.radians(r["refined_rotation_deg"])),
              -np.sin(np.radians(r["refined_rotation_deg"]))],
             [np.sin(np.radians(r["refined_rotation_deg"])),
              np.cos(np.radians(r["refined_rotation_deg"]))]]),
            np.array([r["refined_offset_y"], r["refined_offset_x"]]))
        low, high = field_footprint(
            transform, (float(record.field_height), float(record.field_width)))
        inside = int(((target[:, 0] > low[0]) & (target[:, 0] < high[0])
                      & (target[:, 1] > low[1])
                      & (target[:, 1] < high[1])).sum())
        area = float(np.prod(high - low))
        chance = 1.0 - np.exp(-inside / area * np.pi * float(threshold) ** 2)
        n = int(cell_counts.get(site, 0))
        k = int(link_counts.get(site, 0))
        rows.append({
            "plate": plate, "well": well, "site": site, "cells": n,
            "matched": k, "link_rate": k / max(n, 1),
            "sbs_nuclei_under_field": inside, "chance_rate": chance,
            "refined": int(r["refined"]),
            "refined_inliers": int(r["refined_inliers"])})
    check = pd.DataFrame(rows, columns=[
        "plate", "well", "site", "cells", "matched", "link_rate",
        "sbs_nuclei_under_field", "chance_rate", "refined",
        "refined_inliers"])
    check["link_over_chance"] = check["link_rate"] / check[
        "chance_rate"].clip(lower=1e-6)
    failed = (check["refined"] == 0) & (check["link_over_chance"] < 3.0)
    chance_only = check["link_over_chance"] < 2.0
    check["dropped"] = failed | chance_only
    check["reason"] = np.select(
        [failed, chance_only],
        ["transform refinement failed and links under 3x chance",
         "links under 2x chance"], default="")
    return check


#: The models ``ops_align_model`` may name, the first being the default.
_PH_ALIGN_MODELS = ("tile_affine", "similarity")


def _ph_align_options(settings: Mapping[str, Any]
                      ) -> Tuple[str, int, float, int]:
    """The ``ops_align_*`` settings of the merge phase, checked.

    :param settings: read for ``ops_align_model``,
        ``ops_align_tile_min_nuclei``, ``ops_align_match_radius`` and
        ``ops_align_trim_passes``.
    :returns: ``(model, min_nuclei, radius, passes)``.
    :raises ValueError: on an unknown model or a value below its minimum.
    """
    from .qt.i18n import tr

    model = str(settings.get("ops_align_model") or _PH_ALIGN_MODELS[0]
                ).strip().lower()
    if model not in _PH_ALIGN_MODELS:
        raise ValueError(tr(
            "ops_align_model must be one of {models}; got {value}",
            models=list(_PH_ALIGN_MODELS), value=model))
    min_nuclei = int(_setting_number(settings.get(
        "ops_align_tile_min_nuclei"), "ops_align_tile_min_nuclei", 200))
    radius = float(_setting_number(settings.get(
        "ops_align_match_radius"), "ops_align_match_radius", 3.0))
    passes = int(_setting_number(settings.get(
        "ops_align_trim_passes"), "ops_align_trim_passes", 3))
    if min_nuclei < 3 or radius <= 0 or passes < 1:
        raise ValueError(tr(
            "ops_align_tile_min_nuclei must be at least 3, "
            "ops_align_match_radius above 0 and ops_align_trim_passes at "
            "least 1; got {values}",
            values=[min_nuclei, radius, passes]))
    return model, min_nuclei, radius, passes


def _ph_tile_centres(db: str, plate: str, well: str):
    """The sequencing tile centres of one well, from ``ops_geometry``.

    :param db: the measurements database.
    :param plate: the plate name.
    :param well: the well name.
    :returns: ``(centres, sites)``: ``(T, 2)`` tile centres ``(y, x)`` in
        well pixels, of the first cycle, and each tile's site.
    :raises ValueError: when the well has no ``ops_geometry`` rows.
    """
    from .qt.i18n import tr

    geometry = _well_rows(db, "ops_geometry", plate, well)
    if geometry is None or geometry.empty:
        raise ValueError(tr(
            "ops_align_model 'tile_affine' needs well {well}'s sequencing "
            "tile positions (ops_geometry); run the stitch phase first",
            well=well))
    first = geometry[geometry["cycle"] == geometry["cycle"].min()]
    centres = first[["y", "x"]].to_numpy(float) + np.column_stack(
        [first["tile_height"].to_numpy(float) / 2.0,
         first["tile_width"].to_numpy(float) / 2.0])
    return centres, first["site"].to_numpy(np.int64)


def _ph_merge(db: str, plate: str, well: str, settings: Mapping[str, Any],
              destination: str, gpu: bool) -> Dict[str, Any]:
    """Pair each phenotype cell with its sequencing nucleus; store ``ops_merge``.

    :param db: the measurements database.
    :param plate: the plate name.
    :param well: the well name.
    :param settings: read for ``ops_merge_threshold``, ``ops_ownership``
        and the ``ops_align_*`` settings.
    :param destination: the run folder.
    :param gpu: let the nearest-neighbour search use the card.
    :returns: the merge report.

    Per field: the nucleus centroids are moved into the well frame with the
    stored transform, the transform is re-solved on the field's own nuclei
    against the sequencing nuclei under it
    (:func:`spacr.ops_merge.refine_field_transform`), and the cells are
    paired one to one (:func:`spacr.ops_merge.merge_field`). With
    ``ops_align_model`` ``'tile_affine'`` the re-solved similarity is
    refined further into one affine per field and one per sequencing tile
    group (:func:`spacr.ops_merge.tile_affine_transform`), every nucleus is
    moved by its tile group's affine before it is paired, and
    ``ops_phenotype`` records the model, the device, the field affine and
    the group affines (``align_tile_transforms``, JSON rows of tile site,
    nuclei and the six affine numbers). A sequencing
    nucleus claimed by two fields -- an overlap -- keeps one copy, chosen by
    ``ops_ownership``:

    - ``'closer_match'`` (default) keeps the closer match
      (:func:`spacr.ops_merge.deduplicate_matches`), as before 1.5.1.8;
    - ``'a1b'`` keeps the copy from the field in which the sequencing
      centroid lies deepest (:func:`spacr.ops_merge.deepest_field_matches`);
    - ``'a1'`` keeps only cells whose phenotype nucleus centroid lies
      deepest in their own field, then the closer match.

    With ``'a1'`` or ``'a1b'`` every phenotype cell also gets ownership
    flags in ``ops_ownership_<well>.parquet``: ``in_overlap_margin`` (a
    field overlapping this one holds the nucleus centroid deeper),
    ``any_match`` (the cell had a sequencing candidate), ``owned_link``
    (its link was kept) and ``owned``. A cell with a candidate is owned
    when its link was kept; a cell without one (A1) when it is not in the
    margin. The aggregate phase then pools owned cells only, and leaves out
    those cut by the field's edge (``ph_masks``).
    """
    import pandas as pd

    from .ops_merge import (deduplicate_matches, deepest_field_matches,
                            field_footprint, field_to_well, footprint_depth,
                            merge_field, overlapping_fields,
                            refine_field_transform, tile_affine_to_well,
                            tile_affine_transform)
    from .qt.i18n import tr

    started = time.perf_counter()
    ownership = _ph_ownership_rule(settings)
    align_model, tile_min, match_radius, trim_passes = _ph_align_options(
        settings)
    tiled = align_model == "tile_affine"
    path = _ph_cells_path(destination, well)
    if not os.path.isfile(path):
        raise ValueError(tr(
            "well {well} has no phenotype cell table at {path}; run the "
            "ph_measure phase first", well=well, path=path))
    cells = pd.read_parquet(path, columns=["site", "label", "centroid_row",
                                           "centroid_col"])
    placed = _ph_transforms(db, plate, well, destination)
    objects = _well_rows(db, "ops_objects", plate, well)
    if objects is None or objects.empty:
        raise ValueError(tr(
            "well {well} has no ops_objects rows; run the objects phase "
            "first", well=well))
    target = objects[["centroid_y", "centroid_x"]].to_numpy(float)
    object_ids = objects["object_id"].to_numpy(np.int64)
    threshold = float(_setting_number(settings.get("ops_merge_threshold"),
                                      "ops_merge_threshold", 4.0))
    if tiled:
        tile_centres, tile_sites = _ph_tile_centres(db, plate, well)
        identity = (1.0, np.eye(2), np.zeros(2))
        align_seconds = 0.0
    rows = []
    refined_rows = []
    fields = {}
    by_site = {int(r.site): r for r in placed.itertuples()}
    for site, group in cells.groupby("site", sort=True):
        record = by_site.get(int(site))
        if record is None:
            continue
        seed = (float(record.scale), np.array(
            [[np.cos(np.radians(record.rotation_deg)),
              -np.sin(np.radians(record.rotation_deg))],
             [np.sin(np.radians(record.rotation_deg)),
              np.cos(np.radians(record.rotation_deg))]]),
            np.array([float(record.offset_y), float(record.offset_x)]))
        points = group[["centroid_row", "centroid_col"]].to_numpy(float)
        corners = field_to_well(np.array(
            [[0, 0], [0, record.field_width], [record.field_height, 0],
             [record.field_height, record.field_width]], float), seed)
        pad = 4.0 * threshold + 20.0
        low, high = corners.min(axis=0) - pad, corners.max(axis=0) + pad
        near = np.flatnonzero((target[:, 0] >= low[0]) & (target[:, 0] <= high[0])
                              & (target[:, 1] >= low[1]) & (target[:, 1] <= high[1]))
        transform, inliers, residual, refined = refine_field_transform(
            points, target[near], seed, radius=min(3.0, threshold))
        refined_rows.append({
            "site": int(site), "refined": int(refined),
            "refined_inliers": int(inliers),
            "refined_residual_px": residual,
            "refined_scale": float(transform[0]),
            "refined_rotation_deg": float(np.degrees(np.arctan2(
                transform[1][1, 0], transform[1][0, 0]))),
            "refined_offset_y": float(transform[2][0]),
            "refined_offset_x": float(transform[2][1]),
            "refined_shift_px": float(np.hypot(
                *(field_to_well(np.array([[record.field_height / 2.0,
                                          record.field_width / 2.0]]),
                                transform)
                  - field_to_well(np.array([[record.field_height / 2.0,
                                             record.field_width / 2.0]]),
                                  seed))[0])),
        })
        if tiled:
            clock = time.perf_counter()
            affine, tiles = tile_affine_transform(
                points, target[near], transform, seed, tile_centres,
                refined=bool(refined), min_nuclei=tile_min,
                radius=match_radius, passes=trim_passes)
            moved = tile_affine_to_well(points, affine, tiles, tile_centres)
            align_seconds += time.perf_counter() - clock
            (a11, a12), (a21, a22) = affine[1]
            refined_rows[-1].update({
                "align_model": "tile_affine", "align_device": "cpu",
                "align_a11": float(a11), "align_a12": float(a12),
                "align_a21": float(a21), "align_a22": float(a22),
                "align_ty": float(affine[2][0]),
                "align_tx": float(affine[2][1]),
                "align_tile_groups": int(len(tiles)),
                "align_tile_transforms": json.dumps([
                    [int(tile_sites[tile]), int(t[3]),
                     *(float(v) for v in np.asarray(t[1]).ravel()),
                     float(t[2][0]), float(t[2][1])]
                    for tile, t in sorted(tiles.items())]),
            })
            pairs = merge_field(moved, target[near], identity,
                                threshold=threshold, gpu=gpu)
            footprint = affine
        else:
            pairs = merge_field(points, target[near], transform,
                                threshold=threshold, gpu=gpu)
            footprint = transform
        labels = group["label"].to_numpy(np.int64)
        for i, j, distance in pairs:
            rows.append((plate, well, int(site), int(labels[int(i)]),
                         int(object_ids[near[int(j)]]), float(distance),
                         int(refined)))
        if ownership != "closer_match":
            fields[int(site)] = (footprint, points, labels, pairs, near,
                                 (float(record.field_height),
                                  float(record.field_width)),
                                 moved if tiled else None)
    frame = pd.DataFrame(rows, columns=["plate", "well", "site", "label",
                                        "object_id", "distance", "refined"])
    check_report = None
    if bool(settings.get("ops_drop_failed_fields", False)):
        from .tabular import write_table

        check = _ph_field_check(plate, well, cells, frame, refined_rows,
                                placed, target, by_site, threshold)
        write_table(check, _ph_fieldcheck_path(destination, well),
                    canonicalise=False)
        check_report = {
            "fields": int(len(check)), "dropped": check.loc[
                check["dropped"], "site"].astype(int).tolist(),
            "table": _ph_fieldcheck_path(destination, well)}
    owner_report = None
    if ownership == "closer_match":
        frame, duplicates = deduplicate_matches(frame)
    else:
        prints = {s: field_footprint(f[0], f[5]) for s, f in fields.items()}
        neighbours = overlapping_fields(prints)
        depths, flags = [], []
        for site, (transform, points, labels, pairs, near,
                   _shape, tile_moved) in fields.items():
            low, high = prints[site]
            if len(pairs):
                depths.append(footprint_depth(
                    target[near[pairs[:, 1].astype(int)]], low, high))
            moved = (field_to_well(points, transform) if tile_moved is None
                     else tile_moved)
            own = footprint_depth(moved, low, high)
            other = np.full(len(moved), -np.inf)
            for neighbour in neighbours[site]:
                other = np.maximum(other, footprint_depth(
                    moved, *prints[neighbour]))
            flags.append(pd.DataFrame({
                "site": site, "label": labels, "well_y": moved[:, 0],
                "well_x": moved[:, 1], "own_depth": own,
                "best_other_depth": other}))
        frame["sbs_depth_px"] = (np.concatenate(depths) if depths
                                 else np.empty(0, float))
        candidates = frame
        if ownership == "a1b":
            frame, duplicates = deepest_field_matches(candidates)
        else:
            margin_rows = pd.concat(flags, ignore_index=True) if flags else \
                pd.DataFrame(columns=["site", "label", "own_depth",
                                      "best_other_depth"])
            core = margin_rows.loc[margin_rows["own_depth"]
                                   >= margin_rows["best_other_depth"],
                                   ["site", "label"]]
            frame = candidates.merge(core, on=["site", "label"], how="inner")
            duplicates = int(len(candidates) - len(frame))
        frame, extra = deduplicate_matches(frame)
        duplicates += extra
        owner = (pd.concat(flags, ignore_index=True) if flags else
                 pd.DataFrame(columns=["site", "label", "well_y", "well_x",
                                       "own_depth", "best_other_depth"]))
        owner["in_overlap_margin"] = owner["own_depth"] < owner[
            "best_other_depth"]
        key = pd.MultiIndex.from_frame(owner[["site", "label"]])
        owner["any_match"] = key.isin(pd.MultiIndex.from_frame(
            candidates[["site", "label"]]))
        owner["owned_link"] = key.isin(pd.MultiIndex.from_frame(
            frame[["site", "label"]]))
        owner["owned"] = np.where(owner["any_match"], owner["owned_link"],
                                  ~owner["in_overlap_margin"])
        owner.insert(0, "well", well)
        owner.insert(0, "plate", plate)
        owner["ownership_rule"] = ownership
        from .tabular import write_table

        write_table(owner, _ph_ownership_path(destination, well),
                    canonicalise=False)
        closer, _ = deduplicate_matches(candidates)
        owner_report = {
            "rule": ownership, "candidates": int(len(candidates)),
            "cells_with_candidate": int(owner["any_match"].sum()),
            "margin_rows": int(owner["in_overlap_margin"].sum()),
            "owned": int(owner["owned"].sum()),
            "margin_copies_excluded": int((~owner["owned"]).sum()),
            "switched_from_closer_match": int((~pd.MultiIndex.from_frame(
                frame[["site", "label", "object_id"]]).isin(
                pd.MultiIndex.from_frame(closer[["site", "label",
                                                 "object_id"]]))).sum()),
            "table": _ph_ownership_path(destination, well),
        }
    _replace_well_rows(db, "ops_merge", frame, plate, well)
    refinements = pd.DataFrame(refined_rows)
    if len(refinements):
        keep = placed.drop(columns=[c for c in placed.columns if c != "site"
                                    and (c in refinements.columns
                                         or str(c).startswith("align_"))])
        _replace_well_rows(db, "ops_phenotype",
                           keep.merge(refinements, on="site", how="left"),
                           plate, well)
    report = {
        "cells": int(len(cells)), "merged": int(len(frame)),
        "match_rate": round(len(frame) / max(1, len(cells)), 4),
        "duplicates_dropped": duplicates,
        "fields": int(cells["site"].nunique()),
        "fields_refined": int(refinements["refined"].sum()) if len(refinements) else 0,
        "refined_shift_px_median": (
            round(float(refinements.loc[refinements["refined"] > 0,
                                        "refined_shift_px"].median()), 3)
            if len(refinements) and refinements["refined"].any() else None),
        "distance_median_px": (round(float(frame["distance"].median()), 3)
                               if len(frame) else None),
        "threshold_px": threshold,
        "ownership": owner_report or {"rule": ownership},
        "field_check": check_report,
        **({"align": {
            "model": "tile_affine", "device": "cpu",
            "tile_min_nuclei": tile_min,
            "match_radius_px": match_radius,
            "trim_passes": trim_passes,
            "fields_with_tile_groups": int(sum(
                r["align_tile_groups"] > 0 for r in refined_rows)),
            "tile_groups": int(sum(r["align_tile_groups"]
                                   for r in refined_rows)),
            "seconds_per_field": round(
                align_seconds / max(1, len(refined_rows)), 5),
        }} if tiled else {}),
        "seconds": round(time.perf_counter() - started, 1),
    }
    _say(tr("{well} merge: {merged} of {cells} cells matched to a sequencing "
            "nucleus", well=well, merged=report["merged"],
            cells=report["cells"]))
    return report


def _ph_design(path: str, prefix_length: int):
    """``prefix -> (guide, gene)`` from a guide design table.

    :param path: a CSV/TSV/XLSX with ``sgRNA`` and ``gene_symbol`` (or
        ``gene``) columns.
    :param prefix_length: how many leading bases the decoded barcodes keep.
    :returns: ``(frame, ambiguous)``: one row per unambiguous prefix with
        ``prefix``, ``sgRNA`` and ``gene``, and how many prefixes were
        dropped because two genes share them.
    """
    from .qt.i18n import tr
    from .tabular import read_table

    design = read_table(path, canonicalise=False, report=None)
    guide = next((c for c in ("sgRNA", "sgrna", "guide", "sequence")
                  if c in design.columns), None)
    gene = next((c for c in ("gene_symbol", "gene", "Gene")
                 if c in design.columns), None)
    if guide is None or gene is None:
        raise ValueError(tr(
            "the design table {path} needs an sgRNA and a gene_symbol "
            "column; it has {columns}", path=path,
            columns=list(design.columns)[:20]))
    table = design[[guide, gene]].dropna().astype(str)
    table.columns = ["sgRNA", "gene"]
    table["prefix"] = table["sgRNA"].str.upper().str[:int(prefix_length)]
    genes = table.groupby("prefix")["gene"].nunique()
    ambiguous = set(genes[genes > 1].index)
    table = table[~table["prefix"].isin(ambiguous)].drop_duplicates("prefix")
    return table.reset_index(drop=True), len(ambiguous)


def _ph_feature_columns(columns) -> List[str]:
    """The per-cell feature columns: compartment-prefixed, not positions."""
    skip = ("centroid", "bbox", "coords", "orientation", "label")
    return [c for c in columns
            if str(c).startswith(("nucleus_", "cell_", "cytoplasm_"))
            and not any(s in str(c) for s in skip)]


def _ph_outside_ring_column(feature: str) -> bool:
    """Whether ``feature`` is an ``outside`` profiling column (``*_outside_<stat>``).

    :param feature: a per-cell feature name.
    :returns: True for the background-ring statistics of
        :func:`spacr.measure.measure_outside`, which
        ``ops_drop_nonfinite_rows`` does not count.
    """
    import re

    return re.search(r"_outside_(mean|percentile_\d+)$", str(feature)) is not None


#: A cell with a larger share of non-finite features is left out when
#: ``ops_drop_nonfinite_rows`` is on (Brieflow's ``drop_rows_threshold``).
_PH_NONFINITE_SHARE = 0.01

#: How many feature columns the aggregate phase reads and reduces at once.
#: Medians and robust z scores are per column, so the block size changes only
#: the memory a plate needs: the whole cell table of Funk et al.'s plate A
#: took 46 GB at once and about 8 GB in blocks of 48.
_PH_POOL_BLOCK = 48

#: The settings that turn on a gene-profile option of the aggregate phase
#: (:func:`_ph_profile_options`); each needs the per-cell TVN components.
_PH_PROFILE_SWITCHES: Tuple[str, ...] = (
    "ops_profile_cell_cycle", "ops_profile_shrinkage",
    "ops_profile_generic_axes", "ops_profile_weighting",
    "ops_guide_efficacy", "ops_profile_similarity", "ops_penetrance",
    "ops_escapers", "ops_phase_tests", "ops_profile_space",
    "ops_neighbour_effects",
)

_PH_DEGENERATE_CAUSES = (
    ("distance_to_nucleus",
     "the intensity centre of mass of most cells lies inside the nucleus, "
     "where this distance is 0 by definition"),
    ("euler_number",
     "almost every object has the same topology: 1 for a cell or nucleus "
     "without holes and 0 for a cytoplasm ring"),
    ("frac_low10",
     "most objects have no pixel below the 10th percentile of their field, "
     "which lies in the background"),
    ("frac_high90",
     "most objects have no pixel above the 90th percentile of their field"),
)


def _ph_names(value) -> List[str]:
    """A comma-separated setting, or a list, as stripped names."""
    if value is None:
        return []
    items = value.split(",") if isinstance(value, str) else list(value)
    return [str(item).strip() for item in items if str(item).strip()]


def _ph_genotype_table(path: str, plate: str, wells: Sequence[str]):
    """The called sequencing nuclei of ``wells`` from a genotype table.

    :param path: a Parquet, CSV, TSV or Excel table with ``well``,
        ``object_id`` and ``mapped_guide`` columns, and optionally ``plate``.
    :param plate: rows of another plate are left out when ``plate`` is a
        column.
    :param wells: the wells to keep.
    :returns: ``well, object_id, mapped_guide`` for every nucleus with a
        guide.
    :raises ValueError: when the file is missing, lacks a column, or calls
        one nucleus twice.
    """
    import pandas as pd

    from .qt.i18n import tr

    if not path or not os.path.isfile(str(path)):
        raise ValueError(tr(
            "ops_genotype_table must be a table with well, object_id and "
            "mapped_guide columns; got {path}", path=path))
    wanted = ["plate", "well", "object_id", "mapped_guide"]
    if str(path).lower().endswith(".parquet"):
        import pyarrow.parquet as pq

        names = pq.ParquetFile(str(path)).schema_arrow.names
        table = pd.read_parquet(str(path),
                                columns=[c for c in wanted if c in names])
    else:
        from .tabular import read_table

        table = read_table(str(path), canonicalise=False, report=None)
    missing = [c for c in wanted[1:] if c not in table.columns]
    if missing:
        raise ValueError(tr(
            "the genotype table {path} needs well, object_id and "
            "mapped_guide columns; it lacks {missing}", path=path,
            missing=missing))
    if "plate" in table.columns:
        table = table[table["plate"].astype(str) == str(plate)]
    table = table[table["well"].astype(str).isin([str(w) for w in wells])]
    called = table.loc[table["mapped_guide"].fillna("").astype(str) != "",
                       ["well", "object_id", "mapped_guide"]]
    twice = int(called.duplicated(["well", "object_id"]).sum())
    if twice:
        raise ValueError(tr(
            "the genotype table {path} calls {count} sequencing nuclei more "
            "than once; give each well and object_id one row", path=path,
            count=twice))
    return called.reset_index(drop=True)


def _ph_read_columns(sources, columns: Sequence[str]):
    """Some columns of several cell tables, stacked in table order.

    :param sources: ``(path, names, rows)`` per table: its file, its column
        names and its row count.
    :param columns: the columns to read; a table without one gives NaN.
    :returns: one frame, as stacking the whole tables and selecting the
        columns would.
    """
    import pandas as pd

    frames = []
    for path, names, rows in sources:
        present = [c for c in columns if c in names]
        frame = pd.read_parquet(path, columns=present)
        if not present:
            frame = pd.DataFrame(index=pd.RangeIndex(rows))
        frames.append(frame)
    stacked = pd.concat(frames, ignore_index=True)
    return stacked.loc[:, [c for c in columns if c in stacked.columns]]


def _ph_degenerate_feature(feature: str, values, control, wells
                           ) -> Dict[str, Any]:
    """Why one feature's robust z is undefined in every well.

    :param feature: the feature column.
    :param values: the feature over the pooled cells.
    :param control: True for each control cell.
    :param wells: each cell's well.
    :returns: the feature QC row: the reason and the control statistics
        behind it.
    """
    import pandas as pd

    reference = pd.to_numeric(values[control], errors="coerce")
    finite = reference.dropna()
    row: Dict[str, Any] = {
        "feature": feature, "control_cells": int(control.sum()),
        "control_finite": int(finite.size),
        "control_median": float(finite.median()) if finite.size else None,
        "control_share_at_median": (
            round(float((finite == finite.median()).mean()), 4)
            if finite.size else None)}
    if not finite.size:
        row["reason"] = "no control cell has a finite value"
        return row
    groups = wells[control].loc[finite.index]
    spread = (finite - finite.groupby(groups).transform("median")).abs(
    ).groupby(groups).median()
    row["wells_with_zero_control_mad"] = int((spread == 0).sum())
    reason = ("the median absolute deviation of the control cells is zero "
              "in every well, so the robust z is undefined")
    for family, cause in _PH_DEGENERATE_CAUSES:
        if feature.endswith(family):
            reason = f"{reason}: {cause}"
            break
    row["reason"] = reason
    return row


class _PhFunnel:
    """The per-well cell counts of the aggregate phase's filters, as it runs them.

    Each count is taken from the frame the aggregate phase holds at that
    step, so the funnel cannot disagree with the tables it pooled: a
    reduction recomputed afterwards could. ``owned`` comes from
    ``ops_ownership_<well>`` (A1/A1b); without one every cell is owned.
    """

    def __init__(self, cells, destination: str, wells: Sequence[str]):
        """Count the segmented cells and read each well's ownership.

        :param cells: every cell of the pooled wells (``well``, ``site``,
            ``label``, and ``touches_field_edge`` when the mask pass was
            read).
        :param destination: the run folder.
        :param wells: the pooled wells.
        """
        import pandas as pd

        self.wells = [str(w) for w in wells]
        self.counts: Dict[str, Dict[str, Any]] = {w: {} for w in self.wells}
        self.count("cells_segmented", cells)
        if "touches_field_edge" in cells:
            self.count("cells_without_mask_row",
                       cells[cells["touches_field_edge"].isna()])
        owned = np.ones(len(cells), bool)
        for well in self.wells:
            path = _ph_ownership_path(destination, well)
            if not os.path.isfile(path):
                continue
            table = pd.read_parquet(path, columns=["site", "label", "owned"])
            flags = pd.Series(table["owned"].astype(bool).to_numpy(),
                              index=pd.MultiIndex.from_arrays(
                                  [table["site"].astype(int),
                                   table["label"].astype(int)]))
            mine = (cells["well"].astype(str) == well).to_numpy()
            keys = pd.MultiIndex.from_arrays(
                [cells.loc[mine, "site"].astype(int),
                 cells.loc[mine, "label"].astype(int)])
            owned[mine] = [False if pd.isna(v) else bool(v)
                           for v in flags.reindex(keys)]
        self.owned = pd.Series(owned, index=cells.index)
        self.count("owned", cells[owned])
        self.count("margin_copies_excluded", cells[~owned])

    def count(self, name: str, frame) -> None:
        """Record ``frame``'s rows per well under ``name``."""
        sizes = frame.groupby(frame["well"].astype(str)).size() if len(
            frame) else {}
        for well in self.wells:
            self.counts[well][name] = int(sizes.get(well, 0)) if len(
                frame) else 0

    def cut(self, cells, cut) -> None:
        """Count the owned cells the field's edge cuts, and the whole ones."""
        owned = self.owned.reindex(cells.index).to_numpy()
        cut = cut.to_numpy()
        self.count("owned_cut_excluded", cells[owned & cut])
        self.count("owned_whole", cells[owned & ~cut])
        self.owned = self.owned[~cut].reset_index(drop=True)

    def failed_fields(self, cells, inside, gone) -> None:
        """Count the owned, whole cells of the fields the field check dropped."""
        owned = (self.owned.to_numpy(bool) if len(self.owned) == len(cells)
                 else np.ones(len(cells), bool))
        self.count("registration_failure_cells_excluded",
                   cells[owned & np.asarray(inside, bool)])
        for well in self.wells:
            self.counts[well]["registration_failure_fields"] = sorted(
                int(s) for w, s in zip(gone["well"], gone["site"])
                if str(w) == well)

    def calls(self, cells, control) -> None:
        """Count the controls, mitotic and interphase cells, guides and genes."""
        self.count("nontargeting", cells[control.to_numpy()])
        mitotic = cells["mitotic"].astype(bool).to_numpy() if \
            "mitotic" in cells else np.zeros(len(cells), bool)
        self.count("mitotic", cells[mitotic])
        self.count("interphase", cells[~mitotic])
        for well in self.wells:
            mine = cells[cells["well"].astype(str) == well]
            self.counts[well]["guides"] = int(mine["sgRNA"].nunique())
            self.counts[well]["genes"] = int(mine["gene"].nunique())

    def frame(self, plate: str):
        """One row per well: ``plate``, ``well`` and every count taken."""
        import pandas as pd

        return pd.DataFrame([{"plate": plate, "well": well, **counts}
                             for well, counts in self.counts.items()])


def _ph_aggregate(db: str, plate: str, wells: Sequence[str],
                  settings: Mapping[str, Any], destination: str,
                  keep_cells: bool = False) -> Dict[str, Any]:
    """Reduce merged, guide-called cells to per-guide and per-gene tables.

    :param db: the measurements database.
    :param plate: the plate name.
    :param wells: the wells run.
    :param settings: read for ``ops_design``, ``ops_control_prefix``,
        ``ops_bootstrap``, ``ops_pvalue_null``, ``ops_test_features``,
        ``ops_mitotic_dna_z``, ``ops_genotype_table``, ``ops_pool_wells``,
        ``ops_pool_interphase``, ``ops_feature_qc``, ``ops_ownership``,
        ``ops_flatfield``, ``ops_flatfield_maps``, ``ops_drop_failed_fields``,
        ``ops_drop_nonfinite_rows``, ``ops_profile_normalisation``,
        ``ops_aggregate_level``, ``ops_nt_rows`` and the ``ops_tvn_*``,
        ``ops_cluster_*`` and ``ops_mitotic_model`` options.
    :param destination: where ``ops_guides_<plate>`` and
        ``ops_genes_<plate>`` (CSV and Parquet) are written. A run that pools
        only interphase cells or only some wells adds ``_interphase`` or
        ``_wells-<wells>`` to both names, so it does not replace the full
        tables.
    :param keep_cells: also write ``ops_cells_<plate>[tag].parquet``: each
        pooled cell's ``plate``, ``well``, ``site``, ``label``,
        ``object_id``, ``mapped_guide``, ``sgRNA``, ``gene`` and
        ``mitotic``, which the ``ph_tables`` phase puts on every row.
    :returns: the aggregate report.

    FUNK ET AL. 2022's REDUCTION: every feature is a robust z against the
    same well's non-targeting cells
    (:func:`spacr.gene_measurement_compare.robust_z_to_controls`); a guide
    is the median of its cells; a gene is the median of its guides; and,
    when ``ops_bootstrap`` is above zero, each gene is tested against a null
    of non-targeting guides of the same sizes
    (:func:`spacr.gene_measurement_compare.bootstrap_gene_pvalues`). The
    mitotic index is the share of a gene's cells
    :func:`~spacr.gene_measurement_compare.mitotic_calls` calls mitotic.

    THE FEATURES ARE REDUCED A BLOCK OF COLUMNS AT A TIME
    (``_PH_POOL_BLOCK``). Only the keys and the two mitotic-call columns of
    every cell are held at once. Each block is read from the per-well cell
    tables, normalised and reduced, so the tables are the ones a whole-frame
    reduction gives, value for value.

    ``ops_genotype_table`` replaces the ``ops_barcodes`` calls with another
    decode, joined on ``well`` and ``object_id``. ``ops_pool_wells`` pools a
    subset of the wells; each cell is still normalised against its own
    well. With ``ops_pool_interphase`` the guides and genes summarise the
    interphase cells only, and so do the control cells of the test. With
    ``ops_feature_qc`` on, a feature whose robust z is undefined in every
    well is left out of both tables. It is listed with its reason in
    ``ops_feature_qc_<plate>.csv``.

    With ``ops_ownership`` ``'a1'`` or ``'a1b'`` the merge kept owned links
    only, and cells that touch their field's edge are left out too
    (``ops_maskpass_<well>.parquet`` from ``ph_masks``): such a cell is cut,
    and the whole copy is in the neighbouring field or not imaged at all.
    With ``ops_flatfield='analytic'`` every intensity feature is corrected
    for the field's illumination from the mask pass's gains
    (:func:`spacr.ops_flatfield.correct_features`) before anything else,
    the mitotic call included; with ``'exact'`` the measurement already
    corrected the pixels and nothing more is done here.

    With ``ops_cross_field_ownership`` the genotyped cells are reduced by
    ``ops_cross_field_policy`` (:func:`_ph_cross_field_filter`): one row per
    physical cell by default; the report's ``cross_field`` says how many
    went.

    With ``ops_drop_failed_fields`` the cells of a field whose transform
    re-solve failed and which links under three times chance, or which links
    under twice chance, are left out (the merge phase checked every field);
    the dropped fields and why are written to
    ``dropped_fields_<plate>.csv``. With ``ops_drop_nonfinite_rows`` a cell
    with more than 1 % of its features not finite is left out; the report's
    ``dropped_nonfinite_rows`` counts them. The ``outside`` profiling
    columns are not counted: a cell enclosed by its neighbours has no
    background ring, which is NaN by definition and flagged by
    ``<compartment>_enclosed``, and such a cell is kept.

    With ``ops_profile_normalisation``
    set to ``'tvn'`` the run also writes TVN profiles
    (:mod:`spacr.ops_tvn`) as ``ops_guides_<plate>[tag]_tvn`` and
    ``ops_genes_<plate>[tag]_tvn``; the robust z tables are still written,
    because the per-gene test needs named features.
    ``ops_tvn_fit_cells='interphase'`` fits the TVN on the interphase cells
    and applies it to every pooled cell, so in a run that pools every cell
    the mitotic cells share the interphase space, as the cell-cycle
    profiles' reference has them. ``ops_cluster_benchmark``
    clusters the TVN profiles (or the robust z ones without TVN) and scores
    them (:mod:`spacr.ops_clusters`), and ``ops_mitotic_model`` calls
    mitotic cells with a trained classifier (:mod:`spacr.ops_mitotic`)
    instead of the DNA rule.

    ``ops_aggregate_level='cells'`` makes a gene the median of all of its
    cells, as Brieflow does, instead of the median of its guides.
    ``ops_nt_rows`` makes the non-targeting guides rows of the gene tables:
    ``'barcode'`` one row per guide (Brieflow's layout), ``'pseudo'``
    random pseudo-genes of four guides
    (:func:`spacr.gene_measurement_compare.nt_row_names`).
    """
    import pandas as pd
    import pyarrow.parquet as pq

    from .gene_measurement_compare import (AGGREGATE_LEVELS, NT_ROW_MODES,
                                           aggregate_genes, aggregate_guides,
                                           bootstrap_gene_pvalues,
                                           mitotic_calls, nt_row_names,
                                           robust_z_to_controls)
    from .qt.i18n import tr
    from .tabular import write_table

    started = time.perf_counter()
    level = str(settings.get("ops_aggregate_level") or "guides").strip().lower()
    nt_rows = str(settings.get("ops_nt_rows") or "").strip().lower() or None
    if level not in AGGREGATE_LEVELS or nt_rows not in (None, *NT_ROW_MODES):
        raise ValueError(tr(
            "ops_aggregate_level must be guides or cells and ops_nt_rows "
            "empty, pseudo or barcode; got {level} and {rows}",
            level=level, rows=nt_rows))
    design_path = settings.get("ops_design")
    if not design_path or not os.path.isfile(str(design_path)):
        raise ValueError(tr(
            "ops_design must be the guide design table with sgRNA and "
            "gene_symbol columns; got {path}", path=design_path))
    control_prefix = str(settings.get("ops_control_prefix") or "nontargeting")
    run_wells = [str(w) for w in wells]
    asked = [w.upper() for w in _ph_names(settings.get("ops_pool_wells"))]
    unknown = [w for w in asked if w not in {r.upper() for r in run_wells}]
    if unknown:
        raise ValueError(tr(
            "ops_pool_wells names wells {unknown} that this run does not "
            "have; it has {wells}", unknown=unknown, wells=run_wells))
    pooled = [w for w in run_wells if not asked or w.upper() in asked]
    genotype_path = settings.get("ops_genotype_table") or None
    merged_rows = _rows_for_wells(db, "ops_merge", plate, pooled)
    if genotype_path:
        if merged_rows is None:
            raise ValueError(tr(
                "aggregate needs ops_merge; run the merge phase first"))
        called = _ph_genotype_table(str(genotype_path), plate, pooled)
    else:
        barcodes = _rows_for_wells(db, "ops_barcodes", plate, pooled)
        if merged_rows is None or barcodes is None:
            raise ValueError(tr(
                "aggregate needs ops_merge and ops_barcodes; run the merge "
                "and decode phases first"))
        called = barcodes.loc[
            barcodes["mapped_guide"].fillna("").astype(str) != "",
            ["well", "object_id", "mapped_guide"]]
    length = int(called["mapped_guide"].str.len().mode().iloc[0]) if len(called) else 11
    design, ambiguous = _ph_design(str(design_path), length)
    sources, schemas = [], []
    for well in pooled:
        path = _ph_cells_path(destination, well)
        if os.path.isfile(path):
            meta = pq.ParquetFile(path)
            schemas.append(meta.schema_arrow)
            sources.append((path, set(meta.schema_arrow.names),
                            int(meta.metadata.num_rows)))
    if not sources:
        raise ValueError(tr("no phenotype cell table for wells {wells}",
                            wells=list(pooled)))
    columns: List[str] = []
    for schema in schemas:
        for name in schema.names:
            if name not in columns and not name.startswith("__index_level_"):
                columns.append(name)
    features = _ph_feature_columns(columns)
    single = all(str(schema.field(name).type) == "float"
                 for schema in schemas for name in features
                 if name in schema.names)
    anchor = [] if single else ["_ph_float64_anchor"]
    channels = [c.lower() for c in _ph_channels(settings)]
    dna_column = f"nucleus_{channels[0]}_mean_intensity" if channels else ""
    if dna_column not in columns or "nucleus_area" not in columns:
        dna_column = next((c for c in features if c.startswith("nucleus_")
                           and c.endswith("mean_intensity")), "")
    mitotic_inputs = [c for c in (dna_column, "nucleus_area")
                      if c and c in columns]
    mitotic_model = None
    if settings.get("ops_mitotic_model"):
        from .ops_mitotic import load_mitotic_classifier
        mitotic_model = load_mitotic_classifier(str(settings["ops_mitotic_model"]))
        needed = [*mitotic_model["features"], mitotic_model.get("normalise_by")]
        absent_model = [c for c in needed if c and c not in columns]
        if absent_model:
            raise ValueError(tr(
                "ops_mitotic_model needs columns {absent} that the cell tables "
                "do not have", absent=absent_model))
        extra = [c for c in needed
                 if c and c not in ("well", "site", "label")]
        mitotic_inputs = list(dict.fromkeys([*mitotic_inputs, *extra]))
    cells = _ph_read_columns(sources, ["well", "site", "label",
                                       *mitotic_inputs])
    cells["_row"] = np.arange(len(cells))
    ownership = _ph_ownership_rule(settings)
    flatfield = _ph_flatfield_mode(settings)
    flat_offsets: Dict[str, float] = {}
    flat_record: Optional[Dict[str, Any]] = None
    gain_columns: List[str] = []
    if flatfield == "analytic":
        from .ops_flatfield import load_flat_maps, record_flat_md5

        flat_folder = _ph_flat_folder(settings, destination)
        _maps, flat_offsets, info = load_flat_maps(
            flat_folder, _ph_channels(settings))
        record_flat_md5(destination, info)
        flat_record = {"mode": "analytic", "folder": flat_folder,
                       "md5": info.get("md5"), "offsets": flat_offsets}
        gain_columns = [f"{comp}_{ch}_g" for comp in
                        ("cell", "nucleus", "cytoplasm")
                        for ch in flat_offsets] + [
            f"{comp}_px" for comp in ("cell", "nucleus", "cytoplasm")]
    ownership_counts: Optional[Dict[str, Any]] = None
    if ownership != "closer_match" or gain_columns:
        mask_rows = []
        for well in pooled:
            path = _ph_maskpass_path(destination, well)
            if not os.path.isfile(path):
                raise ValueError(tr(
                    "ops_ownership {rule} and ops_flatfield {mode} need the "
                    "mask pass of well {well} at {path}; run the ph_masks "
                    "phase first", rule=ownership, mode=flatfield, well=well,
                    path=path))
            names = set(pq.ParquetFile(path).schema_arrow.names)
            lacking = [c for c in gain_columns if c not in names]
            if lacking:
                raise ValueError(tr(
                    "the mask pass of well {well} has no flat-field gains "
                    "{lacking}; rerun ph_masks with ops_flatfield "
                    "'analytic'", well=well, lacking=lacking[:3]))
            mask_rows.append(pd.read_parquet(path, columns=[
                "well", "site", "label", "touches_field_edge",
                *gain_columns]))
        masks = pd.concat(mask_rows, ignore_index=True)
        cells = cells.merge(masks, on=["well", "site", "label"], how="left")
        del masks, mask_rows
    funnel = _PhFunnel(cells, destination, pooled)
    if ownership != "closer_match":
        cut = cells["touches_field_edge"].fillna(True).astype(bool)
        funnel.cut(cells, cut)
        ownership_counts = {"rule": ownership,
                            "cells_cut_by_field_edge": int(cut.sum())}
        for well in pooled:
            path = _ph_ownership_path(destination, well)
            if os.path.isfile(path):
                owned = pd.read_parquet(path, columns=["owned"])["owned"]
                ownership_counts.setdefault("margin_copies_excluded", 0)
                ownership_counts["margin_copies_excluded"] += int(
                    (~owned.astype(bool)).sum())
        cells = cells[~cut.to_numpy()].reset_index(drop=True)
    field_report: Optional[Dict[str, Any]] = None
    if bool(settings.get("ops_drop_failed_fields", False)):
        checks = []
        for well in pooled:
            path = _ph_fieldcheck_path(destination, well)
            if not os.path.isfile(path):
                raise ValueError(tr(
                    "ops_drop_failed_fields needs the field check of well "
                    "{well} at {path}; run the merge phase with it on",
                    well=well, path=path))
            checks.append(pd.read_parquet(path))
        check = pd.concat(checks, ignore_index=True)
        gone = check[check["dropped"].astype(bool)]
        target = os.path.join(destination, f"dropped_fields_{plate}.csv")
        write_table(gone[["plate", "well", "site", "cells", "matched",
                          "link_rate", "chance_rate", "link_over_chance",
                          "refined", "reason"]], target, canonicalise=False)
        inside = pd.MultiIndex.from_frame(cells[["well", "site"]]).isin(
            pd.MultiIndex.from_frame(gone[["well", "site"]]))
        field_report = {
            "fields_checked": int(len(check)),
            "fields_dropped": [f"{w}/{int(x)}" for w, x in
                               zip(gone["well"], gone["site"])],
            "cells_dropped": int(inside.sum()), "table": target}
        funnel.failed_fields(cells, inside, gone)
        cells = cells[~inside].reset_index(drop=True)
    cells = cells.merge(merged_rows[["well", "site", "label", "object_id"]],
                        on=["well", "site", "label"], how="inner")
    cells = cells.merge(called, on=["well", "object_id"], how="left")
    with_guide = cells["mapped_guide"].notna()
    funnel.count("linked", cells)
    funnel.count("linked_no_genotype", cells[~with_guide])
    cells = cells[with_guide].merge(
        design.rename(columns={"prefix": "mapped_guide"}),
        on="mapped_guide", how="inner")
    funnel.count("with_guide", cells)
    cross_field_report = None
    if bool(settings.get("ops_cross_field_ownership", False)):
        cells, cross_field_report = _ph_cross_field_filter(
            cells, destination, pooled, settings)
        funnel.count("one_copy_per_cell", cells)
    nonfinite_dropped: Optional[int] = None
    if bool(settings.get("ops_drop_nonfinite_rows", False)):
        bad_count = np.zeros(len(cells), np.int64)
        counted = [f for f in features if not _ph_outside_ring_column(f)]
        for start in range(0, len(counted), _PH_POOL_BLOCK):
            block = counted[start:start + _PH_POOL_BLOCK]
            values = _ph_read_columns(sources, block).iloc[
                cells["_row"].to_numpy()].to_numpy(np.float32)
            bad_count += (~np.isfinite(values)).sum(axis=1)
        broken = bad_count > _PH_NONFINITE_SHARE * len(counted)
        nonfinite_dropped = int(broken.sum())
        funnel.count("dropped_nonfinite_rows", cells[broken])
        cells = cells[~broken].reset_index(drop=True)
    rows = cells["_row"].to_numpy()
    control = cells["gene"].str.lower().str.startswith(control_prefix.lower())
    if nt_rows:
        cells["gene"] = nt_row_names(
            cells["sgRNA"], cells["gene"], rows=nt_rows,
            control_prefix=control_prefix,
            pseudo_prefix=f"{control_prefix}_pseudo_")
    mitotic_z = float(_setting_number(settings.get("ops_mitotic_dna_z"),
                                      "ops_mitotic_dna_z", 3.0))
    if gain_columns and dna_column:
        from .ops_flatfield import correct_features

        cells[dna_column] = correct_features(
            cells, cells, [dna_column], flat_offsets)[dna_column].to_numpy()
    if mitotic_model is not None:
        from .ops_mitotic import predict_mitotic
        model_cells = cells
        if gain_columns:
            from .ops_flatfield import correct_features, correction_inputs
            model_features = list(mitotic_model["features"])
            needed = [c for c in correction_inputs(model_features,
                                                   list(flat_offsets))
                      if c not in cells]
            model_cells = cells.reset_index(drop=True)
            if needed:
                extra_inputs = _ph_read_columns(sources, needed).iloc[
                    cells["_row"].to_numpy()].reset_index(drop=True)
                model_cells = pd.concat([model_cells, extra_inputs], axis=1)
            corrected = correct_features(model_cells, model_cells,
                                         model_features, flat_offsets)
            model_cells = model_cells.copy()
            model_cells[model_features] = corrected.to_numpy()
        cells["mitotic"] = predict_mitotic(mitotic_model, model_cells).to_numpy()
    elif dna_column and "nucleus_area" in cells:
        cells["mitotic"] = mitotic_calls(cells, dna=dna_column,
                                         area="nucleus_area", dna_z=mitotic_z)
    else:
        cells["mitotic"] = False
    funnel.calls(cells, control)
    interphase = bool(settings.get("ops_pool_interphase", False))
    keep = (~cells["mitotic"].astype(bool)).to_numpy() if interphase else \
        np.ones(len(cells), bool)
    boot = int(_setting_number(settings.get("ops_bootstrap"),
                               "ops_bootstrap", 0))
    null = str(settings.get("ops_pvalue_null") or "calibrated").strip().lower()
    if null not in ("calibrated", "cell_bootstrap"):
        raise ValueError(tr(
            "ops_pvalue_null must be calibrated or cell_bootstrap; got "
            "{value}", value=null))
    normalisation = str(settings.get("ops_profile_normalisation")
                        or "robust_z").strip().lower()
    if normalisation not in ("robust_z", "tvn"):
        raise ValueError(tr(
            "ops_profile_normalisation must be robust_z or tvn; got {value}",
            value=normalisation))
    named = _ph_names(settings.get("ops_test_features"))
    absent = [c for c in named if c not in features]
    if absent:
        raise ValueError(tr(
            "ops_test_features names columns {absent} that the cell tables "
            "do not have", absent=absent))
    tested: List[str] = []
    if boot > 0 and control.any():
        tested = named or [c for c in (
            "nucleus_area", dna_column,
            f"nucleus_{channels[0]}_integrated_intensity" if channels else "",
            *(f"cell_{c}_mean_intensity" for c in channels[1:]))
            if c in features]
    feature_qc = bool(settings.get("ops_feature_qc", True))
    meta = cells.loc[:, ["well", "sgRNA", "gene", "mitotic"]]
    pool_rows = (control.to_numpy() & keep)
    control_values = pd.DataFrame(index=pd.RangeIndex(int(pool_rows.sum())))
    guide_parts, gene_parts, degenerate = [], [], []
    gains = None
    if gain_columns:
        from .ops_flatfield import correct_features, correction_inputs

        gains = cells[gain_columns].reset_index(drop=True)
    for start in range(0, len(features), _PH_POOL_BLOCK):
        block = features[start:start + _PH_POOL_BLOCK]
        if gains is None:
            values = _ph_read_columns(sources, block).iloc[rows].reset_index(
                drop=True)
        else:
            needed = correction_inputs(block, list(flat_offsets))
            values = correct_features(
                _ph_read_columns(sources, needed).iloc[rows].reset_index(
                    drop=True), gains, block, flat_offsets)
        frame = pd.concat([meta[["well"]], values], axis=1)
        if anchor:
            frame[anchor[0]] = 0.0
        normalised = robust_z_to_controls(frame, [*block, *anchor],
                                          control=control)[block]
        del frame
        if feature_qc and control.any():
            for feature in block:
                if not normalised[feature].notna().any():
                    degenerate.append(_ph_degenerate_feature(
                        feature, values[feature], control, cells["well"]))
        for feature in block:
            if feature in tested:
                control_values[feature] = normalised.loc[
                    pool_rows, feature].to_numpy()
        scored = pd.concat([meta, normalised], axis=1)[keep]
        guides = aggregate_guides(scored, block, guide="sgRNA", gene="gene",
                                  mitotic=None if interphase else "mitotic")
        genes = aggregate_genes(guides, block, level=level, cells=scored)
        guide_parts.append(guides if not guide_parts else guides[block])
        gene_parts.append(genes if not gene_parts else genes[block])
        del values, normalised, scored
    guides = pd.concat(guide_parts, axis=1)
    genes = pd.concat(gene_parts, axis=1)
    genes = genes[genes["gene"] != "nontargeting_drop"].reset_index(drop=True)
    dropped = [row["feature"] for row in degenerate]
    if dropped:
        guides = guides.drop(columns=dropped)
        genes = genes.drop(columns=dropped)
    tested = [c for c in tested if c not in dropped]
    is_control = guides["gene"].str.lower().str.startswith(
        control_prefix.lower())
    null_parameters = None
    if tested:
        pvalues = bootstrap_gene_pvalues(
            control_values, guides[~is_control], tested, n_boot=boot,
            null=null, control_guides=guides[is_control])
        null_parameters = pvalues.attrs.get("null")
        if "gene" in pvalues.columns:
            genes = genes.merge(pvalues, on="gene", how="left")
    tag = "_interphase" if interphase else ""
    if len(pooled) < len(run_wells):
        tag += "_wells-" + "-".join(pooled)
    outputs = {}
    funnel_path = os.path.join(destination, f"ops_funnel_{plate}{tag}.parquet")
    funnel.frame(plate).to_parquet(funnel_path + ".part", index=False)
    os.replace(funnel_path + ".part", funnel_path)
    outputs["funnel_parquet"] = funnel_path
    if keep_cells:
        target = _ph_cells_table_path(destination, plate, tag)
        kept = cells[["well", "site", "label", "object_id", "mapped_guide",
                      "sgRNA", "gene", "mitotic"]].copy()
        kept.insert(0, "plate", plate)
        kept["mitotic"] = kept["mitotic"].astype(bool)
        kept.to_parquet(target + ".part", index=False)
        os.replace(target + ".part", target)
        outputs["cells_parquet"] = target
    for name, table in (("guides", guides), ("genes", genes)):
        for suffix in ("csv", "parquet"):
            target = os.path.join(destination,
                                  f"ops_{name}_{plate}{tag}.{suffix}")
            write_table(table, target, canonicalise=False)
            outputs[f"{name}_{suffix}"] = target
    tvn_report = None
    profile_report = None
    wants_profiles = any(settings.get(key) for key in _PH_PROFILE_SWITCHES)
    if wants_profiles and normalisation != "tvn":
        raise ValueError(tr(
            "the gene-profile options {keys} work on TVN profiles; set "
            "ops_profile_normalisation to tvn", keys=[
                key for key in _PH_PROFILE_SWITCHES if settings.get(key)]))
    if normalisation == "tvn":
        from .ops_tvn import _aggregate_tvn
        tvn_settings = {key: settings.get(key) for key in (
            "ops_tvn_mode", "ops_tvn_pca_sample", "ops_tvn_seed",
            "ops_tvn_components", "ops_tvn_whiten_epsilon",
            "ops_aggregate_level")}
        extra: List[str] = []
        if wants_profiles:
            tvn_settings["_keep_embedding"] = True
            extra = ["site", "label", "mitotic"]
            integrated = f"nucleus_{channels[0]}_integrated_intensity" \
                if channels else ""
            if integrated not in columns:
                integrated = next((c for c in features if c.startswith(
                    "nucleus_") and c.endswith("integrated_intensity")), "")
            if integrated:
                cells["_dna_integrated"] = _ph_read_columns(
                    sources, [integrated]).iloc[rows][integrated].to_numpy()
                if gain_columns:
                    from .ops_flatfield import correct_features
                    cells["_dna_integrated"] = correct_features(
                        cells.rename(columns={"_dna_integrated": integrated}),
                        cells, [integrated], flat_offsets)[integrated].to_numpy()
                extra.append("_dna_integrated")
        fit_cells = str(settings.get("ops_tvn_fit_cells") or "pooled").strip().lower()
        if fit_cells not in ("pooled", "interphase"):
            raise ValueError(tr(
                "ops_tvn_fit_cells must be pooled or interphase; got {value}",
                value=fit_cells))
        fit = (~cells["mitotic"].astype(bool)).to_numpy() \
            if fit_cells == "interphase" else None
        tvn_guides, tvn_genes, tvn_report = _aggregate_tvn(
            sources, rows, cells[["well", "sgRNA", "gene", *extra]], keep,
            control.to_numpy(), features, tvn_settings, _ph_read_columns,
            fit=fit)
        tvn_genes = tvn_genes[tvn_genes["gene"] != "nontargeting_drop"
                              ].reset_index(drop=True)
        if wants_profiles:
            profile_report = _ph_profile_options(
                settings, tvn_report.pop("_cells"),
                tvn_report.pop("_embedding"), tvn_guides, tvn_genes,
                destination, f"{plate}{tag}", control_prefix)
            outputs.update(profile_report["outputs"])
        for name, table in (("guides", tvn_guides), ("genes", tvn_genes)):
            for suffix in ("csv", "parquet"):
                target = os.path.join(destination,
                                      f"ops_{name}_{plate}{tag}_tvn.{suffix}")
                write_table(table, target, canonicalise=False)
                outputs[f"{name}_tvn_{suffix}"] = target
    cluster_report = None
    if settings.get("ops_cluster_benchmark"):
        from .ops_clusters import _aggregate_benchmark
        benchmark_settings = {key: settings.get(key) for key in (
            "ops_benchmark_corum", "ops_benchmark_kegg", "ops_benchmark_string",
            "ops_reference_clusters", "ops_cluster_target",
            "ops_cluster_resolutions", "ops_cluster_seed",
            "ops_consensus_runs")}
        if tvn_report is not None:
            profile = (tvn_guides, tvn_genes,
                       [c for c in tvn_genes.columns if c.startswith("PC_")],
                       f"{plate}{tag}_tvn")
        else:
            kept = [c for c in features if c not in dropped]
            profile = (guides, genes, kept, f"{plate}{tag}")
        cluster_report = _aggregate_benchmark(
            *profile[:3], benchmark_settings, destination, profile[3],
            control_prefix=control_prefix)
        outputs.update(cluster_report["outputs"])
    if feature_qc:
        target = os.path.join(destination,
                              f"ops_feature_qc_{plate}{tag}.csv")
        write_table(pd.DataFrame(degenerate, columns=[
            "feature", "reason", "control_cells", "control_finite",
            "control_median", "control_share_at_median",
            "wells_with_zero_control_mad"]), target, canonicalise=False)
        outputs["feature_qc_csv"] = target
    flagged: List[str] = []
    if gain_columns:
        from .ops_flatfield import flagged_features

        flagged = flagged_features(features, list(flat_offsets))
    report = {
        "wells": pooled, "cells_merged": int(len(with_guide)),
        "cells_with_guide": int(with_guide.sum()),
        "cells_in_design": int(len(cells)),
        "dropped_fields": field_report,
        "dropped_nonfinite_rows": nonfinite_dropped,
        "cross_field": cross_field_report,
        "cells_pooled": int(keep.sum()),
        "pooled": "interphase" if interphase else "all",
        "genotype_table": str(genotype_path) if genotype_path else None,
        "control_cells": int(control.sum()),
        "guides": int(len(guides)), "genes": int(len(genes)),
        "ambiguous_prefixes": ambiguous, "features": len(features),
        "features_dropped": degenerate,
        "mitotic_rule": {"dna": dna_column, "dna_z": mitotic_z,
                         "area": "nucleus_area", "area_z": 0.0}
        if mitotic_model is None else {
            "classifier": str(settings["ops_mitotic_model"]),
            "features": len(mitotic_model["features"]),
            "validation": mitotic_model.get("summary")},
        "ownership": ownership_counts or {"rule": ownership},
        "flatfield": flat_record or {"mode": flatfield},
        "flatfield_flagged_features": flagged,
        "mitotic_fraction_all": round(float(cells["mitotic"].mean()), 4)
        if len(cells) else None,
        "mitotic_fraction_controls": round(float(
            cells.loc[control, "mitotic"].mean()), 4) if control.any() else None,
        "bootstrap": boot, "bootstrap_features": tested,
        "pvalue_null": null if tested else None,
        "null_parameters": null_parameters,
        "profile_normalisation": normalisation,
        "aggregate_level": level, "nt_rows": nt_rows,
        "tvn": tvn_report,
        "profile_options": profile_report,
        "clusters": cluster_report,
        "outputs": outputs,
        "seconds": round(time.perf_counter() - started, 1),
    }
    _say(tr("{plate} aggregate: {guides} guides, {genes} genes from {cells} "
            "cells", plate=plate, guides=report["guides"],
            genes=report["genes"], cells=report["cells_in_design"]))
    return report


#: The layout name of the merged phenotype store ``ph_merged`` writes, as
#: ``merged/LAYOUT.json`` records it, so a reader can tell it from the
#: one-array-per-field ``.npy`` layout the Measure module writes.
_PH_MERGED_LAYOUT = "ops_merged_v1"

#: The registration modes ``ops_ph_secondary_registration`` may name.
_PH_SECONDARY_MODES = ("none", "phase_correlation")

#: A field's phase-correlation shift that lies further than this, in
#: pixels, from the well's running median is replaced by that median, once
#: five fields have been registered; the reference plates' acquisitions are
#: offset by about 20 to 35 px, and a field that registers 12 px away from
#: its neighbours has registered on noise.
_PH_SECONDARY_MAX_DEVIATION = 12.0


def _ph_secondary(settings: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    """The second-acquisition settings, or None when the plate has one.

    :param settings: read for ``ops_ph_secondary_channels``,
        ``ops_ph_secondary_registration``, ``ops_ph_secondary_window`` and
        ``ops_ph_secondary_fallback``.
    :returns: ``{"channels", "registration", "window", "fallback"}``.
    :raises ValueError: on an unknown registration or fallback mode.
    """
    from .qt.i18n import tr

    channels = [c.upper() for c in _ph_names(
        settings.get("ops_ph_secondary_channels"))]
    if not channels:
        return None
    mode = str(settings.get("ops_ph_secondary_registration")
               or "none").strip().lower()
    if mode not in _PH_SECONDARY_MODES:
        raise ValueError(tr(
            "ops_ph_secondary_registration must be one of {modes}; got "
            "{value}", modes=list(_PH_SECONDARY_MODES), value=mode))
    fallback = str(settings.get("ops_ph_secondary_fallback")
                   or "well_median").strip().lower()
    if fallback not in ("well_median", "none"):
        raise ValueError(tr(
            "ops_ph_secondary_fallback must be well_median or none; got "
            "{value}", value=fallback))
    window = int(_setting_number(settings.get("ops_ph_secondary_window"),
                                 "ops_ph_secondary_window", 1024))
    return {"channels": channels, "registration": mode, "window": window,
            "fallback": fallback}


def _ph_secondary_pair(files: Mapping[str, str], channels: Sequence[str]
                       ) -> Tuple[str, str]:
    """Which primary and which second-acquisition channel to register on.

    :param files: ``channel token -> path`` for one site.
    :param channels: the channels the second acquisition contributes.
    :returns: ``(primary channel, second-acquisition channel)``: a channel
        both acquisitions hold, registered on itself (Funk plate B's
        A594); else the primary's nuclear stain against the first channel
        of the second acquisition's file (plate C's DAPI against A594).
    :raises ValueError: when one of the two acquisitions is missing.
    """
    from .qt.i18n import tr

    primary, second = _ph_split_files(files, channels)
    held_primary = list(_plane_sources(primary))
    held_second = [name for token in second for name in token.split("-")]
    if not held_primary or not held_second:
        raise ValueError(tr(
            "ops_ph_secondary_channels names {channels}, but the site's files "
            "{files} are not two acquisitions", channels=list(channels),
            files=sorted(files)))
    tagged = [c for c in held_second
              if "@" in c and c.split("@")[0] in held_primary]
    if tagged:
        return tagged[0].split("@")[0], tagged[0]
    shared = [c for c in held_second if c in held_primary
              and c.upper() not in {x.upper() for x in channels}]
    if shared:
        return shared[0], shared[0]
    return _NUCLEAR, held_second[0]


def _ph_register_secondary(files: Mapping[str, str], channels: Sequence[str],
                           window: int, unreadable: list
                           ) -> Optional[Dict[str, Any]]:
    """The whole-pixel shift that puts one site's second acquisition on its first.

    :param files: ``channel token -> path`` for the site.
    :param channels: the channels the second acquisition contributes.
    :param window: the side of the centre square the shift is measured on.
    :param unreadable: ``(path, reason)`` is appended for a file that fails.
    :returns: ``{"dy", "dx", "raw_dy", "raw_dx", "error", "pair"}``, or None
        when a file cannot be read.

    The two max projections' centre squares are phase-correlated without
    normalisation (``skimage.registration.phase_cross_correlation``), as
    the plate B and C reference run did, and the shift is rounded to whole
    pixels so no channel is interpolated.
    """
    from skimage.registration import phase_cross_correlation

    reference, moving = _ph_secondary_pair(files, channels)
    primary, second = _ph_split_files(files, channels)
    first = _ph_read_field(primary, (reference,), unreadable)
    other = _ph_read_field(second, (moving,), unreadable)
    if first is None or other is None:
        return None
    height, width = first.shape[:2]
    half = int(window) // 2
    cy, cx = height // 2, width // 2
    a = first[cy - half:cy + half, cx - half:cx + half, 0]
    b = other[cy - half:cy + half, cx - half:cx + half, 0]
    shift, error, _phase = phase_cross_correlation(a, b, normalization=None)
    return {"dy": int(round(float(shift[0]))), "dx": int(round(float(shift[1]))),
            "raw_dy": float(shift[0]), "raw_dx": float(shift[1]),
            "error": float(error), "pair": f"{reference}/{moving}"}


def _ph_secondary_shifts(db: str, plate: str, well: str, phenotype_files,
                         settings: Mapping[str, Any]) -> Dict[str, Any]:
    """Register every placed field's second acquisition; store the shifts.

    :param db: the measurements database.
    :param plate: the plate name.
    :param well: the well name.
    :param phenotype_files: the phenotype index for this well.
    :param settings: the OPS settings (:func:`_ph_secondary`).
    :returns: the report.

    Writes ``secondary_dy``, ``secondary_dx`` and ``secondary_fallback``
    (and the unrounded ``secondary_raw_dy``/``secondary_raw_dx``) into the
    well's ``ops_phenotype`` rows, which every later reader of the field
    applies. The fields are registered in site order and the fallback is
    decided in that order: once five fields have their own shift, a shift
    more than :data:`_PH_SECONDARY_MAX_DEVIATION` px from the median of
    those is replaced by it. A well whose rows already carry the shifts is
    not registered again.
    """
    import pandas as pd

    from .qt.i18n import tr

    started = time.perf_counter()
    spec = _ph_secondary(settings)
    rows = _well_rows(db, "ops_phenotype", plate, well)
    if spec is None or spec["registration"] == "none":
        return {"skipped": "no second acquisition to register"}
    if rows is None or rows.empty:
        raise ValueError(tr(
            "well {well} has no ops_phenotype rows in {db}; run the "
            "phenotype phase first", well=well, db=db))
    if "secondary_dy" in rows.columns and rows["secondary_dy"].notna().all():
        return {"resumed": int(len(rows))}
    files = phenotype_files.get(min(phenotype_files), {}) if phenotype_files else {}
    history: List[Tuple[int, int]] = []
    found: Dict[int, Dict[str, Any]] = {}
    unreadable: list = []
    for site in sorted(int(s) for s in rows["site"]):
        if site not in files:
            continue
        shift = _ph_register_secondary(files[site], spec["channels"],
                                       spec["window"], unreadable)
        if shift is None:
            continue
        used = False
        if spec["fallback"] == "well_median" and len(history) >= 5:
            median = np.median(np.array(history, float), axis=0)
            if np.hypot(shift["dy"] - median[0],
                        shift["dx"] - median[1]) > _PH_SECONDARY_MAX_DEVIATION:
                shift["dy"] = int(round(float(median[0])))
                shift["dx"] = int(round(float(median[1])))
                used = True
        if not used:
            history.append((shift["dy"], shift["dx"]))
        shift["fallback"] = used
        found[site] = shift
    table = rows.drop(columns=[c for c in rows.columns
                               if c.startswith("secondary_")])
    shifts = pd.DataFrame(
        [{"site": s, "secondary_dy": v["dy"], "secondary_dx": v["dx"],
          "secondary_fallback": int(v["fallback"]),
          "secondary_raw_dy": v["raw_dy"], "secondary_raw_dx": v["raw_dx"]}
         for s, v in sorted(found.items())],
        columns=["site", "secondary_dy", "secondary_dx", "secondary_fallback",
                 "secondary_raw_dy", "secondary_raw_dx"])
    table = table.merge(shifts, on="site", how="left")
    _replace_well_rows(db, "ops_phenotype", table, plate, well)
    return {"registered": len(found),
            "fallbacks": int(sum(v["fallback"] for v in found.values())),
            "pair": next(iter(found.values()))["pair"] if found else None,
            "median_shift": ([float(np.median([v["dy"] for v in found.values()])),
                              float(np.median([v["dx"] for v in found.values()]))]
                             if found else None),
            "unreadable": unreadable,
            "seconds": round(time.perf_counter() - started, 1)}


def _ph_secondary_by_site(db: str, plate: str, well: str,
                          settings: Mapping[str, Any]) -> Dict[int, Any]:
    """``site -> secondary`` argument of :func:`_ph_read_field` for one well.

    :param db: the measurements database.
    :param plate: the plate name.
    :param well: the well name.
    :param settings: the OPS settings.
    :returns: an empty dict for a plate of one acquisition; else every
        placed site, with its stored shift (zero without registration).
    """
    spec = _ph_secondary(settings)
    if spec is None:
        return {}
    rows = _well_rows(db, "ops_phenotype", plate, well)
    if rows is None:
        return {}
    out = {}
    for record in rows.to_dict("records"):
        dy = record.get("secondary_dy")
        dx = record.get("secondary_dx")
        out[int(record["site"])] = {
            "channels": spec["channels"],
            "dy": 0 if dy is None or not np.isfinite(dy) else int(dy),
            "dx": 0 if dx is None or not np.isfinite(dx) else int(dx)}
    return out


def _ph_merged_root(settings: Mapping[str, Any], destination: str) -> str:
    """Where ``merged/`` and the crop store go: ``ops_merged_root``, else
    ``dst_root``."""
    return str(settings.get("ops_merged_root") or destination)


def _ph_merged_path(root: str, plate: str, well: str, site: int) -> str:
    """One field's merged arrays: ``merged/phenotype/<plate>/<well>/<site>.npz``."""
    return os.path.join(root, "merged", "phenotype", str(plate), str(well),
                        f"{int(site)}.npz")


def _ph_owned_flags(destination: str, well: str, settings: Mapping[str, Any]):
    """Each cell's ownership, as ``{(site, label): 0 or 1}``, or None.

    :param destination: the run folder.
    :param well: the well name.
    :param settings: the OPS settings.
    :returns: 1 for a cell this field owns, 0 for a copy another field
        owns, from ``ops_ownership_<well>.parquet`` (``ops_ownership``
        ``'a1'`` or ``'a1b'``, :func:`_ph_merge`); None when no ownership
        was decided, so every cell counts as owned. With
        ``ops_cross_field_ownership`` on and a policy other than
        ``'flag_only'``, a copy that is not its physical cell's owner
        (:func:`_ph_cross_field_ownership`) is a 0 as well, so merged/ and
        the crops hold one copy per cell.
    """
    import pandas as pd

    flags = None
    path = _ph_ownership_path(destination, well)
    if os.path.isfile(path):
        owned = pd.read_parquet(path, columns=["site", "label", "owned"])
        flags = {(int(s), int(l)): int(bool(o)) for s, l, o in
                 zip(owned["site"], owned["label"], owned["owned"])}
    cross = _ph_cross_field_path(destination, well)
    if (bool(settings.get("ops_cross_field_ownership", False))
            and _ph_cross_field_policy(settings) != "flag_only"
            and os.path.isfile(cross)):
        table = pd.read_parquet(cross, columns=["site", "label", "is_owner"])
        flags = dict(flags or {})
        for site, label, owner in zip(table["site"], table["label"],
                                      table["is_owner"]):
            key = (int(site), int(label))
            flags[key] = int(flags.get(key, 1) and bool(owner))
    return flags


def _ph_edge_labels(cells: np.ndarray) -> np.ndarray:
    """The labels of ``cells`` that touch the field's border."""
    border = np.concatenate([cells[0], cells[-1], cells[:, 0], cells[:, -1]])
    return np.unique(border[border > 0])


def _ph_image_row(stack: np.ndarray, channels: Sequence[str], nuclei,
                  cells) -> Dict[str, Any]:
    """The field's ``image`` table values (item 814's columns, the cheap ones).

    :param stack: the ``(H, W, C)`` field as stored in ``merged/``.
    :param channels: its channel names.
    :param nuclei: the nucleus labels.
    :param cells: the cell labels.
    :returns: per channel ``channel_<c>_background_median``, ``_p25``,
        ``_mean``, ``_p1``, ``_p99`` and ``_saturated_fraction`` (pixels at
        65,535) over the pixels outside every cell; ``cell_count``,
        ``nucleus_count`` and their ``_area_fraction``.
    """
    outside = cells == 0
    row: Dict[str, Any] = {}
    for index, channel in enumerate(channels):
        name = str(channel).lower()
        plane = stack[..., index]
        background = plane[outside].astype(np.float64)
        if background.size:
            p1, p25, p50, p99 = np.percentile(background, [1, 25, 50, 99])
        else:
            p1 = p25 = p50 = p99 = float("nan")
        row[f"channel_{name}_background_median"] = float(p50)
        row[f"channel_{name}_background_p25"] = float(p25)
        row[f"channel_{name}_mean"] = float(plane.mean(dtype=np.float64))
        row[f"channel_{name}_p1"] = float(p1)
        row[f"channel_{name}_p99"] = float(p99)
        row[f"channel_{name}_saturated_fraction"] = float(
            (plane >= 65535).mean())
    for role, labels in (("cell", cells), ("nucleus", nuclei)):
        present = np.unique(labels[labels > 0])
        row[f"{role}_count"] = int(present.size)
        row[f"{role}_area_fraction"] = float((labels > 0).mean())
    return row


def _ph_merged_field(task: Mapping[str, Any]) -> Dict[str, Any]:
    """Write one field's ``merged/`` arrays; the ``ph_merged`` worker.

    :param task: ``site``, ``files``, ``channels``, ``mask_path``,
        ``out_path``, ``flat_folder`` (or None), ``dtype``, ``owned``
        (``{label: flag}`` or None), ``meta`` and ``secondary``.
    :returns: ``site``, ``cells``, ``unreadable`` and the field's ``image``
        row.
    """
    unreadable: list = []
    stack = _ph_read_field(task["files"], task["channels"], unreadable,
                           task.get("secondary"))
    if stack is None:
        return {"site": task["site"], "cells": None, "unreadable": unreadable}
    with np.load(task["mask_path"]) as saved:
        nuclei = saved["nuclei"].astype(np.int32)
        cells = saved["cells"].astype(np.int32)
    if task.get("flat_folder"):
        from .ops_flatfield import correct_field

        planes, offsets, _inverse = _ph_flat_planes(
            task["flat_folder"], task["channels"], stack.shape[:2])
        stack = correct_field(stack, planes, offsets, task["channels"])
    if task["dtype"] == "uint16":
        stored = np.clip(np.rint(stack), 0, 65535).astype(np.uint16)
    else:
        stored = stack.astype(np.float32)
    cytoplasm = np.where(nuclei > 0, 0, cells).astype(np.int32)
    size = int(cells.max()) + 1 if cells.size else 1
    owned = np.zeros(size, np.uint8)
    present = np.unique(cells[cells > 0])
    given = task.get("owned")
    for label in present:
        owned[label] = 1 if given is None else int(given.get(int(label), 1))
    owned[_ph_edge_labels(cells)] = 2
    image = _ph_image_row(stored, task["channels"], nuclei, cells)
    meta = dict(task["meta"], image=image)
    part = task["out_path"] + ".part.npz"
    os.makedirs(os.path.dirname(task["out_path"]), exist_ok=True)
    np.savez_compressed(part, channels=np.moveaxis(stored, -1, 0),
                        nuclei=nuclei.astype(np.uint32),
                        cells=cells.astype(np.uint32),
                        cytoplasm=cytoplasm.astype(np.uint32), owned=owned,
                        meta=np.array(json.dumps(meta, default=str)))
    os.replace(part, task["out_path"])
    return {"site": task["site"], "cells": int(present.size),
            "unreadable": unreadable, "image": image}


def _ph_read_merged(path: str) -> Dict[str, Any]:
    """One ``ops_merged_v1`` field: its arrays and its decoded ``meta``.

    :param path: a ``merged/phenotype/<plate>/<well>/<site>.npz``.
    :returns: ``channels`` as ``(C, H, W)``, ``nuclei``, ``cells``,
        ``cytoplasm``, ``owned`` and ``meta`` (a dict).
    """
    with np.load(path) as saved:
        out = {name: saved[name] for name in saved.files if name != "meta"}
        out["meta"] = json.loads(str(saved["meta"]))
    return out


def _ph_merged_layout(root: str) -> Optional[Dict[str, Any]]:
    """The ``merged/LAYOUT.json`` record under ``root``, or None.

    :param root: an ``ops_merged_root`` or a folder holding ``merged/``.
    :returns: the record when ``merged/`` is the OPS layout
        (:data:`_PH_MERGED_LAYOUT`), else None -- the Measure module's
        one-``.npy``-per-field layout has no such record.
    """
    path = os.path.join(root, "merged", "LAYOUT.json")
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as handle:
        record = json.load(handle)
    return record if record.get("layout") == _PH_MERGED_LAYOUT else None


def _ph_merged(db: str, plate: str, well: str, phenotype_files,
               settings: Mapping[str, Any], destination: str) -> Dict[str, Any]:
    """Write one well's fields to ``merged/``; the ``ph_merged`` phase.

    :param db: the measurements database.
    :param plate: the plate name.
    :param well: the well name.
    :param phenotype_files: the phenotype index for this well.
    :param settings: read for ``ops_merged_root``, ``ops_merged_flatfield``,
        ``ops_merged_dtype``, ``ops_merged_save_sbs``, ``ops_flatfield_maps``,
        ``ops_ph_channels``, ``ops_ph_sites``, the ``ops_ph_secondary_*``
        options and ``n_workers``.
    :param destination: the run folder.
    :returns: the report.

    One ``np.savez_compressed`` file per placed, segmented field, at
    ``<ops_merged_root>/merged/phenotype/<plate>/<well>/<site>.npz``:
    ``channels`` (``ops_ph_channels`` order, ``(C, H, W)``; flat-field
    corrected as ``(I - b) / F + b`` with ``ops_merged_flatfield`` when the
    plate's maps exist, then rounded to uint16 or kept as float32 by
    ``ops_merged_dtype``); the ``nuclei``, ``cells`` and ``cytoplasm``
    labels at full resolution; ``owned``, one flag per cell label (1 owned
    by this field, 0 a copy another field owns, 2 touching the field's
    edge; see :func:`_ph_owned_flags`); and ``meta``, a JSON string with
    the plate, well, site, the field's transform, the field's ``image``
    row and the spaCR version and commit. ``merged/LAYOUT.json`` names the
    layout.

    Each file is written as ``.part`` and renamed, and a field whose file
    exists is not redone; ``part_options.json`` in the well's folder
    refuses a rerun with other options (:func:`_ph_options_marker`).
    """
    import multiprocessing

    from .qt.i18n import tr
    from .resource_log import _parallel_process_executor as ProcessPoolExecutor
    from .run_journal import _git_hash

    started = time.perf_counter()
    channels = _ph_channels(settings)
    root = _ph_merged_root(settings, destination)
    dtype = str(settings.get("ops_merged_dtype") or "uint16").strip().lower()
    if dtype not in ("uint16", "float32"):
        raise ValueError(tr(
            "ops_merged_dtype must be uint16 or float32; got {value}",
            value=dtype))
    mask_dir = os.path.join(destination, well, "phenotype_masks")
    if not os.path.isdir(mask_dir):
        raise ValueError(tr(
            "well {well} has no masks in {folder}; run ph_segment first",
            well=well, folder=mask_dir))
    files = phenotype_files.get(min(phenotype_files), {}) if phenotype_files else {}
    sites = sorted(int(name[5:-4]) for name in os.listdir(mask_dir)
                   if name.startswith("site-") and name.endswith(".npz")
                   and ".part" not in name)
    wanted = _ph_sites(settings.get("ops_ph_sites"))
    sites = [s for s in sites if (wanted is None or s in wanted)]
    missing_files = [s for s in sites if s not in files]
    sites = [s for s in sites if s in files]
    flat_folder = None
    options: Dict[str, Any] = {"channels": list(channels), "dtype": dtype,
                               "flatfield": None, "secondary": None,
                               "ownership": _ph_ownership_rule(settings),
                               "cross_field": (
                                   _ph_cross_field_policy(settings)
                                   if settings.get("ops_cross_field_ownership")
                                   else None)}
    if bool(settings.get("ops_merged_flatfield", True)):
        folder = _ph_flat_folder(settings, destination)
        if os.path.isfile(os.path.join(folder, "flat_info.json")):
            from .ops_flatfield import load_flat_maps

            flat_folder = folder
            options["flatfield"] = load_flat_maps(folder, channels)[2]["md5"]
    secondary = _ph_secondary_by_site(db, plate, well, settings)
    if secondary:
        options["secondary"] = _ph_secondary(settings)
    out_dir = os.path.dirname(_ph_merged_path(root, plate, well, 0))
    os.makedirs(out_dir, exist_ok=True)
    _ph_options_marker(out_dir, options)
    os.makedirs(os.path.join(root, "merged"), exist_ok=True)
    layout_path = os.path.join(root, "merged", "LAYOUT.json")
    with open(layout_path + ".part", "w", encoding="utf-8") as handle:
        json.dump({"layout": _PH_MERGED_LAYOUT, "channels": list(channels),
                   "masks": ["nuclei", "cells", "cytoplasm"],
                   "flags": {"owned": {"0": "copy owned by another field",
                                       "1": "owned by this field",
                                       "2": "touches the field edge"}},
                   "dtype": dtype, "axes": "CYX"}, handle, indent=1)
    os.replace(layout_path + ".part", layout_path)
    placed = _ph_transforms(db, plate, well, destination)
    by_site = {int(r["site"]): r for r in placed.to_dict("records")}
    owned = _ph_owned_flags(destination, well, settings)
    commit = _git_hash()
    from . import __version__

    tasks = []
    for site in sites:
        out_path = _ph_merged_path(root, plate, well, site)
        if os.path.isfile(out_path):
            continue
        record = by_site.get(site, {})
        transform = {key: record.get(key) for key in record
                     if key in _PH_TRANSFORM_COLUMNS
                     or str(key).startswith(("refined_", "secondary_"))}
        tasks.append({
            "site": site, "files": files[site], "channels": channels,
            "mask_path": os.path.join(mask_dir, f"site-{site}.npz"),
            "out_path": out_path, "flat_folder": flat_folder, "dtype": dtype,
            "owned": (None if owned is None else
                      {label: flag for (s, label), flag in owned.items()
                       if s == site}),
            "secondary": secondary.get(site),
            "meta": {"plate": plate, "well": well, "site": site,
                     "transform": transform, "spacr_version": __version__,
                     "spacr_commit": commit}})
    workers = max(1, min(int(settings.get("n_workers") or 1), len(tasks) or 1))
    results = []
    if workers == 1:
        results = [_ph_merged_field(task) for task in tasks]
    else:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(workers, mp_context=context) as pool:
            results = list(pool.map(_ph_merged_field, tasks))
    unreadable = [item for r in results for item in r["unreadable"]]
    report = {"fields": len(sites), "written": sum(
        1 for r in results if r["cells"] is not None),
              "resumed": len(sites) - len(tasks),
              "fields_without_files": missing_files,
              "unreadable": unreadable, "root": root,
              "flatfield_md5": options["flatfield"], "dtype": dtype,
              "save_sbs": ("not built: ops_merged_save_sbs is reserved for "
                           "item 813's SBS stacks"
                           if settings.get("ops_merged_save_sbs") else False),
              "workers": workers,
              "seconds": round(time.perf_counter() - started, 1)}
    _say(tr("{well} ph_merged: {count} fields written to {folder}",
            well=well, count=report["written"], folder=out_dir))
    return report


#: The compartments the standard tables hold, one table each.
_PH_TABLE_COMPARTMENTS = ("cell", "nucleus", "cytoplasm")

#: The key columns every ``ph_tables`` row carries, in this order.
_PH_TABLE_KEYS = ("prcfo", "plateID", "rowID", "columnID", "fieldID",
                  "object_label", "object_id", "sgRNA", "gene",
                  "genotype_rule", "mitotic", "owned")


def _ph_cells_table_path(destination: str, plate: str, tag: str = "") -> str:
    """Where the aggregate phase writes each pooled cell's genotype and call."""
    return os.path.join(destination, f"ops_cells_{plate}{tag}.parquet")


def _ph_genotype_rule(settings: Mapping[str, Any]) -> str:
    """Which genotype the tables carry: the swapped table or the run's decode."""
    path = settings.get("ops_genotype_table")
    if path:
        return f"ops_genotype_table:{os.path.basename(str(path))}"
    return f"ops_barcodes:{settings.get('ops_cell_rule') or 'mapped'}"


def _ph_prcfo(plate: str, row: str, column: str, site: int, label: int) -> str:
    """A cell's ``prcfo``, the plate as the ``image`` table writes it.

    :func:`spacr.schema.compose_prcfo` escapes an underscore in the plate
    for file names; the tables keep the plate as named, as
    :func:`spacr.measure._image_table_keys` does, so the keys join.
    """
    from . import schema

    return "_".join([str(plate), row, column, schema.field_id(int(site)),
                     schema.object_id(int(label))])


def _ph_owned_cells(destination: str, well: str, cells, settings):
    """Which of a well's cells are owned and whole, as a boolean array.

    :param destination: the run folder.
    :param well: the well name.
    :param cells: the well's cell table (``site``, ``label``).
    :param settings: the OPS settings.
    :returns: True for a cell whose field owns it (:func:`_ph_owned_flags`;
        every cell when no ownership was decided) and that does not touch
        its field's edge (``ops_maskpass_<well>``, or every cell when the
        mask pass was not run).
    """
    import pandas as pd

    keys = pd.MultiIndex.from_arrays([cells["site"].astype(int),
                                      cells["label"].astype(int)])
    owned = np.ones(len(cells), bool)
    flags = _ph_owned_flags(destination, well, settings)
    if flags is not None:
        owned &= np.array([bool(flags.get(key, 0)) for key in keys], bool)
    path = _ph_maskpass_path(destination, well)
    if os.path.isfile(path):
        edges = pd.read_parquet(path, columns=["site", "label",
                                               "touches_field_edge"])
        cut = pd.Series(edges["touches_field_edge"].astype(bool).to_numpy(),
                        index=pd.MultiIndex.from_arrays(
                            [edges["site"].astype(int),
                             edges["label"].astype(int)]))
        owned &= ~np.array([True if pd.isna(v) else bool(v)
                            for v in cut.reindex(keys)], bool)
    return owned


def _ph_table_frames(plate: str, well: str, cells, links, genotype,
                     owned, rule: str, integer_keys: bool = False):
    """The key columns and compartment features of one well's owned cells.

    :param plate: the plate name.
    :param well: the well name.
    :param cells: the well's phenotype cell table.
    :param links: ``site, label, object_id`` from ``ops_merge``.
    :param genotype: ``site, label, sgRNA, gene, mitotic`` per pooled cell,
        or None.
    :param owned: :func:`_ph_owned_cells`.
    :param rule: :func:`_ph_genotype_rule`.
    :param integer_keys: also key each cell by ``cell_key``, the integer
        :func:`spacr.ops_run_infra.integer_cell_key` (item 872, B17).
    :returns: ``(keys, {compartment: features})``, row-aligned.
    """
    import pandas as pd

    from . import schema

    kept = cells[owned].reset_index(drop=True)
    row, column = schema.parse_well(well)
    keys = pd.DataFrame({
        "prcfo": [_ph_prcfo(plate, row, column, s, l)
                  for s, l in zip(kept["site"], kept["label"])],
        "plateID": str(plate), "rowID": row, "columnID": column,
        "fieldID": [schema.field_id(int(s)) for s in kept["site"]],
        "object_label": kept["label"].astype(np.int64).to_numpy()})
    if integer_keys:
        from .ops_run_infra import integer_cell_key

        keys.insert(1, "cell_key", integer_cell_key(
            int(row[1:]), int(column[1:]), kept["site"].to_numpy(),
            kept["label"].to_numpy()))
    lookup = kept[["site", "label"]].astype(np.int64)
    if links is not None and len(links):
        lookup = lookup.merge(links[["site", "label", "object_id"]].astype(
            {"site": np.int64, "label": np.int64}), on=["site", "label"],
            how="left")
    else:
        lookup["object_id"] = np.nan
    if genotype is not None and len(genotype):
        lookup = lookup.merge(genotype[["site", "label", "sgRNA", "gene",
                                        "mitotic"]].astype(
            {"site": np.int64, "label": np.int64}), on=["site", "label"],
            how="left")
    else:
        lookup = lookup.assign(sgRNA=None, gene=None, mitotic=np.nan)
    keys["object_id"] = lookup["object_id"].to_numpy()
    keys["sgRNA"] = lookup["sgRNA"].to_numpy()
    keys["gene"] = lookup["gene"].to_numpy()
    keys["genotype_rule"] = rule
    keys["mitotic"] = lookup["mitotic"].to_numpy()
    keys["owned"] = 1
    features = {}
    for compartment in _PH_TABLE_COMPARTMENTS:
        names = [c for c in kept.columns
                 if str(c).startswith(compartment + "_")
                 and not (compartment == "cell"
                          and str(c).startswith("cell_id"))]
        features[compartment] = kept[names].reset_index(drop=True)
    return keys, features


def _ph_insert_rows(conn, table: str, frame) -> None:
    """Append ``frame`` to ``table`` in one transaction, creating or widening it.

    :param conn: an open sqlite3 connection.
    :param table: the table.
    :param frame: the rows.

    One ``executemany`` per frame inside one ``BEGIN``/``COMMIT`` (item
    810): an autocommit connection would sync every row.
    """
    from .utils import _sqlite_identifier, _sqlite_value

    if frame.empty and len(frame.columns) == 0:
        return
    quoted = _sqlite_identifier(table)
    existing = [r[1] for r in conn.execute(f"PRAGMA table_info({quoted})")]

    def declared(series) -> str:
        """The SQLite type of a column."""
        kind = series.dtype.kind
        return {"i": "INTEGER", "u": "INTEGER", "b": "INTEGER",
                "f": "REAL"}.get(kind, "TEXT")

    if not existing:
        conn.execute(f"CREATE TABLE {quoted} (" + ", ".join(
            f"{_sqlite_identifier(str(c))} {declared(frame[c])}"
            for c in frame.columns) + ")")
    else:
        for column in frame.columns:
            if str(column) not in existing:
                conn.execute(f"ALTER TABLE {quoted} ADD COLUMN "
                             f"{_sqlite_identifier(str(column))} "
                             f"{declared(frame[column])}")
    columns = ", ".join(_sqlite_identifier(str(c)) for c in frame.columns)
    marks = ", ".join("?" for _ in frame.columns)
    values = [[_sqlite_value(v) for v in frame[c].tolist()]
              for c in frame.columns]
    conn.execute("BEGIN")
    try:
        conn.executemany(f"INSERT INTO {quoted} ({columns}) VALUES ({marks})",
                         zip(*values))
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise


def _ph_tables(db: str, plate: str, wells: Sequence[str],
               settings: Mapping[str, Any], destination: str,
               genotype_path: Optional[str] = None) -> Dict[str, Any]:
    """Write spaCR's standard measurement tables; the ``ph_tables`` phase.

    :param db: the run's OPS database (``ops_merge`` is read from it).
    :param plate: the plate name.
    :param wells: the wells run.
    :param settings: read for ``ops_tables_parquet_threshold``,
        ``ops_genotype_table`` and the ownership settings.
    :param destination: the run folder; the tables go to
        ``<dst_root>/measurements/measurements.db``.
    :param genotype_path: the aggregate phase's per-cell genotype table
        (``ops_cells_<plate>.parquet``); found in ``destination`` when None.
    :returns: the report.

    One row per owned, whole cell (:func:`_ph_owned_cells`) in ``cell``,
    ``nucleus`` and ``cytoplasm``: the keys of :data:`_PH_TABLE_KEYS` --
    ``prcfo`` (``plate_rR_cC_fSITE_oLABEL``, :func:`spacr.schema.compose_prcfo`),
    the plate, row, column and field ids, the cell's label, its sequencing
    nucleus (``object_id``, empty when unlinked), the guide and gene the
    aggregate phase pooled it under, which genotype that was, its mitotic
    call and ``owned`` -- then the compartment's features. ``image`` gets
    one row per field from ``merged/`` (:func:`_ph_merged`), when that was
    written.

    ``ops_tables_integer_keys`` adds ``cell_key``, an integer that joins
    as ``prcfo`` does (:func:`spacr.ops_run_infra.integer_cell_key`), and
    indexes it; ``ops_tables_layout`` ``'wide'`` also writes ``cell_wide``,
    one row per cell with the keys and the cell, nucleus and cytoplasm
    features side by side, which equals the three tables joined (item 872,
    B17).

    The rows are written per well in one transaction per table (item 810),
    and the join keys are indexed afterwards (item 811). When the plate
    holds more owned cells than ``ops_tables_parquet_threshold``, the
    features go to ``measurements/<table>/<plate>_<well>.parquet`` instead
    and the database keeps the key columns only. The database is rebuilt
    on every run.
    """
    import pandas as pd

    from .database_concurrency import connect
    from .qt.i18n import tr

    started = time.perf_counter()
    folder = os.path.join(destination, "measurements")
    os.makedirs(folder, exist_ok=True)
    target = os.path.join(folder, "measurements.db")
    for suffix in ("", "-wal", "-shm", "-journal"):
        if os.path.exists(target + suffix):
            os.remove(target + suffix)
    threshold = int(_setting_number(settings.get("ops_tables_parquet_threshold"),
                                    "ops_tables_parquet_threshold", 5_000_000))
    path = genotype_path or _ph_cells_table_path(destination, plate)
    genotype = pd.read_parquet(path) if os.path.isfile(path) else None
    links = _rows_for_wells(db, "ops_merge", plate, wells)
    rule = _ph_genotype_rule(settings)
    integer_keys = bool(settings.get("ops_tables_integer_keys", False))
    layout = str(settings.get("ops_tables_layout") or "compartments").strip(
        ).lower()
    if layout not in ("compartments", "wide"):
        raise ValueError(tr(
            "ops_tables_layout must be compartments or wide; got {value}",
            value=layout))
    plans = []
    total = 0
    for well in wells:
        cell_path = _ph_cells_path(destination, well)
        if not os.path.isfile(cell_path):
            continue
        cells = pd.read_parquet(cell_path, columns=["site", "label"])
        owned = _ph_owned_cells(destination, well, cells, settings)
        plans.append((well, cell_path, owned))
        total += int(owned.sum())
    if not plans:
        raise ValueError(tr("no phenotype cell table for wells {wells}",
                            wells=list(wells)))
    wide_to_parquet = total > threshold
    counts = Counter()
    parquet_files: List[str] = []
    conn = connect(target, timeout=60.0)
    try:
        conn.isolation_level = None
        for pragma in ("journal_mode=WAL", "synchronous=NORMAL",
                       "temp_store=MEMORY", "cache_size=-262144"):
            conn.execute(f"PRAGMA {pragma}")
        for well, cell_path, owned in plans:
            cells = pd.read_parquet(cell_path)
            well_links = (links[links["well"].astype(str) == str(well)]
                          if links is not None else None)
            well_genotype = (genotype[genotype["well"].astype(str) == str(well)]
                             if genotype is not None else None)
            keys, features = _ph_table_frames(plate, well, cells, well_links,
                                              well_genotype, owned, rule,
                                              integer_keys=integer_keys)
            del cells
            if layout == "wide":
                frame = pd.concat([keys, *(features[c] for c in
                                           _PH_TABLE_COMPARTMENTS)], axis=1)
                _ph_insert_rows(conn, "cell_wide", frame)
                counts["cell_wide"] += len(frame)
            for compartment in _PH_TABLE_COMPARTMENTS:
                if wide_to_parquet:
                    sub = os.path.join(folder, compartment)
                    os.makedirs(sub, exist_ok=True)
                    out = os.path.join(sub, f"{plate}_{well}.parquet")
                    pd.concat([keys[["prcfo"]], features[compartment]],
                              axis=1).to_parquet(out + ".part", index=False)
                    os.replace(out + ".part", out)
                    parquet_files.append(out)
                    frame = keys
                else:
                    frame = pd.concat([keys, features[compartment]], axis=1)
                _ph_insert_rows(conn, compartment, frame)
                counts[compartment] += len(frame)
        merged_root = _ph_merged_root(settings, destination)
        image_rows = []
        for well, _cell_path, _owned in plans:
            field_dir = os.path.dirname(_ph_merged_path(merged_root, plate,
                                                        well, 0))
            if not os.path.isdir(field_dir):
                continue
            for name in sorted(os.listdir(field_dir)):
                if not name.endswith(".npz") or ".part" in name:
                    continue
                with np.load(os.path.join(field_dir, name)) as saved:
                    meta = json.loads(str(saved["meta"]))
                image_rows.append(_ph_image_record(plate, well,
                                                   int(meta["site"]),
                                                   meta.get("image") or {}))
        tick = time.perf_counter()
        for compartment in _PH_TABLE_COMPARTMENTS:
            conn.execute(f"CREATE INDEX IF NOT EXISTS ix_{compartment}_prcfo "
                         f"ON {compartment}(prcfo)")
            conn.execute(f"CREATE INDEX IF NOT EXISTS ix_{compartment}_field "
                         f"ON {compartment}(plateID, rowID, columnID, fieldID)")
            conn.execute(f"CREATE INDEX IF NOT EXISTS ix_{compartment}_object "
                         f"ON {compartment}(object_id)")
            conn.execute(f"CREATE INDEX IF NOT EXISTS ix_{compartment}_gene "
                         f"ON {compartment}(gene)")
            if integer_keys:
                conn.execute(f"CREATE INDEX IF NOT EXISTS "
                             f"ix_{compartment}_cell_key ON "
                             f"{compartment}(cell_key)")
        if layout == "wide":
            conn.execute("CREATE INDEX IF NOT EXISTS ix_cell_wide_prcfo ON "
                         "cell_wide(prcfo)")
        index_seconds = time.perf_counter() - tick
    finally:
        conn.close()
    if image_rows:
        from .measure import _write_image_rows

        counts["image"] = _write_image_rows(target, image_rows)
    report = {"db": target, "rows": dict(counts), "owned_cells": total,
              "wide_to_parquet": wide_to_parquet,
              "parquet_threshold": threshold,
              "parquet_files": len(parquet_files),
              "genotype_table": path if genotype is not None else None,
              "genotype_rule": rule,
              "integer_keys": integer_keys, "layout": layout,
              "index_seconds": round(index_seconds, 1),
              "seconds": round(time.perf_counter() - started, 1)}
    _say(tr("{plate} ph_tables: {rows} cells written to {db}", plate=plate,
            rows=counts["cell"], db=target))
    return report


def _ph_image_record(plate: str, well: str, site: int,
                     values: Mapping[str, Any]) -> Dict[str, Any]:
    """One ``image`` row for :func:`spacr.measure._write_image_rows`.

    :returns: ``file_name`` -- the field stem ``<plate>_<well>_<site>``,
        with the plate escaped as merged field names are, which
        :func:`spacr.measure._image_table_keys` parses into item 814's
        keys -- and the field's values.
    """
    from . import schema

    stem = "_".join([schema.escape_filename_component(plate), well, str(site)])
    return {"file_name": stem, **values}


def _ph_hash_fraction(*parts) -> float:
    """A stable number in [0, 1) from ``parts``: blake2b over ``"|".join``.

    The same cell gets the same number in every run and every process, so
    a sample drawn by it does not depend on the order the fields run in.
    """
    import hashlib

    digest = hashlib.blake2b("|".join(map(str, parts)).encode(),
                             digest_size=8)
    return int.from_bytes(digest.digest(), "big") / 2.0 ** 64


def _ph_crop_targets(plate: str, well: str, called, design,
                     settings: Mapping[str, Any]):
    """The sequencing nuclei a crop is cut for: ``object_id -> reason``.

    :param plate: the plate name.
    :param well: the well name.
    :param called: ``object_id, mapped_guide`` of the well's called nuclei,
        in the genotype table's order.
    :param design: :func:`_ph_design`'s ``prefix, sgRNA, gene``.
    :param settings: read for ``ops_crop_per_guide``, ``ops_crop_all_nt``,
        ``ops_crop_seed`` and ``ops_control_prefix``.
    :returns: ``{object_id: "guide_sample" | "control_guide"}``; None when
        ``ops_crop_per_guide`` is None, which keeps every linked cell.

    The rule of the plate A reference run: each called nucleus gets a
    random priority from a generator seeded by
    ``hash(plate, well, ops_crop_seed)``, and the ``ops_crop_per_guide``
    first of each targeting guide are kept; with ``ops_crop_all_nt`` every
    non-targeting nucleus is kept as well.
    """
    per_guide = settings.get("ops_crop_per_guide")
    if per_guide in (None, ""):
        return None
    prefix_gene = dict(zip(design["prefix"], design["gene"]))
    called = called[called["mapped_guide"].fillna("").astype(str) != ""]
    called = called.assign(gene=called["mapped_guide"].map(prefix_gene))
    called = called[called["gene"].notna()].copy()
    seed = int(_setting_number(settings.get("ops_crop_seed"), "ops_crop_seed",
                               0))
    rng = np.random.default_rng(int(_ph_hash_fraction(plate, well, seed)
                                    * 2 ** 32))
    called["priority"] = rng.random(len(called))
    control = str(settings.get("ops_control_prefix") or "nontargeting").lower()
    nt = called["gene"].str.lower().str.startswith(control)
    sample = (called[~nt].sort_values("priority").groupby("mapped_guide")
              .head(int(per_guide)))
    out = {int(o): "guide_sample" for o in sample["object_id"]}
    if bool(settings.get("ops_crop_all_nt", True)):
        out.update({int(o): "control_guide"
                    for o in called.loc[nt, "object_id"]})
    return out


def _ph_cut_window(array: np.ndarray, centre, size: int) -> np.ndarray:
    """The ``size`` window :func:`spacr.utils._crop_center` cuts at ``centre``.

    :param array: a 2-D or ``(H, W, C)`` array.
    :param centre: the integer centre, in ``array``'s pixels.
    :param size: the window side.
    :returns: the window, zero outside ``array``.
    """
    pad = int(size)
    padded = np.pad(array, ((pad, pad), (pad, pad))
                    + ((0, 0),) * (array.ndim - 2))
    top = max(0, int(centre[0]) + pad - size // 2)
    left = max(0, int(centre[1]) + pad - size // 2)
    return padded[top:top + size, left:left + size]


def _ph_crop_cell(stack: np.ndarray, cells: np.ndarray, nuclei: np.ndarray,
                  box, label: int, size: int, with_mask: bool):
    """One cell's crop, cut as spaCR's crop export cuts it.

    :param stack: the ``(H, W, C)`` uint16 field.
    :param cells: the cell labels.
    :param nuclei: the nucleus labels.
    :param box: the cell's ``find_objects`` box.
    :param label: the cell's label.
    :param size: the crop side.
    :param with_mask: also return the ``(size, size)`` uint8 mask, 1 the
        cell and 2 its nucleus.
    :returns: ``(crop (C, size, size), mask or None, (cy, cx), edge)``.

    :func:`spacr.utils._crop_center` on a window of the cell's box grown by
    ``size``: pixels outside the cell are 0 and the crop is centred on the
    cell mask's centroid. Raw counts, no stretch (item 772).
    """
    from scipy import ndimage

    from .utils import _crop_center

    height, width = cells.shape
    y0, y1 = max(0, box[0].start - size), min(height, box[0].stop + size)
    x0, x1 = max(0, box[1].start - size), min(width, box[1].stop + size)
    local = (cells[y0:y1, x0:x1] == label).astype(np.int32)
    centre = np.round(ndimage.center_of_mass(local)).astype(int)
    crop = _crop_center(stack[y0:y1, x0:x1].copy(), local.copy(), size, size)
    mask = None
    if with_mask:
        body = _ph_cut_window(local.astype(np.uint8), centre, size)
        core = _ph_cut_window((nuclei[y0:y1, x0:x1] == label).astype(np.uint8),
                              centre, size)
        mask = body.copy()
        mask[(core > 0) & (body > 0)] = 2
    top, left = y0 + centre[0] - size // 2, x0 + centre[1] - size // 2
    edge = int(top < 0 or left < 0 or top + size > height
               or left + size > width)
    return (np.ascontiguousarray(np.moveaxis(crop, -1, 0)), mask,
            (int(y0 + centre[0]), int(x0 + centre[1])), edge)


def _ph_crop_bytes(crop: np.ndarray, mask: Optional[np.ndarray],
                   channels: Sequence[str]) -> bytes:
    """One crop as a TIFF: page 0 the ``(C, H, W)`` uint16 image, page 1 the mask."""
    import io

    import tifffile

    buffer = io.BytesIO()
    with tifffile.TiffWriter(buffer) as writer:
        writer.write(crop, compression="zstd", photometric="minisblack",
                     planarconfig="separate", predictor=True,
                     metadata={"axes": "CYX", "channels": list(channels)})
        if mask is not None:
            writer.write(mask, compression="zstd", photometric="minisblack")
    return buffer.getvalue()


def _ph_read_crop(shard: str, offset: int, size: int):
    """One crop back from a shard: ``(image (C, H, W), mask or None)``.

    :param shard: the ``.tar`` file.
    :param offset: the member's data offset, from the shard's index.
    :param size: the member's size.
    """
    import io

    import tifffile

    with open(shard, "rb") as handle:
        handle.seek(int(offset))
        data = handle.read(int(size))
    with tifffile.TiffFile(io.BytesIO(data)) as tiff:
        image = tiff.series[0].asarray()
        mask = tiff.series[1].asarray() if len(tiff.series) > 1 else None
    return image, mask


def _ph_shard_folder(root: str, plate: str, well: str, shard: int,
                     cap: int) -> str:
    """The folder of shard ``shard``, split so no folder holds over ``cap`` files.

    Each shard is two files (the tar and its index), so a well folder holds
    ``cap // 2`` shards before the next ones go to ``part-NNNN/``.
    """
    base = os.path.join(root, "data", "cell", str(plate), str(well))
    per_folder = max(1, int(cap) // 2)
    if shard < per_folder:
        return base
    return os.path.join(base, f"part-{shard // per_folder:04d}")


def _ph_write_shard(path: str, members) -> "Any":
    """Write ``members`` (``(name, bytes, record)``) to one tar; its index.

    :returns: the index frame: each record with ``member``, ``offset_data``
        and ``size``. The offsets are read back from the closed tar
        (``TarFile.getmembers``), not recorded while adding: ``addfile``
        works on a copy of the ``TarInfo``, so an offset taken from the
        caller's copy is wrong (the reference run's ``fix_shard_index.py``).
    """
    import io
    import tarfile

    import pandas as pd

    part = path + ".part"
    with tarfile.open(part, "w", format=tarfile.USTAR_FORMAT) as tar:
        for name, data, _record in members:
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mtime = 0
            tar.addfile(info, io.BytesIO(data))
    with tarfile.open(part, "r") as tar:
        where = {m.name: (m.offset_data, m.size) for m in tar.getmembers()}
    rows = []
    for name, _data, record in members:
        offset, size = where[name]
        rows.append({"member": name, "offset_data": int(offset),
                     "size": int(size), **record})
    index = pd.DataFrame(rows)
    index.to_parquet(path[:-len(".tar")] + ".index.parquet.part", index=False)
    os.replace(part, path)
    os.replace(path[:-len(".tar")] + ".index.parquet.part",
               path[:-len(".tar")] + ".index.parquet")
    return index


def _ph_crop_field(task: Mapping[str, Any]) -> Dict[str, Any]:
    """Cut one field's chosen cells; the ``ph_crops`` worker.

    :param task: ``site``, ``files``, ``channels``, ``crop_channels``,
        ``mask_path``, ``labels`` (``{label: (reason, object_id)}``),
        ``size``, ``flat_folder``, ``masks``, ``secondary``, ``plate``,
        ``well``; optionally ``flat_mean_one``, which scales each flat to
        mean 1 over the field before it divides.
    :returns: ``site``, ``members`` (``(name, bytes, record)``) and
        ``unreadable``.
    """
    from scipy import ndimage

    from . import schema

    unreadable: list = []
    stack = _ph_read_field(task["files"], task["channels"], unreadable,
                           task.get("secondary"))
    if stack is None:
        return {"site": task["site"], "members": [], "unreadable": unreadable}
    if task.get("flat_folder"):
        from .ops_flatfield import correct_field

        planes, offsets, _inverse = _ph_flat_planes(
            task["flat_folder"], task["channels"], stack.shape[:2])
        if task.get("flat_mean_one"):
            planes = {c: (p.astype(np.float64) / p.astype(np.float64).mean()
                          ).astype(np.float32) for c, p in planes.items()}
            offsets = {c: float(np.float32(v)) for c, v in offsets.items()}
        stack = correct_field(stack, planes, offsets, task["channels"])
    keep = [task["channels"].index(c) for c in task["crop_channels"]]
    stack = np.clip(np.rint(stack[..., keep]), 0, 65535).astype(np.uint16)
    with np.load(task["mask_path"]) as saved:
        nuclei = saved["nuclei"].astype(np.int32)
        cells = saved["cells"].astype(np.int32)
    boxes = ndimage.find_objects(cells)
    row, column = schema.parse_well(task["well"])
    members = []
    for label in sorted(task["labels"]):
        if label - 1 >= len(boxes) or boxes[label - 1] is None:
            continue
        reason, object_id = task["labels"][label]
        crop, mask, (cy, cx), edge = _ph_crop_cell(
            stack, cells, nuclei, boxes[label - 1], label, task["size"],
            task["masks"])
        prcfo = _ph_prcfo(task["plate"], row, column, task["site"], label)
        name = (f"site-{task['site']}/"
                f"{schema.escape_filename_component(task['plate'])}"
                f"{prcfo[len(str(task['plate'])):]}.tif")
        members.append((name, _ph_crop_bytes(crop, mask,
                                             task["crop_channels"]),
                        {"prcfo": prcfo, "plate": task["plate"],
                         "well": task["well"], "site": int(task["site"]),
                         "label": int(label), "object_id": object_id,
                         "cy": cy, "cx": cx, "window_edge": edge,
                         "reason": reason}))
    return {"site": task["site"], "members": members, "unreadable": unreadable}


def _ph_crops(db: str, plate: str, wells: Sequence[str], phenotype_index,
              settings: Mapping[str, Any], destination: str) -> Dict[str, Any]:
    """Cut raw crops of owned cells into a sharded store; the ``ph_crops`` phase.

    :param db: the measurements database.
    :param plate: the plate name.
    :param wells: the wells run.
    :param phenotype_index: the phenotype tile index.
    :param settings: read for ``ops_crop_per_guide``, ``ops_crop_size``,
        ``ops_crop_channels``, ``ops_crop_flatfield``, ``ops_crop_seed``,
        ``ops_crop_all_nt``, ``ops_crop_control_fraction``,
        ``ops_crop_masks``, ``ops_crop_flat_mean_one``, ``ops_max_files_per_folder``,
        ``ops_merged_root``, ``ops_design``, ``ops_genotype_table`` and the
        ``ops_ph_secondary_*`` options.
    :param destination: the run folder.
    :returns: the report.

    Which cells: owned, whole cells (:func:`_ph_owned_cells`) whose link
    (``ops_merge``) is one of the nuclei :func:`_ph_crop_targets` chose --
    every linked one when ``ops_crop_per_guide`` is None -- plus, with
    ``ops_crop_control_fraction`` above 0, every owned, whole cell whose
    ``hash(plate, well, site, label)`` falls below it, linked or not, so a
    cell a later decoder calls is still represented.

    Each crop is a TIFF member (``site-<n>/<prcfo>.tif``; page 0 the
    ``(C, H, W)`` uint16 crop, page 1 the mask when ``ops_crop_masks``) of a
    tar shard under
    ``<ops_merged_root>/data/cell/<plate>/<well>/shard-NNNN.tar`` with a
    ``shard-NNNN.index.parquet`` (``member``, ``offset_data``, ``size``,
    ``prcfo``, ``site``, ``label``, ``object_id``, ``cy``, ``cx``,
    ``window_edge``, ``reason``). A shard holds at most
    ``ops_max_files_per_folder`` crops and no folder more than that many
    files (item 809). A well with ``CROPS_DONE.json`` for the same options
    is not cut again.
    """
    import multiprocessing

    import pandas as pd

    from .qt.i18n import tr
    from .resource_log import _parallel_process_executor as ProcessPoolExecutor

    started = time.perf_counter()
    channels = _ph_channels(settings)
    crop_channels = tuple(c.upper() for c in _ph_names(
        settings.get("ops_crop_channels"))) or channels
    absent = [c for c in crop_channels if c not in channels]
    if absent:
        raise ValueError(tr(
            "ops_crop_channels names {absent}, which ops_ph_channels "
            "{channels} does not hold", absent=absent,
            channels=list(channels)))
    size = int(_setting_number(settings.get("ops_crop_size"), "ops_crop_size",
                               96))
    cap = int(_setting_number(settings.get("ops_max_files_per_folder"),
                              "ops_max_files_per_folder", 10000))
    fraction = float(_setting_number(settings.get("ops_crop_control_fraction"),
                                     "ops_crop_control_fraction", 0.01))
    with_masks = bool(settings.get("ops_crop_masks", False))
    root = _ph_merged_root(settings, destination)
    flat_folder = None
    if bool(settings.get("ops_crop_flatfield", True)):
        folder = _ph_flat_folder(settings, destination)
        if os.path.isfile(os.path.join(folder, "flat_info.json")):
            flat_folder = folder
    options = {"channels": list(crop_channels), "size": size,
               "per_guide": settings.get("ops_crop_per_guide"),
               "all_nt": bool(settings.get("ops_crop_all_nt", True)),
               "control_fraction": fraction, "masks": with_masks,
               "seed": settings.get("ops_crop_seed") or 0,
               "flatfield": flat_folder, "cap": cap,
               "flat_mean_one": bool(settings.get("ops_crop_flat_mean_one",
                                                  False)),
               "genotype": _ph_genotype_rule(settings)}
    design_path = settings.get("ops_design")
    sampled = (settings.get("ops_crop_per_guide") not in (None, "")
               and bool(design_path) and os.path.isfile(str(design_path)))
    links = _rows_for_wells(db, "ops_merge", plate, wells)
    report: Dict[str, Any] = {"wells": {}, "options": options}
    for well in wells:
        done_path = os.path.join(root, "data", "cell", str(plate), str(well),
                                 "CROPS_DONE.json")
        if os.path.isfile(done_path):
            with open(done_path, encoding="utf-8") as handle:
                done = json.load(handle)
            if done.get("options") == json.loads(json.dumps(options)):
                report["wells"][well] = {"resumed": True,
                                         "crops": done.get("crops")}
                continue
        cell_path = _ph_cells_path(destination, well)
        mask_dir = os.path.join(destination, well, "phenotype_masks")
        files = (phenotype_index.get(well) or {})
        files = files.get(min(files), {}) if files else {}
        if not os.path.isfile(cell_path) or not files:
            report["wells"][well] = {"skipped": "no cell table or no tiles"}
            continue
        cells = pd.read_parquet(cell_path, columns=["site", "label"])
        owned = _ph_owned_cells(destination, well, cells, settings)
        cells = cells[owned]
        link = (links[links["well"].astype(str) == str(well)]
                if links is not None else pd.DataFrame(
                    columns=["site", "label", "object_id"]))
        object_of = {(int(s), int(l)): int(o) for s, l, o in
                     zip(link["site"], link["label"], link["object_id"])}
        targets = None
        if sampled:
            if settings.get("ops_genotype_table"):
                called = _ph_genotype_table(str(settings["ops_genotype_table"]),
                                            plate, [well])
            else:
                called = _well_rows(db, "ops_barcodes", plate, well)
                called = (called[["object_id", "mapped_guide"]]
                          if called is not None else
                          pd.DataFrame(columns=["object_id", "mapped_guide"]))
            guides = called["mapped_guide"].dropna().astype(str)
            guides = guides[guides != ""]
            length = int(guides.str.len().mode().iloc[0]) if len(guides) else 11
            design = _ph_design(str(design_path), length)[0]
            targets = _ph_crop_targets(plate, well, called, design, settings)
        chosen: Dict[int, Dict[int, Tuple[str, Any]]] = {}
        for site, label in zip(cells["site"].astype(int),
                               cells["label"].astype(int)):
            object_id = object_of.get((site, label))
            reason = None
            if object_id is not None:
                if targets is None:
                    reason = "linked"
                elif object_id in targets:
                    reason = targets[object_id]
            if reason is None and fraction > 0 and _ph_hash_fraction(
                    plate, well, site, label) < fraction:
                reason = "random"
            if reason is not None:
                chosen.setdefault(site, {})[label] = (reason, object_id)
        secondary = _ph_secondary_by_site(db, plate, well, settings)
        tasks = [{"site": site, "files": files[site], "channels": channels,
                  "crop_channels": crop_channels,
                  "mask_path": os.path.join(mask_dir, f"site-{site}.npz"),
                  "labels": labels, "size": size, "flat_folder": flat_folder,
                  "masks": with_masks, "secondary": secondary.get(site),
                  "flat_mean_one": options["flat_mean_one"],
                  "plate": plate, "well": well}
                 for site, labels in sorted(chosen.items())
                 if site in files and os.path.isfile(
                     os.path.join(mask_dir, f"site-{site}.npz"))]
        workers = max(1, min(int(settings.get("n_workers") or 1),
                             len(tasks) or 1))
        if workers == 1:
            results = map(_ph_crop_field, tasks)
            pool = None
        else:
            pool = ProcessPoolExecutor(
                workers, mp_context=multiprocessing.get_context("spawn"))
            results = pool.map(_ph_crop_field, tasks)
        shard, pending, crops, unreadable, indexes = 0, [], 0, [], []
        try:
            for result in results:
                unreadable.extend(result["unreadable"])
                pending.extend(result["members"])
                while len(pending) >= cap:
                    folder = _ph_shard_folder(root, plate, well, shard, cap)
                    os.makedirs(folder, exist_ok=True)
                    indexes.append(_ph_write_shard(
                        os.path.join(folder, f"shard-{shard:04d}.tar"),
                        pending[:cap]))
                    crops += min(cap, len(pending))
                    pending = pending[cap:]
                    shard += 1
        finally:
            if pool is not None:
                pool.shutdown()
        if pending:
            folder = _ph_shard_folder(root, plate, well, shard, cap)
            os.makedirs(folder, exist_ok=True)
            indexes.append(_ph_write_shard(
                os.path.join(folder, f"shard-{shard:04d}.tar"), pending))
            crops += len(pending)
            shard += 1
        reasons = Counter(r for labels in chosen.values()
                          for r, _o in labels.values())
        well_report = {"crops": crops, "shards": shard,
                       "reasons": dict(reasons), "unreadable": unreadable,
                       "owned_cells": int(len(cells))}
        os.makedirs(os.path.dirname(done_path), exist_ok=True)
        with open(done_path, "w", encoding="utf-8") as handle:
            json.dump({"options": options, **well_report}, handle, indent=1,
                      default=str)
        report["wells"][well] = well_report
    report["seconds"] = round(time.perf_counter() - started, 1)
    _say(tr("{plate} ph_crops: {count} crops", plate=plate, count=sum(
        w.get("crops") or 0 for w in report["wells"].values())))
    return report



#: The scores ``ops_cross_field_metric`` may name, the first the default.
_PH_CROSS_FIELD_METRICS = ("overlap_fraction", "iou")

#: The policies ``ops_cross_field_policy`` may name, the first the default.
_PH_CROSS_FIELD_POLICIES = ("one_per_cell", "owner_only", "flag_only")

#: How many label images the ownership pass keeps in memory at once: 160
#: uint32 fields of 2,960 px are about 5.6 GB.
_PH_CROSS_FIELD_CACHE = 160


def _ph_cross_field_path(destination: str, well: str) -> str:
    """Where one well's cross-field ownership table is written."""
    return os.path.join(destination, f"ops_cross_field_{well}.parquet")


def _ph_cross_field_edges_path(destination: str, well: str) -> str:
    """Where one well's cross-field copy edges are written."""
    return os.path.join(destination, f"ops_cross_field_edges_{well}.parquet")


def _ph_cross_field_policy(settings: Mapping[str, Any]) -> str:
    """``ops_cross_field_policy``, checked against :data:`_PH_CROSS_FIELD_POLICIES`."""
    from .qt.i18n import tr

    policy = str(settings.get("ops_cross_field_policy")
                 or "one_per_cell").strip().lower()
    if policy not in _PH_CROSS_FIELD_POLICIES:
        raise ValueError(tr(
            "ops_cross_field_policy must be one of {policies}; got {value}",
            policies=list(_PH_CROSS_FIELD_POLICIES), value=policy))
    return policy


def _ph_rotation(degrees: float) -> np.ndarray:
    """The 2x2 rotation ``[[cos, -sin], [sin, cos]]`` of ``degrees``."""
    angle = np.radians(float(degrees))
    return np.array([[np.cos(angle), -np.sin(angle)],
                     [np.sin(angle), np.cos(angle)]])


def _ph_to_well(points, transform) -> np.ndarray:
    """Field ``(row, col)`` points in the well frame: ``s * R @ p + t``."""
    scale, rotation, offset = transform
    return scale * (np.asarray(points, float) @ rotation.T) + offset


def _ph_to_field(points, transform) -> np.ndarray:
    """Well points in a field's pixels: ``R^T (w - t) / s``."""
    scale, rotation, offset = transform
    return ((np.asarray(points, float) - offset) / scale) @ rotation


def _ph_field_record_transform(record: Mapping[str, Any]):
    """A field's transform from its ``ops_phenotype`` row: the re-solved one
    (``refined_*``) when ``refined == 1``, else the stored seed."""
    refined = record.get("refined") == 1
    scale = float(record["refined_scale"] if refined else record["scale"])
    degrees = float(record["refined_rotation_deg"] if refined
                    else record["rotation_deg"])
    offset = np.array([record["refined_offset_y"] if refined
                       else record["offset_y"],
                       record["refined_offset_x"] if refined
                       else record["offset_x"]], float)
    return scale, _ph_rotation(degrees), offset


def _ph_pair_intersections(cells_a: np.ndarray, transform_a,
                           cells_b: np.ndarray, transform_b):
    """``(label_a, label_b, pixels)`` for every pair of labels two fields share.

    :param cells_a: field a's cell labels.
    :param transform_a: its field-to-well transform.
    :param cells_b: field b's cell labels.
    :param transform_b: its transform.
    :returns: three int64 arrays. Every labelled pixel of field a inside
        the box field b covers is moved to the well and on into field b,
        rounded to the nearest pixel; the counts are in field-a pixels.
    """
    height_a, width_a = cells_a.shape
    height_b, width_b = cells_b.shape
    corners = np.array([[0, 0], [0, width_b], [height_b, 0],
                        [height_b, width_b]], float)
    in_a = _ph_to_field(_ph_to_well(corners, transform_b), transform_a)
    r0 = max(0, int(np.floor(in_a[:, 0].min())) - 1)
    r1 = min(height_a, int(np.ceil(in_a[:, 0].max())) + 1)
    c0 = max(0, int(np.floor(in_a[:, 1].min())) - 1)
    c1 = min(width_a, int(np.ceil(in_a[:, 1].max())) + 1)
    empty = (np.empty(0, np.int64),) * 3
    if r1 <= r0 or c1 <= c0:
        return empty
    rows, cols = np.mgrid[r0:r1, c0:c1]
    label_a = cells_a[r0:r1, c0:c1].ravel()
    keep = label_a > 0
    if not keep.any():
        return empty
    points = np.column_stack([rows.ravel()[keep],
                              cols.ravel()[keep]]).astype(float)
    label_a = label_a[keep]
    moved = np.rint(_ph_to_field(_ph_to_well(points, transform_a),
                                 transform_b)).astype(np.int64)
    inside = ((moved[:, 0] >= 0) & (moved[:, 0] < height_b)
              & (moved[:, 1] >= 0) & (moved[:, 1] < width_b))
    label_a, moved = label_a[inside], moved[inside]
    label_b = cells_b[moved[:, 0], moved[:, 1]]
    both = label_b > 0
    if not both.any():
        return empty
    code = (label_a[both].astype(np.int64) << 32) | label_b[both].astype(
        np.int64)
    unique, counts = np.unique(code, return_counts=True)
    return unique >> 32, unique & 0xFFFFFFFF, counts.astype(np.int64)


def _ph_pair_edges(label_a, label_b, inter, area_a, area_b, *,
                   metric: str = "overlap_fraction", min_overlap: float = 0.5):
    """The label pairs of two fields that are copies of one cell.

    :param label_a: field a's labels, one per shared pair.
    :param label_b: field b's labels.
    :param inter: the pairs' shared pixels.
    :param area_a: ``label -> pixels`` in field a.
    :param area_b: ``label -> pixels`` in field b.
    :param metric: ``'overlap_fraction'``, ``inter / min(area_a, area_b)``,
        or ``'iou'``, ``inter / (area_a + area_b - inter)``.
    :param min_overlap: the smallest score of a copy.
    :returns: ``label_a, label_b, inter, score`` for the pairs where each
        label is the other's best partner (most shared pixels, ties to the
        smaller label) and the score reaches ``min_overlap``.
    :raises ValueError: on an unknown metric.
    """
    import pandas as pd

    from .qt.i18n import tr

    if metric not in _PH_CROSS_FIELD_METRICS:
        raise ValueError(tr(
            "ops_cross_field_metric must be one of {metrics}; got {value}",
            metrics=list(_PH_CROSS_FIELD_METRICS), value=metric))
    if len(label_a) == 0:
        return pd.DataFrame(columns=["label_a", "label_b", "inter", "score"])
    frame = pd.DataFrame({"label_a": label_a, "label_b": label_b,
                          "inter": inter})
    best_a = (frame.sort_values(["label_a", "inter", "label_b"],
                                ascending=[True, False, True],
                                kind="mergesort").drop_duplicates("label_a"))
    best_b = (frame.sort_values(["label_b", "inter", "label_a"],
                                ascending=[True, False, True],
                                kind="mergesort").drop_duplicates("label_b"))
    mutual = best_a.merge(best_b[["label_a", "label_b"]],
                          on=["label_a", "label_b"])
    first = mutual["label_a"].map(area_a).astype(float).to_numpy()
    second = mutual["label_b"].map(area_b).astype(float).to_numpy()
    shared = mutual["inter"].astype(float).to_numpy()
    if metric == "iou":
        mutual["score"] = shared / (first + second - shared)
    else:
        mutual["score"] = shared / np.minimum(first, second)
    return mutual[mutual["score"] >= float(min_overlap)].reset_index(drop=True)


def _ph_assign_ownership(cells, edges):
    """Physical cells and their owners from the copies and their edges.

    :param cells: ``site, label, area`` per segmented copy.
    :param edges: ``site_a, label_a, site_b, label_b``.
    :returns: ``cells`` with ``component`` (the connected component of the
        undirected edge graph), ``n_copies``, ``owner_field``,
        ``owner_label``, ``is_owner`` and ``owned_area_share``. The owner is
        the largest copy, ties to the smaller site, then the smaller label.
    """
    import pandas as pd
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    cells = cells.reset_index(drop=True).copy()
    index = pd.Series(np.arange(len(cells)), index=pd.MultiIndex.from_arrays(
        [cells["site"], cells["label"]]))
    if len(edges):
        first = index.reindex(pd.MultiIndex.from_arrays(
            [edges["site_a"], edges["label_a"]])).to_numpy()
        second = index.reindex(pd.MultiIndex.from_arrays(
            [edges["site_b"], edges["label_b"]])).to_numpy()
        known = ~(np.isnan(first.astype(float)) | np.isnan(second.astype(float)))
        first = first[known].astype(np.int64)
        second = second[known].astype(np.int64)
    else:
        first = second = np.empty(0, np.int64)
    count = len(cells)
    graph = coo_matrix((np.ones(len(first)), (first, second)),
                       shape=(count, count))
    _n, component = connected_components(graph, directed=False)
    cells["component"] = component
    cells["n_copies"] = cells.groupby("component")["label"].transform("size")
    order = cells.sort_values(["component", "area", "site", "label"],
                              ascending=[True, False, True, True],
                              kind="mergesort")
    owners = order.drop_duplicates("component")[["component", "site", "label",
                                                 "area"]]
    owners.columns = ["component", "owner_field", "owner_label", "owner_area"]
    cells = cells.merge(owners, on="component", how="left")
    cells["is_owner"] = ((cells["site"] == cells["owner_field"])
                         & (cells["label"] == cells["owner_label"]))
    cells["owned_area_share"] = cells["area"] / cells["owner_area"]
    return cells.drop(columns=["owner_area"])


def _ph_cross_field_fields(fields: Mapping[int, Mapping[str, Any]], *,
                           metric: str, min_overlap: float, load=None,
                           log=None):
    """Cross-field ownership over a set of fields.

    :param fields: ``{site: {"transform", "shape", "areas" (label -> px),
        "cells" (label image or None)}}``.
    :param metric: :func:`_ph_pair_edges`'s score.
    :param min_overlap: its threshold.
    :param load: ``load(site)`` returns a label image not held in
        ``fields``; at most :data:`_PH_CROSS_FIELD_CACHE` are kept.
    :param log: called with a progress line every 250 field pairs.
    :returns: ``(cells, edges, pairs)``: :func:`_ph_assign_ownership`'s
        table, the edges (``site_a, site_b, label_a, label_b, inter,
        score``) in pair order, and how many field pairs overlapped.
    """
    from collections import OrderedDict

    import pandas as pd

    from .ops_merge import field_footprint, overlapping_fields

    sites = sorted(fields)
    prints = {s: field_footprint(fields[s]["transform"], fields[s]["shape"])
              for s in sites}
    neighbours = overlapping_fields(prints)
    pairs = [(a, b) for a in sites for b in neighbours[a] if b > a]
    pairs.sort()
    cache: "OrderedDict[int, np.ndarray]" = OrderedDict()

    def image(site: int) -> np.ndarray:
        """One field's label image, from ``fields`` or the cache."""
        if fields[site].get("cells") is not None:
            return fields[site]["cells"]
        if site in cache:
            cache.move_to_end(site)
            return cache[site]
        cache[site] = load(site)
        while len(cache) > _PH_CROSS_FIELD_CACHE:
            cache.popitem(last=False)
        return cache[site]

    found = []
    for count, (a, b) in enumerate(pairs, start=1):
        label_a, label_b, inter = _ph_pair_intersections(
            image(a), fields[a]["transform"], image(b), fields[b]["transform"])
        edges = _ph_pair_edges(label_a, label_b, inter, fields[a]["areas"],
                               fields[b]["areas"], metric=metric,
                               min_overlap=min_overlap)
        if len(edges):
            edges.insert(0, "site_b", b)
            edges.insert(0, "site_a", a)
            found.append(edges)
        if log and count % 250 == 0:
            log(count, len(pairs))
    edges = (pd.concat(found, ignore_index=True) if found else
             pd.DataFrame(columns=["site_a", "site_b", "label_a", "label_b",
                                   "inter", "score"]))
    for column in ("site_a", "site_b", "label_a", "label_b", "inter"):
        edges[column] = edges[column].astype(np.int64)
    edges["score"] = edges["score"].astype(float)
    cells = pd.concat([pd.DataFrame({
        "site": s, "label": fields[s]["areas"].index.astype(np.int64),
        "area": fields[s]["areas"].to_numpy(float)}) for s in sites],
        ignore_index=True) if sites else pd.DataFrame(
        columns=["site", "label", "area"])
    return _ph_assign_ownership(cells, edges), edges, len(pairs)


def _ph_cross_field_ownership(db: str, plate: str, well: str,
                              settings: Mapping[str, Any], destination: str
                              ) -> Dict[str, Any]:
    """Find every copy of a cell across overlapping fields; the ``ph_ownership`` phase.

    :param db: the measurements database (``ops_phenotype`` is read, never
        written).
    :param plate: the plate name.
    :param well: the well name.
    :param settings: read for ``ops_cross_field_ownership``,
        ``ops_cross_field_metric``, ``ops_cross_field_min_overlap`` and
        ``ops_cross_field_policy``.
    :param destination: the run folder.
    :returns: the report.

    Funk et al.'s fields overlap their neighbours by about 430 px, so a cell
    there is segmented in two fields or more, often whole in one and cut in
    the other. For every pair of overlapping fields
    (:func:`spacr.ops_merge.overlapping_fields`) the cell labels are
    compared pixel by pixel through the two fields' transforms
    (:func:`_ph_pair_intersections`); mutual best partners whose score
    reaches ``ops_cross_field_min_overlap`` are copies of one cell
    (:func:`_ph_pair_edges`), the connected copies are one physical cell,
    and its largest copy owns it (:func:`_ph_assign_ownership`). A field's
    transform is the re-solved one when the merge refined it.

    Writes ``ops_cross_field_<well>.parquet`` (``plate``, ``well``,
    ``site``, ``label``, ``area`` = ``cell_px``, ``touches_field_edge``,
    ``component``, ``n_copies``, ``owner_field``, ``owner_label``,
    ``is_owner``, ``owned_area_share``) and
    ``ops_cross_field_edges_<well>.parquet``. Only sites with both a label
    image and mask-pass rows take part.
    """
    import pandas as pd

    from .qt.i18n import tr
    from .tabular import write_table

    if not bool(settings.get("ops_cross_field_ownership", False)):
        return {"skipped": "ops_cross_field_ownership is off"}
    started = time.perf_counter()
    metric = str(settings.get("ops_cross_field_metric")
                 or "overlap_fraction").strip().lower()
    if metric not in _PH_CROSS_FIELD_METRICS:
        raise ValueError(tr(
            "ops_cross_field_metric must be one of {metrics}; got {value}",
            metrics=list(_PH_CROSS_FIELD_METRICS), value=metric))
    min_overlap = float(_setting_number(
        settings.get("ops_cross_field_min_overlap"),
        "ops_cross_field_min_overlap", 0.5))
    if not 0.0 < min_overlap <= 1.0:
        raise ValueError(tr(
            "ops_cross_field_min_overlap must lie in (0, 1]; got {value}",
            value=min_overlap))
    policy = _ph_cross_field_policy(settings)
    mask_dir = os.path.join(destination, well, "phenotype_masks")
    maskpass = _ph_maskpass_path(destination, well)
    if not os.path.isfile(maskpass):
        raise ValueError(tr(
            "cross-field ownership needs the mask pass of well {well} at "
            "{path}; run the ph_masks phase first", well=well, path=maskpass))
    passed = pd.read_parquet(maskpass, columns=["site", "label", "cell_px",
                                                "touches_field_edge"])
    placed = _well_rows(db, "ops_phenotype", plate, well)
    if placed is None or placed.empty:
        raise ValueError(tr(
            "well {well} has no ops_phenotype rows in {db}; run the "
            "phenotype phase first", well=well, db=db))
    by_site = {site: group for site, group in passed.groupby("site")}
    fields: Dict[int, Dict[str, Any]] = {}
    for record in placed.to_dict("records"):
        site = int(record["site"])
        if site not in by_site or not os.path.isfile(
                os.path.join(mask_dir, f"site-{site}.npz")):
            continue
        table = by_site[site]
        fields[site] = {
            "transform": _ph_field_record_transform(record),
            "shape": (int(record["field_height"]), int(record["field_width"])),
            "areas": pd.Series(table["cell_px"].to_numpy(),
                               index=table["label"].to_numpy(np.int64)),
            "cells": None}

    def load(site: int) -> np.ndarray:
        """One field's cell labels."""
        with np.load(os.path.join(mask_dir, f"site-{site}.npz")) as saved:
            return saved["cells"]

    def log(count: int, total: int) -> None:
        """A progress line."""
        _say(tr("{well} ph_ownership: {count} of {total} field pairs",
                well=well, count=count, total=total))

    cells, edges, pairs = _ph_cross_field_fields(
        fields, metric=metric, min_overlap=min_overlap, load=load, log=log)
    edge = passed.set_index(["site", "label"])["touches_field_edge"]
    cells["touches_field_edge"] = np.array(
        [False if pd.isna(v) else bool(v) for v in edge.reindex(
            pd.MultiIndex.from_arrays([cells["site"], cells["label"]]))], bool)
    cells = cells[["site", "label", "area", "touches_field_edge", "component",
                   "n_copies", "owner_field", "owner_label", "is_owner",
                   "owned_area_share"]]
    cells.insert(0, "well", well)
    cells.insert(0, "plate", plate)
    write_table(cells, _ph_cross_field_path(destination, well),
                canonicalise=False)
    write_table(edges, _ph_cross_field_edges_path(destination, well),
                canonicalise=False)
    components = cells.drop_duplicates("component")
    report = {
        "metric": metric, "min_overlap": min_overlap, "policy": policy,
        "fields": len(fields), "field_pairs": pairs, "edges": int(len(edges)),
        "cells": int(len(cells)), "physical_cells": int(len(components)),
        "cells_with_copies": int((cells["n_copies"] > 1).sum()),
        "non_owner_copies": int((~cells["is_owner"]).sum()),
        "components_with_two_copies_in_one_field": int(cells.loc[
            cells.duplicated(["component", "site"]), "component"].nunique()),
        "table": _ph_cross_field_path(destination, well),
        "seconds": round(time.perf_counter() - started, 1)}
    _say(tr("{well} ph_ownership: {cells} copies are {physical} cells",
            well=well, cells=report["cells"],
            physical=report["physical_cells"]))
    return report


def _ph_cross_field_filter(cells, destination: str, wells: Sequence[str],
                           settings: Mapping[str, Any]):
    """Apply ``ops_cross_field_policy`` to the aggregate's genotyped cells.

    :param cells: the genotyped cells (``well``, ``site``, ``label``).
    :param destination: the run folder.
    :param wells: the pooled wells.
    :param settings: the OPS settings.
    :returns: ``(cells, report)``. ``'one_per_cell'`` keeps one row per
        physical cell, its largest copy (ties to the smaller site, then
        label); ``'owner_only'`` drops every non-owner copy; ``'flag_only'``
        changes nothing. Cells absent from the ownership table keep
        themselves.
    """
    import pandas as pd

    from .qt.i18n import tr

    policy = _ph_cross_field_policy(settings)
    if policy == "flag_only":
        return cells, {"policy": policy}
    tables = []
    for well in wells:
        path = _ph_cross_field_path(destination, well)
        if not os.path.isfile(path):
            raise ValueError(tr(
                "ops_cross_field_ownership needs {path}; run the "
                "ph_ownership phase first", path=path))
        tables.append(pd.read_parquet(path, columns=[
            "well", "site", "label", "component", "area", "is_owner"]))
    owners = pd.concat(tables, ignore_index=True)
    joined = cells.reset_index(drop=True).merge(
        owners, on=["well", "site", "label"], how="left")
    if policy == "owner_only":
        keep = np.array([True if pd.isna(v) else bool(v)
                         for v in joined["is_owner"]], bool)
        return (cells.reset_index(drop=True)[keep].reset_index(drop=True),
                {"policy": policy,
                 "dropped_non_owner_copies": int((~keep).sum())})
    key = joined["well"].astype(str) + ":" + joined["component"].map(
        lambda v: "" if pd.isna(v) else str(int(v)))
    lone = joined["component"].isna()
    key[lone] = "lone:" + pd.Series(np.arange(int(lone.sum())),
                                    index=key[lone].index).astype(str)
    order = joined.assign(_key=key, _position=np.arange(len(joined))).sort_values(
        ["_key", "area", "site", "label"], ascending=[True, False, True, True],
        kind="mergesort")
    kept = np.sort(order.drop_duplicates("_key")["_position"].to_numpy())
    return (cells.reset_index(drop=True).iloc[kept].reset_index(drop=True),
            {"policy": policy,
             "dropped_extra_copies": int(len(joined) - len(kept))})



#: The levels ``ops_reference_counts_level`` may name: decoded sequencing
#: nuclei, or linked, owned phenotype cells. They are never pooled.
_PH_REPORT_LEVELS = ("sbs", "phenotype")

#: The funnel columns of ``funnel_per_well.csv``, each read from the
#: aggregate phase's own count (:class:`_PhFunnel`).
_PH_FUNNEL_COLUMNS = (
    ("segmented", "cells_segmented"), ("whole_and_owned", "owned_whole"),
    ("linked", "linked"), ("with_library_guide", "with_guide"),
    ("interphase", "interphase"), ("decoy_assigned", None),
    ("dropped_failed_fields", "registration_failure_cells_excluded"),
    ("dropped_nonfinite", "dropped_nonfinite_rows"))


def _ph_rho_with_null(spacr, reference, reps: int, rng) -> Dict[str, Any]:
    """Spearman's rho of two count vectors and a shuffled-label control.

    :param spacr: spaCR's counts.
    :param reference: the reference counts, row-aligned.
    :param reps: how many times spaCR's counts are permuted.
    :param rng: the generator the permutations draw from.
    :returns: ``n``, ``spearman``, ``shuffled_mean`` and
        ``shuffled_p95_abs`` (the 95th percentile of abs(rho) under the
        permutations), over the rows where both are finite.
    """
    from scipy.stats import spearmanr

    spacr = np.asarray(spacr, float)
    reference = np.asarray(reference, float)
    finite = np.isfinite(spacr) & np.isfinite(reference)
    spacr, reference = spacr[finite], reference[finite]
    rho = float(spearmanr(spacr, reference)[0])
    null = np.array([spearmanr(rng.permutation(spacr), reference)[0]
                     for _ in range(int(reps))], float)
    return {"n": int(finite.sum()), "spearman": rho,
            "shuffled_mean": float(null.mean()) if null.size else None,
            "shuffled_p95_abs": (float(np.percentile(np.abs(null), 95))
                                 if null.size else None),
            "shuffles": int(reps)}


def _ph_reference_counts(path: str):
    """A published per-guide count table: ``guide``, ``gene``, ``count``.

    :raises ValueError: when a column is missing.
    """
    from .qt.i18n import tr
    from .tabular import read_table

    table = read_table(str(path), canonicalise=False, report=None)
    missing = [c for c in ("guide", "gene", "count") if c not in table.columns]
    if missing:
        raise ValueError(tr(
            "ops_reference_counts {path} needs guide, gene and count "
            "columns; it lacks {missing}", path=path, missing=missing))
    table = table[["guide", "gene", "count"]].copy()
    table["count"] = table["count"].astype(float)
    return table.reset_index(drop=True)


def _ph_field_geometry(placed) -> Tuple[Dict[int, Any], Dict[int, Tuple[int, int]],
                                        Dict[int, List[int]]]:
    """Every placed field's transform, shape and overlapping neighbours.

    :param placed: one well's ``ops_phenotype`` rows.
    :returns: ``(transforms, shapes, neighbours)`` keyed by site; a field's
        neighbours are the other fields whose well-frame bounding boxes
        overlap its own.
    """
    transforms, shapes = {}, {}
    for record in placed.to_dict("records"):
        site = int(record["site"])
        transforms[site] = _ph_field_record_transform(record)
        shapes[site] = (int(record["field_height"]), int(record["field_width"]))
    boxes = {}
    for site, transform in transforms.items():
        h, w = float(shapes[site][0]), float(shapes[site][1])
        corners = _ph_to_well(np.array([[0, 0], [0, w], [h, 0], [h, w]], float),
                              transform)
        boxes[site] = (corners.min(axis=0), corners.max(axis=0))
    sites = sorted(transforms)
    neighbours = {s: [t for t in sites if t != s and bool(np.all(
        np.minimum(boxes[s][1], boxes[t][1]) > np.maximum(boxes[s][0], boxes[t][0])))]
        for s in sites}
    return transforms, shapes, neighbours


def _ph_zone(site: int, points, geometry) -> Tuple[np.ndarray, np.ndarray]:
    """How many other fields cover each point, and its distance to the field edge.

    :param site: the field the points lie in.
    :param points: ``(N, 2)`` field ``(row, col)`` centroids.
    :param geometry: :func:`_ph_field_geometry`.
    :returns: ``(covered, edge_distance)``: the number of other acquired
        fields each point maps inside (``[0, H) x [0, W)``), and
        ``min(row, H - 1 - row, col, W - 1 - col)``.
    """
    transforms, shapes, neighbours = geometry
    points = np.asarray(points, float)
    well = _ph_to_well(points, transforms[site])
    covered = np.zeros(len(points), int)
    for other in neighbours.get(site, []):
        mapped = _ph_to_field(well, transforms[other])
        h, w = shapes[other]
        covered += ((mapped[:, 0] >= 0) & (mapped[:, 0] < h)
                    & (mapped[:, 1] >= 0) & (mapped[:, 1] < w)).astype(int)
    h, w = shapes[site]
    edge = np.minimum.reduce([points[:, 0], h - 1 - points[:, 0],
                              points[:, 1], w - 1 - points[:, 1]])
    return covered, edge


def _ph_rate_rows(frame, keys: Sequence[str], column: str, min_cells: int = 0
                  ) -> List[Dict[str, Any]]:
    """Cells, matched and match % of every group of ``frame``.

    :param frame: one row per unit with a boolean ``column``.
    :param keys: the grouping columns.
    :param column: the matched flag.
    :param min_cells: groups with fewer units are left out.
    :returns: one dict per group, keys first.
    """
    rows = []
    for key, group in frame.groupby(list(keys), observed=True):
        if len(group) < min_cells:
            continue
        key = key if isinstance(key, tuple) else (key,)
        values = group[column].astype(bool)
        rows.append({**dict(zip(keys, key)), "cells": int(len(group)),
                     "matched": int(values.sum()),
                     "match_pct": 100 * float(values.mean())})
    return rows


def _ph_external_assign(external, copies, max_px: float):
    """Assign each external cell to the nearest spaCR copy of its field.

    :param external: ``field, label, y, x, matched`` rows of one well.
    :param copies: the spaCR copies with ``site``, ``label`` and the cell and
        nucleus centroids.
    :param max_px: the largest distance an assignment may span.
    :returns: the external rows with ``site`` and the assigned spaCR
        ``spacr_label`` (-1 when none lies within ``max_px``). Per field
        the nucleus and the cell centroids are both tried and the one with
        the smaller median nearest distance is kept.
    """
    import pandas as pd
    from scipy.spatial import cKDTree

    out = []
    by_site = {int(s): g for s, g in copies.groupby("site")}
    for field, rows in external.groupby("field"):
        table = by_site.get(int(field))
        if table is None:
            continue
        best = None
        for kind in ("nucleus", "cell"):
            if f"{kind}_cy" not in table:
                continue
            xy = table[[f"{kind}_cy", f"{kind}_cx"]].to_numpy(float)
            ok = np.isfinite(xy).all(1)
            if not ok.any():
                continue
            distance, index = cKDTree(xy[ok]).query(
                rows[["y", "x"]].to_numpy(float), distance_upper_bound=max_px)
            found = np.isfinite(distance)
            labels = np.where(found, table["label"].to_numpy()[ok][
                np.minimum(index, ok.sum() - 1)], -1)
            median = (float(np.median(distance[found])) if found.any()
                      else float("nan"))
            if best is None or median < best[0]:
                best = (median, labels)
        if best is not None:
            out.append(rows.assign(site=int(field), spacr_label=best[1]))
    if not out:
        return external.iloc[:0].assign(site=pd.Series(dtype=int),
                                        spacr_label=pd.Series(dtype=int))
    return pd.concat(out, ignore_index=True)


#: How the external comparison of ``ops_report_external_cells`` is matched,
#: written beside its numbers.
_PH_EXTERNAL_MATCHING = (
    "Paired on the same physical cells: each external cell is assigned to the "
    "nearest spaCR copy in the same field within ops_report_external_max_px "
    "and through it to that copy's cross-field component; only components "
    "with at least one external copy are counted, for both pipelines. Both "
    "use the same field transforms, the same overlap-strip rule (the owner "
    "copy's zone) and the same matched rule (any copy matched). The external "
    "match flags are taken as given; their threshold and deduplication are "
    "the external pipeline's own.")


def _ph_overlap_report(db: str, plate: str, wells: Sequence[str],
                       settings: Mapping[str, Any], destination: str,
                       folder: str) -> Dict[str, Any]:
    """Phenotype-genotype match rates in tile-overlap strips and cores (item 915).

    :param db: the measurements database.
    :param plate: the plate.
    :param wells: the wells.
    :param settings: read for ``ops_report_edge_bin_px``,
        ``ops_report_external_cells`` and ``ops_report_external_max_px``.
    :param destination: the run folder, holding ``ops_maskpass_<well>``
        and, when the ``ph_ownership`` phase ran, ``ops_cross_field_<well>``.
    :param folder: where the tables go.
    :returns: the paths written and the counts per well.

    A copy (a segmented cell in one field) lies in an OVERLAP STRIP when its
    centroid maps inside at least one other acquired field, else in the
    CORE (:func:`_ph_zone`). A copy is matched when ``ops_merge`` links it.
    A physical cell (a cross-field component) is matched when any copy is,
    and its zone is its owner copy's. ``overlap_match_by_zone.csv`` has
    both units; ``overlap_match_by_edge_distance.csv`` bins copies by
    their distance to the field edge; ``overlap_match_per_field.csv`` keeps
    fields with at least 50 units per zone.
    """
    import pandas as pd
    from scipy.stats import binomtest

    from .qt.i18n import tr
    from .tabular import read_table, write_table

    bin_px = float(_setting_number(settings.get("ops_report_edge_bin_px"),
                                   "ops_report_edge_bin_px", 100))
    max_px = float(_setting_number(settings.get("ops_report_external_max_px"),
                                   "ops_report_external_max_px", 8.0))
    if bin_px <= 0 or max_px <= 0:
        raise ValueError(tr(
            "ops_report_edge_bin_px and ops_report_external_max_px must be "
            "positive; got {bin} and {px}", bin=bin_px, px=max_px))
    external_path = settings.get("ops_report_external_cells")
    external_all = (read_table(str(external_path), canonicalise=False, report=None)
                    if external_path else None)
    zones, edges, fields, paired, external_info = [], [], [], [], {}
    for well in wells:
        maskpass = _ph_maskpass_path(destination, well)
        placed = _well_rows(db, "ops_phenotype", plate, well)
        if not os.path.isfile(maskpass) or placed is None or placed.empty:
            raise ValueError(tr(
                "the overlap report of well {well} needs its mask pass and "
                "ops_phenotype rows; run ph_masks first", well=well))
        geometry = _ph_field_geometry(placed)
        copies = pd.read_parquet(maskpass)
        copies = copies[copies["site"].astype(int).isin(geometry[0])].copy()
        parts = []
        for site, group in copies.groupby("site", sort=True):
            covered, edge = _ph_zone(int(site), group[["cell_cy", "cell_cx"]], geometry)
            parts.append(group.assign(covered_by_others=covered, edge_dist=edge))
        copies = pd.concat(parts, ignore_index=True) if parts else copies.assign(
            covered_by_others=0, edge_dist=0.0)
        links = _well_rows(db, "ops_merge", plate, well)
        linked = set() if links is None else set(zip(
            links["site"].astype(int), links["label"].astype(int)))
        copies["matched"] = [(int(a), int(b)) in linked for a, b in
                             zip(copies["site"], copies["label"])]
        copies["zone"] = np.where(copies["covered_by_others"] > 0,
                                  "overlap strip", "core")
        copies["well"], copies["unit"] = well, "copy"
        zones += _ph_rate_rows(copies, ["well", "unit", "zone"], "matched")
        copies["edge_dist_bin_px"] = (np.floor(copies["edge_dist"] / bin_px)
                                      * bin_px).astype(int)
        for row in _ph_rate_rows(copies, ["well", "unit", "edge_dist_bin_px"],
                                 "matched"):
            band = copies["edge_dist_bin_px"] == row["edge_dist_bin_px"]
            row["strip_share"] = float((copies.loc[band, "zone"]
                                        == "overlap strip").mean())
            edges.append(row)
        fields += _ph_rate_rows(copies.rename(columns={"site": "field"}),
                                ["well", "unit", "field", "zone"], "matched", 50)
        cross = _ph_cross_field_path(destination, well)
        physical = None
        if os.path.isfile(cross):
            owners = pd.read_parquet(cross, columns=["site", "label", "component",
                                                     "is_owner"])
            joined = copies.merge(owners, on=["site", "label"], how="left").dropna(
                subset=["component"])
            owner = joined[joined["is_owner"].astype(bool)].drop_duplicates(
                "component").set_index("component")
            physical = joined.groupby("component").agg(matched=("matched", "max"))
            physical["zone"] = owner["zone"].reindex(physical.index).fillna("core")
            physical["field"] = owner["site"].reindex(physical.index)
            physical["well"], physical["unit"] = well, "physical_cell"
            zones += _ph_rate_rows(physical, ["well", "unit", "zone"], "matched")
            fields += _ph_rate_rows(physical.dropna(subset=["field"]).astype(
                {"field": int}), ["well", "unit", "field", "zone"], "matched", 50)
        if external_all is None:
            continue
        external = external_all
        if "well" in external_all:
            external = external_all[external_all["well"].astype(str) == str(well)]
        external = external.copy()
        external["matched"] = external["matched"].astype(bool)
        inside = external[external["field"].astype(int).isin(geometry[0])]
        parts = []
        for site, group in inside.groupby("field", sort=True):
            covered, _edge = _ph_zone(int(site), group[["y", "x"]], geometry)
            parts.append(group.assign(zone=np.where(covered > 0, "overlap strip",
                                                    "core")))
        if parts:
            per_copy = pd.concat(parts, ignore_index=True).assign(
                well=well, unit="external_copy")
            zones += _ph_rate_rows(per_copy, ["well", "unit", "zone"], "matched")
        if physical is None:
            raise ValueError(tr(
                "ops_report_external_cells compares physical cells and needs "
                "the ph_ownership phase's table {path}", path=cross))
        assigned = _ph_external_assign(external, copies, max_px)
        assigned = assigned[assigned["spacr_label"] >= 0].merge(
            owners.rename(columns={"label": "spacr_label"}),
            on=["site", "spacr_label"], how="inner")
        theirs = assigned.groupby("component")["matched"].max()
        both_sides = pd.DataFrame({"external": theirs}).join(
            physical[["matched", "zone"]].rename(columns={"matched": "spacr"}),
            how="left")
        both_sides["zone"] = both_sides["zone"].fillna("core")
        for zone in ("core", "overlap strip"):
            group = both_sides[both_sides["zone"] == zone]
            spacr = group["spacr"].fillna(False).astype(bool)
            theirs_ = group["external"].astype(bool)
            only_spacr = int((spacr & ~theirs_).sum())
            only_theirs = int((~spacr & theirs_).sum())
            paired.append({
                "well": well, "zone": zone, "physical_cells": int(len(group)),
                "spacr_match_pct": 100 * float(spacr.mean()) if len(group) else np.nan,
                "external_match_pct": (100 * float(theirs_.mean())
                                       if len(group) else np.nan),
                "both": int((spacr & theirs_).sum()), "spacr_only": only_spacr,
                "external_only": only_theirs,
                "mcnemar_exact_p": (float(binomtest(
                    only_spacr, only_spacr + only_theirs, 0.5).pvalue)
                    if only_spacr + only_theirs else 1.0)})
        external_info[well] = {"external_cells": int(len(external)),
                               "assigned": int(len(assigned)),
                               "max_px": max_px}
    outputs = {}
    rates = ["cells", "matched", "match_pct"]
    for name, rows, columns in (
            ("overlap_match_by_zone", zones, ["well", "unit", "zone", *rates]),
            ("overlap_match_by_edge_distance", edges,
             ["well", "unit", "edge_dist_bin_px", *rates, "strip_share"]),
            ("overlap_match_per_field", fields,
             ["well", "unit", "field", "zone", *rates])):
        outputs[name] = write_table(pd.DataFrame(rows, columns=columns), os.path.join(
            folder, f"{name}.csv"), canonicalise=False)
    if external_all is not None:
        outputs["overlap_external_paired"] = write_table(
            pd.DataFrame(paired), os.path.join(folder, "overlap_external_paired.csv"),
            canonicalise=False)
        with open(os.path.join(folder, "overlap_external_paired.json"), "w",
                  encoding="utf-8") as handle:
            json.dump({"matching": _PH_EXTERNAL_MATCHING, "wells": external_info,
                       "source": str(external_path)}, handle, indent=1)
        outputs["overlap_external_info"] = os.path.join(
            folder, "overlap_external_paired.json")
    return {"outputs": outputs, "zones": zones}


def _ph_with_read_stage(out, read_stage: Mapping[str, Tuple[int, int]]):
    """The funnel with ``sbs_nuclei``, ``sbs_nuclei_with_read`` and its % added.

    :param out: one row per well with a ``well`` column.
    :param read_stage: :func:`_ph_read_stage`.
    :returns: a copy; wells without field stats get NaN.
    """
    out = out.copy()
    wells = out["well"].astype(str)
    nuclei = wells.map(lambda w: read_stage[w][0] if w in read_stage else np.nan)
    reads = wells.map(lambda w: read_stage[w][1] if w in read_stage else np.nan)
    out["sbs_nuclei"] = nuclei.to_numpy()
    out["sbs_nuclei_with_read"] = reads.to_numpy()
    out["sbs_nuclei_with_read_pct"] = (100 * reads / nuclei).to_numpy()
    return out


def _ph_read_stage(db: str, wells: Sequence[str]) -> Dict[str, Tuple[int, int]]:
    """Each well's decoded nuclei and nuclei with at least one read.

    :param db: the measurements database; the decode phase's
        ``results/ops_decode/field_stats_<well>.csv`` sit beside it.
    :param wells: the wells.
    :returns: ``{well: (nuclei, nuclei_with_read)}`` for the wells whose
        field stats carry ``nuclei_with_read`` (a decode run with
        ``ops_report_read_stage`` on).
    """
    from .tabular import read_table

    folder = os.path.join(os.path.dirname(os.path.abspath(db)), "results",
                          "ops_decode")
    out = {}
    for well in wells:
        path = os.path.join(folder, f"field_stats_{well}.csv")
        if not os.path.isfile(path):
            continue
        stats = read_table(path, canonicalise=False, report=None)
        if "nuclei_with_read" in stats:
            out[str(well)] = (int(stats["nuclei"].sum()),
                              int(stats["nuclei_with_read"].sum()))
    return out


def _ph_report(db: str, plate: str, wells: Sequence[str],
               settings: Mapping[str, Any], destination: str,
               aggregate: Optional[Mapping[str, Any]] = None
               ) -> Dict[str, Any]:
    """The run's yield funnel and its counts against a published table; the ``report`` phase.

    :param db: the measurements database.
    :param plate: the plate name.
    :param wells: the wells run.
    :param settings: read for ``ops_reference_counts``,
        ``ops_reference_counts_level``, ``ops_reference_shuffles``,
        ``ops_reference_seed``, ``ops_design``, ``ops_genotype_table``
        and the alpha ``ops_report_read_stage``, ``ops_report_overlap_zones``,
        ``ops_report_edge_bin_px``, ``ops_report_external_cells`` and
        ``ops_report_external_max_px`` (item 915).
    :param destination: the run folder; the files go to
        ``<dst_root>/results/ops_report/``.
    :param aggregate: this run's aggregate report, whose ``outputs`` name
        the funnel and guide tables; the untagged tables in
        ``destination`` when None.
    :returns: the report.

    ``funnel_per_well.csv``: per well, ``segmented``, ``whole_and_owned``,
    ``linked``, ``with_library_guide``, ``interphase``, ``decoy_assigned``
    (linked cells whose call is no guide of the design), ``dropped_failed_fields``
    and ``dropped_nonfinite``, then every count the aggregate phase took
    (:class:`_PhFunnel`), so nothing is recounted another way.

    ``counts_vs_reference.csv`` and ``.json``: spaCR's count of each guide
    of ``ops_reference_counts`` and of each gene, beside the reference's,
    and Spearman's rho with a shuffled-label control
    (:func:`_ph_rho_with_null`; ``ops_reference_shuffles`` permutations
    from ``ops_reference_seed``, guides first, then genes). Each level is
    its own block, named in every row: ``sbs`` counts decoded sequencing
    nuclei per guide (every reference guide, absent ones 0); ``phenotype``
    counts the pooled cells of the guide and gene tables (the guides and
    genes both have). ``spacr_scaled`` is spaCR's count scaled to the
    reference's pooled total.

    With ``ops_report_read_stage`` the funnel gains ``sbs_nuclei``,
    ``sbs_nuclei_with_read`` and ``sbs_nuclei_with_read_pct`` from the
    decode phase's field stats (:func:`_ph_read_stage`); with
    ``ops_report_overlap_zones`` the match rates in overlap strips and
    cores are written (:func:`_ph_overlap_report`).
    """
    import pandas as pd

    from .qt.i18n import tr
    from .tabular import write_table

    started = time.perf_counter()
    folder = os.path.join(destination, "results", "ops_report")
    os.makedirs(folder, exist_ok=True)
    outputs = dict((aggregate or {}).get("outputs") or {})
    funnel_path = outputs.get("funnel_parquet") or os.path.join(
        destination, f"ops_funnel_{plate}.parquet")
    report: Dict[str, Any] = {"folder": folder}
    read_stage = (_ph_read_stage(db, wells)
                  if bool(settings.get("ops_report_read_stage", False)) else None)
    if os.path.isfile(funnel_path):
        funnel = pd.read_parquet(funnel_path)
        out = funnel[["plate", "well"]].copy()
        for name, source in _PH_FUNNEL_COLUMNS:
            if source is None:
                out[name] = (funnel.get("linked", 0)
                             - funnel.get("linked_no_genotype", 0)
                             - funnel.get("with_guide", 0))
            elif source in funnel:
                out[name] = funnel[source]
            elif name == "whole_and_owned" and "owned" in funnel:
                out[name] = funnel["owned"]
            else:
                out[name] = 0
        if read_stage is not None:
            out = _ph_with_read_stage(out, read_stage)
        rest = [c for c in funnel.columns if c not in ("plate", "well")]
        out = pd.concat([out, funnel[rest].add_prefix("aggregate_")], axis=1)
        target = os.path.join(folder, "funnel_per_well.csv")
        write_table(out, target, canonicalise=False)
        report["funnel"] = target
    elif read_stage:
        out = _ph_with_read_stage(pd.DataFrame({"plate": plate, "well": sorted(read_stage)}),
                                  read_stage)
        target = os.path.join(folder, "funnel_per_well.csv")
        write_table(out, target, canonicalise=False)
        report["funnel"] = target
    else:
        report["funnel"] = None
    if read_stage is not None:
        report["read_stage"] = {w: {"nuclei": v[0], "with_read": v[1]}
                                for w, v in read_stage.items()}
    if bool(settings.get("ops_report_overlap_zones", False)):
        report["overlap"] = _ph_overlap_report(db, plate, wells, settings,
                                               destination, folder)["outputs"]
    reference_path = settings.get("ops_reference_counts")
    if not reference_path:
        report["counts_vs_reference"] = {
            "skipped": "no ops_reference_counts table"}
        report["seconds"] = round(time.perf_counter() - started, 1)
        return report
    levels = [v.lower() for v in _ph_names(
        settings.get("ops_reference_counts_level") or "sbs")]
    unknown = [v for v in levels if v not in _PH_REPORT_LEVELS]
    if unknown:
        raise ValueError(tr(
            "ops_reference_counts_level must name {levels}; got {value}",
            levels=list(_PH_REPORT_LEVELS), value=unknown))
    reps = int(_setting_number(settings.get("ops_reference_shuffles"),
                               "ops_reference_shuffles", 1000))
    seed = int(_setting_number(settings.get("ops_reference_seed"),
                               "ops_reference_seed", 0))
    reference = _ph_reference_counts(str(reference_path))
    frames, summary = [], {"reference": str(reference_path), "seed": seed,
                           "shuffles": reps, "levels": {}}
    for level in levels:
        rng = np.random.default_rng(seed)
        if level == "sbs":
            if settings.get("ops_genotype_table"):
                called = _ph_genotype_table(str(settings["ops_genotype_table"]),
                                            plate, wells)
            else:
                called = _rows_for_wells(db, "ops_barcodes", plate, wells)
                if called is None:
                    raise ValueError(tr(
                        "the sbs level needs ops_barcodes; run the decode "
                        "phase first"))
                called = called[called["mapped_guide"].fillna("").astype(str)
                                != ""]
            counts = called.groupby("mapped_guide").size()
            length = int(pd.Series(counts.index).str.len().mode().iloc[0]) \
                if len(counts) else 11
            guides = reference.assign(
                spacr=reference["guide"].astype(str).str.upper().str[:length]
                .map(counts).fillna(0).astype(float))
            genes = guides.groupby("gene")[["count", "spacr"]].sum().reset_index()
        else:
            guide_path = outputs.get("guides_parquet") or os.path.join(
                destination, f"ops_guides_{plate}.parquet")
            gene_path = outputs.get("genes_parquet") or os.path.join(
                destination, f"ops_genes_{plate}.parquet")
            if not (os.path.isfile(guide_path) and os.path.isfile(gene_path)):
                raise ValueError(tr(
                    "the phenotype level needs the aggregate phase's guide "
                    "and gene tables at {path}; run aggregate first",
                    path=guide_path))
            mine = pd.read_parquet(guide_path, columns=["guide", "n_cells"])
            guides = mine.merge(reference, on="guide", how="inner").rename(
                columns={"n_cells": "spacr"})
            guides = guides[["guide", "gene", "count", "spacr"]]
            pooled = reference.groupby("gene")["count"].sum()
            genes = pd.read_parquet(gene_path, columns=["gene", "n_cells"]).merge(
                pooled.rename("count"), left_on="gene", right_index=True,
                how="inner").rename(columns={"n_cells": "spacr"})
        block = {}
        for unit, table in (("guide", guides), ("gene", genes)):
            block[unit] = _ph_rho_with_null(table["spacr"].to_numpy(float),
                                            table["count"].to_numpy(float),
                                            reps, rng)
            total = float(table["spacr"].sum())
            frame = pd.DataFrame({
                "level": level, "unit": unit,
                "key": table[unit].astype(str).to_numpy(),
                "gene": table["gene"].astype(str).to_numpy(),
                "spacr_count": table["spacr"].to_numpy(float),
                "spacr_scaled": (table["spacr"].to_numpy(float)
                                 * float(table["count"].sum()) / total
                                 if total else np.nan),
                "reference_count": table["count"].to_numpy(float)})
            frames.append(frame)
        summary["levels"][level] = block
    table = pd.concat(frames, ignore_index=True)
    target = os.path.join(folder, "counts_vs_reference.csv")
    write_table(table, target, canonicalise=False)
    with open(os.path.join(folder, "counts_vs_reference.json"), "w",
              encoding="utf-8") as handle:
        json.dump(summary, handle, indent=1)
    report["counts_vs_reference"] = {
        "csv": target, "json": os.path.join(folder, "counts_vs_reference.json"),
        **summary}
    report["seconds"] = round(time.perf_counter() - started, 1)
    return report


def _ph_profile_tables(name: str, table, destination: str, label: str,
                       outputs: Dict[str, str]) -> None:
    """Write one profile-option table as CSV and Parquet and record both.

    :param name: the table's role, such as ``genes_phase``.
    :param table: the table.
    :param destination: the output folder.
    :param label: the plate and tag of the run.
    :param outputs: the report's outputs, updated in place.
    """
    from .tabular import write_table

    for suffix in ("csv", "parquet"):
        target = os.path.join(destination, f"ops_{name}_{label}.{suffix}")
        write_table(table, target, canonicalise=False)
        outputs[f"{name}_{suffix}"] = target


def _ph_active_genes(settings: Mapping[str, Any]) -> List[str]:
    """The genes with a phenotype of their own, from ``ops_neighbour_active``.

    :param settings: read for ``ops_neighbour_active``: one table or a
        list of tables with a ``gene`` column; a gene is active when any
        ``*_fdr`` column is below 0.05 or ``below_corrected_p`` is true.
    :returns: the active genes, sorted.
    """
    import pandas as pd

    from .tabular import read_table

    value = settings.get("ops_neighbour_active")
    paths = [value] if isinstance(value, str) else list(value or [])
    active = set()
    for path in paths:
        table = read_table(str(path), canonicalise=False, report=None)
        hit = np.zeros(len(table), bool)
        for column in table.columns:
            if str(column).endswith("_fdr"):
                hit |= (pd.to_numeric(table[column], errors="coerce") < 0.05).to_numpy()
            elif column == "below_corrected_p":
                hit |= table[column].fillna(False).astype(bool).to_numpy()
        active |= set(table["gene"][hit].astype(str))
    return sorted(active)


def _ph_neighbour_effects(settings: Mapping[str, Any], cells, embedding, pcs: Sequence[str],
                          destination: str, control_prefix: str = "nontargeting"
                          ) -> Dict[str, Any]:
    """The neighbour table and neighbour effects of the aggregate phase (item 891).

    :param settings: read for ``ops_neighbour_radius_px``,
        ``ops_neighbour_grid_px``, ``ops_neighbour_receivers``,
        ``ops_neighbour_null_reps`` and ``ops_neighbour_active``.
    :param cells: the profiled cells, with ``well``, ``site``, ``label``,
        ``sgRNA`` and ``gene``, in the row order of ``embedding``.
    :param embedding: their TVN components.
    :param pcs: the component names.
    :param destination: the run folder: each well's
        ``<well>/phenotype_masks/site-<n>.npz`` cell masks and
        ``ops_maskpass_<well>.parquet`` nucleus centroids are read from it,
        and the tables go to its ``results/ops_neighbours``.
    :param control_prefix: the control gene prefix.
    :returns: the report, with ``outputs``.
    :raises ValueError: for a bad setting, a missing mask or centroid, or
        ``'inactive'`` receivers without ``ops_neighbour_active``.

    Every receiver's neighbours are the segmented cells whose mask reaches
    within the radius of its nucleus centroid
    (:func:`spacr.gene_measurement_compare.mask_neighbours`); receivers
    nearer the field edge than the radius are left out, and the field
    centre is half the mask height
    (:func:`spacr.gene_measurement_compare.neighbour_effects`). The tables
    are named as the reference scripts name them: ``nbr_<well>`` and
    ``recv_<well>``, and ``nbr_tests_<design>``, ``nbr_null_<design>``,
    ``nbr_d_<design>`` and ``nbr_V_<design>`` with design ``B`` for
    inactive and ``A`` for control receivers.
    """
    import pandas as pd

    from .gene_measurement_compare import (NEIGHBOUR_RECEIVERS, mask_neighbours,
                                           neighbour_effects)
    from .qt.i18n import tr
    from .tabular import read_table, write_table

    radius = int(_setting_number(settings.get("ops_neighbour_radius_px"),
                                 "ops_neighbour_radius_px", 50))
    grid = int(_setting_number(settings.get("ops_neighbour_grid_px"),
                               "ops_neighbour_grid_px", 2))
    reps = int(_setting_number(settings.get("ops_neighbour_null_reps"),
                               "ops_neighbour_null_reps", 5))
    receivers = str(settings.get("ops_neighbour_receivers") or "inactive").strip().lower()
    if receivers not in NEIGHBOUR_RECEIVERS or radius < 1 or grid < 1:
        raise ValueError(tr(
            "ops_neighbour_receivers must be inactive or nt and the radius and "
            "grid at least 1 pixel; got {receivers}, {radius}, {grid}",
            receivers=receivers, radius=radius, grid=grid))
    active = _ph_active_genes(settings)
    if receivers == "inactive" and not settings.get("ops_neighbour_active"):
        raise ValueError(tr(
            "ops_neighbour_receivers inactive needs ops_neighbour_active, the "
            "tables that say which genes have a phenotype of their own"))
    folder = os.path.join(destination, "results", "ops_neighbours")
    os.makedirs(folder, exist_ok=True)
    outputs: Dict[str, str] = {}
    frame = pd.concat([cells[["well", "site", "label", "sgRNA", "gene"]].reset_index(drop=True),
                       pd.DataFrame(np.asarray(embedding), columns=list(pcs))], axis=1)
    frame["well"] = frame["well"].astype(str)
    wells: Dict[str, Any] = {}
    centre = None
    for well, part in frame.groupby("well", sort=True):
        path = _ph_maskpass_path(destination, well)
        if not os.path.isfile(path):
            raise ValueError(tr(
                "ops_neighbour_effects needs the mask pass of well {well} at "
                "{path}; run the ph_masks phase first", well=well, path=path))
        centroids = read_table(path, canonicalise=False, report=None,
                               columns=["site", "label", "nucleus_cy", "nucleus_cx"])
        located = part[["site", "label"]].merge(centroids, on=["site", "label"], how="left")
        if located["nucleus_cy"].isna().any():
            raise ValueError(tr(
                "{count} cells of well {well} have no nucleus centroid in {path}",
                count=int(located["nucleus_cy"].isna().sum()), well=well, path=path))
        pairs, table = [], []
        for site, here in located.groupby("site", sort=True):
            mask_file = os.path.join(destination, well, "phenotype_masks", f"site-{int(site)}.npz")
            if not os.path.isfile(mask_file):
                raise ValueError(tr(
                    "ops_neighbour_effects needs the cell masks at {path}; run "
                    "the ph_segment phase first", path=mask_file))
            with np.load(mask_file) as stored:
                mask = stored["cells"]
            centre = float(mask.shape[0]) / 2 if centre is None else centre
            p, r = mask_neighbours(mask, here["label"].to_numpy(),
                                   here[["nucleus_cy", "nucleus_cx"]].to_numpy(float),
                                   radius=radius, grid=grid)
            p.insert(0, "site", np.int32(site))
            r.insert(0, "site", np.int32(site))
            pairs.append(p)
            table.append(r)
        pairs = pd.concat(pairs, ignore_index=True)
        table = pd.concat(table, ignore_index=True)
        outputs[f"nbr_{well}"] = write_table(pairs, os.path.join(folder, f"nbr_{well}.parquet"),
                                             canonicalise=False)
        outputs[f"recv_{well}"] = write_table(table, os.path.join(folder, f"recv_{well}.parquet"),
                                              canonicalise=False)
        wells[well] = (part.drop(columns="well").reset_index(drop=True), table, pairs)
    result = neighbour_effects(wells, list(pcs), active=active, receivers=receivers,
                               null_reps=reps, edge_px=float(radius), field_centre=centre)
    design = "B" if receivers == "inactive" else "A"
    for name in ("tests", "null"):
        table = result["tests" if name == "tests" else "nulls"]
        outputs[f"nbr_{name}_{design}"] = write_table(
            table, os.path.join(folder, f"nbr_{name}_{design}.parquet"), canonicalise=False)
    for name in ("d", "V"):
        target = os.path.join(folder, f"nbr_{name}_{design}.npy")
        np.save(target, np.asarray(result[name], np.float32))
        outputs[f"nbr_{name}_{design}"] = target
    target = os.path.join(folder, "nbr_names.npy")
    np.save(target, result["names"])
    outputs["nbr_names"] = target
    tests = result["tests"]
    genes = tests[tests["kind"] == "gene"]
    controls = tests[tests["kind"].str.startswith("control")]
    summary = {"design": design, "receivers": receivers, "radius_px": radius, "grid_px": grid,
               "null_reps": reps, "active_genes": len(active), "field_centre_px": centre,
               "genes_tested": int(genes["T"].notna().sum()),
               "median_exposed": float(genes["n_exposed"].median()) if len(genes) else None,
               "q_clone<0.1": int((genes["q_clone"] < 0.1).sum()),
               "q_shuffle<0.1": int((genes["q_shuffle"] < 0.1).sum()),
               "NTpseudo p_clone<0.05": float((controls["p_clone"] < 0.05).mean())
               if len(controls) else None,
               "NTpseudo p_shuffle<0.05": float((controls["p_shuffle"] < 0.05).mean())
               if len(controls) else None}
    target = os.path.join(folder, f"nbr_summary_{design}.json")
    with open(target, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=1)
    outputs[f"nbr_summary_{design}"] = target
    return {**summary, "outputs": outputs}


def _ph_profile_options(settings: Mapping[str, Any], cells, embedding,
                        guides, genes, destination: str, label: str,
                        control_prefix: str = "nontargeting"
                        ) -> Dict[str, Any]:
    """Apply the gene-profile options to the TVN profiles of a run.

    :param settings: read for the ``ops_profile_*`` options.
    :param cells: the profiled cells in the row order of ``embedding``,
        with ``well``, ``sgRNA`` and ``gene``, and ``site``, ``mitotic``
        and ``_dna_integrated`` when an option needs them.
    :param embedding: the per-cell TVN components.
    :param guides: the TVN guide table.
    :param genes: the TVN gene table.
    :param destination: where the tables are written.
    :param label: the plate and tag, as in ``ops_genes_<label>_tvn``.
    :param control_prefix: the control gene prefix.
    :returns: the report: what each option did, and the tables written.
    """
    import pandas as pd

    from .gene_measurement_compare import (
        CELL_CYCLE_PHASES, aggregate_genes, aggregate_guides,
        cell_cycle_phases, contrastive_weighting, embedding_profiles,
        fusion_profiles, generic_covariates, guide_efficacy_profiles,
        guide_level_similarity, nt_row_names, penetrance_tests,
        phase_columns, phase_profiles, phase_tests,
        remove_generic_axes, responder_profiles, shrink_gene_profiles)
    from .qt.i18n import tr

    report: Dict[str, Any] = {"outputs": {}}
    outputs = report["outputs"]
    columns = [c for c in genes.columns if str(c).startswith("PC_")]
    control = cells["gene"].astype(str).str.lower().str.startswith(
        control_prefix.lower()).to_numpy()
    profile_guides, profile_genes = guides, genes
    pcs = list(columns)
    phases = None
    cycle = None
    axes_asked: List[str] = []
    weighting = None
    similarity = None

    cycle = str(settings.get("ops_profile_cell_cycle") or "").strip() or None
    source = str(settings.get("ops_mitotic_source") or "rule").strip().lower()
    if source not in ("rule", "classifier"):
        raise ValueError(tr("ops_mitotic_source must be rule or classifier; "
                            "got {value}", value=source))
    if source == "classifier" and not settings.get("ops_mitotic_model"):
        raise ValueError(tr(
            "ops_mitotic_source classifier needs ops_mitotic_model, a "
            "trained mitotic classifier"))
    axes_asked = _ph_names(settings.get("ops_profile_generic_axes"))
    needs_phases = bool(cycle) or bool(settings.get("ops_phase_tests")) or any(
        str(a).startswith("shift_") for a in axes_asked)
    if needs_phases:
        if "_dna_integrated" not in cells:
            raise ValueError(tr(
                "cell-cycle phases need the nuclear integrated DNA intensity, "
                "which the cell tables do not have"))
        split = float(_setting_number(
            settings.get("ops_profile_g1_sg2_split_log2"),
            "ops_profile_g1_sg2_split_log2", 0.30))
        phases = cell_cycle_phases(cells, dna="_dna_integrated",
                                   control=control, split_log2=split)["phase"]
        report["cell_cycle"] = {
            "split_log2": split, "mitotic_source": source,
            "fractions": {p: round(float((phases == p).mean()), 4)
                          for p in CELL_CYCLE_PHASES}}
    if cycle:
        frame = pd.concat([cells[["well", "sgRNA", "gene"]].reset_index(drop=True),
                           pd.DataFrame(embedding, columns=pcs)], axis=1)
        frame["phase"] = phases.to_numpy()
        phase_guides, phase_genes = phase_profiles(
            frame, pcs, control=control, guide="sgRNA")
        _ph_profile_tables("guides_phase", phase_guides, destination, label,
                           outputs)
        _ph_profile_tables("genes_phase", phase_genes, destination, label,
                           outputs)
        columns = phase_columns(phase_genes, cycle)
        chosen = CELL_CYCLE_PHASES if cycle == "concat" else (cycle,)
        profile_guides = phase_guides.assign(n_cells=phase_guides[[
            f"{p}_n_cells" for p in chosen]].fillna(0).sum(axis=1).astype(int))
        profile_genes = genes[["gene", "n_guides", "n_cells"]].merge(
            phase_genes[["gene", *columns]], on="gene", how="inner")
        report["cell_cycle"]["profile"] = cycle

    shrinkage = str(settings.get("ops_profile_shrinkage") or "").strip().lower() or None
    if shrinkage not in (None, "eb_nt"):
        raise ValueError(tr("ops_profile_shrinkage must be empty or eb_nt; "
                            "got {value}", value=shrinkage))
    if shrinkage:
        prior = float(_setting_number(settings.get("ops_profile_shrinkage_prior_df"),
                                      "ops_profile_shrinkage_prior_df", 4.0))
        if cycle:
            parts = []
            for p in (CELL_CYCLE_PHASES if cycle == "concat" else (cycle,)):
                chosen_columns = [f"{p}_{c}" for c in pcs]
                sub = profile_guides[profile_guides[f"{p}_n_cells"].fillna(0) >= 3]
                shrunk, _factor, _params = shrink_gene_profiles(
                    sub, chosen_columns, control_prefix=control_prefix,
                    n_cells=sub[f"{p}_n_cells"], prior_df=prior)
                parts.append(shrunk.set_index("gene"))
            shrunk = pd.concat(parts, axis=1).reset_index().dropna()
            report["shrinkage"] = {"prior_df": prior, "per_phase": True}
        else:
            shrunk, factor, params = shrink_gene_profiles(
                profile_guides, columns, control_prefix=control_prefix,
                prior_df=prior)
            report["shrinkage"] = {**params, "median_factor": float(factor.median())}
        profile_genes = genes[["gene", "n_guides", "n_cells"]].merge(
            shrunk, on="gene", how="inner")
        _ph_profile_tables("genes_shrink", profile_genes, destination, label,
                           outputs)

    if axes_asked:
        min_rho = float(_setting_number(
            settings.get("ops_profile_essential_axis_min_rho"),
            "ops_profile_essential_axis_min_rho", 0.30))
        stage = phases.to_numpy() if phases is not None else np.full(len(cells), "G1")
        covariates = generic_covariates(
            genes, cells[["gene"]].assign(phase=stage), control=control)
        profile_genes = profile_genes.dropna(subset=columns).reset_index(drop=True)
        guide_rows = profile_guides[columns].to_numpy(float)
        residual, info, _axis, guide_residual = remove_generic_axes(
            profile_genes[columns].to_numpy(float), profile_genes["gene"],
            covariates, axes=axes_asked, min_rho=min_rho,
            control_prefix=control_prefix, guide_values=guide_rows,
            guide_genes=profile_guides["gene"])
        profile_genes = profile_genes.copy()
        profile_genes[columns] = residual
        profile_guides = profile_guides.copy()
        profile_guides[columns] = guide_residual
        report["generic_axes"] = info
        _ph_profile_tables("genes_generic", profile_genes, destination, label,
                           outputs)

    weighting = str(settings.get("ops_profile_weighting") or "").strip().lower() or None
    if weighting not in (None, "contrastive_wiener"):
        raise ValueError(tr("ops_profile_weighting must be empty or "
                            "contrastive_wiener; got {value}", value=weighting))
    if weighting and settings.get("ops_profile_shrinkage"):
        raise ValueError(tr(
            "ops_profile_weighting and ops_profile_shrinkage both remove what "
            "the control guides explain; stacked, no direction is left, so "
            "choose one"))
    if weighting:
        profile_genes, profile_guides, info = contrastive_weighting(
            profile_genes, profile_guides, columns,
            control_prefix=control_prefix)
        columns = [c for c in profile_genes.columns if str(c).startswith("w_")]
        report["weighting"] = {"directions_kept": info["directions_kept"],
                               "kbar": info["kbar"],
                               "nt_guides": info["nt_guides"]}
        _ph_profile_tables("genes_weighted", profile_genes, destination, label,
                           outputs)
        _ph_profile_tables("guides_weighted", profile_guides, destination,
                           label, outputs)

    efficacy = str(settings.get("ops_guide_efficacy") or "").strip().lower() or None
    if efficacy not in (None, "jacks_blend"):
        raise ValueError(tr("ops_guide_efficacy must be empty or jacks_blend; "
                            "got {value}", value=efficacy))
    if efficacy and not weighting:
        raise ValueError(tr(
            "ops_guide_efficacy runs in the weighted space; set "
            "ops_profile_weighting to contrastive_wiener"))
    if efficacy:
        iters = int(_setting_number(settings.get("ops_guide_efficacy_iters"),
                                    "ops_guide_efficacy_iters", 20))
        emax = float(_setting_number(settings.get("ops_guide_efficacy_emax"),
                                     "ops_guide_efficacy_emax", 1.0))
        profile_genes, per_guide = guide_efficacy_profiles(
            profile_genes, profile_guides, columns=columns, iters=iters,
            emax=emax)
        report["guide_efficacy"] = {
            "iters": iters, "emax": emax,
            "guides_below_half": float((per_guide.dropna() < 0.5).mean())}
        _ph_profile_tables("genes_efficacy", profile_genes, destination, label,
                           outputs)
        _ph_profile_tables("guides_efficacy", per_guide.rename(
            "efficacy").rename_axis("guide").reset_index(), destination,
            label, outputs)

    similarity = str(settings.get("ops_profile_similarity") or "").strip().lower() or None
    if similarity not in (None, "gene_cosine", "guide_level", "responder",
                          "sliced_wasserstein"):
        raise ValueError(tr(
            "ops_profile_similarity must be gene_cosine, guide_level, "
            "responder or sliced_wasserstein; got {value}", value=similarity))
    if similarity == "guide_level":
        null_size = int(_setting_number(settings.get("ops_guide_null_size"),
                                        "ops_guide_null_size", 200000))
        result = guide_level_similarity(
            profile_guides.dropna(subset=columns), columns,
            control_prefix=control_prefix, null_size=null_size)
        report["guide_level"] = result["summary"]
        _ph_profile_tables("gene_pairs", result["pairs"], destination, label,
                           outputs)

    if similarity in ("responder", "sliced_wasserstein"):
        alpha = float(_setting_number(settings.get("ops_profile_responder_alpha"),
                                      "ops_profile_responder_alpha", 0.05))
        frame = pd.concat([cells[["sgRNA", "gene"]].reset_index(drop=True),
                           pd.DataFrame(embedding, columns=pcs)], axis=1)
        use = frame["gene"].to_numpy() != "nontargeting_drop"
        if phases is not None:
            use &= phases.to_numpy() != "M"
        result = responder_profiles(frame, pcs, control=control, guide="sgRNA",
                                    use=use, alpha=alpha)
        for name in ("summary", "genes", "guides"):
            _ph_profile_tables(f"responder_{name}", result[name], destination,
                               label, outputs)
        quantiles = pd.DataFrame(result["sliced_wasserstein"]).add_prefix("sw_")
        quantiles.insert(0, "gene", result["summary"]["gene"].to_numpy())
        _ph_profile_tables("genes_sliced_wasserstein", quantiles, destination,
                           label, outputs)
        report["responders"] = {"alpha": alpha, "genes": int(len(result["summary"]))}

    if settings.get("ops_escapers") and not settings.get("ops_penetrance"):
        raise ValueError(tr("ops_escapers reads the penetrance fit; set "
                            "ops_penetrance too"))
    if settings.get("ops_penetrance"):
        frame = pd.concat([cells[["well", "site", "sgRNA", "gene"]].reset_index(drop=True),
                           pd.DataFrame(embedding, columns=pcs)], axis=1)
        tested = sorted(g for g in pd.unique(frame["gene"].astype(str))
                        if not g.lower().startswith(control_prefix.lower()))
        result = penetrance_tests(
            frame, pcs, tested, control=control,
            folds=int(_setting_number(settings.get("ops_penetrance_folds"),
                                      "ops_penetrance_folds", 5)),
            control_pool=int(_setting_number(settings.get("ops_penetrance_nt_pool"),
                                             "ops_penetrance_nt_pool", 20000)),
            lam=float(_setting_number(settings.get("ops_penetrance_lambda"),
                                      "ops_penetrance_lambda", 0.5)),
            dbic=float(_setting_number(settings.get("ops_escaper_dbic"),
                                       "ops_escaper_dbic", 10.0)),
            min_separation=float(_setting_number(settings.get("ops_escaper_min_sep"),
                                                 "ops_escaper_min_sep", 2.0)))
        for name in ("genes", "controls", "guides"):
            _ph_profile_tables(f"penetrance_{name}", result[name], destination,
                               label, outputs)
        report["penetrance"] = {"genes": int(len(result["genes"])),
                                "controls": int(len(result["controls"]))}
        if settings.get("ops_escapers") and len(result["genes"]):
            escapers = result["genes"][result["genes"]["escaper"]]
            _ph_profile_tables("penetrance_escapers", escapers, destination,
                               label, outputs)
            report["penetrance"]["escapers"] = int(len(escapers))
    if settings.get("ops_phase_tests"):
        null_kind = str(settings.get("ops_phase_null") or "targeting").strip().lower()
        frame = pd.concat([cells[["well", "sgRNA"]].reset_index(drop=True),
                           pd.DataFrame(embedding, columns=pcs)], axis=1)
        frame["gene"] = nt_row_names(cells["sgRNA"], cells["gene"], rows="pseudo",
                                     control_prefix=control_prefix)
        frame["phase"] = phases.to_numpy()
        result = phase_tests(frame, pcs, control=control, null=null_kind)
        _ph_profile_tables("phase_tests", result["tests"], destination, label,
                           outputs)
        _ph_profile_tables("phase_calls", result["calls"], destination, label,
                           outputs)
        report["phase_tests"] = result["summary"]
    if settings.get("ops_neighbour_effects"):
        report["neighbours"] = _ph_neighbour_effects(
            settings, cells, embedding, pcs, destination, control_prefix)
        outputs.update(report["neighbours"].pop("outputs"))

    space = str(settings.get("ops_profile_space") or "").strip()
    if space and space != "tvn":
        kind, _sep, folder = space.partition(":")
        if kind not in ("embedding", "fusion") or not folder:
            raise ValueError(tr(
                "ops_profile_space must be tvn, embedding:<folder> or "
                "fusion:<folder>; got {value}", value=space))
        index = pd.read_parquet(os.path.join(folder, "index.parquet"),
                                columns=["row", "well", "site", "label"])
        exported = np.load(os.path.join(folder, "embeddings.npy"), mmap_mode="r")
        frame = cells[["well", "site", "label", "sgRNA", "gene"]].reset_index(
            drop=True).rename_axis("_cell").reset_index()
        index["well"] = index["well"].astype(str)
        frame["well"] = frame["well"].astype(str)
        matched = index.merge(frame, on=["well", "site", "label"], how="inner")
        if phases is not None:
            matched = matched[phases.to_numpy()[matched["_cell"].to_numpy()] != "M"]
        matched = matched.sort_values("row").reset_index(drop=True)
        matched_control = matched["gene"].astype(str).str.lower().str.startswith(
            control_prefix.lower())
        embedded, info = embedding_profiles(
            exported, matched, control=matched_control,
            rows=matched["row"].to_numpy(),
            whiten_epsilon=float(_setting_number(
                settings.get("ops_tvn_whiten_epsilon"), "ops_tvn_whiten_epsilon", 3.0)))
        _ph_profile_tables("genes_embedding", embedded, destination, label,
                           outputs)
        report["embedding"] = {"folder": folder, "cells_matched": int(len(matched)),
                               "exported_rows": int(len(index)), **info}
        if kind == "fusion":
            measured = pd.concat([matched[["sgRNA", "gene"]],
                                  pd.DataFrame(embedding[matched["_cell"].to_numpy()],
                                               columns=pcs)], axis=1)
            blocks = {"meas": aggregate_genes(aggregate_guides(
                measured, pcs, guide="sgRNA"), pcs),
                      "emb": embedded}
            order = _ph_names(settings.get("ops_fusion_blocks")) or ["meas", "emb"]
            unknown = [b for b in order if b not in blocks]
            if unknown:
                raise ValueError(tr(
                    "ops_fusion_blocks names {unknown}; the blocks are meas "
                    "and emb", unknown=unknown))
            fused = fusion_profiles({b: blocks[b] for b in order})
            _ph_profile_tables("genes_fusion", fused, destination, label,
                               outputs)
            report["embedding"]["fusion_dims"] = {
                b: int(fused.columns.str.startswith(f"f_{b}").sum()) for b in order}
    return report


def _rows_for_wells(db: str, table: str, plate: str, wells: Sequence[str]):
    """Rows of ``table`` for ``plate`` and any of ``wells``, or None."""
    from .ops_store import read_table, row_count

    if row_count(db, table) is None:
        return None
    frame = read_table(db, table)
    keep = (frame["plate"].astype(str) == str(plate)) & frame["well"].astype(
        str).isin([str(w) for w in wells])
    return frame[keep].reset_index(drop=True)


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

        Either folder may be a parent of both halves, and each setting
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
        The alpha phenotype path adds ``"ph_flatfield"``, ``"ph_segment"``,
        ``"ph_measure"``, ``"ph_masks"``, ``"merge"`` and ``"aggregate"``,
        run only when named here and after those four; ``ph_flatfield``
        builds the plate's maps once, before the wells, and ``ph_masks``
        writes each cell's field-edge flags and flat-field gains from the
        saved masks. A run of only these needs ``dst_root``
        and ``plate`` but no ``genotype_source``, takes its wells from
        ``ops_phenotype``, and adds to each well's ``ops_report.json``
        rather than replacing it. ``aggregate`` pools the wells run and
        writes ``ops_guides_<plate>`` and ``ops_genes_<plate>``.
        ``"ph_merged"`` (per well, after ``merge``) writes each field to
        ``merged/`` (:func:`_ph_merged`); ``"ph_tables"`` (after
        ``aggregate``) writes the standard ``measurements/measurements.db``
        tables (:func:`_ph_tables`); ``"ph_crops"`` cuts raw crops of owned
        cells into a sharded store (:func:`_ph_crops`). ``"ph_ownership"`` (per
        well, after ``ph_masks``) finds the copies of each cell across
        overlapping fields (:func:`_ph_cross_field_ownership`).
        ``"report"``, last, writes the yield funnel and the counts against
        a published table to ``results/ops_report/`` (:func:`_ph_report`). With
        ``ops_ph_secondary_registration`` set, each well's second phenotype
        acquisition is registered before its fields are read
        (:func:`_ph_secondary_shifts`). The alpha
        ``decode_qc`` phase (items 876 and 877) fits the false-call model
        and the genotype confidence over the wells decoded, writing
        ``results/ops_decode_qc``; it also runs after ``decode`` when
        ``ops_decode_qc`` or ``ops_genotype_confidence`` is on. The alpha
        ``compare_calls`` phase (item 878) matches spaCR's calls to other
        pipelines' per-cell tables (``ops_compare_tables``) and writes
        ``results/ops_compare``; it also runs after ``decode`` when those
        tables are set. With ``ops_compare_match`` ``"mask_overlap"``
        (item 922) it runs alone and matches spaCR's label masks to a
        reference pipeline's by pixel overlap instead
        (:func:`spacr.ops_decode.run_compare_masks`). The alpha
        profile path, ``"profile"`` and ``"evaluate"``, runs alone: it turns
        the stored cell tables of ``ops_profile_cells`` into gene profiles
        and scores them against reference gene pairs
        (:func:`spacr.ops_tvn.run_profile_phases`); ``"modules"`` builds
        the module atlas of those profiles
        (:func:`spacr.ops_modules.run_module_phase`) and ``"mitocarta"``
        scores clusterings against MitoCarta and an external module table
        (:func:`spacr.ops_clusters.run_mitocarta_benchmark`); like the
        profile path, both hold the BLAS and OpenMP pools to one thread while
        ``ops_deterministic_blas`` is on (the result's
        ``atlas_blas_threads``).
        ``"stress"``, after ``report``,
        times the finished store and writes
        ``results/ops_stress/stress_test.csv``
        (:func:`spacr.ops_run_infra.stress_rows`); ``ops_stress_report``
        runs it too.

        Plate-scale runs (item 872, alpha): ``ops_ph_acquisitions`` names
        each phenotype acquisition's folder and channels
        (:func:`_ph_acquisition_index`); ``ops_ph_pipeline`` and
        ``ops_ph_stage_dir`` stream the fields (:func:`_ph_fields`);
        ``ops_archive_root`` copies each finished well to archive storage,
        verifies it and resumes from its ``WELL_DONE`` marker
        (:func:`_ph_archive_resume`); ``ops_gpu_admission`` waits for a free
        card first (:func:`_ops_admitted_run`).
        The alpha
        ``"flat_nuclear"`` and ``"flat_validate"`` phases (item 912) run
        alone: they build each plate's nuclear-median DNA flat-field map and
        check it on held-out fields (:func:`spacr.ops_flatfield.run_flat_phases`).
        The alpha ``"score"`` phase (item 913) runs alone: it scores gene
        profiles and clusterings with copairs and EFAAR
        (:func:`spacr.ops_scoring.score_ops_profiles`). The profile path's
        alpha ``"cluster_sweep"`` (item 916, the Leiden resolution sweep),
        ``"etest"`` (item 917, the per-gene E-test and its power curve),
        ``"cluster_consensus"`` (item 928, guide-bootstrap consensus
        clustering) and ``"cluster_bench_test"`` (item 929, the calibrated
        consensus test) phases run inside :func:`spacr.ops_tvn.run_profile_phases`. Like the
        profile path, the flat-field and score phases hold the BLAS and
        OpenMP pools to one thread while ``ops_deterministic_blas`` is on
        (the result's ``blas_threads``). With ``ops_layout`` set to
        ``'cpg_cropped'`` (alpha, items 919 and 920) only ``"cpg_index"``,
        ``"cpg_merged"`` and ``"prefix_decode"`` run
        (:func:`spacr.cpg_cropped.run_cpg_phases`,
        :func:`spacr.ops_prefix.run_prefix_decode`); the default phases mean
        the first two, and the decode too under
        ``ops_library_mode='spacer_prefix'``.
    :param library: the guide barcodes, as a sequence or the path of a CSV
        with a ``prefix``, ``barcode`` or ``sequence`` column. When given,
        each barcode is also mapped to its closest guide. Overrides the
        ``ops_library`` setting.
    :returns: ``{"db": path, "wells": {well: {phase: report}}}``.
    :raises ValueError: when the source holds no tiles this can name, an
        unknown phase or well is asked for, or a phase's input is missing.
    """
    settings = dict(settings or {})
    if settings.get("ops_gpu_admission") and not _OPS_ADMITTED.get():
        return _ops_admitted_run(settings, wells=wells, phases=phases,
                                 library=library)
    unknown = [phase for phase in phases if phase not in _ALL_PHASES]
    if unknown:
        raise ValueError(
            f"unknown phases {unknown}; choose from {list(_ALL_PHASES)}")
    if (str(settings.get("ops_layout") or "tiles") == "cpg_cropped"
            or any(phase in _CPG_PHASES for phase in phases)):
        return _run_cpg(settings, phases)
    if any(phase in _PROFILE_PHASES for phase in phases):
        if any(phase not in _PROFILE_PHASES for phase in phases):
            raise ValueError("the profile phases run on their own; run "
                             f"{list(_PROFILE_PHASES)} without the image phases")
        from .ops_tvn import _pinned_blas, run_profile_phases

        profile = [phase for phase in phases
                   if phase in ("profile", "evaluate", "cluster_sweep", "etest",
                                "cluster_consensus", "cluster_bench_test")]
        result = run_profile_phases(settings, profile, report=_say) if profile else {}
        atlas = [phase for phase in phases if phase in ("modules", "mitocarta")]
        if atlas:
            with _pinned_blas(bool(settings.get("ops_deterministic_blas", True))) as pinned:
                if "modules" in atlas:
                    from .ops_modules import run_module_phase

                    result = {**result, "modules": run_module_phase(settings, report=_say)}
                if "mitocarta" in atlas:
                    from .ops_clusters import run_mitocarta_benchmark

                    result = {**result, "mitocarta": run_mitocarta_benchmark(settings, report=_say)}
            result = {**result, "atlas_blas_threads": pinned}
        return result
    if any(phase in _FLAT_PHASES for phase in phases):
        if any(phase not in _FLAT_PHASES for phase in phases):
            raise ValueError("the nuclear flat-field phases run on their own; run "
                             f"{list(_FLAT_PHASES)} without the image phases")
        from .ops_flatfield import run_flat_phases
        from .ops_tvn import _pinned_blas

        with _pinned_blas(bool(settings.get("ops_deterministic_blas", True))) as pinned:
            result = run_flat_phases(settings, phases, report=_say)
        return {**result, "blas_threads": pinned}
    if "score" in phases:
        if list(phases) != ["score"]:
            raise ValueError("the score phase runs on its own")
        from .ops_scoring import score_ops_profiles
        from .ops_tvn import _pinned_blas

        with _pinned_blas(bool(settings.get("ops_deterministic_blas", True))) as pinned:
            result = score_ops_profiles(settings, report=_say)
        return {**result, "blas_threads": pinned}
    if "compare_calls" in phases and str(settings.get("ops_compare_match") or "centroid") == "mask_overlap":
        if list(phases) != ["compare_calls"]:
            raise ValueError("compare_calls with ops_compare_match 'mask_overlap' runs on its own")
        from .ops_decode import run_compare_masks

        return {"compare_calls": run_compare_masks(settings, report=_say)}
    sequencing = any(phase in _PHASES for phase in phases)
    root = settings.get("genotype_source")
    if sequencing and (not isinstance(root, (str, os.PathLike))
                       or not os.path.isdir(str(root))):
        raise ValueError(
            f"genotype_source must be the folder of sequencing tiles; got {root!r}")
    root = str(root) if root else ""
    destination = str(settings.get("dst_root") or root)
    if not destination:
        from .qt.i18n import tr

        raise ValueError(tr(
            "the phenotype phases need dst_root, the folder holding "
            "measurements.db"))
    os.makedirs(destination, exist_ok=True)
    db = os.path.join(destination, "measurements.db")
    plate = str(settings.get("plate")
                or (os.path.basename(os.path.normpath(root)) if root else ""))
    gpu = bool(settings.get("ops_gpu", True))
    if "decode" in phases:
        _spot_detector(settings)
    barcodes = _load_library(library if library is not None
                             else settings.get("ops_library") or None)

    index: Dict[str, Any] = {}
    if sequencing:
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
    elif set(phases) <= {"decode_qc", "compare_calls"}:
        chosen = _decoded_wells(db, plate, wells)
    else:
        placed = _rows_for_wells(db, "ops_phenotype", plate,
                                 [str(w).upper() for w in wells] if wells
                                 else _placed_wells(db, plate))
        if placed is None or placed.empty:
            from .qt.i18n import tr

            raise ValueError(tr(
                "no ops_phenotype rows for plate {plate} in {db}; run the "
                "phenotype phase first", plate=plate, db=db))
        chosen = sorted(set(placed["well"].astype(str)))

    phenotype_root = settings.get("phenotype_source")
    phenotype_index: Dict[str, Any] = {}
    out_acquisitions = None
    wants_tiles = any(p in phases for p in ("phenotype", "ph_segment",
                                            "ph_measure", "ph_flatfield",
                                            "ph_merged", "ph_crops"))
    if wants_tiles and phenotype_root:
        if not os.path.isdir(str(phenotype_root)):
            raise ValueError(
                f"phenotype_source must be the folder of phenotype tiles; "
                f"got {phenotype_root!r}")
        acquisitions = _ph_acquisitions(settings)
        if acquisitions is not None:
            phenotype_index, ignored = _ph_acquisition_index(
                str(phenotype_root), acquisitions)
            out_acquisitions = {"folders": acquisitions["folders"],
                                "register_on": acquisitions["register_on"],
                                "ignored_files": ignored}
            if (len(acquisitions["folders"]) > 1
                    and not settings.get("ops_ph_secondary_channels")):
                settings["ops_ph_secondary_channels"] = ",".join(
                    c for _folder, names in acquisitions["folders"][1:]
                    for c in names)
        else:
            phenotype_index = (_index_tiles(str(phenotype_root), "uncycled")
                               or _index_tiles(str(phenotype_root), "cycled"))
        if not phenotype_index:
            raise ValueError(
                f"no tile under {phenotype_root} is named like "
                "20X_DAPI-GFP-A594-AF750_A1_DAPI-GFP_Site-0.tif "
                "(magnification, channel set, well, this file's channels, "
                "site) or like 20X_c1_A1_DAPI-GFP_Site-0.tif")
        if acquisitions is None:
            _one_acquisition(phenotype_index, "phenotype_source",
                             str(phenotype_root))

    out: Dict[str, Any] = {"db": db, "plate": plate, "wells": {}}
    if out_acquisitions is not None:
        out["ph_acquisitions"] = out_acquisitions
    if "ph_flatfield" in phases:
        out["ph_flatfield"] = _ph_flatfield(plate, chosen, phenotype_index,
                                            settings, destination, db)
        with open(os.path.join(destination, f"ops_flatfield_{plate}.json"),
                  "w", encoding="utf-8") as handle:
            json.dump(out["ph_flatfield"], handle, indent=1, default=str)
    archive_root = settings.get("ops_archive_root") or None
    well_phases = [p for p in phases if p not in _PH_PLATE_PHASES
                   and p not in ("decode_qc", "compare_calls")]
    only_qc = set(phases) <= {"decode_qc", "compare_calls"}
    for well in ([] if only_qc else chosen):
        folder = os.path.join(destination, well)
        report_path = os.path.join(folder, "ops_report.json")
        if settings.get("ops_gpu_admission") and os.path.exists(
                os.path.join(destination, "STOP")):
            out["stopped_before"] = well
            _say(f"STOP found in {destination}; stopping before well {well}")
            break
        if archive_root:
            resumed = _ph_archive_resume(destination, str(archive_root), plate,
                                         well, well_phases, settings)
            if resumed is not None:
                out["wells"][well] = resumed
                continue
        report: Dict[str, Any] = {}
        if not sequencing and os.path.isfile(report_path):
            with open(report_path, encoding="utf-8") as handle:
                report = json.load(handle) or {}
        started = time.perf_counter()
        if sequencing:
            cycle_files = index[well]
            nuclear = [cycle for cycle in sorted(cycle_files)
                       if any(_NUCLEAR in _plane_sources(files)
                              for files in cycle_files[cycle].values())]
            if not nuclear:
                raise ValueError(f"well {well} has no {_NUCLEAR} plane in any cycle")
            reference = nuclear[0]
            report.update({"reference_cycle": reference,
                           "cycles": sorted(cycle_files)})
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
        segment, measure = "ph_segment" in phases, "ph_measure" in phases
        secondary = _ph_secondary(settings)
        if (secondary and secondary["registration"] != "none"
                and well in phenotype_index
                and (segment or measure or "ph_merged" in phases
                     or "ph_crops" in phases)):
            report["ph_secondary"] = _ph_secondary_shifts(
                db, plate, well, phenotype_index[well], settings)
        if segment or measure:
            if well not in phenotype_index:
                report["ph_fields"] = {
                    "skipped": f"no phenotype tiles for well {well} under "
                               f"phenotype_source {phenotype_root!r}"}
            else:
                os.makedirs(folder, exist_ok=True)
                if not os.path.isfile(report_path) and "phenotype" in report:
                    with open(report_path, "w", encoding="utf-8") as handle:
                        json.dump(report, handle, indent=1, default=str)
                report["ph_fields"] = _ph_fields(
                    db, plate, well, phenotype_index[well], settings, gpu,
                    destination, segment=segment, measure=measure)
        if "ph_masks" in phases:
            report["ph_masks"] = _ph_masks(plate, well, settings, destination)
        if "ph_ownership" in phases:
            report["ph_ownership"] = _ph_cross_field_ownership(
                db, plate, well, settings, destination)
        if "merge" in phases:
            report["merge"] = _ph_merge(db, plate, well, settings,
                                        destination, gpu)
        if "ph_merged" in phases:
            report["ph_merged"] = _ph_merged(
                db, plate, well, phenotype_index.get(well) or {}, settings,
                destination)
        report["seconds"] = round(time.perf_counter() - started, 1)
        report["peak_rss_gb"] = _peak_rss_gb()
        os.makedirs(folder, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=1, default=str)
        if archive_root:
            from .ops_run_infra import archive_well

            report["archive"] = archive_well(
                destination, str(archive_root), plate, well,
                phases=well_phases, verify=_ph_archive_verify(settings),
                delete_local=bool(settings.get("ops_archive_delete_local")),
                report=report)
        out["wells"][well] = report
    if _wants_decode_qc(settings, phases):
        from .ops_decode import results_folder, run_decode_qc

        out["decode_qc"] = run_decode_qc(db, plate, chosen, settings, barcodes)
        with open(os.path.join(results_folder(db, "ops_decode_qc"),
                               f"decode_qc_{plate}.json"),
                  "w", encoding="utf-8") as handle:
            json.dump(out["decode_qc"], handle, indent=1, default=str)
    if "compare_calls" in phases or ("decode" in phases
                                     and settings.get("ops_compare_tables")):
        from .ops_decode import results_folder, run_compare_calls

        out["compare_calls"] = run_compare_calls(db, plate, chosen, settings,
                                                 barcodes)
        with open(os.path.join(results_folder(db, "ops_compare"),
                               f"compare_calls_{plate}.json"),
                  "w", encoding="utf-8") as handle:
            json.dump(out["compare_calls"], handle, indent=1, default=str)
    if archive_root and not only_qc and "stopped_before" not in out:
        from .ops_run_infra import mark_plate_done

        out["archive_plate_done"] = mark_plate_done(str(archive_root), plate,
                                                    chosen)
    if "aggregate" in phases:
        out["aggregate"] = _ph_aggregate(db, plate, chosen, settings,
                                         destination,
                                         keep_cells="ph_tables" in phases)
        out["aggregate"]["peak_rss_gb"] = _peak_rss_gb()
        with open(os.path.join(destination, f"ops_aggregate_{plate}.json"),
                  "w", encoding="utf-8") as handle:
            json.dump(out["aggregate"], handle, indent=1, default=str)
    if "ph_tables" in phases:
        out["ph_tables"] = _ph_tables(
            db, plate, chosen, settings, destination,
            ((out.get("aggregate") or {}).get("outputs") or {}).get(
                "cells_parquet"))
    if "ph_crops" in phases:
        out["ph_crops"] = _ph_crops(db, plate, chosen, phenotype_index,
                                    settings, destination)
    if "report" in phases:
        out["report"] = _ph_report(db, plate, chosen, settings, destination,
                                   out.get("aggregate"))
    if "stress" in phases or settings.get("ops_stress_report"):
        from .ops_run_infra import stress_rows, write_stress_report

        rows = stress_rows(destination, plate, chosen, out,
                           crop_root=_ph_merged_root(settings, destination),
                           control_prefix=str(settings.get("ops_control_prefix")
                                              or "nontargeting"))
        out["stress"] = {"csv": write_stress_report(destination, rows),
                         "rows": len(rows)}
    for phase in ("ph_tables", "ph_crops", "report", "stress"):
        if phase in out:
            with open(os.path.join(destination, f"ops_{phase}_{plate}.json"),
                      "w", encoding="utf-8") as handle:
                json.dump(out[phase], handle, indent=1, default=str)
    return out


def _wants_decode_qc(settings: Mapping[str, Any], phases: Sequence[str]) -> bool:
    """Whether the decode QC phase runs (items 876 and 877).

    :param settings: the run's settings.
    :param phases: the phases asked for.
    :returns: True when ``decode_qc`` is named, or the decode runs with
        ``ops_decode_qc`` or ``ops_genotype_confidence`` on.
    """
    from .ops_decode import _flag_or

    if "decode_qc" in phases:
        return True
    return "decode" in phases and (_flag_or(settings, "ops_decode_qc")
                                   or _flag_or(settings, "ops_genotype_confidence"))


def _decoded_wells(db: str, plate: str, wells) -> List[str]:
    """The wells of ``plate`` that ``ops_barcodes`` has rows for.

    :param db: the measurements database.
    :param plate: the plate.
    :param wells: the wells asked for, or None for every decoded well.
    :returns: the wells, sorted.
    :raises ValueError: when nothing was decoded.
    """
    from .ops_store import read_table, row_count

    if wells:
        return sorted(str(w).upper() for w in wells)
    if row_count(db, "ops_barcodes") is None:
        raise ValueError(f"no ops_barcodes in {db}; run the decode phase "
                         "before decode_qc or compare_calls")
    rows = read_table(db, "ops_barcodes")
    return sorted(set(rows.loc[rows["plate"].astype(str) == str(plate),
                               "well"].astype(str)))


#: True inside a run that :func:`_ops_admitted_run` let start, so the run
#: does not wait for the card a second time.
_OPS_ADMITTED: "contextvars.ContextVar[bool]" = contextvars.ContextVar(
    "ops_gpu_admitted", default=False)


def _ops_admitted_run(settings: Mapping[str, Any], *, wells, phases,
                      library) -> Dict[str, Any]:
    """:func:`run_ops` once the GPU admission policy lets it start (item 872, B11).

    :param settings: the OPS settings with ``ops_gpu_admission`` set; read
        for the ``ops_gpu_*`` settings
        (:func:`spacr.gpu_admission.admission_from_settings`). A ``STOP``
        file in ``dst_root`` ends the wait, and the run between wells.
    :param wells: as :func:`run_ops` takes them.
    :param phases: as :func:`run_ops` takes them.
    :param library: as :func:`run_ops` takes it.
    :returns: :func:`run_ops`'s result with ``gpu_admission`` (the wait's
        outcome); when the wait ended without admission, only that and no
        phase run.

    START and FINISH go to ``~/.spacr/gpu/log``; FINISH is written however
    the run ends, with the exception's name when it raised.
    """
    from .gpu_admission import admission_from_settings

    destination = str(settings.get("dst_root") or settings.get("genotype_source")
                      or "")
    plate = str(settings.get("plate") or os.path.basename(
        os.path.normpath(destination)) or "plate")
    admission = admission_from_settings(
        settings, f"spacr-ops-{plate}",
        stop_file=os.path.join(destination, "STOP") if destination else None)
    waited = admission.wait()
    waited["waits"] = admission.waits[-20:]
    if not waited["admitted"]:
        _say(f"GPU admission: not started ({waited})")
        return {"db": os.path.join(destination, "measurements.db"),
                "plate": plate, "wells": {}, "gpu_admission": waited}
    admission.start()
    token = _OPS_ADMITTED.set(True)
    status: Any = "error"
    try:
        out = run_ops(settings, wells=wells, phases=phases, library=library)
        status = 0
    except BaseException as failure:
        status = type(failure).__name__
        raise
    finally:
        _OPS_ADMITTED.reset(token)
        admission.finish(status)
    out["gpu_admission"] = waited
    return out


def _ph_archive_verify(settings: Mapping[str, Any]) -> str:
    """``ops_archive_verify``, checked: ``"md5"`` or ``"size"``."""
    from .ops_run_infra import ARCHIVE_VERIFY_MODES
    from .qt.i18n import tr

    mode = str(settings.get("ops_archive_verify") or "md5").strip().lower()
    if mode not in ARCHIVE_VERIFY_MODES:
        raise ValueError(tr("ops_archive_verify must be md5 or size; got "
                            "{value}", value=mode))
    return mode


def _ph_archive_resume(destination: str, archive_root: str, plate: str,
                       well: str, phases: Sequence[str],
                       settings: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    """A well's report from its archive copy, when the copy is complete (item 872, B9).

    :param destination: the run folder.
    :param archive_root: ``ops_archive_root``.
    :param plate: the plate name.
    :param well: the well name.
    :param phases: the per-well phases this run asks for.
    :param settings: read for ``ops_archive_verify`` and
        ``ops_archive_delete_local``.
    :returns: the archived report with ``archive``, ``{"resumed": True}``
        when the well need not run again; None when it must run: no
        marker, a marker of fewer phases, or a damaged copy whose local
        files are gone. A damaged copy whose local files are still there
        is copied again and verified, without measuring anything again.
    """
    from .ops_run_infra import archive_well, archived_well

    done = archived_well(archive_root, plate, well, phases=phases)
    if done is None:
        return None
    if done["ok"]:
        return {**done["report"], "archive": {"resumed": True,
                                              "marker": done["marker"]}}
    if not all(os.path.isfile(os.path.join(destination, p))
               for p in done["bad"]):
        _say(f"{well}: the archive copy is damaged ({done['bad']}) and the "
             "local files are gone; the well runs again")
        return None
    _say(f"{well}: the archive copy is damaged ({done['bad']}); copying again")
    copied = archive_well(destination, archive_root, plate, well,
                          phases=phases, verify=_ph_archive_verify(settings),
                          delete_local=bool(settings.get(
                              "ops_archive_delete_local")),
                          report=done["report"])
    return {**done["report"], "archive": {**copied, "resumed": True,
                                          "damaged": done["bad"]}}


def _placed_wells(db: str, plate: str) -> List[str]:
    """The wells of ``plate`` that ``ops_phenotype`` has rows for."""
    from .ops_store import read_table, row_count

    if row_count(db, "ops_phenotype") is None:
        return []
    frame = read_table(db, "ops_phenotype")
    return sorted(set(frame.loc[frame["plate"].astype(str) == str(plate),
                                "well"].astype(str)))


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


def _st_parse_window(value) -> Optional[Tuple[float, float, float, float]]:
    """A crop window as ``(x, y, width, height)``, or None for no window.

    :param value: None, an empty string, a sequence of four numbers or a
        string of four numbers separated by commas, semicolons or spaces.
    :returns: the four numbers as floats, or None.
    :raises ValueError: when the value does not hold four numbers or the
        width or height is not positive.
    """
    if value is None:
        return None
    if isinstance(value, str):
        parts = [part for part in value.replace(";", ",").replace(
            " ", ",").split(",") if part.strip()]
        if not parts:
            return None
    else:
        parts = list(value)
    if len(parts) != 4:
        raise ValueError(
            f"A window is x, y, width and height; got {len(parts)} values.")
    window = tuple(float(part) for part in parts)
    if window[2] <= 0 or window[3] <= 0:
        raise ValueError("A window needs a positive width and height.")
    return window


def _st_clip_window(window, shape) -> Tuple[int, int, int, int]:
    """Whole-pixel window bounds clipped to an image.

    :param window: ``(x, y, width, height)`` in pixels.
    :param shape: the image's ``(rows, columns)``.
    :returns: ``(row0, row1, col0, col1)``.
    :raises ValueError: when the window misses the image.
    """
    x, y, width, height = (int(round(float(v))) for v in window)
    row0, row1 = max(0, y), min(int(shape[0]), y + height)
    col0, col1 = max(0, x), min(int(shape[1]), x + width)
    if row1 <= row0 or col1 <= col0:
        raise ValueError(
            f"The window {x}, {y}, {width}, {height} px lies outside the "
            f"{shape[1]}x{shape[0]} px image.")
    return row0, row1, col0, col1


def _st_page_window(handle, page, bounds) -> np.ndarray:
    """One TIFF page's pixels inside ``bounds``, reading only the tiles or
    strips that overlap them.

    The tiles are decoded by tifffile itself, so every compression tifffile
    reads works (Xenium writes JPEG 2000 tiles), and nothing goes through
    zarr: ``TiffFile.aszarr`` does not open under zarr 3 with the tifffile
    releases that predate it, and this path does not care which zarr is
    installed. Pages tifffile cannot address by segment (separate sample
    planes, volumetric tiles) are read whole and cropped.

    :param handle: the open :class:`tifffile.TiffFile`.
    :param page: a page or frame of that file.
    :param bounds: ``(row0, row1, col0, col1)`` in the page's pixels.
    :returns: a ``(rows, columns)`` or ``(rows, columns, samples)`` array.
    """
    row0, row1, col0, col1 = bounds
    key = getattr(page, "keyframe", None) or page
    samples = int(getattr(key, "samplesperpixel", 1) or 1)
    offsets = tuple(getattr(page, "dataoffsets", ()) or ())
    counts = tuple(getattr(page, "databytecounts", ()) or ())
    simple = (offsets and len(offsets) == len(counts)
              and int(getattr(key, "imagedepth", 1) or 1) == 1
              and (samples == 1 or int(key.planarconfig) == 1))
    if not simple:
        whole = np.asarray(page.asarray())
        axes = str(getattr(key, "axes", ""))
        if whole.ndim == 3 and axes.startswith("S"):
            whole = np.moveaxis(whole, 0, -1)
        whole = np.squeeze(whole)
        return np.array(whole[row0:row1, col0:col1])
    length, width = int(key.imagelength), int(key.imagewidth)
    if key.is_tiled:
        seg_rows, seg_cols = int(key.tilelength), int(key.tilewidth)
    else:
        seg_rows = min(int(key.rowsperstrip or length), length)
        seg_cols = width
    across = -(-width // seg_cols)
    shape = (row1 - row0, col1 - col0) + ((samples,) if samples > 1 else ())
    out = np.zeros(shape, dtype=key.dtype)
    decode = key.decode
    tables = getattr(key, "jpegtables", None)
    handle_file = handle.filehandle
    for tile_row in range(row0 // seg_rows, (row1 - 1) // seg_rows + 1):
        for tile_col in range(col0 // seg_cols, (col1 - 1) // seg_cols + 1):
            index = tile_row * across + tile_col
            if index >= len(offsets) or not counts[index]:
                continue
            handle_file.seek(offsets[index])
            raw = handle_file.read(counts[index])
            segment, _where, _shape = decode(raw, index, jpegtables=tables)
            if segment is None:
                continue
            segment = np.asarray(segment)
            segment = segment.reshape(segment.shape[-3:])
            top, left = tile_row * seg_rows, tile_col * seg_cols
            a0, a1 = max(row0, top), min(row1, top + segment.shape[0])
            b0, b1 = max(col0, left), min(col1, left + segment.shape[1])
            if a1 <= a0 or b1 <= b0:
                continue
            piece = segment[a0 - top:a1 - top, b0 - left:b1 - left]
            out[a0 - row0:a1 - row0, b0 - col0:b1 - col0] = (
                piece if samples > 1 else piece[..., 0])
    return out


def _st_read_window(path: str, level: int = 0, window=None
                    ) -> Tuple[np.ndarray, Tuple[int, int]]:
    """One image plane, or the part of it inside a window, and its origin.

    For a TIFF with a window, only the tiles or strips under the window are
    read, plane by plane, and the planes of a z-stack (or of any axis other
    than Y, X and colour samples) are maximum-projected as they arrive, so
    memory holds two windows rather than a whole pyramid level: level 0 of a
    Xenium morphology image is about 33 GB. Other formats are read whole and
    cropped.

    :param path: the image file.
    :param level: the pyramid level of an OME-TIFF, 0 for full resolution.
    :param window: ``(x, y, width, height)`` in pixels of that level, or
        None for the whole plane. It is clipped to the image.
    :returns: ``(image, (x0, y0))``: the 2-D or ``(Y, X, 3|4)`` array and the
        column and row of its top-left pixel in the level.
    """
    window = _st_parse_window(window)
    lower = path.lower()
    if window is None:
        return _st_load_image(path, level), (0, 0)
    if lower.endswith((".tif", ".tiff", ".btf")):
        import tifffile

        with tifffile.TiffFile(path) as handle:
            series = handle.series[0]
            levels = getattr(series, "levels", None) or [series]
            chosen = levels[min(int(level), len(levels) - 1)]
            axes = str(chosen.axes)
            rows = int(chosen.shape[axes.index("Y")])
            cols = int(chosen.shape[axes.index("X")])
            bounds = _st_clip_window(window, (rows, cols))
            pages = list(getattr(chosen, "pages", None) or [])
            if not pages or any(page is None for page in pages):
                pages = [chosen[index] for index in range(len(chosen))]
            image = None
            for page in pages:
                plane = _st_page_window(handle, page, bounds)
                image = plane if image is None else np.maximum(image, plane)
        return image, (bounds[2], bounds[0])
    whole = _st_load_image(path, level)
    bounds = _st_clip_window(window, whole.shape[:2])
    return (np.array(whole[bounds[0]:bounds[1], bounds[2]:bounds[3]]),
            (bounds[2], bounds[0]))


def _st_load_image(path: str, level: int = 0, window=None) -> np.ndarray:
    """Read one image plane: a PNG, JPEG or a level of a (pyramidal) TIFF.

    A z-stack is maximum-projected; colour images keep their channels.

    :param path: the image file.
    :param level: the pyramid level of an OME-TIFF, 0 for full resolution.
    :param window: ``(x, y, width, height)`` in pixels of that level to read
        only that part (see :func:`_st_read_window`); None reads the whole
        plane.
    :returns: a 2-D or ``(Y, X, 3|4)`` array.
    """
    if _st_parse_window(window) is not None:
        return _st_read_window(path, level, window)[0]
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
        """Return the grayscale registration image and its reduction factor."""
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


def _st_replace_image_rows(db: str, table: str, frame, prcf) -> None:
    """Write ``frame`` into ``table``, replacing those images' earlier rows.

    :param db: the measurement database.
    :param table: the table name.
    :param frame: the images' rows, with a ``prcfo`` column.
    :param prcf: the image key whose earlier rows are replaced, or a
        sequence of image keys.
    """
    import pandas as pd

    from .tabular import database_tables, read_table, write_database

    keys = {prcf} if isinstance(prcf, str) else {str(k) for k in prcf}
    parts = [frame]
    if os.path.exists(db) and table in database_tables(db):
        old = read_table(db, table=table, canonicalise=False, report=None)
        if frame.empty and not len(frame.columns):
            if "prcfo" not in old.columns:
                return
            images = old["prcfo"].astype(str).str.rsplit("_o", n=1).str[0]
            if not images.isin(keys).any():
                return
            frame = old.iloc[:0].copy()
            parts = [frame]
        if "prcfo" in old.columns:
            images = old["prcfo"].astype(str).str.rsplit("_o", n=1).str[0]
            old = old[~images.isin(keys).to_numpy()]
        if len(old):
            parts.insert(0, old)
    elif frame.empty and not len(frame.columns):
        return
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
    keys = _st_prcfo(prcf, labels)
    return _st_write_object_counts(db, object_type, keys, [prcf], labels,
                                   genes, matrix, platform)


def _st_write_object_counts(db: str, object_type: str, keys, prcfs, labels,
                            genes, matrix, platform: str) -> Dict[str, Any]:
    """Write per-object gene counts of one or more images in one pass.

    The tables and columns of :func:`_st_write_counts`; every earlier row
    of the images in ``prcfs`` is replaced.

    :param db: the measurement database; created if absent.
    :param object_type: the object type the counts belong to.
    :param keys: the ``prcfo`` key of each matrix row.
    :param prcfs: the image keys whose earlier rows are replaced.
    :param labels: the object labels, one per matrix row.
    :param genes: the gene names, one per matrix column.
    :param matrix: the objects-by-genes counts.
    :param platform: the platform the counts came from.
    :returns: the table names and the number of objects and genes written.
    """
    import pandas as pd

    prcf = list(prcfs)
    totals = np.asarray(matrix.sum(axis=0)).ravel()
    order = np.argsort(-totals, kind="stable")
    wide_genes = order[:_ST_WIDE_GENE_LIMIT]
    wide_genes = np.sort(wide_genes)
    dense = matrix[:, wide_genes].toarray()
    names, seen = [], Counter()
    for i in wide_genes:
        seen[str(genes[i])] += 1
        count = seen[str(genes[i])]
        names.append(f"expr_{genes[i]}" if count == 1
                     else f"expr_{genes[i]}_{count}")
    wide = pd.DataFrame(dense, columns=names)
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
    xy = np.asarray(xy, dtype=float)
    margin = float(radius)
    inside = ((xy[:, 0] >= -margin) & (xy[:, 0] < image.shape[1] + margin)
              & (xy[:, 1] >= -margin) & (xy[:, 1] < image.shape[0] + margin))
    xy = xy[inside] / step
    if values is not None:
        values = np.asarray(values)[inside]
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
    axis.set_xlim(-0.5, shown.shape[1] - 0.5)
    axis.set_ylim(shown.shape[0] - 0.5, -0.5)
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
    A ``window`` crops the platform image before anything else, reading
    only the tiles under it (:func:`_st_read_window`), and the coordinates
    are shifted into the crop, so a crop of a full-resolution Xenium
    morphology image can be the analysis frame. With the user's own image
    the landmarks' source coordinates are pixels of the cropped platform
    image.

    :param request: ``image`` (a platform image name or a file of the user's
        own), ``level``, ``landmarks``, ``window`` (``x, y, width, height``
        of the platform image; empty for the whole image), ``window_unit``
        (``"um"`` for micrometres, else pixels of the chosen level) and
        ``window_image`` (a TIFF file the cropped platform image is written
        to, for segmenting it; empty writes nothing).
    :returns: a dict with ``xy``, ``radius``, ``microns_per_pixel``,
        ``image`` (the analysis image), ``matrix`` (3x3, identity without a
        user image) and ``registration`` (method and quality, and the
        window's origin and size in level pixels when cropped).
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
    window = _st_parse_window(request.get("window"))
    unit = str(request.get("window_unit") or "px").strip().lower()
    if window is not None and unit in ("um", "µm", "micron", "microns"):
        if not microns:
            raise ValueError(
                "A window in micrometres needs the image's pixel size, and "
                "this bundle does not give one; give the window in pixels.")
        window = tuple(value / microns for value in window)
    platform_image, origin = _st_read_window(bundle["images"][name], level,
                                             window)
    info = {"method": "platform", "frame": name, "level": level}
    if window is not None:
        xy = xy - np.asarray(origin, dtype=float)
        info["window"] = {"x": int(origin[0]), "y": int(origin[1]),
                          "width": int(platform_image.shape[1]),
                          "height": int(platform_image.shape[0]),
                          "unit": "px", "level": level}
        target = str(request.get("window_image") or "").strip()
        if target:
            from .tiff_io import write_tiff

            target = os.path.expanduser(target)
            os.makedirs(os.path.dirname(os.path.abspath(target)),
                        exist_ok=True)
            write_tiff(target, platform_image,
                       photometric=("rgb" if platform_image.ndim == 3
                                    else "minisblack"))
            info["window"]["image"] = target
    result = {"xy": xy, "radius": radius, "microns_per_pixel": microns,
              "image": platform_image, "matrix": np.eye(3),
              "registration": info}
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


def _st_pathogen_columns(genes, spec) -> np.ndarray:
    """Which genes are the pathogen's own probes or reads.

    :param genes: the bundle's gene names.
    :param spec: gene names separated by commas or semicolons, or a sequence
        of them; a name ending in ``*`` matches every gene starting with the
        rest (``CHIKV*``). Empty names none.
    :returns: a boolean per gene.
    """
    if isinstance(spec, str):
        entries = [part.strip() for part in spec.replace(";", ",").split(",")]
    else:
        entries = [str(part).strip() for part in (spec or ())]
    entries = [entry for entry in entries if entry]
    names = np.asarray([str(gene) for gene in genes], dtype=object)
    chosen = np.zeros(len(names), dtype=bool)
    for entry in entries:
        if entry.endswith("*"):
            prefix = entry[:-1]
            chosen |= np.array([name.startswith(prefix) for name in names],
                               dtype=bool)
        else:
            chosen |= names == entry
    return chosen


def _st_measured(db: str, table: str):
    """Measure's table of one object type, indexed by ``prcfo``.

    Measure writes ``prcf`` and ``object_label``; the key is composed from
    them as :func:`spacr.io._read_and_join_tables` composes it,
    ``<prcf>_o<label>``, when the table carries no ``prcfo`` column.

    :param db: the measurement database.
    :param table: the object table, ``cell`` or ``nucleus``.
    :returns: the table indexed by ``prcfo``, or None when it is absent or
        carries no key.
    """
    from .tabular import database_tables, read_table

    if not os.path.exists(db) or table not in database_tables(db):
        return None
    measured = read_table(db, table=table, report=None)
    if "prcfo" not in measured.columns:
        if not {"prcf", "object_label"} <= set(measured.columns):
            return None
        measured = measured.assign(prcfo=measured["prcf"].astype(str) + "_o"
                                   + measured["object_label"].astype(
                                       int).astype(str))
    return measured.drop_duplicates("prcfo").set_index("prcfo")


def _st_analyse(summary: Dict[str, Any], obs, matrix, genes,
                request: Mapping[str, Any], output: str, prcf: str,
                host: str, *, by_masks: bool,
                trend_by_masks: bool = True) -> None:
    """Infection calls, infected-versus-uninfected tests, the distance trend,
    the region summary and the ``.h5ad`` export of one set of host objects.

    :param summary: the run summary; ``files``, ``messages`` and
        ``infection`` are filled in.
    :param obs: one row per host object, with ``prcfo``, ``centroid_x``,
        ``centroid_y``, ``parasite_objects`` and ``distance_to_parasite``;
        ``infected`` and ``pathogen_transcripts`` are added.
    :param matrix: the host objects-by-genes counts.
    :param genes: the gene names.
    :param request: read for ``min_pathogen_transcripts``,
        ``pathogen_genes``, ``region_mask`` and ``anndata``.
    :param output: the results folder.
    :param prcf: the file-name prefix of the results.
    :param host: the host object type.
    :param by_masks: infection may be read from overlapped pathogen or
        vacuole objects.
    :param trend_by_masks: the distance trend runs over every object when
        infection comes from masks; otherwise it runs over the uninfected
        objects when infection comes from pathogen genes, against the
        distance already in ``distance_to_parasite``.
    """
    from .tabular import write_table

    files = summary["files"]
    raw_threshold = request.get("min_pathogen_transcripts")
    threshold = max(1, int(raw_threshold)) if raw_threshold not in (
        None, "") else 1
    pathogen_spec = request.get("pathogen_genes") or ""
    pathogen_columns = _st_pathogen_columns(genes, pathogen_spec)
    if (pathogen_spec and not pathogen_columns.any()):
        summary["messages"].append(
            "None of the pathogen genes named is in the data; infection is "
            "read from the pathogen and vacuole masks.")
    from_genes = bool(pathogen_columns.any())
    infection_known = from_genes or by_masks
    host_columns = ~pathogen_columns
    host_matrix = matrix[:, np.flatnonzero(host_columns)]
    host_genes = np.asarray(genes, dtype=object)[host_columns]
    if from_genes:
        pathogen_counts = np.asarray(
            matrix[:, np.flatnonzero(pathogen_columns)].sum(axis=1)).ravel()
        obs["pathogen_transcripts"] = pathogen_counts
        obs["infected"] = pathogen_counts >= threshold
    elif by_masks:
        obs["infected"] = obs["parasite_objects"].to_numpy() >= threshold
    if infection_known:
        infected = obs["infected"].to_numpy(dtype=bool)
        rule = "pathogen_genes" if from_genes else "pathogen_objects"
        summary["infection"] = {
            "rule": rule, "min_pathogen_transcripts": threshold,
            "pathogen_genes": [str(g) for g in
                               np.asarray(genes, dtype=object)[
                                   pathogen_columns]],
            "infected": int(infected.sum()),
            "uninfected": int((~infected).sum())}
        columns = ["prcfo", "parasite_objects"] + (
            ["pathogen_transcripts"] if from_genes else []) + ["infected"]
        files["infection"] = write_table(
            obs[columns].assign(min_pathogen_transcripts=threshold,
                                infection_rule=rule),
            os.path.join(output, f"{prcf}_{host}_infection.csv"))
        try:
            table = _st_compare_groups(host_matrix, host_genes, infected)
            table["min_pathogen_transcripts"] = threshold
            files["infected_vs_uninfected"] = write_table(
                table, os.path.join(output, f"{prcf}_{host}_infected_vs_uninfected.csv"))
            summary["infected_vs_uninfected"] = {
                "n": table.attrs.get("n"), "genes": int(len(table)),
                "significant": int((table["q_value"] < 0.05).sum()),
                "min_pathogen_transcripts": threshold}
        except ValueError as error:
            summary["messages"].append(str(error))
    if (by_masks if trend_by_masks else from_genes):
        try:
            chosen = (np.ones(len(obs), dtype=bool) if trend_by_masks
                      else ~obs["infected"].to_numpy(dtype=bool))
            trend = _st_distance_trend(
                host_matrix[np.flatnonzero(chosen)], host_genes,
                obs["distance_to_parasite"].to_numpy(dtype=float)[chosen])
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
    elif infection_known:
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


def _st_run(request: Mapping[str, Any], *, bundle=None,
            registered=None) -> Dict[str, Any]:
    """Read, register, assign to spaCR objects, write and analyse.

    :param request: ``folder`` (the platform output), ``platform``,
        ``bin_um``, ``min_qv``, ``image``, ``level``, ``landmarks``,
        ``masks`` (``{object type: mask file}``), ``region_mask``, ``db``
        (the measurement database), ``prcf`` (the image key; defaults to the
        first mask's file stem), ``output`` (the results folder; defaults to
        ``spatial_transcriptomics`` beside the database), ``gene`` (coloured
        in the overlay), ``anndata`` (write ``.h5ad`` when anndata is
        installed), ``pathogen_genes`` (the pathogen's own genes, see
        :func:`_st_pathogen_columns`) and ``min_pathogen_transcripts``
        (default 1). A host object is infected when it holds at least
        ``min_pathogen_transcripts`` transcripts (Visium: UMIs) of the
        pathogen genes; with no pathogen genes named, when it overlaps at
        least that many objects of the pathogen or vacuole mask. Pathogen
        genes are left out of the host comparisons. The window keys of
        :func:`_st_register` apply too.
    :param bundle: an already-read bundle, to skip reading it again.
    :param registered: an already-computed :func:`_st_register` result.
    :returns: a summary with the objects written per type, the analyses'
        file paths, the registration and the messages worth showing.
    """
    from .plot import save_figure

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
            points = np.asarray(registered["xy"], dtype=float)
            in_frame = int(((points[:, 0] >= 0) & (points[:, 0] < mask.shape[1])
                            & (points[:, 1] >= 0)
                            & (points[:, 1] < mask.shape[0])).sum())
            written["in_frame"] = in_frame
            written["assigned_fraction_in_frame"] = assigned / max(in_frame, 1)
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
    measured = _st_measured(db, host)
    if measured is not None:
        extra = measured.drop(columns=[c for c in measured.columns
                                       if c in obs.columns])
        obs = obs.join(extra, on="prcfo")
    _st_analyse(summary, obs, matrix, genes, request, output, prcf, host,
                by_masks=bool(parasites) and host not in parasites,
                trend_by_masks=True)
    files = summary["files"]
    from matplotlib.figure import Figure

    figure = Figure(figsize=(7.0, 6.0))
    from .figures.bundle import _register_figure_data
    _register_figure_data(figure, registered["image"], kind="overlay", gene=str(request.get("gene") or ""))
    gene = str(request.get("gene") or "")
    _st_draw_overlay(figure, registered["image"], registered["xy"],
                     _st_gene_values(bundle, gene),
                     radius=registered["radius"], mask=masks[host],
                     title=f"{bundle['platform']} {gene}".strip())
    files["overlay"] = save_figure(figure, os.path.join(
        output, f"{prcf}_overlay.png"))
    summary["obs"] = obs
    return summary



_ST_UNASSIGNED = frozenset({"UNASSIGNED", "-1", "", "nan", "None"})

_ST_NUCLEUS_UM = 8.0

_ST_TISSUE_FRACTION = 0.002


def _st_slide(source: str, platform: str = "auto") -> Dict[str, Any]:
    """The slide image of a platform folder, without reading its counts.

    :param source: a Xenium output bundle or a Space Ranger ``outs`` folder.
    :param platform: ``"auto"``, ``"xenium"``, ``"visium_hd"`` or
        ``"visium"``.
    :returns: ``platform``, ``folder``, ``channels`` (one image file per
        intensity channel; a Xenium ``morphology_focus`` folder gives one
        per file) and ``microns_per_pixel`` of the full-resolution image.
    :raises FileNotFoundError: when the folder holds no slide image.
    """
    import json

    folder = os.path.expanduser(str(source))
    platform = _st_platform(folder) if platform in ("", "auto", None) \
        else str(platform)
    if platform == "xenium":
        pixel_size = 0.2125
        meta = os.path.join(folder, "experiment.xenium")
        if os.path.exists(meta):
            with open(meta, encoding="utf-8") as handle:
                pixel_size = float(json.load(handle).get("pixel_size",
                                                         pixel_size))
        focus = os.path.join(folder, "morphology_focus")
        channels = sorted(
            os.path.join(focus, name) for name in os.listdir(focus)
            if name.lower().endswith((".ome.tif", ".ome.tiff"))) \
            if os.path.isdir(focus) else []
        if not channels:
            single = _st_first(folder, ("morphology_focus.ome.tif",
                                        "morphology_mip.ome.tif",
                                        "morphology.ome.tif"))
            channels = [single] if single else []
        if not channels:
            raise FileNotFoundError(
                f"{folder} holds no Xenium morphology image "
                "(morphology_focus/, morphology_focus.ome.tif, "
                "morphology_mip.ome.tif or morphology.ome.tif).")
        return {"platform": platform, "folder": folder,
                "channels": channels, "microns_per_pixel": pixel_size}
    images = [name for name in sorted(os.listdir(folder))
              if name.lower().endswith((".tif", ".tiff", ".btf"))]
    if not images:
        raise FileNotFoundError(
            f"{folder} holds no full-resolution microscope image (.tif, "
            ".tiff or .btf) beside binned_outputs/.")
    microns = None
    binned = os.path.join(folder, "binned_outputs")
    candidates = ([os.path.join(binned, name, "spatial",
                                "scalefactors_json.json")
                   for name in sorted(os.listdir(binned))]
                  if os.path.isdir(binned) else [])
    candidates.append(os.path.join(folder, "spatial",
                                   "scalefactors_json.json"))
    for path in candidates:
        if os.path.exists(path):
            with open(path, encoding="utf-8") as handle:
                factors = json.load(handle)
            microns = factors.get("microns_per_pixel")
            if microns is None and factors.get("spot_diameter_fullres"):
                microns = 55.0 / float(factors["spot_diameter_fullres"])
            if microns:
                break
    if not microns:
        raise FileNotFoundError(
            f"{folder} has no scalefactors_json.json with the image's "
            "microns_per_pixel.")
    return {"platform": platform, "folder": folder,
            "channels": [os.path.join(folder, images[0])],
            "microns_per_pixel": float(microns)}


def _st_level_geometry(path: str, level: int = 0) -> Dict[str, Any]:
    """Size of one pyramid level and its scale against full resolution.

    :param path: the image file.
    :param level: the level asked for; an image without it gives its
        coarsest level.
    :returns: ``level`` (the one used), ``shape`` (rows, columns),
        ``scale`` (full-resolution pixels per level pixel, x then y),
        ``levels`` (how many there are) and ``rgb``.
    """
    if path.lower().endswith((".tif", ".tiff", ".btf")):
        import tifffile

        with tifffile.TiffFile(path) as handle:
            series = handle.series[0]
            levels = list(getattr(series, "levels", None) or [series])
            used = min(max(int(level), 0), len(levels) - 1)

            def size(item):
                """``(rows, columns)`` of one level of the series."""
                axes = str(item.axes)
                return (int(item.shape[axes.index("Y")]),
                        int(item.shape[axes.index("X")]))

            base, chosen = size(levels[0]), size(levels[used])
            axes = str(levels[used].axes)
            rgb = "S" in axes and int(
                levels[used].shape[axes.index("S")]) in (3, 4)
        return {"level": used, "shape": chosen, "levels": len(levels),
                "scale": (base[1] / chosen[1], base[0] / chosen[0]),
                "rgb": bool(rgb)}
    image = _st_load_image(path)
    return {"level": 0, "shape": tuple(int(v) for v in image.shape[:2]),
            "levels": 1, "scale": (1.0, 1.0), "rgb": image.ndim == 3}


def _st_stain(stack: np.ndarray, stain: str, rgb: bool) -> np.ndarray:
    """The image the segmenter sees.

    :param stack: ``(Y, X, C)`` intensity channels, colour samples for an
        RGB slide.
    :param stain: ``auto``, ``dapi``, ``hematoxylin``, ``rgb`` or ``gray``.
    :param rgb: the slide is a colour (H&E) image.
    :returns: a 2-D float image, nuclei bright, or the colour image itself
        for ``rgb``.
    """
    stain = str(stain or "auto").strip().lower()
    if not rgb:
        return np.asarray(stack[..., 0], dtype=np.float32)
    colour = np.asarray(stack[..., :3])
    if stain == "rgb":
        return colour
    if stain in ("dapi", "gray", "grey"):
        return 255.0 - colour.astype(np.float32).mean(axis=-1)
    from skimage.color import rgb2hed

    return np.clip(rgb2hed(colour)[..., 0], 0, None).astype(np.float32)


def _st_overview(slide: Mapping[str, Any], geometry: Mapping[str, Any],
                 region: Tuple[int, int, int, int], stain: str,
                 side: int = 512) -> np.ndarray:
    """A sample of the region's stain image, for one slide-wide threshold.

    Nine windows of at most ``side`` pixels are read at the chosen level,
    centred at a sixth, a half and five sixths of the region's width and
    height, so the threshold sees nuclei at the resolution they are
    segmented at; a pyramid level's averaging dims nuclei and pulls an Otsu
    threshold into the background.

    :param slide: from :func:`_st_slide`.
    :param geometry: from :func:`_st_level_geometry` for the chosen level.
    :param region: ``(row0, row1, col0, col1)`` in the chosen level.
    :param stain: as :func:`_st_stain` takes it.
    :param side: the edge of each sampled window in pixels.
    :returns: the sampled stain pixels, one row per window.
    """
    path = slide["channels"][0]
    row0, row1, col0, col1 = region
    rgb = bool(geometry["rgb"])
    width = min(int(side), col1 - col0)
    height = min(int(side), row1 - row0)
    pieces = []
    for fy in (1.0 / 6, 0.5, 5.0 / 6):
        for fx in (1.0 / 6, 0.5, 5.0 / 6):
            left = int(min(max(col0, col0 + fx * (col1 - col0) - width / 2),
                           col1 - width))
            top = int(min(max(row0, row0 + fy * (row1 - row0) - height / 2),
                          row1 - height))
            image, _origin = _st_read_window(path, geometry["level"],
                                             (left, top, width, height))
            stack = image if image.ndim == 3 else image[..., None]
            gray = _st_stain(stack, "hematoxylin" if rgb and stain == "rgb"
                             else stain, rgb)
            pieces.append(np.asarray(gray, dtype=np.float32).ravel())
    return np.stack(pieces)


def _st_threshold_nuclei(image: np.ndarray, threshold: float,
                         diameter: float) -> np.ndarray:
    """Nuclei by a threshold and a distance-transform watershed.

    :param image: the 2-D stain image, nuclei bright.
    :param threshold: the slide-wide foreground threshold.
    :param diameter: the expected nucleus diameter in pixels.
    :returns: the label image.
    """
    from scipy import ndimage
    from skimage import feature, filters, measure, morphology, segmentation

    from .measure import _remove_objects_smaller_than

    foreground = filters.median(np.asarray(image, dtype=np.float32) > threshold,
                                morphology.disk(1)).astype(bool)
    smallest = max(4, int(0.1 * np.pi * (diameter / 2.0) ** 2))
    foreground = _remove_objects_smaller_than(foreground, smallest)
    if not foreground.any():
        return np.zeros(image.shape, dtype=np.int32)
    distance = ndimage.distance_transform_edt(foreground)
    peaks = feature.peak_local_max(
        distance, min_distance=max(2, int(diameter / 3.0)),
        labels=measure.label(foreground), exclude_border=False)
    markers = np.zeros(foreground.shape, dtype=np.int32)
    markers[tuple(peaks.T)] = np.arange(1, len(peaks) + 1)
    return segmentation.watershed(-distance, markers,
                                  mask=foreground).astype(np.int32)


def _st_field_stem(plate: str, index: int) -> str:
    """The merged file stem of field ``index`` of a slide.

    :param plate: the slide's plate name, escaped for file names.
    :param index: the window's raster index, from 0.
    :returns: ``<plate>_A01_<index + 1>``.
    """
    return f"{plate}_A01_{int(index) + 1}"


def _st_replace_plate_rows(db: str, table: str, frame, plate: str) -> None:
    """Write ``frame`` into ``table``, replacing that plate's earlier rows.

    :param db: the measurement database.
    :param table: the table name.
    :param frame: the plate's rows, with a ``plateID`` column.
    :param plate: the plate whose earlier rows are replaced.
    """
    import pandas as pd

    from .tabular import database_tables, read_table, write_database

    parts = [frame]
    if os.path.exists(db) and table in database_tables(db):
        old = read_table(db, table=table, canonicalise=False, report=None)
        if frame.empty:
            if "plateID" not in old.columns or not (
                    old["plateID"].astype(str) == str(plate)).any():
                return
            frame = old.iloc[:0].copy()
            parts = [frame]
        if "plateID" in old.columns:
            old = old[old["plateID"].astype(str) != str(plate)]
        if len(old):
            parts.insert(0, old)
    elif frame.empty:
        return
    combined = pd.concat(parts, ignore_index=True) if len(parts) > 1 \
        else frame
    write_database(combined, db, table, if_exists="replace")


def run_spatial_masks(settings: Mapping[str, Any], *,
                      segment=None) -> Dict[str, Any]:
    """Segment a whole Xenium or Visium HD slide into Measure-ready fields.

    The slide (or ``spatial_window`` of it) is cut into overlapping windows
    with :func:`spacr.ops_compose.windows_over`; each window's tiles are
    read with :func:`_st_read_window`, its nuclei segmented and grown into
    cells, and the observations sewn and numbered across the seams with
    :func:`spacr.ops_objects.sew` and :func:`spacr.ops_objects.number`, the
    OPS window code. Every object is kept from one window that saw all of
    it, and every window that keeps an object is written as a field:
    ``stack/<stem>.npy``, ``masks/cell_mask_stack`` and
    ``masks/nucleus_mask_stack``, and ``merged/<stem>.npy`` (the channels,
    then the cell and nucleus masks) with the plane-layout file, the layout
    Mask Generation writes and Measure reads. The fields' places on the
    slide go to the ``spatial_fields`` table and the objects to
    ``spatial_objects`` in ``measurements/measurements.db``.

    :param settings: the Spatial Mask Generation settings
        (:data:`spacr.ops_settings.SPATIAL_MASK_DEFAULTS`).
    :param segment: optional callable taking the stain image and returning
        its nucleus label image, in place of the configured segmenter.
    :returns: the run report, also written as ``spatial_mask_report.json``.
    :raises ValueError: when the source is not a folder or the overlap is
        not smaller than the window.
    """
    import json
    import shutil

    import pandas as pd
    from skimage.segmentation import expand_labels

    from .crops import MERGED_LAYOUT_SIDECAR
    from .ops_compose import windows_over
    from .ops_objects import number, objects_in_window, sew, unseen_records
    from .ops_settings import spatial_mask_defaults
    from .schema import escape_filename_component, parse_field_stem

    started = time.perf_counter()
    settings = spatial_mask_defaults(dict(settings or {}))
    source = settings.get("spatial_source")
    if not source or not os.path.isdir(os.path.expanduser(str(source))):
        raise ValueError(
            "spatial_source must be a Xenium output bundle or a Space Ranger "
            f"outs folder; got {source!r}")
    slide = _st_slide(str(source), str(settings.get("spatial_platform")
                                       or "auto"))
    geometry = _st_level_geometry(slide["channels"][0],
                                  int(settings.get("spatial_level") or 0))
    level = int(geometry["level"])
    rows, cols = geometry["shape"]
    microns = float(slide["microns_per_pixel"]) * float(geometry["scale"][0])
    window = _st_parse_window(settings.get("spatial_window"))
    unit = str(settings.get("spatial_window_unit") or "um").strip().lower()
    if window is not None and unit in ("um", "µm", "micron", "microns"):
        window = tuple(value / microns for value in window)
    row0, row1, col0, col1 = (_st_clip_window(window, (rows, cols))
                              if window is not None else (0, rows, 0, cols))
    canvas = (row1 - row0, col1 - col0)
    size = int(settings.get("spatial_tile_size") or 2048)
    overlap = int(settings.get("spatial_tile_overlap") or 0)
    if overlap >= size:
        raise ValueError(
            f"spatial_tile_overlap {overlap} must be smaller than "
            f"spatial_tile_size {size}.")
    windows = list(windows_over(canvas, size=size, overlap=overlap))
    folder = slide["folder"]
    dst = os.path.expanduser(str(settings.get("spatial_dst")
                                 or os.path.join(folder, "spacr_spatial")))
    sample = str(settings.get("spatial_sample") or "").strip() or \
        os.path.basename(os.path.normpath(folder)) or "slide"
    plate = escape_filename_component(sample)
    for sub in ("stack", "merged", os.path.join("masks", "cell_mask_stack"),
                os.path.join("masks", "nucleus_mask_stack"), "measurements"):
        target = os.path.join(dst, sub)
        os.makedirs(target, exist_ok=True)
        for name in os.listdir(target):
            if name.startswith(f"{plate}_A01_") and name.endswith(".npy"):
                os.remove(os.path.join(target, name))
    scratch = os.path.join(dst, "spatial", "_windows")
    os.makedirs(scratch, exist_ok=True)
    rgb = bool(geometry["rgb"])
    stain = str(settings.get("spatial_stain") or "auto")
    method = str(settings.get("spatial_segmenter") or "cellpose").lower()
    if method not in ("cellpose", "threshold"):
        raise ValueError(
            f"spatial_segmenter must be cellpose or threshold; got {method!r}")
    diameter = settings.get("spatial_diameter")
    nucleus_px = float(diameter) if diameter else _ST_NUCLEUS_UM / microns
    expansion = float(settings.get("spatial_cell_expansion_um") or 0.0)
    expansion_px = expansion / microns
    from skimage.filters import threshold_otsu

    overview = _st_overview(slide, geometry, (row0, row1, col0, col1), stain)
    threshold = float(threshold_otsu(overview)) if np.ptp(overview) > 0 \
        else float(overview.max()) + 1.0
    model = None
    seconds = Counter()
    observations: list = []
    stems: Dict[Tuple[int, int], Tuple[int, str]] = {}
    empty = 0
    channels = 0
    for index, piece in enumerate(windows):
        tick = time.perf_counter()
        bounds = (col0 + piece.left, row0 + piece.top, piece.width,
                  piece.height)
        planes = [_st_read_window(path, level, bounds)[0]
                  for path in slide["channels"]]
        if rgb:
            stack = np.asarray(planes[0])[..., :3]
        else:
            stack = np.dstack([np.asarray(p) if p.ndim == 2
                               else np.asarray(p)[..., 0] for p in planes])
        channels = int(stack.shape[-1])
        image = _st_stain(stack, stain, rgb)
        seconds["read"] += time.perf_counter() - tick
        tissue = (image if image.ndim == 2
                  else _st_stain(stack, "hematoxylin", rgb)) > threshold
        if tissue.mean() < _ST_TISSUE_FRACTION:
            empty += 1
            continue
        tick = time.perf_counter()
        if segment is not None:
            nuclei = np.asarray(segment(image), dtype=np.int32)
        elif method == "threshold":
            nuclei = _st_threshold_nuclei(image, threshold, nucleus_px)
        else:
            if model is None:
                from .model_zoo import _resolve_pinned_model_file

                pinned = _resolve_pinned_model_file(
                    settings, "spatial_cellpose_model", kinds=("cellpose",))
                model = _cellpose_model(
                    {"cellpose_model": str(pinned) if pinned is not None
                     else settings.get("spatial_cellpose_model")},
                    bool(settings.get("spatial_gpu", False)))
            extra: Dict[str, Any] = {}
            if diameter:
                extra["diameter"] = float(diameter)
            if image.ndim == 3:
                extra["channel_axis"] = -1
            nuclei = np.asarray(_governed_eval(model, image,
                                               site='ops_spatial',
                                               settings=settings,
                                               batch_size=16, **extra)[0],
                                dtype=np.int32)
        seconds["segment"] += time.perf_counter() - tick
        cells = (expand_labels(nuclei, distance=expansion_px).astype(np.int32)
                 if expansion_px > 0 else nuclei)
        found = objects_in_window(piece, cells, canvas=canvas)
        if not found:
            empty += 1
            continue
        stem = _st_field_stem(plate, index)
        np.save(os.path.join(dst, "stack", f"{stem}.npy"),
                stack.astype(np.uint16))
        np.save(os.path.join(scratch, f"{stem}_cells.npy"), cells)
        np.save(os.path.join(scratch, f"{stem}_nuclei.npy"), nuclei)
        observations.extend(found)
        stems[piece.offset()] = (index, stem)
    tick = time.perf_counter()
    groups = sew(observations)
    unseen = len(unseen_records(groups))
    objects = number(groups, strict=False)
    seconds["sew"] = time.perf_counter() - tick
    owned: Dict[Tuple[int, int], list] = {}
    for one in objects:
        owned.setdefault(tuple(one.window), []).append(one)
    field_rows, object_rows = [], []
    by_offset = {piece.offset(): piece for piece in windows}
    tick = time.perf_counter()
    for offset, (index, stem) in sorted(stems.items(),
                                        key=lambda item: item[1][0]):
        kept = sorted(owned.get(offset, ()), key=lambda one: one.object_id)
        stack_path = os.path.join(dst, "stack", f"{stem}.npy")
        if not kept:
            os.remove(stack_path)
            continue
        if len(kept) > np.iinfo(np.uint16).max:
            raise ValueError(
                f"{stem} keeps {len(kept)} objects, more than a uint16 mask "
                "holds; lower spatial_tile_size.")
        cells = np.load(os.path.join(scratch, f"{stem}_cells.npy"))
        nuclei = np.load(os.path.join(scratch, f"{stem}_nuclei.npy"))
        lookup = np.zeros(int(max(cells.max(), nuclei.max())) + 1,
                          dtype=np.uint16)
        for local, one in enumerate(kept, start=1):
            lookup[int(one.label)] = local
        cell_plane, nucleus_plane = lookup[cells], lookup[nuclei]
        stack = np.load(stack_path)
        np.save(os.path.join(dst, "masks", "cell_mask_stack", f"{stem}.npy"),
                cell_plane)
        np.save(os.path.join(dst, "masks", "nucleus_mask_stack",
                             f"{stem}.npy"), nucleus_plane)
        np.save(os.path.join(dst, "merged", f"{stem}.npy"),
                np.dstack([stack, cell_plane, nucleus_plane]).astype(
                    np.uint16))
        piece = by_offset[offset]
        prcf = parse_field_stem(stem).prcf
        field_rows.append({
            "plateID": sample, "prcf": prcf, "file_name": stem,
            "field_number": int(index) + 1, "x0": int(col0 + piece.left),
            "y0": int(row0 + piece.top), "width": int(piece.width),
            "height": int(piece.height), "level": level,
            "scale_x": float(geometry["scale"][0]),
            "scale_y": float(geometry["scale"][1]),
            "microns_per_pixel": microns, "platform": slide["platform"],
            "source": folder, "image": slide["channels"][0],
            "channels": channels, "objects": len(kept)})
        for local, one in enumerate(kept, start=1):
            x = float(one.centroid_x) + col0
            y = float(one.centroid_y) + row0
            object_rows.append({
                "plateID": sample, "prcfo": f"{prcf}_o{local}", "prcf": prcf,
                "object_label": local, "slide_object_id": int(one.object_id),
                "centroid_x": x, "centroid_y": y,
                "centroid_x_um": x * microns, "centroid_y_um": y * microns,
                "area": int(one.area), "n_observations": int(
                    one.n_observations)})
    seconds["write"] = time.perf_counter() - tick
    shutil.rmtree(scratch, ignore_errors=True)
    layout = {"version": 1, "intensity_channels": list(range(channels)),
              "mask_plane_order": ["cell", "nucleus"],
              "mask_dims": {"cell": channels, "nucleus": channels + 1}}
    with open(os.path.join(dst, "merged", MERGED_LAYOUT_SIDECAR), "w",
              encoding="utf-8") as handle:
        json.dump(layout, handle, indent=2, sort_keys=True)
        handle.write("\n")
    db = os.path.join(dst, "measurements", "measurements.db")
    _st_replace_plate_rows(db, "spatial_fields",
                           pd.DataFrame(field_rows), sample)
    _st_replace_plate_rows(db, "spatial_objects",
                           pd.DataFrame(object_rows), sample)
    report = {
        "platform": slide["platform"], "source": folder, "dst": dst,
        "db": db, "plate": sample, "image": slide["channels"][0],
        "level": level, "microns_per_pixel": microns,
        "region_px": [int(col0), int(row0), int(canvas[1]), int(canvas[0])],
        "segmenter": "custom" if segment is not None else method,
        "threshold": threshold, "windows": len(windows),
        "empty_windows": empty, "observations": len(observations),
        "objects": len(objects), "unseen_objects": unseen,
        "fields": len(field_rows), "channels": channels,
        "measure": {"src": os.path.join(dst, "merged"),
                    "channels": list(range(channels)),
                    "cell_mask_dim": channels,
                    "nucleus_mask_dim": channels + 1},
        "seconds": {key: round(value, 1) for key, value in seconds.items()},
        "total_seconds": round(time.perf_counter() - started, 1),
        "peak_rss_gb": _peak_rss_gb(),
    }
    with open(os.path.join(dst, "spatial_mask_report.json"), "w",
              encoding="utf-8") as handle:
        json.dump(report, handle, indent=1, default=str)
    _say(f"spatial masks: {len(objects)} objects in {len(field_rows)} fields "
         f"from {len(windows)} windows in {report['total_seconds']} s")
    return report


def _st_slide_xy(bundle: Mapping[str, Any]) -> Tuple[np.ndarray, float]:
    """Every transcript or bin in full-resolution pixels of the slide image.

    Xenium micrometres are divided by the morphology pixel size; Visium HD
    bins are already in full-resolution pixels of the microscope image.

    :param bundle: a bundle from :func:`_st_read_bundle`.
    :returns: ``(xy, radius)``: ``(N, 2)`` x and y, and the spot or bin
        radius in full-resolution pixels (0 for transcripts).
    """
    if bundle["platform"] == "xenium":
        xy, _radius, _size = _st_platform_xy(bundle, "", 0)
        return xy, 0.0
    obs = bundle["obs"]
    xy = np.column_stack([obs["x_full"].to_numpy(dtype=float),
                          obs["y_full"].to_numpy(dtype=float)])
    return xy, 0.5 * float(bundle["spot_diameter_full"])


def _st_field_frame(xy: np.ndarray, field: Mapping[str, Any]) -> np.ndarray:
    """Full-resolution slide pixels in the pixel frame of one field.

    :param xy: ``(N, 2)`` x and y in full-resolution pixels.
    :param field: a ``spatial_fields`` row: ``scale_x``, ``scale_y``
        (full-resolution pixels per field pixel), ``x0`` and ``y0`` (the
        field's origin in pixels of its level).
    :returns: the ``(N, 2)`` field-frame coordinates.
    """
    xy = np.asarray(xy, dtype=float)
    return np.column_stack([
        xy[:, 0] / float(field["scale_x"]) - float(field["x0"]),
        xy[:, 1] / float(field["scale_y"]) - float(field["y0"])])


def _st_subset(bundle: Mapping[str, Any], index: np.ndarray
               ) -> Dict[str, Any]:
    """The bundle restricted to some transcripts or bins.

    :param bundle: a bundle from :func:`_st_read_bundle`.
    :param index: the rows kept.
    :returns: a shallow copy holding only those rows.
    """
    part = dict(bundle)
    if bundle["platform"] == "xenium":
        part["transcripts"] = bundle["transcripts"].iloc[index]
    else:
        part["counts"] = bundle["counts"][index]
        part["obs"] = bundle["obs"].iloc[index].reset_index(drop=True)
    return part


def run_spatial_map(settings: Mapping[str, Any]) -> Dict[str, Any]:
    """Count every transcript or bin into the cells of the spatial fields.

    Reads the whole Xenium bundle (transcripts at ``spatial_min_qv``) or one
    Visium HD bin size, puts every point in full-resolution pixels of the
    slide image (:func:`_st_slide_xy`), then in the frame of each field
    Spatial Mask Generation wrote (:func:`_st_field_frame`), and counts it
    into the object of each plane named in ``spatial_objects``: Xenium
    transcripts by the pixel they fall in, Visium HD bins shared out by the
    fraction of the bin each object covers. Each object is owned by one
    field, so nothing is counted twice across the overlaps. The counts go to
    ``<object>_expression`` (and for Xenium ``<object>_expression_long``,
    for Visium HD ``<object>_spot_coverage``) in the fields'
    ``measurements.db``, keyed by ``prcfo`` like Measure's tables. Infection
    calls, infected-versus-uninfected tests, the distance trend (to the
    nearest infected cell, over the uninfected ones), the region summary,
    the ``.h5ad`` export and an overlay of the field with the most cells go
    to ``spatial_transcriptomics`` beside the masks.

    :param settings: the Spatial Map Barcodes settings
        (:data:`spacr.ops_settings.SPATIAL_MAP_DEFAULTS`).
    :returns: the run report, also written as ``spatial_map_report.json``.
    :raises ValueError: when the masks folder holds no fields of this slide
        or a requested plane is not in the fields.
    """
    import hashlib
    import json
    import tempfile

    import pandas as pd
    from matplotlib.figure import Figure
    from scipy import sparse
    from scipy.spatial import cKDTree

    from .crops import read_merged_plane_layout
    from .ops_settings import spatial_map_defaults
    from .plot import save_figure
    from .tabular import database_tables, read_table

    started = time.perf_counter()
    settings = spatial_map_defaults(dict(settings or {}))
    source = settings.get("spatial_source")
    masks_dir = settings.get("spatial_masks_dir")
    if not source or not os.path.isdir(os.path.expanduser(str(source))):
        raise ValueError(
            "spatial_source must be the Xenium bundle or Space Ranger outs "
            f"folder the masks were made from; got {source!r}")
    if not masks_dir or not os.path.isdir(os.path.expanduser(str(masks_dir))):
        raise ValueError(
            "spatial_masks_dir must be the folder Spatial Mask Generation "
            f"wrote; got {masks_dir!r}")
    masks_dir = os.path.expanduser(str(masks_dir))
    db = os.path.join(masks_dir, "measurements", "measurements.db")
    if not os.path.exists(db) or "spatial_fields" not in database_tables(db):
        raise ValueError(
            f"{masks_dir} has no spatial_fields table; run Spatial Mask "
            "Generation into it first.")
    fields = read_table(db, table="spatial_fields", canonicalise=False,
                        report=None)
    folder = os.path.expanduser(str(source))
    same = fields["source"].astype(str).map(
        lambda path: os.path.realpath(path) == os.path.realpath(folder))
    if same.any():
        fields = fields[same.to_numpy()]
    fields = fields.reset_index(drop=True)
    if not len(fields):
        raise ValueError(f"{masks_dir} holds no fields.")
    merged = os.path.join(masks_dir, "merged")
    layout = read_merged_plane_layout(merged) or {}
    dims = dict(layout.get("mask_dims") or {"cell": int(fields["channels"]
                                                        .iloc[0]),
                                            "nucleus": int(fields["channels"]
                                                           .iloc[0]) + 1})
    kinds = [kind.strip() for kind in str(
        settings.get("spatial_objects") or "cell").split(",") if kind.strip()]
    missing = [kind for kind in kinds if kind not in dims]
    if missing:
        raise ValueError(
            f"The fields have no {', '.join(missing)} plane; they hold "
            f"{', '.join(dims)}.")
    tick = time.perf_counter()
    bundle = _st_read_bundle(
        folder, str(settings.get("spatial_platform") or "auto"),
        bin_um=int(settings.get("spatial_bin_um") or 2),
        min_qv=float(settings.get("spatial_min_qv")
                     if settings.get("spatial_min_qv") is not None else 20.0))
    read_seconds = time.perf_counter() - tick
    platform = bundle["platform"]
    genes = bundle["genes"]
    xy0, radius0 = _st_slide_xy(bundle)
    total = int(len(xy0))
    per_kind: Dict[str, Dict[str, list]] = {
        kind: {"keys": [], "labels": [], "prcfs": [], "matrices": [],
               "coverage": []} for kind in kinds}
    in_fields = np.zeros(total, dtype=bool)
    in_cell = np.zeros(total, dtype=bool)
    best = (None, -1)
    tick = time.perf_counter()
    for _row, field in fields.iterrows():
        frame_xy = _st_field_frame(xy0, field)
        radius = radius0 / float(field["scale_x"])
        width, height = int(field["width"]), int(field["height"])
        near = np.flatnonzero(
            (frame_xy[:, 0] >= -radius) & (frame_xy[:, 0] < width + radius)
            & (frame_xy[:, 1] >= -radius) & (frame_xy[:, 1] < height + radius))
        array = np.load(os.path.join(merged, f"{field['file_name']}.npy"),
                        mmap_mode="r")
        part = _st_subset(bundle, near)
        points = frame_xy[near]
        inside = ((points[:, 0] >= 0) & (points[:, 0] < width)
                  & (points[:, 1] >= 0) & (points[:, 1] < height))
        in_fields[near[inside]] = True
        for kind in kinds:
            mask = np.asarray(array[..., int(dims[kind])], dtype=np.int64)
            labels, matrix, coverage = _st_object_counts(part, points, mask,
                                                         radius)
            if platform == "xenium" and kind == kinds[0]:
                in_cell[near[_st_points_in_mask(points, mask) > 0]] = True
            prcf = str(field["prcf"])
            store = per_kind[kind]
            store["keys"].extend(_st_prcfo(prcf, labels))
            store["labels"].append(np.asarray(labels, dtype=np.int64))
            store["prcfs"].append(prcf)
            store["matrices"].append(matrix)
            if coverage is not None and len(coverage):
                spot = near[coverage["spot"].to_numpy()]
                if kind == kinds[0]:
                    in_cell[spot] = True
                store["coverage"].append(coverage.drop(columns=["spot"]).assign(
                    barcode=bundle["obs"]["barcode"].to_numpy()[spot],
                    prcfo=_st_prcfo(prcf, coverage["object_label"])))
            if kind == kinds[0] and len(labels) > best[1]:
                best = (field, len(labels))
    assign_seconds = time.perf_counter() - tick
    output = os.path.join(masks_dir, "spatial_transcriptomics")
    os.makedirs(output, exist_ok=True)
    plate = str(fields["plateID"].iloc[0])
    summary: Dict[str, Any] = {
        "platform": platform, "source": folder, "db": db, "output": output,
        "plate": plate, "fields": int(len(fields)), "genes": int(len(genes)),
        "points": total, "points_in_fields": int(in_fields.sum()),
        "objects": {}, "files": {}, "messages": []}
    if platform != "xenium":
        summary["bin_um"] = int(settings.get("spatial_bin_um") or 2)
    host_matrix = None
    for kind in kinds:
        store = per_kind[kind]
        labels = (np.concatenate(store["labels"]) if store["labels"]
                  else np.zeros(0, dtype=np.int64))
        matrix = (sparse.vstack(store["matrices"]).tocsr()
                  if store["matrices"] else sparse.csr_matrix((0, len(genes))))
        written = _st_write_object_counts(db, kind, store["keys"],
                                          store["prcfs"], labels, genes,
                                          matrix, platform)
        written["counted"] = float(matrix.sum())
        table = f"{kind}_spot_coverage"
        if store["coverage"] or (platform != "xenium"
                                  and table in database_tables(db)):
            coverage = (pd.concat(store["coverage"], ignore_index=True)
                        if store["coverage"] else pd.DataFrame())
            _st_replace_image_rows(db, table, coverage, store["prcfs"])
            written["tables"] += (table,)
        summary["objects"][kind] = written
        if kind == kinds[0]:
            host_matrix, host_keys, host_labels = matrix, store["keys"], labels
    host = kinds[0]
    if platform == "xenium":
        summary["assigned_fraction_in_fields"] = (
            summary["objects"][host]["counted"]
            / max(summary["points_in_fields"], 1))
        own = bundle["transcripts"]["xenium_cell_id"].astype(str).to_numpy()
        theirs = in_fields & ~np.isin(own, list(_ST_UNASSIGNED))
        if theirs.any():
            summary["xenium_assigned_in_spacr_cell"] = float(
                in_cell[theirs].mean())
            summary["xenium_assigned_in_fields"] = int(theirs.sum())
    else:
        summary["bins_covering_a_cell"] = int(in_cell.sum())
        counts = np.asarray(bundle["counts"].sum(axis=1)).ravel()
        summary["umis_in_fields"] = float(counts[in_fields].sum())
        summary["umi_fraction_assigned"] = (
            summary["objects"][host]["counted"]
            / max(summary["umis_in_fields"], 1.0))
    objects = (read_table(db, table="spatial_objects", canonicalise=False,
                          report=None)
               if "spatial_objects" in database_tables(db) else None)
    obs = pd.DataFrame({"prcfo": host_keys,
                        "prcf": [key.rsplit("_o", 1)[0] for key in host_keys],
                        "object_label": host_labels})
    if objects is not None:
        geometry = objects.drop_duplicates("prcfo").set_index("prcfo")[
            ["centroid_x", "centroid_y", "centroid_x_um", "centroid_y_um"]]
        obs = obs.join(geometry, on="prcfo")
    else:
        for column in ("centroid_x", "centroid_y", "centroid_x_um",
                       "centroid_y_um"):
            obs[column] = np.nan
    obs["expr_total"] = np.asarray(host_matrix.sum(axis=1)).ravel()
    obs["parasite_objects"] = 0
    obs["distance_to_parasite"] = np.nan
    request = {"pathogen_genes": settings.get("spatial_pathogen_genes") or "",
               "min_pathogen_transcripts": settings.get(
                   "spatial_min_pathogen_transcripts"),
               "anndata": bool(settings.get("spatial_anndata", True))}
    pathogen = _st_pathogen_columns(genes, request["pathogen_genes"])
    if pathogen.any() and len(obs):
        raw = request["min_pathogen_transcripts"]
        threshold = max(1, int(raw)) if raw not in (None, "") else 1
        load = np.asarray(host_matrix[:, np.flatnonzero(pathogen)].sum(
            axis=1)).ravel()
        infected = load >= threshold
        centres = obs[["centroid_x_um", "centroid_y_um"]].to_numpy(dtype=float)
        if infected.any() and np.isfinite(centres).all():
            distance, _index = cKDTree(centres[infected]).query(centres)
            obs["distance_to_parasite"] = np.where(infected, 0.0, distance)
    if len(obs):
        measured = _st_measured(db, host)
        if measured is not None:
            extra = measured.drop(columns=[c for c in measured.columns
                                           if c in obs.columns])
            obs = obs.join(extra, on="prcfo")
            summary["joined_measure_rows"] = int(
                obs["prcfo"].isin(measured.index).sum())
        _st_analyse(summary, obs, host_matrix, genes, request, output,
                    plate, host, by_masks=False, trend_by_masks=False)
    else:
        summary["messages"].append("No object received any count.")
    if best[0] is not None:
        field = best[0]
        array = np.load(os.path.join(merged, f"{field['file_name']}.npy"),
                        mmap_mode="r")
        channels = int(field["channels"])
        image = (np.asarray(array[..., :3]).astype(np.uint8)
                 if channels == 3 and str(platform).startswith("visium")
                 else np.asarray(array[..., 0]))
        frame_xy = _st_field_frame(xy0, field)
        gene = str(settings.get("spatial_gene") or "")
        figure = Figure(figsize=(7.0, 6.0))
        from .figures.bundle import _register_figure_data
        _register_figure_data(figure, image, kind="overlay", gene=gene)
        _st_draw_overlay(figure, image, frame_xy,
                         _st_gene_values(bundle, gene),
                         radius=radius0 / float(field["scale_x"]),
                         mask=np.asarray(array[..., int(dims[host])]),
                         title=f"{platform} {field['prcf']} {gene}".strip())
        summary["files"]["overlay"] = save_figure(figure, os.path.join(
            output, f"{field['prcf']}_overlay.png"))
    summary["seconds"] = {"read": round(read_seconds, 1),
                          "assign": round(assign_seconds, 1),
                          "total": round(time.perf_counter() - started, 1)}
    summary["peak_rss_gb"] = _peak_rss_gb()
    report_path = os.path.join(output, "spatial_map_report.json")
    suffixes = {"infection": "_infection.csv",
                "infected_vs_uninfected": "_infected_vs_uninfected.csv",
                "distance_trend": "_distance_trend.csv",
                "region_summary": "_region_summary.csv",
                "anndata": ".h5ad"}
    runs = []
    try:
        if not os.path.islink(report_path):
            with open(report_path, "rb") as handle:
                payload = handle.read(1024 * 1024 + 1)
            previous = json.loads(payload) if len(payload) <= 1024 * 1024 else {}
            ledger = previous.get("analysis_output_ownership", {})
            valid = (isinstance(ledger, dict)
                     and set(ledger) == {"version", "runs"}
                     and type(ledger["version"]) is int
                     and ledger["version"] == 1
                     and isinstance(ledger["runs"], list)
                     and os.path.realpath(previous.get("output", ""))
                     == os.path.realpath(output)
                     and os.path.realpath(previous.get("db", ""))
                     == os.path.realpath(db))
            seen = set()
            for entry in ledger.get("runs", []) if valid else []:
                if (not isinstance(entry, dict)
                        or set(entry) != {"plate", "host", "files"}
                        or not isinstance(entry["plate"], str)
                        or not isinstance(entry["host"], str)
                        or not isinstance(entry["files"], dict)):
                    valid = False
                    break
                identity = (entry["plate"], entry["host"])
                allowed = {f"{entry['plate']}_{entry['host']}{suffix}"
                           for suffix in suffixes.values()}
                if identity in seen or any(
                        not isinstance(name, str) or name not in allowed
                        or "/" in name or "\\" in name
                        or not isinstance(digest, str)
                        or re.fullmatch(r"[0-9a-f]{64}", digest) is None
                        for name, digest in entry["files"].items()):
                    valid = False
                    break
                seen.add(identity)
            if valid:
                runs = ledger["runs"]
    except (OSError, ValueError, TypeError, AttributeError):
        pass
    owned = next((entry for entry in runs
                  if (entry["plate"], entry["host"]) == (plate, host)), None)
    if owned is None:
        owned = {"plate": plate, "host": host, "files": {}}
        runs.append(owned)
    for key, suffix in suffixes.items():
        name = f"{plate}_{host}{suffix}"
        path = summary["files"].get(key)
        if (not isinstance(path, str) or "/" in name or "\\" in name
                or os.path.basename(path) != name or os.path.islink(path)
                or os.path.dirname(os.path.realpath(path))
                != os.path.realpath(output) or not os.path.isfile(path)):
            continue
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
        owned["files"][name] = digest.hexdigest()
    summary["analysis_output_ownership"] = {"version": 1, "runs": runs}
    report = {key: value for key, value in summary.items()}
    descriptor, pending_report = tempfile.mkstemp(
        prefix=".spatial_map_report-", suffix=".json", dir=output)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=1, default=str)
        os.replace(pending_report, report_path)
    finally:
        if os.path.exists(pending_report):
            os.remove(pending_report)
    if not len(obs):
        retired = []
        for name, expected in owned["files"].items():
            path = os.path.join(output, name)
            if (os.path.islink(path) or not os.path.isfile(path)
                    or os.path.dirname(os.path.realpath(path))
                    != os.path.realpath(output)):
                continue
            digest = hashlib.sha256()
            with open(path, "rb") as handle:
                while chunk := handle.read(1024 * 1024):
                    digest.update(chunk)
            if digest.hexdigest() == expected:
                os.remove(path)
                retired.append(name)
        if retired:
            for name in retired:
                del owned["files"][name]
            descriptor, pending_report = tempfile.mkstemp(
                prefix=".spatial_map_report-", suffix=".json", dir=output)
            try:
                with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                    json.dump(report, handle, indent=1, default=str)
                os.replace(pending_report, report_path)
            finally:
                if os.path.exists(pending_report):
                    os.remove(pending_report)
    _say(f"spatial map: {summary['objects'][host]['objects']} {host} objects "
         f"counted from {summary['points_in_fields']} of {total} points")
    summary["obs"] = obs
    return summary
