"""A parameter grid search for the OPS spacer-prefix decode on a few tiles (item 960, alpha).

Before a full run, pick a few representative tiles and score every
combination of a grid of spot-detection and decode settings on them. Each
combination reports, per tile and pooled over the tiles, the spot count, the
read mapping rate, the cell assignment rate, the decoy FDR and the
barcode-shuffle call rate, and how long it took. The last two are the
false-call controls of items 942 and 949, so a setting is never chosen on
yield alone.

The grid uses the decode itself, never a second implementation: each tile's
spots are found once, down to the lowest read threshold of the grid, as
:func:`spacr.ops_prefix.run_prefix_decode` finds them while it tunes, and
each combination is then called, matched and counted by the decode's own
functions. A combination's numbers are therefore the numbers a run at that
setting writes to ``site_summary.csv``.

The default point is the settings as they stand. When they tune the
thresholds per well (``ops_prefix_threshold_tuning='per_well_decoy'``, the
default), the decode runs once on the chosen tiles to find each well's
choice, and every grid point changes only the settings the grid names
around that choice.

Turn on ``ops_grid_search`` and an OPS run of the cropped-site layout runs
the grid instead of the decode. Every grid run is written as CSV
(``ops_grid_<run id>.csv``, identical when the same settings are run
again), with the timings in a second CSV and a plot of yield against the
shuffle call rate that marks the default and the Pareto front. Applying a
point writes its values into the settings with the grid run's ID, tiles and
seed.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
import time
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

#: The settings a grid may vary, with the type each value is read as.
_GRID_KEYS: Dict[str, Callable[[Any], Any]] = {
    "ops_read_threshold": float,
    "ops_read_mode": str,
    "ops_prefix_min_read_quality": float,
    "ops_prefix_max_distance": int,
    "ops_quality_min_reads": int,
    "ops_prefix_crosstalk": str,
}

#: How the tiles are chosen, the first being the default.
_TILE_MODES: Tuple[str, ...] = ("stratified", "edge_centre", "user")

#: An example grid, as ``ops_grid_params`` writes one.
_GRID_EXAMPLE = "ops_read_threshold=40,80,160; ops_read_mode=log,raw"

#: How many tiles are chosen when ``ops_grid_n_tiles`` is empty.
_DEFAULT_TILES = 8

#: The columns of the grid CSV that hold counts, summed over tiles for the
#: pooled row.
_COUNTS = ("spots", "reads", "mapped_reads", "exact", "corrected", "cells_in_labels",
           "cells_real", "cells_decoy", "shuffle_called")

#: The decode settings the run ID is taken from, beside the grid and tiles.
_ID_KEYS = ("ops_library", "ops_library_sequence_col", "ops_library_gene_col",
            "ops_prefix_length", "ops_sbs_cycles", "ops_cycle_mask", "ops_prefix_collision",
            "ops_prefix_metric", "ops_prefix_max_distance", "ops_prefix_crosstalk",
            "ops_quality_min_reads", "ops_prefix_quality_method",
            "ops_random_barcode_control", "ops_control_seed", "ops_prefix_masks",
            "ops_read_threshold", "ops_prefix_decoy_k", "ops_prefix_decoy_seed",
            "ops_prefix_fdr_scale", "ops_prefix_min_read_quality",
            "ops_prefix_threshold_tuning", "ops_prefix_fdr_target_pct",
            "ops_prefix_tuning_grid", "ops_prefix_nearmiss_decoy_k",
            "ops_prefix_fdr_calibration", "ops_read_mode", "cpg_root",
            "ops_decode_auto_tune_fraction", "ops_decode_auto_seed",
            "ops_decode_auto_objective", "ops_decode_auto_f_beta",
            "ops_decode_auto_unreachable", "ops_decode_auto_ci")

#: The libraries a grid worker matches against, set once per process by
#: :func:`_init_worker`.
_LIBRARIES: Dict[str, Any] = {}


def _cast(key: str, value: Any) -> Any:
    """One grid value, read as its setting's type.

    :param key: the setting.
    :param value: the value as written.
    :returns: the value.
    :raises ValueError: for a value of the wrong type, or a read mode other
        than ``log`` or ``raw``.
    """
    kind = _GRID_KEYS[key]
    if isinstance(value, str):
        value = value.strip().strip("'\"")
    try:
        out = kind(float(value)) if kind is int else kind(value)
    except (TypeError, ValueError):
        raise ValueError(f"ops_grid_params: {key} takes {kind.__name__} values; "
                         f"got {value!r}") from None
    if kind is int and float(value) != out:
        raise ValueError(f"ops_grid_params: {key} takes whole numbers; got {value!r}")
    if key == "ops_read_mode":
        out = out.lower()
        if out not in ("log", "raw"):
            raise ValueError("ops_grid_params: ops_read_mode takes 'log' and 'raw'; "
                             "'auto' chooses per well and is the default point itself")
    return out


def _parse_grid(value: Any) -> Dict[str, List[Any]]:
    """Read a parameter grid from a setting, a settings file or the CLI.

    The one parser of every way a grid is given: a dict of lists, a JSON
    object, or text such as ``ops_read_threshold=40,80,160;
    ops_read_mode=log,raw`` (``:`` may stand for ``=``, and a newline for
    ``;``).

    :param value: the grid, or None or empty for no grid.
    :returns: ``{setting: values}`` in the order given, each value read as
        its setting's type and repeats dropped; empty for no grid.
    :raises ValueError: for a setting the grid cannot vary (the message
        lists those it can), an empty value list, or a value of the wrong
        type.
    """
    if value is None or (isinstance(value, str) and not value.strip()):
        return {}
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("{"):
            try:
                value = json.loads(text)
            except json.JSONDecodeError as error:
                raise ValueError(f"ops_grid_params is not valid JSON: {error}") from None
        else:
            value = {}
            for part in text.replace("\n", ";").split(";"):
                if not part.strip():
                    continue
                sep = "=" if "=" in part else ":"
                if sep not in part:
                    raise ValueError(f"ops_grid_params: {part.strip()!r} has no '=': "
                                     "write setting=value,value")
                key, values = part.split(sep, 1)
                value[key.strip()] = [v for v in values.replace("[", "").replace(
                    "]", "").split(",") if v.strip()]
    if not isinstance(value, Mapping):
        raise ValueError("ops_grid_params must be setting=values text or a dict of lists")
    out: Dict[str, List[Any]] = {}
    for key, values in value.items():
        key = str(key).strip()
        if key not in _GRID_KEYS:
            raise ValueError(f"ops_grid_params cannot vary {key!r}; it varies "
                             f"{sorted(_GRID_KEYS)}")
        if not isinstance(values, (list, tuple)):
            values = [values]
        cast: List[Any] = []
        for item in values:
            item = _cast(key, item)
            if item not in cast:
                cast.append(item)
        if not cast:
            raise ValueError(f"ops_grid_params: {key} has no values")
        out[key] = cast
    return out


def _tile_key(plate: str, well_id: str, site: int) -> str:
    """A tile's name in a selection: ``<plate>:<well id>:<site>``."""
    return f"{plate}:{well_id}:{int(site)}"


def _parse_tiles(value: Any, index: pd.DataFrame) -> List[Tuple[str, str, str, int]]:
    """A user's or a saved tile selection, checked against the layout.

    :param value: a list or comma-separated text of tiles, each
        ``plate:well:site``, ``well:site`` or ``site`` (the plate or well
        may be left out when the layout has one).
    :param index: the layout's sites (``plate``, ``well``, ``well_id``,
        ``site``).
    :returns: ``(plate, well, well_id, site)`` per tile, in the order given.
    :raises ValueError: for a tile that is ambiguous or not in the layout.
    """
    parts = value if isinstance(value, (list, tuple)) else str(value).split(",")
    sites = index[["plate", "well", "well_id", "site"]].drop_duplicates()
    out: List[Tuple[str, str, str, int]] = []
    for part in (str(p).strip() for p in parts):
        if not part:
            continue
        bits = part.split(":")
        try:
            site = int(bits[-1])
        except ValueError:
            raise ValueError(f"ops_grid_tiles: {part!r} does not end in a site number") from None
        hit = sites[sites["site"] == site]
        if len(bits) >= 2:
            well = bits[-2]
            hit = hit[(hit["well_id"] == well) | (hit["well"] == well)]
        if len(bits) >= 3:
            hit = hit[hit["plate"] == bits[-3]]
        if len(hit) != 1:
            raise ValueError(f"ops_grid_tiles: {part!r} names "
                             f"{'no' if hit.empty else 'more than one'} site of the layout; "
                             "write plate:well:site")
        row = hit.iloc[0]
        key = (str(row["plate"]), str(row["well"]), str(row["well_id"]), int(row["site"]))
        if key not in out:
            out.append(key)
    if not out:
        raise ValueError("ops_grid_tiles names no tile")
    return out


def _pick_tiles(index: pd.DataFrame, *, n: int = _DEFAULT_TILES, mode: str = "stratified",
               seed: int = 0, tiles: Any = None) -> List[str]:
    """Choose the representative tiles of a grid search.

    :param index: the layout's sites, one row per site or per file, with
        ``plate``, ``well``, ``well_id`` and ``site``.
    :param n: how many tiles; at most the layout's sites.
    :param mode: ``'stratified'`` (the default) takes the wells in turn and a
        seeded random site of each; ``'edge_centre'`` takes, in each well
        in turn, alternately the site furthest from the well's centre and
        the site nearest it, with the sites of a well taken as a square
        raster in site order (a raster and a snake raster give the same
        distances); ``'user'`` takes ``tiles``.
    :param seed: the random seed of ``'stratified'`` and of the tie order
        of ``'edge_centre'``.
    :param tiles: the user's tiles (:func:`_parse_tiles` forms); with any
        mode a selection given here is used as it is.
    :returns: the tiles as ``plate:well_id:site``, in the order chosen.
    :raises ValueError: for an unknown mode, ``'user'`` without tiles, a
        tile not in the layout, or ``n`` below 1.
    """
    if mode not in _TILE_MODES:
        raise ValueError(f"ops_grid_tile_mode must be one of {list(_TILE_MODES)}; got {mode!r}")
    if tiles not in (None, "", []):
        return [_tile_key(p, w, s) for p, _well, w, s in _parse_tiles(tiles, index)]
    if mode == "user":
        raise ValueError("ops_grid_tile_mode 'user' needs ops_grid_tiles, the tiles to use "
                         "(plate:well:site, comma-separated)")
    if int(n) < 1:
        raise ValueError(f"ops_grid_n_tiles must be 1 or more; got {n}")
    sites = (index[["plate", "well_id", "site"]].drop_duplicates()
             .sort_values(["plate", "well_id", "site"]).reset_index(drop=True))
    rng = np.random.default_rng(int(seed))
    queues: List[List[Tuple[str, str, int]]] = []
    for (plate, well_id), group in sites.groupby(["plate", "well_id"], sort=True):
        numbers = [int(s) for s in group["site"]]
        if mode == "stratified":
            order = [numbers[i] for i in rng.permutation(len(numbers))]
        else:
            side = max(1, int(math.ceil(math.sqrt(len(numbers)))))
            centre = (side - 1) / 2.0
            ties = rng.permutation(len(numbers))
            dist = {s: (math.hypot(i // side - centre, i % side - centre), int(ties[i]))
                    for i, s in enumerate(numbers)}
            far = sorted(numbers, key=lambda s: (-dist[s][0], dist[s][1]))
            near = sorted(numbers, key=lambda s: dist[s])
            order, seen = [], set()
            for pair in zip(far, near):
                for s in pair:
                    if s not in seen:
                        seen.add(s)
                        order.append(s)
        queues.append([(str(plate), str(well_id), s) for s in order])
    picked: List[str] = []
    want = min(int(n), sum(len(q) for q in queues))
    while len(picked) < want:
        for queue in queues:
            if queue and len(picked) < want:
                picked.append(_tile_key(*queue.pop(0)))
    return picked


def _run_id(settings: Mapping[str, Any], grid: Mapping[str, List[Any]],
            tiles: Sequence[str]) -> str:
    """The grid run's ID: 12 hex digits of a hash of everything that sets its numbers."""
    from . import __version__

    record = {"grid": grid, "tiles": list(tiles), "spacr": __version__,
              "settings": {key: settings.get(key) for key in _ID_KEYS}}
    text = json.dumps(record, sort_keys=True, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def _grid_points(grid: Mapping[str, List[Any]]) -> List[Dict[str, Any]]:
    """Every combination of the grid, in order; one empty point for no grid."""
    keys = list(grid)
    return [dict(zip(keys, values)) for values in itertools.product(*(grid[k] for k in keys))]


def _default_points(settings: Mapping[str, Any], tiles: Sequence[Tuple[str, str, str, int]],
                    folder: str, say: Callable[[str], None]) -> Tuple[Dict[Tuple[str, str], Dict[str, Any]], Dict[str, Any]]:
    """Each well's default point: the entered settings, or the decode's own per-well choice.

    :param settings: the run's settings.
    :param tiles: the chosen tiles.
    :param folder: the grid run's folder; with tuning the decode runs on the
        chosen tiles under ``<folder>/default_run/<plate>_<well>``.
    :param say: progress lines.
    :returns: ``({(plate, well_id): point}, engine)``: each point holds every
        :data:`_GRID_KEYS` setting; ``engine`` holds the decode's
        ``site_summary.csv`` rows and seconds when it ran.
    """
    from .ops_prefix import _prefix_read_mode, prefix_settings, run_prefix_decode

    chosen = prefix_settings(settings)
    mode = _prefix_read_mode(settings, chosen)
    base = {"ops_read_threshold": chosen["ops_read_threshold"], "ops_read_mode": mode,
            "ops_prefix_min_read_quality": chosen["ops_prefix_min_read_quality"],
            "ops_prefix_max_distance": chosen["ops_prefix_max_distance"],
            "ops_quality_min_reads": chosen["ops_quality_min_reads"],
            "ops_prefix_crosstalk": chosen["ops_prefix_crosstalk"]}
    wells = sorted({(p, w, wid) for p, w, wid, _s in tiles})
    points = {(p, wid): dict(base) for p, _w, wid in wells}
    engine: Dict[str, Any] = {"ran": False, "rows": [], "seconds": 0.0}
    if chosen["ops_prefix_threshold_tuning"] != "per_well_decoy":
        return points, engine
    root = str(settings.get("dst_root") or "")
    for plate, well, well_id in wells:
        dst = os.path.join(folder, "default_run", f"{plate}_{well_id}")
        merged = os.path.join(root, "merged")
        os.makedirs(dst, exist_ok=True)
        link = os.path.join(dst, "merged")
        if root and os.path.isdir(merged) and not os.path.exists(link):
            os.symlink(os.path.abspath(merged), link)
        sites = sorted(s for p, w, _wid, s in tiles if (p, w) == (plate, well))
        started = time.perf_counter()
        out = run_prefix_decode({**settings, "dst_root": dst, "cpg_wells": [well],
                                 "cpg_sites": sites}, report=say)
        engine["seconds"] += time.perf_counter() - started
        engine["ran"] = True
        tuned = out.get("tuning", {}).get(f"{plate}_{well_id}")
        if not tuned:
            raise ValueError(f"the decode did not tune {plate} {well_id} on the grid's tiles")
        best = tuned["chosen"]
        points[(plate, well_id)].update({
            "ops_read_threshold": float(best["ops_read_threshold"]),
            "ops_prefix_min_read_quality": float(best["ops_prefix_min_read_quality"]),
            "ops_prefix_max_distance": int(best["ops_prefix_max_distance"]),
            "ops_read_mode": str(tuned.get("read_mode", "log" if mode == "auto" else mode))})
        summary = pd.read_csv(os.path.join(out["folder"], "site_summary.csv"))
        engine["rows"].extend(summary[(summary["plate"].astype(str) == plate)
                                      & (summary["well"].astype(str) == well_id)]
                              .to_dict("records"))
    return points, engine


def _libraries(settings: Mapping[str, Any]) -> Dict[str, Any]:
    """The whitelist with its decoys, the control and the shuffled whitelists, as the decode builds them."""
    from .ops_decode import SHUFFLE_SEEDS
    from .ops_prefix import (prefix_decoy_library, prefix_nearmiss_decoys, prefix_settings,
                             shuffled_library, spacer_prefix_library)

    chosen = prefix_settings(settings)
    library = spacer_prefix_library(
        chosen["ops_library"], sequence_col=chosen["ops_library_sequence_col"],
        gene_col=chosen["ops_library_gene_col"], prefix_length=chosen["ops_prefix_length"],
        n_cycles=chosen["ops_sbs_cycles"], cycle_mask=chosen["ops_cycle_mask"],
        collision=chosen["ops_prefix_collision"])
    matching = (prefix_decoy_library(library, chosen["ops_prefix_decoy_k"],
                                     chosen["ops_prefix_decoy_seed"])
                if chosen["ops_prefix_decoy_k"] > 0 else library)
    if chosen["ops_prefix_nearmiss_decoy_k"] > 0:
        matching = prefix_nearmiss_decoys(matching, chosen["ops_prefix_nearmiss_decoy_k"],
                                          chosen["ops_prefix_decoy_seed"])
    control = (shuffled_library(library, chosen["ops_control_seed"])
               if chosen["ops_random_barcode_control"] else None)
    shuffles = [shuffled_library(library, chosen["ops_control_seed"] + s)
                for s in range(SHUFFLE_SEEDS)]
    return {"library": library, "matching": matching, "control": control,
            "shuffles": shuffles}


def _init_worker(libraries: Mapping[str, Any]) -> None:
    """Keep the libraries in a grid worker for every tile it scores."""
    _LIBRARIES.clear()
    _LIBRARIES.update(libraries)


def _score_tile(task: Mapping[str, Any]) -> Dict[str, Any]:
    """Find one tile's spots once and score every grid point on them.

    :param task: ``settings``, ``index`` (the tile's files), ``plate``,
        ``well``, ``well_id``, ``site``, ``points`` (the full settings of
        each grid point for this tile's well, default first) and ``loosest``
        (the largest mismatch limit of any point).
    :returns: ``rows`` (one per point, the :func:`spacr.ops_prefix._site_row`
        counts plus spots, mapped reads and the shuffle-called cells),
        ``detect_s`` and ``eval_s`` (one per point).
    """
    from .cpg_cropped import load_cpg_site, merged_path
    from .ops_prefix import (DECOY_GENE, NEARMISS_GENE, QUALITY_CLASSES,
                             RECIPE_SIMPLE_CLASSES, _at_setting, _called_counts,
                             _labelled_cells, _raw_candidates, _shuffle_frame, _site_row,
                             _site_tuning_reads, _spot_candidates, match_prefix_reads,
                             prefix_settings, site_labels)

    settings = task["settings"]
    chosen = prefix_settings(settings)
    plate, well, well_id, site = task["plate"], task["well"], task["well_id"], task["site"]
    points = task["points"]
    matching, control = _LIBRARIES["matching"], _LIBRARIES["control"]
    shuffles = _LIBRARIES["shuffles"]
    method = chosen["ops_prefix_quality_method"]
    class_names = RECIPE_SIMPLE_CLASSES if method == "recipe_simple" else QUALITY_CLASSES
    pinned = task["pinned"]
    started = time.perf_counter()
    loaded = load_cpg_site(task["index"], plate, well, site, sbs_cycles=task["cycles"],
                           sbs_bases=pinned["cpg_sbs_bases"], sbs_dapi=None, parts=("sbs",))
    labels = site_labels(chosen["ops_prefix_masks"], plate=plate, well=well, well_id=well_id,
                         site=site, merged=merged_path(str(settings.get("dst_root") or ""),
                                                       plate, well_id, site))
    stack = loaded["sbs"]
    if labels is not None and np.asarray(labels).shape != stack.shape[1:3]:
        raise ValueError(f"the cell labels are {np.asarray(labels).shape}, the sequencing "
                         f"stack {stack.shape[1:3]}")
    floor = min(p["ops_read_threshold"] for p in points)
    peaks, strength, values = _spot_candidates(stack, floor=floor)
    cell = (np.asarray(labels)[peaks[:, 0], peaks[:, 1]].astype(np.int64)
            if labels is not None and len(peaks) else np.zeros(len(peaks), np.int64))
    held = {"log": (peaks, strength, values, cell)}
    if any(p["ops_read_mode"] == "raw" for p in points):
        held["raw"] = _raw_candidates(stack, held["log"])
    del loaded, stack
    labelled = labels is not None
    n_labelled = _labelled_cells(labels, pd.DataFrame())
    detect_s = time.perf_counter() - started
    memo: Dict[str, tuple] = {}
    shuffle_memo: Dict[Tuple[int, int], Dict[str, tuple]] = {}
    frames: Dict[Tuple[str, float, str], pd.DataFrame] = {}
    rows, eval_s = [], []
    metric = chosen["ops_prefix_metric"]
    for point in points:
        began = time.perf_counter()
        here = {**chosen, **{k: point[k] for k in ("ops_quality_min_reads",)}}
        key = (point["ops_read_mode"], float(point["ops_read_threshold"]),
               point["ops_prefix_crosstalk"])
        if key not in frames:
            frames[key] = _site_tuning_reads(
                held[key[0]], matching, memo, bases=pinned["cpg_sbs_bases"],
                threshold=key[1], crosstalk=key[2], metric=metric,
                max_distance=task["loosest"], read_mode=key[0])
        frame = frames[key]
        distance = int(point["ops_prefix_max_distance"])
        reads = _at_setting(frame, float(point["ops_prefix_min_read_quality"]),
                            distance).reset_index(drop=True)
        if control is not None:
            reads["control_status"] = match_prefix_reads(
                reads["read"].tolist(), control, metric=metric,
                max_distance=distance)["status"].to_numpy()
        called = []
        for s, shuffled_lib in enumerate(shuffles):
            store = shuffle_memo.setdefault((s, distance), {})
            called.append(_called_counts(_shuffle_frame(
                reads, shuffled_lib, store, metric=metric, max_distance=distance),
                method)[0])
        shuffled = float(np.mean(called)) if called else 0.0
        reads.insert(0, "site", int(site))
        reads.insert(0, "well", well_id)
        reads.insert(0, "plate", plate)
        row, _cells = _site_row(
            reads, plate=plate, well_id=well_id, site=site, chosen=here, method=method,
            class_names=class_names, labelled=labelled, n_labelled=n_labelled,
            decoy_k=chosen["ops_prefix_decoy_k"], nearmiss_k=chosen["ops_prefix_nearmiss_decoy_k"],
            scale=chosen["ops_prefix_fdr_scale"],
            shuffled=shuffled if chosen["ops_prefix_fdr_calibration"] == "shuffle" else None)
        hit = reads["status"].isin(("exact", "corrected")) & ~reads["gene"].isin(
            (DECOY_GENE, NEARMISS_GENE))
        row.update({"spots": int(len(frame)), "mapped_reads": int(hit.sum()),
                    "shuffle_called": shuffled})
        if chosen["ops_prefix_decoy_k"] < 1:
            row.setdefault("cells_in_labels", int(n_labelled or row["cells_with_reads"]))
        rows.append(row)
        eval_s.append(time.perf_counter() - began)
    return {"rows": rows, "detect_s": detect_s, "eval_s": eval_s}


def _summarise(rows: Sequence[Mapping[str, Any]], *, k: int, scale: float) -> Dict[str, Any]:
    """The counts and rates of some tiles at one grid point.

    :param rows: the tiles' rows.
    :param k: decoys per prefix.
    :param scale: the decoy FDR scale.
    :returns: the summed counts, ``mapping_pct`` (reads matched to a guide,
        not a decoy, per 100 reads), ``assignment_pct`` (cells called to a
        guide per 100 cells), ``decoy_fdr_pct`` and ``shuffle_rate_pct``.
    """
    from .ops_decode import fdr

    out = {key: float(sum(r.get(key, 0) or 0 for r in rows)) for key in _COUNTS}
    for key in _COUNTS:
        if key != "shuffle_called":
            out[key] = int(out[key])
    cells = max(out["cells_in_labels"], 1)
    out["mapping_pct"] = 100.0 * out["mapped_reads"] / max(out["reads"], 1)
    out["assignment_pct"] = 100.0 * out["cells_real"] / cells
    out["decoy_fdr_pct"] = fdr(out["cells_real"], out["cells_decoy"], k, scale) if k > 0 \
        else float("nan")
    out["shuffle_rate_pct"] = 100.0 * out["shuffle_called"] / cells
    return out


def _pareto(yield_pct: Sequence[float], false_pct: Sequence[float]) -> List[bool]:
    """Which points no other point beats on both yield and false-call rate.

    :param yield_pct: each point's cell assignment rate.
    :param false_pct: each point's shuffle call rate.
    :returns: one flag per point; a point that assigns no cell is never on
        the front, and of identical points only the first is.
    """
    from .ops_decode import _pareto as shared

    flags = shared(list(false_pct), [round(float(v), 9) for v in yield_pct])
    return [bool(flag and float(y) > 0) for flag, y in zip(flags, yield_pct)]


def _plot(table: pd.DataFrame, path: str, title: str) -> str:
    """Draw the pooled points: yield against the shuffle call rate, with the default and the Pareto front."""
    import matplotlib

    matplotlib.use("Agg", force=False)
    from matplotlib.figure import Figure

    figure = Figure(figsize=(5.2, 4.0), dpi=120)
    ax = figure.add_subplot(1, 1, 1)
    ax.scatter(table["shuffle_rate_pct"], table["assignment_pct"], s=22, color="#2a6f97",
               alpha=0.6, label="grid point")
    front = table[table["pareto"]].sort_values("shuffle_rate_pct")
    ax.plot(front["shuffle_rate_pct"], front["assignment_pct"], color="#16324f", lw=1.2,
            marker="o", ms=4, label="Pareto front")
    default = table[table["is_default"]]
    ax.scatter(default["shuffle_rate_pct"], default["assignment_pct"], s=70, marker="*",
               color="#2ec4b6", edgecolor="#16324f", zorder=5, label="current default")
    ax.set_xlabel("Barcode-shuffle call rate (% of cells)")
    ax.set_ylabel("Cells assigned to a guide (%)")
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    figure.tight_layout()
    figure.savefig(path, metadata={"Software": None})
    return path


def _run_grid_search(settings: Mapping[str, Any], *, settings_path: Optional[str] = None,
                    report: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
    """Score a grid of decode settings on a few representative tiles (item 960, alpha).

    :param settings: the OPS settings of a spacer-prefix run on the
        cropped-site layout (``ops_layout='cpg_cropped'``), with
        ``ops_grid_params`` (the grid, :func:`_parse_grid`; empty scores the
        default point alone), ``ops_grid_tiles`` (a saved or user selection),
        ``ops_grid_tile_mode``, ``ops_grid_n_tiles`` and ``ops_grid_seed``
        (:func:`_pick_tiles`); ``n_workers`` tiles are scored at once, fewer
        when free RAM holds fewer (the shared worker guard,
        :func:`spacr.resource_log._guard_workers`).
    :param settings_path: the settings file; the outputs go beside it, else
        to ``<dst_root>/settings``.
    :param report: called with progress lines.
    :returns: ``run_id``, ``tiles``, ``grid``, ``table`` (the pooled rows,
        one per point, default first), ``defaults`` (each well's default
        point), ``default_matches_decode`` (with tuning: whether the
        default point's counts equal the decode's own on every tile, else
        None), the paths ``csv``, ``timing_csv``, ``plot`` and ``json``,
        and ``seconds``.

    Writes ``ops_grid_<run id>.csv``: one row per point and tile and one
    pooled row per point (``tile`` ``'pooled'``), with the point's
    settings, ``spots`` (spots above the read threshold), ``reads`` (after
    the read-quality floor), ``mapped_reads``, ``mapping_pct``,
    ``cells_in_labels``, ``cells_real``, ``cells_decoy``,
    ``assignment_pct``, ``decoy_fdr_pct``, ``shuffle_called``,
    ``shuffle_rate_pct`` (mean of three shuffled whitelists), ``is_default``
    and, on pooled rows, ``pareto``. The same settings write the same file.
    ``ops_grid_<run id>_timing.csv`` holds each tile's spot-finding seconds
    and each point's scoring seconds; ``ops_grid_<run id>.png`` plots the
    pooled points; ``ops_grid_<run id>.json`` records the run.
    :raises ValueError: for a layout other than the cropped sites, a library
        mode other than ``spacer_prefix``, or a bad grid or tile selection.
    """
    from concurrent.futures import ProcessPoolExecutor

    from .cpg_cropped import cpg_site_index
    from .ops_prefix import prefix_settings

    say = report or (lambda text: None)
    started = time.perf_counter()
    if str(settings.get("ops_layout") or "tiles") != "cpg_cropped" or str(
            settings.get("ops_library_mode") or "barcode") != "spacer_prefix":
        raise ValueError("the OPS grid search scores the spacer-prefix decode of the "
                         "cropped-site layout: set ops_layout 'cpg_cropped' and "
                         "ops_library_mode 'spacer_prefix'")
    if not settings.get("dst_root"):
        raise ValueError("the OPS grid search needs dst_root")
    grid = _parse_grid(settings.get("ops_grid_params"))
    chosen = prefix_settings(settings)
    built = cpg_site_index(settings, md5=False)
    index, pinned = built["index"], built["pinned"]
    mode = str(settings.get("ops_grid_tile_mode") or _TILE_MODES[0]).strip().lower()
    seed = int(settings.get("ops_grid_seed") or 0)
    n_tiles = int(settings.get("ops_grid_n_tiles") or _DEFAULT_TILES)
    keys = _pick_tiles(index, n=n_tiles, mode=mode, seed=seed,
                      tiles=settings.get("ops_grid_tiles"))
    tiles = _parse_tiles(keys, index)
    run_id = _run_id(settings, grid, keys)
    folder = (os.path.dirname(os.path.abspath(settings_path)) if settings_path
              else os.path.join(str(settings["dst_root"]), "settings"))
    os.makedirs(folder, exist_ok=True)
    work = os.path.join(str(settings["dst_root"]), "results", "ops_grid_search", run_id)
    os.makedirs(work, exist_ok=True)
    say(f"grid search {run_id}: the default and {sum(1 for p in _grid_points(grid) if p)} grid "
        f"points on {len(tiles)} tiles ({', '.join(keys)})")
    defaults, engine = _default_points(settings, tiles, work, say)
    points_of = {}
    for well_key, base in defaults.items():
        points_of[well_key] = [dict(base)] + [{**base, **p} for p in _grid_points(grid) if p]
    loosest = max(int(p["ops_prefix_max_distance"]) for pts in points_of.values() for p in pts)
    libraries = _libraries(settings)
    cycles = int(max(libraries["library"].positions))
    tasks = [{"settings": dict(settings), "pinned": pinned, "cycles": cycles,
              "index": index[(index["plate"] == p) & (index["well"] == w)
                             & (index["site"] == s)],
              "plate": p, "well": w, "well_id": wid, "site": s,
              "points": points_of[(p, wid)], "loosest": loosest} for p, w, wid, s in tiles]
    workers = max(1, min(int(settings.get("n_workers") or 1), len(tasks)))
    if workers > 1:
        from .resource_log import _array_file_nbytes, _guard_workers

        first = tasks[0]["index"]
        paths = list(first["path"]) if "path" in first.columns else []
        workers = max(1, int(_guard_workers(
            "ops_decode", workers,
            len(paths) * _array_file_nbytes(paths[0]) if paths else 0,
            settings=settings)))
    if workers == 1:
        _init_worker(libraries)
        scored = [_score_tile(task) for task in tasks]
    else:
        import multiprocessing

        with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker,
                                 initargs=(libraries,),
                                 mp_context=multiprocessing.get_context("spawn")) as pool:
            scored = list(pool.map(_score_tile, tasks))
    k, scale = chosen["ops_prefix_decoy_k"], chosen["ops_prefix_fdr_scale"]
    n_points = 1 + sum(1 for p in _grid_points(grid) if p)
    grid_keys = list(_GRID_KEYS)
    lines, timing, pooled = [], [], []
    for i in range(n_points):
        tile_rows = []
        for key, (p, _w, wid, s), result in zip(keys, tiles, scored):
            row = result["rows"][i]
            point = points_of[(p, wid)][i]
            summary = _summarise([row], k=k, scale=scale)
            tile_rows.append(row)
            lines.append({"run_id": run_id, "point": i, "tile": key,
                          "is_default": all(points_of[w][i] == points_of[w][0] for w in points_of),
                          **{g: point[g] for g in grid_keys}, **summary})
            timing.append({"run_id": run_id, "point": i, "tile": key,
                           "detect_s": round(result["detect_s"], 3),
                           "eval_s": round(result["eval_s"][i], 4)})
        varied = points_of[next(iter(points_of))][i]
        same = {g: (varied[g] if all(points_of[w][i][g] == varied[g] for w in points_of)
                    else "per well") for g in grid_keys}
        pooled.append({"run_id": run_id, "point": i, "tile": "pooled",
                       "is_default": all(points_of[w][i] == points_of[w][0] for w in points_of),
                       **same, **_summarise(tile_rows, k=k, scale=scale)})
    table = pd.DataFrame(pooled)
    table["pareto"] = _pareto(table["assignment_pct"].tolist(),
                              table["shuffle_rate_pct"].tolist())
    frame = pd.concat([table, pd.DataFrame(lines)], ignore_index=True)
    stem = os.path.join(folder, f"ops_grid_{run_id}")
    frame.to_csv(f"{stem}.csv", index=False, float_format="%.10g")
    pd.DataFrame(timing).to_csv(f"{stem}_timing.csv", index=False)
    plot = _plot(table, f"{stem}.png", f"OPS grid {run_id}: {len(keys)} tiles")
    matches = None
    if engine["ran"]:
        mine = {(str(r["plate"]), str(r["well"]), int(r["site"])): r
                for r in (result["rows"][0] for result in scored)}
        matches = True
        for row in engine["rows"]:
            ours = mine.get((str(row["plate"]), str(row["well"]), int(row["site"])))
            if ours is None or any(int(row[c]) != int(ours[c]) for c in (
                    "reads", "exact", "corrected", "cells_in_labels", "cells_real",
                    "cells_decoy") if c in row):
                matches = False
    record = {"run_id": run_id, "tiles": keys, "tile_mode": mode, "seed": seed,
              "grid": grid, "points": n_points,
              "defaults": {f"{p}:{w}": v for (p, w), v in defaults.items()},
              "default_matches_decode": matches, "workers": workers,
              "decode_seconds": round(engine["seconds"], 2),
              "seconds": round(time.perf_counter() - started, 2)}
    with open(f"{stem}.json", "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=1, default=str)
    best = table.sort_values(["pareto", "assignment_pct"], ascending=False).iloc[0]
    say(f"grid search {run_id}: default {table.iloc[0]['assignment_pct']:.2f} % assigned at "
        f"shuffle {table.iloc[0]['shuffle_rate_pct']:.3f} %; best on the front "
        f"{best['assignment_pct']:.2f} % at {best['shuffle_rate_pct']:.3f} % (point "
        f"{int(best['point'])}); {f'{stem}.csv'}")
    return {**record, "table": table, "csv": f"{stem}.csv", "timing_csv": f"{stem}_timing.csv",
            "plot": plot, "json": f"{stem}.json"}


def _apply_grid_choice(settings: Mapping[str, Any], result: Any, point: int) -> Dict[str, Any]:
    """Write one grid point's values into the run settings.

    :param settings: the run settings.
    :param result: :func:`_run_grid_search`'s result, or the path of its CSV.
    :param point: the point to apply (``point`` in the table; 0 is the
        default).
    :returns: a copy of the settings with the point's values. A point other
        than the default fixes its values, so the per-well tuning is
        switched off (``ops_prefix_threshold_tuning='none'``) and the read
        mode is entered; a column the point leaves to each well keeps the
        setting. ``ops_grid_run_id``, ``ops_grid_tiles`` and
        ``ops_grid_seed`` record the grid run, and ``ops_grid_search`` is
        switched off so the next run decodes.
    :raises ValueError: for a point the grid run does not have.
    """
    if isinstance(result, (str, os.PathLike)):
        table = pd.read_csv(result)
        table = table[table["tile"] == "pooled"]
        tiles = [t for t in pd.read_csv(result)["tile"].unique() if t != "pooled"]
        record = {"run_id": str(table["run_id"].iloc[0]), "tiles": tiles, "seed": None}
        sibling = os.path.splitext(str(result))[0] + ".json"
        if os.path.isfile(sibling):
            with open(sibling, encoding="utf-8") as handle:
                record.update(json.load(handle))
    else:
        table, record = result["table"], result
    hit = table[table["point"].astype(int) == int(point)]
    if hit.empty:
        raise ValueError(f"grid run {record['run_id']} has no point {point}")
    row = hit.iloc[0]
    out = dict(settings)
    if int(point) != 0:
        for key in _GRID_KEYS:
            value = row[key]
            if isinstance(value, str) and value == "per well":
                continue
            out[key] = _cast(key, value)
        out["ops_prefix_threshold_tuning"] = "none"
    out.update({"ops_grid_run_id": record["run_id"], "ops_grid_tiles": ",".join(record["tiles"]),
                "ops_grid_search": False})
    if record.get("seed") is not None:
        out["ops_grid_seed"] = int(record["seed"])
    return out
