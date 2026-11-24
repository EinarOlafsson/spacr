"""In situ sequencing modes of the spacer-prefix decode: combinatorial barcodes, peak-prioritised calling and read compartments (item 961, alpha).

Pooled screens read with PerturbView, Zombie or T7 in situ sequencing
differ in three ways from a screen that reads the guide itself, as the Funk
et al. and PERISCOPE screens do, and each is a setting here.

* A cell's identity can be a combination of barcode segments read in
  different cycles (``ops_barcode_layout='combinatorial'``). A layout file
  names each segment's cycles, the library column it is read against and its
  role: ``map`` segments identify the construct, ``recomb`` segments must
  pair with the map segment's construct (a read whose segments belong to
  different constructs is flagged ``recombined`` and not used for the call),
  and ``umi`` segments are only recorded.
* One bright amplicon per cell is better called by peak intensity than by
  read count (``ops_call_sort='peak'``; ``'count'`` and ``'quality'`` rank
  by reads or by mean read quality).
* Amplicons that stay in the nucleus are better assigned to the nucleus
  (``ops_read_compartment='nucleus'``; ``'cytoplasm'`` keeps the reads
  outside it), from the nucleus labels of ``ops_prefix_nuclei_masks``.

``ops_insitu_preset='t7'`` sets the T7 combination Brieflow validated: reads
assigned to the nucleus, peak-prioritised calling and one read per cell.

Every segment is matched with the spacer-prefix decode's own matcher, so
there is one matcher. The combinatorial decode reports, per segment, the
mapping rate, the decoy FDR of the map segment and the barcode-shuffle call
rate (every segment's whitelist shuffled together), and per cell the unique
combination rate and the recombination rate. With every setting at its
default the spacer-prefix decode is unchanged.
"""
from __future__ import annotations

import json
import os
import time
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

#: The barcode layouts, the first being the default.
_LAYOUTS = ("single", "combinatorial")

#: The read compartments, the first being the default.
_COMPARTMENTS = ("cell", "nucleus", "cytoplasm")

#: The segment roles of a combinatorial layout.
_ROLES = ("map", "recomb", "umi")

#: The named in situ presets, applied to the keys a run leaves empty.
_PRESETS: Dict[str, Dict[str, Any]] = {
    "t7": {"ops_read_compartment": "nucleus", "ops_call_sort": "peak",
           "ops_quality_min_reads": 1},
}

#: How many shuffled whitelists the combinatorial shuffle control uses.
_SHUFFLE_SEEDS = 3


def _choice(settings: Mapping[str, Any], key: str, allowed: Sequence[str],
            default: Optional[str]) -> Optional[str]:
    """One choice setting, checked; empty gives ``default``."""
    value = settings.get(key)
    if value is None or (isinstance(value, str) and value.strip().lower() in ("", "none")):
        return default
    value = str(value).strip().lower()
    if value not in allowed:
        raise ValueError(f"{key} must be one of {list(allowed)}; got {value!r}")
    return value


def _insitu_options(settings: Mapping[str, Any], chosen: Mapping[str, Any]) -> Dict[str, Any]:
    """The item-961 settings of a run, checked, with the preset applied.

    :param settings: the OPS settings.
    :param chosen: :func:`spacr.ops_prefix.prefix_settings` of the run.
    :returns: ``layout``, ``layout_file``, ``call_sort`` (None for the
        decode's own rule), ``compartment``, ``nuclei_masks``,
        ``recomb_min_quality``, ``min_reads`` (None unless the preset sets
        it), ``preset`` and ``recorded`` (the non-default values, for the
        run record; empty with every default).
    :raises ValueError: for an unknown value, a preset key the run set to
        something else, a combinatorial layout without its file, or a
        call sort with ``ops_prefix_quality_method`` other than ``'spacr'``.
    """
    preset = _choice(settings, "ops_insitu_preset", ("none",) + tuple(_PRESETS), "none")
    values = dict(_PRESETS.get(preset, {}))
    layout = _choice(settings, "ops_barcode_layout", _LAYOUTS, "single")
    from .ops_prefix import _CALL_SORTS

    call_sort = _choice(settings, "ops_call_sort", _CALL_SORTS, values.get("ops_call_sort"))
    compartment = (_choice(settings, "ops_read_compartment", _COMPARTMENTS,
                           values.get("ops_read_compartment")) or "cell")
    q = settings.get("ops_recomb_min_quality")
    recomb_q = 0.0 if q is None or q == "" else float(q)
    min_reads = None
    if "ops_quality_min_reads" in values and int(chosen["ops_quality_min_reads"]) == 2:
        min_reads = int(values["ops_quality_min_reads"])
    layout_file = settings.get("ops_barcode_layout_file") or None
    if layout == "combinatorial" and not layout_file:
        raise ValueError("ops_barcode_layout 'combinatorial' needs ops_barcode_layout_file, "
                         "the table of segments (segment, start, end, role, column)")
    if call_sort is not None and chosen["ops_prefix_quality_method"] != "spacr":
        raise ValueError("ops_call_sort ranks barcodes under ops_prefix_quality_method "
                         "'spacr'; 'recipe_simple' keeps the recipe's own ranking")
    recorded = {key: value for key, value in (
        ("ops_insitu_preset", preset if preset != "none" else None),
        ("ops_barcode_layout", layout if layout != "single" else None),
        ("ops_barcode_layout_file", layout_file),
        ("ops_call_sort", call_sort),
        ("ops_read_compartment", compartment if compartment != "cell" else None),
        ("ops_recomb_min_quality", recomb_q if recomb_q > 0 else None),
        ("ops_quality_min_reads", min_reads)) if value is not None}
    return {"layout": layout, "layout_file": layout_file, "call_sort": call_sort,
            "compartment": compartment,
            "nuclei_masks": settings.get("ops_prefix_nuclei_masks") or None,
            "recomb_min_quality": recomb_q, "min_reads": min_reads, "preset": preset,
            "recorded": recorded}


def _nuclei_labels(options: Mapping[str, Any], *, plate: str, well: str, well_id: str,
                   site: int, destination: str) -> np.ndarray:
    """One site's nucleus labels, from ``ops_prefix_nuclei_masks`` or the merged site file.

    :raises ValueError: when neither holds nucleus labels.
    """
    from .cpg_cropped import merged_path
    from .ops_prefix import _site_nuclei

    found = _site_nuclei(options["nuclei_masks"], plate=plate, well=well, well_id=well_id,
                         site=site, merged=merged_path(destination, plate, well_id, site))
    if found is None:
        raise ValueError(f"ops_read_compartment needs nucleus labels for {plate} {well_id} "
                         f"site {site}: set ops_prefix_nuclei_masks (a path with {{site}}) "
                         "or give the merged site file a 'nuclei' array")
    return np.asarray(found)


def _compartment_labels(labels: Optional[np.ndarray], options: Mapping[str, Any], *,
                        plate: str, well: str, well_id: str, site: int,
                        destination: str) -> Optional[np.ndarray]:
    """The cell labels a read takes under ``ops_read_compartment``.

    :param labels: the site's cell labels, or None.
    :param options: from :func:`_insitu_options`.
    :param plate: the plate.
    :param well: the well folder name.
    :param well_id: the normalised well id.
    :param site: the site.
    :param destination: the run folder (for the merged site file).
    :returns: the labels unchanged for ``'cell'``; for ``'nucleus'`` each
        cell's label only where a nucleus lies, for ``'cytoplasm'`` only
        where none does; None stays None.
    :raises ValueError: when the nucleus labels are missing or not the cell
        labels' grid.
    """
    if labels is None or options["compartment"] == "cell":
        return labels
    nuclei = _nuclei_labels(options, plate=plate, well=well, well_id=well_id, site=site,
                            destination=destination)
    labels = np.asarray(labels)
    if nuclei.shape != labels.shape:
        raise ValueError(f"the nucleus labels are {nuclei.shape}, the cell labels "
                         f"{labels.shape}")
    inside = nuclei > 0
    keep = inside if options["compartment"] == "nucleus" else ~inside
    return np.where(keep, labels, 0).astype(labels.dtype)


def _load_layout(path: str) -> List[Dict[str, Any]]:
    """The segments of a combinatorial layout file (CSV, TSV or JSON).

    :param path: the file; each segment has ``segment``, ``start`` and
        ``end`` (1-based cycles, inclusive), ``role`` (:data:`_ROLES`),
        ``column`` (the library column holding the segment's bases; not
        needed for ``umi``) and optionally ``max_distance`` (default 1) and
        ``metric`` (default ``hamming``).
    :returns: the segments in file order.
    :raises ValueError: for a missing field, overlapping cycles, no ``map``
        segment, or an unknown role or metric.
    """
    if str(path).lower().endswith(".json"):
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
        rows = data.get("segments", data) if isinstance(data, dict) else data
    else:
        from .tabular import read_table

        rows = read_table(str(path), canonicalise=False, report=None).to_dict("records")
    segments, used = [], set()
    for row in rows:
        try:
            seg = {"segment": str(row["segment"]), "start": int(row["start"]),
                   "end": int(row["end"]), "role": str(row["role"]).strip().lower()}
        except KeyError as error:
            raise ValueError(f"ops_barcode_layout_file: a segment has no {error}") from None
        if seg["role"] not in _ROLES:
            raise ValueError(f"ops_barcode_layout_file: role must be one of {list(_ROLES)}; "
                             f"got {seg['role']!r}")
        if not 1 <= seg["start"] <= seg["end"]:
            raise ValueError(f"ops_barcode_layout_file: segment {seg['segment']} has cycles "
                             f"{seg['start']}-{seg['end']}")
        cycles = set(range(seg["start"], seg["end"] + 1))
        if cycles & used:
            raise ValueError(f"ops_barcode_layout_file: segment {seg['segment']} overlaps "
                             "another segment's cycles")
        used |= cycles
        column = row.get("column")
        if seg["role"] != "umi" and (column is None or str(column) in ("", "nan")):
            raise ValueError(f"ops_barcode_layout_file: segment {seg['segment']} needs the "
                             "library column it is read against")
        distance = row.get("max_distance")
        metric = row.get("metric")
        seg.update({"column": None if column is None else str(column),
                    "max_distance": 1 if distance in (None, "") or pd.isna(distance)
                    else int(distance),
                    "metric": "hamming" if metric in (None, "") or pd.isna(metric)
                    else str(metric).strip().lower()})
        if seg["metric"] not in ("hamming", "levenshtein"):
            raise ValueError(f"ops_barcode_layout_file: metric must be hamming or "
                             f"levenshtein; got {seg['metric']!r}")
        segments.append(seg)
    if not any(seg["role"] == "map" for seg in segments):
        raise ValueError("ops_barcode_layout_file needs at least one 'map' segment")
    return segments


def _constructs(table: pd.DataFrame, segments: Sequence[Mapping[str, Any]],
                gene_col: str) -> pd.DataFrame:
    """The library's constructs: each row's segment sequences, name and gene.

    :raises ValueError: for a missing column or a sequence of the wrong length.
    """
    out = pd.DataFrame(index=table.index)
    for seg in segments:
        if seg["role"] == "umi":
            continue
        if seg["column"] not in table.columns:
            raise ValueError(f"the library has no column {seg['column']!r} for segment "
                             f"{seg['segment']}; columns are {list(table.columns)[:20]}")
        seq = table[seg["column"]].astype(str).str.strip().str.upper()
        width = seg["end"] - seg["start"] + 1
        bad = seq[seq.str.len() != width]
        if len(bad):
            raise ValueError(f"segment {seg['segment']} reads {width} cycles but the library "
                             f"holds {bad.iloc[0]!r}")
        out[seg["segment"]] = seq
    if gene_col not in table.columns:
        raise ValueError(f"the library has no column {gene_col!r}")
    out["gene"] = table[gene_col].astype(str)
    named = [s["segment"] for s in segments if s["role"] in ("map", "recomb")]
    out["construct"] = (table["construct"].astype(str) if "construct" in table.columns
                        else out[named].agg("+".join, axis=1))
    return out.drop_duplicates("construct").reset_index(drop=True)


def _segment_library(sequences: Sequence[str], seg: Mapping[str, Any]):
    """One segment's whitelist, built by the spacer-prefix library builder.

    Each sequence is placed at the segment's cycles and the cycles before it
    are masked, so the decode's own matcher reads exactly those cycles.
    """
    from .ops_prefix import spacer_prefix_library

    unique = sorted(set(sequences))
    pad = "N" * (seg["start"] - 1)
    frame = pd.DataFrame({"sgRNA": [pad + s for s in unique], "gene_symbol": unique})
    return spacer_prefix_library(frame, prefix_length=seg["end"], n_cycles=seg["end"],
                                 cycle_mask=list(range(1, seg["start"])))


def _libraries(constructs: pd.DataFrame, segments: Sequence[Mapping[str, Any]], *,
               decoy_k: int, decoy_seed: int) -> Dict[str, Any]:
    """Every segment's whitelist (the map segments with decoys) and its key-to-construct index."""
    from .ops_prefix import prefix_decoy_library

    libs, index = {}, {}
    for seg in segments:
        if seg["role"] == "umi":
            continue
        name = seg["segment"]
        lib = _segment_library(constructs[name].tolist(), seg)
        if seg["role"] == "map" and decoy_k > 0:
            lib = prefix_decoy_library(lib, decoy_k, decoy_seed)
        libs[name] = lib
        table: Dict[str, set] = {}
        for key, construct in zip(constructs[name], constructs["construct"]):
            table.setdefault(key, set()).add(construct)
        index[name] = table
    return {"libs": libs, "index": index,
            "gene": dict(zip(constructs["construct"], constructs["gene"]))}


def _shuffled_constructs(constructs: pd.DataFrame, segments: Sequence[Mapping[str, Any]],
                         seed: int) -> pd.DataFrame:
    """The constructs with every segment's letters shuffled (the shuffle control).

    Each sequence keeps its length and base composition; the segments of one
    construct stay paired, so a shuffled read can only be called the way a
    real one is.
    """
    rng = np.random.default_rng(int(seed))
    out = constructs.copy()
    for seg in segments:
        if seg["role"] == "umi":
            continue
        out[seg["segment"]] = ["".join(np.array(list(s))[rng.permutation(len(s))])
                               for s in out[seg["segment"]]]
    out["construct"] = [f"shuffled{i}" for i in range(len(out))]
    out["gene"] = "control"
    return out


def _combinatorial_reads(calls: Sequence[str], margin: np.ndarray, frame: pd.DataFrame,
                         segments: Sequence[Mapping[str, Any]], built: Mapping[str, Any],
                         recomb_min_quality: float) -> pd.DataFrame:
    """Each read's segment calls, its construct and whether it recombined.

    :param calls: the reads, one letter per cycle from cycle 1.
    :param margin: ``(reads, cycles)`` per-cycle call margins.
    :param frame: one row per read (``y``, ``x``, ``cell``, ``quality``,
        ``peak``), extended and returned.
    :param segments: from :func:`_load_layout`.
    :param built: from :func:`_libraries`.
    :param recomb_min_quality: a recomb segment whose worst cycle margin is
        below this is not checked (Brieflow's ``recomb_q_thresh``).
    :returns: ``frame`` with, per segment, ``<segment>_status``,
        ``<segment>_key`` and ``<segment>_quality`` (``<segment>_seq`` for a
        UMI), then ``construct`` (None unless one construct fits),
        ``gene`` (:data:`spacr.ops_prefix.DECOY_GENE` for a decoy map key),
        ``recomb_checked`` and ``recombined``.
    """
    from .ops_prefix import DECOY_GENE, match_prefix_reads

    out = frame.reset_index(drop=True).copy()
    out["read"] = list(calls)
    n = len(out)
    margin = np.asarray(margin, np.float32).reshape(n, -1) if n else np.zeros((0, 1))
    candidates: List[Optional[set]] = [None] * n
    unmatched = np.zeros(n, bool)
    decoy = np.zeros(n, bool)
    checked = np.zeros(n, bool)
    recombined = np.zeros(n, bool)
    for seg in sorted(segments, key=lambda item: _ROLES.index(item["role"])):
        name = seg["segment"]
        lo, hi = seg["start"] - 1, seg["end"]
        part = margin[:, lo:hi] if n else np.zeros((0, 1))
        worst = np.where(np.isnan(part), np.inf, part).min(axis=1) if n else np.zeros(0)
        out[f"{name}_quality"] = np.where(np.isfinite(worst), worst, 0.0).astype(np.float32)
        if seg["role"] == "umi":
            out[f"{name}_seq"] = [str(c)[lo:hi] for c in calls]
            continue
        matched = match_prefix_reads(list(calls), built["libs"][name], metric=seg["metric"],
                                     max_distance=seg["max_distance"])
        hit = matched["status"].isin(("exact", "corrected")).to_numpy()
        out[f"{name}_status"] = matched["status"].to_numpy()
        out[f"{name}_key"] = matched["prefix"].to_numpy()
        is_decoy = (matched["gene"] == DECOY_GENE).to_numpy() & hit
        for i in range(n):
            key = matched["prefix"].iat[i]
            if seg["role"] == "map":
                if not hit[i]:
                    unmatched[i] = True
                elif is_decoy[i]:
                    decoy[i] = True
                else:
                    found = built["index"][name].get(key, set())
                    candidates[i] = set(found) if candidates[i] is None \
                        else candidates[i] & found
            elif hit[i] and out[f"{name}_quality"].iat[i] >= recomb_min_quality:
                checked[i] = True
                found = built["index"][name].get(key, set())
                if candidates[i] is not None:
                    pairs = candidates[i] & found
                    if not pairs:
                        recombined[i] = True
                    candidates[i] = pairs
    construct, gene = [], []
    for i in range(n):
        if decoy[i] and not unmatched[i]:
            construct.append(None)
            gene.append(DECOY_GENE)
        elif unmatched[i] or recombined[i] or not candidates[i] or len(candidates[i]) != 1:
            construct.append(None)
            gene.append(None)
        else:
            name = next(iter(candidates[i]))
            construct.append(name)
            gene.append(built["gene"][name])
    out["construct"] = construct
    out["gene"] = gene
    out["recomb_checked"] = checked & ~decoy
    out["recombined"] = recombined & ~decoy
    return out


def _combinatorial_cells(reads: pd.DataFrame, *, sort: str, min_reads: int,
                         keys: Sequence[str] = ("plate", "well", "site", "cell")) -> pd.DataFrame:
    """Each cell's construct, from its reads that fit one construct (or a decoy).

    :param reads: from :func:`_combinatorial_reads`, with ``keys``.
    :param sort: ``'count'``, ``'quality'`` or ``'peak'``: how a cell's
        candidates are ranked.
    :param min_reads: reads a ``perfect`` cell needs.
    :param keys: the columns naming a cell.
    :returns: one row per cell with a read: ``keys``, ``reads``,
        ``mapped`` (reads with a construct or a decoy), ``recomb_checked``,
        ``recombined``, ``candidates``, ``construct``, ``gene``, ``decoy``
        (the top candidate is a decoy), ``top_reads``, ``quality``
        (``perfect``, ``great``, ``ambiguous`` or ``none``) and ``called``
        (perfect or great, not a decoy).
    """
    from .ops_prefix import DECOY_GENE

    columns = list(keys) + ["reads", "mapped", "recomb_checked", "recombined", "candidates",
                            "construct", "gene", "decoy", "top_reads", "quality", "called"]
    frame = reads[reads["cell"] > 0] if len(reads) else reads
    if frame.empty:
        return pd.DataFrame(columns=columns)
    rows = []
    for cell_key, group in frame.groupby(list(keys), sort=True):
        label = group["construct"].where(group["construct"].notna(),
                                         group["gene"].where(group["gene"] == DECOY_GENE))
        usable = group[label.notna()].assign(label=label[label.notna()])
        score: Dict[str, float] = {}
        count: Dict[str, int] = {}
        for name, part in usable.groupby("label", sort=True):
            count[name] = int(len(part))
            score[name] = (float(len(part)) if sort == "count" else
                           float(part["quality"].mean()) if sort == "quality" else
                           float(part["peak"].max()))
        ranked = sorted(score, key=lambda k: (-score[k], -count[k], k))
        if not ranked:
            quality = "none"
        elif len(ranked) > 1 and score[ranked[0]] == score[ranked[1]]:
            quality = "ambiguous"
        elif len(ranked) == 1 and count[ranked[0]] >= int(min_reads):
            quality = "perfect"
        else:
            quality = "great"
        top = ranked[0] if ranked else None
        is_decoy = top == DECOY_GENE
        called = quality in ("perfect", "great") and not is_decoy
        rows.append({**dict(zip(keys, cell_key if isinstance(cell_key, tuple) else (cell_key,))),
                     "reads": int(len(group)), "mapped": int(len(usable)),
                     "recomb_checked": int(group["recomb_checked"].sum()),
                     "recombined": int(group["recombined"].sum()),
                     "candidates": len(ranked),
                     "construct": top if called else None,
                     "gene": group.loc[group["construct"] == top, "gene"].iloc[0]
                     if called else None,
                     "decoy": bool(is_decoy and quality in ("perfect", "great")),
                     "top_reads": count.get(top, 0) if top else 0,
                     "quality": quality, "called": bool(called)})
    return pd.DataFrame(rows, columns=columns)


def _segment_rates(reads: pd.DataFrame, segments: Sequence[Mapping[str, Any]],
                   decoy_k: int) -> Dict[str, Dict[str, Any]]:
    """Per segment: reads matched, the mapping rate and, for map segments, the read-level decoy FDR."""
    from .ops_decode import fdr
    from .ops_prefix import DECOY_GENE

    out = {}
    n = int(len(reads))
    for seg in segments:
        if seg["role"] == "umi":
            continue
        name = seg["segment"]
        status = reads[f"{name}_status"] if n else pd.Series([], dtype=object)
        hit = status.isin(("exact", "corrected"))
        row = {"role": seg["role"], "cycles": f"{seg['start']}-{seg['end']}",
               "reads": n, "matched": int(hit.sum()),
               "mapping_pct": 100.0 * float(hit.sum()) / n if n else 0.0}
        if seg["role"] == "map" and decoy_k > 0 and n:
            decoys = int((hit & (reads["gene"] == DECOY_GENE)).sum())
            real = int(hit.sum()) - decoys
            row.update({"decoy_reads": decoys, "read_decoy_fdr_pct": fdr(real, decoys,
                                                                        decoy_k, 1.0)})
        out[name] = row
    return out


def _run_combinatorial_decode(settings: Mapping[str, Any],
                              report: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
    """The ``prefix_decode`` phase under ``ops_barcode_layout='combinatorial'``.

    Each site's spots are found and called as the spacer-prefix decode
    finds and calls them (one read threshold, no tuning), every segment is
    matched with the decode's matcher, and each cell's construct is called
    from its reads. Writes, under ``<dst_root>/results/ops_insitu/``:
    ``reads_<plate>_<well>.parquet``, ``cells_<plate>_<well>.csv``,
    ``site_summary.csv`` and ``insitu_report.json`` (the layout, per well
    and per segment the mapping rate and read-level decoy FDR, the cell
    decoy FDR, the unique combination rate, the recombination rate and the
    barcode-shuffle call rate over :data:`_SHUFFLE_SEEDS` shuffled
    whitelists).

    :param settings: the OPS settings of a cropped-site run.
    :param report: called with progress lines.
    :returns: the output paths and the report.
    :raises ValueError: with per-well tuning or the auto read (they choose
        one key's thresholds), without ``dst_root``, or for a bad layout.
    """
    from .cpg_cropped import cpg_site_index, load_cpg_site, merged_path
    from .ops_decode import fdr
    from .ops_prefix import (_labelled_cells, _prefix_calls, _prefix_read_mode,
                             _raw_candidates, _spot_candidates, prefix_settings, site_labels)
    from .tabular import read_table, write_table

    say = report or (lambda text: None)
    started = time.perf_counter()
    destination = settings.get("dst_root")
    if not destination:
        raise ValueError("the prefix_decode phase needs dst_root")
    chosen = prefix_settings(settings)
    options = _insitu_options(settings, chosen)
    if chosen["ops_prefix_threshold_tuning"] != "none":
        raise ValueError("ops_barcode_layout 'combinatorial' decodes at one read threshold; "
                         "set ops_prefix_threshold_tuning 'none'")
    read_mode = _prefix_read_mode(settings, chosen)
    if read_mode == "auto":
        raise ValueError("ops_barcode_layout 'combinatorial' takes ops_read_mode 'log' or 'raw'")
    sort = options["call_sort"] or "count"
    min_reads = options["min_reads"] or int(chosen["ops_quality_min_reads"])
    segments = _load_layout(options["layout_file"])
    library = read_table(chosen["ops_library"], canonicalise=False, report=None)
    constructs = _constructs(library, segments, chosen["ops_library_gene_col"])
    decoy_k = chosen["ops_prefix_decoy_k"]
    built = _libraries(constructs, segments, decoy_k=decoy_k,
                       decoy_seed=chosen["ops_prefix_decoy_seed"])
    shuffled = [_libraries(_shuffled_constructs(constructs, segments,
                                                chosen["ops_control_seed"] + s),
                           segments, decoy_k=0, decoy_seed=0)
                for s in range(_SHUFFLE_SEEDS)]
    cycles = max(seg["end"] for seg in segments)
    index_built = cpg_site_index(settings, md5=False)
    pinned = index_built["pinned"]
    if cycles > pinned["cpg_sbs_cycles"]:
        raise ValueError(f"the layout reads cycles 1..{cycles}; cpg_sbs_cycles is "
                         f"{pinned['cpg_sbs_cycles']}")
    folder = os.path.join(str(destination), "results", "ops_insitu")
    os.makedirs(folder, exist_ok=True)
    rows, wells, paths = [], {}, {}
    for (plate, well, well_id), sites in index_built["index"].groupby(
            ["plate", "well", "well_id"], sort=True):
        site_reads, site_cells, shuffle_called = [], [], []
        for site in sorted(int(s) for s in sites["site"].unique()):
            loaded = load_cpg_site(index_built["index"], plate, well, site, sbs_cycles=cycles,
                                   sbs_bases=pinned["cpg_sbs_bases"], sbs_dapi=None,
                                   parts=("sbs",))
            stack = loaded["sbs"]
            labels = site_labels(chosen["ops_prefix_masks"], plate=plate, well=well,
                                 well_id=well_id, site=site,
                                 merged=merged_path(str(destination), plate, well_id, site))
            labels = _compartment_labels(labels, options, plate=plate, well=well,
                                         well_id=well_id, site=site,
                                         destination=str(destination))
            candidates = _spot_candidates(stack, floor=chosen["ops_read_threshold"])
            if read_mode == "raw":
                candidates = _raw_candidates(stack, candidates)
            peaks, strength, values = candidates
            del loaded, stack
            calls, quality, margin = _prefix_calls(values, bases=pinned["cpg_sbs_bases"],
                                                   crosstalk=chosen["ops_prefix_crosstalk"],
                                                   read_mode=read_mode, margins=True)
            cell = (np.asarray(labels)[peaks[:, 0], peaks[:, 1]].astype(np.int64)
                    if labels is not None and len(peaks) else np.zeros(len(peaks), np.int64))
            base = pd.DataFrame({"y": peaks[:, 0] if len(peaks) else np.zeros(0, np.int64),
                                 "x": peaks[:, 1] if len(peaks) else np.zeros(0, np.int64),
                                 "cell": cell, "quality": quality,
                                 "peak": np.asarray(strength, np.float32)})
            keep = base["quality"] >= chosen["ops_prefix_min_read_quality"]
            calls = [c for c, k in zip(calls, keep) if k]
            margin = np.asarray(margin)[keep.to_numpy()] if len(base) else margin
            base = base[keep].reset_index(drop=True)
            reads = _combinatorial_reads(calls, margin, base, segments, built,
                                         options["recomb_min_quality"])
            for name, value in (("site", int(site)), ("well", well_id), ("plate", plate)):
                reads.insert(0, name, value)
            cells = _combinatorial_cells(reads, sort=sort, min_reads=min_reads)
            n_cells = _labelled_cells(labels, cells) if labels is not None else int(len(cells))
            called = 0.0
            for lib in shuffled:
                s_reads = _combinatorial_reads(calls, margin, base, segments, lib,
                                               options["recomb_min_quality"])
                for name, value in (("site", int(site)), ("well", well_id), ("plate", plate)):
                    s_reads.insert(0, name, value)
                s_cells = _combinatorial_cells(s_reads, sort=sort, min_reads=min_reads)
                called += float(s_cells["called"].sum()) if len(s_cells) else 0.0
            called /= max(len(shuffled), 1)
            shuffle_called.append(called)
            real = int(cells["called"].sum()) if len(cells) else 0
            decoy = int(cells["decoy"].sum()) if len(cells) else 0
            rows.append({"plate": plate, "well": well_id, "site": int(site),
                         "reads": int(len(reads)), "cells_in_labels": int(n_cells),
                         "cells_with_reads": int(len(cells)), "cells_called": real,
                         "cells_decoy": decoy,
                         "recomb_checked": int(reads["recomb_checked"].sum()) if len(reads) else 0,
                         "recombined": int(reads["recombined"].sum()) if len(reads) else 0,
                         "shuffle_called": called})
            site_reads.append(reads)
            site_cells.append(cells)
            say(f"prefix_decode (combinatorial) {plate} {well_id} site {site}: {len(reads)} "
                f"reads, {real} cells called")
        reads_all = pd.concat(site_reads, ignore_index=True)
        cells_all = pd.concat(site_cells, ignore_index=True)
        stem = f"{plate}_{well_id}"
        paths[f"reads_{stem}"] = os.path.join(folder, f"reads_{stem}.parquet")
        paths[f"cells_{stem}"] = os.path.join(folder, f"cells_{stem}.csv")
        write_table(reads_all, paths[f"reads_{stem}"], canonicalise=False)
        write_table(cells_all, paths[f"cells_{stem}"], canonicalise=False)
        here = [r for r in rows if r["plate"] == plate and r["well"] == well_id]
        n_cells = sum(r["cells_in_labels"] for r in here)
        real = sum(r["cells_called"] for r in here)
        decoys = sum(r["cells_decoy"] for r in here)
        checked = sum(r["recomb_checked"] for r in here)
        wells[stem] = {
            "cells_in_labels": n_cells, "cells_called": real,
            "unique_combination_pct": 100.0 * real / max(n_cells, 1),
            "cell_decoy_fdr_pct": fdr(real, decoys, decoy_k, 1.0) if decoy_k > 0 else None,
            "recombination_pct": 100.0 * sum(r["recombined"] for r in here) / checked
            if checked else None,
            "shuffle_call_rate_pct": 100.0 * sum(r["shuffle_called"] for r in here)
            / max(n_cells, 1),
            "segments": _segment_rates(reads_all, segments, decoy_k)}
    summary = pd.DataFrame(rows)
    paths["site_summary"] = os.path.join(folder, "site_summary.csv")
    write_table(summary, paths["site_summary"], canonicalise=False)
    record = {"layout": segments, "settings": options["recorded"], "read_mode": read_mode,
              "call_sort": sort, "min_reads": min_reads, "wells": wells,
              "shuffle_seeds": [chosen["ops_control_seed"] + s for s in range(_SHUFFLE_SEEDS)],
              "seconds": round(time.perf_counter() - started, 1)}
    paths["report"] = os.path.join(folder, "insitu_report.json")
    with open(paths["report"], "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=1, default=str)
    for stem, row in wells.items():
        say(f"prefix_decode (combinatorial) {stem}: {row['unique_combination_pct']:.2f} % of "
            f"cells called, recombination {row['recombination_pct']}, shuffle "
            f"{row['shuffle_call_rate_pct']:.3f} %")
    return {"folder": folder, "paths": paths, "report": record, "layout": "combinatorial"}
