"""Decode rules, decoy FDR and decode QC for optical pooled screens (alpha).

Purpose and scope
=================
:func:`spacr.ops_engine.run_ops` decodes every field of a well. This module
holds the parts of that decode that the BMC decode-yield study added
(items 873, 876, 877 and 878), so that a settings run reproduces the study's
final rule rather than a script over a field cache:

* the field rule "rule (a) + S2" (:func:`field_calls`): library-calibrated
  base calls, territory attribution, quality-weighted votes, the runner-up
  one-mismatch rule and the pixel fallback, matched against the library and
  a decoy library;
* the decoy FDR report of every decode (:func:`fdr_report`,
  :func:`plate_fdr_summary`);
* the per-field evidence a field's calls were made from, in the cache layout
  the study used (:func:`field_evidence`), which the decode QC reads.

Every rule here is off unless its setting turns it on, so a decode that
asks for none of them is the decode it was before.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

#: The settings that reproduce "rule (a) + S2", the final decode rule of the
#: decode-yield study (2026-11-07): 74.43 % of plate A's nuclei assigned at a
#: decoy FDR of 0.065 per 100. ``ops_decode_preset: rule_a_s2`` applies them
#: to every key a settings file leaves at its default.
RULE_A_S2: Dict[str, Any] = {
    "ops_channel_registration": "spots_refine",
    "ops_registration_refine_px": 3,
    "ops_spot_min_peak": 20.0,
    "ops_spot_overrule_share": 0.5,
    "ops_read_threshold": 315.0,
    "ops_crosstalk": "argmax_median",
    "ops_decode_library_calibration": True,
    "ops_calibration_passes": 2,
    "ops_calibration_min_labels": 30,
    "ops_cell_rule": "mapped",
    "ops_read_mismatches": 0,
    "ops_decode_runner_up": True,
    "ops_runner_up_margin": 0.1,
    "ops_decode_impute_missing_cycles": True,
    "ops_assign_by_cell_territory": True,
    "ops_footprint": 10.0,
    "ops_tie": 1.0,
    "ops_decode_quality_weighted": True,
    "ops_vote_weight_floor": 0.05,
    "ops_min_fraction": 0.6,
    "ops_single_read_quality": 0.06,
    "ops_decode_pixel_fallback": True,
    "ops_pixel_percentile": 97.0,
    "ops_pixel_radius": 10.0,
    "ops_pixel_min_pixels": 2,
    "ops_pixel_min_fraction": 0.6,
    "ops_pixel_quality": 0.1,
    "ops_decoy_library_k": 4,
    "ops_fdr_scale": 1.268,
    "ops_decode_skip_unplaced_sites": True,
}

#: The presets ``ops_decode_preset`` may name.
DECODE_PRESETS: Dict[str, Dict[str, Any]] = {"rule_a_s2": RULE_A_S2}

#: The defaults of the decode keys this module reads, which are the
#: behaviour before item 873: every new rule off.
_DEFAULTS: Dict[str, Any] = {
    "ops_registration_refine_px": 3,
    "ops_spot_min_peak": 20.0,
    "ops_spot_overrule_share": 0.5,
    "ops_decode_library_calibration": False,
    "ops_calibration_passes": 2,
    "ops_calibration_min_labels": 30,
    "ops_decode_runner_up": False,
    "ops_runner_up_margin": 0.1,
    "ops_decode_impute_missing_cycles": True,
    "ops_assign_by_cell_territory": False,
    "ops_tie": 1.0,
    "ops_decode_quality_weighted": False,
    "ops_vote_weight_floor": 0.05,
    "ops_decode_pixel_fallback": False,
    "ops_pixel_percentile": 97.0,
    "ops_pixel_radius": 10.0,
    "ops_pixel_min_pixels": 2,
    "ops_pixel_min_fraction": 0.6,
    "ops_pixel_quality": 0.1,
    "ops_decoy_library_k": 0,
    "ops_fdr_scale": 1.268,
    "ops_decode_skip_unplaced_sites": True,
    "ops_store_field_cache": False,
}

#: How far from a nucleus boundary a fallback pixel is kept at all, in
#: pixels. The pixel calls are normalised over the pixels kept, so this
#: reach is part of the rule; the study's field cache kept 15 px.
_PIXEL_REACH = 15.0

#: The lowest spot strength the field cache keeps, so the read threshold
#: can be swept afterwards; the study's cache kept 100.
_CACHE_FLOOR = 100.0

#: The decode settings whose non-default value routes a well through
#: :func:`field_calls`.
_RULE_SWITCHES = ("ops_decode_library_calibration", "ops_decode_runner_up",
                  "ops_assign_by_cell_territory",
                  "ops_decode_quality_weighted", "ops_decode_pixel_fallback",
                  "ops_store_field_cache")

#: Matchers and library indices built once per process, keyed by the
#: library and decoy sets they hold.
_MATCHERS: Dict[Tuple[Any, int], Any] = {}


def resolve_settings(settings: Mapping[str, Any],
                     defaults: Optional[Mapping[str, Any]] = None
                     ) -> Dict[str, Any]:
    """The settings with ``ops_decode_preset`` applied.

    :param settings: the run's settings.
    :param defaults: the panel defaults, so a key a settings file carries at
        its default still takes the preset's value; None reads
        :data:`spacr.ops_settings.OPS_DEFAULTS`.
    :returns: a copy; unchanged when no preset is named. A key set to
        anything other than its default keeps that value.
    :raises ValueError: for an unknown preset.
    """
    out = dict(settings or {})
    name = str(out.get("ops_decode_preset") or "none").strip().lower()
    if name in ("", "none"):
        return out
    if name not in DECODE_PRESETS:
        raise ValueError(f"ops_decode_preset must be one of "
                         f"{['none'] + sorted(DECODE_PRESETS)}; got {name!r}")
    if defaults is None:
        from .ops_settings import OPS_DEFAULTS as defaults
    for key, value in DECODE_PRESETS[name].items():
        current = out.get(key)
        if current is None or current == "" or current == defaults.get(
                key, _DEFAULTS.get(key)):
            out[key] = value
    return out


def _number(settings: Mapping[str, Any], key: str, cast=float):
    """One numeric setting, its default when empty.

    :param settings: the settings.
    :param key: the key.
    :param cast: ``float`` or ``int``.
    :returns: the value.
    :raises ValueError: when it is not a number.
    """
    value = settings.get(key)
    if value is None or value == "":
        value = _DEFAULTS[key]
    try:
        return cast(float(value)) if cast is int else cast(value)
    except (TypeError, ValueError):
        raise ValueError(f"{key} must be a number; got {value!r}") from None


def _flag(settings: Mapping[str, Any], key: str) -> bool:
    """One boolean setting, its default when empty.

    :param settings: the settings.
    :param key: the key.
    :returns: the value; the strings ``false``, ``0``, ``no`` and ``off`` are
        False.
    """
    value = settings.get(key)
    if value is None or value == "":
        return bool(_DEFAULTS[key])
    if isinstance(value, str):
        return value.strip().lower() not in ("false", "0", "no", "off")
    return bool(value)


def decode_rule(settings: Mapping[str, Any]) -> Dict[str, Any]:
    """The field rule's parameters, read from resolved settings.

    :param settings: settings after :func:`resolve_settings`.
    :returns: the parameters :func:`field_calls` takes, plus ``active`` --
        True when any rule of this module is on, which routes the well
        through :func:`field_calls` -- and ``decoy_k``.
    :raises ValueError: for a negative count or a fraction outside (0, 1].
    """
    rule = {
        "calibration": _flag(settings, "ops_decode_library_calibration"),
        "calibration_passes": _number(settings, "ops_calibration_passes", int),
        "calibration_min_labels": _number(settings,
                                          "ops_calibration_min_labels", int),
        "runner_up": _flag(settings, "ops_decode_runner_up"),
        "runner_up_margin": _number(settings, "ops_runner_up_margin"),
        "impute": _flag(settings, "ops_decode_impute_missing_cycles"),
        "territory": _flag(settings, "ops_assign_by_cell_territory"),
        "tie": _number(settings, "ops_tie"),
        "quality_weighted": _flag(settings, "ops_decode_quality_weighted"),
        "weight_floor": _number(settings, "ops_vote_weight_floor"),
        "pixels": _flag(settings, "ops_decode_pixel_fallback"),
        "pixel_percentile": _number(settings, "ops_pixel_percentile"),
        "pixel_radius": _number(settings, "ops_pixel_radius"),
        "pixel_min_pixels": _number(settings, "ops_pixel_min_pixels", int),
        "pixel_min_fraction": _number(settings, "ops_pixel_min_fraction"),
        "pixel_quality": _number(settings, "ops_pixel_quality"),
        "decoy_k": _number(settings, "ops_decoy_library_k", int),
        "fdr_scale": _number(settings, "ops_fdr_scale"),
        "store_cache": _flag(settings, "ops_store_field_cache"),
    }
    if rule["decoy_k"] < 0 or rule["pixel_min_pixels"] < 1 \
            or rule["calibration_passes"] < 1:
        raise ValueError("ops_decoy_library_k must be 0 or more, and "
                         "ops_pixel_min_pixels and ops_calibration_passes "
                         "1 or more")
    if not 0 < rule["pixel_min_fraction"] <= 1:
        raise ValueError("ops_pixel_min_fraction must lie in (0, 1]; got "
                         f"{rule['pixel_min_fraction']}")
    rule["active"] = bool(rule["decoy_k"] > 0 or any(
        _flag(settings, key) for key in _RULE_SWITCHES))
    return rule


def matcher_for(library: Sequence[str], decoys: Sequence[str]):
    """The matcher over guides and decoys, built once per process.

    :param library: the guide barcodes.
    :param decoys: the decoy barcodes.
    :returns: ``(matcher, decoy_set)``.
    """
    from .ops_sbs import LibraryMatcher

    library = frozenset(library)
    decoys = frozenset(decoys)
    key = (hash(library), hash(decoys))
    found = _MATCHERS.get(key)
    if found is None:
        found = (LibraryMatcher(list(library) + list(decoys)), decoys)
        for stale in [k for k in _MATCHERS if k[0] != "masked"]:
            del _MATCHERS[stale]
        _MATCHERS[key] = found
    return found


def _masked_index(library: Sequence[str]) -> dict:
    """The library's masked index, built once per process.

    :param library: the guide barcodes.
    :returns: ``(position, barcode without it) -> barcodes``.
    """
    from .ops_sbs import _library_index

    key = ("masked", hash(frozenset(library)))
    found = _MATCHERS.get(key)
    if found is None:
        found = _library_index(library)[1]
        _MATCHERS[key] = found
    return found


def _call(values: np.ndarray, rule: Mapping[str, Any], library: Sequence[str],
          crosstalk: str):
    """Base calls for reads or pixels under the rule.

    :param values: ``(N, cycles, 4)`` intensities.
    :param rule: from :func:`decode_rule`.
    :param library: the guide barcodes.
    :param crosstalk: the unmixing method, ``none`` for none.
    :returns: ``(data, barcodes, margin, quality)``.
    """
    from .ops_sbs import called_bases, library_calibrated_bases

    if len(values) == 0:
        cycles = values.shape[1] if values.ndim == 3 else 0
        return (np.zeros((0, cycles, 4), np.float32), [],
                np.zeros((0, cycles), np.float32), np.zeros(0, np.float32))
    if rule.get("calibration") and library:
        data, codes, margin = library_calibrated_bases(
            values, library, passes=int(rule["calibration_passes"]),
            min_labels=int(rule["calibration_min_labels"]),
            masked=_masked_index(library))
    elif crosstalk and crosstalk != "none":
        data, codes, margin = called_bases(values, compensate=True,
                                           method=crosstalk, gpu=False)
    else:
        data, codes, margin = called_bases(values, gpu=False)
    worst = np.where(np.isnan(margin), np.inf, margin).min(1)
    quality = np.where(np.isfinite(worst), worst, 0.0)
    return data, codes, margin, quality


def field_evidence(stack: np.ndarray, filtered: np.ndarray,
                   score: np.ndarray, peaks: np.ndarray,
                   strength: np.ndarray, kept: Sequence[int],
                   task: Mapping[str, Any], *, edge: int,
                   floor: float, pixel_percentile: Optional[float] = None,
                   features: bool = False) -> Dict[str, np.ndarray]:
    """One field's decode evidence, in the study's field-cache layout.

    :param stack: ``(kept cycles, channels, H, W)``, the aligned stack.
    :param filtered: the stack after the -LoG(1 px) filter, clipped at 0.
    :param score: the cross-cycle read score.
    :param peaks: ``(N, 2)`` every local maximum of the score.
    :param strength: ``(N,)`` each peak's strength, its score over the local
        minimum.
    :param kept: the cycles the stack holds, in order.
    :param task: the field's task -- ``cycles``, ``centroids``, ``areas``,
        ``ids`` and ``owned``.
    :param edge: how many pixels from the field edge a peak must lie.
    :param floor: the weakest strength kept.
    :param pixel_percentile: keep the pixels above this percentile of the
        score for the pixel fallback; None keeps none.
    :param features: add the per-spot features the false-call model reads
        (item 876): raw intensity and local background per cycle and
        channel, score sharpness, local focus and spot density.
    :returns: ``peaks``, ``strength``, ``values`` (``(N, cycles, 4)``, NaN for
        a cycle not kept), ``gap``, ``id0``, ``id1``, ``owned0``,
        ``owned_ids``, ``owned_areas`` and ``cycles``, plus ``pix_*`` and
        ``f_*`` when asked.
    """
    from scipy import ndimage

    from .ops_sbs import extract_bases, nearest_boundaries

    cycles = list(task["cycles"])
    height, width = filtered.shape[-2:]
    peaks = np.asarray(peaks).reshape(-1, 2)
    strength = np.asarray(strength).reshape(-1)
    keep = (strength > floor) & (peaks[:, 0] >= edge) \
        & (peaks[:, 0] < height - edge) & (peaks[:, 1] >= edge) \
        & (peaks[:, 1] < width - edge)
    peaks, strength = peaks[keep], strength[keep]
    measured = extract_bases(filtered, peaks, window=1)
    n_channels = filtered.shape[1]
    values = np.full((len(peaks), len(cycles), n_channels), np.nan, np.float32)
    for i, cycle in enumerate(kept):
        values[:, cycles.index(cycle)] = measured[:, i]
    centres = np.asarray(task["centroids"], float).reshape(-1, 2)
    areas = np.asarray(task["areas"], float).reshape(-1)
    ids = np.asarray(task["ids"], np.int64).reshape(-1)
    owned = np.asarray(task["owned"], bool).reshape(-1)
    gap64, index = nearest_boundaries(peaks, centres, areas)
    gap = gap64.astype(np.float32)
    out: Dict[str, np.ndarray] = {
        "peaks": peaks.astype(np.int32), "strength": strength.astype(np.float32),
        "values": values, "gap": gap,
        "id0": np.where(index[:, 0] >= 0, ids[np.maximum(index[:, 0], 0)]
                        if len(ids) else -1, -1),
        "id1": np.where(index[:, 1] >= 0, ids[np.maximum(index[:, 1], 0)]
                        if len(ids) else -1, -1),
        "owned0": np.where(index[:, 0] >= 0, owned[np.maximum(index[:, 0], 0)]
                           if len(owned) else False, False),
        "owned_ids": ids[owned], "owned_areas": areas[owned],
        "cycles": np.asarray(cycles),
    }
    if features and len(peaks):
        py, px = peaks[:, 0], peaks[:, 1]
        raw = np.stack([stack[i][:, py, px].T for i in range(stack.shape[0])], 1)
        background = np.stack([np.stack([
            ndimage.minimum_filter(stack[i, k], 9)[py, px]
            for k in range(stack.shape[1])], 1) for i in range(stack.shape[0])], 1)
        ring = ndimage.uniform_filter(score, 5)[py, px]
        from .ops_cycles import sbs_mean_target

        target = sbs_mean_target(list(stack[0]))
        lap = ndimage.laplace(target.astype(np.float32)) ** 2
        focus = ndimage.uniform_filter(lap, 64)[py, px]
        from scipy.spatial import cKDTree

        density = np.array([len(x) - 1 for x in cKDTree(peaks).query_ball_point(
            peaks.astype(float), 10.0)])
        full_raw = np.full((len(peaks), len(cycles), raw.shape[-1]), np.nan,
                           np.float32)
        full_bg = np.full_like(full_raw, np.nan)
        for i, cycle in enumerate(kept):
            full_raw[:, cycles.index(cycle)] = raw[:, i]
            full_bg[:, cycles.index(cycle)] = background[:, i]
        out.update({"f_raw": full_raw, "f_bg": full_bg,
                    "f_sharp": (score[py, px] / np.maximum(ring, 1e-6)).astype(np.float32),
                    "f_focus": focus.astype(np.float32),
                    "f_density": density.astype(np.int16)})
    if pixel_percentile is not None and len(centres):
        ys, xs = np.nonzero(score > np.percentile(score, float(pixel_percentile)))
        inside = (ys >= edge) & (ys < height - edge) & (xs >= edge) & (xs < width - edge)
        ys, xs = ys[inside], xs[inside]
        pixel_values = np.full((len(ys), len(cycles), n_channels), np.nan,
                               np.float32)
        for i, cycle in enumerate(kept):
            pixel_values[:, cycles.index(cycle)] = filtered[i][:, ys, xs].T
        pixel_gap, pixel_index = nearest_boundaries(
            np.column_stack([ys, xs]), centres, areas)
        near = pixel_gap[:, 0] <= _PIXEL_REACH
        pixel_index = pixel_index[near]
        out.update({
            "pix_yx": np.column_stack([ys, xs]).astype(np.int16)[near],
            "pix_values": pixel_values[near],
            "pix_score": score[ys, xs].astype(np.float16)[near],
            "pix_gap": pixel_gap[near].astype(np.float16),
            "pix_id0": ids[pixel_index[:, 0]],
            "pix_id1": np.where(pixel_index[:, 1] >= 0,
                                ids[np.maximum(pixel_index[:, 1], 0)], -1),
            "pix_owned0": owned[pixel_index[:, 0]],
        })
    return out


def field_calls(evidence: Mapping[str, np.ndarray], rule: Mapping[str, Any],
                library: Sequence[str], matcher, decoys, *,
                threshold: float, footprint: float, max_distance: int,
                min_fraction: float, single_read_quality: float,
                crosstalk: str = "argmax_median",
                evidence_rows: bool = False) -> Dict[str, Any]:
    """Every call of one field under the rule, decoys included.

    :param evidence: from :func:`field_evidence`, or a field-cache ``.npz``
        of the decode-yield study opened with :func:`numpy.load`.
    :param rule: from :func:`decode_rule`.
    :param library: the guide barcodes.
    :param matcher: from :func:`matcher_for`.
    :param decoys: the decoy set from :func:`matcher_for`.
    :param threshold: the read threshold on the spot strength.
    :param footprint: how far beyond a boundary a read may lie, in pixels.
    :param max_distance: the mismatches a read may carry; the runner-up
        rule raises 0 to 1.
    :param min_fraction: the share of the votes the guide must hold.
    :param single_read_quality: the quality a one-read guide needs.
    :param crosstalk: the unmixing method.
    :param evidence_rows: also return ``evidence``, one row per call --
        decoys included -- with the read and pixel support behind it
        (:data:`EVIDENCE_COLUMNS`), which the genotype confidence reads.
    :returns: ``calls`` -- ``{object: {...}}`` with ``guide``, ``decoy``,
        ``source`` (``primary`` or ``pixel_fallback``), ``n_reads``,
        ``n_agreeing``, ``fraction``, ``quality``, ``barcode`` and
        ``n_pixels`` -- for owned objects only; ``nuclei``, the objects this
        field owns; ``assigned_real`` and ``assigned_decoy``; and read
        counts for the well report.
    """
    from collections import Counter

    from .ops_sbs import vote_mapped_reads, vote_pixels

    strength = np.asarray(evidence["strength"])
    selected = strength > threshold
    values = np.asarray(evidence["values"])[selected]
    data, codes, margin, quality = _call(values, rule, library, crosstalk)
    gap = np.asarray(evidence["gap"])[selected]
    first = np.asarray(evidence["id0"])[selected]
    owned_first = np.asarray(evidence["owned0"])[selected]
    near = gap[:, 0] <= footprint
    ambiguous = near & ((gap[:, 1] - gap[:, 0]) < rule["tie"])
    own = near & owned_first & (bool(rule["territory"]) | ~ambiguous)
    rows = np.flatnonzero(own)
    objects = first[rows]
    distance = max(int(max_distance), 1) if rule["runner_up"] else int(max_distance)
    tallies: Optional[dict] = {} if evidence_rows else None
    primary = vote_mapped_reads(
        objects, [codes[i] for i in rows], quality[rows], data[rows],
        margin[rows], matcher, max_distance=distance,
        runner_up_margin=(rule["runner_up_margin"] if rule["runner_up"] else None),
        quality_weighted=bool(rule["quality_weighted"]),
        weight_floor=float(rule["weight_floor"]), min_fraction=min_fraction,
        single_read_quality=single_read_quality,
        impute_missing=bool(rule["impute"]),
        strength=strength[selected][rows], evidence=tallies)
    reads_per = Counter(objects.tolist())
    pixel: Dict[int, dict] = {}
    if rule["pixels"] and "pix_values" in evidence and len(evidence["pix_values"]):
        _, pixel_codes, _, pixel_quality = _call(
            np.asarray(evidence["pix_values"], np.float32),
            {"calibration": False}, library, crosstalk)
        pixel_own = np.asarray(evidence["pix_owned0"]) & (
            np.asarray(evidence["pix_gap"])[:, 0].astype(float)
            <= rule["pixel_radius"])
        chosen = np.flatnonzero(pixel_own)
        pixel = vote_pixels(
            np.asarray(evidence["pix_id0"])[chosen],
            [pixel_codes[i] for i in chosen], pixel_quality[chosen], matcher,
            min_quality=rule["pixel_quality"],
            min_pixels=int(rule["pixel_min_pixels"]),
            min_fraction=rule["pixel_min_fraction"])
    owned = set(np.asarray(evidence["owned_ids"]).tolist())
    calls: Dict[int, dict] = {}
    for owner, hit in pixel.items():
        calls[owner] = {"guide": hit["guide"], "source": "pixel_fallback",
                        "n_agreeing": hit["n_pixels"], "fraction": hit["fraction"],
                        "quality": float("nan"), "barcode": hit["barcode"],
                        "n_pixels": hit["n_pixels"]}
    for owner, hit in primary.items():
        calls[owner] = {"guide": hit["guide"], "source": "primary",
                        "n_agreeing": hit["n_support"], "fraction": hit["fraction"],
                        "quality": hit["quality"], "barcode": hit["barcode"],
                        "n_pixels": pixel[owner]["n_pixels"] if owner in pixel else 0,
                        "n_exact": hit["n_exact"]}
    real = decoy = 0
    kept_calls: Dict[int, dict] = {}
    for owner, hit in calls.items():
        if owner not in owned:
            continue
        hit["decoy"] = hit["guide"] in decoys
        hit["n_reads"] = int(reads_per.get(owner, 0))
        decoy += int(hit["decoy"])
        real += int(not hit["decoy"])
        kept_calls[owner] = hit
    out = {"calls": kept_calls, "nuclei": len(owned), "assigned_real": real,
           "assigned_decoy": decoy, "reads": int(selected.sum()),
           "reads_owned": int(len(rows)), "reads_near": int(near.sum()),
           "reads_ambiguous": int(ambiguous.sum())}
    if evidence_rows:
        out["evidence"] = _evidence_rows(primary, pixel, tallies, owned, decoys,
                                         objects, ambiguous[rows])
    return out


#: The columns of :func:`field_calls`'s evidence rows, after ``well`` and
#: ``site``: the per-call read support of the decode-yield study's
#: ``decode_diag.py``.
EVIDENCE_COLUMNS = ("object_id", "mapped_guide", "is_decoy", "source",
                    "n_reads", "n_matched", "n_guides_seen", "n_exact",
                    "n_corr", "vote_top", "vote_total", "vote_frac",
                    "runner_margin", "q_best", "q_mean", "strength_max",
                    "n_amb", "pix_n", "pix_total", "pix_frac",
                    "field_reads_per_nucleus", "field_nuclei")


def _evidence_rows(primary, pixel, tallies, owned, decoys, objects,
                   ambiguous) -> List[Dict[str, Any]]:
    """One evidence row per owned call, primary first where both exist.

    :param primary: from :func:`spacr.ops_sbs.vote_mapped_reads`.
    :param pixel: from :func:`spacr.ops_sbs.vote_pixels`.
    :param tallies: the vote evidence the primary vote filled.
    :param owned: the objects this field owns.
    :param decoys: the decoy set.
    :param objects: the object of every owned read.
    :param ambiguous: whether each owned read lay near two nuclei.
    :returns: the rows, in the study's order (the set of primary and pixel
        objects).
    """
    from collections import Counter

    reads = Counter(objects.tolist())
    near_two = Counter(objects[ambiguous].tolist())
    density = len(objects) / max(1, len(owned))
    votes, exact, corrected = tallies["votes"], tallies["exact"], tallies["corrected"]
    best, qualities, strength = tallies["best"], tallies["qualities"], tallies["strength"]
    rows = []
    for owner in set(primary) | set(pixel):
        if owner not in owned:
            continue
        if owner in primary:
            hit = primary[owner]
            guide, top, total = hit["guide"], hit["vote_top"], hit["vote_total"]
            margin, seen, source = hit["runner_margin"], hit["n_guides"], "primary"
        else:
            hit = pixel[owner]
            guide, top, total, margin = hit["guide"], 0.0, 0.0, hit["runner_margin"]
            seen = len(votes.get(owner, {}))
            source = "pixel_fallback"
        fallback = pixel.get(owner)
        has_guide = owner in best and guide in best[owner]
        rows.append({
            "object_id": int(owner), "mapped_guide": guide,
            "is_decoy": guide in decoys, "source": source,
            "n_reads": int(reads.get(owner, 0)),
            "n_matched": int(tallies["matched"].get(owner, 0)),
            "n_guides_seen": int(seen),
            "n_exact": int(exact[owner].get(guide, 0)) if owner in exact else 0,
            "n_corr": int(corrected[owner].get(guide, 0)) if owner in corrected else 0,
            "vote_top": float(top), "vote_total": float(total),
            "vote_frac": float(top / total) if total else np.nan,
            "runner_margin": float(margin),
            "q_best": float(best[owner][guide]) if has_guide else np.nan,
            "q_mean": (float(np.mean(qualities[owner][guide]))
                       if owner in qualities and guide in qualities[owner] else np.nan),
            "strength_max": (float(strength[owner][guide])
                             if owner in strength and guide in strength[owner] else np.nan),
            "n_amb": int(near_two.get(owner, 0)),
            "pix_n": int(fallback["n_pixels"]) if fallback and fallback["guide"] == guide else 0,
            "pix_total": int(fallback["n_matched"]) if fallback else 0,
            "pix_frac": (float(fallback["n_pixels"] / fallback["n_matched"])
                         if fallback and fallback["guide"] == guide else 0.0),
            "field_reads_per_nucleus": float(density), "field_nuclei": len(owned),
        })
    return rows


def fdr(real: int, decoy: int, k: float, scale: float) -> float:
    """The decoy-estimated false calls per 100 real calls.

    :param real: real calls.
    :param decoy: decoy calls.
    :param k: decoys per guide.
    :param scale: the calibration of a decoy match to a wrong real match
        (1.268 for the Funk et al. 2022 library); never below 1, so a decoy
        shortfall is never credited.
    :returns: ``100 * max(scale, 1) / k * decoy / max(real, 1)``; NaN without
        decoys.
    """
    if not k:
        return float("nan")
    return 100.0 * max(float(scale), 1.0) / float(k) * decoy / max(int(real), 1)


def fdr_report(field_stats, *, plate: str, well: str, k: int, scale: float,
               skipped_sites: Sequence[int] = ()) -> Dict[str, Any]:
    """One well's decoy FDR report.

    :param field_stats: one row per decoded field with ``nuclei``,
        ``assigned_real`` and ``assigned_decoy``.
    :param plate: the plate.
    :param well: the well.
    :param k: decoys per guide.
    :param scale: the FDR calibration (:func:`fdr`).
    :param skipped_sites: the sites not decoded because the stitch did not
        place them.
    :returns: the report: nuclei, assigned, assigned %, decoy calls, FDR per
        100, and the skipped sites.
    """
    nuclei = int(field_stats["nuclei"].sum()) if len(field_stats) else 0
    real = int(field_stats["assigned_real"].sum()) if len(field_stats) else 0
    decoy = int(field_stats["assigned_decoy"].sum()) if len(field_stats) else 0
    return {"plate": plate, "well": well, "fields": int(len(field_stats)),
            "nuclei": nuclei, "assigned": real, "assigned_decoy": decoy,
            "pct": round(100 * real / max(nuclei, 1), 2),
            "fdr_per_100": round(fdr(real, decoy, k, scale), 3) if k else None,
            "decoy_library_k": int(k), "fdr_scale": float(scale),
            "skipped_sites": sorted(int(s) for s in skipped_sites)}


def plate_fdr_summary(folder: str, *, plate: str, k: int, scale: float,
                      reps: int = 1000, seed: int = 0) -> Dict[str, Any]:
    """The plate's assigned % and FDR, from every well's field stats.

    :param folder: the ``results/ops_decode`` folder holding
        ``field_stats_<well>.csv`` files.
    :param plate: the plate.
    :param k: decoys per guide.
    :param scale: the FDR calibration.
    :param reps: field-bootstrap replicates for the FDR's 95 % interval.
    :param seed: the bootstrap seed.
    :returns: the summary, with per-well rows; empty wells when no stats
        were written.
    """
    import pandas as pd

    from .tabular import read_table

    names = sorted(name for name in os.listdir(folder)
                   if name.startswith("field_stats_") and name.endswith(".csv"))
    if not names:
        return {"plate": plate, "wells": {}}
    stats = pd.concat([read_table(os.path.join(folder, name), canonicalise=False,
                                  report=None) for name in names], ignore_index=True)
    real = stats["assigned_real"].to_numpy()
    decoy = stats["assigned_decoy"].to_numpy()
    rng = np.random.default_rng(seed)
    boot = [fdr(real[i].sum(), decoy[i].sum(), k, scale)
            for i in (rng.integers(0, len(stats), len(stats))
                      for _ in range(int(reps)))] if k else []
    summary = {
        "plate": plate, "nuclei_in_decoded_fields": int(stats["nuclei"].sum()),
        "assigned": int(real.sum()), "assigned_decoy": int(decoy.sum()),
        "pct_of_decoded_field_nuclei": round(100 * real.sum() / max(stats["nuclei"].sum(), 1), 2),
        "fdr_per_100": round(fdr(real.sum(), decoy.sum(), k, scale), 3) if k else None,
        "fdr_ci95": ([round(float(np.percentile(boot, 2.5)), 3),
                      round(float(np.percentile(boot, 97.5)), 3)] if boot else None),
        "per_well": {},
    }
    for well, group in stats.groupby("well"):
        summary["per_well"][str(well)] = {
            "pct": round(100 * group["assigned_real"].sum() / max(group["nuclei"].sum(), 1), 2),
            "fdr_per_100": (round(fdr(group["assigned_real"].sum(),
                                      group["assigned_decoy"].sum(), k, scale), 3)
                            if k else None)}
    return summary


def results_folder(db: str, name: str = "ops_decode") -> str:
    """The folder a decode writes its reports into, created.

    :param db: the measurements database; its folder is ``dst_root``.
    :param name: the subfolder of ``results``.
    :returns: ``<dst_root>/results/<name>``.
    """
    folder = os.path.join(os.path.dirname(os.path.abspath(db)), "results", name)
    os.makedirs(folder, exist_ok=True)
    return folder


def write_well_reports(db: str, plate: str, well: str, decoded: Sequence[Mapping[str, Any]],
                       rule: Mapping[str, Any], skipped_sites: Sequence[int]
                       ) -> Dict[str, Any]:
    """Write one well's field stats, FDR report and the plate summary.

    :param db: the measurements database.
    :param plate: the plate.
    :param well: the well.
    :param decoded: the decoded fields, each with ``site`` and ``rule_a``
        from :func:`field_calls`.
    :param rule: from :func:`decode_rule`.
    :param skipped_sites: the unplaced sites.
    :returns: the well's FDR report, with the paths written under ``paths``.
    """
    import pandas as pd

    from .tabular import write_table

    folder = results_folder(db)
    stats = pd.DataFrame([{"plate": plate, "well": well, "site": int(r["site"]),
                           "nuclei": int(r["rule_a"]["nuclei"]),
                           "assigned_real": int(r["rule_a"]["assigned_real"]),
                           "assigned_decoy": int(r["rule_a"]["assigned_decoy"])}
                          for r in sorted(decoded, key=lambda r: r["site"])],
                         columns=["plate", "well", "site", "nuclei",
                                  "assigned_real", "assigned_decoy"])
    stats_path = os.path.join(folder, f"field_stats_{well}.csv")
    write_table(stats, stats_path, canonicalise=False)
    report = fdr_report(stats, plate=plate, well=well, k=int(rule["decoy_k"]),
                        scale=float(rule["fdr_scale"]),
                        skipped_sites=skipped_sites)
    report_path = os.path.join(folder, f"decode_fdr_{well}.json")
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=1)
    summary = plate_fdr_summary(folder, plate=plate, k=int(rule["decoy_k"]),
                                scale=float(rule["fdr_scale"]))
    summary_path = os.path.join(folder, f"decode_fdr_{plate}.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=1)
    report["paths"] = {"field_stats": stats_path, "fdr": report_path,
                       "plate_summary": summary_path}
    return report


def barcode_rows(plate: str, well: str, decoded: Sequence[Mapping[str, Any]]
                 ) -> List[Dict[str, Any]]:
    """The ``ops_barcodes`` rows of a well decoded by :func:`field_calls`.

    Decoy calls are counted for the FDR and never stored.

    :param plate: the plate.
    :param well: the well.
    :param decoded: the decoded fields, each with ``rule_a`` and
        ``n_cycles``.
    :returns: one row per real call, in object order.
    """
    rows = []
    for result in decoded:
        for object_id, hit in result["rule_a"]["calls"].items():
            if hit["decoy"]:
                continue
            rows.append({"plate": plate, "well": well, "object_id": int(object_id),
                         "barcode": hit["barcode"], "quality": hit["quality"],
                         "n_reads": hit["n_reads"], "n_agreeing": hit["n_agreeing"],
                         "fraction": hit["fraction"],
                         "n_cycles": int(result["n_cycles"]),
                         "mapped_guide": hit["guide"]})
    return sorted(rows, key=lambda row: row["object_id"])


def save_field_cache(folder: str, site: int, evidence: Mapping[str, np.ndarray]) -> str:
    """Write one field's evidence as ``site<NNNN>.npz``.

    :param folder: the well's cache folder, created.
    :param site: the site.
    :param evidence: from :func:`field_evidence`.
    :returns: the path written.
    """
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"site{int(site):04d}.npz")
    np.savez_compressed(path, **evidence)
    return path


def unplaced_sites(cycle_files: Mapping[int, Mapping[int, Any]],
                   placements: Mapping[int, Any]) -> List[int]:
    """The sites with tiles that the stitch did not place.

    :param cycle_files: ``cycle -> site -> channel -> path``.
    :param placements: ``site -> (y, x)``.
    :returns: the unplaced sites, sorted.
    """
    sites = {int(site) for files in cycle_files.values() for site in files}
    return sorted(site for site in sites if site not in placements)

