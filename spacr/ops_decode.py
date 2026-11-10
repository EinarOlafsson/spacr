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
    "ops_decode_qc": False,
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
                  "ops_store_field_cache", "ops_decode_qc")

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
        "store_cache": (_flag(settings, "ops_store_field_cache")
                        or _flag(settings, "ops_decode_qc")),
        "features": _flag(settings, "ops_decode_qc"),
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



#: The features the genotype confidence reads off each call's evidence
#: (item 877), in the order the reference model saw them.
CONFIDENCE_FEATURES = ("log_reads", "n_matched", "n_exact", "n_corr",
                       "vote_frac", "runner_margin", "q_best", "q_mean",
                       "log_strength", "n_amb", "pixel", "pix_n", "pix_frac",
                       "n_guides_seen", "field_reads_per_nucleus",
                       "field_nuclei", "q_best_missing", "vote_frac_missing")

#: The models :func:`genotype_confidence` fits; the setting names them.
CONFIDENCE_MODELS = {"logistic": "LR", "isotonic": "ISO", "hist_gbm": "HGB",
                     "strata": "STRATA"}

#: The seed of the reference confidence run (2026-11-07).
CONFIDENCE_SEED = 20261107


def _confidence_table(calls, folds: int = 5):
    """The evidence table with the confidence model's derived columns.

    :param calls: rows with :data:`EVIDENCE_COLUMNS` plus ``well`` and
        ``site``.
    :param folds: how many folds the field key is split into.
    :returns: a copy with ``fold``, ``pixel``, ``unanimous``, the ``_missing``
        flags, the logs, and the NaN fields filled with 0.
    """
    import hashlib

    d = calls.copy()
    d["fold"] = [int(hashlib.md5(f"{w}|{s}".encode()).hexdigest()[:8], 16) % int(folds)
                 for w, s in zip(d["well"], d["site"])]
    d["pixel"] = (d["source"] == "pixel_fallback").astype(float)
    d["unanimous"] = (d["runner_margin"] >= 0.999).astype(float)
    for column in ("q_best", "q_mean", "strength_max", "vote_frac"):
        d[f"{column}_missing"] = d[column].isna().astype(float)
        d[column] = d[column].fillna(0.0)
    d["log_reads"] = np.log1p(d["n_reads"])
    d["log_strength"] = np.log1p(d["strength_max"])
    return d


class _Logistic:
    """Logistic regression on standardised features, the reference's LR."""

    def __init__(self):
        """Build the scaler and the model."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        self.scaler = StandardScaler()
        self.model = LogisticRegression(C=1.0, max_iter=2000)

    def fit(self, x, y, sample_weight=None):
        """Fit on ``x``.

        :param x: the features.
        :param y: the decoy labels.
        :param sample_weight: per-row weights.
        :returns: self.
        """
        self.scaler.fit(x)
        self.model.fit(self.scaler.transform(x), y, sample_weight=sample_weight)
        return self

    def predict_proba(self, x):
        """Class probabilities.

        :param x: the features.
        :returns: ``(N, 2)``.
        """
        return self.model.predict_proba(self.scaler.transform(x))


def genotype_confidence(calls, *, model: str = "hist_gbm",
                        calibration: str = "target_decoy", bins: int = 10,
                        seed: int = CONFIDENCE_SEED, fdr_scale: float = 1.268,
                        decoy_k: int = 4, folds: int = 5,
                        max_real: int = 1_000_000) -> Dict[str, Any]:
    """Each call's calibrated probability of being correct (item 877).

    A call to a decoy is a known false call. Out of fold, by sequencing
    field (five folds keyed on the field's name, so a field is never scored
    by a model that saw it), four models estimate P(decoy | evidence) from
    each call's read and pixel support (:data:`CONFIDENCE_FEATURES`):
    logistic regression on the features and their squares, an isotonic
    recalibration of that score, histogram gradient boosting, and the decoy
    rate of matched strata. A real call's error is then
    ``c * P(decoy) / P(real)``, ``c = fdr_scale / decoy_k``, and
    ``genotype_confidence`` is one minus it. A control fits the logistic
    model on decoy labels shuffled within fields; its AUC should be 0.5.

    :param calls: one row per call, decoys included, with ``well``, ``site``
        and :data:`EVIDENCE_COLUMNS`, in decode order.
    :param model: which model's error becomes the confidence: ``hist_gbm``,
        ``logistic``, ``isotonic``, ``strata``, or ``auto`` for the lowest
        held-out log-loss.
    :param calibration: ``target_decoy`` converts P(decoy) to the error of a
        real call as above; ``none`` reports ``1 - P(decoy)``.
    :param bins: reliability bins, each expecting an equal number of decoys.
    :param seed: the subsampling, shuffling and boosting seed.
    :param fdr_scale: the decoy calibration (1.268 for the Funk library).
    :param decoy_k: decoys per guide.
    :param folds: cross-fitting folds, keyed on the md5 of ``well|site``.
    :param max_real: the most real calls each fit sees, weighted back.
    :returns: ``table`` (the calls with ``e_LR``, ``e_ISO``, ``e_HGB``,
        ``e_STRATA``, ``e`` and ``genotype_confidence``), ``metrics`` (per
        model AUC, log-loss, Brier and error counts, the chosen model),
        ``reliability`` (bin, predicted, observed, n per model) and
        ``strata``.
    :raises ValueError: for an unknown model or calibration, or calls with
        no decoy.
    """
    import pandas as pd
    from scipy.stats import chi2
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.isotonic import IsotonicRegression
    from sklearn.metrics import roc_auc_score

    if model != "auto" and model not in CONFIDENCE_MODELS:
        raise ValueError(f"ops_confidence_model must be one of "
                         f"{['auto'] + sorted(CONFIDENCE_MODELS)}; got {model!r}")
    if calibration not in ("target_decoy", "none"):
        raise ValueError("ops_confidence_calibration must be 'target_decoy' "
                         f"or 'none'; got {calibration!r}")
    d = _confidence_table(calls, folds)
    y = d["is_decoy"].to_numpy().astype(int)
    if not y.any() or y.all():
        raise ValueError("the genotype confidence needs decoy and real calls; "
                         "decode with ops_decoy_library_k above 0")
    c = float(fdr_scale) / float(decoy_k)
    features = list(CONFIDENCE_FEATURES)
    base = d[features].to_numpy(float)
    squared = np.c_[base, np.clip(base[:, :10], -50, 50) ** 2]
    fold = d["fold"].to_numpy()
    rng = np.random.default_rng(seed)

    def crossfit(make, x, labels, sub_real=max_real):
        """Out-of-fold P(decoy), each fit on its folds' decoys and real calls.

        :param make: builds an unfitted model.
        :param x: the features.
        :param labels: the decoy labels.
        :param sub_real: the most real calls a fit sees.
        :returns: one probability per call.
        """
        p = np.zeros(len(labels))
        for k in range(int(folds)):
            train = fold != k
            real_rows = np.flatnonzero(train & (labels == 0))
            decoy_rows = np.flatnonzero(train & (labels == 1))
            take = rng.choice(real_rows, min(sub_real, len(real_rows)), replace=False)
            rows = np.r_[take, decoy_rows]
            weight = np.r_[np.full(len(take), len(real_rows) / len(take)),
                           np.ones(len(decoy_rows))]
            fitted = make()
            fitted.fit(x[rows], labels[rows], sample_weight=weight)
            p[~train] = fitted.predict_proba(x[~train])[:, 1]
        return p

    p_lr = crossfit(_Logistic, squared, y)
    p_hgb = crossfit(lambda: HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
        min_samples_leaf=200, l2_regularization=1.0, random_state=seed), base, y)
    p_iso = np.zeros(len(y))
    score = np.log(np.clip(p_lr, 1e-9, 1))
    for k in range(int(folds)):
        train = fold != k
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(
            score[train], y[train])
        p_iso[~train] = iso.predict(score[~train])
    strata = (np.clip(d["n_reads"], 0, 5).astype(str) + "|" + d["source"] + "|"
              + d["unanimous"].astype(int).astype(str)).to_numpy()
    p_str = np.zeros(len(y))
    for k in range(int(folds)):
        train = fold != k
        rate = pd.Series(y[train]).groupby(strata[train]).mean()
        p_str[~train] = pd.Series(strata[~train]).map(rate).fillna(
            y[train].mean()).to_numpy()
    shuffled = y.copy()
    groups = (d["well"].astype(str) + "|" + d["site"].astype(str)).to_numpy()
    for _, index in pd.Series(np.arange(len(y))).groupby(groups).indices.items():
        shuffled[index] = y[index][rng.permutation(len(index))]
    p_sh = crossfit(_Logistic, squared, shuffled, sub_real=300_000)

    def error(p):
        """A real call's error from P(decoy).

        :param p: P(decoy).
        :returns: the error, 0 to 1.
        """
        if calibration == "none":
            return np.clip(p, 0, 1)
        return np.clip(c * p / np.clip(1 - p, 1e-12, None), 0, 1)

    def logloss(p, labels):
        """Mean log-loss.

        :param p: P(decoy).
        :param labels: the labels.
        :returns: the loss.
        """
        p = np.clip(p, 1e-9, 1 - 1e-9)
        return float(-(labels * np.log(p) + (1 - labels) * np.log(1 - p)).mean())

    def reliability(p, name):
        """Reliability rows: bins of equal expected decoys.

        :param p: P(decoy).
        :param name: the model.
        :returns: one row per bin.
        """
        order = np.argsort(p)
        cumulative = np.cumsum(p[order])
        edges = np.searchsorted(cumulative, np.linspace(0, cumulative[-1], bins + 1)[1:-1])
        which = np.zeros(len(p), int)
        which[order] = np.searchsorted(edges, np.arange(len(p)), side="right")
        rows = []
        for b in range(int(bins)):
            m = which == b
            nd, nr = int(y[m].sum()), int((1 - y[m]).sum())
            lo = chi2.ppf(0.025, 2 * nd) / 2 if nd > 0 else 0.0
            hi = chi2.ppf(0.975, 2 * nd + 2) / 2
            rows.append({"model": name, "bin": b, "calls": int(m.sum()),
                         "real_calls": nr, "decoy_calls": nd,
                         "predicted_error": float(error(p[m])[y[m] == 0].mean()),
                         "observed_error": c * nd / max(nr, 1),
                         "observed_lo": c * lo / max(nr, 1),
                         "observed_hi": c * hi / max(nr, 1)})
        return rows

    probabilities = {"LR": p_lr, "ISO": p_iso, "HGB": p_hgb, "STRATA": p_str,
                     "control: shuffled decoy labels": p_sh}
    metrics: Dict[str, Any] = {}
    reliability_rows: List[Dict[str, Any]] = []
    for name, p in probabilities.items():
        labels = shuffled if name.startswith("control") else y
        e = error(p)
        metrics[name] = {
            "AUC_decoy": float(roc_auc_score(labels, p)), "logloss": logloss(p, labels),
            "brier": float(((p - labels) ** 2).mean()),
            "predicted_false_real_calls": float(e[labels == 0].sum()),
            "decoy_estimated_false_real_calls": float(c * labels.sum()),
            "mean_error_real": float(e[labels == 0].mean()),
            "real_calls_error_gt_1pct": int((e[labels == 0] > 0.01).sum()),
            "real_calls_error_gt_10pct": int((e[labels == 0] > 0.1).sum())}
        if not name.startswith("control"):
            reliability_rows += reliability(p, name)
    best = min(("LR", "ISO", "HGB", "STRATA"), key=lambda k: metrics[k]["logloss"])
    chosen = best if model == "auto" else CONFIDENCE_MODELS[model]
    metrics.update({"best_by_logloss": best, "chosen": chosen,
                    "rule": ("lowest held-out log-loss on the decoy label"
                             if model == "auto" else f"ops_confidence_model {model}"),
                    "fdr_scale_c": c, "calibration": calibration, "seed": int(seed)})
    table = calls.copy()
    for name in ("LR", "ISO", "HGB", "STRATA"):
        table[f"e_{name}"] = error(probabilities[name])
    table["e"] = table[f"e_{chosen}"]
    table["genotype_confidence"] = 1 - table["e"]
    stratum = pd.DataFrame({"stratum": strata, "is_decoy": d["is_decoy"].to_numpy(),
                            "e": table["e"].to_numpy()})
    strata_table = stratum.groupby("stratum").agg(
        calls=("is_decoy", "size"), decoys=("is_decoy", "sum"),
        model_error=("e", "mean")).reset_index()
    strata_table["observed_error"] = c * strata_table["decoys"] / (
        strata_table["calls"] - strata_table["decoys"]).clip(lower=1)
    return {"table": table, "metrics": metrics,
            "reliability": pd.DataFrame(reliability_rows), "strata": strata_table}


#: The sequence-only control features of the false-call model (item 876):
#: what a call's guide sequence alone says about whether it is a decoy.
SEQUENCE_FEATURES = ("_gc", "_homopolymer", "_frac_G", "_frac_T", "_frac_A",
                     "_frac_C", "nearest_other_guide")

#: The columns of the false-call cell table that are labels, keys or splits,
#: never features.
_QC_DROP = ("field", "obj", "label", "guide", "tp2", "test", "plate",
            "decoy_half")

#: The permissive decode the false-call model is trained on: a low read
#: threshold, matches at up to two mismatches, territory attribution and a
#: plain majority vote, so the decoy calls the strict rule refuses are there
#: to learn from.
QC_PERMISSIVE = {"threshold": 150.0, "footprint": 10.0, "max_distance": 2,
                 "tie": 1.0, "min_fraction": 0.6}


def _field_exact_rate(codes, real_matcher) -> float:
    """The share of a field's reads that match a guide exactly.

    :param codes: the field's called reads.
    :param real_matcher: a matcher over the real guides only.
    :returns: the share, 0 for a field without reads.
    """
    return float(np.mean([real_matcher.match(b, 0)[0] is not None for b in codes])
                 if codes else 0)


def false_call_tables(fields: Sequence[Mapping[str, Any]],
                      library: Sequence[str], decoys: Sequence[str], *,
                      seed: int = 7, tile_centre: float = 740.0,
                      permissive: Optional[Mapping[str, Any]] = None,
                      keep_reads: bool = False):
    """The read and cell tables the false-call model learns from (item 876).

    Each field is decoded permissively (:data:`QC_PERMISSIVE`) against the
    library and the decoys. Every read owned by a field's nucleus becomes a
    row with its match, per-cycle margins and mismatches, spot features and
    position; every owned nucleus becomes a row labelled ``FP`` (its reads
    vote for a decoy), ``TP`` (a guide) or ``TN`` (no call), with the mean
    evidence of its supporting reads and the sequence features of its guide.
    Half the fields of each plate are held out (``test``), and the decoys are
    split by the md5 of their sequence into a training half and an
    evaluation half (``decoy_half``).

    :param fields: ``{"plate", "well", "site", "z"}`` per field, ``z`` the
        field's evidence (:func:`field_evidence` with ``features=True``).
    :param library: the guide barcodes.
    :param decoys: the decoy barcodes.
    :param seed: the held-out field draw's seed (7 in the study).
    :param tile_centre: the tile centre, for each read's radius in the tile.
    :param permissive: overrides of :data:`QC_PERMISSIVE`.
    :param keep_reads: also return the read table, which is about ten times
        the size of the cell table; False returns an empty one.
    :returns: ``(reads, cells)`` DataFrames.
    """
    import hashlib
    from collections import Counter, defaultdict

    import pandas as pd

    from .ops_sbs import LibraryMatcher, called_bases

    rule = {**QC_PERMISSIVE, **(permissive or {})}
    library = sorted(set(library))
    decoy_set = set(decoys)
    real = LibraryMatcher(library)
    combined = LibraryMatcher(library + list(decoys))
    table = np.frombuffer("".join(library).encode(), np.uint8).reshape(len(library), -1)
    nearest: Dict[str, int] = {}

    def nearest_other(guide):
        """The fewest mismatches from a guide to another guide.

        :param guide: a library guide.
        :returns: the distance.
        """
        if guide not in nearest:
            distance = (table != np.frombuffer(guide.encode(), np.uint8)).sum(1)
            nearest[guide] = int(distance[distance > 0].min())
        return nearest[guide]

    rng = np.random.default_rng(seed)
    evaluation = {d for d in decoy_set if int(hashlib.md5(d.encode()).hexdigest(), 16) % 2}
    training = decoy_set - evaluation
    plates = np.array([str(f["plate"]) for f in fields])
    test_field = np.zeros(len(fields), bool)
    for plate in sorted(set(plates)):
        index = np.flatnonzero(plates == plate)
        test_field[rng.choice(index, len(index) // 2, replace=False)] = True
    all_reads, cell_frames = [], []
    cycle_keys = None
    for fi, fd in enumerate(fields):
        z = fd["z"]
        selected = np.asarray(z["strength"]) > rule["threshold"]
        first, _, _ = called_bases(np.asarray(z["values"])[selected], normalise=True, gpu=False)
        data, codes, margin = called_bases(first, compensate=True, method="argmax_median",
                                           normalise=False, gpu=False) if len(first) else \
            (first, [], np.zeros((0, np.asarray(z["values"]).shape[1])))
        gap = np.asarray(z["gap"])[selected]
        first_id = np.asarray(z["id0"])[selected]
        near = (gap[:, 0] <= rule["footprint"]) & np.asarray(z["owned0"])[selected]
        ambiguous = (gap[:, 1] - gap[:, 0]) < rule["tie"]
        peaks = np.asarray(z["peaks"])[selected]
        strength = np.asarray(z["strength"])[selected]
        spot = {k: np.asarray(z[k])[selected] for k in
                ("f_raw", "f_bg", "f_sharp", "f_focus", "f_density") if k in z}
        exact_rate = _field_exact_rate(codes, real)
        areas = dict(zip(np.asarray(z["owned_ids"]).tolist(),
                         np.asarray(z["owned_areas"]).tolist()))
        n_cycles = np.asarray(z["values"]).shape[1]
        cells = defaultdict(list)
        for i in np.flatnonzero(near):
            guide, d = combined.match(codes[i], rule["max_distance"])
            mismatched = ([c for c, (x, yv) in enumerate(zip(codes[i], guide)) if x != yv]
                          if guide else [])
            m = margin[i]
            finite = np.isfinite(m).any()
            row = {"field": fi, "obj": int(first_id[i]), "guide": guide,
                   "d": d if guide else -1,
                   "label_read": "TN" if guide is None else ("FP" if guide in decoy_set else "TP"),
                   "q": float(np.nanmin(m)) if finite else 0.0,
                   "q_mean": float(np.nanmean(m)) if finite else 0.0,
                   "strength": float(strength[i]), "gap0": float(gap[i, 0]),
                   "gap_diff": float(min(gap[i, 1] - gap[i, 0], 50)),
                   "ambiguous": bool(ambiguous[i]),
                   "y": float(peaks[i, 0]), "x": float(peaks[i, 1]),
                   "n_mismatch": len(mismatched),
                   "nearest_other_guide": (nearest_other(guide)
                                           if (guide and guide not in decoy_set) else
                                           (int(((table != np.frombuffer(guide.encode(), np.uint8)).sum(1)).min())
                                            if guide else -1))}
            for c in range(n_cycles):
                row[f"margin_c{c + 1}"] = float(np.nan_to_num(m[c], nan=-1))
                row[f"mismatch_c{c + 1}"] = int(c in mismatched)
                if spot:
                    raw = spot["f_raw"][i, c]
                    background = spot["f_bg"][i, c]
                    row[f"raw_max_c{c + 1}"] = (float(np.nanmax(raw))
                                                if np.isfinite(raw).any() else -1)
                    row[f"bg_c{c + 1}"] = (float(np.nanmean(background))
                                           if np.isfinite(background).any() else -1)
            if spot:
                row["sharpness"] = float(spot["f_sharp"][i])
                row["focus_local"] = float(spot["f_focus"][i])
                row["neighbour_spots"] = int(spot["f_density"][i])
            row["field_exact_rate"] = float(exact_rate)
            row["radius_in_tile"] = float(np.hypot(peaks[i, 0] - tile_centre,
                                                   peaks[i, 1] - tile_centre))
            row["nucleus_area"] = float(areas.get(int(first_id[i]), np.nan))
            row["_seq"] = guide or ""
            if keep_reads:
                all_reads.append(row)
            cells[int(first_id[i])].append(row)
        if cycle_keys is None:
            cycle_keys = ([f"margin_c{i}" for i in range(1, n_cycles + 1)]
                          + [f"mismatch_c{i}" for i in range(1, n_cycles + 1)]
                          + [f"bg_c{i}" for i in range(1, n_cycles + 1)]
                          + [f"raw_max_c{i}" for i in range(1, n_cycles + 1)]
                          + ["sharpness", "focus_local", "neighbour_spots",
                             "radius_in_tile", "y", "x"])
        field_cells = []
        for owner in np.asarray(z["owned_ids"]).tolist():
            reads = cells.get(owner, [])
            matched = [r for r in reads if r["guide"]]
            cell = {"field": fi, "obj": owner, "n_reads": len(reads),
                    "n_matched": len(matched),
                    "nucleus_area": float(areas.get(owner, np.nan)),
                    "field_exact_rate": float(exact_rate)}
            label, guide = "TN", None
            if matched:
                tally = Counter(r["guide"] for r in matched)
                (guide, n), *rest = tally.most_common()
                if not (rest and rest[0][1] == n) and n / len(matched) >= rule["min_fraction"]:
                    label = "FP" if guide in decoy_set else "TP"
                else:
                    guide = None
            cell["label"] = label
            cell["guide"] = guide
            if guide:
                support = [r for r in matched if r["guide"] == guide]
                cell["n_support"] = len(support)
                cell["agreement"] = len(support) / len(matched)
                cell["n_exact_support"] = sum(r["d"] == 0 for r in support)
                cell["tp2"] = label == "TP" and cell["n_exact_support"] >= 2
                cell["min_mismatch"] = min(r["d"] for r in support)
                cell["q_best"] = max(r["q"] for r in support)
                cell["q_mean"] = float(np.mean([r["q_mean"] for r in support]))
                cell["strength_max"] = max(r["strength"] for r in support)
                cell["gap0_min"] = min(r["gap0"] for r in support)
                cell["ambiguous_share"] = float(np.mean([r["ambiguous"] for r in support]))
                cell["nearest_other_guide"] = support[0]["nearest_other_guide"]
                for key in cycle_keys:
                    if key in support[0]:
                        cell[key] = float(np.mean([r[key] for r in support]))
                runs = "".join(ch if i == 0 or guide[i - 1] == ch else "|" + ch
                               for i, ch in enumerate(guide)).split("|")
                cell["_gc"] = (guide.count("G") + guide.count("C")) / len(guide)
                cell["_homopolymer"] = max(len(x) for x in runs)
                for base in "GTAC":
                    cell[f"_frac_{base}"] = guide.count(base) / len(guide)
            field_cells.append(cell)
        cell_frames.append(pd.DataFrame(field_cells))
    reads = pd.DataFrame(all_reads)
    cells = (pd.concat(cell_frames, ignore_index=True, sort=False)
             if cell_frames else pd.DataFrame())
    for frame in (cells, reads):
        if len(frame):
            frame["test"] = test_field[frame["field"].to_numpy()]
            frame["plate"] = plates[frame["field"].to_numpy()]
            frame["decoy_half"] = frame["guide"].map(
                lambda g: "eval" if g in evaluation else ("train" if g in training else ""))
    return reads, cells


def _xgboost_classifier(params: Mapping[str, Any]):
    """An XGBoost classifier, imported only when asked for.

    :param params: the classifier's parameters.
    :returns: the unfitted classifier.
    :raises ImportError: with the install line, when xgboost is absent.
    """
    try:
        import xgboost
    except ImportError as error:
        raise ImportError(
            "ops_decode_qc_model 'xgboost' needs the xgboost package: "
            "pip install xgboost, or set ops_decode_qc_model to 'hist_gbm'"
        ) from error
    return xgboost.XGBClassifier(**params)


def _tree_shap(model, frame) -> Tuple[np.ndarray, float]:
    """Per-call SHAP values of a fitted tree model.

    :param model: a fitted XGBoost or scikit-learn tree classifier.
    :param frame: the features.
    :returns: ``(values, base_value)``. The shap package is used when it is
        installed; for XGBoost without it, the booster's own exact TreeSHAP
        (``pred_contribs``) gives the same values.
    """
    try:
        import shap
    except ImportError:
        shap = None
    if shap is not None:
        explainer = shap.TreeExplainer(model)
        values = explainer.shap_values(frame)
        if isinstance(values, list):
            values = values[-1]
        values = np.asarray(values)
        if values.ndim == 3:
            values = values[..., -1]
        return values, float(np.atleast_1d(explainer.expected_value)[-1])
    if hasattr(model, "get_booster"):
        import xgboost

        contributions = model.get_booster().predict(
            xgboost.DMatrix(frame), pred_contribs=True)
        return contributions[:, :-1], float(contributions[0, -1])
    raise ImportError("SHAP values of a hist_gbm model need the shap package: "
                      "pip install shap")


def false_call_model(cells, *, model: str = "xgboost", seed: int = 0,
                     shap_sample: int = 20000, permutation_repeats: int = 10,
                     n_jobs: int = 1, features: Optional[Sequence[str]] = None,
                     boot: int = 300) -> Dict[str, Any]:
    """What separates a decoy call from a library call (item 876).

    Trained on the training fields' library calls and training-half decoys,
    scored on the held-out fields' library calls and evaluation-half
    decoys, so neither a field nor a decoy sequence is both learnt and
    tested. A sequence-only model (:data:`SEQUENCE_FEATURES`) measures how
    much of the separation comes from how the decoys were built; chance is
    an AUROC of 0.5. Mean absolute SHAP (Lundberg and Lee 2017) and
    permutation importance (average-precision drop) rank the features.

    :param cells: from :func:`false_call_tables`.
    :param model: ``xgboost`` (an optional dependency) or ``hist_gbm``.
    :param seed: the model and resampling seed (0 in the study).
    :param shap_sample: how many held-out calls the SHAP ranking is taken
        over; 0 takes them all.
    :param permutation_repeats: permutation-importance repeats.
    :param n_jobs: model threads; 1 is reproducible bit for bit.
    :param features: the main model's features; None takes every column that
        is not a label, key, split or sequence feature.
    :param boot: field-bootstrap replicates for the AUROC and AP intervals.
    :returns: ``metrics`` (a DataFrame: main, sequence-only control, chance),
        ``shap_mean_abs`` and ``permutation_importance`` (DataFrames),
        ``model`` (the fitted main model) and ``features``.
    :raises ValueError: for an unknown model, or no held-out decoy.
    """
    import pandas as pd
    from sklearn.inspection import permutation_importance
    from sklearn.metrics import average_precision_score, roc_auc_score

    if model not in ("xgboost", "hist_gbm"):
        raise ValueError("ops_decode_qc_model must be 'xgboost' or 'hist_gbm'; "
                         f"got {model!r}")
    sequence = [c for c in cells.columns if c.startswith("_")] + ["nearest_other_guide"]
    main = list(features) if features else [
        c for c in cells.columns if c not in _QC_DROP and c not in sequence]
    assigned = cells[cells["label"].isin(["TP", "FP"])]
    train = assigned[~assigned["test"] & ((assigned["label"] == "TP")
                                          | (assigned["decoy_half"] == "train"))]
    test = assigned[assigned["test"] & ((assigned["label"] == "TP")
                                        | (assigned["decoy_half"] == "eval"))]
    y_train = (train["label"] == "FP").astype(int)
    y_test = (test["label"] == "FP").astype(int).to_numpy()
    if not y_test.any() or not y_train.any():
        raise ValueError("the false-call model needs decoy calls in the training "
                         "and the held-out fields; decode with ops_decoy_library_k "
                         "above 0 over more fields")

    def make():
        """An unfitted classifier of the chosen kind.

        :returns: the classifier.
        """
        if model == "xgboost":
            return _xgboost_classifier(dict(
                n_estimators=400, max_depth=5, learning_rate=0.05, subsample=0.8,
                colsample_bytree=0.8, n_jobs=n_jobs, eval_metric="logloss",
                random_state=seed))
        from sklearn.ensemble import HistGradientBoostingClassifier

        return HistGradientBoostingClassifier(max_iter=400, max_depth=5,
                                              learning_rate=0.05, random_state=seed)

    rng = np.random.default_rng(seed)
    test_fields = test["field"].unique()
    field_of = test["field"].to_numpy()

    def interval(labels, p, score):
        """A field-bootstrap 95 % interval.

        :param labels: the held-out labels.
        :param p: the predictions.
        :param score: the metric.
        :returns: ``(low, high)``.
        """
        values = []
        for _ in range(int(boot)):
            picked = rng.choice(test_fields, len(test_fields))
            index = np.concatenate([np.flatnonzero(field_of == x) for x in picked])
            if labels[index].min() == labels[index].max():
                continue
            values.append(score(labels[index], p[index]))
        return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))

    rows, fitted = [], {}
    for name, columns in (("main_fp_vs_tp", main), ("control_sequence_only", sequence)):
        classifier = make()
        classifier.fit(train[columns].astype(float), y_train)
        p = classifier.predict_proba(test[columns].astype(float))[:, 1]
        rows.append({"model": name, "n_train": int(len(train)), "n_test": int(len(test)),
                     "fp_test": int(y_test.sum()),
                     "auc": round(roc_auc_score(y_test, p), 4),
                     "auc_ci95": interval(y_test, p, roc_auc_score),
                     "average_precision": round(average_precision_score(y_test, p), 4),
                     "ap_ci95": interval(y_test, p, average_precision_score),
                     "fp_prevalence_test": round(float(y_test.mean()), 4)})
        fitted[name] = classifier
    rows.append({"model": "chance", "n_train": int(len(train)), "n_test": int(len(test)),
                 "fp_test": int(y_test.sum()), "auc": 0.5, "auc_ci95": (0.5, 0.5),
                 "average_precision": round(float(y_test.mean()), 4),
                 "ap_ci95": (round(float(y_test.mean()), 4),) * 2,
                 "fp_prevalence_test": round(float(y_test.mean()), 4)})
    best = fitted["main_fp_vs_tp"]
    held_out = test[main].astype(float)
    sample = held_out.sample(min(int(shap_sample), len(held_out)), random_state=seed) \
        if shap_sample else held_out
    values, base_value = _tree_shap(best, sample)
    shap_mean = pd.Series(np.abs(values).mean(0), index=main).sort_values(ascending=False)
    permutation = permutation_importance(
        best, held_out, y_test,
        scoring=lambda est, x, y: average_precision_score(y, est.predict_proba(x)[:, 1]),
        n_repeats=int(permutation_repeats), random_state=seed, n_jobs=1)
    importance = pd.DataFrame({"feature": main,
                               "permutation_ap_drop_mean": permutation.importances_mean,
                               "permutation_ap_drop_sd": permutation.importances_std})
    return {"metrics": pd.DataFrame(rows),
            "shap_mean_abs": shap_mean.rename("mean_abs_shap").rename_axis("feature").reset_index(),
            "shap_base_value": base_value,
            "permutation_importance": importance.sort_values(
                "permutation_ap_drop_mean", ascending=False).reset_index(drop=True),
            "model": best, "features": main, "sequence_features": sequence}


def _write(frame, path: str) -> str:
    """Write a table by its suffix, column names as they are.

    :param frame: the table.
    :param path: the destination.
    :returns: the path written.
    """
    from .tabular import write_table

    return write_table(frame, path, canonicalise=False)


def write_well_calls(db: str, well: str, decoded: Sequence[Mapping[str, Any]]) -> str:
    """Write every call of a well with its evidence, decoys included.

    :param db: the measurements database.
    :param well: the well.
    :param decoded: the decoded fields, each with ``site`` and ``rule_a``
        carrying ``evidence``.
    :returns: the path of ``results/ops_decode/calls_<well>.parquet``.
    """
    import pandas as pd

    rows = [{"well": well, "site": int(r["site"]), **row}
            for r in sorted(decoded, key=lambda r: r["site"])
            for row in r["rule_a"].get("evidence", ())]
    frame = pd.DataFrame(rows, columns=["well", "site"] + list(EVIDENCE_COLUMNS))
    path = os.path.join(results_folder(db), f"calls_{well}.parquet")
    _write(frame, path)
    return path


def _qc_number(settings: Mapping[str, Any], key: str, default, cast=float):
    """One decode-QC setting, its default when empty.

    :param settings: the settings.
    :param key: the key.
    :param default: the default.
    :param cast: ``float`` or ``int``.
    :returns: the value, or None when the default is None and it is empty.
    :raises ValueError: when it is not a number.
    """
    value = settings.get(key)
    if value is None or value == "":
        return default
    try:
        return cast(float(value)) if cast is int else cast(value)
    except (TypeError, ValueError):
        raise ValueError(f"{key} must be a number; got {value!r}") from None


def confidence_outputs(result: Mapping[str, Any], folder: str) -> Dict[str, Any]:
    """Write the genotype-confidence report (item 877).

    :param result: from :func:`genotype_confidence`.
    :param folder: ``results/ops_decode_qc``.
    :returns: ``confidence_metrics.json``'s content, with the paths.
    """
    reliability = result["reliability"]
    chosen = result["metrics"]["chosen"]
    mine = reliability[reliability["model"] == chosen]
    table = mine.rename(columns={"predicted_error": "predicted",
                                 "observed_error": "observed",
                                 "real_calls": "n"})[
        ["bin", "predicted", "observed", "n", "decoy_calls", "observed_lo",
         "observed_hi"]]
    reliability_path = os.path.join(folder, "confidence_reliability.csv")
    _write(table, reliability_path)
    _write(reliability, os.path.join(folder, "confidence_reliability_all_models.csv"))
    ece = float((mine["real_calls"] / max(int(mine["real_calls"].sum()), 1)
                 * (mine["predicted_error"] - mine["observed_error"]).abs()).sum())
    metrics = {"model": chosen, "auc": result["metrics"][chosen]["AUC_decoy"],
               "ece": ece,
               "shuffled_auc": result["metrics"]["control: shuffled decoy labels"]["AUC_decoy"],
               "models": result["metrics"]}
    metrics_path = os.path.join(folder, "confidence_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=1, default=str)
    _write(result["strata"], os.path.join(folder, "confidence_strata.csv"))
    calls_path = os.path.join(folder, "confidence_calls.parquet")
    _write(result["table"], calls_path)
    metrics["paths"] = {"reliability": reliability_path, "metrics": metrics_path,
                        "calls": calls_path}
    return metrics


def _with_confidence(db: str, plate: str, table, minimum) -> Dict[str, Any]:
    """Add ``genotype_confidence`` to ``ops_barcodes``, dropping calls below
    ``minimum``.

    :param db: the measurements database.
    :param plate: the plate.
    :param table: the confidence table: ``well``, ``object_id``,
        ``is_decoy``, ``genotype_confidence``.
    :param minimum: the lowest confidence kept, or None.
    :returns: rows updated and rows dropped.
    """
    from .ops_store import read_table, row_count, write_table

    if row_count(db, "ops_barcodes") is None:
        return {"updated": 0, "dropped": 0}
    barcodes = read_table(db, "ops_barcodes")
    real = table[~table["is_decoy"].astype(bool)][
        ["well", "object_id", "genotype_confidence"]].copy()
    real["plate"] = plate
    barcodes = barcodes.drop(columns=[c for c in ("genotype_confidence",)
                                      if c in barcodes.columns])
    merged = barcodes.merge(real, on=["plate", "well", "object_id"], how="left")
    dropped = 0
    if minimum is not None:
        low = (merged["plate"].astype(str) == str(plate)) & (
            merged["genotype_confidence"] < float(minimum))
        dropped = int(low.sum())
        merged = merged[~low].reset_index(drop=True)
    write_table(db, "ops_barcodes", merged, if_exists="replace")
    return {"updated": int(merged["genotype_confidence"].notna().sum()),
            "dropped": dropped}


def false_call_outputs(result: Mapping[str, Any], cells, folder: str) -> Dict[str, Any]:
    """Write the false-call model's report (item 876).

    :param result: from :func:`false_call_model`.
    :param cells: the cell table it was trained on.
    :param folder: ``results/ops_decode_qc``.
    :returns: the paths and the metric rows.
    """
    paths = {name: os.path.join(folder, name) for name in (
        "model_metrics.csv", "shap_mean_abs.csv", "permutation_importance.csv",
        "model.json", "false_call_cells.parquet")}
    _write(result["metrics"], paths["model_metrics.csv"])
    _write(result["shap_mean_abs"], paths["shap_mean_abs.csv"])
    _write(result["permutation_importance"], paths["permutation_importance.csv"])
    model = result["model"]
    if hasattr(model, "get_booster"):
        model.get_booster().save_model(paths["model.json"])
    else:
        with open(paths["model.json"], "w", encoding="utf-8") as handle:
            json.dump({"model": type(model).__name__,
                       "params": {k: str(v) for k, v in model.get_params().items()},
                       "features": list(result["features"])}, handle, indent=1)
    _write(cells, paths["false_call_cells.parquet"])
    return {"paths": paths, "metrics": result["metrics"].to_dict("records")}


def _cached_fields(destination: str, plate: str, wells: Sequence[str]):
    """The stored field evidence of the wells, in well and site order.

    :param destination: ``dst_root``.
    :param plate: the plate.
    :param wells: the wells.
    :returns: ``{"plate", "well", "site", "z"}`` per field; ``z`` opens lazily.
    """
    fields = []
    for well in sorted(wells):
        folder = os.path.join(destination, well, "decode_cache")
        if not os.path.isdir(folder):
            continue
        for name in sorted(os.listdir(folder)):
            if name.startswith("site") and name.endswith(".npz"):
                fields.append({"plate": plate, "well": well, "site": int(name[4:8]),
                               "z": np.load(os.path.join(folder, name))})
    return fields


def run_decode_qc(db: str, plate: str, wells: Sequence[str],
                  settings: Mapping[str, Any], library: Sequence[str]
                  ) -> Dict[str, Any]:
    """The decode QC phase: the false-call model and the genotype confidence.

    :param db: the measurements database.
    :param plate: the plate.
    :param wells: the wells decoded.
    :param settings: the run's settings; ``ops_decode_qc`` and
        ``ops_genotype_confidence`` choose what runs.
    :param library: the guide barcodes.
    :returns: one report per part run, under ``false_call_model`` and
        ``genotype_confidence``.
    :raises ValueError: when a part's input is missing.
    """
    import pandas as pd

    from .ops_sbs import decoy_library
    from .tabular import read_table

    settings = resolve_settings(settings)
    rule = decode_rule(settings)
    folder = results_folder(db, "ops_decode_qc")
    report: Dict[str, Any] = {}
    if _flag_or(settings, "ops_genotype_confidence"):
        names = [os.path.join(results_folder(db), f"calls_{w}.parquet")
                 for w in sorted(wells)]
        present = [n for n in names if os.path.isfile(n)]
        if not present:
            raise ValueError("ops_genotype_confidence needs the per-call evidence "
                             "the decode writes under results/ops_decode; decode "
                             "with ops_decode_preset rule_a_s2 first")
        calls = pd.concat([read_table(n, canonicalise=False, report=None) for n in present], ignore_index=True)
        result = genotype_confidence(
            calls, model=str(settings.get("ops_confidence_model") or "hist_gbm"),
            calibration=str(settings.get("ops_confidence_calibration") or "target_decoy"),
            bins=_qc_number(settings, "ops_confidence_bins", 10, int),
            seed=_qc_number(settings, "ops_confidence_seed", CONFIDENCE_SEED, int),
            fdr_scale=rule["fdr_scale"], decoy_k=max(rule["decoy_k"], 1))
        metrics = confidence_outputs(result, folder)
        metrics["ops_barcodes"] = _with_confidence(
            db, plate, result["table"],
            _qc_number(settings, "ops_confidence_min", None))
        report["genotype_confidence"] = metrics
    if _flag_or(settings, "ops_decode_qc"):
        split = str(settings.get("ops_decode_qc_split") or "field").strip().lower()
        if split != "field":
            raise ValueError(f"ops_decode_qc_split must be 'field'; got {split!r}")
        destination = os.path.dirname(os.path.abspath(db))
        fields = _cached_fields(destination, plate, wells)
        if not fields:
            raise ValueError("ops_decode_qc needs the field evidence the decode "
                             "stores with ops_decode_qc on; decode again with it")
        decoys = decoy_library(library, k=max(rule["decoy_k"], 1))
        _, cells = false_call_tables(fields, library, decoys)
        chosen = settings.get("ops_decode_qc_features")
        features = ([f.strip() for f in str(chosen).split(",") if f.strip()]
                    if chosen else None)
        result = false_call_model(
            cells, model=str(settings.get("ops_decode_qc_model") or "xgboost"),
            seed=_qc_number(settings, "ops_decode_qc_seed", 0, int),
            shap_sample=_qc_number(settings, "ops_decode_qc_max_calls", 20000, int),
            features=features)
        report["false_call_model"] = false_call_outputs(result, cells, folder)
    return report


def _flag_or(settings: Mapping[str, Any], key: str) -> bool:
    """A boolean QC setting, False when empty.

    :param settings: the settings.
    :param key: the key.
    :returns: the value.
    """
    value = settings.get(key)
    if isinstance(value, str):
        return value.strip().lower() not in ("", "false", "0", "no", "off")
    return bool(value)


#: How far apart, in pixels, two pipelines' nuclei may lie and still be one
#: cell (item 878); the three-way comparison of the decode-yield study used 4.
COMPARE_MATCH_PX = 4.0

#: How far from the tile border a cell must lie to be compared, in pixels.
COMPARE_EDGE_PX = 10


def match_units(tables: Mapping[str, Any], max_px: float = COMPARE_MATCH_PX
                ) -> List[Dict[str, int]]:
    """Group the cells of several call tables into matched units.

    Every pair of tables is linked by mutual nearest neighbours within
    ``max_px``; the links are joined shortest first, and a link that would
    put two cells of one table into a unit is skipped, so a unit holds at
    most one cell per table.

    :param tables: ``name -> DataFrame`` with ``y`` and ``x`` in one frame.
    :param max_px: the largest distance linked.
    :returns: one ``{name: row index}`` per unit, cells left alone included.
    """
    from itertools import combinations

    from scipy.spatial import cKDTree

    names = list(tables)
    edges = []
    for a, b in combinations(names, 2):
        ta, tb = tables[a], tables[b]
        if not len(ta) or not len(tb):
            continue
        pa, pb = ta[["y", "x"]].to_numpy(), tb[["y", "x"]].to_numpy()
        dab, iab = cKDTree(pb).query(pa)
        dba, iba = cKDTree(pa).query(pb)
        for i, (d, j) in enumerate(zip(dab, iab)):
            if d <= max_px and iba[j] == i:
                edges.append((d, (a, i), (b, j)))
    parent: Dict[Tuple[str, int], Tuple[str, int]] = {}
    members: Dict[Tuple[str, int], Dict[str, int]] = {}

    def find(node):
        """The root of a node's unit.

        :param node: ``(name, index)``.
        :returns: the root.
        """
        while parent.get(node, node) != node:
            node = parent[node]
        return node

    for name in names:
        for i in range(len(tables[name])):
            parent[(name, i)] = (name, i)
            members[(name, i)] = {name: i}
    for _, u, v in sorted(edges):
        ru, rv = find(u), find(v)
        if ru == rv or set(members[ru]) & set(members[rv]):
            continue
        parent[rv] = ru
        members[ru] = {**members[ru], **members.pop(rv)}
    return [members[r] for r in members if find(r) == r]


def call_category(calls: Mapping[str, Optional[str]],
                  labels: Optional[Mapping[str, str]] = None) -> Tuple[str, str]:
    """The agreement category of one matched unit.

    :param calls: ``name -> guide or None`` for the tables in the unit.
    :param labels: display names; None uses the table names.
    :returns: ``(category, subcategory)``: ``all3_same``, ``all3_diff``,
        ``two_agree_third_diff``, ``two_agree_third_none``,
        ``two_disagree_third_none``, ``one_only``, ``pair_agree``,
        ``pair_disagree`` or ``none``, and which table it names.
    """
    labels = labels or {name: name for name in calls}
    called = {m: g for m, g in calls.items() if g is not None}
    n = len(called)
    values = list(called.values())
    if n == 0:
        return "none", ""
    if n == 1:
        return "one_only", f"{labels[list(called)[0]]}-only"
    if len(calls) == 2 or n == 2:
        missing = [m for m in calls if calls[m] is None]
        joined = "+".join(labels[m] for m in called)
        if len(set(values)) == 1:
            if len(calls) == 2:
                return "pair_agree", joined
            return "two_agree_third_none", f"{joined} agree, {labels[missing[0]]} none"
        if len(calls) == 2:
            return "pair_disagree", joined
        return "two_disagree_third_none", f"{joined} differ, {labels[missing[0]]} none"
    if len(set(values)) == 1:
        return "all3_same", ""
    if len(set(values)) == 3:
        return "all3_diff", ""
    odd = [m for m in called if values.count(called[m]) == 1][0]
    return "two_agree_third_diff", f"{labels[odd]} differs"


def compare_field(tables: Mapping[str, Any], calls: Mapping[str, Mapping[int, str]],
                  decoys, *, max_px: float = COMPARE_MATCH_PX,
                  labels: Optional[Mapping[str, str]] = None) -> List[Dict[str, Any]]:
    """The matched units of one field and their categories.

    :param tables: ``name -> DataFrame`` of the field's compared cells, with
        ``cell``, ``y`` and ``x``.
    :param calls: ``name -> {cell: guide}``.
    :param decoys: the decoy set.
    :param max_px: the matching distance.
    :param labels: display names for the categories.
    :returns: one row per unit: ``present``, ``<name>_cell``,
        ``<name>_call``, ``<name>_decoy``, ``category``, ``subcategory``.
    """
    names = list(tables)
    single = all(len(n) == 1 for n in names)
    rows = []
    for unit in match_units(tables, max_px):
        present = ("" if single else "+").join(m for m in names if m in unit)
        row: Dict[str, Any] = {"present": present}
        found = {}
        for m in names:
            if m not in unit:
                row.update({f"{m}_cell": np.nan, f"{m}_call": None})
                continue
            cell = int(tables[m].iloc[unit[m]]["cell"])
            guide = calls[m].get(cell)
            found[m] = guide
            row.update({f"{m}_cell": cell, f"{m}_call": guide,
                        f"{m}_decoy": guide in decoys if guide else False})
        category, sub = call_category(found, labels)
        everyone = len(unit) == len(names)
        row["category"] = category if everyone else f"matched_{present}:{category}"
        row["subcategory"] = sub
        rows.append(row)
    return rows


def compare_summary(units, names: Sequence[str], *, decoy_k: int = 4,
                    fdr_scale: float = 1.268,
                    labels: Optional[Mapping[str, str]] = None):
    """Category counts with a decoy FDR per table, and per-table totals.

    :param units: rows from :func:`compare_field` with ``plate`` and
        ``field`` added.
    :param names: the table names, in order.
    :param decoy_k: decoys per guide.
    :param fdr_scale: the FDR calibration.
    :param labels: display names for the columns.
    :returns: ``(categories, tables)`` DataFrames: per plate and category,
        ``n``, ``pct_of_matched_triples``, ``fields`` and each table's real
        and decoy calls and FDR; per plate and table over the units every
        table holds, real calls, assigned % and FDR.
    """
    import pandas as pd

    labels = labels or {name: name for name in names}
    single = all(len(n) == 1 for n in names)
    full = ("" if single else "+").join(names)
    u = units.copy()
    for m in names:
        if f"{m}_call" not in u:
            u[f"{m}_call"] = None
        u[f"{m}_called"] = u[f"{m}_call"].notna()
        u[f"{m}_decoy"] = (u[f"{m}_decoy"].eq(True) if f"{m}_decoy" in u
                           else False)
        u[f"{m}_real"] = u[f"{m}_called"] & ~u[f"{m}_decoy"]
    u["cat_full"] = np.where(u["subcategory"].fillna("") != "",
                             u["category"] + " | " + u["subcategory"].fillna(""),
                             u["category"])
    rows = []
    for plate, g in list(u.groupby("plate")) + [("ALL", u)]:
        triples = int((g["present"] == full).sum())
        for cat, h in g.groupby("cat_full"):
            present = h["present"].iloc[0]
            row = {"plate": plate, "category": cat, "present": present, "n": len(h),
                   "pct_of_matched_triples": (round(100 * len(h) / max(triples, 1), 3)
                                              if present == full else np.nan),
                   "fields": h["field"].nunique()}
            for m in names:
                if not h["present"].str.contains(m, regex=False).any():
                    continue
                real, decoy = int(h[f"{m}_real"].sum()), int(h[f"{m}_decoy"].sum())
                row.update({f"{labels[m]}_real_calls": real,
                            f"{labels[m]}_decoy_calls": decoy,
                            f"{labels[m]}_decoy_FDR_pct": (round(fdr(real, decoy, decoy_k, fdr_scale), 3)
                                                           if real else np.nan)})
            rows.append(row)
    triples = u[u["present"] == full]
    per_table = []
    for plate, g in list(triples.groupby("plate")) + [("ALL", triples)]:
        for m in names:
            real, decoy = int(g[f"{m}_real"].sum()), int(g[f"{m}_decoy"].sum())
            per_table.append({"plate": plate, "table": labels[m],
                              "matched_triples": len(g), "real_calls": real,
                              "assigned_pct": round(100 * real / max(len(g), 1), 2),
                              "decoy_calls": decoy,
                              "decoy_FDR_pct": round(fdr(real, decoy, decoy_k, fdr_scale), 3)
                              if real else np.nan})
    return pd.DataFrame(rows), pd.DataFrame(per_table)


def compare_region(frame, site: int, origin: Tuple[float, float], centres,
                   sites, *, tile: Tuple[int, int], edge: int = COMPARE_EDGE_PX):
    """The cells of one field that it owns and that lie clear of its border.

    :param frame: cells with ``y`` and ``x`` in the tile frame.
    :param site: the field.
    :param origin: the tile's ``(y, x)`` placement in the well.
    :param centres: ``(N, 2)`` every placed tile's centre in the well.
    :param sites: the site of each centre.
    :param tile: the tile ``(height, width)``.
    :param edge: the border margin.
    :returns: the selected rows, re-indexed.
    """
    from scipy.spatial import cKDTree

    if not len(frame):
        return frame.reset_index(drop=True)
    yy = frame["y"].to_numpy() + origin[0]
    xx = frame["x"].to_numpy() + origin[1]
    _, nearest = cKDTree(centres).query(np.column_stack([yy, xx]))
    inside = ((frame["y"] >= edge) & (frame["y"] < tile[0] - edge)
              & (frame["x"] >= edge) & (frame["x"] < tile[1] - edge)).to_numpy()
    return frame[(np.asarray(sites)[nearest] == site) & inside].reset_index(drop=True)


def _compare_tables_setting(value) -> Dict[str, str]:
    """``ops_compare_tables`` as ``name -> path``.

    :param value: a dict, a JSON object string, or ``name=path`` pairs
        separated by commas.
    :returns: the mapping.
    :raises ValueError: when it names no table.
    """
    if isinstance(value, Mapping):
        out = {str(k): str(v) for k, v in value.items()}
    else:
        text = str(value or "").strip()
        if text.startswith("{"):
            out = {str(k): str(v) for k, v in json.loads(text).items()}
        else:
            out = dict(part.split("=", 1) for part in text.split(",") if "=" in part)
            out = {k.strip(): v.strip() for k, v in out.items()}
    if not out:
        raise ValueError("ops_compare_tables must name at least one external "
                         "call table, as name=path or a JSON object")
    return out


def run_compare_calls(db: str, plate: str, wells: Sequence[str],
                      settings: Mapping[str, Any], library: Sequence[str]
                      ) -> Dict[str, Any]:
    """The compare_calls phase: spaCR's genotype calls against other pipelines'.

    Each external table holds one row per cell with ``well``, ``site``,
    ``y`` and ``x`` in that site's tile frame (the sequencing cycle-1 tile)
    and ``call`` (or ``guide``), empty for an uncalled cell, optionally
    ``cell``. spaCR's side is every nucleus of the decoded fields with its
    call from ``results/ops_decode/calls_<well>.parquet``, decoys included.
    Cells are compared where their field owns them and at least
    :data:`COMPARE_EDGE_PX` from its border; matched units are categorised
    and each category gets a decoy FDR per table.

    :param db: the measurements database.
    :param plate: the plate.
    :param wells: the wells decoded.
    :param settings: read for ``ops_compare_tables``,
        ``ops_compare_match_px``, ``ops_compare_fields`` (``well:site``
        pairs separated by commas, or empty for every site of the tables),
        ``ops_decoy_library_k`` and ``ops_fdr_scale``.
    :param library: the guide barcodes, for the decoy set.
    :returns: the report, with the paths written under
        ``results/ops_compare``.
    :raises ValueError: when a table lacks its columns or the decode's
        calls are missing.
    """
    import pandas as pd

    from .ops_sbs import decoy_library
    from .ops_store import read_table as read_ops_table
    from .tabular import read_table

    settings = resolve_settings(settings)
    rule = decode_rule(settings)
    paths = _compare_tables_setting(settings.get("ops_compare_tables"))
    max_px = _qc_number(settings, "ops_compare_match_px", COMPARE_MATCH_PX)
    decoys = set(decoy_library(library, k=max(rule["decoy_k"], 1)))
    external = {}
    for name, path in paths.items():
        frame = read_table(path, canonicalise=False, report=None)
        if "call" not in frame and "guide" in frame:
            frame = frame.rename(columns={"guide": "call"})
        missing = {"well", "site", "y", "x", "call"} - set(frame.columns)
        if missing:
            raise ValueError(f"ops_compare_tables {name}: {path} lacks columns "
                             f"{sorted(missing)}")
        if "cell" not in frame:
            frame = frame.assign(cell=np.arange(1, len(frame) + 1))
        frame["call"] = frame["call"].where(frame["call"].astype(str).str.len() > 0)
        frame["call"] = frame["call"].where(frame["call"].notna(), None)
        external[name] = frame
    wanted = settings.get("ops_compare_fields")
    if wanted:
        fields = [(w.split(":")[0].strip().upper(), int(w.split(":")[1]))
                  for w in (wanted if isinstance(wanted, (list, tuple))
                            else str(wanted).split(",")) if ":" in str(w)]
    else:
        fields = sorted({(str(w), int(s)) for frame in external.values()
                         for w, s in zip(frame["well"], frame["site"])})
    geometry = read_ops_table(db, "ops_geometry")
    geometry = geometry[geometry["plate"].astype(str) == str(plate)]
    objects = read_ops_table(db, "ops_objects")
    objects = objects[objects["plate"].astype(str) == str(plate)]
    names = ["spaCR"] + list(external)
    units = []
    for well, site in fields:
        calls_path = os.path.join(results_folder(db), f"calls_{well}.parquet")
        if not os.path.isfile(calls_path):
            raise ValueError(f"compare_calls needs {calls_path}; decode well {well} "
                             "with ops_decode_preset rule_a_s2 first")
        spacr = read_table(calls_path, canonicalise=False, report=None)
        placed = geometry[(geometry["well"].astype(str) == well)
                          & (geometry["cycle"] == geometry["cycle"].min())]
        tile = (int(placed["tile_height"].iloc[0]), int(placed["tile_width"].iloc[0]))
        centres = np.column_stack([placed["y"] + tile[0] / 2, placed["x"] + tile[1] / 2])
        origin_row = placed[placed["site"] == site]
        if origin_row.empty:
            continue
        origin = (float(origin_row["y"].iloc[0]), float(origin_row["x"].iloc[0]))
        cells = objects[objects["well"].astype(str) == well]
        mine = pd.DataFrame({"cell": cells["object_id"].to_numpy(),
                             "y": cells["centroid_y"].to_numpy() - origin[0],
                             "x": cells["centroid_x"].to_numpy() - origin[1]})
        mine = mine[(mine["y"] > -30) & (mine["y"] < tile[0] + 30)
                    & (mine["x"] > -30) & (mine["x"] < tile[1] + 30)]
        tables = {"spaCR": compare_region(mine, site, origin, centres,
                                          placed["site"].to_numpy(), tile=tile)}
        hits = spacr[spacr["site"] == site]
        calls = {"spaCR": dict(zip(hits["object_id"].astype(int), hits["mapped_guide"]))}
        for name, frame in external.items():
            part = frame[(frame["well"].astype(str) == well) & (frame["site"] == site)]
            tables[name] = compare_region(part[["cell", "y", "x"]], site, origin,
                                          centres, placed["site"].to_numpy(), tile=tile)
            calls[name] = {int(c): g for c, g in zip(part["cell"], part["call"])
                           if isinstance(g, str)}
        for row in compare_field(tables, calls, decoys, max_px=max_px):
            row.update({"plate": plate, "well": well, "site": site,
                        "field": f"{plate}_{well}_{site}"})
            units.append(row)
    frame = pd.DataFrame(units)
    folder = results_folder(db, "ops_compare")
    if frame.empty:
        return {"units": 0, "fields": len(fields)}
    categories, per_table = compare_summary(frame, names, decoy_k=max(rule["decoy_k"], 1),
                                            fdr_scale=rule["fdr_scale"])
    out = {"units": int(len(frame)), "fields": len(fields),
           "paths": {"units": _write(frame, os.path.join(folder, "units.parquet")),
                     "categories": _write(categories, os.path.join(folder, "categories.csv")),
                     "tables": _write(per_table, os.path.join(folder, "tables.csv"))}}
    return out
