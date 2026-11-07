"""Flat-field maps for optical pooled screen phenotype fields, and two ways to apply them.

A 20X phenotype field is not evenly lit: on plate A of Funk et al. 2022 the
corners receive about half the light of the centre. A cell measured in a
corner then reads dimmer than the same cell in the middle, and in a pooled
screen, where each guide's cells sit at random positions, that adds noise to
every intensity feature and biases the cells near the edge.

THE MAP. Per plate and channel, a few hundred fields are read, block-averaged
in 16 x 16 pixel blocks, and the per-block median over the fields is taken.
Cells move from field to field while the illumination does not, so the
median keeps the illumination and loses the cells. It is smoothed with a
Gaussian (sigma 6 blocks, about 96 field pixels); the nuclear channel, whose
sparse bright nuclei leave a noisier median, is then fitted with a 4th-order
2-D polynomial. The camera and background offset ``b`` is the 0.1th
percentile of every block mean, and the map is ``flat = (smooth - b) /
(centre - b)``: 1 at the centre. It is floored at 0.25, so no pixel is
multiplied by more than 4.

    observed = b + flat * (true - b)        corrected = (observed - b) / flat + b

THE EXACT CORRECTION divides every pixel before measuring
(:func:`correct_field`). It needs the raw images again, which on plate A
was 2.1 TB read and about 58 hours of measurement at six workers.

THE ANALYTIC CORRECTION moves the features already measured
(:func:`correct_features`). A mask pass (:func:`mask_pass`) gives, per cell,
compartment and channel, the mean gain ``g`` of ``1 / flat`` over the
compartment's pixels and the pixel count ``n``. Then:

    location (mean, median, mode, min, max, percentiles)   (v - b) g + b
    integrated                                            (v - b n) g + b n
    spread (std, range, iqr)                              v g
    cv, gini                                              recomputed
    everything else                                       unchanged

It is exact when the flat is constant over the cell, which it nearly is: it
varies by under 2 percent across a 70 pixel cell except in the far corners.
On 24 plate A fields (31,078 cells) checked against the exact correction,
location features moved from a median relative error of 6.9 to 0.10
percent, integrated from 6.9 to 0.01 and spread from 9.0 to 0.42. The mode
of a float image has no such rule, so the 12 ``*_mode_intensity`` features
are corrected as locations and flagged (:func:`flagged_features`).
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

# Block edge, in field pixels, of the block means the maps are built from.
FLAT_BLOCK = 16
# Gaussian sigma, in blocks, of the smoothing.
FLAT_SIGMA = 6.0
# Degree of the 2-D polynomial fitted to the nuclear channel's map.
FLAT_POLYNOMIAL_DEGREE = 4
# Percentile of all block means taken as the offset ``b``.
FLAT_OFFSET_PERCENTILE = 0.1
# The smallest flat value; 0.25 caps the gain at 4.
FLAT_FLOOR = 0.25
# Half the edge, in blocks, of the centre window the map is normalised to.
FLAT_CENTRE_HALF = 9

# Statistics that move like a location: ``(v - b) g + b``.
LOCATION_STATISTICS = (
    "mean_intensity", "median_intensity", "mode_intensity", "min_intensity",
    "max_intensity", "percentile_5", "percentile_10", "percentile_25",
    "percentile_75", "percentile_85", "percentile_95")
# Statistics that move like a spread: ``v g``.
SPREAD_STATISTICS = ("std_intensity", "range_intensity", "iqr_intensity")


def field_blocks(stack: np.ndarray, block: int = FLAT_BLOCK) -> np.ndarray:
    """One field as block means, the input :func:`build_flat_maps` takes.

    :param stack: ``(H, W, C)`` intensities of one field.
    :param block: the block edge in pixels; rows and columns past the last
        whole block are dropped.
    :returns: ``(H // block, W // block, C)`` float32 block means.
    """
    stack = np.asarray(stack, np.float32)
    if stack.ndim == 2:
        stack = stack[..., np.newaxis]
    height = stack.shape[0] // block * block
    width = stack.shape[1] // block * block
    return stack[:height, :width].reshape(
        height // block, block, width // block, block,
        stack.shape[2]).mean((1, 3))


def _polynomial_fit(plane: np.ndarray, degree: int) -> np.ndarray:
    """``plane`` replaced by its least-squares 2-D polynomial of ``degree``."""
    size = plane.shape[0]
    yy, xx = np.mgrid[:size, :size] / (size - 1) * 2 - 1
    terms = np.stack([(xx ** i * yy ** j).ravel()
                      for i in range(degree + 1)
                      for j in range(degree + 1 - i)], 1)
    coefficients, *_ = np.linalg.lstsq(terms, plane.ravel(), rcond=None)
    return (terms @ coefficients).reshape(plane.shape)


def build_flat_maps(blocks: np.ndarray, channels: Sequence[str], *,
                    polynomial_channels: Iterable[str] = ("DAPI",),
                    sigma: float = FLAT_SIGMA,
                    degree: int = FLAT_POLYNOMIAL_DEGREE,
                    offset_percentile: float = FLAT_OFFSET_PERCENTILE,
                    floor: float = FLAT_FLOOR,
                    centre_half: int = FLAT_CENTRE_HALF
                    ) -> Tuple[Dict[str, np.ndarray], Dict[str, dict]]:
    """Per-channel flat-field maps from many fields' block means.

    :param blocks: ``(N, h, w, C)`` block means of ``N`` fields
        (:func:`field_blocks`), square blocks grids.
    :param channels: the name of each of the ``C`` channels.
    :param polynomial_channels: channels whose smoothed median is also
        fitted with a 2-D polynomial; the nuclear channel by default.
    :param sigma: Gaussian sigma in blocks.
    :param degree: the polynomial's degree.
    :param offset_percentile: the percentile of all block means taken as
        the offset ``b``.
    :param floor: the smallest flat value, so the gain is at most
        ``1 / floor``.
    :param centre_half: half the edge, in blocks, of the centre window the
        map is normalised to.
    :returns: ``(maps, method)``: ``{channel: (h, w) float32 flat}``, 1 at
        the centre, and ``{channel: {"how", "offset", "centre", "flat_min",
        "flat_max"}}``.

    This is the method of the plate A pilot (seg-QC ``flat.py``) and of the
    plate B to D maps (``build_flat.py``); rebuilt from the same fields it
    gives the same arrays.
    """
    from scipy import ndimage

    blocks = np.asarray(blocks)
    if blocks.ndim != 4 or blocks.shape[3] != len(channels):
        raise ValueError(
            f"blocks must be (fields, h, w, {len(channels)}); got "
            f"{blocks.shape}")
    wanted = {str(c).upper() for c in polynomial_channels}
    size = blocks.shape[1]
    maps: Dict[str, np.ndarray] = {}
    method: Dict[str, dict] = {}
    for index, channel in enumerate(channels):
        median = np.median(blocks[..., index], 0)
        offset = float(np.percentile(blocks[..., index], offset_percentile))
        smooth = ndimage.gaussian_filter(median, sigma)
        if str(channel).upper() in wanted:
            smooth = _polynomial_fit(smooth, degree)
            how = (f"median, Gaussian sigma {sigma:g} (ds px), then "
                   f"{degree}th-order 2D polynomial fit")
        else:
            how = (f"median, Gaussian sigma {sigma:g} (ds px = "
                   f"{sigma * FLAT_BLOCK:g} field px)")
        centre = smooth[size // 2 - centre_half:size // 2 + centre_half,
                        size // 2 - centre_half:size // 2 + centre_half].mean()
        flat = np.clip((smooth - offset) / (centre - offset), floor, None)
        maps[str(channel)] = flat.astype(np.float32)
        method[str(channel)] = {"how": how, "offset": offset,
                                "centre": float(centre),
                                "flat_min": float(flat.min()),
                                "flat_max": float(flat.max())}
    return maps, method


def choose_flat_fields(pool: Sequence[Tuple[str, int]], count: int,
                       seed: int = 0) -> List[Tuple[str, int]]:
    """The fields a plate's maps are built from: a seeded random sample.

    :param pool: every ``(well, site)`` that may be used, in a fixed order.
    :param count: how many to take; all of them when fewer exist.
    :param seed: the seed of :func:`numpy.random.default_rng`.
    :returns: the chosen ``(well, site)``, in ``pool`` order.

    Random positions over the plate, so each map pixel sees cells and gaps
    alike; ``build_flat.py`` of the plate B to D maps draws the same way.
    """
    rng = np.random.default_rng(int(seed))
    take = min(int(count), len(pool))
    return [pool[i] for i in sorted(rng.choice(len(pool), take,
                                               replace=False))]


def write_flat_maps(folder: str, plate: str, maps: Mapping[str, np.ndarray],
                    method: Mapping[str, dict], fields,
                    extra: Optional[Mapping[str, object]] = None) -> dict:
    """Write a plate's maps once, with their md5 and the fields used.

    :param folder: where ``flat_<plate>_<CH>.npy`` and ``flat_info.json``
        go.
    :param plate: the plate name used in the file names.
    :param maps: ``{channel: flat}`` from :func:`build_flat_maps`.
    :param method: its ``method`` record.
    :param fields: the ``(well, site)`` the maps were built from.
    :param extra: more keys for ``flat_info.json``, such as when and from
        which wells the maps were built.
    :returns: the ``flat_info.json`` record.
    :raises FileExistsError: when the folder already holds maps. A map is
        recorded by md5 in every analysis that used it, so it is never
        overwritten; build into a new folder instead.
    """
    if os.path.isfile(os.path.join(folder, "flat_info.json")) or any(
            name.startswith("flat_") and name.endswith(".npy")
            for name in (os.listdir(folder) if os.path.isdir(folder) else [])):
        raise FileExistsError(
            f"{folder} already holds flat-field maps; they are never "
            "overwritten, so build into a new folder")
    os.makedirs(folder, exist_ok=True)
    digests = {}
    for channel, flat in maps.items():
        path = os.path.join(folder, f"flat_{plate}_{channel}.npy")
        np.save(path, np.asarray(flat, np.float32))
        with open(path, "rb") as handle:
            digests[channel] = hashlib.md5(handle.read()).hexdigest()
    by_well: Dict[str, List[int]] = {}
    for well, site in fields:
        by_well.setdefault(str(well), []).append(int(site))
    record = {"plate": plate, "n_fields": len(list(fields)),
              "fields_by_well": {w: sorted(v) for w, v in by_well.items()},
              "n_wells_used": len(by_well), "method": dict(method),
              "md5": digests, "immutable": True, **dict(extra or {})}
    with open(os.path.join(folder, "fields.json"), "w",
              encoding="utf-8") as handle:
        json.dump([[str(w), int(s)] for w, s in fields], handle)
    # Written last: its presence marks the maps complete.
    with open(os.path.join(folder, "flat_info.json"), "w",
              encoding="utf-8") as handle:
        json.dump(record, handle, indent=1)
    return record


def load_flat_maps(folder: str, channels: Sequence[str]
                   ) -> Tuple[Dict[str, np.ndarray], Dict[str, float], dict]:
    """A plate's maps and offsets from a folder :func:`write_flat_maps` wrote.

    :param folder: holds ``flat_info.json`` and one ``flat_*_<CH>.npy`` per
        channel; the plate A pilot's ``seg_qc`` folder has the same layout.
    :param channels: the channels wanted, matched without regard to case.
    :returns: ``(maps, offsets, info)``: ``{channel: flat}`` as stored,
        ``{channel: b}``, and the info record with each map's md5 (computed
        here when the record lacks it).
    :raises FileNotFoundError: when the folder lacks the record or a map.
    """
    info_path = os.path.join(folder, "flat_info.json")
    if not os.path.isfile(info_path):
        raise FileNotFoundError(
            f"{folder} holds no flat_info.json; build the maps first "
            "(phase ph_flatfield)")
    with open(info_path, encoding="utf-8") as handle:
        info = json.load(handle)
    method = {str(k).upper(): v for k, v in (info.get("method") or {}).items()}
    files = {}
    for name in sorted(os.listdir(folder)):
        if name.startswith("flat_") and name.endswith(".npy"):
            files[name[:-4].rsplit("_", 1)[-1].upper()] = os.path.join(
                folder, name)
    maps, offsets, digests = {}, {}, dict(info.get("md5") or {})
    for channel in channels:
        key = str(channel).upper()
        if key not in files or key not in method:
            raise FileNotFoundError(
                f"{folder} has no flat-field map for channel {channel}")
        maps[str(channel)] = np.load(files[key])
        offsets[str(channel)] = float(method[key]["offset"])
        if key not in {str(k).upper() for k in digests}:
            with open(files[key], "rb") as handle:
                digests[key] = hashlib.md5(handle.read()).hexdigest()
    info["md5"] = digests
    info["folder"] = os.path.abspath(folder)
    return maps, offsets, info


def flat_planes(maps: Mapping[str, np.ndarray], shape: Tuple[int, int]
                ) -> Dict[str, np.ndarray]:
    """Each map brought to the field's pixels, bilinearly.

    :param maps: ``{channel: (h, w) flat}``.
    :param shape: ``(H, W)`` of a field.
    :returns: ``{channel: (H, W) float32 flat}``.
    """
    from scipy import ndimage

    out = {}
    height, width = int(shape[0]), int(shape[1])
    for channel, flat in maps.items():
        flat = np.asarray(flat)
        zoom = (height / flat.shape[0] if flat.shape[0] == flat.shape[1]
                else (height / flat.shape[0], width / flat.shape[1]))
        out[channel] = ndimage.zoom(flat, zoom, order=1)[:height, :width]
    return out


def correct_field(stack: np.ndarray, planes: Mapping[str, np.ndarray],
                  offsets: Mapping[str, float], channels: Sequence[str]
                  ) -> np.ndarray:
    """One field corrected pixel by pixel: ``(I - b) / flat + b``.

    :param stack: ``(H, W, C)`` intensities.
    :param planes: ``{channel: (H, W) flat}`` from :func:`flat_planes`.
    :param offsets: ``{channel: b}``.
    :param channels: the stack's channel names, in order.
    :returns: a float32 copy; the input is not changed.

    The offset is added back so a corrected pixel stays on the camera's
    scale and the background keeps its level; only the illumination
    falloff is divided out.
    """
    out = np.asarray(stack, np.float32).copy()
    for index, channel in enumerate(channels):
        offset = offsets[channel]
        out[..., index] = (out[..., index] - offset) / planes[channel] + offset
    return out


def mask_pass(nuclei: np.ndarray, cells: np.ndarray,
              inverse_planes: Optional[Mapping[str, np.ndarray]] = None):
    """Per-cell geometry of one field's masks, and the mean gain under each.

    :param nuclei: the field's nucleus labels, nucleus ``k`` in cell ``k``.
    :param cells: its cell labels.
    :param inverse_planes: ``{channel: (H, W) 1 / flat}`` in float64, or
        None for the geometry only.
    :returns: one row per cell label: ``label``; ``<c>_px`` per compartment
        (cell, nucleus, cytoplasm); ``cell_cy``/``cell_cx`` and
        ``nucleus_cy``/``nucleus_cx``, the pixel centroids; ``top``,
        ``bottom``, ``left``, ``right``, the cell's box; and
        ``touches_field_edge`` and ``nucleus_touches_field_edge``. With
        planes, also ``<c>_<CH>_g``, the mean of ``1 / flat`` over the
        compartment's pixels.

    A cell touching the field's edge is cut: part of it lies outside the
    field, so its area and integrated intensities are too small. Where a
    neighbouring field holds the whole cell, that copy is the one to keep.
    """
    import pandas as pd
    from scipy import ndimage

    cells = np.asarray(cells).astype(np.int64)
    nuclei = np.asarray(nuclei).astype(np.int64)
    height, width = cells.shape
    size = int(max(cells.max(initial=0), nuclei.max(initial=0))) + 1
    cell_counts = np.bincount(cells.ravel(), minlength=size)
    labels = np.flatnonzero(cell_counts[1:]) + 1
    if labels.size == 0:
        return pd.DataFrame({"label": np.empty(0, np.int64)})
    cell_boxes = ndimage.find_objects(cells.astype(np.int32))
    nucleus_boxes = ndimage.find_objects(nuclei.astype(np.int32))
    rows, cols = np.indices(cells.shape, dtype=np.float64)
    cytoplasm = np.where(nuclei > 0, 0, cells)
    record = {"label": labels.astype(np.int64)}
    for name, mask in (("cell", cells), ("nucleus", nuclei),
                       ("cytoplasm", cytoplasm)):
        flat = mask.ravel()
        count = (cell_counts if name == "cell" else
                 np.bincount(flat, minlength=size)).astype(np.float64)
        record[f"{name}_px"] = count[labels]
        with np.errstate(invalid="ignore", divide="ignore"):
            if name != "cytoplasm":
                record[f"{name}_cy"] = (np.bincount(flat, rows.ravel(), size)
                                        / count)[labels]
                record[f"{name}_cx"] = (np.bincount(flat, cols.ravel(), size)
                                        / count)[labels]
            for channel, inverse in (inverse_planes or {}).items():
                record[f"{name}_{channel}_g"] = (
                    np.bincount(flat, np.asarray(inverse).ravel(), size)
                    / count)[labels]
    top = np.array([cell_boxes[l - 1][0].start for l in labels])
    bottom = np.array([cell_boxes[l - 1][0].stop for l in labels])
    left = np.array([cell_boxes[l - 1][1].start for l in labels])
    right = np.array([cell_boxes[l - 1][1].stop for l in labels])
    record.update(top=top, bottom=bottom, left=left, right=right)
    record["touches_field_edge"] = ((top == 0) | (left == 0)
                                    | (bottom == height) | (right == width))
    edge = np.zeros(len(labels), bool)
    for i, label in enumerate(labels):
        box = nucleus_boxes[label - 1] if label - 1 < len(nucleus_boxes) \
            else None
        if box is not None:
            edge[i] = (box[0].start == 0 or box[1].start == 0
                       or box[0].stop == height or box[1].stop == width)
    record["nucleus_touches_field_edge"] = edge
    return pd.DataFrame(record)


def correction_kind(column: str, channels: Sequence[str]) -> Optional[str]:
    """How the analytic correction moves one feature column.

    :param column: a cell table column, ``<compartment>_<channel>_<stat>``.
    :param channels: the channel names; matched without regard to case.
    :returns: ``"loc"``, ``"spread"``, ``"int"``, ``"cv"``, ``"gini"``, or
        None for a column the correction leaves unchanged.
    """
    parts = str(column).split("_", 2)
    if len(parts) < 3 or parts[1] not in {str(c).lower() for c in channels}:
        return None
    statistic = parts[2]
    if statistic in LOCATION_STATISTICS:
        return "loc"
    if statistic in SPREAD_STATISTICS:
        return "spread"
    if statistic == "integrated_intensity":
        return "int"
    if statistic == "cv_intensity":
        return "cv"
    if statistic == "gini_intensity":
        return "gini"
    return None


def correction_inputs(features: Sequence[str], channels: Sequence[str]
                      ) -> List[str]:
    """The columns :func:`correct_features` reads to correct ``features``.

    :param features: the columns to correct.
    :param channels: the channel names.
    :returns: ``features`` plus the mean and std columns their cv and gini
        are recomputed from, in a stable order.
    """
    out = list(features)
    for column in features:
        if correction_kind(column, channels) in ("cv", "gini"):
            compartment, channel = str(column).split("_")[:2]
            for statistic in ("mean_intensity", "std_intensity"):
                extra = f"{compartment}_{channel}_{statistic}"
                if extra not in out:
                    out.append(extra)
    return out


def correct_features(frame, gains, features: Sequence[str],
                     offsets: Mapping[str, float]):
    """The analytic flat-field correction of measured features.

    :param frame: one row per cell, holding ``features`` and the columns
        :func:`correction_inputs` names.
    :param gains: a frame aligned with ``frame`` holding the
        :func:`mask_pass` columns ``<compartment>_<CH>_g`` and
        ``<compartment>_px``.
    :param features: the columns to return.
    :param offsets: ``{channel: b}``, keyed by the channel names the gain
        columns use.
    :returns: a float32 frame of ``features``; a column the correction does
        not move is returned as measured.

    Each kind moves as the module docstring states. Values are computed in
    float64 and the cv and gini from the corrected mean and std before the
    result is cast, which is what the plate A pilot's ``flat_correct.py``
    did.
    """
    channels = list(offsets)
    by_lower = {str(c).lower(): c for c in channels}
    out = frame[list(features)].copy()
    needed = correction_inputs(features, channels)
    work = {}
    for column in needed:
        kind = correction_kind(column, channels)
        if kind is None or kind in ("cv", "gini"):
            continue
        compartment, lower = column.split("_")[:2]
        channel = by_lower[lower]
        gain = gains[f"{compartment}_{channel}_g"].to_numpy(np.float64)
        offset = float(offsets[channel])
        value = frame[column].to_numpy(np.float64)
        if kind == "loc":
            work[column] = (value - offset) * gain + offset
        elif kind == "spread":
            work[column] = value * gain
        else:
            count = gains[f"{compartment}_px"].to_numpy(np.float64)
            work[column] = (value - offset * count) * gain + offset * count
    for column in features:
        kind = correction_kind(column, channels)
        if kind not in ("cv", "gini"):
            continue
        compartment, lower = column.split("_")[:2]
        before = frame[f"{compartment}_{lower}_mean_intensity"].to_numpy(
            np.float64)
        after = work[f"{compartment}_{lower}_mean_intensity"]
        with np.errstate(invalid="ignore", divide="ignore"):
            if kind == "cv":
                work[column] = work[f"{compartment}_{lower}_std_intensity"] \
                    / after
            else:
                gain = gains[f"{compartment}_{by_lower[lower]}_g"].to_numpy(
                    np.float64)
                work[column] = frame[column].to_numpy(np.float64) * before \
                    * gain / after
    for column in features:
        if column in work:
            out[column] = work[column]
    return out.astype(np.float32)


def flagged_features(features: Sequence[str], channels: Sequence[str]
                     ) -> List[str]:
    """The intensity features the analytic correction cannot get right.

    :param features: the feature columns.
    :param channels: the channel names.
    :returns: the ``*_mode_intensity`` columns. The mode of a float image is
        not a location the gain moves predictably, so it is corrected as
        one and flagged rather than trusted.
    """
    return [c for c in features if correction_kind(c, channels) == "loc"
            and str(c).endswith("_mode_intensity")]
