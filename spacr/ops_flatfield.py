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
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

FLAT_BLOCK = 16
FLAT_SIGMA = 6.0
FLAT_POLYNOMIAL_DEGREE = 4
FLAT_OFFSET_PERCENTILE = 0.1
FLAT_FLOOR = 0.25
FLAT_CENTRE_HALF = 9

LOCATION_STATISTICS = (
    "mean_intensity", "median_intensity", "mode_intensity", "min_intensity",
    "max_intensity", "percentile_5", "percentile_10", "percentile_25",
    "percentile_75", "percentile_85", "percentile_95")
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


def flat_planes(maps: Mapping[str, np.ndarray], shape: Tuple[int, int],
                lookups: Optional[Mapping[str, Mapping[str, object]]] = None
                ) -> Dict[str, np.ndarray]:
    """Each map brought to the field's pixels, bilinearly.

    :param maps: ``{channel: (h, w) flat}``.
    :param shape: ``(H, W)`` of a field.
    :param lookups: ``{CHANNEL: {"lookup", "block", "centre_offset"}}``
        (:func:`flat_lookups`); a channel not named is read by zoom, as
        every map before item 912 is.
    :returns: ``{channel: (H, W) float32 flat}``.
    """
    out = {}
    for channel, flat in maps.items():
        how = dict((lookups or {}).get(str(channel).upper()) or {})
        if how.get("lookup", "zoom_endpoints") == "zoom_endpoints":
            out[channel] = flat_to_field(flat, shape)
        else:
            out[channel] = flat_to_field(flat, shape, lookup=str(how["lookup"]),
                                         block=int(how.get("block", FLAT_BLOCK)),
                                         centre_offset=float(how.get("centre_offset", 8.0))
                                         ).astype(np.float32)
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


#: The fits :func:`fit_nuclear_flat` knows, the first being the default.
#: ``local_quadratic_log`` is item 912's map; the other two are the losing
#: arms of its comparison, kept so their maps can be rebuilt.
NUCLEAR_FITS = ("local_quadratic_log", "gaussian", "local_linear")

#: How a map is read at field pixels (:func:`flat_to_field`): map pixel
#: ``i`` at field pixel ``block * i + centre_offset``, or where
#: ``scipy.ndimage.zoom`` puts it, ``i * (H - 1) / (h - 1)``.
FLAT_LOOKUPS = ("block_centre", "zoom_endpoints")

#: Where a fitted map is evaluated: at each block's own centre, or at the
#: field position the zoom lookup will read it from.
FLAT_MAP_GRIDS = ("block_centre", "lookup")

#: The radius bins, in field pixels from the field centre, of the held-out
#: radial table (:func:`radial_table`).
RADIAL_EDGES = (0, 500, 1000, 1400, 1600, 1800, 1900, 2000, 2100)

_LINEAR_TERMS = ((0, 0), (1, 0), (0, 1))
_QUADRATIC_TERMS = ((0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2))
_DEFAULT_RIDGE = {"local_quadratic_log": 1e-6, "local_linear": 1e-9}


def nuclear_median_grid(cy, cx, values, *, block: int = FLAT_BLOCK,
                        size: Union[int, Tuple[int, int]] = 185) -> Tuple[np.ndarray, np.ndarray]:
    """Per map cell, the median of the nuclei's values and how many there are.

    :param cy: nucleus centroid rows, in field pixels.
    :param cx: nucleus centroid columns.
    :param values: one value per nucleus, such as its mean DNA intensity
        with no flat-field applied.
    :param block: field pixels per map cell along each axis.
    :param size: map cells along each axis; a centroid past the last cell
        counts in it.
    :returns: ``(median, count)``, two ``(size, size)`` float64 arrays;
        ``median`` is NaN where no nucleus fell.

    A nucleus is bright where the light is and its own size, so the median
    over the nuclei of a cell follows the illumination, while the pixel
    median of a DNA image is mostly background between nuclei and falls
    less steeply toward the corners. Non-finite values are left out.
    """
    import pandas as pd

    cy = np.asarray(cy, np.float64)
    cx = np.asarray(cx, np.float64)
    values = np.asarray(values, np.float64)
    keep = np.isfinite(values) & np.isfinite(cy) & np.isfinite(cx)
    rows, cols = _grid_shape(size)
    iy = np.clip((cy[keep] // block).astype(int), 0, rows - 1)
    ix = np.clip((cx[keep] // block).astype(int), 0, cols - 1)
    grouped = pd.DataFrame({"k": iy * cols + ix, "v": values[keep]}).groupby("k").v
    median = np.full(rows * cols, np.nan)
    count = np.zeros(rows * cols)
    medians = grouped.median()
    sizes = grouped.size()
    median[medians.index] = medians.to_numpy()
    count[sizes.index] = sizes.to_numpy()
    return median.reshape(rows, cols), count.reshape(rows, cols)


def _grid_shape(size) -> Tuple[int, int]:
    """``(rows, cols)`` of a map or field size given as one number or a pair."""
    if np.ndim(size) == 0:
        return int(size), int(size)
    rows, cols = size
    return int(rows), int(cols)


def normalise_flat(surface: np.ndarray, offset: float, *, floor: float = 0.10,
                   centre_cells: int = 2 * FLAT_CENTRE_HALF
                   ) -> Tuple[np.ndarray, float]:
    """A smooth surface as a flat-field map: 1 at the centre, floored.

    :param surface: the fitted surface on the camera's scale.
    :param offset: the camera offset ``b``.
    :param floor: the smallest map value, so the gain is at most
        ``1 / floor``.
    :param centre_cells: the edge, in map cells, of the central square
        whose mean is 1 after normalising.
    :returns: ``(flat, centre)``: ``clip((surface - b) / (centre - b),
        floor)`` and the centre's mean on the camera's scale.
    """
    rows, cols = surface.shape[:2]
    half = int(centre_cells) // 2
    centre = surface[rows // 2 - half:rows // 2 + half, cols // 2 - half:cols // 2 + half].mean()
    return np.clip((surface - offset) / (centre - offset), floor, None), float(centre)


def lookup_positions(size: int, field_px: int, *, block: int = FLAT_BLOCK,
                     centre_offset: float = 8.0) -> np.ndarray:
    """Where the zoom lookup reads each map pixel, in map-cell units.

    :param size: map pixels along the axis.
    :param field_px: field pixels along the axis.
    :param block: field pixels per map cell.
    :param centre_offset: field-pixel offset of a cell's centre.
    :returns: ``(i * (field_px - 1) / (size - 1) - centre_offset) / block``
        for every map pixel ``i``: ``scipy.ndimage.zoom`` puts map pixel
        ``i`` at field pixel ``i * (field_px - 1) / (size - 1)``, which on
        a 185-cell map of a 2,960 px field is up to 8 px past the centre of
        the cell it was measured on.
    """
    return (np.arange(size) * (int(field_px) - 1) / (size - 1) - centre_offset) / block


def _local_fit(values: np.ndarray, weights: np.ndarray, terms, sigma: float,
               ridge: float) -> np.ndarray:
    """Per map cell, the weighted least-squares coefficients of ``terms``.

    :param values: the value of every cell (anything where the weight is 0).
    :param weights: the weight of every cell before the Gaussian.
    :param terms: ``(power of x, power of y)`` per term.
    :param sigma: the Gaussian weight sigma in map cells; every moment is
        smoothed with ``mode="constant"``.
    :param ridge: added to the diagonal of each normal-equation system.
    :returns: ``(h, w, len(terms))`` coefficients in map-pixel coordinates.
    """
    from scipy import ndimage

    rows, cols = values.shape[:2]
    yy, xx = np.mgrid[:rows, :cols].astype(np.float64)
    count = len(terms)
    moments = np.zeros((rows, cols, count, count))
    right = np.zeros((rows, cols, count))
    for i, (a1, b1) in enumerate(terms):
        right[..., i] = ndimage.gaussian_filter(weights * values * xx ** a1 * yy ** b1, sigma,
                                                mode="constant")
        for j, (a2, b2) in enumerate(terms):
            moments[..., i, j] = ndimage.gaussian_filter(
                weights * xx ** (a1 + a2) * yy ** (b1 + b2), sigma, mode="constant")
    return np.linalg.solve(moments + ridge * np.eye(count), right[..., None])[..., 0]


def _evaluate(coefficients: np.ndarray, terms, ys: np.ndarray, xs: np.ndarray) -> np.ndarray:
    """The polynomial of each cell's coefficients at ``(ys, xs)``."""
    return sum(coefficients[..., i] * xs ** a1 * ys ** b1 for i, (a1, b1) in enumerate(terms))


def fit_nuclear_flat(median: np.ndarray, count: np.ndarray, offset: float, *,
                     fit: str = "local_quadratic_log", sigma_px: float = 32.0,
                     block: int = FLAT_BLOCK, ridge: Optional[float] = None,
                     floor: float = 0.10, centre_cells: int = 2 * FLAT_CENTRE_HALF,
                     map_grid: str = "block_centre",
                     field_px: Optional[Union[int, Tuple[int, int]]] = None,
                     centre_offset: float = 8.0) -> Tuple[np.ndarray, Dict[str, object]]:
    """A flat-field map from per-cell nuclear medians (item 912).

    :param median: per map cell, the median nuclear intensity
        (:func:`nuclear_median_grid`); NaN where empty.
    :param count: per map cell, the nuclei behind the median; the weight.
    :param offset: the camera offset ``b`` of the channel and plate.
    :param fit: ``'local_quadratic_log'`` (default): per cell, a weighted
        least-squares fit of ``log(clip(median - b, 1))`` on
        ``1, x, y, x^2, xy, y^2`` with weights count times a Gaussian of
        ``sigma_px``; ``'local_linear'``: the same with a plane on the
        medians themselves; ``'gaussian'``: the count-weighted Gaussian
        mean of the medians, empty reach filled from the nearest cell.
    :param sigma_px: the Gaussian sigma in field pixels (32 = 2 cells).
    :param block: field pixels per map cell.
    :param ridge: added to each local normal-equation system; None takes
        1e-6 for the quadratic and 1e-9 for the plane.
    :param floor: the smallest map value.
    :param centre_cells: the central square normalised to 1.
    :param map_grid: ``'block_centre'`` evaluates each cell's fit at the
        cell itself; ``'lookup'`` at the field position the zoom lookup
        reads the cell from (:func:`lookup_positions`), with the
        coefficients of the nearest cell, so a map read by zoom is right
        where the fall is steepest. Local fits only.
    :param field_px: field pixels along each axis, for ``'lookup'``;
        ``size * block`` when None.
    :param centre_offset: field-pixel offset of a cell's centre.
    :returns: ``(flat, info)``: the float32 map and ``{"centre",
        "filled_cells", "ridge"}``.
    :raises ValueError: for an unknown fit or grid, or ``'lookup'`` with
        the Gaussian fit.
    """
    from scipy import ndimage

    if fit not in NUCLEAR_FITS:
        raise ValueError(f"the nuclear flat-field fit must be one of {list(NUCLEAR_FITS)}; got {fit!r}")
    if map_grid not in FLAT_MAP_GRIDS:
        raise ValueError(f"the map grid must be one of {list(FLAT_MAP_GRIDS)}; got {map_grid!r}")
    median = np.asarray(median, np.float64)
    rows, cols = median.shape[:2]
    valid = np.isfinite(median)
    weights = np.where(valid, np.asarray(count, np.float64), 0.0)
    values = np.where(valid, median, 0.0)
    sigma = float(sigma_px) / block
    info: Dict[str, object] = {"filled_cells": 0, "ridge": None}
    if fit == "gaussian":
        if map_grid != "block_centre":
            raise ValueError("the Gaussian nuclear fit has no lookup grid; use map_grid 'block_centre'")
        with np.errstate(invalid="ignore", divide="ignore"):
            surface = (ndimage.gaussian_filter(values * weights, sigma, mode="constant")
                       / ndimage.gaussian_filter(weights, sigma, mode="constant"))
        bad = ~np.isfinite(surface)
        if bad.any():
            index = ndimage.distance_transform_edt(bad, return_distances=False, return_indices=True)
            surface = surface[tuple(index)]
        info["filled_cells"] = int(bad.sum())
    else:
        ridge = _DEFAULT_RIDGE[fit] if ridge is None else float(ridge)
        info["ridge"] = ridge
        terms = _QUADRATIC_TERMS if fit == "local_quadratic_log" else _LINEAR_TERMS
        target = np.log(np.clip(values - offset, 1, None)) if fit == "local_quadratic_log" else values
        coefficients = _local_fit(target, weights, terms, sigma, ridge)
        if map_grid == "lookup":
            px_rows, px_cols = _grid_shape(field_px) if field_px else (rows * block, cols * block)
            ty = lookup_positions(rows, px_rows, block=block, centre_offset=centre_offset)
            tx = lookup_positions(cols, px_cols, block=block, centre_offset=centre_offset)
            ys, xs = np.meshgrid(ty, tx, indexing="ij")
            near_y = np.clip(np.rint(ty).astype(int), 0, rows - 1)
            near_x = np.clip(np.rint(tx).astype(int), 0, cols - 1)
            coefficients = coefficients[near_y[:, None], near_x[None, :]]
        else:
            ys, xs = np.mgrid[:rows, :cols].astype(np.float64)
        surface = _evaluate(coefficients, terms, ys, xs)
        if fit == "local_quadratic_log":
            surface = np.exp(surface) + offset
    flat, centre = normalise_flat(surface, offset, floor=floor, centre_cells=centre_cells)
    info["centre"] = centre
    return flat.astype(np.float32), info


def flat_to_field(flat: np.ndarray, shape: Tuple[int, int], *,
                  lookup: str = "zoom_endpoints", block: int = FLAT_BLOCK,
                  centre_offset: float = 8.0) -> np.ndarray:
    """One map read at every field pixel, bilinearly.

    :param flat: the ``(h, w)`` map.
    :param shape: ``(H, W)`` of the field.
    :param lookup: ``'zoom_endpoints'`` is ``scipy.ndimage.zoom`` (map
        pixel ``i`` at field pixel ``i * (H - 1) / (h - 1)``), how every map
        before item 912 was read; ``'block_centre'`` puts map pixel ``i``
        at field pixel ``block * i + centre_offset``, the centre of the
        block it was measured on, and holds the edge value beyond the
        outermost centres.
    :param block: field pixels per map pixel.
    :param centre_offset: field-pixel offset of a block's centre; 8 is the
        convention the maps of item 912 are built with, 7.5 the centre of
        pixels ``16 i .. 16 i + 15`` that Measure's illumination surface
        uses.
    :returns: the ``(H, W)`` plane, in the map's dtype for
        ``'zoom_endpoints'`` and float64 for ``'block_centre'``.
    :raises ValueError: for an unknown lookup.
    """
    from scipy import ndimage

    height, width = int(shape[0]), int(shape[1])
    flat = np.asarray(flat)
    if lookup == "zoom_endpoints":
        zoom = (height / flat.shape[0] if flat.shape[0] == flat.shape[1]
                else (height / flat.shape[0], width / flat.shape[1]))
        return ndimage.zoom(flat, zoom, order=1)[:height, :width]
    if lookup != "block_centre":
        raise ValueError(f"the flat-field lookup must be one of {list(FLAT_LOOKUPS)}; got {lookup!r}")
    flat = flat.astype(np.float64)

    def axis(length: int, cells: int):
        """Lower cell, upper cell and weight of every field pixel on one axis."""
        u = np.clip((np.arange(length) - centre_offset) / block, 0, cells - 1)
        low = np.floor(u).astype(int)
        return low, np.minimum(low + 1, cells - 1), u - low

    y0, y1, fy = axis(height, flat.shape[0])
    x0, x1, fx = axis(width, flat.shape[1])
    rows = flat[y0] * (1 - fy)[:, None] + flat[y1] * fy[:, None]
    return rows[:, x0] * (1 - fx) + rows[:, x1] * fx


def flat_lookups(info: Optional[Mapping[str, object]]) -> Dict[str, Dict[str, object]]:
    """How each channel's map in a ``flat_info.json`` record is read.

    :param info: the record :func:`load_flat_maps` returns, or None.
    :returns: ``{channel: {"lookup", "block", "centre_offset"}}`` for the
        channels whose method names a lookup; a map recorded without one
        (every map before item 912) is read by zoom and is left out.
    """
    out: Dict[str, Dict[str, object]] = {}
    for channel, method in dict((info or {}).get("method") or {}).items():
        if isinstance(method, Mapping) and method.get("lookup"):
            out[str(channel).upper()] = {"lookup": str(method["lookup"]),
                                         "block": int(method.get("block", FLAT_BLOCK)),
                                         "centre_offset": float(method.get("centre_offset", 8.0))}
    return out


def holdout_fields(fields: Sequence[Tuple[str, int]], count: int = 24, seed: int = 0
                   ) -> List[Tuple[str, int]]:
    """Fields kept out of a nuclear map's build, the same number per well.

    :param fields: every ``(well, site)`` with nuclei.
    :param count: how many to hold out, split evenly over the wells
        (``count // wells`` each, at least one).
    :param seed: the seed of :func:`numpy.random.default_rng`, drawn once
        per well in sorted well order.
    :returns: the chosen ``(well, site)``, sorted.
    """
    by_well: Dict[str, List[int]] = {}
    for well, site in fields:
        by_well.setdefault(str(well), []).append(int(site))
    if not by_well:
        return []
    rng = np.random.default_rng(int(seed))
    each = max(1, int(count) // len(by_well))
    out = []
    for well in sorted(by_well):
        sites = sorted(set(by_well[well]))
        take = min(each, len(sites))
        out += [(well, int(s)) for s in rng.choice(sites, take, replace=False)]
    return sorted(out)


def _field_keys(frame) -> np.ndarray:
    """``"well:site"`` of every row of a nucleus table."""
    return (frame["well"].astype(str) + ":" + frame["site"].astype(str)).to_numpy()


def build_nuclear_flat(nuclei, *, column: str, offset: float,
                       holdout: Sequence[Tuple[str, int]] = (),
                       cy: str = "nucleus_cy", cx: str = "nucleus_cx",
                       block: int = FLAT_BLOCK, size: Optional[Union[int, Tuple[int, int]]] = None,
                       fit: str = "local_quadratic_log", sigma_px: float = 32.0,
                       ridge: Optional[float] = None, floor: float = 0.10,
                       centre_cells: int = 2 * FLAT_CENTRE_HALF,
                       map_grid: str = "block_centre",
                       field_px: Optional[Union[int, Tuple[int, int]]] = None,
                       centre_offset: float = 8.0, lookup: str = "block_centre"
                       ) -> Tuple[np.ndarray, Dict[str, object]]:
    """One plate's nuclear-median flat-field map from its per-nucleus table.

    :param nuclei: one row per nucleus with ``well``, ``site``, ``cy``,
        ``cx`` and ``column``, measured with no flat-field applied.
    :param column: the nuclear mean intensity of the channel.
    :param offset: the plate's camera offset of the channel; required.
    :param holdout: ``(well, site)`` fields left out of the build, for
        :func:`validate_nuclear_flat`.
    :param cy: the centroid row column.
    :param cx: the centroid column column.
    :param block: field pixels per map cell.
    :param size: map cells per axis; ``field_px // block`` when None.
    :param fit: see :func:`fit_nuclear_flat`.
    :param sigma_px: see :func:`fit_nuclear_flat`.
    :param ridge: see :func:`fit_nuclear_flat`.
    :param floor: see :func:`fit_nuclear_flat`.
    :param centre_cells: see :func:`fit_nuclear_flat`.
    :param map_grid: see :func:`fit_nuclear_flat`.
    :param field_px: field pixels per axis; from the largest centroid
        rounded up to whole blocks when None.
    :param centre_offset: see :func:`fit_nuclear_flat`.
    :param lookup: how the map is to be read, recorded with it
        (:func:`flat_to_field`).
    :returns: ``(flat, info)``: the float32 map and its record (offset,
        floor, held-out fields, training and held-out nuclei, nuclei per
        cell, centre value, corner mean, percent of cells at the floor,
        fit, sigma, grid, lookup and package versions).
    :raises ValueError: for a missing offset, column or nucleus.
    """
    import pandas as pd
    import scipy

    if offset is None or not np.isfinite(float(offset)):
        raise ValueError("the nuclear-median flat-field needs the plate's camera offset "
                         "(ops_flatfield_dark); it is never taken as 0")
    missing = [c for c in ("well", "site", cy, cx, column) if c not in nuclei.columns]
    if missing:
        raise ValueError(f"the nucleus table lacks {missing}")
    keys = _field_keys(nuclei)
    held = np.isin(keys, [f"{w}:{s}" for w, s in holdout])
    train = nuclei[~held]
    if not len(train):
        raise ValueError("no nucleus is left to build the flat-field map from")
    if field_px is None:
        largest = float(np.nanmax(train[[cy, cx]].to_numpy(np.float64)))
        field_px = (int(largest) // block + 1) * block
    if size:
        size = _grid_shape(size)
    else:
        size = tuple(n // block for n in _grid_shape(field_px))
    if size[0] == size[1] and np.ndim(field_px) == 0:
        size = size[0]
    median, count = nuclear_median_grid(train[cy].to_numpy(), train[cx].to_numpy(),
                                        train[column].to_numpy(), block=block, size=size)
    flat, fitted = fit_nuclear_flat(median, count, float(offset), fit=fit, sigma_px=sigma_px,
                                    block=block, ridge=ridge, floor=floor,
                                    centre_cells=centre_cells, map_grid=map_grid,
                                    field_px=field_px, centre_offset=centre_offset)
    corner = int(centre_cells)
    corners = float(np.mean([flat[:corner, :corner].mean(), flat[:corner, -corner:].mean(),
                             flat[-corner:, :corner].mean(), flat[-corner:, -corner:].mean()]))
    info: Dict[str, object] = {
        "estimator": "nuclear_median", "column": column, "offset": float(offset),
        "floor": float(floor), "fit": fit, "sigma_px": float(sigma_px), "ridge": fitted["ridge"],
        "block": int(block), "size": size if np.ndim(size) == 0 else list(size),
        "field_px": int(field_px) if np.ndim(field_px) == 0 else [int(n) for n in field_px],
        "map_grid": map_grid,
        "lookup": lookup, "centre_offset": float(centre_offset), "centre_cells": int(centre_cells),
        "held_out_fields": [f"{w}:{s}" for w, s in holdout],
        "n_training_nuclei": int(len(train)), "n_heldout_nuclei": int(held.sum()),
        "nuclei_per_cell": {"min": float(count.min()), "median": float(np.median(count)),
                            "corner_mean": float(count[:corner, :corner].mean())},
        "centre_value": fitted["centre"], "filled_cells": fitted["filled_cells"],
        "corner_mean": corners, "flat_min": float(flat.min()), "flat_max": float(flat.max()),
        "at_floor_pct": float(100 * (flat <= floor + 1e-6).mean()),
        "versions": {"numpy": np.__version__, "scipy": scipy.__version__, "pandas": pd.__version__}}
    return flat, info


def write_nuclear_flat(folder: str, plate: str, channel: str, flat: np.ndarray,
                       info: Mapping[str, object], *, base: Optional[str] = None) -> dict:
    """Write a nuclear map as a complete, never-overwritten maps folder.

    :param folder: the new folder; it must not hold maps yet.
    :param plate: the plate name used in file names.
    :param channel: the channel the map corrects.
    :param flat: the map.
    :param info: its record from :func:`build_nuclear_flat`; also written
        next to the map as ``flat_<plate>_<CH>.json``.
    :param base: a maps folder (:func:`write_flat_maps`) whose other
        channels are copied byte for byte, so the new folder replaces it.
    :returns: the folder's ``flat_info.json`` record. Its ``md5_record``
        flag makes every run that uses these maps keep their md5 in
        ``illumination_md5.json`` (:func:`record_flat_md5`).
    :raises FileExistsError: when ``folder`` already holds maps.
    """
    import shutil

    from . import __version__ as spacr_version

    if os.path.isfile(os.path.join(folder, "flat_info.json")) or any(
            n.startswith("flat_") and n.endswith(".npy")
            for n in (os.listdir(folder) if os.path.isdir(folder) else [])):
        raise FileExistsError(f"{folder} already holds flat-field maps; they are never overwritten, "
                              "so build into a new folder")
    os.makedirs(folder, exist_ok=True)
    channel = str(channel)
    method: Dict[str, dict] = {}
    digests: Dict[str, str] = {}
    copied: Dict[str, str] = {}
    if base:
        _maps, offsets, base_info = load_flat_maps(base, [])
        base_method = {str(k).upper(): v for k, v in (base_info.get("method") or {}).items()}
        for name in sorted(os.listdir(base)):
            if not (name.startswith("flat_") and name.endswith(".npy")):
                continue
            key = name[:-4].rsplit("_", 1)[-1].upper()
            if key == channel.upper() or key not in base_method:
                continue
            target = os.path.join(folder, f"flat_{plate}_{key}.npy")
            shutil.copyfile(os.path.join(base, name), target)
            with open(target, "rb") as handle:
                digests[key] = hashlib.md5(handle.read()).hexdigest()
            method[key] = dict(base_method[key])
            copied[key] = os.path.abspath(os.path.join(base, name))
    path = os.path.join(folder, f"flat_{plate}_{channel}.npy")
    np.save(path, np.asarray(flat, np.float32))
    with open(path, "rb") as handle:
        digests[channel.upper()] = hashlib.md5(handle.read()).hexdigest()
    record = {**dict(info), "md5": digests[channel.upper()], "spacr": spacr_version}
    with open(path[:-4] + ".json", "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=1)
    method[channel.upper()] = {
        "how": f"nuclear median, {info.get('fit')}, sigma {info.get('sigma_px')} px",
        "offset": float(info["offset"]), "centre": info.get("centre_value"),
        "flat_min": info.get("flat_min"), "flat_max": info.get("flat_max"),
        "lookup": info.get("lookup", "block_centre"), "block": int(info.get("block", FLAT_BLOCK)),
        "centre_offset": float(info.get("centre_offset", 8.0)), "estimator": "nuclear_median"}
    flat_info = {"plate": plate, "method": method, "md5": digests, "immutable": True,
                 "md5_record": True, "copied_from": copied,
                 "held_out_fields": list(info.get("held_out_fields") or [])}
    with open(os.path.join(folder, "flat_info.json"), "w", encoding="utf-8") as handle:
        json.dump(flat_info, handle, indent=1)
    return flat_info


def record_flat_md5(run_folder: str, info: Mapping[str, object]) -> Optional[dict]:
    """Keep the md5 of the maps a run used, and refuse a changed map.

    :param run_folder: the run's ``dst_root``.
    :param info: the record :func:`load_flat_maps` returns.
    :returns: the record written or found, or None for maps written before
        item 912 (no ``md5_record`` flag), which are left as they were.
    :raises ValueError: when ``illumination_md5.json`` already names a
        different md5 for a channel; the message gives both.
    """
    if not (info or {}).get("md5_record"):
        return None
    path = os.path.join(run_folder, "illumination_md5.json")
    now = {str(k).upper(): str(v) for k, v in dict(info.get("md5") or {}).items()}
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as handle:
            earlier = json.load(handle)
        for channel, digest in now.items():
            before = earlier.get("md5", {}).get(channel)
            if before is not None and before != digest:
                raise ValueError(
                    f"the {channel} flat-field map changed since this run first used it "
                    f"(recorded md5 {before}, now {digest}); rerun into a new dst_root or "
                    "restore the recorded map")
        merged = {**earlier.get("md5", {}), **now}
        if merged == earlier.get("md5"):
            return earlier
        earlier["md5"] = merged
        record = earlier
    else:
        os.makedirs(run_folder, exist_ok=True)
        record = {"md5": now, "folder": info.get("folder")}
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=1)
    return record


def field_nucleus_means(image: np.ndarray, nuclei: np.ndarray,
                        maps: Mapping[str, np.ndarray], offset: float, *,
                        lookups: Optional[Mapping[str, Mapping[str, object]]] = None):
    """Per nucleus of one field: raw mean, mean gain and exact corrected mean.

    :param image: the field's raw intensities of the channel.
    :param nuclei: its nucleus labels.
    :param maps: ``{name: (h, w) flat}`` to compare.
    :param offset: the camera offset ``b``.
    :param lookups: ``{name: {"lookup", "block", "centre_offset"}}``;
        zoom when a name is missing.
    :returns: a frame with ``label``, ``raw_mean`` and per map ``g_<name>``
        (the mean of ``1 / F`` over the nucleus, the analytic gain) and
        ``exact_<name>`` (the mean of ``(I - b) / F + b``, the exact
        correction), F the map read at the field's pixels.
    """
    import pandas as pd

    image = np.asarray(image, np.float64)
    labels_flat = np.asarray(nuclei).astype(np.int64).ravel()
    n = int(labels_flat.max(initial=0)) + 1
    count = np.bincount(labels_flat, minlength=n).astype(np.float64)
    labels = np.nonzero(count)[0]
    labels = labels[labels > 0]
    record = {"label": labels, "raw_mean": (np.bincount(labels_flat, image.ravel(), n) / count)[labels]}
    for name, flat in maps.items():
        how = dict((lookups or {}).get(name) or {})
        plane = flat_to_field(np.asarray(flat, np.float64), image.shape,
                              lookup=str(how.get("lookup", "zoom_endpoints")),
                              block=int(how.get("block", FLAT_BLOCK)),
                              centre_offset=float(how.get("centre_offset", 8.0))).astype(np.float64)
        record[f"g_{name}"] = (np.bincount(labels_flat, (1.0 / plane).ravel(), n) / count)[labels]
        record[f"exact_{name}"] = (np.bincount(labels_flat, ((image - offset) / plane + offset).ravel(), n)
                                   / count)[labels]
    return pd.DataFrame(record)


def radial_table(frame, versions: Sequence[str], *, centre: Tuple[float, float],
                 edges: Sequence[float] = RADIAL_EDGES, bootstrap: int = 2000,
                 seed: int = 20261109, min_bin_cells: int = 100):
    """Median corrected value per radius ring over the centre ring, with CIs.

    :param frame: one row per held-out nucleus with ``field``, ``cy``,
        ``cx`` and one column per version (as measured, analytic and
        exact per map).
    :param versions: the version columns.
    :param centre: the field centre ``(row, col)`` in pixels.
    :param edges: the ring edges in pixels.
    :param bootstrap: field bootstraps for the 95 % intervals.
    :param seed: the bootstrap seed of :func:`numpy.random.default_rng`.
    :param min_bin_cells: rings with fewer nuclei are not scored in the
        summary.
    :returns: ``(table, summary)``: one row per ring and version
        (``radius_bin``, ``n_cells``, ``n_fields``, ``version``,
        ``median_over_centre``, ``ci_lo``, ``ci_hi``, rounded to 4
        decimals) and per version the largest ring deviation from 1 over
        rings with ``min_bin_cells`` nuclei and the median beyond 1,800 px
        over the median within 500 px.
    """
    import pandas as pd

    X = frame[list(versions)].copy()
    X["field"] = frame["field"].to_numpy()
    X["radius"] = np.hypot(frame["cy"].to_numpy(np.float64) - centre[0],
                           frame["cx"].to_numpy(np.float64) - centre[1])
    X["bin"] = pd.cut(X.radius, list(edges))
    versions = list(versions)

    def ratios(part):
        """Median per ring over the first ring's median."""
        medians = part.groupby("bin", observed=False)[versions].median()
        return medians / medians.iloc[0]

    point = ratios(X)
    rng = np.random.default_rng(int(seed))
    by_field = {f: x for f, x in X.groupby("field")}
    names = list(by_field)
    draws = np.array([ratios(pd.concat([by_field[f] for f in rng.choice(names, len(names))])).to_numpy()
                      for _ in range(int(bootstrap))]) if bootstrap else None
    rows = []
    for i, ring in enumerate(point.index):
        sub = X[X.bin == ring]
        for j, version in enumerate(versions):
            if draws is not None:
                lo, hi = np.nanpercentile(draws[:, i, j], [2.5, 97.5])
            else:
                lo = hi = np.nan
            rows.append({"radius_bin": f"{int(ring.left)}-{int(ring.right)}", "n_cells": len(sub),
                         "n_fields": sub.field.nunique(), "version": version,
                         "median_over_centre": round(float(point.iloc[i, j]), 4),
                         "ci_lo": round(float(lo), 4), "ci_hi": round(float(hi), 4)})
    table = pd.DataFrame(rows)
    summary = {}
    for version in versions:
        scored = table[(table.version == version) & (table.n_cells >= min_bin_cells)]
        summary[version] = {
            "max_abs_dev_bins_ge100_cells": round(float((scored.median_over_centre - 1).abs().max()), 4),
            "beyond_1800_whole": round(float(X[X.radius > 1800][version].median()
                                             / X[X.radius <= 500][version].median()), 4)}
    return table, summary


def validate_nuclear_flat(nuclei, fields: Mapping[Tuple[str, int], Tuple[np.ndarray, np.ndarray]],
                          maps: Mapping[str, np.ndarray], *, column: str, offset: float,
                          lookups: Optional[Mapping[str, Mapping[str, object]]] = None,
                          gain_column: Optional[str] = None, cy: str = "nucleus_cy",
                          cx: str = "nucleus_cx", edges: Sequence[float] = RADIAL_EDGES,
                          bootstrap: int = 2000, seed: int = 20261109, min_bin_cells: int = 100,
                          centre: Optional[Tuple[float, float]] = None):
    """Held-out check of flat-field maps by exact re-measurement.

    :param nuclei: the per-nucleus table (``well``, ``site``, ``label``,
        ``cy``, ``cx``, ``column``).
    :param fields: ``{(well, site): (raw image, nucleus labels)}`` of the
        held-out fields; they must not have been used to build the maps.
    :param maps: ``{name: flat}``, the new map and any earlier ones.
    :param column: the run's nuclear mean of the channel, no flat applied.
    :param offset: the camera offset ``b``.
    :param lookups: see :func:`field_nucleus_means`.
    :param gain_column: the run's stored gain of the first map, checked
        against the recomputed one.
    :param cy: the centroid row column.
    :param cx: the centroid column column.
    :param edges: see :func:`radial_table`.
    :param bootstrap: see :func:`radial_table`.
    :param seed: see :func:`radial_table`.
    :param min_bin_cells: see :func:`radial_table`.
    :param centre: the field centre; the image centre when None.
    :returns: ``(cells, table, summary)``: per held-out nucleus the run
        value, analytic ``(v - b) g + b`` and exact values per map; the
        radial table; and the summary with sanity rows (raw mean vs run
        value correlation and median relative difference, and the largest
        difference between recomputed and stored gains).
    :raises ValueError: when a field has no nucleus in the table.
    """
    import pandas as pd

    parts, shape = [], None
    for (well, site), (image, labels) in sorted(fields.items()):
        part = field_nucleus_means(image, labels, maps, offset, lookups=lookups)
        part.insert(0, "site", int(site))
        part.insert(0, "well", str(well))
        parts.append(part)
        shape = np.shape(image)
    measured = pd.concat(parts, ignore_index=True)
    table = nuclei.copy()
    table["well"] = table["well"].astype(str)
    table["site"] = table["site"].astype(int)
    merged = table.merge(measured, on=["well", "site", "label"])
    if not len(merged):
        raise ValueError("no held-out nucleus is in the nucleus table")
    value = merged[column].to_numpy(np.float64)
    columns = {"as measured": value}
    for name in maps:
        columns[f"analytic_{name}"] = (value - offset) * merged[f"g_{name}"].to_numpy() + offset
        columns[f"exact_{name}"] = merged[f"exact_{name}"].to_numpy()
    cells = pd.DataFrame(columns)
    cells["field"] = (merged.well + "_" + merged.site.astype(str)).to_numpy()
    cells["cy"] = merged[cy].to_numpy()
    cells["cx"] = merged[cx].to_numpy()
    if centre is None:
        centre = (shape[0] / 2, shape[1] / 2)
    radial, summary = radial_table(cells, list(columns), centre=centre, edges=edges,
                                   bootstrap=bootstrap, seed=seed, min_bin_cells=min_bin_cells)
    sanity = {"cells": int(len(merged)), "fields": int(merged.groupby(["well", "site"]).ngroups),
              "r_raw_vs_run": float(np.corrcoef(merged.raw_mean, value)[0, 1]),
              "median_abs_rel_raw_vs_run": float(np.median(np.abs(merged.raw_mean / value - 1)))}
    if gain_column and gain_column in merged:
        first = next(iter(maps))
        sanity["max_abs_diff_gain_vs_run"] = float(np.abs(merged[f"g_{first}"] - merged[gain_column]).max())
    summary = {"sanity": sanity,
               "matching": ("every version is scored on the same held-out nuclei of the same fields, none "
                            "used to build any map, with the same rings, centre ring and bootstrap draws"),
               **summary}
    return cells, radial, summary


#: The standalone phases of item 912: build a nuclear-median map per plate,
#: then check it on the held-out fields.
FLAT_PHASES = ("flat_nuclear", "flat_validate")


def _per_plate(value, plate: str):
    """``value[plate]`` for a ``{plate: value}`` mapping, else ``value``."""
    if isinstance(value, Mapping):
        return value.get(plate, value.get(str(plate)))
    return value


def _plate_offset(settings: Mapping[str, object], plate: str, channel: str) -> Tuple[float, str]:
    """The plate's camera offset of ``channel`` and where it came from.

    ``ops_flatfield_dark`` (a number, or ``{plate: number}``) wins; else
    the offset recorded for the channel in the ``ops_flatfield_maps``
    folder's ``flat_info.json``, measured when that plate's maps were built.
    """
    value = _per_plate(settings.get("ops_flatfield_dark"), plate)
    if value is not None:
        return float(value), "ops_flatfield_dark"
    base = settings.get("ops_flatfield_maps")
    if base and os.path.isfile(os.path.join(str(base), "flat_info.json")):
        with open(os.path.join(str(base), "flat_info.json"), encoding="utf-8") as handle:
            method = {str(k).upper(): v for k, v in (json.load(handle).get("method") or {}).items()}
        if channel.upper() in method and method[channel.upper()].get("offset") is not None:
            return float(method[channel.upper()]["offset"]), os.path.join(str(base), "flat_info.json")
    raise ValueError(f"plate {plate}: the nuclear-median flat-field needs the {channel} camera offset; "
                     "set ops_flatfield_dark or point ops_flatfield_maps at the plate's maps")


def _nuclear_column(channel: str) -> str:
    """The cell-table column of a channel's nuclear mean intensity."""
    return f"nucleus_{str(channel).lower()}_mean_intensity"


def nucleus_table(settings: Mapping[str, object], plate: str, channel: str):
    """One plate's per-nucleus table for the nuclear-median map.

    :param settings: read for ``ops_flatfield_nuclei`` (a table, a folder of
        per-well tables, or ``{plate: either}``) and ``dst_root``.
    :param plate: the plate.
    :param channel: the nuclear channel.
    :returns: one row per nucleus with ``well``, ``site``, ``label``,
        ``nucleus_cy``, ``nucleus_cx`` and the channel's nuclear mean.
        With no ``ops_flatfield_nuclei``, the run's own tables are joined
        on ``site`` and ``label``: the mask pass (``ops_maskpass_<well>``,
        centroids and stored gains) and the measured cells
        (``ops_phenotype_cells_<well>``, measured with no flat applied).
    :raises ValueError: when no table is found or a column is missing.
    """
    import glob

    import pandas as pd

    from .tabular import read_table, table_columns

    column = _nuclear_column(channel)
    source = _per_plate(settings.get("ops_flatfield_nuclei"), plate)
    frames = []
    if source:
        paths = (sorted(glob.glob(os.path.join(str(source), "*.parquet"))) if os.path.isdir(str(source))
                 else [str(source)])
        for path in paths:
            frames.append(read_table(path, canonicalise=False, report=None))
    else:
        root = str(settings.get("dst_root") or "")
        for path in sorted(glob.glob(os.path.join(root, "ops_maskpass_*.parquet"))):
            well = os.path.basename(path)[len("ops_maskpass_"):-len(".parquet")]
            cells_path = os.path.join(root, f"ops_phenotype_cells_{well}.parquet")
            if not os.path.isfile(cells_path):
                continue
            wanted = [c for c in ("site", "label", "nucleus_cy", "nucleus_cx", f"nucleus_{channel}_g")
                      if c in table_columns(path, canonicalise=False)]
            masks = read_table(path, canonicalise=False, report=None, columns=wanted)
            cells = read_table(cells_path, canonicalise=False, report=None, columns=["site", "label", column])
            joined = masks.merge(cells, on=["site", "label"])
            joined.insert(0, "well", well)
            frames.append(joined)
    if not frames:
        raise ValueError(f"plate {plate}: no per-nucleus table found; set ops_flatfield_nuclei or run "
                         "ph_measure and ph_masks into dst_root first")
    table = pd.concat(frames, ignore_index=True)
    missing = [c for c in ("well", "site", "label", "nucleus_cy", "nucleus_cx", column)
               if c not in table.columns]
    if missing:
        raise ValueError(f"plate {plate}: the per-nucleus table lacks {missing}")
    table["well"] = table["well"].astype(str)
    return table


def _holdout(settings: Mapping[str, object], plate: str, table) -> List[Tuple[str, int]]:
    """The held-out fields: listed, or drawn per well (:func:`holdout_fields`)."""
    listed = _per_plate(settings.get("ops_flatfield_holdout_fields"), plate)
    if listed:
        if isinstance(listed, str):
            listed = [p for p in listed.replace(";", ",").split(",") if p.strip()]
        return [(str(x).split(":")[0].strip(), int(str(x).split(":")[1])) for x in listed]
    fields = sorted({(str(w), int(s)) for w, s in zip(table["well"], table["site"])})
    return holdout_fields(fields, int(settings.get("ops_flatfield_holdout_n") or 24),
                          int(settings.get("ops_flatfield_seed") or 0))


def _nuclear_options(settings: Mapping[str, object]) -> Dict[str, object]:
    """The keyword arguments of :func:`build_nuclear_flat` the settings give."""
    ridge = settings.get("ops_flatfield_ridge")
    field_px = settings.get("ops_flatfield_field_px")
    options = {"block": int(settings.get("ops_flatfield_block_px") or FLAT_BLOCK),
               "fit": str(settings.get("ops_flatfield_fit") or NUCLEAR_FITS[0]),
               "sigma_px": float(settings.get("ops_flatfield_sigma_px") or 32.0),
               "ridge": None if ridge is None else float(ridge),
               "floor": float(settings.get("ops_flatfield_floor") or 0.10),
               "centre_cells": int(settings.get("ops_flatfield_centre_cells") or 18),
               "map_grid": str(settings.get("ops_flatfield_map_grid") or FLAT_MAP_GRIDS[0]),
               "field_px": None if field_px is None else int(field_px),
               "centre_offset": float(settings.get("ops_flatfield_centre_offset", 8.0)),
               "lookup": str(settings.get("ops_flatfield_lookup") or FLAT_LOOKUPS[0])}
    if options["lookup"] not in FLAT_LOOKUPS:
        raise ValueError(f"ops_flatfield_lookup must be one of {list(FLAT_LOOKUPS)}; got {options['lookup']!r}")
    return options


def _nuclear_folder(settings: Mapping[str, object], plate: str) -> str:
    """Where a plate's nuclear maps go: ``ops_flatfield_nuclear_maps``/<plate>."""
    root = settings.get("ops_flatfield_nuclear_maps") or os.path.join(
        str(settings.get("dst_root") or ""), "flatfield_nuclear")
    return os.path.join(str(root), str(plate))


def _field_image(folder: str, well: str, site: int) -> np.ndarray:
    """A held-out field's raw image of the channel, ``*<well>_<site>.npy``."""
    import glob

    found = sorted(glob.glob(os.path.join(folder, f"*{well}_{site}.npy")))
    exact = [p for p in found if os.path.basename(p)[:-4].split("_")[-2:] == [str(well), str(site)]]
    if not exact:
        raise FileNotFoundError(f"no image of held-out field {well}:{site} in {folder}")
    return np.load(exact[0]).astype(np.float64)


def run_flat_phases(settings: Mapping[str, object], phases: Sequence[str] = FLAT_PHASES,
                    report=None) -> Dict[str, object]:
    """Build and check per-plate nuclear-median flat-field maps (item 912).

    ``flat_nuclear`` builds each plate's map from its per-nucleus table
    (:func:`nucleus_table`, :func:`build_nuclear_flat`), leaving the
    held-out fields out, and writes it with its sidecar record into a new
    folder (:func:`write_nuclear_flat`) that also carries the plate's other
    channel maps from ``ops_flatfield_maps``; point ``ops_flatfield_maps``
    at that folder to use it. ``flat_validate`` re-measures the held-out
    nuclei exactly and analytically with the new map and the earlier one
    (:func:`validate_nuclear_flat`) and writes the radial table.

    :param settings: the OPS settings. Read here: ``plate`` (one plate or a
        comma-separated list), ``dst_root``, ``ops_flatfield_nuclear_channel``
        (required), ``ops_flatfield_nuclei``, ``ops_flatfield_dark``,
        ``ops_flatfield_maps``, ``ops_flatfield_nuclear_maps``, the fit
        settings ``ops_flatfield_block_px``, ``ops_flatfield_fit``,
        ``ops_flatfield_sigma_px``, ``ops_flatfield_ridge``,
        ``ops_flatfield_floor``, ``ops_flatfield_centre_cells``,
        ``ops_flatfield_map_grid``, ``ops_flatfield_field_px``,
        ``ops_flatfield_centre_offset``, ``ops_flatfield_lookup``, the hold-out
        ``ops_flatfield_holdout_fields``, ``ops_flatfield_holdout_n``,
        ``ops_flatfield_seed``, and for the check
        ``ops_flatfield_holdout_images`` (a folder, or ``{plate: folder}``,
        of ``*<well>_<site>.npy`` raw images of the channel),
        ``ops_flatfield_holdout_masks`` (the run folder holding
        ``<well>/phenotype_masks/site-<n>.npz``; ``dst_root`` when empty),
        ``ops_flatfield_compare_maps`` (``{name: .npy}`` read by zoom),
        ``ops_flatfield_radial_edges``, ``ops_flatfield_bootstrap``,
        ``ops_flatfield_bootstrap_seed`` and ``ops_flatfield_min_bin_cells``.
    :param phases: ``"flat_nuclear"``, ``"flat_validate"`` or both.
    :param report: called with progress lines.
    :returns: ``{"plates": {plate: {"folder", "md5", ...}}}``.
    :raises ValueError: for a missing channel, offset or table.
    """
    from .ops_settings import ops_defaults
    from .tabular import write_table

    settings = ops_defaults(dict(settings))
    say = report or (lambda text: None)
    channel = settings.get("ops_flatfield_nuclear_channel")
    if not channel:
        raise ValueError("the nuclear-median flat-field needs ops_flatfield_nuclear_channel, the DNA channel")
    channel = str(channel)
    column = _nuclear_column(channel)
    plates = [p.strip() for p in str(settings.get("plate") or "").split(",") if p.strip()] or [""]
    options = _nuclear_options(settings)
    out: Dict[str, object] = {}
    for plate in plates:
        folder = _nuclear_folder(settings, plate or "plate")
        offset, offset_source = _plate_offset(settings, plate, channel)
        table = nucleus_table(settings, plate, channel)
        record: Dict[str, object] = {"folder": folder}
        if "flat_nuclear" in phases:
            held = _holdout(settings, plate, table)
            flat, info = build_nuclear_flat(table, column=column, offset=offset, holdout=held, **options)
            info["offset_source"] = offset_source
            base = settings.get("ops_flatfield_maps")
            base = str(base) if base and os.path.isfile(os.path.join(str(base), "flat_info.json")) else None
            written = write_nuclear_flat(folder, plate or "plate", channel, flat, info, base=base)
            record.update(md5=written["md5"][channel.upper()], corner_mean=info["corner_mean"],
                          training_nuclei=info["n_training_nuclei"], held_out=len(held))
            say(f"flat_nuclear {plate}: {info['n_training_nuclei']} nuclei, corner {info['corner_mean']:.3f}, "
                f"md5 {record['md5']}")
        if "flat_validate" in phases:
            maps, _offsets, info = load_flat_maps(folder, [channel])
            sidecar = os.path.join(folder, f"flat_{plate or 'plate'}_{channel}.json")
            with open(sidecar, encoding="utf-8") as handle:
                built = json.load(handle)
            held = [(x.split(":")[0], int(x.split(":")[1])) for x in built["held_out_fields"]]
            compare = {"nuclear": maps[channel]}
            lookups = {"nuclear": flat_lookups(info).get(channel.upper(), {})}
            base = settings.get("ops_flatfield_maps")
            if base and os.path.isfile(os.path.join(str(base), "flat_info.json")) \
                    and os.path.abspath(str(base)) != os.path.abspath(folder):
                earlier, _o, earlier_info = load_flat_maps(str(base), [channel])
                compare["earlier"] = earlier[channel]
                lookups["earlier"] = flat_lookups(earlier_info).get(channel.upper(), {})
            for name, path in dict(settings.get("ops_flatfield_compare_maps") or {}).items():
                compare[str(name)] = np.load(str(path))
            images = _per_plate(settings.get("ops_flatfield_holdout_images"), plate)
            if not images:
                raise ValueError("flat_validate needs ops_flatfield_holdout_images, the raw images of the "
                                 "held-out fields")
            masks_root = str(_per_plate(settings.get("ops_flatfield_holdout_masks"), plate)
                             or settings.get("dst_root") or "")
            fields = {}
            for well, site in held:
                with np.load(os.path.join(masks_root, well, "phenotype_masks", f"site-{site}.npz")) as saved:
                    labels = saved["nuclei"]
                fields[(well, site)] = (_field_image(str(images), well, site), labels)
            gain = f"nucleus_{channel}_g" if "earlier" in compare else None
            ordered = ({"earlier": compare.pop("earlier"), **compare} if "earlier" in compare else compare)
            cells, radial, summary = validate_nuclear_flat(
                table, fields, ordered, column=column, offset=offset, lookups=lookups, gain_column=gain,
                edges=[float(e) for e in settings.get("ops_flatfield_radial_edges") or RADIAL_EDGES],
                bootstrap=int(settings.get("ops_flatfield_bootstrap", 2000)),
                seed=int(settings.get("ops_flatfield_bootstrap_seed", 20261109)),
                min_bin_cells=int(settings.get("ops_flatfield_min_bin_cells", 100)))
            target = os.path.join(folder, "validation")
            os.makedirs(target, exist_ok=True)
            write_table(cells, os.path.join(target, "heldout_nuclei.parquet"), canonicalise=False)
            write_table(radial, os.path.join(target, "radial.csv"), canonicalise=False)
            with open(os.path.join(target, "radial_summary.json"), "w", encoding="utf-8") as handle:
                json.dump(summary, handle, indent=1)
            record["validation"] = summary
            say(f"flat_validate {plate}: largest ring deviation "
                f"{summary['exact_nuclear']['max_abs_dev_bins_ge100_cells']}")
        out[plate] = record
    return {"plates": out}
