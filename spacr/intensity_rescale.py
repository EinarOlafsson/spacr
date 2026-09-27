"""Auditable intensity scaling for merged measurement fields.

The measurement worker stores merged arrays as ``uint16`` while label planes
must remain integer identities.  This module makes the conversion decision a
plate-level, serialisable plan so every worker uses the same scale and the
decision can be written into ``measurements.db``.
"""
from __future__ import annotations

import os
from typing import Any, Dict, Iterable, Mapping, Tuple

import numpy as np

from . import schema


UINT16_MAX = float(np.iinfo(np.uint16).max)
PLAN_SETTINGS_KEY = "_intensity_rescale_plan"
MASK_DIM_KEYS = tuple(f"{role}_mask_dim" for role in schema.SEGMENTED_ROLES)


def mask_planes(data: np.ndarray, settings: Mapping[str, Any]) -> set[int]:
    """Return last-axis planes that contain labels rather than intensities.

    :param data: merged field array whose last axis contains image planes.
    :param settings: resolved mask-dimension settings for segmented roles.
    """
    n_planes = int(data.shape[-1])
    found: set[int] = set()
    for key in MASK_DIM_KEYS:
        value = settings.get(key)
        if value is None:
            continue
        try:
            value = int(value)
        except (TypeError, ValueError):
            continue
        if 0 <= value < n_planes:
            found.add(value)
    return found


def signal_max(data: np.ndarray,
               settings: Mapping[str, Any]) -> Tuple[float, bool]:
    """Return the largest finite intensity and whether one exists.

    :param data: merged field array whose last axis contains image planes.
    :param settings: resolved mask-dimension settings used to exclude label
        planes from the intensity scan.
    """
    planes = [index for index in range(int(data.shape[-1]))
              if index not in mask_planes(data, settings)]
    if not planes:
        return 0.0, False
    signal = np.asarray(data[..., planes])
    finite = np.isfinite(signal)
    if not finite.any():
        return 0.0, True
    return max(0.0, float(np.max(signal[finite]))), True


def _plate_id(filename: str, settings: Mapping[str, Any]) -> str:
    """Parse and return the plate identifier encoded in a field filename.

    :param filename: Field filename whose schema metadata is parsed.
    :param settings: Settings supplying the optional ``timelapse`` mode.
    :returns: Parsed plate identifier.
    """
    field = schema.parse_field_stem(
        filename, timelapse=bool(settings.get("timelapse", False)))
    return field.plateID


def _kind(dtype: np.dtype, top: float, has_intensity: bool) -> str:
    """Classify the conversion needed for an image's intensity values.

    :param dtype: NumPy data type of the image array.
    :param top: Largest finite, non-negative intensity in the array.
    :param has_intensity: Whether the image contains an intensity plane.
    :returns: ``no_intensity``, ``identity``, ``fixed_normalized``, or
        ``raw`` according to the conversion the image requires.
    """
    if not has_intensity:
        return "no_intensity"
    if top == 0.0:
        return "identity"
    if np.issubdtype(dtype, np.floating) and top <= 1.0:
        return "fixed_normalized"
    return "raw"


def build_plate_plan(src: os.PathLike | str, filenames: Iterable[str],
                     settings: Mapping[str, Any]) -> Dict[str, Any]:
    """Inspect all fields and return a JSON/pickle-safe plate scaling plan.

    :param src: directory containing the merged array files.
    :param filenames: merged-array filenames to inspect as one plate set.
    :param settings: resolved parsing and mask-plane settings.

    Raw-valued fields on a plate share ``65535 / plate_max`` when any one of
    them exceeds the uint16 ceiling.  Normalised floating-point fields retain
    their well-defined fixed conversion of ``x65535``.  A file that cannot be
    inspected is left in ``failures`` and its worker will make (and record) a
    per-field fallback decision.
    """
    root = os.fspath(src)
    inspected: Dict[str, Dict[str, Any]] = {}
    failures: Dict[str, str] = {}
    plate_maxima: Dict[str, float] = {}

    for filename in sorted(set(os.fspath(name) for name in filenames)):
        path = os.path.join(root, filename)
        try:
            data = np.load(path, mmap_mode="r")
            if data.ndim < 3:
                raise ValueError(
                    f"expected a merged array with a channel axis, got {data.shape}")
            plate = _plate_id(filename, settings)
            top, has_intensity = signal_max(data, settings)
            kind = _kind(data.dtype, top, has_intensity)
            stat = os.stat(path)
            inspected[filename] = {
                "plateID": plate,
                "original_dtype": str(data.dtype),
                "original_intensity_max": top,
                "kind": kind,
                "size_bytes": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
            }
            if kind == "raw":
                plate_maxima[plate] = max(plate_maxima.get(plate, 0.0), top)
        except Exception as exc:
            failures[filename] = f"{type(exc).__name__}: {exc}"

    fields: Dict[str, Dict[str, Any]] = {}
    for filename, item in inspected.items():
        kind = item["kind"]
        plate_top = plate_maxima.get(item["plateID"], 0.0)
        if kind == "fixed_normalized":
            factor, scope, comparable = UINT16_MAX, kind, True
        elif kind == "raw" and plate_top > UINT16_MAX:
            factor, scope, comparable = UINT16_MAX / plate_top, "plate", True
        elif kind == "no_intensity":
            factor, scope, comparable = 1.0, kind, True
        else:
            factor, scope, comparable = 1.0, "identity", True
        fields[filename] = {
            **item,
            "rescale_factor": float(factor),
            "rescale_scope": scope,
            "plate_intensity_max": float(plate_top),
            "comparable_within_plate": bool(comparable),
        }

    return {
        "version": 1,
        "fields": fields,
        "failures": failures,
        "plates": {plate: float(top) for plate, top in plate_maxima.items()},
    }


def fallback_record(data: np.ndarray, filename: str,
                    settings: Mapping[str, Any]) -> Dict[str, Any]:
    """Return the explicitly non-comparable per-field fallback decision.

    :param data: loaded merged field array to characterize.
    :param filename: field filename used to recover the plate identifier.
    :param settings: resolved parsing and mask-plane settings.
    """
    top, has_intensity = signal_max(data, settings)
    kind = _kind(data.dtype, top, has_intensity)
    if kind == "fixed_normalized":
        factor = UINT16_MAX
        comparable = True
        scope = kind
    elif kind == "raw" and top > UINT16_MAX:
        factor = UINT16_MAX / top
        comparable = False
        scope = "field_fallback"
    elif kind == "no_intensity":
        factor, comparable, scope = 1.0, True, kind
    else:
        factor, comparable, scope = 1.0, False, "field_fallback"
    try:
        plate = _plate_id(filename, settings)
    except Exception:
        plate = "error"
    return {
        "plateID": plate,
        "original_dtype": str(data.dtype),
        "original_intensity_max": float(top),
        "kind": kind,
        "rescale_factor": float(factor),
        "rescale_scope": scope,
        "plate_intensity_max": None,
        "comparable_within_plate": bool(comparable),
    }


def resolve_record(data: np.ndarray, filename: str,
                   settings: Mapping[str, Any]) -> Dict[str, Any]:
    """Resolve and verify one worker's record against the precomputed plan.

    :param data: loaded merged field array handled by the worker.
    :param filename: field filename used to locate its planned record.
    :param settings: resolved settings containing the optional plate plan.
    """
    plan = settings.get(PLAN_SETTINGS_KEY)
    if not isinstance(plan, dict) or filename in plan.get("failures", {}):
        return fallback_record(data, filename, settings)
    top, has_intensity = signal_max(data, settings)
    kind = _kind(data.dtype, top, has_intensity)
    try:
        plate = _plate_id(filename, settings)
    except Exception:
        return fallback_record(data, filename, settings)
    plate_top = plan.get("plates", {}).get(plate)

    if kind == "fixed_normalized":
        factor, scope, comparable = UINT16_MAX, kind, True
    elif kind == "no_intensity":
        factor, scope, comparable = 1.0, kind, True
    elif plate_top is None or top > float(plate_top) * (1.0 + 1e-12):
        return fallback_record(data, filename, settings)
    elif float(plate_top) > UINT16_MAX:
        factor, scope, comparable = UINT16_MAX / float(plate_top), "plate", True
    else:
        factor, scope, comparable = 1.0, "identity", True
    return {
        "plateID": plate,
        "original_dtype": str(data.dtype),
        "original_intensity_max": float(top),
        "kind": kind,
        "rescale_factor": float(factor),
        "rescale_scope": scope,
        "plate_intensity_max": float(plate_top) if plate_top is not None else None,
        "comparable_within_plate": bool(comparable),
    }


def needs_warning(factor: float) -> bool:
    """Whether a conversion factor is neither identity nor fixed [0, 1].

    :param factor: multiplicative intensity conversion factor to classify.
    """
    return not (np.isclose(float(factor), 1.0)
                or np.isclose(float(factor), UINT16_MAX))


CALIBRATION_SETTINGS_KEY = "_intensity_calibration_plan"
CALIBRATION_STATISTICS = ("foreground", "median")


def _calibration_wells(settings: Mapping[str, Any]) -> Dict[Tuple[str, str], str]:
    """Return the reference wells as ``{(rowID, columnID): name}``.

    :param settings: settings holding ``intensity_calibration_wells``, a list
        of well names or one comma-separated string of them.
    :raises ValueError: when no well is named or a name is not a well.
    """
    raw = settings.get("intensity_calibration_wells") or []
    if isinstance(raw, str):
        raw = raw.split(",")
    wells: Dict[Tuple[str, str], str] = {}
    for name in raw:
        text = str(name).strip()
        if not text:
            continue
        try:
            wells[schema.parse_well(text)] = text
        except schema.WellParseError as exc:
            raise ValueError(
                f"Setting: intensity_calibration_wells has {text!r}, which "
                f"is not a well name such as A01.") from exc
    if not wells:
        raise ValueError(
            "Setting: intensity_calibration_wells is empty; name the bead or "
            "reference wells imaged on every plate, such as ['A01'].")
    return wells


def _reference_statistic(plane: np.ndarray, offset: float,
                         statistic: str) -> float:
    """Return one plane's reference intensity above the camera offset.

    ``median`` is the median of every pixel, for a uniformly stained
    reference well; ``foreground`` is the median of the pixels above an
    Otsu threshold, for sparse beads on a dark background. Both scale
    linearly with exposure once the offset is removed.

    :param plane: one intensity plane (2-D or a 3-D stack).
    :param offset: camera offset subtracted before the statistic.
    :param statistic: ``foreground`` or ``median``.
    :returns: the statistic, or NaN when the plane has no usable pixels.
    """
    values = np.asarray(plane, dtype=np.float64).ravel() - float(offset)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    if statistic == "median" or np.ptp(values) == 0:
        return float(np.median(values))
    from skimage.filters import threshold_otsu
    above = values[values > threshold_otsu(values)]
    return float(np.median(above)) if above.size else float("nan")


def _build_calibration_plan(src: os.PathLike | str, filenames: Iterable[str],
                            settings: Mapping[str, Any]) -> Dict[str, Any]:
    """Measure the reference wells of every plate and return per-plate gains.

    Every plate is one imaging session and must carry the reference wells.
    The reference statistic of each intensity plane is taken from each
    reference field, in the units Measure works in (after the plate's uint16
    rescale), and the median over a plate's fields is that plate's value.
    The first plate by name is the reference session: its gain is 1 and
    every other plate is scaled so its reference wells match it.

    :param src: directory of merged field arrays.
    :param filenames: the merged-array filenames of this run.
    :param settings: Measure settings holding the ``intensity_calibration_*``
        values, the mask dimensions and the uint16 rescale plan.
    :returns: a JSON-safe plan with the reference plate and, for each plate,
        its reference statistic, gain and number of reference fields.
    :raises ValueError: when a plate has no reference field or a reference
        statistic is not positive.
    """
    statistic = str(settings.get("intensity_calibration_statistic")
                    or "foreground")
    if statistic not in CALIBRATION_STATISTICS:
        raise ValueError(
            f"Setting: intensity_calibration_statistic is {statistic!r}; "
            f"choose one of {', '.join(CALIBRATION_STATISTICS)}.")
    offset = float(settings.get("intensity_calibration_offset") or 0.0)
    wells = _calibration_wells(settings)
    root = os.fspath(src)
    samples: Dict[str, Dict[int, list]] = {}
    plates = set()
    for filename in sorted(set(os.fspath(name) for name in filenames)):
        field = schema.parse_field_stem(
            filename, timelapse=bool(settings.get("timelapse", False)))
        plates.add(field.plateID)
        if (field.rowID, field.columnID) not in wells:
            continue
        data = np.asarray(np.load(os.path.join(root, filename), mmap_mode="r"))
        factor = float(resolve_record(data, filename, settings)["rescale_factor"])
        labels = mask_planes(data, settings)
        per_plane = samples.setdefault(field.plateID, {})
        for plane in range(int(data.shape[-1])):
            if plane in labels:
                continue
            per_plane.setdefault(plane, []).append(_reference_statistic(
                data[..., plane].astype(np.float64) * factor, offset,
                statistic))
    missing = sorted(plates - set(samples))
    if missing:
        raise ValueError(
            f"Intensity calibration: plate(s) {', '.join(missing)} have no "
            f"field in the reference wells {', '.join(sorted(wells.values()))}"
            f". Image the beads or reference wells on every plate, or leave "
            f"intensity_calibration off.")
    reference = sorted(samples)[0]
    reference_values = {plane: float(np.nanmedian(values))
                        for plane, values in samples[reference].items()}
    plan_plates: Dict[str, Dict[str, Any]] = {}
    for plate, per_plane in sorted(samples.items()):
        values = {plane: float(np.nanmedian(v)) for plane, v in per_plane.items()}
        bad = [plane for plane, value in values.items()
               if not np.isfinite(value) or value <= 0
               or not np.isfinite(reference_values.get(plane, np.nan))
               or reference_values.get(plane, 0.0) <= 0]
        if bad:
            raise ValueError(
                f"Intensity calibration: plate {plate} plane(s) {bad} have no "
                f"positive reference intensity above offset {offset:g} in "
                f"the reference wells, so no gain can be computed.")
        plan_plates[plate] = {
            "reference_statistic": {str(p): v for p, v in values.items()},
            "gain": {str(p): reference_values[p] / v
                     for p, v in values.items()},
            "n_reference_fields": len(next(iter(per_plane.values()))),
        }
    return {
        "version": 1,
        "reference_plate": reference,
        "statistic": statistic,
        "offset": offset,
        "wells": sorted(wells.values()),
        "plates": plan_plates,
    }


def _apply_calibration(data: np.ndarray, filename: str,
                       settings: Mapping[str, Any]
                       ) -> Tuple[np.ndarray, Dict[str, Any] | None]:
    """Scale one field's intensity planes by its plate's calibration gains.

    Each intensity plane becomes ``offset + (value - offset) * gain``,
    rounded and clipped to the array's integer range; label planes are left
    untouched. Without a calibration plan in ``settings`` the field is
    returned unchanged.

    :param data: the field in Measure's working dtype (``uint8`` or
        ``uint16``); ``uint8`` fields come back as ``uint16``.
    :param filename: the field's filename, read for its plate.
    :param settings: Measure settings holding the calibration plan.
    :returns: ``(field, provenance)``; provenance is None without a plan.
    """
    plan = settings.get(CALIBRATION_SETTINGS_KEY)
    if not isinstance(plan, dict):
        return data, None
    plate = _plate_id(filename, settings)
    entry = plan["plates"].get(plate)
    if entry is None:
        raise ValueError(f"Intensity calibration: {filename} is on plate "
                         f"{plate}, which has no calibration gain.")
    offset = float(plan["offset"])
    out = np.array(data, dtype=np.uint16 if data.dtype == np.uint8
                   else data.dtype, copy=True)
    top = float(np.iinfo(out.dtype).max) if np.issubdtype(
        out.dtype, np.integer) else np.inf
    clipped: Dict[str, float] = {}
    for key, gain in entry["gain"].items():
        plane = int(key)
        if plane >= out.shape[-1]:
            continue
        values = offset + (out[..., plane].astype(np.float64) - offset) * gain
        clipped[key] = float(np.mean(values > top))
        if np.issubdtype(out.dtype, np.integer):
            values = np.clip(np.rint(values), 0, top)
        out[..., plane] = values.astype(out.dtype)
    record = {
        "plateID": plate,
        "reference_plate": plan["reference_plate"],
        "statistic": plan["statistic"],
        "offset": offset,
        "wells": plan["wells"],
        "gain": entry["gain"],
        "reference_statistic": entry["reference_statistic"],
        "reference_plate_statistic":
            plan["plates"][plan["reference_plate"]]["reference_statistic"],
        "n_reference_fields": entry["n_reference_fields"],
        "clipped_fraction": clipped,
    }
    return out, record
