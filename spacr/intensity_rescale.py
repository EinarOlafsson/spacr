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
    """Return last-axis planes that contain labels rather than intensities."""
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
    """Return the largest finite intensity and whether an intensity plane exists."""
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
    field = schema.parse_field_stem(
        filename, timelapse=bool(settings.get("timelapse", False)))
    return field.plateID


def _kind(dtype: np.dtype, top: float, has_intensity: bool) -> str:
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
    """Return the explicitly non-comparable per-field fallback decision."""
    top, has_intensity = signal_max(data, settings)
    kind = _kind(data.dtype, top, has_intensity)
    if kind == "fixed_normalized":
        factor = UINT16_MAX
        comparable = True  # fixed by definition, even without a pre-pass
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
    """Resolve and verify one worker's record against the precomputed plan."""
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
        # Missing plate or pixels replaced by brighter ones after the scan.
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
    """Whether a conversion factor is neither identity nor fixed [0, 1]."""
    return not (np.isclose(float(factor), 1.0)
                or np.isclose(float(factor), UINT16_MAX))
