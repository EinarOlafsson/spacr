"""Auditable intensity scaling for merged measurement fields.

The measurement worker stores merged arrays as ``uint16`` while label planes
must remain integer identities.  This module makes the conversion decision a
plate-level, serialisable plan so every worker uses the same scale and the
decision can be written into ``measurements.db``.
"""
from __future__ import annotations

from .resource_log import _parallel_thread_executor as ThreadPoolExecutor
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

    def _inspect(filename):
        """Read one field's dtype, intensity maximum and file stats.

        :param filename: merged-array filename inside ``src``.
        :returns: ``(filename, record, None)`` or ``(filename, None, error)``.
        """
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
            return filename, {
                "plateID": plate,
                "original_dtype": str(data.dtype),
                "original_intensity_max": top,
                "kind": kind,
                "size_bytes": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
            }, None
        except Exception as exc:
            from .runctx import _is_overload_failure

            if _is_overload_failure(exc):
                raise
            return filename, None, f"{type(exc).__name__}: {exc}"

    names = sorted(set(os.fspath(name) for name in filenames))
    workers = max(1, min(32, int(settings.get("n_jobs") or 8), len(names) or 1))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(_inspect, names))
    for filename, item, error in results:
        if error is not None:
            failures[filename] = error
            continue
        inspected[filename] = item
        if item["kind"] == "raw":
            plate = item["plateID"]
            plate_maxima[plate] = max(plate_maxima.get(plate, 0.0),
                                      item["original_intensity_max"])
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


def _calibration_input_error(setting, code, value=None):
    """Refuse one invalid calibration input while naming its setting and value."""
    from .qt.i18n import tr

    raise ValueError(tr("Bad settings") + f": {setting}; {code}; {value!r}")


def _calibration_inputs(src, filenames, settings):
    """Resolve explicit acquisition groups and selected local reference paths."""
    import csv
    from pathlib import Path

    names = sorted(set(os.fspath(name) for name in filenames))
    sources = {name: schema.parse_field_stem(
        name, timelapse=bool(settings.get("timelapse", False))) for name in names}
    table = settings.get("intensity_calibration_sessions", "")
    folder = settings.get("intensity_calibration_beads", "")
    table = "" if table is None else table
    folder = "" if folder is None else folder
    for key, value in (("sessions", table), ("beads", folder)):
        if not isinstance(value, str):
            _calibration_input_error("intensity_calibration_" + key, "path", value)
    assignments = {}
    if table.strip():
        column = settings.get("intensity_calibration_session_column", "session")
        if not isinstance(column, str) or not column.strip() or column == "filename":
            _calibration_input_error("intensity_calibration_session_column", "column", column)
        with open(os.path.expanduser(table.strip()), newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            columns = reader.fieldnames or []
            if (len(columns) != len(set(columns)) or "filename" not in columns
                    or column not in columns):
                _calibration_input_error("intensity_calibration_sessions", "headers", columns)
            for row in reader:
                name, session = row.get("filename"), row.get(column)
                if (not isinstance(name, str) or not name.strip()
                        or not isinstance(session, str) or not session.strip()
                        or name != Path(name).name or name in assignments):
                    _calibration_input_error("intensity_calibration_sessions", "row", row)
                assignments[name] = session.strip()
    wells = _calibration_wells(settings)
    root = Path(os.path.expanduser(folder.strip())) if folder.strip() else Path(src)
    references = []
    candidates = sorted(path.name for path in root.iterdir()
                        if path.is_file() and path.suffix.lower() in (".npy", ".tif", ".tiff")) if folder.strip() else names
    for name in candidates:
        field = schema.parse_field_stem(
            Path(name).stem, timelapse=bool(settings.get("timelapse", False)))
        if (field.rowID, field.columnID) in wells:
            references.append({"filename": name, "path": str(root / name),
                               "plateID": field.plateID})
    required = set(names) | {row["filename"] for row in references}
    missing = required - set(assignments) if table.strip() else set()
    if missing:
        _calibration_input_error("intensity_calibration_sessions", "missing_filename", sorted(missing))
    field_sessions = {name: assignments[name] if table.strip() else field.plateID
                      for name, field in sources.items()}
    for row in references:
        row["sessionID"] = assignments[row["filename"]] if table.strip() else row["plateID"]
    return {"field_sessions": field_sessions,
            "field_plates": {name: field.plateID for name, field in sources.items()},
            "references": [row for row in references if row["sessionID"] in set(field_sessions.values())],
            "sessions_csv": os.path.expanduser(table.strip()) if table.strip() else None,
            "external_beads": bool(folder.strip())}


def _external_reference_planes(path, expected, settings):
    """Read merged NPY or unambiguously tagged image-only TIFF reference planes."""
    from pathlib import Path

    if Path(path).suffix.lower() == ".npy":
        data = np.asarray(np.load(path, mmap_mode="r"))
        if data.ndim < 3:
            _calibration_input_error("intensity_calibration_beads", "merged_shape", data.shape)
        planes = tuple(index for index in range(data.shape[-1])
                       if index not in mask_planes(data, settings))
        if planes != expected:
            _calibration_input_error("intensity_calibration_beads", "planes", (planes, expected))
        return [data[..., index] for index in planes]
    import tifffile

    with tifffile.TiffFile(path) as image:
        if len(image.series) != 1:
            _calibration_input_error("intensity_calibration_beads", "tiff_series", path)
        series = image.series[0]
        axes = series.axes
        if (len(axes) != len(set(axes)) or not {"X", "Y"} <= set(axes)
                or set(axes) - set("ZYXCS") or sum(axis in axes for axis in "CS") > 1):
            _calibration_input_error("intensity_calibration_beads", "tiff_axes", axes)
        data = series.asarray()
    channel = next((axes.index(axis) for axis in "CS" if axis in axes), None)
    if channel is None:
        data = data[..., None]
    else:
        data = np.moveaxis(data, channel, -1)
    if data.shape[-1] != len(expected):
        _calibration_input_error("intensity_calibration_beads", "channels", (data.shape, expected))
    return [data[..., index] for index in range(data.shape[-1])]


def _session_calibration_plan(src, filenames, settings):
    """Compute the existing gain statistic for explicitly mapped acquisition sessions."""
    inputs = _calibration_inputs(src, filenames, settings)
    grouped = {}
    for name, session in inputs["field_sessions"].items():
        data = np.asarray(np.load(os.path.join(src, name), mmap_mode="r"))
        if data.ndim < 3:
            _calibration_input_error("intensity_calibration_sessions", "merged_shape", data.shape)
        planes = tuple(index for index in range(data.shape[-1])
                       if index not in mask_planes(data, settings))
        record = resolve_record(data, name, settings)
        factor = float(record["rescale_factor"])
        layout = (planes, factor, record["kind"] == "fixed_normalized")
        if not planes or not np.isfinite(factor) or factor <= 0:
            _calibration_input_error("intensity_calibration_sessions", "working_units", name)
        if session in grouped and grouped[session] != layout:
            _calibration_input_error("intensity_calibration_sessions", "incompatible_working_units", session)
        grouped[session] = layout
    if not grouped:
        _calibration_input_error("intensity_calibration_sessions", "empty_sources")
    if len({layout[0] for layout in grouped.values()}) != 1:
        _calibration_input_error("intensity_calibration_sessions", "incompatible_planes")
    statistic = str(settings.get("intensity_calibration_statistic") or "foreground")
    if statistic not in CALIBRATION_STATISTICS:
        _calibration_input_error("intensity_calibration_statistic", "choice", statistic)
    offset = float(settings.get("intensity_calibration_offset") or 0.0)
    if not np.isfinite(offset):
        _calibration_input_error("intensity_calibration_offset", "finite", offset)
    samples = {}
    for row in inputs["references"]:
        session = row["sessionID"]
        planes, factor, normalized = grouped[session]
        values = _external_reference_planes(row["path"], planes, settings)
        per_plane = samples.setdefault(session, {})
        for index, plane in zip(planes, values):
            finite = np.asarray(plane)[np.isfinite(plane)]
            peak = float(np.max(finite)) if finite.size else 0.0
            is_normalized = np.issubdtype(plane.dtype, np.floating) and 0 < peak <= 1
            if (peak > 0 and normalized != is_normalized) or peak * factor > UINT16_MAX:
                _calibration_input_error("intensity_calibration_beads", "working_units", row["filename"])
            per_plane.setdefault(index, []).append(_reference_statistic(
                np.asarray(plane, dtype=np.float64) * factor, offset, statistic))
    missing = set(grouped) - set(samples)
    if missing:
        _calibration_input_error("intensity_calibration_wells", "missing_session", sorted(missing))
    plate = settings.get("intensity_calibration_reference_plate", "")
    selected = settings.get("intensity_calibration_reference_session", "")
    plate = "" if plate is None else plate
    selected = "" if selected is None else selected
    if not isinstance(plate, str) or not isinstance(selected, str):
        _calibration_input_error("intensity_calibration_reference_session", "selection", (plate, selected))
    plate, selected = plate.strip(), selected.strip()
    candidates = {session for name, session in inputs["field_sessions"].items()
                  if inputs["field_plates"][name] == plate} if plate else set(samples)
    if (not candidates or (selected and selected not in candidates)
            or (plate and not selected and len(candidates) != 1)):
        _calibration_input_error("intensity_calibration_reference_plate", "session_selection", (plate, selected, sorted(candidates)))
    reference = selected or sorted(candidates)[0]
    reference_values = {index: float(np.nanmedian(values))
                        for index, values in samples[reference].items()}
    sessions = {}
    for session, per_plane in sorted(samples.items()):
        values = {index: float(np.nanmedian(v)) for index, v in per_plane.items()}
        if any(not np.isfinite(value) or value <= 0
               or not np.isfinite(reference_values.get(index, np.nan))
               or reference_values.get(index, 0) <= 0 for index, value in values.items()):
            _calibration_input_error("intensity_calibration_wells", "positive_reference", session)
        sessions[session] = {
            "reference_statistic": {str(index): value for index, value in values.items()},
            "gain": {str(index): reference_values[index] / value for index, value in values.items()},
            "n_reference_fields": len(next(iter(per_plane.values()))),
        }
    return {"version": 2, "reference_plate": plate or None,
            "reference_session": reference, "statistic": statistic, "offset": offset,
            "wells": sorted(_calibration_wells(settings).values()), "sessions": sessions,
            "field_sessions": inputs["field_sessions"]}


def _build_calibration_plan(src: os.PathLike | str, filenames: Iterable[str],
                            settings: Mapping[str, Any]) -> Dict[str, Any]:
    """Measure reference wells and return plate or explicit-session gains.

    With blank optional session and bead inputs, every plate is one imaging
    session and must carry the reference wells.
    The reference statistic of each intensity plane is taken from each
    reference field, in the units Measure works in (after the plate's uint16
    rescale), and the median over a plate's fields is that plate's value.
    An explicit intensity_calibration_reference_plate names the reference
    session. A blank value uses the first plate by name. Its gain is 1 and
    every other plate is scaled so its reference wells match it.

    :param src: directory of merged field arrays.
    :param filenames: the merged-array filenames of this run.
    :param settings: Measure settings holding the ``intensity_calibration_*``
        values, the mask dimensions and the uint16 rescale plan.
    Explicit session metadata or external references instead produce a version2
    plan binding every measured filename to its acquisition session. Session
    reference selection and compatible working units are validated before gains.

    :returns: a JSON-safe plan with the selected reference and per-plate or
        per-session reference statistics, gains and reference field counts.
    :raises ValueError: when a plate has no reference field or a reference
        statistic is not positive.
    """
    if any(settings.get(key) for key in ("intensity_calibration_sessions",
                                        "intensity_calibration_beads",
                                        "intensity_calibration_reference_session")):
        return _session_calibration_plan(src, filenames, settings)
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
    selection = settings.get("intensity_calibration_reference_plate")
    if selection is None:
        selection = ""
    if not isinstance(selection, str) or (selection.strip() and selection.strip() not in samples):
        from .qt.i18n import tr

        raise ValueError(tr("Bad settings") + f": intensity_calibration_reference_plate={selection!r}; {tuple(sorted(samples))}")
    reference = selection.strip() or sorted(samples)[0]
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
    """Scale one field intensity planes by its recorded calibration group gains.

    Each intensity plane becomes ``offset + (value - offset) * gain``,
    rounded and clipped to the array's integer range; label planes are left
    untouched. Without a calibration plan in ``settings`` the field is
    returned unchanged.

    :param data: the field in Measure's working dtype (``uint8`` or
        ``uint16``); ``uint8`` fields come back as ``uint16``.
    :param filename: the field filename, mapped to an explicit session when
        present or otherwise read for its plate.
    :param settings: Measure settings holding the calibration plan.
    :returns: ``(field, provenance)``; provenance is None without a plan.
    """
    plan = settings.get(CALIBRATION_SETTINGS_KEY)
    if not isinstance(plan, dict):
        return data, None
    plate = _plate_id(filename, settings)
    session = plan.get("field_sessions", {}).get(filename)
    groups = plan["sessions"] if "sessions" in plan else plan["plates"]
    entry = groups.get(session if "sessions" in plan else plate)
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
            groups[plan.get("reference_session", plan["reference_plate"])]["reference_statistic"],
        "n_reference_fields": entry["n_reference_fields"],
        "clipped_fraction": clipped,
    }
    if "sessions" in plan:
        record.update(sessionID=session, reference_session=plan["reference_session"])
    return out, record
