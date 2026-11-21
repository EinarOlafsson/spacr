"""Store OPS field registration tables as OME-Zarr 0.6 transformations."""
from __future__ import annotations

import json
import math
from numbers import Real
from collections.abc import Mapping, Sequence
from typing import Any

from .ome_zarr import NGFF06_INTRINSIC_SYSTEM, OmeZarrError


def _number(value: Any, column: str) -> float:
    """Copy one finite table number without decimal rounding."""
    from .qt.i18n import tr

    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        result = math.nan
    if not math.isfinite(result):
        raise OmeZarrError(tr(
            "Registration column {column} must contain a finite number; "
            "check the OPS registration table before exporting.", column=column))
    return result


def _missing(value: Any) -> bool:
    """Recognize empty optional cells in OPS dataframe records."""
    return value is None or isinstance(value, Real) and math.isnan(value)


def _values(record: Mapping[str, Any], columns: Sequence[str]) -> list[float] | None:
    """Read a complete optional group; refuse partially populated table rows."""
    from .qt.i18n import tr

    present = [column in record and not _missing(record[column]) for column in columns]
    if not any(present):
        return None
    if not all(present):
        raise OmeZarrError(tr(
            "Registration needs all of {columns}; complete the OPS registration "
            "table before exporting.", columns=", ".join(columns)))
    return [_number(record[column], column) for column in columns]


def _vector(ndim: int, iy: int, ix: int, y: float, x: float,
            fill: float = 0.0) -> list[float]:
    """Embed a row/column vector while leaving the other axes unchanged."""
    result = [fill] * ndim
    result[iy], result[ix] = y, x
    return result


def _affine(values: Sequence[float], ndim: int, iy: int, ix: int) -> list[list[float]]:
    """Embed an exact two-dimensional table affine in the image axes."""
    a11, a12, a21, a22, ty, tx = values
    result = [[float(i == j) for j in range(ndim)] + [0.0] for i in range(ndim)]
    result[iy][iy], result[iy][ix], result[iy][-1] = a11, a12, ty
    result[ix][iy], result[ix][ix], result[ix][-1] = a21, a22, tx
    return result


def registration_metadata(record: Mapping[str, Any], *,
                          axes: Sequence[str] = ("c", "y", "x"),
                          ngff_version: str = "0.6") -> dict[str, list[dict[str, Any]]]:
    """Build OME-Zarr writer metadata from one OPS phenotype table row.

    Coordinates are field pixels in ``y, x`` order. The intrinsic ``physical``
    system must therefore use unit spacing, as merged OPS field stores do.
    The applied ``dy/dx`` rows from items 919/998 map painting pixels to
    SBS pixels directly; logged source shifts are never negated again.
    Placement maps field pixels to SBS well pixels: scale, rotation, then
    translation. OPS selects the refined placement only when ``refined == 1``;
    otherwise the stored seed is used, even when diagnostic refined values exist.
    Secondary shifts map the moving secondary frame to the primary field;
    applied shifts are used, rather than the diagnostic raw estimates.

    Each tile-affine fit maps this field to a well frame estimated from one
    SBS tile. Its distinct ``tile_global_<site>`` system identifies that fit;
    the fits are alternatives and are never composed with each other. The
    global affine is also retained as ``tile_global``, including when the
    tile-group list is empty. Numeric affine entries and shifts are copied
    without fitting or rounding. Sidecar
    tables remain the source of truth. No image pixels are read or changed.

    :param record: one 919/998 applied shift row (``dy/dx``), or one
        ``ops_phenotype`` row, including placement, optional
        ``secondary_dy/dx`` and ``align_tile_transforms`` (the existing JSON
        list of ``[site, count, a11, a12, a21, a22, ty, tx]`` rows).
    :param axes: image axis names, containing exactly one ``y`` and ``x``;
        other axes retain identity coordinates.
    :param ngff_version: must be ``"0.6"`` when registration is requested.
    :returns: ``coordinate_systems`` and ``coordinate_transformations``
        keyword arguments for :func:`spacr.ome_zarr.write_ome_zarr`.
    :raises OmeZarrError: on another version, invalid axes, incomplete table
        groups, duplicate tile sites or non-finite transformation parameters.
    """
    from .qt.i18n import tr

    if ngff_version != "0.6":
        raise OmeZarrError(tr(
            "Registration metadata requires OME-Zarr 0.6; set "
            "ome_zarr_ngff_version to '0.6' or disable ome_zarr_store_registration."))
    names = list(axes)
    if (len(set(names)) != len(names) or names.count("y") != 1
            or names.count("x") != 1 or set(names) - {"c", "t", "z", "y", "x"}):
        raise OmeZarrError(tr(
            "Registration axes must have unique names including y and x; "
            "check the exported image axes."))
    ndim, iy, ix = len(names), names.index("y"), names.index("x")
    axis_metadata = [{"name": name, "type": "space" if name in ("x", "y", "z")
                      else "channel" if name == "c" else "time"} for name in names]
    systems, transforms = [], []
    physical = {"name": NGFF06_INTRINSIC_SYSTEM}
    columns = ("scale", "rotation_deg", "offset_y", "offset_x")
    placement = None
    placement_columns = columns + tuple("refined_" + column for column in columns)
    if any(column in record and not _missing(record[column]) for column in placement_columns):
        from numpy import errstate
        from .ops_engine import _ph_field_record_transform

        try:
            with errstate(invalid="ignore"):
                scale, rotation, offset = _ph_field_record_transform(record)
            scale = _number(scale, "scale/refined_scale")
            dy = _number(offset[0], "offset_y/refined_offset_y")
            dx = _number(offset[1], "offset_x/refined_offset_x")
            matrix = [[_number(value, "rotation_deg/refined_rotation_deg")
                       for value in row] for row in rotation]
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise OmeZarrError(tr(
                "Active registration needs a complete finite scale, rotation and "
                "offset; repair the selected OPS placement in the registration "
                "table before exporting.")) from exc
        placement = scale, matrix, dy, dx
    shift = _values(record, ("dy", "dx"))
    if placement is not None and shift is not None:
        raise OmeZarrError(tr(
            "Registration contains both placement and dy/dx shifts; "
            "export one source table row at a time."))
    if shift is not None:
        systems.append({"name": "sbs", "axes": axis_metadata})
        transforms.append({"type": "translation", "input": physical,
                           "output": {"name": "sbs"},
                           "translation": _vector(ndim, iy, ix, *shift)})
    if placement is not None:
        scale, matrix, dy, dx = placement
        steps = [{"type": "scale", "scale": _vector(ndim, iy, ix, scale, scale, 1.0)}]
        if matrix != [[1.0, 0.0], [0.0, 1.0]]:
            rotation = _affine((*matrix[0], *matrix[1], 0.0, 0.0), ndim, iy, ix)
            steps.append({"type": "rotation", "rotation": [row[:-1] for row in rotation]})
        steps.append({"type": "translation", "translation": _vector(ndim, iy, ix, dy, dx)})
        systems.append({"name": "sbs", "axes": axis_metadata})
        transforms.append({"type": "sequence", "input": physical,
                           "output": {"name": "sbs"}, "transformations": steps})
    secondary = _values(record, ("secondary_dy", "secondary_dx"))
    if secondary is not None:
        systems.append({"name": "secondary", "axes": axis_metadata})
        transforms.append({"type": "translation", "input": {"name": "secondary"},
                           "output": physical,
                           "translation": _vector(ndim, iy, ix, *secondary)})
    tiles = record.get("align_tile_transforms")
    if isinstance(tiles, str):
        try:
            tiles = json.loads(tiles)
        except (TypeError, ValueError) as exc:
            raise OmeZarrError(tr(
                "Tile registration must be a JSON list of eight-value rows; "
                "check align_tile_transforms in the OPS table.")) from exc
    if not _missing(tiles):
        if not isinstance(tiles, (list, tuple)):
            raise OmeZarrError(tr(
                "Tile registration must be a JSON list of eight-value rows; "
                "check align_tile_transforms in the OPS table."))
        seen = set()
        for row in tiles:
            if not isinstance(row, (list, tuple)) or len(row) != 8:
                raise OmeZarrError(tr(
                    "Tile registration must be a JSON list of eight-value rows; "
                    "check align_tile_transforms in the OPS table."))
            site = _number(row[0], "align_tile_transforms.site")
            if not site.is_integer() or site in seen:
                raise OmeZarrError(tr(
                    "Tile registration sites must be unique integers; "
                    "check align_tile_transforms in the OPS table."))
            seen.add(site)
            values = [_number(value, "align_tile_transforms") for value in row[2:]]
            name = "tile_global_" + str(int(site))
            systems.append({"name": name, "axes": axis_metadata})
            transforms.append({"type": "affine", "input": physical,
                               "output": {"name": name}, "affine": _affine(values, ndim, iy, ix)})
    values = _values(record, ("align_a11", "align_a12", "align_a21", "align_a22",
                              "align_ty", "align_tx"))
    if values is not None:
        systems.append({"name": "tile_global", "axes": axis_metadata})
        transforms.append({"type": "affine", "input": physical,
                           "output": {"name": "tile_global"},
                           "affine": _affine(values, ndim, iy, ix)})
    return {"coordinate_systems": systems, "coordinate_transformations": transforms}
