"""Offline, bounded upload decoding for N698's future inference service.

No projections, intensity normalization or implicit channel mixing. TIFF axis
metadata is authoritative; every non-singleton nonspatial axis needs a choice.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image
import tifffile

from spacr.qt.i18n import tr


DEFAULT_MAX_FILE_BYTES = 64 * 1024 * 1024
DEFAULT_MAX_DECODED_BYTES = 256 * 1024 * 1024
DEFAULT_MAX_PLANE_PIXELS = 25_000_000
MAX_PDF_PAGES = 256
DEFAULT_MAX_CZI_SCENES = 4096


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(tr("{name} must be an integer at least {minimum}", name=name, minimum=minimum))
    return int(value)


def _check_size(shape, dtype, max_decoded_bytes):
    if not all(shape) or math.prod(shape) * np.dtype(dtype).itemsize > max_decoded_bytes:
        raise ValueError(tr("The decoded image exceeds the configured byte limit."))


def _file(path, max_file_bytes):
    path = Path(path)
    _integer(max_file_bytes, "max_file_bytes", 1)
    if not path.is_file() or path.stat().st_size > max_file_bytes:
        raise ValueError(tr("The upload is missing or exceeds the configured file size limit."))
    return path


def _legacy_czi_entries(handle, max_decoded_bytes, max_scenes):
    """Validate global legacy canvas headers and retain per-scene absolute bounds."""
    from types import SimpleNamespace
    from spacr.convert import _legacy_czi_series

    try:
        axes = str(handle.axes or "").upper()
        raw_shape, raw_start = tuple(handle.shape), tuple(handle.start)
        if (len(axes) != len(set(axes)) or axes.count("Y") != 1
                or axes.count("X") != 1 or len(axes) != len(raw_shape)
                or len(axes) != len(raw_start)
                or any(isinstance(v, (bool, np.bool_))
                       or not isinstance(v, (int, np.integer)) for v in raw_shape + raw_start)
                or any(v <= 0 for v in raw_shape)):
            raise ValueError
        shape, start = tuple(map(int, raw_shape)), tuple(map(int, raw_start))
        dtype = np.dtype(handle.dtype)
        if dtype.kind not in "buif":
            raise ValueError
    except (AttributeError, KeyError, TypeError, ValueError, OverflowError) as error:
        raise ValueError(tr("The image does not declare unambiguous spatial axes.")) from error
    _check_size(shape, dtype, max_decoded_bytes)
    try:
        canvas = dict(zip(axes, zip(start, shape)))
        blocks = list(handle.filtered_subblock_directory)
        scene_intervals = []
        for block in blocks:
            block_axes = str(block.axes or "").upper()
            block_start, block_shape = tuple(block.start), tuple(block.shape)
            if (set(block_axes) != set(axes) or len(block_axes) != len(set(block_axes))
                    or len(block_axes) != len(block_start)
                    or len(block_axes) != len(block_shape)
                    or any(isinstance(v, (bool, np.bool_))
                           or not isinstance(v, (int, np.integer))
                           for v in block_start + block_shape)
                    or any(v <= 0 for v in block_shape)):
                raise ValueError
            for axis, block_origin, block_size in zip(block_axes, block_start, block_shape):
                origin, size = canvas[axis]
                low = int(block_origin)
                high = low + int(block_size)
                if not origin <= low < high <= origin + size:
                    raise ValueError
            if "S" in block_axes:
                index = block_axes.index("S")
                low = int(block_start[index])
                scene_intervals.append((low, low + int(block_shape[index])))
            else:
                scene_intervals.append((0, 1))
    except (AttributeError, KeyError, TypeError, ValueError, OverflowError) as error:
        raise ValueError(tr("The image does not declare unambiguous spatial axes.")) from error
    scene_count, previous_stop = 0, None
    for low, high in sorted(scene_intervals):
        scene_count += max(0, high - max(low, previous_stop if previous_stop is not None else low))
        previous_stop = max(high, previous_stop if previous_stop is not None else high)
        if scene_count > max_scenes:
            raise ValueError(tr("The CZI exceeds the configured scene limit."))
    try:
        rows = _legacy_czi_series(SimpleNamespace(
            filtered_subblock_directory=blocks, axes=axes, dtype=dtype))
        if "S" not in axes and len(rows) > 1:
            raise ValueError
        entries = []
        for row in rows:
            scene = int(row["czi_scene"])
            if "S" in canvas and not canvas["S"][0] <= scene < sum(canvas["S"]):
                raise ValueError
            bounds = row["czi_bounds"]
            if set(bounds) != set(axes) - {"S"}:
                raise ValueError
            for axis, (low, high) in bounds.items():
                origin, size = canvas[axis]
                if not origin <= low < high <= origin + size:
                    raise ValueError
            own_axes = row["axes"]
            own_shape = [int(size) for size in row["shape"]]
            if (len(own_axes) != len(own_shape)
                    or any(size <= 0 for size in own_shape)):
                raise ValueError
            entries.append({"shape": own_shape, "axes": own_axes,
                            "dtype": str(dtype), "scene_id": scene,
                            "legacy_czi": {"axes": axes, "shape": list(shape),
                                           "start": list(start),
                                           "bounds": {key: list(value) for key, value in bounds.items()}}})
        return entries
    except (AttributeError, KeyError, TypeError, ValueError, OverflowError) as error:
        raise ValueError(tr("The image does not declare unambiguous spatial axes.")) from error


def _read_legacy_czi_scene(handle, info, max_decoded_bytes, max_scenes):
    """Bound the full old-reader allocation before extracting the declared scene."""
    current = _legacy_czi_entries(handle, max_decoded_bytes, max_scenes)
    if info not in current:
        raise ValueError(tr("The image does not declare unambiguous spatial axes."))
    legacy = info["legacy_czi"]
    dtype = np.dtype(info["dtype"])
    _check_size(legacy["shape"], dtype, max_decoded_bytes)
    selection = []
    for axis, origin in zip(legacy["axes"], legacy["start"]):
        if axis == "S":
            selection.append(info["scene_id"] - origin)
        else:
            low, high = legacy["bounds"][axis]
            selection.append(low - origin if axis not in info["axes"]
                             else slice(low - origin, high - origin))
    array = np.asarray(handle.asarray())
    if array.shape != tuple(legacy["shape"]) or array.dtype != dtype:
        raise ValueError(tr("The image does not declare unambiguous spatial axes."))
    array = array[tuple(selection)]
    if array.shape != tuple(info["shape"]):
        raise ValueError(tr("The image does not declare unambiguous spatial axes."))
    return array


def describe_upload(path, *, max_file_bytes=DEFAULT_MAX_FILE_BYTES,
                    max_decoded_bytes=DEFAULT_MAX_DECODED_BYTES,
                    max_scenes=DEFAULT_MAX_CZI_SCENES):
    """Read choices from headers without decoding image pixel buffers.

    PDF entries describe page sizes in points; image entries describe actual
    dimensions/axes. Unsupported optional vendor readers are refused explicitly.
    Legacy CZI scene expansion is bounded by positive ``max_scenes`` (4096 by
    default), independently of the complete-canvas decoded-byte budget. The
    current CZI reader route is unchanged and does not use this legacy-only cap.
    """
    path = _file(path, max_file_bytes)
    _integer(max_decoded_bytes, "max_decoded_bytes", 1)
    _integer(max_scenes, "max_scenes", 1)
    suffix = path.suffix.lower()
    if suffix in (".tif", ".tiff"):
        with tifffile.TiffFile(path) as handle:
            entries = [{"shape": list(series.shape), "axes": series.axes,
                        "dtype": str(series.dtype)} for series in handle.series]
        return {"format": "TIFF", "series": entries}
    if suffix in (".png", ".jpg", ".jpeg", ".bmp"):
        with Image.open(path) as handle:
            if getattr(handle, "n_frames", 1) != 1:
                raise ValueError(tr("Multi-frame ordinary images are unsupported; use TIFF with explicit axes."))
            mode = handle.mode
            if mode not in ("1", "L", "I", "I;16", "I;16L", "I;16B", "F", "RGB", "RGBA", "LA"):
                raise ValueError(tr("This image color mode is unsupported; export a grayscale or RGB image."))
            channels = len(handle.getbands())
            shape = [handle.height, handle.width] + ([channels] if channels > 1 else [])
            axes = "YXC" if channels > 1 else "YX"
            return {"format": handle.format, "series": [{"shape": shape, "axes": axes, "mode": mode}]}
    if suffix == ".czi":
        try:
            import czifile
        except ImportError as error:
            raise ValueError(tr("CZI uploads require the optional czifile reader.")) from error
        with czifile.CziFile(path) as handle:
            if hasattr(handle, "scenes"):
                entries = [{"shape": list(scene.shape), "axes": scene.axes,
                            "dtype": str(scene.dtype), "scene_id": identifier}
                           for identifier, scene in handle.scenes.items()]
            elif hasattr(handle, "filtered_subblock_directory"):
                entries = _legacy_czi_entries(handle, max_decoded_bytes, max_scenes)
            else:
                raise ValueError(tr("CZI uploads require a czifile reader with explicit scene metadata."))
        if not entries:
            raise ValueError(tr("The CZI declares no readable scenes."))
        return {"format": "CZI", "series": entries}
    if suffix == ".nd2":
        try:
            from nd2reader import ND2Reader
        except ImportError as error:
            raise ValueError(tr("ND2 uploads require the optional nd2reader reader.")) from error
        with ND2Reader(str(path)) as handle:
            sizes = dict(handle.sizes)
            if any(int(sizes.get(axis, 0)) < 1 for axis in ("y", "x")):
                raise ValueError(tr("This ND2 reader cannot determine the image spatial dimensions."))
            if set(sizes) - set("tzcyxv"):
                raise ValueError(tr("This ND2 declares unsupported axes."))
            shape = [int(sizes.get(axis, 1)) for axis in "tzcyx"]
            entries = [{"shape": shape, "axes": "TZCYX", "dtype": str(np.dtype(handle.pixel_type)), "position": number}
                       for number in range(int(sizes.get("v", 1)))]
        return {"format": "ND2", "series": entries}
    if suffix == ".pdf":
        try:
            import pypdfium2
        except ImportError as error:
            raise ValueError(tr("PDF uploads require the optional pypdfium2 reader.")) from error
        with pypdfium2.PdfDocument(path) as handle:
            if len(handle) > MAX_PDF_PAGES:
                raise ValueError(tr("The PDF exceeds the configured page limit."))
            entries = []
            for number in range(len(handle)):
                pdf_page = handle[number]
                try:
                    entries.append({"width_points": pdf_page.get_width(), "height_points": pdf_page.get_height()})
                finally:
                    pdf_page.close()
        return {"format": "PDF", "pages": entries}
    raise ValueError(tr("Unsupported upload format: {suffix}", suffix=suffix))


def _selection(shape, axes, indices, channels, max_plane_pixels):
    axes = axes.replace("S", "C") if "S" in axes and "C" not in axes else axes
    if len(axes) != len(shape) or len(set(axes)) != len(axes) or axes.count("Y") != 1 or axes.count("X") != 1:
        raise ValueError(tr("The image does not declare unambiguous spatial axes."))
    shape_by_axis = dict(zip(axes, shape))
    if shape_by_axis["Y"] * shape_by_axis["X"] > max_plane_pixels:
        raise ValueError(tr("The selected image exceeds the configured pixel limit."))
    indices = dict(indices or {})
    nonspatial = set(axes) - set("CYX")
    if set(indices) - nonspatial:
        raise ValueError(tr("Select only declared nonspatial axes by index."))
    chosen = {}
    for axis in sorted(nonspatial):
        if shape_by_axis[axis] > 1 and axis not in indices:
            raise ValueError(tr("Choose an explicit index for image axis {axis}.", axis=axis))
        index = _integer(indices.get(axis, 0), axis)
        if index >= shape_by_axis[axis]:
            raise ValueError(tr("Image axis {axis} index is out of range.", axis=axis))
        chosen[axis] = index
    count = shape_by_axis.get("C", 1)
    selected_channels = list(range(count)) if channels is None else list(channels)
    if not selected_channels:
        raise ValueError(tr("Choose at least one image channel."))
    selected_channels = [_integer(channel, "channel") for channel in selected_channels]
    if len(set(selected_channels)) != len(selected_channels) or max(selected_channels) >= count:
        raise ValueError(tr("Image channels must be distinct and within the declared range."))
    return axes, chosen, selected_channels


def read_upload_plane(
    path, *, series=0, indices=None, channels=None, page=0, pdf_dpi=150,
    max_file_bytes=DEFAULT_MAX_FILE_BYTES,
    max_decoded_bytes=DEFAULT_MAX_DECODED_BYTES,
    max_plane_pixels=DEFAULT_MAX_PLANE_PIXELS,
    max_scenes=DEFAULT_MAX_CZI_SCENES,
):
    """Decode a selected YX/CYX plane with an explicit selection receipt.

    TIFF decodes the complete selected series; legacy CZI decodes its entire
    declared multi-scene canvas. The decoded-byte limit is checked against
    that complete allocation before reading pixels. Legacy scene metadata is
    also limited by ``max_scenes`` before expanding subblock scene intervals. PDF
    pages render on white at the requested DPI; their pixels are page renderings,
    not calibrated microscope acquisitions. No remote files are fetched.
    """
    path = _file(path, max_file_bytes)
    for name, value in (("max_decoded_bytes", max_decoded_bytes), ("max_plane_pixels", max_plane_pixels), ("pdf_dpi", pdf_dpi)):
        _integer(value, name, 1)
    series, page = _integer(series, "series"), _integer(page, "page")
    description = describe_upload(path, max_file_bytes=max_file_bytes,
                                  max_decoded_bytes=max_decoded_bytes, max_scenes=max_scenes)
    if description["format"] == "PDF":
        if series != 0 or indices:
            raise ValueError(tr("PDF input uses page selection, not image series or axes."))
        if page >= len(description["pages"]):
            raise ValueError(tr("PDF page is out of range."))
        info = description["pages"][page]
        shape = [math.ceil(info["height_points"] * pdf_dpi / 72), math.ceil(info["width_points"] * pdf_dpi / 72), 3]
        _check_size(shape, np.uint8, max_decoded_bytes)
        axes, chosen, selected_channels = _selection(shape, "YXC", {}, channels, max_plane_pixels)
        import pypdfium2
        with pypdfium2.PdfDocument(path) as document:
            pdf_page = document[page]
            try:
                bitmap = pdf_page.render(scale=pdf_dpi / 72)
                try:
                    array = np.array(bitmap.to_pil().convert("RGB"))
                finally:
                    bitmap.close()
            finally:
                pdf_page.close()
        provenance = {"page": page, "pdf_dpi": pdf_dpi, "pixels": "PDF page rendering"}
    else:
        if page != 0:
            raise ValueError(tr("Image input uses series and axis selection, not PDF pages."))
        if series >= len(description["series"]):
            raise ValueError(tr("Image series is out of range."))
        info = description["series"][series]
        shape = info["shape"]
        dtype = np.dtype(info["dtype"]) if "dtype" in info else np.dtype({"I": "int32", "F": "float32", "I;16": "uint16", "I;16L": "uint16", "I;16B": "uint16"}.get(info["mode"], "uint8"))
        _check_size(shape, dtype, max_decoded_bytes)
        axes, chosen, selected_channels = _selection(shape, info["axes"], indices, channels, max_plane_pixels)
        if description["format"] == "TIFF":
            with tifffile.TiffFile(path) as handle:
                array = handle.series[series].asarray()
        elif description["format"] == "CZI":
            import czifile
            with czifile.CziFile(path) as handle:
                if "legacy_czi" in info:
                    array = _read_legacy_czi_scene(handle, info, max_decoded_bytes, max_scenes)
                else:
                    array = handle.scenes[info["scene_id"]].asarray()
        elif description["format"] == "ND2":
            from nd2reader import ND2Reader
            with ND2Reader(str(path)) as handle:
                array = np.stack([np.asarray(handle.get_frame_2D(
                    t=chosen["T"], z=chosen["Z"], c=channel, v=info["position"]))
                    for channel in selected_channels])
            return {"image": array, "image_axes": "CYX", "selection": {
                "format": "ND2", "channels": selected_channels, "series": series,
                "position": info["position"], "source_axes": "TZCYX", "indices": chosen}}
        else:
            with Image.open(path) as handle:
                array = np.array(handle)
        provenance = {"series": series, "source_axes": info["axes"], "indices": chosen}
        if description["format"] == "CZI":
            provenance["scene_id"] = info["scene_id"]
    selectors = tuple(chosen.get(axis, slice(None)) for axis in axes)
    array = array[selectors]
    remaining = "".join(axis for axis in axes if axis not in chosen)
    if "C" in remaining:
        array = np.take(array, selected_channels, axis=remaining.index("C"))
        array = np.transpose(array, [remaining.index(axis) for axis in "CYX"])
        output_axes = "CYX"
    else:
        array = np.transpose(array, [remaining.index(axis) for axis in "YX"])
        output_axes = "YX"
    return {"image": array, "image_axes": output_axes,
            "selection": {"format": description["format"], "channels": selected_channels, **provenance}}
