"""N698 plaque-well detection through the actual desktop detector path.

Detectors produce boxes/confidence. The download explicitly omits segmentation
masks, vectors and cell probabilities rather than fabricating unavailable data.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import re
import shutil
import tempfile

import numpy as np
import pandas as pd
import tifffile

from spacr import __version__
from spacr.plaque import DEFAULT_CONFIDENCE, detect_wells
from spacr.qt.i18n import tr
from spacr.tabular import write_table

from .inputs import read_upload_plane
from .results import _publish_staged_zip, _sha256


COLUMNS = ["detection_id", "x0", "y0", "x1", "y1", "confidence",
           "diameter_px", "axis_ratio", "bbox_area_pixels", "bbox_center_x", "bbox_center_y"]


def build_well_result_zip(destination, *, input_file, image, image_axes,
                          wells, model_id, model_version, weights_sha256, settings):
    """Package actual detection geometry and provenance without fake labels."""
    destination, input_file = Path(destination), Path(input_file)
    image = np.asarray(image)
    if image_axes not in ("YX", "CYX") or image.ndim != len(image_axes) or not all(image.shape):
        raise ValueError(tr("The detection image must declare YX or CYX axes."))
    if image.dtype.kind not in "buif" or not np.isfinite(image).all():
        raise ValueError(tr("The detection image must contain finite numeric pixels."))
    if not input_file.is_file() or destination.resolve() == input_file.resolve():
        raise ValueError(tr("The detection result must preserve its original uploaded file."))
    if not str(model_id).strip() or not str(model_version).strip() or not isinstance(weights_sha256, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", weights_sha256):
        raise ValueError(tr("The detection model identity and weight SHA-256 are required."))
    settings = json.loads(json.dumps(settings, allow_nan=False))
    if not isinstance(settings, dict):
        raise ValueError(tr("Detection settings must be a JSON object."))
    rows = []
    height, width = image.shape[-2:]
    for number, well in enumerate(wells, start=1):
        if not (0 <= well.x0 < well.x1 <= width and 0 <= well.y0 < well.y1 <= height):
            raise ValueError(tr("A detected box falls outside the selected image."))
        if not math.isfinite(well.confidence) or not 0 <= well.confidence <= 1:
            raise ValueError(tr("A detection confidence is invalid."))
        rows.append({"detection_id": number, **well.as_dict(),
                     "bbox_area_pixels": well.width * well.height,
                     "bbox_center_x": (well.x0 + well.x1) / 2,
                     "bbox_center_y": (well.y0 + well.y1) / 2})
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="spacr-web-detection-", dir=destination.parent) as directory:
        stage = Path(directory)
        source = stage / "input" / input_file.name
        source.parent.mkdir()
        shutil.copyfile(input_file, source)
        tifffile.imwrite(stage / "image.tif", image, metadata={"axes": image_axes}, photometric="minisblack")
        write_table(pd.DataFrame(rows, columns=COLUMNS), stage / "objects.csv")
        (stage / "boxes.json").write_text(json.dumps(rows, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        unavailable = {"masks": tr("The well detector returns bounding boxes, not instance masks."),
                       "flows": tr("This detector does not produce flow vectors."),
                       "cell_probability": tr("Detection confidence is not a per-pixel cell probability.")}
        text = tr("spaCR well detection result (research use only).\nThe original upload, selected image and actual boxes/confidences are included.\nBounding-box areas/centres describe rectangles, not segmented well pixels.\nDiameter and axis ratio use the desktop Well geometry. No physical scale is assumed.\nMasks, flows and per-pixel cell probabilities are unavailable for this detector.\nModel identity, settings, unavailable-output reasons and file hashes are in manifest.json.\n")
        (stage / "README.txt").write_text(text, encoding="utf-8")
        manifest = {"schema_version": 1, "spacr_version": __version__, "result_kind": "detections",
                    "model": {"id": model_id, "version": model_version, "weights_sha256": weights_sha256.lower()},
                    "settings": settings, "image_axes": image_axes, "shape_yx": [height, width],
                    "detection_count": len(rows), "outputs": {"boxes": {"available": True}}, "unavailable": unavailable,
                    "measurements": {"box_coordinates": "pixels, zero-based, exclusive upper edge", "area": "bounding rectangle pixels", "centre": "bounding rectangle centre"},
                    "files": {path.relative_to(stage).as_posix(): {"sha256": _sha256(path), "bytes": path.stat().st_size}
                              for path in sorted(stage.rglob("*")) if path.is_file()}}
        (stage / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        _publish_staged_zip(stage, destination)
    return destination


def run_well_download(destination, *, input_file, checkpoint, expected_sha256,
                      model_id, model_version, effective_settings, selection=None):
    """Use desktop detect_wells including RGB/BGR conversion and shape filtering."""
    checkpoint = Path(checkpoint).resolve()
    if not checkpoint.is_file() or not isinstance(expected_sha256, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", expected_sha256):
        raise ValueError(tr("Provide a trusted detector checkpoint and its published SHA-256."))
    digest = _sha256(checkpoint)
    if digest != expected_sha256.lower():
        raise ValueError(tr("The detector checkpoint does not match the published SHA-256."))
    if not isinstance(effective_settings, dict) or set(effective_settings) - {"confidence", "imgsz", "min_axis_ratio"}:
        raise ValueError(tr("Unsupported well detector settings were supplied."))
    settings = {"confidence": DEFAULT_CONFIDENCE, "imgsz": 640, "min_axis_ratio": 0.7, **effective_settings}
    for name in ("confidence", "min_axis_ratio"):
        value = settings[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(tr("Detector setting {name} must lie between zero and one.", name=name))
    if isinstance(settings["imgsz"], bool) or not isinstance(settings["imgsz"], int) or not 1 <= settings["imgsz"] <= 2048:
        raise ValueError(tr("Detector image size must be an integer from 1 to 2048."))
    plane = read_upload_plane(input_file, **dict(selection or {}))
    image = plane["image"]
    if plane["image_axes"] == "CYX":
        if image.shape[0] not in (1, 3):
            raise ValueError(tr("Choose one grayscale or three RGB channels for well detection."))
        image = image[0] if image.shape[0] == 1 else np.moveaxis(image, 0, -1)
    wells = detect_wells(image, str(checkpoint), **settings)
    provenance = {**settings, "input_selection": plane["selection"], "pipeline": "spacr.plaque.detect_wells"}
    return build_well_result_zip(
        destination, input_file=input_file, image=plane["image"], image_axes=plane["image_axes"], wells=wells,
        model_id=model_id, model_version=model_version, weights_sha256=digest, settings=provenance)
