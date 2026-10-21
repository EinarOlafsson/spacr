"""Build downloadable N698 results from real inference outputs.

This module performs no inference, network access or model substitution. The
caller supplies the exact selected image plane and actual backend outputs.
Only 2-D segmentation is accepted; detectors need a separate result contract.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
import zipfile

import numpy as np
import pandas as pd
import tifffile
from skimage.measure import regionprops_table

from spacr import __version__
from spacr.tabular import write_table
from spacr.qt.i18n import tr


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validated(image, masks, image_axes, flows, cell_probability, probability_kind):
    image = np.asarray(image)
    masks = np.asarray(masks)
    if image_axes not in ("YX", "CYX") or image.ndim != len(image_axes):
        raise ValueError(tr("image_axes must describe a YX or CYX image"))
    if not all(image.shape) or image.dtype.kind not in "buif":
        raise ValueError(tr("image must contain real numeric pixels"))
    if not np.isfinite(image).all():
        raise ValueError(tr("image contains non-finite pixels"))
    if masks.ndim != 2 or masks.shape != image.shape[-2:]:
        raise ValueError(tr("masks must match the selected image plane"))
    if masks.dtype.kind not in "ui" or np.any(masks < 0):
        raise ValueError(tr("masks must contain nonnegative integer labels"))
    if int(masks.max(initial=0)) > np.iinfo(np.uint32).max:
        raise ValueError(tr("mask labels exceed uint32"))
    masks = masks.astype(np.uint32, copy=False)
    if flows is not None:
        flows = np.asarray(flows)
        if flows.shape != (2, *masks.shape) or flows.dtype.kind not in "uif":
            raise ValueError(tr("flows must have shape (2, Y, X)"))
        if not np.isfinite(flows).all():
            raise ValueError(tr("flows contain non-finite values"))
    if cell_probability is not None:
        cell_probability = np.asarray(cell_probability)
        if cell_probability.shape != masks.shape or cell_probability.dtype.kind not in "uif":
            raise ValueError(tr("cell probability must match the selected image plane"))
        if not np.isfinite(cell_probability).all():
            raise ValueError(tr("cell probability contains non-finite values"))
        if probability_kind not in ("logits", "probability"):
            raise ValueError(tr("declare logits or probability for the backend output"))
        if probability_kind == "probability" and (
            np.any(cell_probability < 0) or np.any(cell_probability > 1)
        ):
            raise ValueError(tr("probabilities must lie between zero and one"))
    elif probability_kind is not None:
        raise ValueError(tr("no probability semantics without an output array"))
    return image, masks, flows, cell_probability


def _publish_staged_zip(stage, destination):
    pending = None
    try:
        descriptor, pending = tempfile.mkstemp(prefix=".spacr-web-result-", suffix=".zip", dir=destination.parent)
        os.close(descriptor)
        with zipfile.ZipFile(pending, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(stage.rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(stage).as_posix())
        with open(pending, "r+b") as handle:
            os.fsync(handle.fileno())
        os.replace(pending, destination)
        pending = None
    finally:
        if pending is not None:
            Path(pending).unlink(missing_ok=True)


def build_result_zip(
    destination, *, input_file, image, masks, image_axes="YX", model_id,
    model_version, weights_sha256, settings, flows=None,
    cell_probability=None, probability_kind=None, unavailable=None,
):
    """Atomically package one selected plane and its true segmentation outputs.

    ``unavailable`` must explain each missing optional backend output, using
    the keys ``flows`` and ``cell_probability``. Logits are exported unchanged
    and explicitly identified as uncalibrated scores. No placeholders are
    synthesized. Measurements use pixel units and zero-based centroid indices.
    The existing destination survives every failure before final publication.
    """
    image, masks, flows, cell_probability = _validated(
        image, masks, image_axes, flows, cell_probability, probability_kind)
    input_file = Path(input_file)
    destination = Path(destination)
    if not input_file.is_file():
        raise ValueError(tr("input_file must be a regular uploaded file"))
    if input_file.resolve() == destination.resolve():
        raise ValueError(tr("the result archive cannot replace the input file"))
    if not str(model_id).strip() or not str(model_version).strip():
        raise ValueError(tr("model identity and version are required"))
    if not isinstance(weights_sha256, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", weights_sha256):
        raise ValueError(tr("weights_sha256 must identify the actual model weights"))
    unavailable = dict(unavailable or {})
    for key, output in (("flows", flows), ("cell_probability", cell_probability)):
        reason = unavailable.get(key)
        if output is None and (not isinstance(reason, str) or not reason.strip()):
            raise ValueError(tr("explain the unavailable {output} output", output=key))
        if output is not None and key in unavailable:
            raise ValueError(tr("{output} cannot be both supplied and unavailable", output=key))
    if set(unavailable) - {"flows", "cell_probability"}:
        raise ValueError(tr("unrecognized unavailable output"))
    settings = json.loads(json.dumps(settings, allow_nan=False))
    if not isinstance(settings, dict):
        raise ValueError(tr("settings must be a JSON object"))
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="spacr-web-result-", dir=destination.parent) as temporary:
        stage = Path(temporary)
        source = stage / "input" / input_file.name
        source.parent.mkdir()
        shutil.copyfile(input_file, source)
        tifffile.imwrite(stage / "image.tif", image, metadata={"axes": image_axes}, photometric="minisblack")
        tifffile.imwrite(stage / "mask.tif", masks, metadata={"axes": "YX"}, photometric="minisblack")
        if flows is not None:
            np.save(stage / "flows.npy", flows, allow_pickle=False)
        if cell_probability is not None:
            np.save(stage / "cell_probability.npy", cell_probability, allow_pickle=False)
        intensity = image if image_axes == "YX" else np.moveaxis(image, 0, -1)
        labels, compact = np.unique(masks, return_inverse=True)
        compact = compact.reshape(masks.shape)
        if labels[0] != 0:
            compact = compact + 1
            labels = np.concatenate(([0], labels))
        properties = regionprops_table(
            compact, intensity_image=intensity,
            properties=("label", "area", "centroid", "intensity_mean"))
        properties["label"] = labels[properties["label"]]
        frame = pd.DataFrame(properties).rename(columns={
            "centroid-0": "centroid_y", "centroid-1": "centroid_x",
            "intensity_mean": "intensity_mean_channel_0",
            **{f"intensity_mean-{channel}": f"intensity_mean_channel_{channel}"
               for channel in range(image.shape[0] if image_axes == "CYX" else 1)},
        })
        write_table(frame, stage / "objects.csv")
        manifest = {
            "schema_version": 1, "spacr_version": __version__,
            "model": {"id": model_id, "version": model_version,
                      "weights_sha256": weights_sha256.lower()},
            "settings": settings, "image_axes": image_axes,
            "shape_yx": list(masks.shape), "object_count": len(frame),
            "measurements": {"area_unit": "pixel^2", "centroid_unit": "pixel",
                             "centroid_origin": 0, "intensity": "selected image values"},
            "outputs": {
                "flows": {"available": flows is not None, "axes": "vector_yx,Y,X"},
                "cell_probability": {"available": cell_probability is not None,
                                     "kind": probability_kind, "calibrated": False},
            },
            "unavailable": unavailable,
        }
        text = tr(
            "spaCR web mask result (research use only)\n"
            "Model: {id}; version: {version}; spaCR: {spacr}\n"
            "Selected image: image.tif; labels: mask.tif; objects: objects.csv.\n"
            "The input/ folder preserves the uploaded bytes.\n"
            "Measurements use pixels, zero-based centroid coordinates and the selected image intensities.\n"
            "flows.npy stores actual backend Y/X vector components when available.\n"
            "cell_probability.npy stores actual backend scores when available.\n"
            "Raw logits are not calibrated probabilities; no calibration is claimed.\n"
            "Missing backend outputs are omitted, with reasons in manifest.json.\n"
            "Exact model weights SHA-256, settings and file hashes are in manifest.json.\n"
            , id=model_id, version=model_version, spacr=__version__)
        (stage / "README.txt").write_text(text, encoding="utf-8")
        manifest["files"] = {
            path.relative_to(stage).as_posix(): {"sha256": _sha256(path), "bytes": path.stat().st_size}
            for path in sorted(stage.rglob("*")) if path.is_file()
        }
        (stage / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        _publish_staged_zip(stage, destination)

    return destination
