"""Actual desktop well-detection path with injected predictor, no GPU weights."""
import hashlib
import io
import json
from types import SimpleNamespace
import zipfile

import numpy as np
import pandas as pd
import pytest
import tifffile

from spacr import plaque
from tools.web_mask_generator import detector, results


@pytest.fixture
def setup(tmp_path, monkeypatch):
    source = tmp_path / "image.tif"
    image = np.arange(3 * 10 * 12, dtype=np.uint16).reshape(3, 10, 12)
    tifffile.imwrite(source, image, metadata={"axes": "CYX"}, photometric="minisblack")
    checkpoint = tmp_path / "well.pt"
    checkpoint.write_bytes(b"controlled fake detector checkpoint")
    calls = []
    box = lambda coordinates, confidence: SimpleNamespace(xyxy=np.asarray([coordinates]), conf=np.asarray([confidence]))
    boxes = [box([5, 5, 10, 10], 0.9), box([1, 1, 4, 4], 0.8), box([0, 0, 11, 1], 0.99)]
    def predict(**kwargs):
        calls.append(kwargs)
        return [SimpleNamespace(boxes=boxes)]
    monkeypatch.setattr(plaque, "_load_detector", lambda path: SimpleNamespace(predict=predict))
    arguments = dict(input_file=source, checkpoint=checkpoint,
                     expected_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                     model_id="toxoplasma_well_detector_v3", model_version="3", effective_settings={})
    return tmp_path / "result.zip", arguments, image, calls, boxes


def test_desktop_detector_keeps_rgb_conversion_geometry_order_and_true_outputs(setup):
    target, arguments, image, calls, _ = setup
    assert detector.run_well_download(target, **arguments) == target
    assert len(calls) == 1
    np.testing.assert_array_equal(calls[0]["source"], np.moveaxis(image, 0, -1)[..., ::-1])
    assert calls[0]["conf"] == 0.25 and calls[0]["imgsz"] == 640
    with zipfile.ZipFile(target) as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    assert set(members) == {"input/image.tif", "image.tif", "objects.csv", "boxes.json", "README.txt", "manifest.json"}
    assert members["input/image.tif"] == arguments["input_file"].read_bytes()
    np.testing.assert_array_equal(tifffile.imread(io.BytesIO(members["image.tif"])), image)
    frame = pd.read_csv(io.BytesIO(members["objects.csv"]))
    assert frame.detection_id.tolist() == [1, 2]
    assert frame.x0.tolist() == [1, 5]
    assert frame.confidence.tolist() == [0.8, 0.9]
    assert frame.bbox_area_pixels.tolist() == [9, 25]
    assert frame.diameter_px.tolist() == [3, 5]
    assert frame.axis_ratio.tolist() == [1, 1]
    assert frame.bbox_center_x.tolist() == [2.5, 7.5]
    manifest = json.loads(members["manifest.json"])
    assert manifest["result_kind"] == "detections" and manifest["detection_count"] == 2
    assert set(manifest["unavailable"]) == {"masks", "flows", "cell_probability"}
    assert manifest["settings"]["pipeline"] == "spacr.plaque.detect_wells"
    for name, metadata in manifest["files"].items():
        assert metadata == {"sha256": hashlib.sha256(members[name]).hexdigest(), "bytes": len(members[name])}
    assert "not segmented" in members["README.txt"].decode()


@pytest.mark.parametrize("channels", [1, None])
def test_empty_detections_keep_csv_schema_and_grey_inputs(setup, channels):
    target, arguments, image, calls, boxes = setup
    boxes.clear()
    if channels == 1:
        tifffile.imwrite(arguments["input_file"], image[:1], metadata={"axes": "CYX"}, photometric="minisblack")
    else:
        tifffile.imwrite(arguments["input_file"], image[0], metadata={"axes": "YX"}, photometric="minisblack")
    detector.run_well_download(target, **arguments)
    assert calls[0]["source"].shape == (10, 12)
    with zipfile.ZipFile(target) as archive:
        frame = pd.read_csv(io.BytesIO(archive.read("objects.csv")))
    assert frame.empty and frame.columns.tolist() == detector.COLUMNS


@pytest.mark.parametrize("changes", [
    {"expected_sha256": "00" * 32}, {"expected_sha256": "bad"},
    {"effective_settings": []}, {"effective_settings": {"diameter": 52}},
    {"effective_settings": {"confidence": 2}}, {"effective_settings": {"confidence": float("nan")}},
    {"effective_settings": {"min_axis_ratio": -1}}, {"effective_settings": {"confidence": True}},
    {"effective_settings": {"imgsz": 0}}, {"effective_settings": {"imgsz": 2.5}},
])
def test_bad_weights_or_settings_never_enter_predictor(setup, changes):
    target, arguments, _, calls, _ = setup
    arguments.update(changes)
    target.write_bytes(b"earlier result")
    with pytest.raises(ValueError):
        detector.run_well_download(target, **arguments)
    assert calls == []
    assert target.read_bytes() == b"earlier result"


def test_two_input_channels_are_not_silently_treated_as_rgb(setup):
    target, arguments, image, calls, _ = setup
    tifffile.imwrite(arguments["input_file"], image[:2], metadata={"axes": "CYX"}, photometric="minisblack")
    with pytest.raises(ValueError, match="three RGB"):
        detector.run_well_download(target, **arguments)
    assert calls == []


@pytest.mark.parametrize("failure", ["csv", "replace", "predict"])
def test_failed_detection_or_export_preserves_prior_zip(setup, monkeypatch, failure):
    target, arguments, _, _, _ = setup
    target.write_bytes(b"earlier result")
    def fail(*args, **kwargs):
        raise OSError("controlled failure")
    if failure == "csv":
        monkeypatch.setattr(detector, "write_table", fail)
    elif failure == "replace":
        monkeypatch.setattr(results.os, "replace", fail)
    else:
        monkeypatch.setattr(plaque, "_load_detector", fail)
    with pytest.raises(OSError):
        detector.run_well_download(target, **arguments)
    assert target.read_bytes() == b"earlier result"
    assert not list(target.parent.glob("spacr-web-detection-*"))
    assert not list(target.parent.glob(".spacr-web-result-*"))


@pytest.mark.parametrize("change", ["axes", "nonfinite", "identity", "settings", "bounds", "confidence", "source"])
def test_malformed_outputs_or_provenance_cannot_replace_good_detection_zip(setup, change):
    target, arguments, image, _, _ = setup
    target.write_bytes(b"earlier result")
    parameters = dict(input_file=arguments["input_file"], image=image, image_axes="CYX",
                      wells=[plaque.Well(1, 1, 4, 4, 0.8)], model_id=arguments["model_id"],
                      model_version=arguments["model_version"], weights_sha256=arguments["expected_sha256"], settings={})
    if change == "axes":
        parameters["image_axes"] = "YXC"
    elif change == "nonfinite":
        parameters["image"] = np.full(image.shape, np.nan)
    elif change == "identity":
        parameters["model_id"] = ""
    elif change == "settings":
        parameters["settings"] = []
    elif change == "bounds":
        parameters["wells"] = [plaque.Well(-1, 0, 4, 4, 0.8)]
    elif change == "confidence":
        parameters["wells"] = [plaque.Well(1, 1, 4, 4, float("nan"))]
    else:
        parameters["input_file"] = target
    with pytest.raises(ValueError):
        detector.build_well_result_zip(target, **parameters)
    assert target.read_bytes() == b"earlier result"
    assert not list(target.parent.glob("spacr-web-detection-*"))
