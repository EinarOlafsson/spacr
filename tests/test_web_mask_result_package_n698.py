"""CPU-only packaging acceptance, independent of any inference service."""
import hashlib
import io
import json
import zipfile

import numpy as np
import pandas as pd
import pytest
import tifffile

from tools.web_mask_generator import results


@pytest.fixture
def package(tmp_path):
    source = tmp_path / "upload.tiff"
    image = np.arange(40, dtype=np.uint16).reshape(2, 4, 5)
    tifffile.imwrite(source, image, metadata={"axes": "CYX"}, photometric="minisblack")
    mask = np.zeros((4, 5), dtype=np.uint32)
    mask[0, :2] = 7
    mask[2:, 3:] = 2**31
    arguments = dict(
        input_file=source, image=image, image_axes="CYX", masks=mask,
        model_id="actual-model", model_version="v4", weights_sha256="ab" * 32,
        settings={"diameter": 63.5, "channels": [0, 1]},
        flows=np.arange(40, dtype=np.float32).reshape(2, 4, 5),
        cell_probability=np.linspace(-3, 5, 20, dtype=np.float32).reshape(4, 5),
        probability_kind="logits")
    return tmp_path / "result.zip", arguments


def unpack(path):
    with zipfile.ZipFile(path) as archive:
        return {name: archive.read(name) for name in archive.namelist()}


def test_actual_arrays_input_measurements_and_provenance_are_preserved(package):
    target, arguments = package
    assert results.build_result_zip(target, **arguments) == target
    members = unpack(target)
    assert set(members) == {"input/upload.tiff", "image.tif", "mask.tif", "flows.npy", "cell_probability.npy", "objects.csv", "README.txt", "manifest.json"}
    assert members["input/upload.tiff"] == arguments["input_file"].read_bytes()
    for name, key in (("image.tif", "image"), ("mask.tif", "masks")):
        np.testing.assert_array_equal(tifffile.imread(io.BytesIO(members[name])), arguments[key])
    for name, key in (("flows.npy", "flows"), ("cell_probability.npy", "cell_probability")):
        np.testing.assert_array_equal(np.load(io.BytesIO(members[name]), allow_pickle=False), arguments[key])
    frame = pd.read_csv(io.BytesIO(members["objects.csv"]))
    assert frame.label.tolist() == [7, 2**31]
    assert frame.area.tolist() == [2, 4]
    assert frame.centroid_y.tolist() == [0, 2.5]
    assert frame.centroid_x.tolist() == [0.5, 3.5]
    for row in frame.itertuples(index=False):
        selected = arguments["masks"] == row.label
        for channel in range(2):
            assert getattr(row, f"intensity_mean_channel_{channel}") == pytest.approx(arguments["image"][channel][selected].mean())
    manifest = json.loads(members["manifest.json"])
    assert manifest["model"] == {"id": "actual-model", "version": "v4", "weights_sha256": "ab" * 32}
    assert manifest["settings"] == arguments["settings"]
    assert manifest["object_count"] == 2
    assert manifest["outputs"]["cell_probability"] == {"available": True, "calibrated": False, "kind": "logits"}
    for name, metadata in manifest["files"].items():
        assert metadata == {"sha256": hashlib.sha256(members[name]).hexdigest(), "bytes": len(members[name])}
    assert "not calibrated" in members["README.txt"].decode()


def test_missing_backend_arrays_are_explained_never_synthesized(package):
    target, arguments = package
    arguments.update(flows=None, cell_probability=None, probability_kind=None,
                     unavailable={"flows": "backend does not expose vectors", "cell_probability": "backend does not expose scores"})
    results.build_result_zip(target, **arguments)
    members = unpack(target)
    assert "flows.npy" not in members and "cell_probability.npy" not in members
    manifest = json.loads(members["manifest.json"])
    assert manifest["unavailable"] == arguments["unavailable"]
    assert not manifest["outputs"]["flows"]["available"]


@pytest.mark.parametrize("occupied", [False, True])
@pytest.mark.parametrize("channel_mode", ["multi", "grey"])
def test_no_objects_full_frame_and_single_channel_images(package, occupied, channel_mode):
    target, arguments = package
    arguments["masks"][:] = 2**31 if occupied else 0
    if channel_mode == "grey":
        arguments.update(image=arguments["image"][0], image_axes="YX")
    results.build_result_zip(target, **arguments)
    frame = pd.read_csv(io.BytesIO(unpack(target)["objects.csv"]))
    assert len(frame) == int(occupied)
    if occupied:
        assert frame.label.tolist() == [2**31]
        assert frame.area.tolist() == [20]
    assert "intensity_mean_channel_0" in frame.columns


@pytest.mark.parametrize("changes", [
    {"image_axes": "YXC"}, {"image": np.zeros((2, 3))},
    {"image": np.full((2, 4, 5), np.nan)},
    {"image": np.full((2, 4, 5), "x")},
    {"masks": np.zeros((3, 5), dtype=int)},
    {"masks": np.full((4, 5), -1)}, {"masks": np.zeros((4, 5), dtype=float)},
    {"masks": np.full((4, 5), 2**32, dtype=np.uint64)},
    {"flows": np.zeros((4, 5))}, {"flows": np.full((2, 4, 5), np.inf)},
    {"cell_probability": np.zeros((5, 4))},
    {"cell_probability": np.full((4, 5), np.nan)},
    {"probability_kind": "confidence"}, {"probability_kind": "probability"},
    {"flows": None}, {"cell_probability": None},
    {"unavailable": {"flows": "missing"}}, {"unavailable": {"masks": "missing"}},
    {"model_id": ""}, {"model_version": ""}, {"weights_sha256": "not-a-digest"},
    {"settings": {"diameter": float("nan")}}, {"settings": []},
])
def test_invalid_results_do_not_replace_existing_download(package, changes):
    target, arguments = package
    target.write_bytes(b"previous valid result")
    arguments.update(changes)
    with pytest.raises((ValueError, TypeError)):
        results.build_result_zip(target, **arguments)
    assert target.read_bytes() == b"previous valid result"
    assert not list(target.parent.glob(".spacr-web-result-*"))
    assert not list(target.parent.glob("spacr-web-result-*"))


@pytest.mark.parametrize("failure", ["table", "zip", "replace"])
def test_disk_or_writer_failure_preserves_previous_download(package, monkeypatch, failure):
    target, arguments = package
    target.write_bytes(b"previous valid result")
    def fail(*args, **kwargs):
        raise OSError("simulated full disk")
    if failure == "table":
        monkeypatch.setattr(results, "write_table", fail)
    elif failure == "zip":
        monkeypatch.setattr(results.zipfile.ZipFile, "write", fail)
    else:
        monkeypatch.setattr(results.os, "replace", fail)
    with pytest.raises(OSError):
        results.build_result_zip(target, **arguments)
    assert target.read_bytes() == b"previous valid result"
    assert not list(target.parent.glob(".spacr-web-result-*"))
    assert not list(target.parent.glob("spacr-web-result-*"))


def test_result_cannot_overwrite_uploaded_input(package):
    target, arguments = package
    source = arguments["input_file"]
    before = source.read_bytes()
    with pytest.raises(ValueError, match="input file"):
        results.build_result_zip(source, **arguments)
    assert source.read_bytes() == before


def test_bounded_probability_scores_keep_their_declared_semantics(package):
    target, arguments = package
    arguments.update(probability_kind="probability", cell_probability=np.linspace(0, 1, 20).reshape(4, 5))
    results.build_result_zip(target, **arguments)
    manifest = json.loads(unpack(target)["manifest.json"])
    assert manifest["outputs"]["cell_probability"]["kind"] == "probability"
    assert manifest["outputs"]["cell_probability"]["calibrated"] is False


def test_unknown_missing_output_and_missing_source_are_refused(package):
    target, arguments = package
    arguments.update(flows=None, cell_probability=None, probability_kind=None,
                     unavailable={"flows": "no vectors", "cell_probability": "no scores", "image": "unknown"})
    with pytest.raises(ValueError, match="unrecognized"):
        results.build_result_zip(target, **arguments)
    arguments["unavailable"].pop("image")
    arguments["input_file"].unlink()
    with pytest.raises(ValueError, match="regular"):
        results.build_result_zip(target, **arguments)
    assert not target.exists()
