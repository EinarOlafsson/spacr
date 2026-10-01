"""Headless ensemble computation, scientific-data preservation and export."""
import csv
import json
import sys

import numpy as np
import pytest
import tifffile
from scipy import ndimage

from spacr import segmentation_uncertainty as uncertainty
from spacr.curation_queue import build_queue


def segment(image):
    return ndimage.label(np.asarray(image) > 100)[0]


def second_segment(image):
    labels = segment(image)
    labels[: labels.shape[0] // 2] = 0
    return labels


@pytest.fixture
def folder(tmp_path):
    (tmp_path / "masks").mkdir()
    image = np.zeros((32, 48), np.uint16)
    image[4:10, 5:12] = 500
    image[22:29, 25:33] = 700
    for name in ("one", "two"):
        tifffile.imwrite(tmp_path / f"{name}.tif", image)
    return tmp_path


def test_ensemble_changes_uncertainty_and_preserves_primary_objects():
    image = np.zeros((32, 48), np.uint16)
    image[5:12, 6:13] = 500
    baseline = uncertainty.compute_uncertainty(image, segment)
    ensemble = uncertainty.compute_uncertainty(image, segment, second_segment=second_segment)
    assert baseline["n_passes"] == 4
    assert ensemble["n_passes"] == 8
    assert set(ensemble["objects"]) == set(baseline["objects"]) == {1}
    assert ensemble["objects"][1] > baseline["objects"][1]
    assert ensemble["map"].shape == image.shape


def test_map_roundtrip_contains_provenance_and_retains_precision(tmp_path):
    image = np.zeros((12, 20), np.uint16)
    image[3:8, 4:9] = 500
    result = uncertainty.compute_uncertainty(image, segment)
    result["map"][0, 0] = np.float32(0.1234567)
    path = uncertainty.save_uncertainty_map(tmp_path / "map.tif", result,
                                          provenance={"primary_model": "primary", "second_model": "second"})
    assert np.array_equal(tifffile.imread(path), result["map"])
    with tifffile.TiffFile(path) as handle:
        metadata = json.loads(handle.pages[0].description)["spacr_uncertainty"]
    assert metadata["primary_model"] == "primary"
    assert metadata["scores"]["n_passes"] == 4
    before = path.read_bytes()
    with pytest.raises(ValueError, match="source image"):
        uncertainty.save_uncertainty_map(path, result, protected_paths=[path])
    assert path.read_bytes() == before


def test_failed_map_write_preserves_existing_map_and_removes_scratch(tmp_path, monkeypatch):
    result = {"map": np.zeros((4, 4), np.float32)}
    path = uncertainty.save_uncertainty_map(tmp_path / "map.tif", result)
    before = path.read_bytes()

    def fail(*args, **kwargs):
        raise OSError("disk failure")

    monkeypatch.setattr(tifffile, "imwrite", fail)
    with pytest.raises(OSError, match="disk failure"):
        uncertainty.save_uncertainty_map(path, result)
    assert path.read_bytes() == before
    assert not list(tmp_path.glob(".uncertainty-*"))


def test_map_cannot_overwrite_other_scientific_image(tmp_path):
    path = tmp_path / "source.tif"
    tifffile.imwrite(path, np.ones((4, 4), np.uint16))
    before = path.read_bytes()
    with pytest.raises(ValueError, match="not an uncertainty map"):
        uncertainty.save_uncertainty_map(path, {"map": np.zeros((4, 4), np.float32)})
    assert path.read_bytes() == before


def test_queue_computes_scores_maps_and_preserves_sources(folder):
    before = {path: path.read_bytes() for path in folder.glob("*.tif")}
    queue = build_queue(folder, order="name", announce=lambda _text: None)
    calls = []

    def factory(model, device, parameters):
        calls.append((model, device))
        return segment if model == "primary" else second_segment

    scores = uncertainty.compute_queue_uncertainty(
        queue, model="primary", second_model="second", map_folder=folder / "uncertainty",
        segmenter_factory=factory)
    assert calls == [("primary", "cpu"), ("second", "cpu")]
    assert set(scores) == {"one", "two"}
    assert all(score["n_passes"] == 8 and "map" not in score for score in scores.values())
    with open(folder / "curate_uncertainty.csv") as handle:
        rows = list(csv.DictReader(handle))
    assert all(row["passes"] == "8" and row["model"] == "primary + second" for row in rows)
    assert len(list((folder / "uncertainty").glob("*.tif"))) == 2
    assert all(path.read_bytes() == data for path, data in before.items())


def test_queue_rejects_duplicate_model_before_inference(folder):
    queue = build_queue(folder, announce=lambda _text: None)
    with pytest.raises(ValueError, match="must differ"):
        uncertainty.compute_queue_uncertainty(queue, model="same", second_model="same")


def test_cli_computes_without_opening_editor(folder, monkeypatch, capsys):
    from spacr import cli_make_masks as cli

    monkeypatch.setattr(uncertainty, "_make_segmenter", lambda *_args: segment)
    monkeypatch.setattr(cli, "open_editor", lambda *_args: pytest.fail("opened Qt"))
    code = cli.main(["--folder", str(folder), "--compute-uncertainty",
                     "--save-uncertainty-maps", "--limit", "1"])
    assert code == 0
    assert "Saved uncertainty scores for 1 fields" in capsys.readouterr().out
    assert len(list((folder / "uncertainty").glob("*.tif"))) == 1


def test_cli_dry_run_never_starts_inference(folder, monkeypatch):
    from spacr import cli_make_masks as cli

    monkeypatch.setattr(uncertainty, "_make_segmenter", lambda *_args: pytest.fail("inference"))
    assert cli.main(["--folder", str(folder), "--compute-uncertainty", "--dry-run"]) == 0
    assert not (folder / "curate_uncertainty.csv").exists()


def test_cli_reports_failed_inference(folder, monkeypatch, capsys):
    from spacr import cli_make_masks as cli

    def fail(*_args):
        raise RuntimeError("checkpoint unavailable")

    monkeypatch.setattr(uncertainty, "_make_segmenter", fail)
    assert cli.main(["--folder", str(folder), "--compute-uncertainty"]) == 1
    assert "checkpoint unavailable" in capsys.readouterr().err


def test_native_model_loader_keeps_cpu_and_inference_settings(monkeypatch):
    import types

    seen = {}

    class Model:
        def __init__(self, pretrained_model, device, gpu, use_bfloat16=True):
            seen.update(model=pretrained_model, device=str(device), gpu=gpu,
                        use_bfloat16=use_bfloat16)

        def eval(self, images, **kwargs):
            seen['parameters'] = kwargs
            image = images[0]
            shape = image.shape
            labels = segment(image)
            flows = [np.zeros(shape + (3,), np.uint8),
                     np.zeros((2,) + shape, np.float32),
                     np.ones(shape, np.float32)]
            return [labels], [flows], None

    monkeypatch.setitem(sys.modules, 'cellpose.models', types.SimpleNamespace(CellposeModel=Model))
    run = uncertainty._make_segmenter('cpsam', 'cpu', {'diameter': 17, 'normalize': False})
    labels, logits, flows = run(np.ones((20, 30), np.uint16) * 500)
    assert seen['device'] == 'cpu' and not seen['gpu'] and not seen['use_bfloat16']
    assert seen['parameters']['diameter'] == 17
    assert seen['parameters']['normalize'] is False
    assert labels.shape == logits.shape == (20, 30)
    assert flows.shape == (2, 20, 30)


def test_headless_queue_never_imports_pyside(folder):
    import subprocess

    script = r"""
import builtins, sys
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.startswith('PySide6'):
        raise AssertionError('Headless scoring imported Qt: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
from spacr.segmentation_uncertainty import compute_queue_uncertainty
from spacr.curation_queue import build_queue
queue = build_queue(sys.argv[1], order='name', announce=lambda text: None)
factory = lambda model, device, params: lambda image: (image > 100).astype('int32')
assert len(compute_queue_uncertainty(queue, segmenter_factory=factory)) == 2
"""
    result = subprocess.run([sys.executable, '-c', script, str(folder)],
                            text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr


def test_failed_later_field_keeps_completed_score(folder):
    queue = build_queue(folder, order="name", announce=lambda _text: None)
    calls = []

    def partial(image):
        calls.append(1)
        if len(calls) > 4:
            raise RuntimeError("second field failed")
        return segment(image)

    with pytest.raises(RuntimeError, match="second field"):
        uncertainty.compute_queue_uncertainty(
            queue, segmenter_factory=lambda *_args: partial)
    with open(folder / "curate_uncertainty.csv") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["stem"] for row in rows] == ["one"]
