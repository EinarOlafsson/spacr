"""Real image reads and shared model settings across Plaque preview entry points."""
from __future__ import annotations

import types

import numpy as np
import pytest
import tifffile
from PIL import Image

pytest.importorskip("PySide6")

from spacr import plaque_papers as pp
from spacr.qt.widgets import plaque_preview as pv


def _figure(tmp_path):
    path = tmp_path / "figure.png"
    Image.fromarray(np.full((160, 200, 3), 80, np.uint8)).save(path)
    return path


def _detect(image, weights, **kwargs):
    return [types.SimpleNamespace(x0=20, y0=20, x1=80, y1=80, confidence=0.9)]


def _labels(crop):
    labels = np.zeros(crop.shape[:2], np.int32)
    labels[5:10, 5:10] = 7
    return labels


@pytest.mark.parametrize("layout", ["grey16", "channel_first", "rgba", "two_channels", "constant_float"])
def test_real_tiff_display_loading_preserves_dimensions_and_channels(tmp_path, layout):
    grey = np.arange(20 * 30, dtype=np.uint16).reshape(20, 30)
    if layout == "grey16":
        array = grey
    elif layout == "channel_first":
        array = np.stack([np.full((20, 30), value, np.uint8) for value in (11, 22, 33)])
    elif layout == "rgba":
        array = np.broadcast_to(np.array([11, 22, 33, 99], np.uint8), (20, 30, 4))
    elif layout == "two_channels":
        array = np.stack([grey, grey * 0], axis=-1)
    else:
        array = np.full((20, 30), 9.5, np.float32)
    path = tmp_path / "image.tif"
    tifffile.imwrite(path, array, photometric="minisblack")
    original = path.read_bytes()
    result = pv.load_display_image(path)
    assert result.shape == (20, 30, 3) and result.dtype == np.uint8
    assert result.flags.c_contiguous and path.read_bytes() == original
    if layout in ("channel_first", "rgba"):
        np.testing.assert_array_equal(result, np.broadcast_to([11, 22, 33], result.shape))
    elif layout == "constant_float":
        assert not result.any()
    else:
        assert result.min() == 0 and result.max() == 255
        np.testing.assert_array_equal(result[..., 0], result[..., 1])
        np.testing.assert_array_equal(result[..., 1], result[..., 2])


def test_tiff_reader_failure_falls_back_to_pillow_without_changing_pixels(tmp_path, monkeypatch):
    path = tmp_path / "fallback.tif"
    pixels = np.broadcast_to(np.array([7, 31, 90], np.uint8), (12, 14, 3)).copy()
    Image.fromarray(pixels).save(path)

    def unavailable(*args, **kwargs):
        raise OSError("decoder unavailable")

    monkeypatch.setattr(tifffile, "imread", unavailable)
    np.testing.assert_array_equal(pv.load_display_image(path), pixels)


@pytest.mark.parametrize("location", ["explicit", "catalogue", "home", "run_cache", "missing", "unknown"])
def test_detector_resolution_uses_local_checkpoints_without_fetching(tmp_path, monkeypatch, location):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    src = tmp_path / "source"
    entry = types.SimpleNamespace(key="local-well", name="weights.pt", path="")
    monkeypatch.setattr(pv, "_catalogue", lambda: [entry])
    candidates = {
        "explicit": tmp_path / "explicit.pt",
        "catalogue": tmp_path / "catalogue.pt",
        "home": tmp_path / "home/.spacr/models/weights.pt",
        "run_cache": src / "plaque_figures/models/weights.pt",
    }
    path = candidates.get(location)
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"local model fixture")
    if location == "catalogue":
        entry.path = str(path)
    requested = str(path) if location == "explicit" else "not-in-zoo" if location == "unknown" else entry.key
    actual, note, found = pv.resolve_detector(requested, src)
    if path:
        assert actual == str(path) and note == ""
        assert found is (None if location == "explicit" else entry)
        assert path.read_bytes() == b"local model fixture"
    else:
        assert actual == "" and note
        assert found is (None if location == "unknown" else entry)
        assert not src.exists()


def test_model_cache_reuses_only_the_latest_checkpoint(monkeypatch):
    from spacr.qt.widgets import preview_contract

    made = []

    def create(path):
        model = object()
        made.append((path, model))
        return model

    monkeypatch.setattr(pv, "_MODELS", {})
    monkeypatch.setattr(preview_contract, "preview_cellpose_model", create)
    first = pv._cellpose_model("one.pt")
    assert pv._cellpose_model("one.pt") is first
    second = pv._cellpose_model("two.pt")
    assert second is not first and pv._MODELS == {"two.pt": second}
    assert [path for path, model in made] == ["one.pt", "two.pt"]


@pytest.mark.parametrize("entry_point", ["plaque", "figure", "well"])
def test_all_default_segmenters_forward_the_same_settings(tmp_path, monkeypatch, entry_point):
    path = _figure(tmp_path)
    settings = {"diameter": 37, "flow_threshold": 0.27, "CP_prob": -2.5,
                "figure_imgsz": [160], "figure_read_text": False}
    seen = []

    def evaluate(crop, **kwargs):
        assert kwargs["diameter"] == 37
        assert kwargs["flow_threshold"] == 0.27
        assert kwargs["cellprob_threshold"] == -2.5
        assert "channel_axis" in kwargs
        seen.append((crop.shape, kwargs))
        return _labels(crop), [], None

    model = types.SimpleNamespace(eval=evaluate)
    monkeypatch.setattr(pv, "resolve_plaque_model", lambda value: ("chosen.pt", "resolved", None))
    monkeypatch.setattr(pv, "_cellpose_model", lambda path: model)
    if entry_point == "plaque":
        result = pv.plaque_pass(path, settings)
        assert len(seen) == 1 and seen[0][0] == (160, 200, 3)
        assert result["count"] == 1 and result["mean_area"] == 25
    elif entry_point == "figure":
        result = pv.figure_pass(path, settings, detect=_detect)
        assert len(seen) == 1 and seen[0][0] == (60, 60, 3)
        assert result["counts"] == [1] and result["mean_areas"] == [25]
    else:
        region = types.SimpleNamespace(x0=20, y0=20, x1=80, y1=80)
        result = pv.segment_well(np.full((160, 200, 3), 80, np.uint8), region, settings)
        assert len(seen) == 1 and seen[0][0] == (60, 60, 3)
        assert result["count"] == 1 and result["mean_area"] == 25
    assert result["note"] == "resolved"


@pytest.mark.parametrize("entry_point", ["plaque", "figure", "well"])
@pytest.mark.parametrize("failure", ["missing", "broken"])
def test_model_errors_are_reported_without_partial_results(tmp_path, monkeypatch, entry_point, failure):
    path = _figure(tmp_path)
    entry = object()
    monkeypatch.setattr(pv, "resolve_plaque_model", lambda settings:
                        ("", "missing weights", entry) if failure == "missing" else ("broken.pt", "", None))

    def broken(path):
        raise RuntimeError("checkpoint corrupt")

    monkeypatch.setattr(pv, "_cellpose_model", broken)
    if entry_point == "plaque":
        result = pv.plaque_pass(path, {})
    elif entry_point == "figure":
        result = pv.figure_pass(path, {}, detect=_detect)
    else:
        result = pv.segment_well(np.zeros((30, 30, 3), np.uint8),
                                 types.SimpleNamespace(x0=0, y0=0, x1=20, y1=20), {})
    assert "error" in result and "labels" not in result and "counts" not in result
    if failure == "missing":
        assert result == {"error": "missing weights", "entry": entry}
    else:
        assert "checkpoint corrupt" in result["error"]


@pytest.mark.parametrize("read_text", [True, False])
def test_figure_pass_reads_text_only_when_requested_and_offsets_outlines(tmp_path, read_text):
    path = _figure(tmp_path)
    words = [pp.Word("WT", 20, 5, 40, 15)]
    reads = []

    def reader(actual):
        reads.append(actual)
        return words

    result = pv.figure_pass(path, {"figure_read_text": read_text, "figure_imgsz": [160]},
                            detect=_detect, read_text=reader, segment=_labels)
    assert reads == ([path] if read_text else [])
    assert result["words"] == (words if read_text else [])
    assert result["counts"] == [1] and result["mean_areas"] == [25]
    assert not np.array_equal(result["overlay"][25:30, 25:30], np.full((5, 5, 3), 80))
    np.testing.assert_array_equal(result["overlay"][:20, :20], np.full((20, 20, 3), 80))


def test_missing_detector_stops_before_reading_or_segmenting(tmp_path, monkeypatch):
    entry = object()
    monkeypatch.setattr(pv, "resolve_detector", lambda *args: ("", "missing detector", entry))
    assert pv.figure_pass(tmp_path / "not-there.png", {}) == {"error": "missing detector", "entry": entry}


def test_resized_labels_keep_integer_ids_and_flow_maps_keep_shape(tmp_path):
    path = _figure(tmp_path)
    mask = np.zeros((16, 20), np.int32)
    mask[2:6, 2:6] = 700
    result = pv.plaque_pass(path, {}, segment=lambda path:
                            (mask, {"flow_rgb": np.zeros((16, 20, 3), np.uint8),
                                    "cellprob": np.ones((16, 20), np.float32)}))
    assert result["labels"].shape == (160, 200)
    assert set(np.unique(result["labels"])) == {0, 700}
    assert result["count"] == 1 and result["areas"] == [1600]
    assert result["flow_rgb"].shape == (160, 200, 3)
    np.testing.assert_array_equal(result["cellprob"], np.ones((160, 200)))
