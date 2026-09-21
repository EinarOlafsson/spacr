"""Plaque Assay's two modes, run side (item 468).

2026-09-21, the maintainer: "two modes, one (plaque mode) for the cropped
images of plaques that just detects plaques and generated a database with a
table with per image values, and a table with per plaque values ... the other
(figure mode) ... the finding of plaques (the yolo model), the detection of
text on the pannel and plaque annotation with that text, and the mask
generation for the plaques."
"""
from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd
import pytest

tifffile = pytest.importorskip("tifffile")

from spacr import submodules as sm  # noqa: E402


def _plaque_folder(tmp_path):
    masks = tmp_path / "masks"
    masks.mkdir()
    labels = np.zeros((64, 64), np.uint16)
    labels[4:14, 4:14] = 1
    labels[30:50, 30:50] = 2
    tifffile.imwrite(masks / "well1.tif", labels)
    return tmp_path


def test_the_default_model_is_the_current_plaque_model():
    from spacr.settings import get_analyze_plaque_settings

    settings = get_analyze_plaque_settings({})
    assert settings["plaque_model"] == "toxoplasma_plaque_v2"
    assert settings["plaque_mode"] == "plaque"


def test_a_cellpose3_refusal_becomes_advice():
    refusal = ValueError("This model does not appear to be a CP4 model. "
                         "CP3 models are not compatible with CP4.")
    explained = sm.explain_cellpose3(refusal, "bundled")
    assert isinstance(explained, sm.Cellpose3Checkpoint)
    assert "toxoplasma_plaque_v2" in str(explained)
    other = ValueError("something else")
    assert sm.explain_cellpose3(other, "x") is other


def test_plaque_mode_writes_a_per_image_and_a_per_plaque_table(tmp_path,
                                                               monkeypatch):
    src = _plaque_folder(tmp_path)
    monkeypatch.setattr(sm, "_resolve_plaque_model", lambda s, fetch=True: "m")
    monkeypatch.setattr("spacr.utils.save_settings", lambda *a, **k: None)
    sm.analyze_plaques({"src": str(src), "masks": False})
    with sqlite3.connect(src / "masks" / "plaques_analysis.db") as db:
        image = pd.read_sql("SELECT * FROM per_image", db)
        plaques = pd.read_sql("SELECT * FROM per_plaque", db)
        legacy = {r[0] for r in db.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
    assert image.loc[0, "plaque_count"] == 2
    assert image.loc[0, "median_area_px"] == pytest.approx(250.0)
    assert sorted(plaques["area_px"]) == [100, 400]
    assert {"summary", "details", "stats"} <= legacy


def test_figure_mode_hands_the_folder_to_the_figure_engine(tmp_path,
                                                            monkeypatch):
    from spacr import plaque_papers

    seen = {}

    def measure(src, dst, **kw):
        seen.update(src=src, dst=dst, **kw)
        return {"figures": 1, "regions": 4, "plaques": 9,
                "database": str(dst), "awaiting_approval": 2}

    monkeypatch.setattr(plaque_papers, "measure_figure_folder", measure)
    monkeypatch.setattr(sm, "_resolve_plaque_model",
                        lambda s, fetch=True: "/models/r5")
    monkeypatch.setattr("spacr.utils.save_settings", lambda *a, **k: None)
    out = sm.analyze_plaques({"src": str(tmp_path), "plaque_mode": "figure",
                              "figure_imgsz": "640, 1280",
                              "confirm_annotations": True,
                              "figure_read_text": False})
    assert out["plaques"] == 9
    assert seen["segmenter"] == "/models/r5"
    assert seen["imgsz"] == (640, 1280)
    assert seen["confirm_each"] is True
    assert seen["read_text"]("any.png") == []
    assert seen["dst"].endswith("plaque_figures")


def test_detector_boxes_on_a_gpu_are_copied_to_the_host(monkeypatch):
    """2026-09-21: Figure mode failed with "can't convert cuda:0 device type
    tensor to numpy" on the maintainer's GPU; every CPU test had passed."""
    from spacr import plaque

    class DeviceTensor:
        def __init__(self, values):
            self.values = np.asarray(values, dtype=float)

        def __array__(self, *args, **kwargs):
            raise TypeError("can't convert cuda:0 device type tensor to numpy")

        def cpu(self):
            return self

        def numpy(self):
            return self.values

    class Box:
        xyxy = DeviceTensor([[1, 2, 41, 42]])
        conf = DeviceTensor([0.9])

    class Model:
        def predict(self, **kw):
            return [type("R", (), {"boxes": [Box()]})()]

    monkeypatch.setattr(plaque, "_load_detector", lambda w: Model())
    wells = plaque.detect_wells(np.zeros((50, 50, 3), np.uint8), "w.pt",
                                min_axis_ratio=0)
    assert [(w.x0, w.y0, w.x1, w.y1) for w in wells] == [(1, 2, 41, 42)]
    assert wells[0].confidence == pytest.approx(0.9)
