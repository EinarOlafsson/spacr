"""Item 424 in Figure mode's preview: conflicts, scale and the PDF text layer.

* When the label beside a well and its panel's legend passage disagree, the
  Wells table says so in the Source column, and the review saved to
  ``figure_annotations.csv`` carries the flag and its reason.
* The Plaques tab says what the sizes are in: mm^2 with the ruler that gave
  them, or pixels.
* A figure fetched from a PDF is read from the PDF's own text before OCR.

Every model is faked.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("PySide6")

from spacr import plaque_papers as pp  # noqa: E402
from spacr.qt.widgets import plaque_preview as ppv  # noqa: E402


class _Box:
    def __init__(self, x0, y0, x1, y1, confidence=0.9):
        self.x0, self.y0, self.x1, self.y1 = x0, y0, x1, y1
        self.confidence = confidence


def _detect(image, weights, confidence=0.25, imgsz=640, min_axis_ratio=0.0):
    return [_Box(100, 100, 180, 180), _Box(220, 100, 300, 180)]


def _read_text(path):
    return [pp.Word("A", 60, 60, 75, 75), pp.Word("WT", 120, 80, 160, 95),
            pp.Word("KO", 240, 80, 280, 95)]


def _segment_crop(crop):
    labels = np.zeros(crop.shape[:2], dtype=np.int32)
    labels[10:20, 10:20] = 1
    return labels


def _png(path: Path) -> Path:
    from PIL import Image

    image = np.full((400, 400, 3), 230, np.uint8)
    image[100:180, 100:180] = 120
    image[100:180, 220:300] = 120
    Image.fromarray(image).save(path)
    return path


@pytest.fixture
def panel(qtbot, monkeypatch):
    monkeypatch.setattr(ppv, "missing_papers_packages", lambda *a, **k: [])
    widget = ppv.PlaquePreviewPanel(threaded=False)
    qtbot.addWidget(widget)
    return widget


def _run(panel, tmp_path, legend, read_text=_read_text):
    _png(tmp_path / "fig1.png")
    (tmp_path / pp.LEGENDS_FILE).write_text(
        f'file,legend\nfig1.png,"{legend}"\n', encoding="utf-8")
    panel.apply_settings({"plaque_mode": "figure"})
    panel.load_source_async(str(tmp_path))
    assert panel.run_preview(detect=_detect, read_text=read_text,
                             segment=_segment_crop)


def test_a_disagreeing_legend_is_marked_conflict_in_the_source_column(
        panel, tmp_path):
    _run(panel, tmp_path, "Plaques.A, plaque assay of KO parasites.")
    wt = panel._table.item(0, ppv.SOURCE_COLUMN)
    ko = panel._table.item(1, ppv.SOURCE_COLUMN)
    assert wt.text().endswith("/ conflict")
    assert "ko" in wt.toolTip() and "panel A" in wt.toolTip()
    assert wt.foreground().color() == ppv.CONFLICT_COLOUR
    assert "conflict" not in ko.text()
    assert "disagree" in panel.preview_status()


def test_the_saved_review_carries_the_conflict(panel, tmp_path):
    _run(panel, tmp_path, "Plaques.A, plaque assay of KO parasites.")
    path = panel.save_annotations()
    with open(path, newline="", encoding="utf-8") as handle:
        rows = {r["region"]: r for r in csv.DictReader(handle)}
    assert rows["1"]["conflict"] == "true"
    assert "ko" in rows["1"]["conflict_reason"]
    assert rows["2"]["conflict"] == "false" and rows["2"]["strength"] == "strong"
    assert rows["1"]["label_text"] == "WT"


def test_without_a_ruler_the_plaques_tab_says_pixels(panel, tmp_path):
    _run(panel, tmp_path, "Plaques.A, plaque assay of KO parasites.")
    assert "sizes are in pixels" in panel.preview_status()
    assert panel.find_plaques_in_all_wells(segment=_segment_crop)
    rows = panel.plaque_table_rows()
    assert rows and all(r["area_mm2"] is None for r in rows)
    assert all("pixels" in r["scale"] for r in rows)
    header = panel._plaque_table.horizontalHeaderItem(
        ppv.PLAQUE_KEYS.index("scale")).text()
    assert header == "Scale"


def test_a_stated_plate_format_gives_whole_wells_a_ruler(panel, tmp_path):
    _run(panel, tmp_path, "Plaques in 6-well plates.A, WT and KO.")
    assert panel.find_plaques_in_all_wells(segment=_segment_crop)
    rows = panel.plaque_table_rows()
    ppm = 80 / 34.8
    assert rows[0]["area_mm2"] == pytest.approx(100 / ppm ** 2)
    assert "well: 6-well (legend)" in rows[0]["scale"]


def test_a_pdf_pages_text_layer_is_read_before_ocr(panel, tmp_path):
    (tmp_path / pp.TEXT_LAYER_FILE).write_text(json.dumps({"fig1.png": [
        ["A", 60, 60, 75, 75], ["WT", 120, 80, 160, 95],
        ["KO", 240, 80, 280, 95]]}), encoding="utf-8")

    def no_ocr(path):
        raise AssertionError("OCR must not run when the text layer answers")
    _run(panel, tmp_path, "Plaques.A, WT and KO.", read_text=no_ocr)
    assert "PDF's own text" in panel.preview_status()
    assert panel._table.item(0, 2).text() == "WT"
