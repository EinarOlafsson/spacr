"""Annotate's field-QC view and the QC dashboard's classifier card.

Both are alpha: hidden until Preferences -> Show alpha features is on. A stub
classifier stands in for the trained network so the screening report carries
known class probabilities on the CPU in milliseconds.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture
def alpha(monkeypatch):
    from spacr.qt import preferences

    state = {"on": False}
    monkeypatch.setattr(preferences, "_get_show_alpha_features",
                        lambda: state["on"])
    return state


@pytest.fixture
def screened(tmp_path, monkeypatch):
    """A project whose report was written with a stub classifier."""
    from spacr import image_quality as iq

    stack = tmp_path / "stack"
    stack.mkdir()
    rng = np.random.default_rng(0)
    for index in range(3):
        field = rng.integers(100, 4000, (96, 96, 1)).astype(np.uint16)
        field[0, 0, 0] = index
        np.save(stack / f"f{index}.npy", field)
    fixed = {0: 0.9, 1: 0.1, 2: 0.2}
    monkeypatch.setattr(iq, "_prepare_qc_classifier",
                        lambda root, policy, paths, channel_ids=None: "stub")
    original = iq._classify_records

    def stub_records(model, image, records, policy, channel_ids=None):
        assert model == "stub"
        score = fixed[int(np.asarray(image)[0, 0, 0])]
        monkeypatch.setattr(iq, "_predict_qc", lambda m, t, w: np.array(
            [[score, 0.0, 0.0, 0.0, 0.0]], np.float32))
        return original(model, image, records, policy, channel_ids)

    monkeypatch.setattr(iq, "_classify_records", stub_records)
    iq.screen_fields(tmp_path, {"image_qc_mode": "report",
                                "image_qc_classifier": True})
    return tmp_path


def test_the_card_counts_classifier_flags(screened):
    from spacr.qt.widgets.qc_summary import read_dashboard

    card = read_dashboard(str(screened)).card("image_qc_classifier")
    assert card is not None and card.verdict == "warn"
    assert "3 channels scored; 1 class flags" in card.headline
    assert any(line.startswith("out_of_focus: 1 of 3") for line in card.detail)


def test_no_card_without_classifier_probabilities(tmp_path):
    from spacr.qt.widgets.qc_summary import read_dashboard

    assert read_dashboard(str(tmp_path)).card("image_qc_classifier") is None


def test_the_dashboard_card_follows_the_alpha_switch(qtbot, screened, alpha):
    from spacr.qt.preferences import _apply_alpha_widgets
    from spacr.qt.screens.qc_dashboard import QCDashboardScreen
    from spacr.qt.widgets.qc_summary import read_dashboard
    from PySide6.QtWidgets import QWidget

    screen = QCDashboardScreen()
    qtbot.addWidget(screen)
    screen._draw(read_dashboard(str(screened)))
    screen.show()
    card = screen.findChild(QWidget, "QCClassifierCard")
    assert card is not None and card.isHidden()
    alpha["on"] = True
    _apply_alpha_widgets(screen)
    assert not card.isHidden()


def test_the_annotate_button_follows_the_alpha_switch(qtbot, alpha):
    from spacr.qt.preferences import _apply_alpha_widgets
    from spacr.qt.screens.annotate import AnnotateScreen

    screen = AnnotateScreen()
    qtbot.addWidget(screen)
    assert screen._btn_field_qc.isHidden()
    alpha["on"] = True
    _apply_alpha_widgets(screen)
    assert not screen._btn_field_qc.isHidden()


def test_field_view_prefills_from_the_classifier_and_saves_labels(qtbot, screened):
    from spacr.image_quality import _parse_qc_labels
    from spacr.qt.screens.annotate import _FieldQCDialog
    from spacr.tabular import read_table

    dialog = _FieldQCDialog()
    qtbot.addWidget(dialog)
    assert dialog.load_folder(str(screened / "stack")) == 3
    assert dialog.current_labels() == {"out_of_focus"}
    assert not dialog._good.isChecked()
    dialog._good.setChecked(True)
    assert dialog.current_labels() == set()
    dialog.show_field(1)
    assert dialog.current_labels() == set() and dialog._good.isChecked()
    dialog._boxes["debris"].setChecked(True)
    dialog._boxes["bubble"].setChecked(True)
    path = dialog.save()
    table = read_table(path, report=None)
    table.columns = [str(c).lower() for c in table.columns]
    table = table.rename(columns={"fieldid": "field"})
    rows = dict(zip(table["field"], table["label"]))
    assert rows == {"f0.npy": "good", "f1.npy": "bubble;debris"}
    assert _parse_qc_labels(rows["f1.npy"]) == {"bubble", "debris"}
    again = _FieldQCDialog()
    qtbot.addWidget(again)
    again.load_folder(str(screened / "stack"))
    again.show_field(1)
    assert again.current_labels() == {"bubble", "debris"}
