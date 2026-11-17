"""Item 963 on screen: the Classify menu and the per-class threshold window.

Driven with real mouse and key events (offscreen): a click on the slider
moves the shown class's threshold, the crops under it are captioned
unassigned and the counts follow, each class keeps its own threshold, and
Save writes the thresholds into the classification settings file. The
Classify button is an alpha feature (``ALPHA_FEATURES[963]``).
"""
from __future__ import annotations

import pandas as pd
import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QPoint, Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402

from spacr import active_learning as al  # noqa: E402
from tests.test_annotate_classifier_choice_963 import build  # noqa: E402


@pytest.fixture
def alpha(monkeypatch):
    from spacr.qt import preferences

    state = {"on": True}
    monkeypatch.setattr(preferences, "_get_show_alpha_features",
                        lambda: state["on"])
    return state


@pytest.fixture
def plate(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    src = tmp_path / "plate1"
    (src / "measurements").mkdir(parents=True)
    build(src / "measurements", per_well=12, labelled_per_well=8,
          images=True)
    return src


def _screen(qtbot, plate):
    from spacr.qt.screens.annotate import AnnotateScreen

    widget = AnnotateScreen()
    qtbot.addWidget(widget)
    widget._settings.image_size = (32, 32)
    widget.resize(900, 700)
    widget._open_source(str(plate))
    qtbot.waitUntil(lambda: bool(widget._page_paths), timeout=10000)
    return widget


@pytest.fixture
def annotate(qtbot, qt_theme_applied, plate, alpha):
    widget = _screen(qtbot, plate)
    yield widget
    if widget._worker is not None:
        widget._worker.stop(wait=True)


def _classified(qtbot, widget, action_name):
    action = next(a for a in widget._classify_menu.actions()
                  if a.objectName() == action_name)
    action.trigger()
    qtbot.waitUntil(lambda: widget._class_threshold_dialog is not None,
                    timeout=120000)
    dialog = widget._class_threshold_dialog
    qtbot.waitExposed(dialog)
    return dialog


def _click_slider(qtbot, slider, fraction):
    x = int(slider.width() * fraction)
    QTest.mouseClick(slider, Qt.LeftButton, Qt.NoModifier,
                     QPoint(x, slider.height() // 2))
    qtbot.wait(10)


def test_the_classify_menu_offers_auto_xgboost_and_resnet50(annotate):
    button = annotate._btn_classify
    assert button.objectName() == "AnnotateClassify"
    assert not button.isHidden()
    assert [a.text() for a in button.menu().actions()] == [
        "Auto classifier", "XGBoost (default)", "ResNet50 on the images…"]


def test_the_classify_button_is_hidden_while_alpha_is_off(
        qtbot, qt_theme_applied, plate, alpha):
    alpha["on"] = False
    widget = _screen(qtbot, plate)
    try:
        assert widget._btn_classify.isHidden()
    finally:
        if widget._worker is not None:
            widget._worker.stop(wait=True)


def test_the_slider_leaves_crops_under_the_threshold_unassigned(
        qtbot, annotate):
    dialog = _classified(qtbot, annotate, "AnnotateClassifyXGBoost")
    first = dialog.current_class()
    assert dialog._slider.value() == 0
    assert all(kept for _c, kept in dialog.tile_states())
    confidences = [c for c, _k in dialog.tile_states()]
    assert confidences == sorted(confidences, reverse=True)
    before = dialog._counts.text()

    _click_slider(qtbot, dialog._slider, 0.9)
    value = dialog._slider.value()
    assert 70 <= value <= 100
    cut = dialog.thresholds()[first]
    assert cut == pytest.approx(value / 100)
    assert dialog._value.text() == f"{cut:.2f}"
    states = dialog.tile_states()
    assert any(not kept for _c, kept in states)
    for (conf, kept), (_tile, caption, _conf) in zip(states, dialog._tiles):
        assert kept == (conf >= cut)
        assert caption.text().endswith("unassigned") == (not kept)
        assert (_tile.graphicsEffect() is not None) == (not kept)
    assert dialog._counts.text() != before

    QTest.keyClick(dialog._slider, Qt.Key_Left)
    assert dialog.thresholds()[first] == pytest.approx((value - 1) / 100)

    dialog._class_box.setFocus()
    QTest.keyClick(dialog._class_box, Qt.Key_Down)
    second = dialog.current_class()
    assert second != first
    assert dialog._slider.value() == 0
    assert all(kept for _c, kept in dialog.tile_states())
    _click_slider(qtbot, dialog._slider, 0.6)
    QTest.keyClick(dialog._class_box, Qt.Key_Up)
    assert dialog.current_class() == first
    assert dialog._slider.value() == value - 1

    saved = dialog.thresholds()
    QTest.mouseClick(dialog._save, Qt.LeftButton)
    from spacr.cli import load_settings_file
    settings = load_settings_file(dialog._settings_path)
    assert settings["class_thresholds"] == {
        k: round(v, 2) for k, v in saved.items()}
    table = pd.read_csv(al._classify_paths(
        settings["db_path"], "annotate")["table"], keep_default_na=False,
        dtype=str)
    unassigned = table["class_thresholded"] == ""
    assert unassigned.any() and (~unassigned).any()


def test_auto_shows_the_comparison_with_its_control_rows(
        qtbot, annotate, monkeypatch):
    monkeypatch.setattr(al, "_CLASSIFY_CANDIDATES",
                        ("xgboost", "logistic_regression"))
    dialog = _classified(qtbot, annotate, "AnnotateClassifyAuto")
    from PySide6.QtWidgets import QTableWidget

    view = dialog.findChild(QTableWidget, "AnnotateClassifierComparison")
    names = [view.item(r, 0).text() for r in range(view.rowCount())]
    assert names[:2] == ["xgboost", "logistic_regression"]
    assert al._CONTROL_MAJORITY in names and al._CONTROL_SHUFFLED in names
    assert "resnet50" in names
    chosen = [view.item(r, 0).text() for r in range(view.rowCount())
              if view.item(r, 4).text().startswith("chosen")]
    assert len(chosen) == 1 and chosen[0] in ("xgboost",
                                              "logistic_regression")


def test_resnet50_hands_the_source_to_the_classify_screen(qtbot, annotate):
    action = next(a for a in annotate._classify_menu.actions()
                  if a.objectName() == "AnnotateClassifyResNet50")
    with qtbot.waitSignal(annotate.train_requested, timeout=5000) as sent:
        action.trigger()
    screen, seed = sent.args
    assert screen == "classify"
    assert seed["model_type"] == "resnet50"
    assert seed["annotation_column"] == annotate._settings.annotation_column
