"""Item 593: Measure's settings regrouped, and its mask QC behind a button.

The maintainer, 2026-09-28: the QC from masks pushed Run, Stop and the other
actions up the actions container; it is now opt-in behind a QC button left
of the 3D and Time switches, opening a popup. Cloud is an alpha sub-category
under the inputs (Measure and Mask generation); Preview & Diagnostics'
settings join Input & Experiment; an Image Preprocessing category holds the
alpha corrections; "Measurement Features" is "Features" with the per-object
analyses nested in it; Postprocessing holds Runtime & Reliability and
Profiling; and time to event is greyed unless timelapse is on.
"""
from __future__ import annotations

import os

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QSettings                              # noqa: E402

from spacr import settings as S                                   # noqa: E402

PREPROCESSING = ("Bleach Correction α", "Spectral Unmixing α",
                 "Image Deconvolution α", "Illumination Correction α",
                 "Image Enhancement α", "Plate Barcode Linkage α")
FEATURES = ("Confluency α", "Cell Cycle α", "Wound Closure α", "Viability α",
            "CellProfiler α", "GPU Measurement α")


@pytest.fixture
def prefs(tmp_path, monkeypatch):
    """Preferences read and written through a throwaway INI file."""
    from spacr.qt import preferences

    path = tmp_path / "alpha.ini"
    monkeypatch.setattr(preferences, "_settings",
                        lambda: QSettings(str(path), QSettings.IniFormat))
    return preferences


def test_the_measure_layout_is_regrouped():
    from spacr.qt.screens.settings_model import (_category_parents_for,
                                                 categories_for_app)

    sections = categories_for_app("measure", S.categories)
    for gone in ("Preview & Diagnostics", "Measurement Features",
                 "Point Spread Function", "Illumination Correction"):
        assert gone not in sections, gone
    inputs = sections["Input & Experiment"]
    for key in ("src", "experiment", "plot", "test_mode", "test_nr"):
        assert key in inputs, key
    assert "cloud_profile" not in inputs
    assert "cloud_profile" in sections["Cloud α"]
    assert "save_measurements" in sections["Features"]

    parents = _category_parents_for("measure")
    assert parents["Cloud α"] == "Input & Experiment"
    for title in PREPROCESSING:
        assert parents[title] == "Image Preprocessing", title
    for title in FEATURES:
        assert parents[title] == "Features", title
    assert parents["Runtime & Reliability"] == "Postprocessing"
    assert parents["Profiling α"] == "Postprocessing"
    assert "Time To Event α" not in parents
    assert _category_parents_for("mask")["Cloud α"] == "Input & Metadata"


def test_the_panel_draws_the_tree():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    QApplication.instance() or QApplication([])
    from spacr.qt.screens.settings_model import SettingsWidgets

    tree = {section.title: section
            for section in SettingsWidgets("measure").build_sections()}
    assert [c.title for c in tree["Input & Experiment"].children] \
        == ["Cloud α"]
    pre = tree["Image Preprocessing"]
    assert pre.own_rows == []
    assert [c.title for c in pre.children] == [
        t for t in PREPROCESSING if t != "Image Enhancement α"]
    assert [c.title for c in tree["Features"].children] == list(FEATURES)
    assert [c.title for c in tree["Postprocessing"].children] == [
        "Runtime & Reliability", "Profiling α"]
    order = list(tree)
    assert order.index("Image Preprocessing") < order.index("Features") \
        < order.index("Postprocessing")


def test_illumination_and_psf_are_alpha_on_measure_only():
    names = S._alpha_names("settings", "measure")
    assert {"illumination_correction", "psf_operation"} <= names
    assert "illumination_correction" not in S._alpha_names(
        "settings", "illumination")


def test_time_to_event_is_greyed_unless_timelapse_is_on(qtbot):
    from spacr.qt.screens.settings_model import SettingsWidgets

    model = SettingsWidgets("measure")
    model.build_sections()
    rules = S.get_setting_dependencies()
    keys = [k for k in S.categories["Time To Event α"] if k in rules]
    assert "time_to_event" in keys and len(keys) >= 10

    assert model.set_value_for_key("timelapse", False)
    model._refresh_setting_dependencies()
    control = model._built_control("time_to_event")
    assert control is not None and not control.isEnabled()

    assert model.set_value_for_key("timelapse", True)
    model._refresh_setting_dependencies()
    assert model._built_control("time_to_event").isEnabled()


def _heading(screen, name):
    for section in screen.rendered_settings_sections():
        if str(section.title()).upper() == name.upper():
            return section
    raise AssertionError(f"no {name!r} card on the form")


def test_an_umbrella_of_alpha_headings_hides_with_them(qtbot, prefs):
    from spacr.qt.screens.app_screen import AppScreen
    from spacr.qt.widget_cleanup import retire_pyqtgraph_menus

    screen = AppScreen("measure")
    try:
        screen._refresh_alpha_visibility()
        assert _heading(screen, "Image Preprocessing").isHidden()
        assert not _heading(screen, "Features").isHidden()
        assert not _heading(screen, "Postprocessing").isHidden()
        assert _heading(screen, "Profiling α").isHidden()

        prefs._set_show_alpha_features(True)
        screen._refresh_alpha_visibility()
        assert not _heading(screen, "Image Preprocessing").isHidden()
        assert not _heading(screen, "Illumination Correction α").isHidden()
    finally:
        retire_pyqtgraph_menus(screen)
        screen.close()
        screen.deleteLater()


def _run_top(screen):
    button = screen._btn_run
    return button.mapTo(screen, button.rect().topLeft()).y()


def test_the_action_bar_stays_put_with_the_qc(qtbot, qt_theme_applied):
    """Run's position is the same with and without the QC installed."""
    from PySide6.QtWidgets import QApplication

    from spacr.qt import prerun
    from spacr.qt.screens.app_screen import AppScreen

    plain = AppScreen("measure")
    qtbot.addWidget(plain)
    plain.resize(1400, 900)
    plain.show()
    QApplication.processEvents()
    without = _run_top(plain)
    actions_without = plain._actions_row.geometry()

    screen = AppScreen("measure")
    qtbot.addWidget(screen)
    screen.resize(1400, 900)
    screen.show()
    banner = prerun.install_qc_banner(screen, threaded=False)
    assert banner is not None
    banner.show()
    QApplication.processEvents()

    assert _run_top(screen) == without
    assert screen._actions_row.geometry() == actions_without

    row = screen._actions_row.layout()
    button = screen._btn_qc
    assert button.text() == "QC"
    at = row.indexOf(button)
    assert at >= 0
    for switch in (getattr(screen, "_dimension_switches", None) or {}).values():
        assert at < row.indexOf(switch)
    for other in (getattr(screen, "_preview_switch", None),
                  getattr(screen, "_ai_switch", None)):
        if other is not None and row.indexOf(other) >= 0:
            assert at < row.indexOf(other)

    dialog = prerun.qc_dialog(screen)
    assert not dialog.isVisible()
    button.click()
    QApplication.processEvents()
    assert dialog.isVisible()
    assert banner.window() is dialog
    assert _run_top(screen) == without
    dialog.close()
