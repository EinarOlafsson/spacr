"""Home's real System refresh action retains the selected panel text size."""

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QSettings, Qt
from PySide6.QtWidgets import QApplication, QLabel

from spacr.qt import gui_scale, preferences, theme
from spacr.qt.widgets import home


def _settle():
    for _ in range(8):
        QApplication.processEvents()


def _readings(panel):
    labels = [label for label in panel.findChildren(QLabel)
              if label.isVisible() and label.text() in {"GPU", "VRAM", "Disk", "12%", "37%",
                                  "4 / 8 GB", "10 / 20 GB"}]
    for label in labels:
        label.ensurePolished()
    assert len(labels) == 6
    return {label.text(): label.fontInfo().pixelSize() for label in labels}


@pytest.mark.parametrize("percent", [70, 160])
def test_system_refresh_keeps_selected_text_size(qtbot, qapp, tmp_path,
                                                monkeypatch, percent):
    store = QSettings(str(tmp_path / "home-text.ini"), QSettings.IniFormat)
    monkeypatch.setattr(preferences, "_settings", lambda: store)
    monkeypatch.setattr(home.HomePage, "_read_the_journal", lambda self: None)
    monkeypatch.setattr(home.QueuedPanel, "queue_items", lambda self: [])
    current = {"gpu": "12%"}
    monkeypatch.setattr(home.SystemPanel, "gpu_util",
                        staticmethod(lambda: current["gpu"]))
    monkeypatch.setattr(home.SystemPanel, "gpu_vram",
                        staticmethod(lambda: "4 / 8 GB"))
    monkeypatch.setattr(home.SystemPanel, "disk_used",
                        staticmethod(lambda: "10 / 20 GB"))
    before_sheet = qapp.styleSheet()
    qapp.setStyleSheet(theme.stylesheet("dark"))
    page = home.HomePage([("measure", "Measure", "Measure objects", "Core")],
                         lambda key: None)
    qtbot.addWidget(page)
    try:
        page.resize(1400, 900)
        page.show()
        _settle()
        base = _readings(page._system)
        page._homeAsideTextSlider.setValue(percent)
        _settle()
        scaled = _readings(page._system)
        assert scaled["GPU"] != base["GPU"]
        current["gpu"] = "37%"
        page._aside_scroll.ensureWidgetVisible(page._system._refresh)
        _settle()
        for _ in range(2):
            qtbot.mouseClick(page._system._refresh, Qt.LeftButton)
            _settle()
            actual = _readings(page._system)
            expected = {("37%" if key == "12%" else key): value
                        for key, value in scaled.items()}
            assert actual == expected
            assert store.value("prefs/home_aside_text", type=float) == percent / 100
            assert page._aside.width() >= gui_scale._actual_minimum_size(page._aside).width()
        page._homeAsideTextSlider.setValue(100)
        _settle()
        qtbot.mouseClick(page._system._refresh, Qt.LeftButton)
        _settle()
        assert _readings(page._system) == {
            ("37%" if key == "12%" else key): value
            for key, value in base.items()}
    finally:
        page.close()
        qapp.setStyleSheet(before_sheet)
