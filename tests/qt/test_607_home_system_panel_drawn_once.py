"""Item 607: with Home's Text size above 100%, the System panel is drawn once.

Found by spacr-d7 while re-recording tutorials, 2026-10-01: with the Text
size slider above 100%, the System panel showed every row twice. A refresh
queued the old rows for deletion but left them painted where they stood, and
at a larger text size the new rows sit at new heights, so both showed.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QEvent, QSettings                         # noqa: E402
from PySide6.QtWidgets import QApplication, QLabel                   # noqa: E402

from spacr.qt.widgets import home as home_mod                        # noqa: E402

APPS = [("mask", "Mask", "Segment cells", "Core")]


@pytest.fixture
def page(tmp_path, monkeypatch, qapp):
    """Home with a throwaway preference store and no GPU probe."""
    from spacr.qt import preferences as prefs
    from spacr.qt import theme

    path = tmp_path / "spacr-607.ini"
    monkeypatch.setattr(prefs, "_settings",
                        lambda: QSettings(str(path), QSettings.IniFormat))
    monkeypatch.setattr(home_mod, "_nvml", lambda: None)
    before = qapp.styleSheet()
    qapp.setStyleSheet(theme.stylesheet("dark"))
    built = home_mod.HomePage(APPS, lambda key: None)
    built.resize(1400, 900)
    built.show()
    _pump()
    yield built
    built.close()
    built.deleteLater()
    qapp.setStyleSheet(before)


def _pump() -> None:
    """Deliver queued shows and paints; deferred deletes are left queued, as
    they are between a refresh and the event loop's next pass."""
    for _ in range(6):
        QApplication.processEvents()


def _shown_labels(panel) -> list:
    return [label.text() for label in panel._box.findChildren(QLabel)
            if label.isVisible()]


@pytest.mark.parametrize("percent", [100, 150, 200])
def test_a_refresh_leaves_one_set_of_system_rows_on_screen(page, percent):
    page._homeAsideTextSlider.setValue(percent)
    page.refresh()
    page.refresh()
    _pump()
    assert _shown_labels(page._system) == [
        "GPU", "n/a", "VRAM", "n/a", "Disk", page._system.disk_used()]


def test_the_rows_a_refresh_replaces_are_hidden_before_they_are_deleted(page):
    page._homeAsideTextSlider.setValue(150)
    layout = page._system.body_layout
    old = [layout.itemAt(i).widget() for i in range(layout.count())]
    page.refresh()
    _pump()
    assert len(old) == 3
    assert all(row.isHidden() for row in old)
    fresh = [layout.itemAt(i).widget() for i in range(layout.count())]
    assert len(fresh) == 3 and all(row.isVisible() for row in fresh)


def test_the_grabbed_panel_matches_a_freshly_built_one(page):
    page._homeAsideTextSlider.setValue(150)
    page.refresh()
    _pump()
    after_refresh = page._system.grab().toImage()
    QApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    _pump()
    after_deletion = page._system.grab().toImage()
    assert after_refresh == after_deletion
