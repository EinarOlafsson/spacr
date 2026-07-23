"""Tests for the Batch-5 onboarding features."""
from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _isolated_qsettings(monkeypatch, qt_theme_applied, tmp_path):
    from PySide6.QtCore import QCoreApplication, QSettings
    QCoreApplication.setOrganizationName("spacr-test")
    QCoreApplication.setApplicationName("qt-onboarding-test")
    QSettings.setDefaultFormat(QSettings.IniFormat)
    QSettings.setPath(QSettings.IniFormat, QSettings.UserScope,
                        str(tmp_path))
    QSettings("spacr", "qt").clear()
    # `_skip_first_launch_tour` (autouse in conftest) marks the tour
    # seen so MainWindow-constructing tests don't fire the overlay,
    # but our `.clear()` above wipes it. Re-mark AFTER clearing.
    try:
        from spacr.qt.first_run import mark_tour_seen
        mark_tour_seen()
    except Exception:
        pass
    yield


# ---------------------------------------------------------------------------
# First-launch tour state
# ---------------------------------------------------------------------------

def test_tour_state_defaults_to_unseen(qt_theme_applied):
    # The _isolated_qsettings fixture marks the tour seen so
    # MainWindow-constructing tests skip the overlay; here we
    # explicitly want the "no state stored" case.
    from spacr.qt.first_run import (
        was_tour_shown, reset_tour_state,
    )
    reset_tour_state()
    assert was_tour_shown() is False


def test_mark_and_reset_tour_state(qt_theme_applied):
    from spacr.qt.first_run import (
        mark_tour_seen, was_tour_shown, reset_tour_state,
    )
    mark_tour_seen()
    assert was_tour_shown() is True
    reset_tour_state()
    assert was_tour_shown() is False


def test_maybe_show_tour_respects_seen_flag(qtbot, qt_theme_applied):
    from spacr.qt.app import MainWindow
    from spacr.qt.first_run import (
        maybe_show_tour, mark_tour_seen,
    )
    win = MainWindow()
    qtbot.addWidget(win)
    mark_tour_seen()
    assert maybe_show_tour(win) is None    # already seen → skipped


def test_maybe_show_tour_force_bypasses_seen(qtbot, qt_theme_applied):
    from spacr.qt.app import MainWindow
    from spacr.qt.first_run import maybe_show_tour, mark_tour_seen
    win = MainWindow()
    qtbot.addWidget(win)
    mark_tour_seen()
    overlay = maybe_show_tour(win, force=True)
    assert overlay is not None
    overlay.close()


def test_default_tour_has_at_least_five_steps():
    from spacr.qt.first_run import DEFAULT_TOUR
    assert len(DEFAULT_TOUR) >= 5
    for step in DEFAULT_TOUR:
        assert step.title and step.body


# ---------------------------------------------------------------------------
# Recent runs on the home page
# ---------------------------------------------------------------------------

def test_home_page_shows_no_recent_runs_when_history_empty(
        qtbot, qt_theme_applied, tmp_path, monkeypatch,
):
    from spacr import run_journal as rj
    monkeypatch.setattr(rj, "runs_root", lambda: tmp_path)
    from spacr.qt.app import MainWindow
    win = MainWindow()
    qtbot.addWidget(win)
    # There's no RECENT RUNS label anywhere in the widget tree
    from PySide6.QtWidgets import QLabel
    labels = [w.text() for w in win.findChildren(QLabel)]
    assert not any("RECENT RUNS" in lbl for lbl in labels)


def test_home_page_shows_recent_runs_when_history_present(
        qtbot, qt_theme_applied, tmp_path, monkeypatch,
):
    from spacr import run_journal as rj
    monkeypatch.setattr(rj, "runs_root", lambda: tmp_path)
    with rj.open_run("mask", {"src": "/tmp/x"}):
        pass
    from spacr.qt.app import MainWindow
    win = MainWindow()
    qtbot.addWidget(win)
    from PySide6.QtWidgets import QLabel
    labels = [w.text() for w in win.findChildren(QLabel)]
    assert any("RECENT RUNS" in lbl for lbl in labels)


# ---------------------------------------------------------------------------
# Empty-state banner on AppScreen
# ---------------------------------------------------------------------------

def test_empty_state_banner_visible_when_src_empty(qtbot, qt_theme_applied):
    from spacr.qt.screens.app_screen import AppScreen
    screen = AppScreen("mask")
    qtbot.addWidget(screen)
    assert screen._empty_state_card is not None


def test_empty_state_banner_hides_when_src_gets_text(qtbot, qt_theme_applied):
    from PySide6.QtWidgets import QLineEdit
    from spacr.qt.screens.app_screen import AppScreen
    screen = AppScreen("mask")
    qtbot.addWidget(screen)
    src_widget = screen._settings_model._widgets.get("src")
    assert isinstance(src_widget, QLineEdit)
    src_widget.setText("/tmp/some/plate")
    for _ in range(3):
        qtbot.wait(30)
    assert not screen._empty_state_card.isVisible()


# ---------------------------------------------------------------------------
# Command palette wire
# ---------------------------------------------------------------------------

def test_command_palette_can_be_constructed(qtbot, qt_theme_applied):
    from spacr.qt.app import MainWindow
    from spacr.qt.command_palette import CommandPalette
    win = MainWindow()
    qtbot.addWidget(win)
    palette = CommandPalette(win)
    qtbot.addWidget(palette)
    # Commands should include at least the apps + Preferences + shortcut cheat
    labels = [c.label for c in palette._commands]
    assert any("Mask" in l for l in labels)
    assert any("Preferences" in l for l in labels)
    assert any("shortcuts" in l.lower() for l in labels)


def test_command_palette_filter_narrows_results(qtbot, qt_theme_applied):
    from spacr.qt.app import MainWindow
    from spacr.qt.command_palette import CommandPalette
    win = MainWindow()
    qtbot.addWidget(win)
    palette = CommandPalette(win)
    qtbot.addWidget(palette)
    palette._on_filter("preferences")
    # After filtering, at least one visible row should mention preferences
    from PySide6.QtWidgets import QListWidgetItem
    visible = [palette._list.item(i).text()
                for i in range(palette._list.count())]
    assert any("Preferences" in v for v in visible)
