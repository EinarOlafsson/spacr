"""The main window's shell: sidebar and tile widths, the command palette, a
window built without its optional systems, and the semantic button roles.

Widths are measured against :func:`spacr.qt.preferences.scaled_px`, which is
the one place the font-scale preference turns into pixels, at both the
100 % design scale and the shipped 150 %.
"""
from __future__ import annotations

import sys
from contextlib import contextmanager

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

from PySide6.QtCore import QSettings, QSize, Qt                  # noqa: E402
from PySide6.QtGui import (QFontMetrics, QIcon, QKeySequence,    # noqa: E402
                           QPixmap, QShortcut)
from PySide6.QtWidgets import QApplication, QPushButton          # noqa: E402

from spacr.qt import app as app_mod                              # noqa: E402
from spacr.qt import button_roles as br                          # noqa: E402
from spacr.qt import preferences                                 # noqa: E402
from spacr.qt.app import (SECTION_ASSAYS, SECTION_CORE,          # noqa: E402
                          MainWindow, Sidebar, make_home_page, visible_apps)
from spacr.qt.preferences import scaled_px                       # noqa: E402
from spacr.qt.widgets.home import AppTile, HomePage              # noqa: E402
from spacr.qt.widgets.tile import HTile                          # noqa: E402

pytestmark = pytest.mark.qt


def _close_owned_screens(window):
    """Close the window's own screens before the window is deleted."""
    for screen in list(getattr(window, "_screens", {}).values()):
        try:
            screen.close()
            screen.deleteLater()
        except RuntimeError:
            pass


def _private_preferences(tmp_path, monkeypatch, name):
    """Point preferences at a private copy of the current store."""
    source = preferences._settings()
    path = str(tmp_path / name)
    private = QSettings(path, QSettings.IniFormat)
    for key in source.allKeys():
        private.setValue(key, source.value(key))
    private.sync()
    monkeypatch.setattr(preferences, "_settings",
                        lambda: QSettings(path, QSettings.IniFormat))


# ---------------------------------------------------------------------------
# Sidebar and tile widths
# ---------------------------------------------------------------------------

def _fake_apps(names):
    return [(f"k{i}", n, f"desc {i}", SECTION_CORE)
            for i, n in enumerate(names)]


def _lay_out(widget):
    """Show and settle a widget so elision (computed in resizeEvent) runs."""
    widget.show()
    QApplication.processEvents()
    widget.resize(widget.width(), widget.height())
    QApplication.processEvents()


@pytest.fixture
def at_font_scale(qapp, qt_theme_applied):
    """Put the font-scale preference and the stylesheet on one scale."""
    from spacr.qt.theme import stylesheet

    def _apply(scale):
        preferences.set_font_scale(scale)
        qapp.setStyleSheet(stylesheet(font_scale=scale))
        qapp.processEvents()

    yield _apply
    qapp.setStyleSheet(stylesheet())
    qapp.processEvents()


def test_the_sidebar_column_sits_between_its_scaled_floor_and_cap(
        qtbot, at_font_scale, monkeypatch, subtests):
    """Short names sit on the floor, a longer one widens, a huge one elides at the cap."""
    huge = "An Extraordinarily Long Hypothetical Module Name For Testing"
    for scale in (1.0, 1.5):
        at_font_scale(scale)
        floor, cap = scaled_px(Sidebar.WIDTH_MIN), scaled_px(Sidebar.WIDTH_MAX)
        assert (floor, cap) == (round(Sidebar.WIDTH_MIN * scale),
                                round(Sidebar.WIDTH_MAX * scale))
        with subtests.test(msg=f"{scale} short names on the floor"):
            monkeypatch.setattr(app_mod, "APPS", _fake_apps(["A", "B", "C"]))
            bar = Sidebar()
            qtbot.addWidget(bar)
            assert bar.width() == floor
            _lay_out(bar)
            assert bar.clipped_items() == []
        with subtests.test(msg=f"{scale} a longer name widens"):
            monkeypatch.setattr(app_mod, "APPS", _fake_apps(
                ["A", "Cellpose Model Comparison Workbench"]))
            bar = Sidebar()
            qtbot.addWidget(bar)
            assert floor < bar.width() <= cap
            _lay_out(bar)
            assert bar.clipped_items() == []
        with subtests.test(msg=f"{scale} a huge name elides at the cap"):
            monkeypatch.setattr(app_mod, "APPS", _fake_apps([huge]))
            bar = Sidebar()
            qtbot.addWidget(bar)
            bar.resize(bar.width(), 400)
            _lay_out(bar)
            assert bar.width() == cap
            clipped = bar.clipped_items()
            assert [b.full_text().strip() for b in clipped] == [huge]
            assert clipped[0].text() != huge and "…" in clipped[0].text()
            assert clipped[0].accessibleName().strip() == huge
    with subtests.test(msg="the real registry fits without clipping"):
        monkeypatch.undo()
        bar = Sidebar()
        qtbot.addWidget(bar)
        width = bar.fitting_width()
        assert scaled_px(Sidebar.WIDTH_MIN) <= width <= scaled_px(
            Sidebar.WIDTH_MAX)
        assert bar.clipped_items() == []


@contextmanager
def _at_design_scale():
    """Keep the page and its pixel contract at the 100 % design scale."""
    from spacr.qt import gui_scale

    settings = preferences._settings()
    keys = ("prefs/font_scale", "prefs/gui_scale")
    previous = {key: settings.value(key) for key in keys
                if settings.contains(key)}
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(gui_scale, "_SCALE", 1.0)
        try:
            preferences.set_font_scale(1.0)
            preferences.set_gui_scale(1.0)
            yield
        finally:
            for key in keys:
                if key in previous:
                    settings.setValue(key, previous[key])
                else:
                    settings.remove(key)


def test_home_tiles_and_icons_are_sized_from_the_font_scale(
        qapp, qt_theme_applied, qtbot, tmp_path, monkeypatch, subtests):
    """Every tile fits the longest name under the scaled cap; an HTile icon is scaled."""
    _private_preferences(tmp_path, monkeypatch, "home.ini")
    preferences.set_easy_mode(False)
    monkeypatch.setattr(preferences, "DEFAULT_SHOW_ALPHA", True)
    with _at_design_scale():
        page = make_home_page()
        qtbot.addWidget(page)
        page.resize(1500, 950)
        page.show()
        qapp.processEvents()
        tiles = {t.text_label: t for t in page.findChildren(AppTile)}
        with subtests.test(msg="longest tile name fits"):
            assert tiles, "the home page drew no tiles"
            name = max(tiles, key=len)
            label = tiles[name].name_label
            assert QFontMetrics(label.font()).horizontalAdvance(name) <= \
                label.available_text_width()
            assert label.is_elided() is False
            assert tiles[name].width() <= scaled_px(HomePage.TILE_MAX_W)
            assert len({tile.width() for tile in tiles.values()}) == 1
    for side in (32, 52):
        with subtests.test(msg=f"HTile icon {side}"):
            pixmap = QPixmap(32, 32)
            pixmap.fill(Qt.red)
            tile = HTile("Curate", "review masks", icon=QIcon(pixmap),
                         icon_size=side)
            qtbot.addWidget(tile)
            assert tile.icon().isNull() is False
            assert tile.iconSize() == QSize(scaled_px(side), scaled_px(side))


# ---------------------------------------------------------------------------
# The command palette, and a window without its optional systems
# ---------------------------------------------------------------------------

@pytest.fixture
def win(qtbot, qt_theme_applied):
    """A live MainWindow, cleaned up by pytest-qt."""
    w = MainWindow()
    qtbot.addWidget(w, before_close_func=_close_owned_screens)
    return w


def test_the_command_palette_filters_restores_and_navigates(win, qtbot,
                                                            subtests):
    """Filters by name or section, an empty filter restores all, Enter opens the app."""
    from spacr.qt.command_palette import CommandPalette

    palette = CommandPalette(win)
    qtbot.addWidget(palette)

    def rows():
        return [palette._list.item(i).text()
                for i in range(palette._list.count())]

    total = palette._list.count()
    assert total > 0
    with subtests.test(msg="no match"):
        palette._on_filter("zzzz-no-such-command")
        assert palette._list.count() == 0
    with subtests.test(msg="empty filter restores"):
        palette._on_filter("")
        assert palette._list.count() == total
    with subtests.test(msg="by section"):
        palette._on_filter(SECTION_ASSAYS)
        found = rows()
        for name in (n for _k, n, _d, s in visible_apps()
                     if s == SECTION_ASSAYS):
            assert any(name in r for r in found), name
    with subtests.test(msg="by name, then navigate"):
        palette._on_filter("plate viewer")
        found = rows()
        assert any("Plate Viewer" in r for r in found)
        assert not any("Mask" in r for r in found), found
        palette._on_activate()
        assert win._status_app_label.text() == "Plate Viewer"
        assert win._stack.currentWidget() is win._screens["plate_view"]


def test_the_window_opens_without_shortcuts_or_the_tour(
        qtbot, qt_theme_applied, monkeypatch, tmp_path, subtests):
    """With shortcuts and first-run refused, only the navigator's Escape remains."""
    import spacr.qt

    _private_preferences(tmp_path, monkeypatch, "optional-shortcuts.ini")
    monkeypatch.setitem(sys.modules, "spacr.qt.shortcuts", None)
    monkeypatch.setitem(sys.modules, "spacr.qt.first_run", None)
    monkeypatch.delattr(spacr.qt, "shortcuts", raising=False)
    monkeypatch.delattr(spacr.qt, "first_run", raising=False)
    for navigator in (False, True):
        with subtests.test(msg=f"navigator={navigator}"):
            preferences.set_easy_mode(navigator)
            w = MainWindow()
            qtbot.addWidget(w, before_close_func=_close_owned_screens)
            assert w._stack.currentWidget() is w._startup
            if navigator:
                escape = w._startup._workflow_escape
                assert escape.key() == QKeySequence(Qt.Key_Escape)
                assert escape.parent() is w._startup
                assert w.findChildren(QShortcut) == [escape]
            else:
                assert w.findChildren(QShortcut) == []


# ---------------------------------------------------------------------------
# Semantic button roles
# ---------------------------------------------------------------------------

def test_the_button_filter_classifies_live_buttons_and_skips_dead_ones(
        qapp, qtbot, monkeypatch, subtests):
    """Object names give a role when text cannot; a dead or dying button is skipped."""
    cases = (
        # (case, text, object name, role, busy)
        ("primary by name", "Proceed with the analysis", "PrimaryButton",
         "positive", False),
        ("danger by name", "Overwrite the database", "DangerButton",
         "negative", False),
        ("delete text beats a primary name", "Delete plate", "PrimaryButton",
         "negative", False),
        ("run text beats a danger name", "Run", "DangerButton", "positive",
         False),
        ("cancel is neutral", "Cancel", "", None, None),
        ("neutral, unnamed", "Overwrite the database", "", None, None),
    )
    for case, text, name, role, busy in cases:
        with subtests.test(msg=case):
            button = QPushButton(text)
            qtbot.addWidget(button)
            button.setObjectName(name)
            assert br._SemanticButtonFilter().classify(button) is None
            assert button.property("buttonActionRole") == role
            assert button.property("buttonActionBusy") is busy
    with subtests.test(msg="a second classification rewires and repolishes nothing"), \
            monkeypatch.context() as patch:
        button = QPushButton("Proceed with the analysis")
        qtbot.addWidget(button)
        button.setObjectName("PrimaryButton")
        repolished, busy_calls = [], []
        patch.setattr(br, "_repolish", lambda target: repolished.append(target))
        patch.setattr(br, "set_button_busy",
                      lambda target, busy: busy_calls.append((target, busy)))
        event_filter = br._SemanticButtonFilter()
        event_filter.classify(button)
        assert repolished == [button]
        event_filter.classify(button)
        assert repolished == [button]
        button.pressed.emit()
        assert busy_calls == [(button, True)]
        assert (button.property("buttonActionRole"),
                button.property("buttonActionBusy")) == ("positive", False)
    with subtests.test(msg="deleted C++ side"):
        from shiboken6 import delete

        dead = QPushButton("Cancel")
        delete(dead)
        assert br._SemanticButtonFilter().classify(dead) is None
    with subtests.test(msg="only a re-entrant deletion is swallowed"):
        button = object()
        answers = iter((True, False, True, True))
        real_alive = br.alive
        real_classify = br._SemanticButtonFilter._classify_live

        def refuse(_self, candidate):
            # Only the synthetic object refuses; the application-wide filter
            # keeps classifying real buttons through the real code.
            if candidate is not button:
                return real_classify(_self, candidate)
            raise RuntimeError("classification failed")

        with monkeypatch.context() as patch:
            patch.setattr(br, "alive", lambda candidate: next(answers)
                          if candidate is button else real_alive(candidate))
            patch.setattr(br._SemanticButtonFilter, "_classify_live", refuse)
            event_filter = br._SemanticButtonFilter()
            assert event_filter.classify(button) is None
            with pytest.raises(RuntimeError, match="classification failed"):
                event_filter.classify(button)
