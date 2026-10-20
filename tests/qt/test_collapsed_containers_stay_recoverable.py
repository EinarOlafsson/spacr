"""A collapsed container can always be opened again, at any GUI or font scale.

A macOS user of 1.5.1.5 collapsed Mask's Actions and System containers. At
75 % GUI scale (the default on screens narrower than 1600 px since
d034b0b873) the folded headings were held at three quarters of their own
height: ``CollapsibleSplitter._apply_limits`` measured the heading in drawn
pixels and handed it to ``setMaximumHeight``, which the GUI-scale layer
scaled again (the same mistake 05d6a523c8 fixed for the pane floors). The
titles were cut to a few pixels, nothing could be clicked, a collapsed pane
cannot be dragged, and the fold was restored on every launch.

Held here, at every GUI and font scale the Preferences sliders allow,
matched and mismatched, at 1440 x 870 and 1024 x 640:

* restored from a saved state with Actions and System folded and damaged
  pane sizes, every collapsed heading is whole and its handle can be
  grabbed (the shared check the distribution smoke runs);
* every collapsible container of every module, collapsed, passes the same
  check (enumerated from the module registry, so a new one is covered);
* a module screen that scrolls when cramped fills its scroll area at every
  GUI scale (it was resized in 100 % units and drew at half size at 50 %,
  squeezing its last collapsed heading);
* a click on a collapsed heading opens it, and dragging its handle does too;
* zero, negative, absurd and non-numeric saved sizes are repaired;
* Help > Window > "Reset this module's layout" restores the defaults.
"""
from __future__ import annotations

import json
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtCore import QEvent, QPoint, QPointF, Qt  # noqa: E402
from PySide6.QtGui import QMouseEvent  # noqa: E402
from PySide6.QtWidgets import QApplication, QVBoxLayout, QWidget  # noqa: E402

from spacr.qt import gui_scale  # noqa: E402
from spacr.qt import preferences as prefs  # noqa: E402
from spacr.qt.widgets import collapsible_splitter as cs  # noqa: E402

pytestmark = pytest.mark.qt

WINDOWS = [(1440, 870), (1024, 640)]

_LOW, _HIGH = prefs.GUI_SCALE_MIN, prefs.GUI_SCALE_MAX
_FONT_LOW, _FONT_HIGH = prefs.FONT_SCALE_MIN, prefs.FONT_SCALE_MAX

#: Every slider stop asked for, both sliders together, then mismatched pairs.
SCALES = ([(value, value) for value in
           (_LOW, 0.5, 0.75, 1.0, 1.25, 1.5, _HIGH)]
          + [(0.5, 1.5), (1.5, 0.5), (_LOW, _FONT_HIGH), (_HIGH, _FONT_LOW),
             (0.75, 1.0), (1.0, 0.75), (0.5, _FONT_HIGH)])


def _settle(qtbot, ms: int = 120) -> None:
    """Let queued re-layouts and timers run."""
    for _turn in range(4):
        QApplication.processEvents()
    qtbot.wait(ms)
    QApplication.processEvents()


def _apply_scale(qapp, gui: float, font: float) -> None:
    """Change GUI and font scale while the app runs, as Preferences does."""
    prefs.set_gui_scale(gui)
    prefs.set_font_scale(font)
    gui_scale.set_gui_scale_live(gui)
    prefs.apply_preferences_to_app(qapp)


@pytest.fixture
def scales_restored(qapp):
    """Put both scales back to 100 % after the test."""
    yield
    _apply_scale(qapp, 1.0, 1.0)


def _main_window(qtbot, size):
    """The real main window at ``size``."""
    import spacr.qt
    from spacr.qt.app import MainWindow

    spacr.qt.register_self_registering_modules()
    prefs.set_preload_policy("on_demand")
    window = MainWindow(initial_app="__home__")
    qtbot.addWidget(window)
    window.resize(*size)
    window.show()
    return window


def _open(window, qtbot, key):
    """Open module ``key`` and return its screen."""
    window._on_nav_selected(key)
    qtbot.waitUntil(lambda: window._stack.currentWidget() is not None,
                    timeout=60000)
    _settle(qtbot, 200)
    return window._stack.currentWidget()


def _press(handle, kind, point, buttons):
    """Send one mouse event of ``kind`` to ``handle`` at ``point``."""
    local = QPointF(point)
    event = QMouseEvent(kind, local, QPointF(handle.mapToGlobal(point)),
                        Qt.LeftButton, buttons, Qt.NoModifier)
    QApplication.sendEvent(handle, event)


def _drag(handle, qtbot, along: int) -> None:
    """Drag ``handle`` by ``along`` pixels along its splitter."""
    vertical = handle.orientation() == Qt.Vertical
    start = handle.rect().center()
    _press(handle, QEvent.MouseButtonPress, start, Qt.LeftButton)
    for step in (1, 2, 3):
        delta = int(along * step / 3)
        _press(handle, QEvent.MouseMove,
               start + (QPoint(0, delta) if vertical else QPoint(delta, 0)),
               Qt.LeftButton)
        QApplication.processEvents()
    end = start + (QPoint(0, along) if vertical else QPoint(along, 0))
    _press(handle, QEvent.MouseButtonRelease, end, Qt.NoButton)
    _settle(qtbot)


def _click(widget, qtbot) -> None:
    """Click ``widget`` in its middle."""
    qtbot.mouseClick(widget, Qt.LeftButton, pos=widget.rect().center())
    _settle(qtbot)


@pytest.mark.parametrize("size", WINDOWS, ids=lambda s: f"{s[0]}x{s[1]}")
def test_restored_folds_on_mask_stay_whole_at_every_scale(
        qapp, qtbot, scales_restored, size):
    prefs.set_folded_panel("mask/Actions", True)
    prefs.set_folded_panel("mask/System", True)
    prefs._settings().setValue(
        "layout/panes/mask::runtime",
        json.dumps({"Actions": 0, "System": -40, "Console": "tall",
                    "Figures": 10 ** 9}))
    window = _main_window(qtbot, size)
    try:
        screen = _open(window, qtbot, "mask")
        split = screen._runtime_splitter
        assert split.is_collapsed("Actions") and split.is_collapsed("System")
        repaired = prefs._settings().value("layout/panes/mask::runtime", "")
        assert not repaired or json.loads(repaired) == {}
        for gui, font in SCALES:
            _apply_scale(qapp, gui, font)
            _settle(qtbot, 200)
            assert cs._collapsed_header_faults(screen) == [], (gui, font)
            assert cs._probe_collapsed_headers(
                screen, lambda: _settle(qtbot)) == [], (gui, font)
            assert split.is_collapsed("Actions"), (gui, font)
        _apply_scale(qapp, 0.75, 0.75)
        _settle(qtbot, 200)
        _click(split.pane("System").folder.heading, qtbot)
        assert not split.is_collapsed("System")
        actions = split.pane("Actions").widget
        handle = split.handle(split.indexOf(actions))
        assert handle.isVisible() and handle.width() > 0
        assert handle.height() > 0
        _drag(handle, qtbot, -40)
        assert not split.is_collapsed("Actions")
        assert prefs.get_folded_panels().get("mask/Actions") is None
    finally:
        window.close()


@pytest.mark.parametrize("size", WINDOWS, ids=lambda s: f"{s[0]}x{s[1]}")
def test_every_module_container_collapses_recoverably(
        qapp, qtbot, scales_restored, size):
    """Every CollapsibleSplitter pane of every registered module, collapsed."""
    from spacr.qt import app as app_mod

    window = _main_window(qtbot, size)
    keys = [entry[0] for entry in app_mod.APPS] + ["make_masks"]
    judged = 0
    try:
        for gui, font in ((1.0, 1.0), (0.5, 1.5), (1.5, 0.5)):
            _apply_scale(qapp, gui, font)
            faults = {}
            for key in keys:
                screen = _open(window, qtbot, key)
                splitters = cs._splitters_under(screen)
                judged += sum(len(split.panes()) for split in splitters)
                found = cs._probe_collapsed_headers(
                    screen, lambda: _settle(qtbot, 60))
                if found:
                    faults[key] = found
            assert faults == {}, json.dumps([gui, font, faults], default=str)
        assert judged > 0
    finally:
        window.close()


@pytest.mark.parametrize("gui", [0.5, 0.75, 1.25])
def test_a_cramped_screen_fills_its_scroll_area_at_any_gui_scale(
        qapp, qtbot, scales_restored, gui):
    """The scroll's content is sized in drawn pixels, not scaled again."""
    from spacr.qt.app import _CrampedScroll

    _apply_scale(qapp, gui, gui)
    window = _main_window(qtbot, WINDOWS[0])
    try:
        screen = _open(window, qtbot, "foreign")
        scroll = screen.findChild(_CrampedScroll)
        assert scroll is not None
        content = scroll.widget()
        room = scroll.viewport().size()
        assert content.width() >= room.width(), (gui, content.size(), room)
        assert content.height() >= room.height(), (gui, content.size(), room)
    finally:
        window.close()


def test_the_reset_action_restores_the_default_layout(
        qapp, qtbot, scales_restored):
    window = _main_window(qtbot, WINDOWS[0])
    try:
        screen = _open(window, qtbot, "mask")
        split = screen._runtime_splitter
        for name in ("Actions", "System"):
            split.pane(name).folder.set_shut(True, by_user=True)
        split.pane("Console").extent = 77
        split._save_extents()
        _settle(qtbot)
        assert prefs.get_folded_panels().get("mask/Actions") is True
        assert cs.get_pane_extents(split._persist_key)
        action = window._act_reset_layout
        assert action.objectName() == "ResetModuleLayoutAction"
        assert action in window._window_menu.actions()
        action.trigger()
        _settle(qtbot)
        assert not split.is_collapsed("Actions")
        assert not split.is_collapsed("System")
        assert "mask/Actions" not in prefs.get_folded_panels()
        assert "mask/System" not in prefs.get_folded_panels()
        assert cs.get_pane_extents(split._persist_key) == {}
        assert split.pane("Console").extent == split.pane(
            "Console").default_extent
    finally:
        window.close()


@pytest.mark.parametrize("gui,font", SCALES)
def test_a_bare_splitter_keeps_its_folded_headings(
        qapp, qtbot, scales_restored, gui, font):
    """The splitter alone, every scale, folded before and after it is shown."""
    from spacr.qt.widgets.card import Card

    _apply_scale(qapp, gui, font)
    host = QWidget()
    qtbot.addWidget(host)
    host.resize(600, 400)
    column = QVBoxLayout(host)
    split = cs.CollapsibleSplitter(Qt.Vertical)
    column.addWidget(split)
    console = QWidget()
    split.add_pane(console, "Console", extent=300, minimum=0)
    cards = []
    for name in ("System", "Actions"):
        card = Card(title=name, foldable=True)
        card.body_layout.addWidget(QWidget())
        card.folder.set_shut(True, by_user=False)
        split.add_pane(card, name, folder=card.folder, stretch=0)
        cards.append(card)
    host.show()
    _settle(qtbot)
    assert cs._collapsed_header_faults(host) == [], (gui, font)
    _apply_scale(qapp, font, gui)
    _settle(qtbot)
    assert cs._collapsed_header_faults(host) == [], (font, gui)
    _click(cards[0].folder.heading, qtbot)
    assert not split.is_collapsed("System")


def test_damaged_saved_sizes_are_repaired_on_load(qapp):
    key = "test-only::repair"
    prefs._settings().setValue(
        f"layout/panes/{key}",
        json.dumps({"A": 0, "B": -3, "C": "x", "D": 10 ** 12, "E": 240}))
    split = cs.CollapsibleSplitter(Qt.Vertical, persist_key=key)
    assert split._stored == {"E": 240}
    assert json.loads(prefs._settings().value(
        f"layout/panes/{key}", "")) == {"E": 240}
    prefs._settings().setValue(f"layout/panes/{key}", "[not json")
    split = cs.CollapsibleSplitter(Qt.Vertical, persist_key=key)
    assert split._stored == {}
    assert not prefs._settings().value(f"layout/panes/{key}", "")
    split.deleteLater()
