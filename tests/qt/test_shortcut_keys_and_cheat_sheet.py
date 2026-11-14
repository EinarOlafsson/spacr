"""Window shortcuts: how keys are spelled, bound and discovered, what each
window-wide key does when its target is missing or broken, and the
cheat-sheet overlay (or fallback dialog) that lists them.

Every handler is driven against real widgets where a widget is what it
touches, and against small stand-ins where the point is a missing or
failing optional piece; each case asserts the exact effect or log record.
"""
from __future__ import annotations

import importlib
import logging

import pytest
from PySide6.QtCore import QEvent, QSize
from PySide6.QtGui import QAction, QKeySequence, QShortcut
from PySide6.QtWidgets import QDialog, QLabel, QLineEdit, QMainWindow, QWidget

from spacr.qt import shortcuts as sc

_LOGGER = "spacr.qt.shortcuts"


@pytest.fixture
def window(qapp):
    """A main window of a fixed size, not shown."""
    win = QMainWindow()
    win.resize(900, 600)
    yield win
    win.close()
    win.deleteLater()


@pytest.fixture
def shown_window(qapp):
    """A shown main window -- the AI key looks for the visible screen."""
    win = QMainWindow()
    win.resize(900, 600)
    win.show()
    yield win
    win.hide()
    win.close()
    win.deleteLater()


@pytest.fixture
def screen_factory(qapp):
    """Real ``AppScreen`` instances built without the full constructor."""
    from spacr.qt.screens.app_screen import AppScreen

    class _StubScreen(AppScreen):
        def __init__(self, parent, switch=None):
            QWidget.__init__(self, parent)
            if switch is not None:
                self._ai_switch = switch

        def showEvent(self, event):
            QWidget.showEvent(self, event)

        def hideEvent(self, event):
            QWidget.hideEvent(self, event)

        def paintEvent(self, event):
            QWidget.paintEvent(self, event)

        def resizeEvent(self, event):
            QWidget.resizeEvent(self, event)

        def eventFilter(self, obj, event):
            return QWidget.eventFilter(self, obj, event)

    return _StubScreen


class _Switch:
    """The AI toggle, recording every ``setChecked``."""

    def __init__(self, checked=False):
        self.checked = bool(checked)
        self.history = []

    def isChecked(self):
        return self.checked

    def setChecked(self, value):
        self.checked = bool(value)
        self.history.append(self.checked)


class _StackWindow:
    """A window-shaped object whose ``_stack`` shows one screen."""

    def __init__(self, screen):
        self._stack = type("Stack", (), {
            "currentWidget": staticmethod(lambda: screen)})()


class _Recorder:
    """An object that records every attribute asked of it and has none."""

    def __init__(self):
        object.__setattr__(self, "asked", [])

    def __getattr__(self, name):
        self.asked.append(name)
        raise AttributeError(name)


def _bound(window):
    """Native spellings of every shortcut bound under ``window``."""
    return [s.key().toString(QKeySequence.NativeText)
            for s in window.findChildren(QShortcut)]


def _record(caplog, message):
    """The single log record whose message is ``message``."""
    found = [r for r in caplog.records if r.getMessage() == message]
    assert len(found) == 1, [r.getMessage() for r in caplog.records]
    return found[0]


def test_native_prints_the_keyboards_own_spelling(monkeypatch, subtests):
    """Keys are spelled as Qt's native text; an empty sequence or one Qt
    refuses is shown exactly as written."""
    for keys in ("Ctrl+H", "Ctrl+Shift+Z", "F1", "Alt+Home", "?"):
        with subtests.test(msg=keys):
            assert sc.native(keys) == QKeySequence(keys).toString(
                QKeySequence.NativeText)
            assert sc.native(keys) != ""

    with subtests.test(msg="empty"):
        assert sc.native("") == ""

    with subtests.test(msg="unparseable"):
        def _refuse(_keys):
            raise ValueError("not a key sequence")

        monkeypatch.setattr(sc, "QKeySequence", _refuse)
        assert sc.native("Mod4+Space") == "Mod4+Space"


def test_install_binds_once_every_key_it_owns_and_none_it_does_not(window,
                                                                    subtests):
    """Each installed key is bound exactly once, even after a second
    install; window actions keep the keys bound elsewhere."""
    sc.install(window)
    sc.install(window)
    bound = _bound(window)
    owned = [spec for spec in sc.installed() if " + " not in spec.keys]
    with subtests.test(msg="installed() excludes actions and gestures"):
        assert owned == [spec for spec in sc.installed()]
        assert not {spec.keys for spec in owned} & sc.BOUND_ELSEWHERE
    for spec in owned:
        with subtests.test(msg=spec.keys):
            assert bound.count(sc.native(spec.keys)) == 1
    for keys in sorted(sc.BOUND_ELSEWHERE):
        with subtests.test(msg=f"left to its action: {keys}"):
            assert sc.native(keys) not in bound


def test_discover_lists_undeclared_runtime_keys_once(window):
    """A runtime shortcut is listed under Other once, an action brings its
    own text, and a declared or empty key is left out."""
    QShortcut(QKeySequence("Ctrl+Alt+Z"), window)
    QShortcut(QKeySequence("Ctrl+Alt+Y"), window)
    QShortcut(QKeySequence("Ctrl+Alt+Y"), window)
    QShortcut(QKeySequence("Ctrl+0"), window)
    QShortcut(QKeySequence(), window)
    action = QAction("&Export figure", window)
    action.setShortcut(QKeySequence("Ctrl+Alt+E"))
    window.addAction(action)

    assert sc.discover(window) == [
        sc.ShortcutSpec(sc.native("Ctrl+Alt+Z"), "(not described)", "Other"),
        sc.ShortcutSpec(sc.native("Ctrl+Alt+Y"), "(not described)", "Other"),
        sc.ShortcutSpec(sc.native("Ctrl+Alt+E"), "Export figure", "Other"),
    ]


def test_window_hooks_step_over_any_failure_and_pin_menu_roles(window,
                                                               monkeypatch,
                                                               subtests):
    """Every optional hook runs in order whichever one raises, and the
    macOS menu-role sweep runs last, even after a failure of its own."""
    modules = ["spacr.qt.widgets.feature_dictionary", "spacr.qt.settings_search",
               "spacr.qt.help_search", "spacr.qt.recipes",
               "spacr.qt.preview_registry", "spacr.qt.walkthrough",
               "spacr.qt.screens.map_barcodes"]
    ran = []

    def _patch(broken):
        for path in modules:
            module = importlib.import_module(path)
            if path == broken:
                def _hook(_window, _path=path):
                    ran.append(_path)
                    raise RuntimeError(f"{_path} is unhappy")
            else:
                def _hook(_window, _path=path):
                    ran.append(_path)
            monkeypatch.setattr(module, "install_window_hooks", _hook)

    for broken in [None] + modules:
        with subtests.test(msg=f"broken={broken}"):
            ran.clear()
            _patch(broken)
            window.pin_all_menu_roles = lambda: ran.append("pin")
            sc._install_window_hooks(window)
            assert ran == modules + ["pin"]

    with subtests.test(msg="a sweep that raises"):
        ran.clear()
        _patch(None)

        def _explode():
            ran.append("pin")
            raise RuntimeError("no menu bar")

        window.pin_all_menu_roles = _explode
        sc._install_window_hooks(window)
        assert ran == modules + ["pin"]

    with subtests.test(msg="no sweep to run"):
        ran.clear()
        del window.pin_all_menu_roles
        sc._install_window_hooks(window)
        assert ran == modules


def test_the_question_mark_goes_to_the_screen_or_the_cheat_sheet(monkeypatch,
                                                                 subtests):
    """A screen that claims ``?`` keeps it; one that declines, raises, has
    no handler, or no stack at all gets the cheat sheet."""
    shown = []
    monkeypatch.setattr(sc, "show_cheat_sheet", lambda parent: shown.append(parent))

    def _screen(result):
        class _Screen:
            def __init__(self):
                self.keys = []

            def handle_key(self, key):
                self.keys.append(key)
                if isinstance(result, Exception):
                    raise result
                return result
        return _Screen()

    # (case, window builder, whether the sheet is shown)
    cases = [
        ("claimed", lambda: _StackWindow(_screen(True)), False),
        ("declined", lambda: _StackWindow(_screen(False)), True),
        ("handler raises", lambda: _StackWindow(_screen(RuntimeError("repaint"))), True),
        ("no handler", lambda: _StackWindow(object()), True),
        ("no current screen", lambda: _StackWindow(None), True),
        ("no stack", object, True),
    ]
    for case, build, sheet in cases:
        with subtests.test(msg=case):
            shown.clear()
            target = build()
            sc._help_key(target)
            assert shown == ([target] if sheet else [])
            screen = getattr(getattr(target, "_stack", None), "currentWidget",
                             lambda: None)()
            if hasattr(screen, "keys"):
                assert screen.keys == ["?"]


def test_a_numbered_key_opens_only_a_visible_app_in_range(monkeypatch,
                                                          subtests):
    """Ctrl+N opens the N-th app when it exists and is visible; out of range,
    hidden or an unreadable registry does nothing."""
    from spacr.qt import app as qt_app

    monkeypatch.setattr(qt_app, "APPS", [("mask", "Mask"), ("measure", "M"),
                                         ("hidden", "H")], raising=False)

    def _visibility(key):
        if key == "broken":
            raise RuntimeError("the registry is gone")
        return key != "hidden"

    monkeypatch.setattr(qt_app, "app_is_visible", _visibility, raising=False)
    cases = [(0, ["mask"]), (1, ["measure"]), (2, []), (3, []), (8, []),
             (-1, [])]
    for idx, expected in cases:
        with subtests.test(msg=f"index {idx}"):
            picked = []
            target = type("W", (), {"_on_nav_selected":
                                    lambda self, key: picked.append(key)})()
            sc._nav_by_index(target, idx)
            assert picked == expected

    with subtests.test(msg="a registry that raises"):
        monkeypatch.setattr(qt_app, "APPS", [("broken", "B")], raising=False)
        picked = []
        target = type("W", (), {"_on_nav_selected":
                                lambda self, key: picked.append(key)})()
        sc._nav_by_index(target, 0)
        assert picked == []

    with subtests.test(msg="a window without the route"):
        monkeypatch.setattr(qt_app, "APPS", [("mask", "Mask")], raising=False)
        route = _Recorder()
        sc._nav_by_index(route, 0)
        assert route.asked == ["_on_nav_selected"]


def test_the_ai_key_toggles_only_the_visible_screens_switch(shown_window,
                                                            screen_factory,
                                                            caplog, subtests):
    """Ctrl+/ flips the first visible AppScreen's switch and back; hidden
    screens, screens without a switch and windows without screens are left
    alone; a window that cannot be searched is logged with its traceback."""
    from spacr.qt.screens.app_screen import AppScreen

    with subtests.test(msg="no app screens at all"):
        sc._toggle_ai(shown_window)
        assert shown_window.findChildren(AppScreen) == []

    hidden_switch, visible_switch = _Switch(), _Switch()
    hidden = screen_factory(shown_window, hidden_switch)
    hidden.hide()
    visible = screen_factory(shown_window, visible_switch)
    visible.show()

    with subtests.test(msg="flips the visible switch and back"):
        sc._toggle_ai(shown_window)
        sc._toggle_ai(shown_window)
        assert visible_switch.history == [True, False]
        assert hidden_switch.history == []

    with subtests.test(msg="a hidden-only window flips nothing"):
        visible.hide()
        sc._toggle_ai(shown_window)
        assert (visible_switch.history, hidden_switch.history) == ([True, False], [])
        hidden.show()
        sc._toggle_ai(shown_window)
        assert hidden_switch.history == [True]
        hidden.hide()

    with subtests.test(msg="a visible screen without a switch"):
        bare = screen_factory(shown_window)
        bare.show()
        sc._toggle_ai(shown_window)
        assert not hasattr(bare, "_ai_switch")
        assert visible_switch.history == [True, False]
        late = _Switch(checked=True)
        bare._ai_switch = late
        sc._toggle_ai(shown_window)
        assert late.history == [False]

    with subtests.test(msg="a window that cannot be searched is logged"):
        searched = []

        class _Hostile:
            def findChildren(self, kind):
                searched.append(kind)
                raise RuntimeError("window is gone")

        with caplog.at_level(logging.DEBUG, logger=_LOGGER):
            sc._toggle_ai(_Hostile())
        assert searched == [AppScreen]
        record = _record(caplog, "could not toggle the AI switch")
        assert record.exc_info[0] is RuntimeError


def test_ctrl_f_focuses_the_settings_search_or_leaves_the_screen_alone(
        qapp, caplog, subtests):
    """The caret goes to the search box with its text selected and the
    settings column revealed; a missing stack or strip stops at that
    question, and a refusing box or reveal is logged."""
    def _screen(entry, reveal=None):
        bar = type("Bar", (), {})()
        bar._input = entry
        screen = type("Screen", (), {})()
        screen._settings_search = bar
        if reveal is not None:
            screen.reveal_settings = reveal
        return screen

    with subtests.test(msg="selects the search text and reveals the column"):
        entry = QLineEdit()
        entry.setText("cell diameter")
        revealed = []
        sc._focus_settings_search(_StackWindow(
            _screen(entry, lambda: revealed.append("reveal"))))
        assert entry.selectedText() == "cell diameter"
        assert revealed == ["reveal"]

    with subtests.test(msg="no search strip"):
        screen = _Recorder()
        sc._focus_settings_search(_StackWindow(screen))
        assert screen.asked == ["_settings_search"]

    with subtests.test(msg="no stack"):
        target = _Recorder()
        sc._focus_settings_search(target)
        assert target.asked == ["_stack"]

    with subtests.test(msg="a reveal that raises still focuses"):
        entry = QLineEdit()
        entry.setText("plot")

        def _explode():
            raise RuntimeError("no column")

        caplog.clear()
        with caplog.at_level(logging.DEBUG, logger=_LOGGER):
            sc._focus_settings_search(_StackWindow(_screen(entry, _explode)))
        assert entry.selectedText() == "plot"
        assert _record(caplog, "could not open the settings column").exc_info[0] \
            is RuntimeError

    with subtests.test(msg="a box that refuses focus"):
        tried = []

        class _Refusing:
            def setFocus(self):
                tried.append("setFocus")
                raise RuntimeError("already deleted")

        caplog.clear()
        with caplog.at_level(logging.DEBUG, logger=_LOGGER):
            sc._focus_settings_search(_StackWindow(_screen(_Refusing())))
        assert tried == ["setFocus"]
        assert _record(caplog, "could not focus the settings search box") \
            .exc_info[0] is RuntimeError


def test_the_cheat_sheet_overlay_lists_every_key_and_fits_the_window(
        window, qapp, subtests):
    """The overlay names every mapped key, says where scoped keys work, never
    outgrows the window, wraps categories when narrow, stays centred and
    leaves the window around the card untouched."""
    with subtests.test(msg="every key, scope and the closing hint"):
        overlay = sc.show_cheat_sheet(window)
        assert type(overlay) is sc.ShortcutOverlay
        assert window._spacr_shortcut_overlay is overlay
        printed = {label.text() for label in overlay.findChildren(QLabel)}
        for spec in sc.mapped():
            assert sc.native(spec.keys) in printed, spec.keys
        scoped = [spec for spec in sc.mapped() if spec.scope != sc.EVERYWHERE]
        assert scoped
        for spec in scoped:
            assert f"{spec.label}  —  {spec.scope}" in printed
        assert {"Keyboard shortcuts", "Press any key to close."} <= printed
        overlay.dismiss()

    def _header_columns(overlay):
        grid = overlay._card_content.layout()
        return sorted({grid.getItemPosition(grid.indexOf(label))[1]
                       for label in overlay.findChildren(QLabel)
                       if label.objectName() == "ShortcutOverlayCategory"})

    for width, height, pairs in ((700, 600, 1), (1280, 800, 2), (1900, 900, 4)):
        with subtests.test(msg=f"{width}x{height}"):
            window.resize(width, height)
            overlay = sc.show_cheat_sheet(window)
            card = overlay._card.geometry()
            assert overlay.size() == QSize(width, height)
            assert card.width() <= width - 24 and card.height() <= height - 24
            assert abs((width - card.width()) // 2 - card.x()) <= 0
            assert _header_columns(overlay) == list(range(0, 2 * pairs, 2))
            overlay.dismiss()

    with subtests.test(msg="the card stays centred on a resize"):
        window.resize(900, 600)
        overlay = sc.show_cheat_sheet(window)
        qapp.processEvents()
        overlay.setGeometry(0, 0, 1400, 1600)
        overlay.resizeEvent(None)
        card = overlay._card.geometry()
        assert abs(card.center().x() - 700) <= 1
        assert abs(card.center().y() - 800) <= 1
        overlay.dismiss()

    with subtests.test(msg="the window around the card is left as it was"):
        qapp.processEvents()
        x, y = 2, window.height() - 3
        before = window.grab().toImage().pixel(x, y)
        overlay = sc.show_cheat_sheet(window)
        qapp.processEvents()
        assert not overlay._card.geometry().contains(x, y)
        assert window.grab().toImage().pixel(x, y) == before
        overlay.dismiss()


def test_the_cheat_sheet_closes_replaces_and_falls_back(window, qapp,
                                                        monkeypatch, subtests):
    """Any key or click closes it, a second sheet replaces the first (even
    a stale one), a dead window still lets it go, and a parentless or
    zero-sized caller gets the modal dialog instead."""
    from shiboken6 import isValid

    for case, close in (("a key", lambda o: o.keyPressEvent(None)),
                        ("a click", lambda o: o.mousePressEvent(None))):
        with subtests.test(msg=f"{case} closes it"):
            overlay = sc.show_cheat_sheet(window)
            assert overlay.isHidden() is False
            close(overlay)
            qapp.processEvents()
            assert overlay.isHidden() is True

    with subtests.test(msg="it follows the window's size"):
        overlay = sc.show_cheat_sheet(window)
        window.resize(1100, 700)
        assert overlay.eventFilter(window, QEvent(QEvent.Resize)) is False
        assert overlay.geometry() == window.rect()
        overlay.dismiss()

    with subtests.test(msg="re-showing replaces the overlay"):
        first = sc.show_cheat_sheet(window)
        second = sc.show_cheat_sheet(window)
        qapp.processEvents()
        assert second is not first
        assert window._spacr_shortcut_overlay is second
        assert first.isHidden() is True and second.isHidden() is False
        second.dismiss()

    with subtests.test(msg="a stale overlay is replaced anyway"):
        class _Stale:
            def dismiss(self):
                raise RuntimeError("wrapped C/C++ object has been deleted")

        window._spacr_shortcut_overlay = _Stale()
        overlay = sc.show_cheat_sheet(window)
        assert type(overlay) is sc.ShortcutOverlay
        assert window._spacr_shortcut_overlay is overlay
        overlay.dismiss()

    with subtests.test(msg="a window already gone dismisses quietly"):
        overlay = sc.show_cheat_sheet(window)

        class _Gone:
            def removeEventFilter(self, _filter):
                raise RuntimeError("wrapped C/C++ object has been deleted")

        overlay._window = _Gone()
        overlay.dismiss()
        assert overlay.isHidden() is True
        qapp.sendPostedEvents(None, QEvent.DeferredDelete)
        assert not isValid(overlay)

    from spacr.qt.widgets.workflow_diagram import DiagramDialog

    opened = []

    def _exec(dialog):
        opened.append((dialog.parent(), dialog.windowTitle()))
        return 0

    monkeypatch.setattr(DiagramDialog, "exec", _exec)
    monkeypatch.setattr(QDialog, "exec", _exec)
    tiny = QWidget()
    tiny.resize(QSize(0, 0))
    for case, parent in (("no parent", None), ("a zero-sized window", tiny)):
        with subtests.test(msg=case):
            opened.clear()
            assert sc.show_cheat_sheet(parent) is None
            assert opened == [(parent, "spaCR — Keyboard shortcuts")]
            assert getattr(parent, "_spacr_shortcut_overlay", None) is None
    tiny.deleteLater()


def test_the_overlay_filter_handles_only_its_own_events(shown_window,
                                                        subtests):
    """A press on the card's viewport dismisses and is swallowed; a release
    there, a press elsewhere, or a resize of the window pass on."""
    overlay = sc.ShortcutOverlay(shown_window)
    overlay.show()
    viewport = overlay._scroll.viewport()
    stranger = QWidget(shown_window)

    with subtests.test(msg="a release on the viewport passes on"):
        assert overlay.eventFilter(viewport, QEvent(QEvent.MouseButtonRelease)) is False
        assert overlay.isVisible() is True

    with subtests.test(msg="a press elsewhere passes on"):
        assert overlay.eventFilter(stranger, QEvent(QEvent.MouseButtonPress)) is False
        assert overlay.isVisible() is True

    with subtests.test(msg="a window resize is followed, not consumed"):
        shown_window.resize(1100, 720)
        assert overlay.eventFilter(shown_window, QEvent(QEvent.Resize)) is False
        assert overlay.geometry() == shown_window.rect()
        card = overlay._card.geometry()
        assert card.width() <= overlay.width() and card.height() <= overlay.height()
        assert abs((overlay.width() - card.width()) // 2 - card.x()) <= 1

    with subtests.test(msg="a press on the viewport dismisses and is swallowed"):
        assert overlay.eventFilter(viewport, QEvent(QEvent.MouseButtonPress)) is True
        assert overlay.isVisible() is False
        assert [w for w in shown_window.findChildren(QWidget)
                if w.objectName() == sc.OVERLAY_NAME and w.isVisible()] == []
