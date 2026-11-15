"""The parts of ``spacr.qt.app`` that stand on their own beside the window.

The dock column (:class:`Sidebar`), the menu-bar drag filter, the app icons,
the pipeline preloader, the bundled fonts, the dialog filters, the app
registry and the native crash dump.  Each test walks a table of cases for one
unit and states the exact value every case must produce; the failure arms
(a broken collaborator) sit beside the working arm they are measured against.
"""
from __future__ import annotations

import importlib
import logging
import os
import sys
import threading
import types

import pytest
from PySide6.QtCore import QEvent, QObject, QPoint, QPointF, Qt
from PySide6.QtGui import QIcon, QMouseEvent
from PySide6.QtWidgets import QPushButton, QWidget

from spacr.crashreport import _reports_dir as _REAL_REPORTS_DIR
from spacr.qt import app as app_mod
from spacr.qt import iconset
from spacr.qt.app import (
    APPS,
    SECTIONS,
    Sidebar,
    _DragsTheWindowByTheMenuBar,
    _ICON_OVERRIDES,
    _PipelinePreloader,
    _icon_for_app,
    _load_bundled_fonts,
    register_app,
    registered_entry,
    unregister_app,
)

pytestmark = pytest.mark.qt

PROBE_KEY = "parts_probe_app"


def _img(icon: QIcon, px: int = 24):
    """Rasterise an icon so two QIcons can be compared by pixels."""
    return icon.pixmap(px, px).toImage()


@pytest.fixture
def sidebar(qapp, qt_theme_applied):
    """A real dock column, deleted afterwards."""
    made = Sidebar()
    yield made
    made.deleteLater()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def test_sidebar_rows_announce_their_key_read_in_full_and_follow_maturity(
        qtbot, qapp, qt_theme_applied, monkeypatch, subtests):
    """Rows emit their own key, no shipping label is elided, hidden is not removed."""
    bar = Sidebar()
    qtbot.addWidget(bar)
    by_key = {b.property("navKey"): b for b in bar.findChildren(QPushButton)}

    with subtests.test(msg="a click emits exactly the row's key"):
        assert "invasion" not in by_key, "Invasion is reached through Toxoplasma"
        for key in ("__home__", "mask", "toxoplasma"):
            with qtbot.waitSignal(bar.nav_selected, timeout=1000) as blocker:
                by_key[key].click()
            assert blocker.args == [key]

    with subtests.test(msg="no shipping label is clipped at full height"):
        bar.resize(bar.width(), 1400)
        bar.show()
        qapp.processEvents()
        assert [b.full_text().strip() for b in bar.clipped_items()] == []

    with subtests.test(msg="rows follow the maturity preference"):
        import spacr.qt.preferences as preferences

        keyed = [b for b in bar._items if b.property("navKey")
                 and b.property("navKey") != "__home__"]
        assert keyed
        monkeypatch.setattr(preferences, "maturity_is_visible", lambda _s: False)
        bar.refresh_visibility()
        assert [b.isHidden() for b in keyed] == [True] * len(keyed)
        assert set(keyed) <= set(bar._items), "hidden, not removed"
        monkeypatch.setattr(preferences, "maturity_is_visible", lambda _s: True)
        bar.refresh_visibility()
        assert any(not b.isHidden() for b in keyed)


def test_sidebar_icons_are_re_inked_and_section_headers_fold(
        sidebar, qtbot, qapp, monkeypatch, subtests):
    """Theme switches re-ink every row; a missing icon set leaves text rows;
    a section header folds, lights on hover and answers a left release."""
    from spacr.qt import preferences as prefs

    with subtests.test(msg="a theme switch re-inks every row"):
        monkeypatch.setattr(prefs, "resolve_effective_theme", lambda: "dark")
        bar = Sidebar()
        qtbot.addWidget(bar)
        rows = {b.property("navKey"): b for b in bar.findChildren(QPushButton)}
        dark = {k: _img(b.icon()) for k, b in rows.items()}
        assert [k for k, b in rows.items() if b.icon().isNull()] == []
        monkeypatch.setattr(prefs, "resolve_effective_theme", lambda: "light")
        bar.refresh_icons()
        assert [k for k, b in rows.items() if _img(b.icon()) == dark[k]] == []

    with subtests.test(msg="refresh_icons reaches every keyed row"):
        sidebar.refresh_icons()
        keyed = [b for b in sidebar._items if b.property("navKey")]
        assert keyed
        assert [b.property("navKey") for b in keyed if b.icon().isNull()] == []

    with subtests.test(msg="no icon set: every app still listed, text only"):
        real = app_mod._icon_for_app
        with monkeypatch.context() as m:
            m.setattr(app_mod, "_icon_for_app", lambda key: None)
            bar = Sidebar()
            qtbot.addWidget(bar)
            rows = [b for b in bar._items
                    if b.property("navKey") not in (None, "", "__home__")]
            assert len(rows) > 5
            assert all(b.icon().isNull() for b in rows)
            bar.refresh_icons()
            assert all(b.icon().isNull() for b in rows)
        assert app_mod._icon_for_app is real
        bar.refresh_icons()
        assert any(not b.icon().isNull() for b in rows)

    section = SECTIONS[0]
    with subtests.test(msg="toggle_section folds and unfolds"):
        assert sidebar.section_is_open(section) is True
        assert sidebar.toggle_section(section) is False
        assert sidebar.section_is_open(section) is False
        assert sidebar.toggle_section(section) is True
        assert sidebar.section_is_open(section) is True

    headers = [child for child in sidebar.findChildren(QWidget)
               if child.property("sectionName")]
    with subtests.test(msg="hover lights a header"):
        assert headers, "the sidebar drew no section headers"
        header = headers[0]
        sidebar.eventFilter(header, QEvent(QEvent.Type.Enter))
        assert header.property("hovered") is True
        sidebar.eventFilter(header, QEvent(QEvent.Type.Leave))
        assert header.property("hovered") is False

    with subtests.test(msg="a left release on a header toggles it"):
        header = headers[0]
        name = str(header.property("sectionName"))
        before = sidebar.section_is_open(name)
        event = QMouseEvent(QEvent.Type.MouseButtonRelease, QPoint(2, 2),
                            Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton,
                            Qt.KeyboardModifier.NoModifier)
        assert sidebar.eventFilter(header, event) is True
        assert sidebar.section_is_open(name) is (not before)
        sidebar.toggle_section(name)
        assert sidebar.section_is_open(name) is before

    with subtests.test(msg="an event on a non-header is passed on"):
        plain = QWidget(sidebar)
        assert sidebar.eventFilter(plain, QEvent(QEvent.Type.Enter)) is False


# ---------------------------------------------------------------------------
# _DragsTheWindowByTheMenuBar
# ---------------------------------------------------------------------------

class _Handle:
    def __init__(self, refuse=False):
        self.moves = 0
        self.refuse = refuse

    def startSystemMove(self):        # noqa: N802 - Qt naming
        self.moves += 1
        if self.refuse:
            raise RuntimeError("the compositor declined")


class _Window(QObject):
    """A QObject, because the filter parents itself to the window."""

    def __init__(self, handle=None, dead=False, borderless=True):
        super().__init__()
        self._handle = handle
        self._dead = dead
        self._borderless = borderless

    def windowHandle(self):           # noqa: N802 - Qt naming
        if self._dead:
            raise RuntimeError("Internal C++ object already deleted.")
        return self._handle


class _Bar:
    """A menu bar stand-in: it knows which points carry an action."""

    def __init__(self, action_at=None, gone=False):
        self._action_at = action_at
        self._gone = gone

    def actionAt(self, _point):       # noqa: N802 - Qt naming
        if self._gone:
            raise RuntimeError("Internal C++ object already deleted.")
        return self._action_at


def _press(button=Qt.MouseButton.LeftButton, kind=QEvent.Type.MouseButtonPress):
    return QMouseEvent(kind, QPointF(40.0, 8.0), button, button,
                       Qt.KeyboardModifier.NoModifier)


@pytest.mark.parametrize("case,window,bar,event,consumed,moves", [
    ("bare strip drags", dict(), dict(), "press", True, 1),
    ("press on a menu opens it", dict(), dict(action_at=object()), "press", False, 0),
    ("right click", dict(), dict(), "right", False, 0),
    ("release passes", dict(), dict(), "release", False, 0),
    ("move passes", dict(), dict(), "move", False, 0),
    ("paint passes", dict(), dict(), "paint", False, 0),
    ("framed window", dict(borderless=False), dict(), "press", False, 0),
    ("no native handle yet", dict(handle=None), dict(), "press", False, None),
    ("bar cannot answer actionAt", dict(), dict(gone=True), "press", False, 0),
    ("compositor refuses", dict(refuse=True), dict(), "press", False, 1),
    ("window torn down", dict(dead=True), dict(), "press", False, 0),
])
def test_the_menu_bar_drag_filter_drags_only_on_a_bare_left_press(
        qapp, case, window, bar, event, consumed, moves):
    """Only a left press on the bare strip of a borderless, live window drags."""
    window = dict(window)
    refuse = window.pop("refuse", False)
    handle = window.pop("handle", _Handle(refuse=refuse))
    drag = _DragsTheWindowByTheMenuBar(_Window(handle, **window))
    events = {
        "press": lambda: _press(),
        "right": lambda: _press(button=Qt.MouseButton.RightButton),
        "release": lambda: _press(kind=QEvent.Type.MouseButtonRelease),
        "move": lambda: _press(kind=QEvent.Type.MouseMove),
        "paint": lambda: QEvent(QEvent.Type.Paint),
    }
    assert drag.eventFilter(_Bar(**bar), events[event]()) is consumed
    if handle is not None:
        assert handle.moves == moves


# ---------------------------------------------------------------------------
# _icon_for_app
# ---------------------------------------------------------------------------

def test_icon_for_app_resolves_overrides_glyphs_and_plugin_icons(
        qapp, tmp_path, monkeypatch, subtests):
    """Every app draws an icon; overrides, glyphs and plugin icons win in order."""
    import spacr.plugins as plugins

    with subtests.test(msg="every registered app has a non-null icon"):
        assert [k for k, *_r in APPS if _icon_for_app(k).isNull()] == []

    with subtests.test(msg="the override makes train_cellpose wear Cellpose"):
        assert _ICON_OVERRIDES == {"train_cellpose": "cellpose_masks.png"}
        assert _img(_icon_for_app("train_cellpose")) == \
            _img(_icon_for_app("cellpose_masks"))
        assert _img(_icon_for_app("mask")) != _img(_icon_for_app("measure"))

    with subtests.test(msg="a forced key gets the themed glyph"):
        with monkeypatch.context() as m:
            m.setattr(app_mod, "_FORCE_GLYPH", {PROBE_KEY})
            assert _img(app_mod._icon_for_app(PROBE_KEY)) == \
                _img(iconset.icon(PROBE_KEY))

    icon_path = tmp_path / "plugin.png"
    from PIL import Image
    Image.new("RGBA", (16, 16), (200, 30, 30, 255)).save(icon_path)
    cases = [
        ("plugin PNG file used as is", lambda key: types.SimpleNamespace(
            icon=str(icon_path)), lambda: QIcon(str(icon_path))),
        ("plugin glyph name is themed", lambda key: types.SimpleNamespace(
            icon="flask"), lambda: iconset.icon("flask")),
        ("plugin without an icon falls back", lambda key: types.SimpleNamespace(
            icon=""), lambda: iconset.app_icon(APPS[0][0])),
        ("unreadable plugin registry falls back",
         lambda key: (_ for _ in ()).throw(RuntimeError("unreadable")),
         lambda: iconset.app_icon(APPS[0][0])),
    ]
    for name, get_app, expected in cases:
        with subtests.test(msg=name), monkeypatch.context() as m:
            m.setattr(plugins, "get_app", get_app)
            key = APPS[0][0] if "falls back" in name else PROBE_KEY
            icon = app_mod._icon_for_app(key)
            assert not icon.isNull()
            assert _img(icon) == _img(expected())


# ---------------------------------------------------------------------------
# _PipelinePreloader
# ---------------------------------------------------------------------------

class _HeldThread:
    """Record ``start`` without running the target concurrently."""

    def __init__(self, target=None, name=None, daemon=None):
        self.target = target
        self.name = name
        self.daemon = daemon
        self.started = False

    def start(self):
        self.started = True


def test_the_preloader_imports_off_thread_reports_each_step_and_finishes_once(
        qapp, qtbot, monkeypatch, subtests):
    """Imports run on the worker, progress arrives on the poll, done fires once."""
    with subtests.test(msg="held worker: the poll delivers the callbacks"):
        imported, steps, done = [], [], []
        real_import = importlib.import_module

        def _fake_import(name):
            imported.append(name)
            if name == "spacr.no_such_module":
                raise ImportError("nope")
            return real_import(name)

        with monkeypatch.context() as m:
            m.setattr(importlib, "import_module", _fake_import)
            m.setattr(app_mod, "threading", types.SimpleNamespace(
                Thread=_HeldThread, Event=threading.Event))
            pre = _PipelinePreloader(on_step=lambda i, n: steps.append((i, n)),
                                     on_done=lambda: done.append(1))
            m.setattr(pre, "_MODULES", ("spacr.no_such_module", "json"))
            assert pre.total() == 2
            pre.start()
            worker = pre._thread
            assert (worker.started, worker.name, worker.daemon) == (
                True, "spacr-preload", True)
            assert imported == []
            pre.start()
            assert pre._thread is worker
            worker.target()
            assert imported == ["spacr.no_such_module", "json"]
            assert (steps, done) == ([], [])
            pre._drain()
            assert steps == [(1, 2), (2, 2)]
            assert done == [1]
            pre._drain()
            assert done == [1]
            assert pre._i == 2

    def explode(*_a):
        raise RuntimeError("the loading screen has closed")

    cases = [
        # (name, modules, on_step: "record" / "raise" / None (no callback),
        #  on_done raises, expected steps)
        ("every step reported, done once", ("json", "os.path",
                                            "no.such.module.at.all"),
         "record", False, [(1, 3), (2, 3), (3, 3)]),
        ("an unimportable module does not stop it",
         ("no.such.module.at.all", "json"), "record", False, [(1, 2), (2, 2)]),
        ("no progress callback: every step drained, done once",
         ("no.such.module.at.all", "json", "os.path"), None, False, []),
        ("a raising progress callback is survived", ("json", "os.path"),
         "raise", False, [(1, 2), (2, 2)]),
        ("a raising completion callback is swallowed, not skipped", ("json",),
         "record", True, [(1, 1)]),
    ]
    for name, modules, step_mode, done_raises, expected in cases:
        with subtests.test(msg=name), monkeypatch.context() as m:
            m.setattr(_PipelinePreloader, "_MODULES", modules)
            steps, done = [], []

            def on_step(i, n, _steps=steps, _raise=step_mode == "raise"):
                _steps.append((i, n))
                if _raise:
                    explode()

            def on_done(_done=done, _raise=done_raises):
                _done.append(1)
                if _raise:
                    explode()

            pre = _PipelinePreloader(
                on_step=on_step if step_mode else None, on_done=on_done)
            pre.start()
            pre.start()
            qtbot.waitUntil(lambda: bool(done), timeout=10000)
            qapp.processEvents()
            assert done == [1]
            assert steps == expected
            assert (pre._i, pre._reported) == (len(modules), len(modules))


# ---------------------------------------------------------------------------
# _load_bundled_fonts
# ---------------------------------------------------------------------------

def test_the_bundled_fonts_are_registered_from_the_font_folders_only(
        qapp, monkeypatch, subtests):
    """Open Sans and the glyph fonts are registered; non-fonts and a missing
    folder register nothing."""
    import PySide6.QtGui as qtgui
    from PySide6.QtGui import QFontDatabase

    with subtests.test(msg="Open Sans reaches QFontDatabase, idempotently"):
        assert _load_bundled_fonts() is None
        families = QFontDatabase.families()
        assert any("Open Sans" in f for f in families)
        _load_bundled_fonts()
        assert QFontDatabase.families() == families

    fonts_dir = os.path.join("resources", "font", "open_sans", "static")
    real_isdir = os.path.isdir

    with subtests.test(msg="the real folders hand over only TTF/OTF files"):
        added: list = []
        with monkeypatch.context() as m:
            m.setattr(qtgui, "QFontDatabase", types.SimpleNamespace(
                addApplicationFont=lambda p: added.append(p) or len(added)))
            _load_bundled_fonts()
        assert added
        assert [p for p in added
                if not p.lower().endswith((".ttf", ".otf"))] == []
        assert {os.path.basename(os.path.dirname(p)) for p in added} == {
            "static", "glyphs"}
        assert any("OpenSans" in os.path.basename(p) for p in added)

    with subtests.test(msg="a listing with non-fonts hands over the fonts only"):
        handed: list = []
        listed = ["OFL.txt", "README.md", "OpenSans-Regular.ttf", "OpenSans.otf"]
        with monkeypatch.context() as m:
            m.setattr(qtgui, "QFontDatabase",
                      types.SimpleNamespace(addApplicationFont=handed.append))
            m.setattr(app_mod, "os", types.SimpleNamespace(
                path=os.path, listdir=lambda d: list(listed)))
            app_mod._load_bundled_fonts()
        fonts = ["OpenSans-Regular.ttf", "OpenSans.otf"]
        assert [os.path.basename(p) for p in handed] == fonts + fonts
        assert [os.path.basename(os.path.dirname(p)) for p in handed] == [
            "static", "static", "glyphs", "glyphs"]

    for name, hidden in [("no Open Sans folder", lambda p: str(p).endswith(fonts_dir)),
                         ("no folder at all", lambda p: True)]:
        with subtests.test(msg=name), monkeypatch.context() as m:
            loaded: list = []
            m.setattr(os.path, "isdir",
                      lambda p, _h=hidden: False if _h(p) else real_isdir(p))
            m.setattr(QFontDatabase, "addApplicationFont",
                      staticmethod(lambda path: loaded.append(path) or -1))
            assert app_mod._load_bundled_fonts() is None
            assert loaded == []


# ---------------------------------------------------------------------------
# install_the_dialog_filters
# ---------------------------------------------------------------------------

def test_dialog_filters_install_in_isolation_and_name_what_ran(
        qapp, monkeypatch, caplog, subtests):
    """Each installer gets the app; a missing or raising one is skipped alone."""
    ran: list = []
    module = types.ModuleType("spacr_fake_filters")
    module.one = lambda app: ran.append(("one", app))
    module.two = lambda app: ran.append(("two", app))

    def bad(_app):
        raise RuntimeError("this filter cannot install here")

    module.bad = bad
    monkeypatch.setitem(sys.modules, "spacr_fake_filters", module)

    cases = [
        ("every installer runs, in order",
         (("spacr_fake_filters", "one"), ("spacr_fake_filters", "two")),
         ("one", "two"), ["one", "two"], None),
        ("a raising installer does not stop the others",
         (("spacr_fake_filters", "one"), ("spacr_fake_filters", "bad"),
          ("spacr_fake_filters", "two")),
         ("one", "two"), ["one", "two"], "could not install bad"),
        ("a module that will not import", (
            ("spacr.this_module_does_not_exist", "whatever"),), (), [],
         "could not install whatever"),
        ("a missing function in a real module", (
            ("spacr.qt.dialogs", "no_such_installer"),), (), [],
         "could not install no_such_installer"),
        ("an empty registry", (), (), [], None),
    ]
    for name, registry, installed, names, logged in cases:
        with subtests.test(msg=name):
            ran.clear()
            caplog.clear()
            monkeypatch.setattr(app_mod, "_DIALOG_FILTERS", registry)
            with caplog.at_level("DEBUG", logger=app_mod.LOG.name):
                assert app_mod.install_the_dialog_filters(qapp) == installed
            assert [n for n, _ in ran] == names
            assert all(a is qapp for _, a in ran)
            if logged:
                assert logged in caplog.text


# ---------------------------------------------------------------------------
# register_app / registered_entry / unregister_app
# ---------------------------------------------------------------------------

def test_the_app_registry_files_refuses_and_resolves_entries(
        qapp, monkeypatch, caplog, subtests):
    """A row lands in its section's block; duplicates and bad entries are refused."""
    section = SECTIONS[0]

    with subtests.test(msg="a row is filed inside its own section's block"):
        row = register_app(PROBE_KEY, "Parts Probe", "a test app", section)
        try:
            assert row[0] == PROBE_KEY
            sections = [r[3] for r in APPS]
            index = [r[0] for r in APPS].index(PROBE_KEY)
            assert sections[index] == section
            first = sections.index(section)
            last = len(sections) - 1 - sections[::-1].index(section)
            assert sections[first:last + 1] == [section] * (last - first + 1)
        finally:
            assert unregister_app(PROBE_KEY) is True
        assert unregister_app(PROBE_KEY) is False
        assert PROBE_KEY not in {r[0] for r in APPS}

    with subtests.test(msg="the same key twice is refused"):
        register_app(PROBE_KEY, "Parts Probe", "a test app", section)
        try:
            with pytest.raises(ValueError, match="already registered"):
                register_app(PROBE_KEY, "Other", "another", section)
        finally:
            unregister_app(PROBE_KEY)

    entries = [
        ("an entry is imported on demand", "json:dumps", None),
        ("a malformed entry names the spelling", "justamodule", "module:function"),
    ]
    for name, entry, refusal in entries:
        with subtests.test(msg=name):
            register_app(PROBE_KEY, "Parts Probe", "a test app", section,
                         entry=entry)
            try:
                if refusal is None:
                    import json
                    assert registered_entry(PROBE_KEY) is json.dumps
                else:
                    with pytest.raises(ValueError, match=refusal):
                        registered_entry(PROBE_KEY)
            finally:
                unregister_app(PROBE_KEY)

    with subtests.test(msg="an uncataloguable translation costs only the name"):
        import spacr.qt.i18n as i18n

        def explode(_source, _values):
            raise ValueError("that row has a blank entry")

        logger = logging.getLogger("spacr.qt.app")
        with monkeypatch.context() as m:
            m.setattr(i18n, "add_translation", explode)
            m.setattr(logger, "propagate", True)
            m.setattr(logger, "disabled", False)
            with caplog.at_level(logging.WARNING, logger="spacr.qt.app"):
                row = register_app(PROBE_KEY, "Parts Probe", "a test app",
                                   section, translations=["x"] * 9)
            try:
                assert row[0] == PROBE_KEY
                assert PROBE_KEY in {r[0] for r in APPS}
                assert "Could not translate app name" in caplog.text
            finally:
                unregister_app(PROBE_KEY)


# ---------------------------------------------------------------------------
# _install_crash_dump
# ---------------------------------------------------------------------------

@pytest.fixture
def native_capture(monkeypatch):
    from spacr import crashreport

    original = crashreport._NATIVE_CAPTURE_HANDLE
    monkeypatch.setattr(crashreport, "_NATIVE_CAPTURE_HANDLE", None)
    monkeypatch.setattr(crashreport, "_NATIVE_EXCEPTION_HOOK", object())
    yield crashreport
    handle = crashreport._NATIVE_CAPTURE_HANDLE
    if handle is not None and handle is not original:
        handle.close()


def _close_capture(native_capture, monkeypatch):
    handle = native_capture._NATIVE_CAPTURE_HANDLE
    if handle is not None:
        handle.close()
    monkeypatch.setattr(native_capture, "_NATIVE_CAPTURE_HANDLE", None)


def test_the_crash_dump_is_kept_open_beside_the_log_or_costs_only_itself(
        tmp_path, monkeypatch, native_capture, subtests):
    """The dump sits beside the log (or in ~/.spacr/logs), stays open, and a
    folder that cannot be made returns ``""``."""
    import faulthandler

    from spacr import logging_util

    with subtests.test(msg="beside the ordinary log, handed to faulthandler"):
        enabled: list = []
        with monkeypatch.context() as m:
            m.setattr(logging_util, "log_dir", lambda: str(tmp_path / "beside"))
            os.makedirs(tmp_path / "beside")
            m.setattr(faulthandler, "enable",
                      lambda file=None, all_threads=True: enabled.append(file))
            path = app_mod._install_crash_dump()
            assert path == str(tmp_path / "beside" / app_mod.CRASH_DUMP_NAME)
            with open(path) as stream:
                assert "=== spaCR session " in stream.read()
            assert enabled == [native_capture._NATIVE_CAPTURE_HANDLE]
            assert not native_capture._NATIVE_CAPTURE_HANDLE.closed
        _close_capture(native_capture, monkeypatch)

    for oversized in (False, True):
        with subtests.test(msg=f"no logging helper (oversized={oversized})"), \
                monkeypatch.context() as m:
            home = tmp_path / f"home{int(oversized)}"
            expected = home / ".spacr" / "logs" / app_mod.CRASH_DUMP_NAME
            expected.parent.mkdir(parents=True)
            prefix = ("x" * native_capture.MAX_LOG_BYTES + "\n"
                      if oversized else "")
            expected.write_text(prefix + "Fatal Python error: Aborted\n")

            def must_not_report(*args, **kwargs):
                raise AssertionError("must not authorize replay")

            m.setattr(native_capture, "_reports_dir", _REAL_REPORTS_DIR)
            m.setattr(native_capture, "_dispatch_error", must_not_report)
            m.setattr(native_capture, "_update_state", must_not_report)
            m.setitem(sys.modules, "spacr.logging_util", None)
            m.setattr(os.path, "expanduser", lambda _p: str(home))
            m.setenv("SPACR_HOME", os.path.join(str(home), ".spacr"))
            m.setattr(faulthandler, "enable",
                      lambda file=None, all_threads=True: None)
            path = app_mod._install_crash_dump()
            assert path == os.path.join(str(home), ".spacr", "logs",
                                        app_mod.CRASH_DUMP_NAME)
            text = expected.read_text()
            assert "Fatal Python error: Aborted" in text
            assert "=== spaCR session " in text
            if oversized:
                assert text.startswith(
                    "\n=== spaCR session history truncated ===\n")
                new_header = text[text.rfind("\n=== spaCR session "):]
                assert expected.stat().st_size == \
                    native_capture.MAX_LOG_BYTES + len(new_header.encode())
            assert not native_capture._NATIVE_CAPTURE_HANDLE.closed
        _close_capture(native_capture, monkeypatch)

    with subtests.test(msg="a log folder that cannot be made"), \
            monkeypatch.context() as m:
        m.setattr(logging_util, "log_dir", lambda: (_ for _ in ()).throw(
            RuntimeError("the log directory cannot be resolved")))
        m.setattr(os, "makedirs", lambda *_a, **_k: (_ for _ in ()).throw(
            OSError("read-only filesystem")))

        def must_not_create(*args, **kwargs):
            raise AssertionError("must not create a fallback folder")

        m.setattr(native_capture.Path, "mkdir", must_not_create)
        assert app_mod._install_crash_dump() == ""
        assert native_capture._NATIVE_CAPTURE_HANDLE is None
