"""The paint diagnostic, and the other module-level helpers of ``spacr.qt.app``,
answer something useful when the part they read cannot be read.

The paint diagnostic (item 408) is a tool for a session that is already
misbehaving, so it has to keep going when a piece of that session will not
answer: every part that fails is NAMED in the report's ``errors`` and the
rest of the report is still returned. These tests break one part at a time
(the grab, the platform name, the version lookup, a preference, the screen,
the sheets, the window list, a child widget, the suspect picker, the folder)
and read the report the user would open.

The same file pins the small readers around it: the fold-host reader that
reads a module's SOURCE rather than importing it, the tile order for an
unknown section, the carried preview state when a replacement cannot take a
value, the pip-less escape command on Windows, and the font scale that
falls back to 1.0.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
import types
from pathlib import PurePosixPath

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtGui import QColor, QImage, QPixmap  # noqa: E402
from PySide6.QtWidgets import QLabel, QStackedWidget, QWidget  # noqa: E402

from spacr.qt import app as app_mod  # noqa: E402
from spacr.qt import hidpi as hidpi_mod  # noqa: E402
from spacr.qt import preferences as prefs_mod  # noqa: E402
from spacr.qt.theme import TRANSPARENT_PROPERTY  # noqa: E402

pytestmark = pytest.mark.qt


# ---------------------------------------------------------------------------
# Stand-ins: a bare window, a display, a screen whose parts will not answer
# ---------------------------------------------------------------------------

class _BareWindow(QWidget):
    """A top-level widget with just enough of a MainWindow for the dump."""

    def __init__(self, backdrop=None):
        super().__init__()
        self._backdrop = backdrop
        self.resize(120, 80)

    def window_backdrop(self):
        return self._backdrop


class _Display:
    """A display whose grab is whatever pixmap the test hands it."""

    def __init__(self, pixmap):
        self._pixmap = pixmap

    def grabWindow(self, _wid):
        return self._pixmap

    def name(self):
        return "probe-display"


class _Liar(QWidget):
    """A child whose visibility cannot be read."""

    def isVisible(self):
        raise RuntimeError("this widget is half torn down")


class _DeadConsole:
    """A console that went away while the dump was being written."""

    def isVisible(self):
        raise RuntimeError("the console is gone")


class _BrokenScreen(QWidget):
    """A current screen whose page fill cannot be read."""

    app_key = "probe"

    def __init__(self):
        super().__init__()
        self._console = _DeadConsole()
        self.label = QLabel("still here", self)
        self.liar = _Liar(self)

    def page_fill(self):
        raise RuntimeError("the page fill is not ready")


def _boom(*_args, **_kwargs):
    raise RuntimeError("this part cannot be read")


def _use_display(monkeypatch, pixmap):
    monkeypatch.setattr(hidpi_mod, "screen_for_widget",
                        lambda _widget=None: _Display(pixmap))


# ---------------------------------------------------------------------------
# The dump
# ---------------------------------------------------------------------------

def test_an_empty_grab_and_an_unreadable_session_are_named_not_fatal(
        qapp, qtbot, tmp_path, monkeypatch):
    """Every broken part lands in ``errors``; the JSON is still written."""
    window = _BareWindow()
    qtbot.addWidget(window)
    window.show()
    _use_display(monkeypatch, QPixmap())
    stub_app = types.SimpleNamespace(styleSheet=_boom)
    monkeypatch.setattr(app_mod, "QApplication", types.SimpleNamespace(
        platformName=_boom, instance=lambda: stub_app,
        topLevelWidgets=_boom))
    monkeypatch.setattr(app_mod, "_git_state", _boom)
    monkeypatch.setattr(prefs_mod, "get_theme", _boom)
    monkeypatch.setattr(app_mod, "_paint_suspects", _boom)

    report = app_mod._dump_paint_diagnostics(window, _out_dir=tmp_path)

    errors = "\n".join(report["errors"])
    for part in ("grab the window from the display",
                 "read the Qt platform",
                 "read the spaCR version and checkout",
                 "read the theme preference",
                 "fingerprint the style sheets",
                 "list the top-level windows",
                 "pick the suspects"):
        assert part in errors
    assert "the grab came back empty" in errors
    assert report["files"]["png"] is None
    assert report["window_backdrop"] == {"present": False}
    assert report["screen"] == {"present": False}
    assert report["widgets"] == []
    assert "theme" not in report["preferences"]
    assert "pane_opacity" in report["preferences"]
    written = json.loads(
        (tmp_path / os.path.basename(report["files"]["json"])).read_text(
            encoding="utf-8"))
    assert written["errors"] == report["errors"]


def test_a_folder_that_cannot_be_made_costs_the_files_not_the_report(
        qapp, qtbot, tmp_path, monkeypatch):
    """A path under a FILE: no folder, no PNG, no JSON -- each one named,
    and the screen's readable widgets are still recorded."""
    blocker = tmp_path / "not_a_folder"
    blocker.write_text("x", encoding="utf-8")
    window = _BareWindow()
    qtbot.addWidget(window)
    window._stack = QStackedWidget(window)
    screen = _BrokenScreen()
    window._stack.addWidget(screen)
    window._stack.resize(120, 80)
    window.show()
    pixmap = QPixmap(12, 8)
    pixmap.fill(QColor("black"))
    _use_display(monkeypatch, pixmap)

    report = app_mod._dump_paint_diagnostics(window,
                                             _out_dir=blocker / "dumps")

    errors = "\n".join(report["errors"])
    assert f"create {blocker / 'dumps'}" in errors
    assert "QPixmap.save returned False" in errors
    assert "read the current screen" in errors
    assert "read the paint state of a _Liar" in errors
    assert "write " in errors
    assert report["files"] == {"json": None, "png": None}
    assert report["screenshot"]["display"] == "probe-display"
    assert report["screen"]["app_key"] == "probe"
    assert report["screen"]["page_fill"] is None
    classes = [record["class"] for record in report["widgets"]]
    assert "QLabel" in classes and "_Liar" not in classes


def test_a_folder_that_is_not_a_path_stops_early_and_says_so(
        qapp, qtbot, monkeypatch):
    window = _BareWindow()
    qtbot.addWidget(window)

    report = app_mod._dump_paint_diagnostics(window, _out_dir=12345)

    assert len(report["errors"]) == 1
    assert report["errors"][0].startswith("stopped early: TypeError")
    assert report["files"] == {"json": None, "png": None}
    assert list(report)[0] == "suspects"


def test_a_window_that_cannot_take_the_key_gets_no_key(monkeypatch):
    monkeypatch.setenv("SPACR_PAINT_DIAG", "1")

    assert app_mod._install_the_paint_diagnostic(object()) is None


# ---------------------------------------------------------------------------
# The small readers behind the dump
# ---------------------------------------------------------------------------

def test_a_value_json_cannot_write_is_written_as_its_text():
    assert app_mod._plain(3) == 3
    assert app_mod._plain(None) is None
    assert app_mod._plain(True) is True
    assert app_mod._plain(PurePosixPath("/data/plate1")) == "/data/plate1"
    assert app_mod._plain(Qt.GlobalColor.red) == str(Qt.GlobalColor.red)


def test_a_rectangle_is_measured_from_the_window_whatever_its_relation(
        qapp, qtbot):
    window = QWidget()
    other = QWidget()
    qtbot.addWidget(window)
    qtbot.addWidget(other)
    window.setGeometry(10, 20, 100, 60)
    other.setGeometry(40, 70, 30, 20)
    window.show()
    other.show()

    assert app_mod._rect_in_window(window, window) == [0, 0, 100, 60]
    x, y, w, h = app_mod._rect_in_window(other, window)
    assert (w, h) == (30, 20)
    delta = other.mapToGlobal(other.rect().topLeft()) - window.mapToGlobal(
        window.rect().topLeft())
    assert (x, y) == (delta.x(), delta.y())


def test_a_path_whose_stop_is_not_an_ancestor_runs_to_the_top(qapp, qtbot):
    top = QWidget()
    top.setObjectName("Top")
    qtbot.addWidget(top)
    child = QLabel("x", top)
    stranger = QWidget()
    qtbot.addWidget(stranger)

    assert app_mod._widget_path(child, stranger) == "QWidget#Top > QLabel"


def test_a_widget_with_no_sheet_above_it_names_no_ancestor(qapp, qtbot):
    top = QWidget()
    qtbot.addWidget(top)
    child = QLabel("x", top)

    assert app_mod._nearest_sheet_ancestor(child) is None
    assert app_mod._the_transparent_rule_is_on_the_ancestry(child) is (
        TRANSPARENT_PROPERTY in qapp.styleSheet())


def test_an_empty_image_has_no_pixels_and_no_black_share(qapp):
    assert app_mod._pixels_of(QImage()) is None
    assert app_mod._near_black_fraction(None, [0, 0, 5, 5], (1.0, 1.0)) is None


def test_a_rectangle_off_the_capture_has_no_black_share(qapp):
    image = QImage(4, 4, QImage.Format.Format_RGBA8888)
    image.fill(QColor("black"))
    pixels = app_mod._pixels_of(image)

    assert app_mod._near_black_fraction(pixels, [0, 0, 4, 4], (1.0, 1.0)) == 1.0
    assert app_mod._near_black_fraction(pixels, [10, 10, 4, 4],
                                        (1.0, 1.0)) is None


def test_a_tagged_widget_with_no_rule_above_it_is_a_suspect():
    record = {"class": "QLabel", "objectName": "Tagged", "geometry": [0, 0, 1, 1],
              TRANSPARENT_PROPERTY: True, "transparent_rule_on_ancestry": False}

    suspects = app_mod._paint_suspects([record], rule_in_play=True)
    assert [s["objectName"] for s in suspects] == ["Tagged"]
    assert "no sheet on its ancestry" in suspects[0]["reasons"][0]
    assert app_mod._paint_suspects([record], rule_in_play=False) == []


def test_a_folder_that_is_not_a_checkout_has_no_git_state(tmp_path):
    assert app_mod._git_state(str(tmp_path)) == (None, None)


# ---------------------------------------------------------------------------
# Fold hosts, read from source
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_hosts(tmp_path, monkeypatch):
    """A package of host modules on ``sys.path`` for the source reader."""
    pkg = tmp_path / "probe_fold_hosts"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "sibling.py").write_text(textwrap.dedent('''
        OTHER = "not the key"
        KEY: str = "sibling_key"
        '''), encoding="utf-8")
    (pkg / "host.py").write_text(textwrap.dedent('''
        from . import sibling
        from . import absent
        import os
        count: int
        os.environ_note = "an attribute target"
        APP_KEY = "probe_host"
        FOLD_ORDER = (absent.KEY, "b")
        FOLDED_APPS = (sibling.KEY, os.sep, "plain")
        '''), encoding="utf-8")
    (pkg / "lonely.py").write_text(textwrap.dedent('''
        HOST_KEY = "lonely_host"
        FOLDED_APPS = "not a tuple"
        '''), encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    for name in list(sys.modules):
        if name.startswith("probe_fold_hosts"):
            monkeypatch.delitem(sys.modules, name)
    return "probe_fold_hosts"


def test_a_constant_is_read_from_an_annotated_assignment(fake_hosts):
    assert app_mod._declared_constant(f"{fake_hosts}.sibling",
                                      "KEY") == "sibling_key"
    assert app_mod._declared_constant(f"{fake_hosts}.sibling",
                                      "MISSING") is None
    assert app_mod._declared_constant("no_such_module_anywhere_xyz",
                                      "KEY") is None


def test_a_fold_list_that_names_what_cannot_be_resolved_is_not_used(
        fake_hosts):
    host, folded = app_mod._declared_folds(f"{fake_hosts}.host")

    assert host == "probe_host"
    assert folded == ()


def test_a_host_with_no_fold_tuple_declares_its_key_alone(fake_hosts):
    assert app_mod._declared_folds(f"{fake_hosts}.lonely") == (
        "lonely_host", ())


def test_folds_skip_hosts_that_are_missing_or_fold_nothing(
        fake_hosts, monkeypatch):
    monkeypatch.setitem(sys.modules, "spacr.qt.widgets.fold_strip", None)
    monkeypatch.setattr(app_mod, "_EXTRA_FOLD_HOSTS", (
        "no_such_module_anywhere_xyz", f"{fake_hosts}.lonely",
        f"{fake_hosts}.host"))

    assert app_mod.folded_children() == {}


def test_a_row_in_an_unknown_section_sorts_after_every_known_one():
    known = app_mod.tile_sort_key(("mask", "Mask", "", app_mod.SECTION_CORE))
    unknown = app_mod.tile_sort_key(("x", "X", "", "No Such Section"))

    assert unknown == (len(app_mod.SECTION_ORDER), 0)
    assert unknown > known


# ---------------------------------------------------------------------------
# Carrying a preview, the pip-less command, the font scale
# ---------------------------------------------------------------------------

class _Stubborn:
    """A replacement preview that will not take a loaded image."""

    _image = None

    def __setattr__(self, name, value):
        if name == "_image":
            raise AttributeError("read only")
        object.__setattr__(self, name, value)


def test_a_preview_carries_what_the_replacement_will_take():
    loaded = types.SimpleNamespace(_image="pixels", _image_path="/a.tif",
                                   _path_full=None, _settings={"c": 1})
    old = types.SimpleNamespace(_live_preview=loaded, _preview_panel=loaded)
    target = _Stubborn()
    fresh = types.SimpleNamespace(_live_preview=target, _preview_panel=None)

    app_mod._carry_preview_state(old, fresh)

    assert target._image is None
    assert target._image_path == "/a.tif"
    assert target._settings == {"c": 1}


def test_on_windows_the_escape_is_a_cmd_line(monkeypatch):
    import spacr.updater as updater

    monkeypatch.setattr(updater, "find_uv",
                        lambda: r"C:\Program Files\spaCR\uv.exe")
    monkeypatch.setattr(app_mod, "os", types.SimpleNamespace(name="nt"))

    said = app_mod._the_missing_pip_escape("error: No module named pip")

    assert said == subprocess.list2cmdline([
        r"C:\Program Files\spaCR\uv.exe", "pip", "install", "--upgrade",
        "--python", sys.executable, "spacr"])
    assert said.startswith('"C:\\Program Files')


def test_an_unreadable_font_scale_reads_as_one(monkeypatch):
    monkeypatch.setattr(prefs_mod, "get_font_scale", _boom)

    assert app_mod._current_font_scale() == 1.0
