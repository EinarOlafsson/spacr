"""CPU coverage for the module head of spacr.gui_elements (lines 1-460).

Covers:
  * ``_register_open_sans`` on every platform branch (Linux / Windows /
    Darwin / unknown) including each failure path — font copy errors,
    ``fc-cache`` / ``xrdb`` failures, an unreadable ``~/.Xresources`` and the
    catch-all outer handler.
  * the module-level registration guard (the ``except`` that logs a warning
    when ``_register_open_sans`` blows up at import time), exercised by
    re-executing the module source with ``logging.getLogger`` rigged to fail.
  * ``restart_gui_app`` — both the relaunch path and the error print.
  * ``set_element_size``'s memoized fast path.
  * the ``set_dark_style`` branches that are skipped by the cached/default
    call: the ``teal`` alias, the non-OpenSans (no font loader) path and the
    per-widget restyling of Label/Button/ScrolledText/OptionMenu.

Everything is offline and writes only inside ``tmp_path`` — HOME is
redirected and ``subprocess.run`` is stubbed so no font cache or X resource
database on the real machine is ever touched.
"""
from __future__ import annotations

import importlib.util
import logging
import os
import shutil
import sys
import types

import pytest

try:
    import spacr.gui_elements as ge
except Exception as e:
    pytest.skip(f"spacr.gui_elements unavailable in this env: {e}",
                allow_module_level=True)

FONT_LOGGER = "spacr.gui.fonts"
TTFS = ("OpenSans-Regular.ttf", "OpenSans-Bold.ttf", "OpenSans-Italic.ttf")
BUNDLED_FONT_DIR = os.path.join(os.path.dirname(ge.__file__),
                                "resources", "font", "open_sans", "static")


# ---------------------------------------------------------------------------
# helpers / fixtures
# ---------------------------------------------------------------------------

class _CollectingHandler(logging.Handler):
    """Captures records straight off the target logger (propagation-proof)."""

    def __init__(self):
        super().__init__(level=logging.DEBUG)
        self.records = []

    def emit(self, record):
        self.records.append(record)

    @property
    def messages(self):
        return [r.getMessage() for r in self.records]

    def has(self, needle, level=None):
        return any(needle in r.getMessage()
                   and (level is None or r.levelno == level)
                   for r in self.records)


@pytest.fixture
def font_log():
    """Attach a collecting handler to the ``spacr.gui.fonts`` logger."""
    lg = logging.getLogger(FONT_LOGGER)
    handler = _CollectingHandler()
    old_level, old_disabled, old_prop = lg.level, lg.disabled, lg.propagate
    lg.addHandler(handler)
    lg.setLevel(logging.DEBUG)
    lg.disabled = False
    prev_disable = logging.root.manager.disable
    logging.disable(logging.NOTSET)
    try:
        yield handler
    finally:
        lg.removeHandler(handler)
        lg.setLevel(old_level)
        lg.disabled = old_disabled
        lg.propagate = old_prop
        logging.disable(prev_disable)


@pytest.fixture(autouse=True)
def closed_figures():
    """Never let an Agg figure leak out of this module."""
    yield
    try:
        import matplotlib.pyplot as plt
        plt.close("all")
    except Exception:
        pass


@pytest.fixture
def fake_home(tmp_path, monkeypatch):
    """Redirect ``~`` so font installation happens inside tmp_path."""
    monkeypatch.setenv("HOME", str(tmp_path))
    assert os.path.expanduser("~") == str(tmp_path)
    return tmp_path


@pytest.fixture
def recorded_subprocess(monkeypatch):
    """Stub ``subprocess.run`` and record the argv of every call."""
    calls = []

    def _run(cmd, *args, **kwargs):
        calls.append(list(cmd))
        return types.SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

    monkeypatch.setattr(ge.subprocess, "run", _run)
    return calls


def _boom(*args, **kwargs):
    raise OSError("injected failure")


class RecordingStyle:
    """Stand-in for ``ttk.Style`` that records every configure/map call.

    ``set_dark_style`` only ever calls ``theme_use``/``configure``/``map`` on
    the style object, so this records exactly what the palette applied
    without depending on the live ttk theme engine.
    """

    def __init__(self):
        self.theme = None
        self.configured = {}
        self.mapped = {}

    def theme_use(self, name):
        self.theme = name

    def configure(self, name, **kwargs):
        self.configured.setdefault(name, {}).update(kwargs)

    def map(self, name, **kwargs):
        self.mapped.setdefault(name, {}).update(kwargs)


# ---------------------------------------------------------------------------
# _register_open_sans — Linux branch
# ---------------------------------------------------------------------------

def test_register_open_sans_linux_installs_fonts_and_xresources(
        fake_home, monkeypatch, recorded_subprocess, font_log):
    """Happy Linux path: fonts copied, fc-cache run, Xft settings appended."""
    monkeypatch.setattr(ge.platform, "system", lambda: "Linux")

    ge._register_open_sans()

    fonts_dir = fake_home / ".local" / "share" / "fonts"
    assert sorted(p.name for p in fonts_dir.iterdir()) == sorted(TTFS)
    for name in TTFS:
        assert (fonts_dir / name).stat().st_size == \
            os.path.getsize(os.path.join(BUNDLED_FONT_DIR, name))

    xres = fake_home / ".Xresources"
    body = xres.read_text()
    assert "Xft.antialias: 1" in body
    assert "Xft.hintstyle: hintslight" in body

    assert [c[0] for c in recorded_subprocess] == ["fc-cache", "xrdb"]
    assert recorded_subprocess[0][-1] == str(fonts_dir)
    assert recorded_subprocess[1] == ["xrdb", "-merge", str(xres)]

    assert font_log.has("Installed 3 OpenSans font(s)")
    assert font_log.has("Font cache updated successfully")
    assert font_log.has("Xft anti-aliasing configured successfully")
    # everything went to DEBUG, nothing leaked at WARNING or above
    assert all(r.levelno == logging.DEBUG for r in font_log.records)


def test_register_open_sans_linux_copy_and_fc_cache_failures(
        fake_home, monkeypatch, font_log):
    """copy2 + subprocess failures are swallowed and logged, not raised."""
    monkeypatch.setattr(ge.platform, "system", lambda: "Linux")
    monkeypatch.setattr(shutil, "copy2", _boom)
    monkeypatch.setattr(ge.subprocess, "run", _boom)

    ge._register_open_sans()  # must not raise

    fonts_dir = fake_home / ".local" / "share" / "fonts"
    assert list(fonts_dir.iterdir()) == []          # nothing was installed
    for name in TTFS:
        assert font_log.has(f"Could not copy {name}: injected failure")
    assert not font_log.has("Installed")            # copied stayed 0
    assert font_log.has("Could not update font cache: injected failure")
    # the file itself is still written; only the `xrdb -merge` call fails
    assert "Xft.antialias: 1" in (fake_home / ".Xresources").read_text()
    assert font_log.has("Could not configure Xft anti-aliasing: injected failure")


def test_register_open_sans_linux_skips_existing_fonts_and_xresources(
        fake_home, monkeypatch, recorded_subprocess, font_log):
    """Already-installed fonts and an already-configured .Xresources are
    left completely untouched."""
    monkeypatch.setattr(ge.platform, "system", lambda: "Linux")
    fonts_dir = fake_home / ".local" / "share" / "fonts"
    fonts_dir.mkdir(parents=True)
    for name in TTFS:
        (fonts_dir / name).write_text("stale placeholder")
    xres = fake_home / ".Xresources"
    xres.write_text("Xft.antialias: 1\n")

    ge._register_open_sans()

    for name in TTFS:
        assert (fonts_dir / name).read_text() == "stale placeholder"
    assert xres.read_text() == "Xft.antialias: 1\n"     # not appended to
    assert [c[0] for c in recorded_subprocess] == ["fc-cache"]  # no xrdb
    assert not font_log.has("Installed")
    assert font_log.has("Xft anti-aliasing already configured")


def test_register_open_sans_linux_unreadable_xresources(
        fake_home, monkeypatch, recorded_subprocess, font_log):
    """A ~/.Xresources that cannot be read (here: it is a directory) is
    reported and the append is attempted anyway, then also reported."""
    monkeypatch.setattr(ge.platform, "system", lambda: "Linux")
    (fake_home / ".Xresources").mkdir()

    ge._register_open_sans()  # must not raise

    assert (fake_home / ".Xresources").is_dir()
    assert font_log.has("Could not read")
    assert font_log.has("Could not configure Xft anti-aliasing")
    # fc-cache still ran; xrdb never did because the write blew up first
    assert [c[0] for c in recorded_subprocess] == ["fc-cache"]


# ---------------------------------------------------------------------------
# _register_open_sans — Windows branch
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("rc, expected", [(1, 2), (0, 0)])
def test_register_open_sans_windows_gdi(monkeypatch, font_log, rc, expected):
    """The Windows branch feeds both bundled TTFs to AddFontResourceW and
    counts only the successful (non-zero) registrations."""
    import ctypes
    seen = []

    def _add_font(path):
        seen.append(path)
        return rc

    monkeypatch.setattr(ge.platform, "system", lambda: "Windows")
    monkeypatch.setattr(ctypes, "windll",
                        types.SimpleNamespace(gdi32=types.SimpleNamespace(
                            AddFontResourceW=_add_font)),
                        raising=False)

    ge._register_open_sans()

    assert [os.path.basename(p) for p in seen] == \
        ["OpenSans-Regular.ttf", "OpenSans-Bold.ttf"]
    assert all(os.path.isfile(p) for p in seen)
    assert font_log.has(f"Loaded {expected} OpenSans font(s) into Windows GDI")


def test_register_open_sans_windows_without_windll(monkeypatch, font_log):
    """No ctypes.windll (i.e. not really Windows) -> warning, no crash."""
    import ctypes
    monkeypatch.setattr(ge.platform, "system", lambda: "Windows")
    monkeypatch.delattr(ctypes, "windll", raising=False)

    ge._register_open_sans()

    assert font_log.has("Could not register fonts on Windows")
    assert not font_log.has("Loaded")


# ---------------------------------------------------------------------------
# _register_open_sans — Darwin / unknown / catch-all
# ---------------------------------------------------------------------------

def test_register_open_sans_darwin_installs_fonts(fake_home, monkeypatch, font_log):
    monkeypatch.setattr(ge.platform, "system", lambda: "Darwin")

    ge._register_open_sans()

    fonts_dir = fake_home / "Library" / "Fonts"
    assert sorted(p.name for p in fonts_dir.iterdir()) == sorted(TTFS)
    assert font_log.has(f"Installed 3 OpenSans font(s) to {fonts_dir}")
    assert not font_log.has("already installed")


def test_register_open_sans_darwin_already_installed(fake_home, monkeypatch, font_log):
    monkeypatch.setattr(ge.platform, "system", lambda: "Darwin")
    fonts_dir = fake_home / "Library" / "Fonts"
    fonts_dir.mkdir(parents=True)
    for name in TTFS:
        (fonts_dir / name).write_text("already here")

    ge._register_open_sans()

    for name in TTFS:
        assert (fonts_dir / name).read_text() == "already here"
    assert font_log.has("OpenSans fonts already installed on macOS")


def test_register_open_sans_darwin_copy_failure(fake_home, monkeypatch, font_log):
    monkeypatch.setattr(ge.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(shutil, "copy2", _boom)

    ge._register_open_sans()  # must not raise

    fonts_dir = fake_home / "Library" / "Fonts"
    assert list(fonts_dir.iterdir()) == []
    for name in TTFS:
        assert font_log.has(f"Could not copy {name}: injected failure")
    # copied == 0 -> the "already installed" message is what gets logged
    assert font_log.has("OpenSans fonts already installed on macOS")


def test_register_open_sans_unknown_platform(monkeypatch, font_log):
    monkeypatch.setattr(ge.platform, "system", lambda: "Plan9")

    ge._register_open_sans()

    assert font_log.has("Font registration not implemented for Plan9")


def test_register_open_sans_outer_failure_is_swallowed(monkeypatch, font_log):
    """Anything unexpected inside the helper is caught by the outer guard."""
    def _explode():
        raise RuntimeError("no platform for you")

    monkeypatch.setattr(ge.platform, "system", _explode)

    ge._register_open_sans()  # must not raise

    assert font_log.has("Font registration failed: no platform for you")


# ---------------------------------------------------------------------------
# module-level registration guard
# ---------------------------------------------------------------------------

def test_module_level_font_registration_failure_is_logged(monkeypatch, font_log):
    """Re-execute gui_elements.py with the *second* ``getLogger`` call (the
    one inside ``_register_open_sans``) rigged to blow up, so the failure
    escapes the helper and hits the module-level ``except``."""
    real_get_logger = logging.getLogger
    state = {"n": 0}

    def _fake_get_logger(name=None):
        if name == FONT_LOGGER:
            state["n"] += 1
            if state["n"] >= 2:
                raise RuntimeError("logger exploded")
        return real_get_logger(name) if name is not None else real_get_logger()

    monkeypatch.setattr(logging, "getLogger", _fake_get_logger)

    spec = importlib.util.spec_from_file_location(
        "spacr._gui_elements_reexec_for_test", ge.__file__)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # the module still finished importing despite the font failure
    assert callable(module.set_dark_style)
    assert module.set_dark_style is not ge.set_dark_style
    assert module._cached_dark_style is None
    assert state["n"] == 2
    assert font_log.has("could not register OpenSans fonts", level=logging.WARNING)
    assert "spacr._gui_elements_reexec_for_test" not in sys.modules


# ---------------------------------------------------------------------------
# restart_gui_app
# ---------------------------------------------------------------------------

def test_restart_gui_app_destroys_root_and_relaunches(monkeypatch):
    launched = []
    created = []

    stub_gui = types.ModuleType("spacr.gui")
    stub_gui.gui_app = lambda: launched.append("gui_app")
    monkeypatch.setitem(sys.modules, "spacr.gui", stub_gui)

    class FakeTk:
        def __init__(self):
            created.append(self)

    monkeypatch.setattr(ge.tk, "Tk", FakeTk)

    class Root:
        destroyed = False

        def destroy(self):
            Root.destroyed = True

    ge.restart_gui_app(Root())

    assert Root.destroyed is True
    assert len(created) == 1          # a fresh Tk root was built
    assert launched == ["gui_app"]    # and the app was relaunched


def test_restart_gui_app_prints_error_on_failure(capsys):
    class Root:
        def destroy(self):
            raise RuntimeError("cannot destroy")

    ge.restart_gui_app(Root())  # must not raise

    out = capsys.readouterr().out
    assert "Error restarting GUI application: cannot destroy" in out


# ---------------------------------------------------------------------------
# set_element_size cache
# ---------------------------------------------------------------------------

def test_set_element_size_returns_cached_dict_without_touching_monitors(monkeypatch):
    sentinel = {'btn_size': 42, 'bar_size': 43, 'settings_width': 44,
                'panel_width': 45, 'panel_height': 46}
    monkeypatch.setattr(ge, "_cached_element_size", sentinel)

    def _no_monitors():
        raise AssertionError("get_monitors must not be called when cached")

    monkeypatch.setattr(ge, "get_monitors", _no_monitors)

    assert ge.set_element_size() is sentinel


# ---------------------------------------------------------------------------
# set_dark_style — non-cached branches
# ---------------------------------------------------------------------------

def test_set_dark_style_teal_alias_bypasses_cache(tk_root):
    """``active_color='teal'`` resolves to #008080, and a call carrying a
    side-effect argument must not poison the module cache."""
    before = ge._cached_dark_style
    style = RecordingStyle()

    out = ge.set_dark_style(style, widgets=[], active_color='teal')

    assert out['active_color'] == '#008080'
    assert out['bg_color'] == '#000000'
    assert out['fg_color'] == '#ffffff'
    assert out['inactive_color'] == '#2B2B2B'
    assert ge._cached_dark_style is before          # cache untouched
    assert style.theme == 'clam'
    assert style.configured['Spacr.Horizontal.TProgressbar']['background'] == '#008080'


def test_set_dark_style_without_font_loader(tk_root):
    """A non-OpenSans family skips spacrFont entirely and hands ttk a plain
    (family, size) font tuple."""
    style = RecordingStyle()

    out = ge.set_dark_style(style, widgets=[], font_family="Helvetica", font_size=11)

    assert out['font_loader'] is None
    assert out['font_family'] == "Helvetica"
    assert out['font_sizes'] == {'small': 10, 'body': 11, 'header': 13, 'title': 17}
    assert style.configured['TLabel']['font'] == ("Helvetica", 11)
    assert style.configured['TLabel']['background'] == '#000000'


def test_set_dark_style_with_font_loader_uses_font_object(tk_root):
    """The OpenSans path hands ttk a real Font object, not a tuple."""
    import tkinter.font as tkFont
    style = RecordingStyle()

    out = ge.set_dark_style(style, widgets=[], font_family="OpenSans", font_size=12)

    assert isinstance(out['font_loader'], ge.spacrFont)
    assert isinstance(style.configured['TLabel']['font'], tkFont.Font)


def test_set_dark_style_restyles_widgets_with_font_loader(tk_root):
    """Label / Button / ScrolledText / OptionMenu all get the palette, and
    the OptionMenu's dropdown menu is restyled too."""
    import tkinter as tk
    from tkinter import scrolledtext

    label = tk.Label(tk_root, text="lbl")
    button = tk.Button(tk_root, text="btn")
    text = scrolledtext.ScrolledText(tk_root, width=10, height=2)
    var = tk.StringVar(master=tk_root, value="a")
    option = tk.OptionMenu(tk_root, var, "a", "b")
    style = RecordingStyle()

    ge.set_dark_style(style, widgets=[label, button, text, option],
                      bg_color="#101010", fg_color="#eeeeee")

    for w in (label, button, option):
        assert w.cget("bg") == "#101010"
        assert w.cget("fg") == "#eeeeee"
    assert text.cget("bg") == "#101010"
    assert text.cget("fg") == "#eeeeee"
    assert text.cget("insertbackground") == "#eeeeee"
    menu = option["menu"]
    # tk hands menu colours back as Tcl border objects -> compare as strings
    assert str(menu.cget("bg")) == "#101010"
    assert str(menu.cget("fg")) == "#eeeeee"
    # font_loader branch -> a *named* Font object, never a "family size" spec
    assert "OpenSans" not in str(label.cget("font"))
    assert "OpenSans" not in str(menu.cget("font"))


def test_set_dark_style_restyles_widgets_without_font_loader(tk_root):
    """Same widgets, no font loader: the literal (family, size) tuple is
    applied to widgets, the ScrolledText and the OptionMenu's menu."""
    import tkinter as tk
    from tkinter import scrolledtext

    label = tk.Label(tk_root, text="lbl")
    button = tk.Button(tk_root, text="btn")
    text = scrolledtext.ScrolledText(tk_root, width=10, height=2)
    var = tk.StringVar(master=tk_root, value="a")
    option = tk.OptionMenu(tk_root, var, "a", "b")
    style = RecordingStyle()

    out = ge.set_dark_style(style, widgets=[label, button, text, option],
                            font_family="Helvetica", font_size=11,
                            bg_color="#202020", fg_color="#dddddd")

    assert out['font_loader'] is None
    assert str(label.cget("font")) == "Helvetica 11"
    assert str(button.cget("font")) == "Helvetica 11"
    assert str(option.cget("font")) == "Helvetica 11"
    assert str(option["menu"].cget("font")) == "Helvetica 11"
    assert str(option["menu"].cget("bg")) == "#202020"
    assert text.cget("bg") == "#202020"
    assert text.cget("insertbackground") == "#dddddd"


def test_set_dark_style_ignores_unstyleable_widgets(tk_root):
    """A widget that matches none of the isinstance branches passes through
    untouched (no exception, no restyle)."""
    import tkinter as tk

    scale = tk.Scale(tk_root, from_=0, to=1)
    original_bg = scale.cget("bg")
    style = RecordingStyle()

    ge.set_dark_style(style, widgets=[scale], bg_color="#303030")

    assert scale.cget("bg") == original_bg
