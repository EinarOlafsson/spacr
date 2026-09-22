"""spaCR, and its setup screen, start dark unless the user chose otherwise.

Maintainer, 2026-09-21: "please always start spacr in dark mode by default,
that goes as well for the spacr startup ... make sure this works from linux,
windows and mac osx. users may change later on but first time and if the
user doesnt change anything, dark mode should be used for main spacr and
spacr start."

"spacr start" is the first-run setup screen (`SetupSlides`), which
`spacr.qt.app.launch` opens before the main window; it wears the theme that
`apply_preferences_to_app` puts on the application, so both follow the one
default.

WHAT CHANGED, AND WHAT IS PINNED HERE:

* The default is ``"dark"`` (it was ``"system"``, so a light Mac, Windows
  or GNOME desktop started spaCR light).
* A stored ``"system"`` counts only if somebody chose it. Before this change
  both Preferences and the setup screen wrote the value their Theme control
  showed, so most stored ``"system"`` values are the old default saved
  unchanged, and they now read as dark. Choosing "Follow system" from now on
  sets a flag that makes it stick. An explicit Light is kept.
* The operating system's own scheme cannot repaint a dark spaCR: the style
  is Fusion unless ``QT_STYLE_OVERRIDE`` says otherwise (macOS and Windows
  native styles draw some controls from the OS scheme), and on Qt 6.8+ the
  platform is told the scheme spaCR is drawing in
  (``QStyleHints.setColorScheme``), which is what makes the macOS title bar,
  native menus and dialogs, and the Windows title bar, match.

ONLY LINUX CAN RUN THIS. The platform branches are exercised with stand-in
application objects that report what macOS, Windows and GTK/KDE report --
the style's name and ``QStyleHints.colorScheme`` -- because spaCR's code
does not branch on ``sys.platform`` for any of it; ``sys.platform`` is
patched as well to prove that.
"""
from __future__ import annotations

import inspect
import os
import sys

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

pytestmark = pytest.mark.qt

from PySide6.QtCore import QSettings, Qt                        # noqa: E402
from PySide6.QtGui import QPalette                              # noqa: E402

from spacr.qt import preferences as prefs                       # noqa: E402
from spacr.qt import theme                                      # noqa: E402

PLATFORMS = (
    ("darwin", "macos"),
    ("win32", "windows11"),
    ("win32", "windowsvista"),
    ("linux", "breeze"),
    ("linux", "fusion"),
)


@pytest.fixture
def store(monkeypatch, tmp_path, qapp):
    """A fresh profile: nothing chosen yet."""
    settings = QSettings(str(tmp_path / "prefs.ini"), QSettings.IniFormat)
    monkeypatch.setattr(prefs, "_settings", lambda: settings)
    return settings


class _Hints:
    """``QStyleHints`` as a platform reports it."""

    def __init__(self, system):
        self.system = system
        self.requested = None
        self.calls = []

    def colorScheme(self):                               # noqa: N802
        return self.requested if self.requested is not None else self.system

    def setColorScheme(self, scheme):                    # noqa: N802
        self.calls.append(("set", scheme))
        self.requested = (None if scheme == Qt.ColorScheme.Unknown
                          else scheme)

    def unsetColorScheme(self):                          # noqa: N802
        self.calls.append(("unset",))
        self.requested = None


class _NoSetHints:
    """A Qt older than 6.8: the scheme can be read and not requested."""

    def __init__(self, system):
        self.system = system

    def colorScheme(self):                               # noqa: N802
        return self.system


class _Style:
    def __init__(self, name):
        self._name = name

    def name(self):
        return self._name


class _App:
    """Enough of a QApplication for the style and scheme decisions."""

    def __init__(self, style, system, hints=None):
        self._style = _Style(style)
        self._hints = hints or _Hints(system)
        self.set_style = []

    def style(self):
        return self._style

    def styleHints(self):                                # noqa: N802
        return self._hints

    def setStyle(self, style):                           # noqa: N802
        self.set_style.append(style.name().lower())
        self._style = _Style(style.name().lower())


def test_a_fresh_profile_is_dark(store):
    assert prefs.DEFAULT_THEME == "dark"
    assert prefs.get_theme() == "dark"
    assert prefs.get_theme_choice() == "dark"
    assert prefs.resolve_effective_theme() == "dark"


def test_the_setup_screen_opens_on_dark_and_saving_it_unchanged_keeps_dark(
        store):
    """The startup screen writes every answer it shows when it is closed,
    so an untouched Theme question must not turn into a choice."""
    from spacr.qt import setup_screen

    answers = setup_screen.current()
    assert answers["theme"] == "dark"
    assert setup_screen.apply(answers) == []
    assert prefs.get_theme() == "dark"


def test_a_stored_follow_system_that_nobody_chose_reads_as_dark(store):
    """The old default, saved unchanged by an older Preferences or setup
    screen. The maintainer's rule is that such a user gets dark."""
    store.setValue("prefs/theme", "system")
    assert prefs.get_theme() == "dark"
    assert prefs.get_theme_choice() == "dark"
    assert store.value("prefs/theme") == "system", (
        "the stored value is read differently, not rewritten")


def test_choosing_follow_system_sticks(store, monkeypatch):
    monkeypatch.setattr(theme, "system_colour_scheme", lambda app=None: "light")
    prefs.set_theme_choice("system")
    assert prefs.get_theme() == "system"
    assert prefs.resolve_effective_theme() == "light"


def test_choosing_something_else_forgets_follow_system(store):
    prefs.set_theme("system")
    prefs.set_theme("dark")
    store.setValue("prefs/theme", "system")
    assert prefs.get_theme() == "dark"


def test_an_explicit_light_is_kept(store):
    prefs.set_theme_choice("light")
    assert prefs.get_theme() == "light"
    assert prefs.resolve_effective_theme() == "light"


def test_follow_system_with_no_answer_from_the_platform_is_dark(
        store, monkeypatch):
    monkeypatch.setattr(theme, "system_colour_scheme", lambda app=None: None)
    prefs.set_theme("system")
    assert prefs.resolve_effective_theme() == "dark"


@pytest.mark.parametrize("platform,style", PLATFORMS)
@pytest.mark.parametrize("system", (Qt.ColorScheme.Light,
                                    Qt.ColorScheme.Dark,
                                    Qt.ColorScheme.Unknown))
def test_the_desktop_scheme_cannot_make_a_fresh_spacr_light(
        store, monkeypatch, platform, style, system):
    """A light Mac, a light Windows app mode, a light GTK/KDE desktop."""
    monkeypatch.setattr(sys, "platform", platform)
    app = _App(style, system)
    real = theme.system_colour_scheme
    monkeypatch.setattr(theme, "system_colour_scheme",
                        lambda _app=None: real(app))

    assert prefs.resolve_effective_theme() == "dark"
    assert theme.use_a_style_that_honours_the_palette(app, environ={}) \
        == "fusion"
    assert theme.hold_the_colour_scheme(app, "dark") is True
    assert app.styleHints().colorScheme() == Qt.ColorScheme.Dark


@pytest.mark.parametrize("name,scheme", (
    ("dark", Qt.ColorScheme.Dark),
    ("light", Qt.ColorScheme.Light),
    ("glass", Qt.ColorScheme.Dark),
    ("cell", Qt.ColorScheme.Dark),
    ("nocturne", Qt.ColorScheme.Dark),
))
def test_the_platform_is_told_the_scheme_the_theme_draws_in(name, scheme):
    app = _App("macos", Qt.ColorScheme.Light)
    assert theme.hold_the_colour_scheme(app, name) is True
    assert app.styleHints().colorScheme() == scheme
    assert theme.scheme_of(name) == ("light" if scheme == Qt.ColorScheme.Light
                                     else "dark")


def test_follow_system_releases_the_scheme_to_the_platform():
    app = _App("macos", Qt.ColorScheme.Light)
    theme.hold_the_colour_scheme(app, "dark")
    theme.hold_the_colour_scheme(app, None)
    assert app.styleHints().colorScheme() == Qt.ColorScheme.Light
    assert app.styleHints().calls[-1] == ("unset",)


@pytest.mark.parametrize("system,answer", (
    (Qt.ColorScheme.Light, "light"),
    (Qt.ColorScheme.Dark, "dark"),
    (Qt.ColorScheme.Unknown, None),
))
def test_the_system_scheme_is_the_platforms_not_spacrs_own(system, answer):
    """Asked after spaCR pinned dark, it must still answer what the desktop
    says, or "Follow system" would follow spaCR."""
    app = _App("macos", system)
    theme.hold_the_colour_scheme(app, "dark")
    assert theme.system_colour_scheme(app) == answer


def test_a_qt_older_than_6_8_is_a_quiet_no_op():
    """PySide6 >= 6.6 is supported; setColorScheme arrived in 6.8. The
    palette and stylesheet carry the colours there on their own."""
    app = _App("windows11", Qt.ColorScheme.Light,
               hints=_NoSetHints(Qt.ColorScheme.Light))
    assert theme.hold_the_colour_scheme(app, "dark") is False
    assert theme.system_colour_scheme(app) == "light"


def test_a_style_somebody_asked_for_is_left_alone():
    app = _App("macos", Qt.ColorScheme.Light)
    assert theme.use_a_style_that_honours_the_palette(
        app, environ={"QT_STYLE_OVERRIDE": "macos"}) == "macos"
    assert app.set_style == []


def test_launch_puts_fusion_on_before_the_theme_and_the_setup_screen():
    """The style has to be in place before the first palette and before the
    setup screen is built, or that screen is drawn in the native style."""
    from spacr.qt import app as qt_app

    source = inspect.getsource(qt_app.launch)
    style = source.index("use_a_style_that_honours_the_palette(app)")
    assert style < source.index("apply_preferences_to_app(app)")
    assert style < source.index("open_setup_if_needed")


def test_the_real_application_gets_the_dark_palette_and_a_dim_disabled_ink(
        qapp):
    try:
        theme.apply_qpalette(qapp, "dark")
        palette = qapp.palette()
        dark = theme.palette_for("dark")
        assert palette.color(QPalette.ColorRole.Window).name() == dark["bg"]
        assert palette.color(QPalette.ColorRole.Text).name() == dark["fg"]
        assert palette.color(QPalette.ColorGroup.Disabled,
                             QPalette.ColorRole.Text).name() == dark["fg_dim"]
        assert palette.color(QPalette.ColorRole.PlaceholderText).name() \
            == dark["fg_dim"]
        hints = qapp.styleHints()
        if hasattr(hints, "setColorScheme"):
            assert hints.colorScheme() in (Qt.ColorScheme.Dark,
                                           Qt.ColorScheme.Unknown), (
                "the offscreen platform may ignore the request; it must "
                "never answer Light for a dark theme")
    finally:
        theme.apply_qpalette(qapp, "dark")
