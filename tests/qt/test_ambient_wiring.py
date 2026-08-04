"""The ambient background, as installed on module screens.

This file tests :mod:`spacr.qt.screens.app_screen`'s side of the feature —
*which* screens get an animated backdrop, what happens when they cannot,
and how a preference change reaches a screen that has been open for an
hour. The animation itself (themes, palettes, frame cost) is
:mod:`spacr.qt.widgets.ambient`'s own test file's problem.

Two groups, deliberately:

``against a stand-in``
    Most tests here run against a controlled ``install_ambient`` so the
    *screen's* decisions can be observed exactly — including the ones
    only visible as an absence, like "the preference being off must mean
    the widget is never constructed", which is not the same claim as
    "the widget is not visible".

``against the real widget``
    :func:`_real_ambient` re-checks the load-bearing structural claims
    against the shipped widget, because a stand-in that lowers itself
    proves nothing about one that does not.
"""
from __future__ import annotations

import sys
import types

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QEvent, QPoint, Qt
from PySide6.QtWidgets import QLabel, QMainWindow, QStackedWidget, QWidget


#: A spread of module screens, not one. The previous round of Qt work
#: was caught by a smoke test that constructed every registered module,
#: and one screen proving the rule is not the same as the rule holding.
#: Sequencing is in the list on purpose — it is the exception.
SAMPLE_APPS = ("mask", "measure", "classify", "regression", "umap",
               "activation", "replication", "map_barcodes")

#: The ones that must end up with the ambient backdrop.
AMBIENT_APPS = tuple(key for key in SAMPLE_APPS if key != "map_barcodes")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _isolated_qsettings(monkeypatch, qt_theme_applied, tmp_path):
    """Route QSettings into a temp .ini so tests don't touch real prefs.

    Nothing here asserts on what survives the round trip — see
    :func:`ambient_prefs` — but building a screen and running a
    preferences save reads (and can write) a dozen settings, and none of
    that belongs in the developer's own configuration.
    """
    from PySide6.QtCore import QSettings

    from spacr.qt import preferences as prefs

    # `preferences._settings()` builds `QSettings(_ORG, _APP)` — the
    # (organization, application) constructor — and that one resolves to the
    # NATIVE location regardless of `setDefaultFormat` and `setPath`.
    # Measured, because the previous version of this fixture assumed otherwise
    # and was inert:
    #
    #     QSettings.setDefaultFormat(IniFormat)
    #     QSettings.setPath(IniFormat, UserScope, tmp)
    #     QSettings("spacr", "qt").fileName()
    #       -> /home/<user>/.config/spacr/qt.conf      # the REAL file
    #     QSettings().fileName()
    #       -> <tmp>/spacr-test/qt-...-test.ini        # redirected fine
    #
    # so the `QSettings("spacr", "qt").clear()` that followed did not clear a
    # temp store, it **erased the developer's real preferences**, once per
    # test. Redirecting the class is not enough; the only reliable isolation
    # is to replace the accessor every reader goes through.
    store = QSettings(str(tmp_path / "prefs.ini"), QSettings.IniFormat)
    monkeypatch.setattr(prefs, "_settings", lambda: store)

    # A belt-and-braces guard, so this can never silently regress into
    # destroying real configuration again: if the store ever resolves outside
    # the temp directory, fail the test rather than write to it.
    resolved = store.fileName()
    assert str(tmp_path) in resolved, (
        f"QSettings isolation failed: {resolved} is outside {tmp_path}. "
        "Refusing to run — this fixture would otherwise write to the "
        "developer's real preferences.")

    try:
        from spacr.qt.first_run import mark_tour_seen
        mark_tour_seen()
    except Exception:
        pass
    yield


@pytest.fixture(autouse=True)
def _no_wallpaper(monkeypatch):
    """Keep the flat-fill path unless a test asks for a picture.

    ``_theme_wallpaper`` reaches into the theme's image cache; every
    test here that is not *about* the wallpaper wants it out of the way.
    """
    from spacr.qt.screens import app_screen
    monkeypatch.setattr(app_screen, "_theme_wallpaper", lambda: None)


class StubAmbient(QWidget):
    """Stand-in for :class:`spacr.qt.widgets.ambient.AmbientWidget`.

    Implements the published API and the same structural promises the
    DNA rain makes (lowered, no focus, no mouse), so a test that finds
    this widget *not* at the bottom of the stack has found something the
    screen did after installing it — which is exactly the failure this
    stand-in exists to be able to see.
    """

    def __init__(self, parent=None, *, theme, palette, backdrop=None):
        super().__init__(parent)
        self.theme = theme
        self.palette_name = palette
        self.backdrop = backdrop
        self.themes_set = []
        self.palettes_set = []
        self.backgrounds_set = []
        self.animating = True
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.setFocusPolicy(Qt.NoFocus)

    def set_theme(self, name):
        self.theme = name
        self.themes_set.append(name)

    def set_palette(self, name):
        self.palette_name = name
        self.palettes_set.append(name)

    def set_background_color(self, color):
        self.backgrounds_set.append(str(color))

    def set_animating(self, on):
        self.animating = bool(on)


@pytest.fixture
def fake_ambient(monkeypatch):
    """Install a controlled ``spacr.qt.widgets.ambient`` and report on it.

    The returned module records every ``install_ambient`` call, so a
    test can distinguish "no ambient widget is visible" from "no ambient
    widget was ever built" — the difference the preference is *for*.
    """
    module = types.ModuleType("spacr.qt.widgets.ambient")
    module.AMBIENT_THEMES = ("blobs", "mesh")
    module.DEFAULT_THEME = "blobs"
    module.DEFAULT_PALETTE = "spacr"
    palettes = {"blobs": ("spacr", "ember"), "mesh": ("steel",)}
    module.palettes_for = lambda theme: palettes.get(theme, ())
    module.theme_label = lambda name: name.title()
    module.palette_label = lambda theme, palette: palette.title()
    module.AmbientWidget = StubAmbient
    module.calls = []

    def install_ambient(host, layout=None, *, theme, palette, backdrop=None):
        module.calls.append({"host": host, "layout": layout, "theme": theme,
                             "palette": palette, "backdrop": backdrop})
        widget = StubAmbient(host, theme=theme, palette=palette,
                             backdrop=backdrop)
        widget.setGeometry(host.rect())
        widget.lower()
        widget.show()
        return widget

    module.install_ambient = install_ambient
    monkeypatch.setitem(sys.modules, "spacr.qt.widgets.ambient", module)
    return module


@pytest.fixture
def ambient_prefs(monkeypatch):
    """Drive the three ambient preferences from a dict.

    The screen reads them through :mod:`spacr.qt.preferences` — lazily,
    inside the method — so patching the module's functions is the honest
    seam: every read still goes through the name the product calls, and
    a test changing its mind mid-way is exactly what the user does in
    the Preferences dialog.

    Deliberately *not* driven by writing real QSettings: whether a value
    survives a round trip through an ini file is
    :mod:`spacr.qt.preferences`'s own claim (and its own test file's),
    and borrowing it here only buys the screen tests a second way to
    fail. Mutate the returned dict to change what the screen sees.
    """
    from spacr.qt import preferences
    state = {"enabled": True, "theme": "blobs", "palette": "spacr"}
    monkeypatch.setattr(preferences, "get_ambient_enabled",
                        lambda: state["enabled"])
    monkeypatch.setattr(preferences, "get_ambient_theme",
                        lambda: state["theme"])
    monkeypatch.setattr(preferences, "get_ambient_palette",
                        lambda: state["palette"])
    return state


@pytest.fixture
def app_theme_restored(qt_theme_applied):
    """Undo anything a test does to the shared QApplication's look.

    ``apply_preferences_to_app`` re-palettes and re-stylesheets the whole
    application, and the QApplication is session-scoped: a test that
    left it in the light theme would take out every later test that
    checks a dark-theme pixel.
    """
    yield
    from spacr.qt.theme import apply_qpalette, stylesheet
    apply_qpalette(qt_theme_applied)
    qt_theme_applied.setStyleSheet(stylesheet())


def _screen(qtbot, app_key, adopted=False):
    """Build one module screen.

    :param adopted: True when the caller is about to reparent it into a
        window it registers instead — registering both would have qtbot
        close a screen its window has already deleted.
    """
    from spacr.qt.screens.app_screen import AppScreen
    screen = AppScreen(app_key)
    if not adopted:
        qtbot.addWidget(screen)
    return screen


def _in_a_window(qtbot, screen):
    """Put ``screen`` where it really lives: a child of MainWindow's stack.

    Not a detail, and not decoration. ``QApplication.setPalette`` sends
    ``ApplicationPaletteChange`` to top-level widgets **only** (Qt
    6.11); a child gets ``PaletteChange``. And once an application
    stylesheet is in force — which is always, in the shipped app — a
    palette change does not reach the child at all until the stylesheet
    is re-applied over it. A theme test run on a bare, parentless screen
    is therefore green for a handler that never fires in production,
    which is exactly what the DNA rain's was.

    Keep the returned window in a local: qtbot only weak-references it,
    and a collected window deletes the C++ half of the screen under the
    test's feet.
    """
    window = QMainWindow()
    stack = QStackedWidget()
    window.setCentralWidget(stack)
    stack.addWidget(screen)
    qtbot.addWidget(window)
    window.resize(900, 600)
    window.show()
    qtbot.waitExposed(window)
    return window


def _save_preferences(qapp):
    """Do what pressing Save in the Preferences dialog does."""
    from spacr.qt.preferences import apply_preferences_to_app
    apply_preferences_to_app(qapp)
    qapp.processEvents()


def _switch_theme(qapp, monkeypatch, theme):
    """Re-theme the running application the way a save does to widgets.

    The palette *and* the stylesheet, in that order, because it is the
    stylesheet re-apply that carries ``PaletteChange`` down to a child
    screen — ``setPalette`` alone does not reach one once an application
    stylesheet is in force, which is always.

    Deliberately narrower than :func:`_save_preferences`: that also runs
    ``apply_ambient_preferences``, which pushes the persisted animation
    settings onto every live ambient widget, and a test asking what *the
    screen* did to the widget cannot tell the two apart afterwards.
    """
    from spacr.qt import preferences
    from spacr.qt.theme import apply_qpalette, stylesheet
    monkeypatch.setattr(preferences, "resolve_effective_theme",
                        lambda: theme)
    apply_qpalette(qapp, theme=theme)
    qapp.setStyleSheet(stylesheet(theme=theme))
    qapp.processEvents()


# ---------------------------------------------------------------------------
# The rule
# ---------------------------------------------------------------------------

def test_the_rule_is_every_screen_without_an_animation_of_its_own():
    """One stated rule, not an incidental ``else`` on the rain's ``if``."""
    from spacr.qt.screens.app_screen import (DNA_RAIN_APPS,
                                             uses_ambient_background)
    from spacr.qt.app import APPS

    assert DNA_RAIN_APPS == {"map_barcodes"}
    for key, _name, _description, _section in APPS:
        assert uses_ambient_background(key) is (key not in DNA_RAIN_APPS)
    # And it is the membership that decides, not the spelling of one key:
    # anything that grows a themed animation of its own steps out by
    # joining that set.
    assert uses_ambient_background("map_barcodes") is False
    assert uses_ambient_background("a_module_that_does_not_exist_yet") is True


@pytest.mark.parametrize("app_key", AMBIENT_APPS)
def test_every_non_sequencing_screen_gets_the_ambient_backdrop(
        qtbot, qt_theme_applied, fake_ambient, app_key):
    screen = _screen(qtbot, app_key)
    assert isinstance(screen._ambient, StubAmbient), (
        f"{app_key}: no ambient backdrop was installed")
    assert screen._ambient.parent() is screen
    assert screen._dna_rain is None, f"{app_key}: got the sequencing rain"


def test_sequencing_keeps_its_own_animation_and_gains_nothing(
        qtbot, qt_theme_applied, fake_ambient):
    """map_barcodes still gets the DNA rain, and never a second one.

    A second animated background behind the rain would fight it — which
    is why this is one rule and not two independent installs.
    """
    screen = _screen(qtbot, "map_barcodes")
    assert screen._dna_rain is not None, "the DNA rain went missing"
    assert screen._ambient is None
    assert fake_ambient.calls == [], \
        "an ambient backdrop was constructed behind the DNA rain"


@pytest.mark.parametrize("app_key", SAMPLE_APPS)
def test_no_screen_ever_carries_both_backdrops(qtbot, qt_theme_applied,
                                               fake_ambient, app_key):
    screen = _screen(qtbot, app_key)
    assert not (screen._ambient is not None and screen._dna_rain is not None)
    assert (screen._ambient is not None) or (screen._dna_rain is not None), (
        f"{app_key}: ended up with no backdrop at all")


def test_sequencing_cannot_acquire_one_through_the_refresh_path(
        qtbot, qt_theme_applied, fake_ambient):
    """The live-refresh path honours the same rule the constructor does."""
    screen = _screen(qtbot, "map_barcodes")
    screen.refresh_ambient_background()
    screen.show()
    assert screen._ambient is None
    assert fake_ambient.calls == []
    assert screen._dna_rain is not None


# ---------------------------------------------------------------------------
# The contract the DNA rain established
# ---------------------------------------------------------------------------

def test_the_backdrop_is_behind_every_sibling_and_ignores_input(
        qtbot, qt_theme_applied, fake_ambient):
    screen = _screen(qtbot, "measure")
    screen.resize(900, 600)
    screen.show()
    qtbot.waitExposed(screen)
    ambient = screen._ambient

    siblings = [child for child in screen.children()
                if isinstance(child, QWidget)]
    assert siblings.index(ambient) == 0 or all(
        not ambient.isAncestorOf(other) for other in siblings)
    # Stacking order, not child order: ask Qt what is on top at a point
    # the backdrop covers. It must never be the answer.
    for point in (QPoint(5, 5), QPoint(screen.width() // 2, 40)):
        on_top = screen.childAt(point)
        assert on_top is not ambient, \
            "the ambient backdrop is in front of the screen content"
    assert ambient.testAttribute(Qt.WA_TransparentForMouseEvents)
    assert ambient.focusPolicy() == Qt.NoFocus


def test_a_successful_install_clears_the_page_surfaces(
        qtbot, qt_theme_applied, fake_ambient):
    """Otherwise the animation runs, costs its frames and reaches nobody.

    Every layout container is an opaque ``bg`` under the blanket QWidget
    rule; one of them is enough to bury the backdrop completely.
    """
    from spacr.qt.theme import TRANSPARENT_PROPERTY
    screen = _screen(qtbot, "measure")
    assert screen._ambient is not None
    pages = [screen._header, screen._body_splitter, screen._settings_scroll,
             screen._settings_scroll.viewport(), screen._settings_content,
             screen._runtime_wrap, screen._console_wrap]
    for page in pages:
        assert page.property(TRANSPARENT_PROPERTY) is True, \
            f"{page.objectName() or type(page).__name__} still paints over it"


def test_the_cards_in_front_of_it_stay_opaque(qtbot, qt_theme_applied,
                                              fake_ambient):
    """Pages go transparent; the surfaces the settings sit on do not."""
    from spacr.qt.theme import TRANSPARENT_PROPERTY
    screen = _screen(qtbot, "measure")
    assert screen._settings_sections
    for section in screen._settings_sections:
        assert not section.property(TRANSPARENT_PROPERTY)


# ---------------------------------------------------------------------------
# It must never stop a screen from opening
# ---------------------------------------------------------------------------

def test_a_raising_install_leaves_the_screen_usable(qtbot, qt_theme_applied,
                                                    fake_ambient):
    """A decorative background is never worth a module that will not open."""
    def boom(*_args, **_kwargs):
        raise RuntimeError("no GPU, no gradients, no nothing")
    fake_ambient.install_ambient = boom

    screen = _screen(qtbot, "measure")
    assert screen._ambient is None
    assert screen._settings_sections, "the settings form did not build"
    failures = [label.text() for label in screen.findChildren(QLabel)
                if label.text().startswith("Failed to build settings")]
    assert not failures
    screen.resize(900, 600)
    screen.show()
    qtbot.waitExposed(screen)
    assert not screen.grab().isNull(), "the screen does not render"
    # And it is not left half-dressed. This used to be asserted as "the
    # header is not transparent, because surfaces are only cleared once
    # something is painting back there" -- but `ModuleHeader.__init__`
    # calls `make_transparent(self)` unconditionally and always has, so
    # the assertion was false the day it was written and this test has
    # been failing since. The rule it was reaching for is now true by
    # construction and stated the other way round: there is always
    # something painting back there, because `paintEvent` fills the page
    # whenever the backdrop failed to install. That is exactly the case
    # under test here, so assert the fill instead of the header.
    from spacr.qt.theme import active_palette
    assert screen.page_fill() is not None, (
        "the ambient install raised, so the screen owns its own page fill; "
        "without it `bg` (#000000 in the dark theme) shows through")


def test_a_failed_install_leaves_no_orphan_widget(qtbot, qt_theme_applied,
                                                  fake_ambient):
    """A half-installed backdrop would keep its parent, and its timer."""
    built = []

    def half_install(host, layout=None, *, theme, palette, backdrop=None):
        widget = StubAmbient(host, theme=theme, palette=palette)
        built.append(widget)
        raise RuntimeError("fell over after parenting the widget")

    fake_ambient.install_ambient = half_install
    screen = _screen(qtbot, "measure")
    qt_theme_applied.processEvents()
    assert screen._ambient is None
    assert len(built) == 1
    assert built[0].parent() is None, "the orphan is still a child of the screen"


def test_a_failing_install_is_not_retried_on_every_palette_event(
        qtbot, qt_theme_applied, fake_ambient, ambient_prefs):
    """Re-applying a stylesheet raises ``PaletteChange`` on every screen.

    An install that just failed will fail the same way a millisecond
    later, so retrying it there turns one broken import into one broken
    import per repaint of the theme.
    """
    attempts = []

    def boom(*_args, **_kwargs):
        attempts.append(1)
        raise RuntimeError("no ambient module here")

    fake_ambient.install_ambient = boom
    screen = _screen(qtbot, "measure")
    assert len(attempts) == 1
    for _ in range(5):
        screen.changeEvent(QEvent(QEvent.PaletteChange))
    screen.show()
    qtbot.waitExposed(screen)
    assert len(attempts) == 1, f"retried {len(attempts)} times"

    # But a preference change is a new question, and gets a new answer.
    ambient_prefs["palette"] = "ember"
    screen.refresh_ambient_background()
    assert len(attempts) == 2


def test_a_missing_ambient_module_is_not_a_broken_screen(qtbot,
                                                         qt_theme_applied,
                                                         monkeypatch):
    """No stand-in at all: the import itself fails."""
    monkeypatch.setitem(sys.modules, "spacr.qt.widgets.ambient", None)
    screen = _screen(qtbot, "measure")
    assert screen._ambient is None
    assert screen._settings_sections
    screen.refresh_ambient_background()          # and the live path too
    assert screen._ambient is None


def test_broken_preferences_do_not_break_the_screen(qtbot, qt_theme_applied,
                                                    fake_ambient, monkeypatch):
    from spacr.qt import preferences

    def boom():
        raise RuntimeError("preferences are gone")
    monkeypatch.setattr(preferences, "get_ambient_enabled", boom)

    screen = _screen(qtbot, "measure")
    assert screen._ambient is None
    assert fake_ambient.calls == []
    screen.refresh_ambient_background()
    assert screen._ambient is None


def test_a_broken_theme_read_is_survivable(qtbot, qt_theme_applied,
                                           fake_ambient, monkeypatch):
    """The enabled flag reads; the theme name does not."""
    from spacr.qt import preferences
    screen = _screen(qtbot, "measure")
    assert screen._ambient is not None

    def boom():
        raise RuntimeError("unreadable")
    monkeypatch.setattr(preferences, "get_ambient_theme", boom)
    screen.refresh_ambient_background()
    assert screen._ambient is not None, "a bad read must not delete the widget"


# ---------------------------------------------------------------------------
# The preference
# ---------------------------------------------------------------------------

def test_the_preference_off_means_never_constructed(qtbot, qt_theme_applied,
                                                    fake_ambient,
                                                    ambient_prefs):
    """Off is off, not built-and-hidden.

    Constructing it and hiding it still pays for the construction, and
    the machine this toggle exists for is the one running Cellpose on
    the GPU behind a 40-plate pipeline.
    """
    ambient_prefs["enabled"] = False

    for app_key in ("measure", "mask", "regression"):
        screen = _screen(qtbot, app_key)
        assert screen._ambient is None, f"{app_key}: installed anyway"
    assert fake_ambient.calls == [], "the widget was built despite the toggle"


def test_the_persisted_theme_and_palette_reach_the_widget(
        qtbot, qt_theme_applied, fake_ambient, ambient_prefs):
    """Whatever Preferences says is what gets built."""
    ambient_prefs["theme"] = "blobs"
    ambient_prefs["palette"] = "ember"

    screen = _screen(qtbot, "measure")
    assert fake_ambient.calls[-1]["theme"] == "blobs"
    assert fake_ambient.calls[-1]["palette"] == "ember"
    assert screen._ambient.theme == "blobs"
    assert screen._ambient.palette_name == "ember"


def test_with_nothing_stored_a_screen_still_gets_a_backdrop(
        qtbot, qt_theme_applied, fake_ambient):
    """Out of the box — real preference reads, nothing persisted.

    The unpatched getters, over settings this file's fixture just
    cleared: the feature is on by default, and the screen installs
    exactly the theme and palette preferences reports rather than a
    default of its own.
    """
    from spacr.qt.preferences import get_ambient_palette, get_ambient_theme
    screen = _screen(qtbot, "measure")
    assert screen._ambient is not None, "the default must be on"
    assert fake_ambient.calls[-1]["theme"] == get_ambient_theme()
    assert fake_ambient.calls[-1]["palette"] == get_ambient_palette()


def test_the_image_themes_hand_their_wallpaper_to_the_backdrop(
        qtbot, qt_theme_applied, fake_ambient, monkeypatch, tmp_path):
    """Same reason the rain gets one: a flat fill hides the photograph."""
    from spacr.qt.screens import app_screen
    wallpaper = tmp_path / "wall.png"
    wallpaper.write_bytes(b"not really a png, and never decoded here")
    monkeypatch.setattr(app_screen, "_theme_wallpaper", lambda: str(wallpaper))

    _screen(qtbot, "measure")
    assert fake_ambient.calls[-1]["backdrop"] == str(wallpaper)


# ---------------------------------------------------------------------------
# Live changes — no restart
# ---------------------------------------------------------------------------

def test_turning_it_off_removes_a_live_backdrop(qtbot, qt_theme_applied,
                                                fake_ambient, ambient_prefs):
    screen = _screen(qtbot, "measure")
    widget = screen._ambient
    assert widget is not None

    ambient_prefs["enabled"] = False
    screen.refresh_ambient_background()

    assert screen._ambient is None
    assert widget.animating is False, "it was left animating"
    assert widget.parent() is None, "it is still a child of the screen"


def test_turning_it_on_builds_one_on_an_open_screen(qtbot, qt_theme_applied,
                                                    fake_ambient,
                                                    ambient_prefs):
    """The screen was built while the preference was off. No restart."""
    ambient_prefs["enabled"] = False
    screen = _screen(qtbot, "measure")
    assert screen._ambient is None

    ambient_prefs["enabled"] = True
    screen.refresh_ambient_background()
    assert isinstance(screen._ambient, StubAmbient)


def test_a_preferences_save_reaches_an_open_child_screen(
        qtbot, qt_theme_applied, fake_ambient, ambient_prefs,
        app_theme_restored):
    """End to end, through the call a Preferences save actually makes.

    Not a synthesised event: which event Qt really delivers to a screen
    buried in MainWindow's stack is the whole question here, and the
    answer is neither obvious nor the one the DNA rain assumed.
    """
    ambient_prefs["enabled"] = False
    screen = _screen(qtbot, "measure", adopted=True)
    window = _in_a_window(qtbot, screen)
    assert screen._ambient is None

    ambient_prefs["enabled"] = True
    _save_preferences(qt_theme_applied)

    assert isinstance(screen._ambient, StubAmbient), (
        "a preferences save did not reach an open screen inside the window")


def test_coming_back_to_a_tab_picks_up_the_change(qtbot, qt_theme_applied,
                                                  fake_ambient,
                                                  ambient_prefs):
    """The second path: a preference written without a theme re-apply."""
    ambient_prefs["enabled"] = False
    screen = _screen(qtbot, "measure")
    assert screen._ambient is None

    ambient_prefs["enabled"] = True
    screen.show()
    qtbot.waitExposed(screen)
    assert isinstance(screen._ambient, StubAmbient)


def test_a_new_palette_reaches_the_open_widget_without_rebuilding_it(
        qtbot, qt_theme_applied, fake_ambient, ambient_prefs):
    screen = _screen(qtbot, "measure")
    widget = screen._ambient
    built = len(fake_ambient.calls)

    ambient_prefs["palette"] = "ember"
    screen.refresh_ambient_background()

    assert screen._ambient is widget, "the widget was rebuilt, not retuned"
    assert widget.palettes_set == ["ember"]
    assert len(fake_ambient.calls) == built


def test_an_unchanged_preference_does_not_restart_the_animation(
        qtbot, qt_theme_applied, fake_ambient, ambient_prefs):
    """A tab switch must not be a restart. showEvent runs this every time."""
    screen = _screen(qtbot, "measure")
    widget = screen._ambient
    for _ in range(5):
        screen.refresh_ambient_background()
    screen.show()
    qtbot.waitExposed(screen)
    assert screen._ambient is widget
    assert widget.themes_set == []
    assert widget.palettes_set == []
    assert len(fake_ambient.calls) == 1


# ---------------------------------------------------------------------------
# Live theme switch
# ---------------------------------------------------------------------------

def test_a_theme_switch_re_fills_the_backdrop(qtbot, qt_theme_applied,
                                              fake_ambient, ambient_prefs,
                                              monkeypatch,
                                              app_theme_restored):
    """The flat fill is captured at construction — dark to light used to
    leave a black rectangle on a white page."""
    from spacr.qt.theme import palette_for
    screen = _screen(qtbot, "measure", adopted=True)
    window = _in_a_window(qtbot, screen)
    widget = screen._ambient
    widget.backgrounds_set.clear()

    _switch_theme(qt_theme_applied, monkeypatch, "light")

    assert widget.backgrounds_set, "the new theme's fill never arrived"
    # `page`, not `bg`: the backdrop fills the surface the panels float on,
    # and that is now its own palette role. Filling with `bg` is what put a
    # black box behind the settings column.
    assert widget.backgrounds_set[-1] == str(palette_for("light")["page"])


def test_a_theme_switch_keeps_the_users_animation_choice(qtbot,
                                                         qt_theme_applied,
                                                         fake_ambient,
                                                         ambient_prefs,
                                                         monkeypatch,
                                                         app_theme_restored):
    """The palette is a Preferences entry. A theme switch is not consent.

    The rain makes the same call about its trail colour, for the same
    reason: silently resetting something the user picked is worse than a
    slightly off-theme one.
    """
    ambient_prefs["palette"] = "ember"
    screen = _screen(qtbot, "measure", adopted=True)
    window = _in_a_window(qtbot, screen)
    widget = screen._ambient
    assert widget.palette_name == "ember"

    _switch_theme(qt_theme_applied, monkeypatch, "light")

    assert widget.palette_name == "ember"
    assert widget.theme == ambient_prefs["theme"]
    assert widget.themes_set == [], "the screen re-imposed an animation theme"
    assert widget.palettes_set == [], "the screen overwrote a chosen palette"
    # It did re-fill the background, which is the part that *is* the
    # theme's to decide.
    assert widget.backgrounds_set


def test_repeated_palette_events_do_not_re_fill_the_backdrop(
        qtbot, qt_theme_applied, fake_ambient):
    """``PaletteChange`` is chatty — re-applying a stylesheet raises it.

    Every ``set_background_color`` costs the DNA rain its whole strip
    cache and a full repaint, so an unchanged theme must cost nothing.
    """
    screen = _screen(qtbot, "measure", adopted=True)
    window = _in_a_window(qtbot, screen)
    widget = screen._ambient
    for _ in range(4):
        screen.changeEvent(QEvent(QEvent.PaletteChange))
    first = list(widget.backgrounds_set)
    assert len(first) <= 1, "the fill was re-applied for an unchanged theme"
    for _ in range(4):
        screen.changeEvent(QEvent(QEvent.PaletteChange))
    assert widget.backgrounds_set == first


def test_unrelated_change_events_are_ignored(qtbot, qt_theme_applied,
                                             fake_ambient, monkeypatch):
    """Only a palette change re-themes; Enabled and Font changes are not one.

    The stand-in used to raise, so the test's only failure mode was an
    exception rather than an assertion — invisible to the suite-hygiene rule,
    and silent about how many times it was called. It counts now, and the
    count is what is asserted.
    """
    from spacr.qt.screens import app_screen
    screen = _screen(qtbot, "measure")

    calls = []
    monkeypatch.setattr(app_screen, "_theme_wallpaper",
                        lambda *a, **k: calls.append((a, k)))
    screen.changeEvent(QEvent(QEvent.EnabledChange))
    screen.changeEvent(QEvent(QEvent.FontChange))
    assert calls == [], "re-themed on an event that is not a palette change"


def test_a_screen_with_no_backdrop_shrugs_off_a_theme_switch(
        qtbot, qt_theme_applied, fake_ambient, ambient_prefs):
    ambient_prefs["enabled"] = False
    screen = _screen(qtbot, "measure")
    assert screen._ambient is None and screen._dna_rain is None
    screen.changeEvent(QEvent(QEvent.PaletteChange))
    screen.changeEvent(QEvent(QEvent.ApplicationPaletteChange))


def test_the_sequencing_rain_still_follows_the_theme(qtbot, qt_theme_applied,
                                                     fake_ambient, monkeypatch,
                                                     app_theme_restored):
    """The refactor that added the ambient path must not drop the rain's —
    and, in fact, this is the first test that drives it the way the app
    does rather than by handing the screen a synthesised event."""
    from spacr.qt.theme import palette_for
    screen = _screen(qtbot, "map_barcodes", adopted=True)
    window = _in_a_window(qtbot, screen)
    rain = screen._dna_rain
    assert rain is not None
    chosen = rain.color().name()

    _switch_theme(qt_theme_applied, monkeypatch, "light")

    assert rain.background_color().name() == \
        palette_for("light")["page"].lower()
    assert rain.color().name() == chosen, \
        "a colour the user picked must survive a theme switch"


# ---------------------------------------------------------------------------
# The real widget
# ---------------------------------------------------------------------------

def _real_ambient():
    """The shipped ambient module, or skip.

    Owned by another module; these are the claims that a stand-in
    cannot make on its behalf.
    """
    return pytest.importorskip("spacr.qt.widgets.ambient")


@pytest.mark.parametrize("app_key", AMBIENT_APPS)
def test_the_real_widget_lands_on_every_module_screen(qtbot, qt_theme_applied,
                                                      app_key):
    ambient = _real_ambient()
    screen = _screen(qtbot, app_key)
    assert isinstance(screen._ambient, ambient.AmbientWidget), (
        f"{app_key}: no real ambient backdrop")
    assert screen._dna_rain is None


def test_the_real_widget_keeps_the_rains_contract(qtbot, qt_theme_applied):
    ambient = _real_ambient()
    screen = _screen(qtbot, "measure")
    screen.resize(900, 600)
    screen.show()
    qtbot.waitExposed(screen)
    widget = screen._ambient
    assert isinstance(widget, ambient.AmbientWidget)
    assert widget.testAttribute(Qt.WA_TransparentForMouseEvents)
    assert widget.focusPolicy() == Qt.NoFocus
    for point in (QPoint(5, 5), QPoint(screen.width() // 2, 40)):
        assert screen.childAt(point) is not widget
    assert not screen.grab().isNull()


def test_the_real_widget_goes_quiet_with_the_screen(qtbot, qt_theme_applied):
    """Zero cost while the pipeline runs on another tab.

    These screens stay open for the length of a 40-plate run; an
    animation that kept ticking behind a hidden one would cost a core
    for nobody. The timer, not the visibility, is the claim.
    """
    _real_ambient()
    screen = _screen(qtbot, "measure")
    screen.resize(600, 400)
    screen.show()
    qtbot.waitExposed(screen)
    widget = screen._ambient
    assert widget.isVisible()
    assert widget.is_running()
    screen.hide()
    qt_theme_applied.processEvents()
    assert not widget.isVisible()
    assert not widget.is_running(), "the animation is still ticking off-screen"
    screen.show()
    qtbot.waitExposed(screen)
    assert widget.is_running(), "and it never came back"


def test_the_real_widget_honours_the_preference(qtbot, qt_theme_applied,
                                                ambient_prefs):
    _real_ambient()
    ambient_prefs["enabled"] = False
    screen = _screen(qtbot, "measure")
    assert screen._ambient is None
    ambient_prefs["enabled"] = True
    screen.refresh_ambient_background()
    assert screen._ambient is not None
