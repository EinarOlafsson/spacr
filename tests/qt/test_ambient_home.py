"""The ambient backdrop on Home, and what shows through the panels.

Two things are pinned here, both of them user-visible requests:

* Home gets the same animation the module screens do — it was the one page
  in the app that stayed flat.
* The page-opacity preference decides how much of that animation shows
  through the surfaces on top of it (the tile pane on Home; the settings
  cards, console and figure panes on a module screen). At 100 % nothing
  shows through and the app looks exactly as it did before the feature;
  below 100 % the animation is visible behind the content.
"""
from __future__ import annotations

import pytest
from PySide6.QtCore import QSettings, Qt

from spacr.qt import preferences as prefs
from spacr.qt import theme
from spacr.qt.widgets.ambient import AmbientWidget
from spacr.qt.widgets.home import HomePage


APPS = [
    ("mask", "Mask", "Segment cells", "Core"),
    ("measure", "Measure", "Measure objects", "Core"),
    ("map_barcodes", "Map Barcodes", "Sequencing", "Core"),
]


@pytest.fixture(autouse=True)
def _isolated_qsettings(monkeypatch, tmp_path):
    """Never touch the developer's real preferences.

    `preferences._settings()` builds `QSettings(_ORG, _APP)`, and that
    constructor resolves to the NATIVE location whatever `setPath` says —
    measured, after a sibling fixture that assumed otherwise silently erased
    the real store once per test. Replacing the accessor is the only reliable
    isolation, and the assertion below refuses to run if it ever stops
    working.
    """
    store = QSettings(str(tmp_path / "prefs.ini"), QSettings.IniFormat)
    monkeypatch.setattr(prefs, "_settings", lambda: store)
    assert str(tmp_path) in store.fileName(), (
        "QSettings isolation failed; refusing to write to real preferences")
    return store


def _home() -> HomePage:
    return HomePage(APPS, lambda key: None)


# ---------------------------------------------------------------------------
# Home gets the backdrop
# ---------------------------------------------------------------------------

def test_home_installs_the_ambient_backdrop(qtbot, qt_theme_applied):
    prefs.set_ambient_enabled(True)
    page = _home()
    qtbot.addWidget(page)
    assert page._ambient is not None
    assert isinstance(page._ambient, AmbientWidget)


def test_the_backdrop_is_behind_everything_and_takes_no_input(
        qtbot, qt_theme_applied):
    """Lowered and inert, or it would eat clicks meant for the tiles."""
    prefs.set_ambient_enabled(True)
    page = _home()
    qtbot.addWidget(page)

    children = page.children()
    assert children.index(page._ambient) == 0, "backdrop is not lowered"
    assert page._ambient.testAttribute(Qt.WA_TransparentForMouseEvents)
    assert page._ambient.focusPolicy() == Qt.NoFocus


def test_the_preference_off_means_not_built_at_all(qtbot, qt_theme_applied):
    """Off must skip construction, not build-then-hide.

    The construction is itself the cost the toggle exists to avoid on a
    machine already running Cellpose on the GPU.
    """
    prefs.set_ambient_enabled(False)
    page = _home()
    qtbot.addWidget(page)
    assert page._ambient is None
    assert page.findChildren(AmbientWidget) == []


def test_home_still_opens_when_the_backdrop_cannot_be_built(
        qtbot, qt_theme_applied, monkeypatch):
    """A decorative background must never stop Home from opening."""
    import spacr.qt.widgets.ambient as ambient_mod

    def _boom(*_a, **_k):
        raise RuntimeError("no GPU, no pixmap, no luck")

    monkeypatch.setattr(ambient_mod, "install_ambient", _boom)
    prefs.set_ambient_enabled(True)

    page = _home()          # must not raise
    qtbot.addWidget(page)
    assert page._ambient is None
    # And nothing half-built was left parented with a live timer.
    assert page.findChildren(AmbientWidget) == []


def test_a_failed_install_leaves_the_page_opaque(qtbot, qt_theme_applied,
                                                 monkeypatch):
    """Surfaces are cleared only after a SUCCESSFUL install.

    Clearing them first would leave a failed page transparent with nothing
    behind it — a worse outcome than simply having no animation.
    """
    import spacr.qt.widgets.ambient as ambient_mod
    monkeypatch.setattr(ambient_mod, "install_ambient",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError()))
    prefs.set_ambient_enabled(True)

    page = _home()
    qtbot.addWidget(page)
    body = page.layout().itemAt(0).widget()
    assert not body.property(theme.TRANSPARENT_PROPERTY)


def test_a_successful_install_clears_the_positioning_containers(
        qtbot, qt_theme_applied):
    """Without this the animation reaches the eye only through layout gaps."""
    prefs.set_ambient_enabled(True)
    page = _home()
    qtbot.addWidget(page)

    body = page.layout().itemAt(0).widget()
    assert body.property(theme.TRANSPARENT_PROPERTY) is True
    assert page._hint_bar.property(theme.TRANSPARENT_PROPERTY) is True


# ---------------------------------------------------------------------------
# What shows through — the page-opacity preference
# ---------------------------------------------------------------------------

def test_pane_opacity_controls_how_much_of_the_backdrop_shows_on_home(
        qtbot, qt_theme_applied):
    """The tile pane is drawn at the requested alpha, not forced opaque.

    This is the mechanism behind "visible behind the settings, console,
    figure and chat when opacity is less than 100": the surfaces on top are
    painted at `effective_pane_alpha()`, so lowering it lets the animation
    through rather than hiding it.
    """
    prefs.set_ambient_enabled(True)
    page = _home()
    qtbot.addWidget(page)

    # `_pane_alpha()` reads the preference LIVE rather than caching it at
    # construction, which is what lets an open page pick up a change on the
    # next restyle instead of needing a rebuild. So the two values have to be
    # sampled against the same page, one setting at a time — comparing two
    # separately-built pages measures nothing, because both of them report
    # whatever the preference says at the moment they are asked.
    prefs.set_pane_opacity(1.0)
    solid = page._pane_alpha()
    prefs.set_pane_opacity(0.4)
    sheer = page._pane_alpha()

    assert solid == pytest.approx(1.0)
    assert sheer < solid, (
        "lowering page opacity did not thin the pane the tiles sit on")
    assert sheer == pytest.approx(0.4), (
        "the flat themes have no legibility floor, so the request should be "
        "honoured exactly")


def test_the_tile_pane_is_cleared_with_the_other_containers(
        qtbot, qt_theme_applied):
    """The box behind the tiles is a container, so it paints nothing.

    This flipped once. It was briefly held out of `_clear_page_surfaces` so
    the opacity preference could dial it — then the instruction settled on
    removing the black boxes behind the tiles outright and letting the TILES
    carry the opacity instead, which is where it can actually be seen.
    """
    prefs.set_ambient_enabled(True)
    prefs.set_pane_opacity(1.0)
    page = _home()
    qtbot.addWidget(page)

    assert page._tabs.property(theme.TRANSPARENT_PROPERTY) is True


@pytest.mark.parametrize("requested", [1.0, 0.75, 0.5, 0.25])
def test_module_surfaces_honour_the_same_preference(requested):
    """The settings cards, console and figure panes thin out together.

    They take their alpha from `panel_alpha`, which the app stylesheet is
    built with (`preferences.apply_theme` passes `surface_opacity=
    get_pane_opacity()`), so one preference moves all of them. On the flat
    themes there is no legibility floor to fight — a dark card fading into a
    dark window cannot make its own white text harder to read — so the
    requested value is honoured exactly.
    """
    for role in ("surface", "surface_alt", "tile"):
        assert theme.panel_alpha("dark", role, requested) == pytest.approx(
            requested), f"{role} did not honour the requested opacity"


def test_an_image_theme_keeps_its_legibility_floor():
    """Space is clamped, and that is correct rather than a bug.

    Its wallpaper can put something bright behind a card, so thinning the
    card past the floor would make its text unreadable. The flat themes have
    no such constraint, which is why they pass the request straight through.
    """
    floored = theme.panel_alpha("space", "surface", 0.0)
    assert floored > 0.5
    assert theme.panel_alpha("space", "surface", 1.0) == pytest.approx(1.0)
