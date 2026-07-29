"""#16j: maturity as a colour, a legend for it, a rim, and the dock.

Four claims, all of them things the user asked for in words and none of
them checkable by reading widget structure alone:

1. **Hovering a module tile lights it in the colour of how finished it
   is** — ``#3B82F6`` stable, ``#FF00FF`` beta, ``#00CEC8`` alpha. The
   hover is asserted by *painting the widget and reading the pixels*,
   because the thing that can go wrong is a QSS selector that never
   matches, and a selector that never matches is invisible to every
   structural assertion there is.
2. **A legend under the right-hand tiles says what the colours mean**,
   in words as well as in swatches — colour alone fails WCAG 1.4.1.
3. **Every tile carries a thin rim** in the theme's ink: white on the
   dark themes, near-black on the light one.
4. **The dock obeys the preference**: revealed on hover, locked open as
   a real column, or not there at all.

On ``QTest.mouseMove``: setting ``WA_UnderMouse`` by hand does *not*
produce a ``:hover`` — ``QAbstractButton`` tracks its own hover flag
from the enter/leave events the platform delivers, so the pointer has to
really be moved. That is the only route that exercises the rule the app
actually ships.
"""
from __future__ import annotations

import pytest

from PySide6.QtCore import QPoint
from PySide6.QtGui import QColor
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QLabel

from spacr.qt import theme
from spacr.qt.app import APPS, MainWindow, app_stage, make_home_page
from spacr.qt.widgets.home import AppTile, StageLegend


THEMES = ("dark", "light")

#: Where a page that is about to be hovered is parked. Far enough from
#: the origin that no window another test left behind can be sitting on
#: the same global coordinates — see ``_themed_page``.
_LONELY_CORNER = QPoint(4000, 2400)


def _themed_page(qtbot, monkeypatch, theme_name: str):
    """A real Home page rendered under ``theme_name``.

    Two halves, and both are needed. The stylesheet goes on the PAGE
    rather than on the QApplication — a global ``setStyleSheet``
    re-polishes every widget any other test left behind, which is slow
    and a good way to crash on a stale one. And the preference is
    patched first, because ``HomePage`` resolves its inline colours
    through ``preferences.resolve_effective_theme()`` at construction:
    without this the page would paint dark panels under a light
    stylesheet, and the pixels read below would be measuring that
    mismatch rather than the rules.
    """
    from spacr.qt import preferences as prefs
    monkeypatch.setattr(prefs, "resolve_effective_theme",
                        lambda: theme_name)
    page = make_home_page()
    qtbot.addWidget(page)
    page.setStyleSheet(theme.stylesheet(theme_name))
    page.resize(1400, 900)
    # Parked well away from the origin, and that is not cosmetic.
    # `QApplicationPrivate::dispatchEnterLeave` decides who is under the
    # pointer with `QApplication::widgetAt(globalPos)` — a *global*
    # lookup. A full test run leaves other top-level windows behind at
    # (0, 0), so a synthetic move onto a tile at the same global
    # coordinates finds one of those instead and this page never gets
    # `WA_UnderMouse` at all. The symptom is a hover test that passes
    # alone and fails in the suite.
    page.move(_LONELY_CORNER)
    page.show()
    qtbot.waitExposed(page)
    return page


def _close_stray_popups() -> None:
    """Dismiss any popup left open by an earlier test.

    While Qt is in popup mode the *application* grabs the mouse and
    every mouse event is redirected to the popup, so a synthetic move
    onto a tile is delivered somewhere else entirely and no hover
    happens. In this file alone the symptom is a hover test that passes
    in isolation and fails in a full run, which is the worst kind of
    flake — so the grab is cleared explicitly rather than hoped away.
    """
    from PySide6.QtWidgets import QApplication
    for _ in range(10):
        popup = QApplication.activePopupWidget()
        if popup is None:
            break
        popup.close()
        QApplication.processEvents()


def _hover(qtbot, tile) -> None:
    """Put a real pointer on ``tile``.

    ``setAttribute(Qt.WA_UnderMouse, True)`` does NOT produce a
    ``:hover`` — ``QAbstractButton`` keeps its own hover flag, set from
    the enter/leave events the platform delivers, and that is what ends
    up in ``QStyleOption.state``. The pointer is parked off the tile
    first so that consecutive hovers in one test cannot be swallowed as
    "the cursor is already there".
    """
    _close_stray_popups()
    window = tile.window()
    window.raise_()
    window.activateWindow()
    QTest.mouseMove(window, QPoint(0, 0))
    qtbot.wait(1)
    QTest.mouseMove(tile, QPoint(tile.width() // 2, 6))
    qtbot.wait(1)
    assert tile.underMouse(), (
        f"the synthetic pointer never reached {tile.text_label} — "
        "something else in this process is holding the mouse, and the "
        "pixels below would be measuring the un-hovered tile")


def _visible_tiles(page) -> list:
    return [t for t in page._tabs.widget(0).findChildren(AppTile)
            if t.isVisible()]


def _one_per_stage(page) -> dict:
    picked: dict = {}
    for tile in _visible_tiles(page):
        picked.setdefault(tile.stage, tile)
    return picked


def _rim_pixel(page, tile) -> QColor:
    """The colour of the tile's left border, halfway down.

    Grabbed from the PAGE, not from the tile. A tile paints
    ``background: transparent`` on the opaque themes, so grabbing it
    alone hands back a pixmap with nothing behind the border, and a
    translucent rim over nothing composites to (0, 0, 0, 0) — a
    measurement of the absence of a backdrop rather than of the rim.
    From the page, the rim is composited over the panel it sits on,
    which is what the eye sees.
    """
    image = page.grab().toImage()
    origin = tile.mapTo(page, QPoint(0, 0))
    y = origin.y() + tile.height() // 2
    # The border is one physical pixel a column or so in, depending on
    # the corner radius: take the one that differs most from the panel
    # just outside the tile.
    behind = image.pixelColor(max(0, origin.x() - 3), y)
    candidates = [image.pixelColor(origin.x() + dx, y) for dx in range(0, 4)]
    return max(candidates,
               key=lambda c: (abs(c.red() - behind.red())
                              + abs(c.green() - behind.green())
                              + abs(c.blue() - behind.blue())))


def _near(a: QColor, b: str, tol: int = 6) -> bool:
    other = QColor(b)
    return (abs(a.red() - other.red()) <= tol
            and abs(a.green() - other.green()) <= tol
            and abs(a.blue() - other.blue()) <= tol)


# ===========================================================================
# 1. The stage is on the tile, and it is the registry's answer
# ===========================================================================

def test_every_tile_carries_the_stage_the_registry_gave_it(qtbot,
                                                           qt_theme_applied):
    """The property the stylesheet selects on, and where it comes from.

    ``AppTile`` never looks a stage up — ``HomePage`` is handed the whole
    mapping by ``make_home_page`` — so this is the join between the two
    tables, asserted on the widget that has to carry it."""
    page = make_home_page()
    qtbot.addWidget(page)
    seen = {}
    for tile in page.findChildren(AppTile):
        seen.setdefault(tile.text_label, set()).add(tile.stage)
    expected = {name: {app_stage(key)} for key, name, *_r in APPS}
    assert seen == expected
    for tile in page.findChildren(AppTile):
        assert tile.property("stage") == tile.stage, (
            "the Qt property is what QSS selects on; the attribute is "
            "what the tests read — they have to be the same string")


def test_the_stage_is_a_word_on_the_tile_not_only_a_colour(qtbot,
                                                           qt_theme_applied):
    """WCAG 1.4.1: colour is never the only carrier of information.

    A screen reader gets "alpha" out of the accessible description and
    the tooltip; a colour-blind user gets it out of the tooltip. The
    hover hue is the fast path, not the only one."""
    page = make_home_page()
    qtbot.addWidget(page)
    for tile in page.findChildren(AppTile):
        word = theme.STAGE_LABEL[tile.stage]
        assert word in tile.accessibleDescription()
        assert word.lower() in tile.toolTip().lower()


# ===========================================================================
# 2. Hovering paints the stage colour — read off the pixels
# ===========================================================================

@pytest.mark.parametrize("theme_name", THEMES)
def test_hovering_a_tile_lights_it_in_its_stage_colour(qtbot, monkeypatch,
                                                       theme_name):
    """The three hues, on the three kinds of module, in both themes.

    All three stages are on the Home tab, which is why this hovers
    there: Core alone has no alpha module in it.
    """
    page = _themed_page(qtbot, monkeypatch, theme_name)
    picked = _one_per_stage(page)
    assert set(picked) == {"stable", "beta", "alpha"}, (
        f"the Home tab does not show all three stages: {sorted(picked)}")

    for stage, tile in picked.items():
        _hover(qtbot, tile)
        rim = _rim_pixel(page, tile)
        assert _near(rim, theme.STAGE_HOVER[stage]), (
            f"{theme_name}: hovering {tile.text_label} ({stage}) drew "
            f"{rim.name()}, not {theme.STAGE_HOVER[stage]}")


def test_the_three_hover_colours_are_the_ones_that_were_asked_for():
    """Recorded verbatim, so a re-hue is a decision and not a drift."""
    assert theme.STAGE_HOVER == {
        "stable": "#3B82F6",
        "beta":   "#FF00FF",
        "alpha":  "#00CEC8",
    }
    assert theme.stage_hover("alpha") == "#00CEC8"
    assert theme.stage_hover("no such stage") == theme.STAGE_HOVER["stable"]


@pytest.mark.parametrize("theme_name", THEMES)
def test_a_tile_that_is_not_hovered_shows_the_rim_instead(qtbot, monkeypatch,
                                                          theme_name):
    """"there should always be a thin white rim (black in white mode)".

    Always: the un-hovered state is the one this is about. It used to be
    ``border: 1px solid transparent``, i.e. nothing at all until you
    hovered, which with the descriptions gone would leave the tiles as
    floating icons with no edges.
    """
    page = _themed_page(qtbot, monkeypatch, theme_name)
    palette = theme.palette_for(theme_name)
    # The rim IS the theme's ink — white on dark, near-black on light —
    # painted at 35 %. Derived, never a literal: a theme added later
    # gets a visible rim without anyone remembering to write one down.
    assert theme.rim_colour(theme_name) == palette["fg"]
    panel = QColor(palette["surface"])
    expected = QColor(theme.composite(palette["fg"], 0.35, palette["surface"]))

    for tile in _visible_tiles(page)[:6]:
        rim = _rim_pixel(page, tile)
        assert not _near(rim, panel.name(), tol=8), (
            f"{theme_name}: {tile.text_label} has no visible rim "
            f"({rim.name()} is the panel colour)")
        if theme_name == "light":
            assert rim.lightness() < panel.lightness(), (
                f"a light-theme rim must be darker ink, got {rim.name()}")
        else:
            assert rim.lightness() > panel.lightness(), (
                f"a dark-theme rim must be lighter ink, got {rim.name()}")
        # …and it is the ink at 35 %, not some other grey.
        assert _near(rim, expected.name(), tol=24), (
            f"{theme_name}: rim {rim.name()} is not the ink at 35 % "
            f"({expected.name()})")


# ===========================================================================
# 3. The legend
# ===========================================================================

def test_the_legend_sits_under_the_right_hand_tiles(qtbot, qt_theme_applied):
    """"a legend under the right side tiles indicating color and module
    state (alpha, beta, stable)" — the user.

    Under: it is the last thing in the aside column, after the panels of
    numbers, because it explains the tiles rather than reporting on the
    machine."""
    page = make_home_page()
    qtbot.addWidget(page)
    legend = page.legend
    assert isinstance(legend, StageLegend)

    aside = legend.parent()
    order = [aside.layout().itemAt(i).widget()
             for i in range(aside.layout().count())]
    widgets = [w for w in order if w is not None]
    assert widgets[-1] is legend, (
        f"the legend is not last in the aside: {widgets}")


def test_the_legend_names_every_stage_and_draws_its_colour(qtbot,
                                                           qt_theme_applied):
    page = make_home_page()
    qtbot.addWidget(page)
    legend = page.legend

    words = {lbl.text() for lbl in legend.findChildren(QLabel)}
    for stage, label in theme.STAGE_LABEL.items():
        assert label in words, f"the legend does not name {stage}"
        assert legend.row_for(stage) is not None
        assert legend.swatch_colour(stage) == theme.STAGE_HOVER[stage]

    # The swatch is filled with the same hex the hover rule uses. Reading
    # it out of the widget's own stylesheet is what stops the legend and
    # the tiles drifting apart.
    for stage in theme.STAGE_HOVER:
        chip = next(lbl for lbl in legend.findChildren(QLabel)
                    if lbl.objectName() == f"StageSwatch_{stage}")
        assert theme.STAGE_HOVER[stage] in chip.styleSheet()
        assert chip.width() > 0 and chip.height() > 0


def test_the_legend_lists_the_unfinished_first(qtbot, qt_theme_applied):
    """Alpha, then beta, then stable.

    The row a user needs to have read before they trust a number is the
    one they should meet first."""
    page = make_home_page()
    qtbot.addWidget(page)
    legend = page.legend
    order = [lbl.text() for lbl in legend.findChildren(QLabel)
             if lbl.text() in set(theme.STAGE_LABEL.values())]
    assert order == ["Alpha", "Beta", "Stable"]


def test_each_legend_row_explains_itself(qtbot, qt_theme_applied):
    page = make_home_page()
    qtbot.addWidget(page)
    for stage, note in theme.STAGE_NOTE.items():
        assert page.legend.row_for(stage).toolTip() == note
        assert note.endswith(".")


# ===========================================================================
# 4. The page-opacity preference
# ===========================================================================

class TestPaneOpacity:
    """The preference is a request; the legibility floor is not.

    ``pane_alpha`` clamps the user's number UP to
    ``pane_alpha_floor`` — the thinnest the panel can be painted with
    the tile names on it still clearing WCAG AA over the brightest pixel
    that theme's wallpaper can present. The solver's *other* bound,
    ``present_scrim_ceiling``, is deliberately NOT applied: it exists to
    keep the wallpaper visible, and a user dragging the slider to 100 %
    is asking for the wallpaper to be hidden.
    """

    def test_the_default_is_solid_except_for_the_glass_material(self):
        assert theme.DEFAULT_PANE_OPACITY == 1.0
        for name in ("dark", "light", "space", "cell"):
            assert theme.pane_alpha(name) == 1.0
            assert theme.pane_alpha(name, None) == 1.0
        # Glass owns a translucent material independently of the preference:
        # 100% means full material strength, not an opaque coloured sheet.
        assert theme.pane_alpha("glass") == \
            theme.scrim_alpha("glass", "surface")
        assert theme.pane_alpha("glass", None) == \
            theme.scrim_alpha("glass", "surface")

    def test_an_opaque_theme_lets_the_user_take_the_box_away(self):
        """Nothing behind a dark panel on the dark theme but more dark
        theme, so no alpha can make text harder to read — the floor is
        zero and the user gets the whole range."""
        for name in ("dark", "light"):
            assert theme.pane_alpha_floor(name) == 0.0
            assert theme.pane_alpha(name, 0.0) == 0.0
            assert theme.pane_alpha(name, 0.5) == 0.5

    def test_an_image_theme_clamps_up_to_the_legible_floor(self):
        for name in theme.IMAGE_THEMES:
            floor = theme.pane_alpha_floor(name)
            # A sufficiently dark, exposure-bounded backdrop (Glass) can
            # remain readable with no pane fill at all.
            assert 0.0 <= floor < 1.0
            assert theme.pane_alpha(name, 0.0) == floor
            assert theme.pane_alpha(name, floor / 2) == floor
            above = min(1.0, floor + 0.1)
            if name == "glass":
                # Glass treats the preference as material strength: even
                # 100% retains the designed translucency.
                designed = theme.scrim_alpha("glass", "surface")
                assert theme.pane_alpha(name, above) == max(
                    floor, above * designed)
            else:
                # Conventional image themes honour the literal opacity.
                assert theme.pane_alpha(name, above) == above

    def test_the_floor_is_where_the_text_stops_clearing_aa(self):
        """Not a magic number: re-derived from the contrast rules."""
        for name in theme.IMAGE_THEMES:
            floor = theme.pane_alpha_floor(name)
            palette = theme.palette_for(name)
            under = theme.scrim_under(name)
            at_floor = theme.composite(palette["surface"], floor, under)
            assert theme.contrast_ratio(palette["fg"], at_floor) >= 4.5
            if floor > 0.0:
                below = theme.composite(
                    palette["surface"], max(0.0, floor - 0.05), under)
                assert theme.contrast_ratio(palette["fg"], below) < \
                    theme.contrast_ratio(palette["fg"], at_floor)

    def test_out_of_range_and_junk_values_are_survivable(self):
        assert theme.pane_alpha("dark", -5) == 0.0
        assert theme.pane_alpha("dark", 42) == 1.0

    def test_the_preference_round_trips(self, tmp_settings):
        from spacr.qt import preferences as prefs
        assert prefs.get_pane_opacity() == 1.0
        prefs.set_pane_opacity(0.4)
        assert prefs.get_pane_opacity() == pytest.approx(0.4)
        prefs.set_pane_opacity("nonsense")
        assert prefs.get_pane_opacity() == 1.0
        prefs.set_pane_opacity(3.0)
        assert prefs.get_pane_opacity() == 1.0

    def test_the_home_pane_is_painted_at_the_effective_alpha(
            self, qtbot, qt_theme_applied, tmp_settings):
        """The rounded box behind the tiles is a SURFACE at that alpha.

        It used to be the page background — the same colour as the
        window — which is a box whose opacity could never show a
        difference on the two opaque themes.
        """
        from spacr.qt import preferences as prefs
        from spacr.qt.widgets.home import _tab_qss

        prefs.set_pane_opacity(0.5)
        page = make_home_page()
        qtbot.addWidget(page)
        alpha = prefs.effective_pane_alpha()
        assert alpha == theme.pane_alpha(prefs.resolve_effective_theme(), 0.5)

        palette = theme.palette_for(prefs.resolve_effective_theme())
        expected = theme.css_color(palette["surface"], alpha)
        qss = page._tabs.styleSheet()
        assert expected in qss
        assert qss == _tab_qss(palette, alpha)

    @pytest.mark.parametrize("name", theme.THEMES)
    def test_the_preference_controls_shared_module_surfaces(self, name):
        requested = 0.35
        palette = theme.palette_for(name)
        qss = theme.stylesheet(name, surface_opacity=requested)
        for role in ("surface", "surface_alt", "surface_hi"):
            alpha = theme.panel_alpha(name, role, requested)
            expected = theme.css_color(palette[role], alpha)
            assert expected in qss
        # Native popup windows never reveal the desktop through themselves.
        elevated = theme.css_color(
            palette["surface_alt"],
            theme.panel_alpha(name, "elevated", requested))
        assert elevated in qss
        assert theme.panel_alpha(name, "elevated", requested) == 1.0

    def test_module_surface_opacity_keeps_each_roles_legibility_floor(self):
        for name in theme.IMAGE_THEMES:
            for role in ("surface", "surface_alt", "surface_hi", "tile"):
                alpha = theme.panel_alpha(name, role, 0.0)
                assert 0.0 <= alpha <= 1.0
                assert alpha == theme.panel_alpha(name, role, -5.0)


# ===========================================================================
# 5. The dock preference
# ===========================================================================

class TestDockModes:
    """One ``Sidebar`` object, three places it can be.

    The object identity is the load-bearing part: the tutorial overlay,
    the command palette and half the suite reach the app list as
    ``window._sidebar``, so locking the dock has to MOVE that widget
    rather than build a second one.
    """

    def test_the_default_is_locked_open(self, tmp_settings):
        from spacr.qt import preferences as prefs
        assert prefs.DEFAULT_DOCK_MODE == "locked"
        assert prefs.get_dock_mode() == "locked"

    def test_an_unknown_mode_falls_back_rather_than_raising(self,
                                                            tmp_settings):
        from spacr.qt import preferences as prefs
        prefs._settings().setValue(prefs._KEY_DOCK_MODE, "sideways")
        assert prefs.get_dock_mode() == "locked"
        with pytest.raises(ValueError):
            prefs.set_dock_mode("sideways")

    def test_an_unreadable_preference_also_falls_back_to_locked(
            self, monkeypatch):
        from spacr.qt import preferences as prefs

        def unreadable():
            raise OSError("settings unavailable")

        monkeypatch.setattr(prefs, "get_dock_mode", unreadable)
        assert MainWindow.dock_mode(object()) == "locked"

    def test_auto_keeps_the_sidebar_in_the_drawer(self, qtbot,
                                                  qt_theme_applied,
                                                  tmp_settings):
        from spacr.qt import preferences as prefs
        prefs.set_dock_mode("auto")
        win = MainWindow()
        qtbot.addWidget(win)
        assert win.dock_mode() == "auto"
        assert win._sidebar.parent() is win._app_drawer
        assert win._app_drawer.is_enabled()
        assert not win._dock_slot.isVisible()
        assert win._act_all_apps.isEnabled()
        win.toggle_app_drawer()
        assert win._app_drawer.is_open()

    def test_locked_makes_it_a_column_that_never_slides(self, qtbot,
                                                        qt_theme_applied,
                                                        tmp_settings):
        from spacr.qt import preferences as prefs
        prefs.set_dock_mode("locked")
        win = MainWindow()
        qtbot.addWidget(win)
        win.resize(1440, 900)
        win.show()
        qtbot.waitExposed(win)

        assert win._dock_slot.isVisible()
        assert win._sidebar.parent() is win._dock_slot
        assert win._sidebar.isVisible()
        assert win._sidebar.width() == win._sidebar.fitting_width()
        # The reveal is disarmed, so the panel cannot also slide in over
        # the column it already is.
        assert not win._app_drawer.is_enabled()
        win.toggle_app_drawer()
        assert not win._app_drawer.is_open()
        # …and clicking a row does not make the column vanish.
        win._on_drawer_navigated("measure")
        assert win._sidebar.isVisible()

        # A window resize must not drag the column back to the drawer's
        # width. `EdgeDrawer.relayout` still runs on every host resize;
        # it has to leave a panel it no longer owns alone.
        width = win._sidebar.width()
        win.resize(1100, 760)
        qtbot.wait(1)
        win._app_drawer.relayout()
        assert win._sidebar.width() == width

    def test_hidden_takes_the_strip_away_and_says_where_to_get_it_back(
            self, qtbot, qt_theme_applied, tmp_settings):
        from spacr.qt import preferences as prefs
        prefs.set_dock_mode("hidden")
        win = MainWindow()
        qtbot.addWidget(win)
        assert not win._dock_slot.isVisible()
        assert not win._app_drawer.is_enabled()
        assert not win._app_drawer.is_open()
        win.toggle_app_drawer()
        assert not win._app_drawer.is_open(), (
            "Ctrl+B opened a dock the user asked not to have")
        assert not win._act_all_apps.isEnabled()
        assert "Preferences" in win._act_all_apps.toolTip()

    def test_hiding_the_dock_leaves_every_app_reachable(self, qtbot,
                                                        qt_theme_applied,
                                                        tmp_settings):
        """A dock you cannot summon must not be a dead end."""
        from spacr.qt import preferences as prefs
        prefs.set_dock_mode("hidden")
        win = MainWindow()
        qtbot.addWidget(win)
        labels: set = set()
        for top in win.menuBar().actions():
            if top.text().replace("&", "") != "spaCR":
                continue
            for act in top.menu().actions():
                if not act.isSeparator():
                    labels.add(act.text())
            break
        assert {name for _k, name, *_r in APPS} <= labels
        drawn = {t.text_label
                 for t in win._startup._tabs.widget(0).findChildren(AppTile)}
        assert drawn == {name for _k, name, *_r in APPS}

    def test_switching_modes_moves_the_same_widget_back_and_forth(
            self, qtbot, qt_theme_applied, tmp_settings):
        from spacr.qt import preferences as prefs
        prefs.set_dock_mode("auto")
        win = MainWindow()
        qtbot.addWidget(win)
        sidebar = win._sidebar

        win.apply_dock_mode("locked")
        assert win._sidebar is sidebar
        assert sidebar.parent() is win._dock_slot

        win.apply_dock_mode("auto")
        assert win._sidebar is sidebar
        assert sidebar.parent() is win._app_drawer
        assert win._app_drawer.is_enabled()
        assert not win._dock_slot.isVisible()

        # Idempotent: applying the same mode twice changes nothing.
        win.apply_dock_mode("auto")
        assert sidebar.parent() is win._app_drawer

    def test_a_disarmed_drawer_ignores_the_hot_strip(self, qtbot,
                                                     qt_theme_applied,
                                                     tmp_settings):
        """``set_enabled(False)`` hides the trigger AND refuses to arm.

        Hiding it alone would leave a keyboard or synthetic enter event
        able to start the dwell timer."""
        from spacr.qt import preferences as prefs
        prefs.set_dock_mode("hidden")
        win = MainWindow()
        qtbot.addWidget(win)
        drawer = win._app_drawer
        assert not drawer._trigger.isVisible()
        drawer.arm()
        assert not drawer._open_timer.isActive()
        assert not drawer.is_open()


class TestPreferencesDialog:
    """Both new controls exist, show the stored value, and are saved.

    A control that is drawn but never read in ``_save`` is the exact bug
    this is here for: the dialog would look right and change nothing.
    """

    def _dock_combo(self, dlg):
        from PySide6.QtWidgets import QComboBox
        from spacr.qt.preferences import VALID_DOCK_MODES
        for combo in dlg.findChildren(QComboBox):
            data = {combo.itemData(i) for i in range(combo.count())}
            if data == set(VALID_DOCK_MODES):
                return combo
        raise AssertionError("no dock-mode combo in the Preferences dialog")

    def _opacity_slider(self, dlg):
        from PySide6.QtWidgets import QSlider
        for slider in dlg.findChildren(QSlider):
            if (slider.minimum(), slider.maximum()) == (0, 100):
                return slider
        raise AssertionError("no page-opacity slider in the dialog")

    def test_the_dock_mode_round_trips_through_the_dialog(
            self, qtbot, qt_theme_applied, tmp_settings, monkeypatch):
        from PySide6.QtWidgets import QDialogButtonBox
        from spacr.qt import preferences as prefs

        monkeypatch.setattr(prefs, "apply_preferences_to_app", lambda *a: None)
        prefs.set_dock_mode("locked")
        dlg = prefs.PreferencesDialog()
        qtbot.addWidget(dlg)
        combo = self._dock_combo(dlg)
        assert combo.currentData() == "locked", (
            "the dialog opened on a mode the user did not choose")
        combo.setCurrentIndex(
            next(i for i in range(combo.count())
                 if combo.itemData(i) == "hidden"))
        dlg.findChild(QDialogButtonBox).accepted.emit()
        assert prefs.get_dock_mode() == "hidden"

    def test_the_opacity_round_trips_through_the_dialog(
            self, qtbot, qt_theme_applied, tmp_settings, monkeypatch):
        from PySide6.QtWidgets import QDialogButtonBox
        from spacr.qt import preferences as prefs

        monkeypatch.setattr(prefs, "apply_preferences_to_app", lambda *a: None)
        prefs.set_pane_opacity(0.6)
        dlg = prefs.PreferencesDialog()
        qtbot.addWidget(dlg)
        slider = self._opacity_slider(dlg)
        assert slider.value() == 60
        slider.setValue(35)
        dlg.findChild(QDialogButtonBox).accepted.emit()
        assert prefs.get_pane_opacity() == pytest.approx(0.35)

    def test_the_slider_says_when_the_theme_will_overrule_it(
            self, qtbot, qt_theme_applied, tmp_settings, monkeypatch):
        """A number the app quietly ignores is worse than no control.

        On an image theme the legibility floor is well above zero, so
        the readout has to admit that "20 %" is going to be honoured as
        something else."""
        from spacr.qt import preferences as prefs
        from PySide6.QtWidgets import QLabel

        monkeypatch.setattr(prefs, "apply_preferences_to_app", lambda *a: None)
        monkeypatch.setattr(prefs, "resolve_effective_theme", lambda: "space")
        dlg = prefs.PreferencesDialog()
        qtbot.addWidget(dlg)
        slider = self._opacity_slider(dlg)
        slider.setValue(10)
        readouts = [lbl.text() for lbl in dlg.findChildren(QLabel)
                    if lbl.text().startswith("10%")]
        assert readouts, "the slider has no readout"
        held = int(round(theme.pane_alpha("space", 0.10) * 100))
        assert f"held at {held}%" in readouts[0], readouts
        # …and when nothing is being overruled it says only the number.
        slider.setValue(100)
        plain = [lbl.text() for lbl in dlg.findChildren(QLabel)
                 if lbl.text() == "100%"]
        assert plain


@pytest.fixture
def tmp_settings(tmp_path, monkeypatch):
    """A throwaway QSettings file, so a test never edits the real prefs."""
    from PySide6.QtCore import QSettings
    from spacr.qt import preferences as prefs

    path = tmp_path / "spacr-test.ini"

    def _fake():
        return QSettings(str(path), QSettings.IniFormat)

    monkeypatch.setattr(prefs, "_settings", _fake)
    yield path
