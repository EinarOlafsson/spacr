"""Theme, backdrop and prose-box work on a screen that is being taken apart.

A palette change is a chatty event -- re-applying the stylesheet raises one --
and it arrives at every screen, including ones already mid-teardown. So the
re-theming paths are asked to touch widgets whose C++ side has gone, to resolve
a theme from preferences that may not answer, and to re-render prose boxes that
were removed with the section they belonged to.

The page's own fill is the other half. A screen animating something of its own
paints nothing, an image theme paints nothing so the picture shows through, and
a theme that cannot be resolved falls back to painting nothing rather than
guessing a colour that would show as a slab.
"""
from __future__ import annotations

import os
import types

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from spacr.qt.screens.app_screen import AppScreen                # noqa: E402

pytestmark = pytest.mark.qt


def _boom(*_args, **_kwargs):
    raise RuntimeError("the widget has been destroyed")


@pytest.fixture
def screen(qtbot):
    widget = AppScreen("regression")
    qtbot.addWidget(widget)
    return widget


class TestTheAmbientBackdrop:

    def test_a_screen_that_already_has_one_does_not_build_a_second(
            self, screen, monkeypatch):
        """A palette event arrives at every screen on every stylesheet apply.

        Building another backdrop each time would stack animations behind
        one page, each with its own timer.
        """
        from spacr.qt import preferences

        monkeypatch.setattr(screen, "_ambient", object())
        monkeypatch.setattr(preferences, "get_ambient_enabled", _boom)

        screen._install_ambient()

        assert screen._ambient is not None

    def test_a_backdrop_that_will_not_stop_animating_is_still_discarded(
            self, screen, monkeypatch):
        """Qt owns it through its parent; dropping the reference is not enough.

        A backdrop left parented keeps painting and keeps its timer, which is
        the whole cost of the failure this swallows.
        """
        widget = types.SimpleNamespace(set_animating=_boom)
        discarded = []
        monkeypatch.setattr("spacr.qt.screens.app_screen._discard_widget",
                            discarded.append)
        monkeypatch.setattr(screen, "_ambient", widget)

        screen._remove_ambient()

        assert discarded == [widget]
        assert screen._ambient is not widget

    def test_an_orphan_backdrop_that_will_not_stop_is_still_taken_away(
            self, screen, monkeypatch, qtbot):
        """A backdrop left over from a failed install has no owner but this."""
        from spacr.qt.widgets.ambient import AmbientWidget

        orphan = AmbientWidget(screen)
        qtbot.addWidget(orphan)
        monkeypatch.setattr(orphan, "set_animating", _boom)
        discarded = []
        monkeypatch.setattr("spacr.qt.screens.app_screen._discard_widget",
                            discarded.append)

        screen._discard_orphan_ambient()

        assert orphan in discarded


class TestRethemingTheBackdrops:

    def test_a_theme_that_cannot_be_resolved_leaves_the_backdrop_alone(
            self, screen, monkeypatch):
        """Its current fill is a better answer than a guessed one."""
        from spacr.qt import preferences

        pushed = []
        monkeypatch.setattr(screen, "_ambient", types.SimpleNamespace(
            set_background_color=pushed.append))
        monkeypatch.setattr(preferences, "resolve_effective_theme", _boom)

        screen._retheme_backdrops()

        assert pushed == []

    def test_a_backdrop_that_will_not_take_the_fill_does_not_stop_the_others(
            self, screen, monkeypatch):
        """Both backdrops are re-themed from one event."""
        taken = []
        monkeypatch.setattr(screen, "_dna_rain", types.SimpleNamespace(
            set_background_color=_boom, set_backdrop=_boom))
        monkeypatch.setattr(screen, "_ambient", types.SimpleNamespace(
            set_background_color=taken.append))
        screen._backdrop_applied = None

        screen._retheme_backdrops()

        assert len(taken) == 1


class TestThePagesOwnFill:

    def test_a_screen_that_animates_something_paints_no_fill(self, screen,
                                                              monkeypatch):
        """A flat fill would paint over the animation it sits behind."""
        monkeypatch.setattr(screen, "_ambient", object())

        assert screen.page_fill() is None

    def test_an_image_theme_paints_no_fill_either(self, screen, monkeypatch):
        """The window paints the wallpaper; the page shows it through."""
        from spacr.qt import preferences

        monkeypatch.setattr(screen, "_ambient", None)
        monkeypatch.setattr(screen, "_dna_rain", None)
        monkeypatch.setattr(preferences, "resolve_effective_theme",
                            lambda: "cell")

        assert screen.page_fill() is None

    def test_a_flat_theme_paints_its_page_colour(self, screen, monkeypatch):
        from spacr.qt import preferences

        monkeypatch.setattr(screen, "_ambient", None)
        monkeypatch.setattr(screen, "_dna_rain", None)
        monkeypatch.setattr(preferences, "resolve_effective_theme",
                            lambda: "dark")

        colour = screen.page_fill()

        assert colour is not None and colour.isValid()

    def test_a_theme_that_will_not_resolve_falls_back_to_no_fill(
            self, screen, monkeypatch):
        """The rendering the page had before this existed, not a slab."""
        from spacr.qt import preferences

        monkeypatch.setattr(screen, "_ambient", None)
        monkeypatch.setattr(screen, "_dna_rain", None)
        monkeypatch.setattr(preferences, "resolve_effective_theme", _boom)

        assert screen.page_fill() is None


class TestRethemingTheProseBoxes:

    def test_a_screen_with_no_prose_boxes_re_themes_nothing(self, screen,
                                                             monkeypatch):
        monkeypatch.setattr(screen, "_section_explainers", {}, raising=False)

        screen._retheme_section_explainers()

    def test_a_palette_that_will_not_resolve_leaves_every_box_as_it_was(
            self, screen, monkeypatch):
        from spacr.qt import theme

        monkeypatch.setattr(screen, "_section_explainers",
                            {"Paths": types.SimpleNamespace(setHtml=_boom)},
                            raising=False)
        monkeypatch.setattr(theme, "active_palette", _boom)

        screen._retheme_section_explainers()

    def test_a_box_that_was_removed_with_its_section_is_skipped(self, screen,
                                                                 monkeypatch):
        rendered = []
        monkeypatch.setattr(screen, "_section_explainers", {
            "Paths": None,
            "General": types.SimpleNamespace(
                setHtml=lambda html: rendered.append(html)),
        }, raising=False)

        screen._retheme_section_explainers()

        assert len(rendered) == 1

    def test_a_box_whose_widget_has_gone_does_not_stop_the_rest(self, screen,
                                                                monkeypatch):
        """A screen being torn down still gets the palette event."""
        rendered = []
        monkeypatch.setattr(screen, "_section_explainers", {
            "Paths": types.SimpleNamespace(setHtml=_boom),
            "General": types.SimpleNamespace(
                setHtml=lambda html: rendered.append(html)),
        }, raising=False)

        screen._retheme_section_explainers()

        assert len(rendered) == 1

    def test_a_screen_with_no_model_box_renders_nothing_for_it(self, screen,
                                                               monkeypatch):
        monkeypatch.setattr(screen, "_model_explainer", None, raising=False)

        screen._refresh_model_explainer()

    def test_a_setting_that_will_not_be_read_falls_back_to_its_default(
            self, screen, monkeypatch):
        """The formula still has to render for the model that IS on the panel."""
        model = screen._settings_model
        monkeypatch.setattr(model, "_read_widget", _boom)

        screen._refresh_model_explainer()

        assert screen._model_explainer.toHtml()


def test_opening_a_sub_heading_opens_the_umbrella_above_it(qtbot):
    """A heading opened inside a collapsed group has to be reachable."""
    from spacr.qt.widgets.section import Section

    parent = Section("Advanced settings")
    qtbot.addWidget(parent)
    parent.set_expanded(False)

    AppScreen._open_the_headings_above(parent, True)

    assert parent.is_expanded() is True


def test_collapsing_a_sub_heading_leaves_the_umbrella_open(qtbot):
    """The user still has the rest of the group in front of them."""
    from spacr.qt.widgets.section import Section

    parent = Section("Advanced settings")
    qtbot.addWidget(parent)
    parent.set_expanded(True)

    AppScreen._open_the_headings_above(parent, False)

    assert parent.is_expanded() is True


def test_a_parent_heading_that_has_gone_is_not_an_error():
    """The signal outlives the section on a screen being torn down."""
    AppScreen._open_the_headings_above(object(), True)
    AppScreen._open_the_headings_above(
        types.SimpleNamespace(is_expanded=_boom), True)


class TestBuildingTheScreenWhenAnEnrolmentFails:

    def test_a_workspace_registry_that_refuses_still_gives_a_screen(
            self, qtbot, monkeypatch):
        """The screen is the module; contributing to a saved run is extra."""
        monkeypatch.setattr(AppScreen, "register_workspace", _boom)

        built = AppScreen("regression")
        qtbot.addWidget(built)

        assert built._settings_model is not None

    def test_a_theme_helper_that_refuses_still_gives_a_screen(self, qtbot,
                                                              monkeypatch):
        from spacr.qt import theme

        monkeypatch.setattr(theme, "take_the_scroll_arrows_off", _boom)

        built = AppScreen("regression")
        qtbot.addWidget(built)

        assert built._settings_model is not None


def test_a_screen_that_opts_into_tabs_mounts_each_category_as_a_page(
        qtbot, monkeypatch):
    """The opt-in exists because a stacked column of categories reads as a wall.

    Every category still knows its own maturity, hint and rows -- only where
    it is MOUNTED changes -- and a tab is already the disclosure, so the
    section inside it is left expanded rather than carrying a second one.
    """
    from PySide6.QtWidgets import QTabWidget

    monkeypatch.setattr(AppScreen, "SETTINGS_AS_TABS", frozenset({"measure"}))
    built = AppScreen("measure")
    qtbot.addWidget(built)

    tabs = built._settings_tabs
    assert isinstance(tabs, QTabWidget)
    assert tabs.count() > 2
    assert all(tabs.tabText(index) for index in range(tabs.count()))
    assert len(built._settings_sections) >= tabs.count()
