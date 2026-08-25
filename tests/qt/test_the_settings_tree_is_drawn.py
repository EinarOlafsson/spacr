"""The settings panel draws the whole tree, not just its outer level.

``SettingsWidgets.build_sections`` returns a ``SettingsSection``: still the
``(title, rows)`` pair every caller unpacks, with ``own_rows`` for the rows a
heading owns itself and ``children`` for the headings nested under it. A
screen that reads only the pair draws every control exactly once but FLAT --
"Advanced settings" renders as a single heading holding a hundred and
twenty-eight rows, and the sub-headings that say which object each row
belongs to are nowhere on screen.

What is pinned here:

* the umbrella and its families are drawn as nested ``Section`` widgets, one
  per entry of ``children``, all the way down to the per-object level;
* no control is drawn twice and none is lost -- the flat reading and the tree
  reading render exactly the same set of setting keys;
* a sub-heading is described by its PATH: "Cell" under "Object filtration" is
  not the "Cell" segmentation category, and a title-only lookup hands it the
  blurb about Cellpose models;
* opening a sub-heading from outside -- the search strip, the command palette
  -- opens the umbrella above it, because an expanded section inside a
  collapsed one shows the user nothing.
"""
from __future__ import annotations

import pytest

from spacr.qt.screens.app_screen import AppScreen
from spacr.qt.screens.settings_model import SettingsWidgets, section_tooltip
from spacr.qt.widgets.section import Section


def _screen(qtbot, app_key: str = "mask") -> AppScreen:
    screen = AppScreen(app_key)
    qtbot.addWidget(screen)
    return screen


def _sections_by_source(screen: AppScreen) -> dict:
    return {
        str(section.property("settingsCategorySource")): section
        for section in screen._settings_sections
    }


def _nested_sections(section: Section) -> list:
    """The ``Section`` widgets drawn directly inside ``section``."""
    return [
        child for child in section.findChildren(Section)
        if child.parentWidget() is not None
        and _closest_section(child.parentWidget()) is section
    ]


def _closest_section(widget):
    while widget is not None:
        if isinstance(widget, Section):
            return widget
        widget = widget.parentWidget()
    return None


def _model_tree(app_key: str = "mask"):
    return SettingsWidgets(app_key).build_sections()


def test_the_umbrella_draws_a_section_for_every_family_and_object(qtbot):
    """Every ``children`` entry of the model becomes a heading on screen."""
    screen = _screen(qtbot)
    drawn = _sections_by_source(screen)
    tree = {spec.title: spec for spec in _model_tree()}
    umbrella_spec = tree["Advanced settings"]
    assert umbrella_spec.children, (
        "the model no longer nests anything under Advanced settings")

    umbrella = drawn["Advanced settings"]
    families = {section.property("settingsCategorySource"): section
                for section in _nested_sections(umbrella)}
    assert set(families) == {child.title for child in umbrella_spec.children}

    for child_spec in umbrella_spec.children:
        family = families[child_spec.title]
        objects = {section.property("settingsCategorySource"): section
                   for section in _nested_sections(family)}
        assert set(objects) == {
            grandchild.title for grandchild in child_spec.children}, (
            f"{child_spec.title} lost its per-object headings")


def test_a_family_holds_its_rows_under_the_object_they_belong_to(qtbot):
    """The rows sit in the sub-heading that owns them, not in the umbrella."""
    screen = _screen(qtbot)
    drawn = _sections_by_source(screen)
    umbrella = drawn["Advanced settings"]

    assert umbrella._row_widgets == [], (
        "the umbrella still holds rows itself; the tree is being flattened")
    for family in _nested_sections(umbrella):
        assert family._row_widgets == [], (
            f"{family.title()} holds rows a sub-heading owns")
        for objects in _nested_sections(family):
            assert objects._row_widgets, (
                f"{objects.title()} was drawn with nothing in it")


def test_drawing_the_tree_renders_every_key_exactly_once(qtbot):
    """A control the user cannot reach, or reaches twice, is the failure."""
    screen = _screen(qtbot)
    keys = []
    for section in screen._settings_sections:
        for _label, field in section._row_widgets:
            keys.append(field.property("settingKey"))
    assert all(keys), "a rendered row carries no setting key"
    assert len(keys) == len(set(keys)), "a setting was drawn under two headings"

    expected = []
    for spec in _model_tree():
        expected.extend(label for label, _widget in spec.rows)
    assert len(keys) == len(expected), (
        "the tree renders a different number of rows than the flat reading")


def test_a_sub_heading_is_described_by_its_path_not_by_its_word(qtbot):
    """"Cell" under a family is not the Cell segmentation category."""
    screen = _screen(qtbot)
    drawn = _sections_by_source(screen)
    umbrella = drawn["Advanced settings"]
    family = _nested_sections(umbrella)[0]
    per_object = {section.property("settingsCategorySource"): section
                  for section in _nested_sections(family)}
    cell = per_object["Cell"]

    spec = next(
        grandchild
        for child in next(spec for spec in _model_tree()
                          if spec.title == "Advanced settings").children
        if child.title == family.property("settingsCategorySource")
        for grandchild in child.children
        if grandchild.title == "Cell"
    )
    expected = section_tooltip(screen.app_key, spec)
    assert expected in cell.header().toolTip()
    assert "Cellpose" not in cell.header().toolTip()

    screen.show_category_hint("Cell")
    assert expected in screen._category_hint.text().replace("&#x27;", "'")


def test_opening_a_sub_heading_opens_the_umbrella_above_it(qtbot):
    """An expanded section inside a collapsed one is a section nobody sees."""
    screen = _screen(qtbot)
    drawn = _sections_by_source(screen)
    umbrella = drawn["Advanced settings"]
    family = _nested_sections(umbrella)[0]
    per_object = _nested_sections(family)[0]

    assert not umbrella.is_expanded()
    per_object.set_expanded(True)
    assert family.is_expanded(), "the family stayed shut over an open heading"
    assert umbrella.is_expanded(), "the umbrella stayed shut over an open one"

    # Closing the sub-heading leaves the group the user was reading open.
    per_object.set_expanded(False)
    assert umbrella.is_expanded()


def test_the_innermost_heading_is_the_one_a_consumer_finds(qtbot):
    """`_settings_sections` is deepest-first, so `isAncestorOf` lands right.

    The command palette reveals a setting by scanning that list for the
    first section that is an ancestor of the widget and expanding it. The
    umbrella is an ancestor of every advanced control, so a list that put it
    before its own sub-headings would expand it and stop, leaving the
    setting inside a heading that is still shut.
    """
    screen = _screen(qtbot)
    drawn = _sections_by_source(screen)
    umbrella = drawn["Advanced settings"]
    family = _nested_sections(umbrella)[0]
    per_object = _nested_sections(family)[0]
    _label, field = per_object._row_widgets[0]

    found = next(section for section in screen._settings_sections
                 if section.isAncestorOf(field))
    assert found is per_object


@pytest.mark.parametrize(
    "code", ["sv", "de", "es", "zh_CN", "pt", "hi", "ko", "is", "fr"])
def test_every_heading_of_the_tree_has_an_exact_translation(code):
    """A heading the panel now draws must not be half-translated.

    Drawing the tree put three headings on screen that nothing rendered
    before, and each is a phrase the word-by-word fallback only partly
    recognises: "Object Filtration (all objects)" came out as "Objekt
    Filtration (all Objekt)". An exact catalog row is the only thing that
    beats that fallback, so each heading is asserted to have one rather
    than merely to change.
    """
    from spacr.qt.i18n import _exact_translation

    for spec in _model_tree():
        for section in spec.walk():
            if section is spec and not section.children:
                continue    # a leaf top-level category is not new here
            assert _exact_translation(section.title, code) is not None, (
                f"{section.title!r} has no exact row in {code}; the "
                "word-by-word fallback will half-translate it")


@pytest.mark.parametrize("app_key", ["measure", "regression", "umap"])
def test_a_module_with_no_nesting_is_drawn_exactly_as_before(qtbot, app_key):
    """The tree is additive: a flat module gains no headings from it."""
    screen = _screen(qtbot, app_key)
    titles = [str(section.property("settingsCategorySource"))
              for section in screen._settings_sections]
    assert titles == [spec.title for spec in _model_tree(app_key)]
    for section in screen._settings_sections:
        assert not _nested_sections(section)
