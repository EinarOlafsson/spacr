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
from PySide6.QtWidgets import QWidget

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


def _model_tree(qtbot, app_key: str = "mask"):
    owner = QWidget()
    qtbot.addWidget(owner)
    return SettingsWidgets(app_key, parent=owner).build_sections()


def test_the_umbrella_draws_a_section_for_every_family_and_object(qtbot):
    """Every current model child becomes a heading in its declared parent."""
    screen = _screen(qtbot, "measure")
    screen._open_every_waiting_heading()
    drawn = _sections_by_source(screen)
    tree = {spec.title: spec for spec in _model_tree(qtbot, "measure")}
    umbrella_spec = tree["Image Preprocessing"]
    assert umbrella_spec.children, "the current correction tree must be nested"
    umbrella = drawn[umbrella_spec.title]
    families = {section.property("settingsCategorySource"): section
                for section in _nested_sections(umbrella)}
    assert set(families) == {child.title for child in umbrella_spec.children}
    assert [section.property("settingsCategorySource")
            for section in _nested_sections(umbrella)] == [
                child.title for child in umbrella_spec.children]
    for child_spec in umbrella_spec.children:
        family = families[child_spec.title]
        objects = {section.property("settingsCategorySource"): section
                   for section in _nested_sections(family)}
        assert set(objects) == {child.title for child in child_spec.children}, (
            f"{child_spec.title} lost its nested headings")
        assert {field.property("settingKey") for _label, field in family._row_widgets} == {
            field.property("settingKey") for _label, field in child_spec.own_rows}


def test_a_family_holds_its_rows_under_the_object_they_belong_to(qtbot):
    """Every row belongs to exactly the current model's declared heading."""
    screen = _screen(qtbot, "measure")
    screen._open_every_waiting_heading()
    drawn = _sections_by_source(screen)
    umbrella_spec = next(spec for spec in _model_tree(qtbot, "measure")
                         if spec.title == "Input & Experiment")
    assert umbrella_spec.own_rows and umbrella_spec.children
    umbrella = drawn[umbrella_spec.title]
    expected_own = {field.property("settingKey")
                    for _label, field in umbrella_spec.own_rows}
    assert {field.property("settingKey") for _label, field in umbrella._row_widgets} == expected_own
    nested_keys = set()
    for child_spec in umbrella_spec.children:
        family = drawn[child_spec.title]
        own = {field.property("settingKey") for _label, field in family._row_widgets}
        expected = {field.property("settingKey") for _label, field in child_spec.own_rows}
        assert own == expected and own, f"{child_spec.title} lost its own rows"
        assert not own & expected_own, "the parent still holds a child's rows"
        assert not own & nested_keys, "a row was drawn under two children"
        nested_keys.update(own)
        assert _closest_section(family.parentWidget()) is umbrella


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
    for spec in _model_tree(qtbot):
        expected.extend(label for label, _widget in spec.rows)
    assert len(keys) == len(expected), (
        "the tree renders a different number of rows than the flat reading")


def test_a_sub_heading_is_described_by_its_path_not_by_its_word(qtbot):
    """A current nested heading receives its model path's exact guidance."""
    screen = _screen(qtbot, "measure")
    screen._open_every_waiting_heading()
    drawn = _sections_by_source(screen)
    umbrella_spec = next(spec for spec in _model_tree(qtbot, "measure")
                         if spec.title == "Input & Experiment")
    spec = next(child for child in umbrella_spec.children if child.title == "Cloud α")
    child = drawn[spec.title]
    assert _closest_section(child.parentWidget()) is drawn[umbrella_spec.title]
    assert spec.path == (umbrella_spec.title, spec.title)
    expected = section_tooltip(screen.app_key, spec)
    assert expected and expected in child.header().toolTip()
    assert "Cellpose" not in child.header().toolTip()
    screen.show_category_hint(spec.title)
    assert expected in screen._category_hint.text().replace("&#x27;", "'")


def test_opening_a_sub_heading_opens_the_umbrella_above_it(qtbot):
    """An expanded child opens its real parent without flattening the tree."""
    screen = _screen(qtbot, "measure")
    screen._open_every_waiting_heading()
    drawn = _sections_by_source(screen)
    umbrella = drawn["Input & Experiment"]
    family = drawn["Cloud α"]
    assert _closest_section(family.parentWidget()) is umbrella
    family.set_expanded(False)
    umbrella.set_expanded(False)
    assert not umbrella.is_expanded()
    family.set_expanded(True)
    assert family.is_expanded(), "the requested child stayed shut"
    assert umbrella.is_expanded(), "the umbrella stayed shut over an open child"
    family.set_expanded(False)
    assert umbrella.is_expanded()


def test_the_innermost_heading_is_the_one_a_consumer_finds(qtbot):
    """The real nested heading precedes its ancestors in consumer lookup."""
    screen = _screen(qtbot, "measure")
    screen._open_every_waiting_heading()
    drawn = _sections_by_source(screen)
    umbrella = drawn["Input & Experiment"]
    per_object = drawn["Cloud α"]
    assert _closest_section(per_object.parentWidget()) is umbrella
    assert per_object._row_widgets
    for _label, field in per_object._row_widgets:
        found = next(section for section in screen._settings_sections
                     if section.isAncestorOf(field))
        assert found is per_object
        assert screen._settings_sections.index(found) < screen._settings_sections.index(umbrella)


@pytest.mark.parametrize(
    "code", ["sv", "de", "es", "zh_CN", "pt", "hi", "ko", "is", "fr"])
def test_every_heading_of_the_tree_has_an_exact_translation(code, qtbot):
    """A heading the panel now draws must not be half-translated.

    Drawing the tree put three headings on screen that nothing rendered
    before, and each is a phrase the word-by-word fallback only partly
    recognises: "Object Filtration (all objects)" came out as "Objekt
    Filtration (all Objekt)". An exact catalog row is the only thing that
    beats that fallback, so each heading is asserted to have one rather
    than merely to change.
    """
    from spacr.qt.i18n import _exact_translation

    for spec in _model_tree(qtbot):
        for section in spec.walk():
            if section is spec and not section.children:
                continue    # a leaf top-level category is not new here
            assert _exact_translation(section.title, code) is not None, (
                f"{section.title!r} has no exact row in {code}; the "
                "word-by-word fallback will half-translate it")


# 2026-09-29 (item 595): Measure is no longer flat -- its settings were
# regrouped into Input & Experiment, Image Preprocessing, Features and
# Postprocessing with sub-headings, as the maintainer asked -- so it left this
# list; tests/qt/test_measure_settings_regrouped.py holds its tree.
@pytest.mark.parametrize("app_key", ["regression", "umap"])
def test_a_module_with_no_nesting_is_drawn_exactly_as_before(qtbot, app_key):
    """The tree is additive: a flat module gains no headings from it."""
    screen = _screen(qtbot, app_key)
    titles = [str(section.property("settingsCategorySource"))
              for section in screen._settings_sections]
    assert titles == [spec.title for spec in _model_tree(qtbot, app_key)]
    for section in screen._settings_sections:
        assert not _nested_sections(section)
