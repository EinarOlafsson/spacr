"""A closed settings category builds its rows, and most of its controls, when opened.

A module screen opens with every settings category closed but one. Counted on
2026-09-22 in a real window, 91 of Classify's 101 settings, 101 of Mask's 132
and 61 of Measure's 63 sat in categories nobody could see, and building their
controls, captions and rows was most of what the settings panel cost at open.
`AppScreen._build_a_waiting_heading` now builds such a category's heading
only; its rows are laid out, and the controls nothing has read are built, the
first time it is opened.

The controls ARE the settings model -- `collect()` reads them, recipes and
imports write them -- so the half these tests hold hardest is that nothing can
tell the difference except by being faster:

  * a plain control that was never built is read from its plan, and that plan
    reads back exactly what the built control would, for every setting of
    every module (the exhaustive test below);
  * `collect()` on a screen that has built almost nothing equals `collect()`
    after every category is opened, and equals a window that built
    everything at once -- and so do the rows, captions, visibility, enabled
    state and help of every category once opened;
  * writing a value builds that one control, and the value is collected and
    shown when the category opens;
  * the search strip and the command palette find and reveal a setting whose
    category has not been built;
  * a category built after a switch to Swedish is in Swedish, and styled by
    the page's sheet;
  * a state rule (the classifier family greying the other family's
    settings) reaches controls built after the rule last ran;
  * closing a screen builds nothing.
"""
from __future__ import annotations

import os
from contextlib import contextmanager

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

pytestmark = pytest.mark.qt

from PySide6.QtWidgets import (QApplication, QFormLayout, QLabel,  # noqa: E402
                               QWidget)


def _pump(rounds: int = 20) -> None:
    app = QApplication.instance()
    for _ in range(rounds):
        app.processEvents()


@contextmanager
def _language(code):
    from spacr.qt import i18n

    before = os.environ.get(i18n.ENV_LANGUAGE)
    os.environ[i18n.ENV_LANGUAGE] = code
    try:
        yield
    finally:
        if before is None:
            os.environ.pop(i18n.ENV_LANGUAGE, None)
        else:
            os.environ[i18n.ENV_LANGUAGE] = before


def _same(a, b) -> bool:
    from spacr.qt.settings_diff import _values_equal

    try:
        return bool(_values_equal(a, b)) and type(a) is type(b)
    except Exception:                                       # noqa: BLE001
        return a == b


def _differing(one: dict, other: dict) -> list:
    return sorted(key for key in set(one) | set(other)
                  if key not in one or key not in other
                  or not _same(one[key], other[key]))


# -- the model on its own -----------------------------------------------------

def test_every_plain_control_reads_back_what_its_plan_says(qapp):
    """The invariant that lets a control be read without being built."""
    from spacr.qt import register_self_registering_modules
    from spacr.qt.screens import settings_model as sm
    from spacr.qt.screens.app_screen import APP_TITLES
    from spacr.settings_spec import convert_settings_dict_for_gui

    register_self_registering_modules()
    checked, wrong = 0, []
    for app_key in sorted(APP_TITLES):
        try:
            model = sm.SettingsWidgets(app_key)
            variables = convert_settings_dict_for_gui(model._defaults)
        except Exception:                                   # noqa: BLE001
            continue
        for key, (kind, options, default) in variables.items():
            route, plan = model._route_control(kind, options, default, key)
            if route != "plain":
                continue
            checked += 1
            control = model._build_plain(plan)
            built = model._read_widget(control)
            unbuilt = sm._value_a_plain_control_holds(plan)
            if not (built == unbuilt and type(built) is type(unbuilt)):
                wrong.append((app_key, key, built, unbuilt))
            control.deleteLater()
    assert checked > 500
    assert not wrong, wrong[:10]


def test_a_model_built_for_its_values_builds_every_control(qapp):
    """Only a screen that asks for waiting gets it; everything else is eager."""
    from spacr.qt.screens.settings_model import SettingsWidgets

    model = SettingsWidgets("classify_merged")
    model.build_sections()
    assert not model._widgets.keys_to_come()
    assert len(model._widgets.built_items()) == len(model._widgets)


def test_the_map_answers_which_settings_exist_without_building(qapp):
    from spacr.qt.screens.settings_model import SettingsWidgets

    model = SettingsWidgets("measure")
    model.categories_may_wait = lambda title, keys: True
    model.build_sections()
    widgets = model._widgets
    waiting = widgets.keys_to_come()
    assert waiting, "nothing waited"
    key = waiting[0]
    assert key in widgets and key in list(widgets)
    assert len(widgets) == len(list(widgets))
    assert not widgets.is_built(key)
    assert widgets.built(key) is None
    control = widgets[key]
    assert widgets.is_built(key) and widgets[key] is control
    rest = widgets.keys_to_come()
    assert rest
    dict(widgets.items())
    assert not widgets.keys_to_come()


def test_collect_reads_waiting_controls_without_building_them(qapp):
    from spacr.qt.screens.settings_model import SettingsWidgets

    lazy = SettingsWidgets("mask")
    lazy.categories_may_wait = lambda title, keys: True
    lazy.build_sections()
    waiting = len(lazy._widgets.keys_to_come())
    assert waiting > 50
    values = lazy.collect()
    assert len(lazy._widgets.keys_to_come()) == waiting
    eager = SettingsWidgets("mask")
    eager.build_sections()
    assert _differing(values, eager.collect()) == []
    assert lazy.modified_keys() == eager.modified_keys()
    assert len(lazy._widgets.keys_to_come()) == waiting


def test_writing_a_waiting_setting_builds_that_control_alone(qapp):
    from spacr.qt.screens.settings_model import SettingsWidgets

    model = SettingsWidgets("mask")
    model.categories_may_wait = lambda title, keys: True
    model.build_sections()
    key = next(key for key in model._widgets.keys_to_come()
               if model._widgets.meta_for(key)["control"] == "int")
    before = len(model._widgets.keys_to_come())
    target = model._widgets.meta_for(key)["range"][1]
    assert model.set_value_for_key(key, target)
    assert model._widgets.is_built(key)
    assert len(model._widgets.keys_to_come()) == before - 1
    assert model.collect()[key] == target


# -- on a real window ---------------------------------------------------------

def _window(qtbot, key, *, eager: bool = False):
    from spacr.qt import register_self_registering_modules
    from spacr.qt.app import MainWindow
    from spacr.qt.screens.app_screen import AppScreen

    register_self_registering_modules()
    window = MainWindow()
    qtbot.addWidget(window)
    window.resize(1400, 900)
    window.show()
    _pump(40)
    if eager:
        original = AppScreen._a_category_may_wait
        AppScreen._a_category_may_wait = lambda self, title, keys=(): False
        try:
            window._on_nav_selected(key)
        finally:
            AppScreen._a_category_may_wait = original
    else:
        window._on_nav_selected(key)
    _pump(60)
    return window, window._screens[key]


def _waiting(screen) -> list:
    return [section for section in screen._settings_sections
            if screen._heading_is_waiting(section)]


def _rows(screen) -> dict:
    """Every category's rows as a user reads them, keyed by title."""
    model = screen._settings_model
    by_widget = {id(widget): key for key, widget
                 in model._widgets.built_items()}
    found = {}
    for section in screen._settings_sections:
        form = getattr(section, "_form", None)
        if not isinstance(form, QFormLayout):
            continue
        rows = []
        for index in range(form.rowCount()):
            field_item = form.itemAt(index, QFormLayout.FieldRole)
            label_item = form.itemAt(index, QFormLayout.LabelRole)
            field = field_item.widget() if field_item else None
            key = by_widget.get(id(field)) if field is not None else None
            if key is None and field is not None:
                key = next((by_widget[id(child)]
                            for child in field.findChildren(QWidget)
                            if id(child) in by_widget), None)
            label = label_item.widget() if label_item else None
            caption = " ".join(each.text()
                               for each in label.findChildren(QLabel)) \
                if label is not None else ""
            control = model._widgets.built(key) if key else None
            rows.append((
                key, caption, form.isRowVisible(index),
                control.isEnabled() if control is not None else None,
                control.toolTip() if control is not None else "",
                type(field).__name__ if field is not None else None,
                control.sizeHint().height() if control is not None else None,
                control.palette().color(control.foregroundRole()).name()
                if control is not None else "",
            ))
        found[(section.property("settingsCategorySource"),
               section.isHidden())] = tuple(rows)
    return found


def test_opening_a_module_leaves_its_closed_categories_unbuilt(qtbot, monkeypatch):
    from spacr.qt.screens.app_screen import _IdlePrebuild

    # Opening must defer the rows; a later idle slice is tested separately.
    # Coverage can make the event pumping below exceed the real idle delay.
    monkeypatch.setattr(_IdlePrebuild, "IDLE_MS", 10 ** 8)
    _window_, screen = _window(qtbot, "classify_merged")
    model = screen._settings_model
    assert len(_waiting(screen)) >= 5
    assert len(model._widgets.keys_to_come()) > len(model._widgets) // 2
    for section in _waiting(screen):
        assert section._form.rowCount() == 0


def test_the_run_is_given_exactly_what_a_window_that_built_everything_gives(
        qtbot):
    for app_key in ("classify_merged", "mask"):
        _lazy_window, lazy = _window(qtbot, app_key)
        waiting = len(lazy._settings_model._widgets.keys_to_come())
        values = lazy._settings_model.collect()
        assert len(lazy._settings_model._widgets.keys_to_come()) == waiting
        _eager_window, eager = _window(qtbot, app_key, eager=True)
        assert not _waiting(eager)
        assert _differing(values, eager._settings_model.collect()) == []
        lazy._open_every_waiting_heading()
        _pump(10)
        assert _differing(values, lazy._settings_model.collect()) == []
        assert _rows(lazy) == _rows(eager), app_key


def test_a_value_written_before_its_category_opens_is_there_when_it_does(
        qtbot):
    _window_, screen = _window(qtbot, "classify_merged")
    model = screen._settings_model
    key = next(key for key in model._widgets.keys_to_come()
               if model._widgets.meta_for(key)["control"] == "int")
    heading = screen._waiting_heading_of[key]
    target = model._widgets.meta_for(key)["range"][1]
    assert model.set_value_for_key(key, target)
    assert screen._heading_is_waiting(heading), "writing built the category"
    assert model.collect()[key] == target
    heading.set_expanded(True)
    _pump(10)
    control = model._widgets[key]
    assert heading._holds(control) and control.value() == target


def test_searching_finds_and_reveals_a_setting_in_a_waiting_category(qtbot):
    _window_, screen = _window(qtbot, "classify_merged")
    bar = screen._settings_search
    key = next(key for key in screen._waiting_heading_of
               if key in screen._settings_model._widgets)
    heading = screen._waiting_heading_of[key]
    assert key in bar.indexed_keys()
    assert bar.section_of(key) is heading
    assert bar.reveal(key)
    _pump(10)
    assert not screen._heading_is_waiting(heading)
    control = screen._settings_model._widgets[key]
    assert heading._holds(control) and control.isVisible()


def test_the_palette_opens_the_waiting_category_it_reveals(qtbot):
    from spacr.qt import command_palette as CP

    window, screen = _window(qtbot, "measure")
    key = next(iter(screen._waiting_heading_of))
    heading = screen._waiting_heading_of[key]
    palette = CP.CommandPalette(window)
    qtbot.addWidget(palette)
    palette._reveal_setting(key)
    _pump(10)
    assert not screen._heading_is_waiting(heading)
    assert heading.is_expanded()
    assert heading._holds(screen._settings_model._widgets[key])


def test_a_category_opened_after_a_switch_to_swedish_is_in_swedish(qtbot):
    from spacr.qt.i18n import retranslate_widget_tree, tr

    window, screen = _window(qtbot, "mask")
    heading = None
    for section in _waiting(screen):
        keys = [key for key, owner in screen._waiting_heading_of.items()
                if owner is section]
        captions = [screen._settings_model._label_for(key) for key in keys]
        if any(tr(caption, "sv") != caption for caption in captions):
            heading = section
            break
    assert heading is not None, "no waiting category has a Swedish caption"
    with _language("sv"):
        retranslate_widget_tree(window, "sv")
        heading.set_expanded(True)
        _pump(10)
        captions = [label.text() for label in heading.findChildren(QLabel)
                    if label.text()]
        english = [text for text in captions if tr(text, "sv") != text]
        assert not english, f"opened in English: {english[:5]}"


def test_a_state_rule_reaches_controls_built_after_it_ran(qtbot):
    """The classifier family greys the other family's settings, built or not."""
    from spacr.classify import FAMILY_SETTINGS

    _window_, screen = _window(qtbot, "classify_merged")
    model = screen._settings_model
    owned = {key for keys in FAMILY_SETTINGS.values() for key in keys}
    waiting = [key for key in model._widgets.keys_to_come() if key in owned]
    assert waiting, "no family-owned setting waited"
    family = model._widgets["classifier_family"]
    other = next(index for index in range(family.count())
                 if index != family.currentIndex())
    family.setCurrentIndex(other)
    _pump(5)
    screen._open_every_waiting_heading()
    _pump(5)
    eager_states = {}
    from spacr.qt.screens.settings_model import SettingsWidgets

    eager = SettingsWidgets("classify_merged")
    eager.build_sections()
    eager._widgets["classifier_family"].setCurrentIndex(other)
    for key in waiting:
        eager_states[key] = eager._widgets[key].isEnabled()
    states = {key: model._widgets[key].isEnabled() for key in waiting}
    assert states == eager_states
    assert not all(states.values()), "the family greyed nothing"


def test_the_dimension_switches_are_offered_for_rows_still_waiting(qtbot):
    _window_, screen = _window(qtbot, "mask")
    waiting = set(screen._waiting_heading_of)
    assert screen.dimension_switch("z") is not None
    assert screen.dimension_switch("t") is not None
    assert any(key in waiting for key in ("z_stack", "t_stack"))
    for section in screen.rendered_settings_sections():
        title = str(section.property("settingsCategorySource") or "")
        if title.startswith("Volumetric"):
            assert section.isHidden()


def test_closing_a_screen_builds_nothing(qtbot, monkeypatch):
    from spacr.qt.screens import settings_model as sm

    _window_, screen = _window(qtbot, "measure")
    built = []
    real = sm.SettingsWidgets._build_controls
    monkeypatch.setattr(sm.SettingsWidgets, "_build_controls",
                        lambda self, keys: built.append(list(keys))
                        or real(self, keys))
    screen._shutdown_settings_widgets()
    screen.close()
    _pump(5)
    assert built == []
