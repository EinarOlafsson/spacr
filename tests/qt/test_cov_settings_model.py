"""The defensive half of the settings panel, driven rather than assumed.

``spacr/qt/screens/settings_model.py`` is 5,469 lines and every module screen
is built out of it.  Measured over the 70 test files that import it, with the
marker expression CI runs, it sat at 92 % -- and the 137 statements nobody
reached were almost all ``except`` arms and "the widget is not what I expect"
guards: exactly the branches that decide whether opening a module shows a
panel or a traceback.

Three shapes recur and are worth naming, because each is a real state:

* A PLUGIN CONTRIBUTES AN APP.  Nine separate helpers ask
  ``spacr.plugins.get_app`` for the module's defaults, categories, tooltips,
  labels and docs URL, and every one of them wraps the call in
  ``except Exception`` because a third-party plugin can raise anything at all.
  No test had ever made one raise, and none had made ``get_app`` succeed
  either.
* A C++ OBJECT OUTLIVES ITS PYTHON HANDLE.  The API-dot installer and the
  tooltip refresher walk widget trees that a screen teardown may already have
  destroyed; ``RuntimeError`` there is the ordinary case, not the exotic one.
* A SETTING'S VALUE IS NOT THE TYPE THE WIDGET EXPECTS.  A settings CSV is
  hand-editable, so every reader has a fallback, and a fallback nobody has run
  is a guess.

Offscreen Qt, no network, no modal dialogs.
"""
from __future__ import annotations

import csv
import os

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QEvent, QSize, Qt                     # noqa: E402
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox,  # noqa: E402
                               QHBoxLayout, QLabel, QLineEdit, QSpinBox,
                               QWidget)

from spacr.qt.screens import settings_model as SM               # noqa: E402


# ---------------------------------------------------------------------------
# a plugin-contributed app
# ---------------------------------------------------------------------------

class _FakeApp:
    """The subset of ``plugins.AppContribution`` this module reads."""

    def __init__(self, defaults="tests.qt.test_cov_settings_model:plugin_defaults",
                 categories=None, tooltips=None, labels=None, docs_url=""):
        self.defaults = defaults
        self.categories = categories or {}
        self.tooltips = tooltips or {}
        self.labels = labels or {}
        self.docs_url = docs_url


def plugin_defaults(_settings=None):
    """Module-level so ``plugins.load_object`` can import it by reference."""
    return {"src": "", "alpha": 1, "beta": "two"}


def _install_plugin(monkeypatch, app):
    import spacr.plugins as plugins
    monkeypatch.setattr(plugins, "get_app", lambda key: app)


def _break_plugin(monkeypatch, exc=RuntimeError("plugin registry is on fire")):
    import spacr.plugins as plugins

    def boom(_key):
        raise exc

    monkeypatch.setattr(plugins, "get_app", boom)


# ---------------------------------------------------------------------------
# resolve_default_settings and its neighbours
# ---------------------------------------------------------------------------

def test_a_plugin_registry_that_raises_is_not_a_broken_module(monkeypatch):
    """A third-party plugin must not be able to stop a built-in from opening."""
    _break_plugin(monkeypatch)

    settings = SM.resolve_default_settings("measure")

    assert isinstance(settings, dict) and settings


def test_a_plugin_whose_defaults_take_no_argument_is_still_called(monkeypatch):
    """``defaults({})`` first, then ``defaults()`` -- both spellings exist."""
    import spacr.plugins as plugins

    calls = []

    def no_argument_defaults():
        calls.append("()")
        return {"src": "", "gamma": 3}

    _install_plugin(monkeypatch, _FakeApp(defaults="ref"))
    monkeypatch.setattr(plugins, "load_object", lambda _ref: no_argument_defaults)

    assert SM.resolve_default_settings("anything") == {"src": "", "gamma": 3}
    assert calls == ["()"]


def test_a_plugin_whose_defaults_are_not_callable_is_refused(monkeypatch):
    """Naming the contribution beats a TypeError from three frames down."""
    import spacr.plugins as plugins

    _install_plugin(monkeypatch, _FakeApp(defaults="pkg:not_a_function"))
    monkeypatch.setattr(plugins, "load_object", lambda _ref: {"already": "a dict"})

    with pytest.raises(TypeError, match="not callable"):
        SM.resolve_default_settings("anything")


def test_a_plugin_whose_defaults_return_a_list_is_refused(monkeypatch):
    """The panel indexes it by key; a list would fail much later and vaguely."""
    import spacr.plugins as plugins

    _install_plugin(monkeypatch, _FakeApp(defaults="pkg:wrong"))
    monkeypatch.setattr(plugins, "load_object", lambda _ref: lambda *_a: ["a", "b"])

    with pytest.raises(TypeError, match="expected dict"):
        SM.resolve_default_settings("anything")


def test_a_module_whose_settings_module_will_not_import_is_only_logged(
        caplog, monkeypatch):
    """The registration seam imports the owning module; it may not be there.

    An optional dependency that will not import should cost that app its
    settings panel, not stop the window opening -- so the failure is logged
    with the app it belongs to and swallowed.
    """
    monkeypatch.setattr(
        SM, "_registered_app_metadata",
        lambda _key: {"defaults_module": "spacr._no_such_module_at_all"})

    with caplog.at_level("WARNING"):
        SM._import_registered_defaults_module("_probe_missing")

    assert any("owns the" in record.getMessage() for record in caplog.records)


def test_has_curated_layout_says_no_when_the_registry_raises(monkeypatch):
    """"I cannot tell" and "no" are the same answer for a layout question."""
    _break_plugin(monkeypatch)

    assert SM.has_curated_layout("a_key_no_builtin_uses") is False


def test_has_curated_layout_says_yes_for_a_plugin_with_categories(monkeypatch):
    _install_plugin(monkeypatch, _FakeApp(categories={"Main": ["alpha"]}))

    assert SM.has_curated_layout("a_key_no_builtin_uses") is True


def test_needs_curated_layout_says_no_when_defaults_will_not_resolve(monkeypatch):
    """An app with no resolvable settings has no panel to judge."""
    monkeypatch.setattr(SM, "resolve_default_settings",
                        lambda _key: (_ for _ in ()).throw(KeyError("nope")))

    assert SM.needs_curated_layout("whatever") is False


def test_categories_for_app_falls_back_when_the_registry_raises(monkeypatch):
    """A plugin that explodes must not empty a built-in module's tabs."""
    _break_plugin(monkeypatch)

    out = SM.categories_for_app("_no_builtin_uses_this", {"Main": ["src"]})

    assert out == {"Main": ["src"]}


def test_a_plugin_supplies_its_own_categories(monkeypatch):
    _install_plugin(monkeypatch, _FakeApp(categories={"Tab": ("alpha", "beta")}))

    assert SM.categories_for_app("x", {"Main": ["src"]}) == {
        "Tab": ["alpha", "beta"]}


def test_external_masks_moves_its_input_keys_out_of_every_other_tab(monkeypatch):
    """The relocation loop removes a key wherever it was, however often."""
    import spacr.plugins as plugins
    monkeypatch.setattr(plugins, "get_app", lambda _key: None)

    out = SM.categories_for_app(
        "external_masks", {"Advanced": ["dst", "recursive", "dst", "keep"]})

    assert list(out)[0] == "Input mapping"
    assert out["Advanced"] == ["keep"]


def test_api_docs_url_falls_back_when_the_registry_raises(monkeypatch):
    _break_plugin(monkeypatch)

    assert isinstance(SM.api_docs_url("measure"), str)


def test_a_plugin_docs_url_wins(monkeypatch):
    _install_plugin(monkeypatch, _FakeApp(docs_url="https://example.invalid/x"))

    assert SM.api_docs_url("x") == "https://example.invalid/x"


# ---------------------------------------------------------------------------
# tooltips and type hints
# ---------------------------------------------------------------------------

def test_tooltips_are_empty_rather_than_fatal_when_settings_will_not_import(
        monkeypatch):
    """``spacr.settings`` registers lazily; it can be mid-import here."""
    import builtins

    real_import = builtins.__import__

    def refuse(name, *args, **kwargs):
        if name == "spacr.settings":
            raise ImportError("settings is half built")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", refuse)

    assert SM.get_tooltips() == {}


def test_a_type_hint_for_no_key_is_empty():
    assert SM._type_hint("") == ""


def test_a_type_hint_is_empty_rather_than_fatal_when_settings_will_not_import(
        monkeypatch):
    import builtins

    real_import = builtins.__import__

    def refuse(name, *args, **kwargs):
        if name == "spacr.settings":
            raise ImportError("settings is half built")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", refuse)

    assert SM._type_hint("src") == ""


def test_a_blank_category_title_has_no_blurb():
    assert SM._category_blurb("measure", "") == ""


def test_an_optional_type_hint_keeps_its_optional_marker_when_translated():
    """The catalog is asked for atoms; the union and the marker are assembled.

    Enumerating every possible union in every language is what this avoids.
    """
    assert SM._type_hint("custom_regex") == "string (optional)"
    assert SM._translated_type_hint("custom_regex", "en") == "string (optional)"


def test_a_type_hint_survives_a_build_with_no_translation_catalogs(monkeypatch):
    """``i18n_catalogs`` is optional; its absence must not blank the hint."""
    import builtins

    real_import = builtins.__import__

    def refuse(name, globals=None, locals=None, fromlist=(), level=0):
        if "i18n_catalogs" in name or "i18n_catalogs" in tuple(fromlist or ()):
            raise ImportError("no catalogs in this build")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", refuse)

    assert SM._translated_body("Some help text", "de", setting_key="src") \
        == "Some help text"
    # a key the compact term catalog does not carry, so the optional
    # generated catalog is the only thing that could have translated it
    assert SM._translated_setting_name(
        "_probe_never_translated_key", "de", "measure") \
        == SM._humanize("_probe_never_translated_key")


# ---------------------------------------------------------------------------
# the API-dot tooltip filter and installer
# ---------------------------------------------------------------------------

def test_a_widget_with_no_api_tooltip_is_left_to_qt(qtbot):
    """The filter must not swallow events for widgets it knows nothing about."""
    widget = QWidget()
    qtbot.addWidget(widget)
    filt = SM._ApiTooltipFilter(widget)

    assert filt.eventFilter(widget, QEvent(QEvent.Enter)) is False


def test_the_native_tooltip_is_suppressed_where_the_clickable_one_shows(qtbot):
    """Both would appear; the native one vanishes when you reach for the link."""
    widget = QWidget()
    qtbot.addWidget(widget)
    widget.setProperty("apiTooltipHtml", "<b>docs</b>")
    filt = SM._ApiTooltipFilter(widget)

    assert filt.eventFilter(widget, QEvent(QEvent.ToolTip)) is True


def test_leaving_a_widget_starts_the_hover_tooltip_hiding(qtbot, monkeypatch):
    widget = QWidget()
    qtbot.addWidget(widget)
    widget.setProperty("apiTooltipHtml", "<b>docs</b>")
    filt = SM._ApiTooltipFilter(widget)

    from spacr.qt.widgets import hover_tooltip

    hidden = []
    monkeypatch.setattr(hover_tooltip.HoverTooltip, "instance",
                        classmethod(lambda cls: type("T", (), {
                            "start_hide": lambda self: hidden.append(True),
                            "show_for": lambda self, *a: None,
                        })()))

    assert filt.eventFilter(widget, QEvent(QEvent.Leave)) is False
    assert hidden == [True]


def test_refreshing_tooltips_on_nothing_is_a_no_op():
    assert SM.refresh_api_tooltips(None) is None


def test_refreshing_tooltips_survives_a_destroyed_tree(qtbot):
    """``findChildren`` on a deleted C++ object is the ordinary teardown race."""
    class _Gone:
        def findChildren(self, *_args, **_kwargs):
            raise RuntimeError("Internal C++ object already deleted")

    assert SM.refresh_api_tooltips(_Gone()) is None


def test_refreshing_tooltips_skips_a_child_that_died_mid_walk(qtbot):
    """One dead widget must not abandon the rest of the panel's help."""
    class _DeadChild:
        def property(self, _name):
            raise RuntimeError("Internal C++ object already deleted")

    class _Root:
        def findChildren(self, *_args, **_kwargs):
            return [_DeadChild()]

        def property(self, _name):
            return None

    assert SM.refresh_api_tooltips(_Root()) is None


def test_a_label_with_no_layout_gets_no_api_dot(qtbot):
    """A label the caller has not parented has nowhere to put the dot."""
    label = QLabel("Source")
    qtbot.addWidget(label)

    assert SM._add_api_dot_to_label(label, "measure", "src", "<b>x</b>") is None
    assert label.property("settingApiDotInstalled") in (None, False)


def test_a_control_with_no_layout_gets_no_api_dot(qtbot):
    field = QCheckBox("Verbose")
    qtbot.addWidget(field)

    assert SM._add_api_dot_to_combined_control(
        field, field, "measure", "verbose", "<b>x</b>") is None


def test_a_control_that_already_has_its_dot_in_this_window_is_left_alone(qtbot):
    host = QWidget()
    qtbot.addWidget(host)
    layout = QHBoxLayout(host)
    field = QCheckBox("Verbose", host)
    layout.addWidget(field)
    SM._add_api_dot_to_combined_control(host, field, "measure", "verbose",
                                        "<b>x</b>")
    first = getattr(field, "_spacr_api_dot", None)
    assert first is not None

    SM._add_api_dot_to_combined_control(host, field, "measure", "verbose",
                                        "<b>y</b>")

    assert getattr(field, "_spacr_api_dot", None) is first


def test_a_control_whose_previous_dot_was_destroyed_gets_a_new_one(qtbot):
    """The stale handle raises ``RuntimeError``; that is not a reason to stop."""
    import shiboken6

    host = QWidget()
    qtbot.addWidget(host)
    layout = QHBoxLayout(host)
    field = QCheckBox("Verbose", host)
    layout.addWidget(field)
    SM._add_api_dot_to_combined_control(host, field, "measure", "verbose",
                                        "<b>x</b>")
    dot = getattr(field, "_spacr_api_dot")
    shiboken6.delete(dot)

    SM._add_api_dot_to_combined_control(host, field, "measure", "verbose",
                                        "<b>y</b>")

    assert getattr(field, "_spacr_api_dot") is not dot


def test_a_label_wrapper_with_no_help_child_is_returned_as_it_is(qtbot):
    wrapper = QWidget()
    qtbot.addWidget(wrapper)
    wrapper.setObjectName("SettingLabelWithInfo")

    assert SM._unwrap_setting_label(wrapper) is wrapper


def test_a_remembered_label_that_was_destroyed_is_looked_up_again(qtbot):
    """The cached handle is the fast path; a dead one must fall through."""
    import shiboken6

    owner = QWidget()
    qtbot.addWidget(owner)
    field = QLineEdit(owner)
    dead = QLabel(owner)
    field.setProperty("settingLabelWidget", dead)
    shiboken6.delete(dead)

    assert SM._setting_label_for_field(owner, field) is None


# ---------------------------------------------------------------------------
# the flow layout behind the chip strips
# ---------------------------------------------------------------------------

def test_taking_an_item_that_is_not_there_returns_nothing(qtbot):
    host = QWidget()
    qtbot.addWidget(host)
    layout = SM._FlowLayout(host)

    assert layout.takeAt(7) is None
    assert layout.hasHeightForWidth() is True
    assert isinstance(layout.sizeHint(), QSize)


def test_a_flow_host_with_no_layout_answers_from_the_base_class(qtbot):
    host = SM._FlowHost()
    qtbot.addWidget(host)

    assert isinstance(host.heightForWidth(120), int)
    assert isinstance(host.sizeHint(), QSize)


def test_a_flow_host_with_a_layout_asks_the_layout(qtbot):
    host = SM._FlowHost()
    qtbot.addWidget(host)
    layout = SM._FlowLayout(host)
    layout.addWidget(QLabel("chip", host))

    assert host.heightForWidth(200) == layout.heightForWidth(200)


def test_committing_an_empty_chip_entry_adds_nothing(qtbot):
    strip = SM._ChipStrip()
    qtbot.addWidget(strip)
    before = list(strip.values())

    strip._entry.setText("   ")
    strip._commit_entry()

    assert list(strip.values()) == before


# ---------------------------------------------------------------------------
# the alphabet selector, which pretends to be a QLineEdit
# ---------------------------------------------------------------------------

def _alphabet(qtbot):
    widget = SM._AlphabetSelect(choices=[("a", "A"), ("b", "B")])
    qtbot.addWidget(widget)
    return widget


def test_the_alphabet_selector_answers_the_line_edit_api(qtbot):
    """Callers that only know ``text``/``setText`` have to keep working."""
    widget = _alphabet(qtbot)

    widget.setText("a")

    assert widget.text() == repr(widget.get_value())


def test_toggling_an_alphabet_button_reports_a_change(qtbot):
    widget = _alphabet(qtbot)
    seen = []
    widget.changed.connect(lambda: seen.append(True))

    widget._on_toggled(True)

    assert seen == [True]


def test_a_bare_scalar_becomes_a_one_member_selection():
    assert SM._AlphabetSelect._as_members(7) == {7}


# ---------------------------------------------------------------------------
# the list editor
# ---------------------------------------------------------------------------

def _list_editor(qtbot, **kwargs):
    widget = SM._ListEditor(**kwargs)
    qtbot.addWidget(widget)
    return widget


def _nested_editor(qtbot, default, allow_none=False):
    widget = SM._ListEditor(key="_probe", default=default, nested_capable=True,
                            allow_none=allow_none, element_type=str)
    qtbot.addWidget(widget)
    return widget


def test_a_nested_list_editor_with_every_group_empty_gives_none(qtbot):
    editor = _nested_editor(qtbot, [[]], allow_none=True)

    assert editor.get_value() is None


def test_a_nested_list_editor_that_may_not_be_none_gives_an_empty_container(
        qtbot):
    editor = _nested_editor(qtbot, [[]], allow_none=False)

    assert editor.get_value() == []


def test_a_nested_list_editor_always_shows_at_least_one_row(qtbot):
    editor = _nested_editor(qtbot, [[]])

    assert editor._strips


def test_the_list_editor_answers_the_line_edit_api(qtbot):
    editor = _list_editor(qtbot, default=["a", "b"], element_type=str)

    editor.setText("c, d")

    assert editor.text() == str(editor.get_value())


def test_a_single_item_list_renders_as_that_item(qtbot):
    editor = _list_editor(qtbot, default=["only"], element_type=str)

    assert editor.text() == "only"


def test_a_typed_element_that_will_not_convert_keeps_its_text(qtbot):
    """A hand-edited CSV is the normal source of this; refuse to lose it."""
    editor = _list_editor(qtbot, default=[1], element_type=int)

    assert editor._cast("not a number") == "not a number"


def test_the_word_none_in_an_untyped_list_is_the_value_none(qtbot):
    editor = _list_editor(qtbot, default=[], element_type=None)

    assert editor._cast("None") is None


def test_a_scalar_becomes_a_one_element_list(qtbot):
    assert SM._ListEditor._as_sequence(5) == [5]


def test_a_literal_that_is_not_a_sequence_becomes_a_one_element_list(qtbot):
    assert SM._ListEditor._as_sequence("7") == [7]


# ---------------------------------------------------------------------------
# list_shape_for
# ---------------------------------------------------------------------------

def test_list_shape_survives_a_settings_module_that_will_not_import(
        monkeypatch):
    import builtins

    real_import = builtins.__import__

    def refuse(name, *args, **kwargs):
        if name == "spacr.settings":
            raise ImportError("half built")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", refuse)

    assert SM.list_shape_for("channels", [1, 2]) is not None


def test_a_tuple_default_that_is_not_declared_a_list_stays_a_tuple():
    """The container the module ships with is the container it gets back.

    A setting whose default is ``(2, 3)`` and whose declared type says nothing
    about lists must not come back as ``[2, 3]``: the pipeline that reads it
    unpacks a pair.
    """
    _nested, _allow_none, element_type, container = SM.list_shape_for(
        "_probe_undeclared_tuple", (2, 3))

    assert container is tuple and element_type is int


def test_a_list_of_flags_is_not_a_list_of_numbers():
    """``bool`` is a subclass of ``int``; the check has to be ordered."""
    shape = SM.list_shape_for("_probe_flags", [True, False])

    assert shape is None or shape[2] is None


# ---------------------------------------------------------------------------
# SettingsWidgets
# ---------------------------------------------------------------------------

def test_building_a_panel_survives_a_plugin_registry_that_raises(monkeypatch):
    _break_plugin(monkeypatch)

    widgets = SM.SettingsWidgets("measure")

    assert widgets.build_sections()


def test_a_label_override_survives_a_plugin_registry_that_raises(monkeypatch):
    widgets = SM.SettingsWidgets("measure")
    _break_plugin(monkeypatch)

    assert isinstance(widgets._label_for("src"), str)


def test_a_plugin_supplies_a_label(monkeypatch):
    widgets = SM.SettingsWidgets("measure")
    _install_plugin(monkeypatch, _FakeApp(labels={"src": "Where the plate is"}))

    assert widgets._label_for("src") == "Where the plate is"


def test_a_tooltip_is_html_formatted_for_its_key():
    widgets = SM.SettingsWidgets("measure")

    assert isinstance(widgets.tooltip_for("src"), str)


def test_a_widget_rendered_without_a_default_is_never_reported_modified():
    """There is nothing to differ from, so saying so would be an invention."""
    widgets = SM.SettingsWidgets("measure")
    widgets._widgets["_not_a_default"] = QLineEdit()

    assert "_not_a_default" not in widgets.modified_keys()


def test_a_widget_that_cannot_be_read_is_skipped_rather_than_fatal():
    """One broken control must not stop the panel reporting the other forty."""
    class _Unreadable(QWidget):
        pass

    widgets = SM.SettingsWidgets("measure")
    widgets._defaults["_probe"] = "x"
    broken = _Unreadable()
    widgets._widgets["_probe"] = broken

    assert isinstance(widgets.modified_keys(), list)


def test_a_hidden_category_is_not_rendered():
    """A module that trains Torch gets no Cellpose tab."""
    widgets = SM.SettingsWidgets("measure")
    titles_before = [title for title, _rows in widgets.build_sections()]
    assert titles_before

    SM._APP_HIDDEN_CATEGORIES.setdefault("measure", set()).add(titles_before[0])
    try:
        titles_after = [title for title, _rows in widgets.build_sections()]
    finally:
        SM._APP_HIDDEN_CATEGORIES["measure"].discard(titles_before[0])

    assert titles_before[0] not in titles_after


def test_a_bool_default_gets_a_toggle_already_set():
    widgets = SM.SettingsWidgets("measure")

    widget = widgets._widget_for("entry", None, True, "_probe_bool")

    assert widget is not None and widget.isChecked() is True


def test_a_list_default_gets_a_list_edit_holding_it():
    widgets = SM.SettingsWidgets("measure")

    widget = widgets._widget_for("entry", None, ["a", "b"],
                                 "_probe_list")

    assert widget is not None
    assert widget.get_value() == ["a", "b"]


def test_setting_a_value_on_a_widget_type_nobody_handles_reports_failure():
    widgets = SM.SettingsWidgets("measure")
    widgets._widgets["_probe"] = QWidget()

    assert widgets.set_value_for_key("_probe", 1) is False


def test_setting_a_value_that_the_widget_refuses_reports_failure():
    widgets = SM.SettingsWidgets("measure")
    widgets._widgets["_probe"] = QSpinBox()

    assert widgets.set_value_for_key("_probe", "not an integer") is False


def test_a_combo_takes_a_value_by_its_visible_text():
    widgets = SM.SettingsWidgets("measure")
    combo = QComboBox()
    combo.addItem("Alpha", "alpha")
    widgets._widgets["_probe"] = combo

    assert widgets.set_value_for_key("_probe", "Alpha") is True
    assert combo.currentText() == "Alpha"


def test_an_editable_combo_keeps_a_value_it_has_never_heard_of():
    """Refusing would silently drop what the settings file actually says."""
    widgets = SM.SettingsWidgets("measure")
    combo = QComboBox()
    combo.setEditable(True)
    combo.addItem("Alpha", "alpha")
    widgets._widgets["_probe"] = combo

    widgets.set_value_for_key("_probe", "something else entirely")

    assert combo.currentText() == "something else entirely"


def test_a_plain_line_edit_takes_none_as_empty_text():
    widgets = SM.SettingsWidgets("measure")
    edit = QLineEdit()
    widgets._widgets["_probe"] = edit

    assert widgets.set_value_for_key("_probe", None) is True
    assert edit.text() == ""


def test_a_hidden_value_is_refused_for_a_key_that_is_not_hidden():
    """Hidden does not mean absent, and absent does not mean hidden."""
    widgets = SM.SettingsWidgets("measure")

    assert widgets.set_hidden_value("src", "/tmp") is False


def test_a_string_that_is_not_the_declared_type_survives_coercion():
    coerce = SM.SettingsWidgets._coerce_to_expected_type

    assert coerce("_probe_never_declared_", "some text") == "some text"


def test_a_boolean_setting_written_as_text_is_read_as_a_boolean():
    coerce = SM.SettingsWidgets._coerce_to_expected_type

    assert coerce("verbose", "True") is True
    assert coerce("verbose", "false") is False


def test_reading_a_widget_nobody_recognises_gives_nothing():
    widgets = SM.SettingsWidgets("measure")

    assert widgets._read_widget(QWidget()) is None


def test_reading_an_empty_line_edit_gives_none_not_an_empty_string():
    """An empty box means "unset"; "" would be a value the pipeline acts on."""
    widgets = SM.SettingsWidgets("measure")

    assert widgets._read_widget(QLineEdit()) is None


def test_reading_an_editable_combo_gives_what_the_user_typed():
    combo = QComboBox()
    combo.setEditable(True)
    combo.addItem("Alpha", "alpha")
    combo.setEditText("typed by hand")
    widgets = SM.SettingsWidgets("measure")

    assert widgets._read_widget(combo) == "typed by hand"


# ---------------------------------------------------------------------------
# the dependency / greying seam
# ---------------------------------------------------------------------------

def test_a_panel_with_no_widgets_has_no_dependency_rules():
    widgets = SM.SettingsWidgets("measure")
    widgets._widgets.clear()

    assert widgets._rules_for_this_panel() == {}


def test_dependency_rules_are_empty_rather_than_fatal_when_settings_refuses(
        monkeypatch):
    widgets = SM.SettingsWidgets("measure")
    widgets.build_sections()
    import spacr.settings as settings

    monkeypatch.setattr(
        settings, "get_setting_dependencies",
        lambda: (_ for _ in ()).throw(RuntimeError("no rules today")))

    assert widgets._rules_for_this_panel() == {}


def test_a_predicate_that_raises_leaves_its_control_enabled(monkeypatch):
    """Greying on a guess hides the control the user needs to fix the run."""
    widgets = SM.SettingsWidgets("measure")
    widgets.build_sections()
    key = next(iter(widgets._widgets))
    control = widgets._widgets[key]
    control.setEnabled(False)
    monkeypatch.setattr(widgets, "_rules_for_this_panel", lambda: {
        key: {"predicate": lambda *_a: (_ for _ in ()).throw(ValueError("x")),
              "sources": ()},
    })

    widgets._refresh_setting_dependencies()

    assert control.isEnabled() is True


def test_a_widget_that_cannot_be_read_does_not_break_the_settings_snapshot():
    """One control raising must not cost the other forty their current values."""
    class _Unreadable(QLineEdit):
        def text(self):
            raise RuntimeError("Internal C++ object already deleted")

    widgets = SM.SettingsWidgets("measure")
    widgets._defaults["_probe"] = "x"
    widgets._widgets["_probe"] = _Unreadable()

    current = widgets._current_dependency_settings()

    assert current["_probe"] == "x"


def test_the_umap_reducer_greying_stops_when_there_is_no_selector():
    widgets = SM.SettingsWidgets("umap")
    widgets._widgets.pop("reduction_method", None)

    assert widgets._refresh_umap_reducer_enablement() is None


def test_the_umap_reducer_greying_stops_on_a_method_it_does_not_know():
    widgets = SM.SettingsWidgets("umap")
    widgets.build_sections()
    selector = widgets._widgets.get("reduction_method")
    if selector is None:
        pytest.skip("umap panel has no reduction_method control in this build")
    if isinstance(selector, QComboBox):
        selector.setEditable(True)
        selector.setEditText("a_method_that_does_not_exist")

    assert widgets._refresh_umap_reducer_enablement() is None


def test_the_classifier_greying_stops_when_the_family_cannot_be_resolved(
        monkeypatch):
    widgets = SM.SettingsWidgets("classify_merged")
    widgets.build_sections()
    import spacr.classify as families

    monkeypatch.setattr(
        families, "resolve_family",
        lambda _s: (_ for _ in ()).throw(ValueError("unknown family")))

    assert widgets._refresh_classifier_family_enablement() is None


def test_changing_the_classifier_family_re_greys_the_panel(monkeypatch):
    widgets = SM.SettingsWidgets("classify_merged")
    widgets.build_sections()
    calls = []
    monkeypatch.setattr(widgets, "_refresh_classifier_family_enablement",
                        lambda: calls.append(True))

    widgets._on_classifier_family_changed()

    assert calls == [True]


def test_the_training_basis_greying_stops_when_the_basis_is_unknown(monkeypatch):
    widgets = SM.SettingsWidgets("classify_merged")
    widgets.build_sections()
    import spacr.training_basis as basis

    monkeypatch.setattr(
        basis, "resolve_basis",
        lambda _s: (_ for _ in ()).throw(ValueError("unknown basis")))

    assert widgets.refresh_training_basis_enablement() is None


# ---------------------------------------------------------------------------
# the plate context read off the user's CSVs
# ---------------------------------------------------------------------------

def test_a_paired_row_that_is_not_a_mapping_is_skipped():
    widgets = SM.SettingsWidgets("regression")

    found = widgets._loaded_table_paths({"paired_data": ["not a dict"]})

    assert found == []


def test_a_single_path_given_as_a_string_is_still_one_path(tmp_path):
    widgets = SM.SettingsWidgets("regression")
    csv_path = tmp_path / "scores.csv"
    csv_path.write_text("plateID,x\np1,1\n")

    found = widgets._loaded_table_paths({"score_data": str(csv_path)})

    assert [path for _index, path in found] == [str(csv_path)]


def test_a_very_large_table_is_left_unknown_rather_than_read(tmp_path):
    """Stalling the GUI to grey one field is a worse answer than "unknown"."""
    widgets = SM.SettingsWidgets("regression")
    big = tmp_path / "big.csv"
    with open(big, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["plateID", "value"])
        row = ["plate1", "x" * 200]
        for _ in range(30_000):
            writer.writerow(row)
    assert os.path.getsize(big) > 5_000_000

    context = widgets._plate_context([(0, str(big))])

    assert context == {"plate_count": None, "has_plate_id": None}


def test_a_csv_the_sniffer_cannot_read_is_parsed_as_ordinary_csv(tmp_path):
    """A one-column file gives the sniffer nothing to sniff."""
    widgets = SM.SettingsWidgets("regression")
    odd = tmp_path / "one_column.csv"
    odd.write_text("plateID\nplate1\nplate2\n")

    context = widgets._plate_context([(0, str(odd))])

    assert context["has_plate_id"] is True
    assert context["plate_count"] == 2


# ---------------------------------------------------------------------------
# tooltip retargeting
# ---------------------------------------------------------------------------

def test_a_layout_hole_is_stepped_over_rather_than_dereferenced(qtbot):
    host = QWidget()
    qtbot.addWidget(host)
    layout = QHBoxLayout(host)
    layout.addStretch(1)
    field = QLineEdit(host)
    layout.addWidget(field)

    found = SM._owning_layout(layout, field)

    assert found is not None and found[0] is layout


def test_a_label_already_holding_a_different_setting_keeps_its_own_help(qtbot):
    """Clearing it here is how eighty settings ended up with help nowhere."""
    root = QWidget()
    qtbot.addWidget(root)
    layout = QHBoxLayout(root)
    label = QLabel("Source", root)
    label.setToolTip("help that belongs to another setting")
    field = QLineEdit(root)
    field.setToolTip("help for this one")
    field.setProperty("settingKey", "src")
    field.setProperty("settingLabelWidget", label)
    layout.addWidget(label)
    layout.addWidget(field)

    SM.retarget_field_tooltips(root)

    assert label.toolTip() == "help that belongs to another setting"


# ---------------------------------------------------------------------------
# the last defensive arms: states a real screen reaches and no test had
# ---------------------------------------------------------------------------

def test_the_merged_classifier_panel_loses_no_setting_to_the_rebuild():
    """The invariant the deleted catch-all was a net for, asserted directly.

    ``classify_merged`` rebuilds its tabs from a literal group table so the
    family choice comes first and each family's settings sit under a heading
    that names it. Any group the five ordering tuples do not mention would be
    dropped by that rebuild -- a whole tab of settings gone from the panel
    with nothing said. The old catch-all copied such a group through, and
    could never run because the tuples enumerate the literal exactly. This
    fails the moment that stops being true.
    """
    widgets = SM.SettingsWidgets("classify_merged")
    sections = widgets.build_sections()
    rendered = {id(widget) for _title, rows in sections
                for _label, widget in rows}
    dropped = sorted(key for key, widget in widgets._widgets.items()
                     if id(widget) not in rendered)

    assert dropped == [], (
        f"the classify_merged rebuild dropped {len(dropped)} setting(s): "
        f"{dropped[:8]}")


def test_no_merged_classifier_heading_names_its_family_twice():
    """"Machine Learning - ML Classifier ..." would read as a stutter."""
    titles = [title for title, _rows
              in SM.SettingsWidgets("classify_merged").build_sections()]

    assert not [t for t in titles if t.startswith("Machine Learning")
                and "ML Classifier" in t]


def test_an_optional_type_hint_is_translated_with_its_optional_word():
    """English short-circuits; the assembly only runs for another language."""
    out = SM._translated_type_hint("custom_regex", "de")

    assert out.endswith(")") and "(" in out
    assert out != "string (optional)" or SM._language_code("de") == "en"


def test_a_remembered_label_whose_c_plus_plus_half_is_gone_is_looked_up_again(
        qtbot):
    """The cached handle is the fast path; a dead one must fall through."""
    import shiboken6

    owner = QWidget()
    qtbot.addWidget(owner)
    field = QLineEdit(owner)
    dead = QLabel(owner)
    field._spacr_setting_label = dead
    shiboken6.delete(dead)

    assert SM._setting_label_for_field(owner, field) is None


def test_a_label_its_layout_does_not_hold_gets_no_dot_and_leaks_no_host(qtbot):
    """``replaceWidget`` answers None; the half-built host must go with it."""
    parent = QWidget()
    qtbot.addWidget(parent)
    QHBoxLayout(parent)
    orphan = QLabel("Source", parent)      # a child, but not IN the layout

    assert SM._add_api_dot_to_label(orphan, "measure", "src", "<b>x</b>") is None
    assert orphan.parentWidget() is parent


def test_a_control_its_layout_does_not_hold_gets_no_dot_either(qtbot):
    parent = QWidget()
    qtbot.addWidget(parent)
    QHBoxLayout(parent)
    orphan = QCheckBox("Verbose", parent)

    assert SM._add_api_dot_to_combined_control(
        parent, orphan, "measure", "verbose", "<b>x</b>") is None
    assert orphan.parentWidget() is parent


def test_a_nested_list_with_no_groups_still_shows_one_empty_row(qtbot):
    """An editor with nothing in it has to offer somewhere to type."""
    editor = SM._ListEditor(key="_probe", default=[["a"]], nested_capable=True,
                            element_type=str)
    qtbot.addWidget(editor)

    editor._rebuild(True, [])

    assert len(editor._strips) == 1


def test_a_control_that_raises_on_read_is_skipped_by_modified_keys():
    """One broken control must not stop the panel reporting the other forty."""
    class _Unreadable(QLineEdit):
        def text(self):
            raise RuntimeError("Internal C++ object already deleted")

    widgets = SM.SettingsWidgets("measure")
    widgets._defaults["_probe"] = "x"
    widgets._widgets["_probe"] = _Unreadable()

    assert "_probe" not in widgets.modified_keys()


def test_the_class_editor_is_handed_the_live_preview_frame():
    """Without it the class editor draws its swatches against nothing."""
    widgets = SM.SettingsWidgets("classify_merged")
    seen = []
    widgets._preview_frame = object()

    class _Editor:
        def __init__(self, **_kwargs):
            pass

        def set_frame(self, frame):
            seen.append(frame)

    import spacr.qt.screens.settings_model as module
    real = module.ClassEditorWidget
    module.ClassEditorWidget = _Editor
    try:
        widgets._widget_for("entry", None, [], "classes")
    finally:
        module.ClassEditorWidget = real

    assert seen == [widgets._preview_frame]


def test_a_list_default_for_a_scalar_setting_still_gets_a_list_widget():
    """``expected_types`` says string; the shipped default is a list anyway."""
    widgets = SM.SettingsWidgets("measure")

    widget = widgets._widget_for("entry", None, ["a", "b"], "src")

    assert widget is not None
    assert widget.get_value() in (["a", "b"], "['a', 'b']")


def test_a_widget_kind_nobody_declared_builds_nothing():
    """Better no control than a text box pretending to be one."""
    widgets = SM.SettingsWidgets("measure")

    assert widgets._widget_for("a_kind_that_does_not_exist", None, 1, "x") is None


def test_coercion_gives_the_text_back_when_settings_will_not_import(monkeypatch):
    """Losing what the user typed is worse than leaving it a string."""
    import builtins

    real_import = builtins.__import__

    def refuse(name, globals=None, locals=None, fromlist=(), level=0):
        if level and "settings" in tuple(fromlist or ()):
            raise ImportError("settings is half built")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", refuse)

    assert SM.SettingsWidgets._coerce_to_expected_type("verbose", "True") \
        == "True"


def test_a_boolean_setting_holding_a_word_that_is_not_a_boolean_is_kept():
    """``continue``, not ``return None``: another declared type may take it."""
    coerce = SM.SettingsWidgets._coerce_to_expected_type

    assert coerce("verbose", "maybe") == "maybe"


def test_a_list_setting_holding_a_scalar_literal_is_kept_as_written():
    """``[1, 2]`` parses to a list; ``5`` parses to an int the key cannot hold."""
    coerce = SM.SettingsWidgets._coerce_to_expected_type

    assert coerce("channels", "5") == "5"


def test_the_umap_reducer_greying_skips_a_setting_the_panel_never_rendered():
    """A key the layout hides is not a control to enable or disable."""
    widgets = SM.SettingsWidgets("umap")
    widgets.build_sections()
    if "reduction_method" not in widgets._widgets:
        pytest.skip("umap panel has no reduction_method control in this build")
    owned = set().union(*SM._UMAP_REDUCER_SETTINGS.values())
    for key in owned:
        widgets._widgets.pop(key, None)

    assert widgets._refresh_umap_reducer_enablement() is None


def test_the_classifier_greying_greys_nothing_when_the_family_is_unknown(
        monkeypatch):
    """An unknown family is the pipeline's error to raise, loudly, at run time."""
    widgets = SM.SettingsWidgets("classify_merged")
    widgets.build_sections()
    assert "classifier_family" in widgets._widgets
    control = next(c for k, c in widgets._widgets.items()
                   if k != "classifier_family")
    control.setEnabled(True)
    import spacr.classify as families
    monkeypatch.setattr(
        families, "resolve_family",
        lambda _s: (_ for _ in ()).throw(ValueError("unknown family")))

    widgets._refresh_classifier_family_enablement()

    assert control.isEnabled() is True


def test_a_nested_layout_is_walked_to_find_the_field(qtbot):
    """The field is usually inside a wrapper's own layout, not the outer one."""
    from PySide6.QtWidgets import QVBoxLayout

    host = QWidget()
    qtbot.addWidget(host)
    outer = QVBoxLayout(host)
    inner = QHBoxLayout()
    outer.addLayout(inner)
    field = QLineEdit(host)
    inner.addWidget(field)

    layout, index = SM._owning_layout(outer, field)

    assert layout is inner and index == 0


def test_an_index_the_layout_answers_nothing_for_is_stepped_over(qtbot):
    """``item.widget()`` on a None item is a crash inside a tooltip refresh.

    Qt answers None for an index that has been emptied under it, which a
    layout being rebuilt while the panel is walked does reach.
    """
    host = QWidget()
    qtbot.addWidget(host)
    field = QLineEdit(host)

    class _Item:
        def widget(self):
            return field

        def layout(self):
            return None

    class _HoleyLayout:
        """Answers None for its first index and the field for its second."""

        def count(self):
            return 2

        def itemAt(self, index):
            return None if index == 0 else _Item()

    layout, index = SM._owning_layout(_HoleyLayout(), field)

    assert index == 1 and isinstance(layout, _HoleyLayout)


def test_the_api_module_registry_waits_when_the_app_is_still_being_built():
    """`spacr.qt.app` imports the screens that import this module.

    Pulling the registry mid-import gets nothing, and the push half of the
    seam delivers every row later -- so returning empty-handed is correct and
    raising would break the import that is halfway through.
    """
    import sys

    class _HalfBuilt:
        pass

    real = sys.modules.get("spacr.qt.app")
    sys.modules["spacr.qt.app"] = _HalfBuilt()
    try:
        assert SM._absorb_registered_api_modules() is None
    finally:
        if real is None:
            sys.modules.pop("spacr.qt.app", None)
        else:
            sys.modules["spacr.qt.app"] = real


def test_a_translated_tooltip_body_comes_from_the_catalog_when_there_is_one(
        monkeypatch):
    """Exact catalog prose wins; word-level translation of a paragraph does not."""
    from spacr.qt import i18n_catalogs

    monkeypatch.setattr(i18n_catalogs, "setting_tooltip",
                        lambda key, source, code: "Der Quellordner.")

    assert SM._translated_body("The source folder.", "de", setting_key="src") \
        == "Der Quellordner."


def test_a_translated_category_blurb_comes_from_the_catalog_too(monkeypatch):
    """The category branch of the same lookup, which has its own catalog call."""
    from spacr.qt import i18n_catalogs

    monkeypatch.setattr(i18n_catalogs, "category_help",
                        lambda source, code: "Eingaben und Ausgaben.")

    assert SM._translated_body("Inputs and outputs.", "de", category=True) \
        == "Eingaben und Ausgaben."


def test_entering_a_widget_shows_the_clickable_tooltip(qtbot, monkeypatch):
    """The hover tooltip is deliberately clickable, so it replaces the native one."""
    widget = QWidget()
    qtbot.addWidget(widget)
    widget.setProperty("apiTooltipHtml", "<b>docs</b>")
    filt = SM._ApiTooltipFilter(widget)

    from spacr.qt.widgets import hover_tooltip

    shown = []

    class _Stub:
        def show_for(self, target, html):
            shown.append((target, html))

        def start_hide(self):
            pass

    monkeypatch.setattr(hover_tooltip.HoverTooltip, "instance",
                        classmethod(lambda cls: _Stub()))

    assert filt.eventFilter(widget, QEvent(QEvent.Enter)) is False
    assert shown == [(widget, "<b>docs</b>")]


def test_a_remembered_label_from_another_window_is_not_reused(qtbot):
    """Two screens can hold the same setting; the label must be this one's."""
    owner = QWidget()
    other = QWidget()
    qtbot.addWidget(owner)
    qtbot.addWidget(other)
    field = QLineEdit(owner)
    stranger = QLabel("Source", other)
    field._spacr_setting_label = stranger

    assert SM._setting_label_for_field(owner, field) is not stranger


def test_a_remembered_label_in_this_window_is_reused(qtbot):
    """The fast path: no walk of every QFormLayout on the screen."""
    owner = QWidget()
    qtbot.addWidget(owner)
    label = QLabel("Source", owner)
    field = QLineEdit(owner)
    field._spacr_setting_label = label

    assert SM._setting_label_for_field(owner, field) is label


def test_an_empty_string_for_a_setting_that_cannot_be_none_stays_a_string():
    """Turning it into None behind the user's back is what this refuses."""
    coerce = SM.SettingsWidgets._coerce_to_expected_type

    assert coerce("verbose", "") == ""
    assert coerce("verbose", "None") == "None"
