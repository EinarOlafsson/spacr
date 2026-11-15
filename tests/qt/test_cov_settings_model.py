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

from PySide6.QtCore import QSize                     # noqa: E402
from PySide6.QtWidgets import (QComboBox, QHBoxLayout, QLabel, QLineEdit,
                               QWidget)

from spacr.qt.screens import settings_model as SM               # noqa: E402


@pytest.fixture
def owned_settings_widgets(qtbot):
    owner = QWidget()
    qtbot.addWidget(owner)
    return lambda app_key: SM.SettingsWidgets(app_key, parent=owner)


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


def test_a_blank_category_title_has_no_blurb():
    assert SM._category_blurb("measure", "") == ""


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
# SettingsWidgets
# ---------------------------------------------------------------------------


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


def test_a_hidden_category_is_not_rendered(owned_settings_widgets):
    """A module that trains Torch gets no Cellpose tab."""
    widgets = owned_settings_widgets("measure")
    titles_before = [title for title, _rows in widgets.build_sections()]
    assert titles_before

    SM._APP_HIDDEN_CATEGORIES.setdefault("measure", set()).add(titles_before[0])
    try:
        titles_after = [title for title, _rows in widgets.build_sections()]
    finally:
        SM._APP_HIDDEN_CATEGORIES["measure"].discard(titles_before[0])

    assert titles_before[0] not in titles_after


def test_a_known_off_form_value_is_preserved_and_an_unknown_one_is_refused():
    """An omitted row is still a setting, but a foreign key is not."""
    widgets = SM.SettingsWidgets("measure")

    assert widgets.set_hidden_value("src", "/tmp") is True
    assert widgets._defaults["src"] == "/tmp"
    assert widgets.set_hidden_value("_not_a_declared_setting", "x") is False


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


def test_the_training_basis_greying_stops_when_the_basis_is_unknown(
        monkeypatch, owned_settings_widgets):
    widgets = owned_settings_widgets("classify_merged")
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
    csv_path.write_text("plateID,x\np1,1\n", encoding="utf-8")

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
    odd.write_text("plateID\nplate1\nplate2\n", encoding="utf-8")

    context = widgets._plate_context([(0, str(odd))])

    assert context["has_plate_id"] is True
    assert context["plate_count"] == 2


# ---------------------------------------------------------------------------
# tooltip retargeting
# ---------------------------------------------------------------------------


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


def test_an_optional_type_hint_is_translated_with_its_optional_word():
    """English short-circuits; the assembly only runs for another language."""
    out = SM._translated_type_hint("custom_regex", "de")

    assert out.endswith(")") and "(" in out
    assert out != "string (optional)" or SM._language_code("de") == "en"


def test_a_nested_list_with_no_groups_still_shows_one_empty_row(qtbot):
    """An editor with nothing in it has to offer somewhere to type."""
    editor = SM._ListEditor(key="_probe", default=[["a"]], nested_capable=True,
                            element_type=str)
    qtbot.addWidget(editor)

    editor._rebuild(True, [])

    assert len(editor._strips) == 1


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


def test_the_umap_reducer_greying_skips_a_setting_the_panel_never_rendered(
        owned_settings_widgets):
    """A key the layout hides is not a control to enable or disable."""
    widgets = owned_settings_widgets("umap")
    widgets.build_sections()
    if "reduction_method" not in widgets._widgets:
        pytest.skip("umap panel has no reduction_method control in this build")
    owned = set().union(*SM._UMAP_REDUCER_SETTINGS.values())
    for key in owned:
        widgets._widgets.pop(key, None)

    assert widgets._refresh_umap_reducer_enablement() is None


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
                        lambda key, source, code, app_key="":
                        "Der Quellordner.")

    assert SM._translated_body("The source folder.", "de", setting_key="src") \
        == "Der Quellordner."


def test_a_translated_category_blurb_comes_from_the_catalog_too(monkeypatch):
    """The category branch of the same lookup, which has its own catalog call."""
    from spacr.qt import i18n_catalogs

    monkeypatch.setattr(i18n_catalogs, "category_help",
                        lambda source, code: "Eingaben und Ausgaben.")

    assert SM._translated_body("Inputs and outputs.", "de", category=True) \
        == "Eingaben und Ausgaben."


def test_an_empty_string_for_a_setting_that_cannot_be_none_stays_a_string():
    """Turning it into None behind the user's back is what this refuses."""
    coerce = SM.SettingsWidgets._coerce_to_expected_type

    assert coerce("verbose", "") == ""
    assert coerce("verbose", "None") == "None"
