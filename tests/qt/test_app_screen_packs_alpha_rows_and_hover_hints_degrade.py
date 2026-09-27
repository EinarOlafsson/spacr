"""Example packs, alpha rows, the live-preview wiring and hover hints.

Pins what the user still gets from an AppScreen when a helper it leans on
is missing or refuses:

* a shipped path the crop re-rooter cannot handle is still re-homed by the
  folder name, and a pack whose paths cannot be re-homed is still applied;
* the example screen's own kind button shows the fetch, and a paired-data
  control that cannot take paths gets the tables as settings instead;
* alpha rows are hidden one by one even when one row has gone, a non-combo
  is never gated, and a combo without a list view or item model keeps its
  entries; the live re-decide after Preferences carries on past a strip or
  model that refuses;
* the Plaque mode switch failing to install does not take the screen down;
* the preview is regrouped when the file naming changes, through the first
  signal that accepts the connection; an owed preview is not regrouped;
* the source-folder chooser writes into a plain src field and refuses
  without one;
* a tooltip event on an ordinary widget is not swallowed; a hovered setting
  whose docs URL cannot be built still gets its hint, without a link;
* the dock's module hint refuses without a key and writes plain text when
  there is no API or tutorial link.
"""
from __future__ import annotations

import os
import sys
import types

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QEvent, QStringListModel  # noqa: E402
from PySide6.QtWidgets import QComboBox, QLabel, QLineEdit, QPushButton, QWidget  # noqa: E402

from spacr.qt.screens.app_screen import AppScreen  # noqa: E402
from spacr.qt.widget_cleanup import retire_pyqtgraph_menus  # noqa: E402

pytestmark = pytest.mark.qt


def _console_text(console) -> str:
    from spacr.qt.widgets.console_panel import _StdoutBlock

    return "\n".join(block.text()
                     for block in console.findChildren(_StdoutBlock))


@pytest.fixture
def screen(qtbot):
    widget = AppScreen("regression")
    try:
        yield widget
    finally:
        retire_pyqtgraph_menus(widget)
        widget.close()
        widget.deleteLater()


@pytest.fixture
def swap_model(screen):
    real = screen._settings_model

    def swap(model):
        screen._settings_model = model
        return model

    yield swap
    screen._settings_model = real


class _Signal:
    def __init__(self, refuse=False):
        self.refuse = refuse
        self.slots = []

    def connect(self, slot):
        if self.refuse:
            raise TypeError("cannot connect")
        self.slots.append(slot)


def _raise(error=ValueError):
    def go(*_args, **_kwargs):
        raise error("refused")
    return go


# --------------------------------------------------------------------------
# example packs


def test_a_path_the_rerooter_cannot_handle_is_rehomed_by_name(
        screen, monkeypatch, tmp_path):
    monkeypatch.setattr("spacr.portable_paths.reroot_crop_path",
                        _raise(OSError))
    destination = tmp_path / "plate"
    loaded = {"src": "/far/away/plate/merged", "n": 3}
    out = screen.reanchor_example_paths(loaded, destination)
    assert out == {"src": str(destination / "merged"), "n": 3}
    assert loaded["src"] == "/far/away/plate/merged"


def test_a_pack_whose_paths_cannot_be_rehomed_is_still_applied(
        screen, monkeypatch, tmp_path):
    report = types.SimpleNamespace(source="gen_settings.csv",
                                   applied=["alpha"], renamed=[("old", "beta")],
                                   dropped=[], elsewhere=[], malformed=0)
    monkeypatch.setattr(
        "spacr.qt.settings_pack.settings_from_pack",
        lambda app_key, root: ({"alpha": "1", "beta": "2", "gamma": "3"},
                               report))
    monkeypatch.setattr(screen, "reanchor_example_paths", _raise(KeyError))
    got = []
    monkeypatch.setattr(screen, "apply_settings_dict",
                        lambda values: got.append(dict(values)) or len(values))
    applied = screen.apply_settings_that_came_with(tmp_path)
    assert got == [{"alpha": "1", "beta": "2"}]
    assert applied == 2


class TestTheExampleScreen:

    @pytest.fixture
    def fetched(self, monkeypatch, tmp_path):
        from spacr import example_data

        got = types.SimpleNamespace(
            counts=["p1_counts.csv"], scores=["p1_dv.csv"],
            folder=str(tmp_path), note=lambda: "Fetched.")
        monkeypatch.setattr(example_data, "missing",
                            lambda folder=None, kind=None: [{"f": 1}])
        return got

    def test_the_kind_button_shows_the_fetch_and_comes_back(
            self, screen, monkeypatch, fetched):
        from spacr import example_data

        button = QPushButton("Score")
        screen._example_scores_button = button
        seen = []

        def fetch(**kwargs):
            seen.append((button.isEnabled(), button.text(), kwargs["kind"]))
            return fetched

        monkeypatch.setattr(example_data, "fetch", fetch)
        screen.load_the_example_screen(download=False, kind="scores")
        assert seen == [(False, "Fetching 1 file(s)…", "scores")]
        assert button.isEnabled()
        assert button.text() == "Load test data…"

    def test_without_a_paired_table_the_tables_go_in_as_settings(
            self, screen, monkeypatch, fetched):
        from spacr import example_data

        monkeypatch.setattr(example_data, "fetch", lambda **_k: fetched)
        screen._example_data_button = None
        widgets = screen._settings_model._widgets
        monkeypatch.setitem(widgets, "paired_data", QLabel("not a table"))
        applied = []
        monkeypatch.setattr(
            screen, "apply_settings_dict",
            lambda values: applied.append(values) or 2)
        answer = screen.load_the_example_screen(download=False)
        assert applied == [{"count_data": ["p1_counts.csv"],
                            "score_data": ["p1_dv.csv"]}]
        assert answer["applied"] == 2
        assert "Paired 1 score table(s) with 1 count table(s)" in (
            _console_text(screen._console))


# --------------------------------------------------------------------------
# alpha rows and choices


def test_a_gone_alpha_row_does_not_stop_the_others_hiding(
        screen, swap_model, monkeypatch):
    asked = []

    def set_row_visible(key, visible):
        asked.append((key, visible))
        if key == "gone":
            raise RuntimeError("Internal C++ object already deleted.")

    swap_model(types.SimpleNamespace(_set_row_visible=set_row_visible))
    monkeypatch.setattr(screen, "_alpha_hidden_keys",
                        lambda: ["gone", "kept"])
    screen._apply_alpha_rows()
    assert asked == [("gone", False), ("kept", False)]


def test_only_a_combo_is_gated(qtbot, monkeypatch):
    monkeypatch.setattr("spacr.settings._alpha_choices", lambda key: {"b"})
    field = QLineEdit("b")
    qtbot.addWidget(field)
    AppScreen._gate_alpha_choices("backend", field)
    assert field.isEnabled() and field.text() == "b"


class _BareCombo(QComboBox):
    def view(self):
        return None


def test_a_combo_without_a_view_or_items_keeps_its_entries(qtbot,
                                                           monkeypatch):
    monkeypatch.setattr("spacr.settings._alpha_choices", lambda key: {"beta"})
    monkeypatch.setattr("spacr.qt.preferences._is_alpha_visible",
                        lambda *a: False)
    combo = _BareCombo()
    qtbot.addWidget(combo)
    combo.setModel(QStringListModel(["stable", "beta"]))
    AppScreen._gate_alpha_choices("backend", combo)
    assert [combo.itemText(i) for i in range(combo.count())] == [
        "stable", "beta"]


def test_a_shut_alpha_choice_is_hidden_and_disabled(qtbot, monkeypatch):
    monkeypatch.setattr("spacr.settings._alpha_choices", lambda key: {"beta"})
    monkeypatch.setattr("spacr.qt.preferences._is_alpha_visible",
                        lambda *a: False)
    combo = QComboBox()
    qtbot.addWidget(combo)
    combo.addItems(["stable", "beta"])
    AppScreen._gate_alpha_choices("backend", combo)
    assert combo.view().isRowHidden(1)
    assert not combo.model().item(1).isEnabled()
    assert combo.model().item(0).isEnabled()


def test_the_live_alpha_redecide_carries_on_past_refusals(
        screen, swap_model, monkeypatch):
    class _Bar:
        def _build_index(self):
            raise ValueError("index refused")

        def apply(self, reopen=True):
            raise ValueError("apply refused")

    was = screen.__dict__.get("_settings_search")
    screen._settings_search = _Bar()
    swap_model(types.SimpleNamespace(refresh_object_visibility=_raise()))
    reached = []
    monkeypatch.setattr(screen, "refresh_maturity_visibility",
                        lambda: reached.append("maturity"))
    monkeypatch.setattr("spacr.qt.preferences._apply_alpha_widgets",
                        lambda root: reached.append(root))
    try:
        screen._refresh_alpha_visibility()
    finally:
        screen._settings_search = was
    assert reached == ["maturity", screen]


def test_the_live_alpha_redecide_without_an_object_rule(screen, swap_model,
                                                        monkeypatch):
    was = screen.__dict__.get("_settings_search")
    screen._settings_search = None
    swap_model(types.SimpleNamespace())
    reached = []
    monkeypatch.setattr(screen, "refresh_maturity_visibility",
                        lambda: reached.append("maturity"))
    monkeypatch.setattr("spacr.qt.preferences._apply_alpha_widgets",
                        lambda root: reached.append(root))
    try:
        screen._refresh_alpha_visibility()
    finally:
        screen._settings_search = was
    assert reached == ["maturity", screen]


# --------------------------------------------------------------------------
# plaque mode and the live preview's naming


def test_a_plaque_switch_that_will_not_install_is_survived(screen,
                                                            monkeypatch):
    tried = []

    def refuse(target):
        tried.append(target)
        raise ImportError("plaque preview missing")

    monkeypatch.setattr(
        "spacr.qt.widgets.plaque_preview.install_plaque_mode", refuse)
    screen._install_plaque_mode()
    assert tried == [screen]


def test_the_naming_regroups_the_preview_through_the_signal_that_takes(
        screen, swap_model, monkeypatch, qtbot):
    regrouped = []
    monkeypatch.setattr(screen, "_part_is_owed", lambda part: False)
    screen._live_preview = types.SimpleNamespace(
        regroup_the_folder=lambda: regrouped.append(True))
    regex = types.SimpleNamespace(currentIndexChanged=_Signal(refuse=True),
                                  textChanged=None,
                                  value_changed=_Signal())
    swap_model(types.SimpleNamespace(_widgets={"custom_regex": regex}))
    screen._wire_live_preview_naming()
    assert len(regex.value_changed.slots) == 1
    regex.value_changed.slots[0]("(?P<well>.*)")
    qtbot.waitUntil(lambda: regrouped == [True], timeout=3000)


def test_a_naming_field_with_no_signal_is_not_followed(
        screen, swap_model, monkeypatch):
    monkeypatch.setattr(screen, "_part_is_owed", lambda part: True)
    silent = types.SimpleNamespace()
    swap_model(types.SimpleNamespace(_widgets={"metadata_type": silent}))
    screen._wire_live_preview_naming()
    timer = screen._live_naming_timer
    assert not timer.isActive()
    assert vars(silent) == {}


def test_a_preview_still_owed_is_not_regrouped(screen, monkeypatch):
    regrouped = []
    monkeypatch.setattr(screen, "_part_is_owed", lambda part: True)
    screen.__dict__["_live_preview_stub"] = None
    monkeypatch.setattr(screen, "_live_preview", types.SimpleNamespace(
        regroup_the_folder=lambda: regrouped.append(True)), raising=False)
    screen._regroup_the_live_preview()
    assert regrouped == []


def test_a_preview_that_cannot_regroup_is_left_alone(screen, monkeypatch):
    monkeypatch.setattr(screen, "_part_is_owed", lambda part: False)
    monkeypatch.setattr(screen, "_live_preview",
                        types.SimpleNamespace(regroup_the_folder=None),
                        raising=False)
    assert screen._regroup_the_live_preview() is None


# --------------------------------------------------------------------------
# choosing the source folder


def test_no_src_field_means_no_chooser(screen, swap_model, monkeypatch):
    from PySide6.QtWidgets import QFileDialog

    opened = []
    monkeypatch.setattr(QFileDialog, "getExistingDirectory",
                        lambda *a: opened.append(a) or "/x")
    swap_model(types.SimpleNamespace(_widgets={}))
    assert screen.choose_source_folder() == ""
    assert opened == []


def test_a_plain_src_field_is_written_directly(screen, swap_model,
                                               monkeypatch, qtbot, tmp_path):
    from PySide6.QtWidgets import QFileDialog

    field = QLineEdit()
    qtbot.addWidget(field)
    monkeypatch.setattr(QFileDialog, "getExistingDirectory",
                        lambda *a: str(tmp_path))
    swap_model(types.SimpleNamespace(_widgets={"src": field}))
    assert screen.choose_source_folder() == str(tmp_path)
    assert field.text() == str(tmp_path)


def test_a_src_that_is_not_a_line_edit_is_not_written(screen, swap_model,
                                                     monkeypatch, tmp_path):
    from PySide6.QtWidgets import QFileDialog

    field = types.SimpleNamespace(text="unchanged")
    monkeypatch.setattr(QFileDialog, "getExistingDirectory",
                        lambda *a: str(tmp_path))
    swap_model(types.SimpleNamespace(_widgets={"src": field}))
    assert screen.choose_source_folder() == str(tmp_path)
    assert field.text == "unchanged"


# --------------------------------------------------------------------------
# hover hints


def test_a_tooltip_on_an_ordinary_widget_is_not_swallowed(screen, qtbot):
    plain = QWidget()
    qtbot.addWidget(plain)
    assert screen.eventFilter(plain, QEvent(QEvent.Type.ToolTip)) is False


@pytest.fixture
def hint_strip(screen):
    if getattr(screen, "_hint_strip", None) is None:
        screen._hint_strip = QLabel()
    return screen._hint_strip


@pytest.fixture
def written(screen, monkeypatch):
    got = []
    monkeypatch.setattr(
        screen, "_write_hint",
        lambda text, url="", hold=False, animated=False:
            got.append((text, url, hold)))
    monkeypatch.setattr("spacr.qt.preferences.get_tooltips_bottom_enabled",
                        lambda: True)
    monkeypatch.setattr("spacr.qt.preferences.get_tooltips_box_enabled",
                        lambda: False)
    return got


def test_a_setting_whose_docs_link_fails_still_gets_its_hint(
        screen, hint_strip, written, monkeypatch, qtbot):
    field = QLineEdit()
    qtbot.addWidget(field)
    field.setProperty("settingKey", "src")
    monkeypatch.setattr("spacr.qt.screens.settings_model.refresh_api_tooltips",
                        lambda obj: None)
    monkeypatch.setattr("spacr.qt.screens.settings_model.api_docs_url",
                        _raise(LookupError))
    monkeypatch.setattr(screen._settings_model, "plain_tooltip_for",
                        lambda key: "Where the images are.")
    screen.eventFilter(field, QEvent(QEvent.Type.Enter))
    assert written == [("Where the images are.", "", True)]


def test_a_caption_hint_is_written_without_a_link(screen, hint_strip,
                                                  written, qtbot):
    caption = QLabel("Source")
    qtbot.addWidget(caption)
    screen._hint_map[caption] = "The caption's help."
    screen.eventFilter(caption, QEvent(QEvent.Type.Enter))
    assert written == [("The caption's help.", "", True)]


def test_a_module_hint_needs_a_key(screen, hint_strip):
    assert screen.show_module_hint("", "Makes masks.") is False


def test_a_module_hint_without_link_modules_writes_nothing(
        screen, hint_strip, monkeypatch):
    monkeypatch.setitem(sys.modules, "spacr.qt.tutorials", None)
    before = hint_strip.text()
    assert screen.show_module_hint("mask", "Makes masks.") is False
    assert hint_strip.text() == before


def test_a_module_hint_without_links_is_plain_text(screen, hint_strip,
                                                   monkeypatch):
    monkeypatch.setattr("spacr.qt.tutorials.tutorial_url", lambda key: "")
    monkeypatch.setattr("spacr.qt.screens.settings_model.api_docs_url",
                        lambda *a: "")
    hint_strip.resize(600, 120)
    assert screen.show_module_hint("mask", "Makes masks & more.") is True
    assert hint_strip.text() == "Makes masks &amp; more."
    assert hint_strip.toolTip() == "Makes masks & more."
