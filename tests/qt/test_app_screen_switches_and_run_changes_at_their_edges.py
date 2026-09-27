"""The runtime switches and the run-change refreshes, at their edges.

Pins what the user sees when the preview, interactive and AI switches and
the results refreshes meet a missing or refusing collaborator:

* the preview switch moves beside the Actions heading when the card has no
  title row to ride on, and a screen without a switch still shows/hides
  the card;
* switching UMAP's explorer off with an empty figure queue leaves the queue
  hidden;
* a configured-provider list that cannot be read means "no preference";
* a settings widget whose shutdown fails does not stop the others;
* raising the results tab works on a figures card that does not fold;
* a run change refreshes the Cells tab even when its montage cannot be
  emptied, and survives a Measurements tab that cannot refresh;
* an imported organelle switch never reshapes a form without organelles;
* Measure's Run stops, and says so in the log, when the crop warning is
  declined.
"""
from __future__ import annotations

import os
import types

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QCheckBox, QHBoxLayout, QWidget  # noqa: E402

from spacr.qt.screens.app_screen import AppScreen  # noqa: E402
from spacr.qt.widget_cleanup import retire_pyqtgraph_menus  # noqa: E402

pytestmark = pytest.mark.qt


def _make(key):
    widget = AppScreen(key)
    return widget


def _retire(widget):
    retire_pyqtgraph_menus(widget)
    widget.close()
    widget.deleteLater()


@pytest.fixture
def screen(qtbot):
    widget = _make("regression")
    try:
        yield widget
    finally:
        _retire(widget)


def _raise(error=ValueError):
    def go(*_args, **_kwargs):
        raise error("refused")
    return go


# --------------------------------------------------------------------------
# the preview switch


@pytest.fixture
def preview(screen, qtbot):
    host = QWidget()
    qtbot.addWidget(host)
    heading = QHBoxLayout(host)
    card = QWidget()
    qtbot.addWidget(card)
    screen._preview_card_attr = "_spare_preview_card"
    screen._spare_preview_card = card
    screen._preview_primed = True
    screen._actions_heading_row = heading
    return types.SimpleNamespace(card=card, heading=heading, host=host)


def test_a_card_without_a_title_row_leaves_the_switch_by_the_heading(
        screen, preview, qtbot):
    switch = QCheckBox("Live")
    qtbot.addWidget(switch)
    screen._preview_switch = switch
    screen._on_preview_switch(True)
    assert preview.heading.indexOf(switch) >= 0
    assert switch.isChecked()
    assert not preview.card.isHidden()


def test_without_a_switch_the_card_still_follows(screen, preview):
    screen._preview_switch = None
    screen._on_preview_switch(False)
    assert preview.card.isHidden()


# --------------------------------------------------------------------------
# UMAP's explorer


def test_turning_the_explorer_off_keeps_an_empty_queue_hidden(
        screen, monkeypatch, qtbot):
    explorer = QWidget()
    qtbot.addWidget(explorer)
    explorer.show()
    shown = []
    queue = types.SimpleNamespace(count=lambda: 0,
                                  show=lambda: shown.append(True),
                                  hide=lambda: None)
    real_if_built = screen._if_built
    monkeypatch.setattr(
        screen, "_if_built",
        lambda name: explorer if name == "_umap_explorer"
        else real_if_built(name))
    screen._figure_queue = queue
    screen._on_interactive_switch(False)
    assert explorer.isHidden()
    assert shown == []


# --------------------------------------------------------------------------
# the AI provider


def test_an_unreadable_provider_list_means_no_preference(monkeypatch,
                                                         screen):
    monkeypatch.setattr("spacr.qt.preferences.get_preferred_provider",
                        lambda: "claude")
    monkeypatch.setattr("spacr.qt.ai.configured_providers",
                        _raise(OSError))
    assert screen._wanted_provider() == ""


# --------------------------------------------------------------------------
# shutting settings widgets down


def test_a_widget_that_cannot_shut_down_does_not_stop_the_rest(screen):
    stopped = []
    real = screen._settings_model
    screen._settings_model = types.SimpleNamespace(_widgets={
        "a": types.SimpleNamespace(shutdown=_raise(RuntimeError)),
        "b": types.SimpleNamespace(shutdown=_raise(TypeError)),
        "c": types.SimpleNamespace(shutdown=lambda: stopped.append("c")),
    })
    try:
        screen._shutdown_settings_widgets()
    finally:
        screen._settings_model = real
    assert stopped == ["c"]


# --------------------------------------------------------------------------
# the results tabs


def test_the_results_tab_is_raised_on_a_card_that_does_not_fold(
        screen, qtbot, monkeypatch):
    tabs = screen._results_tabs
    page = screen._results_page
    assert tabs is not None and page is not None
    other = QWidget()
    tabs.insertTab(0, other, "Other")
    tabs.setCurrentWidget(other)
    card = QWidget()
    qtbot.addWidget(card)
    monkeypatch.setattr(screen, "_figures_card", card)
    screen._raise_the_results_tab()
    assert tabs.currentWidget() is page
    assert not card.isHidden()


def test_a_run_change_refreshes_what_it_can(screen):
    refreshed = []
    screen._cell_montage = types.SimpleNamespace(
        clear=None, refresh=lambda: refreshed.append("cells"),
        shutdown=lambda: None)
    screen._scan_panel = types.SimpleNamespace(
        refresh_databases=_raise(OSError))
    screen._on_loaded_run_changed_refresh_tabs()
    assert refreshed == ["cells"]


# --------------------------------------------------------------------------
# bulk apply and the organelle switches


def test_an_organelle_switch_never_reshapes_a_form_without_organelles(
        screen):
    current = {"organelle_channel": 1, "src": "/data"}
    assert screen._bulk_apply_changes_form_shape(
        {"organelle_channel": 3}, current) is False


# --------------------------------------------------------------------------
# the model chooser and the console fold


def test_a_chosen_model_is_written_into_a_plain_field(screen, monkeypatch,
                                                     qtbot, caplog):
    from PySide6.QtWidgets import QLineEdit

    monkeypatch.setattr("spacr.qt.widgets.model_zoo_picker.choose_model",
                        lambda parent, kinds: "/models/cyto3")
    field = QLineEdit()
    qtbot.addWidget(field)
    screen._choose_a_model_for(field, "cell_model_name")
    assert field.text() == "/models/cyto3"
    stranger = types.SimpleNamespace()
    with caplog.at_level("WARNING"):
        screen._choose_a_model_for(stranger, "cell_model_name")
    assert "no way to write a model path into SimpleNamespace" in caplog.text
    assert vars(stranger) == {}


def test_folding_the_console_gives_its_height_to_the_pane_above(screen,
                                                                qtbot):
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QSplitter

    splitter = QSplitter(Qt.Vertical)
    qtbot.addWidget(splitter)
    above, wrap = QWidget(), QWidget()
    splitter.addWidget(above)
    splitter.addWidget(wrap)
    splitter.resize(300, 600)
    splitter.show()
    splitter.setSizes([300, 300])
    before = splitter.sizes()
    was_wrap = getattr(screen, "_console_wrap", None)
    was_split = getattr(screen, "_console_splitter", None)
    screen._console_wrap = wrap
    screen._console_splitter = None
    try:
        screen._console_folded(True)
        folded = splitter.sizes()
        assert folded[1] < before[1] and folded[0] > before[0]
        assert screen._console_height == before[1]
        screen._console_folded(False)
        assert splitter.sizes()[1] == before[1]
    finally:
        screen._console_wrap = was_wrap
        screen._console_splitter = was_split


# --------------------------------------------------------------------------
# Mask's switches when their homes refuse them


def test_mask_keeps_its_live_and_ops_switches_when_their_homes_refuse(
        qtbot, monkeypatch):
    from spacr.qt.widgets import card as card_module
    from spacr.qt.widgets.ai_toggle_label import AiToggleLabel

    for cls in (card_module.Card, card_module._CardBuiltWhenShown):
        real = cls.add_title_action

        def refuse_the_switch(card, widget, *args, _real=real, **kwargs):
            if isinstance(widget, AiToggleLabel):
                raise RuntimeError("no room on the title bar")
            return _real(card, widget, *args, **kwargs)

        monkeypatch.setattr(cls, "add_title_action", refuse_the_switch)
    installed = []

    def refuse(screen, switch):
        installed.append(switch)
        raise ImportError("the OPS page is not available")

    monkeypatch.setattr("spacr.qt.screens.mask.install_ops_switch", refuse)
    widget = _make("mask")
    try:
        switch = widget._preview_switch
        assert switch is not None
        assert widget._lp_switch is switch
        card = getattr(widget, widget._preview_card_attr)
        assert not card.isAncestorOf(switch), (
            "a card that refused the switch must not be holding it")
        assert installed == [widget._ops_switch]
        assert widget._ops_switch is not None
    finally:
        _retire(widget)


# --------------------------------------------------------------------------
# Measure's Run


def test_measure_stops_when_the_crop_warning_is_declined(qtbot, monkeypatch):
    logged = []
    monkeypatch.setattr("spacr.qt.verbose_logger.log_button_press",
                        lambda name, details: logged.append((name, details)))
    widget = _make("measure")
    try:
        asked = []
        monkeypatch.setattr(widget, "_confirm_crop_choices",
                            lambda settings: asked.append(settings) or False)
        widget._on_run(override={"src": "/nowhere"})
        assert asked == [{"src": "/nowhere"}]
        assert logged[-1] == ("measure.Run",
                              {"result": "cancelled_at_crop_warning"})
        assert getattr(widget, "_thread", None) is None
    finally:
        _retire(widget)
