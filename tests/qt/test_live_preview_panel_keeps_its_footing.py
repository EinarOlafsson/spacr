"""Live preview panel: what the user sees on the less-travelled paths.

Pinned here, with a real panel offscreen:

* a folder where some files follow the naming and one does not gets an
  "image" column after the channel columns, and clicking it, or clicking a
  channel with two objects chosen, changes no channel setting, while
  clicking with an organelle chosen moves the organelle channel;
* following an object onto a channel its field has no file for keeps the
  field on screen;
* a regrouping that finishes after a newer one, or for another folder, is
  not adopted;
* an open Live settings window is not swept into the hidden control store;
* the cycle arrows do nothing with one object chosen;
* ``apply_settings`` with values no control can hold: a word for an
  organelle channel, an ``adjust_cells`` with no truth value, an unknown
  morphology, a number field given text -- the rest still lands;
* a smaller organelle count than the one on screen puts the object picker
  back on the first entry;
* changing the organelle morphology re-offers the methods, even when the
  open settings window cannot re-gate its rows;
* a description written for an organelle setting becomes its tooltip;
* the plaque model resolved off the GUI thread is not adopted after the
  user picked a model, nor after a pass settled it first;
* a model chosen from the zoo that the list already holds is selected, not
  added twice;
* a worker that finishes while a newer pass runs leaves the buttons busy;
* a stale cell-probability result is dropped;
* an enhancement chain is named on the status line;
* the mask comparison with no field, with a multi-channel field, and a
  comparison popup that is dismissed;
* the Live settings window shows every row control again when reopened;
* the "no limit is None" table when the shipped defaults cannot be read.
"""
from __future__ import annotations

import sys
import threading
import types
from pathlib import Path

import numpy as np
import pytest
import tifffile

pytest.importorskip("PySide6")

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import QDialog, QFormLayout  # noqa: E402

from spacr.qt.widgets import live_preview as LP  # noqa: E402

pytestmark = pytest.mark.qt


@pytest.fixture(autouse=True)
def _qapp(qapp):
    """QPixmap aborts the process outright when no QGuiApplication exists."""
    return qapp


def _name(field: int, chan: int) -> str:
    return f"plate1_A01_T0001F{field:03d}L01A01Z01C{chan:02d}.tif"


@pytest.fixture
def mixed_plate(tmp_path: Path) -> Path:
    """Field 1 has channels 1 and 2, field 2 only channel 1, and one file
    does not follow the naming at all."""
    root = tmp_path / "mixed"
    root.mkdir()
    tifffile.imwrite(root / _name(1, 1), np.full((8, 8), 11, np.uint16))
    tifffile.imwrite(root / _name(1, 2), np.full((8, 8), 12, np.uint16))
    tifffile.imwrite(root / _name(2, 1), np.full((8, 8), 21, np.uint16))
    tifffile.imwrite(root / "odd.tif", np.full((8, 8), 99, np.uint16))
    return root


@pytest.fixture
def panel(qtbot):
    widget = LP.LivePreviewPanel(threaded=False)
    qtbot.addWidget(widget)
    return widget


@pytest.fixture
def no_shift(monkeypatch):
    monkeypatch.setattr(LP.LivePreviewPanel, "_shift_held",
                        lambda self: False)


def _headers(panel):
    table = panel._set_table
    return [table.horizontalHeaderItem(c).text()
            for c in range(table.columnCount())]


def _cell(panel, column: int, name: str) -> int:
    """The row whose cell in ``column`` names the file ``name``."""
    table = panel._set_table
    for row in range(table.rowCount()):
        item = table.item(row, column)
        if item is not None and Path(item.data(Qt.UserRole)).name == name:
            return row
    raise AssertionError(f"{name} is not in column {column}")


# ---------------------------------------------------------------------------
# the set table
# ---------------------------------------------------------------------------

def test_an_unreadable_file_gets_its_own_column_after_the_channels(
        panel, mixed_plate):
    assert panel.load_image(mixed_plate / _name(1, 1))
    assert _headers(panel) == ["ch 0", "ch 1", "image"]
    assert panel._column_channels == [0, 1, None]
    row = _cell(panel, 2, "odd.tif")
    assert panel._set_table.item(row, 0) is None, (
        "the odd file has no channel, so it sits under 'image' only")


def test_clicking_the_image_column_sets_no_channel(panel, mixed_plate,
                                                   no_shift):
    panel.load_image(mixed_plate / _name(1, 1))
    panel._cell_channel.setValue(0)
    row = _cell(panel, 2, "odd.tif")
    panel._on_set_cell_clicked(row, 2)
    assert panel._image_path.name == "odd.tif"
    assert panel._cell_channel.value() == 0


def test_clicking_a_channel_with_two_objects_sets_neither(
        panel, mixed_plate, no_shift):
    panel.load_image(mixed_plate / _name(1, 1))
    panel._object_box.setCurrentText("cell + nucleus")
    before = (panel._cell_channel.value(), panel._nucleus_channel.value())
    row = _cell(panel, 1, _name(1, 2))
    panel._on_set_cell_clicked(row, 1)
    assert panel._image_path.name == _name(1, 2), "the channel is shown"
    assert (panel._cell_channel.value(),
            panel._nucleus_channel.value()) == before


def test_clicking_a_channel_with_an_organelle_chosen_moves_its_channel(
        panel, mixed_plate, no_shift):
    panel.load_image(mixed_plate / _name(1, 1))
    panel._object_box.setCurrentText("organelle")
    assert panel._selected_object_types() == ("organelle",)
    panel._organelle_channel.setValue(0)
    row = _cell(panel, 1, _name(1, 2))
    panel._on_set_cell_clicked(row, 1)
    assert panel._organelle_channel.value() == 1


def test_following_onto_a_channel_the_field_lacks_keeps_the_field(
        panel, mixed_plate, no_shift):
    panel.load_image(mixed_plate / _name(1, 1))
    panel._cell_channel.setValue(0)
    row = _cell(panel, 0, _name(2, 1))
    panel._on_set_cell_clicked(row, 0)
    assert panel._image_path.name == _name(2, 1)
    assert panel._set_table.item(row, 1) is None
    panel._cell_channel.setValue(1)
    assert panel._image_path.name == _name(2, 1), (
        "field 2 has no channel 1 file, so the view stays where it is")
    assert int(panel._image.max()) == 21


def test_a_late_or_foreign_regrouping_is_not_adopted(panel, mixed_plate,
                                                     tmp_path):
    panel.load_image(mixed_plate / _name(1, 1))
    assert panel.regroup_the_folder() is True
    sets_before = list(panel._sampler.sets)
    token = panel._regroup_token
    meta, custom = panel._regex_config()
    panel._adopt_the_regrouping(token - 1, mixed_plate, meta, custom,
                                ([], []))
    assert list(panel._sampler.sets) == sets_before, "an older answer"
    panel._adopt_the_regrouping(token, tmp_path, meta, custom, ([], []))
    assert list(panel._sampler.sets) == sets_before, "another folder"
    assert _headers(panel) == ["ch 0", "ch 1", "image"]


# ---------------------------------------------------------------------------
# windows, cycling
# ---------------------------------------------------------------------------

def test_an_open_settings_window_is_not_stowed_away(panel):
    panel.open_live_settings()
    dialog = panel._live_settings_dialog
    try:
        assert dialog.isWindow() and dialog.parentWidget() is panel
        panel._stow_free_widgets()
        assert dialog.parentWidget() is panel
        assert dialog.isVisible()
    finally:
        dialog.close()


def test_the_cycle_arrows_do_nothing_with_one_object(panel):
    panel._object_box.setCurrentText("cell")
    panel._cycle_index = 0
    panel._cycle_next_btn.click()
    panel._cycle_prev_btn.click()
    assert panel._cycle_index == 0
    assert panel._composite_roles == ()
    assert panel._cycle_label.text() == ""


def test_the_settings_window_shows_every_row_control_when_reopened(
        panel, qtbot):
    panel._object_box.setCurrentText("organelle")
    panel.open_live_settings()
    panel._live_settings_dialog.close()
    qtbot.waitUntil(lambda: panel._live_settings_dialog is None
                    or not panel._live_settings_dialog.isVisible())
    panel._live_settings_dialog = None
    panel.open_live_settings()
    dialog = panel._live_settings_dialog
    try:
        for form in dialog.findChildren(QFormLayout):
            for row in range(form.rowCount()):
                item = form.itemAt(row, QFormLayout.FieldRole)
                widget = item.widget() if item is not None else None
                if widget is not None and form.isRowVisible(row):
                    assert not widget.isHidden(), (
                        f"{widget.objectName() or type(widget).__name__} "
                        "came back as a caption over an empty field")
    finally:
        dialog.close()


# ---------------------------------------------------------------------------
# apply_settings with odd values
# ---------------------------------------------------------------------------

def test_values_no_control_can_hold_do_not_stop_the_rest(panel):
    panel.apply_settings({
        "number_of_organelles": 1,
        "organelle_channel": "three",
        "adjust_cells": np.array([1, 0]),
        "organelle_morphology": "blob",
        "organelle_ridge_filter": "sato",
        "organelle_min_size": "big",
        "cell_diameter": 42.0,
    })
    assert panel._diameter.value() == 42.0, "the usable value landed"
    assert "organelle" not in panel._organelle_channel_values
    widgets = panel._organelle_widgets
    assert LP._combo_value(widgets["morphology"]) == "spots", (
        "an unknown morphology leaves the first one selected")
    assert LP._combo_value(widgets["ridge_filter"]) == "sato"
    assert widgets["min_size"].value() == 0


def test_fewer_organelles_puts_the_picker_back_on_cell(panel):
    panel.apply_settings({"number_of_organelles": 2})
    panel._object_box.setCurrentText("organelle 2")
    assert panel._selected_object_types() == ("organelleb",)
    panel.apply_settings({"number_of_organelles": 1})
    assert LP._combo_value(panel._object_box) == "cell"
    assert panel._object_box.findText("organelle 2") < 0


def test_a_new_morphology_offers_its_own_methods(panel):
    methods = panel._organelle_widgets["method"]
    panel._organelle_widgets["morphology"].setCurrentText("network")
    offered = [methods.itemData(i) or methods.itemText(i)
               for i in range(methods.count())]
    assert "ridge" in offered and "hysteresis" in offered
    assert "log" not in offered


def test_a_morphology_change_survives_a_window_that_cannot_regate(
        panel, monkeypatch):
    panel._object_box.setCurrentText("organelle")
    panel.open_live_settings()
    dialog = panel._live_settings_dialog
    try:
        calls = []

        def _refuse():
            calls.append(1)
            raise RuntimeError("rows are gone")

        monkeypatch.setattr(dialog, "refresh_visibility", _refuse)
        panel._organelle_widgets["morphology"].setCurrentText("ring")
        assert calls == [1]
        methods = panel._organelle_widgets["method"]
        offered = [methods.itemData(i) or methods.itemText(i)
                   for i in range(methods.count())]
        assert "ridge" not in offered and "dog" in offered
    finally:
        monkeypatch.undo()
        dialog.close()


def test_an_organelle_description_becomes_its_tooltip(qtbot, monkeypatch):
    import spacr.settings as settings_module

    monkeypatch.setitem(settings_module.descriptions,
                        "organelle_tophat_radius",
                        "Radius of the white top-hat background removal.")
    widget = LP.LivePreviewPanel(threaded=False)
    qtbot.addWidget(widget)
    assert widget._organelle_widgets["tophat_radius"].toolTip() == (
        "Radius of the white top-hat background removal.")


def test_off_is_zero_when_the_shipped_defaults_cannot_be_read(
        panel, monkeypatch):
    import spacr.settings as settings_module

    def _broken(_settings):
        raise RuntimeError("defaults unavailable")

    monkeypatch.setattr(LP.LivePreviewPanel, "_OFF_IS_NONE", None)
    monkeypatch.setattr(settings_module,
                        "set_default_settings_preprocess_generate_masks",
                        _broken)
    assert panel._keys_whose_off_is_none() == frozenset()
    assert panel._off_as_the_run_spells_it("organelle_max_size", 0) == 0


# ---------------------------------------------------------------------------
# the model
# ---------------------------------------------------------------------------

@pytest.fixture
def slow_plaque_resolver(monkeypatch, tmp_path):
    """A plaque resolver that answers only when told to."""
    checkpoint = tmp_path / "plaque_ckpt"
    checkpoint.write_bytes(b"")
    go = threading.Event()
    fake = types.ModuleType("spacr.submodules")
    fake._requested_plaque_model = lambda s: s.get("plaque_model")

    def _resolve(_settings, fetch=True):
        go.wait(20)
        return str(checkpoint)

    fake._resolve_plaque_model = _resolve
    monkeypatch.setitem(sys.modules, "spacr.submodules", fake)
    return go, checkpoint


def test_a_model_picked_while_the_plaque_model_resolves_is_kept(
        qtbot, slow_plaque_resolver, tmp_path):
    """A chosen checkpoint survives an older resolver without local model caches.

    :param qtbot: ownership and event-loop fixture.
    :param slow_plaque_resolver: delayed checkpoint resolution fixture.
    :param tmp_path: isolated alternate checkpoint destination.
    """
    go, checkpoint = slow_plaque_resolver
    widget = LP.LivePreviewPanel(module="analyze_plaques")
    qtbot.addWidget(widget)
    widget.apply_settings({"plaque_model": "zoo:plaque_v2"})
    box = widget._model_box
    alternate = tmp_path / "chosen_checkpoint"
    alternate.write_bytes(b"independent choice")
    other = str(alternate)
    assert other != box.currentText()
    box.addItem(other)
    box.setCurrentText(other)
    assert widget._model_for_this_pass()[0] == other, (
        "the pass settles the pending seed and keeps the user's pick")
    go.set()
    qtbot.waitUntil(lambda: widget._model_jobs.pending_jobs() == 0,
                    timeout=20000)
    qtbot.wait(50)
    assert box.currentText() == other, "the late answer is not adopted"
    assert box.findText(str(checkpoint)) < 0


def test_a_zoo_pick_the_list_already_holds_is_selected_once(
        panel, monkeypatch):
    from spacr.qt.widgets import model_zoo_picker

    box = panel._model_box
    listed = box.itemText(box.count() - 1)
    box.setCurrentIndex(0)
    count = box.count()
    monkeypatch.setattr(model_zoo_picker, "choose_model",
                        lambda parent, kinds=None: listed)
    panel._choose_a_preview_model()
    assert box.currentText() == listed
    assert box.count() == count


# ---------------------------------------------------------------------------
# results arriving
# ---------------------------------------------------------------------------

class _StillRunning:
    """A newer pass's worker, still in flight."""

    def isRunning(self):
        return True


def test_an_old_worker_finishing_leaves_a_newer_pass_busy(panel):
    panel.set_preview_busy(True)
    panel._worker = _StillRunning()
    try:
        panel._on_worker_finished()
        assert not panel._run_btn.isEnabled()
        assert panel._cancel_btn.isEnabled()
    finally:
        panel._worker = None


def test_a_stale_cell_probability_is_dropped(panel):
    panel._run_token = 4
    fresh = {"cell": np.zeros((4, 4), np.float32)}
    panel._on_cellprob_ready(fresh, token=4)
    panel._on_cellprob_ready({"cell": np.ones((4, 4), np.float32)}, token=3)
    assert panel._cellprob.keys() == fresh.keys()
    assert float(panel._cellprob["cell"].max()) == 0.0


def test_an_enhancement_chain_is_named_on_the_status_line(panel, tmp_path):
    path = tmp_path / "field.tif"
    tifffile.imwrite(path, np.full((16, 16), 100, np.uint16))
    panel.load_image(path)
    panel._on_processing_provenance({
        "processing": {"operation": "none"},
        "enhancement": {"order": ["denoise", "deconvolve"],
                        "denoise": {}, "deconvolve": {}},
    })
    mask = np.zeros((16, 16), np.int32)
    mask[4:10, 4:10] = 1
    panel._on_worker_done({"cell": mask}, "")
    status = panel._status.text()
    assert "Enhancement: denoise, deconvolve (preview field)." in status


# ---------------------------------------------------------------------------
# the mask comparison
# ---------------------------------------------------------------------------

def _mask(corner, shape=(16, 16)):
    mask = np.zeros(shape, np.int32)
    mask[corner:corner + 5, corner:corner + 5] = 1
    return mask


def test_with_no_field_there_is_nothing_to_compare(panel):
    assert panel.comparable_masks() == []
    assert panel.comparison_layers() == []
    assert panel.open_mask_comparison() is False
    assert panel._compare_view.isHidden()


def test_a_multichannel_field_offers_each_channel_unticked(panel):
    panel._image = np.random.RandomState(1).randint(
        0, 255, (16, 16, 2)).astype(np.uint16)
    panel._snapshot_run({"cell": _mask(2)}, ["cell=1"])
    panel._snapshot_run({"cell": _mask(8)}, ["cell=1"])
    layers = panel.comparison_layers()
    names = [layer.name for layer in layers]
    assert names[-2:] == ["Channel 1", "Channel 2"]
    assert [layer.ticked for layer in layers[-2:]] == [False, False]
    assert names[-3] == "Field as shown"


def test_dismissing_the_comparison_popup_draws_nothing(panel, monkeypatch):
    from spacr.qt.widgets.mask_comparison import MaskComparisonDialog

    panel._image = np.zeros((16, 16), np.uint16)
    panel._snapshot_run({"cell": _mask(2)}, ["cell=1"])
    panel._snapshot_run({"cell": _mask(8)}, ["cell=1"])
    shown = []

    def _dismissed(dialog):
        shown.append([tick.text() for tick, _ in dialog.rows()])
        return int(QDialog.Rejected.value)

    monkeypatch.setattr(MaskComparisonDialog, "exec", _dismissed)
    assert panel.open_mask_comparison() is False
    assert len(shown) == 1 and len(shown[0]) == 3, (
        "two masks and the field were offered")
    assert panel._compare_view.isHidden()


def test_accepting_the_comparison_popup_draws_the_third_panel(
        panel, monkeypatch):
    """``exec`` answers with an int. The panel compared it against
    ``dialog.Accepted``, which a PySide6 dialog *instance* does not have, so
    closing the popup either way raised AttributeError instead of drawing."""
    from spacr.qt.widgets.mask_comparison import MaskComparisonDialog

    panel._image = np.zeros((16, 16), np.uint16)
    panel._snapshot_run({"cell": _mask(2)}, ["cell=1"])
    panel._snapshot_run({"cell": _mask(8)}, ["cell=1"])
    monkeypatch.setattr(MaskComparisonDialog, "exec",
                        lambda dialog: int(QDialog.Accepted.value))
    assert panel.open_mask_comparison() is True
    assert not panel._compare_view.isHidden()
    assert panel._compare_view.picture() is not None
