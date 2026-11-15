"""The Measure crop preview when the array, the folder or the theme misbehave.

The panel's happy path -- drop a merged array, get a grid of crops -- is
covered elsewhere. What is exercised here is everything the panel promises to
survive: an unreadable ``.npy``, a folder it cannot group, a palette lookup
that raises, a stale worker result arriving after the user moved on, and each
of the sentences it puts on the status line instead of drawing nothing.
"""
from __future__ import annotations

import numpy as np
import pytest
from PySide6.QtCore import QMimeData, QPoint, QPointF, Qt, QUrl
from PySide6.QtGui import (
    QDragEnterEvent,
    QDragMoveEvent,
    QDropEvent,
    QPixmap,
)

from spacr.qt.widgets import measure_preview as MP


def _merged(tmp_path, name="plate1_A01_f1.npy", *, planes=8):
    """A merged array with cell, nucleus, pathogen and organelle planes."""
    data = np.zeros((48, 48, planes), np.float32)
    data[..., :3] = 20
    cell = np.zeros((48, 48), np.int32)
    nucleus = np.zeros_like(cell)
    pathogen = np.zeros_like(cell)
    organelle = np.zeros_like(cell)
    cell[2:18, 2:18] = 1
    nucleus[5:10, 5:10] = 1
    organelle[11:14, 11:14] = 1
    cell[24:42, 24:42] = 2
    pathogen[28:33, 28:33] = 1
    data[..., 4] = cell
    data[..., 5] = nucleus
    data[..., 6] = pathogen
    data[..., 7] = organelle
    path = tmp_path / name
    np.save(path, data)
    return str(path)


@pytest.fixture(autouse=True)
def _application(qapp):
    """Every Qt object built here needs the shared application to exist."""
    return qapp


@pytest.fixture
def panel(qtbot):
    """An unthreaded panel, so every job lands before the call returns."""
    widget = MP.MeasurePreviewPanel(threaded=False)
    qtbot.addWidget(widget)
    return widget


# ---------------------------------------------------------------------------
# presence and categories
# ---------------------------------------------------------------------------

def test_a_minimum_size_makes_a_speck_of_signal_not_count():
    """Below the minimum, an overlapping label is not presence."""
    mask = np.zeros((10, 10), np.int32)
    mask[0:1, 0:2] = 1
    region = np.ones((10, 10), bool)
    data = np.zeros((10, 10, 2), np.float32)
    data[..., 1] = mask

    assert MP._presence_in(data, 1, region, 0) is True
    assert MP._presence_in(data, 1, region, 2) is True
    assert MP._presence_in(data, 1, region, 3) is False


def test_a_non_cell_object_is_labelled_by_its_own_name_and_kept():
    """Only cells get phenotype categories; the rest are named and included."""
    crops = [{"label": 1}, {"label": 2}]
    MP.annotate_crops(crops, None, {"object": "nucleus"})
    assert [c["category"] for c in crops] == ["Nucleus", "Nucleus"]
    assert all(c["included"] for c in crops)


def test_a_cell_dim_past_the_array_leaves_the_crops_uncategorised():
    """There is no cell mask to read, so no phenotype can be claimed."""
    data = np.zeros((8, 8, 2), np.float32)
    crops = [{"label": 1, "bbox": (0, 0, 4, 4)}]
    MP.annotate_crops(crops, data, {"object": "cell", "cell_dim": 5})
    assert "category" not in crops[0]
    MP.annotate_crops(crops, data, {"object": "cell", "cell_dim": None})
    assert "category" not in crops[0]


def test_a_crop_pass_that_raises_is_reported_as_a_crop_failure():
    """The worker returns the message rather than letting the thread die."""
    import spacr.measure as measure_module

    original = measure_module.crop_objects_from_array

    def boom(*args, **kwargs):
        raise ValueError("mask_dim is out of range")

    measure_module.crop_objects_from_array = boom
    try:
        result = MP.compute_crops(np.zeros((4, 4, 2), np.float32),
                                  {"mask_dim": 1}, {})
    finally:
        measure_module.crop_objects_from_array = original
    assert result["crops"] == []
    assert result["error"] == "Crop failed: mask_dim is out of range"


def test_a_null_pixmap_is_returned_unrounded():
    """There is no corner to round on an image that does not exist."""
    empty = QPixmap()
    assert MP._rounded_pixmap(empty) is empty


def test_a_png_mapping_that_cannot_be_resolved_falls_back_to_the_default(
        monkeypatch):
    """A settings dict the resolver rejects still yields a drawable mapping."""
    import spacr.crops as crops_module

    def boom(settings):
        raise KeyError("png_channel_mapping")

    monkeypatch.setattr(crops_module, "resolve_png_channel_mapping", boom)
    assert MP._resolve_png_mapping({}) == MP._default_png_mapping()


# ---------------------------------------------------------------------------
# thumbnails
# ---------------------------------------------------------------------------

def test_a_thumbnail_still_draws_when_the_palette_cannot_be_read(
        qtbot, monkeypatch):
    """A theme lookup that raises falls back to fixed colours, not to nothing."""
    import spacr.qt.theme as theme_module

    monkeypatch.setattr(theme_module, "active_palette",
                        lambda *a, **k: (_ for _ in ()).throw(
                            RuntimeError("no palette yet")))
    thumb = MP._CropThumb(3, included=False)
    qtbot.addWidget(thumb)
    assert "#4A9EFF" in thumb.styleSheet() or "#24262a" in thumb.styleSheet()


def test_clicking_a_thumbnail_announces_which_one_it_was(qtbot):
    """The index travels with the click so the panel can toggle selection."""
    thumb = MP._CropThumb(7)
    qtbot.addWidget(thumb)
    seen = []
    thumb.clicked.connect(seen.append)
    with qtbot.waitSignal(thumb.clicked):
        qtbot.mouseClick(thumb, Qt.LeftButton)
    assert seen == [7]


# ---------------------------------------------------------------------------
# drag and drop
# ---------------------------------------------------------------------------

def _urls(paths):
    """A mime payload of local file URLs.

    The caller keeps the reference: PySide6 drag events do not own the
    :class:`QMimeData` they are handed, and letting it be collected while the
    event is alive segfaults the interpreter.
    """
    mime = QMimeData()
    mime.setUrls([QUrl.fromLocalFile(str(p)) for p in paths])
    return mime


def test_a_drag_of_something_unsupported_is_refused(panel, tmp_path):
    """Only the suffixes this panel can read are accepted."""
    stray = tmp_path / "notes.txt"
    stray.write_text("nothing to preview", encoding="utf-8")
    mime = _urls([stray])

    enter = QDragEnterEvent(QPoint(1, 1), Qt.CopyAction, mime,
                            Qt.LeftButton, Qt.NoModifier)
    panel.dragEnterEvent(enter)
    assert not enter.isAccepted()

    move = QDragMoveEvent(QPoint(1, 1), Qt.CopyAction, mime,
                          Qt.LeftButton, Qt.NoModifier)
    panel.dragMoveEvent(move)
    assert not move.isAccepted()

    dropped = QDropEvent(QPointF(1, 1), Qt.CopyAction, mime,
                         Qt.LeftButton, Qt.NoModifier)
    panel.dropEvent(dropped)
    assert not dropped.isAccepted()
    assert panel._data is None


def test_a_drag_with_no_urls_at_all_is_refused(panel):
    """Text dragged in from elsewhere is not a path."""
    mime = QMimeData()
    mime.setText("/somewhere/else.npy")
    enter = QDragEnterEvent(QPoint(1, 1), Qt.CopyAction, mime,
                            Qt.LeftButton, Qt.NoModifier)
    panel.dragEnterEvent(enter)
    assert not enter.isAccepted()
    assert panel._dropped_path(enter) is None


def test_a_dropped_merged_array_is_accepted_and_loaded(panel, tmp_path):
    """The first supported URL in the drop is the one that gets read."""
    path = _merged(tmp_path)
    mime = _urls([tmp_path / "notes.txt", path])

    dropped = QDropEvent(QPointF(1, 1), Qt.CopyAction, mime,
                         Qt.LeftButton, Qt.NoModifier)
    panel.dropEvent(dropped)
    assert dropped.isAccepted()
    assert panel._data is not None
    assert panel._data_path == path

    move = QDragMoveEvent(QPoint(1, 1), Qt.CopyAction, mime,
                          Qt.LeftButton, Qt.NoModifier)
    panel.dragMoveEvent(move)
    assert move.isAccepted()


# ---------------------------------------------------------------------------
# loading, in flight and out of order
# ---------------------------------------------------------------------------

def test_an_empty_path_submits_no_load(panel):
    """There is nothing to queue, so no token is spent."""
    before = panel._load_token
    assert panel.load_array_async("") is False
    assert panel.load_array_async(None) is False
    assert panel._load_token == before


# ---------------------------------------------------------------------------
# the source selectors
# ---------------------------------------------------------------------------


def test_choosing_the_field_of_view_already_loaded_does_not_reload_it(
        panel, tmp_path):
    """Re-selecting the current entry is not a request for a new read."""
    panel.load_array(_merged(tmp_path))
    before = panel._load_token
    panel._on_fov_changed()
    assert panel._load_token == before

    panel._loading_fov = True
    try:
        panel._on_fov_changed()
        assert panel._load_token == before
    finally:
        panel._loading_fov = False


# ---------------------------------------------------------------------------
# settings in and out
# ---------------------------------------------------------------------------

def test_naming_fewer_than_three_png_channels_blanks_the_rest(panel):
    """An unnamed plane is removed, not left holding its default."""
    panel._png_dims.set_value({"r": 2, "g": None, "b": None})
    assert panel._png_channel_mapping() == {"r": 2, "g": None, "b": None}


def test_a_propagation_callback_that_raises_does_not_take_the_panel_down(panel):
    """The settings screen's problem is not the preview's to die of.

    The callback is really reached, with the settings the panel would have
    sent -- and it keeps being reached afterwards, because a panel that
    quietly unwired a callback that once threw would stop propagating for
    the rest of the session without ever saying so.
    """
    sent = []

    def boom(settings):
        sent.append(settings)
        raise RuntimeError("the settings screen is gone")

    panel.set_propagate_callback(boom)
    panel.propagate_settings()
    assert sent == [panel.settings_for_propagation()]

    before = len(sent)
    panel._propagate_btn.setChecked(True)
    panel._maybe_propagate()
    assert len(sent) > before


def test_previewing_an_object_also_asks_for_it_as_a_crop_output(panel):
    """A previewed object the run would not write is a preview of nothing."""
    assert panel._crop_mode_checks["nucleus"].isChecked() is False
    panel._on_object_changed("nucleus")
    assert panel._crop_mode_checks["nucleus"].isChecked() is True


def test_the_cytoplasm_preview_borrows_the_cell_footprint(panel):
    """Cytoplasm is derived during measurement and has no input slice."""
    panel._object_box.setCurrentText("cytoplasm")
    panel._mask_dims["cell"].setValue(4)
    assert panel._current_mask_dim() == 4


def test_presence_is_unknown_until_an_array_is_loaded(panel, tmp_path):
    """With nothing loaded there is no mask to look in; with one, there is."""
    assert panel._presence("nucleus", np.ones((4, 4), bool)) is None
    assert panel._phenotype_text("nucleus", None) == MP._phenotype_label(
        "nucleus", None)

    panel.load_array(_merged(tmp_path))
    first_cell = np.zeros((48, 48), bool)
    first_cell[2:18, 2:18] = True
    second_cell = np.zeros((48, 48), bool)
    second_cell[24:42, 24:42] = True
    assert panel._presence("nucleus", first_cell) is True
    assert panel._presence("nucleus", second_cell) is False
    assert panel._presence("pathogen", second_cell) is True


# ---------------------------------------------------------------------------
# the preview pass
# ---------------------------------------------------------------------------


def test_a_crop_pass_that_failed_announces_an_empty_result(panel, qtbot):
    """``preview_ready`` fires with None so listeners are not left waiting."""
    seen = []
    panel.preview_ready.connect(seen.append)
    panel._on_crops_ready(panel._crop_token,
                          {"crops": [], "error": "Crop failed: no mask"})
    assert panel._status.text() == "Crop failed: no mask"
    assert seen == [None]


def test_a_header_still_renders_when_the_palette_cannot_be_read(
        panel, monkeypatch):
    """Losing the theme costs the styling, not the heading."""
    import spacr.qt.theme as theme_module

    monkeypatch.setattr(theme_module, "active_palette",
                        lambda *a, **k: (_ for _ in ()).throw(
                            RuntimeError("no palette yet")))
    label = panel._category_header("Infected", [(0, {}), (1, {})])
    assert "Infected" in label.text()
    assert "2" in label.text()


def test_clicking_the_same_thumbnail_twice_deselects_it(panel, tmp_path):
    """Selection is a toggle, and the status line counts what is selected."""
    panel.load_array(_merged(tmp_path))
    panel.refresh()
    panel._on_thumb_clicked(0)
    assert panel._selected == {0}
    assert "selected" in panel._status.text()
    panel._on_thumb_clicked(0)
    assert panel._selected == set()


# ---------------------------------------------------------------------------
# the crop settings dialog
# ---------------------------------------------------------------------------

def test_the_crop_settings_dialog_is_raised_rather_than_opened_twice(
        panel, qtbot):
    """A second press brings the open dialog forward; it does not stack one."""
    panel.open_crop_settings()
    first = panel._crop_settings_dialog
    assert first is not None
    qtbot.addWidget(first)
    qtbot.waitUntil(first.isVisible)

    panel.open_crop_settings()
    assert panel._crop_settings_dialog is first

    first.close()
    panel._clear_crop_settings_dialog()
    assert panel._crop_settings_dialog is None


# ---------------------------------------------------------------------------
# building the panel when the theme or a control will not cooperate
# ---------------------------------------------------------------------------

def test_the_panel_builds_without_a_palette_to_read(qtbot, monkeypatch):
    """The scroll area falls back to a fixed background rather than failing."""
    import spacr.qt.theme as theme_module

    monkeypatch.setattr(theme_module, "active_palette",
                        lambda *a, **k: (_ for _ in ()).throw(
                            RuntimeError("no palette yet")))
    widget = MP.MeasurePreviewPanel(threaded=False)
    qtbot.addWidget(widget)
    assert "#161719" in widget._grid_scroll.viewport().styleSheet()


class _RefusingSignal:
    """A signal that will not accept this slot, as Qt's do on a bad match."""

    def connect(self, _slot):
        raise TypeError("cannot connect this signal to that slot")


class _RecordingSignal:
    def __init__(self):
        self.slots = []

    def connect(self, slot):
        self.slots.append(slot)


class _AwkwardControl:
    """A control whose first-choice signal refuses the panel's slot."""

    def __init__(self):
        self.valueChanged = _RefusingSignal()
        self.currentTextChanged = _RefusingSignal()
        self.editingFinished = _RefusingSignal()
        self.toggled = _RecordingSignal()


def test_a_control_that_refuses_one_signal_is_wired_through_the_next(panel):
    """Connection walks the signal list; a refusal is not the end of it."""
    filtering = _AwkwardControl()
    propagating = _AwkwardControl()
    panel._min_sizes["cell"] = filtering
    panel._cytoplasm = propagating

    panel._connect_controls()

    assert panel._on_setting_changed in filtering.toggled.slots
    assert panel._maybe_propagate in propagating.toggled.slots
