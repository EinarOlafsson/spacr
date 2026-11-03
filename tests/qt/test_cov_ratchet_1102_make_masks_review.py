"""Screen-level arms of Make Masks: channel and role choice, layer events, room folding."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("PySide6")

import shiboken6
from PySide6.QtWidgets import QCheckBox, QComboBox, QLabel, QWidget

from spacr.qt.screens import make_masks as mm

pytestmark = pytest.mark.qt

Screen = mm.MakeMasksScreen


# -- the mask channel and role ------------------------------------------------

def test_choosing_the_channel_already_edited_or_with_nothing_open_changes_nothing():
    shown = []
    fake = SimpleNamespace(_field_channels=None, _canvas=SimpleNamespace(mask=np.zeros((2, 2))),
                           _mask_channel_combo=SimpleNamespace(itemData=lambda index: 1),
                           _mask_channel="kept",
                           _review_another_channel=lambda *a: shown.append(a))
    Screen._on_mask_channel_chosen(fake, 1)
    fake._field_channels = SimpleNamespace(channel=1, view=object())
    Screen._on_mask_channel_chosen(fake, -1)              # no row chosen
    Screen._on_mask_channel_chosen(fake, 1)               # the channel on screen
    assert shown == [] and fake._mask_channel == "kept"
    Screen._on_mask_channel_chosen(fake, 0)               # the same data, another row
    fake._mask_channel_combo = SimpleNamespace(itemData=lambda index: 2)
    Screen._on_mask_channel_chosen(fake, 0)
    assert [call[1] for call in shown] == [2]


def _reviewing(saved=True, **over):
    log = SimpleNamespace(warned=[], synced=0, loaded=0)
    view = {"header": {"path": "/f/a.tif"}, "series": 0, "time": 0, "z": 0, "axes": "YX"}
    fake = SimpleNamespace(
        _save_mask_if_needed=lambda: saved, _save_boxes_if_needed=lambda: True,
        _sync_channel_controls=lambda: setattr(log, "synced", log.synced + 1),
        _warn=lambda title, text: log.warned.append(text),
        _load_current=lambda: setattr(log, "loaded", log.loaded + 1),
        _queue_masks_dir=None, _mask_channel=0)
    fake.__dict__.update(over)
    return fake, SimpleNamespace(view=view), log


def test_another_channel_is_not_opened_when_the_edits_cannot_be_saved(monkeypatch):
    monkeypatch.setattr(mm.engine, "_set_container_image_view",
                        lambda *a, **k: pytest.fail("opened despite the failed save"))
    fake, channels, log = _reviewing(saved=False)
    Screen._review_another_channel(fake, channels, 1)
    assert log.synced == 1 and log.loaded == 0 and fake._mask_channel == 0


def test_another_channel_the_reader_refuses_is_reported_and_the_old_one_stays(monkeypatch):
    def refuse(*args, **kwargs):
        raise ValueError("That channel cannot be read.")

    monkeypatch.setattr(mm.engine, "_set_container_image_view", refuse)
    fake, channels, log = _reviewing()
    Screen._review_another_channel(fake, channels, 1)
    assert log.warned == ["That channel cannot be read."]
    assert log.synced == 1 and log.loaded == 0 and fake._mask_channel == 0
    monkeypatch.setattr(mm.engine, "_set_container_image_view", lambda *a, **k: {})
    Screen._review_another_channel(fake, channels, 1)
    assert fake._mask_channel == 1 and log.loaded == 1


def test_a_role_chosen_with_no_mask_layer_on_screen_only_names_the_role():
    combo = SimpleNamespace(itemData=lambda index: "nucleus")
    fake = SimpleNamespace(_mask_role_combo=combo, _mask_role="object",
                           _canvas=SimpleNamespace(mask_layer=None), _field_stack=[])
    Screen._on_mask_role_chosen(fake, 0)
    assert fake._mask_role == "nucleus"
    fake._canvas = SimpleNamespace(mask_layer=object(), color="kept")
    Screen._on_mask_role_chosen(fake, 0)                   # a layer the stack no longer holds
    assert fake._canvas.color == "kept"


# -- layer events ------------------------------------------------------------------

def test_a_layer_added_or_removed_does_not_restyle_or_schedule_a_repaint():
    canvas = SimpleNamespace(layered=False, layers_version=0, mask_layer=object())
    fake = SimpleNamespace(_building_layers=False, _layers_custom=False,
                           _layers_refresh_pending=False, _canvas=canvas)
    for kind in ("inserted", "removed", "data", "selected"):
        Screen._on_field_layers_changed(fake, SimpleNamespace(kind=kind))
    assert fake._layers_custom is False and canvas.layers_version == 0
    assert fake._layers_refresh_pending is False


def test_a_repaint_due_after_the_canvas_is_gone_is_dropped(qapp):
    canvas = QWidget()
    shiboken6.delete(canvas)
    fake = SimpleNamespace(_layers_refresh_pending=True, _canvas=canvas)
    Screen._apply_layers_refresh(fake)
    assert fake._layers_refresh_pending is False


def test_rebuilding_the_layers_of_a_field_with_no_image_leaves_an_empty_stack():
    from spacr.layers import ImageLayer, LayerStack

    stack = LayerStack()
    stack.append(ImageLayer(np.zeros((2, 2), np.uint16), name="old"))
    canvas = SimpleNamespace(image=None, mask=None, layers_version=0, refresh=lambda: None)
    synced = []
    fake = SimpleNamespace(_field_stack=stack, _canvas=canvas, _field_channels=None,
                           _layers_custom=False, _building_layers=False,
                           _sync_channel_controls=lambda: synced.append(True))
    Screen._rebuild_field_layers(fake)
    assert len(stack) == 0 and canvas.mask_layer is None and canvas.layered is False
    assert canvas.layers_version == 1 and synced == [True] and fake._building_layers is False


# -- small pure arms ------------------------------------------------------------------

def test_a_contrast_preset_without_enhancement_controls_sets_only_the_display_range(monkeypatch):
    values = {}

    class Box:
        def __init__(self, name):
            self.name = name

        def setValue(self, value):
            values[self.name] = value

        def setChecked(self, value):
            values[self.name] = value

    preset = {"lower": 1, "upper": 99, "invert": True, "clahe_tile": 8,
              "clahe_clip": 2.0, "clahe": True}
    monkeypatch.setattr(mm.engine, "CONTRAST_PRESETS", {"strong": preset})
    status = []
    fake = SimpleNamespace(
        _contrast_preset=SimpleNamespace(currentData=lambda: "strong", currentText=lambda: "Strong"),
        _norm_lo=Box("lo"), _norm_hi=Box("hi"), _invert_display=Box("invert"),
        _status_label=SimpleNamespace(setText=status.append))
    Screen._on_contrast_preset(fake, 0)
    assert values == {"lo": 1.0, "hi": 99.0, "invert": True}
    assert status == ["Contrast preset applied: Strong."]


def test_the_saved_mark_is_skipped_when_the_annotation_group_has_none():
    fake = SimpleNamespace(_saved_state=None, _mask_is_saved=lambda: pytest.fail("asked"))
    assert Screen._refresh_saved_state(fake) is None
    label = QLabel()
    fake = SimpleNamespace(_saved_state=label, _mask_is_saved=lambda: True)
    Screen._refresh_saved_state(fake)
    assert label.isHidden() is False


def test_restoring_a_row_that_left_the_box_forgets_it_quietly(qapp):
    box = QComboBox()
    box.addItem("otsu", "otsu")
    gated = {"gone": "was 3D only", "otsu": "tip"}
    fake = SimpleNamespace(_detect_3d_gated={"mode": gated})
    Screen._gate_3d_rows(fake, box, "mode", False, lambda value: True, "{name} is 2D only")
    assert gated == {} and box.itemData(0, mm.Qt.ToolTipRole) == "tip"
    assert box.model().item(0).isEnabled()


def test_the_3d_switch_without_a_stitch_box_only_gates_the_method_rows(qapp):
    toggle = QCheckBox()
    gates = []
    fake = SimpleNamespace(
        _detect_3d=toggle, _cp_model=SimpleNamespace(currentData=lambda: "cpsam"),
        _mask_stage=SimpleNamespace(session=None), _detect_3d_on=lambda: False,
        _gate_3d_rows=lambda box, key, on, supported, reason: gates.append((key, on)),
        _mag_mode=SimpleNamespace(currentData=lambda: "otsu"),
        _sync_otsu_controls=lambda: gates.append("otsu"))
    Screen._sync_detect_3d(fake)
    assert gates == [("mode", False), ("model", False), "otsu"]
    assert not toggle.isEnabled()


def test_a_keep_whose_mask_cannot_be_saved_records_no_verdict():
    stage = SimpleNamespace(session=object(), dirty=True, busy=False, save=lambda: False)
    fake = SimpleNamespace(_mask_stage=stage,
                           _record_nothing=None,
                           _image_files=["a.tif"], _canvas=SimpleNamespace(mask=np.zeros((2, 2))),
                           _save_mask_if_needed=lambda: pytest.fail("went on to the verdict"))
    assert Screen._on_curate(fake, True) is None


def test_a_gpu_backend_that_failed_with_nothing_drawn_yet_falls_back_to_the_cpu():
    said, closed = [], []
    broken = SimpleNamespace(backend="opengl", close=lambda: closed.append(True))
    fake = SimpleNamespace(_gl=broken, _gl_tried=False, _render_state="x", _render_artist=None,
                           _gpu_refused=set(), _backend_choice="cpu", _backend_note="",
                           _say=said.append, _sync_scale_slider=lambda: None,
                           _reason=mm._VolumeStage._reason)
    mm._VolumeStage._gpu_failed(fake, RuntimeError("no context\nsecond line"))
    assert fake._gl is None and fake._gpu_refused == {"opengl"} and closed == [True]
    assert fake._render_state is None
    assert "no context" in fake._backend_note and "second line" not in fake._backend_note
    assert said == [fake._backend_note]


def test_a_cube_line_of_sight_that_misses_the_volume_gives_no_point():
    origin = np.array([0.0, 100.0, 0.0])
    # a line of sight almost along the slice plane, leaving the volume at two different places
    matrix = np.linalg.inv(np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [0.0, 0.0, 0.01]]))
    fake = SimpleNamespace(
        _view_axes=lambda: "axes", _projection=lambda ax: (origin, matrix, np.zeros(3)),
        session=SimpleNamespace(limits=[(0.0, 10.0)] * 3, xyz_spacing=(1.0, 1.0, 1.0)),
        _cursor=(0, 0, 0))
    assert mm._VolumeStage._cube_point(fake, 0.0, 0.0) is None
