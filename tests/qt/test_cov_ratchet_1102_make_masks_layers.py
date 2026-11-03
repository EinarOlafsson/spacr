"""Defensive arms of Make Masks that the layered-canvas and layered-field work added."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6.QtGui import QPixmap  # noqa: E402
from PySide6.QtWidgets import QLabel  # noqa: E402

from spacr.qt.screens import make_masks as mm

pytestmark = pytest.mark.qt

Canvas = mm._MaskCanvas
Screen = mm.MakeMasksScreen


# -- the layered composite ---------------------------------------------------

def test_a_worker_composite_falls_back_to_the_plain_render_when_nothing_is_gathered():
    rgb = np.zeros((4, 5, 3), np.uint8)
    rgb[..., 0] = 200
    rgb[..., 2] = 7
    stack = SimpleNamespace(_render_uint8_gathered=lambda canvas: None,
                            render_uint8=lambda canvas: rgb)
    picture = mm._layers_picture_for(SimpleNamespace(stack=stack, canvas=object()))
    assert picture.shape == (4, 5) and picture.dtype == np.uint32
    assert int(picture[0, 0]) == (0xFF << 24) | (200 << 16) | 7


def test_a_snapshot_for_the_worker_leaves_out_layers_it_cannot_draw():
    from spacr.layers import ImageLayer, LabelsLayer, LayerStack, PointsLayer

    stack = LayerStack()
    stack.append(ImageLayer(np.arange(12, dtype=np.uint16).reshape(3, 4), name="image"))
    labels = LabelsLayer(np.zeros((3, 4), np.uint16), name="mask")
    stack.append(labels)
    stack.append(PointsLayer(np.array([[1.0, 1.0]]), name="points"))
    copy = mm._layer_snapshot(stack, labels, np.ones((3, 4), np.uint16))
    assert copy.names == ("image", "mask")
    assert int(copy["mask"].data.max()) == 1


def test_the_layer_canvas_samples_only_the_labels_chosen_for_display():
    from spacr.layers import LabelsLayer

    mask = np.array([[1, 2], [2, 3]], np.uint16)
    layer = LabelsLayer(mask.copy(), name="mask")
    fake = SimpleNamespace(draws_layers=lambda: True, mask_layer=layer, mask=mask,
                           display_ids=[2])
    canvas = Canvas._layered_canvas(fake, (0, 0, 2, 2), 1, (2, 2))
    assert canvas is not None
    np.testing.assert_array_equal(layer.data, [[0, 2], [2, 0]])
    assert Canvas._layered_canvas(SimpleNamespace(draws_layers=lambda: False),
                                  (0, 0, 2, 2), 1, (2, 2)) is None


def test_a_piece_of_the_layers_is_rendered_plainly_when_nothing_is_gathered():
    rgb = np.zeros((6, 6, 3), np.uint8)
    rgb[..., 1] = 9
    stack = SimpleNamespace(_render_uint8_gathered=lambda canvas, rows, cols: None,
                            render_uint8=lambda canvas: rgb)
    fake = SimpleNamespace(field_layers=stack)
    picture = Canvas._layered_picture(fake, object(), slice(1, 3), slice(2, 5))
    assert picture.shape == (2, 3)
    assert int(picture[0, 0]) == (0xFF << 24) | (9 << 8)


def test_a_kept_composite_is_reused_only_when_it_can_be_patched():
    key = (object(), object(), None, 5)
    shown = np.zeros((8, 8), np.uint16)
    picture = np.zeros((8, 8), np.uint32)
    kept = (key, None, picture, shown, None)
    fake = SimpleNamespace(field_layers=[], _layered_picture=lambda *a, **k: None)
    # nothing kept, or a kept tuple of the old shape
    assert Canvas._recompose_layered(fake, None, key, None, shown) is None
    assert Canvas._recompose_layered(fake, kept[:4], key, None, shown) is None
    # a different layer version
    assert Canvas._recompose_layered(fake, ((key[0], key[1], None, 6), None, picture, shown, None),
                                     key, None, shown) is None
    # no label changed: the kept picture and labels come back as they are
    again = Canvas._recompose_layered(fake, kept, key, None, shown.copy())
    assert again[0] is picture and again[1] is shown
    # most of the view changed: compositing it all is cheaper
    assert Canvas._recompose_layered(fake, kept, key, None, np.ones((8, 8), np.uint16)) is None


def test_a_layer_composite_that_failed_is_logged_and_one_for_a_closed_canvas_is_dropped(caplog):
    emitted = []
    fake = SimpleNamespace(layers_ready=SimpleNamespace(emit=emitted.append))
    with caplog.at_level("WARNING"):
        Canvas._layers_done(fake, "request", None, RuntimeError("boom"))
    assert emitted == [] and "could not be composited" in caplog.text

    def gone(_payload):
        raise RuntimeError("Internal C++ object already deleted.")

    closed = SimpleNamespace(layers_ready=SimpleNamespace(emit=gone))
    Canvas._layers_done(closed, "request", "picture", None)        # swallowed
    Canvas._layers_done(fake, "request", "picture", None)
    assert emitted == [("request", "picture")]


def _taking(serial=3, composed_same=True, mask=None, display_ids=None, key=3):
    shown = np.zeros((4, 4), np.uint16)
    picture = np.full((4, 4), 0xFF000000, np.uint32)
    request = SimpleNamespace(key=key, composed=("then",), shown=shown, size=(4, 4))
    calls = []
    fake = SimpleNamespace(
        _layers_serial=serial, image=np.zeros((4, 4)),
        mask=np.zeros((4, 4), np.uint16) if mask is None else mask,
        enhance_display=False, displayed_source=lambda: "source",
        _viewport_bounds=lambda: (0, 0, 4, 4),
        _composed_key=lambda source, step: ("now", step),
        _same_composition=lambda then, now: composed_same,
        _show_composed=lambda qimg, w, h: calls.append(("show", w, h)),
        refresh=lambda: calls.append("refresh"),
        display_ids=display_ids, _composed=None)
    return fake, request, picture, calls


def test_a_finished_layer_picture_is_shown_only_while_it_still_is_the_view(monkeypatch):
    monkeypatch.setattr(mm, "device_ratio", lambda widget: 1.0)
    # a newer request has replaced it
    fake, request, picture, calls = _taking(serial=4, key=3)
    Canvas._take_layers(fake, (request, picture))
    assert calls == [] and fake._composed is None
    # the view changed while it was drawn
    fake, request, picture, calls = _taking(composed_same=False)
    Canvas._take_layers(fake, (request, picture))
    assert calls == [] and fake._composed is None
    # the mask is shown through its display ids, and an edit made meanwhile redraws
    mask = np.array([[1, 2, 0, 0]] * 4, np.uint16)
    fake, request, picture, calls = _taking(mask=mask, display_ids=[2])
    Canvas._take_layers(fake, (request, picture))
    assert fake._composed[1].width() == 4 and fake._composed[3] is request.shown
    assert calls == [("show", 4, 4), "refresh"]
    # nothing changed meanwhile: no redraw
    fake, request, picture, calls = _taking()
    Canvas._take_layers(fake, (request, picture))
    assert calls == [("show", 4, 4)]


# -- painting layers ------------------------------------------------------------

def _paintable(qapp, **over):
    pixmap = QPixmap(8, 8)
    fake = SimpleNamespace(layer_overlay=SimpleNamespace(width=lambda: 8, height=lambda: 8),
                           mask=np.zeros((8, 8), np.uint16), pixmap=lambda: pixmap,
                           layers_hold_overlay=False, draws_layers=lambda: False)
    fake.__dict__.update(over)
    return fake


def test_the_outline_overlay_is_not_painted_over_layers_or_over_another_size(qapp):
    held = _paintable(qapp, layers_hold_overlay=True, draws_layers=lambda: True)
    assert Canvas._paint_layers(held) is None
    wrong_size = _paintable(
        qapp, layer_overlay=SimpleNamespace(width=lambda: 3, height=lambda: 3))
    assert Canvas._paint_layers(wrong_size) is None


def test_outlines_at_screen_resolution_are_not_painted_when_none_can_be_drawn(qapp, monkeypatch):
    monkeypatch.setattr(mm, "device_ratio", lambda widget: 1.0)
    monkeypatch.setattr(mm, "logical_size", lambda pixmap: pixmap.size())
    monkeypatch.setattr(mm.engine, "_outline_thickness", lambda value: 3.0)
    asked = []
    fake = _paintable(
        qapp, width=lambda: 8, height=lambda: 8, outline_thickness=3.0,
        layer_source=(1, [], None, None, None),
        _viewport_bounds=lambda: (0, 0, 8, 8),
        _outline_picture=lambda bounds, size, thickness: asked.append((bounds, size, thickness)))
    assert Canvas._paint_layers(fake) is None
    assert asked == [((0, 0, 8, 8), (8, 8), 3.0)]


def test_an_outline_view_with_no_outlines_is_remembered_as_empty(monkeypatch):
    monkeypatch.setattr(mm.engine, "_layer_outline_view", lambda *a, **k: None)
    fake = SimpleNamespace(layer_source=(2, [], None, None, None), _outline_view=None)
    assert Canvas._outline_picture(fake, (0, 0, 4, 4), (4, 4), 2.0) is None
    assert fake._outline_view == ((2, (0, 0, 4, 4), (4, 4), 2.0), None)
    # the kept answer is returned without reading the layers again
    monkeypatch.setattr(mm.engine, "_layer_outline_view",
                        lambda *a, **k: pytest.fail("read the layers again"))
    assert Canvas._outline_picture(fake, (0, 0, 4, 4), (4, 4), 2.0) is None


# -- small pure arms --------------------------------------------------------------

def test_an_empty_source_line_keeps_the_stock_minimum(qapp):
    label = mm._SourceLabel()
    assert label.minimumSizeHint() == QLabel().minimumSizeHint()
    label.setWordWrap(True)
    label.setText("Open a folder of images to start")
    assert label.minimumSizeHint() == QLabel.minimumSizeHint(label)


def test_a_metadata_read_asked_to_stop_reads_nothing(tmp_path):
    worker = mm._MaskReaderMetadataWorker(str(tmp_path / "x.tif"), 1)
    worker.isInterruptionRequested = lambda: True
    worker.run()
    assert worker.result is None and worker.error is None


def test_a_volume_mesh_without_a_cancellation_token_is_built_directly():
    class Geometry:
        def geometry_for_label(self, label):
            return ("mesh", label)

    assert mm._volume_mesh((Geometry(), 7)) == ("mesh", 7)


def test_the_opengl_volume_view_needs_pyopengl_to_import(monkeypatch):
    import sys

    monkeypatch.setattr(mm, "find_spec", lambda name: object())
    monkeypatch.setitem(sys.modules, "OpenGL", SimpleNamespace(__path__=[]))
    monkeypatch.setitem(sys.modules, "OpenGL.GL", None)
    assert mm._opengl_volume_importable() is False


def test_the_folded_import_warm_up_survives_an_unreadable_language(monkeypatch):
    from spacr.qt import i18n

    def broken():
        raise RuntimeError("no language")

    monkeypatch.setattr(i18n, "current_language", broken)
    monkeypatch.setattr(mm, "_import_the_cold_modules", lambda: None)
    mm._WARM_LANGUAGE[:] = ["kept"]
    thread = mm._warm_the_folded_imports()
    thread.join(5)
    assert not thread.is_alive() and mm._WARM_LANGUAGE == ["kept"]


def test_a_failed_field_timer_leaves_the_field_untimed(monkeypatch, caplog):
    from spacr import curation_queue

    class Broken:
        def __init__(self, *args):
            raise OSError("no timer")

    monkeypatch.setattr(curation_queue, "_FieldTimer", Broken)
    fake = SimpleNamespace(_folder="/f", _log=None, _field_timer="old")
    with caplog.at_level("WARNING"):
        Screen._start_field_timer(fake, "a.tif", np.zeros((2, 2)))
    assert fake._field_timer is None and "field timer" in caplog.text


def test_a_failed_timing_row_never_stops_a_visit_being_logged(monkeypatch, caplog):
    from spacr import curation_queue

    def broken(*args):
        raise OSError("disk full")

    monkeypatch.setattr(curation_queue, "_log_field_visit", broken)
    fake = SimpleNamespace(_field_timer=object(), _queue=SimpleNamespace(folder="/f"),
                           _log=None, _canvas=SimpleNamespace(mask=np.zeros((2, 2))))
    with caplog.at_level("WARNING"):
        Screen._log_field_visit(fake, "keep")
    assert "curation timing" in caplog.text


def test_a_finished_save_for_a_closed_screen_is_dropped():
    def gone(_request):
        raise RuntimeError("Internal C++ object already deleted.")

    seen = []
    Screen._deliver_mask_write(
        SimpleNamespace(_mask_write_delivered=SimpleNamespace(emit=gone)), {"id": 1})
    Screen._deliver_mask_write(
        SimpleNamespace(_mask_write_delivered=SimpleNamespace(emit=seen.append)), {"id": 2})
    assert seen == [{"id": 2}]


def test_a_cancelled_plane_review_with_no_displayed_field_changes_nothing():
    fake = SimpleNamespace(_pending_curation_status="x", _reader_displayed_field=None)
    Screen._restore_reader_displayed_field(fake)
    assert fake._pending_curation_status is None
    gone = SimpleNamespace(_pending_curation_status="x", _image_files=[],
                           _reader_displayed_field=("/f", "a.tif", None),
                           _field_folders=None, _folder="/other", _current_index=4)
    Screen._restore_reader_displayed_field(gone)
    assert gone._current_index == 4 and gone._folder == "/other"


def test_a_scan_for_layer_objects_stops_when_a_newer_scan_has_begun(monkeypatch):
    read = []
    monkeypatch.setattr(mm.engine, "_layer_has_objects",
                        lambda path, filename: read.append((path, filename)) or True)
    fake = SimpleNamespace(_layer_object_cache={}, _layer_scan_generation=2)
    Screen._scan_layer_objects(fake, ["a.tif", "b.tif"], ["/layer"], 1)
    assert read == [] and fake._layer_object_cache == {}
    Screen._scan_layer_objects(fake, ["a.tif"], ["/layer"], 2)
    assert fake._layer_object_cache == {("/layer", "a.tif"): True}
