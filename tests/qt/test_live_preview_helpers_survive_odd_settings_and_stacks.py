"""Live preview: the module-level helpers on the inputs nobody sent them.

Pinned here, all without a panel:

* an object caption ``organelle x`` whose tail is not a number still reads
  as the first organelle slot;
* the ``object_filters`` bound readers and writer when the setting is not a
  mapping, a stored row is invalid, or a bound value is not a number --
  the preview must answer "no bound" rather than raise;
* :func:`load_source_payload` projecting the loaded channel's z-stack on
  its worker (it runs in a Qt job, where coverage cannot see it, so it is
  called directly here), skipping sets from another folder or with one
  plane, and keeping the single plane when the projection cannot be read;
* the preview worker when the pass found nothing, and when it was
  cancelled before it started;
* the organelle pass when the organelle defaults cannot be filled;
* a backend model that returns no flow and a probability that is not 2-D;
* the ruler drawn over the picture, and a zoom that cannot be applied;
* the model-name checks when the backend tables, the file system or the
  alias table cannot be reached, and the plaque model when its resolver
  cannot be imported or fails.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest
import tifffile

pytest.importorskip("PySide6")

from PySide6.QtCore import QPoint, Qt  # noqa: E402
from PySide6.QtGui import QColor, QImage, QPixmap  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402

from spacr.qt.widgets import live_preview as LP  # noqa: E402
from spacr.qt.widgets.preview_controls import ImageSet  # noqa: E402

pytestmark = pytest.mark.qt


@pytest.fixture(autouse=True)
def _qapp(qapp):
    """QPixmap aborts the process outright when no QGuiApplication exists."""
    return qapp


def _name(field: int, chan: int, z: int = 1) -> str:
    return f"plate1_A01_T0001F{field:03d}L01A01Z{z:02d}C{chan:02d}.tif"


@pytest.fixture
def stack_plate(tmp_path: Path) -> Path:
    """One field, two channels, three planes: channel 1 is 1/2/3, channel 2
    is 10/20/30, so each projection is told apart by value."""
    root = tmp_path / "stacked"
    root.mkdir()
    for chan, step in ((1, 1), (2, 10)):
        for z in range(1, 4):
            tifffile.imwrite(root / _name(1, chan, z),
                             np.full((8, 8), z * step, np.uint16))
    return root


# ---------------------------------------------------------------------------
# object captions and object_filters
# ---------------------------------------------------------------------------

def test_an_organelle_caption_with_a_word_for_a_number_is_the_first_slot():
    assert LP.object_role("organelle x") == "organelle"
    assert LP.object_role("organelle 2") == "organelleb"
    assert LP.object_role("cell") == "cell"


@pytest.mark.parametrize("raw", ["[1, 2]", "{'cell': [", "{'bogus': []}"])
def test_an_unreadable_object_filters_reads_as_no_bound(raw):
    assert LP._bound_from_filters({"object_filters": raw}, "cell",
                                  "min_area") is None


def test_a_bound_is_read_from_its_row():
    settings = {"object_filters": {
        "cell": [{"property": "area", "min": 40.0, "max": None}]}}
    assert LP._bound_from_filters(settings, "cell", "min_area") == 40.0
    assert LP._bound_from_filters(settings, "cell", "max_area") is None


def test_bounds_written_over_a_setting_that_is_not_a_mapping_start_afresh():
    table = LP._bounds_into_filters("[1, 2, 3]", {"cell": {"min_area": 25}})
    assert table == {"cell": [{"property": "area", "min": 25.0,
                               "max": None}]}


def test_an_invalid_stored_row_is_replaced_not_kept():
    existing = {"cell": [{"property": "no_such_property", "min": 1}],
                "nucleus": [{"property": "solidity", "min": 0.9}]}
    table = LP._bounds_into_filters(existing, {"cell": {"max_area": 500}})
    assert table["cell"] == [{"property": "area", "min": None,
                              "max": 500.0}]
    assert table["nucleus"] == [{"property": "solidity", "min": 0.9}], (
        "another object's rows are left alone")


def test_a_bound_that_is_not_a_number_drops_the_objects_rows():
    existing = {"cell": [{"property": "area", "min": 10.0, "max": None}]}
    table = LP._bounds_into_filters(existing, {"cell": {"min_area": "lots"}})
    assert "cell" not in table, (
        "the only row was the area row being replaced, and the new bound "
        "could not be read, so nothing is left to filter the cell on")


# ---------------------------------------------------------------------------
# load_source_payload's projection
# ---------------------------------------------------------------------------

def test_the_payload_projects_the_opened_channels_stack(stack_plate):
    second = stack_plate / _name(1, 2, 2)
    out = LP.load_source_payload(second, project=True)
    assert out["error"] == ""
    assert Path(out["path"]) == second
    assert out["array"].min() == out["array"].max() == 30, (
        "channel 2's planes are 10/20/30; the projection is 30, not the "
        "20 of the plane that was opened")


def test_the_payload_projects_from_cached_sets_and_skips_the_others(
        stack_plate, tmp_path):
    elsewhere = ImageSet(key=("p", "A01", "001"), directory=str(tmp_path),
                         channels={"01": "x.tif"},
                         planes={"01": ["x.tif", "y.tif"]})
    flat = ImageSet(key=("plate1", "A01", "002"), directory=str(stack_plate),
                    channels={"01": _name(1, 1, 1)},
                    planes={"01": [_name(1, 1, 1)]})
    unnamed = ImageSet(key=("plate1", "A01", "003"),
                       directory=str(stack_plate),
                       channels={"02": _name(1, 2, 1)},
                       planes={"02": [_name(1, 2, 1), _name(1, 2, 2)]})
    own = ImageSet(key=("plate1", "A01", "001"), directory=str(stack_plate),
                   channels={"01": _name(1, 1, 1)},
                   planes={"01": [_name(1, 1, z) for z in (1, 2, 3)]})
    out = LP.load_source_payload(
        stack_plate / _name(1, 1, 1), enumerate_sets=False, project=True,
        known_sets=(elsewhere, flat, unnamed, own))
    assert out["sets"] is None, "nothing was enumerated"
    assert out["array"].max() == 3, (
        "only the set that holds the file, in its folder, with more than "
        "one plane is projected: max(1, 2, 3)")


def test_no_matching_set_leaves_the_single_plane(stack_plate, tmp_path):
    elsewhere = ImageSet(key=("p", "A01", "001"), directory=str(tmp_path),
                         channels={"01": "x.tif"},
                         planes={"01": ["x.tif", "y.tif"]})
    out = LP.load_source_payload(stack_plate / _name(1, 2, 3),
                                 enumerate_sets=False, project=True,
                                 known_sets=(elsewhere,))
    assert out["array"].max() == 30, "plane 3 of channel 2, unprojected"
    assert out["array"].min() == 30


def test_a_stack_that_cannot_be_read_leaves_the_single_plane(stack_plate):
    broken = ImageSet(key=("plate1", "A01", "001"), directory=str(stack_plate),
                      channels={"01": _name(1, 1, 2)},
                      planes={"01": [_name(1, 1, 2), "missing_plane.tif"]})
    out = LP.load_source_payload(stack_plate / _name(1, 1, 2),
                                 enumerate_sets=False, project=True,
                                 known_sets=(broken,))
    assert out["error"] == ""
    assert out["array"].max() == 2, "the plane that was opened is shown"


# ---------------------------------------------------------------------------
# the worker and the segmentation pass
# ---------------------------------------------------------------------------

def _drain(worker):
    got = {"provenance": []}
    worker.finished_masks.connect(
        lambda m, e, t: got.setdefault("masks", (m, e, t)))
    worker.provenance_ready.connect(
        lambda record, t: got["provenance"].append(record))
    return got


def test_a_pass_with_no_objects_announces_no_provenance():
    worker = LP._PreviewWorker(
        LP.PreviewRequest(image=np.zeros((6, 6), np.float32),
                          object_types=()), token=3)
    got = _drain(worker)
    worker.run()
    assert got["masks"] == ({}, "", 3)
    assert got["provenance"] == [], "no masks, nothing to describe"


def test_a_pass_cancelled_before_it_starts_reports_the_cancellation():
    request = LP.PreviewRequest(image=np.zeros((6, 6), np.float32))
    request.cancel.set()
    worker = LP._PreviewWorker(request, token=5)
    got = _drain(worker)
    worker.run()
    masks, error, token = got["masks"]
    assert masks is None and token == 5
    assert "cancelled" in error.lower()


def test_a_complete_organelle_setting_segments_when_its_defaults_fail(
        monkeypatch):
    """A settings dict that already names every organelle key is not
    stopped by the default-filler failing: the pass goes on with it."""
    import spacr.settings as settings_module

    settings = {"organelle_morphology": "spots", "organelle_method": "otsu"}
    settings_module._set_organelle_defaults(settings)

    def _refuse(_settings):
        raise RuntimeError("defaults unavailable")

    monkeypatch.setattr(settings_module, "_set_organelle_defaults", _refuse)
    image = np.zeros((64, 64), np.uint16)
    image[20:26, 20:26] = 3000
    image[40:44, 40:44] = 2500
    mask = LP._classical_organelle_mask(image, "organelle", settings)
    assert mask.dtype == np.int32
    assert mask.max() == 2, "both spots are still found"


def test_a_backend_without_flow_or_a_flat_probability_leaves_those_views(
        monkeypatch):
    import spacr._segmentation_backends as backends
    import spacr.object as obj_module

    calls = []

    def _masks_of(model, images, settings, obj, **kwargs):
        calls.append((model, obj))
        mask = np.zeros(images[0].shape[:2], np.int32)
        mask[1:4, 1:4] = 1
        return [mask], [], [np.zeros((2, 8, 8), np.float32)]

    monkeypatch.setattr(obj_module, "_prefixed_model_route",
                        lambda name: ("fake", _masks_of))
    monkeypatch.setattr(backends, "_load_backend",
                        lambda backend, **kw: f"{backend}-model")
    request = LP.PreviewRequest(image=np.ones((8, 8), np.float32),
                                model="fake:thing")
    masks, flows = LP._segment_multi(request)
    assert calls == [("fake-model", "cell")]
    assert masks["cell"].max() == 1
    assert flows == {}, "the backend gave no flow picture"
    assert request.cellprob_maps == {}, (
        "a probability that is not one plane is not shown")


def test_a_backend_with_no_probability_leaves_the_probability_view(
        monkeypatch):
    import spacr._segmentation_backends as backends
    import spacr.object as obj_module

    def _masks_of(model, images, settings, obj, **kwargs):
        return ([np.ones(images[0].shape[:2], np.int32)],
                [np.full(images[0].shape[:2] + (3,), 7, np.uint8)], [])

    monkeypatch.setattr(obj_module, "_prefixed_model_route",
                        lambda name: ("fake", _masks_of))
    monkeypatch.setattr(backends, "_load_backend", lambda backend, **kw: None)
    request = LP.PreviewRequest(image=np.ones((8, 8), np.float32),
                                model="fake:thing")
    masks, flows = LP._segment_multi(request)
    assert masks["cell"].max() == 1
    assert flows["cell"].shape == (8, 8, 3) and np.all(flows["cell"] == 7)
    assert request.cellprob_maps == {}


# ---------------------------------------------------------------------------
# the zoom view
# ---------------------------------------------------------------------------

def _black_view(qtbot):
    view = LP._ZoomView()
    qtbot.addWidget(view)
    view.resize(120, 120)
    pixmap = QPixmap(40, 40)
    pixmap.fill(QColor(0, 0, 0))
    view.set_pixmap(pixmap)
    view.show()
    qtbot.waitExposed(view)
    return view


def _lit_pixels(view) -> int:
    image = view.viewport().grab().toImage().convertToFormat(
        QImage.Format_RGB888)
    lit = 0
    for y in range(image.height()):
        for x in range(image.width()):
            colour = image.pixelColor(x, y)
            if colour.red() + colour.green() + colour.blue() > 150:
                lit += 1
    return lit


def test_the_ruler_is_drawn_over_the_picture(qtbot):
    view = _black_view(qtbot)
    before = _lit_pixels(view)
    view.ruler.set_active(True)
    port = view.viewport()
    start = view.mapFromScene(5, 5)
    end = view.mapFromScene(35, 35)
    QTest.mousePress(port, Qt.LeftButton, pos=start)
    QTest.mouseMove(port, end)
    QTest.mouseRelease(port, Qt.LeftButton, pos=end)
    assert view.ruler.start is not None
    assert view.ruler.length() == pytest.approx(30 * 2 ** 0.5, abs=2)
    assert _lit_pixels(view) > before + 10, "the line is painted"


def test_a_zoom_that_cannot_be_applied_changes_nothing(qtbot):
    view = _black_view(qtbot)
    seen = []
    view.zoom_changed.connect(seen.append)
    scale = view.scale_factor()
    view._apply_zoom(float("nan"), broadcast=True, position=QPoint(10, 10))
    assert view.scale_factor() == scale
    assert seen == []
    assert view._syncing is False, "the guard is released"


# ---------------------------------------------------------------------------
# model names
# ---------------------------------------------------------------------------

def test_a_blank_model_name_is_not_a_model():
    assert LP._is_a_real_model_name("") is False
    assert LP._is_a_real_model_name("   ") is False


def test_a_backend_prefixed_model_is_judged_by_its_backend():
    assert LP._is_a_real_model_name("stardist:") is True
    assert LP._is_a_real_model_name("stardist:/no/such/model") is False
    assert LP._checkpoint_is_missing("stardist:") is False
    assert LP._checkpoint_is_missing("stardist:/no/such/model") is True


def _break_backend_tables(monkeypatch):
    import spacr._segmentation_backends as backends

    def _boom(_name):
        raise RuntimeError("backend table unavailable")

    monkeypatch.setattr(backends, "_cellpose3_choice", _boom)


def _unreadable_file_system(_path):
    raise OSError("file system unavailable")


def test_a_model_name_is_judged_by_its_aliases_when_the_backends_fail(
        monkeypatch):
    _break_backend_tables(monkeypatch)
    monkeypatch.setattr(LP.os.path, "isfile", _unreadable_file_system)
    assert LP._is_a_real_model_name("cyto3") is True
    assert LP._is_a_real_model_name("no_such_model") is False


def test_nothing_is_a_model_when_nothing_can_be_consulted(monkeypatch):
    import spacr.settings as settings_module

    _break_backend_tables(monkeypatch)
    monkeypatch.setattr(LP.os.path, "isfile", _unreadable_file_system)
    monkeypatch.delattr(settings_module, "_CELLPOSE_ALIASES")
    assert LP._is_a_real_model_name("cyto3") is False


def test_a_missing_checkpoint_is_judged_by_its_path_when_the_backends_fail(
        monkeypatch, tmp_path):
    _break_backend_tables(monkeypatch)
    present = tmp_path / "weights.pth"
    present.write_bytes(b"")
    assert LP._checkpoint_is_missing(str(tmp_path / "gone.pth")) is True
    assert LP._checkpoint_is_missing(str(present)) is False
    assert LP._checkpoint_is_missing("cpsam") is False


def test_a_blank_model_key_gives_way_to_the_next_one():
    settings = {"cell_model_name": "   ", "model_name": "cyto3"}
    assert LP._model_the_run_would_use(settings, "cell") == (
        "cyto3", "model_name", True)


def test_the_plaque_model_is_unknown_when_its_resolver_cannot_load(
        monkeypatch):
    monkeypatch.setitem(sys.modules, "spacr.submodules", None)
    assert LP._model_the_run_would_use(
        {"plaque_model": "bundled"}, "cell", "analyze_plaques") == (
            "", "plaque_model", False)


def test_a_plaque_resolver_that_fails_keeps_the_requested_model(monkeypatch):
    fake = types.ModuleType("spacr.submodules")
    fake._requested_plaque_model = lambda s: s.get("plaque_model", "bundled")

    def _resolve(_settings, fetch=True):
        raise RuntimeError("zoo index is corrupt")

    fake._resolve_plaque_model = _resolve
    monkeypatch.setitem(sys.modules, "spacr.submodules", fake)
    assert LP._model_the_run_would_use(
        {"plaque_model": "zoo:plaque_v2"}, "cell", "analyze_plaques") == (
            "zoo:plaque_v2", "plaque_model", False)
