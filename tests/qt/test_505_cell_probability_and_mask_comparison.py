"""Item 505 parts 1 and 3: the cell probability view and the comparison panel.

    "please add the cell probability array here as well ... after having
     generated more than 1 mask the user should be able to over lay the
     masks ... so there are 3 pannels now."
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

from PySide6.QtCore import QEvent, QPoint                     # noqa: E402
from PySide6.QtGui import QImage                              # noqa: E402
from PySide6.QtWidgets import QApplication                    # noqa: E402

from spacr.qt.widgets import live_preview as LP               # noqa: E402
from spacr.qt.widgets import picture_export                   # noqa: E402
from spacr.qt.widgets.mask_comparison import (                # noqa: E402
    IMAGE, MASK, Layer, MaskComparisonDialog, composite)
from spacr.qt.widgets.segmentation_views import render_cellprob  # noqa: E402
from tests.cellpose_api_contract import MISSING_CHANNEL_AXIS  # noqa: E402


def _pixels(view):
    """The RGB bytes a view is showing, as ``H x W x 3``."""
    pixmap = view.picture()
    assert pixmap is not None, "the view shows no picture"
    image = QImage(pixmap.toImage()).convertToFormat(QImage.Format_RGB888)
    w, h, stride = image.width(), image.height(), image.bytesPerLine()
    raw = np.frombuffer(bytes(image.constBits())[:stride * h], np.uint8)
    return raw.reshape(h, stride)[:, :w * 3].reshape(h, w, 3)


def _panel(qtbot, shape=(32, 32)):
    p = LP.LivePreviewPanel()
    qtbot.addWidget(p)
    p._image = np.random.RandomState(0).randint(
        0, 255, shape, dtype=np.uint16)
    return p


def _mask(corner):
    mask = np.zeros((32, 32), np.int32)
    mask[corner:corner + 8, corner:corner + 8] = 1
    return mask


def _logits():
    return np.linspace(-6.0, 6.0, 32 * 32, dtype=np.float32).reshape(32, 32)


# ---------------------------------------------------------------------------
# Part 1: the cell probability view
# ---------------------------------------------------------------------------

def test_cell_probability_is_the_fourth_view():
    assert LP.VIEW_MODES == ("Overlay", "Masks", "Flows", "Cell probability")


def test_the_segmenter_keeps_cellposes_flows_2(monkeypatch):
    """``flows[2]`` of ``model.eval`` is the probability, kept per object."""
    logits = _logits()

    class _Model:
        def eval(self, x, batch_size=8, resample=True, channels=None,
                 channel_axis=MISSING_CHANNEL_AXIS, z_axis=None,
                 normalize=True, invert=False, rescale=None, diameter=None,
                 flow_threshold=0.4, cellprob_threshold=0.0, do_3D=False,
                 anisotropy=None, flow3D_smooth=0, stitch_threshold=0.0,
                 min_size=15, max_size_fraction=0.4, niter=None,
                 augment=False, tile_overlap=0.1, bsize=256,
                 compute_masks=True, progress=None):
            image = x
            rgb = np.zeros(image.shape[:2] + (3,), np.uint8)
            dP = np.zeros((2,) + image.shape[:2], np.float32)
            return np.ones(image.shape[:2], np.uint16), [rgb, dP, logits], None

    monkeypatch.setattr(LP, "preview_cellpose_model", lambda name: _Model())
    request = LP.PreviewRequest(image=np.ones((32, 32), np.uint16))
    masks, flows = LP._segment_multi(request)
    assert set(masks) == {"cell"}
    assert np.array_equal(request.cellprob_maps["cell"], logits)


def test_the_worker_carries_the_probability_in_its_result(monkeypatch, qtbot):
    logits = _logits()
    monkeypatch.setattr(
        LP, "_segment_multi",
        lambda req: ({"cell": _mask(4)}, {}, {"cell": logits}))
    worker = LP._PreviewWorker(
        LP.PreviewRequest(image=np.ones((32, 32), np.uint16)), token=7)
    got = {}
    worker.cellprob_ready.connect(lambda maps, token: got.update(
        maps=maps, token=token))
    worker.run()
    assert got["token"] == 7
    assert np.array_equal(got["maps"]["cell"], logits)


def test_the_fourth_view_shows_the_probability_from_a_faked_result(qtbot):
    p = _panel(qtbot)
    p._masks = {"cell": _mask(4)}
    p._on_cellprob_ready({"cell": _logits()})
    p._view_mode.setCurrentText("Cell probability")
    p._refresh_canvases()
    assert np.array_equal(_pixels(p._mask_view), render_cellprob(_logits()))
    assert p._mask_view.picture_name() == "cell_probability"


def test_a_method_without_a_probability_says_so(qtbot):
    p = _panel(qtbot)
    p._masks = {"organelle": _mask(4)}
    p._processing_provenance = {"methods": {"organelle": "otsu"}}
    p._on_cellprob_ready({})
    p._view_mode.setCurrentText("Cell probability")
    p._refresh_canvases()
    assert p._mask_view.picture() is None
    assert "otsu" in p._mask_view.message()


def test_a_later_run_without_a_probability_clears_the_old_one(qtbot):
    p = _panel(qtbot)
    p._masks = {"cell": _mask(4)}
    p._view_mode.setCurrentText("Cell probability")
    p._on_cellprob_ready({"cell": _logits()})
    assert p._mask_view.picture() is not None
    p._on_cellprob_ready({})
    assert p._mask_view.picture() is None
    assert p._mask_view.message() == "This run gave no cell probability map."


def _right_click_save(view, tmp_path, monkeypatch):
    asked = {}

    def _where(parent, stem):
        asked["stem"] = stem
        return str(tmp_path / f"{stem}.png")

    monkeypatch.setattr(picture_export, "choose_format",
                        lambda widget, point, enabled: ".png")
    monkeypatch.setattr(picture_export, "ask_where_to_save", _where)
    written = view._spacr_picture_menu(QPoint(4, 4))
    return asked.get("stem"), written


def test_the_probability_view_saves_from_a_right_click(qtbot, tmp_path,
                                                       monkeypatch):
    p = _panel(qtbot)
    p._masks = {"cell": _mask(4)}
    p._cellprob = {"cell": _logits()}
    p._view_mode.setCurrentText("Cell probability")
    p._refresh_canvases()
    stem, written = _right_click_save(p._mask_view, tmp_path, monkeypatch)
    assert stem == "cell_probability"
    assert (QImage(written).width(), QImage(written).height()) == (32, 32)


# ---------------------------------------------------------------------------
# Part 3: the comparison panel
# ---------------------------------------------------------------------------

def test_the_button_waits_for_a_second_mask(qtbot):
    p = _panel(qtbot)
    assert p._compare_masks_btn.isHidden()
    p._snapshot_run({"cell": _mask(4)}, ["cell=1"])
    assert p._compare_masks_btn.isHidden(), "one mask is nothing to compare"
    p._snapshot_run({"cell": _mask(12)}, ["cell=1"])
    assert not p._compare_masks_btn.isHidden()
    assert p._compare_view.isHidden(), "the third panel waits for the popup"


def test_the_session_keeps_the_last_eight_masks(qtbot):
    p = _panel(qtbot)
    for run in range(11):
        p._snapshot_run({"cell": _mask(run)}, ["cell=1"])
    kept = p.comparable_masks()
    assert len(kept) == LP.SESSION_MASK_LIMIT == 8
    assert kept[0]["name"].startswith("Run 4")
    assert kept[-1]["name"].startswith("Run 11")


def test_two_masks_composite_to_hand_computed_pixels():
    """Grey 100 under a red mask at 50 % under a green mask at 50 %.

    (0, 0): 100 -> red over it  (100*.5 + 200*.5, 50, 50) = (150, 50, 50)
                -> green over it (75, 50*.5 + 100*.5, 25)  = (75, 75, 25)
    (0, 1): 100 -> green only    (50, 100, 50)
    (1, 1): 100, neither mask covers it.
    """
    image = np.full((2, 2), 100, np.uint8)
    red = np.array([[1, 0], [0, 0]], np.int32)
    green = np.array([[3, 3], [0, 0]], np.int32)
    layers = [Layer("field", IMAGE, image, opacity=1.0),
              Layer("red", MASK, red, colour=(200, 0, 0), opacity=0.5),
              Layer("green", MASK, green, colour=(0, 100, 0), opacity=0.5)]
    out = composite(layers)
    assert out[0, 0].tolist() == [75, 75, 25]
    assert out[0, 1].tolist() == [50, 100, 50]
    assert out[1, 1].tolist() == [100, 100, 100]

    swapped = composite([layers[0], layers[2], layers[1]])
    assert swapped[0, 0].tolist() == [125, 50, 25], (
        "the stacking order must change the answer")
    assert composite([Layer("x", MASK, red, ticked=False)]) is None


def test_the_popup_lists_top_first_and_hands_back_bottom_first(qtbot):
    image = np.full((2, 2), 100, np.uint8)
    dialog = MaskComparisonDialog([
        Layer("b", MASK, np.ones((2, 2), np.int32)),
        Layer("a", MASK, np.ones((2, 2), np.int32)),
        Layer("field", IMAGE, image, opacity=1.0)])
    qtbot.addWidget(dialog)
    assert [layer.name for layer in dialog.chosen()] == ["field", "a", "b"]
    dialog.move(0, 1)
    ticks = [tick.text() for tick, _ in dialog.rows()]
    assert ticks == ["a", "b", "field"]
    dialog.rows()[1][0].setChecked(False)
    dialog.rows()[0][1].setValue(20)
    chosen = dialog.chosen()
    assert [layer.name for layer in chosen] == ["field", "a"]
    assert chosen[-1].opacity == pytest.approx(0.2)


def test_accepting_draws_the_composite_in_a_third_panel(qtbot, tmp_path,
                                                        monkeypatch):
    p = _panel(qtbot)
    p._snapshot_run({"cell": _mask(4)}, ["cell=1"])
    p._snapshot_run({"cell": _mask(8)}, ["cell=1"])

    def _answer(dialog):
        rows = dialog.rows()
        assert [tick.text().split(" · ")[0] for tick, _ in rows[:2]] == \
            ["Run 2", "Run 1"], "the newest mask is listed on top"
        for _, slider in rows[:2]:
            slider.setValue(40)
        return True

    monkeypatch.setattr(p, "_exec_comparison_dialog", _answer)
    assert p.open_mask_comparison() is True
    assert not p._compare_view.isHidden()
    layers = p.comparison_layers()
    for layer in layers[:2]:
        layer.opacity = 0.4
    expected = composite(list(reversed(layers)))
    assert np.array_equal(_pixels(p._compare_view), expected)

    stem, written = _right_click_save(p._compare_view, tmp_path, monkeypatch)
    assert stem == "mask_comparison"
    assert QImage(written).width() == 32


def test_nothing_ticked_puts_the_third_panel_away(qtbot):
    p = _panel(qtbot)
    p._snapshot_run({"cell": _mask(4)}, ["cell=1"])
    p._snapshot_run({"cell": _mask(8)}, ["cell=1"])
    assert p.show_comparison(list(reversed(p.comparison_layers())))
    assert not p._compare_view.isHidden()
    assert p.show_comparison([]) is False
    assert p._compare_view.isHidden()


def test_the_popup_wears_the_card_and_the_rim(qtbot):
    from spacr.qt.preferences import apply_preferences_to_app
    from spacr.qt.widgets.glass import (install_glass_everywhere,
                                        uninstall_glass_everywhere)
    from spacr.qt.widgets.setup_card import SetupCard

    app = QApplication.instance()
    apply_preferences_to_app(app)
    install_glass_everywhere(app)
    dialog = MaskComparisonDialog(
        [Layer("a", MASK, np.ones((2, 2), np.int32))])
    try:
        dialog.show()
        for _ in range(8):
            app.processEvents()
        assert dialog.findChildren(SetupCard)
    finally:
        dialog.close()
        dialog.deleteLater()
        uninstall_glass_everywhere(app)
        app.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        app.processEvents()
