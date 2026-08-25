"""Cellpose-SAM on the field Make Masks has open, and its two intermediates.

Three things have to hold for this to be segmentation a person can steer
rather than a button that reshuffles the mask:

* the settings on the panel are the settings the run uses. A threshold
  that is displayed and then dropped on the way to ``eval`` is worse than
  no control at all, because the user believes they moved it;
* the probability map and the flow field come back on tabs beside the
  mask, INCLUDING when the run found nothing — that is precisely the run
  whose probability map answers why;
* a run that finds nothing, or that fails, leaves the mask alone. A mask
  someone has been correcting for an hour is not something a detection
  may quietly wipe.

The fake models here return exactly the structure cellpose 4.2.1.1 was
measured to return: ``(masks, flows, styles)``, with one image's ``flows``
a three-member list of an ``(H, W, 3)`` uint8 RGB rendering, a
``(2, H, W)`` float32 vector field, and an ``(H, W)`` float32
cell-probability map. The three members are not interchangeable, and
reading them as though they were is what puts the probability map on the
flow tab.
"""
from __future__ import annotations

import os
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pytest

from spacr.qt.screens import make_masks as mm
from spacr.qt.screens.make_masks import (
    CELLPROB_THRESHOLD,
    FLOW_THRESHOLD,
    MakeMasksScreen,
    cellpose_detect,
    cellpose_intermediates,
    flow_rgb,
    stretch_to_uint8,
)

pytestmark = pytest.mark.qt

#: Set to run the one test that loads real Cellpose weights. It is ~50 s
#: on a CPU, which is a minute nobody wants on every suite run.
RUN_REAL_CELLPOSE = os.environ.get("SPACR_CELLPOSE_E2E", "")


def _flows_like_cellpose(shape, cellprob=None):
    """One image's ``flows`` entry, shaped as cellpose 4.2 returns it."""
    rgb = np.zeros(shape + (3,), dtype=np.uint8)
    rgb[..., 0] = 200
    vectors = np.zeros((2,) + shape, dtype=np.float32)
    if cellprob is None:
        cellprob = np.linspace(-6.0, 6.0, int(np.prod(shape)),
                               dtype=np.float32).reshape(shape)
    return [rgb, vectors, cellprob]


class _FakeCellpose:
    """A model that segments two fixed squares and records its kwargs."""

    def __init__(self, labels=None, flows=None):
        self.labels = labels
        self.flows = flows
        self.kwargs = None
        self.batch = None

    def eval(self, x, batch_size=1, normalize=True, channel_axis=None,
             diameter=None, flow_threshold=0.4, cellprob_threshold=0.0,
             min_size=0):
        self.batch = x
        self.kwargs = dict(batch_size=batch_size, normalize=normalize,
                           channel_axis=channel_axis, diameter=diameter,
                           flow_threshold=flow_threshold,
                           cellprob_threshold=cellprob_threshold,
                           min_size=min_size)
        field = np.asarray(x[0])
        labels = self.labels
        if labels is None:
            labels = np.zeros(field.shape, dtype=np.uint16)
            labels[10:30, 10:30] = 1
            labels[40:60, 40:60] = 2
        flows = self.flows
        if flows is None:
            flows = _flows_like_cellpose(field.shape)
        return [labels], [flows], np.zeros(256, dtype=np.float32)


@pytest.fixture
def field_folder(tmp_path: Path) -> Path:
    """Two 80x80 fields with one bright square each, and no masks yet."""
    folder = tmp_path / "fields"
    (folder / "masks").mkdir(parents=True)
    for i in range(2):
        field = np.zeros((80, 80), dtype=np.uint16)
        field[10 + i:40, 10:40] = 3000
        imageio.imwrite(folder / f"f_{i:02d}.tif", field)
    return folder


@pytest.fixture
def screen(qtbot, qt_theme_applied, field_folder):
    """A Make Masks screen with the first field loaded."""
    made = MakeMasksScreen()
    qtbot.addWidget(made)
    made._open_folder(str(field_folder))
    if made._loading:
        made._load_worker.wait(30_000)
        made._on_background_load_finished()
    assert made._canvas.image is not None
    yield made
    made.close_folded()


# ----------------------------------------------------------------------
# Reading what eval hands back
# ----------------------------------------------------------------------
def test_the_three_flow_members_are_told_apart():
    """cellprob comes from flows[2] and the flow picture from flows[0].

    Measured against cellpose 4.2.1.1, whose ``flows`` for one 2-D image
    is ``[RGB (H,W,3) uint8, vectors (2,H,W) float32, cellprob (H,W)
    float32]``. Indexing that list as though its members were the same
    kind of thing is a silent failure: both entries are arrays of the
    right height and width, so the wrong one displays perfectly.
    """
    cellprob = np.full((6, 5), 2.5, dtype=np.float32)
    flows = _flows_like_cellpose((6, 5), cellprob=cellprob)

    got_prob, got_rgb = cellpose_intermediates(flows)

    assert got_prob.shape == (6, 5)
    assert np.array_equal(got_prob, cellprob)
    assert got_rgb.shape == (6, 5, 3)
    # flows[0], not a colouring of flows[2]: it is already a picture.
    assert np.array_equal(got_rgb, flows[0])


def test_a_flow_field_with_no_picture_is_drawn_from_its_vectors():
    """A vector field alone still fills the flow pane.

    ``flows[0]`` is what cellpose 4.2 renders for us, but it is the
    entry most likely to be missing from a wrapper or an older version.
    The vectors carry the same information, so the pane is built from
    them rather than left empty.
    """
    vectors = np.zeros((2, 8, 8), dtype=np.float32)
    vectors[0, :4] = 3.0
    vectors[1, :, :4] = -2.0

    rgb = flow_rgb(vectors)

    assert rgb.shape == (8, 8, 3)
    assert rgb.dtype == np.uint8
    # The two components went to different channels, so a field that
    # points north does not look like one that points east.
    assert rgb[..., 0].tolist() != rgb[..., 1].tolist()
    assert flow_rgb(None) is None
    assert flow_rgb(np.zeros((4, 4))) is None


def test_the_maps_are_stretched_onto_the_display_scale():
    """Both intermediates come back on 0-255, whatever range they were on.

    The probability map runs roughly -12..+8 and the intensity image is
    16-bit counts. Percentile-stretching each is what lets the panes be
    compared with the contrast-stretched image the canvas draws.
    """
    values = np.linspace(-12.0, 8.0, 400, dtype=np.float32).reshape(20, 20)

    stretched = stretch_to_uint8(values, 1.0, 99.0)

    assert stretched.dtype == np.uint8
    assert stretched.min() == 0
    assert stretched.max() == 255
    # Monotone: the brightest pixel of the map is the most probable one.
    assert stretched.ravel()[0] < stretched.ravel()[-1]
    assert stretch_to_uint8(np.zeros((0,))).size == 0


def test_the_field_goes_to_eval_as_a_batch_of_one():
    """One image is passed as ``[field]``, which is what the parser wants.

    Handed a bare 2-D array, cellpose returns a flat three-member flows
    list, and :func:`spacr.spacr_cellpose.parse_cellpose4_output` — the
    repository's own reader of this value — takes ``len(masks)`` for the
    number of images and finds the image height instead. The parse then
    fails on an image that segmented perfectly well.
    """
    model = _FakeCellpose()
    field = np.zeros((80, 80), dtype=np.uint16)

    labels, cellprob, rgb = cellpose_detect(field, model)

    assert isinstance(model.batch, list) and len(model.batch) == 1
    assert model.batch[0].shape == (80, 80)
    assert labels.shape == (80, 80)
    assert sorted(np.unique(labels).tolist()) == [0, 1, 2]
    assert cellprob.shape == (80, 80)
    assert rgb.shape == (80, 80, 3)


def test_every_setting_reaches_eval():
    """The panel's numbers arrive at Cellpose, none of them dropped."""
    model = _FakeCellpose()

    cellpose_detect(np.zeros((40, 40), dtype=np.uint16), model,
                    diameter=45, normalize=False, flow_threshold=0.9,
                    cellprob_threshold=-1.5, min_size=250)

    assert model.kwargs["diameter"] == 45
    assert model.kwargs["normalize"] is False
    assert model.kwargs["flow_threshold"] == pytest.approx(0.9)
    assert model.kwargs["cellprob_threshold"] == pytest.approx(-1.5)
    assert model.kwargs["min_size"] == 250
    # 0 means "work it out from the image", which Cellpose spells None.
    cellpose_detect(np.zeros((40, 40), dtype=np.uint16), model, diameter=0)
    assert model.kwargs["diameter"] is None


def test_a_variadic_eval_keeps_its_settings():
    """A model whose ``eval`` is ``(x, **kw)`` still gets every setting.

    The kwargs are filtered against the signature so a Cellpose that has
    dropped an argument between minor versions does not raise TypeError.
    A signature that NAMES nothing — any wrapper taking ``**kwargs`` —
    matched nothing under that filter, so the run succeeded with the
    thresholds silently discarded: the worst shape this could take, since
    the panel still showed the numbers the user had chosen.
    """
    seen = {}

    class Variadic:
        def eval(self, x, **kwargs):
            seen.update(kwargs)
            field = np.asarray(x[0])
            return ([np.zeros(field.shape, dtype=np.uint16)],
                    [_flows_like_cellpose(field.shape)], None)

    cellpose_detect(np.zeros((16, 16), dtype=np.uint16), Variadic(),
                    flow_threshold=0.75, cellprob_threshold=2.0)

    assert seen["flow_threshold"] == pytest.approx(0.75)
    assert seen["cellprob_threshold"] == pytest.approx(2.0)


# ----------------------------------------------------------------------
# The screen
# ----------------------------------------------------------------------
def test_the_two_intermediates_are_tabs_beside_the_mask(screen):
    """The panes sit on the canvas's own tab strip, resting until a run."""
    from PySide6.QtWidgets import QTabWidget

    tabs = screen._body_splitter.widget(0)

    assert isinstance(tabs, QTabWidget)
    assert [tabs.tabText(i) for i in range(tabs.count())] == [
        "Mask", "Cell probability", "Flows"]
    assert tabs.widget(0) is screen._canvas
    # Enabled, not greyed out: a disabled tab cannot be opened to read
    # the sentence that says why it is empty.
    assert tabs.isTabEnabled(1) and tabs.isTabEnabled(2)
    assert not screen._prob_pane.has_image()
    assert "Cellpose" in screen._prob_pane.text()
    assert "Cellpose" in screen._flow_pane.text()


def test_the_settings_are_cellposes_own_defaults(screen):
    """Nothing is assumed: both thresholds start where Cellpose puts them."""
    assert screen._cp_cellprob.value() == pytest.approx(CELLPROB_THRESHOLD)
    assert screen._cp_flow.value() == pytest.approx(FLOW_THRESHOLD)
    assert CELLPROB_THRESHOLD == 0.0
    assert FLOW_THRESHOLD == 0.4
    assert screen._cp_diameter.value() == 0
    assert screen._cp_normalize.isChecked()
    # The model combo carries names, not translated labels, and cpsam
    # is the one Cellpose 4 always has.
    names = [screen._cp_model.itemData(i)
             for i in range(screen._cp_model.count())]
    assert "cpsam" in names
    assert screen._cp_model.currentData() in names


def test_the_detect_button_is_in_the_one_tool_row(screen):
    """The action rides in the toolbar, so hiding the settings keeps it."""
    assert screen._tool_row_layout.indexOf(screen._btn_cellpose) >= 0
    screen._btn_settings.setChecked(False)
    assert not screen._settings_scroll.isVisible()
    assert screen._btn_cellpose.isEnabled()


def test_a_run_replaces_the_mask_and_fills_both_panes(screen):
    """The mask is what Cellpose found, and both panes say what it saw."""
    screen._cp_loaded["cpsam"] = _FakeCellpose()
    screen._cp_cellprob.setValue(-1.25)
    screen._cp_flow.setValue(0.7)
    screen._combine_mode.setCurrentIndex(
        screen._combine_mode.findData("replace"))

    found = screen.run_cellpose()

    assert found == 2
    mask = screen._canvas.mask
    assert sorted(np.unique(mask).tolist()) == [0, 1, 2]
    assert int((mask == 1).sum()) == 400
    assert int((mask == 2).sum()) == 400
    assert screen._prob_pane.has_image()
    assert screen._flow_pane.has_image()
    # The ledger records the numbers the run was made with, so a mask can
    # be traced back to the settings that produced it.
    edit = screen._log.edits[-1]
    assert edit.kind == "detect"
    assert edit.detail["method"] == "cellpose"
    assert edit.detail["cellprob_threshold"] == pytest.approx(-1.25)
    assert edit.detail["flow_threshold"] == pytest.approx(0.7)
    assert edit.detail["n_objects"] == 2
    # And it is one undo step.
    assert screen._btn_undo.isEnabled()


def test_merge_keeps_what_was_curated_by_hand(screen):
    """A detection folded in as merge adds objects without erasing any."""
    screen._canvas.mask[:] = 0
    screen._canvas.mask[62:72, 62:72] = 5
    screen._cp_loaded["cpsam"] = _FakeCellpose()
    screen._combine_mode.setCurrentIndex(
        screen._combine_mode.findData("merge"))

    screen.run_cellpose()

    mask = screen._canvas.mask
    # The hand-drawn square is still one object of its original size.
    kept = np.bincount(mask.ravel())
    assert (mask[62:72, 62:72] == mask[62, 62]).all()
    assert kept[mask[62, 62]] == 100
    assert len(np.unique(mask)) == 4          # background + 3 objects


def test_a_run_that_finds_nothing_leaves_the_mask_alone(screen):
    """Empty detections do not wipe an hour of correction.

    The probability pane fills anyway. A run that returned no objects is
    exactly the one whose probability map answers the question — did the
    network see nothing, or did the threshold throw away what it saw?
    """
    field = screen._canvas.image
    empty = np.zeros(field.shape, dtype=np.uint16)
    flows = _flows_like_cellpose(field.shape,
                                 cellprob=np.full(field.shape, -6.0,
                                                  dtype=np.float32))
    screen._cp_loaded["cpsam"] = _FakeCellpose(labels=empty, flows=flows)
    screen._canvas.mask[:] = 0
    screen._canvas.mask[5:15, 5:15] = 7
    before = screen._canvas.mask.copy()

    found = screen.run_cellpose()

    assert found == 0
    assert np.array_equal(before, screen._canvas.mask)
    assert screen._prob_pane.has_image()
    assert "no objects" in screen._status_label.text()


def test_a_failed_run_is_reported_and_the_button_comes_back(screen):
    """A Cellpose that raises leaves an editable screen, not a dead one."""
    class Boom:
        def eval(self, x, **kwargs):
            raise RuntimeError("CUDA out of memory")

    screen._cp_loaded["cpsam"] = Boom()
    before = screen._canvas.mask.copy()

    found = screen.run_cellpose()

    assert found == 0
    assert np.array_equal(before, screen._canvas.mask)
    assert screen._btn_cellpose.isEnabled()
    assert "out of memory" in screen._status_label.text()


def test_moving_to_another_field_empties_the_panes(screen):
    """The intermediates belong to one run on one field.

    Left on screen after the field changed they would be a picture of the
    previous image read as a picture of this one, with nothing saying so.
    """
    screen._cp_loaded["cpsam"] = _FakeCellpose()
    screen.run_cellpose()
    screen._view_tabs.setCurrentIndex(1)
    assert screen._prob_pane.has_image()

    screen._on_next()
    if screen._loading:
        screen._load_worker.wait(30_000)
        screen._on_background_load_finished()

    assert screen._current_index == 1
    assert not screen._prob_pane.has_image()
    assert not screen._flow_pane.has_image()
    assert screen._view_tabs.currentIndex() == 0
    assert "Cellpose" in screen._prob_pane.text()


def test_the_model_is_loaded_once_per_session(screen):
    """Loading cpsam costs seconds and hundreds of megabytes."""
    calls = []

    def fake_loader(name):
        calls.append(name)
        return _FakeCellpose()

    screen._cp_loaded.clear()
    original = mm.load_cellpose_model
    mm.load_cellpose_model = fake_loader
    try:
        screen.run_cellpose()
        screen.run_cellpose()
    finally:
        mm.load_cellpose_model = original

    assert calls == ["cpsam"]


def test_cellpose_without_an_open_folder_says_so(qtbot, qt_theme_applied):
    """No field, no run — and a sentence rather than a traceback."""
    made = MakeMasksScreen()
    qtbot.addWidget(made)

    assert made.run_cellpose() == 0
    assert "Open a folder" in made._status_label.text()
    made.close_folded()


@pytest.mark.slow
@pytest.mark.skipif(not RUN_REAL_CELLPOSE,
                    reason="set SPACR_CELLPOSE_E2E=1 (~50 s on CPU)")
def test_real_cellpose_segments_the_open_field(screen, tmp_path):
    """Real weights, real eval, and the objects counted off the mask.

    Four discs of radius 18 (area 1018 px each) came back as four objects
    of 1024-1048 px on cellpose 4.2.1.1, both panes filled, in ~50 s on a
    CPU. Anything that changes how the return value is read fails here
    even when every fake above still passes.
    """
    field = np.zeros((160, 160), dtype=np.uint16)
    yy, xx = np.mgrid[:160, :160]
    for cy, cx in ((40, 40), (40, 110), (110, 45), (115, 115)):
        field[(yy - cy) ** 2 + (xx - cx) ** 2 <= 18 ** 2] = 4000
    screen._canvas.set_image_and_mask(
        field, np.zeros(field.shape, dtype=np.uint16))
    screen._min_area.setValue(50)

    found = screen.run_cellpose()

    assert found == 4
    areas = np.bincount(screen._canvas.mask.ravel())[1:]
    assert len(areas) == 4
    assert all(900 <= int(a) <= 1150 for a in areas), areas.tolist()
    assert screen._prob_pane.has_image()
    assert screen._flow_pane.has_image()
