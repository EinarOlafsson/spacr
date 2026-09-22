"""Item 473: Make Masks detects the way organelle detection does.

Two claims, and they are different claims:

* EVERY ORGANELLE METHOD IS OFFERED -- adaptive, LoG, DoG, ridge,
  hysteresis and U-Net stand in the Mode box beside Otsu, Cellpose and the
  installable backends, each with its own parameters and nobody else's.
* AND IT IS THE SAME CODE. The methods are not reimplemented for the
  magnifier: they reach :func:`spacr.object._segment_single_image`, the
  routine the organelle mask pipeline's workers call. A test that only
  checked the box would pass just as well over a second copy that drifts,
  which is the failure this file is here to make impossible.

Around them, the enhancement chain: a fixed order that is asserted rather
than described, a chain that is part of the request key, a chain that is
written into the ledger beside the mask it produced, and a canvas that can
draw what the detector reads.
"""
from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pytest
from PySide6.QtCore import QPointF

from spacr.qt import detect_chain as dc
from spacr.qt import organelle_modes as om
from spacr.qt.screens import make_masks as mm


def _canvas_xy(screen, img_x: int, img_y: int) -> tuple:
    """The canvas point over image pixel ``(img_x, img_y)``.

    Read off the canvas's own mapping rather than recomputed here, so a
    change to how the field is fitted into the widget cannot leave this
    test hovering over a pixel that is not the one it names.
    """
    canvas = screen._canvas
    height, width = canvas.image.shape[:2]
    pixmap = canvas.pixmap()
    scale = pixmap.width() / width
    left = (canvas.width() - pixmap.width()) / 2
    top = (canvas.height() - pixmap.height()) / 2
    return (left + (img_x + 0.5) * scale, top + (img_y + 0.5) * scale)


def blob_field(size: int = 128) -> np.ndarray:
    """A field of five bright discs on noise, two of them touching."""
    rng = np.random.default_rng(11)
    field = (rng.random((size, size)) * 200).astype(np.uint16)
    yy, xx = np.ogrid[:size, :size]
    for cy, cx in ((30, 30), (30, 90), (90, 30), (60, 60), (60, 74)):
        field[(yy - cy) ** 2 + (xx - cx) ** 2 <= 100] += 4000
    return field


@pytest.fixture
def screen(qtbot, qt_theme_applied, tmp_path: Path):
    """A Make Masks screen with one field open."""
    folder = tmp_path / "fields"
    folder.mkdir()
    imageio.imwrite(folder / "a.tif", blob_field())
    made = mm.MakeMasksScreen()
    qtbot.addWidget(made)
    assert made._open_folder(str(folder))
    try:
        yield made
    finally:
        made._magnifier.close()
        made.close_folded()


def test_the_mode_box_offers_every_organelle_method(screen):
    """The six methods stand in the box, in organelle detection's names."""
    offered = [screen._mag_mode.itemData(i)
               for i in range(screen._mag_mode.count())]
    for mode in ("adaptive", "log", "dog", "ridge", "hysteresis", "unet"):
        assert mode in offered, f"{mode} is not offered"
        assert mode in mm._MAGNIFIER_SEGMENTERS
    assert offered[0] == "otsu", "Otsu stays the first row"
    if "cellpose" in offered:
        assert offered.index("adaptive") < offered.index("cellpose"), \
            "the methods that need nothing installed come first"


def test_every_organelle_method_is_a_legal_organelle_method():
    """No mode is offered that the organelle engine would refuse.

    The engine validates ``(morphology, method)`` before it loads an image
    (:func:`spacr.object._validate_organelle_settings`); a mode whose pair
    it refuses would be a row in the box that can only fail.
    """
    from spacr.object import _validate_organelle_settings

    for mode, morphology in om.MODE_MORPHOLOGY.items():
        _validate_organelle_settings(morphology, mode)


def test_a_method_runs_through_the_organelle_engine_and_not_a_copy(
        screen, monkeypatch):
    """The magnifier's adaptive IS organelle detection's adaptive.

    Proved by intercepting the engine's own entry point: if the screen had
    a second implementation this call would never arrive, and the test
    would fail with "the engine was never asked".
    """
    from spacr import object as object_module

    seen = {}
    real = object_module._segment_single_image

    def watched(img, settings):
        """Record what the engine was asked, then answer normally."""
        seen["settings"] = dict(settings)
        return real(img, settings)

    monkeypatch.setattr(object_module, "_segment_single_image", watched)
    request = mm._MagnifierRequest(
        key=("k",), crop=blob_field(64), box=(0, 0, 64, 64),
        shape=(64, 64), mode="adaptive", sensitivity=0.0, bright=True,
        min_area=20, model_name="cpsam", diameter=0, colour=(1, 2, 3),
        method_params=om.DEFAULT_PARAMS._replace(adaptive_block=31))
    labels, used, note = mm._segment_region(request)

    assert used == "adaptive" and note == "", note
    assert seen, "the organelle engine was never asked"
    assert seen["settings"]["organelle_method"] == "adaptive"
    assert seen["settings"]["organelle_morphology"] == "irregular"
    assert seen["settings"]["organelle_adaptive_block_size"] == 31
    assert seen["settings"]["organelle_min_area"] == 20
    assert labels.shape == (64, 64)


@pytest.mark.parametrize("mode", sorted(set(om.modes()) - {"unet"}))
def test_each_classical_method_finds_objects_on_a_real_looking_field(mode):
    """Every method runs on a field and returns labels of its shape."""
    field = blob_field()
    labels = om.segment(field, mode, om.DEFAULT_PARAMS, min_area=10)
    assert labels.shape == field.shape
    assert int(np.asarray(labels).max()) > 0, f"{mode} found nothing"


def test_a_mode_shows_its_own_parameters_and_no_others(screen):
    """The card is the mode's parameters, from one list the engine reads."""
    for mode in om.modes():
        screen._mag_mode.setCurrentIndex(screen._mag_mode.findData(mode))
        shown = {field for field, widget in screen._method_widgets.items()
                 if screen._method_form.isRowVisible(widget)}
        assert shown == set(om.PARAMETERS_FOR[mode]), mode
        assert screen._method_note.text(), f"{mode} says nothing about itself"


def test_the_card_stays_and_says_so_for_a_mode_that_reads_none_of_it(screen):
    """Otsu leaves the category in place with a sentence in it.

    A category that comes and goes is one whose place and folded state
    cannot be learned; see :meth:`MakeMasksScreen._sync_method_controls`.
    """
    screen._mag_mode.setCurrentIndex(screen._mag_mode.findData("otsu"))
    assert screen._methods_card.isVisibleTo(screen._settings_scroll)
    assert not any(screen._method_form.isRowVisible(widget)
                   for widget in screen._method_widgets.values())
    assert "Adaptive threshold" in screen._method_note.text()


def test_the_guidance_a_method_shows_comes_from_legal_methods():
    """What a method suits is read out of spaCR's own table, not retyped."""
    from spacr.organelle_types import LEGAL_METHODS, morphologies_for_method

    assert morphologies_for_method("ridge") == ("network",)
    assert set(morphologies_for_method("log")) == {
        morphology for morphology, methods in LEGAL_METHODS.items()
        if "log" in methods}
    assert om.guidance("ridge").startswith("Suits ")
    assert "filaments" in om.guidance("ridge")


def test_the_filament_width_box_is_parsed_and_never_raises():
    """A list of scales is typed, so it is read forgivingly."""
    assert mm._parsed_sigmas("1, 2, 3") == (1.0, 2.0, 3.0)
    assert mm._parsed_sigmas("2 4") == (2.0, 4.0)
    assert mm._parsed_sigmas("") == om.DEFAULT_PARAMS.ridge_sigmas
    assert mm._parsed_sigmas("nonsense") == om.DEFAULT_PARAMS.ridge_sigmas
    assert mm._parsed_sigmas("-1, 0, 2") == (2.0,)


def test_the_chain_order_is_fixed_and_written_down():
    """The order is a constant, and it is the order prepare runs in."""
    assert dc.CHAIN_ORDER == (
        "percentile stretch", "background", "denoise", "contrast", "sharpen",
        "detect", "morphology", "split")

    field = blob_field(64).astype(np.float32)
    chain = dc.Chain(background="tophat", background_radius=15,
                     denoise="gaussian", denoise_strength=1.5, gamma=0.7,
                     sharpen=True, sharpen_amount=0.8)
    step_by_step = dc._sharpen(
        dc._contrast(dc._denoise(dc._background(field, chain), chain), chain),
        chain)
    np.testing.assert_allclose(dc.prepare(field, chain), step_by_step,
                               rtol=1e-6)


def test_an_empty_chain_hands_back_the_very_array():
    """Nothing switched on must not cost a float copy of the field."""
    field = blob_field(32)
    assert dc.prepare(field, dc.NO_CHAIN) is field
    labels = np.zeros((4, 4), dtype=np.int32)
    labels[1:3, 1:3] = 1
    assert dc.finish(labels, dc.NO_CHAIN) is labels


@pytest.mark.parametrize("chain", [
    dc.Chain(background="rolling_ball", background_radius=12),
    dc.Chain(background="tophat", background_radius=12),
    dc.Chain(denoise="gaussian", denoise_strength=2.0),
    dc.Chain(denoise="median", denoise_strength=2.0),
    dc.Chain(denoise="bilateral"),
    dc.Chain(denoise="nlm"),
    dc.Chain(gamma=0.5),
    dc.Chain(clahe=True, clahe_tile=16),
    dc.Chain(equalize=True),
    dc.Chain(sharpen=True),
])
def test_every_pre_detection_step_changes_the_image_and_keeps_its_shape(chain):
    """Each step is real: it runs, it moves pixels, it keeps the field."""
    field = blob_field(64)
    out = dc.prepare(field, chain)
    assert out.shape == field.shape
    assert not np.array_equal(out, field.astype(np.float32))


def test_the_split_is_offered_to_a_method_that_is_not_otsu():
    """Two touching discs come back as ONE object, then as two with split.

    The dumb-bell is exactly what the Otsu category's "Split objects that
    touch" was added for, and item 473 is that box offered to the other
    methods: the same distance-transform watershed
    (:func:`spacr.object._watershed_split`), reached from the chain.
    """
    field = np.zeros((64, 96), dtype=np.uint16)
    yy, xx = np.ogrid[:64, :96]
    for cx in (36, 58):
        field[(yy - 32) ** 2 + (xx - cx) ** 2 <= 169] = 5000
    joined = (field > 0).astype(np.int32)

    assert int(dc.finish(joined, dc.NO_CHAIN).max()) == 1
    assert int(dc.finish(joined, dc.Chain(split=True),
                         intensity=field).max()) == 2


def test_morphology_joins_and_separates_what_was_detected():
    """Closing bridges a broken object; opening breaks a thin bridge."""
    broken = np.zeros((32, 32), dtype=np.int32)
    broken[10:20, 6:14] = 1
    broken[10:20, 17:25] = 2
    closed = dc.finish(broken, dc.Chain(morphology="close",
                                        morphology_radius=3))
    assert int(closed.max()) == 1, "closing did not join the two pieces"

    bridged = np.zeros((32, 32), dtype=np.int32)
    bridged[10:20, 6:14] = 1
    bridged[10:20, 17:25] = 1
    bridged[14:16, 14:17] = 1
    opened = dc.finish(bridged, dc.Chain(morphology="open",
                                         morphology_radius=2))
    assert int(opened.max()) == 2, "opening did not break the bridge"


def test_the_otsu_mode_is_left_to_its_own_split_and_fill(monkeypatch):
    """The chain's last two stages are not applied on top of Otsu's.

    Otsu has had "Split objects that touch" of its own since item 419;
    running the chain's split after it would read Otsu's objects back as
    one foreground and join a pair Otsu had just separated.
    """
    called = {}

    def watched(labels, chain, intensity=None):
        """Record that the chain's post steps were asked."""
        called["yes"] = True
        return labels

    monkeypatch.setattr(dc, "finish", watched)
    request = mm._MagnifierRequest(
        key=("k",), crop=blob_field(64), box=(0, 0, 64, 64), shape=(64, 64),
        mode="otsu", sensitivity=0.0, bright=True, min_area=10,
        model_name="cpsam", diameter=0, colour=(1, 2, 3),
        chain=dc.Chain(split=True, morphology="close"))
    _labels, used, _note = mm._segment_region(request)

    assert used == "otsu"
    assert "yes" not in called, "the chain's split ran on top of Otsu's"


def test_a_whole_image_run_of_a_new_method_can_be_cancelled():
    """Item 407's ticket covers the new modes, at the steps between filters.

    The classical methods have no tiles to count, so what is asserted is
    the part that matters to a person: a Cancel is seen, and the run stops
    with :class:`mm._RunCancelled` instead of finishing an answer nobody
    wants.
    """
    ticket = mm._RunTicket()
    field = blob_field(96)
    request = mm._MagnifierRequest(
        key=("k",), crop=field, box=(0, 0, 96, 96), shape=(96, 96),
        mode="adaptive", sensitivity=0.0, bright=True, min_area=20,
        model_name="cpsam", diameter=0, colour=(1, 2, 3), scope="image",
        ticket=ticket, chain=dc.Chain(gamma=0.8))

    labels, used, note = mm._segment_region(request)
    assert used == "adaptive" and note == ""
    assert labels.shape == field.shape

    ticket.cancel()
    with pytest.raises(mm._RunCancelled):
        mm._segment_region(request)


def test_the_chain_and_the_parameters_are_part_of_the_request_key(screen):
    """A cached answer under one chain must not answer for another."""
    magnifier = screen._magnifier
    magnifier.set_enabled(True)
    screen._canvas.resize(600, 400)
    screen._canvas.refresh()
    magnifier.hover(QPointF(*_canvas_xy(screen, 60, 60)))
    assert magnifier.build_request() is not None, "no box under the mouse"
    before = magnifier.build_request().key
    screen._enh_clahe.setChecked(True)
    assert magnifier.build_request().key != before, \
        "the chain is not in the key"

    screen._enh_clahe.setChecked(False)
    assert magnifier.build_request().key == before
    screen._method_widgets["adaptive_block"].setValue(71)
    assert magnifier.build_request().key != before, \
        "a method parameter is not in the key"


def test_the_chain_records_only_what_ran_and_the_order_it_ran_in():
    """A ledger entry says what was done, not what was left alone."""
    assert dc.provenance(dc.NO_CHAIN) == {"enhancement": "none"}
    recorded = dc.provenance(
        dc.Chain(background="tophat", background_radius=20, split=True),
        percentile_stretch=True)["enhancement"]
    assert recorded["percentile_stretch"] is True
    assert recorded["background"] == "tophat"
    assert recorded["background_radius"] == 20
    assert recorded["split"] is True
    assert recorded["order"] == list(dc.CHAIN_ORDER)
    assert "gamma" not in recorded and "denoise" not in recorded


def test_the_mask_a_detect_button_makes_carries_the_chain(screen):
    """Otsu detect writes the chain into the field's ledger."""
    screen._enh_gamma.setValue(0.6)
    screen._enh_split.setChecked(True)
    screen._on_detect_otsu()
    entries = [edit for edit in screen._log._edits if edit.kind == "detect"]
    assert entries, "Otsu detect recorded nothing"
    enhancement = entries[-1].detail["enhancement"]
    assert enhancement["gamma"] == pytest.approx(0.6)
    assert enhancement["split"] is True
    assert enhancement["order"] == list(dc.CHAIN_ORDER)


def test_the_heavy_steps_say_they_are_heavy(screen):
    """A step that costs minutes says so before it is switched on."""
    assert dc.heavy_steps(dc.NO_CHAIN) == ()
    assert "non-local means" in dc.heavy_steps(dc.Chain(denoise="nlm"))
    assert om.HEAVY_MODES["unet"]

    small = dc.Chain(background="tophat", background_radius=5)
    default = dc.Chain(background="rolling_ball", background_radius=50)
    exact = dc.Chain(background="rolling_ball", background_radius=50,
                     background_scale=1.0)
    assert dc.heavy_steps(small) == (), \
        "a small background radius is not slow and must not say it is"
    assert dc.heavy_steps(default) == (), \
        "the default radius at the default scale is about a second"
    assert dc.heavy_steps(exact), \
        "the exact background at the default radius is fifteen seconds"

    assert screen._enh_heavy.text() == ""
    screen._enh_denoise.setCurrentIndex(
        screen._enh_denoise.findData("nlm"))
    assert "non-local means" in screen._enh_heavy.text()


def test_the_canvas_can_draw_what_the_detector_reads(qtbot, screen):
    """"Show the enhanced image" changes the picture and nothing else.

    The picture is built off the GUI thread, so the first ask hands back
    the field as loaded and the enhanced one arrives on
    :attr:`_MaskCanvas.enhanced_ready`; that is the behaviour
    :func:`test_a_slow_background_never_runs_on_the_gui_thread` is about.
    """
    canvas = screen._canvas
    loaded = np.array(canvas.image, copy=True)
    screen._enh_gamma.setValue(0.4)

    with qtbot.waitSignal(canvas.enhanced_ready, timeout=20000):
        screen._enh_show.setChecked(True)
        canvas.enhanced_picture()

    drawn = canvas.enhanced_picture()
    assert drawn is not None
    assert not np.array_equal(drawn, canvas.image)
    np.testing.assert_array_equal(
        canvas.image, loaded,
        err_msg="the enhanced view changed the data underneath")
    assert drawn.dtype == canvas.image.dtype

    screen._enh_show.setChecked(False)
    assert canvas.enhance_display is False
    canvas.close_enhancer()


def test_the_chain_is_described_in_the_words_a_caption_uses():
    """The caption names the steps, not the fields that hold them."""
    assert dc.step_names(dc.NO_CHAIN) == ()
    assert dc.step_names(
        dc.Chain(background="tophat", gamma=0.5, clahe=True, split=True),
        percentile_stretch=True) == (
        "percentile stretch", "top-hat background", "gamma 0.50", "CLAHE",
        "split touching objects")


def test_the_compare_window_shows_the_raw_and_the_enhanced_side_by_side(
        screen):
    """One click, two pictures, and the list of what ran between them."""
    screen._enh_clahe.setChecked(True)
    screen._on_compare_enhanced()
    dialog = screen._compare_dialog
    try:
        assert dialog is not None and dialog.isVisible()
        assert "CLAHE" in dialog.caption.text()
        left, right = (pane.pixmap().toImage() for pane in dialog.panes)
        assert not left.isNull() and not right.isNull()
        assert left != right, "the two pictures are the same picture"
    finally:
        dialog.close()


def test_the_compare_window_says_so_when_nothing_is_switched_on(screen):
    """An empty chain is reported rather than shown as two same pictures."""
    screen._on_compare_enhanced()
    dialog = screen._compare_dialog
    try:
        assert "No enhancement step" in dialog.caption.text()
    finally:
        dialog.close()


def test_denoising_needs_no_pywavelets_and_a_broken_step_is_skipped():
    """Reported 2026-09-22: hovering the image filled the console with
    'PyWavelets is not installed' -- scikit-image's estimate_sigma needs an
    optional package spaCR does not carry, and the exception came out of a
    mouse-move. The noise estimate is numpy's now, and any step that cannot
    run is skipped rather than raised."""
    import numpy as np

    from spacr.qt import detect_chain as dc

    rng = np.random.default_rng(0)
    field = rng.normal(0.5, 0.05, (48, 48)).astype(np.float32)
    assert dc._noise_sigma(field) > 0
    assert dc._noise_sigma(np.zeros((8, 8), np.float32)) == 0.0

    chain = dc.Chain(denoise="nl_means", denoise_strength=1.0)
    out = dc.prepare(field, chain)
    assert out.shape == field.shape and out.dtype == np.float32

    def explode(image, chain):
        raise ImportError("PyWavelets is not installed")

    original = dc._denoise
    dc._denoise = explode
    try:
        kept = dc.prepare(field, chain)
    finally:
        dc._denoise = original
    assert kept.shape == field.shape, "the chain carries on without that step"


def test_a_step_that_is_off_runs_nothing_at_all():
    """The crash of 2026-09-22: a step nobody asked for, on every field.

    :func:`spacr.qt.detect_chain._denoise` used to fall through to
    non-local means for ``denoise="none"``, so ANY active chain -- a
    background, a gamma -- ran a minutes-long denoiser over the whole
    field, from a mouse-move. It was reported as "the rolling ball crashes
    the program right away". What is asserted is the shape of the fix: an
    off step hands back the very array it was given, so it cannot be
    running anything.
    """
    field = blob_field(64).astype(np.float32)
    assert dc._denoise(field, dc.NO_CHAIN) is field
    assert dc._denoise(field, dc.Chain(background="tophat")) is field
    assert dc._contrast(field, dc.NO_CHAIN) is field
    assert dc._contrast(field, dc.Chain(denoise="median")) is field
    assert dc._background(field, dc.NO_CHAIN) is field


def test_the_background_is_estimated_small_and_1_0_is_exact():
    """The scale buys the speed; 1.00 buys scikit-image's own answer.

    A background is what varies slowly, so estimating it on a smaller copy
    and scaling the surface back up is sound. At 1.00 nothing is scaled and
    the call is scikit-image's, which this checks to the bit rather than to
    a tolerance -- that is the promise the tooltip makes.
    """
    from skimage.morphology import disk, white_tophat
    from skimage.restoration import rolling_ball

    field = blob_field(256).astype(np.float32)
    exact_ball = dc.Chain(background="rolling_ball", background_radius=20,
                          background_scale=1.0)
    exact_hat = dc.Chain(background="tophat", background_radius=20,
                         background_scale=1.0)
    np.testing.assert_array_equal(
        dc.prepare(field, exact_ball),
        np.clip(field - rolling_ball(field, radius=20), 0.0, None))
    np.testing.assert_array_equal(
        dc.prepare(field, exact_hat),
        white_tophat(field, disk(20)))

    small = dc.Chain(background="rolling_ball", background_radius=20,
                     background_scale=0.5)
    scaled = dc.prepare(field, small)
    assert scaled.shape == field.shape
    assert not np.array_equal(scaled, dc.prepare(field, exact_ball)), \
        "a scaled estimate that equals the exact one is not being scaled"
    assert dc.background_surface(field, small).shape == field.shape
    assert np.array_equal(dc.background_surface(field, dc.NO_CHAIN),
                          np.zeros_like(field))


def test_a_tiny_region_is_never_estimated_on_a_thumbnail():
    """Below a floor the copy stops being scaled: a box is small already.

    The magnifier's box is a few dozen pixels; halving it would leave a
    surface with no detail to follow, and the exact call on something that
    small costs nothing anyway.
    """
    box = blob_field(48).astype(np.float32)
    chain = dc.Chain(background="rolling_ball", background_radius=10,
                     background_scale=0.25)
    from skimage.restoration import rolling_ball

    np.testing.assert_allclose(
        dc.background_surface(box, chain),
        rolling_ball(box, radius=10), rtol=1e-6)


def test_the_readout_never_runs_the_enhancement_chain(screen, monkeypatch):
    """A mouse move must not be able to start a background estimate.

    The readout is re-read on EVERY move. Pointing it at the chain's
    output is what made choosing a rolling ball freeze the window, so this
    fails if anything in that path reaches :func:`detect_chain.prepare`.
    """
    from PySide6.QtCore import QPointF

    ran = []
    real = dc.prepare
    monkeypatch.setattr(dc, "prepare",
                        lambda image, chain: ran.append(chain) or real(
                            image, chain))
    screen._enh_background.setCurrentIndex(
        screen._enh_background.findData("rolling_ball"))
    screen._enh_background_radius.setValue(50)
    ran.clear()

    screen._canvas.resize(600, 400)
    screen._canvas.refresh()
    for x in range(20, 60, 8):
        screen._canvas.update_readout(QPointF(*_canvas_xy(screen, x, x)))
    assert ran == [], \
        "the readout ran the chain: " + repr(ran[:1])


def test_a_slow_background_never_runs_on_the_gui_thread(qtbot, screen):
    """The enhanced picture is asked for, not waited for.

    "Show the enhanced image" with a background subtraction switched on
    must return at once with the field as loaded, and deliver the enhanced
    picture afterwards. Measured rather than asserted about: the call has
    to be far quicker than the work it starts.
    """
    import time

    canvas = screen._canvas
    screen._enh_background.setCurrentIndex(
        screen._enh_background.findData("rolling_ball"))
    screen._enh_background_radius.setValue(40)
    screen._enh_show.setChecked(True)

    started = time.monotonic()
    first = canvas.enhanced_picture()
    asked_in = time.monotonic() - started

    assert asked_in < 0.5, f"enhanced_picture blocked for {asked_in:.2f} s"
    np.testing.assert_array_equal(first, canvas.detection_base())
    with qtbot.waitSignal(canvas.enhanced_ready, timeout=30000):
        pass
    assert not np.array_equal(canvas.enhanced_picture(),
                              canvas.detection_base())
    assert canvas.close_enhancer()
