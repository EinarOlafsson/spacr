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
from spacr.qt import mask_engine as engine
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
    with pytest.raises(ValueError, match="method must be one of"):
        _validate_organelle_settings("spots", "not-a-detection-method")


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


def test_one_category_shows_one_family_and_hides_the_rest(screen):
    """Item 473's fold: Otsu, Object detection and the methods are one.

    The category stays put whatever is chosen -- a category that comes and
    goes is one whose place and folded state cannot be learned -- and the
    group inside it follows the method.
    """
    titles = [title for title, _section in screen._settings_categories]
    assert "Detection method" in titles
    assert not {"Otsu", "Object detection", "Detection methods"} & set(titles)

    for mode, family in (("otsu", "threshold"), ("li", "threshold"),
                         ("sauvola", "threshold"),
                         ("propagate", "propagate"),
                         ("adaptive", "organelle"),
                         ("cellpose", "cellpose")):
        if screen._mag_mode.findData(mode) < 0:
            continue
        screen._mag_mode.setCurrentIndex(screen._mag_mode.findData(mode))
        assert screen._methods_card.isVisibleTo(screen._settings_scroll)
        shown = {name for name, group in screen._method_groups.items()
                 if group.isVisibleTo(screen._methods_card)}
        assert shown == {family}, f"{mode} showed {shown}"
        assert screen._method_note.text(), f"{mode} says nothing about itself"


def test_a_folded_old_category_stays_folded_after_the_rename():
    """Three titles a user folded become the one that replaced them."""
    for old in ("Otsu", "Object detection", "Detection methods",
                "Cellpose-SAM"):
        assert mm._RENAMED_CATEGORIES[old] == "Detection method"


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
    screen._btn_apply.setChecked(True)
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
    screen._btn_apply.click()
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
        qtbot, screen):
    """One click, two pictures, and the list of what ran between them."""
    screen._enh_clahe.setChecked(True)
    screen._on_compare_enhanced()
    dialog = screen._compare_dialog
    qtbot.waitUntil(lambda: screen._comparison_request is None)
    try:
        assert dialog is not None and dialog.isVisible()
        assert "CLAHE" in dialog.caption.text()
        left, right = (view._item.pixmap().toImage()
                       for view in dialog.views)
        assert not left.isNull() and not right.isNull()
        assert left != right, "the two pictures are the same picture"
    finally:
        dialog.close()


def test_the_compare_window_says_so_when_nothing_is_switched_on(qtbot, screen):
    """An empty chain is reported rather than shown as two same pictures."""
    screen._on_compare_enhanced()
    dialog = screen._compare_dialog
    qtbot.waitUntil(lambda: screen._comparison_request is None)
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


def test_the_compare_window_is_a_window_to_look_in(qtbot, screen):
    """Reported too small to see: it opens big, resizes, and remembers.

    The first version put two 360 px thumbnails in a fixed 780x440 dialog.
    What is asserted is the three things that fixes: a usable opening size,
    pictures that grow with the window rather than staying put, and the
    size being kept for next time.
    """
    screen._enh_clahe.setChecked(True)
    screen._on_compare_enhanced()
    dialog = screen._compare_dialog
    qtbot.waitUntil(lambda: screen._comparison_request is None)
    qtbot.addWidget(dialog)
    try:
        assert dialog.width() >= 1000 and dialog.height() >= 600
        assert dialog.isSizeGripEnabled()

        small = [view.size() for view in dialog.views]
        dialog.resize(1400, 900)
        qtbot.waitUntil(lambda: dialog.views[0].width() > small[0].width(),
                        timeout=2000)
        assert all(view.width() > was.width()
                   for view, was in zip(dialog.views, small)), \
            "the pictures did not grow with the window"
        dialog.close()
    finally:
        dialog.deleteLater()

    assert mm._remembered_compare_size() == (1400, 900)
    again = mm._ComparePreview(blob_field(64), blob_field(64), "x")
    qtbot.addWidget(again)
    try:
        assert (again.width(), again.height()) == (1400, 900)
    finally:
        again.deleteLater()


def test_zooming_one_picture_zooms_the_other(qtbot, screen):
    """Raw and enhanced stay in register, however the zoom is driven."""
    screen._on_compare_enhanced()
    dialog = screen._compare_dialog
    qtbot.waitUntil(lambda: screen._comparison_request is None)
    qtbot.addWidget(dialog)
    dialog.show()
    try:
        left, right = dialog.views
        assert right in left._linked and left in right._linked

        dialog.zoom(2.0)
        assert left.zoom_factor() == pytest.approx(right.zoom_factor()), \
            "the two pictures are at different zooms"
        zoomed = left.zoom_factor()

        left.horizontalScrollBar().setValue(
            left.horizontalScrollBar().maximum())
        assert (right.horizontalScrollBar().value()
                == left.horizontalScrollBar().value()), \
            "panning one picture left the other behind"

        right.zoom_by(1.0 / 2.0)
        assert left.zoom_factor() == pytest.approx(right.zoom_factor())
        assert left.zoom_factor() < zoomed, "the second view drove nothing"

        dialog.fit()
        assert left.zoom_factor() == pytest.approx(right.zoom_factor())
    finally:
        dialog.close()
        dialog.deleteLater()


def test_the_zoom_view_is_the_one_the_qc_browser_already_had():
    """One fit-zoom-pan view in the codebase, not a second copy of it."""
    from spacr.qt.widgets import qc_field_browser
    from spacr.qt.widgets.zoom_view import ZoomableImageView

    assert qc_field_browser._FieldView is ZoomableImageView


def test_every_cpu_threshold_algorithm_is_offered_and_runs(screen):
    """Li's cross entropy and the rest, each a row and each a real cut."""
    from spacr.qt import cpu_modes as cm

    offered = [screen._mag_mode.itemData(i)
               for i in range(screen._mag_mode.count())]
    for mode in ("li", "yen", "triangle", "isodata", "mean", "minimum",
                 "multiotsu", "sauvola", "niblack", "propagate"):
        assert mode in offered, f"{mode} is not offered"
        assert mode in mm._MAGNIFIER_SEGMENTERS
        assert cm.guidance(mode).startswith("Suits "), mode

    field = blob_field(160)
    levels = set()
    for algorithm in engine.GLOBAL_THRESHOLDS:
        levels.add(round(engine._otsu_levels(
            field, algorithm=algorithm)[0], 3))
    assert len(levels) > 3, \
        "the algorithms are all returning the same level; one is not running"


def test_a_threshold_algorithm_is_one_number_and_not_a_second_engine():
    """Choosing Li changes where the level comes from and nothing else.

    Proved by running Otsu and Li through the same call and checking that
    the ONLY difference is the level: at Li's level, the Otsu algorithm
    gives Li's mask to the pixel.
    """
    field = blob_field(160)
    li_level = engine._otsu_levels(field, algorithm="li", smoothing=1.0)[0]
    otsu_level = engine._otsu_levels(field, algorithm="otsu",
                                     smoothing=1.0)[0]
    assert li_level != otsu_level

    by_name = engine._otsu_instances(field, min_area=10, smoothing=1.0,
                                     algorithm="li")
    by_hand = engine._otsu_instances(
        field, min_area=10, smoothing=1.0, algorithm="otsu",
        correction=li_level / otsu_level)
    np.testing.assert_array_equal(by_name, by_hand)


@pytest.mark.parametrize("mode", ["li", "yen", "triangle", "isodata",
                                  "mean", "minimum", "sauvola", "niblack"])
def test_each_algorithm_runs_in_the_box_and_on_the_button(screen, mode):
    """One algorithm, two places, one answer to what it means."""
    field = blob_field(96)
    request = mm._MagnifierRequest(
        key=("k",), crop=field, box=(0, 0, 96, 96), shape=(96, 96),
        mode=mode, sensitivity=0.0, bright=True, min_area=10,
        model_name="cpsam", diameter=0, colour=(1, 2, 3), otsu_window=31)
    labels, used, note = mm._segment_region(request)
    assert used == mode and note == "", note
    assert labels.shape == field.shape

    whole, seeds = screen._cpu_detect(field, mode, screen._otsu_settings())
    assert seeds is None
    assert whole.shape == field.shape


def test_multi_otsu_is_asked_for_by_a_class_count(screen):
    """Multi-Otsu is the class count, and the screen will not leave it at 2."""
    from spacr.qt import cpu_modes as cm

    assert cm.engine_algorithm("multiotsu") == "otsu"
    assert cm.engine_algorithm("li") == "li"

    screen._otsu_classes.setValue(2)
    screen._mag_mode.setCurrentIndex(screen._mag_mode.findData("multiotsu"))
    assert screen._otsu_classes.value() >= 3, \
        "Multi-Otsu was left on two classes, which is plain Otsu"


def test_the_k_box_is_shown_only_for_the_algorithms_that_read_it(screen):
    """Sauvola and Niblack read k; nothing else does."""
    for mode, wanted in (("otsu", False), ("li", False), ("sauvola", True),
                         ("niblack", True), ("propagate", False)):
        screen._mag_mode.setCurrentIndex(screen._mag_mode.findData(mode))
        assert screen._otsu_local_k.isEnabled() is wanted, mode


def test_maxima_and_propagate_grows_one_object_per_centre():
    """Four bright centres, four objects, however they are stopped."""
    from spacr.qt import cpu_modes as cm

    size = 200
    rng = np.random.default_rng(5)
    field = (rng.random((size, size)) * 80).astype(np.float32)
    yy, xx = np.ogrid[:size, :size]
    for cy, cx in ((60, 60), (60, 100), (140, 60), (140, 140)):
        field += 3000 * np.exp(
            -(((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * 18.0 ** 2)))

    for stop, value in (("seed_fraction", 0.4), ("percentile", 95.0),
                        ("threshold", 0.0), ("absolute", 800.0)):
        params = cm.DEFAULT_PARAMS._replace(
            propagate_sigma=3.0, propagate_min_distance=15,
            propagate_seed_level=95.0, propagate_stop=stop,
            propagate_stop_value=value)
        found = cm.propagate(field, params, min_area=50)
        assert found.seeds == 4, f"{stop} found {found.seeds} centres"
        assert int(found.labels.max()) == 4, \
            f"{stop} grew {int(found.labels.max())} objects from 4 centres"
        if stop == "seed_fraction":
            assert found.level is None, \
                "a per-centre rule has no single level to report"
        else:
            assert found.level is not None

    assert set(engine.PROPAGATE_STOPS) == {
        "seed_fraction", "absolute", "percentile", "threshold"}


def test_the_two_touching_objects_a_threshold_cannot_split_come_apart():
    """The reason propagation exists, on a pair that shares a bright bridge."""
    from spacr.qt import cpu_modes as cm

    field = np.zeros((80, 140), dtype=np.float32)
    yy, xx = np.ogrid[:80, :140]
    for cx in (50, 90):
        field += 2000 * np.exp(
            -(((yy - 40) ** 2 + (xx - cx) ** 2) / (2 * 14.0 ** 2)))

    one = engine._otsu_instances(field, min_area=50, split_touching=False)
    assert int(one.max()) == 1, "the pair is meant to threshold as one blob"

    params = cm.DEFAULT_PARAMS._replace(
        propagate_sigma=2.0, propagate_min_distance=12,
        propagate_seed_level=90.0)
    found = cm.propagate(field, params, min_area=50)
    assert found.seeds == 2 and int(found.labels.max()) == 2


def test_the_propagation_says_how_many_centres_it_found(screen):
    """The number the user tunes against reaches the status line."""
    screen._mag_mode.setCurrentIndex(screen._mag_mode.findData("propagate"))
    screen._propagate_widgets["propagate_seed_level"].setValue(99.0)
    screen._propagate_widgets["propagate_min_distance"].setValue(6)
    screen._on_detect_otsu()
    said = screen._status_label.text()
    assert "centre" in said, said


def test_the_propagation_records_its_settings_with_the_mask(screen):
    """A propagated mask carries the four steps that made it."""
    from spacr.qt import cpu_modes as cm

    screen._mag_mode.setCurrentIndex(screen._mag_mode.findData("propagate"))
    screen._propagate_widgets["propagate_stop"].setCurrentIndex(
        screen._propagate_widgets["propagate_stop"].findData("percentile"))
    screen._propagate_widgets["propagate_stop_value"].setValue(92.0)
    screen._on_detect_otsu()
    entries = [edit for edit in screen._log._edits if edit.kind == "detect"]
    assert entries, "the propagation recorded nothing"
    detail = entries[-1].detail
    assert detail["method"] == "propagate"
    assert detail["method_parameters"]["propagate_stop"] == "percentile"
    assert detail["method_parameters"]["propagate_stop_value"] == 92.0
    assert set(cm.PARAMETERS_FOR["propagate"]) <= set(
        detail["method_parameters"])


def test_the_stop_value_and_the_stop_algorithm_are_never_both_live(screen):
    """A control being read and one being ignored must not look alike."""
    screen._mag_mode.setCurrentIndex(screen._mag_mode.findData("propagate"))
    stop = screen._propagate_widgets["propagate_stop"]
    stop.setCurrentIndex(stop.findData("seed_fraction"))
    assert screen._propagate_widgets["propagate_stop_value"].isEnabled()
    assert not screen._propagate_widgets[
        "propagate_stop_algorithm"].isEnabled()
    stop.setCurrentIndex(stop.findData("threshold"))
    assert not screen._propagate_widgets["propagate_stop_value"].isEnabled()
    assert screen._propagate_widgets["propagate_stop_algorithm"].isEnabled()
