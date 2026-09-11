"""A percentile box stores what it shows, so its decimals are its precision.

Reported 2026-09-10 -- "the normalization percentiles should be able to have
as many desimals as the user wants, now they are capped at 2 desimals."

THE CAP WAS NEVER CHOSEN. ``QDoubleSpinBox`` ships with two decimals and
three separate pairs of percentile boxes never overrode it:
``LivePreviewPanel._lo_pct``/``_hi_pct``, ``MeasurePreviewPanel``'s pair, and
the annotator's ``_pct_lo``/``_pct_hi``. ``make_masks.py`` and
``percentile_pair.py`` had both already settled on six; these three were
simply missed.

AND IT ROUNDS THE VALUE, NOT THE DISPLAY. A ``QDoubleSpinBox`` stores what
it shows, so typing 99.995 into a two-decimal box leaves 100.0 behind -- the
top of the range, which is *no stretch at all*. The user asked for a
narrower clip and got a wider one, silently.

WHY THE TOP END SPECIFICALLY. The whole point of an upper percentile is to
clip a few saturated pixels without touching real signal. On a 4-megapixel
field 99.9 spares 4,000 pixels, 99.99 spares 400 and 99.999 spares 40 --
three very different images. Two decimals cannot express the difference
between the last two, so a user chasing a handful of hot pixels has nothing
between "99.99" and "off".

**Every assertion here is paired with a control**, because the neighbouring
file in this directory had a guard that quietly stopped reproducing its own
defect and passed for months on staging that no longer worked. A
two-decimal box built inside the test is the chance baseline: it must LOSE
the value the shipped boxes keep. If the control ever stops losing it, the
measurement is no longer measuring anything and these tests say so rather
than going green.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QDoubleSpinBox

#: What the user typed in the report's units. Three decimals, which is one
#: more than Qt's default can hold -- the smallest value that demonstrates
#: the defect.
TYPED = 99.995

#: The neighbouring rung of the ladder, for the "does it actually change the
#: picture" half.
COARSER = 99.99


@pytest.fixture(autouse=True)
def _qapp(qapp):
    """Constructing any of these panels needs a QGuiApplication."""
    return qapp


def _two_decimal_control() -> QDoubleSpinBox:
    """A box carrying Qt's default precision: the chance baseline.

    Not ``setDecimals(2)`` -- left untouched, which is exactly the state the
    three shipped pairs were in. If this ever holds ``TYPED`` intact then Qt
    changed its default and every assertion below is vacuous.
    """
    box = QDoubleSpinBox()
    box.setRange(0.0, 100.0)
    return box


def _field_with_a_hot_tail() -> np.ndarray:
    """Four megapixels of background under 400 pixels of increasing heat.

    Sized so the two percentiles under test cannot land on the same value:
    99.99 of 4,000,000 leaves the top 400 above it and 99.995 leaves the top
    200, so one cut falls at the bottom of the hot tail and the other
    halfway up it. Every hot pixel carries a distinct value for the same
    reason -- a flat tail would make both percentiles the same number and
    the comparison would pass on a tie.
    """
    rng = np.random.default_rng(391)
    image = rng.integers(90, 110, size=(2000, 2000)).astype(np.uint16)
    flat = image.reshape(-1)
    hot = rng.choice(flat.size, size=400, replace=False)
    flat[hot] = np.arange(1000, 1400, dtype=np.uint16)
    return image


def test_qt_still_defaults_to_two_decimals():
    """The control is a control. Everything below rests on this."""
    assert _two_decimal_control().decimals() == 2
    control = _two_decimal_control()
    control.setValue(TYPED)
    assert control.value() != TYPED, (
        "a box at Qt's default precision kept 99.995 intact, so it is no "
        "longer a baseline for anything and these tests prove nothing")


def test_the_shared_constant_is_what_the_boxes_use():
    """Six, and one opinion about six rather than four of them."""
    from spacr.qt.screens.make_masks import PERCENTILE_DECIMALS
    from spacr.qt.widgets.percentile_pair import DECIMALS

    assert DECIMALS == PERCENTILE_DECIMALS == 6


def test_the_live_preview_percentiles_keep_what_was_typed():
    """The pair named in the report."""
    from spacr.qt.widgets.live_preview import LivePreviewPanel

    panel = LivePreviewPanel()
    try:
        for box in (panel._lo_pct, panel._hi_pct):
            assert box.decimals() == 6
        panel._hi_pct.setValue(TYPED)
        assert panel._hi_pct.value() == pytest.approx(TYPED, abs=1e-9)
        # The lower box's range stops at 50, so it gets its own end of the
        # ladder rather than the same number.
        panel._lo_pct.setValue(0.0001)
        assert panel._lo_pct.value() == pytest.approx(0.0001, abs=1e-9)
    finally:
        panel.deleteLater()


def test_the_crop_preview_percentiles_keep_what_was_typed():
    """The same pair on the measure screen, missed the same way."""
    from spacr.qt.widgets.measure_preview import MeasurePreviewPanel

    panel = MeasurePreviewPanel(threaded=False)
    try:
        for box in (panel._lo_pct, panel._hi_pct):
            assert box.decimals() == 6
        panel._hi_pct.setValue(TYPED)
        assert panel._hi_pct.value() == pytest.approx(TYPED, abs=1e-9)
    finally:
        panel.deleteLater()


def test_the_annotator_does_not_round_a_stored_percentile_on_the_way_in():
    """The annotator's pair edits settings IN PLACE, which doubles the cost.

    ``AnnotateSettings.percentiles`` is a plain float pair on disk and can
    already hold 99.9999. A two-decimal box rounded it while LOADING, and
    the dialog then wrote the rounded value back on the next save -- so
    opening the settings dialog and pressing OK was enough to lose precision
    the user never touched.
    """
    from spacr.qt.screens.annotate import AnnotateSettings, _SettingsDialog

    settings = AnnotateSettings()
    settings.percentiles = [0.0001, TYPED]
    dialog = _SettingsDialog(settings)
    try:
        assert dialog._pct_lo.decimals() == 6
        assert dialog._pct_hi.decimals() == 6
        assert dialog._pct_hi.value() == pytest.approx(TYPED, abs=1e-9)
        assert dialog._pct_lo.value() == pytest.approx(0.0001, abs=1e-9)
    finally:
        dialog.deleteLater()


def test_the_decimals_are_set_before_the_range_and_the_value():
    """Ordering, which is the half that looks like a broken control.

    ``setDecimals`` after ``setValue`` rounds the value already stored, so a
    box can carry six decimals and still hold 100.0. The panels set the
    precision first; this pins that by doing it the wrong way round and
    showing the value does not survive.
    """
    wrong = QDoubleSpinBox()
    wrong.setRange(0.0, 100.0)
    wrong.setValue(TYPED)          # rounded here, at two decimals
    wrong.setDecimals(6)           # too late: 100.0 is already what is held
    assert wrong.value() != pytest.approx(TYPED, abs=1e-9)

    right = QDoubleSpinBox()
    right.setDecimals(6)
    right.setRange(0.0, 100.0)
    right.setValue(TYPED)
    assert right.value() == pytest.approx(TYPED, abs=1e-9)


def test_the_extra_decimals_change_the_picture():
    """Precision the renderer cannot use would not be worth the control.

    The claim under test is the one from the report: 99.995 is a different
    image from 99.99. Both are pushed through the panel's own
    ``_to_uint8`` -- the function ``_refresh_canvases`` calls -- rather than
    a reimplementation of the stretch.
    """
    from spacr.qt.widgets.live_preview import _to_uint8

    field = _field_with_a_hot_tail()
    coarse = _to_uint8(field, normalise=True, lo_pct=2.0, hi_pct=COARSER)
    fine = _to_uint8(field, normalise=True, lo_pct=2.0, hi_pct=TYPED)

    assert coarse.shape == fine.shape
    assert not np.array_equal(coarse, fine), (
        f"{COARSER} and {TYPED} rendered byte-identical images, so the "
        "decimals the user gained buy nothing and the field is precise "
        "about a distinction the renderer cannot make")

    # And the direction is the one the report describes: a HIGHER upper
    # percentile clips fewer pixels, so it maps more of the range onto the
    # dark end and the picture gets dimmer overall.
    assert fine.mean() < coarse.mean()
