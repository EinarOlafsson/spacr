"""The fold buttons are half again as big, and the gap grew with them.

Growing the buttons while leaving the gap alone makes a strip MORE crowded,
not less: the same gap between larger marks is a smaller share of the strip.
So the ratio is what is asserted here, not only the sizes -- a later change
that raises the button edge and forgets the gap fails.
"""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from spacr.qt.widgets.fold_strip import (                       # noqa: E402
    BUTTON_PX, GAP_PX, ICON_PX, FoldStrip,
)

#: What the strip measured before the maintainer called it crowded.
WAS_BUTTON_PX, WAS_ICON_PX, WAS_GAP_PX = 34, 20, 6

#: A narrow window. A masthead also carries a title and the Live and AI
#: buttons, so the strip may not take the whole width at this size.
NARROW_PX = 900


def _strip(count):
    return FoldStrip([(f"key_{i}", lambda *_: None) for i in range(count)])


def test_the_button_is_half_again_as_big(qapp):
    assert BUTTON_PX == round(WAS_BUTTON_PX * 1.5)
    assert ICON_PX == round(WAS_ICON_PX * 1.5)


def test_the_gap_grew_with_them(qapp):
    """The point of the change: the strip must not read tighter than before."""
    assert GAP_PX / BUTTON_PX == pytest.approx(WAS_GAP_PX / WAS_BUTTON_PX,
                                               abs=0.02)
    assert GAP_PX > WAS_GAP_PX


def test_the_icon_still_sits_inside_its_plate(qapp):
    """The hover fill reads as a plate behind the mark, not a border on it."""
    assert ICON_PX < BUTTON_PX
    assert BUTTON_PX - ICON_PX >= 12


@pytest.mark.parametrize("count", [1, 2, 3, 5, 8])
def test_the_strip_is_exactly_its_buttons_and_gaps(qapp, count):
    strip = _strip(count)
    expected = count * BUTTON_PX + max(0, count - 1) * GAP_PX
    assert strip.sizeHint().width() == expected
    for button in strip.buttons:
        assert button.size().width() == BUTTON_PX
        assert button.size().height() == BUTTON_PX


def test_every_shipped_strip_fits_a_narrow_window(qapp):
    """Larger buttons are what would push a masthead into a scroll.

    Asked of the strips that SHIP rather than of a round number: the widest
    host folds three modules, and a strip is measured against half a narrow
    window because the masthead also carries a heading and its own buttons.
    """
    import importlib
    import pkgutil

    import spacr.qt.screens as screens

    widest = 0
    for found in pkgutil.iter_modules(screens.__path__):
        try:
            module = importlib.import_module(
                "spacr.qt.screens." + found.name)
        except Exception:                               # noqa: BLE001
            continue
        folds = getattr(module, "FOLDED_APPS", ()) or ()
        if not folds:
            continue
        widest = max(widest, len(folds))
        width = _strip(len(folds)).sizeHint().width()
        assert width <= NARROW_PX // 2, (
            f"{found.name} folds {len(folds)} modules wanting {width}px "
            f"of a {NARROW_PX}px window")
    assert widest >= 3, "no host folds anything; this test proves nothing"


def test_the_strip_says_when_it_would_overflow(qapp):
    """The ceiling, so a future fold does not silently break a masthead.

    Seven buttons at this size already want more than half a narrow window.
    A host that folds that many needs the strip to wrap or scroll first --
    this records where the limit is rather than leaving it to be found.
    """
    budget = NARROW_PX // 2
    fits = [n for n in range(1, 12)
            if _strip(n).sizeHint().width() <= budget]
    assert max(fits) == 7, (
        f"the strip now fits {max(fits)} buttons in {budget}px, not 7")
