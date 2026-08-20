"""Instruction 138 — the prose fills the width it is given.

Asked twice, once per box:

    "in model and inference in regression module the text in the text box
     should span the width of the textbox"
    "in permutation test in regression module the text in the text box should
     span the width of the textbox"

Delivered by 144's move to rich text, and this is the test that keeps it.
The two constraints used to be unwinnable together in plain text, where
wrapping is per WIDGET: NoWrap kept a formula on one line and left the prose
hard-wrapped at a fixed column, and turning it on broke every prose line a
SECOND time at a different column. In rich text wrapping is per BLOCK, so the
prose reflows and the `<pre>` formula does not.

The measurement is the laid-out HEIGHT at four widths, taken on THE BOX'S OWN
document. Two traps, both hit while writing this:

* line counts are not available until a layout has run, and a document that
  has not been laid out reports zero lines for everything -- which reads as a
  passing test;
* `toHtml()` does not round-trip a `<pre>`. Qt serialises it as a paragraph
  with `white-space: pre-wrap`, so a document re-parsed from that string wraps
  the formula that the live one holds on a single line. Measuring the string
  reported a bug that does not exist on screen.

And a third, in the measurement itself: a document re-measured at a second
width reports a stale height. Each width gets a fresh clone.
"""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

pytestmark = pytest.mark.qt

from PySide6.QtGui import QTextDocument                          # noqa: E402

#: Wide enough that the un-wrappable formula is not what sets the width. The
#: narrowest the settings pane goes in practice is about 300.
WIDTHS = (320, 460, 640, 900)


def _laid_out(box, width: int):
    """A CLONE of the box's own document, laid out at ``width``.

    A clone per width, not one document re-measured: calling `setTextWidth`
    twice on the same document returns a stale height for the second width --
    measured here as 540, 624, 540, 470 for widths that reflow monotonically
    from 820 to 470 when each is laid out fresh. A clone carries the block
    formats, which is what `toHtml()` does not (see the module docstring).
    """
    document = box.document().clone()
    document.setTextWidth(width)
    document.size()                     # forces the layout
    return document


def _heights(box) -> list:
    return [_laid_out(box, w).size().height() for w in WIDTHS]


def _code_blocks(document) -> list:
    """``(line count, non-breakable)`` for every patsy-formula block."""
    out, block = [], document.begin()
    while block.isValid():
        if block.text().strip().startswith("y ~"):
            out.append((block.layout().lineCount(),
                        block.blockFormat().nonBreakableLines()))
        block = block.next()
    return out


def _screen(qtbot):
    from spacr.qt.screens.app_screen import AppScreen

    screen = AppScreen("regression")
    qtbot.addWidget(screen)
    return screen


def test_the_model_box_reflows_instead_of_keeping_a_fixed_column(qtbot):
    heights = _heights(_screen(qtbot)._model_explainer)

    # Strictly shorter at every step: a box twice as wide holds the same
    # prose in fewer lines. A hard-wrapped paragraph would be the same
    # height at all four and leave the right-hand side empty.
    assert heights == sorted(heights, reverse=True)
    assert heights[0] > heights[-1] * 1.5


def test_the_prose_actually_uses_the_width_it_is_given(qtbot):
    """`idealWidth` is what the text WANTS; it should track the box."""
    box = _screen(qtbot)._model_explainer
    for width in WIDTHS:
        ideal = _laid_out(box, width).idealWidth()
        assert ideal > width * 0.9, f"the text stopped short at width {width}"


def test_the_code_formula_is_never_broken_however_narrow_the_box(qtbot):
    """A formula pasted into a methods section arrives as the line it was."""
    box = _screen(qtbot)._model_explainer
    # 240 is narrower than the pane ever gets, deliberately: the formula is
    # what must not break, so it is tested past the point the prose is
    # comfortable.
    for width in (240,) + WIDTHS:
        blocks = _code_blocks(_laid_out(box, width))
        assert blocks, "the box printed no patsy formula at all"
        for lines, non_breakable in blocks:
            assert lines == 1, f"the formula wrapped at width {width}"
            assert non_breakable is True


def test_every_prose_box_on_the_screen_reflows_not_only_the_model_one(qtbot):
    """138 was asked twice — Model & Inference AND Permutation Test."""
    screen = _screen(qtbot)
    boxes = getattr(screen, "_section_explainers", {}) or {}
    assert boxes, "the regression screen built no prose boxes at all"

    for title, box in boxes.items():
        if box is None:
            continue
        heights = _heights(box)
        assert heights == sorted(heights, reverse=True), (
            f"the {title} box does not reflow")


def test_the_boxes_are_rich_text_so_the_two_wraps_stop_fighting(qtbot):
    from PySide6.QtWidgets import QTextEdit

    screen = _screen(qtbot)
    box = screen._model_explainer
    assert isinstance(box, QTextEdit)
    assert box.lineWrapMode() == QTextEdit.WidgetWidth
    # And still copyable, which is why it is not a QLabel.
    assert box.isReadOnly()
    assert box.textInteractionFlags() != 0
