"""The explainer text spans the width of its box, and formulas never break.

Instruction 138, asked for on 2026-08-18 once per box: "in model and inference
in regression module the text in the text box should span the width of the
textbox", and the same for Permutation Test.

It did not. `_wrap_block` ran `textwrap.wrap` at a fixed 54 columns and the
widget was set to NoWrap, so the paragraph was 54 characters wide whatever the
pane was: widening the settings pane widened the box and left its right-hand
side empty.
"""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

pytestmark = pytest.mark.qt

FORMULA_STARTS = ("y ~", "rho =", "minimise")


def _every_explainer():
    from spacr.qt.screens.settings_model import (REGRESSION_LEVELS,
                                                 regression_model_explainer)
    from spacr.regression_spec import REGRESSION_TYPES

    for family in sorted(REGRESSION_TYPES):
        for level in REGRESSION_LEVELS:
            yield family, level, regression_model_explainer(family, level)


def test_no_paragraph_is_hard_wrapped_any_more():
    """A prose line is one paragraph, so the widget can reflow it.

    The old shape is what this rules out: many lines all around 54
    characters, which is a paragraph that has already been broken and cannot
    be un-broken by any box width.
    """
    for family, level, text in _every_explainer():
        prose = [line for line in text.splitlines()
                 if line.startswith("    ")
                 and not line.strip().startswith(FORMULA_STARTS)]
        if len(prose) < 3:
            continue
        long_ones = [line for line in prose if len(line) > 90]
        assert long_ones, (
            f"{family}/{level}: every prose line is short, so the paragraph "
            f"is still hard-wrapped and no box width can reflow it")


def test_the_box_wraps_at_its_own_width(qtbot):
    """The property the request asked for, MEASURED by resizing.

    A box that reflows gets shorter as it gets wider. Measured on the mixed
    explainer: 51 lines at 500px, 39 at 760px, 33 at 1100px.
    """
    from PySide6.QtGui import QFontDatabase
    from PySide6.QtWidgets import QPlainTextEdit

    from spacr.qt.screens.settings_model import regression_model_explainer

    box = QPlainTextEdit()
    qtbot.addWidget(box)
    box.setLineWrapMode(QPlainTextEdit.WidgetWidth)
    box.setFont(QFontDatabase.systemFont(QFontDatabase.FixedFont))
    box.setPlainText(regression_model_explainer("mixed", "both"))
    box.show()

    heights = []
    for width in (500, 760, 1100):
        box.resize(width, 300)
        qtbot.wait(1)
        heights.append(box.document().documentLayout().documentSize().height())

    assert heights[0] > heights[1] > heights[2], heights


def test_the_box_the_panel_builds_wraps_at_its_width(qtbot):
    """Not a box built by this test -- the one the screen actually installs."""
    from PySide6.QtWidgets import QPlainTextEdit

    from spacr.qt.screens.app_screen import AppScreen

    screen = AppScreen("regression")
    qtbot.addWidget(screen)
    assert screen._section_explainers
    for title, box in screen._section_explainers.items():
        assert box.lineWrapMode() == QPlainTextEdit.WidgetWidth, title


def test_no_formula_can_ever_wrap():
    """The reason the box exists. A formula split at an arbitrary column is
    not a formula, and copying one into a methods section is the point of
    having it on screen."""
    from spacr.qt.screens.settings_model import explainer_width

    floor = explainer_width()
    for family, level, text in _every_explainer():
        for line in text.splitlines():
            if line.strip().startswith(FORMULA_STARTS):
                assert len(line) <= floor, (
                    f"{family}/{level}: the formula is {len(line)} characters "
                    f"and the box floor is {floor}, so it can wrap")


def test_the_floor_is_measured_from_the_explainers_not_declared():
    """A formula added to any explainer must widen the floor with it, rather
    than becoming the first thing to wrap."""
    from spacr.qt.screens import settings_model

    floor = settings_model.explainer_width()
    longest = max(
        (len(line) for _f, _l, text in _every_explainer()
         for line in text.splitlines()
         if line.strip().startswith(FORMULA_STARTS)), default=0)
    assert floor >= longest
    assert floor >= settings_model._EXPLAINER_WIDTH


def test_the_permutation_box_is_prose_and_reflows(qtbot):
    """It has no formula, so nothing in it is unbreakable -- and it must
    still fill the box rather than sitting at a fixed column."""
    from spacr.qt.screens.settings_model import section_explainer

    text = section_explainer("regression", "Permutation Test")
    assert text
    prose = [line for line in text.splitlines() if line.strip()]
    assert any(len(line) > 90 for line in prose), (
        "the permutation box is still hard-wrapped")
