"""A rejected assumption is red; an accepted one and a caution are not.

Instruction 225: "whan a test has a broken assumption the text showing that
that assumption is broken should be red".

THE SUMMARY ALREADY SAID "REJECTED" and the word was doing no work, because
it was the same colour as everything around it:

    Breusch-Pagan p = 1 (not rejected at 0.05) ... consistent with equal
    variance
    ... D'Agostino K2 p = 4.96e-157 (REJECTED at 0.05)
    Durbin-Watson = 1.22 (2 is none) -- positive autocorrelation

Three lines, one of them fatal to the p-values, and nothing on screen said
which.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

pytestmark = pytest.mark.qt

SUMMARY = """ASSUMPTIONS
Breusch-Pagan p = 1 (not rejected at 0.05) -- consistent with equal variance
D'Agostino K2 p = 4.96e-157 (REJECTED at 0.05)
max leverage 1 against the 2p/n guide of 0.887 (455 above it)

RECOMMENDATIONS
  ! inference: 'nonparametric'
  - grna_statistic: 'rank'
"""


def _colour_of(widget, needle):
    """The colour the line containing ``needle`` is drawn in, or ``None``.

    Reads the RENDERED document rather than the source, and searches both
    kinds of body. MOST OF A SUMMARY IS A TABLE, not a text block --
    anything shaped "label: value" becomes a row -- which is how the first
    version of this test failed against a highlighter that was working
    perfectly on the one body that was not a table.
    """
    from PySide6.QtWidgets import QPlainTextEdit, QTextBrowser

    # ONE TYPE PER CALL: PySide6's findChildren takes a type, not a tuple of
    # them, and passing a tuple raises rather than matching nothing.
    views = (list(widget.findChildren(QPlainTextEdit))
             + list(widget.findChildren(QTextBrowser)))
    for view in views:
        block = view.document().begin()
        while block.isValid():
            if needle in block.text():
                layout = block.layout()
                formats = layout.formats() if layout else []
                for span in formats:
                    colour = span.format.foreground().color()
                    if colour.isValid() and colour.alpha():
                        return colour.name()
                return None
            block = block.next()
    raise AssertionError(f"no line containing {needle!r}")


@pytest.fixture
def view(qtbot):
    from PySide6.QtWidgets import QPlainTextEdit

    from spacr.qt.widgets.folding_summary import FoldingSummaryView

    widget = FoldingSummaryView()
    qtbot.addWidget(widget)
    widget.setPlainText(SUMMARY)
    return widget


class TestOnlyTheRejectedLineIsRed:

    def test_the_rejected_assumption_is_coloured(self, view):
        from spacr.qt.theme import active_palette

        colour = _colour_of(view, "REJECTED at")
        assert colour is not None
        assert colour.lower() == active_palette()["error"].lower()

    def test_an_accepted_assumption_is_not(self, view):
        assert _colour_of(view, "not rejected at") is None

    def test_a_caution_is_not(self, view):
        """A leverage count above a rule of thumb is a caution, not a
        failure. Colour that fires on everything means nothing."""
        assert _colour_of(view, "max leverage") is None

    def test_a_blocking_recommendation_is_coloured(self, view):
        from spacr.qt.theme import active_palette

        colour = _colour_of(view, "! inference")
        assert colour is not None
        assert colour.lower() == active_palette()["error"].lower()

    def test_a_suggestion_is_not(self, view):
        assert _colour_of(view, "- grna_statistic") is None


class TestTheWholeLine:

    def test_it_is_not_only_the_matched_word(self, view):
        """"REJECTED at 0.05" is the verdict on the sentence it sits in.
        Colouring three words inside a grey line reads as emphasis rather
        than as a state."""
        from PySide6.QtWidgets import QPlainTextEdit, QTextBrowser

        for one in (list(view.findChildren(QPlainTextEdit))
                    + list(view.findChildren(QTextBrowser))):
            block = one.document().begin()
            while block.isValid():
                if "REJECTED at" in block.text():
                    spans = block.layout().formats()
                    assert spans
                    assert spans[0].start == 0
                    assert spans[0].length == len(block.text())
                    return
                block = block.next()
        raise AssertionError("no rejected line")


class TestItSurvivesTheThingsThatBreakHighlighters:

    def test_the_highlighter_is_held(self, view):
        """A QSyntaxHighlighter that is garbage collected highlights
        nothing, silently -- the only way one ever fails."""
        from PySide6.QtWidgets import QPlainTextEdit

        assert any(getattr(one, "_spacr_highlighter", None) is not None
                   for one in view.findChildren(QPlainTextEdit))

    def test_the_colour_comes_from_the_palette(self):
        import inspect

        from spacr.qt.widgets import folding_summary

        body = inspect.getsource(folding_summary.FoldingSummaryView._block)
        assert "active_palette" in body
        for literal in ("#f00", "#ff0000", "Qt.red", "'red'"):
            assert literal not in body, literal

    def test_a_summary_with_nothing_rejected_is_untouched(self, qtbot):
        from PySide6.QtWidgets import QPlainTextEdit

        from spacr.qt.widgets.folding_summary import FoldingSummaryView

        widget = FoldingSummaryView()
        qtbot.addWidget(widget)
        widget.setPlainText("ASSUMPTIONS\nall good\nnothing to report\n")
        assert _colour_of(widget, "all good") is None
