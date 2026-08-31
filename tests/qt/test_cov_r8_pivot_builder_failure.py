"""The pivot panel refusing a table it cannot build.

`recompute` has two failure arms and they say different things. A
`PivotError` is the pivot module's own refusal -- it knows why, so its
message is shown as it stands. Anything else is unexpected, so the panel
says "could not build that table" and puts the detail in the log.

Both clear the previous result first. A panel that kept the last good
table on screen under a failure notice would be showing numbers from a
different question.
"""
from __future__ import annotations

import pandas as pd
import pytest

from spacr.qt.widgets import pivot_builder as PB

pytestmark = pytest.mark.qt


@pytest.fixture()
def panel(qtbot):
    widget = PB.PivotPanel()
    qtbot.addWidget(widget)
    widget._frame = pd.DataFrame({
        "plateID": ["p1", "p1", "p2", "p2"],
        "rowID": ["r1", "r2", "r1", "r2"],
        "cell_area": [1.0, 2.0, 3.0, 4.0],
    })
    # A NON-EMPTY SPEC. `recompute` refuses an empty one before it ever
    # calls `pivot`, with its own "drop a column onto Rows" message -- so
    # without this the failure arms below are never reached.
    spec = PB.PivotSpec(rows=("plateID",), cols=(), values=("cell_area",),
                        aggs=("mean",), quantile=0.5)
    widget.spec = lambda: spec
    return widget


class TestWhenThePivotRefuses:

    def test_a_pivot_error_is_shown_as_it_stands(self, panel, monkeypatch):
        """The pivot module knows why; the panel does not paraphrase."""
        def refuse(_frame, _spec):
            raise PB.PivotError("group by something that is in the table")

        monkeypatch.setattr(PB, "pivot", refuse)
        assert panel.recompute() is None
        assert panel.notice.text() == (
            "group by something that is in the table")
        assert panel._result is None

    def test_an_unexpected_failure_is_named_generically(self, panel,
                                                        monkeypatch, caplog):
        """THE UNCOVERED ARM.

        Not a refusal the pivot module recognises -- a bug, or a frame of
        a shape nobody anticipated. The panel still has to stay on
        screen and say something, and the detail goes to the log rather
        than into a notice nobody can act on.
        """
        def explode(_frame, _spec):
            raise ValueError("reshape of a non-numeric column")

        monkeypatch.setattr(PB, "pivot", explode)
        with caplog.at_level("INFO"):
            assert panel.recompute() is None

        text = panel.notice.text()
        assert text.startswith("could not build that table")
        assert "reshape of a non-numeric column" in text
        assert panel._result is None

    def test_a_failure_clears_the_table_that_was_there(self, panel,
                                                       monkeypatch):
        """A panel showing the previous table under a failure notice
        would be showing numbers from a different question."""
        cleared = []
        monkeypatch.setattr(panel.table, "set_result",
                            lambda result: cleared.append(result))
        monkeypatch.setattr(
            PB, "pivot",
            lambda *_a: (_ for _ in ()).throw(ValueError("no")))

        panel.recompute()
        assert cleared == [None], "the stale result was left on screen"
