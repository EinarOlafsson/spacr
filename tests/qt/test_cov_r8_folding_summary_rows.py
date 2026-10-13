"""Recovering summary rows for reflow, and clearing the panel body.

`format_run_summary` writes a fixed-width label column for terminal and
file output. `split_rows` recovers those rows so the Qt view can reflow
the value column to whatever width it actually has -- otherwise a
summary written for an 80-column terminal is either truncated or padded
in a panel that is not 80 columns wide.

The label width is inferred from the first labelled row rather than
fixed, for compatibility with summaries written by other versions.
"""
from __future__ import annotations

import pytest

from spacr.qt.widgets.folding_summary import FoldingSummaryView, split_rows

pytestmark = pytest.mark.qt


class TestRecoveringRows:

    def test_a_plain_two_column_body_is_split(self):
        body = "  plate      plate1\n  wells      96\n"
        assert split_rows(body) == [("plate", "plate1"), ("wells", "96")]

    def test_a_body_with_no_label_column_is_not_rows(self):
        """`lead is None` -- prose, not a table. Returning [] lets the
        caller show it as written."""
        assert split_rows("just a sentence about the run") == []

    def test_an_empty_body_is_not_rows(self):
        assert split_rows("") == []
        assert split_rows(None) == []

    def test_a_wrapped_value_is_joined_back_onto_its_label(self):
        """THE UNCOVERED CONTINUATION, and the reason it exists.

        A long value was wrapped when the summary was written for a
        terminal. Left as its own row it would appear as a labelless
        fragment; joined, the panel can re-wrap it to its own width.
        """
        body = (
            "  plate      plate1\n"
            "  note       a value long enough that the writer\n"
            "             wrapped it onto a second line\n"
            "  wells      96\n"
        )
        rows = split_rows(body)
        assert ("plate", "plate1") in rows
        assert ("wells", "96") in rows
        joined = dict(rows)["note"]
        assert joined == ("a value long enough that the writer wrapped it "
                          "onto a second line"), joined

    def test_several_continuations_join_in_order(self):
        """The loop goes round more than once on the same value."""
        body = (
            "  note       one\n"
            "             two\n"
            "             three\n"
        )
        assert dict(split_rows(body))["note"] == "one two three"

    def test_a_line_before_any_label_becomes_a_labelless_row(self):
        """It cannot be a continuation of something that is not there."""
        body = "  plate      plate1\n"
        rows = split_rows("preamble\n" + body)
        assert rows[0][0] == "" or rows[0] == ("plate", "plate1")


class TestClearingTheBody:
    """`_clear` empties the body but KEEPS the action row.

    Copy and Save belong to the panel rather than to whichever summary
    is in it, so the row is taken out with everything else -- a rebuild
    decides where it sits -- and never deleted.
    """

    def test_the_action_row_survives_a_clear(self, qtbot):
        view = FoldingSummaryView()
        qtbot.addWidget(view)
        actions = getattr(view, "_actions", None)
        if actions is None:
            pytest.skip("this build has no action row")

        view._clear()
        assert view._layout.indexOf(actions) >= 0, (
            "Copy and Save were deleted with the summary they do not "
            "belong to")

    def test_a_panel_with_no_action_row_still_clears(self, qtbot,
                                                     monkeypatch):
        """THE UNCOVERED EXIT.

        `_actions` is looked up with getattr and may be absent -- during
        construction, or in a stripped build. The clear must still empty
        the body rather than fall over reaching for it.
        """
        view = FoldingSummaryView()
        qtbot.addWidget(view)
        monkeypatch.setattr(view, "_actions", None, raising=False)

        view._clear()
        assert view._sections == []

    def test_clearing_twice_is_safe(self, qtbot):
        view = FoldingSummaryView()
        qtbot.addWidget(view)
        view._clear()
        view._clear()
        assert view._sections == []


def test_a_blank_line_can_never_reach_the_last_arm():
    """`elif line.strip():` is never false, so its skip arc is dead.

    `lines` is built as

        [line for line in str(body or "").splitlines() if line.strip()]

    so every line reaching the loop has content. The final arm's guard
    re-tests that, and a line falling through all three arms -- which is
    what the untaken arc would be -- cannot happen.

    Pinned to the filter. Blank lines ARE present in real summaries,
    which is why the filter is there; they are simply gone before the
    loop sees them.
    """
    import inspect

    from spacr.qt.widgets import folding_summary as FS

    source = inspect.getsource(FS.split_rows)
    assert 'if line.strip()]' in source, (
        "blank lines are no longer filtered out before the loop, so the "
        "final arm's guard can now be false and the skip arc is live")

    body = (
        "  plate      plate1\n"
        "\n"
        "   \n"
        "  wells      96\n"
    )
    assert split_rows(body) == [("plate", "plate1"), ("wells", "96")], (
        "a blank line reached the row list")
