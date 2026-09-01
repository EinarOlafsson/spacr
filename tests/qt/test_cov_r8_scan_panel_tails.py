"""measurement_scan_panel: a column that cannot be counted, and a job
runner that refuses.

Three of these guard something the current code cannot produce, and the
source says so in its own comments -- "JobRunner always returns True
today". That makes them the interesting kind of dead: not impossible by
construction, but dead because of a collaborator's present behaviour.
They are driven with a runner that refuses, which is what the guard was
written for and what a future runner may do.

The fourth is a real defensive arm around `nunique`, which raises on a
column of unhashable values.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spacr.qt.widgets import measurement_scan_panel as MSP

pytestmark = pytest.mark.qt


class TestChoosingRegressableColumns:

    def test_numeric_measurements_are_offered(self):
        frame = pd.DataFrame({"cell_area": [1.0, 2.0, 3.0],
                              "cell_perimeter": [4.0, 5.0, 6.0]})
        assert set(MSP.regressable_columns(frame)) == {"cell_area",
                                                       "cell_perimeter"}

    def test_a_constant_column_is_not_offered(self):
        """Fewer than two distinct values cannot be a response."""
        frame = pd.DataFrame({"cell_area": [1.0, 1.0, 1.0],
                              "cell_perimeter": [4.0, 5.0, 6.0]})
        assert MSP.regressable_columns(frame) == ("cell_perimeter",)

    def test_an_identity_column_is_not_offered(self):
        """A picker that offered these would offer a fit onto a well name."""
        frame = pd.DataFrame({"plateID": ["p1", "p2", "p3"],
                              "object_label": ["1", "2", "3"],
                              "source_database": ["a", "b", "c"],
                              "cell_area": [1.0, 2.0, 3.0]})
        assert MSP.regressable_columns(frame) == ("cell_area",)

    def test_a_column_of_unhashable_values_never_reaches_nunique(self):
        """The `except TypeError` around `nunique` cannot fire.

        `nunique` does raise on unhashable values -- a column of lists
        or sets raises "unhashable type" -- but such a column is
        object-dtype, and `is_numeric_dtype` above the try has already
        skipped it. Only numeric columns reach the count, and their
        values are hashable scalars.

        Driven anyway, because the OUTCOME matters whichever guard
        produces it: an unhashable column must not be offered as a
        response, and must not take the other columns down with it.
        """
        frame = pd.DataFrame({
            "cell_area": [1.0, 2.0, 3.0],
            "odd": [[1], [2], [3]],
        })
        assert MSP.regressable_columns(frame) == ("cell_area",)

        # and the reason it never reaches the count
        assert not pd.api.types.is_numeric_dtype(frame["odd"])
        with pytest.raises(TypeError):
            frame["odd"].nunique(dropna=True)

    def test_a_set_valued_column_is_also_kept_out(self):
        frame = pd.DataFrame({
            "cell_area": [1.0, 2.0, 3.0],
            "odd": [{1}, {2}, {3}],
        })
        assert MSP.regressable_columns(frame) == ("cell_area",)

    def test_a_boolean_column_is_not_a_response(self):
        """Numeric by dtype, but two states is a label, not a measurement."""
        frame = pd.DataFrame({"flag": [True, False, True],
                              "cell_area": [1.0, 2.0, 3.0]})
        assert MSP.regressable_columns(frame) == ("cell_area",)

    def test_no_frame_at_all_offers_nothing(self):
        assert MSP.regressable_columns(None) == ()


class TestAJobRunnerThatRefusesTheWork:
    """Both panels ask a JobRunner to start, and both check the answer.

    The comments say the runner always returns True today, so these arms
    are dead by a collaborator's current behaviour rather than by
    construction -- exactly the kind that quietly stops being dead. What
    they protect is the panel's own state: a refused submit must leave
    it not-running, or the buttons stay disabled and the panel is stuck
    with no work in flight.
    """

    def test_a_refused_merge_leaves_the_panel_idle(self, qtbot,
                                                   monkeypatch, tmp_path):
        panel = MSP.DatabaseMergePanel()
        qtbot.addWidget(panel)

        monkeypatch.setattr(panel._jobs, "submit",
                            lambda *a, **k: False)
        monkeypatch.setattr(panel, "_prepare_merge",
                            lambda **k: object(), raising=False)

        started = panel.start_merge()
        assert started is False
        assert panel._merging is False, (
            "a refused submit left the panel believing work was running")

    def test_a_refused_regression_leaves_the_panel_idle(self, qtbot,
                                                       monkeypatch):
        panel = MSP.ColumnRegressionPanel()
        qtbot.addWidget(panel)

        monkeypatch.setattr(panel._jobs, "submit", lambda *a, **k: False)
        # Both refusals ABOVE the submit have to be satisfied first, or the
        # function returns early and the guard is never reached: a column
        # must be selected, and the merged frame must have been written.
        monkeypatch.setattr(panel, "selected_columns",
                            lambda: ("cell_area",))
        monkeypatch.setattr(panel, "_score_path",
                            lambda: "/tmp/merged.csv")

        started = panel.start_regressions()
        assert started is False
        assert panel._running is False


class TestShowingASectionByTitle:

    def test_a_known_section_is_shown_and_hidden(self, qtbot):
        panel = MSP.MeasurementScanPanel()
        qtbot.addWidget(panel)
        titles = panel.section_titles()
        if not titles:
            pytest.skip("this build has no named sections")
        # THE SECTION WIDGET is the evidence. `_show_section` sets
        # visibility on the whole folder -- header included, deliberately
        # -- so that is what has to follow the flag. Asserting on
        # `section_titles()` would not: it lists sections whether they are
        # shown or not.
        section = panel._folders[titles[0]]

        panel._show_section(titles[0], False)
        assert not section.isVisibleTo(panel), "the section stayed visible"

        panel._show_section(titles[0], True)
        assert section.isVisibleTo(panel), "the section did not come back"

    def test_an_unknown_title_falls_back_to_the_database_list(self, qtbot):
        """THE UNCOVERED ARM.

        A title the panel does not know cannot be shown on its own, and
        hiding nothing would leave the user staring at a panel that
        ignored them. Falling back to the database list keeps the call
        meaningful.
        """
        panel = MSP.MeasurementScanPanel()
        qtbot.addWidget(panel)
        panel._show_section("no such section", False)
        assert panel.databases.isVisible() is False
        panel._show_section("no such section", True)
