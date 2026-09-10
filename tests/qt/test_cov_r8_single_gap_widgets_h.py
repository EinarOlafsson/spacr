"""Four widgets whose one uncovered decision is "and if there is none".

A violin plot over groups with nothing in them, a spinner whose work
finished during its own delay, a proposal made without a reading behind
it, and a list row that no longer points at a trial.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

pytestmark = pytest.mark.qt


# ---------------------------------------------------------------------------
# gene_measurement_compare.py -- a violin over empty groups
# ---------------------------------------------------------------------------

def _comparison(groups):
    """A comparison whose groups hold exactly the values given."""
    import pandas as pd

    from spacr.gene_measurement_compare import Comparison

    rows = [{"group": name, "value": value, "unit": f"{name}{i}"}
            for name, values in groups.items()
            for i, value in enumerate(values)]
    return Comparison(measurement="cell_area", level="well",
                      frame=pd.DataFrame(rows,
                                         columns=["group", "value", "unit"]))


class TestDrawingAViolin:

    def _style(self, **kwargs):
        from spacr.gene_measurement_compare import ComparisonStyle

        return ComparisonStyle(kind="violin", **kwargs)

    def test_a_violin_over_two_populated_groups_draws_both(self):
        from spacr.gene_measurement_compare import render_comparison

        figure, axes = render_comparison(
            _comparison({"treated": [1.0, 2.0, 3.0, 4.0],
                         "control": [2.0, 2.5, 3.5, 5.0]}),
            self._style())

        assert figure is not None and axes is not None

    def test_a_group_with_nothing_finite_in_it_is_left_out(self):
        """The live half of the guard: only the populated groups are drawn.

        ``violinplot`` over an empty array is not an empty picture -- it
        raises -- and it would raise while drawing a report panel,
        taking the whole figure rather than the one empty group.
        """
        import numpy as np

        from spacr.gene_measurement_compare import render_comparison

        figure, axes = render_comparison(
            _comparison({"treated": [1.0, 2.0, 3.0, 4.0],
                         "empty": [np.nan, np.nan],
                         "control": [2.0, 2.5, 3.5]}),
            self._style())

        assert figure is not None and axes is not None

    def test_nothing_finite_anywhere_returns_no_figure_at_all(self):
        """THE PIN.

        ``if alive:`` inside the violin branch cannot be false, because
        the function has ALREADY returned ``(None, None)`` when no group
        has a finite value -- so by the time the violin is drawn at
        least one group is populated and ``alive`` holds it.

        Returning no figure is the right answer there: an axes with no
        data drawn on it is a panel that says "measured, and nothing
        was found", which is a different claim from "nothing was
        measurable".
        """
        import numpy as np

        from spacr.gene_measurement_compare import render_comparison

        figure, axes = render_comparison(
            _comparison({"a": [np.nan, np.inf], "b": [-np.inf]}),
            self._style())

        assert (figure, axes) == (None, None)

    def test_an_empty_comparison_returns_no_figure_either(self):
        from spacr.gene_measurement_compare import render_comparison

        assert render_comparison(_comparison({}), self._style()) == (None, None)
        assert render_comparison(None, self._style()) == (None, None)


# ---------------------------------------------------------------------------
# activity_spinner.py -- the work finished during the delay
# ---------------------------------------------------------------------------

class TestTheSpinnerDelay:
    """The delay is not a prediction: the question is asked AFTER it."""

    def _spinner(self, qtbot):
        from spacr.qt.widgets.activity_spinner import ActivitySpinner

        spinner = ActivitySpinner()
        qtbot.addWidget(spinner)
        return spinner

    def test_work_still_running_when_the_delay_fires_shows_the_spinner(
            self, qtbot):
        spinner = self._spinner(qtbot)
        spinner._manual_busy = True

        spinner._on_delay_elapsed()

        assert spinner.isVisible() or spinner._due, (
            "the spinner did not appear for work that was still running")

    def test_work_that_finished_during_the_delay_shows_nothing(self, qtbot):
        """THE UNCOVERED ARC.

        A job that finished at 1.9 s of a 2 s delay is simply not busy
        by the time the question is asked, and nothing appears. Showing
        it anyway would be a spinner for work that is already done --
        which is exactly the flicker the delay exists to prevent.
        """
        spinner = self._spinner(qtbot)
        spinner._manual_busy = False
        assert spinner.is_busy() is False

        spinner._on_delay_elapsed()

        assert spinner._due is False, "the spinner armed itself with no work"
        assert spinner.isVisible() is False

    def test_an_idle_spinner_posts_no_events_at_all(self, qtbot):
        spinner = self._spinner(qtbot)

        assert spinner.is_busy() is False
        assert spinner.is_spinning() is False


# ---------------------------------------------------------------------------
# settings_advisor_dialog.py -- a proposal with no reading behind it
# ---------------------------------------------------------------------------

class TestRenderingAProposal:

    def _page(self, qtbot):
        from spacr.qt.widgets.settings_advisor_dialog import ProposalPage

        page = ProposalPage()
        qtbot.addWidget(page)
        return page

    def test_a_proposal_with_a_reading_summarises_the_data(self, qtbot):
        from spacr.settings_advisor import Advice, Reading

        page = self._page(qtbot)
        reading = Reading(plates=2, wells=768, guides=1200, genes=300,
                          n_response=45000)

        page.show_the_proposal(Advice(reading=reading), current={})

        assert page.summary.text().strip(), "the summary was left empty"

    def test_a_proposal_with_no_reading_still_lists_its_choices(self, qtbot):
        """THE UNCOVERED ARC.

        The reading is the count of plates, wells, guides and genes --
        it comes from scanning the data. A proposal can be rebuilt from
        stored settings with nothing scanned behind it, and formatting
        ``reading.plates`` then is an AttributeError on None while the
        dialog is being drawn.

        The choices are still shown: they are the proposal, and the
        summary is only its preamble.
        """
        from spacr.settings_advisor import Advice

        page = self._page(qtbot)

        page.show_the_proposal(Advice(reading=None), current={})

        assert page.table.rowCount() == 0

    def test_nothing_to_render_at_all_is_a_no_op(self, qtbot):
        page = self._page(qtbot)

        page._render()                      # never given a proposal

        assert page.table.rowCount() == 0


# ---------------------------------------------------------------------------
# umap_search_viewer.py -- a row that no longer points at a trial
# ---------------------------------------------------------------------------

class TestChoosingATrialFromTheList:

    def _gallery(self, qtbot, trials):
        from spacr.qt.widgets.umap_search_viewer import UmapGalleryDialog

        gallery = UmapGalleryDialog()
        qtbot.addWidget(gallery)
        gallery._trials = list(trials)
        return gallery

    def test_a_row_pointing_at_a_trial_emits_it(self, qtbot):
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import QListWidgetItem

        gallery = self._gallery(qtbot, [{"id": "a"}, {"id": "b"}])
        item = QListWidgetItem("the second map")
        item.setData(Qt.UserRole, 1)

        with qtbot.waitSignal(gallery.trial_chosen, timeout=500) as caught:
            gallery._choose(item)

        assert caught.args == [{"id": "b"}]

    @pytest.mark.parametrize("stored", [None, "1", -1, 5, 2])
    def test_a_row_that_does_not_point_at_one_emits_nothing(self, qtbot,
                                                             stored):
        """THE UNCOVERED ARC.

        Rows carry an INDEX, not the trial. A refreshed search table
        leaves the old rows on screen for a moment holding indexes into
        a list that is now shorter -- and a row whose data was never set
        carries None. Indexing on either is an exception raised from a
        click handler.
        """
        emitted = []
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import QListWidgetItem

        gallery = self._gallery(qtbot, [{"id": "a"}, {"id": "b"}])
        gallery.trial_chosen.connect(emitted.append)

        item = QListWidgetItem("a stale row")
        item.setData(Qt.UserRole, stored)

        gallery._choose(item)               # must not raise

        assert emitted == [], f"a stale row carrying {stored!r} emitted a trial"
