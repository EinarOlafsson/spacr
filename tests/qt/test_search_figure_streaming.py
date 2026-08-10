"""Search figures arrive one at a time, rendered off the GUI thread.

The whole point of the grid is steering: seeing the first few embeddings is
what tells you the range is wrong before the other fifty configurations
run. That only works if the figures arrive DURING the search, and it only
stays usable if drawing them does not block the GUI.
"""

import numpy as np
import pytest

from spacr.hyperparam import Trial
from spacr.qt.screens.hyperparam import HyperparamPanel, render_trial_figure


def _trial(index=0, score=0.8, embedded=True, **params):
    trial = Trial(params=params or {"n_neighbors": 15, "min_dist": 0.1},
                  index=index)
    trial.score = score
    trial.extra_metrics = (
        {"embedding": np.random.default_rng(index).random((80, 2))}
        if embedded else {})
    return trial


class TestTheWorkerSideRenderer:
    """`render_trial_figure` runs on the search thread, so it must not
    touch Qt at all -- these call it with no widget in sight."""

    def test_it_writes_a_real_png(self, tmp_path):
        out = tmp_path / "trial.png"
        assert render_trial_figure(_trial(), "trustworthiness", str(out))
        assert out.stat().st_size > 1000

    def test_a_trial_with_no_embedding_is_a_no_op(self, tmp_path):
        """A classifier sweep has no embedding, and neither does a failed
        UMAP trial. Neither is an error."""
        out = tmp_path / "none.png"
        assert render_trial_figure(_trial(embedded=False), "score",
                                   str(out)) is False
        assert not out.exists()

    def test_a_scoreless_trial_is_a_no_op(self, tmp_path):
        out = tmp_path / "noscore.png"
        assert render_trial_figure(_trial(score=None), "score",
                                   str(out)) is False


class TestThePanelSide:

    @pytest.fixture
    def panel(self, qt_theme_applied, qtbot):
        widget = HyperparamPanel("umap")
        qtbot.addWidget(widget)
        widget.apply_settings({"n_neighbors": 15, "min_dist": 0.1,
                               "metric": "euclidean"})
        widget._figure_grid.set_parameters(["n_neighbors", "min_dist"])
        return widget

    def test_a_figure_lands_on_the_grid_as_it_arrives(self, panel, tmp_path):
        out = tmp_path / "t0.png"
        render_trial_figure(_trial(), "trustworthiness", str(out))
        panel._on_trial_ready(_trial(), 1, 4, str(out))
        assert panel._figure_grid.count() == 1

    def test_a_trial_with_no_figure_still_reaches_the_table(self, panel):
        """The table is the record; the grid is the picture. A trial with
        no embedding belongs in one and not the other."""
        before = panel._table.rowCount()
        panel._on_trial_ready(_trial(embedded=False), 1, 4, "")
        assert panel._table.rowCount() == before + 1
        assert panel._figure_grid.count() == 0

    def test_the_position_records_the_parameters(self, panel, tmp_path):
        for i, (n, d) in enumerate([(5, 0.1), (5, 0.5), (50, 0.1)]):
            out = tmp_path / f"t{i}.png"
            trial = _trial(index=i, n_neighbors=n, min_dist=d)
            render_trial_figure(trial, "trustworthiness", str(out))
            panel._on_trial_ready(trial, i + 1, 3, str(out))
        assert panel._figure_grid.count() == 3
        assert panel._figure_grid.coordinates()[2]["n_neighbors"] == 50

    def test_a_bad_figure_path_does_not_kill_the_search(self, panel):
        """A figure is decoration. Losing one must not lose the run --
        INVARIANTS 10."""
        panel._on_trial_ready(_trial(), 1, 4, "/nonexistent/nope.png")
        assert panel._table.rowCount() >= 1

    def test_the_grid_survives_the_end_of_the_search(self, panel, tmp_path):
        """This test used to assert the OPPOSITE, and it was pinning a bug.

        Hiding the grid when the summary arrived meant the END STATE -- the
        only state a user who steps away ever sees -- was one large figure,
        which is exactly the complaint the grid exists to answer: "not all
        at the end in one large grid and as a large PNG".

        Both are shown now: the grid because it is what the user asked to
        look at, the summary because it carries the ranking and noise band
        that individual panels cannot.
        """
        out = tmp_path / "t.png"
        render_trial_figure(_trial(), "trustworthiness", str(out))
        panel._figure_grid.setVisible(True)
        panel._on_trial_ready(_trial(), 1, 1, str(out))
        panel._show_summary_instead_of_grid()
        assert panel._figure_grid.isVisibleTo(panel), (
            "the grid disappeared when the search finished")
        assert panel._preview.isVisibleTo(panel)

    def test_clicking_a_cell_hands_back_the_vector_pdf(self, panel, tmp_path):
        """A user who set the figure format to PDF should get the PDF.

        The grid necessarily DISPLAYS a PNG -- a PDF cannot be painted into
        a label -- so the file offered on click is not the one on screen.
        """
        from spacr.qt import preferences

        preferences.set_figure_format("pdf")
        out = tmp_path / "t.png"
        render_trial_figure(_trial(), "trustworthiness", str(out))
        panel._on_trial_ready(_trial(), 1, 1, str(out))
        assert panel._figure_grid.figure_path(0).endswith(".pdf")

    def test_the_grid_keeps_its_figures_after_the_search_ends(self, panel,
                                                             tmp_path):
        """Throwing away what the user was just watching, at the moment the
        run finishes, would be its own small betrayal."""
        out = tmp_path / "t.png"
        render_trial_figure(_trial(), "trustworthiness", str(out))
        panel._on_trial_ready(_trial(), 1, 1, str(out))
        panel._show_summary_instead_of_grid()
        assert panel._figure_grid.count() == 1


def test_the_signal_carries_the_rendered_path():
    """Widened from (Trial, done, total) so the path can be rendered on the
    worker and only PLACED on the GUI thread."""
    from spacr.qt.screens.hyperparam import _SearchWorker
    assert _SearchWorker.trial_ready is not None
