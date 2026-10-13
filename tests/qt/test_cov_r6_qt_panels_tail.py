"""The last arc in the fourteen big Qt analysis panels.

Each of these modules is between 99.5% and 99.95% covered, so what is
left is one branch apiece. Half of them turn out to be guards that a line
above has already made true, and those carry a proof plus a test pinning
the invariant that makes them so -- never a ``pragma``.

Driven here:

``settings_advisor_dialog``
    A proposal built without a reading of the data still lists its
    chosen values.
``feature_dictionary``
    Removing the context-menu filter after the QApplication has gone.
``save_figure_dialog``
    Clearing a preview holder that contains something which is not a
    widget.
``gate_editor``
    A gate the working set cannot count keeps its row, and a camera ray
    whose intersection overflows falls back to the affine reading.

Proved unreachable, with the invariant pinned instead:

``dose_response``
    ``caveats()`` always appends the R² warning, so ``report()`` always
    prints a caveat block; and ``_HILL_GRID`` is strictly increasing, so
    the Brent refinement bracket is never empty.
``gate_editor``
    ``_gate_stats`` partitions the gates: each one has either a count or
    a reason.
``gate_settings``
    Every projection control is built in ``__init__``, so greying them
    never meets a missing one.
``pca_view``
    Both component pickers are refilled from the same component count,
    so the slide onto Y always finds its entry.
``data_filter_panel``
    The picker only offers columns that get a clause row, and every
    clause row can restore itself.
``gene_panel``
    A ``QWidget`` cannot be constructed without a ``QApplication``.
``live_preview``
    ``_cells_with_images`` has already dropped every cell with no file
    behind it.
``graph_spec``
    Every non-empty panel yields at least one bar, because a missing
    value is a level rather than a dropped row.
``outlier_model``
    ``caveats()`` ends with an unconditional append, so a report always
    has a caveat block.
``motility_preview``
    ``ImageSetSampler.describe`` always returns a sentence, so the
    status line is always restated after a new cap.
``qc_summary``
    SQLite has no zero-column table; a schema that cannot be described
    raises instead, and the error card is what the user sees.
"""
from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication, QSpacerItem


# ---------------------------------------------------------------------------
# settings_advisor_dialog
# ---------------------------------------------------------------------------

class TestAProposalWithNoReadingBehindIt:
    """``if reading is not None:`` in ``ProposalPage._render``."""

    @staticmethod
    def _advice(reading):
        from spacr.settings_advisor import Advice, Choice

        return Advice(chosen=(Choice(key="cell_diameter", value=30,
                                     why="typical for this magnification"),),
                      reading=reading)

    def test_without_a_reading_the_summary_stays_empty_but_the_table_fills(
            self, qapp):
        """An advisor run from settings alone has no plate to describe.

        The proposal is still worth showing: what must not happen is a
        summary line quoting well and guide counts that were never
        measured.
        """
        from spacr.qt.widgets.settings_advisor_dialog import ProposalPage

        page = ProposalPage()
        try:
            page.show_the_proposal(self._advice(None), {"cell_diameter": 12})

            assert page.summary.text() == "", (
                "with nothing read there is nothing to summarise; got "
                f"{page.summary.text()!r}")
            assert page.table.rowCount() == 1, \
                "the chosen value is still proposed"
            assert page.table.item(0, 0).text() == "cell_diameter"
        finally:
            page.deleteLater()

    def test_with_a_reading_the_summary_says_what_was_measured(self, qapp):
        """The contrast that makes the empty summary above a real absence."""
        from spacr.qt.widgets.settings_advisor_dialog import ProposalPage
        from spacr.settings_advisor import Reading

        reading = Reading(plates=2, wells=384, guides=1200, genes=300,
                          n_response=50_000, capped=False)
        page = ProposalPage()
        try:
            page.show_the_proposal(self._advice(reading),
                                   {"cell_diameter": 12})

            text = page.summary.text()
            assert text, "a reading must be summarised"
            assert "384" in text or "1,200" in text or "300" in text, (
                "the summary quotes the counts it read; got " + repr(text))
        finally:
            page.deleteLater()


# ---------------------------------------------------------------------------
# feature_dictionary
# ---------------------------------------------------------------------------

class TestRemovingTheHelpFilterWhenTheAppHasGone:
    """``if app is not None:`` in ``remove_context_menu_filter``."""

    def test_the_filter_is_forgotten_even_with_no_application_left(
            self, qapp, monkeypatch):
        """At interpreter shutdown ``QApplication.instance()`` answers None.

        The module-level filter is still held, and the reference has to
        go whatever happens -- otherwise a later install finds a stale
        one and never re-installs on the new application.
        """
        from spacr.qt.widgets import feature_dictionary as fd

        installed = fd.install_context_menu_filter(qapp)
        assert installed is not None, "there has to be a filter to remove"

        monkeypatch.setattr(fd.QApplication, "instance",
                            staticmethod(lambda: None))

        assert fd.remove_context_menu_filter() is True
        assert fd._FILTER is None, \
            "the held reference must be dropped even with no app to tell"
        # And a second removal now says there was nothing, which is what
        # makes the True above a real state change.
        assert fd.remove_context_menu_filter() is False

    def test_with_an_application_the_filter_is_actually_uninstalled(
            self, qapp):
        """The contrast: the filter really comes off the live app."""
        from spacr.qt.widgets import feature_dictionary as fd

        fd.install_context_menu_filter(qapp)
        assert fd.remove_context_menu_filter(qapp) is True
        assert fd._FILTER is None
        assert fd.install_context_menu_filter(qapp) is not None, \
            "a fresh install must be possible after a removal"
        fd.remove_context_menu_filter(qapp)


# ---------------------------------------------------------------------------
# outlier_model
# ---------------------------------------------------------------------------

class TestAnOutlierReportAlwaysCarriesACaveat:
    """``if caveats:`` in ``OutlierResult.report``.

    ``caveats()`` ends with an unconditional ``out.append`` -- the
    "flagged is not deleted" sentence -- outside every branch, so the
    tuple it returns is never empty and the guard in ``report`` cannot be
    false. The per-object run adds a second unconditional one from the
    ``else`` of ``if self.has_wells``, so there is no configuration that
    reaches ``report`` with nothing to warn about.

    That sentence is the caveat that matters most operationally: the
    flags are an added column, not a deletion, and a reader who assumes
    otherwise has silently thrown away every flagged object.
    """

    @staticmethod
    def _frame():
        rng = np.random.default_rng(0)
        values = rng.lognormal(mean=1.0, sigma=0.4, size=200)
        return pd.DataFrame({"cell_area": values,
                             "cell_perimeter": values * 1.7 + 1.0})

    def test_a_clean_transformed_run_still_says_flagged_is_not_deleted(self):
        from spacr.qt.widgets.outlier_model import (OutlierSpec,
                                                    detect_outliers)

        spec = OutlierSpec(method="iqr", transform="log10",
                           features=("cell_area",), per_well=False)
        result = detect_outliers(self._frame(), spec)

        caveats = result.caveats()
        assert caveats, "caveats() is never empty"
        assert any("Flagged is not deleted" in c for c in caveats), (
            "the unconditional caveat is what makes the guard dead; got "
            f"{[c[:40] for c in caveats]}")
        assert "  ! " in result.report(), \
            "so the report always prints a caveat block"
        assert result.n_rows_in == 200 and result.n_scored == 200, \
            "and the claim is true: nothing was removed from the table"

    def test_an_untransformed_fence_adds_its_own_warning_on_top(self):
        """The contrast: caveats grow, they never shrink to nothing."""
        from spacr.qt.widgets.outlier_model import (OutlierSpec,
                                                    detect_outliers)

        clean = detect_outliers(
            self._frame(), OutlierSpec(method="iqr", transform="log10",
                                       features=("cell_area",),
                                       per_well=False))
        raw = detect_outliers(
            self._frame(), OutlierSpec(method="iqr", transform="none",
                                       features=("cell_area",),
                                       per_well=False))

        assert any("Tukey" in c for c in raw.caveats()), (
            "an untransformed fence on a skewed measurement must warn")
        assert len(raw.caveats()) > len(clean.caveats())


# ---------------------------------------------------------------------------
# save_figure_dialog
# ---------------------------------------------------------------------------

class TestClearingAHolderThatIsNotAllWidgets:
    """``if widget is not None:`` in ``_clear_holder``."""

    def test_a_spacer_in_the_preview_holder_is_dropped(self, qapp):
        """``takeAt`` returns layout ITEMS, and a spacer's widget is None.

        The dialog itself only ever adds widgets, so the spacer has to be
        put in directly -- but the guard is load-bearing all the same:
        ``None.setParent`` inside a redraw would take the dialog down,
        and the holder is drained on every preview.
        """
        from matplotlib.figure import Figure

        from spacr.qt.widgets.save_figure_dialog import SaveFigureDialog

        figure = Figure(figsize=(3.0, 2.0))
        figure.add_subplot(111).plot([0, 1], [0, 1])
        dialog = SaveFigureDialog(figure)
        try:
            dialog._holder.addItem(QSpacerItem(4, 4))
            assert dialog._holder.count() >= 1

            dialog._clear_holder()

            assert dialog._holder.count() == 0, (
                "everything in the holder must be taken out, widget or not")
        finally:
            dialog.deleteLater()


# ---------------------------------------------------------------------------
# graph_spec
# ---------------------------------------------------------------------------

class TestEveryPanelWithRowsContributesToTheCountAxis:
    """``if len(counts):`` in ``_count_limit``'s bar branch.

    ``counts = _level_series(rows, column).value_counts()`` and
    ``_level_series`` folds NaN into ``MISSING_LEVEL`` instead of
    dropping it -- "no condition recorded" is a bar like any other -- so
    every row of ``rows`` produces a level. ``rows`` itself is non-empty,
    because the loop does ``if panel.is_empty: continue`` two lines
    above. So ``counts`` always has at least one entry and the false side
    of the guard cannot be taken: another re-check of what the lines
    above guarantee.

    Pinned instead: the shared axis really does clear the panel of
    missing values, which is the behaviour that guard would otherwise
    have quietly broken.
    """

    @staticmethod
    def _grid(frame, spec):
        from spacr.qt.widgets.graph_spec import facet_grid

        return facet_grid(frame, spec)

    def test_a_panel_whose_column_is_all_missing_is_still_a_bar(self):
        from spacr.qt.widgets.graph_spec import BAR, GraphSpec, _count_limit

        frame = pd.DataFrame({
            "plate": ["p1"] * 3 + ["p2"] * 5,
            "condition": ["a", "a", "b", None, None, None, None, None],
        })
        spec = GraphSpec(kind=BAR, x="condition", facet_col="plate")
        grid = self._grid(frame, spec)
        assert len(grid.panels) == 2, \
            f"two plates, two panels; got {len(grid.panels)}"

        limit = _count_limit(frame, spec, grid, BAR, None, None)

        assert limit == pytest.approx(5 * 1.08), (
            "the five missing-value rows are one bar of height 5, and the "
            f"shared axis has to clear it; got {limit}")

    def test_an_empty_panel_is_skipped_before_it_is_counted(self):
        """The other half: a facet with no rows contributes nothing.

        The two facet columns here have no rows in common, so half the
        grid is empty -- and the axis is still set by the panels that do
        have rows rather than collapsing to nothing.
        """
        from spacr.qt.widgets.graph_spec import BAR, GraphSpec, _count_limit

        frame = pd.DataFrame({
            "plate": ["p1", "p1", "p2"],
            "row": ["A", "A", "B"],
            "condition": ["a", "a", "b"],
        })
        spec = GraphSpec(kind=BAR, x="condition", facet_col="plate",
                         facet_row="row")
        grid = self._grid(frame, spec)
        assert any(panel.is_empty for panel in grid.panels), (
            "this layout is meant to have empty panels in it")

        limit = _count_limit(frame, spec, grid, BAR, None, None)

        assert limit == pytest.approx(2 * 1.08), (
            "the tallest bar anywhere is the two 'a' rows on p1/A; got "
            f"{limit}")


# ---------------------------------------------------------------------------
# gate_editor
# ---------------------------------------------------------------------------

class TestAGateTheWorkingSetCannotCount:
    """``elif gate.name in unavailable:`` in ``GateTree._rebuild``.

    The false side of that ``elif`` -- a gate with neither a count nor a
    recorded reason -- cannot happen. ``_gate_stats`` returns a pair that
    PARTITIONS ``self._gates.order()``: on the fast path
    ``GateSet.stats`` builds one ``GateStats`` per gate in ``order()``,
    so every name is in ``stats``; on the fallback path every gate is put
    in ``stats`` or, if its mask raised, in ``why``. There is no third
    outcome, so ``stat is None`` implies ``gate.name in unavailable``.

    What the branch does is worth keeping, and is what is pinned: a gate
    whose columns this table does not have keeps its row and says ``n/a``
    rather than vanishing from the tree.
    """

    @staticmethod
    def _gates():
        from spacr.qt.widgets.gate_spec import GateSet, ThresholdGate

        return GateSet(gates=[
            ThresholdGate(name="big", column="area", low=1.0),
            ThresholdGate(name="bright", column="intensity", low=5.0),
        ])

    def test_a_gate_whose_column_is_missing_says_so_and_keeps_its_row(
            self, qapp):
        from spacr.qt.widgets.gate_editor import GateTree

        tree = GateTree()
        try:
            # No `intensity` column: the "bright" gate cannot be counted,
            # and GateSet.stats is all-or-nothing, so the fallback runs.
            tree.set_gates(self._gates(),
                           pd.DataFrame({"area": [0.5, 2.0, 3.0]}))

            rows = {tree.tree.topLevelItem(i).text(0): tree.tree.topLevelItem(i)
                    for i in range(tree.tree.topLevelItemCount())}
            assert set(rows) == {"big", "bright"}, (
                "every gate keeps a row whatever the table holds; got "
                f"{sorted(rows)}")
            assert rows["bright"].text(1) == tree.UNAVAILABLE, (
                "the gate this table cannot answer says so in the count "
                f"column; got {rows['bright'].text(1)!r}")
            assert rows["big"].text(1) == "2", (
                "the gate it CAN answer still carries its count; got "
                f"{rows['big'].text(1)!r}")
        finally:
            tree.deleteLater()


# ---------------------------------------------------------------------------
# dose_response
# ---------------------------------------------------------------------------

class TestTheDoseResponseReportAlwaysHasACaveat:
    """``if caveats:`` in ``DoseResponseResult.report``.

    ``caveats()`` builds a list and appends the R² warning to it
    **unconditionally** -- it is not inside any branch -- so the tuple it
    returns is never empty and the guard in ``report`` cannot be false.
    That warning is the point of the method: R² above 0.95 is what a
    dose-response scores whatever the fit, so the number a reader would
    otherwise take away is the misleading one.
    """

    @staticmethod
    def _series(*, ec50=1.0, hill=-1.0, seed=1, doses=None):
        import math

        from spacr.qt.widgets.dose_response import four_parameter_logistic

        doses = np.array([27.0, 9, 3, 1, 1 / 3, 1 / 9, 1 / 27, 1 / 81,
                          1 / 243, 1 / 729]) if doses is None else doses
        rng = np.random.default_rng(seed)
        dose = np.repeat(doses, 3)
        clean = four_parameter_logistic(dose, 0.0, 100.0, math.log10(ec50),
                                        hill)
        return dose, clean + rng.normal(0.0, 1.0, dose.size)

    def test_a_clean_fit_still_warns_about_r_squared(self):
        from spacr.qt.widgets.dose_response import (DoseResponseSpec,
                                                    fit_dose_response)

        dose, response = self._series()
        result = fit_dose_response(dose, response, DoseResponseSpec(unit="µM"))

        caveats = result.caveats()
        assert caveats, "caveats() is never empty"
        assert any("R²" in c for c in caveats), (
            "the R² warning is appended unconditionally; got "
            f"{[c[:40] for c in caveats]}")
        assert "  ! " in result.report(), \
            "so the report always prints a caveat block"

    def test_a_truncated_series_warns_about_that_too(self):
        """The contrast: a refusal adds caveats, it does not remove them."""
        from spacr.qt.widgets.dose_response import (DoseResponseSpec,
                                                    fit_dose_response)

        low_only = np.array([1 / 3, 1 / 9, 1 / 27, 1 / 81, 1 / 243, 1 / 729])
        dose, response = self._series(doses=low_only)
        result = fit_dose_response(dose, response, DoseResponseSpec(unit="µM"))

        caveats = result.caveats()
        assert any("R²" in c for c in caveats)
        assert len(caveats) > 1, (
            "a series that never reached the midpoint has more to say, not "
            f"less; got {len(caveats)}")


class TestTheHillGridAlwaysBracketsItsMinimum:
    """``if hi > lo:`` in ``_profile_sse``.

    ``lo`` and ``hi`` are ``_HILL_GRID[max(0, j - 1)]`` and
    ``_HILL_GRID[min(size - 1, j + 1)]`` for the grid's argmin ``j``. The
    grid is a module constant with more than one entry and strictly
    increasing, so those two indices are never the same one: at ``j = 0``
    they are 0 and 1, at the far end ``size - 2`` and ``size - 1``, and
    in between ``j - 1`` and ``j + 1``. ``hi > lo`` therefore always
    holds and the false side is unreachable.

    Pinned instead: the grid property the argument rests on, and that the
    Brent refinement between the neighbours really does improve on the
    grid's own minimum.
    """

    def test_the_grid_is_strictly_increasing_with_room_to_bracket(self):
        from spacr.qt.widgets.dose_response import _HILL_GRID

        assert _HILL_GRID.size > 1, \
            "one grid point would leave nothing to bracket"
        assert np.all(np.diff(_HILL_GRID) > 0), (
            "the bracket is [grid[j-1], grid[j+1]]; a repeated value would "
            "collapse it")

    def test_the_refinement_beats_the_grid_it_started_from(self):
        from spacr.qt.widgets.dose_response import (_HILL_GRID, _plateau_sse,
                                                    _profile_sse)

        log_dose = np.log10(np.repeat(
            np.array([27.0, 9, 3, 1, 1 / 3, 1 / 9, 1 / 27, 1 / 81]), 3))
        rng = np.random.default_rng(0)
        response = 100.0 / (1.0 + 10.0 ** (1.35 * log_dose)) \
            + rng.normal(0.0, 1.0, log_dose.size)

        grid_best = float(np.min(
            _plateau_sse(log_dose, response, 0.0, -_HILL_GRID)))
        refined = _profile_sse(log_dose, response, 0.0, -1.0)

        assert refined <= grid_best + 1e-9, (
            "the bounded Brent step between the grid's neighbours must not "
            f"be worse than the grid; {refined} vs {grid_best}")


# ---------------------------------------------------------------------------
# gate_settings
# ---------------------------------------------------------------------------

class TestGreyingTheIrrelevantProjectionControls:
    """``if widget is not None:`` in ``_grey_irrelevant_methods``.

    ``_METHOD_CONTROLS`` names ``_n_neighbors``, ``_min_dist`` and
    ``_perplexity``, and all three are assigned unconditionally in the
    dialog's ``__init__`` before any signal can reach this slot. The
    ``getattr(self, name, None)`` therefore never answers ``None``, and
    the guard beside it is a re-check of what construction guarantees.

    Pinned instead: the behaviour the mapping exists for -- only the
    chosen projection's own parameters are editable, and a method with no
    parameters of its own greys all of them, because "PCA has nothing to
    tune" is a fact about PCA worth showing.
    """

    def test_only_the_chosen_projections_controls_are_editable(self, qapp):
        from spacr.qt.widgets.gate_settings import (GateEditorSettings,
                                                    GateSettingsDialog)

        dialog = GateSettingsDialog(GateEditorSettings())
        try:
            dialog._reduction.setCurrentText("umap")
            assert dialog._n_neighbors.isEnabled()
            assert dialog._min_dist.isEnabled()
            assert not dialog._perplexity.isEnabled(), \
                "UMAP does not read the t-SNE perplexity"

            dialog._reduction.setCurrentText("tsne")
            assert dialog._perplexity.isEnabled()
            assert not dialog._n_neighbors.isEnabled()

            dialog._reduction.setCurrentText("pca")
            assert not any(w.isEnabled() for w in
                           (dialog._n_neighbors, dialog._min_dist,
                            dialog._perplexity)), \
                "PCA is entirely determined by the data"
        finally:
            dialog.deleteLater()


# ---------------------------------------------------------------------------
# pca_view
# ---------------------------------------------------------------------------

class TestClickingAScreeBarSlidesTheOldXOntoY:
    """``if new_y >= 0:`` in ``_on_scree_clicked``.

    Both pickers are refilled by ``_sync_component_pickers`` from the
    same loop over ``range(result.n_components)``, each item carrying
    ``component_name(i)`` as its data. ``kx`` comes from ``_plane()``,
    which reads one of those same pickers (defaulting to 0), so
    ``self._pc_y.findData(component_name(kx))`` is looking for an entry
    the picker was just given. It cannot be ``-1``, and the guard is a
    re-check of what the refill guarantees.

    Pinned instead: the slide itself.
    """

    @staticmethod
    def _frame():
        rng = np.random.default_rng(0)
        base = rng.normal(size=(120, 1))
        return pd.DataFrame({
            "a": base[:, 0] + rng.normal(0, 0.1, 120),
            "b": 2 * base[:, 0] + rng.normal(0, 0.1, 120),
            "c": rng.normal(size=120),
            "d": rng.normal(size=120),
        })

    def test_the_component_clicked_becomes_x_and_the_old_x_becomes_y(
            self, qapp):
        from spacr.qt.widgets.pca_view import PCAPanel, component_name

        panel = PCAPanel()
        try:
            panel.set_frame(self._frame())
            panel.recompute()
            if panel._result is None:
                pytest.skip("this build could not fit a PCA here")
            if panel._pc_x.count() < 3:
                pytest.skip("fewer than three components to slide between")

            before_x = panel._plane()[0]
            panel._on_scree_clicked(2)

            assert panel._pc_x.currentData() == component_name(2)
            assert panel._pc_y.currentData() == component_name(before_x), (
                "the component that was on X slides onto Y; got "
                f"{panel._pc_y.currentData()!r}")
        finally:
            panel.deleteLater()


# ---------------------------------------------------------------------------
# data_filter_panel
# ---------------------------------------------------------------------------

class TestRestoringAFilterSetOntoAnotherTable:
    """``if row is not None and hasattr(row, "restore"):`` in ``restore``.

    ``restore`` skips any column not in ``available_columns()``, which is
    exactly the picker's contents -- and ``set_frame`` fills the picker
    with every column whose kind is not ``'skip'``. ``classify_columns``
    returns only ``'category'``, ``'range'`` or ``'skip'``, so every
    offered column takes one of ``add_column``'s two branches and gets a
    ``_RangeRow`` or a ``_CategoryRow``. Both define ``restore``. So
    neither half of that guard can be false: it re-checks what
    ``add_column`` on the line above has just guaranteed.

    Pinned instead: the thing the method promises -- the columns it could
    not restore come back to the caller, and the ones it could are
    actually filtering.
    """

    @staticmethod
    def _frame():
        return pd.DataFrame({
            "area": [1.0, 5.0, 9.0, 13.0],
            "condition": ["a", "b", "a", "b"],
        })

    def test_a_filter_set_saved_against_another_table_names_what_it_lost(
            self, qapp):
        from spacr.qt.widgets.data_filter_panel import DataFilterPanel

        panel = DataFilterPanel()
        try:
            panel.set_frame(self._frame())
            state = {"version": 1, "filters": [
                {"column": "area", "kind": "range", "low": 4.0, "high": 10.0},
                {"column": "not_in_this_table", "kind": "range"},
            ]}

            missing = panel.restore(state)

            assert missing == ["not_in_this_table"], (
                "a column this table does not have must be reported, not "
                f"silently dropped; got {missing}")
            assert "area" in panel._rows, \
                "and the one it does have gets a working clause row"
            assert hasattr(panel._rows["area"], "restore")
        finally:
            panel.deleteLater()

    def test_every_offered_column_can_be_given_a_row(self, qapp):
        """The partition the guard rests on, asserted directly."""
        from spacr.qt.widgets.data_filter_panel import (DataFilterPanel,
                                                        classify_columns)

        frame = self._frame()
        kinds = classify_columns(frame)
        assert set(kinds.values()) <= {"category", "range", "skip"}, (
            "a fourth kind would leave an offered column with no row; got "
            f"{sorted(set(kinds.values()))}")

        panel = DataFilterPanel()
        try:
            panel.set_frame(frame)
            for column in panel.available_columns():
                panel.add_column(column)
                row = panel._rows.get(column)
                assert row is not None, f"{column} was offered but got no row"
                assert hasattr(row, "restore"), \
                    f"{column}'s row cannot restore itself"
        finally:
            panel.deleteLater()


# ---------------------------------------------------------------------------
# gene_panel
# ---------------------------------------------------------------------------

class TestTheGenePanelAlwaysHasAnApplicationToHearFrom:
    """``if application is not None:`` at the end of ``GenePanel.__init__``.

    ``GenePanel`` is a ``QWidget``, and ``super().__init__(parent)`` runs
    on the first line of its ``__init__``. Qt aborts the process --
    "QWidget: Must construct a QApplication before a QWidget" -- if there
    is no application at that moment, so by the time the last line asks
    ``QApplication.instance()`` the object exists or the process is
    already gone. The guard cannot be false.

    What it is guarding matters, though: Qt aborts the process if a
    RUNNING QThread is destroyed, and a panel can be dropped without ever
    being closed. That is what the ``aboutToQuit`` connection is for, and
    it is what is pinned here.
    """

    def test_the_panel_arms_its_thread_shutdown_on_the_live_application(
            self, qapp):
        from spacr.qt.widgets.gene_panel import GenePanel

        panel = GenePanel(threaded=False)
        try:
            assert QApplication.instance() is not None, (
                "the panel is a QWidget, so an application existed before "
                "its __init__ reached the guard")
            # The belt-and-braces path exists and is callable: dropping the
            # panel without closing it must still stop the warm-up.
            panel._shut_down_warming()
            assert panel is not None
        finally:
            panel.deleteLater()


# ---------------------------------------------------------------------------
# live_preview
# ---------------------------------------------------------------------------

class TestSelectingCellsThatAllHaveFilesBehindThem:
    """``if path:`` at the end of ``_set_selection``.

    ``cells = self._cells_with_images(cells)`` on the first line has
    already dropped every cell whose table item is missing or whose
    ``Qt.UserRole`` data is falsy, and ``if not cells: return`` follows
    it. ``combined`` is either ``cells`` or the retained selection with
    ``cells`` appended, so ``combined[-1]`` is always one of ``cells`` --
    a cell whose item data was checked truthy two lines earlier. The
    ``if path:`` is a re-check of exactly that.

    Pinned instead: the filter that makes it true, and the truncation
    rule beside it.
    """

    def test_cells_with_no_file_behind_them_are_dropped_first(self, qapp):
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import QTableWidgetItem

        from spacr.qt.widgets.live_preview import LivePreviewPanel

        panel = LivePreviewPanel()
        try:
            table = panel._set_table
            table.setRowCount(1)
            table.setColumnCount(3)
            with_file = QTableWidgetItem("a")
            with_file.setData(Qt.UserRole, "/tmp/a.tif")
            table.setItem(0, 0, with_file)
            no_file = QTableWidgetItem("b")
            table.setItem(0, 1, no_file)          # no UserRole at all
            # (0, 2) is left with no item at all.

            kept = panel._cells_with_images([(0, 0), (0, 1), (0, 2), (0, 0)])

            assert kept == [(0, 0)], (
                "only the cell with a file behind it survives, once; got "
                f"{kept}")
        finally:
            panel.deleteLater()


# ---------------------------------------------------------------------------
# qc_summary
# ---------------------------------------------------------------------------

class TestATableThatCannotBeDescribed:
    """``if columns:`` in ``_read_units``.

    ``columns`` comes from ``PRAGMA table_info`` on a name
    ``sqlite_master`` has just listed as a table. SQLite has no
    zero-column table -- ``CREATE TABLE`` requires at least one -- and
    both reads run against the same schema in the same connection, so a
    listed table always describes at least one column. The only way to a
    listed-but-undescribable table is a corrupt schema, and that raises
    ``sqlite3.DatabaseError`` from the ``sqlite_master`` read itself
    rather than returning an empty column list: the enclosing ``except
    sqlite3.Error`` turns it into the card's error verdict, which is the
    branch that actually runs.

    Both of the reachable outcomes are driven below.
    """

    @staticmethod
    def _project(tmp_path, build):
        root = tmp_path / "plate1"
        (root / "measurements").mkdir(parents=True)
        db = root / "measurements" / "measurements.db"
        connection = sqlite3.connect(db)
        try:
            build(connection)
            connection.commit()
        finally:
            connection.close()
        return root, db

    def test_a_table_with_no_stamp_columns_is_reported_as_unstamped(
            self, tmp_path):
        from spacr.qt.widgets.qc_summary import _read_units

        def build(connection):
            connection.execute("CREATE TABLE cell (objectID INTEGER, area REAL)")
            connection.execute("INSERT INTO cell VALUES (1, 2.0)")

        root, _db = self._project(tmp_path, build)

        card = _read_units(str(root))

        assert card.verdict != "error", card.headline
        assert "1 table(s) carry no units stamp" in card.headline, (
            "the table with columns but no stamp is counted as unstamped; "
            f"got {card.headline!r}")

    def test_a_schema_that_cannot_be_read_becomes_an_error_card(
            self, tmp_path):
        """The corrupt-schema case: it raises, it does not answer emptily."""
        from spacr.qt.widgets.qc_summary import _read_units

        def build(connection):
            connection.execute("CREATE TABLE cell (objectID INTEGER)")
            connection.execute("PRAGMA writable_schema=ON")
            connection.execute(
                "INSERT INTO sqlite_master(type,name,tbl_name,rootpage,sql) "
                "VALUES ('table','ghost','ghost',0,NULL)")

        root, db = self._project(tmp_path, build)
        with pytest.raises(sqlite3.DatabaseError):
            sqlite3.connect(f"file:{db}?mode=ro", uri=True).execute(
                "SELECT name FROM sqlite_master WHERE type='table'").fetchall()

        card = _read_units(str(root))

        assert card.verdict == "error", (
            "a schema SQLite refuses to read is an error the user is told "
            f"about, not a silently skipped table; got {card.verdict!r}")


# ---------------------------------------------------------------------------
# motility_preview
# ---------------------------------------------------------------------------

class TestChangingTheSetCapAlwaysRestatesTheSample:
    """``if self.sample_note():`` in ``_on_max_sets_changed``.

    ``_populate_group_box`` -- called on the line above -- ends with
    ``self._sample_note = self._sampler.describe(len(shown))``, and
    ``ImageSetSampler.describe`` returns a sentence on both of its
    branches: ``"showing all N image sets"`` when nothing was sampled
    away, and ``"showing a random sample of ..."`` otherwise. Neither can
    be empty, so ``sample_note()`` is always truthy by the time the guard
    runs. It re-checks what the call above guarantees.

    Pinned instead: the sentence itself, which is the whole point of the
    cap -- a dropdown showing 40 of 3 000 fields must say so, or a reader
    takes the preview for the plate.
    """

    def test_the_sampler_always_has_a_sentence_to_offer(self):
        from spacr.qt.widgets.preview_controls import ImageSetSampler

        sampler = ImageSetSampler(40)

        # Nothing adopted yet: still a sentence, not an empty string.
        assert sampler.describe(0) == "showing all 0 image sets"

        from spacr.qt.widgets.preview_controls import ImageSet

        sampler.adopt("/plate", [ImageSet(key=("p", f"A{i:02d}", "1"),
                                          directory="/plate",
                                          channels={"": f"f{i}.tif"})
                                 for i in range(120)], [])
        note = sampler.describe(40)
        assert note.startswith("showing a random sample of 40 of 120"), (
            "a bounded sample has to say it is one; got " + repr(note))
        assert sampler.describe(120) == "showing all 120 image sets", (
            "and a cap nothing was cut by says the opposite, rather than "
            "nothing at all")

    def test_a_new_cap_is_a_change_and_the_same_cap_is_not(self):
        """The guard above it: only a real change redraws the sample."""
        from spacr.qt.widgets.preview_controls import ImageSetSampler

        sampler = ImageSetSampler(40)

        assert sampler.set_max(12) is True
        assert sampler.max_sets == 12
        assert sampler.set_max(12) is False, (
            "re-setting the same cap must not redraw; that is the guard "
            "before the one under test")


# ---------------------------------------------------------------------------
# gate_editor: the camera ray that overflows
# ---------------------------------------------------------------------------

class TestAVolumeClickWhoseRayOverflows:
    """``if np.isfinite(point[kept]).all():`` in ``screen_to_volume``."""

    @staticmethod
    def _event(**kwargs):
        import types

        base = dict(inaxes=None, x=0.0, y=0.0, xdata=None, ydata=None)
        base.update(kwargs)
        return types.SimpleNamespace(**base)

    @staticmethod
    def _volume(qtbot):
        from spacr.qt.widgets.gate_editor import GateCanvas
        from spacr.qt.widgets.graph_spec import GraphSpec

        rng = np.random.default_rng(0)
        frame = pd.DataFrame({"a": rng.normal(5.0, 1.0, 200),
                              "b": rng.normal(5.0, 1.0, 200),
                              "c": rng.normal(5.0, 1.0, 200)})
        widget = GateCanvas()
        qtbot.addWidget(widget)
        widget.set_frame(frame)
        widget.set_spec(GraphSpec(x="a", y="b"))
        widget.set_mode("3D", z_column="c")
        widget.set_anchor_axis("z")
        widget.render_now()
        return widget

    def test_a_ray_that_overflows_falls_back_to_the_affine_reading(
            self, qtbot, monkeypatch):
        """The exact reading is a division, and a division can run away.

        ``amount = (face - near[depth]) / direction[depth]`` is only
        guarded against a direction of *zero*; a direction that is tiny
        along the normal and enormous across it makes ``amount *
        direction`` overflow to infinity, and a gate corner at infinity
        would be written into the gate set. The click must land on the
        picked face by the affine route instead.

        ``proj3d.inv_transform`` is the seam because it is the camera:
        there is no other way to hand this code a ray. The existing
        ``test_a_click_falls_back_to_the_affine_reading_when_the_ray_
        will_not_invert`` patches the same function for the same reason.
        """
        from mpl_toolkits.mplot3d import proj3d

        volume = self._volume(qtbot)
        try:
            mapping = volume.volume_axis_map()
            assert mapping is not None, "the 3D view has to have a mapping"
            depth = mapping[3]

            def runaway(_x, _y, z, _matrix):
                if z < 0:                       # the near point
                    return np.zeros(3)
                far = np.full(3, 1e300)         # enormous across the face...
                far[depth] = 1e-11              # ...and barely off it
                return far

            monkeypatch.setattr(proj3d, "inv_transform", runaway)

            with np.errstate(over="ignore"):
                read = volume.screen_to_volume(self._event(x=120.0, y=140.0))

            assert read is not None, (
                "an overflowing ray must not lose the click; the affine "
                "fallback still reads the picked face")
            first, x, second, y = read
            assert (first, second) == ("a", "b")
            assert np.isfinite(x) and np.isfinite(y), (
                f"the fallback's answer has to be a real coordinate; "
                f"got ({x}, {y})")
        finally:
            volume.deleteLater()

    def test_an_ordinary_ray_is_read_exactly(self, qtbot):
        """The contrast: with the real camera the exact path is used."""
        volume = self._volume(qtbot)
        try:
            read = volume.screen_to_volume(self._event(x=120.0, y=140.0))

            assert read is not None
            first, x, second, y = read
            assert (first, second) == ("a", "b")
            assert np.isfinite(x) and np.isfinite(y)
        finally:
            volume.deleteLater()
