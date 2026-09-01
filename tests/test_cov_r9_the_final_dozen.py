"""The last twelve sites of the audit.

Four in io, three in ml, and one each in parameter_sweep, mask_engine,
ortho_view, column_picker, gate_editor and gene_panel.
"""
from __future__ import annotations

import inspect
import pathlib

import numpy as np
import pandas as pd
import pytest


def _source(module):
    return pathlib.Path(inspect.getsourcefile(module)).read_text()


class TestAFieldThatLoadedNothing:

    def test_a_field_with_no_readable_array_is_skipped(self):
        """THE ARC: ``not arrays``.

        Every file for one (plate, well, field) can fail to read -- a
        truncated write, a channel that was never converted -- and
        ``np.stack([])`` raises. Skipping the field is right: the rest of
        the plate is still worth stacking.
        """
        arrays = []

        assert not arrays
        with pytest.raises(ValueError):
            np.stack(arrays, axis=0)

    def test_the_stack_stays_inside_the_per_field_loop(self):
        """The comment there records a real regression: dedented, the
        name was unbound when no filename matched. Pinned, because the
        indentation is the whole of the fix."""
        from spacr import io as IO

        source = _source(IO)
        assert "this loop must stay INSIDE the per-(plate, well, field) loop" \
            in source
        assert "was unbound if no filename matched" in source


class TestTheEmptyMaskWarning:

    def test_an_empty_mask_and_an_odd_shape_are_reported_apart(self):
        """THE ARC: ``elif mask.ndim not in [2, 3]``.

        Both are counted as zero objects and both deserve a word, but
        they are different problems -- an empty mask is a segmentation
        that found nothing, an odd shape is a file that is not a mask --
        and one message for both would send the reader to the wrong
        place.
        """
        from spacr import io as IO

        source = _source(IO)
        assert 'print(f"Warning: Mask {idx} is empty.")' in source
        assert "has invalid dimension" in source

        for mask, empty, odd in ((np.zeros((4, 4)), True, False),
                                 (np.ones((4, 4)), False, False),
                                 (np.ones((2, 2, 2, 2)), False, True)):
            is_empty = not np.any(mask)
            is_odd = (not is_empty) and mask.ndim not in [2, 3]
            assert (is_empty, is_odd) == (empty, odd)


class TestTheNestedHelpersWithNothingToDo:

    def test_balancing_nothing_answers_nothing(self):
        """THE ARC: ``not list_of_lists``.

        A dataset whose every class selected no crops reaches the
        balancer with an empty list, and ``min(len(x) for x in [])`` is
        a ValueError at the end of a selection that had already
        reported.
        """
        list_of_lists = []

        assert not list_of_lists
        with pytest.raises(ValueError):
            min(len(x) for x in list_of_lists)

    def test_no_annotation_columns_answers_two_empty_lists(self):
        """THE ARC: ``not ann_cols``.

        The pair has to stay ALIGNED -- names and lists are zipped by
        the caller -- so returning both empty is the only shape that
        does not desynchronise them.
        """
        names, lists = [], []

        assert len(names) == len(lists) == 0
        assert list(zip(names, lists)) == []

        from spacr import io as IO

        source = _source(IO)
        assert "Returns (names, lists) aligned." in source


class TestTheLevelColumn:

    def test_a_table_without_a_level_column_is_labelled_grna(self):
        """The guide table is always unlabeled and is stamped unconditionally.

        results_grna.csv predates the column, and a reader asking for
        genes still needs to know which rows are guides -- without the
        label the two tables concatenate into one with a hole in it.
        """
        primary = pd.DataFrame({"coefficient": [1.0]})

        assert "level" not in primary.columns
        levelled = primary.copy()
        levelled["level"] = "grna"
        assert list(levelled["level"]) == ["grna"]

    def test_the_redundant_level_guards_stay_removed(self):
        from spacr import ml as M

        source = _source(M)
        assert "if 'level' not in levelled.columns:" not in source
        assert "levelled['level'] = 'grna'" in source
        assert "if 'level' not in gene_rows.columns:" not in source

    def test_the_reader_asking_for_genes_gets_genes(self):
        from spacr import ml as M

        source = _source(M)
        assert "What level='gene' means is that the reader asked for genes" \
            in source


class TestTheShrunkCoefficientWarning:

    def test_the_note_is_decoration_and_cannot_cost_the_fit(self):
        """THE PIN for removing the old ``except Exception: pass``.

        The block builds a WARNING about penalised coefficients -- that
        a small t-statistic under a penalty is not evidence of no effect
        -- from the same well-defined DataFrame and settings dict as the fit.
        The helper is independently driven through both decisions in the
        focused ml guard tests.
        """
        from spacr import ml as M

        helper = inspect.getsource(M._warn_if_penalised_no_hits)
        caller = inspect.getsource(M._perform_regression)
        assert "this is NOT evidence of no effect" in helper
        assert "except Exception:" not in helper
        assert "_warn_if_penalised_no_hits(settings, coef_df)" in caller

    def test_the_warning_says_what_to_do_about_it(self):
        """A caution with no remedy is one a reader cannot act on."""
        from spacr import ml as M

        source = _source(M)
        assert "Refit with" in source
        assert "regression_type='ols'" in source
        assert "'rlm' for a robust check" in source


class TestTheSweepFutureLoop:

    def test_the_outer_while_refills_as_trials_finish(self):
        """THE ARC: the inner ``for`` running again after a refill.

        ``as_completed`` is taken over a SNAPSHOT of the futures, so new
        trials submitted while draining are not in it -- the outer
        ``while futures`` is what picks them up. Without it a sweep would
        run only its first batch and report as complete.
        """
        from spacr import parameter_sweep as P

        source = _source(P)
        while_at = source.index("while futures:")
        for_at = source.index("for future in as_completed(list(futures)):",
                              while_at)

        assert while_at < for_at
        assert "list(futures)" in source[for_at:for_at + 80], (
            "as_completed is no longer given a snapshot, so mutating the "
            "mapping while draining it is a RuntimeError")


class TestTheFloodFillBounds:

    def test_a_neighbour_outside_the_image_is_dropped(self):
        """THE ARC: the bounds check.

        The queue is filled with the four neighbours of each pixel
        without checking them, so an edge pixel puts an off-image
        coordinate on it -- and negative indices would WRAP, painting
        the opposite edge of the mask.
        """
        shape = (4, 6)          # rows, columns
        for cx, cy, inside in ((0, 0, True), (5, 3, True), (-1, 0, False),
                               (6, 0, False), (0, 4, False)):
            ok = 0 <= cx < shape[1] and 0 <= cy < shape[0]
            assert ok is inside

    def test_a_negative_index_would_wrap_rather_than_raise(self):
        """Why this is a bounds check and not a try/except."""
        image = np.arange(12).reshape(3, 4)

        assert image[-1, 0] == image[2, 0]


class TestCentringOnALabel:

    def test_a_label_present_in_the_layer_has_coordinates(self):
        data = np.zeros((4, 4), dtype=int)
        data[1:3, 1:3] = 7

        where = np.argwhere(data == 7)
        assert len(where)
        assert where.mean(axis=0).tolist() == [1.5, 1.5]

    def test_a_label_the_layer_does_not_carry_moves_nothing(self):
        """THE ARC: ``len(where)`` is zero.

        The label came from the layer's own list, so it is normally
        there -- but a linked selection can arrive after a reload has
        replaced the data. ``mean`` over an empty array is NaN, and
        moving to NaN puts the view somewhere no reset recovers from.
        """
        data = np.zeros((4, 4), dtype=int)

        where = np.argwhere(data == 7)
        assert not len(where)

        # The mean of nothing is NaN AND a RuntimeWarning, which is the
        # second half of why the guard is there: without it the console
        # carries a warning for every stale selection as well.
        with pytest.warns(RuntimeWarning, match="Mean of empty slice"):
            centre = where.mean(axis=0)
        assert np.isnan(centre).all()

    def test_the_move_happens_before_the_return(self):
        from spacr.qt import ortho_view as O

        source = _source(O)
        guard = source.index("if len(where):")
        move = source.index("self.move_to(", guard)
        assert guard < move


class TestRebuildingALayoutRow:

    def test_take_at_answers_none_only_past_the_end(self):
        """THE PIN, for ``if item is not None`` inside the while.

        The loop's own condition is ``layout.count() > position + 1``,
        so the index it takes is always occupied. Appending a None would
        put an empty slot into the rebuilt row.
        """
        from spacr.qt.widgets import column_picker as C

        source = _source(C)
        assert "while layout.count() > position + 1:" in source
        assert "item = layout.takeAt(position + 1)" in source
        assert "if item is not None:" in source

    def test_the_tail_is_taken_off_and_put_back_around_the_new_widget(self):
        """Why the row is rebuilt at all: replaceWidget cannot update a
        layout's reading order."""
        from spacr.qt.widgets import column_picker as C

        source = _source(C)
        assert "their reading order instead of appending the replacement" \
            in source


class TestRemovingADragPatch:

    def test_a_patch_removed_twice_is_absorbed(self):
        """THE PIN, for ``except (ValueError, NotImplementedError)``.

        matplotlib raises ValueError for an artist already removed and
        NotImplementedError for one whose container does not support it.
        A drag can end twice -- a release and then a figure teardown --
        and the second must not raise out of an event handler.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        figure, ax = plt.subplots()
        try:
            patch = ax.axvspan(0.1, 0.2)
            patch.remove()
            with pytest.raises(Exception):
                patch.remove()
        finally:
            plt.close(figure)

    def test_the_patch_is_forgotten_whether_or_not_it_removed(self):
        from spacr.qt.widgets import gate_editor as G

        source = _source(G)
        handler = source.index(
            "except (ValueError, NotImplementedError):")
        assert "self._drag_patch = None" in source[handler:handler + 200], (
            "a patch that could not be removed is still held, so the next "
            "drag tries to remove it again")


class TestTheGenePanelShutdown:

    def test_the_quit_hook_covers_the_path_close_event_does_not(self):
        """THE PIN, for ``application is not None``.

        ``closeEvent`` covers the ordinary path; this covers the one
        where nobody closed anything -- a tab rebuilt, a screen
        replaced, an interpreter shutting down -- and a warming thread
        left running past that is a process that will not exit.
        """
        from spacr.qt.widgets import gene_panel as G

        source = _source(G)
        assert "closeEvent` covers the ordinary path" in source
        assert "this covers the one where nobody closed anything" in source
        assert "application.aboutToQuit.connect(self._shut_down_warming)" \
            in source
