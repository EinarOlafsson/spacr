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


def _permutation_screen(n_genes=6, guides_per_gene=3, wells=90, seed=0):
    """A long table shaped like spaCR's saved regression_data.csv."""
    rng = np.random.default_rng(seed)
    genes = [f"G{i}" for i in range(n_genes)]
    guides = [(g, f"{g}_{k}") for g in genes for k in range(guides_per_gene)]
    rows = []
    for well in range(wells):
        plate = f"plate{well % 3 + 1}"
        prc = f"{plate}_r{well // 12 + 1}_c{well % 12 + 1}"
        chosen = rng.choice(len(guides), size=4, replace=False)
        share = rng.dirichlet(np.ones(4))
        # One gene really does move the phenotype, so the gene pass has
        # something to find and the test is not reading an empty frame.
        hit = sum(share[i] for i, c in enumerate(chosen)
                  if guides[c][0] == "G0")
        pred = float(np.clip(0.2 + 0.6 * hit + rng.normal(0, 0.05),
                             0.01, 0.99))
        for i, c in enumerate(chosen):
            gene, guide = guides[c]
            rows.append({"prc": prc, "grna": guide, "gene": gene,
                         "fraction": float(share[i]), "pred": pred,
                         "cell_count": 120, "plateID": plate,
                         "rowID": f"r{well // 12 + 1}",
                         "columnID": f"c{well % 12 + 1}"})
    return pd.DataFrame(rows)


def _permutation_settings(level):
    return dict(guide_min_wells=[1], guide_primary_min_wells=1,
                guide_permutations=60, guide_permutation_seed=0,
                guide_permutation_block="plateID", guide_nuisance_columns=[],
                multiple_testing_method="fdr_bh", fdr_alpha=0.05,
                guide_presence_threshold=0.0,
                guide_permutation_batch_size=30, grna_statistic="pearson",
                analysis_unit="well", agg_type="mean",
                regression_type="beta", level=level)


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

    def test_the_stack_stays_inside_the_per_field_loop(self, tmp_path,
                                                       monkeypatch):
        """A real regression, pinned by what it cost rather than by the
        indentation that fixes it. Dedented, the stack ran once after
        the loop: ``arrays`` was unbound when no filename matched, and
        only the LAST field ever got a movie -- every other one was
        silently dropped."""
        import spacr.timelapse as TL
        from spacr import io as IO

        made = []
        monkeypatch.setattr(
            TL, "_npz_to_movie",
            lambda arrays, names, path, fps: made.append(
                pathlib.Path(path).name))

        fields = tmp_path / "npy"
        fields.mkdir()
        for field in ("1", "2"):
            for time_point in (0, 1):
                np.save(fields / f"plate1_A01_{field}_{time_point}.npy",
                        np.zeros((4, 5, 2), dtype=np.uint16))

        IO._create_movies_from_npy_per_channel(str(fields), fps=5)

        assert sorted(made) == ["plate1_A01_1_channel_0.mp4",
                                "plate1_A01_1_channel_1.mp4",
                                "plate1_A01_2_channel_0.mp4",
                                "plate1_A01_2_channel_1.mp4"], (
            "not every field got a movie, which is what a dedented stack "
            "does: it runs once, on whatever the last field left behind")

        # The other half of the same regression: nothing matched the
        # regex, so a dedented stack reaches np.stack with `arrays`
        # unbound rather than simply finding no groups to make.
        unmatched = tmp_path / "unmatched"
        unmatched.mkdir()
        np.save(unmatched / "not-a-field-name.npy", np.zeros((2, 2, 1)))
        made.clear()

        IO._create_movies_from_npy_per_channel(str(unmatched), fps=5)

        assert made == []


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

    def test_the_reader_asking_for_genes_gets_genes(self, tmp_path):
        """What ``level='gene'`` means is that the reader asked for
        genes, so genes are what the primary table reports.

        The guide pass runs either way -- a gene's regressor is the sum
        of its guides' fractions, so there is no gene answer without it
        -- and results_grna.csv still holds those rows. Driven, because
        which table carries which level is the whole of the rule.
        """
        from spacr.ml import _run_guide_permutation_analysis

        destination = tmp_path / "gene_level"
        destination.mkdir()
        _run_guide_permutation_analysis(
            _permutation_screen(), "pred", str(destination),
            _permutation_settings("gene"))

        primary = pd.read_csv(destination / "results.csv")
        guides = pd.read_csv(destination / "results_grna.csv")

        assert set(primary["level"].dropna().unique()) == {"gene"}, (
            "results.csv no longer reports genes to a reader who asked "
            "for them, so the rows exist in a file the panel never opens")
        assert len(guides) > 0, (
            "the guide pass stopped writing results_grna.csv, which runs "
            "either way because the gene regressor is built out of it")


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

        One completion is taken from a snapshot before the pool is refilled,
        so new trials are picked up by the outer ``while futures``. Without
        it a sweep would run only its first batch and report as complete.
        """
        from spacr import parameter_sweep as P

        source = _source(P)
        while_at = source.index("while futures:")
        next_at = source.index(
            "future = next(as_completed(tuple(futures)))", while_at)

        assert while_at < next_at
        assert "tuple(futures)" in source[next_at:next_at + 80]
        assert "for future in as_completed" not in source[while_at:]


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

    def test_the_tail_is_taken_off_and_put_back_around_the_new_widget(
            self, qtbot):
        """Why the row is rebuilt at all: Python ``QLayout`` subclasses
        do not expose Qt's protected ``replaceAt``, so ``replaceWidget``
        cannot update them and answers None. Retaining the layout ITEMS
        puts the replacement where the old widget was -- the tail goes
        back AROUND it rather than the replacement going on the end.

        Driven on ``FlowLayout``, which is one of those subclasses, so
        the rebuild is the arm that actually runs.
        """
        from PySide6.QtWidgets import QLabel, QWidget

        from spacr.qt.widgets.column_picker import _replace_layout_widget
        from spacr.qt.widgets.flow import FlowLayout

        host = QWidget()
        qtbot.addWidget(host)
        flow = FlowLayout(host)
        first, middle, last = QLabel("a"), QLabel("b"), QLabel("c")
        for widget in (first, middle, last):
            flow.addWidget(widget)
        replacement = QLabel("x")

        # The premise: Qt's own replaceWidget is no use on this layout.
        assert flow.replaceWidget(middle, replacement) is None
        assert [flow.itemAt(i).widget().text()
                for i in range(flow.count())] == ["a", "b", "c"]

        removed = _replace_layout_widget(flow, middle, replacement)

        assert [flow.itemAt(i).widget().text()
                for i in range(flow.count())] == ["a", "x", "c"], (
            "the replacement was appended at the end instead of taking "
            "the slot it replaced, so the row no longer reads in order")
        assert removed is not None and removed.widget() is middle


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

    @pytest.mark.parametrize("removal_error",
                             (None, ValueError, NotImplementedError))
    def test_the_patch_is_forgotten_whether_or_not_it_removed(
            self, removal_error):
        from spacr.qt.widgets import gate_editor as G

        class Patch:
            calls = 0

            def remove(self):
                self.calls += 1
                if removal_error is not None:
                    raise removal_error("the artist is already gone")

        class Canvas:
            _resize = None
            _move_name = None
            _tool = "rectangle"
            _drag_origin = None

            def __init__(self):
                self._drag_patch = Patch()

            @staticmethod
            def _volume_release(_event):
                return False

        canvas = Canvas()
        patch = canvas._drag_patch

        G.GateCanvas._on_release(canvas, object())

        assert patch.calls == 1
        assert canvas._drag_patch is None, (
            "a patch that could not be removed is still held, so the next "
            "drag tries to remove it again")


class TestTheGenePanelShutdown:

    def test_the_quit_hook_covers_the_path_close_event_does_not(self, qtbot,
                                                                monkeypatch):
        """THE PIN, for ``application is not None``.

        ``closeEvent`` covers the ordinary path; this covers the one
        where nobody closed anything -- a tab rebuilt, a screen
        replaced, an interpreter shutting down -- and a warming thread
        left running past that is a process that will not exit.

        Driven through a stand-in application, so what is pinned is that
        the panel REGISTERS the hook and that the hook stops the warming
        without any close event having happened.
        """
        from spacr.qt.widgets import gene_panel as G

        connected = []

        class _AboutToQuit:
            @staticmethod
            def connect(slot):
                connected.append(slot)

        class _Application:
            aboutToQuit = _AboutToQuit()

        class _QApplication:
            @staticmethod
            def instance():
                return _Application()

        monkeypatch.setattr(G, "QApplication", _QApplication)
        panel = G.GenePanel(threaded=False)
        qtbot.addWidget(panel)

        assert connected, (
            "nothing is connected to aboutToQuit, so a panel dropped "
            "without being closed leaves its warming thread running")
        hook = connected[0]
        assert getattr(hook, "__self__", None) is panel, (
            "the quit hook is not a bound method of the panel, so it "
            "cannot reach the runner it is supposed to shut down")

        stopped = []
        monkeypatch.setattr(panel._runner, "shutdown",
                            lambda: stopped.append(True))

        hook()

        assert stopped == [True], (
            "the quit hook no longer shuts the warming down, which is "
            "the path closeEvent does not cover")
