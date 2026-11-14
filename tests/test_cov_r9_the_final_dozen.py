"""The last twelve sites of the audit.

Four in io, three in ml, and one each in parameter_sweep, mask_engine,
ortho_view, column_picker, gate_editor and gene_panel.
"""
from __future__ import annotations


import numpy as np
import pandas as pd
import pytest


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


class TestTheLevelColumn:

    def case_the_reader_asking_for_genes_gets_genes(self, tmp_path):
        """What ``level='gene'`` means is that the reader asked for
        genes, so genes are what the primary table reports.

        The guide pass runs either way -- a gene's regressor is the sum
        of its guides' fractions, so there is no gene answer without it
        -- and results_grna.csv still holds those rows. Driven, because
        which table carries which level is the whole of the rule.
        """
        _seen = []
        from spacr.ml import _run_guide_permutation_analysis

        destination = tmp_path / "gene_level"
        destination.mkdir()
        _run_guide_permutation_analysis(
            _permutation_screen(), "pred", str(destination),
            _permutation_settings("gene"))

        primary = pd.read_csv(destination / "results.csv")
        guides = pd.read_csv(destination / "results_grna.csv")

        _seen.append(keep(set(primary["level"].dropna().unique())))
        _seen.append(bool(
            len(guides) > 0))
        return _seen


class TestRebuildingALayoutRow:

    def case_the_tail_is_taken_off_and_put_back_around_the_new_widget(
            self, qtbot):
        """Why the row is rebuilt at all: Python ``QLayout`` subclasses
        do not expose Qt's protected ``replaceAt``, so ``replaceWidget``
        cannot update them and answers None. Retaining the layout ITEMS
        puts the replacement where the old widget was -- the tail goes
        back AROUND it rather than the replacement going on the end.

        Driven on ``FlowLayout``, which is one of those subclasses, so
        the rebuild is the arm that actually runs.
        """
        _seen = []
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
        _seen.append(bool(
            flow.replaceWidget(middle, replacement) is None))
        _seen.append(keep([flow.itemAt(i).widget().text()
                for i in range(flow.count())]))

        removed = _replace_layout_widget(flow, middle, replacement)

        _seen.append(keep([flow.itemAt(i).widget().text()
                for i in range(flow.count())]))
        _seen.append(bool(
            removed is not None and removed.widget() is middle))
        return _seen


class TestRemovingADragPatch:

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

    def case_the_quit_hook_covers_the_path_close_event_does_not(self, qtbot,
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
        _seen = []
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

        _seen.append(bool(
            connected))
        hook = connected[0]
        _seen.append(bool(
            getattr(hook, "__self__", None) is panel))

        stopped = []
        monkeypatch.setattr(panel._runner, "shutdown",
                            lambda: stopped.append(True))

        hook()

        _seen.append(keep(stopped))
        return _seen


from tests.observed_case import keep, observe  # noqa: E402

@pytest.mark.parametrize(("case", "expected"), [
    pytest.param((TestTheLevelColumn, 'case_the_reader_asking_for_genes_gets_genes'), [{'gene'}, True], id="the_reader_asking_for_genes_gets_genes"),
    pytest.param((TestRebuildingALayoutRow, 'case_the_tail_is_taken_off_and_put_back_around_the_new_widget'), [True, ['a', 'b', 'c'], ['a', 'x', 'c'], True], id="the_tail_is_taken_off_and_put_back_around_the_new_widget"),
    pytest.param((TestTheGenePanelShutdown, 'case_the_quit_hook_covers_the_path_close_event_does_not'), [True, True, [True]], id="the_quit_hook_covers_the_path_close_event_does_not"),
])
def test_r9_the_final_dozen_contracts(request, case, expected):
    """One contract per row: the case records what it checked, the row states it."""
    observed = observe(case, request)
    assert observed == expected


def _centring_outcome(array, label):
    """Select ``label`` through the real handler and return what the view did.

    The handler is driven unbound on a stand-in view, with a labels layer
    whose key lookup says object ``'key7'`` is ``label`` -- so the only thing
    that decides whether the view moves is whether ``label`` is in the data.
    """
    import types
    import warnings

    from spacr.layers import LabelsLayer
    from spacr.qt import ortho_view as O

    moves = []

    class _Spacing:
        @staticmethod
        def world_map(centre):
            return {"y": float(centre[0]), "x": float(centre[1])}

    class _Field:
        @staticmethod
        def object_key(value):
            return "key7" if value == label else "other"

    class _Layer(LabelsLayer):
        def __init__(self):
            pass

        data = array
        field = _Field()
        spacing = _Spacing()
        selected_label = 0

        def labels(self):
            return [label]

    layer = _Layer()
    view = types.SimpleNamespace(_stack=[layer],
                                 move_to=lambda **where: moves.append(where))
    selection = types.SimpleNamespace(keys=["key7"])
    with warnings.catch_warnings():
        warnings.simplefilter("error")      # a mean of nothing would raise here
        O.OrthoView.on_linked_selection_changed(view, selection)
    return moves, layer.selected_label


_PRESENT = np.zeros((4, 4), dtype=int)
_PRESENT[1:3, 1:3] = 7


@pytest.mark.parametrize(("array", "label", "expected"), [
    pytest.param(_PRESENT, 7, ([{"y": 1.5, "x": 1.5}], 7),
                 id="a-label-in-the-data-moves-the-view-to-its-centre"),
    pytest.param(np.zeros((4, 4), dtype=int), 7, ([], 7),
                 id="a-label-the-data-no-longer-carries-moves-nothing"),
])
def test_a_linked_selection_centres_only_on_an_object_that_is_still_there(
        array, label, expected):
    """The ``len(where)`` guard: a stale selection must not move the view to NaN."""
    observed = _centring_outcome(array, label)
    assert observed == expected
