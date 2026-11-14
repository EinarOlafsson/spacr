"""The volcano plot's restyle dialogs, its drag selection, and its q colouring.

Each one-question dialog has two outcomes that both matter: an answer reaches
every scatter, and a cancel changes nothing.  A modified drag selects what it
encloses while a plain one stays pyqtgraph's pan.  The q colouring, opacity and
correction say in the caption exactly what the picture carries.  One general
test per behaviour family, each walking a case table with ``subtests``.
"""
from __future__ import annotations

import builtins
import types

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")

import pyqtgraph as pg  # noqa: E402
from PySide6.QtCore import QPointF, Qt  # noqa: E402
from PySide6.QtGui import QColor  # noqa: E402
from PySide6.QtWidgets import QCheckBox, QFileDialog, QInputDialog, QMenu  # noqa: E402

from spacr.qt.widgets import fast_plots as fp  # noqa: E402

pytestmark = pytest.mark.qt

N = 40


def _screen(seed: int = 0) -> pd.DataFrame:
    """A coefficient table with one real hit, so a threshold line is drawn."""
    rng = np.random.default_rng(seed)
    p_values = rng.uniform(0.001, 0.99, N)
    p_values[0] = 1e-8
    return pd.DataFrame({
        "feature": [f"fraction:grna[{i}_1]" for i in range(N)],
        "coefficient": rng.normal(0, 0.5, N),
        "p_value": p_values,
        "well_count": rng.integers(4, 96, N).astype("float64"),
        "n_guides": rng.integers(1, 4, N),
        "condition": list(rng.choice(["nc", "pc", "other"], N, p=[0.1, 0.1, 0.8])),
    })


def _coefficients(n=12, *, p=None, feature=True):
    """A coefficient table the shape ``perform_regression`` writes."""
    frame = pd.DataFrame({
        "grna": [f"g{i}" for i in range(n)],
        "coefficient": np.linspace(-2.0, 2.0, n),
        "p_value": np.linspace(1e-6, 0.9, n) if p is None else np.asarray(p, dtype=float),
    })
    if feature:
        frame.insert(0, "feature", [f"fraction:grna[g{i}]" for i in range(n)])
    return frame


@pytest.fixture
def make(qtbot):
    """A VolcanoPlot; drawn over ``_screen()`` unless ``drawn=False``."""
    def build(drawn=True):
        plot = fp.VolcanoPlot()
        qtbot.addWidget(plot)
        if drawn:
            plot.set_results(_screen(), effect_threshold=0.6)
        return plot
    return build


def _answers(mp, name, *answers):
    queue = iter(answers)
    mp.setattr(QInputDialog, name, staticmethod(lambda *a, **k: next(queue)))


def _scatters(widget):
    return [item for item in widget.plot.plotItem.items
            if isinstance(item, pg.ScatterPlotItem)]


def _brush_names(item):
    return [brush.color().name() for brush in item.data["brush"]]


def _legend_labels(widget):
    legend = getattr(widget.plot.plotItem, "legend", None)
    return [] if legend is None else [label.text for _, label in legend.items]


def _legend_box(widget):
    return [box for box in widget.findChildren(QCheckBox)
            if box.text().startswith("legend")][0]


def test_a_restyle_answer_reaches_every_scatter_and_a_cancel_nothing(make, monkeypatch, subtests):
    """Size, opacity, colour, labels, cut and split: applied on a full OK, untouched on Cancel."""
    def sizes(p):
        return [item.opts["size"] for item in p._scatter_items()]

    def opacities(p):
        return [item.opacity() for item in p._scatter_items()]

    def brushes(p):
        return [item.opts["brush"].color().name() for item in p._scatter_items()]

    cases = [
        # case_id, ask, patch, read, expected (None = unchanged)
        ("a_point_size", "_ask_point_size", ("getDouble", (21.0, True)), sizes, 21.0),
        ("a_cancelled_point_size", "_ask_point_size", ("getDouble", (21.0, False)), sizes, None),
        ("an_opacity", "_ask_opacity", ("getDouble", (0.25, True)), opacities, 0.25),
        ("a_cancelled_opacity", "_ask_opacity", ("getDouble", (0.25, False)), opacities, None),
        ("a_point_colour", "_ask_point_colour", ("pick", QColor("#ff0055")), brushes, "#ff0055"),
        ("a_cancelled_colour", "_ask_point_colour", ("pick", QColor()), brushes, None),
    ]
    for case_id, ask, (name, answer), read, expected in cases:
        with subtests.test(msg=case_id):
            plot = make()
            before = read(plot)
            with monkeypatch.context() as mp:
                if name == "pick":
                    mp.setattr(fp, "pick_colour", lambda *a, _c=answer, **k: _c)
                else:
                    _answers(mp, name, answer)
                getattr(plot, ask)()
            after = read(plot)
            if expected is None:
                assert after == before
            elif isinstance(expected, float):
                assert after == pytest.approx([expected] * len(before))
            else:
                assert after == [expected] * len(before)
    with subtests.test(msg="both_axis_labels_are_applied"):
        plot = make()
        with monkeypatch.context() as mp:
            _answers(mp, "getText", ("effect size", True), ("-log10 p", True))
            plot._ask_labels()
        assert (plot.plot.getAxis("bottom").labelText,
                plot.plot.getAxis("left").labelText) == ("effect size", "-log10 p")
    with subtests.test(msg="cancelling_the_second_label_keeps_the_first"):
        plot = make()
        before = (plot.plot.getAxis("bottom").labelText, plot.plot.getAxis("left").labelText)
        with monkeypatch.context() as mp:
            _answers(mp, "getText", ("effect size", True), ("-log10 p", False))
            plot._ask_labels()
        assert (plot.plot.getAxis("bottom").labelText,
                plot.plot.getAxis("left").labelText) == before
    with subtests.test(msg="cancelling_the_first_label_asks_nothing_more"):
        plot = make()
        asked = []
        with monkeypatch.context() as mp:
            mp.setattr(QInputDialog, "getText",
                       staticmethod(lambda *a, **k: asked.append(1) or ("", False)))
            plot._ask_labels()
        assert asked == [1]
    for case_id, answer, expected in (("an_effect_size_cut_is_reported", (2.5, True), [2.5]),
                                      ("a_cancelled_cut_calls_nobody", (2.5, False), [])):
        with subtests.test(msg=case_id):
            plot = make()
            chosen: list = []
            with monkeypatch.context() as mp:
                _answers(mp, "getDouble", answer)
                plot._ask_threshold_multiplier(1.0, chosen.append)
            assert chosen == expected
    for case_id, answers, expected in (
            ("a_split_takes_both_ends", [(4.0, True), (7.0, True)], (4.0, 7.0)),
            ("cancelling_the_first_end", [(4.0, False)], None),
            ("cancelling_the_second_end", [(4.0, True), (7.0, False)], None)):
        with subtests.test(msg=case_id):
            plot = make()
            with monkeypatch.context() as mp:
                _answers(mp, "getDouble", *answers)
                plot._ask_y_split()
            assert plot.y_split() == expected


class _Drag:
    """The parts of a pyqtgraph drag event the wrapper actually reads."""

    def __init__(self, modifier, down, now, finish=True):
        self._modifier, self._down, self._now, self._finish = modifier, down, now, finish
        self.accepted = False

    def modifiers(self):
        return self._modifier

    def button(self):
        return Qt.LeftButton

    def accept(self):
        self.accepted = True

    def buttonDownPos(self):
        return self._down

    def pos(self):
        return self._now

    def isFinish(self):
        return self._finish


def test_a_modified_drag_selects_and_a_plain_one_pans(make, subtests):
    """The rubber band wraps pyqtgraph's drag once and only for modified drags."""
    def banded():
        plot = make()
        box = plot.plot.getViewBox()
        handed_back: list = []
        box.mouseDragEvent = lambda event, axis=None: handed_back.append(event)
        box._spacr_band = False
        plot._install_rubber_band()
        return plot, box, handed_back

    with subtests.test(msg="a_plain_drag_is_handed_back"):
        plot, box, handed_back = banded()
        event = _Drag(Qt.NoModifier, QPointF(0, 0), QPointF(10, 10))
        box.mouseDragEvent(event)
        assert (handed_back, event.accepted, plot.selected_keys()) == ([event], False, [])
    with subtests.test(msg="a_drag_in_progress_draws_and_selects_nothing"):
        plot, box, handed_back = banded()
        event = _Drag(Qt.ControlModifier, QPointF(0, 0), QPointF(10, 10), finish=False)
        assert box.mouseDragEvent(event) is None
        assert (handed_back, event.accepted, plot.selected_keys()) == ([], True, [])
    with subtests.test(msg="a_finished_drag_over_everything_selects_every_point"):
        plot, box, _ = banded()
        (low_x, high_x), (low_y, high_y) = plot.axis_limits()
        span_x, span_y = high_x - low_x, high_y - low_y
        event = _Drag(Qt.ShiftModifier,
                      box.mapFromView(QPointF(low_x - span_x, low_y - span_y)),
                      box.mapFromView(QPointF(high_x + span_x, high_y + span_y)))
        box.mouseDragEvent(event)
        assert event.accepted is True
        assert sorted(plot.selected_keys()) == sorted(_screen()["feature"])
    with subtests.test(msg="installed_once_not_once_per_call"):
        plot, box, _ = banded()
        wrapped = box.mouseDragEvent
        plot._install_rubber_band()
        assert box.mouseDragEvent is wrapped


def test_the_volcano_show_as_menu_degrades_without_its_table(make, monkeypatch, subtests):
    """A failed redraw keeps the plot; no graph-type table means no submenu."""
    with subtests.test(msg="a_redraw_that_fails_leaves_the_plot"):
        plot = make()
        before = len(plot._scatter_items())
        plot._show_as_kind("bar")
        assert len(plot._scatter_items()) == before
    with subtests.test(msg="no_graph_type_table_no_submenu"):
        plot = make()
        plot.spec = types.SimpleNamespace(frame=_screen(), group="condition",
                                          value="coefficient", kind="bar")
        real_import = builtins.__import__

        def blocked(name, globals=None, locals=None, fromlist=(), level=0):
            if level and name == "graph_types":
                raise ImportError("graph types are unavailable")
            return real_import(name, globals, locals, fromlist, level)

        menu = QMenu()
        with monkeypatch.context() as mp:
            mp.setattr(builtins, "__import__", blocked)
            assert plot._offer_graph_kinds(menu) is None
        assert menu.actions() == []


def test_the_q_encoding_and_correction_are_stated_in_the_caption(make, subtests):
    """Ramp, opacity, compartments and correction: the caption names what is drawn."""
    missing = QColor(fp.MISSING_COLOUR).name()
    with subtests.test(msg="an_unknown_correction_is_kept_but_not_checked"):
        plot = make(drawn=False)
        frame = _coefficients()
        frame["q_value"] = 0.5
        plot.set_results(frame, run_method="peeking-until-significant")
        assert (plot.run_correction(), plot.correction()) == (
            "peeking-until-significant", "fdr_bh")
        assert "WARNING" not in plot.caption()
        plot.set_results(frame, run_method="bh")
        assert plot.correction() == "fdr_bh"
        assert "WARNING" in plot.caption()
    with subtests.test(msg="every_p_underflowed_paints_every_point_missing"):
        plot = make(drawn=False)
        plot.set_q_colour("ramp")
        assert plot.set_results(_coefficients(8, p=[0.0] * 8)) == 8
        _legend_box(plot).setChecked(True)
        assert set(_brush_names(_scatters(plot)[0])) == {missing}
        assert "no q (8)" in _legend_labels(plot)
    with subtests.test(msg="one_shared_q_is_one_colour_a_spread_is_many"):
        plot = make(drawn=False)
        plot.set_q_colour("ramp")
        assert plot.set_results(_coefficients(8, p=[0.4] * 8)) == 8
        flat = set(_brush_names(_scatters(plot)[0]))
        assert len(flat) == 1 and flat != {missing}
        assert plot.set_results(_coefficients(8)) == 8
        assert len(set(_brush_names(_scatters(plot)[0]))) > 1
    with subtests.test(msg="no_q_at_all_draws_no_ramp_key"):
        plot = make(drawn=False)
        plot.set_q_colour("ramp")
        plot.set_results(_coefficients(8, p=[np.nan] * 8))
        _legend_box(plot).setChecked(True)
        assert [t for t in _legend_labels(plot) if t.startswith("q ")] == []
        assert plot.set_results(_coefficients(8)) == 8
        assert [t for t in _legend_labels(plot) if t.startswith("q ")] != []
    with subtests.test(msg="the_ramp_names_the_compartment_colouring_it_displaced"):
        plot = make(drawn=False)
        plot.set_results(_coefficients(), compartment="nucleus")
        assert "colouring is off while it is" not in plot.caption()
        plot.set_q_colour("ramp")
        plot.set_results(_coefficients(), compartment="nucleus")
        assert "the nucleus colouring is off while it is" in plot.caption()
    with subtests.test(msg="all_compartments_with_no_genes_has_no_legend"):
        plot = make(drawn=False)
        frame = _coefficients(feature=False)
        assert plot.set_results(frame, compartment=fp.FastPlot._all_compartments()) == 12
        assert _legend_box(plot).isEnabled() is False
        assert plot.set_results(frame) == 12
        assert _legend_box(plot).isEnabled() is True
        assert "legend (2)" in _legend_box(plot).text()
    with subtests.test(msg="opacity_by_q_is_captioned_even_with_nothing_drawn"):
        plot = make(drawn=False)
        plot.set_q_mark("opacity")
        assert plot.set_results(_coefficients(6, p=[np.nan] * 6)) == 0
        assert "Point OPACITY is the evidence against the null" in plot.caption()
        assert plot.set_results(_coefficients(6)) == 6
    with subtests.test(msg="a_ticked_legend_returns_after_a_redraw"):
        plot = make(drawn=False)
        plot.set_results(_coefficients())
        _legend_box(plot).setChecked(True)
        assert sorted(_legend_labels(plot)) == ["called (1)", "not called (11)"]
        plot.set_correction("bonferroni")
        assert set(_legend_labels(plot)) == {"called (1)", "not called (11)"}
        assert "Bonferroni" in plot.caption()


def test_the_corrected_table_is_written_only_with_q_values_and_a_path(make, tmp_path, monkeypatch, subtests):
    """``_write_corrected_table``: None for a cancel or no q values, else the path."""
    def save_to(mp, path):
        mp.setattr(QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: (path, "")))

    with subtests.test(msg="a_cancelled_dialog_writes_nothing"):
        plot = make(drawn=False)
        plot.set_results(_coefficients())
        with monkeypatch.context() as mp:
            save_to(mp, "")
            assert plot._write_corrected_table() is None
        assert list(tmp_path.iterdir()) == []
    with subtests.test(msg="an_empty_reload_has_nothing_to_write"):
        plot = make(drawn=False)
        plot.set_results(_coefficients())
        assert plot.set_results(pd.DataFrame()) == 0
        with monkeypatch.context() as mp:
            save_to(mp, str(tmp_path / "no.csv"))
            assert plot._write_corrected_table() is None
        assert not (tmp_path / "no.csv").exists()
    with subtests.test(msg="a_chosen_path_gets_q_and_local_fdr"):
        plot = make(drawn=False)
        plot.set_results(_coefficients())
        written = tmp_path / "recorrected.csv"
        with monkeypatch.context() as mp:
            save_to(mp, str(written))
            assert plot._write_corrected_table() == str(written)
        saved = pd.read_csv(written)
        assert len(saved) == 12
        assert {"q_value", "local_fdr"} <= set(saved.columns)
