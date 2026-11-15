"""Table-driven tests of the result-sheet panels, plates, summary and fast renderer.

The result sheet turns one coefficient table into panels, a plate heatmap and
a summary paragraph, and the fast renderer writes the same panels with
pyqtgraph.  One general test per unit walks a case table with pytest's
``subtests`` fixture: the volcano's labels and highlight, which panels a table
supports (``available``), the per-plate well matrices, the summary paragraph,
when the fast renderer forces Qt offscreen, and where it writes a file.  Every
expected value is exact: labels, answers, matrices and whole sentences.
"""
from __future__ import annotations

import os

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from spacr.figures import fast_render as fr  # noqa: E402
from spacr.figures import panels, plates  # noqa: E402
from spacr.figures.summary import summarise  # noqa: E402

NAN = float("nan")


@pytest.fixture
def ax():
    """One fresh axes, closed afterwards."""
    figure = plt.figure()
    try:
        yield figure.add_subplot(111)
    finally:
        plt.close(figure)


def _called():
    """Five significant, large-effect coefficients, one of them unnamed."""
    return pd.DataFrame({
        "gene": ["GRA14", "ROP18", NAN, "MYR1", "ASP5"],
        "coefficient": [2.0, -2.2, 2.4, 1.8, -1.9],
        "p_value": [1e-8, 1e-9, 1e-10, 1e-7, 1e-6],
    })


COUNTS = ["3 up", "2 down", "0 not called"]


def test_volcano_labels_named_hits_and_outlines_only_a_present_highlight(
        subtests, monkeypatch):
    """Named hits are labelled, NaN never is; a highlight outlines one point."""
    # (case id, label_top, highlight, labels, single-point outlines)
    cases = [
        ("the unnamed strongest point is not labelled nan", 5, None,
         ["p = 0.05", "ROP18", "GRA14", "MYR1", "ASP5", *COUNTS], []),
        ("a highlighted coefficient is outlined", 0, "ROP18",
         ["p = 0.05", *COUNTS], [[[-2.2, 9.0]]]),
        ("a highlight that names nothing draws no outline", 0,
         "not_in_this_screen", ["p = 0.05", *COUNTS], []),
        ("no highlight draws no outline", 0, None, ["p = 0.05", *COUNTS], []),
    ]
    # pandas 3 can keep the float NaN in a nominally string-like Series;
    # force that shape on every pandas version.
    monkeypatch.setattr(panels, "label_series",
                        lambda frame: frame["gene"].astype(object))
    for case, top, highlight, labels, outlines in cases:
        with subtests.test(msg=case):
            figure, axes = plt.subplots()
            before = len(axes.collections)
            panel = panels.volcano(axes, _called(), effect_threshold=None,
                                   label_top=top, highlight=highlight)
            assert (panel.drawn, panel.reason) == (True, "")
            assert [text.get_text() for text in axes.texts] == labels
            assert [c.get_offsets().tolist() for c in axes.collections[before:]
                    if len(c.get_offsets()) == 1] == outlines
            plt.close(figure)


SCREEN = pd.DataFrame({
    "feature": [f"fraction:grna[g{i}]" for i in range(30)],
    "gene": [f"G{i // 3}" for i in range(30)],
    "coefficient": np.linspace(-2.0, 2.0, 30),
    "p_value": np.linspace(1e-6, 0.9, 30),
})
GENES = pd.DataFrame({"gene": ["A", "B", "C"], "coefficient": [1.0, -1.0, 0.2],
                      "p_value": [0.001, 0.002, 0.5]})
NO_EFFECT = pd.DataFrame({"p_value": [0.01, 0.4], "gene": ["A", "B"]})


def _answers(true):
    return {key: key in true for key in panels.SHEET_ORDER}


def test_available_answers_per_panel_without_drawing_or_raising(
        subtests, monkeypatch):
    """Each panel answers True/False; a broken panel is False, not an error."""
    def explode(_ax, _frame):
        raise RuntimeError("this panel is broken")

    effect = ("volcano", "effect_rank", "effect_distribution", "p_histogram",
              "qq")
    # (case id, table, broken panel, panels that can be drawn)
    cases = [
        ("a per-guide screen supports all but the controls", SCREEN, None,
         effect + ("agreement",)),
        ("a gene-level table has no guide agreement", GENES, None, effect),
        ("a table with no effect supports only the p histogram", NO_EFFECT,
         None, ("p_histogram",)),
        ("a panel that raises is unavailable, the rest are not", GENES, "qq",
         effect[:-1]),
    ]
    assert panels.SHEET_ORDER == ("volcano", "effect_rank",
                                  "effect_distribution", "controls",
                                  "agreement", "p_histogram", "qq")
    for case, table, broken, drawable in cases:
        with subtests.test(msg=case), monkeypatch.context() as patch:
            if broken:
                patch.setitem(panels.REGISTRY, broken, explode)
            caller = plt.figure()
            figures_before = set(plt.get_fignums())
            try:
                assert panels.available(table) == _answers(drawable)
                assert set(plt.get_fignums()) == figures_before
            finally:
                plt.close(caller)


def test_well_matrices_paint_only_wells_with_numbers(subtests):
    """Plates are named, unreadable wells and non-numbers stay NaN."""
    # (case id, rows, column, keyword arguments, names, matrices, shape)
    cases = [
        ("identifiers that name no well give no matrices",
         pd.DataFrame({"prc": ["x_y_z", "x_y_w"], "v": [1.0, 2.0]}), "v", {},
         ["x"], [], (0, 0)),
        ("a plate with no numeric value is drawn as absent",
         pd.DataFrame({"prc": ["p1_r1_c1", "p1_r1_c2", "p2_r1_c1",
                               "p2_r1_c2"], "v": [1.0, 2.0, "n/a", ""]}),
         "v", {}, ["p1", "p2"],
         [[[1.0, 2.0, NAN], [NAN, NAN, NAN]],
          [[NAN, NAN, NAN], [NAN, NAN, NAN]]], (2, 3)),
        ("keys without a plate prefix are read as one plate",
         pd.DataFrame({"prc": ["p1_r1_c1", "r2_c2"], "value": [4.0, 8.0]}),
         "value", {"plates": ["p1"]}, ["p1"],
         [[[4.0, NAN, NAN], [NAN, NAN, NAN]]], (2, 3)),
    ]
    for case, rows, column, options, names, matrices, shape in cases:
        with subtests.test(msg=case):
            got_names, got, got_shape = plates.well_matrices(rows, column,
                                                             **options)
            assert (got_names, tuple(got_shape)) == (names, shape)
            assert len(got) == len(matrices)
            for matrix, expected in zip(got, matrices):
                np.testing.assert_array_equal(matrix, np.array(expected))


def _guides(coefficients, p_values):
    n = len(coefficients)
    return pd.DataFrame({"feature": [f"fraction:gene[g{i}]" for i in range(n)],
                         "gene": [f"g{i}" for i in range(n)],
                         "coefficient": coefficients, "p_value": p_values})


TESTED = "{n} coefficients were tested. {k} pass uncorrected p ≤ 0.05 " \
         "(NO correction was applied)."


def test_summarise_writes_exactly_what_was_applied(subtests):
    """The paragraph quotes the cut it applied, or none, or says nothing."""
    cases = [
        ("a table with no fitted effect gets no paragraph",
         pd.DataFrame({"feature": ["fraction:gene[abc]"], "p_value": [0.01]}),
         {}, ""),
        ("the effect-size cut you set is applied and quoted",
         _guides([3.0, -2.5, 0.2, 0.1, -0.05, 1.9],
                 [1e-6, 1e-6, 1e-6, 0.4, 0.6, 0.7]),
         {"effect_threshold": 2.0},
         TESTED.format(n=6, k=3) + " Of those, 2 also clear the effect-size "
         "cut of 2 (the value you set) — the rest are detectable but "
         "smaller than the spread of the guides that target nothing. "
         "Strongest: g0 (+3), g1 (-2.5)."),
        ("no cut lists every significant hit and quotes no cut",
         _guides([3.0, 0.2, 0.1, -0.05], [1e-6, 1e-6, 0.4, 0.6]),
         {"effect_threshold": None},
         TESTED.format(n=4, k=2) + " Strongest: g0 (+3), g1 (+0.2)."),
    ]
    for case, table, options, paragraph in cases:
        with subtests.test(msg=case):
            assert summarise(table, **options) == paragraph


# ---------------------------------------------------------------------------
# the fast renderer
# ---------------------------------------------------------------------------

class _RunningApplication:
    """Enough of a QApplication for the module and for pytest-qt."""

    def processEvents(self):  # noqa: N802 (Qt naming)
        return None


class _StandInApplication:
    """A QApplication class whose instance is already running."""

    running = _RunningApplication()

    @classmethod
    def instance(cls):
        return cls.running


def test_fast_renderer_goes_offscreen_only_without_a_display(
        monkeypatch, subtests):
    """A display keeps Qt's platform; no display installs ``offscreen``."""
    pytest.importorskip("pyqtgraph")
    qtwidgets = pytest.importorskip("PySide6.QtWidgets")
    for case, display, platform in (
            ("a display is not pushed offscreen", ":0", None),
            ("no display renders offscreen", None, "offscreen")):
        with subtests.test(msg=case), monkeypatch.context() as patch:
            patch.setattr(qtwidgets, "QApplication", _StandInApplication)
            patch.setattr(fr, "qt_application", lambda: None)
            patch.setattr(fr, "_APPLICATION", None)
            for name in ("QT_QPA_PLATFORM", "WAYLAND_DISPLAY", "DISPLAY"):
                patch.delenv(name, raising=False)
            if display:
                patch.setenv("DISPLAY", display)
            assert fr._pyqtgraph_ready(create=True) == (True, "")
            assert fr._APPLICATION is _StandInApplication.running
            assert os.environ.get("QT_QPA_PLATFORM") == platform


class _Plot:
    """A scene that records where it was asked to export itself."""

    def __init__(self, written=None, error=None):
        self.asked = []
        self._written = written
        self._error = error

    def export(self, destination):
        self.asked.append(destination)
        if self._error:
            raise self._error
        return self._written


def test_render_with_pyqtgraph_writes_where_asked_and_never_raises(
        tmp_path, monkeypatch, subtests):
    """A destination is exported (folder made only if named) or refused."""
    monkeypatch.chdir(tmp_path)
    nested = str(tmp_path / "figures" / "volcano.png")
    # (case id, plot, destination, path, drawn, reason, folders made)
    cases = [
        ("a bare filename needs no folder", _Plot(), "volcano.png",
         "volcano.png", True, "", []),
        ("a named folder is made", _Plot(), nested, nested, True, "",
         ["figures"]),
        ("the exporter's own answer is the path", _Plot(written="other.png"),
         "volcano.png", "other.png", True, "", ["figures"]),
        ("no destination is a reason", _Plot(), None, None, False,
         "no destination was given", ["figures"]),
        ("an exporter that raises is a reason",
         _Plot(error=OSError("disk full")), "volcano.png", None, False,
         "OSError: disk full", ["figures"]),
    ]
    for case, plot, destination, path, drawn, reason, folders in cases:
        with subtests.test(msg=case):
            panel = fr._render_with_pyqtgraph("volcano", None, destination,
                                              plot=plot, announce=False)
            assert (panel.key, panel.renderer, panel.path, panel.drawn,
                    panel.reason) == ("volcano", "pyqtgraph", path, drawn,
                                      reason)
            assert plot.asked == ([] if destination is None
                                  else [destination])
            assert sorted(p.name for p in tmp_path.iterdir()
                          if p.is_dir()) == folders
