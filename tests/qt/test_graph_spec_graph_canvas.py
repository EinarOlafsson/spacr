"""How the graph builder's canvas and the PCA view render a graph_spec ``GraphSpec``.

The canvas is rendered for real -- a matplotlib figure on the offscreen
platform -- and the assertions are about the coordinates and artists that
come out, so a branch that draws nothing cannot pass by not raising.  The
quiet paths are pinned too: nothing to jitter, a brush on a panel that is not
there, a preview patch already gone, a PCA plane naming a component the
result does not have.

One general test per behaviour family, each walking a case table with
pytest's ``subtests`` fixture.
"""
from __future__ import annotations

import types

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("matplotlib")

from matplotlib.backend_bases import MouseEvent  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from spacr.qt.linked_selection import LinkedSelection  # noqa: E402
from spacr.qt.widgets import graph_builder as gb  # noqa: E402
from spacr.qt.widgets.graph_spec import (  # noqa: E402
    BAR, HISTOGRAM, SCATTER, GraphSpec,
)

pytestmark = pytest.mark.qt


def _frame(rows=60):
    rng = np.random.default_rng(5)
    return pd.DataFrame({
        "plateID": [f"p{i % 2 + 1}" for i in range(rows)],
        "rowID": [f"r{i % 3 + 1}" for i in range(rows)],
        "columnID": [f"c{i % 4 + 1}" for i in range(rows)],
        "area": rng.normal(100, 10, rows),
        "intensity": rng.normal(size=rows),
        "gene": [f"g{i % 3}" for i in range(rows)],
        "cell_count": [i % 5 for i in range(rows)],
        "fieldID": [f"f{i % 2 + 1}" for i in range(rows)],
        "object_label": list(range(rows)),
    })


@pytest.fixture
def make(qtbot):
    """A fresh GraphCanvas over ``_frame()`` showing ``spec`` immediately."""
    def build(spec=None):
        canvas = gb.GraphCanvas(link=LinkedSelection(), source="graph_spec")
        qtbot.addWidget(canvas)
        canvas.set_frame(_frame())
        if spec is not None:
            canvas.set_spec(spec, immediate=True)
        return canvas
    return build


def test_the_spec_places_every_row_and_keeps_the_rest(make, subtests):
    """set_channel edits one channel; _xy places rows by value or level index."""
    frame = _frame()
    with subtests.test(msg="one_channel_keeps_the_rest"):
        canvas = make(GraphSpec(x="area", y="intensity", kind=SCATTER))
        canvas.set_channel("colour", "gene")
        assert (canvas.spec.x, canvas.spec.y, canvas.spec.colour, canvas.spec.kind) == (
            "area", "intensity", "gene", SCATTER)
    with subtests.test(msg="an_axis_with_no_column_sits_at_zero"):
        canvas = make(GraphSpec(x="area", kind=HISTOGRAM))
        x, y = canvas._xy(canvas._frame)
        np.testing.assert_array_equal(y, np.zeros(len(frame)))
        np.testing.assert_allclose(x, frame["area"].to_numpy(float))
    with subtests.test(msg="a_categorical_axis_is_the_level_index"):
        canvas = make(GraphSpec(x="gene", y="area", kind=SCATTER))
        x, y = canvas._xy(canvas._frame)
        np.testing.assert_array_equal(x, np.arange(len(frame)) % 3)
        np.testing.assert_allclose(y, frame["area"].to_numpy(float))
    with subtests.test(msg="no_measurement_means_nothing_to_jitter"):
        canvas = make(GraphSpec(x="gene", y="", kind=BAR))
        ax = Figure().add_subplot(111)
        canvas._draw_jitter(ax, canvas._render_data.frame, gb.active_palette(), over_bars=True)
        assert (len(ax.collections), len(ax.lines)) == (0, 0)
    with subtests.test(msg="the_grid_notice_is_relayed"):
        canvas = make(GraphSpec(x="area", y="intensity", kind=SCATTER))
        said = canvas._notice_text(canvas._render_data, types.SimpleNamespace(
            notice="only the first 6 panels are drawn"))
        assert "only the first 6 panels are drawn" in said


def test_every_kind_draws_its_marks_and_quiet_gestures_draw_nothing(make, subtests):
    """Histogram stacks per level, aggregates draw marks, stray gestures are no-ops."""
    with subtests.test(msg="histogram_stacked_by_a_categorical_colour"):
        canvas = make(GraphSpec(x="area", colour="gene", kind=HISTOGRAM, bins=8))
        figure = canvas.figure()
        assert [bars.get_label() for bars in figure.axes[0].containers] == ["g0", "g1", "g2"]
        assert [t.get_text() for t in figure.legends[0].get_texts()] == ["g0", "g1", "g2"]
        assert len(figure.axes[0].patches) == 24
    for kind in ("jitter", "box", "violin", "bar_jitter"):
        with subtests.test(msg=f"aggregate_{kind}"):
            ax = make(GraphSpec(x="gene", y="area", kind=kind)).figure().axes[0]
            assert (len(ax.collections) + len(ax.patches) + len(ax.lines)
                    + len(ax.containers)) > 0
    with subtests.test(msg="brush_on_a_panel_that_is_not_there"):
        canvas = make(GraphSpec(x="area", y="intensity", kind=SCATTER))
        assert canvas.brush(0.0, 0.0, 1.0, 1.0, row=9, col=9) is None
    with subtests.test(msg="a_preview_patch_already_gone"):
        canvas = make(GraphSpec(x="area", y="intensity", kind=SCATTER))

        class Detached:
            def remove(self):
                raise ValueError("artist is not in the axes")

        canvas._drag_patch = Detached()
        canvas._drag_origin = None
        canvas._on_release(MouseEvent("button_release_event", canvas._canvas, 5.0, 5.0, 1))
        assert canvas._drag_patch is None


def test_the_pca_plane_exists_only_when_a_result_backs_it(qtbot, subtests):
    """PCAScoresCanvas.plane() is (i, j) for components the result has, else None."""
    from spacr.qt.widgets.pca_model import PCASpec, pca
    from spacr.qt.widgets.pca_view import PCAScoresCanvas

    rng = np.random.default_rng(3)
    n = 120
    shift = np.where(np.arange(n) < n // 2, -1.0, 1.0)
    size = shift + rng.normal(0.0, 0.3, n)
    frame = pd.DataFrame({
        "plateID": ["p1"] * n, "rowID": ["r1"] * n, "columnID": ["c1"] * n,
        "fieldID": ["f1"] * n, "object_label": np.arange(n),
        "area": 800.0 + 60.0 * size,
        "perimeter": 110.0 + 7.0 * size + rng.normal(0.0, 0.5, n),
        "intensity": 5000.0 + 300.0 * rng.normal(0.0, 1.0, n),
    })
    result = pca(frame, PCASpec(features=("area", "perimeter", "intensity")))
    canvas = PCAScoresCanvas()
    qtbot.addWidget(canvas)
    cases = [
        ("no_result_yet", "PC2", False, None),
        ("result_backs_pc1_pc2", "PC2", True, (0, 1)),
        ("pc3_exists", "PC3", True, (0, 2)),
        ("pc9_does_not_exist", "PC9", True, None),
    ]
    for case_id, y, with_result, want in cases:
        with subtests.test(msg=case_id):
            if with_result and canvas.plane() is None and case_id == "result_backs_pc1_pc2":
                canvas.set_spec(GraphSpec(x="PC1", y=y, kind=SCATTER))
                canvas.set_result(result, result.scores_frame(frame))
            else:
                canvas.set_spec(GraphSpec(x="PC1", y=y, kind=SCATTER))
            assert canvas.plane() == want
            if want is None and with_result:
                assert canvas.arrow_scale == 0.0
