"""How five linked views answer a shared selection or filter, early or late.

A selection made in one view (``spacr.selection.Selection``) and the shared
filter (``spacr.selection.DataFilter``) are pushed into every other open view.
Each view has to do the right thing when the push arrives before it has
anything to draw (ignore it, schedule nothing, raise nothing) and when it
arrives after (restyle or redraw, state the narrowed population, return the
right rows).  The views driven here are the graph builder canvas, the trellis
canvas, the image scatter screen, the PCA screen and the database browser,
each with a private ``LinkedSelection`` so no other open view is touched.
Offscreen Qt, synchronous reads, tiny synthetic tables.
"""
from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("matplotlib")
pytest.importorskip("PIL")

from PIL import Image                                            # noqa: E402

from spacr.qt.linked_selection import LinkedSelection            # noqa: E402
from spacr.selection import (CategoryFilter, DataFilter,         # noqa: E402
                             Selection, as_key_index)

pytestmark = pytest.mark.qt


def _keyed(rows, **columns):
    """``rows`` objects keyed plate/row/column/field/label, plus ``columns``."""
    frame = pd.DataFrame({
        "plateID": [f"p{i % 2 + 1}" for i in range(rows)],
        "rowID": [f"r{i % 3 + 1}" for i in range(rows)],
        "columnID": [f"c{i % 4 + 1}" for i in range(rows)],
        "fieldID": [f"f{i % 2 + 1}" for i in range(rows)],
        "object_label": list(range(rows)),
    })
    for name, values in columns.items():
        frame[name] = values
    return frame


# ---------------------------------------------------------------------------
# graph builder canvas
# ---------------------------------------------------------------------------

def test_graph_canvas_restyles_or_redraws_for_a_linked_selection(
        qtbot, monkeypatch, subtests):
    """No chart ignores it, an aggregate redraws, a scatter restyles its overlay."""
    from spacr.qt.widgets import graph_builder as gb
    from spacr.qt.widgets.graph_spec import BAR, SCATTER, GraphSpec

    rng = np.random.default_rng(5)
    frame = _keyed(60, area=rng.normal(100, 10, 60),
                   intensity=rng.normal(size=60),
                   gene=[f"g{i % 3}" for i in range(60)])

    with subtests.test(msg="a selection before the first draw is ignored"):
        empty = gb.GraphCanvas(link=LinkedSelection(), source="early")
        qtbot.addWidget(empty)
        empty.on_linked_selection_changed(
            Selection(keys=as_key_index(["p1_r1_c1_f1_0"]), source="elsewhere"))
        assert empty._selected_mask is None

    cases = [
        ("an aggregate chart redraws rather than restyling",
         GraphSpec(x="gene", y="area", kind=BAR), False, [True], None),
        ("a scatter restyles its marks without a redraw",
         GraphSpec(x="area", y="intensity", kind=SCATTER), True, [], 3),
    ]
    for case_id, spec, live, redraws, selected in cases:
        with subtests.test(msg=case_id):
            canvas = gb.GraphCanvas(link=LinkedSelection(), source="graph")
            qtbot.addWidget(canvas)
            canvas.set_frame(frame)
            canvas.set_spec(spec, immediate=True)
            assert canvas._live_highlight is live
            drawn = []
            monkeypatch.setattr(canvas, "render_now",
                                lambda: drawn.append(True))
            canvas.link.set_selection(
                Selection.from_frame(frame.head(3), source="elsewhere"))
            assert drawn == redraws
            if selected is not None:
                assert int(canvas._selected_mask.sum()) == selected


# ---------------------------------------------------------------------------
# trellis canvas
# ---------------------------------------------------------------------------

def test_trellis_canvas_brushes_one_panel_and_ignores_early_selections(
        qtbot, subtests):
    """A brush returns that panel's keys only; a selection before a chart changes nothing."""
    from spacr.qt.widgets.graph_spec import CONTINUOUS, GraphSpec
    from spacr.qt.widgets.trellis_spec import TrellisSpec
    from spacr.qt.widgets.trellis_view import TrellisCanvas

    layout = (("p1", "a"), ("p1", "a"), ("p1", "b"),
              ("p2", "a"), ("p2", "a"), ("p2", "a"))
    keyed = pd.DataFrame([
        {"plateID": plate, "rowID": "r1", "columnID": "c1", "fieldID": "f1",
         "object_label": index + 1, "gene": gene,
         "area": 10.0 * index + 10.0, "intensity": 100.0 - 5.0 * index}
        for index, (plate, gene) in enumerate(layout)])
    scatter = TrellisSpec(graph=GraphSpec(
        x="area", y="intensity", facet_row="plateID", facet_col="gene",
        roles={"area": CONTINUOUS, "intensity": CONTINUOUS}))

    def canvas_with(spec, link=None):
        widget = TrellisCanvas(
            link=link if link is not None else LinkedSelection())
        qtbot.addWidget(widget)
        widget.set_frame(keyed)
        widget.set_trellis_spec(spec)
        return widget

    cases = [((0, 0), [0, 1]), ((0, 1), [2]), ((1, 0), [3, 4, 5])]
    for (row, col), members in cases:
        with subtests.test(msg=f"brushing panel ({row}, {col})"):
            canvas = canvas_with(scatter)
            picked = canvas.brush(-1e9, -1e9, 1e9, 1e9, row=row, col=col,
                                  publish=False)
            expected = Selection.from_frame(keyed.iloc[members])
            assert sorted(picked.keys) == sorted(expected.keys)
            assert picked.source == canvas.link_source

    with subtests.test(msg="a selection arriving before a chart is ignored"):
        link = LinkedSelection()
        canvas = canvas_with(TrellisSpec(), link)
        assert canvas.trellis is None
        link.set_selection(Selection.from_frame(keyed.iloc[:1], source="other"))
        assert canvas.trellis is None
        assert canvas.panel_axes() == {}


# ---------------------------------------------------------------------------
# image scatter screen
# ---------------------------------------------------------------------------

def test_image_scatter_states_a_narrowed_population(qtbot, tmp_path, subtests):
    """A filter re-plots and is stated once a table is plotted, and is silent before."""
    from spacr.qt.crop_thumbs import CropThumbnails
    from spacr.qt.screens import image_scatter as isc

    crops = tmp_path / "crops"
    crops.mkdir()
    rows = []
    for index in range(4):
        path = crops / f"object{index}.png"
        Image.fromarray(np.full((16, 16, 3), 30 * (index + 1),
                                dtype=np.uint8)).save(path)
        rows.append({"plateID": "plate1", "rowID": "r1", "columnID": "c1",
                     "fieldID": "f1", "object_label": index + 1,
                     "cell_area": 100.0 + index,
                     "cell_intensity": 5.0 - index, "png_path": str(path)})
    frame = pd.DataFrame(rows)
    db_path = tmp_path / "measurements.db"
    con = sqlite3.connect(db_path)
    try:
        frame.drop(columns=["png_path"]).to_sql("cell", con, index=False)
        frame[["plateID", "rowID", "columnID", "fieldID", "object_label",
               "png_path"]].rename(columns={"object_label": "cell_id"}).to_sql(
            "png_list", con, index=False)
    finally:
        con.close()
    keys = [f"plate1_r1_c1_f1_cell{i + 1}" for i in range(4)]
    narrowed = DataFilter([CategoryFilter("rowID", ("r1",))])

    with subtests.test(msg="a filter with nothing plotted changes nothing"):
        idle = isc.ImageScatterScreen(threaded=False)
        qtbot.addWidget(idle)
        idle.on_linked_filter_changed(narrowed)
        assert (idle.status.text(), len(idle.canvas)) == ("", 0)

    with subtests.test(msg="a narrowed population is stated and re-plotted"):
        screen = isc.ImageScatterScreen(threaded=False)
        qtbot.addWidget(screen)
        screen.resize(420, 320)
        screen.canvas.resize(300, 260)
        screen._db.setText(str(db_path))
        screen._thumbs = CropThumbnails(str(db_path))
        screen.set_frame(frame.drop(columns=["png_path"]), keys=keys,
                         paths=dict(zip(keys, frame["png_path"])),
                         x="cell_area", y="cell_intensity")
        screen.on_linked_filter_changed(narrowed)
        assert screen.status.text() == f"Filter: {narrowed.describe()}"
        assert len(screen.canvas) == 4
        assert [screen.key_at(i) for i in range(4)] == keys


# ---------------------------------------------------------------------------
# PCA screen
# ---------------------------------------------------------------------------

def test_pca_screen_filters_only_a_loaded_table(qtbot, caplog, subtests):
    """No table schedules nothing; a filter naming a missing column is logged, not applied."""
    from spacr.qt.screens import pca as pca_screen

    rng = np.random.default_rng(7)
    shift = np.repeat([0.0, 1.0], 30)
    features = pd.DataFrame({
        "plateID": ["p1"] * 60, "rowID": ["r1"] * 60,
        "columnID": [f"c{i % 4 + 1}" for i in range(60)],
        "fieldID": ["f1"] * 60, "object_label": list(range(60)),
        "f1": 10.0 + 4.0 * shift + rng.normal(scale=0.4, size=60),
        "f2": 20.0 - 3.0 * shift + rng.normal(scale=0.4, size=60),
        "f3": 5.0 + 2.0 * shift + rng.normal(scale=0.4, size=60),
        "f4": rng.normal(scale=1.0, size=60),
        "gene": ["control"] * 30 + ["treated"] * 30,
    })

    def screen_on(link):
        made = pca_screen.PCAScreen(link=link, threaded=False)
        qtbot.addWidget(made)
        return made

    with subtests.test(msg="a filter change with no table starts no timer"):
        link = LinkedSelection()
        screen = screen_on(link)
        link.set_filter(DataFilter([CategoryFilter("gene", ("control",))]))
        assert not screen._refilter.isActive()
        assert screen._filtered() is None

    with subtests.test(msg="a filter naming a missing column is reported"):
        link = LinkedSelection()
        screen = screen_on(link)
        screen.set_frame(features)
        link.set_filter(DataFilter([CategoryFilter("no_such_column", ("x",))]))
        caplog.clear()
        with caplog.at_level("INFO", logger="spacr.qt.screens.pca"):
            got = screen._filtered()
        assert len(got) == len(features)
        assert "the shared filter does not apply to this table" in caplog.text


# ---------------------------------------------------------------------------
# database browser
# ---------------------------------------------------------------------------

def test_db_browser_ignores_a_selection_before_a_table_is_loaded(qtbot):
    """A link push that arrives before anything is open selects nothing and raises nothing."""
    from spacr.qt.screens.db_browser import DbBrowserScreen

    screen = DbBrowserScreen(threaded=False)
    qtbot.addWidget(screen)

    assert screen.on_linked_selection_changed(
        Selection(keys=as_key_index(["plate1_r1_c1_f1_1"]))) is None
    assert screen.selected_rows() == []
