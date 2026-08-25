"""Enlarging the Image UMAP container enlarges the PLOT, in both directions.

A UMAP of a few thousand crops is unreadable at panel size, and the only
alternative to enlarging it is shrinking the thumbnails -- which is the
opposite of what makes it readable. So the container has to be draggable, and
what the drag has to reach is the coordinate space rather than the frame
around it.

What these tests hold is the second half of that: a resize must not re-embed.
A resize that moved the points would lose the reader's place, and the whole
value of enlarging a projection is that the neighbourhoods stay where they
were while there is more room to tell them apart.
"""

import sqlite3

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("PySide6")
pytest.importorskip("matplotlib")

from PySide6.QtCore import Qt

from spacr.qt.widgets.umap_explorer import ImageUmapExplorer


@pytest.fixture
def payload(tmp_path):
    database = tmp_path / "measurements.db"
    paths = []
    for index, colour in enumerate(((255, 0, 0), (0, 255, 0),
                                    (0, 0, 255), (255, 255, 0))):
        path = tmp_path / f"object_{index}.png"
        Image.new("RGB", (24, 24), colour).save(path)
        paths.append(path)
    with sqlite3.connect(database) as connection:
        connection.execute(
            "CREATE TABLE png_list (png_path TEXT PRIMARY KEY, plateID TEXT)")
        connection.executemany(
            "INSERT INTO png_list VALUES (?, 'plate1')",
            [(str(path),) for path in paths])
    return {
        "embedding": np.array([[0.0, 0.0], [0.2, 0.1],
                               [2.0, 2.0], [2.2, 2.1]]),
        "labels": np.array([0, 0, 1, 1]),
        "records": [{"image": path, "display_name": path.name,
                     "db_path": database, "db_png_path": str(path)}
                    for path in paths],
    }


@pytest.fixture
def explorer(qt_theme_applied, qtbot, payload):
    widget = ImageUmapExplorer()
    qtbot.addWidget(widget)
    # WIDE ENOUGH THAT THE DIVIDER CAN MOVE. The sidebar carries a width
    # floor of its own (the image preview), and while the splitter is narrow
    # enough to hold it at that floor the handle has nowhere to travel -- so
    # a narrower fixture would be testing the floor rather than the drag.
    widget.resize(1400, 420)
    widget.show()
    widget.set_payload(payload)
    qt_theme_applied.processEvents()
    return widget


def _plot_inches(explorer):
    width, height = explorer._figure.get_size_inches()
    return float(width), float(height)


class TestTheDragReachesThePlot:

    def test_a_taller_container_gives_a_taller_plot(self, explorer,
                                                    qt_theme_applied):
        """Enlarging the figure container may mean vertically, and a
        horizontal divider on its own cannot do that -- so the plot has to
        follow its container's HEIGHT as well as its width."""
        before = _plot_inches(explorer)[1]
        explorer.resize(explorer.width(), explorer.height() + 400)
        qt_theme_applied.processEvents()
        assert _plot_inches(explorer)[1] > before

    def test_a_wider_container_gives_a_wider_plot(self, explorer,
                                                  qt_theme_applied):
        before = _plot_inches(explorer)[0]
        explorer.resize(explorer.width() + 400, explorer.height())
        qt_theme_applied.processEvents()
        assert _plot_inches(explorer)[0] > before

    def test_moving_the_divider_moves_the_plot_and_not_only_its_frame(
            self, explorer, qt_theme_applied):
        """A canvas with a fixed size policy would let the splitter move the
        frame while the axes stayed the size they were."""
        splitter = explorer._body_splitter
        chart, sidebar = splitter.sizes()
        before = _plot_inches(explorer)[0]
        floor = splitter.widget(1).minimumSizeHint().width()
        step = sidebar - floor - 10
        assert step > 0, "the fixture is too narrow for the divider to move"
        splitter.setSizes([chart + step, sidebar - step])
        qt_theme_applied.processEvents()
        assert _plot_inches(explorer)[0] > before


class TestTheEmbeddingSurvivesTheDrag:

    def test_no_point_moves_when_the_container_grows(self, explorer,
                                                     qt_theme_applied,
                                                     payload):
        """A resize that re-embedded would be a resize that lost the
        reader's place -- the spread grows, the neighbourhoods do not
        change."""
        before = explorer._scatter.get_offsets().copy()
        explorer.resize(explorer.width() + 300, explorer.height() + 300)
        qt_theme_applied.processEvents()
        after = explorer._scatter.get_offsets()
        assert np.allclose(np.asarray(before), np.asarray(after))
        assert np.allclose(np.asarray(after), payload["embedding"])


class TestTheDividerIsARealHandle:

    def test_it_is_a_horizontal_splitter_between_chart_and_sidebar(
            self, explorer):
        splitter = explorer._body_splitter
        assert splitter.orientation() == Qt.Horizontal
        assert splitter.count() == 2

    def test_neither_side_can_be_collapsed_out_of_existence(self, explorer):
        """A divider dragged to the edge that then cannot be dragged back is
        a divider that has eaten the sidebar."""
        assert not explorer._body_splitter.childrenCollapsible()

    def test_the_handle_says_it_can_be_dragged(self, explorer):
        """The cursor over it is the only thing that says so before the
        first drag."""
        handle = explorer._body_splitter.handle(1)
        assert handle is not None
        assert handle.cursor().shape() == Qt.SplitHCursor
