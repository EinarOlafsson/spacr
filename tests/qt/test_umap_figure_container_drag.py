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
    # floor of its own -- the widest of its buttons -- and while the splitter
    # is narrow enough to hold it at that floor the handle has nowhere to
    # travel towards the sidebar, so a narrower fixture would be testing the
    # floor rather than the drag.
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
        """The cursor over it, and the hover text, are what say so before the
        first drag. The cursor alone only helps a reader who is already on
        the handle, which on a hairline is the hard part."""
        handle = explorer._body_splitter.handle(1)
        assert handle is not None
        assert handle.cursor().shape() == Qt.SplitHCursor
        assert "Drag" in handle.toolTip()

    def test_the_handle_can_be_hit_without_becoming_a_bar(self, explorer):
        """A hairline is the house style for every divider in the app; a
        hairline GRAB AREA is a divider the reader has to hunt for. The
        handle keeps the painted line and widens the target around it."""
        from spacr.qt.theme import SPACING, active_palette

        handle = explorer._body_splitter.handle(1)
        assert handle.width() >= SPACING["sm"]

        strip = handle.grab().toImage()
        row = strip.height() // 2
        line = active_palette()["border_soft"].lower()
        painted = [x for x in range(strip.width())
                   if strip.pixelColor(x, row).name().lower() == line]
        assert len(painted) <= 2, (
            "the handle paints a slab; the widened grab area is supposed to "
            "stay transparent around a one-pixel line")


class TestThePreviewShowsTheWholeCrop:

    def test_a_crop_wider_than_the_sidebar_is_scaled_not_clipped(
            self, explorer, qt_theme_applied, tmp_path):
        """A ``QLabel`` clips a pixmap wider than itself and says nothing.

        The preview exists to be looked at, and a crop with its edges cut off
        looks exactly like a crop that ends there -- so the clipping is
        invisible in the one place it matters. The object under study is
        often the thing at the edge.
        """
        big = tmp_path / "big.png"
        Image.new("RGB", (600, 600), (10, 200, 10)).save(big)
        explorer.set_payload({
            "embedding": np.array([[0.0, 0.0], [1.0, 1.0]]),
            "labels": np.array([0, 1]),
            "records": [{"image": big, "display_name": big.name}] * 2,
        })
        qt_theme_applied.processEvents()

        explorer.show_point(0)
        qt_theme_applied.processEvents()

        preview = explorer._preview
        assert preview.source_pixmap().width() > preview.width(), (
            "the fixture is too wide for this to be testing anything")
        assert preview.pixmap().width() <= preview.width()
        assert preview.pixmap().height() <= preview.height()
        assert preview.pixmap().width() > 0

    def test_the_preview_does_not_shrink_itself_over_repeated_resizes(
            self, explorer, qt_theme_applied, tmp_path):
        """A label whose preferred size is the pixmap it is showing walks
        down to nothing: the layout offers the hint, the label rescales to
        it, and the next hint is smaller again."""
        big = tmp_path / "big.png"
        Image.new("RGB", (400, 400), (200, 30, 30)).save(big)
        explorer.set_payload({
            "embedding": np.array([[0.0, 0.0], [1.0, 1.0]]),
            "labels": np.array([0, 1]),
            "records": [{"image": big, "display_name": big.name}] * 2,
        })
        explorer.show_point(0)
        qt_theme_applied.processEvents()
        settled = explorer._preview.width()

        for _ in range(5):
            explorer.resize(explorer.width(), explorer.height())
            qt_theme_applied.processEvents()

        assert explorer._preview.width() == settled
