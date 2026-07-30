from __future__ import annotations

import sqlite3

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("PySide6")
pytest.importorskip("matplotlib")

from spacr.qt.widgets.umap_explorer import ImageUmapExplorer


def _payload(tmp_path):
    database = tmp_path / "measurements.db"
    image_paths = []
    for index, color in enumerate(((255, 0, 0), (0, 255, 0),
                                   (0, 0, 255), (255, 255, 0))):
        path = tmp_path / f"object_{index}.png"
        Image.new("RGB", (24, 24), color).save(path)
        image_paths.append(path)
    with sqlite3.connect(database) as connection:
        connection.execute(
            "CREATE TABLE png_list "
            "(png_path TEXT PRIMARY KEY, plateID TEXT)")
        connection.executemany(
            "INSERT INTO png_list VALUES (?, 'plate1')",
            [(str(path),) for path in image_paths],
        )
    embedding = np.array([
        [0.0, 0.0], [0.2, 0.1], [2.0, 2.0], [2.2, 2.1]])
    return {
        "embedding": embedding,
        "labels": np.array([0, 0, 1, 1]),
        "records": [
            {
                "image": path,
                "display_name": path.name,
                "db_path": database,
                "db_png_path": str(path),
            }
            for path in image_paths
        ],
    }, database


def _db_values(database, column="umap_annotation"):
    with sqlite3.connect(database) as connection:
        return connection.execute(
            f'SELECT "{column}" FROM png_list ORDER BY png_path'
        ).fetchall()


def test_click_preview_and_lasso_annotation(qtbot, tmp_path):
    payload, database = _payload(tmp_path)
    explorer = ImageUmapExplorer()
    qtbot.addWidget(explorer)
    explorer.set_payload(payload)

    explorer.show_point(0)
    assert not explorer._preview.pixmap().isNull()
    assert "cluster 0" in explorer._point_label.text()

    explorer._on_lasso([
        (-0.2, -0.2), (0.5, -0.2), (0.5, 0.5), (-0.2, 0.5)])
    assert explorer._selected.tolist() == [0, 1]
    explorer._value.setValue(8)
    with qtbot.waitSignal(explorer.annotation_finished, timeout=3000) as signal:
        explorer._write_selected()

    assert signal.args == [2, 0]
    assert _db_values(database) == [(8,), (8,), (None,), (None,)]


def test_cluster_selection_and_propagation(qtbot, tmp_path):
    payload, database = _payload(tmp_path)
    explorer = ImageUmapExplorer()
    qtbot.addWidget(explorer)
    explorer.set_payload(payload)

    explorer._cluster_box.setCurrentIndex(2)
    assert explorer._selected.tolist() == [2, 3]
    with qtbot.waitSignal(explorer.annotation_finished, timeout=3000) as signal:
        explorer._write_clusters()

    assert signal.args == [4, 0]
    assert _db_values(database) == [(0,), (0,), (1,), (1,)]


def test_rejects_misaligned_payload(qtbot):
    explorer = ImageUmapExplorer()
    qtbot.addWidget(explorer)

    with pytest.raises(ValueError, match="equal lengths"):
        explorer.set_payload({
            "embedding": [[0, 0], [1, 1]],
            "labels": [0],
            "records": [{}, {}],
        })


def test_embedding_matches_container_theme_and_uses_viridis(
        qtbot, tmp_path, monkeypatch):
    from matplotlib.colors import to_rgba
    from spacr.qt import theme

    palette = dict(theme.DARK_PALETTE)
    palette.update({
        "surface_alt": "#24272b",
        "fg": "#ffffff",
    })
    monkeypatch.setattr(theme, "active_palette", lambda: palette)
    payload, _database = _payload(tmp_path)
    explorer = ImageUmapExplorer()
    qtbot.addWidget(explorer)

    explorer.set_payload(payload)

    assert explorer._figure.get_facecolor() == to_rgba("#24272b")
    assert explorer._axes.get_facecolor() == to_rgba("#24272b")
    assert explorer._axes.spines["left"].get_edgecolor() == to_rgba(
        "#ffffff")
    assert explorer._scatter.get_cmap().name == "viridis"


def test_display_settings_control_points_lines_and_splitter(qtbot, tmp_path):
    from matplotlib.colors import to_rgba

    payload, _database = _payload(tmp_path)
    payload["display"] = {
        "point_size": 73,
        "point_color": "#4cc9f0",
        "point_alpha": 0.4,
        "outline_width": 0.6,
        "canvas_width": 760,
        "sidebar_width": 340,
    }
    explorer = ImageUmapExplorer()
    qtbot.addWidget(explorer)
    explorer.resize(1200, 700)
    explorer.show()
    explorer.set_payload(payload)
    qtbot.wait(1)

    assert np.all(explorer._scatter.get_sizes() == 73)
    assert explorer._scatter.get_alpha() == pytest.approx(0.4)
    assert explorer._scatter.get_facecolors()[0] == pytest.approx(
        to_rgba("#4cc9f0", alpha=0.4))
    assert explorer._selection_artist.get_linewidths()[0] == pytest.approx(0.6)
    assert explorer._point_size.value() == 73
    assert explorer._point_color.text() == "#4cc9f0"
    sizes = explorer._body_splitter.sizes()
    assert sizes[0] > sizes[1]


def test_display_controls_update_live_and_scroll_zooms(qtbot, tmp_path):
    payload, _database = _payload(tmp_path)
    explorer = ImageUmapExplorer()
    qtbot.addWidget(explorer)
    explorer.set_payload(payload)

    explorer._point_size.setValue(41)
    explorer._point_color.setText("orange")
    explorer._point_color.editingFinished.emit()
    assert np.all(explorer._scatter.get_sizes() == 41)

    before = np.ptp(explorer._axes.get_xlim())
    event = type("_Scroll", (), {
        "inaxes": explorer._axes,
        "xdata": 1.0,
        "ydata": 1.0,
        "button": "up",
    })()
    explorer._on_scroll(event)
    assert np.ptp(explorer._axes.get_xlim()) < before


def test_close_cancels_a_pending_canvas_draw(qtbot, tmp_path):
    payload, _database = _payload(tmp_path)
    explorer = ImageUmapExplorer()
    qtbot.addWidget(explorer)
    explorer.set_payload(payload)
    explorer._canvas._draw_pending = True

    explorer.close()

    assert explorer._canvas._draw_pending is False
    assert explorer._lasso is None
