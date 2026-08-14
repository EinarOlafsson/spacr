"""The Image UMAP coordinate viewer, grid, and row-to-map identity."""
from __future__ import annotations

import numpy as np
import pytest
from PySide6.QtCore import QPoint, Qt

from spacr.hyperparam import SearchResult, Trial
from spacr.qt.screens.hyperparam import HyperparamPanel
from spacr.qt.widgets.umap_search_viewer import (
    BACKGROUND, UmapEmbeddingView, UmapGalleryDialog, axis_frame,
    project_points, thumbnail_image,
)


def _embedding(dimensions=3, shift=0.0):
    rng = np.random.default_rng(7)
    return rng.normal(size=(45, dimensions)) + float(shift)


def _trial(index=0, dimensions=3, shift=0.0, backend="cpu"):
    embedding = _embedding(dimensions, shift)
    return Trial(
        params={"n_neighbors": 5 + index * 10, "min_dist": 0.1},
        score=0.9 - index * 0.1,
        index=index,
        extra_metrics={
            "embedding": embedding,
            "backend": backend,
            "n_components": dimensions,
        },
    )


def test_projection_preserves_aspect_ratio_in_a_rectangular_view():
    square = np.array(((-1.0, -1.0), (-1.0, 1.0),
                       (1.0, -1.0), (1.0, 1.0)))
    points, _depth = project_points(square, 800, 300)
    assert np.ptp(points[:, 0]) == pytest.approx(np.ptp(points[:, 1]))


@pytest.mark.parametrize("dimensions", [2, 3])
def test_axis_frame_has_one_labelled_axis_per_embedding_dimension(dimensions):
    frame = axis_frame(_embedding(dimensions), 640, 480,
                       yaw=0.2, pitch=-0.1)
    assert frame["dimensions"] == dimensions
    assert len(frame["axes"]) == dimensions
    assert [axis[2] for axis in frame["axes"]] == [
        f"Dimension {index}" for index in range(1, dimensions + 1)]


def test_thumbnail_has_the_required_black_background(qapp):
    image = thumbnail_image(_embedding(3), size=120)
    corner = image.pixelColor(0, 0)
    assert corner.red() == BACKGROUND.red()
    assert corner.green() == BACKGROUND.green()
    assert corner.blue() == BACKGROUND.blue()


@pytest.mark.parametrize("dimensions", [2, 3])
def test_viewer_holds_both_supported_dimensions(qtbot, dimensions):
    view = UmapEmbeddingView()
    qtbot.addWidget(view)
    view.set_embedding(_embedding(dimensions), backend="cuml")
    assert view.dimensions == dimensions
    assert view.coordinates.shape == (45, 3)


def test_viewer_refuses_a_cluster_label_count_from_another_map(qtbot):
    view = UmapEmbeddingView()
    qtbot.addWidget(view)
    with pytest.raises(ValueError, match="one value per UMAP point"):
        view.set_embedding(_embedding(2), labels=np.zeros(4))


def test_appearance_changes_rendering_without_refitting_coordinates(qtbot):
    view = UmapEmbeddingView()
    qtbot.addWidget(view)
    view.set_embedding(_embedding(2))
    before = view.coordinates
    view.set_appearance({
        "marker": "diamond", "size": 7.5, "alpha": 0.45,
        "cmap": "viridis",
    })
    assert view.appearance == {
        "marker": "diamond", "size": 7.5, "alpha": 0.45,
        "cmap": "viridis",
    }
    assert np.array_equal(view.coordinates, before)


def test_dragging_a_3d_view_changes_its_camera(qtbot):
    view = UmapEmbeddingView()
    qtbot.addWidget(view)
    view.resize(500, 400)
    view.show()
    view.set_embedding(_embedding(3))
    before = (view._yaw, view._pitch)
    qtbot.mousePress(view, Qt.LeftButton, pos=view.rect().center())
    qtbot.mouseMove(view, pos=view.rect().center() + QPoint(45, 30))
    qtbot.mouseRelease(view, Qt.LeftButton)
    assert (view._yaw, view._pitch) != before


def test_gallery_click_returns_the_exact_trial_not_a_rebuilt_map(qtbot):
    trials = [_trial(0, shift=0.0), _trial(1, shift=12.0)]
    dialog = UmapGalleryDialog(trials)
    qtbot.addWidget(dialog)
    chosen = []
    dialog.trial_chosen.connect(chosen.append)
    item = dialog.list.item(1)
    dialog._choose(item)
    assert chosen == [trials[1]]
    assert chosen[0].extra_metrics["embedding"] is \
        trials[1].extra_metrics["embedding"]


def test_panel_row_click_loads_that_rows_stored_coordinates(
        qtbot, qt_theme_applied):
    panel = HyperparamPanel("umap")
    qtbot.addWidget(panel)
    first, second = _trial(0, shift=0.0), _trial(1, shift=20.0)
    result = SearchResult(
        trials=[first, second], best=first, metric="trustworthiness")
    panel._result = result
    panel._rebuild_table(result)

    panel._table.selectRow(1)
    assert panel._displayed_trial is second
    assert np.array_equal(
        panel._umap_explorer.view.coordinates[:, :3],
        second.extra_metrics["embedding"],
    )


def test_panel_grid_click_loads_the_chosen_row(qtbot, qt_theme_applied):
    panel = HyperparamPanel("umap")
    qtbot.addWidget(panel)
    first, second = _trial(0), _trial(1, backend="cuml")
    result = SearchResult(
        trials=[first, second], best=first, metric="trustworthiness")
    panel._result = result
    panel._rebuild_table(result)
    dialog = panel.open_umap_grid()
    qtbot.addWidget(dialog)

    dialog._choose(dialog.list.item(1))
    assert panel._displayed_trial is second
    assert panel._table.selectionModel().selectedRows()[0].row() == 1


def test_cluster_this_map_updates_only_the_selected_trial(
        qtbot, qt_theme_applied):
    panel = HyperparamPanel("umap")
    qtbot.addWidget(panel)
    rng = np.random.default_rng(11)
    coords = np.vstack((rng.normal(-4, 0.1, (30, 2)),
                        rng.normal(4, 0.1, (30, 2))))
    first, second = _trial(0, dimensions=2), _trial(1, dimensions=2)
    second.extra_metrics["embedding"] = coords
    result = SearchResult(
        trials=[first, second], best=first, metric="trustworthiness")
    panel._result = result
    panel._rebuild_table(result)
    panel._table.selectRow(1)
    panel._cluster_size.setValue(8)

    assert panel.cluster_selected()
    assert "cluster_labels" not in first.extra_metrics
    assert second.extra_metrics["cluster_labels"].shape == (60,)
    assert second.extra_metrics["n_clusters"] == 2


def test_search_request_carries_gpu_dimensions_and_cluster_walk(
        qtbot, qt_theme_applied):
    panel = HyperparamPanel("umap")
    qtbot.addWidget(panel)
    panel._value_edits["n_neighbors"].setText("5")
    panel._value_edits["min_dist"].setText("")
    panel._value_edits["metric"].setText("")
    panel._set_gpu_checked(True)
    panel._dimensions.setCurrentText("3D")
    panel._cluster_during.setChecked(True)
    captured = []

    def search(request, _on_trial, _should_stop):
        captured.append(request)
        return SearchResult(space=request.space, metric=request.criterion)

    panel.set_search_fn(search)
    with qtbot.waitSignal(panel.search_finished, timeout=5000):
        assert panel.run_search()
    request = captured[0]
    assert request.umap_backend == "cuml"
    assert request.umap_components == 3
    assert request.cluster_during_search is True
    assert request.cluster_sizes == panel.cluster_walk_sizes()


def test_table_names_backend_and_cluster_count_per_row(
        qtbot, qt_theme_applied):
    panel = HyperparamPanel("umap")
    qtbot.addWidget(panel)
    trial = _trial(0, backend="cuml")
    trial.extra_metrics["n_clusters"] = 4
    result = SearchResult(
        trials=[trial], best=trial, metric="trustworthiness")
    panel._result = result
    panel._rebuild_table(result)
    assert panel._table.item(
        0, panel.COLUMNS.index("backend")).text() == "cuml"
    assert panel._table.item(
        0, panel.COLUMNS.index("clusters")).text() == "4"


def test_umap_parameter_columns_sort_numerically(qtbot, qt_theme_applied):
    panel = HyperparamPanel("umap")
    qtbot.addWidget(panel)
    first, second = _trial(0), _trial(1)
    first.params.update(n_neighbors=100, min_dist=0.5)
    second.params.update(n_neighbors=9, min_dist=0.05)
    panel._rebuild_table(SearchResult(
        trials=[first, second], best=first, metric="trustworthiness"))
    panel._table.sortItems(
        panel.COLUMNS.index("neighbors"), Qt.AscendingOrder)
    assert [panel._table.item(row, panel.COLUMNS.index("neighbors")).text()
            for row in range(2)] == ["9", "100"]
    panel._table.sortItems(
        panel.COLUMNS.index("min dist"), Qt.DescendingOrder)
    assert [panel._table.item(row, panel.COLUMNS.index("min dist")).text()
            for row in range(2)] == ["0.5", "0.05"]


def test_umap_hides_the_old_max_panels_figure_control(
        qtbot, qt_theme_applied):
    panel = HyperparamPanel("umap")
    qtbot.addWidget(panel)
    assert panel._plot_panel_controls.isHidden()
    assert panel._figure_grid is None
