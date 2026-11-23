"""Native UMAP style observes real Qt painting without recomputing a projection."""
import numpy as np
import pytest
from spacr.figures import style
from spacr.qt.widgets.umap_search_viewer import UmapEmbeddingView, UmapExplorer, project_points

pytestmark = pytest.mark.qt


@pytest.mark.parametrize('dimensions', [2, 3])
def test_native_embedding_observes_actual_paint_and_retains_coordinates_and_camera(qtbot, dimensions):
    """Keep the original camera projection exactly; observe physical fonts, brushes and bounds, and report unvalidated 3D occlusion rather than certifying it."""
    coords = np.random.default_rng(171).normal(size=(48, dimensions))
    labels = np.repeat([-1, 0, 1], 16)
    view = UmapEmbeddingView(); qtbot.addWidget(view); view.resize(440, 440)
    view.set_embedding(coords, labels=labels, caption='Fixed embedding', backend='fixture')
    original = view.coordinates.tobytes()
    camera = (view._yaw, view._pitch, view._zoom)
    expected, depth = project_points(view._coords, view.width(), view.height(),
        yaw=view._yaw, pitch=view._pitch if dimensions == 3 else 0., zoom=view._zoom)
    observed = view._spacr_style_observation()
    assert observed['backend'] == 'QPainter'
    np.testing.assert_array_equal(observed['projected_points'], expected)
    np.testing.assert_array_equal(observed['depth'], depth)
    assert observed['text_bounds_checked']
    assert not observed['text_geometry']['outside']
    assert observed['background'] == [1., 1., 1., 1.]
    assert all(font['family'] == 'Open Sans' and font['size'] >= 8 and font['weight'] == 300 for font in observed['fonts'])
    assert all(abs(colour[3]-.6) < 1e-6 for mark in observed['marks'] for colour in mark['colours'])
    assert all(edge['visible'] for edge in observed['frame'])
    assert any('fixture' in entry['label'] for entry in view._spacr_legend_entries)
    report = style.finalize_figure(view)
    if dimensions == 3:
        assert not report.compliant
        assert any('three-dimensional' in problem for problem in report.problems)
        assert 'completed presentation checks' not in report.checks
    assert view.coordinates.tobytes() == original
    assert (view._yaw, view._pitch, view._zoom) == camera


def test_native_manual_appearance_remains_exact_and_is_diagnosed(qtbot):
    """An explicit .35 viridis diamond remains manual appearance; finalization reports its palette and alpha without repainting it into the default."""
    view = UmapEmbeddingView(); qtbot.addWidget(view); view.resize(440,440)
    view.set_embedding(np.random.default_rng(171).normal(size=(24,2)), labels=np.arange(24)%3)
    values = {'marker':'diamond', 'size':7.5, 'alpha':.35, 'cmap':'viridis'}
    view.set_appearance(values)
    source = view.coordinates.tobytes()
    report = style.finalize_figure(view)
    assert any('sixty percent' in problem for problem in report.problems)
    assert any('shared data palette' in problem for problem in report.problems)
    assert view.appearance == values
    assert view.coordinates.tobytes() == source


def test_native_shell_shows_caption_and_camera_help_outside_the_plot(qtbot):
    """Expose backend and camera instructions in a separate label rather than painting text over the embedding."""
    shell = UmapExplorer(); qtbot.addWidget(shell)
    shell.view.set_embedding(np.random.default_rng(171).normal(size=(12,3)), caption='Fixed embedding', backend='fixture')
    assert shell.legend.parent() is shell
    assert 'Fixed embedding' in shell.legend.text() and 'fixture' in shell.legend.text()
    assert 'drag to spin' in shell.legend.text()
    assert shell.legend.text() == shell.view.toolTip()


def test_clearing_a_native_map_clears_the_external_partition_and_paint_observation(qtbot):
    """A cancelled or empty search must not retain the preceding partition legend or certify its old painted points."""
    shell=UmapExplorer();qtbot.addWidget(shell)
    shell.view.set_embedding(np.random.default_rng(171).normal(size=(12,2)), labels=np.zeros(12,dtype=int), caption='Previous map')
    assert shell.view._spacr_style_observation()['marks']
    shell.view.clear('Search cancelled.')
    assert 'Search cancelled.' in shell.legend.text()
    assert 'Previous map' not in shell.legend.text() and 'Cluster 0' not in shell.legend.text()
    assert shell.view._spacr_style_observation()['specification'] is None
    assert not style.finalize_figure(shell.view).compliant
