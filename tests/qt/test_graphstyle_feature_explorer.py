"""Feature exploration retains original class-bin counts and ranking values under shared presentation."""
import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure
from spacr.figures import style
from spacr.qt import preferences
from spacr.qt.widgets.feature_explorer import FeatureExplorerPanel
from spacr.qt.widgets.feature_rank import ExplorerSpec, distributions, rank_features

pytestmark = pytest.mark.qt


@pytest.mark.parametrize('number', [1, 3])
@pytest.mark.parametrize('manual', [False, True])
def test_feature_distributions_keep_original_histogram_counts_and_use_shared_geometry_labels_and_opacity(qtbot, monkeypatch, number, manual):
    """The baseline used .55-alpha bars, seven-point ticks and wide strips; require actual 40-mm square areas within half a figure pixel, while preserving every original bin count, centre, Matplotlib-converted width, rank score and source value. The original bar primitive retains a one-ULP width conversion; compare its actual widths exactly, with no tolerance change."""
    monkeypatch.setattr(preferences, 'get_figure_style', lambda: {'font_size':12.,'tick_size':12.,'label_size':12.,'legend_size':12.} if manual else {})
    monkeypatch.setattr(preferences, 'get_figure_style_per_graph', dict)
    index = np.arange(16, dtype=float)
    frame = pd.DataFrame({'cls': np.repeat(['a','b'],8)})
    for column in range(number):
        frame['feature_'+str(column)] = index*(column+1)+np.cos(index+column)
    frame.attrs['units'] = {name:'dimensionless' for name in frame.columns if name!='cls'}
    spec = ExplorerSpec(label='cls', bins=8, n_permutations=0)
    result = rank_features(frame,spec)
    source = frame.to_csv(index=False)
    scores = [(score.feature,score.score,score.auc,score.cohen_d,score.ks,score.mutual_info) for score in result.scores]
    widget = FeatureExplorerPanel()
    qtbot.addWidget(widget)
    widget.resize(1000,760)
    widget._frame, widget._spec = frame, spec
    widget._draw(result)
    widget.show()
    qtbot.waitExposed(widget)
    assert len(widget._figure.axes) == number
    for ax, score in zip(widget._figure.axes, result.scores):
        assert ax._spacr_style_spec.kind == 'continuous'
        comparison = dict(ax._spacr_style_spec.comparison)
        assert comparison['graph_type'] == 'histogram'
        assert ax.get_xlabel() == score.feature and ax.get_ylabel() == 'Observation count'
        assert not ax.get_title()
        assert all(spine.get_visible() for spine in ax.spines.values())
        assert all(patch.get_alpha() == .6 for patch in ax.patches)
        assert min(text.get_fontsize() for text in ax.get_xticklabels()+ax.get_yticklabels()) >= (12. if manual else 8.)
        area = ax.get_position()
        inches = widget._figure.get_size_inches()
        assert area.width*inches[0] == pytest.approx(40/25.4, abs=.5/widget._figure.dpi)
        assert area.height*inches[1] == pytest.approx(40/25.4, abs=.5/widget._figure.dpi)
        edges, counts = distributions(frame,score.feature,result.label,bins=spec.bins)
        centres = (edges[:-1]+edges[1:])/2.
        width = float(np.diff(edges).mean())*.9
        assert len(ax.patches) == 8*len(counts)
        original = Figure().add_subplot()
        for values in counts.values():
            original.bar(centres, values, width=width, alpha=.55, linewidth=0.)
        for index, values in enumerate(counts.values()):
            bars = ax.patches[index*8:(index+1)*8]
            np.testing.assert_array_equal([bar.get_height() for bar in bars],values)
            np.testing.assert_array_equal([bar.get_x() for bar in bars],
                [bar.get_x() for bar in original.patches[index*8:(index+1)*8]])
            np.testing.assert_array_equal([bar.get_width() for bar in bars],
                [bar.get_width() for bar in original.patches[index*8:(index+1)*8]])
        units = dict(ax._spacr_style_spec.legend)['units']
        assert dict(units) == {'x':'dimensionless','y':'observations'}
    letters = [text.get_text() for ax in widget._figure.axes for text in ax.texts if text.get_gid()=='spacr-panel-letter']
    assert letters == ([] if number==1 else ['A','B','C'])
    assert frame.to_csv(index=False) == source
    assert [(score.feature,score.score,score.auc,score.cohen_d,score.ks,score.mutual_info) for score in result.scores] == scores


@pytest.mark.parametrize('standalone', [False, True])
@pytest.mark.parametrize('case', ['ordinary', 'nonfinite', 'many', 'manual'])
def test_feature_recipe_replays_original_bins_counts_styles_and_complete_page(qtbot, monkeypatch, standalone, case):
    """Recreate the same original histogram primitives, labels, manual fonts and frozen ranking descriptions. Eight panels reproduce the measured viewport overflow; the captured page must contain them. Matplotlib rebuilds saved rectangles through Bbox.from_bounds, then adjusts the active square box during drawing (measured one ULP); compare the original rectangle through its exact existing conversion and retain the existing half-pixel active physical-geometry gate. The inverse legend transform allows four float64 epsilon, as in the existing control-chart replay gate."""
    import json
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.transforms import Bbox
    from spacr.figures import bundle

    monkeypatch.setattr(preferences, 'get_figure_style', dict)
    monkeypatch.setattr(preferences, 'get_figure_style_per_graph', dict)
    values = np.arange(24, dtype=float)
    frame = pd.DataFrame({'cls':np.repeat(['a','b','c'],8)})
    for index in range(8 if case=='many' else 3):
        frame['measurement_'+str(index)] = values*(index+1)+np.sin(values+index)
    if case=='nonfinite':
        frame.loc[[0,8], 'measurement_0'] = np.nan
        frame.loc[17, 'measurement_1'] = np.inf
    widget = FeatureExplorerPanel()
    qtbot.addWidget(widget)
    widget._frame = frame
    widget._spec = ExplorerSpec(label='cls',bins=8,n_permutations=0)
    result = rank_features(frame,widget._spec)
    widget._draw(result)
    if case=='manual':
        widget._figure.axes[0].xaxis.label.set_fontsize(12.)
        widget._figure.axes[0].patches[0].set_alpha(.42)
        widget._figure.axes[0].patches[0].set_facecolor('#C878A8')
    data, specification = bundle._figure_record(widget._figure)
    before = data.to_csv(index=False)
    with style._file_axes_geometry(widget._figure):
        expected = style._capture_appearance(widget._figure)
    captured = json.loads(json.dumps(bundle._capture_view(widget._figure,specification)))
    assert captured['appearance'] == expected
    assert captured['feature_explorer_recipe']['bins'] == widget._spec.bins
    assert [record['feature'] for record in captured['feature_explorer_recipe']['features']] == [score.feature for score in result.scores[:8]]
    if standalone:
        namespace = {'__name__':'portable_feature_probe'}
        exec(compile(bundle._recreate_script(captured),'recreate_figure.py','exec',dont_inherit=True),namespace)
        draw = namespace['_draw']
    else:
        draw = bundle._draw
    restored = Figure()
    draw(restored,data,captured)
    FigureCanvasAgg(restored).draw()
    assert restored._spacr_replay_problems == ()
    assert list(restored.get_size_inches()) == captured['appearance']['size']
    assert data.to_csv(index=False) == before
    assert len(restored.axes) == len(widget._figure.axes)
    with style._file_axes_geometry(widget._figure):
        for index, (original, replayed) in enumerate(zip(widget._figure.axes,restored.axes,strict=True)):
            assert replayed.get_xlim() == original.get_xlim() and replayed.get_ylim() == original.get_ylim()
            assert replayed.get_xlabel() == original.get_xlabel() and replayed.get_ylabel() == original.get_ylabel()
            assert replayed.get_position(original=True).bounds == Bbox.from_bounds(*captured['appearance']['axes'][index]['rect']).bounds
            np.testing.assert_allclose(np.array(replayed.get_position().bounds)*np.tile(restored.get_size_inches(),2),
                np.array(original.get_position().bounds)*np.tile(widget._figure.get_size_inches(),2),rtol=0,atol=.5/restored.dpi)
            assert replayed.xaxis.label.get_fontsize() == original.xaxis.label.get_fontsize()
            assert len(replayed.patches) == len(original.patches)
            for first, second in zip(original.patches,replayed.patches,strict=True):
                assert (second.get_x(),second.get_y(),second.get_width(),second.get_height()) == (first.get_x(),first.get_y(),first.get_width(),first.get_height())
                assert style._artist_appearance(second) == style._artist_appearance(first)
            original_key = original.get_legend()
            replayed_key = replayed.get_legend()
            assert (original_key is None) == (replayed_key is None)
            if original_key is not None:
                expected_key = style._legend_appearance(original_key)
                actual_key = style._legend_appearance(replayed_key)
                np.testing.assert_allclose(actual_key.pop('anchor'),expected_key.pop('anchor'),rtol=0,atol=4*np.finfo(float).eps)
                assert actual_key == expected_key


def test_feature_archive_entry_point_requires_neither_spacr_nor_qt(qtbot, monkeypatch, tmp_path):
    """Run the real archived script with both spaCR and Qt imports refused; retain exact source CSV bytes and vector-only bars."""
    import os
    import subprocess
    import sys
    import zipfile
    from spacr.figures import bundle

    monkeypatch.setattr(preferences, 'get_figure_style', dict)
    monkeypatch.setattr(preferences, 'get_figure_style_per_graph', dict)
    frame = pd.DataFrame({'cls':['a']*8+['b']*8,'measurement':np.arange(16,dtype=float)+np.sin(np.arange(16))})
    widget = FeatureExplorerPanel()
    qtbot.addWidget(widget)
    widget._frame = frame
    widget._spec = ExplorerSpec(label='cls',bins=8,n_permutations=0)
    widget._draw(rank_features(frame,widget._spec))
    archive = bundle._save_zip(widget._figure,str(tmp_path/'feature.zip'),formats=['svg'])
    folder = tmp_path/'portable'
    with zipfile.ZipFile(archive) as saved:
        assert saved.read('data.csv').decode() == frame.to_csv(index=False)
        saved.extractall(folder)
    guard = tmp_path/'guard'
    guard.mkdir()
    (guard/'sitecustomize.py').write_text(
        "import sys\n"
        "class RejectApplication:\n"
        "    def find_spec(self, fullname, path=None, target=None):\n"
        "        if fullname.split('.')[0] in ('spacr', 'PySide6', 'PyQt5', 'PyQt6'):\n"
        "            raise ImportError('feature replay must not import spaCR or Qt')\n"
        "sys.meta_path.insert(0, RejectApplication())\n")
    process = subprocess.run([sys.executable,'recreate_figure.py','svg','pdf'],cwd=folder,
        env=dict(os.environ,PYTHONPATH=str(guard),MPLBACKEND='Agg'),capture_output=True,text=True,timeout=60)
    assert process.returncode == 0, process.stdout+process.stderr
    vector = (folder/'recreated.svg').read_text()
    assert '<svg' in vector and '<image' not in vector
    assert 'measurement' in vector and 'Observation count' in vector
    assert (folder/'recreated.pdf').read_bytes().startswith(b'%PDF')


@pytest.mark.parametrize('standalone', [False, True])
def test_feature_replay_refuses_changed_class_identities(qtbot, monkeypatch, standalone):
    """A class rename or CSV type conversion must not silently assign a captured colour to a different class."""
    from spacr.figures import bundle

    monkeypatch.setattr(preferences, 'get_figure_style', dict)
    monkeypatch.setattr(preferences, 'get_figure_style_per_graph', dict)
    frame = pd.DataFrame({'cls':['a']*8+['b']*8,'measurement':np.arange(16,dtype=float)})
    widget = FeatureExplorerPanel()
    qtbot.addWidget(widget)
    widget._frame = frame
    widget._spec = ExplorerSpec(label='cls',bins=8,n_permutations=0)
    widget._draw(rank_features(frame,widget._spec))
    _, specification = bundle._figure_record(widget._figure)
    captured = bundle._capture_view(widget._figure,specification)
    changed = frame.assign(cls=frame['cls'].replace({'a':'renamed'}))
    draw = bundle._draw
    if standalone:
        namespace = {'__name__':'feature_binding_probe'}
        exec(compile(bundle._recreate_script(captured),'recreate_figure.py','exec',dont_inherit=True),namespace)
        draw = namespace['_draw']
    with pytest.raises(ValueError,match='class identities do not match'):
        draw(Figure(),changed,captured)


@pytest.mark.parametrize('dialog', ['export', 'settings'])
def test_explicit_page_dimensions_release_automatic_export_geometry(qtbot, monkeypatch, dialog):
    """Both page controls must preserve an explicit 9-by-7-inch page rather than replacing it with a retained automatic layout."""
    from PySide6.QtWidgets import QFormLayout, QLabel
    from spacr.qt.widgets.save_figure_dialog import style_for_file
    from spacr.qt.widgets.figure_settings import FigureSettingsDialog
    from tests.test_graphstyle_live_page_export import _overflowing_figure

    figure, _ = _overflowing_figure()
    if dialog=='export':
        style_for_file(figure,width=9.,height=7.)
    else:
        widget = FigureSettingsDialog(figure)
        qtbot.addWidget(widget)
        fields = {}
        for form in widget.findChildren(QFormLayout):
            for row in range(form.rowCount()):
                label = form.itemAt(row,QFormLayout.LabelRole)
                field = form.itemAt(row,QFormLayout.FieldRole)
                if label and field and isinstance(label.widget(),QLabel):
                    fields[label.widget().text()] = field.widget()
        fields['Width (in)'].setValue(9.)
        fields['Height (in)'].setValue(7.)
    assert figure._spacr_live_geometry is None
    positions = [tuple(ax.get_position(original=True).bounds) for ax in figure.axes]
    with style._file_axes_geometry(figure) as expanded:
        assert not expanded
        assert tuple(figure.get_size_inches()) == (9.,7.)
        assert [tuple(ax.get_position(original=True).bounds) for ax in figure.axes] == positions



def test_eight_feature_panels_scroll_without_clipping_or_changing_source_values(qtbot, monkeypatch):
    """Eight 40-mm panels exceed a 500-by-420-pixel window; require real scrolling and every full panel inside the canvas, keeping the existing half-pixel physical-size margin and all original observations. Clearing the plot must release its automatic minimum size."""
    from PySide6.QtWidgets import QApplication

    monkeypatch.setattr(preferences,'get_figure_style',dict)
    monkeypatch.setattr(preferences,'get_figure_style_per_graph',dict)
    values = np.arange(16,dtype=float)
    frame = pd.DataFrame({'cls':['a']*8+['b']*8})
    for index in range(8):
        frame['measurement_'+str(index)] = values*(index+1)+np.sin(values+index)
    source = frame.to_csv(index=False)
    widget = FeatureExplorerPanel()
    qtbot.addWidget(widget)
    widget.resize(500,420)
    widget._frame = frame
    widget._spec = ExplorerSpec(label='cls',bins=8,n_permutations=0)
    widget._draw(rank_features(frame,widget._spec))
    widget.show()
    qtbot.waitExposed(widget)
    QApplication.processEvents()
    widget._canvas.draw()
    assert widget._plot_scroll.widget() is widget._canvas
    assert widget._plot_scroll.verticalScrollBar().maximum() > 0
    renderer = widget._canvas.get_renderer()
    page = widget._figure.bbox
    assert len(widget._figure.axes) == 8
    for ax in widget._figure.axes:
        box = ax.get_tightbbox(renderer)
        assert page.contains(box.x0,box.y0) and page.contains(box.x1,box.y1)
        physical = ax.get_window_extent(renderer)
        assert physical.width/widget._figure.dpi == pytest.approx(40/25.4,abs=.5/widget._figure.dpi)
        assert physical.height/widget._figure.dpi == pytest.approx(40/25.4,abs=.5/widget._figure.dpi)
    assert frame.to_csv(index=False) == source
    widget._frame = None
    widget._draw(rank_features(frame,widget._spec))
    QApplication.processEvents()
    assert not widget._figure.axes
    assert widget._figure._spacr_live_geometry is None
    assert widget._canvas.minimumWidth() == widget._canvas.minimumHeight() == 0
