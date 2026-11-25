"""Control-chart presentation retains every precomputed QC result and drawn coordinate."""
import numpy as np
import pandas as pd
import pytest
from spacr.figures import style
from spacr.qt import preferences, theme
from spacr.qt.screens import control_chart as screen
from spacr.qt.widgets import control_chart as engine

pytestmark = pytest.mark.qt


def _result(negative=False, *, case=None):
    """Run the unchanged small-control QC engine on a fixed campaign with a late drift."""
    rng = np.random.default_rng(113)
    source = pd.DataFrame([{'plate': 'P' + str(plate + 1).zfill(2), 'order': plate,
        'value': 7. if case == 'constant' else (-10. if negative else 100.) + max(0, plate - 19) * .7 + float(rng.normal(0., .3))}
        for plate in range(30) for well in range(2 + plate % 4 if case == 'unequal' else 3)])
    return engine.control_chart(source, engine.ControlChartSpec(value='value', plate='plate', order='order'))


@pytest.fixture
def canvas(qtbot, monkeypatch):
    """Keep Qt preferences isolated while exercising the actual owned production canvas."""
    monkeypatch.setattr(preferences, 'get_figure_style', dict)
    monkeypatch.setattr(preferences, 'get_figure_style_per_graph', dict)
    monkeypatch.setattr(screen, 'active_palette', lambda: theme.palette_for('light'))
    made = screen.ControlChartCanvas()
    qtbot.addWidget(made)
    return made


def test_control_chart_declares_default_geometry_and_exact_original_result(canvas):
    """The default mean/limit/flag values remain untouched while a square plot uses readable text, shared opacity, grey controls and black references."""
    result = _result()
    before = result.points_frame().to_csv(index=False), result.violations_frame().to_csv(index=False)
    canvas.set_result(result)
    axis = canvas.figure.axes[0]
    specification = axis._spacr_style_spec
    assert specification.kind == 'continuous'
    assert all(spine.get_visible() for spine in axis.spines.values())
    assert axis.get_position().width * canvas.figure.get_figwidth() == pytest.approx(40 / 25.4, abs=.5 / canvas.figure.dpi)
    assert axis.get_position().height * canvas.figure.get_figheight() == pytest.approx(40 / 25.4, abs=.5 / canvas.figure.dpi)
    assert axis.get_ylim()[0] == 0.
    assert all(artist.get_alpha() == .6 for artist in (*axis.lines, *axis.collections))
    np.testing.assert_array_equal(axis.lines[-1].get_ydata(), result.values)
    assert axis.lines[-1].get_color() == style.Palette.CONTROL
    assert axis.lines[-1].get_markerfacecolor() == style.Palette.CONTROL
    assert all(line.get_color() == '#000000' for line in axis.lines[:-1])
    assert not axis.texts
    assert before == (result.points_frame().to_csv(index=False), result.violations_frame().to_csv(index=False))
    assert not style.figure_problems(canvas.figure)


def test_control_chart_uses_supplied_units_and_keeps_negative_values(canvas):
    """Units are explicit display metadata; negative control values and all original outer limits remain present after layout."""
    result = _result(negative=True)
    canvas.set_result(result, unit='µm', quantity='Control size')
    axis = canvas.figure.axes[0]
    assert axis.get_ylabel() == 'Control size (µm)'
    assert axis.get_ylim()[0] <= min(result.values.min(), result.lower.min())
    np.testing.assert_array_equal(axis.lines[-1].get_ydata(), result.values)
    np.testing.assert_array_equal(axis.lines[0].get_ydata(), result.upper)
    np.testing.assert_array_equal(axis.lines[1].get_ydata(), result.lower)
    assert dict(dict(axis._spacr_style_spec.legend)['units'])['y'] == 'µm'
    assert not style.figure_problems(canvas.figure)


def test_empty_control_canvas_uses_the_shared_status_owner(canvas):
    """A refusal state is an explicitly declared status panel, with no invented data, reference, measurement unit or chart frame."""
    canvas.set_result(None, message='No control measurements are available for the selected campaign.')
    axis = canvas.figure.axes[0]
    assert axis._spacr_style_spec.kind == 'schematic'
    assert dict(axis._spacr_style_spec.comparison)['status'] is True
    assert not axis.lines and not axis.collections and not axis.images
    assert axis.texts[0].get_gid() == 'spacr-status'
    assert not axis.axison
    assert canvas.figure.get_facecolor() == (1., 1., 1., 1.)
    assert canvas.figure.patch.get_alpha() == 1.
    assert not style.finalize_figure(canvas.figure).problems


def test_graph_builder_empty_state_uses_the_same_shared_status_owner(qtbot, monkeypatch):
    """A long missing-binding message is a status sentence, rather than an undocumented chart or a ten-word schematic."""
    from spacr.qt.widgets.graph_builder import GraphCanvas

    monkeypatch.setattr(preferences, 'get_figure_style', dict)
    monkeypatch.setattr(preferences, 'get_figure_style_per_graph', dict)
    widget = GraphCanvas()
    qtbot.addWidget(widget)
    widget._figure.clear()
    widget._render_message('Choose the columns and a measurement before drawing this chart.')
    axis = widget._figure.axes[0]
    assert dict(axis._spacr_style_spec.comparison)['status'] is True
    assert axis.texts[0].get_gid() == 'spacr-status'
    assert not style.finalize_figure(widget._figure).problems


@pytest.mark.parametrize('standalone', [False, True])
@pytest.mark.parametrize('case', ['normal', 'negative', 'constant', 'unequal'])
def test_control_chart_recipe_replays_original_bands_limits_flags_and_keys(canvas, case, standalone):
    """Local and standalone replay must keep every frozen QC band/limit, flag coordinate, no-fill state, label and legend while leaving all registered source values unchanged. Figure-relative anchors allow four float64 epsilon for the inverse display transform."""
    import json
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from spacr.figures import bundle

    result = _result(negative=case == 'negative', case=case)
    canvas.set_result(result, unit='µm', quantity='Control size')
    canvas.canvas.draw()
    frame, specification = bundle._figure_record(canvas.figure)
    captured = json.loads(json.dumps(bundle._capture_view(canvas.figure, specification)))
    assert captured['control_chart_recipe']['schema'] == 1
    before = frame.to_csv(index=False)
    if standalone:
        namespace = {'__name__': 'standalone_control_chart'}
        exec(compile(bundle._recreate_script(captured), 'recreate_figure.py', 'exec', dont_inherit=True), namespace)
        draw = namespace['_draw']
    else:
        draw = bundle._draw
    restored = Figure()
    draw(restored, frame, captured)
    FigureCanvasAgg(restored).draw()
    assert not restored._spacr_replay_problems, restored._spacr_replay_problems
    assert frame.to_csv(index=False) == before
    original = canvas.figure.axes[0]
    copied = restored.axes[0]
    assert copied.get_box_aspect() == original.get_box_aspect()
    assert copied.get_xlim() == original.get_xlim() and copied.get_ylim() == original.get_ylim()
    assert copied.get_xlabel() == original.get_xlabel() and copied.get_ylabel() == original.get_ylabel()
    assert len(copied.lines) == len(original.lines)
    assert len(copied.collections) == len(original.collections)
    for first, second in zip(original.lines, copied.lines, strict=True):
        np.testing.assert_array_equal(second.get_xydata(), first.get_xydata())
        assert style._artist_appearance(second) == style._artist_appearance(first)
    for first, second in zip(original.collections, copied.collections, strict=True):
        np.testing.assert_array_equal(second.get_offsets(), first.get_offsets())
        for path, replayed_path in zip(first.get_paths(), second.get_paths(), strict=True):
            np.testing.assert_array_equal(replayed_path.vertices, path.vertices)
            np.testing.assert_array_equal(replayed_path.codes, path.codes)
        assert style._artist_appearance(second) == style._artist_appearance(first)
    original_key = style._legend_appearance(original.get_legend())
    replayed_key = style._legend_appearance(copied.get_legend())
    np.testing.assert_allclose(replayed_key.pop('anchor'), original_key.pop('anchor'),
                               rtol=0, atol=4 * np.finfo(float).eps)
    assert replayed_key == original_key


def test_control_chart_replay_refuses_a_different_plate_order(canvas):
    """Frozen baseline membership and rule positions cannot silently be applied to a permuted campaign."""
    from matplotlib.figure import Figure
    from spacr.figures import bundle

    canvas.set_result(_result(), unit='µm')
    frame, specification = bundle._figure_record(canvas.figure)
    captured = bundle._capture_view(canvas.figure, specification)
    assert captured['control_chart_recipe']['schema'] == 1
    changed = frame.iloc[::-1].reset_index(drop=True)
    with pytest.raises(ValueError, match='plate identities and order'):
        bundle._draw(Figure(), changed, captured)


def test_control_chart_archive_entry_point_requires_neither_spacr_nor_qt(canvas, tmp_path):
    """The real archive entry point renders frozen QC bands and rule marks with both application and Qt imports rejected."""
    import os
    import subprocess
    import sys
    import zipfile
    from spacr.figures import bundle

    canvas.set_result(_result(), unit='µm', quantity='Control size')
    canvas.canvas.draw()
    archive = bundle._save_zip(canvas.figure, str(tmp_path / 'control.zip'), formats=['svg'])
    folder = tmp_path / 'portable'
    with zipfile.ZipFile(archive) as saved:
        saved.extractall(folder)
    guard = tmp_path / 'guard'
    guard.mkdir()
    (guard / 'sitecustomize.py').write_text(
        "import sys\n"
        "class RejectApplication:\n"
        "    def find_spec(self, fullname, path=None, target=None):\n"
        "        if fullname.split('.')[0] in ('spacr', 'PySide6', 'PyQt5', 'PyQt6'):\n"
        "            raise ImportError('portable replay must not import spaCR or Qt')\n"
        "sys.meta_path.insert(0, RejectApplication())\n")
    process = subprocess.run([sys.executable, 'recreate_figure.py', 'svg', 'pdf'], cwd=folder,
        env=dict(os.environ, PYTHONPATH=str(guard), MPLBACKEND='Agg'), capture_output=True, text=True, timeout=60)
    assert process.returncode == 0, process.stdout + process.stderr
    vector = (folder / 'recreated.svg').read_text()
    assert '<svg' in vector and '<image' not in vector
    assert 'Control size' in vector and 'Control series' in vector
    assert (folder / 'recreated.pdf').read_bytes().startswith(b'%PDF')


def test_every_control_rule_ring_clears_the_physical_frame(canvas):
    """Large concentric flags must fit inside a 40 mm panel after the line origin includes zero; a data-coordinate-only bounds test misses their clipped strokes."""
    result = _result()
    canvas.set_result(result, unit='µm')
    canvas.canvas.draw()
    axis = canvas.figure.axes[0]
    gap = .5 * canvas.figure.dpi / 72.
    for bounds in style._marker_display_bounds(axis):
        assert bounds.x0 >= axis.bbox.x0 + gap - .01
        assert bounds.y0 >= axis.bbox.y0 + gap - .01
        assert bounds.x1 <= axis.bbox.x1 - gap + .01
        assert bounds.y1 <= axis.bbox.y1 - gap + .01
    assert axis.get_ylim()[0] == 0.
    np.testing.assert_array_equal(axis.lines[-1].get_ydata(), result.values)


def test_tight_control_chart_crop_keeps_full_long_label_and_external_key(canvas, tmp_path):
    """The tight PNG must keep the long Z-prime axis title and external legend with a white border; Matplotlib otherwise clips the label at the top despite a valid live layout."""
    from PIL import Image
    from spacr import plot

    canvas.set_result(_result(), unit='dimensionless', quantity='Z-prime assay separation')
    canvas.canvas.draw()
    axis = canvas.figure.axes[0]
    artists = style._tight_export_artists(canvas.figure)
    assert axis.yaxis.label in artists and axis.get_legend() in artists
    target = tmp_path / 'control.png'
    plot.save_figure(canvas.figure, target, fmt='png', dpi=300, save_mode='screen',
                     integrity=False, announce_colours=False, bbox_inches='tight', facecolor='white')
    with Image.open(target) as saved:
        pixels = np.asarray(saved.convert('RGB'))
    assert np.all(pixels[:20] == 255) and np.all(pixels[-20:] == 255)
    assert np.all(pixels[:, :20] == 255) and np.all(pixels[:, -20:] == 255)
