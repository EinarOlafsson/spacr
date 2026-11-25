"""Purity presentation retains fixed embeddings, row bindings, scalar values and independent control shapes."""
import numpy as np
import pytest
from spacr.figures import style
from spacr.qt import preferences
from spacr.qt.widgets.annotation_umap_tab import PurityScatter

pytestmark = pytest.mark.qt


@pytest.fixture
def purity_plot(qtbot, monkeypatch):
    """Isolate preferences while using the real native annotation scatter."""
    monkeypatch.setattr(preferences, 'get_figure_style', dict)
    monkeypatch.setattr(preferences, 'get_figure_style_per_graph', dict)
    made = PurityScatter()
    qtbot.addWidget(made)
    made.resize(760, 560)
    return made


def test_purity_points_use_shared_geometry_opacity_and_keep_exact_rows_values_and_shapes(purity_plot):
    """Only appearance changes: finite source positions, row identities, purity values and PC/NC/annotated symbols remain exact, with four square-point marks at sixty percent opacity."""
    from spacr.qt.widgets.annotation_umap_tab import GROUP_SYMBOLS, ANNOTATED

    coordinates = np.array([[0., .1], [1., .8], [np.nan, .3], [-.4, -.6]])
    values = np.array([0., 1., .5, np.nan])
    before = coordinates.tobytes(), values.tobytes()
    count = purity_plot.set_embedding(coordinates, values, ['a', 'b', 'c', 'd'], ['PC', 'NC', None, None])
    assert count == 4
    assert purity_plot._spacr_style_spec.kind == 'continuous'
    assert dict(purity_plot._spacr_style_spec.comparison)['graph_type'] == 'embedding'
    frame = purity_plot.frame()
    np.testing.assert_array_equal(frame[['umap_1', 'umap_2']].to_numpy(), coordinates)
    np.testing.assert_array_equal(frame['purity'], values)
    points = purity_plot._scatter_items()[0].points()
    assert [point.data() for point in points] == [0, 1, 3]
    assert [point.symbol() for point in points] == [GROUP_SYMBOLS['PC'], GROUP_SYMBOLS['NC'], GROUP_SYMBOLS[ANNOTATED]]
    assert all(point.size() ** 2 * (72. / purity_plot.logicalDpiX()) ** 2 == pytest.approx(4.) for point in points)
    assert all(point.brush().color().alphaF() == pytest.approx(.6, abs=1/65535) for point in points)
    assert points[-1].brush().color().name().lower() == style.Palette.BACKGROUND.lower()
    assert not purity_plot.plot.plotItem.titleLabel.text
    assert before == (coordinates.tobytes(), values.tobytes())
    assert 'dimensionless' in str(dict(purity_plot._spacr_style_spec.legend)['units'])


def test_purity_keeps_constant_value_refusal_and_clears_stale_scalar_metadata(purity_plot):
    """A constant purity remains an explicit colour refusal after a previous mapped run; it cannot inherit the old colour-column or range note."""
    positions = np.array([[0., 0.], [1., 1.], [-1., .3]])
    purity_plot.set_embedding(positions, [0., .5, 1.], ['a', 'b', 'c'], ['PC', 'NC', None])
    assert purity_plot._colour_column is not None
    purity_plot.set_embedding(positions, [.5, .5, .5], ['a', 'b', 'c'], ['PC', 'NC', None])
    assert purity_plot._colour_column is None
    assert 'not drawn as a colour' in purity_plot.status()
    assert 'Coloured by' not in purity_plot.status()
    np.testing.assert_array_equal(purity_plot.frame()['purity'], [.5, .5, .5])


@pytest.mark.parametrize('tamper', ['key_colour', 'key_bounds', 'key_label', 'point_colour', 'key_view', 'key_ticks'])
def test_native_purity_scalar_validation_observes_actual_key_and_points_without_repair(purity_plot, tamper):
    """Changing a real scalar cell, source mark, endpoint or quantity label must be detected from actual artists while all source values remain exact and the deliberate change stays visible."""
    import pyqtgraph as pg

    purity_plot.set_embedding(np.array([[0., 0.], [1., 1.], [-1., .3]]),
                              [0., .5, 1.], ['a', 'b', 'c'], ['PC', 'NC', None])
    key = purity_plot._spacr_scalar_key
    assert key is not None
    findings, allowed = style._native_scalar_key_problems(purity_plot, purity_plot._spacr_style_spec)
    assert not findings and len(allowed) == 256
    original = purity_plot.frame().to_csv(index=False)
    cells = key._spacr_scalar_cells
    if tamper == 'key_colour':
        brushes = list(cells.opts['brushes'])
        brushes[7] = pg.mkBrush('#ff0000')
        cells.setOpts(brushes=brushes)
    elif tamper == 'key_bounds':
        high = cells.opts['y1'].copy()
        high[-1] += .1
        cells.setOpts(y1=high)
    elif tamper == 'key_label':
        key.setLabel('right', 'Unrelated quantity')
    elif tamper == 'point_colour':
        purity_plot._scatter_items()[0].points()[0].setBrush(pg.mkBrush('#ff0000'))
    elif tamper == 'key_ticks':
        key.getAxis('right').setTicks([[(0., '0'), (.5, '999'), (1., '1')]])
    else:
        key.getViewBox().setRange(yRange=(0., 1.1), padding=0.)
    findings, allowed = style._native_scalar_key_problems(purity_plot, purity_plot._spacr_style_spec)
    assert findings and not allowed
    assert purity_plot.frame().to_csv(index=False) == original
    if tamper == 'key_label':
        assert key.getAxis('right').labelText == 'Unrelated quantity'
    elif tamper == 'key_view':
        assert key.getViewBox().viewRange()[1] == [0., 1.1]
    elif tamper == 'key_bounds':
        assert cells.opts['y1'][-1] == 1.1


def test_purity_manual_colormap_and_font_remain_actual_choices(purity_plot):
    """An explicit non-house scalar map remains painted and diagnosed; changing the common font size reaches the real key without changing any scalar values."""
    purity_plot.set_embedding(np.array([[0., 0.], [1., 1.], [-1., .3]]),
                              [0., .5, 1.], ['a', 'b', 'c'], ['PC', 'NC', None])
    before = purity_plot.frame().to_csv(index=False)
    purity_plot.colour_by_column('purity', 'viridis')
    lookup = style._native_colour_table('viridis', 256)
    point = purity_plot._scatter_items()[0].points()[0]
    assert point.brush().color().getRgb()[:3] == tuple(int(value) for value in lookup[0, :3])
    findings, allowed = style._native_scalar_key_problems(purity_plot, purity_plot._spacr_style_spec)
    assert any('manual maps' in finding for finding in findings) and not allowed
    purity_plot.set_font_size(12)
    key = purity_plot._spacr_scalar_key
    assert key.getAxis('right').style['tickFont'].pointSizeF() == 12.
    assert key.getAxis('right').labelStyle['font-size'] == '12pt'
    assert purity_plot.frame().to_csv(index=False) == before


@pytest.mark.parametrize('fail', [False, True])
def test_temporary_native_export_style_restores_the_actual_scalar_key(purity_plot, fail):
    """A preview or failed export must restore the key font, label styles and pens exactly along with the main axes; temporary font size cannot leak into the live key."""
    purity_plot.set_embedding(np.array([[0., 0.], [1., 1.], [-1., .3]]),
                              [0., .5, 1.], ['a', 'b', 'c'], ['PC', 'NC', None])
    axis = purity_plot._spacr_scalar_key.getAxis('right')
    before = (axis.style['tickFont'].toString(), dict(axis.labelStyle), axis.pen(), axis.tickPen(), axis.textPen())
    source = purity_plot.frame().to_csv(index=False)
    try:
        with purity_plot._dressed_for_the_file(font_size=14, text_colour='#234567'):
            assert axis.style['tickFont'].pointSizeF() == 14.
            if fail:
                raise RuntimeError('injected preview failure')
    except RuntimeError as error:
        assert fail and str(error) == 'injected preview failure'
    assert (axis.style['tickFont'].toString(), dict(axis.labelStyle), axis.pen(), axis.tickPen(), axis.textPen()) == before
    assert purity_plot.frame().to_csv(index=False) == source


@pytest.mark.parametrize('remove_unit', [False, True])
def test_changing_the_native_colour_column_binds_its_actual_quantity_and_never_borrows_units(purity_plot, remove_unit):
    """The existing choose-any-column action must replace the purity key with its actual selected column; missing metadata stays unspecified rather than inheriting the previous quantity unit."""
    purity_plot.set_embedding(np.array([[0., 0.], [1., 1.], [-1., .3]]),
                              [0., .5, 1.], ['a', 'b', 'c'], ['PC', 'NC', None])
    if remove_unit:
        del purity_plot.frame().attrs['units']['umap_1']
    before = purity_plot.frame().to_csv(index=False)
    purity_plot.colour_by_column('umap_1', 'Blues')
    comparison = dict(purity_plot._spacr_style_spec.comparison)
    assert comparison['colour_column'] == 'umap_1' and comparison['colour_quantity'] == 'UMAP 1'
    assert comparison['colour_unit'] == (None if remove_unit else 'dimensionless')
    key = purity_plot._spacr_scalar_key
    assert 'UMAP 1' in key.getAxis('right').labelText
    assert 'Positive control' not in key.getAxis('right').labelText
    assert key.getViewBox().viewRange()[1] == [-1., 1.]
    findings, allowed = style._native_scalar_key_problems(purity_plot, purity_plot._spacr_style_spec)
    assert bool(findings) is remove_unit
    assert bool(allowed) is not remove_unit
    assert purity_plot.frame().to_csv(index=False) == before


def test_explicit_native_key_label_font_ticks_and_view_survive_a_colormap_change(purity_plot):
    """A colour-map change must keep explicit key appearance and diagnose its mismatched labels or range, rather than silently repairing the user choices."""
    from PySide6.QtGui import QFont

    purity_plot.set_embedding(np.array([[0., 0.], [1., 1.], [-1., .3]]),
                              [0., .5, 1.], ['a', 'b', 'c'], ['PC', 'NC', None])
    key = purity_plot._spacr_scalar_key
    axis = key.getAxis('right')
    font = QFont(axis.style['tickFont'])
    font.setPointSizeF(10.)
    axis.setStyle(tickFont=font)
    axis.setLabel('Explicit custom key')
    ticks = [[(0., 'start'), (.5, 'middle'), (1., 'end')]]
    axis.setTicks(ticks)
    key.getViewBox().setRange(yRange=(-.2, 1.2), padding=0.)
    before = purity_plot.frame().to_csv(index=False)
    purity_plot.colour_by_column('purity', 'viridis')
    assert axis.labelText == 'Explicit custom key'
    assert axis.style['tickFont'].toString() == font.toString()
    assert axis._tickLevels == ticks
    assert key.getViewBox().viewRange()[1] == [-.2, 1.2]
    findings, allowed = style._native_scalar_key_problems(purity_plot, purity_plot._spacr_style_spec)
    assert findings and not allowed
    assert purity_plot.frame().to_csv(index=False) == before
