"""Linked statistical axes retain separate quantities and one physical plotting area."""
import numpy as np
import pytest
from matplotlib.figure import Figure

from spacr.figures import style


def _linked_figure():
    """Build an explicitly declared categorical estimate and cumulative percentage overlay."""
    with style.figure_style('print'):
        figure = Figure(); primary = figure.add_subplot()
        bars = primary.bar(range(3), [50., 30., 20.], width=.6, color=style.Palette.ACCENT, alpha=.6)
        for index, bar in enumerate(bars):
            bar.set_gid(f'spacr-condition:{index}:bar')
        primary.set_xticks(range(3), ['PC1', 'PC2', 'PC3'])
        primary.set_xlim(-.5, 2.5)
        primary.set_xlabel('Principal component'); primary.set_ylabel('Variance (%)')
        specification = style.resolve_style('print', 'categorical', conditions=['PC1','PC2','PC3'],
            comparison={'representation':'estimates'}, legend={'caption':'Variance explained by each component.',
            'reference':'Total variance (100%).', 'quantities':{'y':'variance'}, 'units':{'y':'%'}})
        style.apply_axes_style(primary, specification, layout=False)
        overlay = primary.twinx()
        line, = overlay.plot(range(3), [50., 80., 100.], color=style.Palette.TEAL, alpha=.6)
        overlay.set_ylim(0., 105.)
        overlay.set_ylabel('Cumulative variance (%)')
        overlay_spec = style.resolve_style('print', 'axis_overlay',
            comparison={'overlay_of':0, 'shared_axis':'x'},
            legend={'caption':'Cumulative variance on the shared component axis.',
                    'reference':'Total variance (100%).', 'quantities':{'x':'component', 'y':'cumulative variance'},
                    'units':{'x':'dimensionless', 'y':'%'}})
        style.apply_axes_style(overlay, overlay_spec, layout=False)
    return figure, primary, overlay, line


def test_linked_axes_share_physical_slots_and_fit_both_sets_of_labels_without_changing_values():
    """An overlay is one chart area; fit both axis labels while preserving its separate numerical range and exact cumulative values."""
    figure, primary, overlay, line = _linked_figure()
    values = line.get_xydata().tobytes()
    limits = (primary.get_xlim(), primary.get_ylim(), overlay.get_ylim())
    assert overlay._spacr_style_spec.kind == 'axis_overlay'
    style.pack_axes_geometry([primary])
    assert primary.get_position().width*figure.get_size_inches()[0] == pytest.approx(.66, abs=.5/300)
    assert overlay.get_position().bounds == primary.get_position().bounds
    assert (primary.get_xlim(), primary.get_ylim(), overlay.get_ylim()) == limits
    assert line.get_xydata().tobytes() == values
    result = style.finalize_figure(figure)
    assert result.compliant, result.problems


def test_an_unlinked_axis_cannot_claim_overlay_geometry():
    """Declaring overlay metadata cannot bypass geometry validation for an independent axes."""
    figure, primary, overlay, line = _linked_figure()
    independent = figure.add_axes(primary.get_position())
    independent._spacr_style_spec = overlay._spacr_style_spec
    before = line.get_xydata().tobytes()
    findings = style.finalize_figure(figure, diagnostics=False)
    assert any('Link the overlay' in problem for problem in findings.problems)
    assert not findings.compliant
    assert line.get_xydata().tobytes() == before
