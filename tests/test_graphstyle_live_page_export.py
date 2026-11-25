"""Ordinary file exports contain packed live panels and restore the interactive state."""
import copy

import numpy as np
import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from PIL import Image

from spacr.figures import style
from spacr.plot import _checked_savefig, save_figure


def _overflowing_figure():
    """Use eight 40-mm panels in a 4-by-3-inch viewport, an exact geometric overflow with no numerical tolerance or timing gate."""
    with style.figure_style('print'):
        figure = Figure(figsize=(4., 3.), dpi=100)
        axes = figure.subplots(4, 2).ravel()
        for index, ax in enumerate(axes):
            ax.scatter([.2, .8], [index, index+1], s=4, alpha=.6)
            ax.set_xlabel('Measured quantity'); ax.set_ylabel('Observation')
            spec = style.resolve_style('print', 'continuous')
            style.apply_axes_style(ax, spec, layout=False)
            style.panel_letter(ax, index)
        style.pack_axes_geometry(axes)
        packed = tuple(figure.get_size_inches())
        style._retain_live_axes_geometry(figure, (4., 3.))
    FigureCanvasAgg(figure)
    return figure, packed


def _live_state(figure):
    """Capture exact geometry, source coordinates, limits and automatic-layout metadata."""
    return dict(size=tuple(figure.get_size_inches()),
        positions=[tuple(ax.get_position(original=True).bounds) for ax in figure.axes],
        limits=[(ax.get_xlim(), ax.get_ylim(), ax.get_autoscalex_on(), ax.get_autoscaley_on()) for ax in figure.axes],
        values=[item.get_offsets().tobytes() for ax in figure.axes for item in ax.collections],
        layout=copy.deepcopy(figure._spacr_live_geometry),
        diagnostics=getattr(figure,'_spacr_style_diagnostics',None))


@pytest.mark.parametrize('writer', [save_figure, _checked_savefig])
@pytest.mark.parametrize('refuse', [False, True])
def test_ordinary_writer_uses_a_complete_packed_page_and_restores_live_state(tmp_path, monkeypatch, writer, refuse):
    """The measured eight-panel live page clipped at the top; at the actual savefig boundary every panel and letter must lie inside the page. Check exact viewport, limits, arrays, cache and live-validation restoration after success and an injected write failure."""
    figure, packed = _overflowing_figure()
    figure._spacr_style_diagnostics = style.FigureDiagnostics("live", ("Live viewport needs scrolling",), ())
    before = _live_state(figure)
    actual_save = figure.savefig
    written = []

    def check_page(path, **kwargs):
        """Inspect the real export page, then perform or refuse the actual writer."""
        assert tuple(figure.get_size_inches()) == tuple(np.maximum(packed, before['size']))
        canvas = figure.canvas
        canvas.draw()
        renderer = canvas.get_renderer()
        page = figure.bbox
        for ax in figure.axes:
            area = ax.get_tightbbox(renderer)
            assert page.contains(area.x0, area.y0) and page.contains(area.x1, area.y1)
        written.append(tuple(figure.get_size_inches()))
        if refuse:
            raise OSError('injected write refusal')
        return actual_save(path, **kwargs)

    monkeypatch.setattr(figure, 'savefig', check_page)
    output = tmp_path/'packed.png'
    kwargs = {'fmt':'png'} if writer is save_figure else {'format':'png'}
    if refuse:
        with pytest.raises(OSError, match='injected write refusal'):
            writer(figure, output, dpi=100, integrity=False, **kwargs)
    else:
        writer(figure, output, dpi=100, integrity=False, **kwargs)
        with Image.open(output) as image:
            assert image.height > 300
    assert len(written) == 1
    assert _live_state(figure) == before


@pytest.mark.parametrize('writer', [save_figure, _checked_savefig])
def test_manual_axes_placement_releases_automatic_export_page(tmp_path, monkeypatch, writer):
    """A later exact manual rectangle must release automatic page expansion and remain unchanged by an ordinary write."""
    figure, _ = _overflowing_figure()
    figure.axes[0].set_position([.1, .2, .3, .4])
    size = tuple(figure.get_size_inches())
    positions = [tuple(ax.get_position(original=True).bounds) for ax in figure.axes]
    observed = []

    def check_page(path, **kwargs):
        """Observe explicit user geometry at the actual writer boundary."""
        observed.append(tuple(figure.get_size_inches()))
        assert [tuple(ax.get_position(original=True).bounds) for ax in figure.axes] == positions

    monkeypatch.setattr(figure, 'savefig', check_page)
    kwargs = {'fmt':'png'} if writer is save_figure else {'format':'png'}
    writer(figure, tmp_path/'manual.png', integrity=False, **kwargs)
    assert observed == [size]
    assert tuple(figure.get_size_inches()) == size
