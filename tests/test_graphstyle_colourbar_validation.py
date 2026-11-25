"""Actual scalar keys remain matched to plotted values after explicit edits."""
import numpy as np
import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import PowerNorm
from matplotlib.figure import Figure
from spacr.figures import style

MESSAGE='Match the colour key to the plotted scalar values and opacity'


def _figure(constant=False):
    """Create a shared scalar key with a complete declared heatmap and a resolved renderer."""
    figure=Figure();FigureCanvasAgg(figure);ax=figure.subplots()
    values=np.full((2,2),2.) if constant else np.arange(4.).reshape(2,2)
    image=style.heatmap_image(ax,values,cmap="Blues",vmin=float(values.min()),vmax=float(values.max()))
    key=style.attach_colourbar(ax,image,label='Signal',unit='a.u.')
    style.apply_axes_style(ax,style.resolve_style('print','heatmap',
        legend={'caption':'Signal in arbitrary units.','reference':'supplied range',
                'quantities':{'value':'Signal'},'unit':'a.u.'}))
    figure.canvas.draw()
    return figure,ax,image,key


def _state(image,key):
    """Capture numerical values and actual manual source/key choices around read-only validation."""
    return (image.get_array().data.tobytes(),np.ma.getmaskarray(image.get_array()).tobytes(),
        image.get_clim(),image.get_cmap().name,image.norm.__class__.__name__,
        getattr(image.norm,'gamma',None),image.get_alpha(),key.mappable.get_clim(),key.cmap.name,
        key.solids.get_alpha(),key.solids.get_facecolors().tobytes())


@pytest.mark.parametrize('edit',['source_cmap','source_limits','source_norm','key_cmap','key_faces','key_alpha'])
def test_key_mismatch_is_diagnosed_without_recolouring_or_changing_values(edit):
    """Each edit leaves a real stale or altered key; validation must report the mismatch and preserve all values and explicit choices byte-for-byte."""
    figure,ax,image,key=_figure()
    if edit=='source_cmap':image.set_cmap('Blues_r')
    elif edit=='source_limits':image.set_clim(0.,6.)
    elif edit=='source_norm':image.set_norm(PowerNorm(.5,vmin=0.,vmax=3.))
    elif edit=='key_cmap':key.mappable.set_cmap('Reds')
    elif edit=='key_faces':key.solids.set_array(None);key.solids.set_facecolor('red')
    elif edit=='key_alpha':key.solids.set_alpha(.2)
    figure.canvas.draw();before=_state(image,key)
    report=style.finalize_figure(figure,diagnostics=False)
    assert MESSAGE in report.problems
    assert _state(image,key)==before


@pytest.mark.parametrize('constant',[False,True])
def test_unchanged_key_keeps_its_true_range_and_has_no_mapping_diagnostic(constant):
    """Both 0..3 scales and the single real constant value 2 must pass matching checks without expanding the source normalization."""
    figure,ax,image,key=_figure(constant)
    before=_state(image,key)
    report=style.finalize_figure(figure,diagnostics=False)
    assert MESSAGE not in report.problems
    assert _state(image,key)==before


def test_key_body_fonts_and_contrast_are_checked_without_repair():
    """A six-point bold serif white key label must produce size, family, weight and contrast findings while retaining the explicit text style."""
    figure,ax,image,key=_figure()
    label=key.ax.yaxis.label
    label.set_fontfamily('DejaVu Serif');label.set_fontweight('bold');label.set_fontsize(6.);label.set_color('white')
    report=style.finalize_figure(figure,diagnostics=False)
    assert 'Use bundled Open Sans for every colour-key text element' in report.problems
    assert 'Use at least eight points for every colour-key text element' in report.problems
    assert 'Use the declared body font weight for colour-key text' in report.problems
    assert 'Use readable text contrast against the colour-key background' in report.problems
    assert label.get_fontfamily()==['DejaVu Serif'] and label.get_fontsize()==6. and label.get_fontweight()=='bold' and label.get_color()=='white'
