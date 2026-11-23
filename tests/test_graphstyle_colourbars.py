"""Shared numeric color keys preserve source normalization and physical panel geometry."""
import numpy as np
import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.colors import Normalize, TwoSlopeNorm
from spacr.figures import style


@pytest.mark.parametrize('limits',[(0.,3.),(2.,2.),(-2.,8.)])
def test_attached_color_key_preserves_the_source_norm_arrays_and_actual_color_mapping(limits):
    """A real Matplotlib colorbar must not expand a constant source normalization or move an explicitly centered reference."""
    figure=Figure();FigureCanvasAgg(figure);ax=figure.add_subplot()
    norm=TwoSlopeNorm(0.,vmin=-2.,vmax=8.) if limits==(-2.,8.) else Normalize(*limits)
    values=np.ma.array([[limits[0],np.nan],[limits[1],limits[0]]])
    image=ax.imshow(values,norm=norm,cmap='Blues',alpha=.6)
    original=norm.vmin,norm.vmax,getattr(norm,'vcenter',None),image.get_array().data.tobytes(),np.ma.getmaskarray(image.get_array()).tobytes()
    key=style.attach_colourbar(ax,image,label='Signal',unit='a.u.')
    spec=style.resolve_style('print','scatter',legend={'caption':'Synthetic signal.', 'reference':'supplied range','quantities':{'x':'coordinate','y':'coordinate'},'unit':'dimensionless'})
    style.apply_axes_style(ax,spec)
    figure.canvas.draw()
    assert (norm.vmin,norm.vmax,getattr(norm,'vcenter',None),image.get_array().data.tobytes(),np.ma.getmaskarray(image.get_array()).tobytes())==original
    assert ax.get_position().width*figure.get_size_inches()[0]==pytest.approx(40/25.4,abs=.5/figure.dpi)
    assert ax.get_position().height*figure.get_size_inches()[1]==pytest.approx(40/25.4,abs=.5/figure.dpi)
    assert key.ax.get_position().x0>ax.get_position().x1
    assert key.ax.get_ylabel()=='Signal (a.u.)'
    if limits[0]==limits[1]:
        assert [text.get_text() for text in key.ax.get_yticklabels()]==['2']
        np.testing.assert_array_equal(key.cmap(.5),image.cmap(image.norm(2.)))
    else:
        np.testing.assert_array_equal(key.cmap(key.norm(np.array(limits))),image.cmap(image.norm(np.array(limits))))
    assert key.solids.get_alpha()==.6
    assert all(text.get_fontsize()>=8 and text.get_fontweight()=='light' for text in key.ax.get_yticklabels())
    assert 'Open Sans' in key.ax.yaxis.label.get_fontfamily()


def test_attached_color_keys_pack_with_multiple_primary_panels_without_overlap():
    """Two independently scaled panels retain square axes and leave measured room for both keys, labels and tick values."""
    figure=Figure();FigureCanvasAgg(figure);axes=figure.subplots(1,2)
    keys=[]
    for index,ax in enumerate(axes):
        image=ax.imshow(np.arange(4).reshape(2,2)+index*10,cmap='Blues',alpha=.6)
        keys.append(style.attach_colourbar(ax,image,label='Signal',unit='a.u.'))
        style.apply_axes_style(ax,style.resolve_style('print','scatter',legend={'caption':'Synthetic signal','reference':'range','quantities':{'x':'x','y':'y'},'unit':'dimensionless'}),layout=False)
    style.pack_axes_geometry(axes)
    figure.canvas.draw();renderer=figure.canvas.get_renderer()
    for ax,key in zip(axes,keys):
        assert ax.get_position().width*figure.get_size_inches()[0]==pytest.approx(40/25.4,abs=.5/figure.dpi)
        assert key.ax.get_position().x0>ax.get_position().x1
        assert figure.bbox.contains(*key.ax.get_tightbbox(renderer).corners().max(axis=0))
    assert keys[0].ax.get_tightbbox(renderer).x1<axes[1].get_tightbbox(renderer).x0


def test_empty_color_mapping_keeps_its_unknown_limits_without_inventing_a_key():
    """A zero-cell or all-missing scalar image has no finite range to explain; leave both source normalization and figure axes untouched."""
    figure=Figure();ax=figure.add_subplot()
    image=ax.imshow(np.full((2,2),np.nan),cmap='Blues')
    image.norm.vmin=None;image.norm.vmax=None
    original=tuple(figure.axes),image.get_array().data.tobytes()
    assert style.attach_colourbar(ax,image,label='Signal') is None
    assert image.get_clim()==(None,None)
    assert (tuple(figure.axes),image.get_array().data.tobytes())==original
