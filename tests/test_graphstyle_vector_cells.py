"""Numerical cells export as vectors while micrographs and source state remain intact."""
import io
import json
import numpy as np
import pytest
from matplotlib.figure import Figure
from spacr.figures import style
from spacr import plot


def _state(ax):
    """Capture the rendered source boundary, excluding ordinary first-draw tick/font initialization from export comparisons."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    original=ax.figure.canvas
    try:
        FigureCanvasAgg(ax.figure).draw()
    finally:
        ax.figure.set_canvas(original)
    return (tuple(map(id,ax.images)),tuple(map(id,ax.collections)),ax.get_xlim(),ax.get_ylim(),
        ax.get_autoscalex_on(),ax.get_autoscaley_on(),ax.dataLim.get_points().tobytes(),ax.ignore_existing_data_limits,
        tuple((image.get_array().data.tobytes(),np.ma.getmaskarray(image.get_array()).tobytes(),image.get_clim()) for image in ax.images),
        json.dumps(style._capture_appearance(ax.figure),sort_keys=True))


@pytest.mark.parametrize('origin',['upper','lower'])
@pytest.mark.parametrize('extension',['svg','pdf'])
def test_shared_numeric_image_exports_real_vector_cells_without_changing_its_source(tmp_path,origin,extension):
    """A masked 2x3 source retains exact bytes, original normalization and every artist property through the real shared writer; SVG/PDF contain no embedded raster image."""
    figure=Figure();ax=figure.add_subplot()
    style.heatmap_image(ax,np.array([[1.,2.,np.nan],[3.,4.,5.]]),cmap='Blues',vmin=1.,vmax=5.,origin=origin,extent=(-2.,4.,10.,14.))
    before=_state(ax)
    target=tmp_path/('cells.'+extension)
    plot.save_figure(figure,target,fmt=extension,dpi=300,save_mode='screen',integrity=False,announce_colours=False,preferences=False)
    data=target.read_bytes()
    assert (b'<image' not in data) if extension=='svg' else (b'/Subtype /Image' not in data)
    assert _state(ax)==before


def test_numeric_cell_export_restores_source_state_after_an_injected_write_failure():
    """Temporary vector cells must not survive a failed writer or change the original view, mask, normalization, autoscale flags or replay capture."""
    figure=Figure();ax=figure.add_subplot()
    image=style.heatmap_image(ax,[[1.,np.nan],[2.,3.]],cmap='Blues',vmin=1.,vmax=3.)
    before=_state(ax)
    with pytest.raises(RuntimeError,match='injected'):
        with style._vector_numeric_cells(figure,'svg'):
            assert not image.get_visible()
            raise RuntimeError('injected')
    assert _state(ax)==before


def test_untagged_microscopy_image_keeps_its_original_raster_output():
    """A two-dimensional micrograph is not inferred to be a numerical heatmap from shape or datatype."""
    figure=Figure();ax=figure.add_subplot();ax.imshow(np.arange(16).reshape(4,4),cmap='gray')
    before=_state(ax);target=io.StringIO()
    with style._vector_numeric_cells(figure,'svg'):
        figure.savefig(target,format='svg')
    assert '<image' in target.getvalue()
    assert _state(ax)==before


@pytest.mark.parametrize('origin',['upper','lower'])
def test_vector_cell_coordinates_and_colors_match_the_source_image(origin):
    """Verify all six literal cell boundaries and actual RGBA faces, including the missing cell, against the existing source transform and color mapping."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    figure=Figure();FigureCanvasAgg(figure);ax=figure.add_subplot()
    values=np.array([[1.,2.,np.nan],[3.,4.,5.]])
    image=style.heatmap_image(ax,values,cmap='Blues',vmin=1.,vmax=5.,origin=origin,extent=(-2.,4.,10.,14.))
    before=_state(ax)
    with style._vector_numeric_cells(figure,'svg'):
        mesh=next(mark for mark in ax.collections if hasattr(mark,'get_coordinates'))
        levels=[14.,12.,10.] if origin=='upper' else [10.,12.,14.]
        expected=np.array([[[x,y] for x in [-2.,0.,2.,4.]] for y in levels])
        np.testing.assert_array_equal(mesh.get_coordinates(),expected)
        np.testing.assert_array_equal(mesh.get_array().filled(np.nan),values)
        figure.canvas.draw()
        np.testing.assert_array_equal(mesh.get_facecolors(),image.to_rgba(image.get_array(),alpha=.6).reshape(-1,4))
        assert mesh.norm is not image.norm and mesh.norm.callbacks is not image.norm.callbacks
    assert _state(ax)==before


@pytest.mark.parametrize('choice',['rasterized','interpolation','per_pixel_alpha'])
def test_explicit_manual_image_rendering_keeps_its_original_primitive(choice):
    """Preserve explicit rasterization, interpolated pixels and per-cell alpha without forcing an incompatible vector conversion."""
    figure=Figure();ax=figure.add_subplot()
    image=style.heatmap_image(ax,[[1.,2.],[3.,4.]],cmap='Blues',vmin=1.,vmax=4.)
    if choice=='rasterized':
        image.set_rasterized(True)
    elif choice=='interpolation':
        image.set_interpolation('bilinear')
    else:
        image.set_alpha(np.full((2,2),.35))
    before=_state(ax)
    with style._vector_numeric_cells(figure,'svg'):
        assert image.get_visible()
        assert not any(hasattr(mark,'get_coordinates') for mark in ax.collections)
    assert _state(ax)==before


def test_per_cell_opacity_and_interpolation_replay_as_json_data():
    """Preserve manually chosen opacity arrays, interpolation and rasterization through captured JSON appearance without changing numerical image values."""
    figure=Figure();ax=figure.add_subplot();values=np.arange(4.).reshape(2,2)
    image=style.heatmap_image(ax,values,cmap='Blues',vmin=0.,vmax=3.)
    opacity=np.array([[.2,.3],[.4,.5]])
    image.set_alpha(opacity);image.set_interpolation('bilinear');image.set_rasterized(True)
    saved=json.loads(json.dumps(style._artist_appearance(image)))
    other=ax.imshow(values.copy())
    original=other.get_array().tobytes()
    style._restore_artist_appearance(other,saved)
    np.testing.assert_array_equal(other.get_alpha(),opacity)
    assert other.get_interpolation()=='bilinear' and other.get_rasterized()
    assert other._spacr_numeric_cells
    assert other.get_array().tobytes()==original
