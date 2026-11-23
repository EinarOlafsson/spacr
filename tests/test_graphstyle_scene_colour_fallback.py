"""Unsupported native scalar colour semantics retain the original Matplotlib renderer."""
import numpy as np
import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.colors import TwoSlopeNorm

from spacr.figures import scene,style

pytestmark=pytest.mark.qt


@pytest.mark.parametrize('case',['centred','horizontal','opaque_missing'])
def test_scene_declines_colour_semantics_it_cannot_reproduce_without_changing_values(qapp,monkeypatch,tmp_path,case):
    """The native linear lookup shifted a declared zero and changed missing-cell colours; unsupported mappings must return the established Matplotlib fallback before any native file is written."""
    figure=Figure(figsize=(3.,3.));ax=figure.add_subplot()
    values=np.array([[-2.,0.,7.],[1.,2.,np.nan]])
    ramp=style.diverging_cmap() if case=='centred' else style.heatmap_cmap()
    if case=='opaque_missing':ramp.set_bad(ramp(0.))
    else:ramp.set_bad((0.,0.,0.,0.))
    image=ax.imshow(values,cmap=ramp,norm=TwoSlopeNorm(vmin=-2.,vcenter=0.,vmax=7.) if case=='centred' else None)
    if case=='horizontal':figure.colorbar(image,ax=ax,orientation='horizontal')
    FigureCanvasAgg(figure);figure.canvas.draw()
    before=image.get_array().data.tobytes(),np.ma.getmaskarray(image.get_array()).tobytes(),image.get_clim()
    calls=[]

    def ready():
        """Use the real available Qt backend independently of the application gallery."""
        return True,''

    def write(*args,**kwargs):
        """Record an invalid native write if unsupported colour semantics are accepted."""
        calls.append(args)
        return str(tmp_path/'should-not-be-written.png')

    monkeypatch.setattr(scene,'pyqtgraph_ready',ready)
    monkeypatch.setattr(scene,'export_scene',write)
    written,report=scene.render_figure(figure,tmp_path/'original.png',mode='screen',dpi=150,announce=False)
    assert written is None
    assert not calls
    assert 'NotImplementedError' in report.missing
    assert any('Matplotlib rendering' in note for note in report.notes)
    assert (image.get_array().data.tobytes(),np.ma.getmaskarray(image.get_array()).tobytes(),image.get_clim())==before


def test_constant_key_retains_its_single_true_value_through_the_original_renderer(qapp, monkeypatch, tmp_path):
    """A detached constant key may carry a display-only expanded norm; the original source bounds and single tick must survive without new native endpoint labels."""
    figure = Figure(figsize=(3., 3.))
    axis = figure.add_subplot()
    values = np.full((2, 3), .5)
    image = axis.imshow(values, alpha=.6, cmap='Blues')
    key = style.attach_colourbar(axis, image, label='Purity (dimensionless)',
                                 unit='dimensionless', constant_colour=style.Palette.ACCENT)
    FigureCanvasAgg(figure).draw()
    before = image.get_array().tobytes(), image.get_clim(), key.get_ticks().tolist()
    assert image.get_clim() == (.5, .5) and key.get_ticks().tolist() == [.5]
    calls = []

    def ready():
        """Use the real available Qt adapter for its fallback decision."""
        return True, ''

    def write(*args, **kwargs):
        """Record any incorrect native write of invented endpoints."""
        calls.append(args)
        return str(tmp_path / 'incorrect.png')

    monkeypatch.setattr(scene, 'pyqtgraph_ready', ready)
    monkeypatch.setattr(scene, 'export_scene', write)
    written, report = scene.render_figure(figure, tmp_path / 'constant.png', mode='screen', dpi=150, announce=False)
    assert written is None and not calls
    assert any('Constant scalar colour keys' in note for note in report.notes)
    assert before == (image.get_array().tobytes(), image.get_clim(), key.get_ticks().tolist())
