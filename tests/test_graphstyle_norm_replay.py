"""Portable appearance retains actual scalar normalization instead of replaying a linear substitute."""
import json
import numpy as np
import pytest
from matplotlib import colors
from matplotlib.figure import Figure
from spacr.figures import style,bundle


def _normalizations():
    """Supply standard real normalization owners with literal asymmetric bounds and non-default parameters."""
    return [colors.Normalize(-2.,8.,clip=True),colors.TwoSlopeNorm(1.,vmin=-2.,vmax=8.),
        colors.CenteredNorm(vcenter=1.,halfrange=7.,clip=True),colors.PowerNorm(.4,vmin=-2.,vmax=8.,clip=True),
        colors.LogNorm(.1,10.,clip=True),colors.SymLogNorm(.3,linscale=1.4,vmin=-2.,vmax=8.,base=2.,clip=True),
        colors.BoundaryNorm([-2.,0.,1.,8.],8,extend='both'),colors.NoNorm(vmin=-2.,vmax=8.,clip=True)]


@pytest.mark.parametrize('norm',_normalizations(),ids=lambda norm:type(norm).__name__)
@pytest.mark.parametrize('portable',[False,True])
def test_standard_normalization_replay_preserves_exact_type_parameters_and_rgba(norm,portable):
    """Every recorded map must reproduce exactly the source RGBA at the original values, not merely its endpoints, while source value/mask bytes remain unchanged."""
    values=np.array([[.1,.4],[1.,8.]])
    figure=Figure();ax=figure.subplots();source=ax.imshow(values,cmap='Blues',norm=norm,alpha=.6)
    expected=source.cmap(source.norm(values));before=source.get_array().data.tobytes()
    saved=json.loads(json.dumps(style._capture_appearance(figure)))
    assert saved['axes'][0]['images'][0]['norm']['kind']==type(norm).__name__
    target=Figure();replay=target.subplots();image=replay.imshow(values.copy(),cmap='viridis')
    if portable:
        namespace={'__name__':'standalone_norm'}
        exec(compile(bundle._appearance_script(),'appearance.py','exec',dont_inherit=True),namespace)
        problems=namespace['_restore_appearance'](target,saved)
    else:problems=style._restore_appearance(target,saved)
    assert not problems
    assert type(image.norm) is type(norm)
    assert image.get_clim()==source.get_clim()
    np.testing.assert_array_equal(image.cmap(image.norm(values)),expected)
    assert image.norm.clip==norm.clip
    assert image.get_array().data.tobytes()==source.get_array().data.tobytes()==before
    np.testing.assert_array_equal(np.ma.getmaskarray(image.get_array()),np.ma.getmaskarray(source.get_array()))


def test_custom_normalization_gets_an_explicit_replay_finding():
    """An arbitrary custom transform cannot silently replay as Normalize and receive an empty success report."""
    class SquaredNorm(colors.Normalize):
        """A deliberately custom source transform that must not be guessed by the archive."""
        def __call__(self,value,clip=None):
            """Square the standard normalized coordinates for a visibly different map."""
            return super().__call__(value,clip=clip)**2
    values=np.array([[0.,.2],[.4,1.]])
    figure=Figure();source=figure.subplots().imshow(values,norm=SquaredNorm(0.,1.),cmap='Blues')
    saved=json.loads(json.dumps(style._capture_appearance(figure)))
    target=Figure();target.subplots().imshow(values.copy())
    problems=style._restore_appearance(target,saved)
    assert 'Captured appearance does not support this colour normalization' in problems
    np.testing.assert_array_equal(source.get_array(),values)
    assert isinstance(source.norm,SquaredNorm)
