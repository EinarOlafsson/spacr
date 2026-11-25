"""Control distribution defaults preserve source values and linked row identities."""
import numpy as np
import pytest

pytestmark=pytest.mark.qt


@pytest.mark.parametrize('count',[1,20,21,1001])
def test_control_defaults_use_shared_forms_slots_and_explicit_grey_control_roles_without_changing_rows(qtbot,monkeypatch,count):
    """The unconfigured control widget bypassed automatic forms; require exact source effects/keys, every finite point through 999, no individual points at1000, .22-inch slots and explicitly grey controls with counts outside ticks."""
    import pyqtgraph as pg
    from spacr import graph_types
    from spacr.figures import style
    from spacr.qt.widgets.fast_plots import ControlSeparation

    monkeypatch.setattr(graph_types,'chosen_for',lambda _shape:'')
    groups={'alpha':np.r_[np.linspace(-.5,-.1,count),np.nan],
            'beta':np.r_[np.linspace(.6,1.,count),np.nan]}
    keys={label:[f'{label}_{i}' for i in range(len(values))] for label,values in groups.items()}
    before=np.r_[groups['alpha'],groups['beta']].tobytes()
    widget=ControlSeparation();qtbot.addWidget(widget)
    assert widget.set_groups(groups,keys=keys)==2*count
    widget.resize(760,560);widget.show();qtbot.waitExposed(widget)
    assert widget._effects.tobytes()==before
    assert list(widget._keys)==keys['alpha']+keys['beta']
    assert widget.mark()=='auto'
    spec=widget._spacr_style_spec
    assert spec.kind=='categorical' and spec.sample_counts==(count,count) and spec.controls==(0,1)
    assert widget.plot.getAxis('bottom')._category_labels==('alpha','beta')
    points=[item for item in widget.plot.plotItem.items if isinstance(item,pg.ScatterPlotItem)]
    assert sum(len(item.points()) for item in points)==(2*count if count<1000 else 0)
    with widget._held_at_policy_geometry() as applied:
        assert applied
        observed=widget._spacr_style_observation()
        assert observed['width_inches']==pytest.approx(.44,abs=.5/widget.logicalDpiX())
        report=style.finalize_figure(widget)
        assert not any('forms' in value or 'condition' in value or 'sixty percent' in value for value in report.problems),report.problems
    widget.set_mark('jitter')
    assert widget.mark()=='jitter'
    assert widget._effects.tobytes()==before
