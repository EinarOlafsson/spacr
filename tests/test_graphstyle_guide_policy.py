"""Guide-count relationships retain every source value and explicit opening-view cutoff."""
import numpy as np
import pandas as pd
import pytest

pytestmark=pytest.mark.qt


@pytest.mark.parametrize('outlier',[False,True])
def test_guide_agreement_preserves_quantitative_coordinates_keys_and_declares_its_opening_view(qtbot,monkeypatch,outlier):
    """Keep the original numeric guide-count jitter, agreement fractions, source rows and99th-percentile opening cutoff while registering the quantitative chart, .6 marks and explicit0.5 reference."""
    from spacr import graph_types
    from spacr.figures import style
    from spacr.qt.widgets.fast_plots import GuideAgreementPlot

    monkeypatch.setattr(graph_types,'chosen_for',lambda _shape:'')
    frame=pd.DataFrame({'feature':[f'gene_{i}' for i in range(80)],
        'n_guides':np.resize([1.,2.,3.,4.],80),'concordance':np.linspace(.2,.95,80)})
    if outlier:frame.loc[79,'n_guides']=1000.
    frame['single_guide']=frame.n_guides==1.
    before=frame.to_csv(index=False)
    widget=GuideAgreementPlot();qtbot.addWidget(widget)
    assert widget.set_support(frame)==len(frame)
    widget.resize(760,560);widget.show();qtbot.waitExposed(widget)
    assert widget.mark()=='jitter'
    expected=frame.n_guides.to_numpy()+np.random.default_rng(0).uniform(-.22,.22,len(frame))
    np.testing.assert_array_equal([widget._row_xy[i][0] for i in range(len(frame))],expected)
    np.testing.assert_array_equal([widget._row_xy[i][1] for i in range(len(frame))],frame.concordance)
    assert list(widget._keys)==frame.feature.tolist()
    assert widget._spacr_style_spec.kind=='continuous'
    assert not style._quantity_unit_problems(widget._spacr_style_spec)
    with widget._held_at_policy_geometry() as applied:
        assert applied
        observed=widget._spacr_style_observation()
        assert observed['width_inches']==pytest.approx(observed['height_inches'],abs=.5/widget.logicalDpiX())
        assert all(mark['role']!='unregistered' for mark in observed['marks'])
        assert all(colour[3]*mark['opacity']==pytest.approx(.6,abs=1/255) for mark in observed['marks'] for colour in mark['colours'])
        assert not any('viewport' in value for value in style.finalize_figure(widget).problems)
    comparison=dict(widget._spacr_style_spec.comparison)
    if outlier:
        assert comparison['display_range_reason'] and comparison['hidden_point_count']==1
        assert any('99' in entry['label'] for entry in widget._spacr_legend_entries)
    widget.plot.setXRange(1.5,2.5,padding=0.)
    manual=widget.plot.getViewBox().viewRange()
    assert any('viewport' in value for value in style.finalize_figure(widget).problems)
    assert widget.plot.getViewBox().viewRange()==manual
    assert frame.to_csv(index=False)==before
