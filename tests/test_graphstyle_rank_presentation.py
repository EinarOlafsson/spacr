"""The approved native rank form retains exact estimates and physical category slots."""
import numpy as np
import pandas as pd
import pyqtgraph as pg
import pytest
from spacr.figures import style
from spacr.qt.widgets.fast_plots import EffectRankPlot

pytestmark = pytest.mark.qt


@pytest.mark.parametrize('count', [1, 2, 3])
def test_native_rank_short_category_axes_keep_slots_and_put_the_default_title_in_the_caption(qtbot, count):
    """The measured 'Effect rank' title imposed a 62.78125-pixel minimum, stretching one/two .22-inch slots to .65397 inches at 96 dpi; require the actual slot width within half a logical pixel, with exact source estimates and intervals."""
    frame = pd.DataFrame(dict(gene=[str(i) for i in range(count)], coefficient=np.arange(count)+1.,
        std_err=np.full(count, .1), q_value=np.full(count, .1)))
    widget = EffectRankPlot()
    qtbot.addWidget(widget)
    widget.resize(720, 480)
    widget.set_results(frame, label_column='gene', drop_untested=False)
    widget.show()
    qtbot.waitExposed(widget)
    source = widget.frame().to_csv(index=False)
    with widget._held_at_policy_geometry():
        assert widget.plot.getViewBox().geometry().width()/widget.logicalDpiX() == pytest.approx(count*.22, abs=.5/widget.logicalDpiX())
    assert not widget.plot.plotItem.titleLabel.text
    assert 'Effect rank' in dict(widget._spacr_style_spec.legend)['caption']
    points = widget._scatter_items()[0].points()
    assert all(point.size()**2*(72./widget.logicalDpiX())**2 == pytest.approx(4.) for point in points)
    reference = next(item for item in widget.plot.plotItem.items if isinstance(item, pg.InfiniteLine))
    assert reference.pen.color().name() == '#000000'
    assert reference.pen.widthF()*72/widget.logicalDpiX() == pytest.approx(style.WEIGHTS['reference'])
    assert reference.pen.color().alphaF()*reference.opacity() == pytest.approx(.6, abs=1/65535)
    np.testing.assert_array_equal(widget._effects, frame.coefficient)
    np.testing.assert_array_equal(widget._half, 1.96*frame.std_err)
    assert widget.frame().to_csv(index=False) == source


def test_manual_native_rank_title_remains_and_is_reported_by_the_shared_finalizer(qtbot):
    """An explicit in-graph title remains exact across redraws and validation; a compliance result must report that manual choice instead of silently moving or deleting it."""
    widget = EffectRankPlot()
    qtbot.addWidget(widget)
    widget.resize(720, 480)
    frame = pd.DataFrame(dict(gene=['a','b','c'], coefficient=[1.,2.,3.], std_err=[.1,.2,.3], q_value=[.1,.2,.3]))
    widget.set_results(frame, label_column='gene', drop_untested=False)
    widget.plot.setTitle('Explicit chart title')
    widget.set_results(frame, label_column='gene', drop_untested=False)
    source = widget.frame().to_csv(index=False)
    result = style.finalize_figure(widget, diagnostics=False)
    assert 'Move the plot title into the figure legend' in result.problems
    assert widget.plot.plotItem.titleLabel.text == 'Explicit chart title'
    assert widget.frame().to_csv(index=False) == source
