"""The standalone grouped producer shares policy without changing its original analysis."""
import json
from unittest.mock import patch
import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure
from matplotlib.collections import PathCollection
from spacr import plot
from spacr.figures import style,bundle
from spacr.qt import preferences


def _draw(monkeypatch,count,kind='jitter_box',spread='std'):
    """Draw real standalone grouped observations with explicit units/reference and reversed category order."""
    monkeypatch.setattr(preferences,'get_figure_style',dict);monkeypatch.setattr(preferences,'get_figure_style_per_graph',dict)
    frame=pd.DataFrame({'group':np.repeat(['a','b','c'],count),
        'value':np.tile(np.arange(count,dtype=float)+1.,3)+np.repeat([0.,2.,4.],count)})
    frame.attrs.update(units={'value':'a.u.'},reference='a',experimental_unit='well')
    before=frame.copy(deep=True)
    with patch.object(plot,'_show_if_interactive'):
        figure,results=plot.create_grouped_plot(frame,'group','value',graph_type=kind,
            order=['c','a','b'],error_bar_type=spread)
    pd.testing.assert_frame_equal(frame,before,check_exact=True)
    assert figure._spacr_data.to_csv(index=False)==frame.to_csv(index=False)
    assert not results.empty
    return figure,figure.axes[0],results


@pytest.mark.parametrize('count,form',[(1,'bar'),(20,'bar'),(21,'box'),(1001,'violin')])
def test_standalone_grouped_auto_uses_shared_semantics_and_physical_slots(monkeypatch,count,form):
    """The separately implemented public producer must declare three .22-inch slots, real sample counts, explicit grey reference and original automatic forms."""
    figure,ax,results=_draw(monkeypatch,count)
    spec=ax._spacr_style_spec
    assert spec.controls==(1,) and spec.sample_counts==(count,)*3
    assert ax._spacr_condition_kinds==(form,)*3
    assert ax.get_position().width*figure.get_figwidth()==pytest.approx(.66,abs=.5/figure.dpi)
    assert ax.get_title()==''
    assert not style._condition_presentation_problems(ax,spec)
    assert not style.figure_problems(figure)
    assert any('well' in entry['label'] for entry in ax._spacr_legend_entries)
    plot.plt.close(figure)


@pytest.mark.parametrize('kind',['bar','box','violin','jitter'])
def test_standalone_manual_forms_keep_their_type_and_declared_observations(monkeypatch,kind):
    """Explicit forms stay explicit while their already computed artists gain shared widths, colors, fonts and real finite counts."""
    figure,ax,results=_draw(monkeypatch,21,kind)
    spec=ax._spacr_style_spec
    assert dict(spec.comparison)['graph_type']==kind
    assert spec.controls==(1,) and spec.sample_counts==(21,)*3
    assert ax.get_position().width*figure.get_figwidth()==pytest.approx(.66,abs=.5/figure.dpi)
    assert ax.get_title()==''
    if kind=='bar':assert [item.get_width() for item in ax.patches]==pytest.approx([.6]*3,abs=1e-12)
    if kind=='jitter':assert all(np.array_equal(item.get_sizes(),[4.]) for item in ax.collections if isinstance(item,PathCollection) and len(item.get_offsets()))
    plot.plt.close(figure)


@pytest.mark.parametrize('spread',['std','sem'])
def test_standalone_group_line_replay_preserves_original_error_widths_and_centres(monkeypatch,spread):
    """The original line std/sem selection must survive portable replay with exact centre/uncertainty coordinates and explicit reference-marker colours."""
    figure,ax,results=_draw(monkeypatch,21,'line',spread)
    frame,spec=bundle._figure_record(figure)
    assert spec['grouped_line']['spread_func']==spread
    captured=json.loads(json.dumps(bundle._capture_view(figure,spec)))
    namespace={'__name__':'portable_standalone_groups'}
    exec(compile(bundle._recreate_script(captured),'recreate_figure.py','exec',dont_inherit=True),namespace)
    target=Figure();replay=namespace['_draw'](target,frame,captured)
    assert not target._spacr_replay_problems
    assert len(replay.lines)==len(ax.lines) and len(replay.collections)==len(ax.collections)
    for original,copied in zip(ax.lines,replay.lines):np.testing.assert_array_equal(copied.get_xydata(),original.get_xydata())
    for original,copied in zip(ax.collections,replay.collections):
        for expected,actual in zip(original.get_paths(),copied.get_paths()):np.testing.assert_array_equal(actual.vertices,expected.vertices)
        if isinstance(original,PathCollection):np.testing.assert_array_equal(copied.get_offsets(),original.get_offsets())
    assert not style._quantity_unit_problems(ax._spacr_style_spec)
    plot.plt.close(figure);plot.plt.close(target)
