"""Saved legend and figure-note appearance stays exact without changing source marks."""
import json
import numpy as np
import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from spacr.figures import style,bundle


def _scene():
    """Create the same literal line, point and bar data for source and replay."""
    from spacr.figure_font import use_open_sans_for_figures

    use_open_sans_for_figures()
    figure=Figure(figsize=(4.,3.),dpi=150);FigureCanvasAgg(figure);ax=figure.subplots()
    line,=ax.plot([0.,1.],[2.,3.],color='#0A77B8',alpha=.6,label='Line')
    points=ax.scatter([.2,.8],[2.3,2.7],s=9.,color='#8C8C8C',alpha=.6,label='Points')
    bars=ax.bar([.5],[2.4],width=.2,color='#0A9878',alpha=.6,label='Bar')
    return figure,ax,[line,points,bars]


def _coordinates(ax):
    """Copy every original numerical coordinate independently of legend artists."""
    return ([line.get_xydata().copy() for line in ax.lines],
        [item.get_offsets().copy() for item in ax.collections],
        [(item.get_x(),item.get_y(),item.get_width(),item.get_height()) for item in ax.patches])


@pytest.mark.parametrize('portable',[False,True])
def test_axis_legend_replay_keeps_manual_keys_fonts_frame_and_external_position(portable):
    """Retain actual Line2D/PathCollection/Rectangle keys, .35 manual alpha, text styling and external placement; allow half a pixel only for renderer bounds."""
    figure,ax,handles=_scene()
    legend=ax.legend(handles,['Curve','Observations','Estimate'],loc='upper left',bbox_to_anchor=(1.02,1.),
        title='Measured groups',frameon=True,ncols=1,fontsize=9.,title_fontsize=10.,markerscale=1.3,
        borderpad=.7,labelspacing=.8,handletextpad=.9)
    legend.get_frame().set_facecolor('#EEEFFF');legend.get_frame().set_edgecolor('purple');legend.get_frame().set_alpha(.7)
    for index,text in enumerate(legend.get_texts()):
        text.set_fontfamily('Open Sans');text.set_fontweight('light');text.set_color(['black','purple','navy'][index])
    legend.get_title().set_fontfamily('Open Sans');legend.get_title().set_fontweight('regular')
    for handle in legend.legend_handles:handle.set_alpha(.35)
    figure.canvas.draw();before=_coordinates(ax)
    expected=[style._artist_appearance(item) for item in legend.legend_handles]
    bounds=legend.get_window_extent().bounds
    captured=json.loads(json.dumps(style._capture_appearance(figure)))
    assert captured['axes'][0]['legend']['labels']==['Curve','Observations','Estimate']
    target,replay,_=_scene()
    if portable:
        namespace={'__name__':'standalone_legend'}
        exec(compile(bundle._appearance_script(),'appearance.py','exec',dont_inherit=True),namespace)
        problems=namespace['_restore_appearance'](target,captured)
    else:problems=style._restore_appearance(target,captured)
    assert not problems
    target.canvas.draw();actual=replay.get_legend()
    assert actual is not None
    assert [text.get_text() for text in actual.get_texts()]==['Curve','Observations','Estimate']
    assert [style._artist_appearance(item) for item in actual.legend_handles]==expected
    for original,copied in zip(legend.legend_handles,actual.legend_handles):
        np.testing.assert_array_equal(copied.get_transform().get_matrix(),original.get_transform().get_matrix())
    assert [style._text_appearance(text) for text in actual.get_texts()]==[style._text_appearance(text) for text in legend.get_texts()]
    assert style._text_appearance(actual.get_title())==style._text_appearance(legend.get_title())
    assert style._artist_appearance(actual.get_frame())==style._artist_appearance(legend.get_frame())
    assert actual.get_window_extent().bounds==pytest.approx(bounds,abs=.5)
    after=_coordinates(replay)
    for source,copied in zip(before[:2],after[:2]):
        assert len(source)==len(copied)
        for expected_values,actual_values in zip(source,copied):np.testing.assert_array_equal(actual_values,expected_values)
    assert after[2]==before[2]


def test_figure_legend_and_notes_replay_in_the_original_order():
    """Global notes and shared keys keep their real figure coordinates, order and manual fonts instead of disappearing from an otherwise successful replay."""
    figure,ax,handles=_scene()
    figure.legend(handles[:2],['Shared curve','Shared observations'],loc='lower center',bbox_to_anchor=(.5,1.02),ncols=2,frameon=False)
    note=figure.text(.2,.03,'A fixed figure note',ha='left',va='bottom',rotation=7.,fontfamily='Open Sans',fontsize=11.,fontweight='semibold',color='purple')
    note.set_alpha(.8)
    captured=json.loads(json.dumps(style._capture_appearance(figure)))
    assert len(captured['figure_legends'])==1 and len(captured['figure_texts'])==1
    target,replay,_=_scene()
    assert not style._restore_appearance(target,captured)
    assert len(target.legends)==1 and len(target.texts)==1
    assert [text.get_text() for text in target.legends[0].get_texts()]==['Shared curve','Shared observations']
    assert style._text_appearance(target.texts[0])==style._text_appearance(note)
    np.testing.assert_allclose(target.texts[0].get_position(),note.get_position(),rtol=0,atol=1e-14)
    assert _coordinates(replay)[2]==_coordinates(ax)[2]


def test_unsupported_custom_key_is_reported_without_adding_data_artists():
    """Composite error-bar legend keys need a dedicated vector replay; an unsupported primary handle must produce an explicit finding, with no fake data artists."""
    figure,ax,_=_scene()
    ax.errorbar([.3],[2.5],yerr=.2,fmt='o',capsize=2.,label='Interval')
    ax.legend()
    captured=json.loads(json.dumps(style._capture_appearance(figure)))
    target,replay,_=_scene()
    replay.errorbar([.3],[2.5],yerr=.2,fmt='o',capsize=2.,label='Interval')
    before=_coordinates(replay)
    problems=style._restore_appearance(target,captured)
    assert 'Captured appearance does not support this custom legend key' in problems
    after=_coordinates(replay)
    assert len(after[0])==len(before[0]) and len(after[1])==len(before[1]) and after[2]==before[2]
    for original,restored in zip(before[0],after[0]):np.testing.assert_array_equal(restored,original)
