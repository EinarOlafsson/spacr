"""Explicit synthetic observations test the actual normal boundary validator."""
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from verify_staged_lesson import check_complete_native_trace


def observation(final=False,seek_at=None):
    seek_at=(1.82 if final else 1.72) if seek_at is None else seek_at
    visual={'scenes':[{'speech_start':0,'scene_end':4,'duration':4,'native_live_video':True,'idle_tail_start':.9},
                       {'speech_start':4,'scene_end':8,'duration':4,'native_live_video':True,'idle_tail_start':.9}]}
    spoken={'scenes':[{'speech_start':0,'speech_end':1,'scene_end':1.52,'sentences':[{'audible_end':.82}]},
                       {'speech_start':1.52,'speech_end':2.52,'scene_end':3.04,'sentences':[{'audible_start':1.62,'audible_end':2.34}]}]}
    trace=[]
    for k in range(7,65):
        wall=k*.05
        seeking=not final and seek_at<=wall<seek_at+.05
        ended=final and wall>=seek_at
        if wall<seek_at:
            video=wall;audio=min(wall,1);paused=wall>=1
        elif final:
            video=4;audio=1;paused=True
        else:
            video=4+max(0,wall-seek_at-.05)
            audio=1.52+max(0,wall-seek_at-.05);paused=seeking
        trace.append(dict(wall=wall*1000,audio=audio,video=video,rate=1,audioRate=1,
                          audioPaused=paused,videoPaused=ended,videoEnded=ended,
                          seeking=seeking,correcting=seeking,error=None))
    events=[{'type':'seeking','video':4,'wall':seek_at*1000,'audio':1}]
    return trace,events,visual,spoken


def test_real_validator_accepts_measured_one_second_idle_transition():
    args=observation();r=check_complete_native_trace(*args,0)
    assert r['actual_sound_gap_seconds']==pytest.approx(1.05)
    assert r['idle_padding_skipped'] and not r['entire_recorded_tail_played']


def test_real_validator_requires_actual_final_eof_after_trailing_pause():
    args=observation(final=True);r=check_complete_native_trace(*args,0,True)
    assert r['actual_sound_gap_seconds']==pytest.approx(1.03,abs=.03)
    assert r['final_native_eof_observed']


@pytest.mark.parametrize('mutation',['missing_seek','wrong_destination','unplanned_event',
    'unfinished_speech','short_gap','long_gap','speed','error','midscene_jump','missing_eof'])
def test_malformed_or_unplanned_transition_never_passes(mutation):
    final=mutation=='missing_eof'
    trace,events,visual,spoken=observation(final=final,
        seek_at=1.2 if mutation=='short_gap' else 2 if mutation=='long_gap' else None)
    if mutation=='missing_seek':events=[]
    elif mutation=='wrong_destination':events[0]['video']=5
    elif mutation=='unplanned_event':events.append({'type':'seeking','video':.8,'wall':500})
    elif mutation=='unfinished_speech':
        for row in trace:
            if row['video']<4:row['audio']=min(row['audio'],.9)
    elif mutation=='speed':trace[10]['audioRate']=1.1
    elif mutation=='error':trace[10]['error']='decode failed'
    elif mutation=='midscene_jump':
        for row in trace[5:]:
            if row['video']<4:row['video']+=.8
    elif mutation=='missing_eof':
        for row in trace:row['videoEnded']=False
    with pytest.raises(AssertionError):check_complete_native_trace(trace,events,visual,spoken,0,final)


def test_complete_actual_transition_event_covers_a_missed_periodic_end_sample():
    trace,events,v,a=observation()
    for row in trace:
        if row['video']<4:row['audio']=min(row['audio'],.99)
    events.append({'type':'idle-cut','audio':1.001,'video':1.72,'wall':1720})
    assert check_complete_native_trace(trace,events,v,a,0)['idle_padding_skipped']


def test_incomplete_actual_transition_event_never_authorizes_a_cut():
    trace,events,v,a=observation()
    for row in trace:
        if row['video']<4:row['audio']=min(row['audio'],.9)
    events.append({'type':'idle-cut','audio':.9,'video':1.72,'wall':1720})
    with pytest.raises(AssertionError):check_complete_native_trace(trace,events,v,a,0)


def test_picture_can_preseek_then_hold_during_the_one_second_quiet_window():
    trace,events,v,a=observation()
    for row in trace:
        wall=row['wall']/1000
        if wall>=1:
            row['video']=4+max(0,wall-1.72)
            row['audio']=1.52+max(0,wall-1.72)
            row['videoPaused']=row['audioPaused']=wall<1.72
            row['seeking']=row['correcting']=1<=wall<1.05
    events=[{'type':'idle-cut','audio':1,'video':1,'wall':1000},
            {'type':'seeking','audio':1.52,'video':4,'wall':1000},
            {'type':'pause','video':1,'wall':1000}]
    result=check_complete_native_trace(trace,events,v,a,0)
    assert result['actual_sound_gap_seconds']==pytest.approx(1)


@pytest.mark.parametrize('position,accepted',[(1.15,True),(.9,False),(1.49,False),(1.7,False)])
def test_elastic_rate_is_allowed_only_in_proven_interclip_quiet(position,accepted):
    trace,events,v,a=observation()
    quiet=dict(trace[20],audio=position,audioRate=.7)
    trace[20]=quiet
    if accepted:
        assert check_complete_native_trace(trace,events,v,a,0)['idle_padding_skipped']
    else:
        with pytest.raises(AssertionError):check_complete_native_trace(trace,events,v,a,0)


def test_unproven_idle_tail_must_remain_complete_and_cannot_seek():
    trace,events,v,a=observation()
    del v['scenes'][0]['idle_tail_start']
    with pytest.raises(AssertionError):check_complete_native_trace(trace,events,v,a,0)
    trace,events,v,a=observation(final=True,seek_at=4)
    del v['scenes'][0]['idle_tail_start'];events=[]
    trace.append(dict(trace[-1],wall=4000,video=4,videoPaused=True,videoEnded=True))
    result=check_complete_native_trace(trace,events,v,a,0,True)
    assert result['entire_recorded_tail_played'] and not result['idle_padding_skipped']


def test_known_fence_after_complete_speech_keeps_full_source_tail():
    trace,events,v,a=observation(final=True,seek_at=4)
    v['scenes'][0]['idle_tail_start']=1.6;events=[]
    trace.append(dict(trace[-1],wall=4000,video=4,videoPaused=True,videoEnded=True))
    result=check_complete_native_trace(trace,events,v,a,0,True)
    assert result['entire_recorded_tail_played'] and not result['idle_padding_skipped']
