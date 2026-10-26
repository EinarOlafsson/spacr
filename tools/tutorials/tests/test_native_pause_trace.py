"""Explicit synthetic observations test the actual normal boundary validator."""
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from verify_staged_lesson import (check_complete_native_trace, staged_idle_binding,
                                  require_loaded_idle_timings)
from stage_lesson import write
from copy import deepcopy
import hashlib


@pytest.fixture
def idle_sources(tmp_path):
    master, clock = tmp_path / 'master.mp4', tmp_path / 'clock.json'
    master.write_bytes(b'accepted native master')
    timing = dict(lesson='74_import_images', kind='native_visual_timing', native_live_video=True,
                  coverage=dict(accepted=True), scenes=[dict(scene=i + 1, visual='repeated',
                  speech_start=i * 4, scene_end=(i + 1) * 4, duration=4, native_live_video=True)
                  for i in range(2)])
    write(clock, timing)
    lesson = dict(id='74_import_images', silent='74_import_images/video/master.mp4',
                  visual_timings='74_import_images/video/clock.json')
    policy = dict(schema=1, lessons=[dict(lesson=lesson['id'],
        master_sha256=hashlib.sha256(master.read_bytes()).hexdigest(),
        visual_timing_sha256=hashlib.sha256(clock.read_bytes()).hexdigest(),
        scenes=[dict(visual='repeated', source_start=i * 4, source_end=(i + 1) * 4,
                     idle_tail_start=.8) for i in range(2)])])
    spoken = dict(scenes=[dict(speech_start=i * 2, speech_end=i * 2 + 1) for i in range(2)])
    return lesson, master, clock, policy, spoken


def test_staged_policy_binds_repeated_visuals_to_exact_master_and_clocks(idle_sources):
    lesson, master, clock, policy, spoken = idle_sources
    before = deepcopy(policy)
    bound, timing = staged_idle_binding(lesson, master, clock, policy, required=True)
    assert bound['native_idle_tails']['scenes'] == policy['lessons'][0]['scenes']
    assert [scene['idle_tail_start'] for scene in timing['scenes']] == [.8, .8]
    require_loaded_idle_timings(deepcopy(timing), timing, spoken)
    assert policy == before and 'native_idle_tails' not in lesson


@pytest.mark.parametrize('source', ['master', 'clock'])
def test_stale_default_policy_is_rejected_not_silently_omitted(idle_sources, source):
    lesson, master, clock, policy, _ = idle_sources
    field = 'master_sha256' if source == 'master' else 'visual_timing_sha256'
    policy['lessons'][0][field] = '0' * 64
    with pytest.raises(ValueError, match='source bytes changed'):
        staged_idle_binding(lesson, master, clock, policy)


def test_strict_idle_acceptance_requires_all_reviewed_scene_fences(idle_sources):
    lesson, master, clock, policy, _ = idle_sources
    policy['lessons'][0]['scenes'].pop()
    with pytest.raises(ValueError, match='every scene'):
        staged_idle_binding(lesson, master, clock, policy, required=True)


@pytest.mark.parametrize('change', ['loaded_fence', 'short_speech'])
def test_actual_loaded_fences_and_selected_full_speech_are_required(idle_sources, change):
    lesson, master, clock, policy, spoken = idle_sources
    _, expected = staged_idle_binding(lesson, master, clock, policy, required=True)
    actual = deepcopy(expected)
    if change == 'loaded_fence':
        actual['scenes'][0].pop('idle_tail_start')
    else:
        spoken['scenes'][0]['speech_end'] = .5
    with pytest.raises(ValueError):
        require_loaded_idle_timings(actual, expected, spoken)


def test_strict_gap_rejects_unplanned_long_tail_which_old_scope_retains():
    visual = dict(scenes=[dict(speech_start=0, scene_end=4, duration=4),
                          dict(speech_start=4, scene_end=8, duration=4)])
    spoken = dict(scenes=[dict(speech_start=0, speech_end=1, sentences=[dict(audible_end=.82)]),
                          dict(speech_start=1.72, speech_end=2.72,
                               sentences=[dict(audible_start=1.82)])])
    rows = []
    for tick in range(110):
        now = tick * .05
        rows.append(dict(wall=now * 1000, video=now, audio=min(now, 1) if now < 4 else 1.72 + now - 4,
                         rate=1, audioRate=1, audioPaused=1 <= now < 4, videoPaused=False,
                         videoEnded=False, seeking=False, correcting=False, error=None))
    assert check_complete_native_trace(rows, [], visual, spoken, 0)['actual_sound_gap_seconds'] > 3
    with pytest.raises(AssertionError, match='actual_sound_gap_outside_policy'):
        check_complete_native_trace(rows, [], visual, spoken, 0, require_gap=True)


def observation(final=False,seek_at=None):
    seek_at=(1.82 if final else 1.72) if seek_at is None else seek_at
    visual={'scenes':[{'speech_start':0,'scene_end':4,'duration':4,'native_live_video':True,'idle_tail_start':.9},
                       {'speech_start':4,'scene_end':8,'duration':4,'native_live_video':True,'idle_tail_start':.9}]}
    spoken={'scenes':[{'speech_start':0,'speech_end':1,'scene_end':1.72,'sentences':[{'audible_end':.82}]},
                       {'speech_start':1.72,'speech_end':2.72,'scene_end':3.24,'sentences':[{'audible_start':1.82,'audible_end':2.54}]}]}
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
            audio=1.72+max(0,wall-seek_at-.05);paused=seeking
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
            row['audio']=1.72+max(0,wall-1.72)
            row['videoPaused']=row['audioPaused']=wall<1.72
            row['seeking']=row['correcting']=1<=wall<1.05
    events=[{'type':'idle-cut','audio':1,'video':1,'wall':1000},
            {'type':'seeking','audio':1.72,'video':4,'wall':1000},
            {'type':'pause','video':1,'wall':1000}]
    result=check_complete_native_trace(trace,events,v,a,0)
    assert result['actual_sound_gap_seconds']==pytest.approx(1)


@pytest.mark.parametrize('position,accepted',[(1.15,True),(.9,False),(1.69,False),(1.7,False)])
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


@pytest.mark.parametrize('source_gap,actual_gap,accepted',[
    (.8,.66,True),(.8,.8,True),(.8,.94,True),
    (.8,.64,False),(.8,.96,False),(.8,1.1,False),
    (1.0,.8,False),(1.4,.8,False),(1.4,1.1,True)])
def test_native_gap_preserves_short_source_pause_and_shortens_only_long_pauses(source_gap,actual_gap,accepted):
    trace,events,visual,spoken=observation(seek_at=actual_gap+.67)
    # Actual continuous synthetic audio observations, with the following clip
    # shifted together with its metadata; the measured wall gap is independent.
    delta=source_gap-1.0
    following=spoken['scenes'][1]
    for key in ('speech_start','speech_end','scene_end'):following[key]+=delta
    for key in ('audible_start','audible_end'):following['sentences'][0][key]+=delta
    spoken['scenes'][0]['scene_end']+=delta
    for row in trace:
        if row['video']>=4:row['audio']+=delta
    if accepted:
        result=check_complete_native_trace(trace,events,visual,spoken,0)
        assert result['actual_sound_gap_seconds']==pytest.approx(actual_gap,abs=.015)
    else:
        with pytest.raises(AssertionError,match='actual_sound_gap_outside_policy'):
            check_complete_native_trace(trace,events,visual,spoken,0)


def test_final_eof_keeps_one_second_policy_without_a_following_sound():
    trace,events,visual,spoken=observation(final=True)
    # The source trailing PCM is only .18s; final picture EOF still follows
    # the existing one-second final-cut rule, not an interscene minimum.
    assert spoken['scenes'][0]['speech_end']-spoken['scenes'][0]['sentences'][-1]['audible_end']==pytest.approx(.18)
    assert check_complete_native_trace(trace,events,visual,spoken,0,True)['final_native_eof_observed']
    trace,events,visual,spoken=observation(final=True,seek_at=1.62)
    with pytest.raises(AssertionError,match='actual_sound_gap_outside_policy'):
        check_complete_native_trace(trace,events,visual,spoken,0,True)
