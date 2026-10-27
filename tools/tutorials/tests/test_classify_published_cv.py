"""Synthetic reader/row guards; no published media or capture claims."""
from copy import deepcopy
import json
from pathlib import Path

import pytest
import published_native_mix as mix


@pytest.fixture
def cv_component(tmp_path,monkeypatch):
    repo=tmp_path/'repo';canonical=repo/'tools/tutorials/lessons/41_classify.json'
    canonical.parent.mkdir(parents=True)
    english=dict(id='41_classify',scenes=[dict(visual='cv_24_batch_figure' if i==5 else f'current-{i}') for i in range(12)])
    canonical.write_text(json.dumps(english));monkeypatch.setattr(mix,'REPO',repo)
    baseline=tmp_path/'published';baseline.mkdir()
    for name in ['release-manifest.json','publication-receipt.json']:(baseline/name).write_text('{}')
    original=dict(kind='native_visual_timing',lesson='10_classify_cv',native_live_video=True,
        coverage=dict(accepted=True),source_tracks=[{} for i in range(50)],
        staged_native_edit=dict(fresh_capture=False,raw_capture_revalidated=False),
        scenes=[dict(scene=i+1,visual='gui_24_batch_figure' if i==9 else f'source-{i}',
                     speech_start=i*10,scene_end=(i+1)*10,duration=10,native_live_video=True) for i in range(11)])
    original['scenes'][9].update(edited=True,source_keep_frames=[[30,165]],kept_frames=135,final_frame_hold_frames=165)
    bindings={key:dict(path=f'media_host/10_classify_cv/video/{name}',sha256=sha,bytes=123)
              for key,name,sha in [('master','master.mp4','a'*64),('timing','clock.json','b'*64)]}
    policy=dict(schema=1,lessons=[dict(lesson='10_classify_cv',master_sha256='a'*64,visual_timing_sha256='b'*64,
        scenes=[dict(visual='gui_24_batch_figure',source_start=90,source_end=100,idle_tail_start=4.5)])])
    tracks=[dict(scenes=[dict(speech_start=0,speech_end=6,duration=7) for _ in range(12)]) for _ in range(50)]
    calls=[]
    import build_audio_patch_candidate
    def published(*args):calls.append('published-complete')
    monkeypatch.setattr(build_audio_patch_candidate,'require_published_script_baseline',published)
    def inputs(path,identity):
        assert calls==['published-complete'] and identity=='10_classify_cv'
        return baseline,original,bindings,{}
    monkeypatch.setattr(mix,'_published_inputs',inputs)
    return baseline,english,tracks,policy,original,calls


def test_published_cv_reader_plans_only_accepted_motion_and_result_hold(cv_component):
    baseline,english,tracks,policy,original,calls=cv_component
    before=deepcopy((tracks,original))
    component,row=mix.classify_overview_cv_interval(baseline,english,tracks,source_policy=policy)
    assert calls==['published-complete']
    assert row['source_start_frame']==2700 and row['source_keep_frames']==[[0,135]]
    assert row['kept_frames']==135 and row['final_frame_hold_frames']==75 and row['duration']==7
    assert component['source_scene']==10 and component['visual']=='cv_24_batch_figure'
    assert component['fresh_capture'] is component['raw_capture_revalidated'] is False
    assert (tracks,original)==before


@pytest.mark.parametrize('fault',['unpublished','wrong_raw_excerpt','wrong_fence','one_short_voice','foreign_target'])
def test_published_cv_reader_fails_closed_before_any_encoding(cv_component,monkeypatch,fault):
    baseline,english,tracks,policy,original,calls=cv_component
    if fault=='unpublished':
        import build_audio_patch_candidate
        def rejected(*args):raise ValueError('No normal full published checkpoint')
        monkeypatch.setattr(build_audio_patch_candidate,'require_published_script_baseline',rejected)
    elif fault=='wrong_raw_excerpt':original['scenes'][9]['source_keep_frames']=[[0,300]]
    elif fault=='wrong_fence':policy['lessons'][0]['scenes'][0]['idle_tail_start']=5
    elif fault=='one_short_voice':tracks[-1]['scenes'][5]['speech_end']=4.49
    else:english['scenes'][5]['visual']='another-result'
    with pytest.raises(ValueError):
        mix.classify_overview_cv_interval(baseline,english,tracks,source_policy=policy)
