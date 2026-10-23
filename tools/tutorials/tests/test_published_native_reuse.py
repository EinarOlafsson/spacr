"""Explicit nonmedia fixtures exercise published-source reuse, never capture proof."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from test_native_live_timing import frozen_native_stage
import published_native_reuse as reuse
from native_live_timing import checked_native_timing
from check_completed_matrix import digest


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value),encoding='utf-8')


@pytest.fixture
def retained_stage(frozen_native_stage,tmp_path,monkeypatch):
    old,oldroot=frozen_native_stage
    stage=tmp_path/'new-stage';root=stage/'production/05_home'
    root.mkdir(parents=True)
    shutil.copytree(old/'catalog',stage/'catalog')
    shutil.copytree(oldroot/'audio',root/'audio')
    shutil.copy2(oldroot/'lesson.en.json',root/'lesson.en.json')
    repo=tmp_path/'canonical';canonical=json.loads((root/'lesson.en.json').read_text())
    write(repo/'tools/tutorials/lessons/05_home.json',canonical)
    monkeypatch.setattr(reuse,'REPO',repo)
    baseline=tmp_path/'baseline'
    master=baseline/'media_host/05_home/video/05_home_silent.mp4'
    master.parent.mkdir(parents=True);master.write_bytes(b'explicit published nonmedia unit fixture')
    timing=master.with_name('native-live-timings.json')
    rows=[{'scene':i+1,'speech_start':i*3,'speech_end':(i+1)*3,'scene_end':(i+1)*3,
           'duration':3,'native_live_video':True,'visual':sc['visual'],
           'text':'Earlier narration, superseded without changing this video.'}
          for i,sc in enumerate(canonical['scenes'])]
    write(timing,{'schema':1,'kind':'native_visual_timing','native_live_video':True,
        'complete_native_clips':True,'lesson':'05_home','fps':30,'total_duration':6,
        'playback_speed':1,'looped_or_stretched':False,'scenes':rows,
        'native_clip_sources':[{'scene':1,'clip_sha256':'original published reference retained'}],
        'coverage':{'accepted':True}})
    records=[{'path':p.relative_to(baseline).as_posix(),'bytes':p.stat().st_size,'sha256':digest(p)}
             for p in (master,timing)]
    manifest=baseline/'release-manifest.json';write(manifest,{'files':records})
    commit='a'*40
    write(baseline/'publication-receipt.json',{'commit':commit,'tag':'explicit-unit-fixture',
       'media_root':'https://huggingface.co/datasets/einarolafsson/spacr-tutorials/resolve/'+commit,
       'manifest_sha256':digest(manifest),'media_files':2,'readback':{'commit':commit,'passed':True,
       'files_expected':2,'metadata_matched':2,'downloaded_sha256_matched':2,
       'metadata_failures':[],'download_failures':[]}})
    return stage,root,baseline,repo


def test_source_current_fifty_tracks_reuse_exact_published_clock_without_fake_frames(retained_stage):
    stage,root,baseline,repo=retained_stage
    path=reuse.prepare(stage,'05_home',baseline)
    result=json.loads(path.read_text())
    assert result['coverage']['track_count']==50
    assert result['coverage']['published_scene_clocks_preserved']
    assert result['scenes'][0]['scene_end']==3
    assert result['scenes'][0]['text']=='First control.'
    assert result['published_native_reuse']['fresh_capture'] is False
    assert result['published_native_reuse']['raw_capture_revalidated'] is False
    assert checked_native_timing(stage,'05_home')==path
    assert not (root/'frames.json').exists()
    assert all('image' not in s and 'clip' not in s for s in json.loads((root/'scenes.json').read_text())['scenes'])
    assert digest(root/'video/05_home_silent.mp4')==digest(baseline/'media_host/05_home/video/05_home_silent.mp4')


@pytest.mark.parametrize('mutation',['baseline_bytes','visual_order','duration_overflow',
    'track_source','canonical_catalog','staged_master','stored_clock','fresh_capture_claim'])
def test_reuse_rejects_changed_sources_or_uncovered_narration(retained_stage,mutation):
    stage,root,baseline,repo=retained_stage
    path=reuse.prepare(stage,'05_home',baseline)
    if mutation=='baseline_bytes':
        p=baseline/'media_host/05_home/video/native-live-timings.json';p.write_text(p.read_text()+' ')
    elif mutation=='visual_order':
        p=root/'lesson.en.json';data=json.loads(p.read_text());data['scenes'].reverse()
        write(p,data);write(repo/'tools/tutorials/lessons/05_home.json',data)
    elif mutation in ('duration_overflow','track_source'):
        p=root/'audio/en/af_heart.json';data=json.loads(p.read_text())
        if mutation=='duration_overflow':data['scenes'][0]['duration']=4
        else:data['scenes'][0]['text']='Wrong current instruction.'
        write(p,data)
    elif mutation=='canonical_catalog':
        p=root/'lesson.en.json';data=json.loads(p.read_text());data['scenes'][0]['narration']='New canonical sentence.'
        write(p,data);write(repo/'tools/tutorials/lessons/05_home.json',data)
    elif mutation=='staged_master':
        # Never modify the hardlinked baseline. Preserve this staged name first.
        p=root/'video/05_home_silent.mp4';p.rename(p.with_suffix('.preserved.mp4'));p.write_bytes(b'changed unit fixture')
    elif mutation=='stored_clock':
        data=json.loads(path.read_text());data['scenes'][0]['scene_end']=2;write(path,data)
    else:
        p=root/reuse.RECORD;data=json.loads(p.read_text());data['fresh_capture']=True;write(p,data)
    with pytest.raises(ValueError):checked_native_timing(stage,'05_home')


def test_preparation_never_overwrites_existing_outputs(retained_stage):
    stage,root,baseline,repo=retained_stage
    reuse.prepare(stage,'05_home',baseline)
    with pytest.raises(FileExistsError):reuse.prepare(stage,'05_home',baseline)


def policy_stage(retained_stage, repeated=False):
    stage,root,baseline,repo=retained_stage
    if repeated:
        canonical=json.loads((root/'lesson.en.json').read_text())
        canonical['scenes'][1]['visual']=canonical['scenes'][0]['visual']
        write(root/'lesson.en.json',canonical);write(repo/'tools/tutorials/lessons/05_home.json',canonical)
        for path in (stage/'catalog').glob('lessons_*.json'):
            catalog=json.loads(path.read_text());catalog['lessons'][0]['scenes'][1]['visual']=canonical['scenes'][0]['visual'];write(path,catalog)
        clock=baseline/'media_host/05_home/video/native-live-timings.json'
        original=json.loads(clock.read_text());original['scenes'][1]['visual']=original['scenes'][0]['visual'];write(clock,original)
        manifest=baseline/'release-manifest.json';m=json.loads(manifest.read_text())
        for record in m['files']:
            if record['path'].endswith('native-live-timings.json'):record.update(bytes=clock.stat().st_size,sha256=digest(clock))
        write(manifest,m);receipt=baseline/'publication-receipt.json';r=json.loads(receipt.read_text());r['manifest_sha256']=digest(manifest);write(receipt,r)
    # Full speech bounds, not scene padding, determine essential-action coverage.
    for path in (root/'audio').rglob('*.json'):
        track=json.loads(path.read_text())
        for index,scene in enumerate(track['scenes']):scene.update(speech_start=index*2,speech_end=index*2+1.5)
        write(path,track)
    current=reuse.prepare(stage,'05_home',baseline)
    original_path=baseline/'media_host/05_home/video/native-live-timings.json'
    original=json.loads(original_path.read_text())
    entry=dict(lesson='05_home',master_sha256=digest(root/'video/05_home_silent.mp4'),visual_timing_sha256=digest(original_path),
        scenes=[dict(visual=s['visual'],source_start=s['speech_start'],source_end=s['scene_end'],idle_tail_start=.2+index*.2)
                for index,s in enumerate(original['scenes'])])
    policy=repo/'tools/tutorials/idle_tail_policy.json';write(policy,dict(schema=1,lessons=[entry]))
    return stage,root,baseline,repo,current,policy,entry


@pytest.mark.parametrize('repeated',[False,True])
def test_reused_policy_changes_only_timing_hash_and_preserves_exact_indexed_fences(retained_stage,repeated):
    stage,root,baseline,repo,current,policy,entry=policy_stage(retained_stage,repeated)
    old_policy=policy.read_bytes();old_record=(root/reuse.RECORD).read_bytes()
    result=reuse.reused_policy(stage,'05_home')
    assert result==root/'reused-idle-tail-policy.json'
    assert json.loads(result.read_text())==dict(schema=1,lessons=[dict(entry,visual_timing_sha256=digest(current))])
    assert policy.read_bytes()==old_policy and (root/reuse.RECORD).read_bytes()==old_record
    assert digest(root/'video/05_home_silent.mp4')==digest(baseline/'media_host/05_home/video/05_home_silent.mp4')
    if repeated:
        fences=json.loads(result.read_text())['lessons'][0]['scenes']
        assert fences[0]['visual']==fences[1]['visual'] and [s['source_start'] for s in fences]==[0,3]
        assert [s['idle_tail_start'] for s in fences]==[.2,.4]


@pytest.mark.parametrize('mutation',['old_timing_hash','old_master_hash','old_fence_clock','missing_fence','shortest_speech','stored_clock','integer_flag','staged_master'])
def test_reused_policy_refuses_stale_source_or_uncovered_action_without_writing(retained_stage,mutation):
    stage,root,baseline,repo,current,policy,entry=policy_stage(retained_stage)
    if mutation=='old_timing_hash':entry['visual_timing_sha256']='f'*64
    elif mutation=='old_master_hash':entry['master_sha256']='f'*64
    elif mutation=='old_fence_clock':entry['scenes'][1].update(source_start=3.1)
    elif mutation=='missing_fence':entry['scenes'].pop()
    elif mutation=='shortest_speech':entry['scenes'][1]['idle_tail_start']=1.6
    elif mutation in ('stored_clock','integer_flag'):
        clock=json.loads(current.read_text());clock['scenes'][1]['native_live_video']=False if mutation=='stored_clock' else 1;write(current,clock)
    else:
        master=root/'video/05_home_silent.mp4';master.rename(master.with_suffix('.preserved.mp4'));master.write_bytes(b'changed staged source')
    write(policy,dict(schema=1,lessons=[entry]))
    with pytest.raises(ValueError):reuse.reused_policy(stage,'05_home')
    assert not (root/'reused-idle-tail-policy.json').exists()


def test_reused_policy_never_overwrites_an_existing_result(retained_stage):
    stage,root,*_=policy_stage(retained_stage)
    result=root/'reused-idle-tail-policy.json';result.write_text('preserve incomplete prior output')
    with pytest.raises(FileExistsError):reuse.reused_policy(stage,'05_home')
    assert result.read_text()=='preserve incomplete prior output'


def test_write_policy_cli_requires_the_actual_reuse_record_baseline(retained_stage,monkeypatch):
    stage,root,*_=policy_stage(retained_stage)
    monkeypatch.setattr(sys,'argv',['published_native_reuse.py','--stage',str(stage),'--lesson','05_home','--baseline',str(stage/'wrong-baseline'),'--write-idle-policy'])
    with pytest.raises(ValueError,match='baseline must match'):reuse.main()
    assert not (root/'reused-idle-tail-policy.json').exists()
