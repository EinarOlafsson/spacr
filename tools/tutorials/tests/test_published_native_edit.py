"""Published-source planning contracts and one tiny real 1x frame-edit regression."""
from copy import deepcopy
from pathlib import Path
import pytest
import published_native_edit as edit


@pytest.fixture
def accepted_staged_source(tmp_path,monkeypatch,source_plan):
    """Synthetic normal acceptance inputs; no media decode or capture claims."""
    import native_live_timing,check_completed_matrix
    from stage_lesson import write
    import hashlib,json
    english,original,tracks,plan,_=deepcopy(source_plan)
    identity=english['id'];source=tmp_path/'accepted';target=tmp_path/'edited';repo=tmp_path/'repo'
    root=source/'production'/identity;out=target/'production'/identity
    proofs=[dict(language='en',voice=f'fixture{i}',audio_sha256=str(i),timing_sha256=str(i)) for i in range(50)]
    original.update(lesson=identity,kind='native_visual_timing',native_live_video=True,
                    coverage=dict(accepted=True),source_tracks=proofs)
    write(root/'lesson.en.json',english);write(out/'lesson.en.json',english)
    write(repo/'tools/tutorials/lessons'/f'{identity}.json',english)
    write(root/'scenes.json',dict(scenes=[dict(narration=s['narration']) for s in english['scenes']]))
    write(root/'visual.json',dict(scenes=[]))
    clock=root/'video/native-live-timings.json';write(clock,original)
    master=root/'video'/f'{identity}_silent.mp4';master.write_bytes(b'synthetic accepted master, not media')
    capture=tmp_path/'original-capture'/'captures'/'mask';capture.mkdir(parents=True)
    for name in ['frames.json','provenance.json','host_gate_summary.json']:
        write(capture/name,{'synthetic':True})
    write(capture.parent.parent/'exact-source.json',dict(sha256={'driver.py':'fixture'}))
    write(source/'build-module-stage-report.json',dict(lesson=identity,sources=[str(capture)]))
    write(root/'current-video-acceptance.json',dict(lesson=identity,master_sha256=edit.digest(master),
        timing_sha256=edit.digest(clock),scene_specification_sha256=edit.digest(root/'scenes.json'),
        canonical_english_sha256=hashlib.sha256(json.dumps(english,sort_keys=True,ensure_ascii=False).encode()).hexdigest(),
        native_fifty_voice_timeline=True,master_full_decode_passed=True,
        master_frame_count_matches_scene_boundaries=True,master_all_presentation_times_checked=True))
    from audit_staged_catalogs import CATALOGS
    for name in CATALOGS:
        for base in [source,target]:write(base/'catalog'/name,dict(lessons=[english]))
    monkeypatch.setattr(edit,'REPO',repo)
    monkeypatch.setattr(native_live_timing,'checked_native_timing',lambda stage,identity:clock)
    monkeypatch.setattr(native_live_timing,'verified_narration_tracks',lambda *args:(tracks,proofs))
    monkeypatch.setattr(check_completed_matrix,'reconcile_visual_sources',lambda *args:None)
    # This source branch must never call the published-baseline loader.
    monkeypatch.setattr(edit,'_published_inputs',lambda *args:pytest.fail('Staged source is not published'))
    _,_,bindings,_=edit._staged_inputs(source,identity)
    plan.update(master_sha256=bindings['master']['sha256'],visual_timing_sha256=bindings['timing']['sha256'])
    policy=dict(schema=1,lessons=[dict(lesson=identity,master_sha256=bindings['master']['sha256'],
        visual_timing_sha256=bindings['timing']['sha256'],scenes=[dict(visual='untouched',
        source_start=10,source_end=20,idle_tail_start=1.5)])])
    policy_path=tmp_path/'reviewed-source-policy.json';write(policy_path,policy)
    return source,target,identity,plan,policy_path,capture


def test_staged_source_keeps_untouched_animation_and_generates_rebound_fences(accepted_staged_source):
    from stage_lesson import write,read
    source,target,identity,plan,policy,capture=accepted_staged_source
    root,_,bindings,provenance,spec,timing=edit.inputs(target,identity,None,plan,
        staged_source=source,source_idle_policy=policy)
    assert spec['kind']=='edited_staged_native_intervals'
    assert 'baseline_root' not in provenance and provenance['source_stage_root']==str(source)
    assert any(row['path']==str(capture/'provenance.json') for row in provenance['source_files'])
    first,untouched=timing['scenes']
    assert first['kept_frames']==60 and untouched['duration']==10 and not untouched['edited']
    assert untouched['source_keep_frames']==[[0,300]]
    master=root/'video'/f'{identity}_silent.mp4';master.parent.mkdir(parents=True);master.write_bytes(b'edited fixture')
    write(root/'video/native-live-timings.json',timing)
    original=read(source/bindings['timing']['path'])
    generated=edit.edited_policy(root,identity,timing,bindings,original=original,policy=read(policy))
    assert [row['idle_tail_start'] for row in generated['lessons'][0]['scenes']]==[2,1.5]
    assert generated['lessons'][0]['master_sha256']==edit.digest(master)
    assert generated['lessons'][0]['visual_timing_sha256']==edit.digest(root/'video/native-live-timings.json')


@pytest.mark.parametrize('fault',['finish_master','finish_incomplete','missing_capture','missing_fence',
                                  'late_fence','audio_changed','catalog_changed','same_stage'])
def test_staged_edit_rejects_unbound_source_or_unsafe_fences_before_encoding(accepted_staged_source,monkeypatch,fault):
    from stage_lesson import read,write
    import native_live_timing
    source,target,identity,plan,policy,capture=accepted_staged_source
    root=source/'production'/identity
    if fault.startswith('finish'):
        path=root/'current-video-acceptance.json';data=read(path)
        if fault=='finish_master':data['master_sha256']='0'*64
        else:data['master_full_decode_passed']=False
        write(path,data)
    elif fault=='missing_capture':
        (capture/'provenance.json').rename(capture/'provenance.held.json')
    elif fault in ['missing_fence','late_fence']:
        data=read(policy)
        if fault=='missing_fence':data['lessons'][0]['scenes']=[]
        else:data['lessons'][0]['scenes'][0]['idle_tail_start']=2.1
        write(policy,data)
    elif fault=='audio_changed':
        original=native_live_timing.verified_narration_tracks
        tracks,proofs=deepcopy(original(None,None,None));proofs[0]['audio_sha256']='changed'
        monkeypatch.setattr(native_live_timing,'verified_narration_tracks',lambda *args:(tracks,proofs))
    elif fault=='catalog_changed':
        path=target/'catalog/lessons_en.json';data=read(path);data['unrelated']='change';write(path,data)
    else:target=source
    with pytest.raises(ValueError):
        edit.inputs(target,identity,None,plan,staged_source=source,source_idle_policy=policy)
    assert not (target/'production'/identity/'published-edit-render').exists()

@pytest.fixture
def source_plan():
    english={'id':'07_mask','scenes':[{'visual':'organize','narration':'Choose real fields.'},
                                   {'visual':'untouched','narration':'Next real action.'}]}
    original={'scenes':[{'visual':'organize','speech_start':0,'scene_end':10,'duration':10,'native_live_video':True},
                        {'visual':'untouched','speech_start':10,'scene_end':20,'duration':10,'native_live_video':True}]}
    tracks=[{'scenes':[{'speech_start':0,'speech_end':2+i/100,'duration':2.52+i/100},
                      {'speech_start':3,'speech_end':5,'duration':2.52}]} for i in range(50)]
    bindings={'master':{'sha256':'a'*64},'timing':{'sha256':'b'*64}}
    plan={'schema':1,'lesson':'07_mask','master_sha256':'a'*64,'visual_timing_sha256':'b'*64,
          'scenes':[{'visual':'organize','keep_frames':[[0,30],[240,270]],
                    'confirmed_idle_frames':[[30,240],[270,300]],'final_result_frame':269}]}
    return english,original,tracks,plan,bindings

def test_shortest_voice_bound_1x_actions_final_result_padding_and_untouched_clocks(source_plan):
    rows=edit.plan_intervals(*source_plan)
    a,b=rows
    assert a['kept_frames']==60 and a['final_frame_hold_frames']==31
    assert a['duration']==91/30 and a['speech_start']==0 and a['scene_end']==91/30
    assert b['source_start_frame']==300 and b['source_keep_frames']==[[0,300]]
    assert b['duration']==10 and b['speech_start']==91/30 and b['scene_end']==391/30
    assert not b['edited'] and b['final_frame_hold_frames']==0

@pytest.mark.parametrize('fault',['unreviewed_gap','overlap','fractional','wrong_result','too_long',
                                  'master_hash','timing_hash','unknown','order','missing_voice','untouched_overflow'])
def test_fail_closed_without_every_source_frame_idle_or_kept_and_all_voice_fit(source_plan,fault):
    english,original,tracks,plan,bindings=deepcopy(source_plan)
    row=plan['scenes'][0]
    if fault=='unreviewed_gap':row['confirmed_idle_frames'][0][0]=31
    elif fault=='overlap':row['confirmed_idle_frames'][0][0]=29
    elif fault=='fractional':row['keep_frames'][0][0]=0.0
    elif fault=='wrong_result':row['final_result_frame']=268
    elif fault=='too_long':tracks[0]['scenes'][0]['speech_end']=1.99
    elif fault=='master_hash':plan['master_sha256']='c'*64
    elif fault=='timing_hash':plan['visual_timing_sha256']='c'*64
    elif fault=='unknown':row['visual']='missing'
    elif fault=='order':english['scenes'].reverse()
    elif fault=='missing_voice':tracks.pop()
    else:tracks[0]['scenes'][1]['duration']=10.1;tracks[0]['scenes'][1]['speech_end']=13.1
    with pytest.raises(ValueError):edit.plan_intervals(english,original,tracks,plan,bindings)

@pytest.mark.parametrize('source_start',[6,7])
@pytest.mark.parametrize('terminal_seekable',[False,True])
def test_real_later_scene_encode_kept_sequence_final_clone_no_overwrite_or_source_change(tmp_path,source_start,terminal_seekable):
    import subprocess,hashlib,json
    source=tmp_path/'source.mkv';output=tmp_path/'edited.mp4'
    width,height=64,48
    # An actual lossless source makes every original frame independently identifiable.
    levels=[10+12*i for i in range(18)]
    raw=b''.join(bytes([level])*(width*height*3) for level in levels)
    subprocess.run(['ffmpeg','-nostdin','-n','-v','error','-threads','2','-f','rawvideo',
        '-pixel_format','rgb24','-video_size',f'{width}x{height}','-framerate','30',
        '-i','pipe:0','-frames:v','18','-c:v','ffv1','-threads','2',str(source)],input=raw,check=True)
    before=hashlib.sha256(source.read_bytes()).hexdigest()
    row={'source_start_frame':source_start,'source_keep_frames':[[0,2],[4,6]],
         'kept_frames':4,'final_frame_hold_frames':2}
    edit.encode_scene(source,row,output,terminal_seekable=terminal_seekable)
    decoded=subprocess.check_output(['ffmpeg','-nostdin','-v','error','-threads','2',
        '-i',str(output),'-f','rawvideo','-pix_fmt','rgb24','-threads','2','pipe:1'])
    size=width*height*3;frames=[decoded[i:i+size] for i in range(0,len(decoded),size)]
    assert len(frames)==6 and all(len(frame)==size for frame in frames)
    observed=[]
    for frame in frames:
        mean=sum(frame)/size;index=min(range(len(levels)),key=lambda i:abs(levels[i]-mean))
        assert abs(levels[index]-mean)<3  # Ordinary H264 colour conversion, never a time tolerance.
        observed.append(index)
    assert observed==[source_start,source_start+1,source_start+4,source_start+5,source_start+5,source_start+5]
    assert frames[-1]==frames[-2]==frames[-3]
    if terminal_seekable:
        probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0',
            '-show_entries','frame=key_frame,best_effort_timestamp_time','-of','json',str(output)]))
        assert len(probe['frames'])==len(frames)
        terminal=probe['frames'][-2]
        assert terminal['key_frame']==1
        assert float(terminal['best_effort_timestamp_time'])==pytest.approx((len(frames)-2)/30,abs=1e-6)
    rendered=hashlib.sha256(output.read_bytes()).hexdigest()
    with pytest.raises(subprocess.CalledProcessError):edit.encode_scene(source,row,output)
    assert hashlib.sha256(output.read_bytes()).hexdigest()==rendered
    assert hashlib.sha256(source.read_bytes()).hexdigest()==before

# Reuse the existing genuine 50-track NONMEDIA unit fixture. No new fake
# publication protocol or manufactured native capture metadata is introduced.
from test_published_native_reuse import retained_stage
from test_native_live_timing import frozen_native_stage

def test_inputs_reject_changed_published_bytes_before_render(retained_stage,monkeypatch):
    stage,root,baseline,repo=retained_stage;monkeypatch.setattr(edit,'REPO',repo)
    source=baseline/'media_host/05_home/video/05_home_silent.mp4'
    source.write_bytes(b'changed explicit nonmedia fixture')
    with pytest.raises(ValueError,match='source differs'):
        edit.inputs(stage,'05_home',baseline,{})

def test_product_fences_preserve_only_source_bound_untouched_and_use_new_result_end(retained_stage,monkeypatch):
    import json
    from check_completed_matrix import digest
    from stage_lesson import write
    stage,root,baseline,repo=retained_stage;monkeypatch.setattr(edit,'REPO',repo)
    original=edit.read(baseline/'media_host/05_home/video/native-live-timings.json')
    bindings={name:{'path':f'media_host/05_home/video/{file}','sha256':digest(baseline/f'media_host/05_home/video/{file}')}
              for name,file in [('master','05_home_silent.mp4'),('timing','native-live-timings.json')]}
    old=original['scenes'][0]
    write(repo/'tools/tutorials/idle_tail_policy.json',{'schema':1,'lessons':[dict(lesson='05_home',
        master_sha256=bindings['master']['sha256'],visual_timing_sha256=bindings['timing']['sha256'],
        scenes=[dict(visual=old['visual'],source_start=0,source_end=3,idle_tail_start=1)])]})
    write(root/edit.RECORD,{'baseline_root':str(baseline)})
    (root/'video').mkdir(exist_ok=True)
    (root/'video/05_home_silent.mp4').write_bytes(b'explicit edited nonmedia fixture')
    timing={'scenes':[dict(visual=old['visual'],speech_start=0,scene_end=3,edited=False),
                      dict(visual=original['scenes'][1]['visual'],speech_start=3,scene_end=5,edited=True,kept_frames=30)]}
    write(root/'video/native-live-timings.json',timing)
    result=edit.edited_policy(root,'05_home',timing,bindings)
    entry=result['lessons'][0]
    assert entry['master_sha256']==digest(root/'video/05_home_silent.mp4')
    assert entry['visual_timing_sha256']==digest(root/'video/native-live-timings.json')
    assert [x['idle_tail_start'] for x in entry['scenes']]==[1,1]
    assert [x['source_start'] for x in entry['scenes']]==[0,3]
    assert str(baseline) not in json.dumps(result)
    changed=edit.read(repo/'tools/tutorials/idle_tail_policy.json');changed['lessons'][0]['master_sha256']='c'*64
    write(repo/'tools/tutorials/idle_tail_policy.json',changed)
    with pytest.raises(ValueError,match='source bytes changed'):edit.edited_policy(root,'05_home',timing,bindings)

@pytest.fixture
def checked_edit_stage(retained_stage,monkeypatch):
    from stage_lesson import write,read
    from check_completed_matrix import digest
    stage,root,baseline,repo=retained_stage;monkeypatch.setattr(edit,'REPO',repo)
    # Extend the existing explicit NONMEDIA 50-track fixture with measured-clock
    # fields needed by this planner. Never claim these are real audio/captures.
    for path in (root/'audio').glob('*/*.json'):
        track=read(path)
        for i,scene in enumerate(track['scenes']):
            scene.update(speech_start=i*2,speech_end=i*2+1.48)
        write(path,track)
    bindings={name:digest(baseline/f'media_host/05_home/video/{file}') for name,file in
              [('master','05_home_silent.mp4'),('timing','native-live-timings.json')]}
    visual=read(root/'lesson.en.json')['scenes'][0]['visual']
    plan={'schema':1,'lesson':'05_home','master_sha256':bindings['master'],
          'visual_timing_sha256':bindings['timing'],'scenes':[dict(visual=visual,
          keep_frames=[[0,30]],confirmed_idle_frames=[[30,90]],final_result_frame=29)]}
    root,_,_,source,spec,timing=edit.inputs(stage,'05_home',baseline,plan)
    (root/'video').mkdir(exist_ok=True)
    # A genuine tiny image master exercises the actual poster decoder; production
    # 4K/frame/timeline acceptance belongs to normal finish, not this unit fixture.
    import subprocess
    master=root/'video/05_home_silent.mp4'
    subprocess.run(['ffmpeg','-nostdin','-n','-v','error','-f','lavfi','-i',
        'color=c=red:s=64x48:r=30:d=0.1','-an','-c:v','libx264','-threads','1',
        '-pix_fmt','yuv420p',str(master)],check=True)
    write(root/'scenes.json',spec);timing['visual_spec_sha256']=digest(root/'scenes.json')
    write(root/edit.RECORD,dict(schema=1,kind='edited_published_native_intervals',lesson='05_home',
        fresh_capture=False,raw_capture_revalidated=False,**source,edit=plan,rendered_master_sha256=digest(master)))
    write(root/'video/native-live-timings.json',timing)
    return stage,root


def test_edited_matrix_dispatch_requires_current_fifty_track_and_source_bindings(checked_edit_stage):
    from check_completed_matrix import reconcile_visual_sources
    from stage_lesson import read
    stage,root=checked_edit_stage
    result=reconcile_visual_sources(stage,'05_home',read(root/'lesson.en.json'))
    assert result['fresh_capture'] is False and result['raw_capture_revalidated'] is False
    assert not (root/'visual.json').exists()
    audio=root/'audio/en/af_heart.m4a';audio.write_bytes(audio.read_bytes()+b'changed unit fixture')
    with pytest.raises(ValueError,match='Narration media or source changed'):
        reconcile_visual_sources(stage,'05_home',read(root/'lesson.en.json'))


def test_edited_matrix_rejects_missing_explicit_record(checked_edit_stage):
    from check_completed_matrix import reconcile_visual_sources
    from stage_lesson import read
    stage,root=checked_edit_stage
    (root/edit.RECORD).rename(root/(edit.RECORD+'.preserved'))
    with pytest.raises(ValueError,match='exclusive source/interval record'):
        reconcile_visual_sources(stage,'05_home',read(root/'lesson.en.json'))


def test_edited_poster_dispatch_validates_actual_bindings_then_decodes_first_frame(checked_edit_stage,monkeypatch):
    import importlib.util,sys
    from PIL import Image
    stage,root=checked_edit_stage
    spec=importlib.util.spec_from_file_location('edited_poster_test',Path(__file__).resolve().parents[1]/'authoring/tools/prepare_web_media.py')
    tool=importlib.util.module_from_spec(spec);spec.loader.exec_module(tool)
    monkeypatch.setattr(tool,'PRODUCTION',stage/'production')
    monkeypatch.setattr(sys,'argv',['prepare_web_media.py','--lessons','05_home'])
    assert tool.main()==0
    with Image.open(root/'poster.jpg') as image:
        r,g,b=image.convert('RGB').getpixel((32,24));assert r>230 and g<20 and b<20
    original=(root/'poster.jpg').read_bytes()
    audio=root/'audio/en/af_heart.m4a';audio.write_bytes(audio.read_bytes()+b'changed unit fixture')
    with pytest.raises(ValueError,match='Narration media or source changed'):tool.main()
    assert (root/'poster.jpg').read_bytes()==original


def test_reviewed_setup_omissions_keep_legacy_objects_identical(source_plan):
    legacy=edit.plan_intervals(*source_plan)
    args=deepcopy(source_plan);row=args[3]['scenes'][0]
    row.update(omitted_setup_frames=[[30,240]],omitted_setup_reason='Omit redundant private path setup.')
    row['confirmed_idle_frames']=[[270,300]]
    current=edit.plan_intervals(*args)
    assert current[0].pop('omitted_setup_frames')==[[30,240]]
    assert current[0].pop('omitted_setup_reason')=='Omit redundant private path setup.'
    assert current[0]['confirmed_idle_frames']==[[270,300]]
    current[0]['confirmed_idle_frames']=deepcopy(legacy[0]['confirmed_idle_frames'])
    assert current==legacy
    assert edit.plan_intervals(*source_plan)==legacy
    assert all('omitted_setup_frames' not in r and 'omitted_setup_reason' not in r for r in legacy)


def test_reviewed_setup_can_cover_all_removed_frames_without_false_idle(source_plan):
    args=deepcopy(source_plan);row=args[3]['scenes'][0]
    row.update(omitted_setup_frames=row['confirmed_idle_frames'],omitted_setup_reason='Redundant setup/navigation.')
    row['confirmed_idle_frames']=[]
    result=edit.plan_intervals(*args)[0]
    assert result['confirmed_idle_frames']==[] and result['kept_frames']==60
    assert result['omitted_setup_frames']==[[30,240],[270,300]]


@pytest.mark.parametrize('fault',['missing_reason','missing_frames','blank_reason','wrong_reason_type',
                                  'empty_setup','overlap','gap','fractional','legacy_empty_idle','too_long'])
def test_reviewed_setup_omissions_reject_incomplete_or_dishonest_partition(source_plan,fault):
    args=deepcopy(source_plan);row=args[3]['scenes'][0]
    row.update(omitted_setup_frames=[[30,240]],omitted_setup_reason='Redundant setup.')
    row['confirmed_idle_frames']=[[270,300]]
    if fault=='missing_reason':del row['omitted_setup_reason']
    elif fault=='missing_frames':del row['omitted_setup_frames']
    elif fault=='blank_reason':row['omitted_setup_reason']='  '
    elif fault=='wrong_reason_type':row['omitted_setup_reason']=True
    elif fault=='empty_setup':row['omitted_setup_frames']=[]
    elif fault=='overlap':row['omitted_setup_frames'][0][0]=29
    elif fault=='gap':row['omitted_setup_frames'][0][0]=31
    elif fault=='fractional':row['omitted_setup_frames'][0][0]=30.0
    elif fault=='legacy_empty_idle':
        del row['omitted_setup_frames'];del row['omitted_setup_reason'];row['confirmed_idle_frames']=[]
    elif fault=='too_long':args[2][0]['scenes'][0]['speech_end']=1.99
    with pytest.raises(ValueError):edit.plan_intervals(*args)


@pytest.fixture
def repeated_source_plan(source_plan):
    args=deepcopy(source_plan)
    args[0]['scenes'][1]['visual']='organize'
    args[1]['scenes'][1]['visual']='organize'
    args[3]['scenes'][0]['source_scene']=1
    return args


def test_repeated_visual_edits_resolve_by_source_scene_and_preserve_other_instance(repeated_source_plan):
    args=repeated_source_plan
    row=deepcopy(args[3]['scenes'][0]);row['source_scene']=2
    args[3]['scenes'].append(row)
    rows=edit.plan_intervals(*args)
    assert [r['source_start_frame'] for r in rows]==[0,300]
    assert all(r['edited'] and r['source_keep_frames']==[[0,30],[240,270]] for r in rows)
    args[3]['scenes'].pop(0)
    rows=edit.plan_intervals(*args)
    assert not rows[0]['edited'] and rows[1]['edited']


def test_optional_source_scene_preserves_legacy_unique_plan_objects(source_plan):
    expected=edit.plan_intervals(*source_plan)
    args=deepcopy(source_plan);args[3]['scenes'][0]['source_scene']=1
    assert edit.plan_intervals(*args)==expected


@pytest.mark.parametrize('fault',['missing_index','zero','out_of_range','bool','fractional','wrong_visual',
                                  'duplicate','clock_duration','clock_fractional','clock_number'])
def test_repeat_identity_rejects_ambiguous_invalid_duplicate_and_changed_source_clock(repeated_source_plan,fault):
    args=repeated_source_plan;row=args[3]['scenes'][0]
    if fault=='missing_index':del row['source_scene']
    elif fault=='zero':row['source_scene']=0
    elif fault=='out_of_range':row['source_scene']=3
    elif fault=='bool':row['source_scene']=True
    elif fault=='fractional':row['source_scene']=1.0
    elif fault=='wrong_visual':row['visual']='unknown'
    elif fault=='duplicate':args[3]['scenes'].append(deepcopy(row))
    elif fault=='clock_duration':args[1]['scenes'][0]['duration']=9
    elif fault=='clock_fractional':args[1]['scenes'][0].update(speech_start=.01,scene_end=10.01)
    elif fault=='clock_number':args[1]['scenes'][0]['scene']=2
    with pytest.raises(ValueError):edit.plan_intervals(*args)


def test_inherited_repeat_fences_use_original_clock_and_index_after_prior_edit(retained_stage,monkeypatch):
    from stage_lesson import write,read
    from check_completed_matrix import digest
    stage,root,baseline,repo=retained_stage;monkeypatch.setattr(edit,'REPO',repo)
    path=baseline/'media_host/05_home/video/native-live-timings.json'
    original=read(path);a,b=original['scenes']
    b['visual']=a['visual'];write(path,original)
    bindings={name:{'path':f'media_host/05_home/video/{file}','sha256':digest(baseline/f'media_host/05_home/video/{file}')}
              for name,file in [('master','05_home_silent.mp4'),('timing','native-live-timings.json')]}
    fences=[dict(visual=s['visual'],source_start=s['speech_start'],source_end=s['scene_end'],idle_tail_start=f)
            for s,f in zip(original['scenes'],[.5,1.5])]
    write(repo/'tools/tutorials/idle_tail_policy.json',{'schema':1,'lessons':[dict(lesson='05_home',
        master_sha256=bindings['master']['sha256'],visual_timing_sha256=bindings['timing']['sha256'],scenes=fences)]})
    write(root/edit.RECORD,{'baseline_root':str(baseline)})
    (root/'video').mkdir(exist_ok=True);write(root/'video/native-live-timings.json',{})
    (root/'video/05_home_silent.mp4').write_bytes(b'explicit nonmedia edited fixture')
    timing={'scenes':[dict(visual=a['visual'],speech_start=0,scene_end=2,edited=True,kept_frames=30),
                      dict(visual=b['visual'],speech_start=2,scene_end=5,edited=False)]}
    result=edit.edited_policy(root,'05_home',timing,bindings)['lessons'][0]['scenes']
    assert [f['idle_tail_start'] for f in result]==[1,1.5]
    assert [f['source_start'] for f in result]==[0,2]
