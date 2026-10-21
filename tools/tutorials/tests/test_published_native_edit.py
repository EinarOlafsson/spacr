"""Published-source planning contracts and one tiny real 1x frame-edit regression."""
from copy import deepcopy
from pathlib import Path
import pytest
import published_native_edit as edit

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
def test_real_later_scene_encode_kept_sequence_final_clone_no_overwrite_or_source_change(tmp_path,source_start):
    import subprocess,hashlib
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
    edit.encode_scene(source,row,output)
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
