"""Bounded metadata-only terminal visual and matrix dispatch checks."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json
import sys

import pytest
import published_native_edit as visual
import check_completed_matrix as matrix


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def prefix(tmp_path, monkeypatch):
    stage=tmp_path/'stage';baseline=tmp_path/'baseline';repo=tmp_path/'repo'
    root=stage/'production/05_home';root.mkdir(parents=True);baseline.mkdir()
    english=dict(id='05_home',scenes=[dict(visual=f'v{i}',narration=f'Text {i}') for i in range(38)])
    put(root/'lesson.en.json',english);put(repo/'tools/tutorials/lessons/05_home.json',english)
    clock=dict(lesson='05_home',kind='native_visual_timing',native_live_video=True,
        coverage=dict(accepted=True),scenes=[dict(scene=i+1,visual=f'v{i}' if i<38 else '24_spaceout_field',
            speech_start=i*20,speech_end=(i+1)*20,scene_end=(i+1)*20,duration=20,
            native_live_video=i!=2) for i in range(39)])
    bindings={key:dict(path=f'media_host/05_home/video/{name}',sha256=mark*64,bytes=1)
        for key,name,mark in [('master','05_home_silent.mp4','a'),('timing','native-live-timings.json','b')]}
    source=dict(baseline_root=str(baseline),published_bindings=bindings,
        original_visual_clock=dict(path=bindings['timing']['path'],sha256=bindings['timing']['sha256']))
    tracks=[]
    for voice in range(50):
        tracks.append(dict(scenes=[dict(speech_start=i*20,speech_end=i*20+19.9,
            duration=20.8 if i==37 else 20,sentences=[dict(audible_end=i*20+19.8)]) for i in range(38)]))
    proofs=[dict(language='en',voice=f'voice{i}',audio_sha256='c'*64,timing_sha256='d'*64) for i in range(50)]
    audio=root/'published-terminal-scene-removal.json';put(audio,{'source':'bound'})
    stub=SimpleNamespace(RECORD=audio.name,
        _source=lambda *args:(deepcopy(source),deepcopy(english),deepcopy(clock)),
        checked_terminal_narration=lambda *args,**kwargs:(deepcopy(tracks),deepcopy(proofs)))
    monkeypatch.setitem(sys.modules,'published_terminal_prefix',stub)
    monkeypatch.setattr(visual,'REPO',repo)
    return SimpleNamespace(stage=stage,baseline=baseline,root=root,english=english,clock=clock,
        bindings=bindings,source=source,tracks=tracks,proofs=proofs,audio=audio,stub=stub)


def inputs(p):
    return visual._terminal_inputs(p.stage,'05_home',p.baseline)


def test_full_original_frames_classes_and_clocks_are_preserved(prefix):
    *_,timing=inputs(prefix)
    assert len(timing['scenes'])==38 and timing['preserve_visual_intervals'] is True
    for i,row in enumerate(timing['scenes']):
        assert row['source_start_frame']==i*600
        assert row['source_keep_frames']==[[0,600]] and row['kept_frames']==600
        assert row['native_live_video'] is prefix.clock['scenes'][i]['native_live_video']
        assert row['original_published_interval']==prefix.clock['scenes'][i]
        assert row['speech_start']==i*20
        assert row['final_frame_hold_frames']==(24 if i==37 else 0)
    assert timing['scenes'][-1]['scene_end']==760.8


@pytest.mark.parametrize('fault',['frame_clock','removed_visual','short_track','nonfinite_audible','missing_audible','audio_eof','clock_binding'])
def test_source_or_audio_tamper_rejected(prefix,fault):
    if fault=='frame_clock':prefix.clock['scenes'][4]['scene_end']+=.01
    elif fault=='removed_visual':prefix.clock['scenes'][-1]['visual']='another'
    elif fault=='short_track':prefix.tracks[0]['scenes'].pop()
    elif fault=='nonfinite_audible':prefix.tracks[0]['scenes'][-1]['sentences'][-1]['audible_end']=float('nan')
    elif fault=='missing_audible':prefix.tracks[0]['scenes'][-1]['sentences']=[]
    elif fault=='audio_eof':prefix.tracks[0]['scenes'][-1]['duration']=20
    else:prefix.source['original_visual_clock']['sha256']='f'*64
    with pytest.raises(ValueError):inputs(prefix)


def test_checked_visual_record_rejects_changed_audio_binding(prefix):
    root,_,_,source,spec,timing=inputs(prefix)
    master=root/'video/05_home_silent.mp4';master.parent.mkdir();master.write_bytes(b'fixture')
    put(root/'scenes.json',spec)
    put(root/visual.RECORD,dict(schema=1,kind=visual.TERMINAL_KIND,lesson='05_home',
        fresh_capture=False,raw_capture_revalidated=False,**source,edit=None,
        rendered_master_sha256=visual.digest(master)))
    timing['visual_spec_sha256']=visual.digest(root/'scenes.json')
    put(root/'video/native-live-timings.json',timing)
    assert visual.checked_edited_native_timing(prefix.stage,'05_home').is_file()
    put(prefix.audio,{'source':'changed'})
    with pytest.raises(ValueError,match='audio record'):visual.checked_edited_native_timing(prefix.stage,'05_home')


@pytest.mark.parametrize('fault',[None,'clock','fence','master'])
def test_only_bound_original_fences_are_inherited(prefix,monkeypatch,fault):
    import idle_tail_policy
    root,_,bindings,_,_,timing=inputs(prefix)
    master=root/'video/05_home_silent.mp4';master.parent.mkdir();master.write_bytes(b'fixture')
    put(root/'video/native-live-timings.json',timing)
    entry=dict(lesson='05_home',master_sha256=bindings['master']['sha256'],
        visual_timing_sha256=bindings['timing']['sha256'],scenes=[
        dict(visual=prefix.clock['scenes'][i]['visual'],source_start=i*20,source_end=(i+1)*20,idle_tail_start=5)
        for i in (10,37,38)])
    if fault=='clock':entry['scenes'][1]['source_start']+=.1
    elif fault=='fence':entry['scenes'][1]['idle_tail_start']=20
    elif fault=='master':entry['master_sha256']='f'*64
    monkeypatch.setattr(idle_tail_policy,'load_policy',lambda *_:dict(schema=1,lessons=[entry]))
    if fault:
        with pytest.raises(ValueError):visual.edited_policy(root,'05_home',timing,bindings,original=prefix.clock)
    else:
        policy=visual.edited_policy(root,'05_home',timing,bindings,original=prefix.clock)
        fences=policy['lessons'][0]['scenes']
        assert len(fences)==2 and fences[-1]==dict(visual='v37',source_start=740,source_end=760.8,idle_tail_start=5)


def matrix_stage(p):
    put(p.root/'scenes.json',dict(kind=visual.TERMINAL_KIND))
    put(p.root/visual.RECORD,dict(kind=visual.TERMINAL_KIND))


def test_published_order_requires_checked_terminal_provenance(prefix):
    matrix_stage(prefix);calls=[]
    def checked(*args,**kwargs):
        calls.append(kwargs);return prefix.tracks,prefix.proofs
    prefix.stub.checked_terminal_narration=checked
    assert matrix._terminal_published_voice_order(prefix.stage,'05_home') is True
    assert calls==[{'decode':False}]
    inv={'en':['af_heart','af_aoede']}
    catalog=dict(prefix.english,narration_voices={'en':['af_aoede','af_heart']})
    matrix.reconcile_english_catalog(catalog,prefix.english,inv,published_voice_order=True)
    with pytest.raises(ValueError):matrix.reconcile_english_catalog(catalog,prefix.english,inv)
    catalog['narration_voices']['en']=['af_heart','af_heart']
    with pytest.raises(ValueError):matrix.reconcile_english_catalog(catalog,prefix.english,inv,published_voice_order=True)


@pytest.mark.parametrize('fault',['missing','kind','proof_error','inventory','retained'])
def test_terminal_matrix_route_cannot_bypass_proof(prefix,fault):
    matrix_stage(prefix)
    if fault=='missing':prefix.audio.rename(prefix.audio.with_suffix('.held'))
    elif fault=='kind':put(prefix.root/visual.RECORD,dict(kind='unknown'))
    elif fault=='proof_error':
        def rejected(*args,**kwargs):raise ValueError('source changed')
        prefix.stub.checked_terminal_narration=rejected
    elif fault=='inventory':prefix.proofs.pop()
    with pytest.raises(ValueError):matrix._terminal_published_voice_order(prefix.stage,'05_home',retained_narration=fault=='retained')


def poster_stage(tmp_path, monkeypatch):
    import prepare_web_media as posters
    root=tmp_path/'production/05_home'
    video=root/'video/05_home_silent.mp4';video.parent.mkdir(parents=True)
    video.write_bytes(b'unit fixture: validated master path')
    put(root/'scenes.json',dict(kind=visual.TERMINAL_KIND,scenes=[dict(visual='first_retained')]))
    monkeypatch.setattr(posters,'PRODUCTION',root.parent)
    monkeypatch.setattr(sys,'argv',['prepare_web_media','--lessons','05_home'])
    return posters,root,video


def test_terminal_poster_reads_validated_master_first_frame(tmp_path,monkeypatch):
    import io
    from PIL import Image
    posters,root,video=poster_stage(tmp_path,monkeypatch)
    events=[]
    monkeypatch.setattr(visual,'checked_edited_native_timing',
        lambda stage,lesson:events.append(('validated',stage,lesson)))
    frame=io.BytesIO();Image.new('RGB',(48,32),(36,110,182)).save(frame,format='PNG')
    def decode(command,**kwargs):
        assert events==[('validated',root.parent.parent,'05_home')]
        assert command[command.index('-i')+1]==str(video)
        assert command[command.index('-frames:v')+1]=='1' and '-ss' not in command
        assert kwargs['check'] is True and kwargs['stdout']==posters.subprocess.PIPE
        events.append(('first_master_frame',))
        return SimpleNamespace(stdout=frame.getvalue())
    monkeypatch.setattr(posters.subprocess,'run',decode)
    assert posters.main()==0
    with Image.open(root/'poster.jpg') as actual:
        assert actual.size==(48,32)
        assert all(abs(a-b)<=3 for a,b in zip(actual.getpixel((20,20)),(36,110,182)))
    assert events[-1]==('first_master_frame',)


def test_terminal_poster_validation_failure_precedes_decode_or_output(tmp_path,monkeypatch):
    posters,root,_=poster_stage(tmp_path,monkeypatch)
    def reject(*args):raise ValueError('terminal source changed')
    def forbidden(*args,**kwargs):pytest.fail('ffmpeg must not run before source validation')
    monkeypatch.setattr(visual,'checked_edited_native_timing',reject)
    monkeypatch.setattr(posters.subprocess,'run',forbidden)
    with pytest.raises(ValueError,match='terminal source changed'):posters.main()
    assert not (root/'poster.jpg').exists()
