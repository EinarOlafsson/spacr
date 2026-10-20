"""A retained installer composite keeps complete fresh Home and legacy provenance."""
from copy import deepcopy
from pathlib import Path
import json

import pytest

from test_retained_visual_refresh import retained
import retained_visual_refresh as refresh
from audit_staged_catalogs import CATALOGS
from stage_lesson import read, write


@pytest.fixture
def mixed(retained):
    r = retained
    identity = '04_platform_installers'
    root = r['root'].with_name(identity);r['root'].rename(root)
    old_media = r['baseline'] / 'media_host' / r['identity']
    media = old_media.with_name(identity);old_media.rename(media)
    for base in (media, root):
        (base / 'video' / (r['identity'] + '_silent.mp4')).rename(base / 'video' / (identity + '_silent.mp4'))
    original = r['original'];repo = refresh.REPO
    frames = [600,600,600,600,600,600,693,818,840,879,600,600,600,1377]
    assert sum(frames) == 10007
    spec = read(original / 'scenes.json');spec['lesson'] = identity;spec['scenes'] = spec['scenes'][:14]
    for scene in spec['scenes']:scene.pop('clip', None)
    origins = {};clocks = [];elapsed = 0
    for index,(scene,count) in enumerate(zip(spec['scenes'],frames)):
        visual = {7:'14_current_nightly_performance',8:'13_current_nightly_home',9:'05_doctor'}.get(index,f'visual_{index}')
        origins[visual] = dict(kind='current_nightly' if index in (7,8) else 'published_installation',
                               image_sha256=scene['capture_sha256'],installed_identity=None)
        clocks.append(dict(scene=index+1,speech_start=elapsed/30,speech_end=(elapsed+count)/30,
            scene_end=(elapsed+count)/30,duration=count/30,visual=visual,native_live_video=index in (7,8),
            text=scene['narration'],longest_narration_duration=count/30-.01))
        elapsed += count
        if index in (7,8):
            for suffix in ('.mp4','.json'):
                path=original/'clips'/(visual+suffix);path.parent.mkdir(exist_ok=True)
                path.write_bytes(('old'+visual+suffix).encode());dest=root/path.relative_to(original);dest.write_bytes(path.read_bytes())
            scene['clip']=dict(video='clips/'+visual+'.mp4',sha256=refresh.digest(original/'clips'/(visual+'.mp4')),
                receipt='clips/'+visual+'.json',receipt_sha256=refresh.digest(original/'clips'/(visual+'.json')),
                duration=33.234 if index==7 else 32.367)
    spec['capture_source']['frame_origins']=origins;write(original/'scenes.json',spec)
    lesson=dict(id=identity,scenes=[dict(narration=s['narration']) for s in spec['scenes']])
    write(repo/'tools/tutorials/lessons'/(identity+'.json'),lesson);write(root/'lesson.en.json',lesson)
    for name in CATALOGS:
        catalog=dict(lessons=[lesson]);write(r['baseline']/'web/catalog'/name,catalog);write(r['stage']/'catalog'/name,catalog)
    old_timing=dict(lesson=identity,fps=30,total_duration=10007/30,scenes=clocks,
        visual_spec_sha256=refresh.digest(original/'scenes.json'),coverage=dict(accepted=True),
        native_clip_sources=[dict(scene=i+1,clip_sha256=spec['scenes'][i]['clip']['sha256'],
            receipt_sha256=spec['scenes'][i]['clip']['receipt_sha256']) for i in (7,8)])
    old_timing_path=media/'video/native-live-timings.json';write(old_timing_path,old_timing)
    old_master=media/'video'/(identity+'_silent.mp4')
    baseline_manifest=r['baseline']/'release-manifest.json';base=read(baseline_manifest)
    for row in base['files']:
        path=row['path'].replace(r['identity'],identity);row['path']=path
        actual=r['baseline']/path;row.update(sha256=refresh.digest(actual),bytes=actual.stat().st_size)
    write(baseline_manifest,base)
    pubpath=r['baseline']/'publication-receipt.json';pub=read(pubpath);pub['manifest_sha256']=refresh.digest(baseline_manifest);write(pubpath,pub)
    source=r['stage']/'live/captures/home';source.mkdir(parents=True)
    live=source.parent.parent;name='13_current_nightly_home';N=845;commit='1'*40
    def put(path,data):path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data);return path
    png=(root/'cards/pip_conda_01_create.png').read_bytes()
    put(source/(name+'.png'),png);put(source/(name+'.mp4'),b'new whole native handoff');put(source/(name+'.native.mp4'),b'raw native')
    put(source/(name+'.pointer.json'),b'[]')
    gate=dict(video=str(source/(name+'.native.mp4')),size=[3840,2160],fps=30.,frames=N,duration_s=N/30,
        passed=True,reasons=[],limits=dict(min_fps=27.9),distinct_fps=29.2,
        cursor=dict(jump_count=0,jumps=[],found_fraction=1.,max_step_px=20.,bound_px=115.2),top_bar=dict(hidden=True))
    for key in ('app_freeze_seconds','app_slow_seconds','app_unknown_seconds','app_gui_sampler_gaps_s','app_stall_windows_s','adapter_bypasses'):gate[key]=[]
    write(source/(name+'.gate.json'),gate);write(source/(name+'.host.json'),dict(adapter_bypasses=[],app_gui_sampler_gaps_s=[]))
    capture=dict(video=name+'.mp4',sha256=refresh.digest(source/(name+'.mp4')),size=[3840,2160],duration=N/30,
        full_decode_passed=True,audio_streams=0,target_fps=30,encoded_fps='30/1',frames=[dict(pts=i) for i in range(N)],
        native=dict(video=name+'.native.mp4',sha256=refresh.digest(source/(name+'.native.mp4')),size=[3840,2160]),
        gate=dict(passed=True,reasons=[]),adapter_bypasses=[])
    write(source/(name+'.capture.json'),capture)
    frame=dict(image=name+'.png',sha256=refresh.digest(source/(name+'.png')),clip=dict(video=name+'.mp4',
        sha256=capture['sha256'],native_video=name+'.native.mp4',native_sha256=capture['native']['sha256'],
        receipt=name+'.capture.json',receipt_sha256=refresh.digest(source/(name+'.capture.json')),gate=dict(passed=True,reasons=[])))
    write(source/'frames.json',{name:frame})
    write(source/'provenance.json',dict(commit=commit,module='home',completed_capture=True,app_source_modified=False,selected_scenes=name))
    sourcefile=put(repo/'spacr/capture_source.py',b'unchanged app/source fixture')
    write(live/'exact-source.json',dict(commit=commit,fresh=True,private_profile=True,sha256={'spacr/capture_source.py':refresh.digest(sourcefile)}))
    write(live/'host-run.json',dict(app_exit_code=0,recorder_rc=0,failure=None,gate_failed=[],clips={name:dict(passed=True,reasons=[],distinct_fps=29.2,file=name+'.gate.json')}))
    write(live/'gnome-host/cleanup.json',dict(exit_code=0,survivors_after_cleanup=[]))
    profile={key:False for key in ('show_alpha','show_alpha_features','show_alpha_species','spaceout','anonymous_reports_enabled','anonymous_report_notice_due')}
    write(live/'capture-profile.json',profile)
    locations=dict(video=source/(name+'.mp4'),native_video=source/(name+'.native.mp4'),receipt=source/(name+'.capture.json'),
        gate=source/(name+'.gate.json'),facts=source/(name+'.host.json'),pointer=source/(name+'.pointer.json'),poster=source/(name+'.png'),
        frames_manifest=source/'frames.json',provenance=source/'provenance.json',host_run=live/'host-run.json',
        cleanup=live/'gnome-host/cleanup.json',profile=live/'capture-profile.json',exact_source=live/'exact-source.json')
    native=dict(capture_root=str(source),source_commit=commit,**{key:refresh.binding(path) for key,path in locations.items()})
    approval_path=live/'root-native-acceptance.json'
    approval=dict(accepted=True,source_commit=commit,source_scene=name,actual_native_frames=N,requested_minimum_seconds=28)
    approval.update({key:native[key] for key in ('video','gate','host_run','provenance','profile','cleanup','frames_manifest','poster','exact_source')})
    approval['capture_receipt']=native['receipt'];write(approval_path,approval)
    expected=deepcopy(spec)
    put(root/'fresh/home.png',png);put(root/'fresh/home.mp4',(source/(name+'.mp4')).read_bytes())
    put(root/'fresh/home.capture.json',(source/(name+'.capture.json')).read_bytes())
    expected['scenes'][8].update(image='fresh/home.png',capture_sha256=refresh.digest(root/'fresh/home.png'),
        clip=dict(video='fresh/home.mp4',sha256=native['video']['sha256'],receipt='fresh/home.capture.json',
                  receipt_sha256=native['receipt']['sha256'],duration=N/30))
    origin_home=dict(kind='current_nightly',capture=str(source),source_visual=name,image_sha256=native['poster']['sha256'],
        installed_identity=None,capture_source_commit=commit,capture_provenance=native['provenance'],representation='fresh_full_native_clip')
    card=root/'cards/05_doctor.png';put(card,png)
    write(card.parent/'frames.json',{'05_doctor':dict(image=card.name,sha256=refresh.digest(card),buttons=[],kind='generated_command_reference_not_terminal_output')})
    expected['scenes'][9].update(image='cards/05_doctor.png',capture_sha256=refresh.digest(card))
    origin_card=dict(kind='reference_guidance',capture=str(card.parent),source_visual='05_doctor',image_sha256=refresh.digest(card),
        installed_identity=None,representation='generated_command_reference_not_terminal_output',generator_provenance=refresh.binding(card.parent/'provenance.json'))
    card_approval=r['stage']/'card-approval.json';write(card_approval,dict(lesson=identity,scene=10,accepted=True,
        terminal_output_fabricated=False,installation_or_environment_commands_executed=False,image=refresh.binding(card),
        generator_provenance=origin_card['generator_provenance']))
    historical=r['stage']/'recorded-doctor.json';write(historical,[dict(scene='05_doctor',accepted=True,returncode=0,
        command=['spacr-doctor','--no-gpu-probe'],output='13 passed, 1 warnings, 0 failed, 0 errored, 4 skipped\nczi: missing pylibCZIrw\ninstaller selected cpu; torch currently uses cpu\nno GPU probe was run')])
    expected['capture_source']['frame_origins'].update({'13_current_nightly_home':origin_home,'05_doctor':origin_card})
    expected['capture_source'].update(completed_capture=False,frame_pixels_modified=True);write(root/'scenes.json',expected)
    policy=r['stage']/'policy.json';write(policy,dict(schema=1,kind='retained_mixed_visual_refresh_policy',lesson=identity,
        accepted=True,minimum_native_frames=840,fps=30,whole_native_required=True,other_scene_durations_preserved=True,
        audio_and_catalogs_retained=True,supersedes_sha256='48959f634aff43cbec64c1c3b416e0e97ad4a4ff711bda8537c41d2f3a004e42'))
    master=root/'video'/(identity+'_silent.mp4');render=r['stage']/'render.json';log=r['stage']/'overlay.log'
    write(render,dict(command=refresh.mixed_overlay_command(old_master,root/'fresh/home.mp4',card,master,N),
        returncode=0,output_log=refresh.binding(log)))
    replacements=[]
    for index,kind,first,count,origin in [(8,'strict_full_native_clip',5111,N,origin_home),
            (9,'generated_command_reference_not_terminal_output',5111+N,879,origin_card)]:
        visual=clocks[index]['visual'];replacements.append(dict(scene=index+1,kind=kind,first_frame=first,frame_count=count,
            original_image_sha256=spec['scenes'][index]['capture_sha256'],new_image_sha256=expected['scenes'][index]['capture_sha256'],
            original_origin=origins[visual],new_origin=origin))
    replacements[0].update(native=native,native_visual_acceptance=refresh.binding(approval_path),new_video_sha256=native['video']['sha256'],new_receipt_sha256=native['receipt']['sha256'])
    replacements[1].update(card_visual_acceptance=refresh.binding(card_approval),historical_doctor=refresh.binding(historical))
    receipt=dict(schema=2,kind='retained_visual_refresh',lesson=identity,baseline_root=str(r['baseline']),original_root=str(original),
        baseline_manifest=refresh.binding(baseline_manifest),publication_receipt=refresh.binding(pubpath),original_spec=refresh.binding(original/'scenes.json'),
        original_timing=refresh.binding(old_timing_path),original_master=refresh.binding(old_master),new_spec=refresh.binding(root/'scenes.json'),
        new_master=refresh.binding(master),render_log=refresh.binding(render),policy=refresh.binding(policy),replacements=replacements)
    receipt_path=root/'retained-visual-refresh.json'
    def seal():
        frame=read(source/'frames.json')[name];frame['clip']['receipt_sha256']=refresh.digest(source/(name+'.capture.json'));write(source/'frames.json',{name:frame})
        (root/'fresh/home.capture.json').write_bytes((source/(name+'.capture.json')).read_bytes())
        for key,path in locations.items():native[key]=refresh.binding(path)
        approval=read(approval_path);approval.update({key:native[key] for key in ('video','gate','host_run','provenance','profile','cleanup','frames_manifest','poster','exact_source')});approval['capture_receipt']=native['receipt'];write(approval_path,approval)
        replacements[0]['native_visual_acceptance']=refresh.binding(approval_path);replacements[0]['new_receipt_sha256']=native['receipt']['sha256']
        current=read(root/'scenes.json');current['scenes'][8]['clip']['receipt_sha256']=native['receipt']['sha256'];write(root/'scenes.json',current)
        receipt['new_spec']=refresh.binding(root/'scenes.json');receipt['render_log']=refresh.binding(render);receipt['policy']=refresh.binding(policy)
        write(receipt_path,receipt);write(root/'video/native-live-timings.json',refresh.mixed_retained_timing(old_timing,
            refresh.digest(root/'scenes.json'),refresh.digest(receipt_path),refresh.digest(baseline_manifest),
            native['video']['sha256'],native['receipt']['sha256'],N))
    seal()
    return dict(stage=r['stage'],root=root,baseline=r['baseline'],identity=identity,source=source,live=live,
        original=original,receipt=receipt,receipt_path=receipt_path,old_timing=old_timing,N=N,seal=seal,
        policy=policy,render=render,approval=approval_path,historical=historical)


def test_full_native_latency_tail_shifts_clocks_without_trimming_other_scenes(mixed):
    r=mixed;path=refresh.checked_retained_visual_timing(r['stage'],r['identity'],r['baseline']);t=read(path)
    assert round(t['total_duration']*30)==9167+r['N']==10012
    assert t['scenes'][8]['duration']==845/30
    for index,(before,after) in enumerate(zip(r['old_timing']['scenes'],t['scenes'])):
        assert after['text']==before['text']
        if index!=8:assert after['duration']==before['duration']
    assert t['native_clip_sources'][0]==r['old_timing']['native_clip_sources'][0]
    assert 'complete_native_clips' not in t
    assert t['preserve_visual_intervals'] is True
    graph=read(r['render'])['command'];graph=graph[graph.index('-filter_complex')+1]
    assert '[1:v]setpts=PTS-STARTPTS[home]' in graph and '[1:v]trim' not in graph
    assert 'end_frame=879' in graph and 'start_frame=6830:end_frame=10007' in graph


@pytest.mark.parametrize('change',['short','noncontiguous','badfps','gatefail','freeze','slow','unknown','samplergap',
 'stall','bypass','cursor','topbar','profile','provenance','cleanup','source','approval','recordonlypolicy',
 'other_scene','legacy_performance','timing','missing_playback_mode','wrong_range','renderfailed','missing_policy','card_history'])
def test_mixed_refresh_rejects_missing_or_drifted_actual_evidence(mixed,change):
    r=mixed;name='13_current_nightly_home';source=r['source'];root=r['root']
    if change in ('short','noncontiguous','badfps'):
        p=source/(name+'.capture.json');j=read(p)
        if change=='short':j['frames']=j['frames'][:839];j['duration']=839/30
        if change=='noncontiguous':j['frames'][-1]['pts']-=1
        if change=='badfps':j['encoded_fps']='60/1'
        write(p,j)
    elif change in ('gatefail','freeze','slow','unknown','samplergap','stall','bypass','cursor','topbar'):
        p=source/(name+'.gate.json');j=read(p)
        if change=='gatefail':j.update(passed=False,reasons=['capture loss'])
        if change in ('freeze','slow','unknown','samplergap','stall','bypass'):
            key=dict(freeze='app_freeze_seconds',slow='app_slow_seconds',unknown='app_unknown_seconds',
                     samplergap='app_gui_sampler_gaps_s',stall='app_stall_windows_s',bypass='adapter_bypasses')[change];j[key]=[1]
        if change=='cursor':j['cursor'].update(jump_count=1,jumps=[1])
        if change=='topbar':j['top_bar']['hidden']=False
        write(p,j)
    elif change in ('profile','provenance','cleanup'):
        p={'profile':r['live']/'capture-profile.json','provenance':source/'provenance.json','cleanup':r['live']/'gnome-host/cleanup.json'}[change];j=read(p)
        if change=='profile':j['show_alpha']=True
        if change=='provenance':j['completed_capture']=False
        if change=='cleanup':j['survivors_after_cleanup']=[123]
        write(p,j)
    elif change=='source':
        p=source.parent.parent/'exact-source.json';j=read(p);j['sha256']['spacr/capture_source.py']='0'*64;write(p,j)
    elif change=='approval':p=r['approval'];j=read(p);j['accepted']=False;write(p,j)
    elif change=='recordonlypolicy':write(r['policy'],dict(accepted=True,lesson=r['identity']))
    elif change=='missing_policy':r['receipt']['policy']['sha256']='0'*64
    elif change in ('other_scene','legacy_performance'):
        p=root/'scenes.json';j=read(p)
        if change=='other_scene':j['scenes'][3]['narration']='Changed'
        else:j['scenes'][7]['clip']['duration']=33.235
        write(p,j)
    elif change in ('timing','missing_playback_mode'):
        p=root/'video/native-live-timings.json';j=read(p)
        if change=='timing':j['scenes'][7]['duration']=33.
        else:del j['preserve_visual_intervals']
        write(p,j)
        with pytest.raises(ValueError):refresh.checked_retained_visual_timing(r['stage'],r['identity'],r['baseline'])
        return
    elif change in ('wrong_range','renderfailed'):
        p=r['render'];j=read(p)
        if change=='wrong_range':j['command'][j['command'].index('-filter_complex')+1]='[1:v]trim=end_frame=840[v]'
        else:j['returncode']=1
        write(p,j)
    elif change=='card_history':
        p=r['historical'];j=read(p);j[0]['returncode']=1;write(p,j);r['receipt']['replacements'][1]['historical_doctor']=refresh.binding(p)
    r['seal']()
    if change=='missing_policy':
        j=read(r['receipt_path']);j['policy']['sha256']='0'*64;write(r['receipt_path'],j)
    with pytest.raises(ValueError):refresh.checked_retained_visual_timing(r['stage'],r['identity'],r['baseline'])


def test_short_requested_native_frame_count_cannot_build_a_concat():
    with pytest.raises(ValueError):refresh.mixed_overlay_command('old','native','card','out',839)
