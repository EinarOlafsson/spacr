"""Explicit Classify overview: current navigation/ML plus one published CV excerpt."""
from copy import deepcopy
from pathlib import Path
import math
import os
import re

from stage_lesson import REPO,read,write
from check_completed_matrix import digest
from published_native_edit import encode_scene
from published_platform_native_refresh import rule,accepted_gate,reviewed_keep

LESSON='41_classify'
REVISION='classify-overview-v1'
RECORD='published-native-mix.json'
VISUALS=('main_00_home','main_02_family_choices','main_03_tabular_family',
    'ml_02_data_choice_load','ml_25_ml_figure_02','cv_24_batch_figure',
    'main_05_classifier_evaluation_tooltip','main_05_explain_cv_tooltip',
    'main_05_activation_tooltip','main_05_train_compare_tooltip',
    'main_06_feature_explorer_nested','ml_25_ml_figure_02')


def identity(value):
    from published_native_mix import identity as normal
    return normal(value)


def reviewed_result_quality(selection,gate,keep,count):
    """Admit only the reviewed two clean seconds of the original ML result."""
    if (selection.get('reviewed_result_capture_loss') is not True
            or selection.get('source_scene') not in (5,12)
            or selection.get('source_visual')!='25_ml_figure_02'
            or keep!=[[540,600]] or selection.get('final_result_frame')!=599
            or selection.get('omitted_setup_frames')!=[[0,540]]
            or selection.get('confirmed_idle_frames')!=[[600,count]]
            or not isinstance(selection.get('omitted_setup_reason'),str)
            or not selection['omitted_setup_reason'].strip()
            or gate.get('passed') is not False or gate.get('fps')!=30
            or gate.get('frames')!=count or gate.get('limits',{}).get('min_fps')!=27.9
            or not gate.get('capture_loss_seconds') or not gate.get('reasons')
            or any(not reason.startswith('choppy (capture loss):') for reason in gate['reasons'])):
        raise ValueError('Require the exact explicitly reviewed ML result capture-loss cut')
    for key in ('app_admission_freeze_seconds','app_freeze_seconds','app_slow_seconds',
                'app_unknown_seconds','app_gui_sampler_gaps_s','app_stall_windows_s','adapter_bypasses'):
        if gate.get(key)!=[]:
            raise ValueError('Reviewed ML result cut cannot waive application freezes/gaps/bypasses')
    cursor=gate.get('cursor',{});bar=gate.get('top_bar',{})
    if (gate['limits'].get('min_cursor_found')!=.9
            or cursor.get('found_fraction',-1)<.9 or cursor.get('jump_count')!=0
            or cursor.get('jumps')!=[] or bar.get('hidden') is not True
            or bar.get('visible_in')!=[] or bar.get('sampled_frames',0)<=0):
        raise ValueError('Reviewed ML result cut retains normal cursor/top-bar limits')
    rates=gate.get('distinct_per_second',[]);seconds=[18,19]
    if any(second in gate['capture_loss_seconds'] or second>=len(rates)
           or type(rates[second]) not in (int,float) or not math.isfinite(rates[second])
           or rates[second]<27.9 for second in seconds):
        raise ValueError('Every kept ML source second must independently meet 27.9 fps')
    return dict(scope='reviewed_ml_result_kept_seconds',raw_gate_passed=False,
        kept_frames=deepcopy(keep),kept_seconds=seconds,kept_distinct_per_second=[rates[i] for i in seconds],
        min_fps=27.9,omitted_setup_frames=deepcopy(selection['omitted_setup_frames']),
        confirmed_idle_frames=deepcopy(selection['confirmed_idle_frames']))


def selected_capture(selection,accepted):
    number=selection['source_scene']
    if type(number) is not int or not 1<=number<=12 or number==6:
        raise ValueError('Only the eleven indexed current main/ML roles are capture inputs')
    visual=VISUALS[number-1]
    role,raw=visual.split('_',1)
    if selection.get('visual')!=visual or selection.get('source_visual')!=raw:
        raise ValueError('Overview source role or indexed visual changed')
    capture=Path(selection['capture']).resolve()
    if capture.name!='classify_merged' or capture.parent.name!='captures':
        raise ValueError('Require an actual current unified Classify capture')
    run=capture.parent.parent
    paths={key:capture/(raw+suffix) for key,suffix in [('image','.png'),('video','.mp4'),
        ('native_video','.native.mp4'),('receipt','.capture.json'),('gate','.gate.json'),
        ('host','.host.json'),('pointer','.pointer.json')]}
    paths.update(frames=capture/'frames.json',provenance=capture/'provenance.json',
        scientific=capture/'scientific_acceptance.json',exact=run/'exact-source.json',
        profile=run/'capture-profile.json',host_run=run/'host-run.json',
        summary=capture/'host_gate_summary.json')
    if role=='ml':paths['batch']=capture/'batch_acceptance.json'
    if set(selection['files'])!=set(paths):
        raise ValueError('Bind every original capture/scientific/profile input')
    for key,path in paths.items():
        if (path.is_symlink() or not path.is_file() or selection['files'][key]!=
                dict(sha256=digest(path),bytes=path.stat().st_size)):
            raise ValueError('Actual overview source bytes changed: '+key)
    exact,provenance,host_run=(read(paths[key]) for key in ('exact','provenance','host_run'))
    if (not exact.get('fresh') or not exact.get('private_profile') or not exact.get('sha256')
            or provenance.get('module')!='classify_merged' or provenance.get('commit')!=exact.get('commit')
            or provenance.get('app_source_modified') is not False
            or host_run.get('app_exit_code')!=0 or host_run.get('recorder_rc')!=0
            or host_run.get('failure') is not None):
        raise ValueError('Overview needs actual successful source/host provenance')
    inventory=exact['sha256']
    if any(not isinstance(key,str) or Path(key).is_absolute() or '..' in Path(key).parts
           or not isinstance(value,str) or not re.fullmatch('[0-9a-f]{64}',value)
           for key,value in inventory.items()):
        raise ValueError('Invalid exact current capture inventory')
    profile=read(paths['profile'])
    if (profile.get('theme')!='dark' or profile.get('field')!='data_art_impulse_lens'
            or profile.get('field_density')!=.1 or profile.get('field_density_shipped')!=.1
            or profile.get('popup_backdrop')!='drift' or profile.get('requested_field_held_fps')!=60
            or any(profile.get(key) is not False for key in ('show_alpha','show_alpha_features',
                'show_alpha_species','spaceout','anonymous_reports_enabled','anonymous_report_notice_due'))):
        raise ValueError('Actual overview profile differs from its visible scope')
    scientific=read(paths['scientific'])
    if scientific.get('accepted') is not True:
        raise ValueError('Actual workflow scientific acceptance failed')
    batch=None
    if role=='main':
        from compose_classify_overview import check_navigation
        check_navigation(scientific)
    else:
        batch=read(paths['batch'])
        if batch.get('accepted') is not True:
            raise ValueError('Current ML result needs its actual complete batch acceptance')
    frame=read(paths['frames'])[raw]
    clip,receipt,gate=frame['clip'],read(paths['receipt']),read(paths['gate'])
    summary=read(paths['summary'])['clips'][raw]
    if (summary.get('file')!=paths['gate'].name or clip.get('gate')!=summary
            or any(summary.get(key)!=gate.get(key) for key in
                ('passed','reasons','distinct_fps','app_freeze_seconds','capture_loss_seconds'))):
        raise ValueError('Actual selected gate and source summary disagree')
    if (frame['image']!=paths['image'].name or frame['sha256']!=digest(paths['image'])
            or clip['video']!=paths['video'].name or clip['sha256']!=digest(paths['video'])
            or clip['receipt']!=paths['receipt'].name or clip['receipt_sha256']!=digest(paths['receipt'])
            or clip['native_video']!=paths['native_video'].name or clip['native_sha256']!=digest(paths['native_video'])
            or receipt.get('native',{}).get('sha256')!=clip['native_sha256']
            or receipt.get('native',{}).get('video')!=paths['native_video'].name
            or receipt.get('sha256')!=clip['sha256'] or receipt.get('size')!=[3840,2160]
            or receipt.get('full_decode_passed') is not True or receipt.get('audio_streams')!=0):
        raise ValueError('Actual overview clip/receipt identity changed')
    from native_live_timing import complete_native_duration
    import struct
    header=paths['image'].read_bytes()[:24]
    if header[:16]!=b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR' or struct.unpack('>II',header[16:24])!=(3840,2160):
        raise ValueError('Selected source image must be the actual native 4K frame')
    count=round(complete_native_duration(receipt,30)*30)
    partition={key:value for key,value in selection.items()
               if key not in ('source_visual','reviewed_result_capture_loss')}
    keep,edited=reviewed_keep(partition,count)
    quality=None
    if 'reviewed_result_capture_loss' in selection:
        quality=reviewed_result_quality(selection,gate,keep,count)
    else:
        accepted_gate(gate,capture,accepted)
    return dict(selection=deepcopy(selection),video=str(paths['video']),frames=count,
        keep=keep,edited=edited,inventory=inventory,source_commit=exact['commit'],
        scientific=scientific,batch=batch,source_provenance=provenance,kept_quality=quality)


def compatible_takes(captures):
    """Retakes may differ physically, never in their workflow/scientific proof."""
    for prefix in ('main_','ml_'):
        selected=[row for row in captures if row['selection']['visual'].startswith(prefix)]
        first=selected[0]
        if any(row['scientific']!=first['scientific'] or row['batch']!=first['batch']
               for row in selected[1:]):
            raise ValueError('Current workflow retakes have different scientific/batch acceptance content')


def same_source(captures,component,plan):
    wanted=plan.get('source_commit')
    allowed=plan.get('recorder_delta_paths',[])
    reason=plan.get('recorder_delta_reason')
    commits=plan.get('source_commits',[wanted])
    if (not isinstance(wanted,str) or not re.fullmatch('[0-9a-f]{40}',wanted)
            or not isinstance(allowed,list) or len(set(allowed))!=len(allowed)
            or any(not isinstance(name,str) or not name.startswith('tools/tutorials/capture')
                   or not name.endswith('.py') or '..' in Path(name).parts for name in allowed)
            or (allowed and (not isinstance(reason,str) or not reason.strip()))
            or not isinstance(commits,list) or not commits or len(set(commits))!=len(commits)
            or any(not isinstance(value,str) or not re.fullmatch('[0-9a-f]{40}',value) for value in commits)
            or (not allowed and commits!=[wanted])):
        raise ValueError('Require exact source commits and explicitly reviewed capture-only differences')
    lineage=component['source_capture_lineage']
    exacts=[row for row in lineage.get('source_files',[]) if Path(row.get('path','')).name=='exact-source.json']
    if not exacts:raise ValueError('Published CV needs its original exact-source lineage')
    original=None
    for row in exacts:
        path=Path(row['path'])
        if (not path.is_absolute() or path.is_symlink() or not path.is_file()
                or dict(path=str(path),sha256=digest(path),bytes=path.stat().st_size)!=row):
            raise ValueError('Published CV original exact-source binding changed')
        exact=read(path)
        if exact.get('commit')!=wanted or (original is not None and exact.get('sha256')!=original):
            raise ValueError('Published CV original application identity changed')
        original=exact.get('sha256')
    if not isinstance(original,dict) or not original:
        raise ValueError('Published CV original source inventory missing')
    observed=set();actual={wanted}
    for row in captures:
        inventory=row['inventory'];actual.add(row['source_commit'])
        if set(inventory)!=set(original):
            raise ValueError('Capture source inventories differ')
        observed.update(name for name,value in inventory.items() if value!=original[name])
    if observed!=set(allowed) or actual!=set(commits):
        raise ValueError('Actual source commits or recorder delta differ from the exact reviewed scope')
    return dict(source_commits=sorted(actual),recorder_delta_paths=sorted(observed),
        recorder_delta_reason=reason if allowed else None)


def plan_rows(english,tracks,captures,component,cv_row):
    if ([scene.get('visual') for scene in english['scenes']]!=list(VISUALS)
            or len(tracks)!=50 or any(len(track.get('scenes',[]))!=12 for track in tracks)
            or [row['selection']['source_scene'] for row in captures]!=[i for i in range(1,13) if i!=6]):
        raise ValueError('Require exact twelve overview roles and all fifty current tracks')
    selected={row['selection']['source_scene']:row for row in captures}
    rows=[];cursor=0
    for index,authored in enumerate(english['scenes'],1):
        spoken=[track['scenes'][index-1] for track in tracks]
        full=[float(row['speech_end'])-float(row['speech_start']) for row in spoken]
        durations=[float(row['duration']) for row in spoken]
        if any(not math.isfinite(value) or value<=0 for value in full+durations) or any(a>b for a,b in zip(full,durations)):
            raise ValueError('Invalid complete current speech timing')
        if index==6:
            row=deepcopy(cv_row);count=135;fence=4.5
            row.update(source_kind='published_native_cv_interval',fresh_capture=False)
        else:
            capture=selected[index];keep=capture['keep'];count=sum(b-a for a,b in keep)
            fence=capture['selection']['idle_tail_start']
            if capture['edited'] and count>math.floor(min(full)*30+1e-8):
                raise ValueError('Reviewed action cuts exceed the shortest complete overview speech')
            row=dict(source_kind='current_native_capture',fresh_capture=True,native_live_video=True,
                source_start_frame=0,source_keep_frames=deepcopy(keep),kept_frames=count,
                edited=capture['edited'],native_capture=deepcopy(capture))
        if type(fence) not in (int,float) or not math.isfinite(fence) or not 0<=fence<=min(min(full),count/30):
            raise ValueError('Every essential action fence must fit all fifty speech clips')
        required=max(durations)
        if index==12:
            for voice in spoken:
                sentences=voice.get('sentences');last=sentences[-1].get('audible_end') if sentences else None
                start,end=float(voice['speech_start']),float(voice['speech_end'])
                if type(last) not in (int,float) or not math.isfinite(last) or not start<=last<=end:
                    raise ValueError('Final overview result hold needs validated last audible timing')
                required=max(required,last-start+1)
        frames=max(count,math.ceil(required*30-1e-8))
        row.update(scene=index,visual=authored['visual'],text=authored['narration'],
            speech_start=cursor/30,speech_end=(cursor+frames)/30,scene_end=(cursor+frames)/30,
            duration=frames/30,final_frame_hold_frames=frames-count,idle_tail_start=fence,
            longest_narration_duration=max(durations))
        rows.append(row);cursor+=frames
    return rows


def inputs(stage,baseline,plan,policy):
    from native_live_timing import verified_narration_tracks
    from published_native_mix import classify_overview_cv_interval
    stage=Path(stage).resolve();baseline=Path(baseline).resolve();root=stage/'production'/LESSON
    if (stage==baseline or stage.is_relative_to(baseline) or baseline.is_relative_to(stage)
            or root.is_symlink() or (root/'video').is_symlink() or not root.resolve().is_relative_to(stage)
            or any((root/name).exists() for name in ['staged-native-edit.json','published-native-edit.json','published-native-reuse.json','published-retained-narration.json'])):
        raise ValueError('Overview requires its independent current-track mixed stage')
    english=read(root/'lesson.en.json')
    if (english!=read(REPO/'tools/tutorials/lessons'/f'{LESSON}.json')
            or plan.get('schema')!=1 or plan.get('lesson')!=LESSON or plan.get('revision')!=REVISION
            or plan.get('english_sha256')!=identity(english)):
        raise ValueError('Overview plan differs from exact current canonical source')
    selections=plan.get('captures',[])
    if (not isinstance(selections,list) or [selection.get('source_scene') for selection in selections]
            !=[index for index in range(1,13) if index!=6]):
        raise ValueError('Require exactly the eleven ordered current main/ML roles')
    tracks,proofs=verified_narration_tracks(stage,LESSON,english)
    component,cv_row=classify_overview_cv_interval(baseline,english,tracks,source_policy=policy)
    if plan.get('published_cv')!=component:
        raise ValueError('Reviewed plan must bind the actual fully published CV component')
    accepted=rule(plan)
    captures=[selected_capture(selection,accepted) for selection in selections]
    compatible_takes(captures)
    source_compatibility=same_source(captures,component,plan)
    rows=plan_rows(english,tracks,captures,component,cv_row)
    spec=dict(lesson=LESSON,kind='mixed_published_native_intervals',size=[3840,2160],fps=30,
        scenes=[dict(visual=row['visual'],narration=row['text'],published_source=row) for row in rows])
    lineage=dict(reference_revision=REVISION,published_cv=component,plan_sha256=identity(plan),
        source_compatibility=source_compatibility,
        fresh_capture=False,raw_capture_revalidated=False,fresh_native_scene_count=11,
        fresh_unique_clip_count=len({(row['selection']['capture'],row['selection']['source_visual']) for row in captures}),
        published_cv_scene_count=1,narration_source_kind='current_verified_fifty_tracks')
    timing=dict(schema=1,kind='native_visual_timing',lesson=LESSON,native_live_video=True,
        complete_native_clips=True,preserve_visual_intervals=True,fps=30,playback_speed=1,
        looped_or_stretched=False,total_duration=rows[-1]['scene_end'],scenes=rows,
        source_tracks=proofs,english_sha256=identity(english),published_native_mix=lineage,
        coverage=dict(accepted=True,gaps=[],track_count=50,fresh_native_scene_count=11,published_cv_scene_count=1))
    return root,baseline,spec,timing


def record_for(baseline,plan,policy,master):
    return dict(schema=1,kind='mixed_published_native_intervals',lesson=LESSON,
        reference_revision=REVISION,baseline_root=str(baseline),overview_plan=plan,
        source_idle_policy=policy,fresh_capture=False,raw_capture_revalidated=False,
        narration_source_kind='current_verified_fifty_tracks',rendered_master_sha256=digest(master))


def render(stage,baseline,plan,policy):
    root,baseline,spec,timing=inputs(stage,baseline,plan,policy)
    target=root/'video'/f'{LESSON}_silent.mp4';clock=target.with_name('native-live-timings.json');work=root/'mixed-render'
    if any(os.path.lexists(path) for path in [target,clock,work,root/'scenes.json',root/RECORD,root/'mixed-idle-tail-policy.json']):
        raise FileExistsError('Preserve earlier overview outputs and partial renders')
    from render_visual_master import concat
    work.mkdir();target.parent.mkdir(parents=True,exist_ok=True);parts=[]
    component=timing['published_native_mix']['published_cv']
    for row in timing['scenes']:
        source=baseline/component['published_master']['path'] if row['scene']==6 else row['native_capture']['video']
        part=work/f'scene-{row["scene"]:02}.mp4'
        encode_scene(source,row,part,terminal_seekable=row['scene']==12);parts.append(part)
    concat(parts,target,work/'concat.txt')
    if inputs(stage,baseline,plan,policy)[2:]!=(spec,timing):raise ValueError('Overview sources changed during rendering')
    write(root/'scenes.json',spec);timing['visual_spec_sha256']=digest(root/'scenes.json')
    write(root/RECORD,record_for(baseline,plan,policy,target));write(clock,timing)
    fences=[dict(visual=row['visual'],source_start=row['speech_start'],source_end=row['scene_end'],
                 idle_tail_start=row['idle_tail_start']) for row in timing['scenes']]
    write(root/'mixed-idle-tail-policy.json',dict(schema=1,lessons=[dict(lesson=LESSON,
        master_sha256=digest(target),visual_timing_sha256=digest(clock),scenes=fences)]))
    return clock


def checked(stage):
    root=Path(stage)/'production'/LESSON;record=read(root/RECORD)
    root,baseline,spec,timing=inputs(stage,record['baseline_root'],record['overview_plan'],record['source_idle_policy'])
    if record!=record_for(baseline,record['overview_plan'],record['source_idle_policy'],root/'video'/f'{LESSON}_silent.mp4') or read(root/'scenes.json')!=spec:
        raise ValueError('Overview component record changed')
    timing['visual_spec_sha256']=digest(root/'scenes.json');clock=root/'video/native-live-timings.json'
    if read(clock)!=timing:raise ValueError('Overview timing or current fifty-track proof changed')
    return clock
