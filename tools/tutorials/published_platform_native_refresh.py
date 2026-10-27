"""Replace only Platform's two native companions; retain twelve published intervals."""
from copy import deepcopy
from pathlib import Path
import importlib.util
import math
import os
import re

from stage_lesson import REPO, read, write
from check_completed_matrix import digest
from published_native_reuse import _published_inputs, interval_spec
from published_retained_narration import checked_published_narration, RECORD as AUDIO_RECORD
from published_native_edit import encode_scene

LESSON = '04_platform_installers'
REVISION = 'native-companions-v1'
RULE_SHA = '81196710e6a442c91f65f924858f849f8ed8659fa4b7faf8df2f0c82677a15f8'
SLOTS = {8: '14_current_nightly_performance', 9: '13_current_nightly_home'}
RECORD = 'published-native-mix.json'


def identity(value):
    from published_native_mix import identity as normal
    return normal(value)


def rule(plan):
    """Use the existing exact campaign rule, never a second near-miss threshold."""
    record = plan['admission_rule']
    path = Path(record['path'])
    if (not path.is_absolute() or path.is_symlink() or record.get('sha256') != RULE_SHA
            or digest(path) != RULE_SHA):
        raise ValueError('Require the reviewed owning campaign admission rule')
    spec = importlib.util.spec_from_file_location('_platform_campaign_gate', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.accepted


def accepted_gate(gate, captures, accepted):
    """Bind normal admission while retaining independent raw freeze rejection."""
    if (not accepted(gate, str(captures))
            or any(gate.get(key) for key in ('app_admission_freeze_seconds',
                'app_freeze_seconds', 'app_slow_seconds', 'app_unknown_seconds'))):
        raise ValueError('Selected native companion failed normal admission or raw freeze evidence')


def reviewed_keep(selection, source_frames):
    """Apply the existing native-edit half-open partition contract to a raw clip."""
    required={'keep_frames','confirmed_idle_frames','final_result_frame'}
    setup_fields={'omitted_setup_frames','omitted_setup_reason'}
    if set(selection)-{'source_scene','visual','capture','files','idle_tail_start'}-required-setup_fields:
        raise ValueError('Unknown reviewed native selection fields')
    supplied=set(selection)&(required|setup_fields)
    if not supplied:
        return [[0,source_frames]],False
    has_setup=bool(supplied&setup_fields)
    if supplied!=(required|setup_fields if has_setup else required):
        raise ValueError('Reviewed native cuts require complete interval/result fields')
    keep,idle=selection['keep_frames'],selection['confirmed_idle_frames']
    setup=selection.get('omitted_setup_frames')
    if has_setup and (not isinstance(selection['omitted_setup_reason'],str)
                      or not selection['omitted_setup_reason'].strip()):
        raise ValueError('Omitted setup requires an explicit reviewed reason')
    def intervals(value,allow_empty=False):
        return (isinstance(value,list) and (bool(value) or allow_empty)
            and all(isinstance(p,list) and len(p)==2 and all(type(x) is int for x in p)
                    and 0<=p[0]<p[1]<=source_frames for p in value)
            and all(a[1]<b[0] for a,b in zip(value,value[1:])))
    if (not intervals(keep) or not intervals(idle,allow_empty=has_setup)
            or (has_setup and not intervals(setup))):
        raise ValueError('Invalid reviewed native frame intervals')
    ordered=sorted([(a,b) for a,b in keep]+[(a,b) for a,b in idle]+[(a,b) for a,b in (setup or [])])
    if ordered[0][0]!=0 or ordered[-1][1]!=source_frames or any(a[1]!=b[0] for a,b in zip(ordered,ordered[1:])):
        raise ValueError('Every raw frame must be kept, confirmed idle or omitted setup without overlap')
    if type(selection['final_result_frame']) is not int or selection['final_result_frame']!=keep[-1][1]-1:
        raise ValueError('Hold only the reviewed final kept result frame')
    return deepcopy(keep),True


def selected_capture(selection, accepted):
    number, visual = selection['source_scene'], selection['visual']
    if SLOTS.get(number) != visual:
        raise ValueError('Only indexed Performance8 and Home9 may be replaced')
    capture = Path(selection['capture']).resolve()
    if capture.name != 'home' or capture.parent.name != 'captures':
        raise ValueError('Expected an actual Home capture directory')
    run = capture.parent.parent
    paths = {key: capture / (visual + suffix) for key, suffix in
             [('image', '.png'), ('video', '.mp4'), ('native_video', '.native.mp4'),
              ('receipt', '.capture.json'), ('gate', '.gate.json'), ('host', '.host.json'),
              ('pointer', '.pointer.json')]}
    paths.update(frames=capture/'frames.json', provenance=capture/'provenance.json',
                 exact=run/'exact-source.json', profile=run/'capture-profile.json',
                 host_run=run/'host-run.json', summary=capture/'host_gate_summary.json')
    if set(selection['files']) != set(paths):
        raise ValueError('Bind every original selected capture input')
    for name, path in paths.items():
        expected = selection['files'][name]
        if (path.is_symlink() or not path.is_file() or expected !=
                dict(sha256=digest(path), bytes=path.stat().st_size)):
            raise ValueError('Selected capture bytes changed: '+name)
    exact, provenance, host_run = (read(paths[key]) for key in ('exact','provenance','host_run'))
    if (not exact.get('fresh') or not exact.get('private_profile') or not exact.get('sha256')
            or provenance.get('commit') != exact.get('commit') or provenance.get('module') != 'home'
            or provenance.get('app_source_modified') is not False
            or host_run.get('app_exit_code') != 0 or host_run.get('recorder_rc') != 0
            or host_run.get('failure') is not None):
        raise ValueError('Actual capture source/host provenance failed')
    profile = read(paths['profile'])
    if (profile.get('theme') != 'dark' or profile.get('field') != 'data_art_impulse_lens'
            or profile.get('field_density') != .1 or profile.get('field_density_shipped') != .1
            or profile.get('popup_backdrop') != 'drift' or profile.get('requested_field_held_fps') != 60
            or any(profile.get(k) is not False for k in ('show_alpha','show_alpha_features',
                'show_alpha_species','spaceout','anonymous_reports_enabled','anonymous_report_notice_due'))):
        raise ValueError('Actual capture profile differs from its normal visible scope')
    inventory = exact['sha256']
    if any(not isinstance(k,str) or Path(k).is_absolute() or '..' in Path(k).parts
           or not isinstance(v,str) or not re.fullmatch('[0-9a-f]{64}',v)
           for k,v in inventory.items()):
        raise ValueError('Invalid actual source inventory')
    frame = read(paths['frames'])[visual]
    clip, receipt, gate = frame['clip'], read(paths['receipt']), read(paths['gate'])
    summary = read(paths['summary'])['clips'][visual]
    if (summary.get('file') != paths['gate'].name or clip.get('gate') != summary
            or any(summary.get(k) != gate.get(k) for k in
                   ('passed','reasons','distinct_fps','app_freeze_seconds','capture_loss_seconds'))):
        raise ValueError('Selected clip and normal gate summary disagree')
    accepted_gate(gate, capture, accepted)
    if (frame['image'] != paths['image'].name or frame['sha256'] != digest(paths['image'])
            or clip['video'] != paths['video'].name or clip['receipt'] != paths['receipt'].name
            or clip['sha256'] != digest(paths['video']) or clip['receipt_sha256'] != digest(paths['receipt'])
            or clip['native_video'] != paths['native_video'].name
            or clip['native_sha256'] != digest(paths['native_video'])
            or receipt.get('native',{}).get('sha256') != clip['native_sha256']
            or receipt.get('native',{}).get('video') != paths['native_video'].name
            or receipt.get('sha256') != clip['sha256'] or receipt.get('size') != [3840,2160]
            or receipt.get('full_decode_passed') is not True or receipt.get('audio_streams') != 0):
        raise ValueError('Native handoff/receipt/poster identity changed')
    from native_live_timing import complete_native_duration
    duration = complete_native_duration(receipt,30)
    keep,_ = reviewed_keep(selection,round(duration*30))
    import struct
    header = paths['image'].read_bytes()[:24]
    if header[:16] != b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR' or struct.unpack('>II',header[16:24]) != (3840,2160):
        raise ValueError('Native poster must be the actual full4K PNG')
    fence = selection['idle_tail_start']
    if (type(fence) not in (int,float) or not math.isfinite(fence)
            or not 0 <= fence <= sum(b-a for a,b in keep)/30):
        raise ValueError('Require the actual source-bound reviewed action fence')
    return dict(selection=deepcopy(selection), frames=round(duration*30),
                video=str(paths['video']), source_commit=exact['commit'], inventory=inventory,
                source_run_completed=provenance.get('completed_capture') is True,
                source_provenance=provenance, native_live_video=True, fresh_capture=True)


def compatible_sources(replacements, plan):
    first = replacements[0]['inventory']
    observed = set()
    for row in replacements[1:]:
        current = row['inventory']
        if set(current) != set(first):
            raise ValueError('Selected companion source inventories differ')
        observed.update(k for k in first if first[k] != current[k])
    allowed = plan.get('recorder_delta_paths', [])
    if (len(allowed) != len(set(allowed)) or observed != set(allowed)
            or any(not p.startswith('tools/tutorials/capture') or not p.endswith('.py')
                   or '..' in Path(p).parts for p in allowed)
            or (observed and not plan.get('recorder_delta_reason'))):
        raise ValueError('Only exact explicitly reviewed capture-driver deltas are compatible')


def plan_rows(english, original, tracks, replacements, fences):
    import sys
    sys.path.insert(0,str(REPO/'tools/tutorials/authoring/tools'))
    from render_platform_maintenance_references import VISUALS
    if (english.get('id') != LESSON or len(tracks) != 50
            or [s['visual'] for s in english['scenes']] != list(VISUALS)
            or [s['visual'] for s in original['scenes']] != list(VISUALS)
            or any(type(s.get('native_live_video')) is not bool for s in original['scenes'])
            or [s.get('native_live_video') for s in original['scenes']] !=
               [i in SLOTS for i in range(1,15)]
            or [r['selection']['source_scene'] for r in replacements] != [8,9]
            or any(len(t['scenes']) != 14 for t in tracks) or set(fences) != set(VISUALS)):
        raise ValueError('Require exact Platform14, fifty tracks, twelve held and two indexed native companions')
    by_number = {r['selection']['source_scene']:r for r in replacements}
    rows, cursor = [], 0
    for index, (authored, old) in enumerate(zip(english['scenes'],original['scenes']),1):
        voices = [t['scenes'][index-1] for t in tracks]
        full = [float(v['speech_end'])-float(v['speech_start']) for v in voices]
        durations = [float(v['duration']) for v in voices]
        if any(not math.isfinite(x) or x<=0 for x in full+durations) or any(a>b+1e-9 for a,b in zip(full,durations)):
            raise ValueError('Invalid unchanged complete-speech timing')
        new = by_number.get(index)
        keep,edited = reviewed_keep(new['selection'],new['frames']) if new else ([[0,round(old['duration']*30)]],False)
        count = sum(b-a for a,b in keep)
        fence = new['selection']['idle_tail_start'] if new else fences[authored['visual']]
        if (type(fence) not in (int,float) or not math.isfinite(fence) or fence<0 or fence > min(full)+1e-9
                or (new and fence>count/30+1e-9)):
            raise ValueError('Essential action fence exceeds shortest complete speech')
        if edited and count>math.floor(min(full)*30+1e-8):
            raise ValueError('Reviewed essential frames exceed shortest complete speech')
        hold = max(0,math.ceil(max(durations)*30-1e-8)-count) if new else 0
        if not new and max(durations)>count/30+1e-9:
            raise ValueError('Retained reference narration exceeds unchanged published interval')
        if index == len(english['scenes']):
            required = max(durations)
            for spoken in voices:
                sentences = spoken.get('sentences')
                if not isinstance(sentences,list) or not sentences:
                    raise ValueError('Final reference hold requires validated last audible timing')
                last = sentences[-1].get('audible_end')
                start,end = float(spoken['speech_start']),float(spoken['speech_end'])
                if type(last) not in (int,float) or not math.isfinite(last) or not start<=last<=end:
                    raise ValueError('Final reference hold requires validated last audible timing')
                required = max(required,last-start+1)
            hold = max(hold,math.ceil(required*30-1e-8)-count,0)
        row = deepcopy(old)
        row.update(scene=index,visual=authored['visual'],text=authored['narration'],
            speech_start=cursor/30,speech_end=(cursor+count+hold)/30,
            scene_end=(cursor+count+hold)/30,duration=(count+hold)/30,
            source_start_frame=0 if new else round(old['speech_start']*30),
            source_keep_frames=keep,kept_frames=count,final_frame_hold_frames=hold,
            edited=edited,fresh_capture=new is not None,longest_narration_duration=max(durations),
            source_kind='current_native_companion' if new else 'unchanged_published_interval',
            original_source=deepcopy(old))
        if new:row['native_companion']=deepcopy(new)
        if edited:row['source_kind']='reviewed_native_companion_edit'
        if not new and hold:
            row['source_kind']='published_interval_with_final_result_hold'
        rows.append(row);cursor+=count+hold
    return rows


def inputs(stage, baseline, plan, policy):
    from idle_tail_policy import bind_policy
    stage = Path(stage).resolve();root = stage/'production'/LESSON
    baseline, original, bindings, source = _published_inputs(baseline,LESSON)
    if stage==baseline or stage.is_relative_to(baseline) or baseline.is_relative_to(stage):
        raise ValueError('Require an independent retained narration stage')
    if (root.is_symlink() or (root/'video').is_symlink() or not root.resolve().is_relative_to(stage)
            or any((root/name).exists() for name in ('published-native-reuse.json',
                      'published-native-edit.json','staged-native-edit.json'))):
        raise ValueError('Native companions require an exclusive contained mixed stage')
    english = read(root/'lesson.en.json')
    if (english != read(REPO/'tools/tutorials/lessons'/f'{LESSON}.json')
            or Path(read(root/AUDIO_RECORD)['baseline_root']).resolve()!=baseline):
        raise ValueError('Unchanged canonical and fifty-track baseline must match')
    if (plan.get('schema')!=1 or plan.get('lesson')!=LESSON or plan.get('revision')!=REVISION
            or plan.get('english_sha256')!=identity(english)
            or plan.get('master_sha256')!=bindings['master']['sha256']
            or plan.get('visual_timing_sha256')!=bindings['timing']['sha256']):
        raise ValueError('Native companion plan differs from exact canonical/published source')
    interval_spec(english,original)
    tracks, proofs = checked_published_narration(stage,LESSON)
    accepted = rule(plan)
    selections = plan['replacements']
    if ([s.get('source_scene') for s in selections]!=[8,9]
            or [s.get('visual') for s in selections]!=list(SLOTS.values())):
        raise ValueError('Require only indexed companion8 and9, in order')
    replacements = [selected_capture(s,accepted) for s in selections]
    compatible_sources(replacements,plan)
    lesson = dict(id=LESSON,silent=bindings['master']['path'].removeprefix('media_host/'),
                  visual_timings=bindings['timing']['path'].removeprefix('media_host/'))
    verified = bind_policy(policy,list(bindings.values()),[lesson],lambda _:original)[0]
    fences = {s['visual']:s['idle_tail_start'] for s in verified['native_idle_tails']['scenes']}
    rows = plan_rows(english,original,tracks,replacements,fences)
    spec = dict(lesson=LESSON,kind='mixed_published_native_intervals',size=[3840,2160],fps=30,
                scenes=[dict(visual=s['visual'],narration=s['narration'],published_source=r)
                        for s,r in zip(english['scenes'],rows)])
    lineage = dict(**{k:v for k,v in source.items() if k!='baseline_root'},
        source_master_sha256=bindings['master']['sha256'],source_timing_sha256=bindings['timing']['sha256'],
        reference_revision=REVISION,plan_sha256=identity(plan),source_idle_policy_sha256=identity(policy),
        fresh_capture=False,raw_capture_revalidated=False,fresh_native_scene_count=2,
        retained_published_scene_count=12)
    timing = dict(schema=1,kind='native_visual_timing',lesson=LESSON,native_live_video=True,
        complete_native_clips=True,preserve_visual_intervals=True,
        fps=30,playback_speed=1,looped_or_stretched=False,
        total_duration=rows[-1]['scene_end'],scenes=rows,source_tracks=proofs,
        english_sha256=identity(english),published_native_mix=lineage,
        coverage=dict(accepted=True,gaps=[],track_count=50,source_native_scene_count=2,
                      retained_published_scene_count=12))
    return root,baseline,bindings,source,spec,timing


def record_for(source,plan,policy,master):
    return dict(schema=1,kind='mixed_published_native_intervals',lesson=LESSON,**source,
                fresh_capture=False,raw_capture_revalidated=False,reference_revision=REVISION,
                native_companions_plan=plan,source_idle_policy=policy,
                rendered_master_sha256=digest(master))


def render(stage,baseline,plan,policy):
    root,baseline,bindings,source,spec,timing = inputs(stage,baseline,plan,policy)
    target=root/'video'/f'{LESSON}_silent.mp4';clock=target.with_name('native-live-timings.json')
    work=root/'mixed-render'
    if any(os.path.lexists(p) for p in (target,clock,root/'scenes.json',root/RECORD,
                                      root/'mixed-idle-tail-policy.json',work)):
        raise FileExistsError('Preserve existing companion outputs and partial render')
    work.mkdir();target.parent.mkdir(parents=True,exist_ok=True)
    from render_visual_master import concat
    parts=[]
    for row in timing['scenes']:
        path=work/f'scene-{row["scene"]:02}.mp4'
        video = row['native_companion']['video'] if row['fresh_capture'] else baseline/bindings['master']['path']
        encode_scene(video,row,path,terminal_seekable=row['scene']==14)
        parts.append(path)
    concat(parts,target,work/'concat.txt')
    if inputs(stage,baseline,plan,policy)[4:]!=(spec,timing):
        raise ValueError('Companion inputs changed during rendering')
    write(root/'scenes.json',spec);timing['visual_spec_sha256']=digest(root/'scenes.json')
    write(root/RECORD,record_for(source,plan,policy,target));write(clock,timing)
    source_entry=next(s for s in policy['lessons'] if s['lesson']==LESSON)
    old_fences={s['visual']:s['idle_tail_start'] for s in source_entry['scenes']}
    for selection in plan['replacements']:old_fences[selection['visual']]=selection['idle_tail_start']
    rows=[dict(visual=r['visual'],source_start=r['speech_start'],source_end=r['scene_end'],
               idle_tail_start=old_fences[r['visual']]) for r in timing['scenes']]
    write(root/'mixed-idle-tail-policy.json',dict(schema=1,lessons=[dict(lesson=LESSON,
        master_sha256=digest(target),visual_timing_sha256=digest(clock),scenes=rows)]))
    return clock


def checked(stage):
    root=Path(stage)/'production'/LESSON;record=read(root/RECORD)
    root,_,_,source,spec,timing=inputs(stage,record['baseline_root'],
                                    record['native_companions_plan'],record['source_idle_policy'])
    if record!=record_for(source,record['native_companions_plan'],record['source_idle_policy'],
                           root/'video'/f'{LESSON}_silent.mp4') or read(root/'scenes.json')!=spec:
        raise ValueError('Native companion source record changed')
    timing['visual_spec_sha256']=digest(root/'scenes.json');clock=root/'video/native-live-timings.json'
    if read(clock)!=timing:raise ValueError('Native companion clock or fifty-track proof changed')
    return clock
