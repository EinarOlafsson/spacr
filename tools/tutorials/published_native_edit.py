"""Explicit 1x edits of verified published or accepted staged footage."""
from copy import deepcopy
import argparse,hashlib,json,math,os,re,subprocess,sys
from pathlib import Path
from published_native_reuse import _published_inputs,interval_spec
from stage_lesson import REPO,read,write
from check_completed_matrix import digest

RECORD='published-native-edit.json'
STAGED_RECORD='staged-native-edit.json'

def _staged_inputs(stage,identity):
    """Own an accepted private master without calling it a publication."""
    from native_live_timing import checked_native_timing
    from check_completed_matrix import reconcile_visual_sources
    stage=Path(stage).resolve();root=stage/'production'/identity
    if not re.fullmatch(r'[0-9]+_[a-z0-9_]+',identity) or not root.resolve().is_relative_to(stage):
        raise ValueError('Expected one staged lesson inside its source stage')
    if any((root/name).exists() for name in [RECORD,STAGED_RECORD,'published-native-reuse.json','published-native-mix.json']):
        raise ValueError('Staged edit source must be the original accepted capture stage')
    english=read(root/'lesson.en.json')
    if english!=read(REPO/'tools/tutorials/lessons'/f'{identity}.json'):
        raise ValueError('Staged edit source differs from current canonical narration')
    timing_path=checked_native_timing(stage,identity)
    if timing_path is None:raise ValueError('Staged edits require a checked native source clock')
    reconcile_visual_sources(stage,identity,english)
    master=root/'video'/f'{identity}_silent.mp4';spec=root/'scenes.json'
    acceptance_path=root/'current-video-acceptance.json';acceptance=read(acceptance_path)
    expected={'lesson':identity,'master_sha256':digest(master),'timing_sha256':digest(timing_path),
        'scene_specification_sha256':digest(spec),'canonical_english_sha256':hashlib.sha256(json.dumps(
            english,sort_keys=True,ensure_ascii=False).encode()).hexdigest()}
    if (any(acceptance.get(k)!=v for k,v in expected.items())
            or any(acceptance.get(k) is not True for k in ['native_fifty_voice_timeline',
                'master_full_decode_passed','master_frame_count_matches_scene_boundaries',
                'master_all_presentation_times_checked'])):
        raise ValueError('Staged native source needs exact normal finish acceptance')
    report_path=stage/'build-module-stage-report.json';report=read(report_path)
    if report.get('lesson')!=identity or not report.get('sources'):
        raise ValueError('Staged source needs its original capture composition report')
    files={master,timing_path,spec,root/'visual.json',root/'lesson.en.json',acceptance_path,report_path}
    if any(not p.resolve().is_relative_to(stage) for p in files):
        raise ValueError('Accepted staged source files must stay inside their source stage')
    for value in report['sources']:
        capture=Path(value)
        if not capture.is_absolute():raise ValueError('Original capture source must be absolute')
        exact=capture/'exact-source.json'
        if not exact.exists():exact=capture.parent.parent/'exact-source.json'
        required=[capture/'frames.json',capture/'provenance.json',capture/'host_gate_summary.json',exact]
        if any(not p.is_file() for p in required) or not read(exact).get('sha256'):
            raise ValueError('Staged source capture provenance is incomplete')
        files.update(required)
    def binding(path):
        if path.is_symlink():raise ValueError('Staged source files must be regular immutable inputs')
        return dict(path=str(path.resolve()),sha256=digest(path),bytes=path.stat().st_size)
    source=dict(source_stage_root=str(stage),source_files=[binding(p) for p in sorted(files)])
    bindings={key:dict(path=path.relative_to(stage).as_posix(),sha256=digest(path),bytes=path.stat().st_size)
              for key,path in [('master',master),('timing',timing_path)]}
    return stage,read(timing_path),bindings,source


def _staged_policy(policy,identity,original,bindings,rows,tracks):
    """Validate reviewed untouched fences before encoding a single new frame."""
    from idle_tail_policy import bind_policy
    if (not isinstance(policy,dict) or set(policy)!={'schema','lessons'} or policy['schema']!=1
            or not isinstance(policy['lessons'],list)):
        raise ValueError('Invalid staged source policy')
    selected=[entry for entry in policy['lessons'] if entry.get('lesson')==identity]
    records=[dict(row,path='media_host/'+row['path'].removeprefix('production/').removeprefix('media_host/')) for row in bindings.values()]
    lesson={'id':identity,'silent':bindings['master']['path'].removeprefix('production/').removeprefix('media_host/'),
            'visual_timings':bindings['timing']['path'].removeprefix('production/').removeprefix('media_host/')}
    bound=bind_policy(dict(schema=1,lessons=selected),records,[lesson],lambda _:original)[0]
    inherited={}
    for fence in bound.get('native_idle_tails',{}).get('scenes',[]):
        matches=[i for i,scene in enumerate(original['scenes']) if scene['visual']==fence['visual']
                 and abs(scene['speech_start']-fence['source_start'])<=1e-7
                 and abs(scene['scene_end']-fence['source_end'])<=1e-7]
        inherited[matches[0]]=fence['idle_tail_start']
    for index,row in enumerate(rows):
        fence=row['kept_frames']/30 if row['edited'] else inherited.get(index)
        if fence is None or any(fence>float(t['scenes'][index]['speech_end'])-
                              float(t['scenes'][index]['speech_start']) for t in tracks):
            raise ValueError('Every staged action fence must fit all fifty complete speech clips')
    return inherited

def plan_intervals(english,original,tracks,edit,bindings):
    """All indices are source-scene-relative integer frames, half-open."""
    interval_spec(english,original)  # Normal exact visual order and count check.
    if (set(edit)!={'schema','lesson','master_sha256','visual_timing_sha256','scenes'}
            or edit['schema']!=1 or edit['lesson']!=english['id']
            or edit['master_sha256']!=bindings['master']['sha256']
            or edit['visual_timing_sha256']!=bindings['timing']['sha256']):
        raise ValueError('Reviewed edit is not bound to this published source')
    if len(tracks)!=50:raise ValueError('Every current voice must be checked')
    edits={}
    for selected in edit['scenes']:
        matches=[i for i,source in enumerate(original['scenes']) if source['visual']==selected['visual']]
        if 'source_scene' in selected:
            number=selected['source_scene']
            if type(number) is not int or not 1<=number<=len(original['scenes']) or number-1 not in matches:
                raise ValueError('Edited source scene index/visual changed')
            index=number-1
        elif len(matches)==1:index=matches[0]
        else:raise ValueError('Unknown or ambiguous edited visual requires source_scene')
        source=original['scenes'][index]
        start,end,duration=(float(source[k]) for k in ('speech_start','scene_end','duration'))
        if (source.get('scene',index+1)!=index+1 or not all(math.isfinite(v) for v in (start,end,duration))
                or start<0 or duration<=0 or abs(end-start-duration)>1e-8
                or any(abs(v*30-round(v*30))>1e-5 for v in (start,end))):
            raise ValueError('Edited source scene clock changed')
        if index in edits:raise ValueError('Duplicate edited source scene')
        edits[index]=selected
    if not edits:raise ValueError('Missing edit scenes')
    rows=[];cursor=0
    for index,(source,authored) in enumerate(zip(original['scenes'],english['scenes'])):
        source_frames=round(source['duration']*30)
        spans=[];durations=[]
        for track in tracks:
            if len(track['scenes'])!=len(original['scenes']):raise ValueError('Voice scene count changed')
            spoken=track['scenes'][index]
            span=float(spoken['speech_end'])-float(spoken['speech_start'])
            duration=float(spoken['duration'])
            if not math.isfinite(span) or not math.isfinite(duration) or not 0<span<=duration:
                raise ValueError('Invalid current voice speech interval')
            spans.append(span);durations.append(duration)
        selected=edits.get(index)
        keep=[[0,source_frames]];idle=[];setup=None
        if selected is not None:
            required={'visual','keep_frames','confirmed_idle_frames','final_result_frame'}
            if 'source_scene' in selected:required.add('source_scene')
            setup_fields={'omitted_setup_frames','omitted_setup_reason'}
            has_setup=bool(set(selected)&setup_fields)
            if (set(selected)!=(required|setup_fields if has_setup else required)
                    or source.get('native_live_video') is not True):
                raise ValueError('Only explicit reviewed native scenes can be edited')
            keep=selected['keep_frames'];idle=selected['confirmed_idle_frames']
            setup=selected.get('omitted_setup_frames')
            if has_setup and (not isinstance(selected['omitted_setup_reason'],str) or not selected['omitted_setup_reason'].strip()):
                raise ValueError('Omitted setup requires an explicit reviewed reason')
            def intervals(value,allow_empty=False):
                return (isinstance(value,list) and (bool(value) or allow_empty) and all(isinstance(p,list) and len(p)==2
                    and all(type(x) is int for x in p) and 0<=p[0]<p[1]<=source_frames for p in value)
                    and all(a[1]<b[0] for a,b in zip(value,value[1:])))
            if (not intervals(keep) or not intervals(idle,allow_empty=has_setup)
                    or (has_setup and not intervals(setup))):
                raise ValueError('Invalid reviewed frame intervals')
            ordered=sorted([(a,b,'kept') for a,b in keep]+[(a,b,'idle') for a,b in idle]
                           +[(a,b,'setup') for a,b in (setup or [])])
            if ordered[0][0]!=0 or ordered[-1][1]!=source_frames or any(a[1]!=b[0] for a,b in zip(ordered,ordered[1:])):
                raise ValueError('Every source frame must be kept, confirmed idle or explicitly omitted setup, without overlap')
            if (type(selected['final_result_frame']) is not int
                    or selected['final_result_frame']!=keep[-1][1]-1):
                raise ValueError('Hold only the reviewed final kept result frame')
            if sum(b-a for a,b in keep)>math.floor(min(spans)*30+1e-8):
                raise ValueError('Essential action frames do not fit the shortest full speech clip')
        kept=sum(b-a for a,b in keep)
        if selected is None and max(durations)>source_frames/30+1e-9:
            raise ValueError('New narration exceeds an untouched published interval')
        required_duration=max(durations)
        if selected is not None and index==len(original['scenes'])-1:
            # EOF has an explicit one-second sound-to-end policy. Hold the
            # reviewed result frame; never stretch recorded motion or audio.
            for track in tracks:
                spoken=track['scenes'][index]
                sentences=spoken.get('sentences')
                if not isinstance(sentences,list) or not sentences:
                    raise ValueError('Edited final hold requires validated last audible timing')
                last=sentences[-1].get('audible_end')
                start=float(spoken['speech_start']);end=float(spoken['speech_end'])
                if (type(last) not in (int,float) or not math.isfinite(last)
                        or not start<=last<=end):
                    raise ValueError('Edited final hold requires validated last audible timing')
                required_duration=max(required_duration,last-start+1)
        frames=source_frames if selected is None else max(kept,math.ceil(required_duration*30-1e-8))
        rows.append(dict(scene=index+1,visual=source['visual'],text=authored['narration'],
            speech_start=cursor/30,speech_end=(cursor+frames)/30,scene_end=(cursor+frames)/30,
            duration=frames/30,native_live_video=source.get('native_live_video') is True,
            longest_narration_duration=max(durations),source_start_frame=round(source['speech_start']*30),
            source_keep_frames=keep,confirmed_idle_frames=idle,kept_frames=kept,
            final_frame_hold_frames=frames-kept,edited=selected is not None))
        if setup is not None:
            rows[-1].update(omitted_setup_frames=deepcopy(setup),
                            omitted_setup_reason=selected['omitted_setup_reason'])
        cursor+=frames
    return rows


def inputs(stage,identity,baseline,edit,*,staged_source=None,source_idle_policy=None):
    from native_live_timing import verified_narration_tracks
    stage=Path(stage).resolve();root=stage/'production'/identity
    if staged_source is not None:
        if baseline is not None or source_idle_policy is None:
            raise ValueError('Choose explicit staged source and reviewed source policy, not a published baseline')
        baseline,original,bindings,source=_staged_inputs(staged_source,identity)
        source_idle_policy=Path(source_idle_policy).resolve()
        source['source_idle_policy']=dict(path=str(source_idle_policy),sha256=digest(source_idle_policy))
    else:
        baseline,original,bindings,source=_published_inputs(baseline,identity)
        if source_idle_policy is not None:
            source_idle_policy=Path(source_idle_policy).resolve()
            source['source_idle_policy']=dict(path=str(source_idle_policy),sha256=digest(source_idle_policy))
    if not root.resolve().is_relative_to(stage) or not (root/'video').resolve().is_relative_to(root.resolve()):
        raise ValueError('Published edit destinations must stay inside their independent stage')
    if stage==baseline or stage.is_relative_to(baseline) or baseline.is_relative_to(stage):
        raise ValueError('Edited outputs require an independent stage, never a published-source tree')
    if (root/'published-native-reuse.json').exists():
        raise ValueError('Choose edited footage or unchanged master reuse, not both')
    english=read(root/'lesson.en.json')
    if english!=read(REPO/'tools/tutorials/lessons'/f'{identity}.json'):
        raise ValueError('Edited narration must match current canonical source')
    tracks,proofs=verified_narration_tracks(stage,identity,english)
    if staged_source is not None:
        if proofs!=original.get('source_tracks'):
            raise ValueError('Staged edits must retain all fifty original audio/timing bytes')
        from audit_staged_catalogs import CATALOGS
        for name in CATALOGS:
            before=read(baseline/'catalog'/name);after=read(stage/'catalog'/name)
            if before!=after:raise ValueError('Staged edits must retain every catalog object')
    rows=plan_intervals(english,original,tracks,edit,bindings)
    kind='edited_staged_native_intervals' if staged_source is not None else 'edited_published_native_intervals'
    if source_idle_policy is not None:
        _staged_policy(read(source_idle_policy),identity,original,bindings,rows,tracks)
    spec={'lesson':identity,'kind':kind,'size':[3840,2160],'fps':30,
          'scenes':[{'visual':a['visual'],'narration':a['narration'],'published_frames':deepcopy(r)}
                    for a,r in zip(english['scenes'],rows)]}
    public={k:v for k,v in source.items() if k!='baseline_root'}
    timing={'schema':1,'kind':'native_visual_timing','lesson':identity,'native_live_video':True,
        'complete_native_clips':True,'fps':30,'playback_speed':1,'looped_or_stretched':False,
        'total_duration':rows[-1]['scene_end'],'scenes':rows,'source_tracks':proofs,
        'english_sha256':hashlib.sha256(json.dumps(english,sort_keys=True,ensure_ascii=False).encode()).hexdigest(),
        ('staged_native_edit' if staged_source is not None else 'published_native_edit'):{**public,'source_master_sha256':bindings['master']['sha256'],
            'source_timing_sha256':bindings['timing']['sha256'],
            'edit_plan_sha256':hashlib.sha256(json.dumps(edit,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
            'fresh_capture':False,'raw_capture_revalidated':False},
        'coverage':{'accepted':True,'gaps':[],'track_count':50,
            'native_scene_count':sum(r['native_live_video'] for r in rows),
            'action_speed':1,'edited_actions_fit_shortest_speech':True}}
    return root,baseline,bindings,source,spec,timing


def encode_scene(source,row,output,*,terminal_seekable=False):
    """Keep each reviewed source interval at 1x; duplicate ONLY its final result."""
    filters=[];labels=[]
    for i,(a,b) in enumerate(row['source_keep_frames']):
        filters.append(f'[0:v]trim=start_frame={a}:end_frame={b},setpts=PTS-STARTPTS[p{i}]')
        labels.append(f'[p{i}]')
    # Every kept frame remains exactly one 30fps frame. Preserve the final
    # kept frame at EOF before appending the reviewed result-frame clones.
    filters.append(''.join(labels)+f'concat=n={len(labels)}:v=1:a=0,settb=1/30,setpts=N,fps=30:eof_action=pass,tpad=stop_mode=clone:stop={row["final_frame_hold_frames"]}[out]')
    frames=row['kept_frames']+row['final_frame_hold_frames']
    seek_options=[]
    if terminal_seekable:
        if frames<2:raise ValueError('Terminal native seek needs two genuine output frames')
        # The player resumes the final two frames after its quiet deadline.
        # An IDR at that exact destination avoids decoding a cold terminal GOP.
        seek_options=['-g','60','-force_key_frames',f'{(frames-2)/30:.12f}','-forced-idr','1']
    subprocess.run(['ffmpeg','-nostdin','-n','-v','error','-threads','2',
        '-ss',f'{row["source_start_frame"]/30:.12f}',
        '-t',f'{row["source_keep_frames"][-1][1]/30:.12f}','-i',str(source),
        '-filter_complex_threads','2','-filter_complex',';'.join(filters),'-map','[out]','-an',
        '-frames:v',str(frames),'-c:v','libx264',
        '-preset','veryfast','-crf','18','-threads','2','-pix_fmt','yuv420p',
        *seek_options,'-video_track_timescale','90000',str(output)],check=True)


def render(stage,identity,baseline,edit,*,staged_source=None,source_idle_policy=None):
    options=dict(staged_source=staged_source,source_idle_policy=source_idle_policy)
    root,baseline,bindings,source,spec,timing=inputs(stage,identity,baseline,edit,**options)
    record_name=STAGED_RECORD if staged_source is not None else RECORD
    target=root/'video'/f'{identity}_silent.mp4';timing_path=target.with_name('native-live-timings.json')
    work=root/'published-edit-render'
    if any(os.path.lexists(p) for p in [target,timing_path,root/RECORD,root/STAGED_RECORD,root/'scenes.json',root/'edited-idle-tail-policy.json',work]):
        raise FileExistsError('Preserve existing edited outputs and partial renders')
    work.mkdir();target.parent.mkdir(parents=True,exist_ok=True)
    source_master=baseline/bindings['master']['path'];parts=[]
    for row in timing['scenes']:
        part=work/f'scene-{row["scene"]:02}.mp4'
        encode_scene(source_master,row,part,terminal_seekable=True)
        parts.append(part)
    # Reuse the normal exact frame-count/rational-duration concatenation.
    sys.path.insert(0,str(REPO/'tools/tutorials/authoring/tools'))
    from render_visual_master import concat
    concat(parts,target,work/'concat.txt')
    # Recheck source/current audio before declaring the normal stage complete.
    _,_,_,_,again,checked=inputs(stage,identity,None if staged_source is not None else baseline,edit,**options)
    if again!=spec or checked!=timing:raise ValueError('Edit inputs changed during rendering')
    write(root/'scenes.json',spec)
    timing['visual_spec_sha256']=digest(root/'scenes.json')
    write(root/record_name,dict(schema=1,kind=spec['kind'],lesson=identity,
        fresh_capture=False,raw_capture_revalidated=False,**source,edit=edit,
        rendered_master_sha256=digest(target)))
    write(timing_path,timing)
    if source_idle_policy is not None:
        original=read(baseline/bindings['timing']['path'])
        policy=edited_policy(root,identity,timing,bindings,original=original,policy=read(source_idle_policy))
        from idle_tail_policy import bind_policy
        entry=policy['lessons'][0]
        records=[dict(path=f'media_host/{identity}/video/{identity}_silent.mp4',sha256=entry['master_sha256']),
                 dict(path=f'media_host/{identity}/video/native-live-timings.json',sha256=entry['visual_timing_sha256'])]
        bind_policy(policy,records,[dict(id=identity,silent=f'{identity}/video/{identity}_silent.mp4',
            visual_timings=f'{identity}/video/native-live-timings.json')],lambda _:timing)
    else:policy=edited_policy(root,identity,timing,bindings)
    write(root/'edited-idle-tail-policy.json',policy)
    return timing_path



def edited_policy(root,identity,timing,bindings,*,original=None,policy=None):
    """Carry only previously bound untouched fences; edited result ends are explicit."""
    from idle_tail_policy import bind_policy,load_policy
    if original is None:original=read(Path(read(root/RECORD)['baseline_root'])/bindings['timing']['path'])
    selected=[e for e in (load_policy(REPO) if policy is None else policy)['lessons'] if e['lesson']==identity]
    inherited={}
    if selected:
        policy_bindings=({k:dict(v,path='media_host/'+v['path'].removeprefix('production/').removeprefix('media_host/')) for k,v in bindings.items()}
                         if policy is not None else bindings)
        source_lesson={'id':identity,'silent':policy_bindings['master']['path'].removeprefix('media_host/'),
                       'visual_timings':policy_bindings['timing']['path'].removeprefix('media_host/')}
        verified=bind_policy({'schema':1,'lessons':selected},list(policy_bindings.values()),
                             [source_lesson],lambda _:original)
        for fence in verified[0]['native_idle_tails']['scenes']:
            matches=[i for i,source in enumerate(original['scenes'])
                     if source['visual']==fence['visual']
                     and abs(source['speech_start']-fence['source_start'])<=1e-7
                     and abs(source['scene_end']-fence['source_end'])<=1e-7]
            if len(matches)!=1:raise ValueError('Inherited fence source scene clock is ambiguous or changed')
            inherited[matches[0]]=fence['idle_tail_start']
    fences=[]
    for index,row in enumerate(timing['scenes']):
        fence=row['kept_frames']/30 if row['edited'] else inherited.get(index)
        if fence is not None:
            fences.append(dict(visual=row['visual'],source_start=row['speech_start'],
                               source_end=row['scene_end'],idle_tail_start=fence))
    return {'schema':1,'lessons':[dict(lesson=identity,
        master_sha256=digest(root/'video'/f'{identity}_silent.mp4'),
        visual_timing_sha256=digest(root/'video/native-live-timings.json'),scenes=fences)]}


def checked_edited_native_timing(stage,identity):
    root=Path(stage)/'production'/identity;record=read(root/RECORD)
    policy=record.get('source_idle_policy',{}).get('path')
    root,_,_,source,spec,timing=inputs(stage,identity,record['baseline_root'],record['edit'],source_idle_policy=policy)
    expected=dict(schema=1,kind='edited_published_native_intervals',lesson=identity,
        fresh_capture=False,raw_capture_revalidated=False,**source,edit=record['edit'],
        rendered_master_sha256=digest(root/'video'/f'{identity}_silent.mp4'))
    if expected!=record or read(root/'scenes.json')!=spec:raise ValueError('Edited published source or rendered output changed')
    timing['visual_spec_sha256']=digest(root/'scenes.json')
    path=root/'video/native-live-timings.json'
    if read(path)!=timing:raise ValueError('Edited frame clocks/current narration changed')
    return path


def checked_staged_native_timing(stage,identity):
    root=Path(stage)/'production'/identity;record=read(root/STAGED_RECORD)
    if any((root/name).exists() for name in [RECORD,'published-native-reuse.json','published-native-mix.json']):
        raise ValueError('Choose edited staged footage only, not a second published source route')
    root,_,_,source,spec,timing=inputs(stage,identity,None,record['edit'],
        staged_source=record['source_stage_root'],source_idle_policy=record['source_idle_policy']['path'])
    expected=dict(schema=1,kind='edited_staged_native_intervals',lesson=identity,
        fresh_capture=False,raw_capture_revalidated=False,**source,edit=record['edit'],
        rendered_master_sha256=digest(root/'video'/f'{identity}_silent.mp4'))
    if expected!=record or read(root/'scenes.json')!=spec:
        raise ValueError('Edited staged source, capture bindings or output changed')
    timing['visual_spec_sha256']=digest(root/'scenes.json')
    path=root/'video/native-live-timings.json'
    if read(path)!=timing:raise ValueError('Edited staged clocks or current narration changed')
    return path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',type=Path,required=True);p.add_argument('--lesson',required=True)
    source=p.add_mutually_exclusive_group(required=True)
    source.add_argument('--baseline',type=Path);source.add_argument('--staged-source',type=Path)
    p.add_argument('--source-idle-policy',type=Path)
    p.add_argument('--reviewed-edit',type=Path,required=True)
    a=p.parse_args();print(render(a.stage,a.lesson,a.baseline,read(a.reviewed_edit),
        staged_source=a.staged_source,source_idle_policy=a.source_idle_policy))
if __name__=='__main__':main()
