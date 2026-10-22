"""Explicit 1x edits of verified published footage, never new capture evidence."""
from copy import deepcopy
import argparse,hashlib,json,math,os,subprocess,sys
from pathlib import Path
from published_native_reuse import _published_inputs,interval_spec
from stage_lesson import REPO,read,write
from check_completed_matrix import digest

RECORD='published-native-edit.json'

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
        frames=source_frames if selected is None else max(kept,math.ceil(max(durations)*30-1e-8))
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


def inputs(stage,identity,baseline,edit):
    from native_live_timing import verified_narration_tracks
    stage=Path(stage).resolve();root=stage/'production'/identity
    baseline,original,bindings,source=_published_inputs(baseline,identity)
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
    rows=plan_intervals(english,original,tracks,edit,bindings)
    spec={'lesson':identity,'kind':'edited_published_native_intervals','size':[3840,2160],'fps':30,
          'scenes':[{'visual':a['visual'],'narration':a['narration'],'published_frames':deepcopy(r)}
                    for a,r in zip(english['scenes'],rows)]}
    public={k:v for k,v in source.items() if k!='baseline_root'}
    timing={'schema':1,'kind':'native_visual_timing','lesson':identity,'native_live_video':True,
        'complete_native_clips':True,'fps':30,'playback_speed':1,'looped_or_stretched':False,
        'total_duration':rows[-1]['scene_end'],'scenes':rows,'source_tracks':proofs,
        'english_sha256':hashlib.sha256(json.dumps(english,sort_keys=True,ensure_ascii=False).encode()).hexdigest(),
        'published_native_edit':{**public,'source_master_sha256':bindings['master']['sha256'],
            'source_timing_sha256':bindings['timing']['sha256'],
            'edit_plan_sha256':hashlib.sha256(json.dumps(edit,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
            'fresh_capture':False,'raw_capture_revalidated':False},
        'coverage':{'accepted':True,'gaps':[],'track_count':50,
            'native_scene_count':sum(r['native_live_video'] for r in rows),
            'action_speed':1,'edited_actions_fit_shortest_speech':True}}
    return root,baseline,bindings,source,spec,timing


def encode_scene(source,row,output):
    """Keep each reviewed source interval at 1x; duplicate ONLY its final result."""
    filters=[];labels=[]
    for i,(a,b) in enumerate(row['source_keep_frames']):
        filters.append(f'[0:v]trim=start_frame={a}:end_frame={b},setpts=PTS-STARTPTS[p{i}]')
        labels.append(f'[p{i}]')
    # Every kept frame remains exactly one 30fps frame. Preserve the final
    # kept frame at EOF before appending the reviewed result-frame clones.
    filters.append(''.join(labels)+f'concat=n={len(labels)}:v=1:a=0,settb=1/30,setpts=N,fps=30:eof_action=pass,tpad=stop_mode=clone:stop={row["final_frame_hold_frames"]}[out]')
    subprocess.run(['ffmpeg','-nostdin','-n','-v','error','-threads','2',
        '-ss',f'{row["source_start_frame"]/30:.12f}',
        '-t',f'{row["source_keep_frames"][-1][1]/30:.12f}','-i',str(source),
        '-filter_complex_threads','2','-filter_complex',';'.join(filters),'-map','[out]','-an',
        '-frames:v',str(row['kept_frames']+row['final_frame_hold_frames']),'-c:v','libx264',
        '-preset','veryfast','-crf','18','-threads','2','-pix_fmt','yuv420p',
        '-video_track_timescale','90000',str(output)],check=True)


def render(stage,identity,baseline,edit):
    root,baseline,bindings,source,spec,timing=inputs(stage,identity,baseline,edit)
    target=root/'video'/f'{identity}_silent.mp4';timing_path=target.with_name('native-live-timings.json')
    work=root/'published-edit-render'
    if any(os.path.lexists(p) for p in [target,timing_path,root/RECORD,root/'scenes.json',root/'edited-idle-tail-policy.json',work]):
        raise FileExistsError('Preserve existing edited outputs and partial renders')
    work.mkdir();target.parent.mkdir(parents=True,exist_ok=True)
    source_master=baseline/bindings['master']['path'];parts=[]
    for row in timing['scenes']:
        part=work/f'scene-{row["scene"]:02}.mp4';encode_scene(source_master,row,part);parts.append(part)
    # Reuse the normal exact frame-count/rational-duration concatenation.
    sys.path.insert(0,str(REPO/'tools/tutorials/authoring/tools'))
    from render_visual_master import concat
    concat(parts,target,work/'concat.txt')
    # Recheck source/current audio before declaring the normal stage complete.
    _,_,_,_,again,checked=inputs(stage,identity,baseline,edit)
    if again!=spec or checked!=timing:raise ValueError('Edit inputs changed during rendering')
    write(root/'scenes.json',spec)
    timing['visual_spec_sha256']=digest(root/'scenes.json')
    write(root/RECORD,dict(schema=1,kind='edited_published_native_intervals',lesson=identity,
        fresh_capture=False,raw_capture_revalidated=False,**source,edit=edit,
        rendered_master_sha256=digest(target)))
    write(timing_path,timing)
    write(root/'edited-idle-tail-policy.json',edited_policy(root,identity,timing,bindings))
    return timing_path



def edited_policy(root,identity,timing,bindings):
    """Carry only previously bound untouched fences; edited result ends are explicit."""
    from idle_tail_policy import bind_policy,load_policy
    original=read(Path(read(root/RECORD)['baseline_root'])/bindings['timing']['path'])
    selected=[e for e in load_policy(REPO)['lessons'] if e['lesson']==identity]
    inherited={}
    if selected:
        source_lesson={'id':identity,'silent':bindings['master']['path'].removeprefix('media_host/'),
                       'visual_timings':bindings['timing']['path'].removeprefix('media_host/')}
        verified=bind_policy({'schema':1,'lessons':selected},list(bindings.values()),
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
    root,_,_,source,spec,timing=inputs(stage,identity,record['baseline_root'],record['edit'])
    expected=dict(schema=1,kind='edited_published_native_intervals',lesson=identity,
        fresh_capture=False,raw_capture_revalidated=False,**source,edit=record['edit'],
        rendered_master_sha256=digest(root/'video'/f'{identity}_silent.mp4'))
    if expected!=record or read(root/'scenes.json')!=spec:raise ValueError('Edited published source or rendered output changed')
    timing['visual_spec_sha256']=digest(root/'scenes.json')
    path=root/'video/native-live-timings.json'
    if read(path)!=timing:raise ValueError('Edited frame clocks/current narration changed')
    return path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',type=Path,required=True);p.add_argument('--lesson',required=True)
    p.add_argument('--baseline',type=Path,required=True);p.add_argument('--reviewed-edit',type=Path,required=True)
    a=p.parse_args();print(render(a.stage,a.lesson,a.baseline,read(a.reviewed_edit)))
if __name__=='__main__':main()
