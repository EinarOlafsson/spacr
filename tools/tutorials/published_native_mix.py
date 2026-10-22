"""Compose verified published intervals with six explicit Home reference cards."""
from copy import deepcopy
import argparse,hashlib,json,math,os,sys
from pathlib import Path
from stage_lesson import REPO,read,write
from check_completed_matrix import digest
from published_native_reuse import _published_inputs
from published_native_edit import plan_intervals,encode_scene

RECORD='published-native-mix.json'
CARDS={'home_module_embeddings','home_module_experiment_design','home_module_dose_response',
       'home_module_qc_dashboard','home_module_gate_editor','24_spaceout_field'}

def identity(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False).encode()).hexdigest()


def plan_mix(english,original,tracks,cards,edit,bindings,fences):
    visuals=[s['visual'] for s in english['scenes']];source=[s['visual'] for s in original['scenes']]
    if (english['id']!='05_home' or len(visuals)!=39 or len(set(visuals))!=39 or len(source)!=33
            or len(set(source))!=33 or set(cards)!=CARDS or CARDS & set(source)
            or [v for v in visuals if v not in CARDS]!=source):
        raise ValueError('Require exact current39 order, all original33 in order and only six authored cards')
    if (set(edit)!={'schema','lesson','master_sha256','visual_timing_sha256','scenes'}
            or edit['schema']!=1 or edit['lesson']!=english['id']
            or edit['master_sha256']!=bindings['master']['sha256']
            or edit['visual_timing_sha256']!=bindings['timing']['sha256']):
        raise ValueError('Native edit source binding changed')
    selected={s['visual']:s for s in edit['scenes']}
    if len(selected)!=len(edit['scenes']) or set(selected)-set(source):raise ValueError('Unknown or duplicate native edits')
    if len(tracks)!=50 or any(len(t['scenes'])!=39 for t in tracks):raise ValueError('Require all current fifty39-scene tracks')
    by_source={s['visual']:s for s in original['scenes']};rows=[];cursor=0
    for index,authored in enumerate(english['scenes']):
        visual=authored['visual'];spoken=[t['scenes'][index] for t in tracks]
        spans=[float(s['speech_end'])-float(s['speech_start']) for s in spoken]
        durations=[float(s['duration']) for s in spoken]
        if any(not math.isfinite(x) or x<=0 for x in spans+durations) or any(a>b for a,b in zip(spans,durations)):
            raise ValueError('Invalid current narration intervals')
        if visual in CARDS:
            card=cards[visual]
            if card.get('kind')!='authored_reference_card' or card.get('narration_sha256')!=identity(authored['narration']):
                raise ValueError('Inserted visual must be a current authored reference card')
            frames=math.ceil(max(durations)*30-1e-8)
            row=dict(native_live_video=False,reference_card=deepcopy(card),edited=False,
                     kept_frames=0,final_frame_hold_frames=frames)
        elif visual in selected:
            one=deepcopy(original);one['scenes']=[by_source[visual]]
            current={'id':english['id'],'scenes':[authored]}
            subset=[dict(scenes=[s]) for s in spoken]
            selection={**edit,'scenes':[selected[visual]]}
            row=plan_intervals(current,one,subset,selection,bindings)[0]
            frames=round(row['duration']*30)
        else:
            source_row=by_source[visual];kept=round(source_row['duration']*30)
            frames=max(kept,math.ceil(max(durations)*30-1e-8))
            if frames>kept and (visual not in fences or fences[visual]>min(spans)):
                raise ValueError('Longer narration requires a reviewed early idle/result fence before final-frame hold: '+visual)
            row=dict(native_live_video=source_row.get('native_live_video') is True,
                source_start_frame=round(source_row['speech_start']*30),source_keep_frames=[[0,kept]],
                confirmed_idle_frames=[],kept_frames=kept,final_frame_hold_frames=frames-kept,edited=False)
        row.update(scene=index+1,visual=visual,text=authored['narration'],speech_start=cursor/30,
                   speech_end=(cursor+frames)/30,scene_end=(cursor+frames)/30,duration=frames/30,
                   longest_narration_duration=max(durations))
        rows.append(row);cursor+=frames
    return rows


def inputs(stage,baseline,edit,source_policy):
    from native_live_timing import verified_narration_tracks
    from idle_tail_policy import bind_policy
    stage=Path(stage).resolve();root=stage/'production/05_home'
    baseline,original,bindings,source=_published_inputs(baseline,'05_home')
    if (stage==baseline or stage.is_relative_to(baseline) or baseline.is_relative_to(stage)
            or not root.resolve().is_relative_to(stage)
            or not (root/'video').resolve().is_relative_to(root.resolve())
            or (root/'reference-cards').is_symlink()
            or any((root/name).exists() for name in ['published-native-reuse.json','published-native-edit.json'])):
        raise ValueError('Mixed publication needs its independent exclusive stage')
    english=read(root/'lesson.en.json')
    if english!=read(REPO/'tools/tutorials/lessons/05_home.json'):raise ValueError('Canonical Home source changed')
    sys.path.insert(0,str(REPO/'tools/tutorials/authoring/tools'))
    from render_home_reference_cards import validate_cards
    cards=validate_cards(root/'reference-cards',english,REPO)
    card_contract=read(root/'reference-cards/cards.json')
    tracks,proofs=verified_narration_tracks(stage,'05_home',english)
    lesson={'id':'05_home','silent':bindings['master']['path'].removeprefix('media_host/'),
            'visual_timings':bindings['timing']['path'].removeprefix('media_host/')}
    verified=bind_policy(source_policy,list(bindings.values()),[lesson],lambda _:original)[0]
    fences={s['visual']:s['idle_tail_start'] for s in verified.get('native_idle_tails',{}).get('scenes',[])}
    if edit is None:edit=dict(schema=1,lesson='05_home',master_sha256=bindings['master']['sha256'],
                             visual_timing_sha256=bindings['timing']['sha256'],scenes=[])
    rows=plan_mix(english,original,tracks,cards,edit,bindings,fences)
    spec=dict(lesson='05_home',kind='mixed_published_native_intervals',size=[3840,2160],fps=30,
        scenes=[dict(visual=a['visual'],narration=a['narration'],published_source=deepcopy(r))
                for a,r in zip(english['scenes'],rows)])
    timing=dict(schema=1,kind='native_visual_timing',lesson='05_home',native_live_video=True,
        complete_native_clips=True,fps=30,playback_speed=1,looped_or_stretched=False,
        total_duration=rows[-1]['scene_end'],scenes=rows,source_tracks=proofs,english_sha256=identity(english),
        published_native_mix=dict(**{k:v for k,v in source.items() if k!='baseline_root'},
          source_master_sha256=bindings['master']['sha256'],source_timing_sha256=bindings['timing']['sha256'],
          edit_sha256=identity(edit),cards_contract_sha256=identity(card_contract),
          source_idle_policy_sha256=identity(source_policy),fresh_capture=False,raw_capture_revalidated=False),
        coverage=dict(accepted=True,gaps=[],track_count=50,source_native_scene_count=33,reference_card_count=6))
    return root,baseline,bindings,source,spec,timing,edit,fences


def product_policy(root,timing,fences):
    rows=[]
    for row in timing['scenes']:
        fence=0 if 'reference_card' in row else row['kept_frames']/30 if row['edited'] else fences.get(row['visual'])
        if fence is not None:
            rows.append(dict(visual=row['visual'],source_start=row['speech_start'],source_end=row['scene_end'],idle_tail_start=fence))
    return dict(schema=1,lessons=[dict(lesson='05_home',master_sha256=digest(root/'video/05_home_silent.mp4'),
        visual_timing_sha256=digest(root/'video/native-live-timings.json'),scenes=rows)])


def render(stage,baseline,edit,source_policy):
    root,baseline,bindings,source,spec,timing,edit,fences=inputs(stage,baseline,edit,source_policy)
    target=root/'video/05_home_silent.mp4';work=root/'mixed-render';clock=target.with_name('native-live-timings.json')
    if any(os.path.lexists(p) for p in [target,clock,root/'scenes.json',root/RECORD,root/'mixed-idle-tail-policy.json',work]):
        raise FileExistsError('Preserve existing mixed outputs and incomplete render parts')
    work.mkdir();target.parent.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(REPO/'tools/tutorials/authoring/tools'))
    from render_visual_master import concat,encode_still
    parts=[]
    for row in timing['scenes']:
        part=work/f'scene-{row["scene"]:02}.mp4'
        if 'reference_card' in row:
            encode_still(root/'reference-cards'/row['reference_card']['image'],row['duration'],part,30)
        else:encode_scene(baseline/bindings['master']['path'],row,part)
        parts.append(part)
    concat(parts,target,work/'concat.txt')
    checked=inputs(stage,baseline,edit,source_policy)
    if checked[4:6]!=(spec,timing):raise ValueError('Current cards/native/audio inputs changed during rendering')
    write(root/'scenes.json',spec);timing['visual_spec_sha256']=digest(root/'scenes.json')
    write(root/RECORD,dict(schema=1,kind='mixed_published_native_intervals',lesson='05_home',fresh_capture=False,raw_capture_revalidated=False,**source,
        edit=edit,source_idle_policy=source_policy,rendered_master_sha256=digest(target)))
    write(clock,timing);write(root/'mixed-idle-tail-policy.json',product_policy(root,timing,fences))
    return clock


def checked_mixed_native_timing(stage,identity_):
    if identity_!='05_home':raise ValueError('This reviewed mixed source is Home only')
    root=Path(stage)/'production/05_home';record=read(root/RECORD)
    root,_,_,source,spec,timing,_,_=inputs(stage,record['baseline_root'],record['edit'],record['source_idle_policy'])
    expected=dict(schema=1,kind='mixed_published_native_intervals',lesson='05_home',fresh_capture=False,raw_capture_revalidated=False,**source,edit=record['edit'],
        source_idle_policy=record['source_idle_policy'],rendered_master_sha256=digest(root/'video/05_home_silent.mp4'))
    if record!=expected or read(root/'scenes.json')!=spec:raise ValueError('Mixed source/current frames changed')
    timing['visual_spec_sha256']=digest(root/'scenes.json');clock=root/'video/native-live-timings.json'
    if read(clock)!=timing:raise ValueError('Mixed source clock/current voice bindings changed')
    return clock


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',type=Path,required=True)
    p.add_argument('--baseline',type=Path,required=True);p.add_argument('--reviewed-edit',type=Path)
    p.add_argument('--source-idle-policy',type=Path)
    a=p.parse_args();print(render(a.stage,a.baseline,read(a.reviewed_edit) if a.reviewed_edit else None,
        read(a.source_idle_policy) if a.source_idle_policy else dict(schema=1,lessons=[])))
if __name__=='__main__':main()
