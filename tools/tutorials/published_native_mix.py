"""Compose verified published intervals with exact Home or Image UMAP reference cards."""
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

UMAP_SOURCE = ('01_module', '19_setting_src', '19_setting_tables', '19_setting_row_limit',
    '19_setting_reduction_method', '19_setting_n_neighbors', '19_setting_random_seed',
    '19_setting_clustering', '19_setting_plot_images', '23_batch_finished', '24_batch_figure',
    '25_interactive_embedding', '26_point_0', '26_point_200')
UMAP_CARD = '27_fold_image_scatter'


def source_mapping(english, original, cards, reference_revision=None):
    """Only explicitly reviewed native/reference-card arrangements."""
    if reference_revision is not None:
        from render_platform_maintenance_references import LESSON,REVISION,CARDS,checked_lesson,checked_source
        if english['id']!=LESSON or reference_revision!=REVISION:
            raise ValueError('Maintenance revision is explicit and reviewed only for Platform Installers')
        checked_lesson(english);checked_source(original)
        if set(cards)!=set(CARDS) or any(c.get('reference_revision')!=REVISION for c in cards.values()):
            raise ValueError('Require exactly ten reviewed maintenance reference cards')
        return [s['visual'] for s in original['scenes'] if s['visual'] not in CARDS],set(CARDS)
    visuals = [s['visual'] for s in english['scenes']]
    source = [s['visual'] for s in original['scenes']]
    if english['id'] == '05_home':
        if (len(visuals)!=39 or len(set(visuals))!=39 or len(source)!=33
                or len(set(source))!=33 or set(cards)!=CARDS or CARDS & set(source)
                or [v for v in visuals if v not in CARDS]!=source):
            raise ValueError('Require exact current39 order, all original33 in order and only six authored cards')
        return source, CARDS
    if english['id'] == '15_image_umap':
        if (source != [*UMAP_SOURCE, '27_review_embedding']
                or visuals != [*UMAP_SOURCE, UMAP_CARD] or set(cards) != {UMAP_CARD}
                or any(s.get('native_live_video') is not True for s in original['scenes'])):
            raise ValueError('Require exact Image UMAP fourteen native scenes and one linked-tool reference card')
        return list(UMAP_SOURCE), {UMAP_CARD}
    if english['id'] == '21_model_compare':
        from render_model_compare_reference_cards import VISUALS, CARDS as COMPARE_CARDS
        if (source != list(VISUALS) or visuals != list(VISUALS) or set(cards) != set(COMPARE_CARDS)
                or any(type(s.get('native_live_video')) is not bool for s in original['scenes'])
                or [s['native_live_video'] for s in original['scenes']] != [True,True,False,False,False,False]):
            raise ValueError('Require exact six Model Compare scenes, two native/four held, and only three reviewed reference slots')
        return [v for v in source if v not in COMPARE_CARDS], set(COMPARE_CARDS)
    if english['id'] == '04_platform_installers':
        from render_platform_doctor_reference_card import VISUALS, CARD
        flags=[s.get('native_live_video') for s in original['scenes']]
        if (source!=list(VISUALS) or visuals!=list(VISUALS) or set(cards)!={CARD}
                or any(type(flag) is not bool for flag in flags)
                or flags!=[False]*7+[True,True]+[False]*5
                or round(original['scenes'][9]['duration']*30)!=879):
            raise ValueError('Require exact fourteen Platform scenes, two native/twelve held and only the original879-frame Doctor reference')
        return [v for v in source if v!=CARD], {CARD}
    if english['id'] == '02_install_spacr':
        from render_installation_reference_cards import checked_lesson, checked_source, CARDS
        checked_lesson(english);checked_source(original)
        if set(cards)!=set(CARDS):raise ValueError('Require only the three reviewed Installation card slots')
        return [v for v in source if v not in CARDS], set(CARDS)
    raise ValueError('Mixed published source is only reviewed for Home, Image UMAP, Model Compare and Platform Installers')


def identity(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False).encode()).hexdigest()


def plan_mix(english,original,tracks,cards,edit,bindings,fences,reference_revision=None):
    source, card_visuals = source_mapping(english, original, cards,reference_revision)
    count = len(english['scenes'])
    if (set(edit)!={'schema','lesson','master_sha256','visual_timing_sha256','scenes'}
            or edit['schema']!=1 or edit['lesson']!=english['id']
            or edit['master_sha256']!=bindings['master']['sha256']
            or edit['visual_timing_sha256']!=bindings['timing']['sha256']):
        raise ValueError('Native edit source binding changed')
    selected={s['visual']:s for s in edit['scenes']}
    if english['id'] in {'21_model_compare','04_platform_installers','02_install_spacr'} and selected:
        raise ValueError('Reference refresh preserves all retained source intervals unchanged')
    if len(selected)!=len(edit['scenes']) or set(selected)-set(source):raise ValueError('Unknown or duplicate native edits')
    if len(tracks)!=50 or any(len(t['scenes'])!=count for t in tracks):raise ValueError('Require all current fifty complete scene tracks')
    by_source={s['visual']:s for s in original['scenes']};rows=[];cursor=0
    if english['id']=='04_platform_installers':
        if reference_revision is None:
            from render_platform_doctor_reference_card import MASTER_SHA
        else:
            from render_platform_maintenance_references import MASTER_SHA,TIMING_SHA,VISUALS
            if bindings['timing']['sha256']!=TIMING_SHA or fences!={v:(11.3 if i==8 else 1.0 if i==9 else 0) for i,v in enumerate(VISUALS,1)}:
                raise ValueError('Maintenance source timing or all fourteen reviewed fences changed')
        if bindings['master']['sha256']!=MASTER_SHA:raise ValueError('Accepted Platform master changed')
    if english['id']=='02_install_spacr':
        from render_installation_reference_cards import MASTER_SHA, TIMING_SHA
        if bindings['master']['sha256']!=MASTER_SHA or bindings['timing']['sha256']!=TIMING_SHA:
            raise ValueError('Accepted Installation master or clock changed')
        if set(fences)!=set(by_source) or any(fences[v]!=0 for v in card_visuals):
            raise ValueError('Require all twenty-one existing Installation fences and zero card fences')
    for index,authored in enumerate(english['scenes']):
        visual=authored['visual'];spoken=[t['scenes'][index] for t in tracks]
        spans=[float(s['speech_end'])-float(s['speech_start']) for s in spoken]
        durations=[float(s['duration']) for s in spoken]
        if any(not math.isfinite(x) or x<=0 for x in spans+durations) or any(a>b for a,b in zip(spans,durations)):
            raise ValueError('Invalid current narration intervals')
        if visual in card_visuals:
            card=cards[visual]
            if card.get('kind')!='authored_reference_card' or card.get('narration_sha256')!=identity(authored['narration']):
                raise ValueError('Inserted visual must be a current authored reference card')
            frames=math.ceil(max(durations)*30-1e-8)
            if english['id'] in {'04_platform_installers','02_install_spacr'}:
                frames=round(by_source[visual]['duration']*30)
                if max(durations)>frames/30+1e-9:raise ValueError('Retained Doctor speech exceeds its original interval')
            row=dict(native_live_video=False,reference_card=deepcopy(card),edited=False,
                     kept_frames=0,final_frame_hold_frames=frames)
        elif visual in selected:
            expected=next(i for i,s in enumerate(original['scenes'],1) if s['visual']==visual)
            source_row=by_source[visual]
            if 'scene' in source_row and (type(source_row['scene']) is not int or source_row['scene']!=expected):
                raise ValueError('Native original source index differs from published visual')
            one=deepcopy(original);one['scenes']=[deepcopy(source_row)]
            if 'scene' in one['scenes'][0]:one['scenes'][0]['scene']=1
            current={'id':english['id'],'scenes':[authored]}
            subset=[dict(scenes=[s]) for s in spoken]
            chosen=deepcopy(selected[visual])
            if 'source_scene' in chosen:
                if type(chosen['source_scene']) is not int or chosen['source_scene']!=expected:raise ValueError('Native edit source index differs from published visual')
                chosen['source_scene']=1
            selection={**edit,'scenes':[chosen]}
            row=plan_intervals(current,one,subset,selection,bindings)[0]
            frames=round(row['duration']*30)
        else:
            source_row=by_source[visual];kept=round(source_row['duration']*30)
            frames=max(kept,math.ceil(max(durations)*30-1e-8))
            if english['id'] in {'04_platform_installers','02_install_spacr'} and frames!=kept:
                raise ValueError('Retained Platform speech exceeds its unchanged source clock')
            if frames>kept and (visual not in fences or fences[visual]>min(spans)):
                raise ValueError('Longer narration requires a reviewed early idle/result fence before final-frame hold: '+visual)
            row=dict(native_live_video=source_row.get('native_live_video') is True,
                source_start_frame=round(source_row['speech_start']*30),source_keep_frames=[[0,kept]],
                confirmed_idle_frames=[],kept_frames=kept,final_frame_hold_frames=frames-kept,edited=False)
        row.update(scene=index+1,visual=visual,text=authored['narration'],speech_start=cursor/30,
                   speech_end=(cursor+frames)/30,scene_end=(cursor+frames)/30,duration=frames/30,
                   longest_narration_duration=max(durations))
        if english['id'] in {'04_platform_installers','02_install_spacr'}:
            row.update({key:deepcopy(by_source[visual][key]) for key in ('speech_start','speech_end','scene_end','duration')})
        rows.append(row);cursor+=frames
    return rows


def inputs(stage,baseline,edit,source_policy,identity_='05_home',reference_revision=None):
    from native_live_timing import verified_narration_tracks
    from idle_tail_policy import bind_policy
    stage=Path(stage).resolve();root=stage/'production'/identity_
    baseline,original,bindings,source=_published_inputs(baseline,identity_)
    if (stage==baseline or stage.is_relative_to(baseline) or baseline.is_relative_to(stage)
            or not root.resolve().is_relative_to(stage)
            or not (root/'video').resolve().is_relative_to(root.resolve())
            or (root/'reference-cards').is_symlink()
            or any((root/name).exists() for name in ['published-native-reuse.json','published-native-edit.json'])):
        raise ValueError('Mixed publication needs its independent exclusive stage')
    english=read(root/'lesson.en.json')
    if english!=read(REPO/'tools/tutorials/lessons'/(identity_+'.json')):raise ValueError('Canonical mixed lesson source changed')
    sys.path.insert(0,str(REPO/'tools/tutorials/authoring/tools'))
    if identity_ == '05_home':
        from render_home_reference_cards import validate_cards
    elif identity_ == '15_image_umap':
        from render_umap_reference_card import validate_cards
    elif identity_ == '21_model_compare':
        from render_model_compare_reference_cards import validate_cards
    elif identity_ == '04_platform_installers':
        if reference_revision is None:
            from render_platform_doctor_reference_card import validate_cards
        elif reference_revision=='maintenance-v1':
            from render_platform_maintenance_references import validate_cards
        else:raise ValueError('Unknown Platform reference revision')
    elif identity_ == '02_install_spacr':
        from render_installation_reference_cards import validate_cards
    else:
        raise ValueError('Unreviewed mixed lesson')
    if reference_revision is not None and identity_!='04_platform_installers':raise ValueError('Maintenance revision is Platform-only')
    cards=validate_cards(root/'reference-cards',english,REPO)
    card_contract=read(root/'reference-cards/cards.json')
    if identity_ in {'21_model_compare','04_platform_installers','02_install_spacr'}:
        from published_retained_narration import RECORD as NARRATION_RECORD, checked_published_narration
        if Path(read(root/NARRATION_RECORD)['baseline_root']).resolve()!=baseline:
            raise ValueError('Reference refresh requires unchanged fifty-track narration from this exact published baseline')
        tracks,proofs=checked_published_narration(stage,identity_)
    else:
        tracks,proofs=verified_narration_tracks(stage,identity_,english)
    lesson={'id':identity_,'silent':bindings['master']['path'].removeprefix('media_host/'),
            'visual_timings':bindings['timing']['path'].removeprefix('media_host/')}
    verified=bind_policy(source_policy,list(bindings.values()),[lesson],lambda _:original)[0]
    fences={s['visual']:s['idle_tail_start'] for s in verified.get('native_idle_tails',{}).get('scenes',[])}
    if edit is None:edit=dict(schema=1,lesson=identity_,master_sha256=bindings['master']['sha256'],
                             visual_timing_sha256=bindings['timing']['sha256'],scenes=[])
    rows=plan_mix(english,original,tracks,cards,edit,bindings,fences,reference_revision)
    spec=dict(lesson=identity_,kind='mixed_published_native_intervals',size=[3840,2160],fps=30,
        scenes=[dict(visual=a['visual'],narration=a['narration'],published_source=deepcopy(r))
                for a,r in zip(english['scenes'],rows)])
    timing=dict(schema=1,kind='native_visual_timing',lesson=identity_,native_live_video=True,
        complete_native_clips=True,fps=30,playback_speed=1,looped_or_stretched=False,
        total_duration=rows[-1]['scene_end'],scenes=rows,source_tracks=proofs,english_sha256=identity(english),
        published_native_mix=dict(**{k:v for k,v in source.items() if k!='baseline_root'},
          source_master_sha256=bindings['master']['sha256'],source_timing_sha256=bindings['timing']['sha256'],
          edit_sha256=identity(edit),cards_contract_sha256=identity(card_contract),
          source_idle_policy_sha256=identity(source_policy),fresh_capture=False,raw_capture_revalidated=False),
        coverage=dict(accepted=True,gaps=[],track_count=50,source_native_scene_count=len(original['scenes'])-int(identity_=='15_image_umap'),reference_card_count=len(cards)))
    if identity_ in {'21_model_compare','04_platform_installers','02_install_spacr'}:
        timing['coverage']['source_native_scene_count']=1 if identity_=='02_install_spacr' else 2
        timing['published_native_mix']['replaced_source_scenes']=[
            dict(source_scene=i+1,visual=s['visual'],reason=('Installation reference wording clarified; original commands and clock unchanged' if identity_=='02_install_spacr' else 'Historical CPU diagnostic reference wording clarified; command and results unchanged' if identity_=='04_platform_installers' else 'Authoring terminal replaced by clearly labelled reference from unchanged saved results'))
            for i,s in enumerate(original['scenes']) if s['visual'] in cards]
    if reference_revision is not None:
        timing['published_native_mix']['reference_revision']=reference_revision
        for replaced in timing['published_native_mix']['replaced_source_scenes']:
            replaced['reason']='Authored installation/maintenance reference; retained original commands/results and unchanged source clock'
    if identity_ == '15_image_umap':
        timing['published_native_mix']['omitted_source_scenes'] = [dict(source_scene=15,visual='27_review_embedding',
            reason='Redundant completed embedding view; native scenes11–14 retain real completed result and crop inspection. Replaced by authored linked-tool reference.') ]
    return root,baseline,bindings,source,spec,timing,edit,fences


def product_policy(root,timing,fences):
    identity_ = timing['lesson']
    rows=[]
    for row in timing['scenes']:
        fence=0 if 'reference_card' in row else row['kept_frames']/30 if row['edited'] else fences.get(row['visual'])
        if fence is not None:
            rows.append(dict(visual=row['visual'],source_start=row['speech_start'],source_end=row['scene_end'],idle_tail_start=fence))
    return dict(schema=1,lessons=[dict(lesson=identity_,master_sha256=digest(root/'video'/(identity_+'_silent.mp4')),
        visual_timing_sha256=digest(root/'video/native-live-timings.json'),scenes=rows)])


def render(stage,baseline,edit,source_policy,identity_='05_home',reference_revision=None,native_companions_plan=None):
    if reference_revision=='native-companions-v1':
        if identity_!='04_platform_installers' or edit is not None or native_companions_plan is None:
            raise ValueError('Native companion refresh requires only its explicit Platform plan')
        from published_platform_native_refresh import render as render_companions
        return render_companions(stage,baseline,native_companions_plan,source_policy)
    if native_companions_plan is not None:raise ValueError('Companion plan requires its explicit Platform revision')
    root,baseline,bindings,source,spec,timing,edit,fences=inputs(stage,baseline,edit,source_policy,identity_,reference_revision)
    target=root/'video'/(identity_+'_silent.mp4');work=root/'mixed-render';clock=target.with_name('native-live-timings.json')
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
    checked=inputs(stage,baseline,edit,source_policy,identity_,reference_revision)
    if checked[4:6]!=(spec,timing):raise ValueError('Current cards/native/audio inputs changed during rendering')
    write(root/'scenes.json',spec);timing['visual_spec_sha256']=digest(root/'scenes.json')
    record=dict(schema=1,kind='mixed_published_native_intervals',lesson=identity_,fresh_capture=False,raw_capture_revalidated=False,**source,
        edit=edit,source_idle_policy=source_policy,rendered_master_sha256=digest(target))
    if reference_revision is not None:record['reference_revision']=reference_revision
    write(root/RECORD,record)
    write(clock,timing);write(root/'mixed-idle-tail-policy.json',product_policy(root,timing,fences))
    return clock


def checked_mixed_native_timing(stage,identity_):
    record_path=Path(stage)/'production'/identity_/RECORD
    if record_path.is_file() and read(record_path).get('reference_revision')=='native-companions-v1':
        if identity_!='04_platform_installers':raise ValueError('Native companions are Platform-only')
        from published_platform_native_refresh import checked
        return checked(stage)
    if identity_ not in {'05_home','15_image_umap','21_model_compare','04_platform_installers','02_install_spacr'}:raise ValueError('Unreviewed mixed source')
    root=Path(stage)/'production'/identity_;record=read(root/RECORD)
    if 'reference_revision' in record and record['reference_revision']!='maintenance-v1':raise ValueError('Unknown explicit reference revision')
    root,_,_,source,spec,timing,_,_=inputs(stage,record['baseline_root'],record['edit'],record['source_idle_policy'],identity_,record.get('reference_revision'))
    expected=dict(schema=1,kind='mixed_published_native_intervals',lesson=identity_,fresh_capture=False,raw_capture_revalidated=False,**source,edit=record['edit'],
        source_idle_policy=record['source_idle_policy'],rendered_master_sha256=digest(root/'video'/(identity_+'_silent.mp4')))
    if 'reference_revision' in record:expected['reference_revision']=record['reference_revision']
    if record!=expected or read(root/'scenes.json')!=spec:raise ValueError('Mixed source/current frames changed')
    timing['visual_spec_sha256']=digest(root/'scenes.json');clock=root/'video/native-live-timings.json'
    if read(clock)!=timing:raise ValueError('Mixed source clock/current voice bindings changed')
    return clock


def canonical_source_policy(identity_):
    """Select one existing reviewed binding; inputs() still verifies its source."""
    if identity_ not in {'04_platform_installers','21_model_compare','02_install_spacr'}:
        raise ValueError('Canonical source policy is reviewed only for Platform and Model Compare')
    policy=read(REPO/'tools/tutorials/idle_tail_policy.json')
    if not isinstance(policy,dict) or set(policy)!={'schema','lessons'} or policy['schema']!=1 or not isinstance(policy['lessons'],list):
        raise ValueError('Require canonical schema1 idle policy')
    entries=[row for row in policy['lessons'] if isinstance(row,dict) and row.get('lesson')==identity_]
    if len(entries)!=1:raise ValueError('Require exactly one existing source policy')
    return dict(schema=1,lessons=deepcopy(entries))


def platform_source_policy():
    """Preserve the established Platform-only default."""
    return canonical_source_policy('04_platform_installers')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',type=Path,required=True)
    p.add_argument('--baseline',type=Path,required=True);p.add_argument('--reviewed-edit',type=Path)
    source=p.add_mutually_exclusive_group()
    source.add_argument('--source-idle-policy',type=Path)
    source.add_argument('--canonical-source-policy',action='store_true',help='Explicitly bind the one existing canonical04/21/02 source entry')
    p.add_argument('--lesson',choices=['05_home','15_image_umap','21_model_compare','04_platform_installers','02_install_spacr'],default='05_home')
    p.add_argument('--reference-revision',choices=['maintenance-v1','native-companions-v1'],help='Explicit Platform reference or two-native-companion contract; default Doctor-only unchanged')
    p.add_argument('--native-companions-plan',type=Path,help='Exact reviewed two-clip source bindings and action fences')
    a=p.parse_args();print(render(a.stage,a.baseline,read(a.reviewed_edit) if a.reviewed_edit else None,
        read(a.source_idle_policy) if a.source_idle_policy else canonical_source_policy(a.lesson) if a.canonical_source_policy else platform_source_policy() if a.lesson=='04_platform_installers' else dict(schema=1,lessons=[]),a.lesson,a.reference_revision,read(a.native_companions_plan) if a.native_companions_plan else None))
if __name__=='__main__':main()
