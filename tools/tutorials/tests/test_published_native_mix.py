"""Mixed-source planning and genuine normal generated-card validation, no capture claims."""
from copy import deepcopy
from pathlib import Path
import importlib.util,json,sys
import pytest
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'authoring/tools')]
import published_native_mix as mix

@pytest.fixture
def source_plan():
    original={'scenes':[dict(visual=f'native_{i}',speech_start=i*3,scene_end=(i+1)*3,duration=3,native_live_video=True) for i in range(33)]}
    scenes=[dict(visual=s['visual'],narration='Existing feature '+s['visual']) for s in original['scenes']]
    cards={}
    for i,v in enumerate(sorted(mix.CARDS)):
        authored=dict(visual=v,narration='Authored reference '+v);scenes.insert(3+i*4,authored)
        cards[v]=dict(visual=v,kind='authored_reference_card',narration_sha256=mix.identity(authored['narration']))
    english={'id':'05_home','scenes':scenes}
    tracks=[dict(scenes=[dict(speech_start=i*2,speech_end=i*2+1.48,duration=2) for i in range(39)]) for _ in range(50)]
    bindings={'master':{'sha256':'a'*64},'timing':{'sha256':'b'*64}}
    edit=dict(schema=1,lesson='05_home',master_sha256='a'*64,visual_timing_sha256='b'*64,scenes=[])
    return english,original,tracks,cards,edit,bindings,{}

def test_exact_six_reference_cards_insert_without_reordering_or_truncating_native_source(source_plan):
    rows=mix.plan_mix(*source_plan)
    assert len(rows)==39 and sum('reference_card' in r for r in rows)==6
    native=[r for r in rows if 'reference_card' not in r]
    assert [r['visual'] for r in native]==[f'native_{i}' for i in range(33)]
    assert [r['source_start_frame'] for r in native]==[i*90 for i in range(33)]
    assert all(r['source_keep_frames']==[[0,90]] and r['final_frame_hold_frames']==0 for r in native)
    assert all(not r['native_live_video'] and r['duration']==2 for r in rows if 'reference_card' in r)
    assert rows[-1]['scene_end']==111 and all(a['scene_end']==b['speech_start'] for a,b in zip(rows,rows[1:]))

@pytest.mark.parametrize('fault',['unknown_added','missing_native','duplicate','reorder','noncard',
    'card_text','source_hash','missing_voice','native_edit_unknown'])
def test_mix_rejects_missing_or_changed_source_mapping(source_plan,fault):
    english,original,tracks,cards,edit,bindings,fences=deepcopy(source_plan)
    if fault=='unknown_added':english['scenes'][3]['visual']='unknown_card'
    elif fault=='missing_native':english['scenes'].pop(0)
    elif fault=='duplicate':english['scenes'][1]['visual']=english['scenes'][0]['visual']
    elif fault=='reorder':english['scenes'][0],english['scenes'][1]=english['scenes'][1],english['scenes'][0]
    elif fault=='noncard':cards[next(iter(cards))]['kind']='gui_capture'
    elif fault=='card_text':cards[next(iter(cards))]['narration_sha256']='c'*64
    elif fault=='source_hash':edit['master_sha256']='c'*64
    elif fault=='missing_voice':tracks.pop()
    else:edit['scenes']=[{'visual':'missing_native'}]
    with pytest.raises(ValueError):mix.plan_mix(english,original,tracks,cards,edit,bindings,fences)

def test_new_long_voice_can_hold_only_an_actual_reviewed_early_idle_result(source_plan):
    a=deepcopy(source_plan);a[2][0]['scenes'][0].update(speech_end=4,duration=4.52)
    with pytest.raises(ValueError,match='reviewed early idle'):mix.plan_mix(*a)
    a[-1]['native_0']=1
    row=mix.plan_mix(*a)[0]
    assert row['kept_frames']==90 and row['final_frame_hold_frames']==46
    assert row['duration']==136/30
    a[-1]['native_0']=1.49
    with pytest.raises(ValueError,match='reviewed early idle'):mix.plan_mix(*a)

@pytest.mark.parametrize('native_index',[0,20])
def test_approved_native_interior_edit_still_requires_shortest_full_speech_fit(source_plan,native_index):
    a=deepcopy(source_plan)
    visual=f'native_{native_index}'
    scene_index=next(i for i,s in enumerate(a[0]['scenes']) if s['visual']==visual)
    a[4]['scenes']=[dict(visual=visual,keep_frames=[[0,30]],confirmed_idle_frames=[[30,90]],final_result_frame=29)]
    untouched=deepcopy(a[1])
    row=next(r for r in mix.plan_mix(*a) if r['visual']==visual)
    assert a[1]==untouched and all('scene' not in r for r in a[1]['scenes'])
    assert row['source_start_frame']==native_index*90
    assert row['kept_frames']==30 and row['final_frame_hold_frames']==30 and row['edited']
    a[2][0]['scenes'][scene_index]['speech_end']=a[2][0]['scenes'][scene_index]['speech_start']+.99
    with pytest.raises(ValueError,match='shortest full speech'):mix.plan_mix(*a)

@pytest.fixture
def generated_cards(tmp_path):
    # One normal generator invocation per consuming test fixture, no model/GUI.
    from stage_lesson import REPO
    import render_home_reference_cards as generator
    canonical=REPO/'tools/tutorials/lessons/05_home.json'
    old=sys.argv
    try:
        sys.argv=['render_home_reference_cards.py','--repo',str(REPO),'--lesson',str(canonical),
                  '--destination',str(tmp_path/'reference-cards')]
        generator.main()
    finally:sys.argv=old
    return generator,tmp_path/'reference-cards',json.loads(canonical.read_text()),REPO

def test_normal_six_cards_reject_pixel_definition_and_narration_tampering(generated_cards,monkeypatch):
    generator,folder,english,repo=generated_cards
    cards=generator.validate_cards(folder,english,repo)
    assert set(cards)==mix.CARDS and all(r['kind']=='authored_reference_card' for r in cards.values())
    changed=deepcopy(english);next(s for s in changed['scenes'] if s['visual'] in mix.CARDS)['narration']+=' changed'
    with pytest.raises(ValueError,match='source binding'):generator.validate_cards(folder,changed,repo)
    definition=deepcopy(generator.DEFINITIONS);key=next(iter(definition));title,lines,refs=definition[key]
    definition[key]=(title+' changed',lines,refs)
    with monkeypatch.context() as patch:
        patch.setattr(generator,'DEFINITIONS',definition)
        with pytest.raises(ValueError,match='source binding'):generator.validate_cards(folder,english,repo)
    # A changed source-reference identity is rejected independently of pixels.
    contract=folder/'cards.json';original=contract.read_text();data=json.loads(original)
    data['cards'][0]['references'][0]['sha256']='c'*64;contract.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='bytes, source or narration'):generator.validate_cards(folder,english,repo)
    contract.write_text(original)
    image=folder/next(iter(cards.values()))['image'];image.write_bytes(image.read_bytes()+b'changed PNG bytes')
    with pytest.raises(ValueError,match='bytes, source or narration'):generator.validate_cards(folder,english,repo)


@pytest.fixture
def umap_plan():
    source=[*mix.UMAP_SOURCE, '27_review_embedding']
    original={'scenes':[dict(scene=i+1,visual=v,speech_start=i*3,scene_end=(i+1)*3,duration=3,native_live_video=True)
                        for i,v in enumerate(source)]}
    english={'id':'15_image_umap','scenes':[dict(visual=v,narration='Feature '+v)
             for v in [*mix.UMAP_SOURCE,mix.UMAP_CARD]]}
    cards={mix.UMAP_CARD:dict(visual=mix.UMAP_CARD,kind='authored_reference_card',
        narration_sha256=mix.identity(english['scenes'][-1]['narration']))}
    tracks=[dict(scenes=[dict(speech_start=i*2,speech_end=i*2+1.48,duration=2)
                        for i in range(15)]) for _ in range(50)]
    bindings={'master':{'sha256':'a'*64},'timing':{'sha256':'b'*64}}
    edit=dict(schema=1,lesson='15_image_umap',master_sha256='a'*64,visual_timing_sha256='b'*64,scenes=[])
    return english,original,tracks,cards,edit,bindings,{}


def test_umap_replaces_only_redundant_last_view_with_full_frame_linked_tool_card(umap_plan):
    rows=mix.plan_mix(*umap_plan)
    assert [r['visual'] for r in rows[:-1]]==list(mix.UMAP_SOURCE)
    assert [r['source_start_frame'] for r in rows[:-1]]==[i*90 for i in range(14)]
    assert all(r['source_keep_frames']==[[0,90]] and r['native_live_video'] for r in rows[:-1])
    assert rows[13]['visual']=='26_point_200' and rows[13]['source_start_frame']==1170
    assert rows[-1]['visual']==mix.UMAP_CARD and not rows[-1]['native_live_video']
    assert rows[-1]['reference_card']['kind']=='authored_reference_card'
    assert 'source_start_frame' not in rows[-1] and 'focus' not in rows[-1]
    assert rows[-1]['duration']==2 and rows[-1]['scene_end']==44


@pytest.mark.parametrize('fault',['unknown_source','source_reorder','wrong_discarded_view',
    'early_card','unknown_card','native_flag','card_kind','card_text','source_hash',
    'missing_voice','discarded_view_edit'])
def test_umap_mix_rejects_nonreviewed_replacement_or_changed_proofs(umap_plan,fault):
    a=deepcopy(umap_plan);english,original,tracks,cards,edit,bindings,fences=a
    if fault=='unknown_source':original['scenes'][0]['visual']='other_source'
    elif fault=='source_reorder':original['scenes'][0],original['scenes'][1]=original['scenes'][1],original['scenes'][0]
    elif fault=='wrong_discarded_view':original['scenes'][-1]['visual']='real_only_result'
    elif fault=='early_card':english['scenes'][-1],english['scenes'][0]=english['scenes'][0],english['scenes'][-1]
    elif fault=='unknown_card':cards['other_card']=cards.pop(mix.UMAP_CARD)
    elif fault=='native_flag':original['scenes'][13]['native_live_video']=False
    elif fault=='card_kind':cards[mix.UMAP_CARD]['kind']='gui_capture'
    elif fault=='card_text':cards[mix.UMAP_CARD]['narration_sha256']='c'*64
    elif fault=='source_hash':edit['master_sha256']='c'*64
    elif fault=='missing_voice':tracks.pop()
    else:edit['scenes']=[dict(visual='27_review_embedding')]
    with pytest.raises(ValueError):mix.plan_mix(*a)


def test_umap_indexed_interior_edit_preserves_original_clock_and_shortest_voice_limit(umap_plan):
    a=deepcopy(umap_plan);visual=mix.UMAP_SOURCE[13]
    a[4]['scenes']=[dict(source_scene=14,visual=visual,keep_frames=[[0,30]],
        confirmed_idle_frames=[[30,90]],final_result_frame=29)]
    untouched=deepcopy(a[1])
    row=mix.plan_mix(*a)[13]
    assert a[1]==untouched, 'Virtual one-row indexing must not mutate published input clocks/identity'
    assert row['source_start_frame']==1170 and row['kept_frames']==30 and row['edited']
    a[4]['scenes'][0]['source_scene']=13
    with pytest.raises(ValueError,match='source index'):mix.plan_mix(*a)
    a[4]['scenes'][0]['source_scene']=14
    a[2][0]['scenes'][13]['speech_end']=a[2][0]['scenes'][13]['speech_start']+.99
    with pytest.raises(ValueError,match='shortest full speech'):mix.plan_mix(*a)


@pytest.mark.parametrize('wrong_index',[13,15,0,True,'14',None])
def test_umap_interior_edit_rejects_wrong_original_source_index(umap_plan,wrong_index):
    a=deepcopy(umap_plan);visual=mix.UMAP_SOURCE[13]
    a[1]['scenes'][13]['scene']=wrong_index
    a[4]['scenes']=[dict(source_scene=14,visual=visual,keep_frames=[[0,30]],
        confirmed_idle_frames=[[30,90]],final_result_frame=29)]
    original=deepcopy(a[1])
    with pytest.raises(ValueError,match='original source index'):mix.plan_mix(*a)
    assert a[1]==original


def test_umap_normal_card_binds_actual_header_sources_and_preserves_canonical_focus(tmp_path,monkeypatch):
    from stage_lesson import REPO
    import render_umap_reference_card as generator
    canonical=REPO/'tools/tutorials/lessons/15_image_umap.json'
    english=json.loads(canonical.read_text());before=canonical.read_bytes()
    destination=tmp_path/'reference-cards'
    monkeypatch.setattr(sys,'argv',['render_umap_reference_card.py','--repo',str(REPO),
        '--lesson',str(canonical),'--destination',str(destination)])
    generator.main()
    cards=generator.validate_cards(destination,english,REPO)
    assert set(cards)=={mix.UMAP_CARD} and canonical.read_bytes()==before
    assert english['scenes'][-1]['focus_buttons']=='FoldButton'
    assert cards[mix.UMAP_CARD]['references'][0]['path']=='spacr/qt/screens/image_umap.py'
    changed=deepcopy(english);changed['scenes'][-1]['narration']+=' changed'
    with pytest.raises(ValueError,match='source binding'):generator.validate_cards(destination,changed,REPO)
    contract=destination/'cards.json';value=json.loads(contract.read_text())
    value['cards'][0]['references'][0]['sha256']='c'*64;contract.write_text(json.dumps(value))
    with pytest.raises(ValueError,match='bytes, source or narration'):generator.validate_cards(destination,english,REPO)



def test_explicit_canonical_source_selects_whole_unique_binding_without_mutation(tmp_path,monkeypatch):
    path=tmp_path/'tools/tutorials/idle_tail_policy.json';path.parent.mkdir(parents=True)
    entries=[dict(lesson=lesson,master_sha256='a'*64,visual_timing_sha256='b'*64,
                  scenes=[dict(visual='actual',source_start=12.0,source_end=22.0,idle_tail_start=9.5)])
             for lesson in ('04_platform_installers','21_model_compare')]
    monkeypatch.setattr(mix,'REPO',tmp_path)
    path.write_text(json.dumps(dict(schema=1,lessons=entries)))
    for entry in entries:
        selected=mix.canonical_source_policy(entry['lesson'])
        assert selected==dict(schema=1,lessons=[entry])
        selected['lessons'][0]['scenes'][0]['idle_tail_start']=0
        assert mix.canonical_source_policy(entry['lesson'])['lessons'][0]==entry
    for invalid in ([],[entries[1],entries[1]]):
        path.write_text(json.dumps(dict(schema=1,lessons=invalid)))
        with pytest.raises(ValueError,match='exactly one'):mix.canonical_source_policy('21_model_compare')
    with pytest.raises(ValueError,match='reviewed only'):mix.canonical_source_policy('05_home')


def test_explicit_canonical_cli_passes_whole_binding_and_excludes_file_input(tmp_path,monkeypatch):
    path=tmp_path/'tools/tutorials/idle_tail_policy.json';path.parent.mkdir(parents=True)
    entry=dict(lesson='21_model_compare',master_sha256='a'*64,visual_timing_sha256='b'*64,scenes=[])
    path.write_text(json.dumps(dict(schema=1,lessons=[entry])))
    monkeypatch.setattr(mix,'REPO',tmp_path);calls=[]
    monkeypatch.setattr(mix,'render',lambda *args:calls.append(args))
    argv=['mix','--stage',str(tmp_path/'stage'),'--baseline',str(tmp_path/'baseline'),'--lesson','21_model_compare','--canonical-source-policy']
    monkeypatch.setattr(sys,'argv',argv);mix.main()
    assert len(calls)==1 and calls[0][3:]==(dict(schema=1,lessons=[entry]),'21_model_compare')
    monkeypatch.setattr(sys,'argv',argv+['--source-idle-policy',str(tmp_path/'unused.json')])
    with pytest.raises(SystemExit):mix.main()
    assert len(calls)==1


def test_canonical_flag_does_not_change_legacy_defaults_or_explicit_file(tmp_path,monkeypatch):
    path=tmp_path/'tools/tutorials/idle_tail_policy.json';path.parent.mkdir(parents=True)
    platform=dict(lesson='04_platform_installers',master_sha256='a'*64,visual_timing_sha256='b'*64,scenes=[])
    path.write_text(json.dumps(dict(schema=1,lessons=[platform])))
    custom=tmp_path/'reviewed.json';custom_policy=dict(schema=1,lessons=[dict(lesson='21_model_compare',scenes=[])])
    custom.write_text(json.dumps(custom_policy));monkeypatch.setattr(mix,'REPO',tmp_path);calls=[]
    monkeypatch.setattr(mix,'render',lambda *args:calls.append(args))
    for lesson in ('05_home','15_image_umap','21_model_compare','04_platform_installers'):
        argv=['mix','--stage',str(tmp_path/'stage'),'--baseline',str(tmp_path/'baseline'),'--lesson',lesson]
        monkeypatch.setattr(sys,'argv',argv);mix.main()
        expected=dict(schema=1,lessons=[platform] if lesson=='04_platform_installers' else [])
        assert calls[-1][3:]==(expected,lesson)
    monkeypatch.setattr(sys,'argv',argv[:-1]+['21_model_compare','--source-idle-policy',str(custom)]);mix.main()
    assert calls[-1][3:]==(custom_policy,'21_model_compare')


@pytest.fixture
def installation_plan():
    from render_installation_reference_cards import VISUALS, CARDS, BOUNDS, MASTER_SHA, TIMING_SHA
    original=dict(lesson='02_install_spacr',kind='native_visual_timing',fps=30,native_live_video=True,
        playback_speed=1,looped_or_stretched=False,total_duration=16614/30,coverage=dict(accepted=True,gaps=[]),
        scenes=[dict(scene=i,visual=v,speech_start=a/30,speech_end=b/30,scene_end=b/30,duration=(b-a)/30,
            native_live_video=i==16) for i,(v,(a,b)) in enumerate(zip(VISUALS,BOUNDS),1)])
    english=dict(id='02_install_spacr',scenes=[dict(visual=v,narration='Feature '+v) for v in VISUALS])
    tracks=[dict(scenes=[dict(speech_start=i*2,speech_end=i*2+1.48,duration=2) for i in range(21)]) for _ in range(50)]
    cards={v:dict(visual=v,kind='authored_reference_card',narration_sha256=mix.identity(english['scenes'][VISUALS.index(v)]['narration'])) for v in CARDS}
    bindings=dict(master=dict(sha256=MASTER_SHA),timing=dict(sha256=TIMING_SHA))
    edit=dict(schema=1,lesson=english['id'],master_sha256=MASTER_SHA,visual_timing_sha256=TIMING_SHA,scenes=[])
    return english,original,tracks,cards,edit,bindings,{v:0 for v in VISUALS}


def test_installation_refresh_preserves_twenty_one_clocks_and_eighteen_complete_source_intervals(installation_plan):
    original=deepcopy(installation_plan)
    rows=mix.plan_mix(*installation_plan)
    assert installation_plan==original
    assert [r['scene'] for r in rows]==list(range(1,22))
    assert [i for i,r in enumerate(rows,1) if 'reference_card' in r]==[2,3,6]
    assert [i for i,r in enumerate(rows,1) if r['native_live_video']]==[16]
    for row,source in zip(rows,original[1]['scenes']):
        assert all(row[k]==source[k] for k in ('speech_start','speech_end','scene_end','duration'))
        assert row['edited'] is False
        if 'reference_card' not in row:
            assert row['source_keep_frames']==[[0,round(source['duration']*30)]]
            assert row['final_frame_hold_frames']==0


@pytest.mark.parametrize('fault',['extra_card','missing_card','order','native_flag','index','clock','master','timing',
    'source_edit','missing_voice','card_text','card_kind','missing_fence','card_fence','card_overflow','retained_overflow'])
def test_installation_refresh_rejects_changed_contracts(installation_plan,fault):
    a=deepcopy(installation_plan);english,source,tracks,cards,edit,bindings,fences=a
    if fault=='extra_card':cards['unknown']=next(iter(cards.values()))
    elif fault=='missing_card':cards.pop(next(iter(cards)))
    elif fault=='order':english['scenes'][0],english['scenes'][1]=english['scenes'][1],english['scenes'][0]
    elif fault=='native_flag':source['scenes'][0]['native_live_video']=True
    elif fault=='index':source['scenes'][0]['scene']=2
    elif fault=='clock':source['scenes'][1]['speech_start']+=1/30
    elif fault=='master':edit['master_sha256']=bindings['master']['sha256']='c'*64
    elif fault=='timing':edit['visual_timing_sha256']=bindings['timing']['sha256']='c'*64
    elif fault=='source_edit':edit['scenes']=[dict(visual=english['scenes'][0]['visual'])]
    elif fault=='missing_voice':tracks.pop()
    elif fault=='card_text':next(iter(cards.values()))['narration_sha256']='c'*64
    elif fault=='card_kind':next(iter(cards.values()))['kind']='gui_capture'
    elif fault=='missing_fence':fences.pop(next(iter(fences)))
    elif fault=='card_fence':fences[next(iter(cards))]=.1
    else:
        index=1 if fault=='card_overflow' else 0
        tracks[0]['scenes'][index].update(speech_end=10000,duration=10000)
    with pytest.raises(ValueError):mix.plan_mix(*a)


def test_installation_card_definitions_change_only_reviewed_labels():
    import render_pip_instruction_cards as renderer
    from render_installation_reference_cards import CARDS
    definitions={name:(title,rows,note) for name,title,rows,note in renderer.instruction_cards('1.5.1.3')}
    assert set(CARDS.values())=={'07_create_environment','08_activate_environment','10_launch_commands'}
    assert definitions['10_launch_commands'][1]==[
        ('Normal launch — first-run questions may appear','spacr'),
        ('Optional: skip first-run setup','spacr --no-setup')]
    assert definitions['07_create_environment'][1][0][1]=='python3.12 -m venv .venv'
    assert definitions['08_activate_environment'][1][0][1]=='source .venv/bin/activate'


@pytest.mark.parametrize('fault',['pixels','source_clock','definition','narration','extra_file'])
def test_installation_generated_card_contract_rejects_tampering(installation_plan,tmp_path,monkeypatch,fault):
    import render_installation_reference_cards as generator
    import published_native_reuse as reuse
    from stage_lesson import REPO
    english,original,*_=installation_plan
    # Unit-only published-input boundary: no real candidate, media decode or capture receipt.
    baseline=tmp_path/'baseline';baseline.mkdir()
    source=baseline/'clock.json';source.write_text(json.dumps(original))
    monkeypatch.setattr(generator,'TIMING_SHA',generator.digest(source))
    bindings=dict(master=dict(sha256=generator.MASTER_SHA),timing=dict(sha256=generator.TIMING_SHA,path='clock.json'))
    monkeypatch.setattr(reuse,'_published_inputs',lambda *_:(baseline,original,bindings,{}))
    folder=tmp_path/'cards';generator.render(REPO,english,baseline,folder)
    assert set(generator.validate_cards(folder,english,REPO))==set(generator.CARDS)
    if fault=='pixels':
        image=folder/(next(iter(generator.CARDS))+'.png');image.write_bytes(image.read_bytes()+b'changed')
    elif fault=='source_clock':source=folder/'source-native-timings.json';source.write_bytes(source.read_bytes()+b' ')
    elif fault=='definition':
        previous=generator.definitions
        def changed(repo):
            renderer,rows=previous(repo);rows=deepcopy(rows);visual=next(iter(rows));title,lines,note=rows[visual]
            rows[visual]=(title+' changed',lines,note);return renderer,rows
        monkeypatch.setattr(generator,'definitions',changed)
    elif fault=='narration':english['scenes'][1]['narration']+=' changed'
    else:(folder/'unexpected.png').write_bytes(b'not a reviewed card')
    with pytest.raises(ValueError):generator.validate_cards(folder,english,REPO)
