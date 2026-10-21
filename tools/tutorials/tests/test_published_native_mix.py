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
    row=next(r for r in mix.plan_mix(*a) if r['visual']==visual)
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
