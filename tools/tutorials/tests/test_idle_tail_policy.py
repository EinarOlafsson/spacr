"""Product-policy bindings; fixtures represent metadata, never capture acceptance."""
from copy import deepcopy
from pathlib import Path
import sys
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from idle_tail_policy import bind_policy, validate_catalog_policy


def fixture():
    lesson={'id':'07_mask','silent':'07_mask/video/master.mp4','visual_timings':'07_mask/video/native.json'}
    records=[{'path':'media_host/'+lesson['silent'],'sha256':'a'*64},
             {'path':'media_host/'+lesson['visual_timings'],'sha256':'b'*64}]
    timing={'lesson':'07_mask','kind':'native_visual_timing','native_live_video':True,'coverage':{'accepted':True},
            'scenes':[{'visual':'table','speech_start':120.,'scene_end':163.}]}
    policy={'schema':1,'lessons':[{'lesson':'07_mask','master_sha256':'a'*64,'visual_timing_sha256':'b'*64,
              'scenes':[{'visual':'table','source_start':120.,'source_end':163.,'idle_tail_start':26.2}]}]}
    return policy,records,[lesson],lambda _:deepcopy(timing)


def test_source_bound_fence_is_product_metadata_and_survives_unrelated_refresh():
    policy,records,lessons,load=fixture()
    bound=bind_policy(policy,records,lessons,load)
    assert 'native_idle_tails' not in lessons[0]
    assert bound[0]['native_idle_tails']['schema']==1
    assert bind_policy({'schema':1,'lessons':[]},records,bound,load)==bound
    validate_catalog_policy(records,bound,load)


def test_explicit_empty_policy_never_invents_idle_fences():
    _,records,lessons,load=fixture()
    assert bind_policy({'schema':1,'lessons':[]},records,lessons,load)==lessons


@pytest.mark.parametrize('change',['master','timing','clock','unknown','negative','outside','nan','duplicate'])
def test_changed_source_or_invalid_fence_fails_closed(change):
    policy,records,lessons,load=fixture();entry=policy['lessons'][0];fence=entry['scenes'][0]
    if change=='master':records[0]['sha256']='c'*64
    elif change=='timing':records[1]['sha256']='c'*64
    elif change=='clock':fence['source_start']+=1
    elif change=='unknown':fence['visual']='missing'
    elif change=='negative':fence['idle_tail_start']=-1
    elif change=='outside':fence['idle_tail_start']=43
    elif change=='nan':fence['idle_tail_start']=float('nan')
    elif change=='duplicate':entry['scenes'].append(deepcopy(fence))
    with pytest.raises(ValueError):bind_policy(policy,records,lessons,load)


def test_inherited_fence_cannot_silently_follow_replaced_video():
    args=fixture();bound=bind_policy(*args);args[1][0]['sha256']='c'*64
    with pytest.raises(ValueError):bind_policy({'schema':1,'lessons':[]},args[1],bound,args[3])


def test_default_policy_is_required_and_explicit_empty_fixture_is_supported(tmp_path):
    import json
    from idle_tail_policy import load_policy
    with pytest.raises(FileNotFoundError):load_policy(tmp_path)
    canonical=tmp_path/'tools/tutorials/idle_tail_policy.json';canonical.parent.mkdir(parents=True)
    expected=fixture()[0];canonical.write_text(json.dumps(expected))
    empty=tmp_path/'empty-policy.json';empty.write_text('{"schema":1,"lessons":[]}')
    assert load_policy(tmp_path)==expected
    assert load_policy(tmp_path,empty)=={'schema':1,'lessons':[]}


def test_thin_checkpoint_preserves_hash_schema_checks_without_clock_claim():
    policy,records,lessons,load=fixture();bound=bind_policy(policy,records,lessons,load)
    validate_catalog_policy(records,bound,None)
    records[1]['sha256']='c'*64
    with pytest.raises(ValueError):validate_catalog_policy(records,bound,None)


def test_full_clock_validation_rejects_changed_scene_even_with_declared_hash():
    policy,records,lessons,load=fixture();bound=bind_policy(policy,records,lessons,load)
    def changed(path):
        result=load(path);result['scenes'][0]['scene_end']+=1;return result
    with pytest.raises(ValueError):validate_catalog_policy(records,bound,changed)
    def missing(path):raise FileNotFoundError(path)
    with pytest.raises(FileNotFoundError):validate_catalog_policy(records,bound,missing)


def test_actual_normal_binder_output_is_accepted_by_actual_runtime_helper():
    import json,os,subprocess
    node=os.environ['SPACR_TUTORIAL_NODE']
    policy,records,lessons,load=fixture();bound=bind_policy(policy,records,lessons,load)
    source=(Path(__file__).resolve().parents[1]/'authoring/web/app_v2.js').read_text()
    helper='function boundNativeIdleTimings('+source.split('function boundNativeIdleTimings(',1)[1].split('async function loadLessonDetail(',1)[0]
    js=helper+'console.log(JSON.stringify(boundNativeIdleTimings('+json.dumps(load(None))+','+json.dumps(bound[0])+','+json.dumps('b'*64)+').scenes[0].idle_tail_start));'
    assert json.loads(subprocess.check_output([node,'-e',js],text=True))==26.2


@pytest.mark.parametrize('thin',[False,True])
def test_normal_candidate_validation_supports_thin_checkpoint_but_rejects_bad_present_json(tmp_path,thin):
    import json,hashlib
    from audit_staged_catalogs import CATALOGS
    from validate_candidate import validate
    from append_staged_lessons import JS_PREFIX,JS_SUFFIX
    root=tmp_path/'candidate';web=root/'web';web.mkdir(parents=True)
    policy,_,lessons,load=fixture();lesson=lessons[0];lesson['scenes']=[{'narration':'Metadata fixture'}]
    timing=load(None);timing_bytes=json.dumps(timing).encode();master_bytes=b'media identity fixture, not a playable capture'
    sha=lambda data:hashlib.sha256(data).hexdigest()
    records=[{'path':'media_host/'+lesson['silent'],'sha256':sha(master_bytes),'bytes':len(master_bytes)},
             {'path':'media_host/'+lesson['visual_timings'],'sha256':sha(timing_bytes),'bytes':len(timing_bytes)}]
    policy['lessons'][0].update(master_sha256=records[0]['sha256'],visual_timing_sha256=records[1]['sha256'])
    bound=bind_policy(policy,records,lessons,load)
    for filename in CATALOGS:
        dest=web/'catalog'/filename;dest.parent.mkdir(exist_ok=True);dest.write_text(json.dumps({'lessons':lessons}))
    (web/'lesson_catalog.js').write_text(JS_PREFIX+json.dumps({'lessons':bound})+JS_SUFFIX)
    for path in web.rglob('*'):
        if path.is_file():records.append({'path':path.relative_to(root).as_posix(),'sha256':sha(path.read_bytes()),'bytes':path.stat().st_size})
    manifest={'routes':1,'ready_lessons':1,'coming_soon':[],'catalog_languages':len(CATALOGS),'files':records}
    (root/'release-manifest.json').write_text(json.dumps(manifest))
    if not thin:
        for record,raw in zip(records[:2],[master_bytes,timing_bytes]):
            dest=root/record['path'];dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(raw)
    assert validate(root,include_hosted_media=not thin)['ready']==1
    if thin:
        # Full validation cannot claim a clock read when hosted JSON is absent.
        with pytest.raises(ValueError):validate(root,include_hosted_media=True)
    dest=root/records[1]['path'];dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text('{"wrong":"present bytes"}')
    with pytest.raises(ValueError):validate(root,include_hosted_media=not thin)


def repeated_visual_fixture():
    policy,records,lessons,load=fixture();timing=load(None)
    timing['scenes']=[
        {'visual':'table','speech_start':0.,'scene_end':12.},
        {'visual':'other','speech_start':12.,'scene_end':30.},
        {'visual':'table','speech_start':30.,'scene_end':44.},
        {'visual':'table','speech_start':70.,'scene_end':95.}]
    # Policy order deliberately differs from source order; visuals alone cannot bind it.
    policy['lessons'][0]['scenes']=[
        {'visual':'table','source_start':70.,'source_end':95.,'idle_tail_start':8.},
        {'visual':'table','source_start':0.,'source_end':12.,'idle_tail_start':4.},
        {'visual':'table','source_start':30.,'source_end':44.,'idle_tail_start':6.}]
    return policy,records,lessons,lambda _:deepcopy(timing)


def runtime_fence_values(timing,lesson):
    import json,os,subprocess
    source=(Path(__file__).resolve().parents[1]/'authoring/web/app_v2.js').read_text()
    helper='function boundNativeIdleTimings('+source.split('function boundNativeIdleTimings(',1)[1].split('async function loadLessonDetail(',1)[0]
    js=helper+'console.log(JSON.stringify(boundNativeIdleTimings('+json.dumps(timing)+','+json.dumps(lesson)+','+json.dumps('b'*64)+').scenes.map(s=>s.idle_tail_start??null)));'
    return json.loads(subprocess.check_output([os.environ['SPACR_TUTORIAL_NODE'],'-e',js],text=True))


def test_three_repeated_visuals_bind_exact_clocks_and_actual_runtime_indices():
    policy,records,lessons,load=repeated_visual_fixture()
    bound=bind_policy(policy,records,lessons,load)
    assert bound[0]['native_idle_tails']['scenes']==policy['lessons'][0]['scenes']
    assert bind_policy({'schema':1,'lessons':[]},records,bound,load)==bound
    validate_catalog_policy(records,bound,None)  # Thin checkpoint makes no clock-read claim.
    validate_catalog_policy(records,bound,load)
    assert runtime_fence_values(load(None),bound[0])==[4.,None,6.,8.]


@pytest.mark.parametrize('change',['duplicate','near_duplicate','ambiguous','mismatch'])
def test_repeated_visual_bad_binding_rejected_by_binder_and_runtime(change):
    policy,records,lessons,load=repeated_visual_fixture();timing=load(None)
    fences=policy['lessons'][0]['scenes']
    if change in ('duplicate','near_duplicate'):
        fences.append(deepcopy(fences[0]))
        if change=='near_duplicate':fences[-1]['source_start']+=5e-8
    elif change=='ambiguous':timing['scenes'].append(deepcopy(timing['scenes'][0]))
    else:fences[0]['source_end']+=2e-7  # Outside unchanged normal clock tolerance.
    with pytest.raises(ValueError):bind_policy(policy,records,lessons,lambda _:deepcopy(timing))
    # Exercise runtime independently of the normal binder's refusal; no partial fence leakage.
    lesson=deepcopy(lessons[0]);lesson['native_idle_tails']={'schema':1,**policy['lessons'][0]}
    for scene in timing['scenes']:scene['idle_tail_start']=1.  # Stale timing fields must be cleared.
    assert runtime_fence_values(timing,lesson)==[None]*len(timing['scenes'])
    if change in ('duplicate','near_duplicate'):
        with pytest.raises(ValueError):bind_policy(policy,records,lessons,None)
