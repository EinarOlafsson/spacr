"""Offline exercise of exact verifier functions; no HTTP or browser."""
import ast
import hashlib
import json
import re
from pathlib import Path
import pytest
import sys
import urllib.parse
NORMAL = Path(__file__).resolve().parents[1] / 'tools/tutorials'
sys.path.insert(0,str(NORMAL))
from archived import ARCHIVED_IDS, visible
SOURCE = Path(__file__).resolve().parents[1] / 'tools/verify_tutorial_live.py'
names = {'_sha256','_catalog_payload','_voice_inventory','source_equivalent_assets',
         'expected_visible_lessons','static_audit'}
tree = ast.parse(SOURCE.read_text())
selected = ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names],type_ignores=[])

def bundle(tmp_path, *, projected=True):
    lessons = [{'id':f'active_{n}','scenes':[{}]} for n in range(59)]
    old = [{'id':n,'scenes':[{}]} for n in ARCHIVED_IDS]
    assert len(old)==25
    cat = {'lessons':lessons,'lesson_aliases':{'02_conda_install':'02_install_spacr','03_pip_install':'02_install_spacr'}}
    def encode(d):return ('window.SPACR_LESSON_CATALOG = Object.freeze('+json.dumps(d)+');').encode()
    localcat = {**cat,'lessons':lessons if projected else lessons+old}
    (tmp_path/'lesson_catalog.js').write_bytes(encode(localcat))
    index = b'<script src="lesson_catalog.js?v=catalog"></script><script src="voice_catalog.js?v=voices"></script><script src="app_v2.js?v=player"></script>'
    voices = ('\n'.join(f'    id: "lang{n}",' for n in range(8))+'\n'+'\n'.join(f'      {{ id: "voice{n}"' for n in range(50))).encode()
    payload={'index.html':index,'app_v2.js':b'"use strict";','voice_catalog.js':voices,'lesson_catalog.js':encode(cat)}
    for name,value in payload.items():
        if name!='lesson_catalog.js':(tmp_path/name).write_bytes(value)
    scope=dict(json=json,re=re,hashlib=hashlib,urllib=urllib,LOCAL=tmp_path,ROOT=tmp_path,visible=visible,
               EXPECTED_CACHE_KEY='catalog',EXPECTED_VOICE_KEY='voices',EXPECTED_APP_KEY='player',RETIRED_VOICES=set())
    exec(compile(selected,str(SOURCE),'exec'),scope)
    scope['_fetch']=lambda url,timeout:payload[url.rsplit('/',1)[-1]]
    return scope,payload,cat,encode

@pytest.mark.parametrize('projected',[True,False])
def test_visible_count_uses_normal_archive_projection(tmp_path,projected):
    scope,_,_,_=bundle(tmp_path,projected=projected)
    report=scope['static_audit']('https://fixture.invalid/',timeout=1,compare_local=projected)
    assert report['lessons']==59 and report['scenes']==59

@pytest.mark.parametrize('fault',['extra','foreign_identity','scene_count','index_hash'])
def test_inventory_and_source_hash_failures_remain_strict(tmp_path,fault):
    scope,payload,cat,encode=bundle(tmp_path)
    if fault=='extra':cat['lessons'].append({'id':'foreign','scenes':[{}]})
    elif fault=='foreign_identity':cat['lessons'][0]['id']='foreign'
    elif fault=='scene_count':cat['lessons'][0]['scenes'].append({})
    else:payload['index.html']+=b'<meta name="foreign" content="changed">'
    payload['lesson_catalog.js']=encode(cat)
    with pytest.raises(AssertionError):scope['static_audit']('https://fixture.invalid/',timeout=1)

def test_mobile_counter_is_derived_not_fixed():
    mobile=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='mobile_browser_audit')
    counter=next(n for n in ast.walk(mobile) if isinstance(n,ast.Assert) and 'lessonPosition' in ast.unparse(n.test))
    assert 'expected_visible_lessons()' in ast.unparse(counter.test)
    assert not any(isinstance(n,ast.Constant) and n.value==84 for n in ast.walk(mobile))

HF_SCRIPT = '<script>window.huggingface={variables:{"SPACE_CREATOR_USER_ID":"651dd0c2e956b4858bd2f10e"}};</script>'
HF_METADATA = ('<meta property="og:title" content="spaCR nightly documentation">'
 '<meta property="og:description" content="Nightly preview of the spaCR documentation">'
 '<meta property="og:image" content="https://cdn-thumbnails.huggingface.co/social-thumbnails/spaces/einarolafsson/spacr-docs-nightly-v2.png">'
 '<meta name="twitter:card" content="summary_large_image">'
 '<meta name="twitter:image" content="https://cdn-thumbnails.huggingface.co/social-thumbnails/spaces/einarolafsson/spacr-docs-nightly-v2.png">')

@pytest.mark.parametrize('fault',[None,'extra_script','extra_content','changed_creator',
                                 'changed_description','missing_tag','authored_tags','moved_script'])
def test_hf_normalization_removes_only_exact_known_host_injection(tmp_path,fault):
    scope,payload,_,_=bundle(tmp_path)
    local = '<head><title>Authored title</title><meta name="description" content="Keep authored metadata"></head>'+payload['index.html'].decode()
    if fault=='authored_tags': local=local.replace('</head>',HF_METADATA+'</head>')
    (tmp_path/'index.html').write_text(local)
    remote=local.replace('<head>','<head>'+HF_SCRIPT,1).replace('</head>',HF_METADATA+'</head>',1)
    if fault=='extra_script':remote=remote.replace('<title>','<script>foreign()</script><title>')
    elif fault=='extra_content':remote += '<p>Changed lesson content</p>'
    elif fault=='changed_creator':remote=remote.replace('651dd0c2e956b4858bd2f10e','000000000000000000000000')
    elif fault=='changed_description':remote=remote.replace('Nightly preview of the spaCR documentation','Unrelated description')
    elif fault=='missing_tag':remote=remote.replace('<meta name="twitter:card" content="summary_large_image">','',1)
    elif fault=='moved_script':remote=remote.replace('<head>'+HF_SCRIPT,'<head>').replace('</head>',HF_SCRIPT+'</head>')
    payload['index.html']=remote.encode()
    normalized=scope['source_equivalent_assets'](payload)
    if fault is None:
        assert normalized['index.html']==local.encode()
        assert scope['static_audit']('https://fixture.invalid/',timeout=1)['hashes_match_local']
    else:
        assert normalized['index.html']!=local.encode()
        with pytest.raises(AssertionError):scope['static_audit']('https://fixture.invalid/',timeout=1)
