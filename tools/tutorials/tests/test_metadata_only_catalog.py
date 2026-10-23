"""Reviewed metadata-only corrections preserve complete published scene/media objects."""
from copy import deepcopy
from pathlib import Path
import hashlib,json,sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from build_appended_candidate import CATALOGS,synchronize_metadata,synchronize_links,METADATA_FIELDS

def hash_source(source):return hashlib.sha256(json.dumps(source,sort_keys=True,ensure_ascii=False).encode()).hexdigest()

@pytest.fixture
def metadata_case():
 source=dict(id='59_example',title='Title',section='Tools',app_key='anndata_export',host_app_key='measure',description='Description',objectives=['Inspect the output'],prerequisite='Choose the folder.',
  scenes=[dict(visual='repeated',narration='First.',hold_after=.7),dict(visual='repeated',narration='Second.',hold_after=.7)])
 old=deepcopy(source);source['prerequisite']='Choose the measured project folder.'
 catalogs={};reviews={}
 for name in CATALOGS:
  language=name.split('_',1)[1].removesuffix('.json');lesson=deepcopy(old)
  lesson.update(narration_voices={'en':['af_heart']},silent='59_example/video/master.mp4',visual_timings='59_example/video/native-live-timings.json')
  if language!='en':
   lesson['scenes'][0]['speech_text']='Preserve exact historical spoken plan.'
   lesson['scenes'][1]['custom']={'old':'reviewed'}
   reviews[source['id'],language]=dict(language=language,lesson=source['id'],english_sha256=hash_source(source),
    title=old['title'],section=old['section'],description='Reviewed description '+language,objectives=['Reviewed objective '+language],prerequisite='Reviewed setup '+language,
    scenes=['First.','Second.'])
  catalogs[name]={'lessons':[lesson,dict(id='unrelated',scenes=[dict(narration='Keep untouched.')])]}
 return catalogs,source,reviews

def test_all_reviewed_metadata_changes_preserve_every_scene_voice_media_and_unselected_field(metadata_case):
 catalogs,source,reviews=metadata_case;snapshot=deepcopy(metadata_case)
 result=synchronize_metadata(catalogs,[source],reviews)
 for name in CATALOGS:
  before=catalogs[name]['lessons'][0];after=result[name]['lessons'][0]
  assert {k:v for k,v in before.items() if k not in METADATA_FIELDS}=={k:v for k,v in after.items() if k not in METADATA_FIELDS}
  assert result[name]['lessons'][1]==catalogs[name]['lessons'][1]
  language=name.split('_',1)[1].removesuffix('.json');review=source if language=='en' else reviews[source['id'],language]
  assert {k:after[k] for k in METADATA_FIELDS}=={k:review[k] for k in METADATA_FIELDS}
 assert metadata_case==snapshot

@pytest.mark.parametrize('fault',['canonical_scene','canonical_title','missing_catalog','missing_review','stale_review','review_narration','review_title','duplicate_lesson','missing_lesson','empty_metadata','missing_scenes','missing_route','missing_title','invalid_scene','invalid_id'])
def test_metadata_only_rejects_any_unreviewed_or_nonmetadata_drift(metadata_case,fault):
 catalogs,source,reviews=deepcopy(metadata_case);language=next(lang for identity,lang in reviews)
 name=next(n for n in CATALOGS if n.split('_',1)[1].removesuffix('.json')==language)
 if fault=='missing_scenes':source.pop('scenes')
 elif fault=='missing_route':source.pop('host_app_key')
 elif fault=='missing_title':source.pop('title')
 elif fault=='invalid_scene':source['scenes'][0]['narration']=None
 elif fault=='invalid_id':source['id']='../59_example'
 elif fault=='canonical_scene':source['scenes'][0]['hold_after']=.9
 elif fault=='canonical_title':source['title']='Changed title'
 elif fault=='missing_catalog':catalogs.pop(name)
 elif fault=='missing_review':reviews.pop((source['id'],language))
 elif fault=='stale_review':reviews[source['id'],language]['english_sha256']='0'*64
 elif fault=='review_narration':reviews[source['id'],language]['scenes'][0]='Different speech.'
 elif fault=='review_title':reviews[source['id'],language]['title']='Changed title'
 elif fault=='duplicate_lesson':catalogs[name]['lessons'].append(deepcopy(catalogs[name]['lessons'][0]))
 elif fault=='missing_lesson':catalogs[name]['lessons'].pop(0)
 else:reviews[source['id'],language]['objectives']=[]
 snapshot=deepcopy(catalogs)
 with pytest.raises(ValueError):synchronize_metadata(catalogs,[source],reviews)
 assert catalogs==snapshot

def test_caption_only_scene_structures_remain_positional_and_byte_identical(metadata_case):
 catalogs,source,reviews=metadata_case
 caption=next(n for n in CATALOGS if n.startswith('captions_'))
 old=catalogs[caption]['lessons'][0]
 for key in ('title','section','description','objectives','prerequisite'):old.pop(key,None)
 result=synchronize_metadata(catalogs,[source],reviews)
 assert result[caption]['lessons'][0]['scenes']==old['scenes']
 assert result[caption]['lessons'][0]['prerequisite']

def test_metadata_path_does_not_weaken_link_only_guard(metadata_case):
 catalogs,source,reviews=metadata_case
 with pytest.raises(ValueError,match='Link-only'):synchronize_links(catalogs,[source])


@pytest.mark.parametrize('mode,field',[('--refresh-links','link_ids'),('--refresh-metadata','metadata_ids')])
def test_explicit_preserved_only_cli_modes_need_no_media_refresh(monkeypatch,mode,field):
 import build_appended_candidate as builder
 calls=[]
 monkeypatch.setattr(builder,'build',lambda *args,**kwargs:calls.append((args,kwargs)))
 builder.main(['--stage','/unused/stage','--baseline','/unused/baseline',mode,'59_example'])
 assert len(calls)==1 and calls[0][0][2]==[]
 assert calls[0][1][field]==['59_example'] and calls[0][1]['refresh_ids']==[]


def test_cli_without_any_explicit_selection_rejected(monkeypatch):
 import build_appended_candidate as builder
 monkeypatch.setattr(builder,'build',lambda *args,**kwargs:pytest.fail('No selection must not build'))
 with pytest.raises(SystemExit) as error:builder.main(['--stage','/unused/stage','--baseline','/unused/baseline'])
 assert error.value.code==2
