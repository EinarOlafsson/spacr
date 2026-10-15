from pathlib import Path
import hashlib
import json
import sys
import tempfile

root=Path.cwd()
scratch=Path('/mnt/wd4tb/scratch/f561-classifier-application-20261015')
sys.path.insert(0,str(root/'tools'))
import build_i18n_catalogs as runtime
runtime.canonical_sources=lambda *args,**kwargs: (_ for _ in ()).throw(AssertionError('No full collector'))
import build_documentation_i18n as api
inventory=json.loads((scratch/'scoped-api-inventory.json').read_text())
docs={key:row['source'] for key,row in inventory['changes'].items()}
prior=Path('/mnt/wd4tb/scratch/f561-classifier-backend-20261015/api-drafts')
result={}
manifest=json.loads((scratch/'api-drafts-manifest.json').read_text())
for lang in manifest['languages']:
    path=scratch/'api-drafts'/f'{lang}.json'
    draft=json.loads(path.read_text())
    for row in draft['records']:
        row['context']=api._api_translation_source(row['source'])
    path.write_text(json.dumps(draft,ensure_ascii=False,indent=2)+'\n')
    manifest['languages'][lang]['sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    old=json.loads((prior/f'{lang}.json').read_text())
    oldstream=[row for row in old['records'] if row['label']=='spacr.torch_artifacts.load_model_artifact#2']
    assert len(oldstream)==1
    with tempfile.TemporaryDirectory(dir=scratch,prefix='review-input-') as directory:
        reviewroot=Path(directory);(reviewroot/lang).mkdir()
        (reviewroot/lang/'old.json').write_text(json.dumps({'schema':1,'language':lang,'records':oldstream}))
        api.REVIEWED_API_DIR=reviewroot
        try:
            api.reviewed_api_block_translations(docs,lang)
        except ValueError as exc:
            assert 'stale reviewed API source' in str(exc),str(exc)
            refusal=str(exc)
        else:
            raise AssertionError('Old stream source must refuse')
    result[lang]={'prior_file_sha256':hashlib.sha256((prior/f'{lang}.json').read_bytes()).hexdigest(),'old_stream_source_sha256':oldstream[0]['source_sha256'],'normal_loader_refusal':refusal,'normal_retirement_required_only':'spacr.torch_artifacts.load_model_artifact#2','draft_context_from_normal_helper':True}
(scratch/'normal-retirement-proof.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
(scratch/'api-drafts-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('Nine exact old stream source rows refused by normal loader; original81 inputs untouched;27 new contexts normalized')
