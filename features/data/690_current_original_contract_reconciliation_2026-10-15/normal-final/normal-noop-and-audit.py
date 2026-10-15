from pathlib import Path
import ast, hashlib, importlib.abc, json, sys, tempfile
root=Path.cwd(); scratch=Path('/mnt/wd4tb/scratch/n690-current-contract-audit-20261015')
class NoModels(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in {'torch','transformers','tensorflow','cellpose'}:
   raise RuntimeError('Model import forbidden: '+fullname)
sys.meta_path.insert(0,NoModels());sys.path.insert(0,str(root/'tools'))
import build_i18n_catalogs as runtime
original=runtime.canonical_sources; snapshot=None; calls=0
sha=lambda b:hashlib.sha256(b).hexdigest()
def canonical():
 global snapshot,calls
 if snapshot is None:
  calls+=1; snapshot=original(); print('Actual normal runtime context collected once',flush=True)
 return snapshot
runtime.canonical_sources=canonical
import build_documentation_i18n as api
def forbid(*a,**k):raise RuntimeError('Full API collector/model translation forbidden')
api.public_docstrings=forbid;api._translate_api_documents=forbid;api._translate_batches=forbid
manifest=json.loads((scratch/'fr-existing-api-review-manifest.json').read_text()); source=root/manifest['source_file']
assert sha(source.read_bytes())==manifest['source_file_sha256']
node=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='download_tem_example')
docs={manifest['symbol']:api._clean_doc(node)};assert api._source_hash(docs[manifest['symbol']])==manifest['doc_sha256']
review=root/'docs/i18n/reviewed/api/fr/2026-10-15-tem-callback-thread.json'; rows=json.loads(review.read_text())['records'];assert len(rows)==1
row=rows[0];assert row['source']==manifest['source'] and row['translation']==manifest['target'] and row['context']==api._api_translation_source(row['source'])
api_before={str(p.relative_to(root)):sha(p.read_bytes()) for p in api.API_DIR.glob('*.json')}
rt_before={str(p.relative_to(root)):sha(p.read_bytes()) for p in (root/'spacr/qt/i18n_catalogs').glob('*.py')}
sourcepins={str(p.relative_to(root)):sha(p.read_bytes()) for folder in ('spacr','tools') for p in (root/folder).rglob('*.py')}
accepted=api.reviewed_api_block_translations(docs,'fr');assert accepted[row['source']]==row['translation']
reused=api.reusable_api_translations(docs,'fr');assert set(reused)==set(docs)
overlaid=api._reviewed_api_overlay(docs,reused,'fr');assert overlaid==reused
api.write_language(docs,'fr',overlaid,preserve_other_symbols=True)
for p,h in api_before.items():assert sha((root/p).read_bytes())==h,p
print('Normal reviewed/reuse/overlay/write_language PASS; all ten API artifacts byte-identical',flush=True)
with tempfile.TemporaryDirectory(dir=scratch,prefix='actual-tem-fr-api-audit-') as temp:
 for language in ('en','fr'):
  payload=json.loads((api.API_DIR/f'{language}.json').read_text());payload['symbols']={k:payload['symbols'][k] for k in docs};(Path(temp)/f'{language}.json').write_text(json.dumps(payload,ensure_ascii=False))
 api.API_DIR=Path(temp);result=api.audit(docs,['fr']);assert result==0,result
if snapshot is not None:
 fresh=original();calls+=1;assert fresh==snapshot
assert snapshot is not None
for pins in (api_before,rt_before,sourcepins):
 for p,h in pins.items():assert sha((root/p).read_bytes())==h,p
receipt={'scope':'N690 exact existing French callback paragraph, normal reviewed/reuse/overlay/write_language no-op then ordinary actual selected one-symbol EN+FR api.audit; SUBSET not global API audit.','result':result,'runtime_canonical_calls':calls,'fresh_original_canonical_equal':True,'source_docs':docs,'manifest_sha256':sha((scratch/'fr-existing-api-review-manifest.json').read_bytes()),'review_sha256':sha(review.read_bytes()),'sourcepins':sourcepins,'all_API_bytes_unchanged':api_before,'all_runtime_bytes_unchanged':rt_before,'no_model_import_or_full_API_collector':True}
(scratch/'normal-noop-selected-audit-receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n');print('N690 ordinary selected API audit PASS; source/artifact pins preserved',flush=True)
