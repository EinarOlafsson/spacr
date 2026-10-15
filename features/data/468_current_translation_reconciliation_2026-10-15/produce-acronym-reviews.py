from pathlib import Path
import ast,hashlib,importlib.abc,json,sys
root=Path.cwd();scratch=Path('/mnt/wd4tb/scratch/n468-current-contract-audit-20261015')
class NoHeavy(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in {'torch','transformers','PySide6','PyQt6','PyQt5','tensorflow','cellpose'}:raise RuntimeError('Heavy import forbidden: '+fullname)
sys.meta_path.insert(0,NoHeavy());sys.path.insert(0,str(root/'tools'))
import build_i18n_catalogs as runtime
calls=[]
def forbid(*a,**k):calls.append('forbidden');raise RuntimeError('Fullcollector/modeldecode forbidden')
runtime.canonical_sources=forbid
import build_documentation_i18n as api
api.public_docstrings=forbid;api._translate_api_documents=forbid;api._translate_batches=forbid
sha=lambda b:hashlib.sha256(b).hexdigest();manifest=json.loads((scratch/'acronym-draft-manifest.json').read_text());peerpath=scratch/'api-review-drafts/root-independent-acronym-peer.json';peer=json.loads(peerpath.read_text());assert peer['decision']=='ACCEPT all6 exact existing-target records'
p=root/manifest['source_file'];assert sha(p.read_bytes())==manifest['source_file_sha256'];tree=ast.parse(p.read_text());nodes={}
for n in tree.body:
 if isinstance(n,ast.FunctionDef) and n.name=='paper_folder_name':nodes['spacr.qt.widgets.plaque_preview.paper_folder_name']=n
 if isinstance(n,ast.ClassDef) and n.name=='PlaquePreviewPanel':
  for child in n.body:
   if isinstance(child,ast.FunctionDef) and child.name=='fetch_paper':nodes['spacr.qt.widgets.plaque_preview.PlaquePreviewPanel.fetch_paper']=child
docs={k:api._clean_doc(n) for k,n in nodes.items()};assert len(docs)==2
for k,v in docs.items():assert api._source_hash(v)==manifest['source_docs_sha256'][k]
before={str(p.relative_to(root)):p.read_bytes() for p in api.API_DIR.glob('*.json')};runtimepins={str(p.relative_to(root)):sha(p.read_bytes()) for p in (root/'spacr/qt/i18n_catalogs').glob('*.py')};sourcepins={str(p.relative_to(root)):sha(p.read_bytes()) for p in [root/manifest['source_file'],root/'tools/build_documentation_i18n.py',root/'tools/build_i18n_catalogs.py']};result={}
for lang,info in manifest['drafts'].items():
 draftpath=scratch/'api-review-drafts'/f'{lang}.json';raw=draftpath.read_bytes();assert sha(raw)==info['sha256']==peer['input_files_sha256'][lang];payload=json.loads(raw);old=json.loads(before[str((api.API_DIR/f'{lang}.json').relative_to(root))]);records=payload['records']
 for row in records:
  key,index=row['label'].rsplit('#',1);sb,_=api.translatable_blocks(docs[key]);tb,_=api.translatable_blocks(old['symbols'][key]['text']);assert sb[int(index)]==row['source'] and tb[int(index)]==row['translation'];assert row['context']==api._api_translation_source(row['source'])
 payload['review_kind']='AI technical exact-existing-target review and independent root technical peer; no human/native signoff';payload['independent_peer']={'receipt_sha256':sha(peerpath.read_bytes()),'draft_sha256':sha(raw),'decision':'ACCEPT'};payload['note']='Both exact identifier-input API labels retain current correct target bytes; normal source/context-bound review resolves generic acronym copied-phrase false positive without a source/target/gate change.'
 review=api.REVIEWED_API_DIR/lang/'2026-10-15-plaque-paper-identifiers.json';assert not review.exists();api._write_json(review,payload);accepted=api.reviewed_api_block_translations(docs,lang)
 for row in records:assert accepted[row['source']]==row['translation']
 reused=api.reusable_api_translations(docs,lang);assert set(reused)==set(docs);overlaid=api._reviewed_api_overlay(docs,reused,lang);assert overlaid==reused
 api.write_language(docs,lang,overlaid,preserve_other_symbols=True)
 result[lang]={'normal_review_labels':2,'changed_targets':0,'normal_review_overlay_noop':True,'all_existing_target_metadata_bytes_preserved':True,'review_path':str(review.relative_to(root)),'review_sha256':sha(review.read_bytes())}
for p,b in before.items():assert (root/p).read_bytes()==b,p
for p,h in runtimepins.items():assert sha((root/p).read_bytes())==h,p
for p,h in sourcepins.items():assert sha((root/p).read_bytes())==h,p
assert not calls
receipt={'scope':'Six exact-existing-target API review inputs; guarded ordinary reusable→reviewedoverlay→write_language(...preserve_other_symbols=True). Writer normal no-op; every existingAPIartifact byte-identical.','source_docs':docs,'normal_result':result,'all_API_bytes_unchanged':{p:sha(b) for p,b in before.items()},'runtimepins_unchanged':runtimepins,'sourcepins_unchanged':sourcepins,'forbidden_calls':calls,'peer_sha256':sha(peerpath.read_bytes()),'limitations':'Generic acronym refusals retained as baseline; existing normal exact-reviewed gates unchanged. No whole/current API audit yet; separate guardedselected audit follows.'}
(scratch/'normal-noop-producer-receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n');print(json.dumps(result),flush=True)
