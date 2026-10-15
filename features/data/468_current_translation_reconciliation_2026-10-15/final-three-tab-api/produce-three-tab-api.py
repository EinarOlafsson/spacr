from pathlib import Path
import ast,hashlib,importlib.abc,json,sys
root=Path.cwd();s=Path('/mnt/wd4tb/scratch/n468-current-contract-audit-20261015')
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
sha=lambda b:hashlib.sha256(b).hexdigest();m=json.loads((s/'three-tab-api-manifest.json').read_text());peerpath=s/'three-tab-api-drafts/root-independent-27-target-peer.json';peer=json.loads(peerpath.read_text());assert peer['decision']=='ACCEPT all27 exact source-bound new targets'
p=root/m['source_file'];assert sha(p.read_bytes())==m['source_sha256'];c=next(n for n in ast.parse(p.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='PlaqueSettingsDialog');key='spacr.qt.widgets.plaque_preview.PlaqueSettingsDialog';doc=api._autoapi_class_doc(c);assert doc==m['actual_normal_API_doc'];docs={key:doc};blocks,layout=api.translatable_blocks(doc)
before={str(p.relative_to(root)):p.read_bytes() for p in api.API_DIR.glob('*.json')};rt={str(p.relative_to(root)):sha(p.read_bytes()) for p in (root/'spacr/qt/i18n_catalogs').glob('*.py')};review_before={str(p.relative_to(root)):p.read_bytes() for p in api.REVIEWED_API_DIR.rglob('*.json')};old_en=json.loads(before[str((api.API_DIR/'en.json').relative_to(root))]);out={}
for lang,info in m['drafts'].items():
 draft=s/'three-tab-api-drafts'/f'{lang}.json';raw=draft.read_bytes();assert sha(raw)==info['sha256']==peer['input_files_sha256'][lang];data=json.loads(raw);data['review_kind']='AI technical source/translation review with independent root peer; no native/human signoff';data['independent_peer']={'receipt_sha256':sha(peerpath.read_bytes()),'draft_sha256':sha(raw),'decision':'ACCEPT'}
 for row in m['retirement_proposals']:
  if not row['file'].startswith(f'docs/i18n/reviewed/api/{lang}/'):continue
  q=root/row['file'];old=json.loads(q.read_text());r=old['records'][row['row_index']];assert r==row['old_record'];r['retired']=True;r['retired_reason']='2026-10-15: actual Plaque settings has three tabs; Figure and Text detection are both hidden in Plaque mode. Exact current source-bound replacement recorded separately; original review/history retained.';api._write_json(q,old)
 q=api.REVIEWED_API_DIR/lang/'2026-10-15-plaque-settings-three-tabs.json';assert not q.exists();api._write_json(q,data)
 reviewed=api.reviewed_api_block_translations(docs,lang);_,hist,_,unproven=api._proven_api_history(docs,lang);assert unproven==0
 values=[]
 for i,b in enumerate(blocks):
  value=reviewed.get(b,hist[key].get(b))
  assert value is not None,(lang,i)
  assert api._reviewed_api_block_valid(b,value,lang),(lang,i)
  values.append(value)
 target=api.rebuild_document(layout,values);api.write_language(docs,lang,{key:target},preserve_other_symbols=True)
 old=json.loads(before[str((api.API_DIR/f'{lang}.json').relative_to(root))]);new=json.loads((api.API_DIR/f'{lang}.json').read_text());assert {k:v for k,v in old.items() if k!='symbols'}=={k:v for k,v in new.items() if k!='symbols'};assert {k:v for k,v in old['symbols'].items() if k!=key}=={k:v for k,v in new['symbols'].items() if k!=key}
 old_blocks,_=api.translatable_blocks(old['symbols'][key]['text']);assert all(values[i]==old_blocks[i] for i in [1,3,5]);out[lang]={'changed_blocks':[0,2,4],'other_three_blocks_unchanged':True,'foreign_symbols_preserved':len(old['symbols'])-1}
full_en={k:r['text'] for k,r in old_en['symbols'].items()};full_en[key]=doc;api._write_english_api_manifest(full_en);new_en=json.loads((api.API_DIR/'en.json').read_text());assert {k:v for k,v in old_en['symbols'].items() if k!=key}=={k:v for k,v in new_en['symbols'].items() if k!=key}
for p,h in rt.items():assert sha((root/p).read_bytes())==h
changed_reviews={r['file'] for r in m['retirement_proposals']}
for p,b in review_before.items():
 if p not in changed_reviews:assert (root/p).read_bytes()==b,p
 else:
  old=json.loads(b);new=json.loads((root/p).read_bytes());assert {k:v for k,v in old.items() if k!='records'}=={k:v for k,v in new.items() if k!='records'}
  for oldrow,newrow in zip(old['records'],new['records']):assert oldrow=={k:v for k,v in newrow.items() if k not in {'retired','retired_reason'}} if oldrow!=newrow else True
assert sha((root/m['source_file']).read_bytes())==m['source_sha256'];assert not calls
(s/'three-tab-normal-producer-receipt.json').write_text(json.dumps({'scope':'One rendered class, three changed source blocks, nine locales; ordinary reviewed/history/write_language and full preserved incoming English mapping via normal English writer.','result':out,'source_sha256':m['source_sha256'],'runtime_pins_unchanged':rt,'foreign_reviews_unchanged':len(review_before)-len(changed_reviews),'retired_exact_old_records':27,'new_records':27,'forbidden_calls':calls,'peer_sha256':sha(peerpath.read_bytes())},indent=2)+'\n');print('PASS normal three-tab API producer; nine locales, foreign records/runtime preserved',flush=True)
