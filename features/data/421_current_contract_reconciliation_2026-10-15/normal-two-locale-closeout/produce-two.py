from pathlib import Path
import ast,hashlib,importlib.abc,json,sys
root=Path.cwd();scratch=Path('/mnt/wd4tb/scratch/n421-current-contract-audit-20261015')
class NoHeavy(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in {'torch','transformers','PySide6','PyQt6','PyQt5','cellpose','tensorflow'}:raise RuntimeError('No heavy import in metadata producer: '+fullname)
sys.meta_path.insert(0,NoHeavy());sys.path.insert(0,str(root/'tools'))
import build_i18n_catalogs as runtime
calls=[]
def forbidden(*a,**k):calls.append('forbidden');raise RuntimeError('No fullcollector/modeldecode')
runtime.canonical_sources=forbidden
import build_documentation_i18n as api
api.public_docstrings=forbidden;api._translate_api_documents=forbidden;api._translate_batches=forbidden
sha=lambda b:hashlib.sha256(b).hexdigest();symbol='spacr.qt.widgets.measure_input_table.MeasureInputTable.add_dropped';sourcepath=root/'spacr/qt/widgets/measure_input_table.py';node=next(n for n in ast.walk(ast.parse(sourcepath.read_text())) if isinstance(n,ast.FunctionDef) and n.name=='add_dropped');doc=api._clean_doc(node);docs={symbol:doc}
manifest=json.loads((scratch/'two-target-manifest.json').read_text());assert sha(sourcepath.read_bytes())==manifest['source_file_sha256'];assert api._source_hash(doc)==manifest['source_doc_sha256']
base={p.name:p.read_bytes() for p in api.API_DIR.glob('*.json')};english=json.loads(base['en.json']);assert english['symbols'][symbol]['text']==doc
protected=list((root/'spacr').rglob('*.py'))+list((root/'tools').rglob('*.py'));pins={str(p.relative_to(root)):sha(p.read_bytes()) for p in protected}
peerpath=scratch/'root-independent-two-target-peer.json';peer=json.loads(peerpath.read_text());assert peer['decision']=='ACCEPT'
results={}
for lang in ['sv','hi']:
 draftpath=root/f'features/data/421_current_contract_reconciliation_2026-10-15/api-drafts/{lang}.json';raw=draftpath.read_bytes();payload=json.loads(raw);row=payload['records'][0];assert sha(raw)==manifest['drafts'][lang]['sha256'] and sha(row['translation'].encode())==manifest['drafts'][lang]['target_sha256'];assert peer['drafts'][lang]['sha256']==sha(raw) and peer['drafts'][lang]['target_sha256']==sha(row['translation'].encode())
 assert row['label']==symbol+'#1' and row['source']==api.translatable_blocks(doc)[0][1] and row['context']==api._api_translation_source(row['source'])
 old=json.loads(base[lang+'.json']);oldrec=old['symbols'][symbol];assert oldrec['source_sha256']==api._source_hash(doc);assert oldrec['source_blocks_sha256']==api._source_block_hashes(doc);assert oldrec['translation_source_blocks_sha256']==api._translation_source_block_hashes(doc)
 payload['review_kind']='AI technical translation and independent root technical peer; no native/human signoff';payload['independent_peer']={'decision':'ACCEPT','receipt_sha256':sha(peerpath.read_bytes()),'input_sha256':sha(raw)};payload['note']='Exact existing add_dropped#1 copied-English block corrected after independent technical peer; source and all unrelated API blocks retained.'
 path=api.REVIEWED_API_DIR/lang/'2026-10-15-hand-drawn-drop-worker.json';assert not path.exists();api._write_json(path,payload)
 accepted=api.reviewed_api_block_translations(docs,lang);assert accepted[row['source']]==row['translation'];reused=api.reusable_api_translations(docs,lang);assert reused=={symbol:oldrec['text']};overlaid=api._reviewed_api_overlay(docs,reused,lang)
 ob,layout=api.translatable_blocks(oldrec['text']);nb,_=api.translatable_blocks(overlaid[symbol]);assert len(ob)==len(nb)
 assert ob[1]==row['source'];assert all(nb[i]==(row['translation'] if i==1 else ob[i]) for i in range(len(ob)));assert overlaid[symbol]==oldrec['text'].replace(ob[1],row['translation'],1)
 api.write_language(docs,lang,overlaid,preserve_other_symbols=True);after=json.loads((api.API_DIR/f'{lang}.json').read_text());assert set(after['symbols'])==set(old['symbols']);foreign=set(old['symbols'])-{symbol}
 for k in old:
  if k!='symbols':assert after[k]==old[k]
 for k in foreign:assert after['symbols'][k]==old['symbols'][k],k
 assert {k:v for k,v in after['symbols'][symbol].items() if k!='text'}=={k:v for k,v in oldrec.items() if k!='text'}
 results[lang]={'blocks_repaired':1,'unchanged_blocks':len(ob)-1,'foreign_symbols_preserved':len(foreign),'foreign_symbols_sha256':sha(json.dumps({k:old['symbols'][k] for k in sorted(foreign)},sort_keys=True,ensure_ascii=False).encode()),'review_path':str(path.relative_to(root)),'review_sha256':sha(path.read_bytes()),'source_context_normal_review_gates':True,'generic_source_context_valid':api._api_block_valid(row['source'],row['translation'],lang) and api._api_block_valid(row['context'],row['translation'],lang)}
for name,b in base.items():
 if name not in ['sv.json','hi.json']:assert (api.API_DIR/name).read_bytes()==b,name
for p,h in pins.items():assert sha((root/p).read_bytes())==h,p
assert not calls
receipt={'scope':'Normal exactsame-symbol SV/HI reusable→reviewedoverlay→write_language(preserve_other_symbols=True); no fullcollector/decoder, no source/runtime/Help mutation.','results':results,'sourcepins':pins,'eight_other_API_files_byteidentical':{n:sha(b) for n,b in base.items() if n not in ['sv.json','hi.json']},'source_symbol':symbol,'doc':doc,'parent_peer_sha256':sha(peerpath.read_bytes()),'forbidden_calls':calls}
(scratch/'normal-producer-receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n');print(json.dumps(results),flush=True)
