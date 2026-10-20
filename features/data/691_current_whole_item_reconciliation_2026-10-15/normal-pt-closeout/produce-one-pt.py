from pathlib import Path
import ast, hashlib, importlib.abc, json, os, sys
root=Path.cwd();scratch=Path('/mnt/wd4tb/scratch/n691-current-whole-item-audit-20261015')
class BlockHeavy(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in {'torch','transformers','PySide6','PyQt6','PyQt5','tensorflow','cellpose'}:raise RuntimeError('Forbidden heavyweight import: '+fullname)
sys.meta_path.insert(0,BlockHeavy());sys.path.insert(0,str(root/'tools'))
import build_i18n_catalogs as runtime
calls=[]
def forbid(*a,**k):
 calls.append('forbidden collector/decoder');raise RuntimeError('Full collector/model decode forbidden in scoped PT writer')
runtime.canonical_sources=forbid
import build_documentation_i18n as api
api.public_docstrings=forbid
api._translate_api_documents=forbid
api._translate_batches=forbid
sha=lambda b:hashlib.sha256(b).hexdigest()
sourcepath=root/'spacr/model_zoo.py';raw=sourcepath.read_bytes();tree=ast.parse(raw);node=next(n for n in api._module_scope_nodes(tree.body) if isinstance(n,ast.FunctionDef) and n.name=='mitonet_weights');symbol='spacr.model_zoo.mitonet_weights';doc=api._clean_doc(node);docs={symbol:doc}
assert api._is_visible_function_name(node.name,module_is_package=False) and api._is_rendered_autoapi_entry(sourcepath,'spacr.model_zoo',symbol)
lang='pt';all_api=list(api.API_DIR.glob('*.json'));before_bytes={p.name:p.read_bytes() for p in all_api};before=json.loads(before_bytes['pt.json']);english=json.loads(before_bytes['en.json']);oldrecord=before['symbols'][symbol];assert english['symbols'][symbol]['text']==doc
inputpath=root/'features/data/691_current_whole_item_reconciliation_2026-10-15/pt-draft/2026-10-15-mitonet-verified-cache.json';inputbytes=inputpath.read_bytes();draft=json.loads(inputbytes);row=draft['records'][0];peerpath=scratch/'root-independent-pt-target-peer.json';peer=json.loads(peerpath.read_text());assert peer['decision']=='ACCEPT' and peer['input_file_sha256']==sha(inputbytes) and peer['target_sha256']==sha(row['translation'].encode())
assert row['label']==symbol+'#1' and row['source_sha256']==api._source_hash(row['source']); assert api._api_translation_source(row['source'])==row['context']
sourceblocks,_=api.translatable_blocks(doc);oldblocks,_=api.translatable_blocks(oldrecord['text']);assert sourceblocks[1]==row['source'] and oldblocks[1]==row['source']
assert api._reviewed_api_block_valid(row['source'],row['translation'],lang) and api._reviewed_api_block_valid(row['context'],row['translation'],lang)
assert api._syntax_preserved(row['source'],row['translation'])
assert oldrecord['source_sha256']==api._source_hash(doc) and oldrecord['source_blocks_sha256']==api._source_block_hashes(doc) and oldrecord['translation_source_blocks_sha256']==api._translation_source_block_hashes(doc)
protected=[root/'spacr/model_zoo.py',root/'tools/build_i18n_catalogs.py',root/'tools/build_documentation_i18n.py',root/'tools/nested_helper_docs.py']+list((root/'spacr/qt/i18n_catalogs').glob('*.py'))
pins={str(p.relative_to(root)):sha(p.read_bytes()) for p in protected}
reviewpath=api.REVIEWED_API_DIR/lang/inputpath.name;assert not reviewpath.exists();payload=json.loads(inputbytes);payload['review_kind']='AI technical translation and independent root technical peer; no human/native signoff';payload['independent_peer']={'decision':'ACCEPT','receipt':str(peerpath),'receipt_sha256':sha(peerpath.read_bytes()),'reviewed_input_file_sha256':sha(inputbytes)};payload['note']='One exact copied-English Portuguese API paragraph repaired; independent technical peer accepted. No source changes; all other paragraphs and unrelated generated API records retained.'
api._write_json(reviewpath,payload)
accepted=api.reviewed_api_block_translations(docs,lang);assert accepted[row['source']]==row['translation']
reused=api.reusable_api_translations(docs,lang);assert reused=={symbol:oldrecord['text']}
overlaid=api._reviewed_api_overlay(docs,reused,lang);newblocks,_=api.translatable_blocks(overlaid[symbol]);assert len(newblocks)==len(oldblocks)==len(sourceblocks)
assert all(newblocks[i]==(row['translation'] if i==1 else oldblocks[i]) for i in range(len(oldblocks)))
assert overlaid[symbol]==oldrecord['text'].replace(oldblocks[1],row['translation'],1)
api.write_language(docs,lang,overlaid,preserve_other_symbols=True)
after=json.loads((api.API_DIR/'pt.json').read_text());assert set(after)==set(before)
for k in before:
 if k!='symbols':assert after[k]==before[k]
assert set(after['symbols'])==set(before['symbols'])
foreign=[k for k in before['symbols'] if k!=symbol]
for k in foreign:assert after['symbols'][k]==before['symbols'][k],k
newrecord=after['symbols'][symbol];assert {k:v for k,v in newrecord.items() if k!='text'}=={k:v for k,v in oldrecord.items() if k!='text'}
assert newrecord['text']==overlaid[symbol]
for name,data in before_bytes.items():
 if name!='pt.json':assert (api.API_DIR/name).read_bytes()==data,name
for p,h in pins.items():assert sha((root/p).read_bytes())==h,p
assert not calls
receipt={'date':'2026-10-15','source_sha256':sha(raw),'source_doc_sha256':api._source_hash(doc),'normal_functions':['_module_scope_nodes/_clean_doc/visibility/rendered predicates','reviewed_api_block_translations','_reviewed_api_block_valid source+context','reusable_api_translations','_reviewed_api_overlay','write_language(preserve_other_symbols=True)'],'selected_acceptance':'Exact one repaired block through ordinary reviewed API loader and source/context/syntax/semantic/degeneracy/model-artifact gates; metadata/source/body/context block pins exact. This is not a full API audit.','normal_generic_valid':api._api_block_valid(row['source'],row['translation'],lang),'generic_limit':'Borrowed technical vocabulary is accepted by existing exact-reviewed path, no new gate exception.','counts':{'repaired_blocks':1,'symbol_total_blocks':len(oldblocks),'other_blocks_preserved':len(oldblocks)-1,'foreign_PT_symbols_preserved':len(foreign),'other_API_files_byte_identical':len(before_bytes)-1},'input_sha256':sha(inputbytes),'installed_review_path':str(reviewpath.relative_to(root)),'installed_review_sha256':sha(reviewpath.read_bytes()),'independent_peer_sha256':sha(peerpath.read_bytes()),'foreign_PT_symbols_before_after_sha256':sha(json.dumps({k:before['symbols'][k] for k in foreign},sort_keys=True,ensure_ascii=False).encode()),'other_API_files_sha256':{n:sha(b) for n,b in before_bytes.items() if n!='pt.json'},'protected_source_runtime_pins':pins,'PT_before_sha256':sha(before_bytes['pt.json']),'PT_after_sha256':sha((api.API_DIR/'pt.json').read_bytes()),'forbidden_collector_decoder_calls':calls,'scope_limits':'No full API/runtime collection/audit, model decoding, tests, model/data/network/GPU work; original receipts reused and immutable. No source/runtime/Help or foreign catalog changes.'}
(scratch/'normal-producer-receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(receipt['counts']),flush=True)
