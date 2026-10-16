from pathlib import Path
import ast,hashlib,importlib.util,sys,json,subprocess,os
p=Path('/mnt/wd4tb/scratch/ci-providers-api-label-diagnosis-20261015');root=Path('/mnt/wd4tb/spacr-worktrees/codex-n703-cache-cleanup-gui-20261015');sys.path.insert(0,str(root/'tools'));s=importlib.util.spec_from_file_location('historical_provider_proof',root/'tools/build_documentation_i18n.py');t=importlib.util.module_from_spec(s);s.loader.exec_module(t)
d=json.loads((p/'historical-en-prehosted-records.json').read_text());env=dict(os.environ,GIT_NO_LAZY_FETCH='1');proof=[];treecache={};D=json.loads((p/'diagnosis.json').read_text());pack=json.loads((p/'author-blocks.json').read_text())
for symbol,r in d['found'].items():
 rec=r['original_english_record'];source=rec['text'];assert t._source_hash(source)==rec['source_sha256'];assert t._source_block_hashes(source)==rec['source_blocks_sha256']
 if r['commit'] not in treecache:
  raw=subprocess.check_output(['git','-C',str(root),'show',r['commit']+':spacr/qt/ai/providers.py'],env=env);treecache[r['commit']]=(ast.parse(raw),hashlib.sha256(raw).hexdigest())
 tree,sourcehash=treecache[r['commit']];name=symbol.removeprefix('spacr.qt.ai.providers').lstrip('.');node=tree if not name else next(n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==name)
 actual=ast.get_docstring(node);assert t._source_hash(actual)==rec['source_sha256'],symbol
 blocks,_=t.translatable_blocks(source);reuses=[]
 for row in pack[symbol]:
  if not row['reused_targets']:continue
  matches=[i for i,b in enumerate(blocks) if b==row['source']];assert len(matches)==1
  for lang,entry in row['reused_targets'].items():
   old=json.loads(subprocess.check_output(['git','-C',str(root),'show',D['public_revision']+':'+D['locales'][lang]['manifest_path']],env=env))['symbols'][symbol]
   assert old['source_sha256']==rec['source_sha256'] and old['source_blocks_sha256']==rec['source_blocks_sha256']
   hist=t._historical_api_block_translations(rec,old,lang);assert hist[row['source']]==entry['translation'],(symbol,lang,row['index'])
  reuses.append({'new_index':row['index'],'old_index':matches[0],'source_sha256':row['source_sha256'],'nine_locale_normal_history_valid':True})
 proof.append({'symbol':symbol,'original_commit':r['commit'],'original_source_file_sha256':sourcehash,'original_en_blob':r['blob_oid'],'old_source_sha256':rec['source_sha256'],'actual_ast_doc_source_hash_matches':True,'original_english_record_hashes_valid':True,'reused_blocks':reuses})
result={'proof':proof,'reused_blocks':sum(len(r['reused_blocks']) for r in proof),'normal_tool_sha256':hashlib.sha256((root/'tools/build_documentation_i18n.py').read_bytes()).hexdigest(),'scope':'Finite four-original-record proof only; no source history mutation, canonical/public inventory, writer or audit.'};assert result['reused_blocks']==8
(p/'historical-en-normal-reuse-proof.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
