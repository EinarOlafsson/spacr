from pathlib import Path
import ast, hashlib, json, shutil, subprocess, sys, tempfile
root=Path.cwd(); scratch=Path('/mnt/wd4tb/scratch/n691-current-whole-item-audit-20261015');sys.path.insert(0,str(root/'tools'))
import build_i18n_catalogs as normal
normal.canonical_sources=lambda *a,**k: (_ for _ in ()).throw(AssertionError('Global runtime collector forbidden'))
import build_documentation_i18n as api
import nested_helper_docs as nested
sha=lambda b:hashlib.sha256(b).hexdigest()
langs=['sv','de','es','zh_CN','fr','pt','hi','ko','is']
paths=['spacr/model_zoo.py','spacr/qt/screens/make_masks.py','spacr/qt/widgets/model_zoo_picker.py']
trees={p:ast.parse((root/p).read_text()) for p in paths}
keys=['spacr.model_zoo.'+n for n in ['is_third_party','mitonet_instances','mitonet_weights','segment_mitonet']]
docs={}
for n in api._module_scope_nodes(trees[paths[0]].body):
 if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and 'spacr.model_zoo.'+n.name in keys:
  assert api._is_visible_function_name(n.name,module_is_package=False) and api._is_rendered_autoapi_entry(root/paths[0],'spacr.model_zoo','spacr.model_zoo.'+n.name)
  docs['spacr.model_zoo.'+n.name]=api._clean_doc(n)
assert len(docs)==4
with tempfile.TemporaryDirectory(dir=scratch,prefix='owned-api-snapshot-') as d:
 snap=Path(d)
 for p in paths:
  q=snap/p;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes((root/p).read_bytes())
 definitions=nested.inventory(snap,ignore_patterns=api.AUTOAPI_IGNORE)
 selected={e.module for e in definitions if not e.ignored_by} & set(nested.ENABLED_MODULES)
 nested_matches=[e.qualified_key for e in nested.entries(definitions,modules=selected) if any(x in e.qualified_key.lower() for x in ['mitonet','is_third_party'])]
 assert not nested_matches,nested_matches
english=json.loads((root/'docs/source/_static/i18n/api/en.json').read_text())['symbols']; result={}
for k,doc in docs.items():assert english[k]['text']==doc
for lang in langs:
 review=api.reviewed_api_block_translations(docs,lang); compiled=json.loads((root/f'docs/source/_static/i18n/api/{lang}.json').read_text())['symbols']; values={}
 for k,doc in docs.items():
  current=compiled[k];assert current['source_sha256']==english[k]['source_sha256']; sb,_=api.translatable_blocks(doc);tb,_=api.translatable_blocks(current['text']);assert len(sb)==len(tb)
  assert current['source_blocks_sha256']==api._source_block_hashes(doc)
  assert current['translation_source_blocks_sha256']==api._translation_source_block_hashes(doc)
  blockproof=[{'index':i,'source_sha256':sha(s.encode()),'translation_sha256':sha(t.encode()),'explicit_review':review.get(s)==t,'normal_generic_source_valid':api._api_block_valid(s,t,lang),'normal_generic_context_valid':api._api_block_valid(api._api_translation_source(s),t,lang)} for i,(s,t) in enumerate(zip(sb,tb))]
  values[k]={'source_sha256':current['source_sha256'],'target_sha256':sha(current['text'].encode()),'blocks':len(sb),'block_checks':blockproof,'explicit_review_count':sum(x['explicit_review'] for x in blockproof),'normal_current_source_and_context_pins':True}
 result[lang]=values
(scratch/'api.json').write_text(json.dumps({'symbols':docs,'normal_eligible_nested_matches':nested_matches,'languages':result,'scope':'Four actual public API symbols; normal visibility and rendered predicates, exact docbody/English source and ordinary scoped reviewed block loader/compiled targets. Nested inventory only three owning files. No full API collector.'},ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'normal_current4APIpins9':'PASS','generic_invalid':{lang:{k:[b['index'] for b in v['block_checks'] if not b['normal_generic_source_valid'] or not b['normal_generic_context_valid']] for k,v in symbols.items()} for lang,symbols in result.items()}}),flush=True)
