from pathlib import Path
import ast,hashlib,json,sys,tempfile
root=Path.cwd();scratch=Path('/mnt/wd4tb/scratch/n468-current-contract-audit-20261015');sys.path.insert(0,str(root/'tools'))
import build_i18n_catalogs as runtime
runtime.canonical_sources=lambda *a,**k:(_ for _ in ()).throw(RuntimeError('No global runtime collector'))
import build_documentation_i18n as api
import nested_helper_docs as nested
sha=lambda b:hashlib.sha256(b).hexdigest();paths=['spacr/plaque.py','spacr/qt/widgets/plaque_preview.py'];core={'FieldRow','FieldTable','field_table_destination','field_table_settings','write_field_table_project','measure_from_field_table'}
docs={};symbols={}
for path in paths:
 p=root/path;module=path[:-3].replace('/','.');tree=ast.parse(p.read_text());items={}
 docs[module]=api._clean_doc(tree)
 for n in api._module_scope_nodes(tree.body):
  if not isinstance(n,(ast.ClassDef,ast.FunctionDef,ast.AsyncFunctionDef)):continue
  if False:continue
  visible=not n.name.startswith('_') if isinstance(n,ast.ClassDef) else api._is_visible_function_name(n.name,module_is_package=False)
  if not visible:continue
  key=module+'.'+n.name
  if not api._is_rendered_autoapi_entry(p,module,key):continue
  doc=api._autoapi_class_doc(n) if isinstance(n,ast.ClassDef) else api._clean_doc(n)
  if doc:docs[key]=doc
  items[key]={'AST_sha256':sha(ast.dump(n).encode()),'line':n.lineno}
  if isinstance(n,ast.ClassDef):
   docs.update({k:v for k,v in api._additional_assignment_docs(n.body,key).items() if api._is_rendered_autoapi_entry(p,module,k)})
   for child in api._module_scope_nodes(n.body):
    if isinstance(child,(ast.FunctionDef,ast.AsyncFunctionDef)) and api._is_visible_function_name(child.name,module_is_package=False):
     ck=key+'.'+child.name;cd=api._clean_doc(child)
     if cd and api._is_rendered_autoapi_entry(p,module,ck):docs[ck]=cd
 docs.update({k:v for k,v in api._additional_assignment_docs(tree.body,module).items() if api._is_rendered_autoapi_entry(p,module,k)})
 symbols[path]={'source_sha256':sha(p.read_bytes()),'owning_public_AST':items}
with tempfile.TemporaryDirectory(dir=scratch,prefix='source-nested-snapshot-') as d:
 snapshot=Path(d)
 for path in paths:
  q=snapshot/path;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes((root/path).read_bytes())
 definitions=nested.inventory(snapshot,ignore_patterns=api.AUTOAPI_IGNORE);selected={e.module for e in definitions if not e.ignored_by}&set(nested.ENABLED_MODULES)
 for e in nested.entries(definitions,modules=selected):
  if False:continue
  docs[e.qualified_key]=e.docstring
assert all(docs.values());(scratch/'actual-docs.json').write_text(json.dumps(docs,ensure_ascii=False,indent=2)+'\n')
langs=['sv','de','es','zh_CN','fr','pt','hi','ko','is'];english=json.loads((root/'docs/source/_static/i18n/api/en.json').read_text())['symbols'];result={};problems=[]
for k,v in docs.items():
 if k not in english or english[k]['text']!=v:problems.append({'language':'en','symbol':k,'reason':'missing or current doc differs'})
for lang in langs:
 compiled=json.loads((root/f'docs/source/_static/i18n/api/{lang}.json').read_text())['symbols'];review=api.reviewed_api_block_translations(docs,lang);records={}
 for k,doc in docs.items():
  current=compiled.get(k,{})
  pins=current.get('source_sha256')==api._source_hash(doc) and current.get('source_blocks_sha256')==api._source_block_hashes(doc) and current.get('translation_source_blocks_sha256')==api._translation_source_block_hashes(doc)
  if not pins:problems.append({'language':lang,'symbol':k,'reason':'source/context pins stale/missing'})
  sb,_=api.translatable_blocks(doc);tb,_=api.translatable_blocks(current.get('text',''));bp=[]
  if len(sb)!=len(tb):problems.append({'language':lang,'symbol':k,'reason':'blockcount mismatch'})
  for i,(s,t) in enumerate(zip(sb,tb)):
   reviewed=review.get(s)==t;valid=a=api._api_block_valid(s,t,lang);cvalid=api._api_block_valid(api._api_translation_source(s),t,lang)
   b={'index':i,'source':s,'source_sha256':sha(s.encode()),'target':t,'target_sha256':sha(t.encode()),'exact_review':reviewed,'normal_generic_source_valid':valid,'normal_generic_context_valid':cvalid};bp.append(b)
   if not reviewed and (not valid or not cvalid):problems.append({'language':lang,'symbol':k,'index':i,'reason':'unreviewed normal generic gate rejects','source':s,'target':t,'copied_English':api._normalized_block(s)==api._normalized_block(t)})
  records[k]={'current_source_context_block_pins':pins,'blocks':bp}
 result[lang]=records
(scratch/'api-audit.json').write_text(json.dumps({'scope':'Current exact rendered public two owning Plaque backend/preview modules, publicclass/member/APIvalue prose and actual enabled nested helpers. Subset only. Normal visibility/rendered/clean doc/class constructor/normal scoped reviewed loader, exact compiledsource/context/blockpins; lowlevelgeneric source/context gate diagnostics, NOT full api.audit. No globalcollection/model/tests.','sourcepins':symbols,'symbols':sorted(docs),'languages':result,'problems':problems},ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'symbols':len(docs),'problems':[{'lang':p['language'],'symbol':p['symbol'],'index':p.get('index'),'reason':p['reason'],'copiedEnglish':p.get('copied_English')} for p in problems]},ensure_ascii=False),flush=True)
