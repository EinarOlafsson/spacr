from pathlib import Path
import ast,hashlib,json,re,subprocess,sys,tempfile
r=Path.cwd();s=Path('/mnt/wd4tb/scratch/n197-current-contract-audit-20261015');old='8fc2aa5601847ccd8f7cff287c36c6a9ee00ba53';sha=lambda b:hashlib.sha256(b).hexdigest()
def nodes(text):
 out={}
 def walk(body,p=''):
  for n in body:
   if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)):out[p+n.name]=n;walk(n.body,p+n.name+'.')
 walk(ast.parse(text).body);return out
def nodoc(n):
 n=ast.parse(ast.unparse(n)).body[0]
 for x in ast.walk(n):
  if isinstance(x,(ast.ClassDef,ast.FunctionDef,ast.AsyncFunctionDef)) and x.body and isinstance(x.body[0],ast.Expr) and isinstance(x.body[0].value,ast.Constant) and isinstance(x.body[0].value.value,str):x.body.pop(0)
 return ast.dump(n)
p='spacr/qt/shortcuts.py';source=(r/p).read_bytes();a=nodes(source);b=nodes(subprocess.check_output(['git','show',old+':'+p]));wanted=['mapped','native','_screen_specs','_screen_label','install','show_cheat_sheet','ShortcutOverlay'];proof={}
for key in wanted:
 assert key in a,key
 exact=nodoc(a[key])==nodoc(b[key]);assert exact,key
 proof[key]={'nondoc_AST_sha256':sha(nodoc(a[key]).encode()),'nondoc_AST_identical_hosted':True}
test='tests/qt/test_the_hotkey_map_is_the_whole_map.py';raw=(r/test).read_bytes();assert raw==subprocess.check_output(['git','show',old+':'+test]);passes={};fails=[]
with Path('/mnt/wd4tb/scratch/ci-38699600639-current-triage-20261015/failed.log').open(errors='replace') as f:
 for line in f:
  if test not in line:continue
  m=re.search(r'(PASSED|FAILED) ('+re.escape(test)+r'::\S+)',line)
  if m:
   if m[1]=='PASSED':passes[m[2]]=passes.get(m[2],0)+1
   else:fails.append(m[2])
assert not fails
# Read only the exact existing Help action statements, preserving no unrelated app ownership.
appcur=ast.parse((r/'spacr/qt/app.py').read_text());appold=ast.parse(subprocess.check_output(['git','show',old+':spacr/qt/app.py']))
def action(tree):
 out=[]
 for n in ast.walk(tree):
  if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='act_keys' for t in n.targets):out.append(ast.dump(n))
  if isinstance(n,ast.Expr) and any(isinstance(x,ast.Name) and x.id=='act_keys' for x in ast.walk(n)):out.append(ast.dump(n))
 return out
assert action(appcur)==action(appold)
cp='spacr/qt/command_palette.py';c=nodes((r/cp).read_text());d=nodes(subprocess.check_output(['git','show',old+':'+cp]));palettekey=next(k for k in c if k.endswith('._open_shortcuts'));assert nodoc(c[palettekey])==nodoc(d[palettekey])
sys.path.insert(0,str(r/'tools'));import build_i18n_catalogs as normal;import build_documentation_i18n as api
langs=['sv','de','es','zh_CN','fr','pt','hi','ko','is'];sources={'Previous page','Next page'};locale={}
with tempfile.TemporaryDirectory(dir=s,prefix='scoped-normal-runtime-inputs-') as tmp:
 normal.REVIEWED_RUNTIME_DIR=Path(tmp);normal.canonical_sources=lambda:{'setting_labels':{},'setting_tooltips':{},'ui':sources}
 for lang in langs:
  review=r/f'docs/i18n/reviewed/runtime/{lang}/2026-10-15-n197-annotation-page-navigation.json';payload=json.loads(review.read_text());assert len(payload['records'])==2
  q=Path(tmp)/lang;q.mkdir();(q/'owned.json').write_bytes(review.read_bytes());normal.reviewed_runtime_translations.cache_clear();reviewed=normal.reviewed_runtime_translations(lang)
  tree=ast.parse((r/f'spacr/qt/i18n_catalogs/{lang}.py').read_text());tables={n.targets[0].id:ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id in {'UI','SOURCE_HASHES'}};checks=[]
  for text in sources:
   target=tables['UI'][text];assert reviewed[text]==target;assert tables['SOURCE_HASHES'][('UI',text)]==sha(text.encode());assert normal._translation_candidate_valid(text,target,lang,force=True);checks.append({'source':text,'target':target,'target_sha256':sha(target.encode())})
  locale[lang]={'checks':checks,'normal_scoped_review_and_candidate_PASS':True,'review_sha256':sha(review.read_bytes())}
mod='spacr.qt.shortcuts';docs={mod:api._clean_doc(ast.parse(source))}
for name in ['ShortcutSpec','mapped','native','show_cheat_sheet']:
 node=a[name];docs[mod+'.'+name]=api._autoapi_class_doc(node) if isinstance(node,ast.ClassDef) else api._clean_doc(node)
english=json.loads((r/'docs/source/_static/i18n/api/en.json').read_text())['symbols'];apiproof={}
for k,doc in docs.items():assert english[k]['text']==doc,k
for lang in langs:
 records=json.loads((r/f'docs/source/_static/i18n/api/{lang}.json').read_text())['symbols'];apiproof[lang]={}
 for k,doc in docs.items():
  rec=records[k];assert rec['source_sha256']==api._source_hash(doc) and rec['source_blocks_sha256']==api._source_block_hashes(doc) and rec['translation_source_blocks_sha256']==api._translation_source_block_hashes(doc),(lang,k)
  apiproof[lang][k]={'source_sha256':rec['source_sha256'],'source_context_block_pins_current':True,'target_sha256':sha(rec['text'].encode())}
receipt={'item':'N197','scope':'Exact originalHelp/map/scopes/NativeText/threeentrypoints plusactual pagecaption closure; readonly AST/history/ordinary scoped metadata. No tests, collectors, writers, models or source/item/tracker edits.','source_file':p,'source_sha256':sha(source),'hosted':old,'owning_nondoc_AST':proof,'actual_Help_action_AST_identical_hosted':True,'actual_palette_method_AST_identical_hosted':True,'original_test':{'file':test,'sha256':sha(raw),'byte_identical_hosted':True,'assertions':len([n for n in ast.walk(ast.parse(raw)) if isinstance(n,ast.Assert)]),'unique_hosted_PASS':len(passes),'PASS_nodes':passes,'FAIL_nodes':fails},'nine_page_caption_normal_review_compiled_checks':locale,'actual_rendered_API_scoped_pin_checks':apiproof,'earlier_current_two_node_contract_fix':'features/data/ci_shortcut_current_contracts_2026-10-15/receipt.json','required_actual_native_platform_run':False,'remaining_original_gap':'None proven; normal eighteenpage target closure is already present inactualcompiledcatalogs. All originalfunctional clauses source-bound toexistinghostedpassingchecks. BroaderglobalN678/nativeCI requirements are separate, not invented here.'};(s/'source-history-runtime-api-proof.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n');print(json.dumps({'owning_methods':len(proof),'original_unique_hosted_PASS':len(passes),'page_caption_checks':18,'API_symbols':len(docs),'languages':9,'remaining_original_gap':False}),flush=True)
