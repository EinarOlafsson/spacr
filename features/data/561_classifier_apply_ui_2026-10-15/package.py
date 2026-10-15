from pathlib import Path
import ast, gzip, hashlib, io, json, shutil, subprocess, sys, tokenize
root=Path.cwd(); scratch=Path('/mnt/wd4tb/scratch/home-b-f561-classifier-apply-ui-20261015')
out=root/'features/data/561_classifier_apply_ui_2026-10-15'; out.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(root/'tools'))
import build_documentation_i18n as b
import nested_helper_docs as n
rel='spacr/qt/screens/embeddings.py'; module='spacr.qt.screens.embeddings'
base=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
before=subprocess.check_output(['git','show',f'{base}:{rel}'],text=True)
after=(root/rel).read_text()
sha=lambda s:hashlib.sha256(s.encode() if isinstance(s,str) else s).hexdigest()
def write(name,value): (out/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
def public(text):
 tree=ast.parse(text); result={}
 if b._clean_doc(tree): result[module]=b._clean_doc(tree)
 for node in b._module_scope_nodes(tree.body):
  if not isinstance(node,(ast.ClassDef,ast.FunctionDef,ast.AsyncFunctionDef)): continue
  visible=(not node.name.startswith('_') if isinstance(node,ast.ClassDef) else b._is_visible_function_name(node.name,module_is_package=False))
  key=f'{module}.{node.name}'
  if not visible or not b._is_rendered_autoapi_entry(root/rel,module,key): continue
  result[key]=b._autoapi_class_doc(node) if isinstance(node,ast.ClassDef) else b._clean_doc(node)
  if isinstance(node,ast.ClassDef):
   for child in b._module_scope_nodes(node.body):
    if isinstance(child,(ast.FunctionDef,ast.AsyncFunctionDef)) and b._is_visible_function_name(child.name,module_is_package=False):
     ck=f'{key}.{child.name}'
     if b._is_rendered_autoapi_entry(root/rel,module,ck): result[ck]=b._clean_doc(child)
 return {k:v for k,v in result.items() if v}
def helpers(text,label):
 snap=scratch/f'{label}-snapshot'; p=snap/rel; p.parent.mkdir(parents=True,exist_ok=True); p.write_text(text)
 definitions=n.inventory(snap,ignore_patterns=b.AUTOAPI_IGNORE)
 assert module in n.ENABLED_MODULES
 return {e.qualified_key:e.docstring for e in n.entries(definitions,modules=(module,))}
old_public,new_public=public(before),public(after)
old_helpers,new_helpers=helpers(before,'before'),helpers(after,'after')
old={**old_public,**old_helpers};new={**new_public,**new_helpers};changes=[]
for key,text in new.items():
 if old.get(key)==text: continue
 blocks,layout=b.translatable_blocks(text)
 changes.append({'symbol':key,'kind':'new_nested_helper' if key not in old and key in new_helpers else 'changed_public','before':old.get(key),'after':text,'after_sha256':sha(text),'blocks':blocks})
assert {r['symbol'] for r in changes if r['kind']=='changed_public'}=={f'{module}.EmbeddingsScreen.is_busy'}
assert sum(r['kind']=='new_nested_helper' for r in changes)==5
write('api-dependencies.json',{'base':base,'source_file':rel,'source_sha256':sha(after),'scope':'Normal direct visibility/render helpers and normal nested inventory/entries on exact one-file before/after snapshots; no full collector, runtime import, or generated catalogs.','normal_helpers':['build_documentation_i18n._is_visible_function_name','_is_rendered_autoapi_entry','_clean_doc','_autoapi_class_doc','translatable_blocks','nested_helper_docs.inventory','nested_helper_docs.entries(modules=(module,))'],'unchanged_public':sorted(k for k in new_public if old_public.get(k)==new_public[k]),'unchanged_nested':sorted(k for k in new_helpers if old_helpers.get(k)==new_helpers[k]),'removed':sorted(set(old)-set(new)),'changes':changes})
def literals(text):
 tree=ast.parse(text); parents={c:p for p in ast.walk(tree) for c in ast.iter_child_nodes(p)}; found={}
 for node in ast.walk(tree):
  if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='tr' and node.args and isinstance(node.args[0],ast.Constant) and isinstance(node.args[0].value,str):
   owner=node;names=[]
   while owner in parents:
    owner=parents[owner]
    if isinstance(owner,(ast.ClassDef,ast.FunctionDef,ast.AsyncFunctionDef)): names.append(owner.name)
   found.setdefault(node.args[0].value,[]).append({'line':node.lineno,'owner':'.'.join([module,*reversed(names)])})
 return found
oldtr,newtr=literals(before),literals(after)
write('runtime-message-dependencies.json',{'source_file':rel,'source_sha256':sha(after),'scope':'Literal tr inventory only. Canonical eligibility, global reuse, nine-language inputs and normal generation remain separate.','new_literals':[{'source':s,'source_sha256':sha(s),'occurrences':newtr[s]} for s in sorted(set(newtr)-set(oldtr))],'removed_literals':sorted(set(oldtr)-set(newtr)),'existing_literal_new_uses':[{'source':s,'occurrences':newtr[s]} for s in sorted(set(newtr)&set(oldtr)) if len(newtr[s])>len(oldtr[s])]})
comments=lambda t:[x.string for x in tokenize.generate_tokens(io.StringIO(t).readline) if x.type==tokenize.COMMENT]
assert comments(before)==comments(after)
logs={}
for name in ['before.log','before.time','progress-before.log','after.log','after.time','final.log','final.time','read-only-view.log','read-only-view.time']:
 data=(scratch/name).read_bytes(); (out/(name+'.gz')).write_bytes(gzip.compress(data,mtime=0)); logs[name]={'sha256':sha(data),'compressed_file':name+'.gz'}
write('receipt.json',{'base':base,'backend_source_commit':'03647f0827274824ab26c361259dc022a858d342','source_file':rel,'source_sha256':sha(after),'test_file':'tests/qt/test_own_dino_classifier_apply_controls.py','test_sha256':sha((root/'tests/qt/test_own_dino_classifier_apply_controls.py').read_bytes()),'source_scope':[rel],'test_scope':['tests/qt/test_own_dino_classifier_apply_controls.py'],'baseline':'2 FAIL in2.94s: absent saved-classifier control/method. Exact unchanged hosted slim-bar guard separately FAIL in7.78s naming both training bars.','intermediate':'12 PASS /4 FAIL in9.42s: numeric-only sort key passed string predictions; control restoration missed worker retirement after cancelled/prestart jobs. Corrected real caller/lifecycle behavior.','final':'16 PASS in8.42s:13 application/progress-label cases plus exact slim-bar guard and two existing real-worker training-progress tests. After read-only table/PlainText provenance refinement, the one actual matrix/view case PASS in2.46s; no broad replay.','distinct_cases':16,'cap':'4G; GPU hidden; Qt offscreen; thread limits1; shared heavy.lock','scope_claim':'Real Qt controls, JobRunner, actual CSV tabular export, raw-argument/row-order boundary and result/cancellation lifecycle; inert classifier adapter only. No checkpoint decode, scientific inference/training, network or downloads.','adapter_contract':'Full original raw NHWC object passed with CPU and batch_size only. Saved channel order/scale applied by backend exactly once. Results have dedicated views/export, never inserted into teacher picker or logistic/embedding route.','cancellation':'Delivery suppression only: backend has no cancelled/progress arguments. Active-only25ms GUI timer observes actual runner retirement, stops when idle/closed; overlapping training/application prevented. Close uses normal bounded JobRunner shutdown/parking.','progress_ci':'Both QProgressBars hide painted text; separate word-wrapped PlainText labels retain exact captions; bar format and accessibleDescription reflect same text. Numeric progress unchanged.','alpha':'All new literal ObjectNames inherit existing registered EmbeddingsDinoControls F561 alpha container.','previous_training_inputs':'e68a4d7 preserved unchanged; source-followup changes two is_busy blocks requiring exact18 review replacements. Other accepted training inputs/source docstrings remain unchanged.','comments_exact':True,'api_changed_bodies':len(changes),'api_changed_blocks':sum(len(x['blocks']) for x in changes),'new_tr_literals':len(set(newtr)-set(oldtr)),'logs':logs})
shutil.copyfile(scratch/'package.py',out/'package.py')
print(json.dumps({'api_bodies':len(changes),'api_blocks':sum(len(x['blocks']) for x in changes),'new_literal_tr':len(set(newtr)-set(oldtr)),'source_sha256':sha(after),'evidence':str(out)}))
