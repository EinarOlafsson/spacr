from pathlib import Path
import ast, gzip, hashlib, io, json, shutil, subprocess, sys, tokenize
root=Path.cwd(); scratch=Path('/mnt/wd4tb/scratch/home-b-f561-classifier-ui-20261015')
out=root/'features/data/561_classifier_ui_2026-10-15'; out.mkdir(parents=True,exist_ok=True)
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
    entries=n.entries(definitions,modules=(module,))
    return {e.qualified_key:e.docstring for e in entries}
old_public,new_public=public(before),public(after)
old_helpers,new_helpers=helpers(before,'before'),helpers(after,'after')
old={**old_public,**old_helpers}; new={**new_public,**new_helpers}
changes=[]
for key,text in new.items():
    if old.get(key)==text: continue
    blocks,layout=b.translatable_blocks(text)
    changes.append({'symbol':key,'kind':'new_nested_helper' if key not in old and key in new_helpers else 'changed_public','before':old.get(key),'after':text,'after_sha256':sha(text),'blocks':blocks})
assert {r['symbol'] for r in changes if r['kind']=='changed_public'}=={f'{module}.EmbeddingsScreen.is_busy',f'{module}.EmbeddingsScreen.closeEvent'}
assert sum(r['kind']=='new_nested_helper' for r in changes)==6
write('api-dependencies.json',{'base':base,'source_file':rel,'source_sha256':sha(after),'scope':'Normal direct visibility/render helpers and normal nested inventory/entries on exact one-file before/after snapshots; no full collector, runtime import, or generated catalogs.','normal_helpers':['build_documentation_i18n._is_visible_function_name','_is_rendered_autoapi_entry','_clean_doc','_autoapi_class_doc','translatable_blocks','nested_helper_docs.inventory','nested_helper_docs.entries(modules=(module,))'],'unchanged_public':sorted(k for k in new_public if old_public.get(k)==new_public[k]),'unchanged_nested':sorted(k for k in new_helpers if old_helpers.get(k)==new_helpers[k]),'removed':sorted(set(old)-set(new)),'changes':changes})
def literals(text):
    tree=ast.parse(text); parents={c:p for p in ast.walk(tree) for c in ast.iter_child_nodes(p)}; found={}
    for node in ast.walk(tree):
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='tr' and node.args and isinstance(node.args[0],ast.Constant) and isinstance(node.args[0].value,str):
            owner=node; names=[]
            while owner in parents:
                owner=parents[owner]
                if isinstance(owner,(ast.ClassDef,ast.FunctionDef,ast.AsyncFunctionDef)): names.append(owner.name)
            found.setdefault(node.args[0].value,[]).append({'line':node.lineno,'owner':'.'.join([module,*reversed(names)])})
    return found
oldtr,newtr=literals(before),literals(after)
write('runtime-message-dependencies.json',{'source_file':rel,'source_sha256':sha(after),'scope':'Literal tr inventory only. Canonical eligibility, global reuse, nine-language inputs and normal generation remain separate owner work.','new_literals':[{'source':s,'source_sha256':sha(s),'occurrences':newtr[s]} for s in sorted(set(newtr)-set(oldtr))],'removed_literals':sorted(set(oldtr)-set(newtr)),'existing_literal_new_uses':[{'source':s,'occurrences':newtr[s]} for s in sorted(set(newtr)&set(oldtr)) if len(newtr[s])>len(oldtr[s])]})
def functions(text):
    result={}
    def visit(node,parents):
        for child in ast.iter_child_nodes(node):
            names=parents
            if isinstance(child,(ast.ClassDef,ast.FunctionDef,ast.AsyncFunctionDef)):
                names=(*parents,child.name)
                if not isinstance(child,ast.ClassDef): result['.'.join(names)]=ast.dump(child,include_attributes=False)
            visit(child,names)
    visit(ast.parse(text),()); return result
oldf,newf=functions(before),functions(after)
unchanged=['EmbeddingsScreen._use_embeddings','EmbeddingsScreen._on_dino_progress','EmbeddingsScreen._on_dino_failed','EmbeddingsScreen._on_dino_finished','EmbeddingsScreen._show_dino_result']
unchanged=[k for k in unchanged if k in oldf]
assert all(oldf[k]==newf[k] for k in unchanged)
comments=lambda t:[x.string for x in tokenize.generate_tokens(io.StringIO(t).readline) if x.type==tokenize.COMMENT]
assert comments(before)==comments(after)
logs={}
for name in ['before.log','before.time','after.log','after.time','dialog-after.log','dialog-after.time','identity-before.log','identity-before.time','identity-after.log','identity-after.time']:
    data=(scratch/name).read_bytes(); (out/(name+'.gz')).write_bytes(gzip.compress(data,mtime=0)); logs[name]={'sha256':sha(data),'compressed_file':name+'.gz'}
shutil.copyfile(scratch/'identity-before-source.py.gz',out/'identity-before-source.py.gz')
write('receipt.json',{'base':base,'backend_source_commit':'9f46e93d5a839214bc3e203cb3084132f7dd5c1a','source_file':rel,'source_sha256':sha(after),'test_file':'tests/qt/test_own_dino_classifier_controls.py','test_sha256':sha((root/'tests/qt/test_own_dino_classifier_controls.py').read_bytes()),'source_scope':[rel],'new_test_scope':['tests/qt/test_own_dino_classifier_controls.py'],'baseline':'3 FAIL, absent classifier controls/method, 2.64s','intermediate':'36 PASS and 1 test-fixture enum error in 6.45s; actual Qt QDialog.DialogCode.Accepted correction only; that one actual-dialog test then PASS in 2.61s','identity_baseline':'1 actual FAIL in 2.68s: pandas inferred label 01 as 1 and acquisition group 001 as 1','final':'23 classifier cases PASS in 3.92s after reader-only explicit string identity fix; existing 15 pretraining/alpha cases passed the earlier owning cohort and were not rerun after the unrelated reader-only fix','distinct_cases':38,'final_wall_seconds':7.0,'final_peak_rss_kib':852944,'cap':'4G; GPU hidden; Qt offscreen; thread limits1; shared heavy.lock','training_claim':'No real training, checkpoint decode, model downloads or scientific validation. Inert trainer boundary with real Qt controls/JobRunner/signals/cancellation.','consumer_dependency':'Saved classifier consumption is backend-owned and pending. Result is not inserted into the teacher Foundation picker. This is the explicit training/settings/background-lifecycle scope, not full F561 completion.','alpha':'New literal ObjectNames inherit existing EmbeddingsDinoControls registered F561 container; no new settings keys or registry entries.','unchanged_existing_functions':unchanged,'comments_exact':True,'source_changed_after_final_behavior_run':'Only grammar in existing is_busy doc: its final result -> their final results. Executable behavior is identical to final23-pass source.','api_changed_bodies':len(changes),'api_changed_blocks':sum(len(x['blocks']) for x in changes),'new_tr_literals':len(set(newtr)-set(oldtr)),'logs':logs})
shutil.copyfile(scratch/'package.py',out/'package.py')
print(json.dumps({'api_bodies':len(changes),'api_blocks':sum(len(x['blocks']) for x in changes),'new_literal_tr':len(set(newtr)-set(oldtr)),'source_sha256':sha(after),'evidence':str(out)}))
