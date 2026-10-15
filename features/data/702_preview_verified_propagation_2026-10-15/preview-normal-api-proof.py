from pathlib import Path
import ast,hashlib,json,sys,tempfile,subprocess
r=Path.cwd();s=Path('/mnt/wd4tb/scratch/n702-picker-reference-ui-20261015');sys.path.insert(0,str(r/'tools'));import build_i18n_catalogs as runtime
runtime.canonical_sources=lambda *a,**k:(_ for _ in ()).throw(RuntimeError('No fullcanonical collector'))
import build_documentation_i18n as api;import nested_helper_docs as nested
api.public_docstrings=lambda *a,**k:(_ for _ in ()).throw(RuntimeError('No fullAPI collector'))
paths=['spacr/qt/widgets/live_preview.py','spacr/qt/screens/app_screen.py','spacr/qt/preview_registry.py'];sha=lambda b:hashlib.sha256(b).hexdigest();texts={p:{'before':subprocess.check_output(['git','show','HEAD:'+p],text=True),'after':(r/p).read_text()} for p in paths}
def docs(p,text):
 path=r/p;module=p[:-3].replace('/','.');tree=ast.parse(text);out={module:api._clean_doc(tree)}
 for n in api._module_scope_nodes(tree.body):
  if not isinstance(n,(ast.ClassDef,ast.FunctionDef,ast.AsyncFunctionDef)):continue
  visible=not n.name.startswith('_') if isinstance(n,ast.ClassDef) else api._is_visible_function_name(n.name,module_is_package=False)
  key=module+'.'+n.name
  if not visible or not api._is_rendered_autoapi_entry(path,module,key):continue
  d=api._autoapi_class_doc(n) if isinstance(n,ast.ClassDef) else api._clean_doc(n)
  if d:out[key]=d
  if isinstance(n,ast.ClassDef):
   out.update({k:v for k,v in api._additional_assignment_docs(n.body,key).items() if api._is_rendered_autoapi_entry(path,module,k)})
   for child in api._module_scope_nodes(n.body):
    if isinstance(child,(ast.FunctionDef,ast.AsyncFunctionDef)) and api._is_visible_function_name(child.name,module_is_package=False):
     ck=key+'.'+child.name;cd=api._clean_doc(child)
     if cd and api._is_rendered_autoapi_entry(path,module,ck):out[ck]=cd
 out.update({k:v for k,v in api._additional_assignment_docs(tree.body,module).items() if api._is_rendered_autoapi_entry(path,module,k)})
 return out
public={}
for p,x in texts.items():
 before,after=docs(p,x['before']),docs(p,x['after']);assert before==after,p;public[p]={'normal_rendered_public_docs_unchanged':True,'count':len(after),'docs_sha256':{k:sha(v.encode()) for k,v in after.items()}}
versions={}
for version in ['before','after']:
 with tempfile.TemporaryDirectory(dir=s,prefix='normal-nested-'+version+'-') as d:
  snap=Path(d)
  for p,x in texts.items():q=snap/p;q.parent.mkdir(parents=True,exist_ok=True);q.write_text(x[version])
  inventory=nested.inventory(snap,ignore_patterns=api.AUTOAPI_IGNORE);modules={e.module for e in inventory if not e.ignored_by}&set(nested.ENABLED_MODULES);versions[version]={e.qualified_key:e.docstring for e in nested.entries(inventory,modules=modules)}
assert versions['before']==versions['after']

allowed={'spacr/qt/widgets/live_preview.py':{'LivePreviewPanel':{'__init__','propagate_settings','_choose_a_preview_model','apply_settings','_on_model_or_object_changed'}},'spacr/qt/screens/app_screen.py':{'AppScreen':{'_propagate_live_settings','_propagate_verified_live_settings'}},'spacr/qt/preview_registry.py':{'_PreviewHost':{'on_propagate'}}}
for p,classes in allowed.items():
 old,new=[ast.parse(texts[p][k]) for k in ['before','after']]
 for t in [old,new]:
  for clsname,methodnames in classes.items():
   c=next(n for n in t.body if isinstance(n,ast.ClassDef) and n.name==clsname)
   c.body=[n for n in c.body if not isinstance(n,ast.FunctionDef) or n.name not in methodnames]
  if p.endswith('app_screen.py'):
   t.body=[n for n in t.body if not isinstance(n,ast.FunctionDef) or n.name not in {'_build_live_preview_card','_fill_live_preview_card'}]
 assert ast.dump(old)==ast.dump(new),p
 def trs(t):
  return {n.args[0].value for n in ast.walk(t) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='tr' and n.args and isinstance(n.args[0],ast.Constant)}
 assert trs(ast.parse(texts[p]['before']))==trs(ast.parse(texts[p]['after']))
receipt={'scope':'Ordinary scoped rendered-public and nested inventory over three actual owning modules; no full canonical/API collector, model, writer or tests.','public_docs':public,'normal_nested_entries_unchanged':len(versions['after']),'API_newtargets':0,'runtime_newtargets':0,'source_boundary':'Only five existing LivePreviewPanel method bodies; two global actual LivePreview constructors; AppScreen._propagate_live_settings body and one documented new private method; _PreviewHost.on_propagate guard body. All existing public signatures/docs and tr literals unchanged.','source_sha256':{p:sha(v['after'].encode()) for p,v in texts.items()}}
(s/'preview-normal-api-proof.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps({'API_newtargets':0,'runtime_newtargets':0,'normal_nested_unchanged':len(versions['after']),'sources':receipt['source_sha256']}))
