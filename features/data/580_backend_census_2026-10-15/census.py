import ast,hashlib,json,subprocess,sys
from pathlib import Path
root=Path.cwd();out=Path('/mnt/wd4tb/scratch/home-b-f580-backend-census-20261015');sys.path.insert(0,str(root/'tools'))
import build_documentation_i18n as docs
import nested_helper_docs as nested
import api_visibility
base='0d814310629^';files=['spacr/intensity_rescale.py','spacr/measure.py','spacr/settings.py'];scopes={};bodies={};literals={};changed=[]
for label in ['before','after']:
 stage=out/label;current={};literal=set()
 for name in files:
  raw=subprocess.check_output(['git','show',base+':'+name]) if label=='before' else (root/name).read_bytes()
  p=stage/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw);tree=ast.parse(raw);module=name[:-3].replace('/','.')
  for node in docs._module_scope_nodes(tree.body):
   if not isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)):continue
   visible=not node.name.startswith('_') if isinstance(node,ast.ClassDef) else docs._is_visible_function_name(node.name,module_is_package=False)
   if visible and docs._is_rendered_autoapi_entry(p,module,module+'.'+node.name):current[module+'.'+node.name]=docs._clean_doc(node)
  for n in ast.walk(tree):
   if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='tr' and n.args and isinstance(n.args[0],ast.Constant) and isinstance(n.args[0].value,str):literal.add(n.args[0].value)
  if label=='after':
   for node in tree.body:
    if isinstance(node,ast.FunctionDef) and node.name in ['_calibration_input_error','_calibration_inputs','_external_reference_planes','_session_calibration_plan','_build_calibration_plan','_apply_calibration','_calibration_reference_hashes']:
     changed.append({'symbol':module+'.'+node.name,'public_symbol':api_visibility.public_symbol(module+'.'+node.name),'normal_top_level_visible':docs._is_visible_function_name(node.name,module_is_package=False)})
 inv=nested.inventory(stage,ignore_patterns=docs.AUTOAPI_IGNORE);mods={e.module for e in inv}&nested.ENABLED_MODULES
 scopes[label]={e.qualified_key:{'docstring':e.docstring,'signatures':e.signatures} for e in nested.entries(inv,modules=mods)};bodies[label]=current;literals[label]=literal
assert bodies['before']==bodies['after'];assert scopes['before']==scopes['after'];assert literals['before']==literals['after'];assert all(not r['public_symbol'] and not r['normal_top_level_visible'] for r in changed)
r={'source_owner':'0d81431062978ca6db5fb731d95bb97ed3b819e6','source_sha256':{n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in files},'normal_visible_public_bodies_unchanged':len(bodies['after']),'normal_enabled_nested_entries_unchanged':len(scopes['after']),'changed_private_top_level_definitions':changed,'backend_literal_tr_set_unchanged':sorted(literals['after']),'additional_runtime_sources':0,'additional_api_bodies':0,'scope':'Normal top-level visibility and nested helper policy over exact three backend files; GUI/settings18 source arrivals separately reviewed. No global collector/writer.'}
(out/'receipt.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
