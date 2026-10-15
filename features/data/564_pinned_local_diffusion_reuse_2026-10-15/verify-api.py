import ast,hashlib,json,pathlib,subprocess,sys
root=pathlib.Path('/mnt/wd4tb/spacr-worktrees/codex-f564-pretrained-reuse-20261015');sys.path.insert(0,str(root/'tools'));import nested_helper_docs as normal
out=pathlib.Path('/mnt/wd4tb/scratch/f564-pretrained-backbone-plan-20261015');pairs={};selected=normal.ENABLED_MODULES.intersection({'spacr.attribution','spacr.deep_spacr'})
tool=root/'tools/build_documentation_i18n.py';predicate=next(n for n in ast.parse(tool.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='_is_visible_function_name');ns={};exec(compile(ast.Module(body=[predicate],type_ignores=[]),str(tool),'exec'),ns);visible=ns[predicate.name]
public={}
for scope in ['before','after']:
 stage=out/('api-scoped-'+scope);(stage/'spacr').mkdir(parents=True,exist_ok=True);public[scope]={}
 for name in ['attribution.py','deep_spacr.py']:
  data=subprocess.check_output(['git','-C',str(root),'show','20a4313bcd9d58bc9267c1b80d3b51c02206635d:spacr/'+name]) if scope=='before' else (root/'spacr'/name).read_bytes()
  (stage/'spacr'/name).write_bytes(data);tree=ast.parse(data)
  public[scope][name]={'module':ast.get_docstring(tree),'entries':{n.name:ast.get_docstring(n) for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)) and visible(n.name,module_is_package=False)}}
 entries=normal.entries(normal.inventory(stage,ignore_patterns=()),modules=selected)
 pairs[scope]={e.qualified_key:{'signatures':e.signatures,'docstring':e.docstring} for e in entries}
assert pairs['before']==pairs['after'];assert public['before']==public['after']
receipt={'method':'Actual normal nested_helper_docs.inventory/entries on exact two-file snapshots with enabled-module subset, plus exact AST normal _is_visible_function_name public/module prose comparison. No whole collector or application import.','normal_nested_tool_sha256':hashlib.sha256((root/'tools/nested_helper_docs.py').read_bytes()).hexdigest(),'normal_visibility_tool_sha256':hashlib.sha256(tool.read_bytes()).hexdigest(),'modules_enabled':sorted(selected),'entry_count':len(pairs['after']),'nested_entries_equal':True,'public_module_docs_equal':True,'nested_entries':pairs['after'],'correction':'Earlier direct-top-level-only proof overlooked new nested fingerprint despite attribution being enabled; removed that nested arrival by documented top-level SHA helper, no policy exemption. Historical proof preserved as superseded.','source_sha256':{n:hashlib.sha256((root/'spacr'/n).read_bytes()).hexdigest() for n in ['attribution.py','deep_spacr.py']}}
(out/'corrected-nested-api-inventory-proof.json').write_text(json.dumps(receipt,indent=2)+'\n');print('PASS',len(pairs['after']))
