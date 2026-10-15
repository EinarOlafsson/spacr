from pathlib import Path
import ast,hashlib,inspect,json,subprocess
public=Path('/mnt/wd4tb/spacr-worktrees/codex-f562-coordinate-identity-20261015')
owner=Path('/mnt/wd4tb/spacr-worktrees/codex-n180-app-registration-20261015')
relative='spacr/qt/screens/app_screen.py'
old=subprocess.check_output(['git','show','633705c04e934b5eeec58c5c1db83d09431654f3:'+relative],cwd=owner)
new=(owner/relative).read_bytes()
before=ast.parse(old);after=ast.parse(new)
oldclass=next(node for node in before.body if isinstance(node,ast.ClassDef) and node.name=='AppScreen')
newclass=next(node for node in after.body if isinstance(node,ast.ClassDef) and node.name=='AppScreen')
def methods(node):return {child.name:child for child in node.body if isinstance(child,(ast.FunctionDef,ast.AsyncFunctionDef))}
a,b=methods(oldclass),methods(newclass)
added=set(b)-set(a);assert added=={'_workspace_database_control'}
assert ast.get_docstring(oldclass,clean=False)==ast.get_docstring(newclass,clean=False)
for name in a:
 assert ast.get_docstring(a[name],clean=False)==ast.get_docstring(b[name],clean=False)
 assert ast.dump(a[name].args)==ast.dump(b[name].args)
 assert (ast.dump(a[name].returns) if a[name].returns is not None else None)==(ast.dump(b[name].returns) if b[name].returns is not None else None)
helper=b['_workspace_database_control'];name=helper.name
visible=(not name.startswith('_')) or (name!='__init__' and name.startswith('__') and name.endswith('__'))
assert not visible
parents={child:node for node in ast.walk(after) for child in ast.iter_child_nodes(node)}
enclosing=parents.get(helper);ancestors=[]
while enclosing is not None:ancestors.append(type(enclosing).__name__);enclosing=parents.get(enclosing)
assert not any(kind in {'FunctionDef','AsyncFunctionDef'} for kind in ancestors)
manifest_path=public/'docs/source/_static/i18n/api/en.json';manifest=json.loads(manifest_path.read_text());prefix='spacr.qt.screens.app_screen.AppScreen'
assert manifest['schema']==2 and prefix+'._workspace_database_control' not in manifest['symbols']
def clean(node):return inspect.cleandoc(ast.get_docstring(node,clean=False) or '').strip()
class_source=clean(newclass)+'\n'+clean(b['__init__'])
checked={}
for key,source in [(prefix,class_source),(prefix+'.register_workspace',clean(b['register_workspace'])),(prefix+'.unregister_workspace',clean(b['unregister_workspace']))]:
 record=manifest['symbols'][key];digest=hashlib.sha256(source.encode()).hexdigest();assert record['text']==source and record['source_sha256']==digest;assert set(record)=={'source_sha256','source_blocks_sha256','text'};checked[key]={'source_sha256':digest,'exact_current_source_text_matches_english':True,'metadata_fields':sorted(record)}
paths=['tools/build_documentation_i18n.py','tools/nested_helper_docs.py','tools/api_visibility.py','docs/source/conf.py','docs/source/_static/i18n/api/en.json']
receipt={'reviewer':'hard_overnight independent scoped AST/JSON static peer; no app/generator imports or canonical inventory','decision':'No API translation/English-manifest regeneration is required by this exact source delta. New helper is excluded by both public and nested-helper policies; existing eligible prose/signatures unchanged.','owner_source_before_sha256':hashlib.sha256(old).hexdigest(),'owner_source_after_sha256':hashlib.sha256(new).hexdigest(),'owner_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=owner,text=True).strip(),'public_policy_pin':subprocess.check_output(['git','rev-parse','HEAD'],cwd=public,text=True).strip(),'policy_and_manifest_sha256':{path:hashlib.sha256((public/path).read_bytes()).hexdigest() for path in paths},'helper':{'symbol':prefix+'._workspace_database_control','docstring':clean(helper),'source_doc_sha256':hashlib.sha256(clean(helper).encode()).hexdigest(),'normal_public_function_visible':visible,'lexical_ancestor_kinds':ancestors,'nested_inventory_requires_enclosing_function':True,'helper_has_enclosing_function':False,'committed_english_symbol_absent':True},'existing_eligible_source_hash_evidence':checked,'exact_policy_evidence':{'public_docstrings':'tools/build_documentation_i18n.py:4239 child admission calls _is_visible_function_name before _merge_source_doc','visibility':'tools/build_documentation_i18n.py:4108 permits ordinary public names or documented __dunder__ except __init__; single underscore helper returns False','class_prose':'tools/build_documentation_i18n.py:4077 combines class/constructor prose only; arbitrary private method docs are not class prose','nested_helpers':'tools/nested_helper_docs.py:364-376 requires an enclosing FunctionDef/AsyncFunctionDef; direct AppScreen method has only ClassDef/Module ancestors','manifest':'tools/build_documentation_i18n.py:5503-5510 schema2 hashes only canonical text and prose blocks; no wholemodule/line/signature hash','hash':'tools/build_documentation_i18n.py:5034 hashes exact UTF8 canonical text'},'limits':['No normal API collector/generator/audit, runtime import, Sphinx build, Qt or acceptance cohort executed by this peer.','Does not claim whole current API manifest/global CI is clean; scoped current English matches class/register/unregister text and hash.','Runtime source safety/GUI tests and normal source publication remain owner/root responsibilities.']}
Path('/mnt/wd4tb/scratch/n180-app-registration-20261015/hard-api-eligibility-peer.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({'helper_public_visible':False,'nested_helper_eligible':False,'existing_api_sources_and_hashes_match':list(checked)}))
