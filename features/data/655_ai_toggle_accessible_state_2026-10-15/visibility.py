import ast,hashlib,io,json,subprocess,sys,tokenize
from pathlib import Path
repo=Path('/mnt/wd4tb/spacr-worktrees/codex-home-b-n180-final-help-20261015')
scratch=Path('/mnt/wd4tb/scratch/n655-ai-toggle-state-20261015')
sys.path.insert(0,str(repo/'tools'))
from api_visibility import public_symbol
from nested_helper_docs import inventory
path='spacr/qt/widgets/ai_toggle_label.py'
module='spacr.qt.widgets.ai_toggle_label'
before=subprocess.check_output(['git','show','b54a3b8ac:'+path],cwd=repo)
after=(repo/path).read_bytes()
a,b=ast.parse(before),ast.parse(after)
old=next(n for n in a.body if isinstance(n,ast.ClassDef) and n.name=='AiToggleLabel')
new=next(n for n in b.body if isinstance(n,ast.ClassDef) and n.name=='AiToggleLabel')
old_methods={n.name:n for n in old.body if isinstance(n,ast.FunctionDef)}
new_methods={n.name:n for n in new.body if isinstance(n,ast.FunctionDef)}
assert set(old_methods)==set(new_methods)
for name,node in old_methods.items():
 assert ast.get_docstring(node)==ast.get_docstring(new_methods[name]),name
 if name not in {'setChecked','mousePressEvent'}:
  assert ast.dump(node,include_attributes=False)==ast.dump(new_methods[name],include_attributes=False),name
assert ast.get_docstring(a)==ast.get_docstring(b)
assert ast.get_docstring(old)==ast.get_docstring(new)
comments=lambda raw:[t.string for t in tokenize.tokenize(io.BytesIO(raw).readline) if t.type==tokenize.COMMENT]
assert comments(before)==comments(after)
new_names=[]
for node in b.body:
 if isinstance(node,ast.ClassDef) and node.name=='_AccessibleAiToggle':
  new_names.append(module+'.'+node.name)
  new_names.extend(module+'.'+node.name+'.'+n.name for n in node.body if isinstance(n,ast.FunctionDef))
 elif isinstance(node,ast.FunctionDef) and node.name=='_ai_toggle_accessible_factory':new_names.append(module+'.'+node.name)
policy={name:public_symbol(name) for name in new_names}
assert not any(policy.values())
mini=scratch/'inventory-root'
owner=mini/path
owner.parent.mkdir(parents=True,exist_ok=True)
owner.write_bytes(after)
helpers=inventory(mini,ignore_patterns=())
assert helpers==()
sha=lambda raw:hashlib.sha256(raw).hexdigest()
result={'source_sha256':sha(after),'baseline_source_sha256':sha(before),'public_method_docs_unchanged':True,'module_and_class_docs_unchanged':True,'other_existing_methods_ast_unchanged':True,'changed_existing_methods':['setChecked','mousePressEvent'],'comments_unchanged':True,'actual_public_symbol_calls':policy,'actual_scoped_nested_inventory':[],'scope':'Normal policy on every new declaration and normal helper inventory on an isolated one-file source root; no full repository collector or generated output.'}
(scratch/'visibility.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
