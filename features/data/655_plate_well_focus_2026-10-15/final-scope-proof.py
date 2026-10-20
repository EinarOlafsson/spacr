import ast
import copy
from dataclasses import asdict
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tokenize
root=Path('/mnt/wd4tb/spacr-worktrees/codex-home-b-n180-final-help-20261015')
out=Path(__file__).resolve().parent
sys.path.insert(0,str(root/'tools'))
from api_visibility import public_symbol
from nested_helper_docs import inventory
path='spacr/qt/widgets/plate_map_picker.py'
base='db808733c0'
before=subprocess.check_output(['git','show',base+':'+path],cwd=root)
after=(root/path).read_bytes()
a,b=ast.parse(before),ast.parse(after)
old=next(n for n in a.body if isinstance(n,ast.ClassDef) and n.name=='_Well')
new=next(n for n in b.body if isinstance(n,ast.ClassDef) and n.name=='_Well')
old_methods={n.name:n for n in old.body if isinstance(n,ast.FunctionDef)}
new_methods={n.name:n for n in new.body if isinstance(n,ast.FunctionDef)}
added=sorted(new_methods.keys()-old_methods.keys())
assert added==['_cache_focus_style','changeEvent','focusInEvent','focusOutEvent','paintEvent']
for name,node in old_methods.items():
 assert ast.get_docstring(node)==ast.get_docstring(new_methods[name])
 if name!='__init__':assert ast.dump(node)==ast.dump(new_methods[name]),name
normal=copy.deepcopy(b)
for n in list(normal.body):
 if isinstance(n,ast.ImportFrom) and n.module=='PySide6.QtGui':normal.body.remove(n)
 elif isinstance(n,ast.ImportFrom) and n.module=='PySide6.QtCore':n.names=[v for v in n.names if v.name!='QEvent']
 elif isinstance(n,ast.ClassDef) and n.name=='_Well':
  n.body=[v for v in n.body if not isinstance(v,ast.FunctionDef) or v.name not in added]
  init=next(v for v in n.body if isinstance(v,ast.FunctionDef) and v.name=='__init__')
  init.body=[v for v in init.body if not (isinstance(v,ast.ImportFrom) and v.module=='figure_grid_view') and not (isinstance(v,ast.Assign) and isinstance(v.targets[0],ast.Attribute) and v.targets[0].attr=='_strong_focus_ring') and not (isinstance(v,ast.Expr) and isinstance(v.value,ast.Call) and isinstance(v.value.func,ast.Attribute) and v.value.func.attr=='_cache_focus_style')]
assert ast.dump(a)==ast.dump(normal)
comments=lambda raw:[t.string for t in tokenize.tokenize(io.BytesIO(raw).readline) if t.type==tokenize.COMMENT]
assert comments(before)==comments(after)
policy={'spacr.qt.widgets.plate_map_picker._Well.'+name:public_symbol('spacr.qt.widgets.plate_map_picker._Well.'+name) for name in added}
assert not any(policy.values())
results={}
for name,raw in [('before',before),('after',after)]:
 mini=out/('inventory-'+name);p=mini/path;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
 results[name]=inventory(mini,ignore_patterns=())
semantic = lambda rows: [{key:value for key,value in asdict(row).items() if key not in ('lineno','end_lineno')} for row in rows]
assert semantic(results['before'])==semantic(results['after'])
result={'base_commit':base,'source_sha256':hashlib.sha256(after).hexdigest(),'baseline_sha256':hashlib.sha256(before).hexdigest(),'precise_new_paint_cache_removal_restores_full_module_AST':True,'all_existing_docstrings_comments_mouse_selection_methods_unchanged':True,'actual_public_symbol_calls':policy,'scoped_helper_inventory_unchanged':True,'existing_helper_count':len(results['before']),'helper_location_change_only':[{ 'key':old.qualified_key, 'before':[old.lineno,old.end_lineno], 'after':[new.lineno,new.end_lineno]} for old,new in zip(results['before'],results['after'])],'scope':'Actual normal visibility and one-file helper inventory only; no full collector, target import or generation.'}
(out/'final-scope-proof.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
