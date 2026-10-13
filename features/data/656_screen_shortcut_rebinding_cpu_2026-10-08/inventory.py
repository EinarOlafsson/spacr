import ast
import hashlib
import json
import subprocess
import sys
from pathlib import Path
root=Path('/mnt/wd4tb/spacr-worktrees/codex-screen-shortcuts-20261008')
scratch=Path('/mnt/wd4tb/scratch/656-screen-shortcuts-20261008')
sys.path.insert(0,str(root/'tools'))
import build_documentation_i18n as docs
import build_i18n_catalogs as ui
paths=['spacr/qt/shortcuts.py','spacr/qt/screens/annotate.py','spacr/qt/screens/make_masks.py','spacr/qt/widgets/qc_field_browser.py']
baseline=scratch/'baseline'; baseline.mkdir(exist_ok=True)
for relative in paths:
    path=baseline/relative; path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes(subprocess.check_output(['git','show','fb846a1d992:'+relative],cwd=root))
import os
for path in (root/'spacr').rglob('*.py'):
    relative=path.relative_to(root)
    target=baseline/relative
    if not target.exists():
        target.parent.mkdir(parents=True,exist_ok=True)
        os.link(path,target)
previous_root=docs.ROOT
docs.ROOT=baseline
before=docs.public_docstrings()
docs.ROOT=root
after=docs.public_docstrings()
prefixes=tuple(relative[:-3].replace('/','.') for relative in paths)
scoped={key:body for key,body in after.items() if key.startswith(prefixes)}
cache=scratch/'normal-source-cache.json'
canonical=json.loads(cache.read_text()) if cache.exists() else ui.canonical_sources()
cache.write_text(json.dumps(canonical))
from spacr.qt.i18n import _ROWS, _TERM_ROWS
normal_ui=set(canonical['ui']) | set(_ROWS) | set(_TERM_ROWS)
def tr_literals(path):
    found=set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='tr' and node.args and isinstance(node.args[0],ast.Constant) and isinstance(node.args[0].value,str):
            found.add(node.args[0].value)
    return found
added=set().union(*(tr_literals(root/relative)-tr_literals(baseline/relative) for relative in paths))
result={
    'baseline':'fb846a1d992',
    'source_sha256':{relative:hashlib.sha256((root/relative).read_bytes()).hexdigest() for relative in paths},
    'full_before_api_count':len(before),
    'full_after_api_count':len(after),
    'full_after_runtime_counts':{key:len(value) for key,value in canonical.items()},
    'scoped_api_added':sorted(set(scoped)-set(before)),
    'scoped_api_removed':sorted(key for key in before if key.startswith(prefixes) and key not in scoped),
    'scoped_api_changed':sorted(key for key in set(scoped)&set(before) if scoped[key]!=before[key]),
    'new_literal_tr_captions':sorted(added),
    'new_caption_membership':{caption:caption in normal_ui for caption in sorted(added)},
}
(scratch/'inventory.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
