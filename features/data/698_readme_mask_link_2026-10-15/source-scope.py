import ast
import hashlib
import json
from pathlib import Path
import subprocess

root = Path.cwd()
base = 'd9e71d7be8f5f517048d744a1cc758d813434eb1'
files = ['packaging/generate_readme_visuals.py', 'tools/readme_i18n.py', 'tests/test_readme_presentation.py']
records = {}
for file in files:
    old_text = subprocess.check_output(['git', 'show', f'{base}:{file}'], text=True)
    text = (root / file).read_text()
    old, current = ast.parse(old_text), ast.parse(text)
    if file.startswith('packaging/'):
        owner = next(n for n in current.body if isinstance(n, ast.FunctionDef) and n.name == '_readme_model_zoo_block')
        assert isinstance(owner.body[-3], ast.Assign) and owner.body[-3].targets[0].id == 'site_language'
        assert isinstance(owner.body[-2], ast.AugAssign) and owner.body[-2].target.id == 'lines'
        del owner.body[-3:-1]
    elif file.startswith('tools/'):
        owner = next(n for n in current.body if isinstance(n, ast.AnnAssign) and getattr(n.target, 'id', '') == 'MODEL_ZOO_LABELS')
        for language in owner.value.values:
            index = next(i for i,k in enumerate(language.keys) if k.value == 'Mask generator')
            del language.keys[index]
            del language.values[index]
    else:
        current.body = [n for n in current.body if not (isinstance(n, ast.FunctionDef) and n.name == 'test_model_zoo_links_to_the_matching_site_mask_generator')]
    assert ast.dump(old, include_attributes=False) == ast.dump(current, include_attributes=False), file
    records[file] = {'before_sha256': hashlib.sha256(old_text.encode()).hexdigest(), 'after_sha256': hashlib.sha256(text.encode()).hexdigest(), 'unrelated_AST_exact': True}
Path('/mnt/wd4tb/scratch/home-b-n698-readme-link-20261015/source-scope.json').write_text(json.dumps({'baseline': base, 'status': 'PASS', 'files': records}, indent=2) + '\n')
print(json.dumps({'status': 'PASS', 'files': len(records)}))
