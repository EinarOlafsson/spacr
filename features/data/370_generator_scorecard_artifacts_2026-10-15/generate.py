"""Call only the existing normal model-card and Model Zoo section producers."""
from pathlib import Path
import hashlib
import importlib.util
import json
import subprocess

root = Path('/mnt/wd4tb/spacr-worktrees/codex-home-b-f370-generator-20261015')
import sys
sys.path.insert(0, str(root))
from tools import build_model_cards
spec = importlib.util.spec_from_file_location('f370_visuals', root / 'packaging/generate_readme_visuals.py')
visuals = importlib.util.module_from_spec(spec)
spec.loader.exec_module(visuals)
paths = sorted((root / 'docs/models').rglob('*')) + [visuals.MODEL_ZOO_TABLE, visuals.MODEL_ZOO_SECTIONS]
before = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.is_file()}
changed = list(build_model_cards.write())
visuals.MODEL_ZOO_TABLE.write_text(visuals._model_zoo_table() + '\n', encoding='utf-8')
visuals.MODEL_ZOO_SECTIONS.write_text(visuals._model_zoo_sections(), encoding='utf-8')
after = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.is_file()}
assert set(before) == set(after)
receipt = {'source_commit': subprocess.check_output(['git','rev-parse','HEAD'], cwd=root, text=True).strip(),
           'normal_calls': ['tools.build_model_cards.write', 'packaging.generate_readme_visuals._model_zoo_table', 'packaging.generate_readme_visuals._model_zoo_sections'],
           'model_cards_write_result': changed,
           'before': before, 'after': after,
           'changed': [p for p in before if before[p] != after[p]]}
Path('/mnt/wd4tb/scratch/home-b-f370-generator-20261015/generated.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps({'changed': receipt['changed'], 'model_cards_write_result': changed}))
