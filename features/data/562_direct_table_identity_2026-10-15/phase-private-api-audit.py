import sys,json,hashlib,subprocess
from pathlib import Path
root=Path('/mnt/wd4tb/spacr-worktrees/codex-n698-public-backend-integration-20261015');sys.path[:0]=[str(root/'tools'),str(root)]
import build_documentation_i18n as b
key='spacr.measure._phases_by_torch';docs=b.public_docstrings();english=json.loads((root/'docs/source/_static/i18n/api/en.json').read_text())['symbols']
r={'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),'normal_public_docstrings_contains_helper':key in docs,'committed_English_API_contains_helper':key in english,'normal_visibility_predicate':b._is_visible_function_name('_phases_by_torch',module_is_package=False),'normal_prefix_matches':[name for name in docs if name.startswith(key)],'tool_sha256':hashlib.sha256((root/'tools/build_documentation_i18n.py').read_bytes()).hexdigest(),'runtime_caption_not_used_as_API_proof':True}
Path('/mnt/wd4tb/scratch/f535-torch-api-source-audit-20261015/receipt.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,sort_keys=True))
