from pathlib import Path
import hashlib
import json
import runpy
import sys

sys.path.insert(0, str(Path('tools/tutorials/authoring/tools').resolve()))
stage = Path('/media/carruthers/mnt3/codex/scratch/docs-completion-20261005/tutorial-installation-completion-r1')
freeze = json.loads((stage / 'frozen-narration-inputs.json').read_text())
def verify():
    for path, expected in freeze.items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected, path
verify()
sys.argv = ['render_all_voices.py', '--lessons', '1,3,4', '--languages', 'all', '--voices', 'all', '--threads', '2', '--device', 'cuda']
try:
    runpy.run_path('tools/tutorials/authoring/tools/render_all_voices.py', run_name='__main__')
except SystemExit as result:
    verify()
    raise
