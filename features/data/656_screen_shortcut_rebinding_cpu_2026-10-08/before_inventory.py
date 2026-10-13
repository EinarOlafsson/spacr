import hashlib
import json
import sys
from pathlib import Path
baseline=Path('/mnt/wd4tb/scratch/656-screen-shortcuts-20261008/baseline')
root=Path('/mnt/wd4tb/spacr-worktrees/codex-screen-shortcuts-20261008')
sys.path.insert(0,str(root/'tools'))
import spacr.qt.shortcuts as shortcuts
assert Path(shortcuts.__file__).resolve() == baseline/'spacr/qt/shortcuts.py'
import build_i18n_catalogs as ui
ui.ROOT=baseline
result=ui.canonical_sources()
Path('/mnt/wd4tb/scratch/656-screen-shortcuts-20261008/before-normal-source-cache.json').write_text(json.dumps(result))
print(json.dumps({'baseline':'fb846a1d992','shortcuts_file':shortcuts.__file__, 'shortcuts_sha256':hashlib.sha256(Path(shortcuts.__file__).read_bytes()).hexdigest(), 'runtime_counts':{key:len(value) for key,value in result.items()}},indent=2))
