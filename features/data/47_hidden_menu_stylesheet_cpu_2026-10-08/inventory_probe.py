import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
root=Path('/mnt/wd4tb/spacr-worktrees/codex-serial-field-fade-20261008')
scratch=Path('/mnt/wd4tb/scratch/serial-field-fade-20261008')
mode=sys.argv[1]
source_root=root
if mode=='before':
    source_root=scratch/'inventory-baseline'
    source_root.mkdir(exist_ok=True)
    for path in (root/'spacr').rglob('*.py'):
        target=source_root/path.relative_to(root)
        target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists():
            if path==root/'spacr/qt/theme.py':
                target.write_bytes(subprocess.check_output(['git','show','21362b61978:spacr/qt/theme.py'],cwd=root))
            else: os.link(path,target)
    for name in ('packaging','resources'):
        target=source_root/name
        if not target.exists() and not target.is_symlink(): target.symlink_to(root/name,target_is_directory=True)
if mode=='before':
    for path in (root/'spacr/resources').rglob('*'):
        if not path.is_file(): continue
        target=source_root/path.relative_to(root)
        if not target.exists() and not target.is_symlink():
            target.parent.mkdir(parents=True,exist_ok=True)
            target.symlink_to(path)
sys.path.insert(0,str(source_root))
sys.path.insert(1,str(root/'tools'))
import spacr.qt.theme as theme
assert Path(theme.__file__).resolve()==source_root/'spacr/qt/theme.py'
import build_documentation_i18n as docs
import build_i18n_catalogs as ui
docs.ROOT=source_root
ui.ROOT=source_root
api=docs.public_docstrings()
runtime=ui.canonical_sources()
(scratch/(mode+'-inventory.json')).write_text(json.dumps({'api':api,'runtime':runtime},sort_keys=True))
print(json.dumps({'mode':mode,'theme_sha256':hashlib.sha256(Path(theme.__file__).read_bytes()).hexdigest(),'api_count':len(api),'runtime_counts':{k:len(v) for k,v in runtime.items()}},indent=2))
