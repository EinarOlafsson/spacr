from pathlib import Path
import importlib.util
import json
import os
import shlex
import subprocess
import sys
import tempfile

root = Path.cwd()
spec = importlib.util.spec_from_file_location('archive_fixture', root / 'tests/test_archive_package.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
results = []
with tempfile.TemporaryDirectory(prefix='archive paths with spaces ') as tmp:
    tmp = Path(tmp)
    source = m.screen.__wrapped__(tmp)
    for copied in (False, True):
        package = m.rep._write_archive_package(source, tmp / str(copied), m._form(source), copy_images=copied)
        text = (package / 'README.txt').read_text()
        command = next(line for line in text.splitlines() if line.startswith('python -c '))
        args = shlex.split(command)
        args[0] = sys.executable
        env = dict(os.environ, PYTHONPATH=str(root), CUDA_VISIBLE_DEVICES='', QT_QPA_PLATFORM='offscreen')
        checked = subprocess.run(args, cwd=package, env=env, text=True, capture_output=True, timeout=45)
        assert checked.returncode == 0, checked.stdout + checked.stderr
        assert 'Archive checks passed' in checked.stdout
        original = (package / 'archive_manifest.json').read_bytes()
        (package / 'archive_manifest.json').write_bytes(original + b' ')
        damaged = subprocess.run(args, cwd=package, env=env, text=True, capture_output=True, timeout=45)
        assert damaged.returncode != 0
        assert 'does not match' in damaged.stdout
        results.append({'copy_images':copied,'valid_exit':checked.returncode,'tampered_exit':damaged.returncode,'valid_stdout':checked.stdout.strip(),'tampered_stdout':damaged.stdout.strip()})
print(json.dumps({'results':results,'source':'private synthetic fixture only','network_writes':False}, indent=2))
