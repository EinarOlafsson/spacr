"""Verify every archived payload, including sparse-checkout Git blob mode."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--git', action='store_true')
args = parser.parse_args()
folder = Path(__file__).resolve().parent
root = folder.parents[2]
prefix = folder.relative_to(root).as_posix()
def read(name):
    if args.git:
        return subprocess.check_output(['git', '-C', str(root), 'show', 'HEAD:' + prefix + '/' + name])
    return (folder / name).read_bytes()
manifest = json.loads(read('manifest.json'))
for name, item in manifest.items():
    data = read(name)
    assert len(data) == item['bytes'], name
    assert hashlib.sha256(data).hexdigest() == item['sha256'], name
print(f"Verified {len(manifest)} payloads")
