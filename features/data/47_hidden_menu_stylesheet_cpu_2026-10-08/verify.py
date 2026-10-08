import argparse
import gzip
import hashlib
import json
import subprocess
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--git', action='store_true')
parser.add_argument('--current', action='store_true')
args = parser.parse_args()
root = Path(subprocess.check_output(['git', 'rev-parse', '--show-toplevel'], text=True).strip())
relative = 'features/data/47_hidden_menu_stylesheet_cpu_2026-10-08'
def read(path):
    if args.git:
        return subprocess.check_output(['git', 'show', 'HEAD:' + path], cwd=root)
    return (root / path).read_bytes()
manifest = json.loads(read(relative + '/manifest.json'))
for path, expected in manifest['payloads'].items():
    assert hashlib.sha256(read(relative + '/' + path)).hexdigest() == expected, path
receipt = json.loads(read(relative + '/receipt.json'))
if args.current:
    for path, expected in receipt['source_bindings'].items():
        assert hashlib.sha256(read(path)).hexdigest() == expected, path
before = json.loads(gzip.decompress(read(relative + '/before-inventory.json.gz')))
after = json.loads(gzip.decompress(read(relative + '/after-inventory.json.gz')))
assert before == after
print('PASS', len(manifest['payloads']), 'payload hashes;', len(receipt['source_bindings']), 'source bindings;', 'complete API/runtime equality')
