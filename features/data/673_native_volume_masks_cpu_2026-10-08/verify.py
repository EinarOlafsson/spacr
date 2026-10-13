import argparse, gzip, hashlib, json, subprocess
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--git', action='store_true')
p.add_argument('--frozen', action='store_true')
args = p.parse_args()
root = Path(__file__).resolve().parent
repo = Path(subprocess.check_output(['git', 'rev-parse', '--show-toplevel'], cwd=root, text=True).strip())
relative = root.relative_to(repo)

def read(path):
    local = root / path
    if local.exists():
        return local.read_bytes()
    if args.git:
        return subprocess.check_output(['git', 'show', 'HEAD:' + str(relative / path)], cwd=repo)
    raise FileNotFoundError(local)

m = json.loads(read('manifest.json'))
r = json.loads(read('receipt.json'))
for item in m['payloads']:
    data = read(item['path'])
    assert len(data) == item['bytes'] and hashlib.sha256(data).hexdigest() == item['sha256'], item['path']
for path, digest in r['current_bindings'].items():
    frozen = gzip.decompress(read('source/after/' + path + '.gz'))
    assert hashlib.sha256(frozen).hexdigest() == digest, path
    if not args.frozen:
        current = subprocess.check_output(['git', 'show', 'HEAD:' + path], cwd=repo) if args.git else (repo / path).read_bytes()
        assert hashlib.sha256(current).hexdigest() == digest, path
v = json.loads(read('volume-visual-receipt.json'))
assert v['source_bytes_unchanged'] and v['exact_mask_geometry_roundtrip']
assert v['dialog_native_destroyed'] and v['worker_idle'] and v['top_level_widgets_after'] == 0
print('Verified', len(m['payloads']), 'payloads,', len(r['current_bindings']), 'frozen bindings;', 'current bindings checked' if not args.frozen else 'historical checkpoint only')
