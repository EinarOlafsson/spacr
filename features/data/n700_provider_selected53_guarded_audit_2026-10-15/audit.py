import gzip
import hashlib
import importlib.abc
import json
import pathlib
import subprocess
import sys
import traceback

base = pathlib.Path(__file__).resolve().parent
config = json.loads((base / 'config.json').read_text())
root = pathlib.Path(config['root'])
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip() == config['head']
assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=root, text=True)
for name, digest in config['pins'].items():
    assert sha(root / name) == digest, name
snapshot = pathlib.Path(config['source_snapshot'])
assert sha(snapshot) == config['source_snapshot_sha256']
all_docs = json.loads(gzip.decompress(snapshot.read_bytes()))
docs = {key: all_docs[key] for key in config['selected_symbols']}
assert len(docs) == 53
assert set(docs) == {key for key in all_docs if key == 'spacr.qt.ai.providers' or key.startswith('spacr.qt.ai.providers.')}

class RefuseModels(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'transformers', 'sentencepiece', 'huggingface_hub', 'cellpose', 'tensorflow'}:
            raise RuntimeError('Forbidden model/provider import: ' + fullname)
sys.meta_path.insert(0, RefuseModels())
def network_guard(event, args):
    if event in {'socket.connect', 'socket.getaddrinfo'}:
        raise RuntimeError('Forbidden network event: ' + event)
sys.addaudithook(network_guard)
sys.path.insert(0, str(root / 'tools'))
import build_documentation_i18n as tool
import build_i18n_catalogs as runtime
def refuse_collection(*args, **kwargs):
    raise RuntimeError('Forbidden full canonical collection; no synthetic context')
runtime.canonical_sources = refuse_collection
tool._translate_blocks = refuse_collection
tool._translate_batches = refuse_collection
actual_api = tool.API_DIR
projection = base / 'projected-api'
projection.mkdir(exist_ok=False)
proof = {}
canonical = lambda value: json.dumps(value, ensure_ascii=False, sort_keys=True).encode('utf-8')
for language in ['en', *tool.MODEL_SPECS]:
    actual_path = actual_api / (language + '.json')
    full = json.loads(actual_path.read_text())
    projected = dict(full)
    projected['symbols'] = {key: full['symbols'][key] for key in docs}
    assert {key:value for key,value in projected.items() if key != 'symbols'} == {key:value for key,value in full.items() if key != 'symbols'}
    record_hashes = {key: hashlib.sha256(canonical(value)).hexdigest() for key,value in projected['symbols'].items()}
    path = projection / (language + '.json')
    path.write_text(json.dumps(projected, ensure_ascii=False, indent=2, sort_keys=True) + '\n')
    readback = json.loads(path.read_text())
    assert set(readback['symbols']) == set(docs)
    assert all(canonical(readback['symbols'][key]) == canonical(full['symbols'][key]) for key in docs)
    assert canonical({key:value for key,value in readback.items() if key != 'symbols'}) == canonical({key:value for key,value in full.items() if key != 'symbols'})
    proof[language] = {'actual_sha256': sha(actual_path), 'projection_sha256': sha(path), 'records': record_hashes, 'all_records_and_headers_exact': True}
result = {'scope': config['scope'], 'head': config['head'], 'symbols': list(docs), 'projection': proof, 'source_snapshot_sha256': config['source_snapshot_sha256'], 'original_full_audit': 'REFUSED stale SV spacr.qt.widgets.glass.glass#2; retained separately; no global pass claim', 'status': 'RUNNING'}
try:
    tool.API_DIR = projection
    assert pathlib.Path(tool.ROOT).resolve() == root
    result['ordinary_audit_exit'] = tool.audit(docs, list(tool.MODEL_SPECS))
    result['status'] = 'PASS' if result['ordinary_audit_exit'] == 0 else 'REFUSED'
except Exception as exc:
    result['status'] = 'REFUSED'
    result['exception'] = {'type': type(exc).__name__, 'message': str(exc), 'traceback': traceback.format_exc()}
finally:
    tool.API_DIR = actual_api
    changed = [name for name,digest in config['pins'].items() if sha(root / name) != digest]
    result['changed_actual_inputs'] = changed
    result['all_actual_input_pins_unchanged'] = not changed
    result['pin_count'] = len(config['pins'])
    if changed:
        result['status'] = 'REFUSED'
    (base / 'receipt.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({key:value for key,value in result.items() if key != 'projection'}, ensure_ascii=False, indent=2), flush=True)
raise SystemExit(0 if result['status'] == 'PASS' else 1)
