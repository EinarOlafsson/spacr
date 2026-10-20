import hashlib
import json
import pathlib
import sys

root = pathlib.Path('/mnt/wd4tb/spacr-worktrees/codex-n702-picker-pin-provenance-20261015')
sys.path.insert(0, str(root / 'tools'))
class DenyPackage:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'spacr' or fullname.startswith('spacr.'):
            raise RuntimeError('No package import permitted in proposal validator: ' + fullname)
sys.meta_path.insert(0, DenyPackage())
import build_i18n_catalogs as b
def refuse_capture(*args, **kwargs):
    raise RuntimeError('No canonical collector/model allowed for owner proposal')
b.canonical_sources = refuse_capture
b._translate_batches = refuse_capture
folder = pathlib.Path(__file__).parent
manifest = json.loads((folder / 'manifest.json').read_text())
results = {}
for language, digest in manifest['draft_sha256'].items():
    path = folder / (language + '.json')
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    row = json.loads(path.read_text())['records'][0]
    assert row['source_sha256'] == hashlib.sha256(row['source'].encode()).hexdigest()
    reasons = sorted(b._translation_rejection_reasons(row['source'], row['translation'], language, force=False))
    results[language] = {'ordinary_candidate_valid': not reasons, 'reasons': reasons}
assert not any(name == 'spacr' or name.startswith('spacr.') for name in sys.modules)
(folder / 'candidate-validation.json').write_text(json.dumps({'results': results, 'no_collector_or_model': True, 'no_package_import': True}, indent=2) + '\n')
print(json.dumps(results, ensure_ascii=False), flush=True)
assert all(item['ordinary_candidate_valid'] for item in results.values())
