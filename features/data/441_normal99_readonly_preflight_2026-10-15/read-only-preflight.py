"""Validate actual N441 reviews and external N698 proposals without writing inputs."""
import hashlib
import json
import socket
import subprocess
import sys
from pathlib import Path

config_path = Path(sys.argv[1])
config = json.loads(config_path.read_text())
assert config['ready'] is True, 'Pending source/config review; execution disabled'
root = Path(config['root'])
assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip() == config['input_head']
assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=root)
def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
def verify_inputs():
    assert all(digest(root / name) == value for name, value in config['required_pins'].items())
    assert all(digest(root / name) == value for name, value in config['generated_unchanged'].items())
    assert all(digest(Path(name)) == value for name, value in config['proposal_pins'].items())
verify_inputs()
original_connect = socket.socket.connect
original_connect_ex = socket.socket.connect_ex
def connect(self, address):
    if self.family in (socket.AF_INET, socket.AF_INET6):
        raise RuntimeError('No network permitted in read-only preflight')
    return original_connect(self, address)
def connect_ex(self, address):
    if self.family in (socket.AF_INET, socket.AF_INET6):
        raise RuntimeError('No network permitted in read-only preflight')
    return original_connect_ex(self, address)
socket.socket.connect = connect
socket.socket.connect_ex = connect_ex
sys.path[:0] = [str(root / 'tools'), str(root)]
import build_i18n_catalogs as b
def refuse_write_or_model(*args, **kwargs):
    raise RuntimeError('No catalog writer or translation model permitted in read-only preflight')
b.write_english = refuse_write_or_model
b.write_language = refuse_write_or_model
b._translate_batches = refuse_write_or_model
original_canonical = b.canonical_sources
sources = original_canonical()
assert all(source in sources['ui'] and digest_value == hashlib.sha256(source.encode()).hexdigest()
           for source, digest_value in config['owned_sources'].items())
assert config['proposal_source'] in sources['ui'], 'Actual collector must own N698 literal'
b.canonical_sources = lambda: sources
receipt = {'input_head': config['input_head'], 'driver_sha256': digest(Path(__file__)),
           'config_sha256': digest(config_path), 'read_only': True, 'locales': {},
           'external_proposal_installed': False}
problems = []
expected_fields = {'table', 'key', 'source', 'source_sha256', 'translation'}
for language in b.MODEL_SPECS:
    result = {'installed_N441': 0, 'external_N698': 0}
    try:
        reviewed = b.reviewed_runtime_translations(language)
        document = json.loads((root / 'docs/i18n/reviewed/runtime' / language / config['review_filename']).read_text())
        assert document['schema'] == 1 and document['language'] == language
        assert len(document['records']) == 10
        assert {row['source'] for row in document['records']} == config['owned_sources'].keys()
        for row in document['records']:
            assert set(row) == expected_fields and row['table'] == 'ui' and row['key'] == row['source']
            assert row['source_sha256'] == config['owned_sources'][row['source']]
            assert reviewed[row['source']] == row['translation']
            assert b._contextualize(row['translation'], language, row['source']) == row['translation']
            assert not b._translation_rejection_reasons(row['source'], row['translation'], language,
                                                        force=b._looks_translatable(row['source']))
            result['installed_N441'] += 1
        proposal = json.loads((Path(config['proposal_folder']) / (language + '.json')).read_text())
        assert proposal['schema'] == 1 and proposal['language'] == language
        assert len(proposal['records']) == 1
        row = proposal['records'][0]
        assert set(row) == expected_fields
        assert row['table'] == 'ui' and row['key'] == row['source'] == config['proposal_source']
        assert row['source'] in sources['ui']
        assert row['source_sha256'] == hashlib.sha256(row['source'].encode()).hexdigest()
        assert b._contextualize(row['translation'], language, row['source']) == row['translation']
        reasons = b._translation_rejection_reasons(row['source'], row['translation'], language,
                                                   force=b._looks_translatable(row['source']))
        assert not reasons, sorted(reasons)
        assert row['source'] not in reviewed or reviewed[row['source']] == row['translation']
        result['external_N698'] = 1
    except Exception as exc:
        result['error'] = type(exc).__name__ + ': ' + str(exc)
        problems.append((language, result['error']))
    receipt['locales'][language] = result
    print(language + ': ' + json.dumps(result, ensure_ascii=False), flush=True)
assert original_canonical() == sources, 'Fresh actual collector differs'
verify_inputs()
receipt['fresh_actual_canonical_equal'] = True
receipt['all_input_and_generated_bytes_unchanged'] = True
receipt['problems'] = problems
Path(config['receipt']).write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
assert not problems, problems
assert sum(row['installed_N441'] for row in receipt['locales'].values()) == 90
assert sum(row['external_N698'] for row in receipt['locales'].values()) == 9
print('Read-only normal99 preflight terminal PASS; no files installed or generated', flush=True)
