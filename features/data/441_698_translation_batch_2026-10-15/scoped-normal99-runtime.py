"""Normal N441 catalog closure; requires an independently frozen ready config."""
import hashlib
import importlib.util
import json
import subprocess
import sys
import types
from pathlib import Path

config_path = Path(sys.argv[1])
config = json.loads(config_path.read_text())
assert config['ready'] is True, 'Preparation only; final source approval required'
root = Path(config['root'])
assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip() == config['input_head']
assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=root), 'Input tree must be clean'
def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
def check_inputs():
    assert all(file_hash(root / name) == value for name, value in config['required_pins'].items()), 'Source/review/tool changed'
check_inputs()
assert all(file_hash(root / name) == value for name, value in config['generated_baseline_pins'].items()), 'Generated baseline differs before collector'
sys.path[:0] = [str(root / 'tools'), str(root)]
import build_i18n_catalogs as b
original_canonical = b.canonical_sources
sources = original_canonical()
expected = b._source_hashes(sources)
owned_sources = config['owned_sources']
assert len(owned_sources) == 11
assert all(source in sources['ui'] and hashlib.sha256(source.encode()).hexdigest() == digest
           for source, digest in owned_sources.items())
# Ordinary review loaders share a genuine current snapshot; no gate is replaced.
b.canonical_sources = lambda: sources
tables = {'SETTING_LABELS': 'setting_labels', 'SETTING_TOOLTIPS': 'setting_tooltips',
          'MODULE_SUMMARIES': 'module_summaries', 'CATEGORY_HELP': 'categories', 'UI': 'ui'}
prepared = {}
receipt = {'input_head': config['input_head'], 'config_sha256': file_hash(config_path),
           'driver_sha256': file_hash(Path(__file__)), 'locales': {}}
for language in b.MODEL_SPECS:
    owned = {}
    for filename in config['review_filenames']:
        document = json.loads((root / 'docs/i18n/reviewed/runtime' / language / filename).read_text())
        for row in document['records']:
            assert row['table'] == 'ui' and row['key'] == row['source']
            assert row['source_sha256'] == owned_sources[row['source']]
            assert row['source'] not in owned
            owned[row['source']] = row['translation']
    assert owned.keys() == owned_sources.keys()
    reviewed = b.reviewed_runtime_translations(language)
    assert all(reviewed[source] == target for source, target in owned.items())
    old = types.ModuleType('committed_' + language)
    baseline = subprocess.check_output(['git', 'show', config['input_head'] + ':spacr/qt/i18n_catalogs/' + language + '.py'], cwd=root)
    exec(compile(baseline, '<committed-catalog-' + language + '>', 'exec'), old.__dict__)
    translations, foreign, foreign_hashes = {}, {}, {}
    for table, section in tables.items():
        values = sources[section]
        pairs = values.items() if isinstance(values, dict) else ((value, value) for value in values)
        for key, source in pairs:
            identity = (table, str(key))
            if source in owned:
                continue
            assert key in getattr(old, table), (language, 'missing foreign target', identity)
            assert old.SOURCE_HASHES.get(identity) == expected[identity], (language, 'stale foreign source', identity)
            target = getattr(old, table)[key]
            assert source not in translations or translations[source] == target, (language, source)
            translations[source] = target
            foreign[identity] = target
            foreign_hashes[identity] = old.SOURCE_HASHES[identity]
        # Do not silently discard an unrelated committed row during normal generation.
        for key in getattr(old, table):
            identity = (table, str(key))
            if table == 'UI' and key in owned:
                continue
            assert identity in foreign, (language, 'foreign row would disappear', identity)
    for source, target in owned.items():
        assert b._translation_candidate_valid(source, target, language, force=False), (language, source)
    translations.update(owned)
    prepared[language] = (translations, foreign, foreign_hashes, owned)
    print(language + ': all ordinary review/source/current/foreign gates passed', flush=True)
check_inputs()
b.write_english(sources)
for language, (translations, foreign, foreign_hashes, owned) in prepared.items():
    path = b.write_language(language, sources, translations)
    spec = importlib.util.spec_from_file_location('new_' + language, path)
    new = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(new)
    assert all(getattr(new, table)[key] == target for (table, key), target in foreign.items())
    assert all(new.SOURCE_HASHES[identity] == digest for identity, digest in foreign_hashes.items())
    assert all(new.UI[source] == target for source, target in owned.items())
    receipt['locales'][language] = {'owned': len(owned), 'foreign_preserved': len(foreign)}
assert b.audit(sources, tuple(b.MODEL_SPECS)) == 0
receipt['normal_full_nine_runtime_audit_exit'] = 0
import i18n_caption_arrivals as arrivals
receipt['normal_arrivals'] = arrivals.refresh_arrival_contract()
assert original_canonical() == sources, 'Fresh original canonical collector differs'
check_inputs()
receipt['all_input_pins_unchanged'] = config['required_pins']
Path(config['receipt']).write_text(json.dumps(receipt, indent=2) + '\n')
print('N441 plus model-scale normal99 closure terminal PASS', flush=True)
