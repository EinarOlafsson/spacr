"""Run the normal F561 reviewed-only, identity-scoped runtime closure."""
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import types

root = Path(sys.argv[1]).resolve()
scratch = Path(sys.argv[2]).resolve()
scratch.mkdir(parents=True, exist_ok=True)
sys.path[:0] = [str(root / 'tools'), str(root)]
import build_i18n_catalogs as b

started = time.monotonic()
base = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
manifest = json.loads((scratch / 'sources.json').read_text())
source_names = list(manifest['source_files_sha256']) + [
    'tools/build_i18n_catalogs.py',
    'tools/i18n_caption_arrivals.py']
def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
source_pins = {name: digest(root / name) for name in source_names}
assert all(source_pins[name] == value for name, value in manifest['source_files_sha256'].items())
review_pins = {str(path.relative_to(root)): digest(path)
               for language in b.MODEL_SPECS
               for path in sorted((b.REVIEWED_RUNTIME_DIR / language).glob('*.json'))}

def refuse_decode(*args, **kwargs):
    raise AssertionError('Reviewed-only closure must not decode model translations')
b._translate_batches = refuse_decode
original_canonical = b.canonical_sources
canonical_started = time.monotonic()
sources = original_canonical()
canonical_seconds = time.monotonic() - canonical_started
expected = b._source_hashes(sources)
(scratch / 'canonical-snapshot.json.gz').write_bytes(gzip.compress(
    (json.dumps(sources, ensure_ascii=False, indent=2) + '\n').encode(), mtime=0))
b.canonical_sources = lambda: sources
sections = {'SETTING_LABELS': 'setting_labels', 'SETTING_TOOLTIPS': 'setting_tooltips',
            'MODULE_SUMMARIES': 'module_summaries', 'CATEGORY_HELP': 'categories', 'UI': 'ui'}
table_for = {value: key for key, value in sections.items()}
owned_sources = {(table_for[row['table']], row['key']): row['source'] for row in manifest['sources']}
assert len(owned_sources) == 1
assert {table: sum(ident[0] == table for ident in owned_sources) for table in table_for.values()} == {
    'SETTING_LABELS': 0, 'SETTING_TOOLTIPS': 0, 'MODULE_SUMMARIES': 0, 'CATEGORY_HELP': 0, 'UI': 1}

def pairs(section):
    value = sources[section]
    return value.items() if isinstance(value, dict) else ((item, item) for item in value)
current = {(table, str(key)): source for table, section in sections.items() for key, source in pairs(section)}
assert all(current.get(ident) == source for ident, source in owned_sources.items())

def old_catalog(language):
    module = types.ModuleType('baseline_' + language)
    text = subprocess.check_output(['git', 'show', base + ':spacr/qt/i18n_catalogs/' + language + '.py'], cwd=root, text=True)
    exec(compile(text, 'committed_' + language, 'exec'), module.__dict__)
    return module

def read_catalog(path):
    module = types.ModuleType('generated_catalog')
    exec(compile(path.read_text(), str(path), 'exec'), module.__dict__)
    return module

old_en = old_catalog('en')
removed_sources = {('UI', text) for text in manifest['possibly_removed_ui']}
removed_sources = removed_sources - set(expected)
assert all(ident in old_en.SOURCE_HASHES for ident in removed_sources)
receipt_removed = sorted(removed_sources)
assert set(old_en.SOURCE_HASHES) - set(expected) <= set(owned_sources) | removed_sources, 'Foreign English keys removed'
assert all(value == expected[ident] for ident, value in old_en.SOURCE_HASHES.items() if ident not in owned_sources and ident not in removed_sources), 'Foreign English source hash changed'
assert set(expected) - set(old_en.SOURCE_HASHES) <= set(owned_sources) | removed_sources, 'Foreign English arrivals'
prepared = {}
receipt = {'base': base, 'mode': 'Normal reviewed loader and writers, exact (table,key) ownership, full nine-locale audit',
           'owned_sources': manifest['sources'], 'source_sha256': source_pins,
           'active_review_sha256': review_pins, 'locales': {},
           'initial_canonical_seconds': canonical_seconds,
           'canonical_counts': {key: len(value) for key, value in sources.items()}, 'removed_ui_identities': receipt_removed}
print(json.dumps({'canonical_counts': receipt['canonical_counts'], 'seconds': canonical_seconds}), flush=True)
for language in b.MODEL_SPECS:
    filenames = [root / 'docs/i18n/reviewed/runtime' / language / ('2026-10-15-' + stem + '-home-b.json') for stem in ['f561-classifier-backend']]
    rows = [row for filename in filenames for row in json.loads(filename.read_text())['records']]
    assert len(rows) == 1
    owned = {}
    for row in rows:
        ident = (table_for[row['table']], row['key'])
        assert ident not in owned
        assert owned_sources[ident] == row['source']
        assert row['source_sha256'] == expected[ident]
        owned[ident] = row['translation']
    assert set(owned) == set(owned_sources)
    reviewed = b.reviewed_runtime_translations(language)
    assert all(reviewed[owned_sources[ident]] == target for ident, target in owned.items())
    old = old_catalog(language)
    assert set(old.SOURCE_HASHES) - set(expected) <= set(owned) | removed_sources, (language, 'Foreign keys removed')
    translations, untouched, missing = {}, {}, []
    for ident, source in current.items():
        table, key = ident
        if ident in owned:
            continue
        previous = getattr(old, table)
        if key not in previous or old.SOURCE_HASHES.get(ident) != expected[ident]:
            missing.append(ident)
            continue
        target = previous[key]
        assert source not in translations or translations[source] == target, (language, 'Conflicting foreign targets', source)
        translations[source] = target
        untouched[ident] = target
    assert not missing, (language, 'Foreign targets are absent or source-stale', missing)
    for ident, target in owned.items():
        source = owned_sources[ident]
        assert source not in translations or translations[source] == target, (language, 'Owned source collides with foreign identity', ident)
        assert b._translation_candidate_valid(source, target, language, force=False), (language, ident)
        translations[source] = target
    prepared[language] = (translations, untouched, owned, old)
    receipt['locales'][language] = {
        'owned_records': len(owned), 'new_owned_records': len(set(owned) - set(old.SOURCE_HASHES)),
        'existing_owned_records': len(set(owned) & set(old.SOURCE_HASHES)),
        'unrelated_preserved_records': len(untouched)}
    print(language + ': all normal reviews and exact identity/source preflight passed', flush=True)

assert all(digest(root / name) == value for name, value in source_pins.items())
assert all(digest(root / name) == value for name, value in review_pins.items())
b.write_english(sources)
new_en = read_catalog(b.CATALOG_DIR / 'en.py')
assert all(new_en.SOURCE_HASHES[ident] == value for ident, value in old_en.SOURCE_HASHES.items() if ident not in owned_sources and ident not in removed_sources)
for language, (translations, untouched, owned, old) in prepared.items():
    new = read_catalog(b.write_language(language, sources, translations))
    assert all(getattr(new, table)[key] == value for (table, key), value in untouched.items()), language
    assert all(getattr(new, table)[key] == value for (table, key), value in owned.items()), language
    assert all(new.SOURCE_HASHES[ident] == old.SOURCE_HASHES[ident] for ident in untouched), language
    assert new.SOURCE_HASHES == expected
    excluded = set(sections) | {'SOURCE_HASHES'}
    old_meta = {k: v for k, v in vars(old).items() if not k.startswith('__') and k not in excluded}
    new_meta = {k: v for k, v in vars(new).items() if not k.startswith('__') and k not in excluded}
    assert old_meta == new_meta, (language, 'Catalog metadata changed')
    print(language + ': 1 reviewed record exact; all foreign targets/hashes/metadata retained', flush=True)
result = b.audit(sources, list(b.MODEL_SPECS))
assert result == 0, result
fresh_started = time.monotonic()
assert original_canonical() == sources, 'Fresh original canonical result changed'
assert all(digest(root / name) == value for name, value in source_pins.items()), 'Source/tool changed'
assert all(digest(root / name) == value for name, value in review_pins.items()), 'Review input changed'
receipt.update({'full_nine_locale_audit': result, 'original_canonical_calls': 2,
                'fresh_canonical_seconds': time.monotonic() - fresh_started,
                'elapsed_seconds': time.monotonic() - started,
                'catalog_sha256': {language: digest(b.CATALOG_DIR / (language + '.py')) for language in ['en', *b.MODEL_SPECS]}})
(scratch / 'runtime-receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
print('PASS: 9 reviewed records, full nine-locale audit, fresh canonical equality and all unrelated preservation', flush=True)
