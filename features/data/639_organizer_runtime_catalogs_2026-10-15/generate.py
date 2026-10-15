"""Normal reviewed-only N639 runtime producer; execute only after owner release.

All nine locale preflights precede writes. API catalogs, source history and Help
are preservation-only. This script neither translates with models nor collects
API documentation. The published prerequisite must be merged before sealing
an executable plan; a pending plan deliberately refuses execution.
"""
from __future__ import annotations
import hashlib
import importlib
import json
import subprocess
import sys
import time
import types
from pathlib import Path

ROOT = Path.cwd().resolve()
PLAN_PATH = Path(sys.argv[1]).resolve()
PLAN_RAW = PLAN_PATH.read_bytes()
PLAN = json.loads(PLAN_RAW)
assert PLAN['state'] == 'authorized_baseline_ready', 'Pending foreign writer/public baseline; execution refused'
assert str(ROOT) == PLAN['repository']
OUT = Path(sys.argv[2]).resolve()
assert OUT.is_relative_to(Path('/mnt/wd4tb/scratch'))
OUT.mkdir(parents=True, exist_ok=False)
sys.path[:0] = [str(ROOT / 'tools'), str(ROOT)]
import build_i18n_catalogs as runtime
import i18n_caption_arrivals as arrivals
assert Path(runtime.__file__).resolve() == ROOT / 'tools/build_i18n_catalogs.py'
ORIGINAL_CANONICAL = runtime.canonical_sources
TABLES = {'SETTING_LABELS': 'setting_labels', 'SETTING_TOOLTIPS': 'setting_tooltips',
          'MODULE_SUMMARIES': 'module_summaries', 'CATEGORY_HELP': 'categories', 'UI': 'ui'}
STATS = {'status': 'started', 'prepared_locales': [], 'catalog_writes_started': False,
         'original_runtime_collections': 0, 'model_decode_requests': 0,
         'api_collection_calls': 0, 'help_generation_calls': 0}
START = time.monotonic()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_bytes())


def namespace(raw, label):
    result = types.ModuleType(label)
    exec(compile(raw, label, 'exec'), result.__dict__)
    return result


def refuse_decode(*args, **kwargs):
    STATS['model_decode_requests'] += 1
    raise RuntimeError('Unreviewed translation/model decoding requested')


def foreign(mapping, owned):
    return {key: value for key, value in mapping.items() if key not in owned}


def metadata(module, excluded):
    return {key: value for key, value in vars(module).items()
            if key.isupper() and key not in excluded}


def check_pins(outputs=False):
    assert PLAN_PATH.read_bytes() == PLAN_RAW, 'Plan changed'
    assert sha(Path(__file__).read_bytes()) == PLAN['driver_sha256'], 'Driver changed'
    for name, wanted in {**PLAN['input_pins'], **PLAN['preserved_artifacts']}.items():
        assert sha((ROOT / name).read_bytes()) == wanted, (name, 'source/review/tool/preserved artifact drift')
    if outputs:
        for name, wanted in PLAN['baseline_outputs'].items():
            assert sha((ROOT / name).read_bytes()) == wanted, (name, 'baseline drift')


def main():
    check_pins(outputs=True)
    subprocess.run(['git', 'merge-base', '--is-ancestor', PLAN['public_base'], 'HEAD'], check=True)
    assert set(PLAN['languages']) == set(runtime.MODEL_SPECS)
    owned_sources = PLAN['owned_sources']
    assert len(owned_sources) == len(set(owned_sources)) == 21
    owned_ids = {('UI', source) for source in owned_sources}
    for name, wanted in PLAN['source_files'].items():
        assert sha((ROOT / name).read_bytes()) == wanted
    runtime._translate_batches = refuse_decode
    runtime.reviewed_runtime_translations.cache_clear()
    STATS['original_runtime_collections'] += 1
    sources = ORIGINAL_CANONICAL()
    assert set(owned_sources) <= set(sources['ui']), 'Owned source absent from actual normal canonical inventory'
    runtime.canonical_sources = lambda: sources
    expected_hashes = runtime._source_hashes(sources)
    (OUT / 'canonical-snapshot.json').write_text(json.dumps(sources, ensure_ascii=False, indent=2) + '\n')
    old = {lang: namespace((runtime.CATALOG_DIR / (lang + '.py')).read_bytes(), 'before_' + lang)
           for lang in ['en', *PLAN['languages']]}
    old_arrivals = arrivals.load_arrival_contract()
    prepared, proofs = {}, {}
    for lang in PLAN['languages']:
        payload = read_json(ROOT / PLAN['review_files'][lang]['path'])
        assert payload['language'] == lang and payload['schema'] == 1
        records = payload['records']
        assert len(records) == 21
        assert all(row['table'] == 'ui' and row['source'] == row['key'] for row in records)
        wanted = {row['key']: row['translation'] for row in records}
        assert len(wanted) == 21 and set(wanted) == set(owned_sources)
        for row in records:
            assert row['source_sha256'] == sha(row['source'].encode())
        reviewed = runtime.reviewed_runtime_translations(lang)
        assert {source: reviewed[source] for source in owned_sources} == wanted
        assert not (set(old[lang].SOURCE_HASHES) - set(expected_hashes)), (lang, 'Foreign source removal')
        translations, unchanged = {}, {}
        for table, section in TABLES.items():
            current = sources[section]
            assert set(getattr(old[lang], table)) <= set(current), (lang, table, 'Foreign row removal')
            pairs = current.items() if isinstance(current, dict) else ((value, value) for value in current)
            for key, source in pairs:
                identity = table, str(key)
                if identity in owned_ids:
                    continue
                assert key in getattr(old[lang], table), (lang, identity, 'Foreign missing target')
                assert old[lang].SOURCE_HASHES.get(identity) == expected_hashes[identity], (lang, identity, 'Foreign stale source')
                target = getattr(old[lang], table)[key]
                assert source not in translations or translations[source] == target, (lang, source, 'Conflicting foreign reuse')
                translations[source] = target
                unchanged[identity] = target
        for source, target in wanted.items():
            assert source not in translations or translations[source] == target, (lang, source, 'Shared-source conflict')
            assert runtime._translation_candidate_valid(source, target, lang, force=runtime._looks_translatable(source))
            assert runtime._contextualize(target, lang, source) == target
            translations[source] = target
        prepared[lang] = translations, unchanged, wanted
        proofs[lang] = {'reviewed_owned_rows': 21, 'unchanged_runtime_rows': len(unchanged)}
        STATS['prepared_locales'].append(lang)
        print(lang + ': normal reviewed-source and foreign-preservation preflight passed', flush=True)
    assert len(prepared) == 9
    check_pins(outputs=True)
    STATS['catalog_writes_started'] = True
    runtime.write_english(sources)
    for lang, (translations, unchanged, wanted) in prepared.items():
        path = runtime.write_language(lang, sources, translations)
        after = namespace(path.read_bytes(), 'after_' + lang)
        assert after.SOURCE_HASHES == expected_hashes
        assert set(after.SOURCE_HASHES) == set(old[lang].SOURCE_HASHES) | owned_ids
        for source, target in wanted.items():
            assert after.UI[source] == target
        for (table, key), target in unchanged.items():
            assert getattr(after, table)[key] == target
            assert after.SOURCE_HASHES[(table, key)] == old[lang].SOURCE_HASHES[(table, key)]
        assert metadata(after, set(TABLES) | {'SOURCE_HASHES'}) == metadata(old[lang], set(TABLES) | {'SOURCE_HASHES'})
    english = namespace((runtime.CATALOG_DIR / 'en.py').read_bytes(), 'after_en')
    assert metadata(english, {'UI_SOURCES', 'SOURCE_HASHES'}) == metadata(old['en'], {'UI_SOURCES', 'SOURCE_HASHES'})
    assert english.UI_SOURCES == old['en'].UI_SOURCES | set(owned_sources)
    assert english.SOURCE_HASHES == expected_hashes
    assert foreign(english.SOURCE_HASHES, owned_ids) == foreign(old['en'].SOURCE_HASHES, owned_ids)
    assert runtime.audit(sources, PLAN['languages']) == 0
    STATS['original_runtime_collections'] += 1
    assert ORIGINAL_CANONICAL() == sources, 'Fresh original canonical inventory changed'
    check_pins()
    importlib.invalidate_caches()
    STATS['arrival_modules_refreshed'] = []
    for lang in PLAN['languages']:
        name = 'spacr.qt.i18n_catalogs.' + lang
        sys.modules.pop(name, None)
        fresh = importlib.import_module(name)
        for source, target in prepared[lang][2].items():
            assert fresh.UI[source] == target
        STATS['arrival_modules_refreshed'].append(name)
    arrival_counts = arrivals.refresh_arrival_contract()
    after_arrivals = arrivals.load_arrival_contract()
    assert after_arrivals['baseline'] == old_arrivals['baseline']
    for section in ('retired', 'changed'):
        assert after_arrivals[section] == old_arrivals[section], (section, 'Foreign arrival drift')
    omit_owned = lambda rows: [row for row in rows if (row['table'], row['key']) not in owned_ids]
    assert omit_owned(after_arrivals['added']) == omit_owned(old_arrivals['added'])
    assert foreign(after_arrivals, {'added', 'changed', 'retired'}) == foreign(old_arrivals, {'added', 'changed', 'retired'})
    added = {(row['table'], row['key']): row for row in after_arrivals['added']}
    assert owned_ids <= added.keys()
    check_pins()
    assert STATS['model_decode_requests'] == 0
    report = {'plan_sha256': sha(PLAN_RAW), 'baseline_public': PLAN['public_base'],
              'source_files': PLAN['source_files'], 'locales': proofs,
              'normal_runtime_audit': 0, 'normal_arrival_counts': arrival_counts,
              'foreign_targets_hashes_and_metadata_preserved': True,
              'api_history_and_help_byte_preserved': PLAN['preserved_artifacts'],
              'output_sha256': {name: sha((ROOT / name).read_bytes()) for name in PLAN['baseline_outputs']},
              'limitations': 'Runtime-only producer. Actual compiled-caption and organizer consumer execution is separate; no API/Help generation or scientific claim.'}
    (OUT / 'receipt.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')


try:
    main()
    STATS['status'] = 'pass'
except Exception as exc:
    STATS['status'] = 'failed'
    STATS['exception'] = repr(exc)
    raise
finally:
    STATS['elapsed_seconds'] = time.monotonic() - START
    (OUT / 'run.json').write_text(json.dumps(STATS, indent=2) + '\n')
    runtime.canonical_sources = ORIGINAL_CANONICAL
    runtime.reviewed_runtime_translations.cache_clear()
