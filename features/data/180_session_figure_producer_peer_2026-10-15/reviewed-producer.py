"""Authorized execution only: normal N180 API/runtime closure, then Help.

Run from the immutable, merged private repository with plan.json and an empty
scratch output directory. No production source, review input or validator is
modified. All nine API/runtime preflights precede the first catalog write.
"""
import ast
import hashlib
import importlib
import json
import subprocess
import sys
import time
import types
from pathlib import Path

ROOT = Path.cwd()
PLAN_PATH = Path(sys.argv[1]).resolve()
OUT = Path(sys.argv[2]).resolve()
OUT.mkdir(parents=True, exist_ok=False)
PLAN_RAW = PLAN_PATH.read_bytes()
PLAN = json.loads(PLAN_RAW)
sys.path[:0] = [str(ROOT / 'tools'), str(ROOT)]
import build_documentation_i18n as api
import build_i18n_catalogs as runtime
import i18n_caption_arrivals as arrivals

STATS = {'status': 'started', 'api_prepared': [], 'runtime_prepared': [],
         'catalog_writes_started': False, 'model_decode_requests': 0,
         'full_api_collection_requests': 0, 'original_runtime_collections': 0}
START = time.monotonic()
ORIGINAL_CANONICAL = runtime.canonical_sources
TABLES = {'SETTING_LABELS': 'setting_labels', 'SETTING_TOOLTIPS': 'setting_tooltips',
          'MODULE_SUMMARIES': 'module_summaries', 'CATEGORY_HELP': 'categories', 'UI': 'ui'}


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


def refuse_full_api(*args, **kwargs):
    STATS['full_api_collection_requests'] += 1
    raise RuntimeError('Full API source collection is outside this selected closure')


def check_pins(include_outputs=False):
    assert PLAN_PATH.read_bytes() == PLAN_RAW, 'Plan changed'
    assert sha(Path(__file__).read_bytes()) == PLAN['driver_sha256'], 'Driver changed'
    for name, wanted in PLAN['input_pins'].items():
        assert sha(Path(name).read_bytes()) == wanted, (name, 'source/review/tool drift')
    if include_outputs:
        for name, wanted in PLAN['baseline_outputs'].items():
            assert sha(Path(name).read_bytes()) == wanted, (name, 'baseline/output drift')


def foreign(payload, owned):
    return {key: value for key, value in payload.items() if key not in owned}


def catalog_metadata(module):
    names = set(TABLES) | {'SOURCE_HASHES'}
    return {key: value for key, value in vars(module).items()
            if key.isupper() and key not in names}


def main():
    check_pins(include_outputs=True)
    subprocess.run(['git', 'merge-base', '--is-ancestor', PLAN['public_base'], 'HEAD'], check=True)
    selected = PLAN['source_docs']
    keys = set(selected)
    assert len(keys) == 2 and len(PLAN['changed_blocks']) == 7
    assert set(keys) == set(PLAN['new_symbols'])
    assert set(api.MODEL_SPECS) == set(runtime.MODEL_SPECS) == set(PLAN['languages'])
    caption = PLAN['runtime_source']
    for name, wanted in PLAN['source_files'].items():
        assert sha(Path(name).read_bytes()) == wanted
    tree = ast.parse(Path('spacr/qt/widgets/figure_queue.py').read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'FigureQueue')
    for symbol, expected in selected.items():
        name = symbol.rsplit('.', 1)[1]
        method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == name)
        assert api._is_visible_function_name(name, module_is_package=False)
        assert api._is_rendered_autoapi_entry(Path('spacr/qt/widgets/figure_queue.py').resolve(),
                                             'spacr.qt.widgets.figure_queue', symbol)
        assert api._clean_doc(method) == expected
    api.public_docstrings = refuse_full_api
    api._translate_blocks = refuse_decode
    runtime._translate_batches = refuse_decode
    STATS['original_runtime_collections'] += 1
    sources = ORIGINAL_CANONICAL()
    assert caption in sources['ui'], 'Owned caption not in actual normal source inventory'
    runtime.canonical_sources = lambda: sources
    expected_hashes = runtime._source_hashes(sources)
    (OUT / 'canonical-snapshot.json').write_text(json.dumps(sources, ensure_ascii=False, indent=2) + '\n')
    (OUT / 'before-help_api_index.py').write_bytes(Path('spacr/qt/help_api_index.py').read_bytes())
    old_api = {lang: read_json(api.API_DIR / (lang + '.json')) for lang in ['en', *PLAN['languages']]}
    assert all(not keys.intersection(payload['symbols']) for payload in old_api.values()), 'Expected new API methods already exist'
    all_docs = {key: row['text'] for key, row in old_api['en']['symbols'].items()}
    all_docs.update(selected)
    history_path = Path('docs/i18n/api_source_history.json')
    old_history = read_json(history_path)
    old_arrivals = arrivals.load_arrival_contract()
    prepared_api, prepared_runtime, proofs = {}, {}, {}
    old_runtime = {lang: namespace((runtime.CATALOG_DIR / (lang + '.py')).read_bytes(), 'before_' + lang)
                   for lang in ['en', *PLAN['languages']]}
    owned_id = ('UI', caption)
    for lang in PLAN['languages']:
        reviewed = api.reviewed_api_block_translations(selected, lang)
        translated = api._translate_api_documents(selected, lang, OUT / 'no-models', types.SimpleNamespace(), reuse_history=True)
        seen = 0
        for symbol, body in selected.items():
            sb = api.translatable_blocks(body)[0]
            tb = api.translatable_blocks(translated[symbol])[0]
            assert len(sb) == len(tb)
            for source, target in zip(sb, tb):
                assert target == reviewed[source]
                seen += 1
        assert seen == 7
        prepared_api[lang] = translated
        STATS['api_prepared'].append(lang)
        review_file = Path('docs/i18n/reviewed/runtime') / lang / '2026-10-15-n180-session-figures.json'
        records = read_json(review_file)['records']
        assert len(records) == 1 and records[0]['table'] == 'ui' and records[0]['key'] == caption
        owned_target = records[0]['translation']
        assert runtime.reviewed_runtime_translations(lang)[caption] == owned_target
        old = old_runtime[lang]
        translations, unchanged = {}, {}
        assert set(old.SOURCE_HASHES) - set(expected_hashes) == set(), (lang, 'foreign source removal')
        for table, section in TABLES.items():
            current = sources[section]
            assert set(getattr(old, table)) <= set(current), (lang, table, 'foreign table row removal')
            pairs = current.items() if isinstance(current, dict) else ((value, value) for value in current)
            for key, source in pairs:
                ident = (table, str(key))
                if ident == owned_id:
                    continue
                assert key in getattr(old, table) and old.SOURCE_HASHES.get(ident) == expected_hashes[ident], (lang, ident, 'foreign stale/missing source')
                value = getattr(old, table)[key]
                assert source not in translations or translations[source] == value, (lang, source, 'ambiguous foreign target')
                translations[source] = value
                unchanged[ident] = value
        assert caption not in translations or translations[caption] == owned_target
        assert runtime._translation_candidate_valid(caption, owned_target, lang, force=False)
        translations[caption] = owned_target
        prepared_runtime[lang] = (translations, unchanged, owned_target)
        STATS['runtime_prepared'].append(lang)
        proofs[lang] = {'reviewed_api_blocks': 7, 'reviewed_runtime_rows': 1,
                        'unchanged_runtime_rows': len(unchanged)}
        print(lang + ': API and runtime normal preflight passed', flush=True)
    assert len(prepared_api) == len(prepared_runtime) == 9
    check_pins(include_outputs=True)
    STATS['catalog_writes_started'] = True
    for lang in PLAN['languages']:
        api.write_language(selected, lang, prepared_api[lang], preserve_other_symbols=True)
        after = read_json(api.API_DIR / (lang + '.json'))
        assert foreign(after['symbols'], keys) == old_api[lang]['symbols']
        assert foreign(after, {'symbols'}) == foreign(old_api[lang], {'symbols'})
        for symbol, body in selected.items():
            row = after['symbols'][symbol]
            assert row['text'] == prepared_api[lang][symbol]
            assert row['source_sha256'] == api._source_hash(body)
            assert row['source_blocks_sha256'] == api._source_block_hashes(body)
            assert row['translation_source_blocks_sha256'] == api._translation_source_block_hashes(body)
    api._write_english_api_manifest(all_docs)
    after_en = read_json(api.API_DIR / 'en.json')
    assert foreign(after_en['symbols'], keys) == old_api['en']['symbols']
    assert foreign(after_en, {'symbols'}) == foreign(old_api['en'], {'symbols'})
    for symbol, text in selected.items():
        assert after_en['symbols'][symbol]['text'] == text
    assert read_json(history_path) == old_history, 'Unrelated API history changed'
    runtime.write_english(sources)
    for lang, (translations, unchanged, owned_target) in prepared_runtime.items():
        path = runtime.write_language(lang, sources, translations)
        after = namespace(path.read_bytes(), 'after_' + lang)
        old = old_runtime[lang]
        assert after.UI[caption] == owned_target
        assert set(after.SOURCE_HASHES) == set(old.SOURCE_HASHES) | {owned_id}
        for (table, key), value in unchanged.items():
            assert getattr(after, table)[key] == value
            assert after.SOURCE_HASHES[(table, key)] == old.SOURCE_HASHES[(table, key)]
        assert catalog_metadata(after) == catalog_metadata(old), (lang, 'catalog metadata drift')
    after_en_runtime = namespace((runtime.CATALOG_DIR / 'en.py').read_bytes(), 'after_en')
    old_en_runtime = old_runtime['en']
    for name in ('SETTING_LABELS', 'SETTING_TOOLTIPS', 'CATEGORY_SOURCES', 'MODULE_SUMMARIES'):
        assert getattr(after_en_runtime, name) == getattr(old_en_runtime, name)
    assert after_en_runtime.UI_SOURCES == old_en_runtime.UI_SOURCES | {caption}
    assert foreign(after_en_runtime.SOURCE_HASHES, {owned_id}) == foreign(old_en_runtime.SOURCE_HASHES, {owned_id})
    assert runtime.audit(sources, PLAN['languages']) == 0
    STATS['original_runtime_collections'] += 1
    assert ORIGINAL_CANONICAL() == sources, 'Fresh original canonical inventory changed'
    check_pins()
    importlib.invalidate_caches()
    STATS['arrival_modules_refreshed'] = []
    for lang in PLAN['languages']:
        name = 'spacr.qt.i18n_catalogs.' + lang
        sys.modules.pop(name, None)
        fresh_module = importlib.import_module(name)
        assert fresh_module.UI[caption] == prepared_runtime[lang][2]
        STATS['arrival_modules_refreshed'].append(name)
    arrivals.refresh_arrival_contract()
    after_arrivals = arrivals.load_arrival_contract()
    assert after_arrivals['baseline'] == old_arrivals['baseline']
    for name in ('retired', 'changed'):
        assert after_arrivals[name] == old_arrivals[name], (name, 'foreign arrival drift')
    omit_owned = lambda rows: [row for row in rows if (row['table'], row['key']) != owned_id]
    assert omit_owned(after_arrivals['added']) == omit_owned(old_arrivals['added'])
    assert foreign(after_arrivals, {'added', 'retired', 'changed'}) == foreign(old_arrivals, {'added', 'retired', 'changed'})
    check_pins()
    with (OUT / 'help-generation.log').open('w') as log:
        subprocess.run([sys.executable, 'tools/build_help_search_index.py'], check=True, stdout=log, stderr=subprocess.STDOUT)
    check_pins()
    assert not STATS['model_decode_requests'] and not STATS['full_api_collection_requests']
    report = {'plan_sha256': sha(PLAN_RAW), 'source_files': PLAN['source_files'],
              'baseline_public': PLAN['public_base'], 'locales': proofs,
              'api_history_unchanged': True, 'foreign_arrivals_preserved': True,
              'normal_runtime_audit': 0, 'normal_help_exit': 0,
              'limitations': 'Selected API source claim only; actual Help consumer checks and strict named API arrival acceptance remain separate.'}
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
