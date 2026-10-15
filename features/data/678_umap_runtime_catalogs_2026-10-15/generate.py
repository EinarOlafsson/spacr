"""Write explicitly owned reviewed editorial caption changes with normal runtime tools."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import types

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--root', type=Path, required=True)
parser.add_argument('--receipt', type=Path, required=True)
parser.add_argument('--manifest', type=Path, required=True)
args = parser.parse_args()
root = args.root.resolve()
sys.path[:0] = [str(root / 'tools'), str(root)]
import build_i18n_catalogs as builder

manifest = json.loads(args.manifest.read_text())
owned_sources = set(manifest['current_sources'])
retired_sources = set(manifest['retired_sources'])
assert not owned_sources & retired_sources
review_names = manifest['review_names']
tables = {'SETTING_LABELS': 'setting_labels', 'SETTING_TOOLTIPS': 'setting_tooltips',
          'MODULE_SUMMARIES': 'module_summaries', 'CATEGORY_HELP': 'categories', 'UI': 'ui'}
receipt = {
    'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
    'mode': 'Normal exact-source reviewed_runtime_translations and write_language; no model inference',
    'scope': 'Explicit current/retired editorial UI sources; every unrelated entry preserved',
    'retired_sources': sorted(retired_sources),
    'review_kind': 'AI technical review, no native-speaker or human linguistic acceptance claimed',
    'owned_sources': sorted(owned_sources), 'locales': {}, 'passed': False,
}

def committed(language):
    module = types.ModuleType('committed_' + language)
    text = subprocess.check_output(['git', 'show', f'HEAD:spacr/qt/i18n_catalogs/{language}.py'],
                                   cwd=root, text=True)
    exec(compile(text, 'committed_' + language, 'exec'), module.__dict__)
    return module

try:
    canonical = builder.canonical_sources
    sources = canonical()
    expected = builder._source_hashes(sources)
    assert owned_sources.issubset(sources['ui'])
    builder.canonical_sources = lambda: sources
    owned_idents = {('UI', source) for source in owned_sources | retired_sources}
    assert retired_sources.isdisjoint(sources['ui']), 'Declared retired source is still canonical'
    before_english = committed('en')
    receipt['unrelated_english_source_drift'] = [
        list(key) for key in sorted(set(before_english.SOURCE_HASHES) | set(expected))
        if key not in owned_idents and before_english.SOURCE_HASHES.get(key) != expected.get(key)
    ]
    assert not receipt['unrelated_english_source_drift'], 'Unrelated English sources are stale; refusing scoped generation'
    for table, section in tables.items():
        if table == 'UI':
            assert set(before_english.UI_SOURCES) - owned_sources - retired_sources == set(sources[section]) - owned_sources
        elif table == 'CATEGORY_HELP':
            assert before_english.CATEGORY_SOURCES == frozenset(sources[section])
        else:
            assert getattr(before_english, table) == sources[section], table
    prepared = {}
    for language in builder.MODEL_SPECS:
        owned = {}
        review_hashes = {}
        for review_name in review_names:
            path = root / 'docs/i18n/reviewed/runtime' / language / review_name
            review = json.loads(path.read_text())
            review_hashes[review_name] = hashlib.sha256(path.read_bytes()).hexdigest()
            for row in review['records']:
                assert row['source'] not in owned or owned[row['source']] == row['translation']
                owned[row['source']] = row['translation']
        assert set(owned) == owned_sources, (language, 'Exact owner review source set differs')
        reviewed = builder.reviewed_runtime_translations(language)
        assert all(reviewed[source] == target for source, target in owned.items()), language
        old = committed(language)
        assert {k: v for k, v in old.SOURCE_HASHES.items() if k not in owned_idents} == {
            k: v for k, v in expected.items() if k not in owned_idents
        }, (language, 'unrelated source hash drift')
        translations = {}
        untouched = {}
        missing = []
        for table, section in tables.items():
            values = sources[section]
            pairs = dict(values) if isinstance(values, dict) else {s: s for s in values}
            previous = getattr(old, table)
            assert not (set(previous) - set(pairs) - (retired_sources if table == 'UI' else set())), (language, table, 'unrelated obsolete rows')
            for key, source in pairs.items():
                ident = (table, str(key))
                if ident in owned_idents:
                    continue
                if key not in previous or old.SOURCE_HASHES.get(ident) != expected[ident]:
                    missing.append(ident)
                    continue
                value = previous[key]
                assert source not in translations or translations[source] == value, (language, source)
                translations[source] = value
                untouched[ident] = value
        assert not missing, (language, 'unrelated targets not source-current', missing)
        for source, target in owned.items():
            assert builder._translation_candidate_valid(source, target, language, force=False), (language, source)
        translations.update(owned)
        prepared[language] = translations, untouched, owned
        receipt['locales'][language] = {
            'review_file_hashes': review_hashes,
            'normal_review_lookup_passed': True, 'normal_candidate_gates_passed': True,
            'selected_rows': len(owned), 'unrelated_source_current_rows': len(untouched),
            'formatted_placeholders_passed': True,
        }
        print(f'{language}: {len(owned)} exact reviews valid; {len(untouched)} unrelated targets preserved', flush=True)

    assert canonical() == sources, 'Source changed during generation'
    builder.write_english(sources)
    for language, (translations, untouched, owned) in prepared.items():
        path = builder.write_language(language, sources, translations)
        spec = importlib.util.spec_from_file_location('generated_' + language, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        assert module.SOURCE_HASHES == expected, language
        assert all(getattr(module, table)[key] == value for (table, key), value in untouched.items()), language
        assert all(module.UI[source] == target for source, target in owned.items()), language
        for table, section in tables.items():
            assert set(getattr(module, table)) == set(sources[section]), (language, table)
        receipt['locales'][language].update(unrelated_targets_preserved=True,
                                          all_source_hashes_current=True, exact_key_coverage=True)
        print(f'{language}: normal writer output verified; only owned targets added/updated', flush=True)
    assert builder.audit(sources, ()) == 0
    receipt['english_normal_source_audit'] = 'PASS'
    receipt['full_unrelated_translation_safety_audit'] = 'Not rerun: unrelated translations preserved exactly'
    receipt['passed'] = True
finally:
    args.receipt.write_text(json.dumps(receipt, indent=2) + '\n')
