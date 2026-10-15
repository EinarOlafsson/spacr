"""Normal scoped runtime writer; requires frozen independently approved inputs."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import types

args = argparse.ArgumentParser()
args.add_argument('--root', type=Path, required=True)
args.add_argument('--approved-inputs', type=Path, required=True)
args.add_argument('--approved-inputs-sha256', required=True)
args.add_argument('--write', action='store_true')
options = args.parse_args()
root = options.root.resolve()
scratch = Path(__file__).resolve().parent
review_dir = root / 'docs/i18n/reviewed/runtime'
languages = ('sv', 'de', 'es', 'zh_CN', 'pt', 'hi', 'ko', 'is', 'fr')
groups = {
    '2026-10-15-complete-curation-status.json': 8,
    '2026-10-15-source-picker-complete-captions.json': 6,
    '2026-10-15-hand-drawn-measurement-outputs.json': 9,
}
record_fields = {'table', 'key', 'source_sha256', 'source', 'translation'}
fragment = 'no unjudged fields are left in this queue'


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def sha(path):
    return digest(path.read_bytes())


def record_sha(value):
    return digest(json.dumps(value, ensure_ascii=False, sort_keys=True).encode())


def source_pins():
    paths = list((root / 'spacr').rglob('*.py'))
    paths += [root / 'tools' / name for name in (
        'build_i18n_catalogs.py', 'i18n_caption_arrivals.py', 'i18n_reviewed_ui.py')]
    paths.append(root / 'packaging/i18n/en.json')
    generated = {root / 'spacr/qt/i18n_catalogs' / (language + '.py')
                 for language in ('en',) + languages}
    return {str(p.relative_to(root)): sha(p) for p in paths if p not in generated}


def review_pins():
    return {str(p.relative_to(root)): sha(p) for language in languages
            for p in sorted((review_dir / language).glob('*.json'))}


driver_sha256 = sha(Path(__file__))
arrival_path = root / 'tests/data/release_contracts/runtime_caption_arrivals_2026_10_08.json.gz'
arrival_input_sha256 = sha(arrival_path)
approved_path = options.approved_inputs.resolve()
approved_raw = approved_path.read_bytes()
assert digest(approved_raw) == options.approved_inputs_sha256, 'Approved manifest bytes drift'
approved = json.loads(approved_raw)
assert approved['schema'] == 1 and set(approved['groups']) == set(groups)
approved_sources = set()
for name, count in groups.items():
    group = approved['groups'][name]
    assert group['count'] == count
    assert len(group['sources']) == len(set(group['sources'])) == count
    assert not approved_sources.intersection(group['sources'])
    approved_sources.update(group['sources'])
assert len(approved_sources) == 23
expected_owned_paths = {str((review_dir / language / name).relative_to(root))
                        for language in languages for name in groups}
assert set(approved['owned_files']) == expected_owned_paths
assert set(approved['retirements']) == set(languages)
pins = source_pins()
initial_review_pins = review_pins()
input_head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
owned = {}
for language in languages:
    targets = {}
    for name, count in groups.items():
        path = review_dir / language / name
        raw = path.read_bytes()
        proof = approved['owned_files'][str(path.relative_to(root))]
        assert digest(raw) == proof['sha256'], ('Approved owned file drift', path)
        payload = json.loads(raw)
        assert payload['language'] == language and payload['schema'] == 1
        rows = payload['records']
        assert len(rows) == count and record_sha(rows) == proof['records_sha256']
        assert {row['source'] for row in rows} == set(approved['groups'][name]['sources'])
        for row in rows:
            assert set(row) == record_fields
            assert row['table'] == 'ui' and row['source'] == row['key']
            assert digest(row['source'].encode()) == row['source_sha256']
            assert row['source'] not in targets, (language, row['source'])
            targets[row['source']] = row['translation']
    assert set(targets) == approved_sources
    owned[language] = targets
assert pins == source_pins(), 'Source changed while reading approved owned inputs'
assert initial_review_pins == review_pins(), 'Review inputs changed while reading approved inputs'
assert sha(approved_path) == options.approved_inputs_sha256
assert sha(Path(__file__)) == driver_sha256, 'Driver bytes drift'
assert sha(arrival_path) == arrival_input_sha256, 'Arrival input drift before normal refresh'
sys.path[:0] = [str(root / 'tools'), str(root)]
import build_i18n_catalogs as b
assert Path(b.__file__).resolve() == root / 'tools/build_i18n_catalogs.py'
assert tuple(b.MODEL_SPECS) == languages
original_canonical = b.canonical_sources
assert pins == source_pins(), 'Source changed before actual canonical collection'
assert initial_review_pins == review_pins(), 'Review inputs changed before actual canonical collection'
sources = original_canonical()
assert pins == source_pins(), 'Source changed during actual canonical collection'
assert initial_review_pins == review_pins(), 'Review inputs changed during actual canonical collection'
assert approved_sources <= set(sources['ui'])
assert fragment not in sources['ui'], 'Old fragment remains canonical; refuse retirement'
expected = b._source_hashes(sources)
b.canonical_sources = lambda: sources

# Validate every retirement and destination before any review-file mutation.
retirements = {}
post_review_pins = dict(initial_review_pins)
for language in languages:
    path = review_dir / language / '2026-10-13-701-hide-judged-fields.json'
    proof = approved['retirements'][language]
    assert proof['path'] == str(path.relative_to(root))
    raw = path.read_bytes()
    assert digest(raw) == proof['file_sha256'], ('Approved retirement file drift', path)
    payload = json.loads(raw)
    assert payload['schema'] == 1 and payload['language'] == language
    assert isinstance(payload['records'], list)
    rows = [row for row in payload['records'] if row.get('source') == fragment]
    assert len(rows) == 1
    row = rows[0]
    assert set(row) == record_fields
    assert row['key'] == fragment and row['table'] == 'ui'
    assert row['source_sha256'] == digest(fragment.encode())
    assert record_sha(row) == proof['row_sha256'], ('Approved retirement row drift', path)
    retired = dict(payload)
    retired['records'] = [record for record in payload['records'] if record is not row]
    assert len(retired['records']) == len(payload['records']) - 1
    archive = path.parent / 'archive/2026-10-15-complete-curation-status' / path.name
    assert not archive.exists() and not archive.is_symlink(), 'Never overwrite retirement provenance'
    new_raw = (json.dumps(retired, ensure_ascii=False, indent=2) + '\n').encode()
    retirements[language] = (path, raw, payload, retired, archive, new_raw)
    post_review_pins[str(path.relative_to(root))] = digest(new_raw)

# Validate approved ownership and every foreign identity before retirement/writes.
tables = {'SETTING_LABELS': 'setting_labels', 'SETTING_TOOLTIPS': 'setting_tooltips',
          'MODULE_SUMMARIES': 'module_summaries', 'CATEGORY_HELP': 'categories', 'UI': 'ui'}
prepared = {}
for language, targets in owned.items():
    old = types.ModuleType('committed_' + language)
    baseline = subprocess.check_output(
        ['git', 'show', input_head + ':spacr/qt/i18n_catalogs/' + language + '.py'],
        cwd=root, text=True)
    exec(compile(baseline, 'committed_catalog_' + language, 'exec'), old.__dict__)
    translations = {}
    untouched = {}
    for table, section in tables.items():
        values = sources[section]
        pairs = values.items() if isinstance(values, dict) else ((s, s) for s in values)
        for key, source in pairs:
            if table == 'UI' and key in targets:
                continue
            ident = (table, str(key))
            previous = getattr(old, table)
            assert key in previous and old.SOURCE_HASHES.get(ident) == expected[ident], (
                language, ident, 'foreign source drift')
            value = previous[key]
            assert source not in translations or translations[source] == value
            translations[source] = value
            untouched[ident] = value
    for source, target in targets.items():
        assert source not in translations or translations[source] == target, (
            language, source, 'owned UI text conflicts with foreign identity value')
    translations.update(targets)
    prepared[language] = (translations, untouched, targets)

assert pins == source_pins(), 'Source changed before review-file mutation'
assert initial_review_pins == review_pins(), 'Review inputs changed before retirement'
assert sha(approved_path) == options.approved_inputs_sha256
assert sha(Path(__file__)) == driver_sha256, 'Driver bytes drift'
assert sha(arrival_path) == arrival_input_sha256, 'Arrival input drift before normal refresh'
if not options.write:
    raise SystemExit('Approved ownership/foreign preservation/all9retirement preflights PASS; --write required')
for language, (path, raw, original, retired, archive, new_raw) in retirements.items():
    assert path.read_bytes() == raw
    archive.parent.mkdir(parents=True, exist_ok=True)
    with archive.open('xb') as handle:
        handle.write(raw)
    assert archive.read_bytes() == raw, 'Full retirement archive readback mismatch'
    path.write_bytes(new_raw)
assert review_pins() == post_review_pins, 'Retirement changed unrelated review-file bytes'
assert pins == source_pins(), 'Source changed during retirement'
b.reviewed_runtime_translations.cache_clear()
for language, targets in owned.items():
    reviewed = b.reviewed_runtime_translations(language)
    assert all(reviewed[key] == value for key, value in targets.items())
    for source, target in targets.items():
        assert b._translation_candidate_valid(source, target, language, force=False), (
            language, source)
    print(language + ': normal review and source-current foreign-preservation preflight PASS', flush=True)
assert pins == source_pins(), 'Source changed before ordinary artifact writes'
assert review_pins() == post_review_pins, 'Review inputs changed before ordinary writes'
assert sha(approved_path) == options.approved_inputs_sha256
assert sha(Path(__file__)) == driver_sha256, 'Driver bytes drift'
assert sha(arrival_path) == arrival_input_sha256, 'Arrival input drift before normal refresh'
receipt = {'input_head': input_head, 'driver_sha256': driver_sha256,
           'pre_refresh_arrival_sha256': arrival_input_sha256,
           'approved_inputs_sha256': options.approved_inputs_sha256, 'owned_sources': 23,
           'retired_source': fragment, 'retirement_count': 9, 'locales': {}}
b.write_english(sources)
for language, (translations, untouched, targets) in prepared.items():
    assert pins == source_pins() and review_pins() == post_review_pins
    path = b.write_language(language, sources, translations)
    new = types.ModuleType('new_' + language)
    exec(compile(path.read_bytes(), str(path), 'exec'), new.__dict__)
    assert new.SOURCE_HASHES == expected
    assert all(getattr(new, table)[key] == value for (table, key), value in untouched.items())
    assert all(new.UI[key] == target for key, target in targets.items())
    receipt['locales'][language] = {'owned_rows': len(targets),
                                   'unrelated_preserved': len(untouched),
                                   'catalog_sha256': sha(path)}
assert b.audit(sources, tuple(b.MODEL_SPECS)) == 0

# Arrival tooling must see written source bytes, not an earlier imported catalog.
import spacr.qt.i18n_catalogs as runtime_catalogs
for language in ('en',) + languages:
    path = root / 'spacr/qt/i18n_catalogs' / (language + '.py')
    name = 'spacr.qt.i18n_catalogs.' + language
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    fresh = importlib.util.module_from_spec(spec)
    exec(compile(path.read_bytes(), str(path), 'exec'), fresh.__dict__)
    sys.modules[name] = fresh
    setattr(runtime_catalogs, language, fresh)
runtime_catalogs._module.cache_clear()
runtime_catalogs._english.cache_clear()
import i18n_caption_arrivals as arrivals
assert sha(arrival_path) == arrival_input_sha256, 'Arrival input drift before refresh'
previous_arrival = arrivals.load_arrival_contract()
receipt['normal_arrivals'] = arrivals.refresh_arrival_contract()
assert arrivals.load_arrival_contract()['baseline'] == previous_arrival['baseline']
assert original_canonical() == sources, 'Fresh original canonical collector drift'
assert pins == source_pins(), 'Source changed during ordinary producer'
assert review_pins() == post_review_pins, 'Review inputs changed during ordinary producer'
assert sha(approved_path) == options.approved_inputs_sha256
assert sha(Path(__file__)) == driver_sha256, 'Driver bytes drift during normal producer'
for language, (path, raw, original, retired, archive, new_raw) in retirements.items():
    assert archive.read_bytes() == raw, 'Retirement archive changed during normal producer'
receipt['post_refresh_arrival_sha256'] = sha(arrival_path)
for language, row in receipt['locales'].items():
    assert sha(root / 'spacr/qt/i18n_catalogs' / (language + '.py')) == row['catalog_sha256']
receipt['english_catalog_sha256'] = sha(root / 'spacr/qt/i18n_catalogs/en.py')
receipt['source_sha256'] = pins
receipt['review_input_sha256'] = post_review_pins
receipt['pre_retirement_review_input_sha256'] = initial_review_pins
receipt['retirement_provenance'] = {
    language: {'path': str(path.relative_to(root)), 'old_file_sha256': digest(raw),
               'archive': str(archive.relative_to(root)), 'archive_sha256': sha(archive),
               'retired_exact_row': [row for row in original['records'] if row['source'] == fragment][0],
               'unrelated_rows_preserved': len(retired['records'])}
    for language, (path, raw, original, retired, archive, new_raw) in retirements.items()}
receipt['normal_nine_runtime_audit_exit'] = 0
(scratch / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
print('Normal runtime generation/audit/arrival and fresh original source proof PASS', flush=True)
