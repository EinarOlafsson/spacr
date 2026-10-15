"""Package reviewed N639 inputs after final source freeze; never write catalogs."""
from __future__ import annotations

import ast
import hashlib
import json
import pathlib
import subprocess
import sys


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    plan_path = pathlib.Path(sys.argv[1]).resolve()
    plan = read_json(plan_path)
    if plan.get('state') != 'final_source_frozen':
        raise SystemExit('REFUSED: final organizer source/pins are not frozen')
    root = pathlib.Path(plan['repository']).resolve()
    scratch = plan_path.parent
    out = pathlib.Path(plan['output']).resolve()
    if not out.is_relative_to(pathlib.Path('/mnt/wd4tb/scratch')) or out.exists():
        raise SystemExit('REFUSED: output must be a new WD scratch directory')
    source_pin = plan['source_commit']
    if not source_pin or len(source_pin) != 40:
        raise SystemExit('REFUSED: exact final source commit required')
    subprocess.run(['git', '-C', str(root), 'cat-file', '-e', source_pin + '^{commit}'], check=True)
    for name, expected in plan['draft_files'].items():
        assert sha((scratch / name).read_bytes()) == expected, name
    for name, expected in plan['source_files'].items():
        assert isinstance(expected, str) and len(expected) == 64, name
        raw = (root / name).read_bytes()
        assert sha(raw) == expected, name
        committed = subprocess.check_output(['git', '-C', str(root), 'show', f'{source_pin}:{name}'])
        assert committed == raw, f'uncommitted source drift: {name}'
    for name, expected in plan['tool_files'].items():
        assert isinstance(expected, str) and len(expected) == 64, name
        assert sha((root / name).read_bytes()) == expected, name
    sources = read_json(scratch / 'sources.json')['sources'] + read_json(scratch / 'additional-sources.json')['sources']
    initial = read_json(scratch / 'targets.json')
    extra = read_json(scratch / 'additional-targets.json')
    languages = plan['languages']
    assert set(initial) == set(extra) == set(languages)
    assert len(sources) == 21 and len({r['source'] for r in sources}) == 21
    targets = {lang: initial[lang] + extra[lang] for lang in languages}
    assert all(len(rows) == 21 and all(isinstance(x, str) for x in rows) for rows in targets.values())
    baseline_targets = {lang: list(rows) for lang, rows in targets.items()}
    followup = read_json(scratch / plan['target_followup'])
    changed = set()
    for row in followup['records']:
        lang, index = row['language'], row['index']
        assert (lang, index) not in changed
        assert sources[index]['source'] == row['source']
        assert sha(row['source'].encode()) == row['source_sha256']
        assert targets[lang][index] == row['old_target']
        assert sha(row['old_target'].encode()) == row['old_target_sha256']
        assert sha(row['target'].encode()) == row['target_sha256']
        targets[lang][index] = row['target']
        changed.add((lang, index))
    assert changed == {(lang, index) for lang in ('hi', 'ko', 'zh_CN') for index in (7, 9, 13)}
    assert sum(targets[lang][i] == baseline_targets[lang][i] for lang in languages for i in range(21)) == 180

    sys.path.insert(0, str(root / 'tools'))
    import build_i18n_catalogs as runtime
    assert pathlib.Path(runtime.__file__).resolve() == root / 'tools/build_i18n_catalogs.py'
    scopes = {
        'channel_sorting.build_plan': ('spacr/channel_sorting.py', ('build_plan',)),
        'SetReport': ('spacr/channel_sorting.py', ('SetReport', 'summary')),
        '_few': ('spacr/channel_sorting.py', ('_few',)),
        'OrganizeForMeasureDialog._plan_summary': ('spacr/qt/widgets/organize_for_measure.py', ('OrganizeForMeasureDialog', '_plan_summary')),
    }
    found = {}
    for owner, (name, qualname) in scopes.items():
        node = ast.parse((root / name).read_text(encoding='utf-8'))
        for part in qualname:
            candidates = [n for n in node.body if isinstance(n, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == part]
            assert len(candidates) == 1, (owner, part)
            node = candidates[0]
        literals = {}
        for call in ast.walk(node):
            if not isinstance(call, ast.Call) or runtime._call_name(call) != 'tr':
                continue
            for arg in runtime._candidate_arguments(call, 'tr'):
                for source in runtime._literal_strings(arg, {}):
                    if runtime._looks_translatable(source):
                        literals.setdefault(source, []).append(call.lineno)
        found[owner] = literals
    source_proof = []
    for row in sources:
        assert row['source'] in found[row['owner']], row
        source_proof.append({'owner': row['owner'], 'source': row['source'], 'source_sha256': sha(row['source'].encode()), 'lines': found[row['owner']][row['source']]})

    prior_root = root / 'docs/i18n/reviewed/runtime'
    prior_hashes = {}
    overlaps = []
    for lang in languages:
        owned = {row['source']: target for row, target in zip(sources, targets[lang])}
        for path in sorted((prior_root / lang).glob('*.json')):
            prior_hashes[str(path.relative_to(root))] = sha(path.read_bytes())
            for record in read_json(path).get('records', []):
                key = record.get('key')
                if record.get('table') != 'ui' or key not in owned:
                    continue
                assert record.get('source') == key and record.get('source_sha256') == sha(key.encode()), ('stale owned prior input', path, key)
                assert record.get('translation') == owned[key], ('conflicting owned prior input', path, key)
                overlaps.append({'language': lang, 'path': str(path.relative_to(root)), 'source': key, 'translation_sha256': sha(owned[key].encode())})

    out.mkdir(parents=True)
    review_root = out / 'docs/i18n/reviewed/runtime'
    files = {}
    for lang in languages:
        payload = {
            'schema': 1,
            'language': lang,
            'review_kind': 'AI technical translations independently reviewed by Home-B coordinator; no native-speaker or human-review claim',
            'source_commit': source_pin,
            'source_file_sha256': plan['source_files'],
            'tool_file_sha256': plan['tool_files'],
            'records': [
                {'table': 'ui', 'key': row['source'], 'source': row['source'], 'source_sha256': sha(row['source'].encode()), 'translation': target}
                for row, target in zip(sources, targets[lang])
            ],
        }
        path = review_root / lang / plan['basename']
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n')
        files[str(path.relative_to(out))] = sha(path.read_bytes())
    original_canonical = runtime.canonical_sources
    original_dir = runtime.REVIEWED_RUNTIME_DIR
    checks = []
    try:
        runtime.REVIEWED_RUNTIME_DIR = review_root
        runtime.canonical_sources = lambda: {'ui': tuple(row['source'] for row in sources)}
        runtime.reviewed_runtime_translations.cache_clear()
        for lang in languages:
            actual = runtime.reviewed_runtime_translations(lang)
            expected = {row['source']: target for row, target in zip(sources, targets[lang])}
            assert actual == expected, lang
            checks.append({'language': lang, 'normal_scoped_review_records': len(actual)})
    finally:
        runtime.canonical_sources = original_canonical
        runtime.REVIEWED_RUNTIME_DIR = original_dir
        runtime.reviewed_runtime_translations.cache_clear()
    for name, expected in {**plan['source_files'], **plan['tool_files'], **prior_hashes}.items():
        assert sha((root / name).read_bytes()) == expected, name
    for name, expected in plan['draft_files'].items():
        assert sha((scratch / name).read_bytes()) == expected, name
    receipt = {
        'scope': 'Input-only normal reviewed loader over 21 exact AST-bound tr literals and isolated review files; no full canonical/global extraction claim',
        'source_commit': source_pin,
        'source_files': plan['source_files'],
        'tool_files': plan['tool_files'],
        'source_proof': source_proof,
        'plan_sha256': sha(plan_path.read_bytes()),
        'script_sha256': sha(pathlib.Path(__file__).read_bytes()),
        'normal_scoped_loader': checks,
        'records': 189,
        'placeholder_order_followups': 9,
        'other_accepted_targets_byte_preserved': 180,
        'files': files,
        'identical_active_review_overlaps': overlaps,
        'prior_active_review_files_preserved': len(prior_hashes),
        'catalog_writes': False,
        'full_collector_calls': 0,
        'remaining': 'Coordinator full canonical source ownership, normal all-nine generation/audit/arrivals and actual consumers are separate.'
    }
    (out / 'receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'status': 'pass', 'records': 189, 'normal_review_locales': 9, 'output': str(out)}))


if __name__ == '__main__':
    main()
