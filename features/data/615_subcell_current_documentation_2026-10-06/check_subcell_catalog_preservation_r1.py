from pathlib import Path
import ast
import hashlib
import json
import runpy
import subprocess
import build_documentation_i18n as api

scratch = Path('/media/carruthers/mnt3/codex/scratch/docs-completion-20261005')
baseline = scratch / 'subcell-rybg-documentation-baseline-r1'
inventory = json.loads((scratch / 'subcell-rybg-documentation-inventory-r4.json').read_text())
runtime_reviews = json.loads((scratch / 'subcell-rybg-reviewed-runtime-r4.json').read_text())
languages = ('en', 'sv', 'de', 'es', 'zh_CN', 'pt', 'hi', 'ko', 'is', 'fr')
digest = lambda data: hashlib.sha256(data).hexdigest()
assert digest(Path('spacr/embeddings.py').read_bytes()) == inventory['source_sha256']
report = {'application_source_sha256': inventory['source_sha256'], 'API': {}, 'runtime': {}, 'reviewed_files': {}}
for language in languages:
    path = Path('docs/source/_static/i18n/api') / (language + '.json')
    before = json.loads((baseline / 'api' / path.name).read_text())['symbols']
    after = json.loads(path.read_text())['symbols']
    assert len(before) == 13181 and len(after) == 13182
    assert set(after) - set(before) == set(inventory['API_added'])
    assert not set(before) - set(after)
    assert {key for key in before if before[key] != after[key]} == set(inventory['API_changed'])
    if language != 'en':
        reviewed_path = Path('docs/i18n/reviewed/api') / language / '2026-10-06-subcell-rybg.json'
        reviewed = json.loads(reviewed_path.read_text())['records']
        assert len(reviewed) == 8
        for row in reviewed:
            symbol, _, block_index = row['label'].rpartition('#')
            blocks, _ = api.translatable_blocks(after[symbol]['text'])
            assert blocks[int(block_index)] == row['translation'], (language, row['label'])
    report['API'][language] = {'preserved_complete_records': 13177, 'changed_existing_records': inventory['API_changed'], 'added': inventory['API_added'], 'catalog_sha256': digest(path.read_bytes())}
    path = Path('spacr/qt/i18n_catalogs') / (language + '.py')
    before = runpy.run_path(str(baseline / 'runtime' / path.name))
    after = runpy.run_path(str(path))
    table = 'UI_SOURCES' if language == 'en' else 'UI'
    assert set(after[table]) - set(before[table]) == set(inventory['new_runtime'])
    assert set(before[table]) - set(after[table]) == set(inventory['retired_runtime'])
    for key, value in before.items():
        if key.isupper() and key not in (table, 'SOURCE_HASHES'):
            assert after[key] == value, (language, key)
    preserved = 0
    for key, value in before['SOURCE_HASHES'].items():
        if key == ('UI', inventory['retired_runtime'][0]):
            continue
        assert after['SOURCE_HASHES'][key] == value
        preserved += 1
    assert preserved == 9859 and len(after['SOURCE_HASHES']) == 9880
    if language != 'en':
        for key, value in before[table].items():
            if key not in inventory['retired_runtime']:
                assert after[table][key] == value, (language, key)
        for row in runtime_reviews[language]:
            assert after[table][row['source']] == row['translation'], (language, row)
        assert after[table]['ER (Y)'] == 'ER (Y)'
    report['runtime'][language] = {'all_prior_unchanged_complete_rows': preserved, 'new_reviewed_messages': 21, 'retired': inventory['retired_runtime'], 'current_rows': 9880, 'catalog_sha256': digest(path.read_bytes())}
    print('PASS complete catalog preservation', language, flush=True)
for lane in ('api', 'runtime'):
    old_root = baseline / ('reviewed-' + lane)
    current_root = Path('docs/i18n/reviewed') / lane
    changed_files = []
    original_count = 0
    for old_path in sorted(old_root.rglob('*.json')):
        relative = old_path.relative_to(old_root)
        path = current_root / relative
        assert path.exists(), path
        original_count += 1
        if path.read_bytes() == old_path.read_bytes():
            continue
        candidates = [p for p in (current_root / relative.parts[0] / 'archive').rglob(path.name) if p.read_bytes() == old_path.read_bytes()]
        assert candidates, relative
        before = json.loads(old_path.read_text())
        after = json.loads(path.read_text())
        assert {key: value for key, value in before.items() if key != 'records'} == {key: value for key, value in after.items() if key != 'records'}
        if lane == 'api':
            assert len(before['records']) == len(after['records'])
            for old, new in zip(before['records'], after['records']):
                if old == new:
                    continue
                symbol = old['label'].rpartition('#')[0]
                assert symbol in inventory['API_changed']
                if old['label'] != new['label']:
                    assert {k: v for k, v in old.items() if k != 'label'} == {k: v for k, v in new.items() if k != 'label'}
                else:
                    assert new['retired'] is True and new['retired_reason']
                    assert old == {k: v for k, v in new.items() if k not in ('retired', 'retired_reason')}
        else:
            assert after['records'] == [row for row in before['records'] if row['source'] not in inventory['retired_runtime']]
        changed_files.append({'active': str(relative), 'exact_complete_original_archive': str(candidates[0].relative_to(current_root)), 'original_sha256': digest(old_path.read_bytes())})
    report['reviewed_files'][lane] = {'all_complete_original_files_retained_byte_exact': original_count, 'active_source_based_changes': changed_files}
def nodes(source):
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str):
            node.body.pop(0)
    return {node.name: ast.dump(node) for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
home = subprocess.check_output(['git', 'show', '3ce39ac7b:spacr/embeddings.py'], text=True)
private = subprocess.check_output(['git', 'show', '58df6f5b9:spacr/embeddings.py'], text=True)
current = nodes(Path('spacr/embeddings.py').read_text())
owned = {'_weights_on_disk', 'encoder_entry', '_retrieval_scorecard'}
home_nodes, private_nodes = nodes(home), nodes(private)
assert set(current) == set(home_nodes)
for name, value in current.items():
    assert value == (private_nodes[name] if name in owned else home_nodes[name]), name
subprocess.run(['git', 'diff', '--quiet', '58df6f5b9', '--', 'tools/tutorials', 'docs/source/_extra/tutorials', 'spacr/qt/tutorial'], check=True)
report['all_Home_inference_code_preserved_excluding_documentation_and_three_exact_Root_owned_functions'] = True
report['all_tutorial_source_and_media_byte_preserved'] = True
report['passed'] = True
(scratch / 'subcell-catalog-preservation-r1.json').write_text(json.dumps(report, indent=2) + '\n')
print('PASS every catalog, complete historical review and accepted tutorial; application runtime AST preserved', flush=True)
