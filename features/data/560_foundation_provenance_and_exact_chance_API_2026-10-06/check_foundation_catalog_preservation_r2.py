from pathlib import Path
import ast
import hashlib
import json
import runpy
import subprocess

import build_documentation_i18n as api
import build_i18n_catalogs as runtime

scratch = Path('/media/carruthers/mnt3/codex/scratch/docs-completion-20261005')
baseline = scratch / 'foundation-api-refresh-baseline-r1'
delta = json.loads((scratch / 'foundation-api-source-delta-r1.json').read_text())
languages = ('en', 'sv', 'de', 'es', 'zh_CN', 'pt', 'hi', 'ko', 'is', 'fr')
reviewed = json.loads((scratch / 'foundation-api-reviewed-inputs-r2.json').read_text())
report = {'API': {}, 'runtime': {}, 'reviewed_files': {}}
for language in languages:
    path = Path('docs/source/_static/i18n/api') / (language + '.json')
    before = json.loads((baseline / 'api' / path.name).read_text())['symbols']
    after = json.loads(path.read_text())['symbols']
    assert set(before) == set(after) and len(after) == 13181
    assert {key for key in before if before[key] != after[key]} == {'spacr.embeddings.encoder_entry'}
    if language != 'en':
        old_blocks, _ = api.translatable_blocks(before['spacr.embeddings.encoder_entry']['text'])
        new_blocks, _ = api.translatable_blocks(after['spacr.embeddings.encoder_entry']['text'])
        assert len(old_blocks) == len(new_blocks) == 7
        assert {index for index in range(7) if old_blocks[index] != new_blocks[index]} == {2, 3}
        for entry in reviewed['languages'][language]['API']:
            index = int(entry['label'].rsplit('#', 1)[1])
            assert new_blocks[index] == entry['translation']
    report['API'][language] = {'unchanged_complete_records': 13180,
                               'current_complete_records': len(after),
                               'exact_changed_symbol': 'spacr.embeddings.encoder_entry',
                               'catalog_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    path = Path('spacr/qt/i18n_catalogs') / (language + '.py')
    before = runpy.run_path(str(baseline / 'runtime' / path.name))
    after = runpy.run_path(str(path))
    table = 'UI_SOURCES' if language == 'en' else 'UI'
    assert set(after[table]) - set(before[table]) == set(delta['runtime_added'])
    assert not set(before[table]) - set(after[table])
    for key, value in before.items():
        if key.isupper() and key not in (table, 'SOURCE_HASHES'):
            assert after[key] == value, (language, key)
    for key, value in before['SOURCE_HASHES'].items():
        assert after['SOURCE_HASHES'][key] == value
    if language != 'en':
        assert all(after[table][key] == value for key, value in before[table].items())
        for entry in reviewed['languages'][language]['runtime']:
            assert after[table][entry['source']] == entry['translation']
    assert len(after['SOURCE_HASHES']) == 9860
    assert len(before['SOURCE_HASHES']) == 9857
    report['runtime'][language] = {'all_prior_complete_rows_and_hashes_identical': 9857,
                                   'new_source_bound_notes': 3, 'current_rows': 9860,
                                   'catalog_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    print('PASS', language, '13180 complete API entries and all 9857 existing runtime rows exact', flush=True)
for lane in ('api', 'runtime'):
    old_root = baseline / ('reviewed-' + lane)
    new_root = Path('docs/i18n/reviewed') / lane
    old_paths = {path.relative_to(old_root) for path in old_root.rglob('*.json')}
    new_paths = {path.relative_to(new_root) for path in new_root.rglob('*.json')}
    assert old_paths <= new_paths
    for path in old_paths:
        assert (old_root / path).read_bytes() == (new_root / path).read_bytes()
    assert len(new_paths - old_paths) == 9
    report['reviewed_files'][lane] = {'all_prior_complete_files_and_order_byte_identical': len(old_paths),
                                     'new_direct_AI_review_files': 9}
old_nodes = {node.name: ast.dump(node) for node in ast.parse((baseline / 'embeddings.py').read_text()).body
             if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
new_nodes = {node.name: ast.dump(node) for node in ast.parse(Path('spacr/embeddings.py').read_text()).body
             if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
assert set(old_nodes) == set(new_nodes)
assert {key for key in old_nodes if old_nodes[key] != new_nodes[key]} == {
    '_weights_on_disk', 'encoder_entry', '_retrieval_scorecard', '_timm_encoder'}
home_source = subprocess.check_output(['git', 'show', 'df5b9bb7f9b0eb490cf91e72bb7b84e1d2902337:spacr/embeddings.py'], text=True)
home_nodes = {node.name: ast.dump(node) for node in ast.parse(home_source).body
              if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
assert new_nodes['_timm_encoder'] == home_nodes['_timm_encoder']
subprocess.run(['git', 'diff', '--quiet', '11d3a451b', '--', 'tools/tutorials',
                'docs/source/_extra/tutorials', 'spacr/qt/tutorial'], check=True)
report['all_other_inference_functions_except_coordinated_Home_timm_resize_AST_exact'] = True
report['Home_timm_resize_AST_exact_to_df5b9bb7f'] = True
report['Mask_Measure_Home_Conda_and_all_other_tutorial_source_and_media_unchanged'] = True
(scratch / 'foundation-api-catalog-preservation-r2.json').write_text(json.dumps(report, indent=2) + '\n')
print('PASS all catalogs, previous reviews, inference functions and tutorials preserved', flush=True)
