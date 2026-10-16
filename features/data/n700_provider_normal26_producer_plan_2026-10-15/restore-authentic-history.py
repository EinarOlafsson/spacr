import json
import os
import pathlib
import shutil
import subprocess


def restore(tool, root, stage, full_current_texts, packet_path):
    api = root / 'docs/source/_static/i18n/api'
    history = root / 'docs/i18n/api_source_history.json'
    packet = json.loads(packet_path.read_text())
    assert not packet['missing'] and len(packet['found']) == 4
    old_history = json.loads(history.read_text())
    assert old_history['schema'] == 1 and len(old_history['sources']) == 14
    original_commit = '6e386930baaadf374319d8e7a0e521cc5f53eded'
    original = json.loads(subprocess.check_output(
        ['git', '-C', str(root), 'show', original_commit + ':docs/source/_static/i18n/api/en.json'],
        env=dict(os.environ, GIT_NO_LAZY_FETCH='1'),
    ))['symbols']
    additions = {}
    staged_api = stage / 'docs/source/_static/i18n/api'
    staged_api.mkdir(parents=True, exist_ok=False)
    for language in ['en', *tool.MODEL_SPECS]:
        shutil.copyfile(api / (language + '.json'), staged_api / (language + '.json'))
    staged_history = stage / 'docs/i18n/api_source_history.json'
    staged_history.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(history, staged_history)
    staged_en = json.loads((staged_api / 'en.json').read_text())
    for symbol, evidence in packet['found'].items():
        record = evidence['original_english_record']
        assert evidence['commit'] == original_commit
        assert record == original[symbol]
        projection = {field: record[field] for field in ('source_sha256', 'source_blocks_sha256', 'text')}
        digest = projection['source_sha256']
        assert digest == tool._source_hash(projection['text'])
        assert projection['source_blocks_sha256'] == tool._source_block_hashes(projection['text'])
        assert digest not in old_history['sources']
        additions[digest] = projection
        staged_en['symbols'][symbol] = projection
    assert len(additions) == 4
    (staged_api / 'en.json').write_text(json.dumps(staged_en, ensure_ascii=False, indent=2, sort_keys=True) + '\n')
    old_root, old_api = tool.ROOT, tool.API_DIR
    try:
        tool.ROOT, tool.API_DIR = stage, staged_api
        tool._write_english_api_manifest(full_current_texts)
    finally:
        tool.ROOT, tool.API_DIR = old_root, old_api
    assert (staged_api / 'en.json').read_bytes() == (api / 'en.json').read_bytes()
    for language in tool.MODEL_SPECS:
        assert (staged_api / (language + '.json')).read_bytes() == (api / (language + '.json')).read_bytes()
    generated = json.loads(staged_history.read_text())
    assert generated == {'schema': 1, 'sources': dict(old_history['sources'], **additions)}
    assert len(generated['sources']) == 18
    shutil.copyfile(staged_history, history)
    return {'old_records': 14, 'authentic_records_added': 4, 'restored_records': 18,
            'original_commit': original_commit, 'added_hashes': sorted(additions),
            'normal_generated_history_equals_exact_union': True,
            'actual_english_and_nine_locales_untouched_at_restore': True}
