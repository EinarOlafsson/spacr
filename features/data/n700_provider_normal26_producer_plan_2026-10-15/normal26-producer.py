import argparse
import gzip
import hashlib
import importlib.abc
import importlib.util
import json
import pathlib
import socket
import subprocess
import sys
import traceback

config_path = pathlib.Path(sys.argv[1]).resolve()
config = json.loads(config_path.read_text())
assert config['ready'] is True, 'Pending plan only: Root must reseal and authorize execution'
root = pathlib.Path(config['root']).resolve()
out = pathlib.Path(config['out']).resolve()
out.mkdir(parents=True, exist_ok=True)
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip() == config['head']
assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=root, text=True)
for name, digest in config['pins'].items():
    assert sha(root / name) == digest, name
for name, digest in config['driver_pins'].items():
    assert sha(config_path.parent / name) == digest, name

class RefuseModels(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'transformers', 'sentencepiece', 'huggingface_hub', 'tensorflow', 'cellpose'}:
            raise RuntimeError('Forbidden model/provider import: ' + fullname)
sys.meta_path.insert(0, RefuseModels())
def audit_guard(event, args):
    if event in {'socket.connect', 'socket.getaddrinfo'}:
        raise RuntimeError('Forbidden network event: ' + event)
    if event == 'subprocess.Popen':
        executable, argv = args[:2]
        if not (pathlib.Path(str(executable)).name == 'git' or
                (str(executable) == '/sbin/ldconfig' and argv == ['/sbin/ldconfig', '-p'])):
            raise RuntimeError('Forbidden subprocess: ' + repr(argv))
sys.addaudithook(audit_guard)
sys.path.insert(0, str(root / 'tools'))
import build_documentation_i18n as tool
import build_i18n_catalogs as runtime
def refuse(*args, **kwargs):
    raise RuntimeError('Forbidden decoder or full runtime canonical collection; preserve refusal')
runtime.canonical_sources = refuse
tool._translate_blocks = refuse
tool._translate_batches = refuse

packet_dir = root / 'features/data/n700_provider_utilities_api_inputs_2026-10-15'
diagnosis = json.loads((packet_dir / 'diagnosis.json').read_text())
split = json.loads((packet_dir / 'author-split.json').read_text())
selected = split['easy_symbols'] + split['ci_symbols']
assert len(selected) == len(set(selected)) == 26
docs = {key: diagnosis['public_provider_current_sources'][key] for key in selected}
api = root / 'docs/source/_static/i18n/api'
before_api = {language: json.loads((api / (language + '.json')).read_text()) for language in ['en', *tool.MODEL_SPECS]}
en_before = (api / 'en.json').read_bytes()
full_current_texts = {key: value['text'] for key, value in before_api['en']['symbols'].items()}
assert all(full_current_texts[key] == source for key, source in docs.items())
restore_spec = importlib.util.spec_from_file_location('authentic_history_restore', config_path.parent / 'restore-authentic-history.py')
restore_module = importlib.util.module_from_spec(restore_spec)
restore_spec.loader.exec_module(restore_module)
result = {'head': config['head'], 'selected_symbols': selected, 'phases': [], 'status': 'RUNNING'}
def save():
    (out / 'receipt.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
def immutable_inputs():
    allowed = {'docs/i18n/api_source_history.json', *['docs/source/_static/i18n/api/' + language + '.json' for language in tool.MODEL_SPECS]}
    for name, digest in config['pins'].items():
        if name not in allowed:
            assert sha(root / name) == digest, 'Changed protected input: ' + name
try:
    result['restoration'] = restore_module.restore(tool, root, out / 'restore-stage', full_current_texts, packet_dir / 'historical-en-prehosted-records.json')
    restored_history = (root / 'docs/i18n/api_source_history.json').read_bytes()
    result['restored_history_sha256'] = hashlib.sha256(restored_history).hexdigest()
    result['phases'].append('normal staged18-record history restoration PASS')
    save()
    prepared = {}
    preflights = {}
    reuse = json.loads((root / 'features/data/n700_provider_core_api_drafts_2026-10-15/reused-blocks.json').read_text())
    for language in tool.MODEL_SPECS:
        reviewed = tool.reviewed_api_block_translations(docs, language)
        _stored, historical, from_head, unproven = tool._proven_api_history(docs, language)
        assert unproven == 0, (language, unproven)
        for key, source in docs.items():
            blocks, _ = tool.translatable_blocks(source)
            assert all(block in reviewed or block in historical.get(key, {}) for block in blocks), (language, key, 'uncovered block')
        translated = tool._translate_api_documents(docs, language, pathlib.Path('/nonexistent-model-root-never-used'), argparse.Namespace(device='cpu', batch_size=24, beams=4, threads=1), reuse_history=True)
        assert set(translated) == set(docs)
        for key, source in docs.items():
            source_blocks, _ = tool.translatable_blocks(source)
            target_blocks, _ = tool.translatable_blocks(translated[key])
            assert len(source_blocks) == len(target_blocks), (language, key, 'layout')
            for block, target in zip(source_blocks, target_blocks):
                if block in reviewed:
                    assert target == reviewed[block]
                    assert tool._reviewed_api_block_valid(block, target, language)
                    assert tool._reviewed_api_block_valid(tool._api_translation_source(block), target, language)
                else:
                    assert historical[key][block] == target
        for key, rows in reuse.items():
            target_blocks, _ = tool.translatable_blocks(translated[key])
            for row in rows:
                assert target_blocks[row['index']] == row['reused_targets'][language]['translation']
        prepared[language] = translated
        preflights[language] = {'status': 'PASS', 'reviewed_sources': len(reviewed), 'history_from_head': from_head, 'history_unproven': unproven, 'reused_target_values': 8}
    result['preflights'] = preflights
    result['phases'].append('ALL9 normal history/translation preflights PASS before first locale write')
    immutable_inputs()
    save()
    for language, translated in prepared.items():
        tool.write_language(docs, language, translated, preserve_other_symbols=True)
        current = json.loads((api / (language + '.json')).read_text())
        old = before_api[language]
        assert {k:v for k,v in current.items() if k != 'symbols'} == {k:v for k,v in old.items() if k != 'symbols'}
        assert {k:v for k,v in current['symbols'].items() if k not in docs} == {k:v for k,v in old['symbols'].items() if k not in docs}
        for key, source in docs.items():
            record = current['symbols'][key]
            assert record['text'] == translated[key]
            assert record['source_sha256'] == tool._source_hash(source)
            assert record['source_blocks_sha256'] == tool._source_block_hashes(source)
            assert record['translation_source_blocks_sha256'] == tool._translation_source_block_hashes(source)
    assert (api / 'en.json').read_bytes() == en_before
    assert (root / 'docs/i18n/api_source_history.json').read_bytes() == restored_history
    immutable_inputs()
    result['phases'].append('normal9 language writers +foreign/header/English/history preservation PASS')
    save()
    print('PHASE unchanged FULL9 ordinary API+README audit', flush=True)
    full_docs = tool.public_docstrings()
    snapshot = json.dumps(full_docs, ensure_ascii=False, indent=2).encode('utf-8')
    assert json.loads(snapshot) == full_docs
    snapshot_path = out / 'full-current-source-docs.json.gz'
    snapshot_path.write_bytes(gzip.compress(snapshot, mtime=0))
    assert json.loads(gzip.decompress(snapshot_path.read_bytes())) == full_docs
    result['full_source_snapshot'] = {'path': snapshot_path.name,
        'json_sha256': hashlib.sha256(snapshot).hexdigest(),
        'gzip_sha256': sha(snapshot_path), 'symbols': len(full_docs)}
    english_symbols = before_api['en']['symbols']
    missing_english = sorted(set(full_docs) - set(english_symbols))
    extra_english = sorted(set(english_symbols) - set(full_docs))
    stale_english = sorted(key for key in set(full_docs) & set(english_symbols)
        if english_symbols[key].get('text') != full_docs[key]
        or english_symbols[key].get('source_sha256') != tool._source_hash(full_docs[key])
        or english_symbols[key].get('source_blocks_sha256') != tool._source_block_hashes(full_docs[key]))
    result['global_english_diagnostic'] = {'missing': missing_english,
        'extra': extra_english, 'stale': stale_english,
        'equal': not (missing_english or extra_english or stale_english)}
    save()
    assert all(full_docs[key] == source for key, source in docs.items())
    result['full_source_inventory_symbols'] = len(full_docs)
    result['full9_audit_exit'] = tool.audit(full_docs, list(tool.MODEL_SPECS))
    result['status'] = 'PASS' if result['full9_audit_exit'] == 0 else 'REFUSED'
except Exception as exc:
    result['status'] = 'REFUSED'
    result['error'] = {'type': type(exc).__name__, 'message': str(exc), 'traceback': traceback.format_exc()}
finally:
    try:
        immutable_inputs()
        result['protected_pins_unchanged'] = True
    except Exception as exc:
        result['protected_pins_unchanged'] = False
        result['preservation_error'] = repr(exc)
        result['status'] = 'REFUSED'
    result['actual_changed_paths'] = subprocess.check_output(['git', 'diff', '--name-only'], cwd=root, text=True).splitlines()
    allowed = {'docs/i18n/api_source_history.json', *['docs/source/_static/i18n/api/' + language + '.json' for language in tool.MODEL_SPECS]}
    assert set(result['actual_changed_paths']) <= allowed, result['actual_changed_paths']
    save()
print(json.dumps({key:value for key,value in result.items() if key != 'selected_symbols'}, ensure_ascii=False, indent=2), flush=True)
raise SystemExit(0 if result['status'] == 'PASS' else 1)
