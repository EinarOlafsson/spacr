import hashlib
import importlib.abc
import json
import pathlib
import subprocess
import sys
import traceback

root = pathlib.Path('/mnt/wd4tb/spacr-worktrees/codex-n703-cache-cleanup-gui-20261015')
scratch = pathlib.Path('/mnt/wd4tb/scratch/n700-provider-core-authoring-20261015')
class RefuseModelImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'transformers', 'sentencepiece', 'huggingface_hub', 'cellpose', 'tensorflow'}:
            raise RuntimeError('Forbidden model/provider import: ' + fullname)
sys.meta_path.insert(0, RefuseModelImports())
def network_guard(event, args):
    if event in {'socket.connect', 'socket.getaddrinfo'}:
        raise RuntimeError('Forbidden network event: ' + event)
sys.addaudithook(network_guard)
sys.path.insert(0, str(root / 'tools'))
import build_documentation_i18n as tool
import build_i18n_catalogs as runtime
def refuse_collection(*args, **kwargs):
    raise RuntimeError('Forbidden full canonical collection: retain guarded refusal; no substitute context')
runtime.canonical_sources = refuse_collection
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
paths = subprocess.check_output(['git', 'ls-files', '-z'], cwd=root).decode().split('\0')
paths = [p for p in paths if p.startswith(('spacr/', 'tools/', 'docs/i18n/', 'docs/source/_static/i18n/', 'packaging/i18n/')) or p.startswith('README')]
def pins():
    return {p: sha(root / p) for p in paths if (root / p).is_file()}
before = pins()
docs_all = json.loads(pathlib.Path('/mnt/wd4tb/scratch/ci-providers-api-label-diagnosis-20261015/diagnosis.json').read_text())['public_provider_current_sources']
inventory = json.loads((root / 'features/data/n700_provider_core_api_drafts_2026-10-15/normal-author-sources.json').read_text())
docs = {key: docs_all[key] for key in inventory}
assert len(docs) == 15
result = {'scope': 'Guarded ordinary 15-symbol reviewed-input preflight only; not full API audit', 'head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(), 'symbols': list(docs), 'languages': {}, 'before': before}
try:
    for language in ['sv', 'de', 'es', 'zh_CN', 'fr', 'pt', 'hi', 'ko', 'is']:
        payload = json.loads((root / 'docs/i18n/reviewed/api' / language / '2026-10-15-n700-provider-core.json').read_text())
        checks = []
        for record in payload['records']:
            checks.append({'label': record['label'], 'source_valid': tool._reviewed_api_block_valid(record['source'], record['translation'], language), 'context_valid': tool._reviewed_api_block_valid(record['context'], record['translation'], language)})
        result['languages'][language] = {'individual': checks}
        try:
            reviewed = tool.reviewed_api_block_translations(docs, language)
            result['languages'][language]['ordinary_loader'] = {'status': 'PASS', 'source_count': len(reviewed)}
        except Exception as exc:
            result['languages'][language]['ordinary_loader'] = {'status': 'REFUSED', 'type': type(exc).__name__, 'message': str(exc)}
            if 'Forbidden' in str(exc):
                raise
    result['status'] = 'PASS' if all(all(row['source_valid'] and row['context_valid'] for row in language['individual']) and language['ordinary_loader']['status'] == 'PASS' for language in result['languages'].values()) else 'REFUSED'
except Exception as exc:
    result['status'] = 'REFUSED'
    result['exception'] = {'type': type(exc).__name__, 'message': str(exc), 'traceback': traceback.format_exc()}
finally:
    result['after'] = pins()
    result['all_input_bytes_unchanged'] = result['before'] == result['after']
    (scratch / 'guarded-preflight-result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({key: value for key, value in result.items() if key not in {'before', 'after'}}, ensure_ascii=False, indent=2))
assert result['all_input_bytes_unchanged']
raise SystemExit(0 if result['status'] == 'PASS' else 1)
