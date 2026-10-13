"""Check complete old API records and the exact four YOLO arrivals."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path('/media/carruthers/mnt3/codex/spacr-worktrees/docs-completion-20261005')
SCRATCH = Path('/media/carruthers/mnt3/codex/scratch/docs-completion-20261005')
sys.path.insert(0, str(ROOT / 'tools'))
import build_documentation_i18n as api

expected = {'spacr.qt.mask_engine.' + name for name in (
    'export_yolo_boxes', 'load_yolo_boxes', 'save_yolo_boxes', 'yolo_box_lines')}
source = api.public_docstrings()
commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
results = {}
for language in ('en', 'sv', 'de', 'es', 'zh_CN', 'pt', 'hi', 'ko', 'is', 'fr'):
    relative = f'docs/source/_static/i18n/api/{language}.json'
    before = json.loads(subprocess.check_output(['git', 'show', f'{commit}:{relative}'], cwd=ROOT))['symbols']
    path = ROOT / relative
    original_bytes = path.read_bytes()
    after = json.loads(original_bytes)['symbols']
    assert len(before) == 13177 and len(after) == 13181
    assert after.keys() - before.keys() == expected
    assert not before.keys() - after.keys()
    assert all(record == after[symbol] for symbol, record in before.items()), language
    for symbol in expected:
        assert after[symbol]['source_sha256'] == api._source_hash(source[symbol])
        if language != 'en':
            assert after[symbol]['translation_source_blocks_sha256'] == api._translation_source_block_hashes(source[symbol])
    assert path.read_bytes() == original_bytes, 'Catalog changed during preservation verification'
    results[language] = {'catalog_sha256': hashlib.sha256(original_bytes).hexdigest(),
                         'previous': 13177, 'current': 13181, 'added': sorted(expected), 'removed': [],
                         'all_previous_complete_records_preserved': True,
                         'new_source_and_translation_context_hashes_current': True}
    print(language, 'all 13,177 old complete records preserved; exactly four source-current arrivals', flush=True)
(SCRATCH / '662-api-preservation-proof.json').write_text(json.dumps(
    {'baseline_commit': commit, 'languages': results, 'final_strict_generator_audit_pending': True},
    ensure_ascii=False, indent=2) + '\n')
