from pathlib import Path
import hashlib
import json
import subprocess
import sys

scratch = Path('/media/carruthers/mnt3/codex/scratch/docs-completion-20261005')
sys.path.insert(0, str(Path('tools').resolve()))
import build_documentation_i18n as api

commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
source = api.public_docstrings()
expected = {'spacr.doctor', 'spacr.doctor.check_qt_extra', 'spacr.qt.run'}
reports = {}
for language in ('en', 'sv', 'de', 'es', 'zh_CN', 'pt', 'hi', 'ko', 'is', 'fr'):
    path = Path(f'docs/source/_static/i18n/api/{language}.json')
    before = json.loads(subprocess.check_output(['git', 'show', f'{commit}:{path}']))['symbols']
    data = path.read_bytes()
    after = json.loads(data)['symbols']
    assert before.keys() == after.keys() and len(after) == 13181
    changed = {symbol for symbol in before if before[symbol] != after[symbol]}
    assert changed == expected, (language, changed)
    for symbol in expected:
        assert after[symbol]['source_sha256'] == api._source_hash(source[symbol])
        if language != 'en':
            assert after[symbol]['translation_source_blocks_sha256'] == api._translation_source_block_hashes(source[symbol])
    assert path.read_bytes() == data
    reports[language] = {'catalog_sha256': hashlib.sha256(data).hexdigest(), 'symbols': 13181, 'changed': sorted(changed), 'other_complete_records_preserved': 13178, 'new_source_and_translation_context_hashes_current': True}
    print(language, 'exactly three current GUI-install records changed; 13,178 complete records preserved', flush=True)
(scratch / 'core-gui-api-preservation-proof.json').write_text(json.dumps({'source_commit': commit, 'languages': reports}, indent=2) + '\n')
