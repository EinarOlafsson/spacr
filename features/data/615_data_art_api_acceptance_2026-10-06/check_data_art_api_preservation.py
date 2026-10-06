from pathlib import Path
import hashlib
import json
import subprocess
import sys

sys.path.insert(0, str(Path('tools').resolve()))
import build_documentation_i18n as builder

scratch = Path('/media/carruthers/mnt3/codex/scratch/docs-completion-20261005')
baseline = '3053b081512d6f598a1aba76dac2393de31ba48a'
delta = json.loads((scratch / 'data-art-api-delta-r1.json').read_text())
assert not delta['added'] and not delta['removed']
expected = set(delta['changed'])
assert len(expected) == 12
docs = builder.public_docstrings()
reports = {}
for language in ('en', 'sv', 'de', 'es', 'zh_CN', 'pt', 'hi', 'ko', 'is', 'fr'):
    path = Path(f'docs/source/_static/i18n/api/{language}.json')
    before = json.loads(subprocess.check_output(['git', 'show', f'{baseline}:{path}']))['symbols']
    after = json.loads(path.read_text())['symbols']
    assert before.keys() == after.keys() and len(after) == 13181
    changed = {key for key in before if before[key] != after[key]}
    assert changed == expected, (language, changed ^ expected)
    for key in expected:
        assert after[key]['source_sha256'] == builder._source_hash(docs[key])
        if language != 'en':
            assert after[key]['translation_source_blocks_sha256'] == builder._translation_source_block_hashes(docs[key])
    reports[language] = {'current_symbols': 13181, 'changed_symbols': sorted(expected),
                         'complete_other_records_preserved': 13169,
                         'new_source_and_context_hashes_current': True,
                         'catalog_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    print(language, '12 current records; all 13,169 other complete records preserved', flush=True)
(scratch / 'data-art-api-preservation-proof.json').write_text(json.dumps({'source_commit': baseline, 'languages': reports}, indent=2) + '\n')
