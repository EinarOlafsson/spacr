from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys

scratch = Path('/media/carruthers/mnt3/codex/scratch/docs-completion-20261005')
root = Path.cwd()
names = subprocess.check_output(['git', 'ls-files', '-z']).split(b'\0')
total = sum((root / os.fsdecode(name)).stat().st_size for name in names if name)
size = round(total / 1048576)
assert 1020 < size < 1040
text = Path('README.rst').read_text()
old_line = 'The nightly tracked tree is a 821 MB checkout (measured 2026-09-26),'
assert text.count(old_line) == 1
text = text.replace(old_line, f'The nightly tracked tree is a {size} MB checkout (measured 2026-10-05),')
assert text.count('# either, and no docs, tests, tools or example data.') == 1
text = text.replace('# either, and no docs, tests, tools or example data.',
                    '# either, and no docs, tests, tools, features or example data.')
Path('README.rst').write_text(text)
for language in ('sv', 'de', 'es', 'zh_CN', 'pt', 'hi', 'ko', 'is', 'fr'):
    prior = json.loads(Path(f'docs/i18n/reviewed/readme/{language}/2026-09-26-checkout-size.json').read_text())
    old = prior['records'][0]
    source = old['source'].replace('821 MB', f'{size} MB').replace('2026-09-26', '2026-10-05')
    target = old['translation'].replace('821', str(size)).replace('2026-09-26', '2026-10-05')
    assert source != old['source'] and target != old['translation']
    payload = {'schema': 1, 'language': language,
        'review_method': 'Direct Codex AI numeric review against current tracked files and actual nightly clone measurements; prior reviewed wording preserved. No native-speaker signoff.',
        'records': [{'label': 'dated-clone-and-current-checkout-size', 'source': source,
                     'source_sha256': hashlib.sha256(source.encode()).hexdigest(), 'translation': target}]}
    dest = Path(f'docs/i18n/reviewed/readme/{language}/2026-10-05-checkout-size.json')
    assert not dest.exists()
    dest.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n')
    print(language, 'dated checkout size source-bound review', flush=True)
proof = {'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
         'tracked_bytes_before_readme_edit': total, 'stated_MiB': size,
         'historical_2026_09_15_download_claims_retained': True,
         'normal_measurement_log': str(scratch / 'clone-measure-20261005-r1.log'),
         'normal_measurement_download_columns_not_used': 'The helper double-counts repeated 100% progress lines, clips full-clone totals at 32-bit limits and does not see checkout-triggered partial-clone downloads. Current checkout sizes and sparse exclusions were measured; download claims remain explicitly historical.'}
(scratch / 'readme-checkout-size-source-proof.json').write_text(json.dumps(proof, indent=2) + '\n')
print('Measured current tracked checkout:', size, 'MiB', flush=True)
