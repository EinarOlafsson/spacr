import collections
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT = Path.cwd()
OUT = Path('/mnt/wd4tb/scratch/home-b-n698-readme-link-20261015')
BASE = 'd9e71d7be8f5f517048d744a1cc758d813434eb1'
before = json.loads((OUT / 'before-artifacts.json').read_text())
labels = {'en': 'Mask generator', **json.loads((OUT / 'accepted-site-labels.json').read_text())}
records = []
changed_existing = []
allowed = set()
for language, label in labels.items():
    readme = 'README.rst' if language == 'en' else f'docs/i18n/readme/README.{language}.rst'
    sidecar = f'docs/readme_images/{language}/generated-model-zoo.rst'
    allowed.update((readme, sidecar))
    old = subprocess.check_output(['git', 'show', f'{BASE}:{readme}'], text=True)
    new = (ROOT / readme).read_text()
    begin, end = '.. spacr-model-zoo-begin', '.. spacr-model-zoo-end'
    old_head, _, old_rest = old.partition(begin)
    new_head, _, new_rest = new.partition(begin)
    old_models, _, old_tail = old_rest.partition(end)
    new_models, _, new_tail = new_rest.partition(end)
    assert old_head == new_head and old_tail == new_tail, (language, 'foreign README drift')
    old_links = collections.Counter(re.findall(r'href="([^"]+)"', old_models))
    new_links = collections.Counter(re.findall(r'href="([^"]+)"', new_models))
    page_prefix = 'docs/readme_text/' if language == 'en' else '../../readme_text/'
    panel = f'{page_prefix}{language}/generated-model-zoo-3.md'
    assert new_links - old_links == {panel: 1} and not old_links - new_links
    route = '' if language == 'en' else language.replace('_', '-') + '/'
    target = f'https://einarolafsson.github.io/projects/spacr/{route}#mask-generator'
    text_page = ROOT / f'docs/readme_text/{language}/generated-model-zoo-3.md'
    assert text_page.read_text() == f'[{label}](<{target}>)\n'
    old_source = subprocess.check_output(['git', 'show', f'{BASE}:{sidecar}'], text=True)
    assert (ROOT / sidecar).read_text() == old_source.rstrip() + f'\n\n`{label} <{target}>`__\n'
    records.append({'language': language, 'README': readme, 'unchanged_prior_model_links': sum(old_links.values()), 'new_readable_panel': str(text_page.relative_to(ROOT)), 'website_target': target, 'label': label})
for name, digest in before.items():
    current = ROOT / name
    assert current.is_file(), ('removed previous artifact', name)
    if hashlib.sha256(current.read_bytes()).hexdigest() != digest:
        changed_existing.append(name)
        assert name in allowed, ('foreign artifact changed', name)
assert set(changed_existing) == allowed
result = {'status': 'PASS', 'baseline': BASE, 'routes': records, 'old_artifact_count': len(before), 'changed_existing': sorted(changed_existing), 'unchanged_old_artifacts': len(before) - len(changed_existing), 'source_hashes': {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in ['packaging/generate_readme_visuals.py', 'tools/readme_i18n.py', 'tests/test_readme_presentation.py']}, 'publication': 'Private; actual website deployment/readback is a separate owner gate.'}
(OUT / 'artifact-readback.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({'status': result['status'], 'languages': len(records), 'unchanged_old_artifacts': result['unchanged_old_artifacts'], 'changed_existing': len(changed_existing)}))
