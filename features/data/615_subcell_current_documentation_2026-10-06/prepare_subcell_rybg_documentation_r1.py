from pathlib import Path
import hashlib
import json
import runpy
import shutil
import build_documentation_i18n as api
import build_i18n_catalogs as runtime
scratch = Path('/media/carruthers/mnt3/codex/scratch/docs-completion-20261005')
base = scratch / 'subcell-rybg-documentation-baseline-r1'
base.mkdir(exist_ok=False)
for label, folder in [('api', Path('docs/source/_static/i18n/api')), ('runtime', Path('spacr/qt/i18n_catalogs'))]:
    shutil.copytree(folder, base / label, ignore=shutil.ignore_patterns('__pycache__'))
for lane in ('api', 'runtime'):
    shutil.copytree(Path('docs/i18n/reviewed') / lane, base / ('reviewed-' + lane))
old = json.loads((base / 'api/en.json').read_text())['symbols']
current = api.public_docstrings()
added = sorted(set(current) - set(old))
removed = sorted(set(old) - set(current))
changed = sorted(key for key in set(current) & set(old) if current[key] != old[key]['text'])
assert not removed
messages = []
for key in added + changed:
    blocks, _ = api.translatable_blocks(current[key])
    prior = [] if key in added else api.translatable_blocks(old[key]['text'])[0]
    for index, block in enumerate(blocks):
        if block not in prior and api._api_block_requires_translation(block):
            messages.append({'label': key + '#' + str(index), 'source': block})
canonical = runtime.canonical_sources()
values = set(runtime._unique_translation_sources(canonical))
prior_hashes = runpy.run_path(str(base / 'runtime/en.py'))['SOURCE_HASHES']
new_runtime = sorted(values - set(prior_hashes))
removed_runtime = sorted(set(prior_hashes) - values)
report = {'baseline_commit': '58df6f5b9', 'integrated_Home_commit': '3ce39ac7b',
          'source_sha256': hashlib.sha256(Path('spacr/embeddings.py').read_bytes()).hexdigest(),
          'old_API_symbols': len(old), 'current_API_symbols': len(current), 'API_added': added,
          'API_changed': changed, 'API_removed': removed, 'new_API_blocks': messages,
          'new_runtime': new_runtime, 'retired_runtime': removed_runtime}
(scratch / 'subcell-rybg-documentation-inventory-r1.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
print('API added:', added)
print('API changed:', changed)
print('New translatable API blocks:', len(messages))
print('New runtime messages:', len(new_runtime))
print('Retired runtime messages:', removed_runtime)
for row in messages:
    print(row)
for source in new_runtime:
    print('UI:', source)
