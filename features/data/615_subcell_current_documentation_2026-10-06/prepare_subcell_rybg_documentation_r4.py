from pathlib import Path
import hashlib
import json
import runpy
import shutil
import build_documentation_i18n as api
import build_i18n_catalogs as runtime
scratch = Path('/media/carruthers/mnt3/codex/scratch/docs-completion-20261005')
base = scratch / 'subcell-rybg-documentation-baseline-r1'
assert base.is_dir()
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
current_hashes = runtime._source_hashes(canonical)
new_runtime_keys = sorted(set(current_hashes) - set(prior_hashes))
removed_runtime_keys = sorted(set(prior_hashes) - set(current_hashes))
new_runtime = sorted({key[1] for key in new_runtime_keys})
removed_runtime = sorted({key[1] for key in removed_runtime_keys})
assert set(new_runtime) <= values
assert all(key[0] == 'UI' for key in new_runtime_keys + removed_runtime_keys)
report = {'baseline_commit': '58df6f5b9', 'integrated_Home_commit': '3ce39ac7b',
          'source_sha256': hashlib.sha256(Path('spacr/embeddings.py').read_bytes()).hexdigest(),
          'old_API_symbols': len(old), 'current_API_symbols': len(current), 'API_added': added,
          'API_changed': changed, 'API_removed': removed, 'new_API_blocks': messages,
          'new_runtime': new_runtime, 'new_runtime_keys': new_runtime_keys,
          'retired_runtime': removed_runtime, 'current_runtime_rows': len(current_hashes)}
(scratch / 'subcell-rybg-documentation-inventory-r4.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
print('API added:', added)
print('API changed:', changed)
print('New translatable API blocks:', len(messages))
print('New runtime messages:', len(new_runtime))
print('Retired runtime messages:', removed_runtime)
for row in messages:
    print(row)
for source in new_runtime:
    print('UI:', source)
