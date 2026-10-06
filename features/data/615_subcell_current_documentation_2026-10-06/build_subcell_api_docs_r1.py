from pathlib import Path
import hashlib
import json
from sphinx.cmd.build import build_main

scratch = Path('/media/carruthers/mnt3/codex/scratch/docs-completion-20261005')
out = scratch / 'subcell-api-docs-all-r1'
assert not out.exists()
inventory = json.loads((scratch / 'subcell-rybg-documentation-inventory-r4.json').read_text())
digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
source_sha = digest(Path('spacr/embeddings.py'))
assert source_sha == inventory['source_sha256']
catalogs = {path.name: digest(path) for path in Path('docs/source/_static/i18n/api').glob('*.json')}
assert len(catalogs) == 10
for path in Path('docs/source/_static/i18n/api').glob('*.json'):
    assert len(json.loads(path.read_text())['symbols']) == 13182
rc = build_main(['-W', '-E', '-b', 'html', '--keep-going', 'docs/source', str(out)])
assert digest(Path('spacr/embeddings.py')) == source_sha
assert {path.name: digest(path) for path in Path('docs/source/_static/i18n/api').glob('*.json')} == catalogs
assert rc == 0, rc
(scratch / 'subcell-full-Sphinx-r1.json').write_text(json.dumps({'passed': True, 'application_source_sha256': source_sha, 'catalogs_sha256': catalogs, 'normal_strict_complete_Sphinx': True, 'exit_code': rc}, indent=2) + '\n')
