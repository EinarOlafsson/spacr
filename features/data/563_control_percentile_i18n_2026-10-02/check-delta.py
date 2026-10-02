import hashlib
import json
import subprocess
from pathlib import Path

p = Path(__file__).parent
targets = json.loads((p / 'final-targets.json').read_text())
source = 'Control percentile'
out = {}
for lang in ['en', *targets]:
    path = Path('spacr/qt/i18n_catalogs') / f'{lang}.py'
    before, after = {}, {}
    exec(subprocess.check_output(['git', 'show', 'HEAD:' + str(path)], text=True), before)
    exec(path.read_text(), after)
    delta = {}
    for table, old in before.items():
        if table.startswith('__') or not isinstance(old, (dict, set, frozenset)):
            continue
        new = after[table]
        if old == new:
            continue
        added, removed = set(new) - set(old), set(old) - set(new)
        changed = {key for key in set(old) & set(new) if isinstance(old, dict) and old[key] != new[key]}
        expected = {('UI', source)} if table == 'SOURCE_HASHES' else {source}
        assert added == expected, (lang, table, added)
        assert not removed and not changed, (lang, table, removed, changed)
        delta[table] = {'added': 1, 'removed': 0, 'changed_existing': 0}
    assert delta, lang
    if lang != 'en':
        assert after['UI'][source] == targets[lang], (lang, after['UI'][source])
    out[lang] = {'delta': delta, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
(p / 'catalog-diff-proof.json').write_text(json.dumps(out, indent=2) + '\n')
print(json.dumps(out, indent=2))
