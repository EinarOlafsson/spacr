from pathlib import Path
import json
import runpy
scratch = Path('/media/carruthers/mnt3/codex/scratch/docs-completion-20261005')
sources = json.loads((scratch / 'subcell-rybg-documentation-inventory-r4.json').read_text())['new_runtime']
rows = {}
for language in ('sv', 'de', 'es'):
    values = runpy.run_path('spacr/qt/i18n_catalogs/' + language + '.py')['UI']
    rows[language] = [values[source] for source in sources]
    print(language, json.dumps(rows[language], ensure_ascii=False, indent=2))
(scratch / 'subcell-rybg-runtime-first-candidates-r1.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2) + '\n')
