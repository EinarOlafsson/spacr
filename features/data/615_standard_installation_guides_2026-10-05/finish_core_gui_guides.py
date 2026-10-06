import hashlib
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, 'tools')
import build_guide_i18n as g

scratch = Path('/media/carruthers/mnt3/codex/scratch/docs-completion-20261005')
pot = scratch / 'core-gui-guide-gettext'
targets = json.loads((scratch / 'core-gui-guide-targets.json').read_text())
source_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
prepared = {}
for language in g.LANGUAGES:
    glossary = g.build_glossary(language, pot)
    worklist = scratch / f'core-gui-guide-worklist-{language}.json'
    assert g.export_worklist(language, worklist) == 6
    rows = json.loads(worklist.read_text())
    assert [r['domain'] for r in rows] == ['features', 'features', 'features', 'index', 'installer_guide', 'installer_guide']
    filled = []
    for index, (row, draft) in enumerate(zip(rows, targets[language])):
        target = draft
        for token, name in [('box', 'Box'), ('save', 'Save boxes')]:
            target = target.replace('@' + token + '@', glossary[name])
        if index == 0:
            names = ['Brush', 'Erase', 'Erase object', 'Wand +', 'Wand −', 'Draw', 'Box', 'Divide', 'Zoom', 'Recrop', 'Ruler', 'Levels']
            import re
            pattern = re.compile('|'.join(re.escape(n) for n in sorted(names, key=len, reverse=True)))
            target = pattern.sub(lambda m: glossary.get(m.group(), m.group()), target)
        assert '@' not in target
        assert not g.message_problems(row['msgid'], target, glossary), (language, index, g.message_problems(row['msgid'], target, glossary))
        filled.append(target)
    before = {str(p): {m.id:m.string for m in g.read_catalog(p) if m.id and m.string and not m.fuzzy} for p in (g.LOCALE_DIR / language / 'LC_MESSAGES').glob('*.po')}
    prepared[language] = rows, filled, before

proof = {}
for language, (rows, filled, before) in prepared.items():
    strings = scratch / f'core-gui-guide-filled-{language}.json'
    strings.write_text(json.dumps(filled, ensure_ascii=False, indent=2) + '\n')
    count, errors = g.import_worklist(language, scratch / f'core-gui-guide-worklist-{language}.json', strings=strings, reviewer='codex')
    assert count == 6 and not errors, (language, count, errors)
    preserved = 0
    for path, previous in before.items():
        current = {m.id:m.string for m in g.read_catalog(Path(path)) if m.id}
        for source, target in previous.items():
            assert current[source] == target, (language, source)
            preserved += 1
    review = {'schema':1, 'language':language, 'source_commit':source_commit,
              'review_method':'Direct Codex AI technical translation and review; no independent peer or native-speaker signoff. Core Qt installation, optional extras, eleven tools and independent Box/YOLO export checked; displayed bold controls inserted from current runtime glossary.',
              'source_files':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in map(Path, ['docs/source/features.rst','docs/source/index.rst','docs/source/installer_guide.rst'])},
              'records':[{'domain':r['domain'],'source':r['msgid'],'source_sha256':hashlib.sha256(r['msgid'].encode()).hexdigest(),'target':t,'ui':r['ui']} for r,t in zip(rows,filled)],
              'preserved_existing_translations':preserved}
    dest = Path(f'docs/i18n/reviewed/guides/{language}/2026-10-05-core-gui-boxes.json')
    dest.write_text(json.dumps(review,ensure_ascii=False,indent=2)+'\n')
    proof[language] = {'new_messages':count,'preserved':preserved}
    print(language, proof[language], flush=True)
audit = g.audit(pot, g.LANGUAGES)
for language, result in audit['languages'].items():
    assert result['total'] == result['translated'] == 2965, (language,result)
    assert not result['stale'] and not result['invalid'] and not result['label_missing'], (language,result)
(scratch/'core-gui-guide-proof.json').write_text(json.dumps({'source_commit':source_commit,'reviews':proof,'audit':audit},ensure_ascii=False,indent=2)+'\n')
print('PASS: all nine languages, six current passages, 2965 current messages; every prior accepted translation retained',flush=True)
