from pathlib import Path
import hashlib
import json
import shutil
import sys

ROOT = Path('/mnt/wd4tb/spacr-worktrees/codex-n703-debian-adapter-20261015')
BASE = Path('/mnt/wd4tb/scratch/n702-cluster-guide-20261015')
sys.path.insert(0, str(ROOT / 'tools'))
import build_guide_i18n as builder


def identity(catalog):
    return {m.id: [m.string, sorted(m.flags), m.user_comments]
            for m in catalog if m.id}


source = json.loads((BASE / 'draft.json').read_text())
targets = json.loads((BASE / 'targets.json').read_text())
ssh_targets = json.loads((BASE / 'existing-ssh-targets.json').read_text())
ssh_sources = ['use a verified persistent SSH session with GUI keyboard-interactive prompts instead of the external OpenSSH command.', 'non-secret SSH configuration overrides, such as user, port, identity-file path and jump host; passwords and OTPs are forbidden.']
templates = BASE / 'templates'
templates.mkdir(exist_ok=True)
shutil.copyfile(BASE / 'pot/distributed_execution.pot', templates / 'distributed_execution.pot')
locales = BASE / 'generated-locales'
inputs = BASE / 'inputs'
inputs.mkdir(exist_ok=True)
receipt = {'normal_functions': ['extract', 'update_language', 'import_worklist', 'audit'],
           'scope': 'Two new N702 messages plus two pre-existing missing N699 SSH messages; all prior catalog messages exact.',
           'source_sha256': hashlib.sha256((ROOT / 'docs/source/distributed_execution.rst').read_bytes()).hexdigest(),
           'locales': {}}
pending = []
for lang in builder.LANGUAGES:
    relative = Path(lang) / 'LC_MESSAGES/distributed_execution.po'
    original = ROOT / 'docs/i18n/guides' / relative
    generated = locales / relative
    generated.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(original, generated)
    prior = identity(builder.read_catalog(original))
    rows = [{'domain': 'distributed_execution', 'msgid': msgid, 'msgstr': target}
            for msgid, target in zip([source['title'], source['paragraph']] + ssh_sources, targets[lang] + ssh_targets[lang])]
    for row in rows:
        assert row['msgid'] not in prior
        assert not builder.message_problems(row['msgid'], row['msgstr'], builder.load_glossary(lang)), lang
    worklist = inputs / f'{lang}.json'
    worklist.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + '\n')
    builder.update_language(lang, templates, locale_dir=locales)
    applied, rejected = builder.import_worklist(lang, worklist, locale_dir=locales, reviewer='codex')
    assert applied == 4 and not rejected, (lang, applied, rejected)
    current = identity(builder.read_catalog(generated))
    assert set(current) == set(prior) | {r['msgid'] for r in rows}, lang
    assert all(current[k] == v for k, v in prior.items()), lang
    for row in rows:
        assert current[row['msgid']][0] == row['msgstr']
        assert 'fuzzy' not in current[row['msgid']][1]
    receipt['locales'][lang] = {'prior_messages_preserved': len(prior), 'added_reviewed': 4,
                              'generated_sha256': hashlib.sha256(generated.read_bytes()).hexdigest()}
    pending.append((generated, original))
report = builder.audit(templates, builder.LANGUAGES, locale_dir=locales)
(BASE / 'normal-page-audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
for lang, result in report['languages'].items():
    assert not result['invalid'] and not result['label_missing'], (lang, result)
    assert result['stale'] == 0 and result['translated'] == result['total'], (lang, result)
for generated, original in pending:
    shutil.copyfile(generated, original)
receipt['passed'] = True
(BASE / 'generation-receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(receipt, sort_keys=True))
