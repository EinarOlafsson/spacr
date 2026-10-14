#!/usr/bin/env python3
"""Mechanically bind future notice drafts; no linguistic/native acceptance."""
from pathlib import Path
import sys

WORKTREE = Path('/media/carruthers/mnt3/claude/spacr-lesson20-report-20261013')
for _entry in (str(WORKTREE), str(WORKTREE / 'tools')):
    while _entry in sys.path:
        sys.path.remove(_entry)
sys.path[:0] = [str(WORKTREE), str(WORKTREE / 'tools')]

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import re
import string

ACCEPTED = WORKTREE / 'features/data/700_home_future_notice_design_acceptance_2026-10-14/accepted-future-design-english.json'
ACCEPTED_SHA = '9e2f2511745a8e692cb8172c18eebc79c6755468a2798bd0e60336a33850b72a'
OBLIGATIONS = Path('/media/carruthers/mnt3/claude/n676_stages/work/n700_notice_locale_preflight_corrected_20261014.json')
OBLIGATIONS_SHA = '36b29301ea33a917dfeb4f4bc25312f15e1214f8e115fcd84ce10c02296e0166'
LOCALES = ('de', 'es', 'fr', 'hi', 'is', 'ko', 'pt', 'sv', 'zh_CN')
IDS = tuple(f'ZC{i:02d}' for i in range(1, 15))
STATUS = 'AI technical draft; not native/canonical acceptance'
TOP_KEYS = {'schema_version', 'locale', 'status', 'records'}
RECORD_KEYS = {'id', 'source_english', 'source_sha256_utf8', 'display_predicate',
               'translated_body', 'translated_sha256_utf8', 'semantic_obligations_preserved'}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'duplicate JSON key: {key}')
        result[key] = value
    return result


def load(path):
    raw = path.read_bytes()
    data = json.loads(raw, object_pairs_hook=unique_object)
    return data, {'path': str(path), 'sha256': sha(raw), 'bytes': len(raw)}


def placeholders(text):
    """Count named Python-format, %-format and shell-variable placeholders.

    These accepted bodies currently have none. Future accepted bodies with
    other placeholder syntaxes require explicit verifier review, not guessing.
    """
    found = Counter()
    for _, field, spec, conversion in string.Formatter().parse(text):
        if field is not None:
            found[('format', field, spec, conversion)] += 1
    for token in re.findall(r'%\([A-Za-z_][A-Za-z0-9_]*\)[#0 +\-]*[0-9]*(?:\.[0-9]+)?[a-zA-Z]|(?<!%)%[sdr]|\$\{[A-Za-z_][A-Za-z0-9_]*\}|\$[A-Za-z_][A-Za-z0-9_]*', text):
        found[('literal', token)] += 1
    return found


def verify(draft_dir):
    checks, bindings = [], []

    def check(name, condition, detail=None):
        checks.append({'check': name, 'passed': bool(condition), 'detail': detail})

    accepted, binding = load(ACCEPTED)
    bindings.append(binding)
    obligations, binding = load(OBLIGATIONS)
    bindings.append(binding)
    check('accepted complete file SHA', bindings[0]['sha256'] == ACCEPTED_SHA)
    check('corrected obligations complete file SHA', bindings[1]['sha256'] == OBLIGATIONS_SHA)
    # Fail closed before trusting changed English or obligations.
    if not all(c['passed'] for c in checks):
        raise ValueError('Pinned English/obligation file identity changed')
    notices = accepted['exact_notices']
    obligations_rows = obligations['records']
    source = {x['id']: x for x in notices}
    obligation_by_id = {x['id']: x for x in obligations_rows}
    check('exact 14 accepted IDs', len(notices) == 14 and set(source) == set(IDS))
    check('exact 14 obligation IDs', len(obligations_rows) == 14 and set(obligation_by_id) == set(IDS))
    check('exact nine API locales', obligations['target_locales'] == list(LOCALES))
    check('126 planned bodies', obligations['planned_records'] == 126)
    for notice_id in IDS:
        src, obligation = source[notice_id], obligation_by_id[notice_id]
        check(f'{notice_id}: accepted UTF8 hash', sha(src['text'].encode('utf-8')) == src['text_sha256'])
        check(f'{notice_id}: obligations source binding',
              obligation['accepted_English'] == src['text'] and
              obligation['accepted_English_sha256'] == src['text_sha256'] and
              obligation['required_display_predicate'] == src['required_display_predicate'])
    total = 0
    for locale in LOCALES:
        path = draft_dir / f'{locale}.json'
        try:
            draft, binding = load(path)
            bindings.append(binding)
            check(f'{locale}: exact packet fields; titles/activation excluded', isinstance(draft, dict) and set(draft) == TOP_KEYS)
            check(f'{locale}: exact integer schema', type(draft.get('schema_version')) is int and draft['schema_version'] == 1)
            check(f'{locale}: exact locale', draft.get('locale') == locale)
            check(f'{locale}: draft-only status', draft.get('status') == STATUS)
            rows = draft.get('records')
            if not isinstance(rows, list):
                raise ValueError('records must be a list')
            names = [x.get('id') if isinstance(x, dict) else None for x in rows]
            check(f'{locale}: exact unique 14 IDs', len(names) == 14 and Counter(names) == Counter(IDS))
            total += len(rows)
            for index, row in enumerate(rows):
                if not isinstance(row, dict):
                    raise ValueError(f'record {index} is not an object')
                name = row.get('id')
                check(f'{locale}/{name}: exact body-only record fields', set(row) == RECORD_KEYS)
                if name not in source:
                    check(f'{locale}/{name}: known notice', False)
                    continue
                src, obligation = source[name], obligation_by_id[name]
                check(f'{locale}/{name}: exact English/hash', row.get('source_english') == src['text'] and row.get('source_sha256_utf8') == src['text_sha256'])
                # Equality preserves strings OR structured predicate values.
                check(f'{locale}/{name}: exact display predicate', row.get('display_predicate') == src['required_display_predicate'])
                body = row.get('translated_body')
                check(f'{locale}/{name}: nonblank translated body', isinstance(body, str) and bool(body.strip()))
                if isinstance(body, str):
                    check(f'{locale}/{name}: translated UTF8 hash', row.get('translated_sha256_utf8') == sha(body.encode('utf-8')))
                    try:
                        check(f'{locale}/{name}: placeholder multiset', placeholders(body) == placeholders(src['text']))
                    except ValueError as exc:
                        check(f'{locale}/{name}: placeholder parsing', False, str(exc))
                check(f'{locale}/{name}: exact obligation declaration', row.get('semantic_obligations_preserved') == obligation['semantic_obligations'])
        except (OSError, ValueError, KeyError, TypeError) as exc:
            check(f'{locale}: readable valid complete packet', False, str(exc))
    check('126 bodies across nine exact-locale inputs', total == 126)
    failures = [c for c in checks if not c['passed']]
    return {'schema_version': 1, 'created_at_utc': datetime.now(timezone.utc).isoformat(),
            'status': 'mechanical_pass_only' if not failures else 'mechanical_fixes_required',
            'verifier': {'path': str(Path(__file__).resolve()), 'sha256': sha(Path(__file__).read_bytes())},
            'sys_path_first': sys.path[:2], 'source_bindings': bindings,
            'locales': list(LOCALES), 'expected_bodies': 126, 'observed_bodies': total,
            'checks': checks, 'failures': failures,
            'limitations': ['Mechanical binding only; declarations do not prove translated meaning or native language quality.',
                            'No canonical/source application, activation, provider/runtime behavior, cost certification or locale acceptance.',
                            'No app imports, tests, generators, catalogs, network, GUI or GPU used.',
                            'Only exact named locale files are read; titles/helpers and extra record or packet fields are rejected.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--draft-dir', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--output', type=Path, required=True, help='New external receipt; existing file is never overwritten')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output already exists; preserve it and select a new path')
    try:
        receipt = verify(args.draft_dir)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(2, f'Verification failed before complete receipt: {exc}\n')
    with args.output.open('x', encoding='utf-8') as handle:
        json.dump(receipt, handle, indent=2, ensure_ascii=False)
        handle.write('\n')
    print(json.dumps({'output': str(args.output), 'sha256': sha(args.output.read_bytes()),
                      'checks': len(receipt['checks']), 'failures': len(receipt['failures'])}))
    return 1 if receipt['failures'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
