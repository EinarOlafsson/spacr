from pathlib import Path
import copy
import hashlib
import json
import sys

sys.path.insert(0, str(Path('tools').resolve()))
import build_documentation_i18n as builder

scratch = Path('/media/carruthers/mnt3/codex/scratch/docs-completion-20261005')
delta = json.loads((scratch / 'data-art-api-delta-r1.json').read_text())
docs = builder.public_docstrings()
changed = set(delta['changed'])
assert len(changed) == 12
reports = {}
for language in builder.MODEL_SPECS:
    changes = []
    for path in sorted((Path('docs/i18n/reviewed/api') / language).glob('*.json')):
        payload = json.loads(path.read_text())
        original = copy.deepcopy(payload)
        for record in payload['records']:
            if record.get('retired'):
                continue
            symbol, _, raw_index = record['label'].rpartition('#')
            if symbol not in changed:
                continue
            blocks, _ = builder.translatable_blocks(docs[symbol])
            source = record['source']
            index = int(raw_index)
            if index < len(blocks) and blocks[index] == source and builder._api_translation_source(source) == record['context']:
                continue
            previous = copy.deepcopy(record)
            assert record['source_sha256'] == hashlib.sha256(source.encode()).hexdigest()
            if source in blocks and builder._api_translation_source(source) == record['context']:
                record['label'] = symbol + '#' + str(blocks.index(source))
                record.setdefault('label_history', []).append({'label': previous['label'], 'date': '2026-10-05',
                    'reason': 'Unchanged reviewed source and translation moved to another block index after the data-art documentation update.'})
                action = 'relocated_unchanged_source'
            else:
                record['retired'] = True
                record['retired_reason'] = '2026-10-05: replacement data-art documentation rewrites this English block. Original source, translation and reviewer attribution are preserved as historical evidence; the new source requires a new translation and review.'
                action = 'retired_changed_source'
            for field in ('source', 'source_sha256', 'context', 'translation', 'review_basis'):
                if field in previous:
                    assert record[field] == previous[field]
            changes.append({'path': str(path), 'old_label': previous['label'], 'current_label': record['label'],
                            'action': action, 'source_sha256': record['source_sha256'],
                            'original_translation_sha256': hashlib.sha256(record['translation'].encode()).hexdigest()})
        if payload != original:
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n')
    builder.reviewed_api_block_translations(docs, language)
    reports[language] = changes
    print(language, len(changes), 'historical records migrated; complete current review loader passes', flush=True)
(scratch / 'data-art-api-review-migration-proof.json').write_text(json.dumps({'source_commit': '3053b081512d6f598a1aba76dac2393de31ba48a', 'languages': reports}, ensure_ascii=False, indent=2) + '\n')
