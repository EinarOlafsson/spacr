from pathlib import Path
import hashlib
import json
import build_documentation_i18n as api
import write_reviewed_api_record as writer
scratch = Path('/media/carruthers/mnt3/codex/scratch/docs-completion-20261005')
inventory = json.loads((scratch / 'subcell-rybg-documentation-inventory-r4.json').read_text())
docs = api.public_docstrings()
allowed = set(inventory['API_changed'])
report = {}
for language in ('sv', 'de', 'es', 'zh_CN', 'pt', 'hi', 'ko', 'is', 'fr'):
    changes = []
    folder = Path('docs/i18n/reviewed/api') / language
    for path in sorted(folder.glob('*.json')):
        raw = path.read_bytes()
        doc = json.loads(raw)
        moves = []
        for row in doc['records']:
            if row.get('retired'):
                continue
            symbol, _, old_index = row['label'].rpartition('#')
            if symbol not in allowed:
                continue
            blocks, _ = api.translatable_blocks(docs[symbol])
            matches = [index for index, block in enumerate(blocks) if block == row['source']]
            if len(matches) == 1:
                label = symbol + '#' + str(matches[0])
                if label == row['label']:
                    continue
                resolved = writer.record(label, row['translation'])
                for field in ('source', 'source_sha256', 'context', 'translation'):
                    assert resolved[field] == row[field], (language, field, row)
                assert api._reviewed_api_block_valid(resolved['source'], resolved['translation'], language)
                moves.append({'old_label': row['label'], 'current_label': label,
                              'canonical_source_hash_context_translation_exact': True})
                row['label'] = label
            else:
                assert not matches, (language, row)
                row['retired'] = True
                row['retired_reason'] = '2026-10-06 SubCell four-plane documentation replaces this exact source; complete original review evidence is retained in the adjacent archive.'
                moves.append({'retired_label': row['label'], 'source_sha256': row['source_sha256']})
        if not moves:
            continue
        archive = folder / 'archive/2026-10-06-subcell-rybg-channel-count' / path.name
        archive.parent.mkdir(parents=True, exist_ok=True)
        assert not archive.exists()
        archive.write_bytes(raw)
        assert archive.read_bytes() == raw
        path.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + '\n')
        changes.append({'path': str(path), 'complete_original_archive': str(archive),
                        'complete_original_sha256': hashlib.sha256(raw).hexdigest(), 'moves': moves})
    assert changes, language
    report[language] = changes
    api.reviewed_api_block_translations(docs, language)
    print('PASS exact-source review label rebinding and normal full review admission', language)
(scratch / 'subcell-rybg-API-review-label-rebinding-r3.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
