from pathlib import Path
import json
import sys

sys.meta_path = [finder for finder in sys.meta_path
                 if '__editable__' not in (getattr(finder, '__module__', '') or type(finder).__module__)]
sys.path.insert(0, str(Path.cwd()))
sys.path.insert(0, str(Path('tools').resolve()))
import build_documentation_i18n as builder
import write_reviewed_api_record as writer

scratch = Path('/media/carruthers/mnt3/codex/scratch/docs-completion-20261005')
payload = json.loads((scratch / 'data-art-api-reviewed-targets-r1.json').read_text())
docs = builder.public_docstrings()
prepared = {}
for language, records in payload['languages'].items():
    assert len(records) == 16
    entries = [writer.record(record['label'], record['translation']) for record in records]
    for old, entry in zip(records, entries):
        assert old['source'] == entry['source'] and old['context'] == entry['context']
        assert builder._reviewed_api_block_valid(entry['source'], entry['translation'], language)
        assert builder._reviewed_api_block_valid(entry['context'], entry['translation'], language)
    assert not (writer.REVIEWED / language / '2026-10-05-data-art-replacement.json').exists()
    prepared[language] = entries
for language, entries in prepared.items():
    path = writer.write(language, '2026-10-05-data-art-replacement', entries)
    reviewed = builder.reviewed_api_block_translations(docs, language)
    assert all(reviewed[entry['source']] == entry['translation'] for entry in entries)
    print(language, 'sixteen exact current reviewed inputs admitted through normal helper', flush=True)
print('PASS: all-nine complete current reviewed-input loaders; no generated catalog hand edits', flush=True)
