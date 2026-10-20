"""Regenerate one tooltip API symbol using existing source-proven targets."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path('/mnt/wd4tb/spacr-worktrees/codex-home-b-tooltip-api-20261014')
SCRATCH = Path('/mnt/wd4tb/scratch/home-b-tooltip-api-20261014')
sys.path.insert(0, str(ROOT / 'tools'))
import build_documentation_i18n as builder

SYMBOL = 'spacr.qt.widgets.hover_tooltip.HoverTooltip.leaveEvent'
OLD_HASH = 'fe4414ab341f2ffd0d4d8200249699f8ef180a75295a041759318d445d0373ff'
review_files = {
    'sv': '2026-09-25-api-debt-008.json',
    'de': '2026-09-25-api-debt-007.json',
    'es': '2026-09-25-api-debt-010.json',
    'zh_CN': '2026-09-25-api-debt-006.json',
    'pt': '2026-09-25-api-debt-010.json',
    'hi': '2026-09-25-api-debt-010.json',
    'ko': '2026-09-25-api-debt-008.json',
    'is': '2026-09-25-api-debt-011.json',
    'fr': '2026-09-25-api-debt-010.json',
}

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

source = builder.public_docstrings()[SYMBOL]
docs = {SYMBOL: source}
english_path = builder.API_DIR / 'en.json'
english_digest = digest(english_path)
english = json.loads(english_path.read_text())['symbols'][SYMBOL]
assert english == builder._english_manifest(docs)['symbols'][SYMBOL]
source_blocks, layout = builder.translatable_blocks(source)
assert len(source_blocks) == 2
history_path = ROOT / 'docs/i18n/api_source_history.json'
history_digest = digest(history_path)
history = builder._archived_english_api_records()[OLD_HASH]
old_blocks, _ = builder.translatable_blocks(history['text'])
assert source_blocks == [old_blocks[0], old_blocks[2]]
report = {'source_commit': subprocess.check_output(
    ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
    'symbol': SYMBOL, 'model_decode_required': False,
    'review': 'AI technical review of paragraph deletion and record reindex; '
              'no human or native-speaker signoff; existing translation provenance retained',
    'locales': {}}

for language, filename in review_files.items():
    path = builder.REVIEWED_API_DIR / language / filename
    before = path.read_text()
    payload = json.loads(before)
    matches = [r for r in payload['records'] if r.get('label') == SYMBOL + '#2']
    assert len(matches) == 1
    assert matches[0]['source'] == source_blocks[1]
    assert matches[0]['source_sha256'] == builder._source_hash(source_blocks[1])
    old_label = '"label": "' + SYMBOL + '#2"'
    new_label = '"label": "' + SYMBOL + '#1"'
    assert before.count(old_label) == 1 and new_label not in before
    path.write_text(before.replace(old_label, new_label))
    after = json.loads(path.read_text())
    matches[0]['label'] = SYMBOL + '#1'
    assert after == payload, 'Only the unchanged review block index may move'

for language in review_files:
    path = builder.API_DIR / f'{language}.json'
    before = json.loads(path.read_text())
    old_record = before['symbols'][SYMBOL]
    assert old_record['source_sha256'] == OLD_HASH
    old_targets, _ = builder.translatable_blocks(old_record['text'])
    assert len(old_targets) == 3
    kept = builder._historical_api_block_translations(history, old_record, language)
    assert kept[source_blocks[0]] == old_targets[0]
    assert builder.reviewed_api_block_translations(docs, language)[source_blocks[1]] == old_targets[2]
    for canonical, target in zip(source_blocks, (old_targets[0], old_targets[2])):
        assert builder._api_block_valid(canonical, target, language)
        assert builder._api_block_valid(builder._api_translation_source(canonical), target, language)
    expected = builder.rebuild_document(layout, [old_targets[0], old_targets[2]])
    translations = builder._translate_api_documents(
        docs, language, SCRATCH / 'no-models-needed', argparse.Namespace(), reuse_history=True)
    assert translations == {SYMBOL: expected}
    builder.write_language(docs, language, translations, preserve_other_symbols=True)
    after = json.loads(path.read_text())
    record = after['symbols'][SYMBOL]
    assert record['source_sha256'] == builder._source_hash(source)
    assert record['source_blocks_sha256'] == builder._source_block_hashes(source)
    assert record['translation_source_blocks_sha256'] == builder._translation_source_block_hashes(source)
    assert record['text'] == expected
    assert builder.reusable_api_translations(docs, language) == translations
    before['symbols'][SYMBOL] = record
    assert before == after, 'Unrelated symbols and catalog metadata must be identical'
    report['locales'][language] = {
        'unchanged_other_symbols': len(after['symbols']) - 1,
        'old_blocks': 3, 'new_blocks': 2,
        'kept_paragraphs_verbatim': True,
        'source_and_context_gates': 'PASS',
        'review_record_reindex': '#2 -> #1',
        'catalog_sha256': digest(path),
    }
    print(language, 'PASS: removed obsolete paragraph; all other symbols unchanged', flush=True)

assert digest(english_path) == english_digest
assert digest(history_path) == history_digest
report['english_unchanged_and_current'] = True
report['history_preserved'] = True
(SCRATCH / 'receipt.json').write_text(json.dumps(report, indent=2) + '\n')
print('PASS: nine locales, 18 gated retained blocks, all unrelated records unchanged', flush=True)
