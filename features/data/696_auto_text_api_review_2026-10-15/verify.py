"""Verify the approved AUTO_TEXT source, nine reviews and normal API writer.

Use --source for the approved object_settings_grid.py snapshot while the widget
source is still private. --output must name a new disposable scratch directory.
This writes no repository catalogs and needs no translation model or network.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'tools'))
import build_documentation_i18n as builder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT / 'spacr/qt/widgets/object_settings_grid.py')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    receipt = json.loads(Path(__file__).with_name('receipt.json').read_text())
    source_bytes = args.source.read_bytes()
    assert hashlib.sha256(source_bytes).hexdigest() == receipt['approved_snapshot_sha256']
    symbol = receipt['symbol']
    source = builder._additional_assignment_docs(
        ast.parse(source_bytes).body, 'spacr.qt.widgets.object_settings_grid')[symbol]
    assert builder._source_hash(source) == receipt['source_sha256']
    blocks, _ = builder.translatable_blocks(source)
    assert len(blocks) == 1
    block = blocks[0]
    context = builder._api_translation_source(block)
    assert builder._source_hash(block) == receipt['source_block_sha256']
    assert builder._source_hash(context) == receipt['context_sha256']
    args.output.mkdir(parents=True, exist_ok=False)
    builder.API_DIR = args.output
    docs = {symbol: source}
    for language, evidence in receipt['languages'].items():
        reviewed = builder.reviewed_api_block_translations(docs, language)
        assert set(reviewed) == {block}
        target = reviewed[block]
        assert hashlib.sha256(target.encode()).hexdigest() == evidence['translation_sha256']
        for canonical in (block, context):
            assert builder._api_block_valid(canonical, target, language)
            assert builder._reviewed_api_block_valid(canonical, target, language)
        translated = builder._translate_api_documents(
            docs, language, args.output / 'no-models-needed', argparse.Namespace())
        assert translated == {symbol: target}
        builder.write_language(docs, language, translated)
        record = json.loads((args.output / f'{language}.json').read_text())['symbols'][symbol]
        assert record['source_sha256'] == builder._source_hash(source)
        assert record['source_blocks_sha256'] == builder._source_block_hashes(source)
        assert record['translation_source_blocks_sha256'] == builder._translation_source_block_hashes(source)
        assert record['text'] == target
        assert builder.reusable_api_translations(docs, language) == translated
        print(language, 'PASS: source/context gates, exact review, normal writer and hashes', flush=True)
    assert args.source.read_bytes() == source_bytes
    print('PASS: nine locales; approved source unchanged; no repository catalogs written', flush=True)


if __name__ == '__main__':
    main()
