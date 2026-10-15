"""Refresh the exact N644 public paragraph using normal API builder functions."""
import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--root', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
arguments = parser.parse_args()
ROOT = arguments.root.resolve()
SCRATCH = arguments.output.resolve()
SCRATCH.mkdir(parents=True, exist_ok=False)
SYMBOL = 'spacr.qt.widgets.plaque_preview.resolve_detector'
SOURCE_PIN = '3f137038049a04de6b909e9d75db516332317b20'
FILE_HASH = '5f72095c13cbb59c35deea8ccce89c6d98967f6c6e25bf004c76f5f87b80bdb2'
sys.path.insert(0, str(ROOT / 'tools'))
import build_documentation_i18n as builder

targets = {
    'sv': 'Letar där körningen letar: själva sökvägen, zoo-postens egen sökväg, den aktiva gemensamma modellcachen och körningens modellcache under ``<src>/plaque_figures/models``.',
    'de': 'Sucht dort, wo der Lauf sucht: der Pfad selbst, der eigene Pfad des Zoo-Eintrags, der aktive gemeinsame Modell-Cache und der Modell-Cache des Laufs unter ``<src>/plaque_figures/models``.',
    'es': 'Busca donde busca la ejecución: la propia ruta, la ruta propia de la entrada del zoo, la caché compartida de modelos activa y la caché de modelos de la ejecución en ``<src>/plaque_figures/models``.',
    'zh_CN': '在运行时查找的相同位置查找：路径本身、模型库条目自身的路径、当前使用的共享模型缓存，以及运行在 ``<src>/plaque_figures/models`` 下的模型缓存。',
    'pt': 'Procura onde a execução procura: o próprio caminho, o caminho próprio da entrada do zoo, o cache compartilhado de modelos ativo e o cache de modelos da execução em ``<src>/plaque_figures/models``.',
    'hi': 'वहीं देखता है जहाँ रन देखता है: स्वयं पथ, zoo प्रविष्टि का अपना पथ, सक्रिय साझा मॉडल कैश और ``<src>/plaque_figures/models`` के अंतर्गत रन का मॉडल कैश।',
    'ko': '처리 세션이 찾는 곳을 찾습니다: 경로 자체, 모델 목록 항목 자체의 경로, 현재 사용 중인 공유 모델 캐시, 그리고 ``<src>/plaque_figures/models`` 아래의 처리 세션 모델 캐시.',
    'is': 'Leitar þar sem keyrslan leitar: að slóðinni sjálfri, eigin slóð færslunnar í líkanasafninu, virka sameiginlega líkanaskyndiminninu og líkanaskyndiminni keyrslunnar undir ``<src>/plaque_figures/models``.',
    'fr': 'Cherche là où cherche l’exécution : le chemin lui-même, le chemin propre de l’entrée du zoo, le cache partagé de modèles actif et le cache de modèles de l’exécution sous ``<src>/plaque_figures/models``.',
}

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

source_path = ROOT / 'spacr/qt/widgets/plaque_preview.py'
assert digest(source_path) == FILE_HASH
tree = ast.parse(source_path.read_text())
node = next(n for n in builder._module_scope_nodes(tree.body)
            if isinstance(n, ast.FunctionDef) and n.name == 'resolve_detector')
assert builder._is_visible_function_name(node.name, module_is_package=False)
assert builder._is_rendered_autoapi_entry(source_path, 'spacr.qt.widgets.plaque_preview', SYMBOL)
assert not builder._is_visible_function_name('_start_download', module_is_package=False)
source = builder._clean_doc(node)
docs = {SYMBOL: source}
blocks, layout = builder.translatable_blocks(source)
assert len(blocks) == 5
changed = blocks[1]
context = builder._api_translation_source(changed)
english_path = builder.API_DIR / 'en.json'
english_before = json.loads(english_path.read_text())
old_english = english_before['symbols'][SYMBOL]
old_blocks, _ = builder.translatable_blocks(old_english['text'])
assert len(old_blocks) == 5
assert [i for i in range(5) if old_blocks[i] != blocks[i]] == [1]
assert old_blocks[1].replace('``~/.spacr/models``', 'the active shared model cache') == changed
assert 'spacr.qt.widgets.plaque_preview.PlaquePreviewPanel._start_download' not in english_before['symbols']
history_path = ROOT / 'docs/i18n/api_source_history.json'
history_hash = digest(history_path)
report = {
    'source_pin': SOURCE_PIN,
    'source_file': str(source_path.relative_to(ROOT)),
    'source_file_sha256': FILE_HASH,
    'symbol': SYMBOL,
    'old_english': old_english,
    'new_source_sha256': builder._source_hash(source),
    'new_block_sha256': builder._source_hash(changed),
    'new_context_sha256': builder._source_hash(context),
    'review_kind': 'AI technical review (Codex); no human or native-speaker signoff',
    'model_decode_required': False,
    'private_start_download_excluded': True,
    'locales': {},
}

for language, target in targets.items():
    for input_text in (changed, context):
        assert builder._reviewed_api_block_valid(input_text, target, language), (language, 'review gate')
        assert builder._api_block_valid(input_text, target, language), (language, 'ordinary gate')
    try:
        builder.reviewed_api_block_translations(docs, language)
    except ValueError as exc:
        assert 'stale reviewed API source' in str(exc) and SYMBOL + '#1' in str(exc), str(exc)
        report['locales'][language] = {'before_rejection': str(exc).replace(str(ROOT) + '/', '')}
    else:
        raise AssertionError((language, 'expected source-drift rejection before refresh'))

for language, target in targets.items():
    retired = []
    for path in sorted((builder.REVIEWED_API_DIR / language).glob('*.json')):
        payload = json.loads(path.read_text())
        before = copy.deepcopy(payload)
        for record in payload['records']:
            if record.get('label') != SYMBOL + '#1' or record.get('retired'):
                continue
            assert record['source'] == old_blocks[1]
            original = copy.deepcopy(record)
            record['retired'] = True
            record['retired_reason'] = ('N644 active shared model cache replaces the fixed ~/.spacr/models path; '
                                        '2026-10-15 source-bound replacement is recorded separately. '
                                        'Original review and target retained unchanged.')
            retired.append({'path': str(path.relative_to(ROOT)), 'record': original})
        if before != payload:
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n')
    assert len(retired) == 1, (language, retired)
    report['locales'][language]['retired_review'] = retired[0]
    payload = {
        'schema': 1,
        'language': language,
        'review_kind': 'AI technical review (Codex), no human or native-speaker signoff',
        'note': 'N644 public resolve_detector cache-location paragraph. Only the fixed cache path changes '
                'to the active shared model cache. Search locations/order and per-run cache path remain '
                'unchanged. Exact source pin ' + SOURCE_PIN + '; plaque_preview.py SHA256 ' + FILE_HASH + '.',
        'records': [{
            'label': SYMBOL + '#1',
            'source': changed,
            'source_sha256': builder._source_hash(changed),
            'context': context,
            'translation': target,
            'review_basis': 'AI technical comparison against the current English paragraph and prior locale '
                            'target; replaces only the old fixed path with active shared model cache. '
                            'Preserves local path, model-zoo entry path and literal per-run model path. '
                            'No native-speaker review claimed.',
        }],
    }
    path = builder.REVIEWED_API_DIR / language / '2026-10-15-active-model-cache.json'
    assert not path.exists()
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n')

for language, target in targets.items():
    path = builder.API_DIR / f'{language}.json'
    before = json.loads(path.read_text())
    old_record = before['symbols'][SYMBOL]
    assert old_record['source_sha256'] == old_english['source_sha256']
    old_targets, _ = builder.translatable_blocks(old_record['text'])
    assert len(old_targets) == 5
    expected_targets = old_targets.copy()
    expected_targets[1] = target
    expected = builder.rebuild_document(layout, expected_targets)
    reviewed = builder.reviewed_api_block_translations(docs, language)
    assert reviewed[changed] == target
    translated = builder._translate_api_documents(
        docs, language, SCRATCH / 'no-models-needed', argparse.Namespace(), reuse_history=True)
    assert translated == {SYMBOL: expected}
    builder.write_language(docs, language, translated, preserve_other_symbols=True)
    after = json.loads(path.read_text())
    record = after['symbols'][SYMBOL]
    assert record['text'] == expected
    assert record['source_sha256'] == builder._source_hash(source)
    assert record['source_blocks_sha256'] == builder._source_block_hashes(source)
    assert record['translation_source_blocks_sha256'] == builder._translation_source_block_hashes(source)
    assert builder.reusable_api_translations(docs, language) == translated
    before['symbols'][SYMBOL] = record
    assert before == after, 'Unrelated symbols and catalog metadata must be identical'
    report['locales'][language].update({
        'unchanged_other_symbols': len(after['symbols']) - 1,
        'unchanged_blocks_verbatim': [0, 2, 3, 4],
        'changed_block': 1,
        'new_target': target,
        'ordinary_and_reviewed_canonical_and_context_gates': 'PASS',
        'normal_writer_and_reusable_readback': 'PASS',
        'catalog_sha256': digest(path),
    })
    print(language, 'PASS: only cache-location paragraph changed; other symbols/metadata unchanged', flush=True)

english_docs = {key: record['text'] for key, record in english_before['symbols'].items()}
english_docs[SYMBOL] = source
english_after = builder._english_manifest(english_docs)
english_expected = copy.deepcopy(english_before)
english_expected['symbols'][SYMBOL] = builder._english_manifest(docs)['symbols'][SYMBOL]
assert english_after == english_expected
builder._write_json(english_path, english_after)
assert digest(history_path) == history_hash
assert digest(source_path) == FILE_HASH
report['english_other_symbols_unchanged'] = len(english_after['symbols']) - 1
report['english_catalog_sha256'] = digest(english_path)
report['source_history_sha256_unchanged'] = history_hash
report['normal_english_writer'] = '_english_manifest + _write_json; scoped source overlay preserves unrelated history'
report['base_commit'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
(SCRATCH / 'receipt.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
print('PASS: English and nine locales current; 36 unchanged locale blocks retained verbatim; history unchanged', flush=True)
