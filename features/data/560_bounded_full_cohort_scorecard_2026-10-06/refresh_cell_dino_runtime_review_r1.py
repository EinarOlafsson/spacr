from pathlib import Path
import json
import sys

sys.meta_path = [finder for finder in sys.meta_path
                 if '__editable__' not in (getattr(finder, '__module__', '') or type(finder).__module__)]
sys.path.insert(0, str(Path.cwd()))
sys.path.insert(0, str(Path('tools').resolve()))
import spacr
assert Path(spacr.__file__).resolve().parent == Path('spacr').resolve()
import build_i18n_catalogs as builder
import write_reviewed_api_record as writer

scratch = Path('/media/carruthers/mnt3/codex/scratch/docs-completion-20261005')
delta = json.loads((scratch / 'scorecard-api-runtime-source-delta-r1.json').read_text())
assert not delta['api_changed']
assert len(delta['runtime_added']) == len(delta['runtime_retired']) == 1
old_source, source = delta['runtime_retired'][0], delta['runtime_added'][0]
replacements = {
    'sv': ('Cell-DINO:s vikter har inte publicerats ännu.', 'Cell-DINO kräver en officiell kontrollpunkt och stöds ännu inte av denna version.'),
    'de': ('Die Gewichte von Cell-DINO sind noch nicht veröffentlicht.', 'Cell-DINO benötigt einen offiziellen Checkpoint und wird von dieser Version noch nicht unterstützt.'),
    'es': ('Los pesos de Cell-DINO aún no se han publicado.', 'Cell-DINO requiere un punto de control oficial y esta versión aún no lo admite.'),
    'zh_CN': ('Cell-DINO 的权重尚未发布。', 'Cell-DINO 需要官方检查点，此版本尚不支持。'),
    'pt': ('Os pesos de Cell-DINO ainda não foram publicados.', 'Cell-DINO requer um checkpoint oficial e ainda não é suportado por esta versão.'),
    'hi': ('Cell-DINO के वज़न अभी प्रकाशित नहीं हुए हैं।', 'Cell-DINO के लिए आधिकारिक चेकपॉइंट आवश्यक है और इस संस्करण में अभी इसका समर्थन नहीं है।'),
    'ko': ('Cell-DINO의 가중치는 아직 공개되지 않았습니다.', 'Cell-DINO에는 공식 체크포인트가 필요하며 이 버전에서는 아직 지원되지 않습니다.'),
    'is': ('Vægi Cell-DINO hafa ekki enn verið birt.', 'Cell-DINO þarf opinbera vistun af líkaninu og er ekki enn stutt í þessari útgáfu.'),
    'fr': ('Les poids de Cell-DINO ne sont pas encore publiés.', 'Cell-DINO nécessite un point de contrôle officiel et n’est pas encore pris en charge par cette version.'),
}
prepared = {}
for language, (old_clause, clause) in replacements.items():
    path = writer.REVIEWED_RUNTIME / language / '2026-09-27-runtime-codex-delta.json'
    original = json.loads(path.read_text())
    records = [r for r in original['records'] if r['source'] == old_source]
    assert len(records) == 1
    assert records[0]['translation'].count(old_clause) == 1
    translation = records[0]['translation'].replace(old_clause, clause)
    entry = writer.runtime_record('ui', source, translation)
    prepared[language] = (path, original, entry)

proof = {}
for language, (path, original, entry) in prepared.items():
    retained = [r for r in original['records'] if r['source'] != old_source]
    updated = dict(original)
    updated['records'] = retained
    path.write_text(json.dumps(updated, ensure_ascii=False, indent=2, sort_keys=True) + '\n')
    writer.write(language, path.stem, [entry])
    reviewed = builder.reviewed_runtime_translations(language)
    assert reviewed[source] == entry['translation']
    current = json.loads(path.read_text())
    assert len(current['records']) == len(original['records'])
    assert all(r in current['records'] for r in retained)
    proof[language] = {'unrelated_reviewed_records_preserved': len(retained),
                       'one_source_bound_caption_replaced': True,
                       'source': source, 'translation': entry['translation']}
    print(language, 'normal reviewed input accepted; all other reviewed records preserved', flush=True)
(scratch / 'cell-dino-runtime-review-input-proof-r1.json').write_text(json.dumps(
    {'review_method': 'Direct Codex AI technical review; no native-speaker signoff',
     'languages': proof}, ensure_ascii=False, indent=2) + '\n')
