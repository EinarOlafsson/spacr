from pathlib import Path
import json
import build_documentation_i18n as api
import write_reviewed_api_record as writer
scratch = Path('/media/carruthers/mnt3/codex/scratch/docs-completion-20261005')
labels = ['spacr.embeddings._subcell_model.Pooled.forward#0', 'spacr.embeddings._subcell_model.Pooled.forward#1']
targets = {
    'sv': ['``(n, k, h, w)`` till ``(n, 1536)``, med de två huvudena sammanfogade.', 'Kontrollpunkten avgör om ``k`` är två eller fyra kanaler.'],
    'de': ['``(n, k, h, w)`` nach ``(n, 1536)``, wobei die beiden Köpfe zusammengefügt werden.', 'Der Checkpoint bestimmt, ob ``k`` zwei oder vier Kanäle umfasst.'],
    'es': ['De ``(n, k, h, w)`` a ``(n, 1536)``, concatenando las dos cabezas.', 'El punto de control determina si ``k`` corresponde a dos o cuatro canales.'],
    'zh_CN': ['从 ``(n, k, h, w)`` 转换为 ``(n, 1536)``，拼接两个头的输出。', '检查点决定 ``k`` 是两个通道还是四个通道。'],
    'pt': ['De ``(n, k, h, w)`` para ``(n, 1536)``, concatenando as duas cabeças.', 'O checkpoint determina se ``k`` corresponde a dois ou quatro canais.'],
    'hi': ['``(n, k, h, w)`` से ``(n, 1536)`` तक, दोनों हेड के आउटपुट जोड़कर।', 'चेकपॉइंट निर्धारित करता है कि ``k`` दो चैनल हैं या चार।'],
    'ko': ['``(n, k, h, w)``에서 ``(n, 1536)``으로 변환하며 두 헤드의 출력을 이어 붙입니다.', '체크포인트에 따라 ``k``가 두 채널인지 네 채널인지 결정됩니다.'],
    'is': ['Úr ``(n, k, h, w)`` í ``(n, 1536)``, með hausana tvo samtengda.', 'Vistun líkansins ákvarðar hvort ``k`` táknar tvær eða fjórar rásir.'],
    'fr': ['De ``(n, k, h, w)`` à ``(n, 1536)``, en concaténant les deux têtes.', 'Le point de contrôle détermine si ``k`` correspond à deux ou quatre canaux.'],
}
prepared = {}
for language, translations in targets.items():
    rows = [writer.record(label, target) for label, target in zip(labels, translations)]
    for row in rows:
        assert api._reviewed_api_block_valid(row['source'], row['translation'], language)
        assert api._reviewed_api_block_valid(row['context'], row['translation'], language)
    prepared[language] = rows
for language, rows in prepared.items():
    path = writer.write(language, '2026-10-06-subcell-rybg', rows)
    assert len(json.loads(path.read_text())['records']) == 8
    print('PASS exact two/four-channel pool documentation review', language)
(scratch / 'subcell-rybg-reviewed-pool-API-r1.json').write_text(json.dumps(prepared, ensure_ascii=False, indent=2) + '\n')
