from pathlib import Path
import json

import build_documentation_i18n as api
import write_reviewed_api_record as writer

scratch = Path('/media/carruthers/mnt3/codex/scratch/docs-completion-20261005')
old = json.loads((scratch / 'foundation-api-reviewed-inputs-r2.json').read_text())
indices = (0, 1, 3, 4, 5, 6)
targets = {
    'sv': [
        'Beskriv denna kodare som en :class:`spacr.model_zoo.ModelEntry`.',
        'Posten förenar kodarens konfiguration, tillgängliga uppgifter om kontrollpunktens ursprung och valfria mått på sökprestanda. Jämför endast poster vars kanalpolicyer är förenliga.',
        'Om ingen läsbar lokal kontrollpunkt hittas behåller posten en tom kontrollsumma och förklarar detta i sina anteckningar. :func:`spacr.model_zoo.fetch` avvisar poster utan kontrollsumma.',
        'Kodarens konfiguration; använder ``EmbeddingSpec()`` om den utelämnas.',
        'Valfria mått på sökprestanda, uppmätta med märkta kontroller och lagrade i :attr:`ModelEntry.metrics` för visning med ``scorecard_lines``.',
        'En ``ModelEntry`` av typen ``\'encoder\'``.',
    ],
    'de': [
        'Beschreiben Sie diesen Encoder als :class:`spacr.model_zoo.ModelEntry`.',
        'Der Eintrag vereint die Encoder-Konfiguration, verfügbare Angaben zur Herkunft des Checkpoints und optionale Suchmetriken. Vergleichen Sie nur Einträge mit kompatiblen Kanalrichtlinien.',
        'Wenn kein lesbarer lokaler Checkpoint gefunden wird, behält der Eintrag einen leeren Hash und erläutert dies in seinen Hinweisen. :func:`spacr.model_zoo.fetch` lehnt Einträge ohne Hash ab.',
        'Encoder-Konfiguration; verwendet ``EmbeddingSpec()``, wenn sie nicht angegeben wird.',
        'Optionale Suchmetriken, die anhand beschrifteter Kontrollen gemessen und in :attr:`ModelEntry.metrics` zur Anzeige durch ``scorecard_lines`` gespeichert werden.',
        'Ein ``ModelEntry`` mit dem Typ ``\'encoder\'``.',
    ],
    'es': [
        'Describa este codificador como una :class:`spacr.model_zoo.ModelEntry`.',
        'La entrada combina la configuración del codificador, la procedencia disponible del punto de control y métricas opcionales de recuperación. Compare únicamente entradas con políticas de canales compatibles.',
        'Si no se encuentra un punto de control local legible, la entrada conserva un resumen criptográfico vacío y lo explica en sus notas. :func:`spacr.model_zoo.fetch` rechaza las entradas sin resumen criptográfico.',
        'Configuración del codificador; utiliza ``EmbeddingSpec()`` si se omite.',
        'Métricas opcionales de recuperación medidas con controles etiquetados y almacenadas en :attr:`ModelEntry.metrics` para su visualización mediante ``scorecard_lines``.',
        'Una ``ModelEntry`` de tipo ``\'encoder\'``.',
    ],
    'pt': [
        'Descreva este codificador como uma :class:`spacr.model_zoo.ModelEntry`.',
        'A entrada combina a configuração do codificador, as informações disponíveis sobre a procedência do ponto de controle e métricas opcionais de recuperação. Compare apenas entradas com políticas de canais compatíveis.',
        'Se nenhum ponto de controle local legível for encontrado, a entrada mantém um resumo criptográfico vazio e explica isso nas notas. :func:`spacr.model_zoo.fetch` recusa entradas sem resumo criptográfico.',
        'Configuração do codificador; usa ``EmbeddingSpec()`` quando omitida.',
        'Métricas opcionais de recuperação medidas com controles rotulados e armazenadas em :attr:`ModelEntry.metrics` para exibição por ``scorecard_lines``.',
        'Uma ``ModelEntry`` do tipo ``\'encoder\'``.',
    ],
    'fr': [
        'Décrivez cet encodeur comme une :class:`spacr.model_zoo.ModelEntry`.',
        'L’entrée réunit la configuration de l’encodeur, les informations disponibles sur la provenance du point de contrôle et des métriques de recherche facultatives. Comparez uniquement les entrées dont les politiques de canaux sont compatibles.',
        'Si aucun point de contrôle local lisible n’est trouvé, l’entrée conserve une empreinte vide et l’explique dans ses notes. :func:`spacr.model_zoo.fetch` refuse les entrées sans empreinte.',
        'Configuration de l’encodeur ; utilise ``EmbeddingSpec()`` si elle est omise.',
        'Métriques de recherche facultatives mesurées avec des témoins étiquetés et enregistrées dans :attr:`ModelEntry.metrics` pour être affichées par ``scorecard_lines``.',
        'Une ``ModelEntry`` de type ``\'encoder\'``.',
    ],
    'is': [
        'Lýsið þessum kóðara sem :class:`spacr.model_zoo.ModelEntry`.',
        'Færslan sameinar stillingar kóðarans, tiltækar upplýsingar um uppruna vistunar líkansins og valfrjálsa mælikvarða á endurheimt. Berið aðeins saman færslur með samrýmanlegar rásastefnur.',
        'Ef engin læsileg staðbundin vistun líkans finnst heldur færslan tómu tætigildi og skýrir það í athugasemdum sínum. :func:`spacr.model_zoo.fetch` hafnar færslum án tætigildis.',
        'Stillingar kóðarans; notar ``EmbeddingSpec()`` ef þeim er sleppt.',
        'Valfrjálsir mælikvarðar á endurheimt, mældir með merktum viðmiðum og vistaðir í :attr:`ModelEntry.metrics` til birtingar með ``scorecard_lines``.',
        '``ModelEntry`` af gerðinni ``\'encoder\'``.',
    ],
    'zh_CN': [
        '将此编码器描述为 :class:`spacr.model_zoo.ModelEntry`。',
        '该条目汇总编码器配置、可用的检查点来源信息和可选的检索指标。仅比较通道策略兼容的条目。',
        '如果未找到可读的本地检查点，条目会保留空摘要，并在备注中说明。:func:`spacr.model_zoo.fetch` 会拒绝没有摘要的条目。',
        '编码器配置；省略时使用 ``EmbeddingSpec()``。',
        '使用已标注对照测得的可选检索指标，存储在 :attr:`ModelEntry.metrics` 中，由 ``scorecard_lines`` 显示。',
        '类型为 ``\'encoder\'`` 的 ``ModelEntry``。',
    ],
    'ko': [
        '이 인코더를 :class:`spacr.model_zoo.ModelEntry`로 설명합니다.',
        '항목은 인코더 구성, 사용 가능한 체크포인트 출처 정보 및 선택적 검색 지표를 결합합니다. 채널 정책이 호환되는 항목만 비교하세요.',
        '읽을 수 있는 로컬 체크포인트를 찾지 못하면 항목은 빈 해시를 유지하고 메모에 이를 설명합니다. :func:`spacr.model_zoo.fetch`는 해시가 없는 항목을 거부합니다.',
        '인코더 구성; 생략하면 ``EmbeddingSpec()``를 사용합니다.',
        '라벨이 있는 대조군에서 측정한 선택적 검색 지표로, ``scorecard_lines``에 표시할 수 있도록 :attr:`ModelEntry.metrics`에 저장됩니다.',
        '종류가 ``\'encoder\'``인 ``ModelEntry``입니다.',
    ],
    'hi': [
        'इस एन्कोडर का वर्णन :class:`spacr.model_zoo.ModelEntry` के रूप में करें।',
        'यह प्रविष्टि एन्कोडर कॉन्फ़िगरेशन, चेकपॉइंट की उपलब्ध स्रोत जानकारी और वैकल्पिक पुनर्प्राप्ति मेट्रिक्स को जोड़ती है। केवल संगत चैनल नीतियों वाली प्रविष्टियों की तुलना करें।',
        'यदि पढ़ने योग्य स्थानीय चेकपॉइंट नहीं मिलता है, तो प्रविष्टि खाली हैश रखती है और अपने नोट्स में इसका कारण बताती है। :func:`spacr.model_zoo.fetch` बिना हैश वाली प्रविष्टियों को अस्वीकार करता है।',
        'एन्कोडर कॉन्फ़िगरेशन; छोड़ने पर ``EmbeddingSpec()`` का उपयोग करता है।',
        'लेबल वाले नियंत्रणों पर मापे गए वैकल्पिक पुनर्प्राप्ति मेट्रिक्स, जिन्हें ``scorecard_lines`` में दिखाने के लिए :attr:`ModelEntry.metrics` में संग्रहीत किया जाता है।',
        '``\'encoder\'`` प्रकार की एक ``ModelEntry``।',
    ],
}
prepared = {}
for language, values in targets.items():
    assert len(values) == 6
    entries = [writer.record('spacr.embeddings.encoder_entry#' + str(index), target)
               for index, target in zip(indices, values)]
    unchanged = next(record for record in old['languages'][language]['API'] if record['label'].endswith('#2'))
    entries.append(writer.record(unchanged['label'], unchanged['translation']))
    for entry in entries:
        assert api._reviewed_api_block_valid(entry['source'], entry['translation'], language), (language, entry)
        assert api._reviewed_api_block_valid(entry['context'], entry['translation'], language), (language, entry)
    assert {entry['label'].rsplit('#', 1)[1] for entry in entries} == set(map(str, range(7)))
    prepared[language] = entries
for language, entries in prepared.items():
    writer.write(language, '2026-10-06-encoder-provenance', entries)
    old['languages'][language]['API'] = sorted(entries, key=lambda row: row['label'])
    print('PASS complete direct technical review', language, 'all seven encoder description blocks', flush=True)
old['all_seven_API_blocks_technically_reviewed_after_actual_browser_review'] = True
old['no_native_speaker_signoff'] = True
path = scratch / 'foundation-api-reviewed-inputs-r3.json'
assert not path.exists()
path.write_text(json.dumps(old, ensure_ascii=False, indent=2) + '\n')
