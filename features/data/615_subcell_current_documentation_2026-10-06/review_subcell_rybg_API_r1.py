from pathlib import Path
import json
import build_documentation_i18n as api
import write_reviewed_api_record as writer
scratch = Path('/media/carruthers/mnt3/codex/scratch/docs-completion-20261005')
source = json.loads((scratch / 'subcell-rybg-documentation-inventory-r3.json').read_text())['new_API_blocks']
assert len(source) == 6
translations = {
'sv': [
'Spara endast ett fullständigt val av fyra index och stäng sedan.',
'För ``subcell_rybg``, välj :data:`CHANNEL_PROJECT`, ange fyra uttryckliga, olika ``channels`` i ordningen mikrotubuli (R), ER (Y), DNA (B), protein (G), och sätt ``normalize=False``. Beskärningarna måste vara minst 16 pixlar i varje rumslig dimension. De ursprungliga dimensionerna bevaras; kodaren använder författarnas min-max-normalisering av hela beskärningen.',
'Namnet på en timm-kodare eller en registrerad grundmodellskodare.',
'Koda beskärningar med formen ``(n, h, w, k)`` i batcher anpassade till modellen.',
'float32-beskärningar med kanalerna sist. SubCell R/Y/B/G använder ursprungliga intensitetsvärden; andra grundmodeller förväntar sig [0, 1].',
'Min-max-normalisera modellens förväntade bildplan och koda sedan.'
],
'de': [
'Nur eine vollständige Auswahl von vier Indizes speichern und anschließend schließen.',
'Für ``subcell_rybg`` wählen Sie :data:`CHANNEL_PROJECT`, geben vier explizite, unterschiedliche ``channels`` in der Reihenfolge Mikrotubuli (R), ER (Y), DNA (B), Protein (G) an und setzen ``normalize=False``. Die Bildausschnitte müssen in jeder räumlichen Dimension mindestens 16 Pixel groß sein. Ihre ursprünglichen Abmessungen bleiben erhalten; der Encoder verwendet die Min-Max-Normalisierung der Autoren für den gesamten Bildausschnitt.',
'Name eines timm-Encoders oder eines registrierten Foundation-Encoders.',
'Bildausschnitte der Form ``(n, h, w, k)`` in modellgerechten Batches codieren.',
'float32-Bildausschnitte mit den Kanälen an letzter Stelle. SubCell R/Y/B/G verwendet ursprüngliche Intensitätswerte; andere Foundation-Modelle erwarten [0, 1].',
'Die vom Modell erwarteten Bildebenen Min-Max-normalisieren und anschließend codieren.'
],
'es': [
'Guardar únicamente una selección completa de cuatro índices y cerrar después.',
'Para ``subcell_rybg``, elija :data:`CHANNEL_PROJECT`, indique cuatro ``channels`` explícitos y distintos en el orden microtúbulos (R), ER (Y), DNA (B), proteína (G), y establezca ``normalize=False``. Los recortes deben tener al menos 16 píxeles en cada dimensión espacial. Se conservan las dimensiones originales; el codificador aplica la normalización mínimo-máximo de los autores sobre todo el recorte.',
'Nombre de un codificador timm o de un codificador de modelo fundacional registrado.',
'Codificar recortes de forma ``(n, h, w, k)`` en lotes adecuados para el modelo.',
'Recortes float32 con los canales en la última dimensión. SubCell R/Y/B/G utiliza valores de intensidad originales; los demás modelos fundacionales esperan [0, 1].',
'Normalizar por mínimo y máximo los planos que espera el modelo y después codificarlos.'
],
'zh_CN': [
'仅保存完整的四个索引选择，然后关闭。',
'对于 ``subcell_rybg``，请选择 :data:`CHANNEL_PROJECT`，按微管 (R)、ER (Y)、DNA (B)、蛋白质 (G) 的顺序明确指定四个互不相同的 ``channels``，并设置 ``normalize=False``。裁剪图像的每个空间维度至少为 16 像素。保留原始尺寸；编码器对整个裁剪图像应用作者的最小值-最大值归一化方法。',
'timm 编码器或已注册基础模型编码器的名称。',
'以适合模型的批次编码形状为 ``(n, h, w, k)`` 的裁剪图像。',
'float32 裁剪图像，通道位于最后一个维度。SubCell R/Y/B/G 使用原始强度值；其他基础模型要求数值位于 [0, 1]。',
'对模型所需的图像平面进行最小值-最大值归一化，然后编码。'
],
'pt': [
'Guardar apenas uma seleção completa de quatro índices e depois fechar.',
'Para ``subcell_rybg``, escolha :data:`CHANNEL_PROJECT`, forneça quatro ``channels`` explícitos e distintos na ordem microtúbulos (R), ER (Y), DNA (B), proteína (G), e defina ``normalize=False``. Os recortes devem ter pelo menos 16 pixels em cada dimensão espacial. As dimensões originais são preservadas; o codificador aplica a normalização mínimo-máximo dos autores ao recorte inteiro.',
'Nome de um codificador timm ou de um codificador de modelo fundacional registrado.',
'Codificar recortes de formato ``(n, h, w, k)`` em lotes adequados ao modelo.',
'Recortes float32 com os canais na última dimensão. SubCell R/Y/B/G usa valores de intensidade originais; os outros modelos fundacionais esperam [0, 1].',
'Normalizar pelo mínimo e máximo os planos esperados pelo modelo e depois codificá-los.'
],
'hi': [
'केवल चार सूचकांकों का पूरा चयन सहेजें, फिर बंद करें।',
'``subcell_rybg`` के लिए :data:`CHANNEL_PROJECT` चुनें, माइक्रोट्यूब्यूल (R), ER (Y), DNA (B), प्रोटीन (G) के क्रम में चार स्पष्ट और अलग ``channels`` दें, और ``normalize=False`` सेट करें। क्रॉप की प्रत्येक स्थानिक विमा कम से कम 16 पिक्सेल होनी चाहिए। मूल आकार बनाए रखे जाते हैं; एन्कोडर पूरे क्रॉप पर लेखकों का न्यूनतम-अधिकतम सामान्यीकरण लागू करता है।',
'timm एन्कोडर या पंजीकृत फ़ाउंडेशन एन्कोडर का नाम।',
'``(n, h, w, k)`` आकार के क्रॉप को मॉडल के अनुकूल बैचों में एन्कोड करें।',
'float32 क्रॉप, जिनमें चैनल अंतिम विमा में हैं। SubCell R/Y/B/G मूल तीव्रता मान उपयोग करता है; अन्य फ़ाउंडेशन मॉडल [0, 1] की अपेक्षा करते हैं।',
'मॉडल के अपेक्षित इमेज प्लेन का न्यूनतम-अधिकतम सामान्यीकरण करें, फिर एन्कोड करें।'
],
'ko': [
'네 인덱스를 모두 선택한 경우에만 저장한 다음 닫습니다.',
'``subcell_rybg``에서는 :data:`CHANNEL_PROJECT`를 선택하고 미세소관 (R), ER (Y), DNA (B), 단백질 (G) 순서로 서로 다른 네 ``channels``를 명시적으로 지정한 뒤 ``normalize=False``로 설정하세요. 크롭의 각 공간 차원은 최소 16픽셀이어야 합니다. 원래 크기를 유지하며, 인코더는 전체 크롭에 저자들의 최소-최대 정규화를 적용합니다.',
'timm 인코더 또는 등록된 파운데이션 인코더의 이름입니다.',
'``(n, h, w, k)`` 형태의 크롭을 모델에 적합한 배치로 인코딩합니다.',
'채널이 마지막 차원에 있는 float32 크롭입니다. SubCell R/Y/B/G는 원래 강도 값을 사용하며 다른 파운데이션 모델은 [0, 1]을 기대합니다.',
'모델이 기대하는 영상 평면을 최소-최대 정규화한 다음 인코딩합니다.'
],
'is': [
'Vista aðeins fullbúið val fjögurra vísitalna og loka síðan.',
'Fyrir ``subcell_rybg`` skal velja :data:`CHANNEL_PROJECT`, tilgreina fjögur aðskilin ``channels`` sérstaklega í röðinni örpíplur (R), ER (Y), DNA (B), prótein (G), og stilla ``normalize=False``. Hver rúmvídd skurðar verður að vera að minnsta kosti 16 pixlar. Upprunalegum víddum er haldið; kóðarinn beitir lágmarks-hámarksstöðlun höfunda á allan skurðinn.',
'Heiti timm-kóðara eða skráðs grunnlíkanakóðara.',
'Kóða skurði með lögunina ``(n, h, w, k)`` í lotum sem henta líkaninu.',
'float32-skurðir með rásir í síðustu vídd. SubCell R/Y/B/G notar upprunaleg styrkgildi; önnur grunnlíkön búast við [0, 1].',
'Lágmarks-hámarksstaðla myndplönin sem líkanið gerir ráð fyrir og kóða síðan.'
],
'fr': [
'Enregistrer uniquement un choix complet de quatre indices, puis fermer.',
'Pour ``subcell_rybg``, choisissez :data:`CHANNEL_PROJECT`, indiquez quatre ``channels`` explicites et distincts dans l’ordre microtubules (R), ER (Y), DNA (B), protéine (G), et définissez ``normalize=False``. Les recadrages doivent mesurer au moins 16 pixels dans chaque dimension spatiale. Les dimensions originales sont conservées ; l’encodeur applique la normalisation minimum-maximum des auteurs à l’ensemble du recadrage.',
'Nom d’un encodeur timm ou d’un encodeur de modèle de fondation enregistré.',
'Encoder les recadrages de forme ``(n, h, w, k)`` par lots adaptés au modèle.',
'Recadrages float32 avec les canaux dans la dernière dimension. SubCell R/Y/B/G utilise les valeurs d’intensité originales ; les autres modèles de fondation attendent [0, 1].',
'Normaliser par le minimum et le maximum les plans attendus par le modèle, puis les encoder.'
],
}
prepared = {}
for language, targets in translations.items():
    assert len(targets) == len(source)
    rows = [writer.record(row['label'], target) for row, target in zip(source, targets)]
    for row in rows:
        assert api._reviewed_api_block_valid(row['source'], row['translation'], language), (language, row)
        assert api._reviewed_api_block_valid(row['context'], row['translation'], language), (language, row)
    prepared[language] = rows
for language, rows in prepared.items():
    path = writer.write(language, '2026-10-06-subcell-rybg', rows)
    doc = json.loads(path.read_text())
    doc['review_method'] = 'Direct Codex AI technical translation and review of native four-plane mapping, spatial minimum, original intensities and normalization; no native-speaker signoff.'
    path.write_text(json.dumps(doc, ensure_ascii=False, indent=2, sort_keys=True) + '\n')
    print('PASS normal reviewed API admission', language, len(rows))
(scratch / 'subcell-rybg-reviewed-API-r1.json').write_text(json.dumps(prepared, ensure_ascii=False, indent=2) + '\n')
