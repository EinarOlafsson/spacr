from pathlib import Path
import json
import build_i18n_catalogs as runtime
import write_reviewed_api_record as writer
scratch = Path('/media/carruthers/mnt3/codex/scratch/docs-completion-20261005')
sources = json.loads((scratch / 'subcell-rybg-documentation-inventory-r4.json').read_text())['new_runtime']
assert len(sources) == 21
targets = {
'sv': [
'En modell tränad på mikroskopibilder i stället för fotografier. OpenPhenom och ChAda-ViT tar valfritt antal färgningskanaler. SubCells tvåplansmodell tar DNA och sedan protein; fyrplansmodellen kräver uttrycklig mappning av mikrotubuli, ER, DNA och protein. Vikterna laddas ned en gång. Cell-DINO kräver en officiell kontrollpunkt och stöds ännu inte av denna version. Standard: None (använd basnätverket).',
'En vald SubCell-kanal saknas i de inlästa bildutsnitten. Öppna Kanaler… och mappa de fyra bildplanen igen.',
'Välj fyra olika kanaler i bildutsnitten enligt SubCells officiella ordning. Färgningens identitet gissas inte utifrån kanalens position.',
'Kanal {position} (index {index})', 'Kanaler…',
'Välj en kanal i bildutsnitten för varje SubCell-bildplan: mikrotubuli, ER, DNA och protein.',
'Välj kanal…', 'Välj fyra olika kanaler i bildutsnitten för SubCell.',
'Välj fyra olika kanaler i bildutsnitten innan du kör SubCells fyrplansmodell.', 'DNA (B)', 'ER (Y)',
'Fyra mappade bildplan (en körning)',
'Läs in bildutsnitt med minst fyra kanaler innan du mappar SubCells bildplan för mikrotubuli, ER, DNA och protein.',
'Mappa fyra kanaler i bildutsnitten till mikrotubuli, ER, DNA och protein innan du kör SubCells fyrplansmodell.',
'Mikrotubuli (R)', 'Proteinmarkör (G)', 'SubCell (CZI / Lundberglaboratoriet ViT-B/16, R/Y/B/G)',
'SubCells kanalmappning har sparats i ordningen mikrotubuli, ER, DNA, protein.', 'SubCells fyrplanskanaler',
'SubCells fyrplansmodell kräver bildutsnitt med minst fyra kanaler. Läs först in en lämplig källa för bildutsnitten.',
'De nya bildutsnitten har färre kanaler. Öppna Kanaler… igen för att mappa SubCells fyra bildplan.'
],
'de': [
'Ein auf Mikroskopiebildern statt Fotografien trainiertes Modell. OpenPhenom und ChAda-ViT verwenden beliebig viele Färbungskanäle. Das Zweikanalmodell von SubCell verwendet zuerst DNA und dann Protein; das Vierkanalmodell erfordert eine explizite Zuordnung zu Mikrotubuli, ER, DNA und Protein. Die Gewichte werden einmal heruntergeladen. Cell-DINO benötigt einen offiziellen Checkpoint und wird von dieser Version noch nicht unterstützt. Standard: None (das Backbone verwenden).',
'Ein ausgewählter SubCell-Kanal fehlt in den geladenen Bildausschnitten. Öffnen Sie Kanäle… und ordnen Sie die vier Bildebenen erneut zu.',
'Ordnen Sie vier unterschiedliche Kanäle der Bildausschnitte in der offiziellen SubCell-Reihenfolge zu. Die Färbung wird nicht aus der Kanalposition abgeleitet.',
'Kanal {position} (Index {index})', 'Kanäle…',
'Wählen Sie für jede SubCell-Bildebene einen Kanal der Bildausschnitte: Mikrotubuli, ER, DNA und Protein.',
'Kanal auswählen…', 'Wählen Sie vier unterschiedliche Kanäle der Bildausschnitte für SubCell.',
'Wählen Sie vor dem Start des Vierkanalmodells von SubCell vier unterschiedliche Kanäle der Bildausschnitte.', 'DNA (B)', 'ER (Y)',
'Vier zugeordnete Bildebenen (ein Durchlauf)',
'Laden Sie Bildausschnitte mit mindestens vier Kanälen, bevor Sie die SubCell-Bildebenen für Mikrotubuli, ER, DNA und Protein zuordnen.',
'Ordnen Sie vor dem Start des Vierkanalmodells von SubCell vier Kanäle der Bildausschnitte den Mikrotubuli, ER, DNA und dem Protein zu.',
'Mikrotubuli (R)', 'Proteinmarker (G)', 'SubCell (CZI / Lundberg-Labor ViT-B/16, R/Y/B/G)',
'Die SubCell-Kanalzuordnung wurde in der Reihenfolge Mikrotubuli, ER, DNA, Protein gespeichert.', 'SubCell-Kanäle für vier Bildebenen',
'Das Vierkanalmodell von SubCell benötigt Bildausschnitte mit mindestens vier Kanälen. Laden Sie zuerst eine geeignete Bildausschnittquelle.',
'Die neuen Bildausschnitte haben weniger Kanäle. Öffnen Sie Kanäle… erneut, um die vier SubCell-Bildebenen zuzuordnen.'
],
'es': [
'Un modelo entrenado con imágenes de microscopía en lugar de fotografías. OpenPhenom y ChAda-ViT admiten cualquier número de canales de tinción. El modelo de dos planos de SubCell utiliza DNA y después proteína; el modelo de cuatro planos requiere asignar explícitamente microtúbulos, ER, DNA y proteína. Los pesos se descargan una vez. Cell-DINO requiere un punto de control oficial y esta versión aún no lo admite. Predeterminado: None (usar la red base).',
'Un canal de SubCell seleccionado no existe en los recortes cargados. Abra Canales… y vuelva a asignar los cuatro planos de imagen.',
'Asigne cuatro canales distintos de los recortes en el orden oficial de SubCell. La identidad de la tinción no se deduce de la posición del canal.',
'Canal {position} (índice {index})', 'Canales…',
'Elija un canal de los recortes para cada plano de SubCell: microtúbulos, ER, DNA y proteína.',
'Elegir canal…', 'Elija cuatro canales distintos de los recortes para SubCell.',
'Elija cuatro canales distintos de los recortes antes de ejecutar el modelo de cuatro planos de SubCell.', 'DNA (B)', 'ER (Y)',
'Cuatro planos asignados (una pasada)',
'Cargue recortes con al menos cuatro canales antes de asignar los planos de microtúbulos, ER, DNA y proteína de SubCell.',
'Asigne cuatro canales de los recortes a microtúbulos, ER, DNA y proteína antes de ejecutar el modelo de cuatro planos de SubCell.',
'Microtúbulos (R)', 'Proteína (G)', 'SubCell (CZI / laboratorio de Lundberg ViT-B/16, R/Y/B/G)',
'Asignación de canales de SubCell guardada en el orden microtúbulos, ER, DNA, proteína.', 'Canales de los cuatro planos de SubCell',
'El modelo de cuatro planos de SubCell necesita recortes con al menos cuatro canales. Cargue primero una fuente de recortes adecuada.',
'Los nuevos recortes tienen menos canales. Vuelva a abrir Canales… para asignar los cuatro planos de SubCell.'
],
'zh_CN': [
'使用显微图像而非普通照片训练的模型。OpenPhenom 和 ChAda-ViT 支持任意数量的染色通道。SubCell 的双平面模型先接收 DNA，再接收蛋白质；四平面模型要求明确映射微管、ER、DNA 和蛋白质通道。权重只需下载一次。Cell-DINO 需要官方检查点，此版本尚不支持。默认值为 None（使用骨干网络）。',
'所选的 SubCell 通道不在已加载的裁剪图像中。请打开“通道…”并重新映射四个图像平面。',
'按 SubCell 的官方顺序指定四个不同的裁剪图像通道。不会根据通道位置推测染色类型。',
'通道 {position}（索引 {index}）', '通道…',
'为每个 SubCell 图像平面选择裁剪图像通道：微管、ER、DNA 和蛋白质。',
'选择通道…', '为 SubCell 选择四个不同的裁剪图像通道。',
'运行 SubCell 四平面模型前，请选择四个不同的裁剪图像通道。', 'DNA (B)', 'ER (Y)',
'四个已映射平面（一次前向计算）',
'映射 SubCell 的微管、ER、DNA 和蛋白质平面前，请加载至少含四个通道的裁剪图像。',
'运行 SubCell 四平面模型前，请将四个裁剪图像通道映射至微管、ER、DNA 和蛋白质。',
'微管 (R)', '蛋白质 (G)', 'SubCell (CZI / Lundberg 实验室 ViT-B/16, R/Y/B/G)',
'SubCell 通道映射已按微管、ER、DNA、蛋白质的顺序保存。', 'SubCell 四平面通道',
'SubCell 四平面模型需要至少含四个通道的裁剪图像。请先加载合适的裁剪图像来源。',
'新裁剪图像的通道数量较少。请重新打开“通道…”以映射 SubCell 的四个平面。'
],
'pt': [
'Um modelo treinado com imagens de microscopia em vez de fotografias. OpenPhenom e ChAda-ViT aceitam qualquer número de canais de coloração. O modelo de dois planos do SubCell usa DNA e depois proteína; o modelo de quatro planos exige o mapeamento explícito de microtúbulos, ER, DNA e proteína. Os pesos são baixados uma vez. Cell-DINO exige um checkpoint oficial e ainda não é compatível com esta versão. Padrão: None (usar a rede base).',
'Um canal selecionado do SubCell não existe nos recortes carregados. Abra Canais… e mapeie novamente os quatro planos de imagem.',
'Atribua quatro canais distintos dos recortes na ordem oficial do SubCell. A identidade da coloração não é inferida pela posição do canal.',
'Canal {position} (índice {index})', 'Canais…',
'Escolha um canal dos recortes para cada plano do SubCell: microtúbulos, ER, DNA e proteína.',
'Escolher canal…', 'Escolha quatro canais distintos dos recortes para o SubCell.',
'Escolha quatro canais distintos dos recortes antes de executar o modelo de quatro planos do SubCell.', 'DNA (B)', 'ER (Y)',
'Quatro planos mapeados (uma passagem)',
'Carregue recortes com pelo menos quatro canais antes de mapear os planos de microtúbulos, ER, DNA e proteína do SubCell.',
'Mapeie quatro canais dos recortes para microtúbulos, ER, DNA e proteína antes de executar o modelo de quatro planos do SubCell.',
'Microtúbulos (R)', 'Proteína (G)', 'SubCell (CZI / laboratório Lundberg ViT-B/16, R/Y/B/G)',
'Mapeamento de canais do SubCell salvo na ordem microtúbulos, ER, DNA, proteína.', 'Canais dos quatro planos do SubCell',
'O modelo de quatro planos do SubCell exige recortes com pelo menos quatro canais. Carregue primeiro uma fonte de recortes adequada.',
'Os novos recortes têm menos canais. Abra Canais… novamente para mapear os quatro planos do SubCell.'
],
'hi': [
'साधारण फ़ोटोग्राफ़ के बजाय माइक्रोस्कोपी इमेज पर प्रशिक्षित मॉडल। OpenPhenom और ChAda-ViT कितने भी स्टेन चैनल स्वीकार करते हैं। SubCell का दो-प्लेन मॉडल पहले DNA, फिर प्रोटीन लेता है; चार-प्लेन मॉडल में माइक्रोट्यूब्यूल, ER, DNA और प्रोटीन की स्पष्ट मैपिंग आवश्यक है। वेट एक बार डाउनलोड होते हैं। Cell-DINO को आधिकारिक चेकपॉइंट चाहिए और यह संस्करण अभी उसे समर्थन नहीं देता। डिफ़ॉल्ट: None (बैकबोन नेटवर्क का उपयोग करें)।',
'चुना गया SubCell चैनल लोड किए गए क्रॉप में नहीं है। चैनल… खोलें और चारों इमेज प्लेन फिर से मैप करें।',
'SubCell के आधिकारिक क्रम में क्रॉप के चार अलग चैनल निर्धारित करें। चैनल की स्थिति से स्टेन की पहचान का अनुमान नहीं लगाया जाता।',
'चैनल {position} (सूचकांक {index})', 'चैनल…',
'प्रत्येक SubCell इमेज प्लेन के लिए क्रॉप चैनल चुनें: माइक्रोट्यूब्यूल, ER, DNA और प्रोटीन।',
'चैनल चुनें…', 'SubCell के लिए क्रॉप के चार अलग चैनल चुनें।',
'SubCell का चार-प्लेन मॉडल चलाने से पहले क्रॉप के चार अलग चैनल चुनें।', 'DNA (B)', 'ER (Y)',
'चार मैप किए गए प्लेन (एक पास)',
'SubCell के माइक्रोट्यूब्यूल, ER, DNA और प्रोटीन प्लेन मैप करने से पहले कम से कम चार चैनल वाले क्रॉप लोड करें।',
'SubCell का चार-प्लेन मॉडल चलाने से पहले क्रॉप के चार चैनल माइक्रोट्यूब्यूल, ER, DNA और प्रोटीन पर मैप करें।',
'माइक्रोट्यूब्यूल (R)', 'प्रोटीन (G)', 'SubCell (CZI / Lundberg प्रयोगशाला ViT-B/16, R/Y/B/G)',
'SubCell चैनल मैपिंग माइक्रोट्यूब्यूल, ER, DNA, प्रोटीन के क्रम में सहेजी गई।', 'SubCell के चार-प्लेन चैनल',
'SubCell के चार-प्लेन मॉडल को कम से कम चार चैनल वाले क्रॉप चाहिए। पहले उपयुक्त क्रॉप स्रोत लोड करें।',
'नए क्रॉप में कम चैनल हैं। SubCell के चार प्लेन मैप करने के लिए चैनल… फिर से खोलें।'
],
'ko': [
'일반 사진 대신 현미경 영상으로 학습한 모델입니다. OpenPhenom과 ChAda-ViT는 염색 채널 수에 제한이 없습니다. SubCell의 두 평면 모델은 DNA 다음에 단백질을 받으며, 네 평면 모델은 미세소관, ER, DNA, 단백질을 명시적으로 매핑해야 합니다. 가중치는 한 번 다운로드합니다. Cell-DINO는 공식 체크포인트가 필요하며 이 버전에서는 아직 지원하지 않습니다. 기본값: None (백본 네트워크 사용).',
'선택한 SubCell 채널이 불러온 크롭에 없습니다. 채널…을 열고 네 영상 평면을 다시 매핑하세요.',
'SubCell의 공식 순서대로 서로 다른 크롭 채널 네 개를 지정하세요. 채널 위치로 염색 종류를 추측하지 않습니다.',
'채널 {position} (인덱스 {index})', '채널…',
'각 SubCell 영상 평면의 크롭 채널을 선택하세요: 미세소관, ER, DNA, 단백질.',
'채널 선택…', 'SubCell에 사용할 서로 다른 크롭 채널 네 개를 선택하세요.',
'SubCell의 네 평면 모델을 실행하기 전에 서로 다른 크롭 채널 네 개를 선택하세요.', 'DNA (B)', 'ER (Y)',
'매핑된 네 평면 (한 번의 실행)',
'SubCell의 미세소관, ER, DNA, 단백질 평면을 매핑하기 전에 채널이 네 개 이상인 크롭을 불러오세요.',
'SubCell의 네 평면 모델을 실행하기 전에 크롭 채널 네 개를 미세소관, ER, DNA, 단백질에 매핑하세요.',
'미세소관 (R)', '단백질 (G)', 'SubCell (CZI / Lundberg 연구실 ViT-B/16, R/Y/B/G)',
'SubCell 채널 매핑을 미세소관, ER, DNA, 단백질 순서로 저장했습니다.', 'SubCell 네 평면 채널',
'SubCell의 네 평면 모델에는 채널이 네 개 이상인 크롭이 필요합니다. 먼저 적합한 크롭 소스를 불러오세요.',
'새 크롭의 채널 수가 더 적습니다. 채널…을 다시 열고 SubCell의 네 평면을 매핑하세요.'
],
'is': [
'Líkan þjálfað á smásjármyndum fremur en ljósmyndum. OpenPhenom og ChAda-ViT taka við hvaða fjölda litunarrása sem er. Tveggja myndplana líkan SubCell tekur DNA og síðan prótein; fjögurra myndplana líkanið krefst skýrrar vörpunar örpíplna, ER, DNA og próteins. Vigtir eru sóttar einu sinni. Cell-DINO þarf opinbera vistun líkans og þessi útgáfa styður það ekki enn. Sjálfgefið: None (nota grunnnetið).',
'Valin SubCell-rás er ekki í innlesnum myndskurðum. Opnið Rásir… og varpið myndplönunum fjórum aftur.',
'Veljið fjórar ólíkar rásir myndskurðanna í opinberri röð SubCell. Ekki er giskað á litun út frá stöðu rásar.',
'Rás {position} (vísitala {index})', 'Rásir…',
'Veljið rás myndskurðanna fyrir hvert SubCell-myndplan: örpíplur, ER, DNA og prótein.',
'Velja rás…', 'Veljið fjórar ólíkar rásir myndskurðanna fyrir SubCell.',
'Veljið fjórar ólíkar rásir myndskurðanna áður en fjögurra myndplana líkan SubCell er keyrt.', 'DNA (B)', 'ER (Y)',
'Fjögur vörpuð myndplön (ein keyrsla)',
'Lesið inn myndskurði með að minnsta kosti fjórum rásum áður en myndplönum SubCell fyrir örpíplur, ER, DNA og prótein er varpað.',
'Varpið fjórum rásum myndskurðanna á örpíplur, ER, DNA og prótein áður en fjögurra myndplana líkan SubCell er keyrt.',
'Örpíplur (R)', 'Prótein (G)', 'SubCell (CZI / Lundberg rannsóknarstofa ViT-B/16, R/Y/B/G)',
'Rásavörpun SubCell vistuð í röðinni örpíplur, ER, DNA, prótein.', 'Rásir fjögurra myndplana SubCell',
'Fjögurra myndplana líkan SubCell þarf myndskurði með að minnsta kosti fjórum rásum. Lesið fyrst inn viðeigandi myndskurðagjafa.',
'Nýju myndskurðirnir hafa færri rásir. Opnið Rásir… aftur til að varpa myndplönunum fjórum í SubCell.'
],
'fr': [
'Un modèle entraîné sur des images de microscopie plutôt que sur des photographies. OpenPhenom et ChAda-ViT acceptent un nombre quelconque de canaux de marquage. Le modèle à deux plans de SubCell utilise DNA puis la protéine ; le modèle à quatre plans exige d’associer explicitement les microtubules, ER, DNA et la protéine. Les poids sont téléchargés une fois. Cell-DINO nécessite un point de contrôle officiel et cette version ne le prend pas encore en charge. Valeur par défaut : None (utiliser le réseau de base).',
'Un canal SubCell sélectionné n’existe pas dans les recadrages chargés. Ouvrez Canaux… et réassociez les quatre plans d’image.',
'Attribuez quatre canaux distincts des recadrages dans l’ordre officiel de SubCell. L’identité du marquage n’est pas déduite de la position du canal.',
'Canal {position} (indice {index})', 'Canaux…',
'Choisissez un canal des recadrages pour chaque plan SubCell : microtubules, ER, DNA et protéine.',
'Choisir un canal…', 'Choisissez quatre canaux distincts des recadrages pour SubCell.',
'Choisissez quatre canaux distincts des recadrages avant d’exécuter le modèle à quatre plans de SubCell.', 'DNA (B)', 'ER (Y)',
'Quatre plans associés (un passage)',
'Chargez des recadrages contenant au moins quatre canaux avant d’associer les plans de microtubules, ER, DNA et protéine de SubCell.',
'Associez quatre canaux des recadrages aux microtubules, ER, DNA et à la protéine avant d’exécuter le modèle à quatre plans de SubCell.',
'Marquage des microtubules (R)', 'Protéine (G)', 'SubCell (CZI / laboratoire Lundberg ViT-B/16, R/Y/B/G)',
'Association des canaux SubCell enregistrée dans l’ordre microtubules, ER, DNA, protéine.', 'Canaux des quatre plans SubCell',
'Le modèle à quatre plans de SubCell nécessite des recadrages contenant au moins quatre canaux. Chargez d’abord une source de recadrages adaptée.',
'Les nouveaux recadrages ont moins de canaux. Rouvrez Canaux… pour associer les quatre plans de SubCell.'
],
}
canonical = {source: writer.runtime_record('ui', source, '') for source in sources}
prepared = {}
for language, values in targets.items():
    assert len(values) == len(sources), (language, len(values))
    runtime.reviewed_runtime_translations(language)
    rows = [{**canonical[source], 'translation': target} for source, target in zip(sources, values)]
    for row in rows:
        assert runtime._contextualize(row['translation'], language, row['source']) == row['translation'], (language, row)
        reasons = runtime._translation_rejection_reasons(row['source'], row['translation'], language, force=runtime._looks_translatable(row['source']))
        assert not reasons, (language, reasons, row)
    prepared[language] = rows
for language, rows in prepared.items():
    path = writer.write(language, '2026-10-06-subcell-rybg', rows)
    document = json.loads(path.read_text())
    document['review_method'] = 'Direct Codex AI technical translation and review of all 21 new channel-mapping captions, preserving official R/Y/B/G markers, image-plane/crop/network meanings and source-bound format fields; no native-speaker signoff.'
    path.write_text(json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + '\n')
    print('PASS directly reviewed SubCell runtime input', language, len(rows), flush=True)
(scratch / 'subcell-rybg-reviewed-runtime-r4.json').write_text(json.dumps(prepared, ensure_ascii=False, indent=2) + '\n')
