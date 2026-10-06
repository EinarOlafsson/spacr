from pathlib import Path
import json

import build_documentation_i18n as api
import build_i18n_catalogs as runtime
import write_reviewed_api_record as writer

scratch = Path('/media/carruthers/mnt3/codex/scratch/docs-completion-20261005')
targets = {
    'sv': [
        'Posten registrerar ryggraden, kanalpolicyn och kontrollsumman för läsbara lokala kontrollpunktsfiler utan att ladda ned eller läsa in en modell. Offentliga ryggrader använder cachemetadata från timm; OpenPhenom och ChAda-ViT använder låsta Hugging Face-revisioner; SubCell använder sin officiella kontrollpunkt i torch hub; en lokal DINO-kodare använder sin konfigurerade kontrollpunktssökväg. Poster för grundmodeller anger den ursprungliga leverantören och kontrollpunktens URL. En kontrollsumma ensam verifierar inte en modell och fastställer inte ursprunget för dess träningsdata.',
        'När ingen läsbar kontrollpunkt kan hittas lokalt finns posten ändå och anger detta i sina anteckningar, med en tom kontrollsumma -- vilket :func:`spacr.model_zoo.fetch` redan behandlar som ett avslag, inte ett godkännande. En post som tyst påstod sig ha en kontrollsumma den inte hade beräknat vore sämre än en som medger att den ännu inte kan göra det.',
        'Kanalpolicy: {policy}. Dimensioner från olika policyer är inte jämförbara.',
        'Ingen kontrollsumma: ingen läsbar lokal kontrollpunkt kunde hittas i denna miljö.',
        'Inget utvärderingskort. Mät återhämtning mot märkta fenotypkontroller och bifoga resultaten.'
    ],
    'de': [
        'Der Eintrag erfasst das Backbone, die Kanalrichtlinie und den Hash lesbarer lokaler Checkpoint-Dateien, ohne ein Modell herunterzuladen oder zu laden. Öffentliche Backbones verwenden Cache-Metadaten von timm; OpenPhenom und ChAda-ViT verwenden festgelegte Hugging Face-Revisionen; SubCell verwendet seinen offiziellen Checkpoint im torch hub; ein lokaler DINO-Encoder verwendet seinen konfigurierten Checkpoint-Pfad. Einträge für Grundlagenmodelle nennen den ursprünglichen Anbieter und die Checkpoint-URL. Eine Prüfsumme allein verifiziert kein Modell und belegt nicht die Herkunft seiner Trainingsdaten.',
        'Wenn lokal kein lesbarer Checkpoint ermittelt werden kann, bleibt der Eintrag bestehen und weist in seinen Hinweisen mit einem leeren Hash darauf hin -- was :func:`spacr.model_zoo.fetch` bereits als Ablehnung statt als Freigabe behandelt. Ein Eintrag, der stillschweigend eine nicht berechnete Prüfsumme behauptet, wäre schlechter als einer, der zugibt, sie noch nicht berechnen zu können.',
        'Kanalrichtlinie: {policy}. Dimensionen aus unterschiedlichen Richtlinien sind nicht vergleichbar.',
        'Keine Prüfsumme: In dieser Umgebung konnte kein lesbarer lokaler Checkpoint ermittelt werden.',
        'Keine Bewertung. Messen Sie die Suche anhand beschrifteter Phänotypkontrollen und fügen Sie die Ergebnisse hinzu.'
    ],
    'es': [
        'La entrada registra la red base, la política de canales y el resumen criptográfico de los bytes legibles del punto de control local, sin descargar ni cargar un modelo. Las redes base públicas usan los metadatos de caché de timm; OpenPhenom y ChAda-ViT usan revisiones fijadas de Hugging Face; SubCell usa su punto de control oficial de torch hub; un codificador DINO local usa la ruta de punto de control configurada. Las entradas de modelos fundacionales indican el proveedor original y la URL del punto de control. Una suma de comprobación por sí sola no verifica un modelo ni establece la procedencia de sus datos de entrenamiento.',
        'Cuando no se puede localizar un punto de control legible localmente, la entrada sigue existiendo y lo indica en sus notas, con un resumen vacío -- que :func:`spacr.model_zoo.fetch` ya trata como un rechazo y no como una aprobación. Una entrada que afirmara silenciosamente tener una suma de comprobación que no ha calculado sería peor que una que admite que aún no puede calcularla.',
        'Política de canales: {policy}. Las dimensiones de distintas políticas no son comparables.',
        'Sin suma de comprobación: no se pudo localizar un punto de control local legible en este entorno.',
        'Sin evaluación. Mida la recuperación frente a controles de fenotipo etiquetados y adjunte los resultados.'
    ],
    'zh_CN': [
        '此条目记录骨干网络、通道策略和可读取的本地检查点字节的摘要，不下载或加载模型。公开的骨干网络使用 timm 缓存元数据；OpenPhenom 和 ChAda-ViT 使用固定的 Hugging Face 修订版本；SubCell 使用其官方 torch hub 检查点；本地 DINO 编码器使用其配置的检查点路径。基础模型条目注明原始提供者及检查点 URL。仅有校验和并不能验证模型，也不能确定其训练数据的来源。',
        '如果无法在本地找到可读取的检查点，条目仍然存在，并在备注中说明这一点，同时将摘要留空 -- :func:`spacr.model_zoo.fetch` 已将这种情况视为拒绝，而非通过。一个默默声称拥有未经计算的校验和的条目，比承认目前尚无法计算校验和的条目更糟。',
        '通道策略：{policy}。不同策略的维度不可比较。',
        '无校验和：在此环境中无法找到可读取的本地检查点。',
        '无评估结果。请针对带有标签的表型对照测量检索性能，并附上结果。'
    ],
    'pt': [
        'A entrada regista a rede base, a política de canais e o resumo dos bytes legíveis do checkpoint local, sem transferir nem carregar um modelo. As redes base públicas usam os metadados de cache do timm; OpenPhenom e ChAda-ViT usam revisões fixadas do Hugging Face; SubCell usa o seu checkpoint oficial do torch hub; um codificador DINO local usa o caminho de checkpoint configurado. As entradas de modelos fundacionais indicam o fornecedor original e a URL do checkpoint. Uma soma de verificação, por si só, não verifica um modelo nem estabelece a proveniência dos seus dados de treino.',
        'Quando não é possível localizar um checkpoint legível localmente, a entrada continua a existir e indica-o nas notas, com um resumo vazio -- que :func:`spacr.model_zoo.fetch` já trata como recusa e não como aprovação. Uma entrada que afirmasse silenciosamente ter uma soma de verificação que não calculou seria pior do que uma que admite ainda não a poder calcular.',
        'Política de canais: {policy}. As dimensões de políticas diferentes não são comparáveis.',
        'Sem soma de verificação: não foi possível localizar um checkpoint local legível neste ambiente.',
        'Sem avaliação. Meça a recuperação com controlos de fenótipo rotulados e anexe os resultados.'
    ],
    'hi': [
        'यह प्रविष्टि मॉडल को डाउनलोड या लोड किए बिना बैकबोन, चैनल नीति और पढ़े जा सकने वाले स्थानीय चेकपॉइंट के बाइट का हैश दर्ज करती है। सार्वजनिक बैकबोन timm का कैश मेटाडेटा उपयोग करते हैं; OpenPhenom और ChAda-ViT निश्चित Hugging Face संशोधनों का उपयोग करते हैं; SubCell अपने आधिकारिक torch hub चेकपॉइंट का उपयोग करता है; स्थानीय DINO एन्कोडर अपने कॉन्फ़िगर किए गए चेकपॉइंट पथ का उपयोग करता है। फ़ाउंडेशन मॉडल की प्रविष्टियाँ मूल प्रदाता और चेकपॉइंट URL बताती हैं। केवल चेकसम से मॉडल सत्यापित नहीं होता और उसके प्रशिक्षण डेटा की उत्पत्ति स्थापित नहीं होती।',
        'जब स्थानीय रूप से पढ़ने योग्य चेकपॉइंट नहीं मिल सकता, तब भी प्रविष्टि मौजूद रहती है और अपनी टिप्पणियों में खाली हैश के साथ यह बताती है -- जिसे :func:`spacr.model_zoo.fetch` पहले से स्वीकृति के बजाय अस्वीकृति मानता है। ऐसी प्रविष्टि जो चुपचाप उस चेकसम का दावा करे जिसकी उसने गणना नहीं की है, उस प्रविष्टि से बदतर होगी जो स्वीकार करती है कि वह अभी गणना नहीं कर सकती।',
        'चैनल नीति: {policy}। अलग-अलग नीतियों के आयाम तुलनीय नहीं हैं।',
        'कोई चेकसम नहीं: इस परिवेश में पढ़ने योग्य स्थानीय चेकपॉइंट नहीं मिल सका।',
        'कोई मूल्यांकन नहीं। लेबल किए गए फ़ीनोटाइप नियंत्रणों के विरुद्ध पुनर्प्राप्ति मापें और परिणाम संलग्न करें।'
    ],
    'ko': [
        '이 항목은 모델을 다운로드하거나 로드하지 않고 백본, 채널 정책 및 읽을 수 있는 로컬 체크포인트 바이트의 해시를 기록합니다. 공개 백본은 timm 캐시 메타데이터를 사용하고, OpenPhenom과 ChAda-ViT는 고정된 Hugging Face 리비전을 사용하며, SubCell은 공식 torch hub 체크포인트를 사용합니다. 로컬 DINO 인코더는 설정된 체크포인트 경로를 사용합니다. 파운데이션 모델 항목에는 원래 제공자와 체크포인트 URL이 표시됩니다. 체크섬만으로는 모델이 검증되지 않으며 학습 데이터의 출처도 입증되지 않습니다.',
        '로컬에서 읽을 수 있는 체크포인트를 찾을 수 없어도 항목은 유지되며 빈 해시와 함께 주석에 이를 알립니다 -- :func:`spacr.model_zoo.fetch`는 이미 이를 통과가 아닌 거부로 처리합니다. 계산하지 않은 체크섬이 있다고 조용히 주장하는 항목은 아직 계산할 수 없음을 인정하는 항목보다 나쁩니다.',
        '채널 정책: {policy}. 서로 다른 정책의 차원은 비교할 수 없습니다.',
        '체크섬 없음: 이 환경에서 읽을 수 있는 로컬 체크포인트를 찾을 수 없습니다.',
        '평가 결과 없음. 라벨이 있는 표현형 대조군을 대상으로 검색 성능을 측정하고 결과를 첨부하세요.'
    ],
    'is': [
        'Færslan skráir grunnnetið, rásastefnuna og tætigildi læsilegra bæta í staðbundinni vistun líkansins án þess að sækja eða hlaða líkani. Opinber grunnnet nota lýsigögn úr skyndiminni timm; OpenPhenom og ChAda-ViT nota fastákveðnar Hugging Face-útgáfur; SubCell notar opinbera vistun sína í torch hub; staðbundinn DINO-kóðari notar stillta slóð að vistun líkansins. Færslur grunnlíkana tilgreina upprunalega veitandann og URL vistunarinnar. Gátsumma ein og sér staðfestir hvorki líkan né uppruna þjálfunargagna þess.',
        'Þegar engin læsileg vistun líkans finnst á vélinni er færslan áfram til og segir frá því í athugasemdum sínum með tómu tætigildi -- sem :func:`spacr.model_zoo.fetch` meðhöndlar þegar sem höfnun en ekki samþykki. Færsla sem héldi þegjandi fram gátsummu sem hún hefði ekki reiknað væri verri en færsla sem viðurkennir að hún geti það ekki enn.',
        'Rásastefna: {policy}. Víddir úr ólíkum stefnum eru ekki samanburðarhæfar.',
        'Engin gátsumma: engin læsileg staðbundin vistun líkans fannst í þessu umhverfi.',
        'Ekkert mat. Mælið endurheimt með merktum svipgerðarviðmiðum og bætið niðurstöðunum við.'
    ],
    'fr': [
        'L’entrée enregistre le réseau de base, la politique des canaux et l’empreinte des octets lisibles du point de contrôle local, sans télécharger ni charger de modèle. Les réseaux de base publics utilisent les métadonnées du cache de timm ; OpenPhenom et ChAda-ViT utilisent des révisions fixées de Hugging Face ; SubCell utilise son point de contrôle officiel de torch hub ; un encodeur DINO local utilise le chemin de point de contrôle configuré. Les entrées des modèles de fondation indiquent le fournisseur d’origine et l’URL du point de contrôle. Une somme de contrôle ne suffit pas à vérifier un modèle ni à établir la provenance de ses données d’entraînement.',
        'Lorsqu’aucun point de contrôle lisible ne peut être trouvé localement, l’entrée existe toujours et l’indique dans ses notes, avec une empreinte vide -- que :func:`spacr.model_zoo.fetch` traite déjà comme un refus plutôt qu’une validation. Une entrée qui prétendrait discrètement disposer d’une somme de contrôle qu’elle n’a pas calculée serait pire qu’une entrée qui admet ne pas encore pouvoir la calculer.',
        'Politique des canaux : {policy}. Les dimensions issues de politiques différentes ne sont pas comparables.',
        'Aucune somme de contrôle : aucun point de contrôle local lisible n’a pu être trouvé dans cet environnement.',
        'Aucune évaluation. Mesurez la recherche à partir de témoins phénotypiques étiquetés et joignez les résultats.'
    ],
}
sources = [
    'Channel policy: {policy}. Dimensions from different policies are not comparable.',
    'No checksum: no readable local checkpoint could be resolved in this environment.',
    'No scorecard. Measure retrieval against labelled phenotype controls and attach the results.',
]
docs = api.public_docstrings()
prepared = {}
for language, translations in targets.items():
    assert len(translations) == 5
    api_entries = [writer.record('spacr.embeddings.encoder_entry#' + str(index), target)
                   for index, target in zip((2, 3), translations[:2])]
    for entry in api_entries:
        assert api._reviewed_api_block_valid(entry['source'], entry['translation'], language), (language, entry)
        assert api._reviewed_api_block_valid(entry['context'], entry['translation'], language), (language, entry)
    ui_entries = [writer.runtime_record('ui', source, target)
                  for source, target in zip(sources, translations[2:])]
    prepared[language] = (api_entries, ui_entries)
for language, (api_entries, ui_entries) in prepared.items():
    for entries in (api_entries, ui_entries):
        path = writer.write(language, '2026-10-06-encoder-provenance', entries)
        document = json.loads(path.read_text())
        document['review_method'] = 'Direct Codex AI technical translation and review; no native-speaker signoff'
        path.write_text(json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + '\n')
    reviewed_api = api.reviewed_api_block_translations(docs, language)
    reviewed_runtime = runtime.reviewed_runtime_translations(language)
    assert all(reviewed_api[entry['source']] == entry['translation'] for entry in api_entries)
    assert all(reviewed_runtime[entry['source']] == entry['translation'] for entry in ui_entries)
    print('PASS', language, 'two API blocks and three source-bound runtime notes', flush=True)
(scratch / 'foundation-api-reviewed-inputs-r1.json').write_text(json.dumps({
    'review_method': 'Direct Codex AI technical translation and review; no native-speaker signoff',
    'languages': {language: {'API': rows[0], 'runtime': rows[1]} for language, rows in prepared.items()},
}, ensure_ascii=False, indent=2) + '\n')
