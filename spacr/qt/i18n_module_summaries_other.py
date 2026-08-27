"""Portuguese, Icelandic and French translations of module summaries.

The keys mirror the built-in rows in :data:`spacr.qt.app.APPS`. Keeping these
longer scientific descriptions in a dedicated catalog makes them easier for
fluent speakers to review without mixing them into the general UI catalog.
"""

from __future__ import annotations

_BUILTIN_APP_KEYS = (
    "mask",
    "timelapse",
    "motility",
    "measure",
    "annotate",
    "classify_merged",
    "map_barcodes",
    "regression",
    "align",
    "convert",
    "foreign",
    "external_masks",
    "queue",
    "batch",
    "distributed_jobs",
    "db_browser",
    "model_compare",
    "model_zoo",
    "plate_view",
    "agreement",
    "umap",
    "activation",
    "train_compare",
    "classifier_evaluation",
    "run_history",
    "report",
    "analyze_plaques",
    "recruitment",
    "invasion",
    "replication",
)


#: Portuguese, Icelandic and French one-line summaries for every built-in
#: spaCR module. Product names, file formats, model names and established
#: technical abbreviations remain unchanged so they match the UI and API.
MODULE_SUMMARIES_OTHER: dict[str, dict[str, str]] = {
    "pt": {
        "mask": "Gerar máscaras de segmentação para células, núcleos, patógenos e organelas a partir de imagens de microscopia usando Cellpose e alternativas compatíveis",
        "timelapse": "Segmentar e rastrear objetos ao longo dos quadros de uma série temporal",
        "motility": "Quantificar a velocidade e a retidão das trajetórias e estratificar os resultados por estado de infecção",
        "measure": "Quantificar características de intensidade e morfologia por objeto",
        "annotate": "Atribuir anotações a imagens de objetos individuais e armazená-las no banco de dados do projeto",
        "classify_merged": "Treinar classificadores com PyTorch em recortes de imagem ou com gradient boosting em características medidas",
        "map_barcodes": "Associar códigos de barras de sequenciamento aos dados da triagem",
        "regression": "Análise de regressão das pontuações da triagem",
        "align": "Registrar e unir blocos de imagem em um mosaico gravado de forma incremental com uso limitado de memória",
        "convert": "Converter imagens ND2, CZI, LIF e OME-TIFF para o layout TIFF Yokogawa e registrar os mapeamentos dos arquivos de origem",
        "foreign": "Importar imagens, máscaras e uma tabela de medições externas para um projeto spaCR, mapeando as colunas de origem para os campos do spaCR",
        "external_masks": "Importar imagens e máscaras de rótulos externas como um projeto spaCR medido e pronto para anotação",
        "queue": "Executar a mesma sequência de processamento em várias placas",
        "batch": "Colocar módulos, placas e configurações em fila para execução sequencial sem supervisão",
        "distributed_jobs": "Enviar e monitorar execuções do spaCR em estações de trabalho SSH, Slurm ou comandos de nuvem/HPC",
        "db_browser": "Navegar, filtrar e exportar tabelas de measurements.db",
        "model_compare": "Comparar dois modelos Cellpose nos mesmos campos usando máscaras lado a lado, diferenças na contagem de objetos e o índice Rand ajustado (ARI)",
        "model_zoo": "Navegar, verificar, baixar e avaliar modelos Cellpose e classificadores em campos selecionados",
        "plate_view": "Visualizar medições como mapas de calor de placas e detectar efeitos de borda",
        "agreement": "Calcular o κ de Cohen ou de Fleiss entre colunas de anotação e revisar os recortes com anotações discordantes",
        "umap": "Visualizar embeddings UMAP com imagens como glifos",
        "activation": "Gerar mapas de ativação de classe para as previsões do classificador de imagens",
        "train_compare": "Comparar curvas e configurações de várias execuções de treinamento",
        "classifier_evaluation": "Avaliar previsões em dados reservados, CV aninhada, calibração, vazamento de dados e métricas por placa",
        "run_history": "Pesquisar configurações, resultados, avisos, falhas e métricas de desempenho das execuções",
        "report": "Gerar relatórios HTML ou PDF compartilháveis com resultados de QC, figuras, estatísticas, configurações e versões do software",
        "analyze_plaques": "Quantificar medições de ensaios de placas de lise",
        "recruitment": "Quantificar medições de recrutamento molecular",
        "invasion": "Quantificar parasitas aderidos e invadidos com coloração diferencial de duas cores e calcular a eficiência de invasão por poço",
        "replication": "Quantificar parasitas por vacúolo e calcular as taxas de replicação por condição",
    },
    "is": {
        "mask": "Búa til aðgreiningargrímur fyrir frumur, kjarna, sýkla og frumulíffæri úr smásjármyndum með Cellpose og studdum valkostum",
        "timelapse": "Hluta myndir og rekja hluti milli ramma í tímaröð",
        "motility": "Mæla hraða og beinleika ferla og lagskipta niðurstöðum eftir sýkingarástandi",
        "measure": "Magnmæla styrk og formfræðilega eiginleika hvers hlutar",
        "annotate": "Úthluta merkingum á myndir af stökum hlutum og vista þær í verkefnisgagnagrunninum",
        "classify_merged": "Þjálfa flokkara með PyTorch á myndbútum eða gradient boosting á mældum eiginleikum",
        "map_barcodes": "Tengja raðgreiningarstrikamerki við skimunargögn",
        "regression": "Aðhvarfsgreining á skimunarstigum",
        "align": "Samstilla og sauma myndflísar í mósaík sem er skrifað í áföngum með takmarkaðri minnisnotkun",
        "convert": "Umbreyta ND2-, CZI-, LIF- og OME-TIFF-myndum í Yokogawa TIFF-skipulag og skrá vörpun á frumskrár",
        "foreign": "Flytja ytri myndir, grímur og mælingatöflu inn í spaCR-verkefni og varpa upprunadálkum á spaCR-reiti",
        "external_masks": "Flytja myndir og ytri merkigrímur inn sem mælt spaCR-verkefni sem er tilbúið til merkingar",
        "queue": "Keyra sama vinnsluferli á mörgum plötum",
        "batch": "Setja einingar, plötur og stillingar í biðröð fyrir sjálfvirka raðkeyrslu",
        "distributed_jobs": "Senda inn og fylgjast með spaCR-keyrslum á SSH-vinnustöðvum, Slurm eða með skýja-/HPC-skipunum",
        "db_browser": "Skoða, sía og flytja út töflur úr measurements.db",
        "model_compare": "Bera saman tvö Cellpose-líkön á sömu myndsviðum með grímum hlið við hlið, mun á hlutafjölda og leiðréttum Rand-stuðli (ARI)",
        "model_zoo": "Skoða, sannreyna, sækja og afkastaprófa Cellpose- og flokkunarlíkön á völdum myndsviðum",
        "plate_view": "Sýna mælingar sem hitakort platna og greina jaðaráhrif",
        "agreement": "Reikna Cohen eða Fleiss κ milli merkingardálka og fara yfir myndskurði með ósamræmdum merkingum",
        "umap": "Sýna UMAP-innfellingar með myndum sem táknum",
        "activation": "Búa til flokkavirkjunarkort fyrir spár myndflokkara",
        "train_compare": "Bera saman þjálfunarferla og stillingar margra keyrslna",
        "classifier_evaluation": "Meta spár á fráteknum gögnum, hreiðrað CV, kvörðun, gagnaleka og mælikvarða fyrir hverja plötu",
        "run_history": "Leita í keyrslustillingum, úttökum, viðvörunum, bilunum og afkastamælingum",
        "report": "Búa til deilanlegar HTML- eða PDF-skýrslur með QC-niðurstöðum, myndum, tölfræði, stillingum og hugbúnaðarútgáfum",
        "analyze_plaques": "Magnmæla niðurstöður skellugreininga",
        "recruitment": "Magnmæla sameindaaðsöfnun",
        "invasion": "Mæla áfasta og innrásna sníkla með tveggja lita mismunalitun og reikna innrásarhlutfall fyrir hvern brunn",
        "replication": "Mæla fjölda sníkla í hverri frymisbólu og reikna fjölgunarhraða fyrir hvert skilyrði",
    },
    "fr": {
        "mask": "Générer des masques de segmentation des cellules, noyaux, agents pathogènes et organites à partir d'images de microscopie avec Cellpose et les méthodes compatibles",
        "timelapse": "Segmenter et suivre les objets au fil des images d'une série temporelle",
        "motility": "Quantifier la vitesse et la rectitude des trajectoires et stratifier les résultats selon l’état d’infection",
        "measure": "Quantifier les caractéristiques d'intensité et de morphologie de chaque objet",
        "annotate": "Attribuer des annotations aux images d'objets individuels et les enregistrer dans la base de données du projet",
        "classify_merged": "Entraîner des classifieurs avec PyTorch sur des vignettes ou par gradient boosting sur des caractéristiques mesurées",
        "map_barcodes": "Associer les codes-barres de séquençage aux données de criblage",
        "regression": "Analyse de régression des scores de criblage",
        "align": "Recaler et assembler les tuiles d'image dans une mosaïque écrite progressivement avec une utilisation mémoire limitée",
        "convert": "Convertir les images ND2, CZI, LIF et OME-TIFF au format TIFF Yokogawa et consigner leur correspondance avec les fichiers sources",
        "foreign": "Importer des images, des masques et une table de mesures externes dans un projet spaCR, en associant les colonnes sources aux champs spaCR",
        "external_masks": "Importer des images et des masques d'étiquettes externes comme projet spaCR mesuré et prêt à être annoté",
        "queue": "Exécuter la même chaîne de traitement sur plusieurs plaques",
        "batch": "Mettre en file d'attente les modules, plaques et paramètres pour une exécution séquentielle sans surveillance",
        "distributed_jobs": "Soumettre et surveiller des exécutions spaCR sur des stations de travail SSH, Slurm ou au moyen de commandes cloud/HPC",
        "db_browser": "Parcourir, filtrer et exporter les tables de measurements.db",
        "model_compare": "Comparer deux modèles Cellpose sur les mêmes champs à l’aide de masques côte à côte, des écarts de nombre d’objets et de l’indice de Rand ajusté (ARI)",
        "model_zoo": "Parcourir, vérifier, télécharger et évaluer des modèles Cellpose et de classification sur les champs sélectionnés",
        "plate_view": "Visualiser les mesures sous forme de cartes thermiques de plaques et détecter les effets de bord",
        "agreement": "Calculer le κ de Cohen ou de Fleiss entre les colonnes d’annotation et examiner les vignettes dont les annotations divergent",
        "umap": "Visualiser les plongements UMAP avec des images comme glyphes",
        "activation": "Générer des cartes d’activation de classe pour les prédictions d’un classificateur d’images",
        "train_compare": "Comparer les courbes et les paramètres de plusieurs entraînements",
        "classifier_evaluation": "Évaluer les prédictions sur données réservées, la CV imbriquée, l'étalonnage, les fuites de données et les métriques par plaque",
        "run_history": "Rechercher les paramètres, résultats, avertissements, échecs et métriques de performance des exécutions",
        "report": "Générer des rapports HTML ou PDF partageables contenant les résultats de QC, figures, statistiques, paramètres et versions du logiciel",
        "analyze_plaques": "Quantifier les mesures des essais de plaques de lyse",
        "recruitment": "Quantifier les mesures de recrutement moléculaire",
        "invasion": "Quantifier les parasites attachés et ayant pénétré par coloration différentielle à deux couleurs, puis calculer l’efficacité d’invasion par puits",
        "replication": "Quantifier les parasites par vacuole et calculer les taux de réplication par condition",
    },
}


def validate_module_summaries_other() -> None:
    """Raise :class:`AssertionError` if this parallel catalog is incomplete."""
    expected_languages = {"pt", "is", "fr"}
    assert set(MODULE_SUMMARIES_OTHER) == expected_languages
    assert len(_BUILTIN_APP_KEYS) == len(set(_BUILTIN_APP_KEYS)) == 30

    expected_keys = set(_BUILTIN_APP_KEYS)
    key_sets = {frozenset(summaries) for summaries in MODULE_SUMMARIES_OTHER.values()}
    assert key_sets == {frozenset(expected_keys)}
    for language_code, summaries in MODULE_SUMMARIES_OTHER.items():
        assert len(summaries) == 30, language_code
        assert all(isinstance(text, str) and text.strip() for text in summaries.values())
        assert all("http://" not in text and "https://" not in text for text in summaries.values())


validate_module_summaries_other()
