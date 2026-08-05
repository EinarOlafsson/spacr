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
    "classify",
    "ml_analyze",
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
    "make_masks",
    "train_cellpose",
    "cellpose_masks",
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
        "mask": "Gerar máscaras Cellpose para células, núcleos e patógenos",
        "timelapse": "Segmentar e rastrear objetos ao longo dos quadros de uma série temporal",
        "motility": "Ensaio automatizado de motilidade: rastrear a velocidade e realizar o controle de qualidade da infecção",
        "measure": "Medir características de intensidade e morfologia de objetos individuais",
        "annotate": "Anotar imagens de objetos individuais em uma grade e salvá-las no banco de dados",
        "classify": "Treinar CNNs/Transformers com Torch para classificar objetos individuais",
        "ml_analyze": "Aprendizado de máquina clássico (XGBoost/random forest/…) aplicado às características da triagem",
        "map_barcodes": "Associar códigos de barras de sequenciamento aos dados da triagem",
        "regression": "Análise de regressão das pontuações da triagem",
        "align": "Registrar blocos em uma única tela montada, gravada de forma incremental para que um mosaico de 20000 × 20000 nunca precise caber inteiro na RAM",
        "convert": "Converter ND2/CZI/LIF/OME-TIFF em TIFFs Yokogawa: visualizar o mapeamento e depois criar um arquivo que remeta aos originais",
        "foreign": "Importar imagens, máscaras e uma tabela de medições de outra fonte para um projeto spaCR, mapeando suas colunas para a estrutura do spaCR",
        "external_masks": "Transformar imagens e máscaras de rótulos geradas externamente em um projeto spaCR medido e pronto para anotação",
        "queue": "Encadear várias placas no mesmo fluxo de processamento",
        "batch": "Colocar quaisquer módulos, placas e configurações em fila e executá-los durante a noite",
        "distributed_jobs": "Enviar e monitorar execuções do spaCR em estações de trabalho SSH, Slurm ou comandos de nuvem/HPC",
        "db_browser": "Navegar e exportar measurements.db sem usar a CLI sqlite3",
        "make_masks": "Realizar o ajuste fino de modelos Cellpose para o seu conjunto de dados",
        "train_cellpose": "Treinar modelos Cellpose personalizados",
        "cellpose_masks": "Gerar máscaras com Cellpose",
        "model_compare": "Executar dois modelos Cellpose nos mesmos campos: comparar máscaras lado a lado e diferenças na contagem de objetos e no ARI",
        "model_zoo": "Navegar, verificar, baixar e avaliar modelos Cellpose e classificadores em três dos seus campos",
        "plate_view": "Exibir qualquer medição como um mapa de calor da placa e detectar efeitos de borda",
        "agreement": "Calcular o kappa de Cohen/Fleiss entre colunas de anotação e revisar as discordâncias",
        "umap": "Gerar embeddings UMAP com imagens como glifos",
        "activation": "Gerar mapas de ativação",
        "train_compare": "Sobrepor as curvas de várias execuções de treinamento e comparar lado a lado as diferenças entre suas configurações",
        "classifier_evaluation": "Avaliar previsões em dados reservados, CV aninhada, calibração, vazamento de dados e métricas por placa",
        "run_history": "Pesquisar configurações, arquivos, avisos, falhas e desempenho de todas as tarefas",
        "report": "Criar com um clique um HTML/PDF compartilhável com parecer de QC, figuras, estatísticas, configurações e versões",
        "analyze_plaques": "Analisar dados de ensaios de placas de lise",
        "recruitment": "Analisar dados de recrutamento",
        "invasion": "Coloração externa/interna em duas cores: parasitas aderidos em comparação com invadidos e eficiência de invasão por poço",
        "replication": "Endodiogenia: parasitas por vacúolo, convertidos em taxa de replicação por condição",
    },
    "is": {
        "mask": "Búa til Cellpose-grímur fyrir frumur, kjarna og sýkla",
        "timelapse": "Hluta myndir og rekja hluti milli ramma í tímaröð",
        "motility": "Sjálfvirkt hreyfanleikapróf: rekja hraða og framkvæma gæðaeftirlit með sýkingu",
        "measure": "Mæla styrk og formfræðilega eiginleika stakra hluta",
        "annotate": "Merkja myndir af stökum hlutum í hnitaneti og vista þær í gagnagrunni",
        "classify": "Þjálfa Torch CNN/Transformer-líkön til að flokka staka hluti",
        "ml_analyze": "Hefðbundið vélanám (XGBoost/random forest/…) á eiginleikum úr skimun",
        "map_barcodes": "Tengja raðgreiningarstrikamerki við skimunargögn",
        "regression": "Aðhvarfsgreining á skimunarstigum",
        "align": "Samstilla myndflísar í einn samsettan myndflöt sem er skrifaður í áföngum svo 20000 × 20000 mósaík þurfi aldrei allt að rúmast í RAM",
        "convert": "Umbreyta ND2/CZI/LIF/OME-TIFF í Yokogawa TIFF-skrár: forskoða vörpunina og búa síðan til vörpunarskrá sem vísar aftur í frumskrárnar",
        "foreign": "Flytja inn myndir, grímur og mælingatöflu frá öðrum aðila í spaCR-verkefni og varpa dálkum þeirra á gagnaskipan spaCR",
        "external_masks": "Breyta myndum og ytri merkigrímum í mælt spaCR-verkefni sem er tilbúið til merkingar",
        "queue": "Keyra margar plötur í röð í gegnum sama vinnsluferli",
        "batch": "Setja hvaða einingar, plötur og stillingar sem er í biðröð og keyra þær yfir nótt",
        "distributed_jobs": "Senda inn og fylgjast með spaCR-keyrslum á SSH-vinnustöðvum, Slurm eða með skýja-/HPC-skipunum",
        "db_browser": "Skoða og flytja út measurements.db án sqlite3 CLI",
        "make_masks": "Fínstilla Cellpose-líkön fyrir gagnasafnið þitt",
        "train_cellpose": "Þjálfa sérsniðin Cellpose-líkön",
        "cellpose_masks": "Búa til grímur með Cellpose",
        "model_compare": "Keyra tvö Cellpose-líkön á sömu myndsviðum: bera saman grímur hlið við hlið og mun á hlutafjölda og ARI",
        "model_zoo": "Skoða, sannreyna, sækja og afkastaprófa Cellpose- og flokkunarlíkön á þremur myndsviðum þínum",
        "plate_view": "Sýna hvaða mælingu sem er sem hitakort plötu og greina jaðaráhrif",
        "agreement": "Reikna Cohen/Fleiss kappa milli merkingardálka og fara yfir ágreining",
        "umap": "Búa til UMAP-innfellingar með myndum sem táknum",
        "activation": "Búa til virknikort",
        "train_compare": "Leggja ferla margra þjálfunarkeyrslna yfir hvern annan og bera saman stillingamun hlið við hlið",
        "classifier_evaluation": "Meta spár á fráteknum gögnum, hreiðrað CV, kvörðun, gagnaleka og mælikvarða fyrir hverja plötu",
        "run_history": "Leita í stillingum, skrám, viðvörunum, bilunum og afköstum allra verka",
        "report": "Búa með einum smelli til deilanlegt HTML/PDF með niðurstöðu QC, myndum, tölfræði, stillingum og útgáfum",
        "analyze_plaques": "Greina gögn úr skellugreiningu",
        "recruitment": "Greina gögn um aðsöfnun",
        "invasion": "Tveggja lita ytri/innri litun: áfastir miðað við innrásna sníkla og innrásarhlutfall fyrir hvern brunn",
        "replication": "Endodyogeny: fjöldi sníkla í hverri frymisbólu, metinn sem fjölgunarhraði fyrir hvert skilyrði",
    },
    "fr": {
        "mask": "Générer des masques Cellpose pour les cellules, les noyaux et les agents pathogènes",
        "timelapse": "Segmenter et suivre les objets au fil des images d'une série temporelle",
        "motility": "Test de motilité automatisé : suivre la vitesse et effectuer le contrôle qualité de l'infection",
        "measure": "Mesurer les caractéristiques d'intensité et de morphologie de chaque objet",
        "annotate": "Annoter les images d'objets individuels dans une grille et les enregistrer dans la base de données",
        "classify": "Entraîner des CNN/Transformers avec Torch pour classer des objets individuels",
        "ml_analyze": "Apprentissage automatique classique (XGBoost/random forest/…) sur les caractéristiques de criblage",
        "map_barcodes": "Associer les codes-barres de séquençage aux données de criblage",
        "regression": "Analyse de régression des scores de criblage",
        "align": "Recaler les tuiles dans un canevas assemblé unique, écrit progressivement afin qu'une mosaïque de 20000 × 20000 n'ait jamais à tenir entièrement dans la RAM",
        "convert": "Convertir les formats ND2/CZI/LIF/OME-TIFF en TIFF Yokogawa : prévisualiser la correspondance, puis créer un fichier de correspondance vers les originaux",
        "foreign": "Importer les images, les masques et la table de mesures d'une autre source dans un projet spaCR, en faisant correspondre leurs colonnes à la structure de spaCR",
        "external_masks": "Transformer des images et des masques d'étiquettes générés à l'extérieur en un projet spaCR mesuré et prêt pour l'annotation",
        "queue": "Enchaîner plusieurs plaques dans le même pipeline",
        "batch": "Mettre en file d'attente les modules, plaques et paramètres souhaités et les exécuter pendant la nuit",
        "distributed_jobs": "Soumettre et surveiller des exécutions spaCR sur des stations de travail SSH, Slurm ou au moyen de commandes cloud/HPC",
        "db_browser": "Parcourir et exporter measurements.db sans utiliser la CLI sqlite3",
        "make_masks": "Affiner des modèles Cellpose pour votre jeu de données",
        "train_cellpose": "Entraîner des modèles Cellpose personnalisés",
        "cellpose_masks": "Générer des masques avec Cellpose",
        "model_compare": "Exécuter deux modèles Cellpose sur les mêmes champs : comparer les masques côte à côte ainsi que les écarts de nombre d'objets et d'ARI",
        "model_zoo": "Parcourir, vérifier, télécharger et évaluer des modèles Cellpose et de classification sur trois de vos champs",
        "plate_view": "Afficher toute mesure sous forme de carte thermique de plaque et détecter les effets de bord",
        "agreement": "Calculer le kappa de Cohen/Fleiss entre les colonnes d'annotation et examiner les désaccords",
        "umap": "Générer des plongements UMAP avec des images comme glyphes",
        "activation": "Générer des cartes d'activation",
        "train_compare": "Superposer les courbes de plusieurs entraînements et comparer côte à côte les différences entre leurs paramètres",
        "classifier_evaluation": "Évaluer les prédictions sur données réservées, la CV imbriquée, l'étalonnage, les fuites de données et les métriques par plaque",
        "run_history": "Rechercher les paramètres, fichiers, avertissements, échecs et performances de chaque tâche",
        "report": "Créer en un clic un HTML/PDF partageable avec le verdict de QC, les figures, les statistiques, les paramètres et les versions",
        "analyze_plaques": "Analyser les données d'essais de plaques de lyse",
        "recruitment": "Analyser les données de recrutement",
        "invasion": "Coloration extérieure/intérieure à deux couleurs : parasites attachés ou ayant pénétré, et efficacité d'invasion par puits",
        "replication": "Endodyogénie : parasites par vacuole, convertis en taux de réplication pour chaque condition",
    },
}


def validate_module_summaries_other() -> None:
    """Raise :class:`AssertionError` if this parallel catalog is incomplete."""
    expected_languages = {"pt", "is", "fr"}
    assert set(MODULE_SUMMARIES_OTHER) == expected_languages
    assert len(_BUILTIN_APP_KEYS) == len(set(_BUILTIN_APP_KEYS)) == 34

    expected_keys = set(_BUILTIN_APP_KEYS)
    key_sets = {frozenset(summaries) for summaries in MODULE_SUMMARIES_OTHER.values()}
    assert key_sets == {frozenset(expected_keys)}
    for language_code, summaries in MODULE_SUMMARIES_OTHER.items():
        assert len(summaries) == 34, language_code
        assert all(isinstance(text, str) and text.strip() for text in summaries.values())
        assert all("http://" not in text and "https://" not in text for text in summaries.values())


validate_module_summaries_other()
