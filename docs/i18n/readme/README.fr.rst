|Docs| |Tutorials| |PyPI| |Python| |Tests| |Qt| |Source| |Issues| |License| |DOI|

.. |Docs| image:: https://github.com/EinarOlafsson/spacr/actions/workflows/pages/pages-build-deployment/badge.svg
   :target: https://einarolafsson.github.io/spacr/
   :alt: Documentation
.. |Tutorials| image:: https://img.shields.io/badge/Tutorials-Interactive%20walkthrough-4A9EFF
   :target: https://einarolafsson.github.io/spacr/tutorials/
   :alt: Tutoriels interactifs
.. |PyPI| image:: https://img.shields.io/pypi/v/spacr
   :target: https://pypi.org/project/spacr/
   :alt: Version PyPI
.. |Python| image:: https://img.shields.io/badge/Python-3.9%E2%80%933.14-3776AB?logo=python&logoColor=white
   :target: https://pypi.org/project/spacr/
   :alt: Python 3.9 à 3.14
.. |Tests| image:: https://github.com/EinarOlafsson/spacr/actions/workflows/tests.yml/badge.svg
   :target: https://github.com/EinarOlafsson/spacr/actions/workflows/tests.yml
   :alt: Suite de tests
.. |Qt| image:: https://img.shields.io/badge/GUI-Qt%20%28PySide6%29-41CD52
   :target: https://einarolafsson.github.io/spacr/
   :alt: Interface Qt
.. |Source| image:: https://img.shields.io/badge/GitHub-Source-181717?logo=github
   :target: https://github.com/EinarOlafsson/spacr
   :alt: Code source GitHub
.. |Issues| image:: https://img.shields.io/github/issues/EinarOlafsson/spacr
   :target: https://github.com/EinarOlafsson/spacr/issues
   :alt: Tickets GitHub
.. |License| image:: https://img.shields.io/github/license/EinarOlafsson/spacr
   :target: https://github.com/EinarOlafsson/spacr/blob/main/LICENSE
   :alt: Licence PolyForm Noncommercial
.. |DOI| image:: https://img.shields.io/badge/DOI-10.5281%2Fzenodo.21343317-blue
   :target: https://doi.org/10.5281/zenodo.21343317
   :alt: DOI Zenodo
.. |Release| image:: https://img.shields.io/github/v/release/EinarOlafsson/spacr?label=Installers
   :target: https://github.com/EinarOlafsson/spacr/releases/latest
   :alt: Derniers installateurs
.. |CondaRecipe| image:: https://img.shields.io/badge/conda--forge-recipe-44A833?logo=anaconda
   :target: https://github.com/EinarOlafsson/spacr/tree/main/conda-forge/recipe
   :alt: Recette conda-forge

.. image:: https://raw.githubusercontent.com/EinarOlafsson/spacr/main/spacr/resources/icons/logo_spacr.png
   :alt: spaCR
   :align: center
   :width: 360

spaCR
=====

Langues: `English <../../../README.rst>`_ · `Svenska <README.sv.rst>`_ ·
`Deutsch <README.de.rst>`_ ·
`Español <README.es.rst>`_ ·
`简体中文 <README.zh_CN.rst>`_ ·
`Português <README.pt.rst>`_ ·
`हिन्दी <README.hi.rst>`_ ·
`한국어 <README.ko.rst>`_ ·
`Íslenska <README.is.rst>`_ ·
`Français <README.fr.rst>`_

`Informations sur les modèles de traduction <../TRANSLATION_MODELS.md>`_

**Analyse spatiale des phénotypes de criblages CRISPR.**

spaCR segmente et mesure les cellules individuelles dans des images de microscopie à haut contenu, associe chaque cellule au gRNA qu’elle a reçu et indique quels gènes ont modifié le phénotype. Les images de plaques et les lectures FASTQ constituent les entrées ; les mesures par objet, les classificateurs entraînés, les tailles d’effet par guide et par gène et une liste de résultats classés constituent les sorties.

Pour les criblages CRISPR groupés fondés sur l’imagerie, ce flux couvre l’ensemble du parcours. Avec des images de microscopie à haut contenu mais sans criblage, les étapes de segmentation, de mesure, d’annotation et de classification peuvent être exécutées indépendamment.

Les images, masques, recadrages, mesures, annotations, prédictions, codes-barres et identifiants de puits sont conservés dans un même projet SQLite, ce qui permet de relier chaque valeur d’un résultat à son objet d’origine.

Exécutez spaCR comme application de bureau ou sans interface graphique sur une station de travail, un serveur ou un cluster. Les deux modes utilisent les mêmes modules et CUDA est activé automatiquement lorsqu’un module le prend en charge.


Vue d’ensemble du flux de travail
---------------------------------

|Tutorials|

.. image:: https://raw.githubusercontent.com/EinarOlafsson/spacr/main/spacr/resources/icons/flow_chart_v3.png
   :alt: Flux de travail spaCR et organisation des sorties
   :align: center

Les images de microscopie (TIFF, OME-TIFF, LIF, CZI, ND2) et les lectures de séquençage (FASTQ) alimentent des flux complémentaires d’analyse d’images et d’association des codes-barres. Les tables d’objets, recadrages, annotations, prédictions, identités des guides, résultats de QC et résumés par puits sont ensuite analysés ensemble.


Démarrage rapide
----------------

.. code-block:: bash

   conda create -n spacr python=3.12 -y
   conda activate spacr
   python -m pip install --upgrade pip
   python -m pip install "spacr[qt]"
   spacr

spaCR prend en charge Python **3.9 à 3.14** (à l’exception de Python 3.14.1, exclu par torchvision). Python 3.12 offre le plus grand choix de paquets scientifiques facultatifs. Linux est recommandé pour les flux de travail CUDA ; macOS et Windows sont également pris en charge.


Détails de l’installation
-------------------------

|Release| |PyPI| |CondaRecipe|

**(beta) Installateurs de bureau légers:**

.. spacr-installer-links-begin

* `Windows 10/11: télécharger SpaCR 1.5.0.4 <https://github.com/EinarOlafsson/spacr/releases/download/v1.5.0.4/SpaCR-1.5.0.4-Windows-Online-Setup.exe>`_
* `macOS 11+ (Intel et Apple silicium): télécharger SpaCR 1.5.0.4 <https://github.com/EinarOlafsson/spacr/releases/download/v1.5.0.4/SpaCR-1.5.0.4-macOS-Universal-Online.pkg>`_
* `64 bits Linux: télécharger SpaCR 1.5.0.4 <https://github.com/EinarOlafsson/spacr/releases/download/v1.5.0.4/SpaCR-1.5.0.4-Linux-x86_64-Online.run>`_

.. spacr-installer-links-end

Programmes d’installation légers — ni conda ni installation Python existante requis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Pendant l’installation, l’installateur télécharge un environnement privé Python 3.12, Qt, PyTorch, spaCR et les dépendances scientifiques ; conda et une installation Python existante ne sont donc pas nécessaires. La version portable pour CPU est utilisée par défaut afin d’éviter le téléchargement inopiné de plusieurs gigaoctets de bibliothèques CUDA. Windows propose l’accélération NVIDIA comme composant facultatif, Linux accepte ``--torch-backend auto`` et le wheel PyTorch standard pour macOS conserve l’accélération Apple MPS.

L’aide, la progression et les messages d’erreur de l’installateur suivent la langue du système d’exploitation dans les dix langues de spaCR : anglais, suédois, allemand, espagnol, chinois simplifié, portugais, hindi, coréen, islandais et français. Les paramètres régionaux non pris en charge utilisent l’anglais.

Sous Linux, rendez l’installateur téléchargé exécutable avant de l’ouvrir :

.. code-block:: bash

   chmod +x SpaCR-*-Linux-x86_64-Online.run
   ./SpaCR-*-Linux-x86_64-Online.run

Sous macOS, ouvrez le fichier ``.pkg`` téléchargé. Si Gatekeeper bloque l’installateur bêta actuel parce qu’il n’est pas notarié, ouvrez **Réglages Système → Confidentialité et sécurité**, choisissez **Ouvrir quand même** pour spaCR, puis relancez le paquet.

Avant de remplacer une installation antérieure, l’installateur vérifie spaCR, Qt, PyTorch et la cohérence des dépendances. Une mise à jour interrompue laisse donc l’environnement fonctionnel précédent en place. Un journal de diagnostic nommé ``install.log`` est conservé dans le répertoire d’installation privé de spaCR.

Application de bureau depuis PyPI
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   python -m pip install "spacr[qt]"
   spacr

Installation sans interface graphique ou sur serveur
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   python -m pip install spacr
   spacr-run --list

Branche de développement la plus récente
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   git clone https://github.com/EinarOlafsson/spacr.git
   cd spacr
   git switch nightly
   python -m pip install -e ".[qt]"

Environnements conda
~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   conda create -n spacr python=3.12 pip -y
   conda activate spacr
   python -m pip install "spacr[qt]"

Fonctionnalités facultatives
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Installez uniquement les extras dont votre workflow a besoin :

.. code-block:: bash

   python -m pip install "spacr[trackastra]"    # transformer tracking
   python -m pip install "spacr[ultrack]"       # global-optimization tracking
   python -m pip install "spacr[btrack]"        # btrack timelapse tracking
   python -m pip install "spacr[attribution]"   # TorchCAM methods
   python -m pip install "spacr[boosting]"      # LightGBM and CatBoost
   python -m pip install "spacr[zernike]"       # Zernike measurements
   python -m pip install "spacr[napari]"        # napari mask correction
   python -m pip install "spacr[czi,nd2,lif]"   # vendor file readers

Les extras disponibles dépendent de la version de Python. Sous Python 3.13, ultrack limite ``spacr[all]`` et la contrainte NumPy de TorchCAM limite l’extra ``attribution`` ; le paquet principal et l’application Qt ne sont pas affectés. Sous Python 3.14, btrack est disponible par l’intermédiaire de son extra. Le convertisseur CZI pylibCZIrw est facultatif et non testé ; la lecture des fichiers CZI avec czifile reste disponible.

L'interface Tk est toujours installée sous la forme ``spacr-legacy`` mais n'est plus développée.


Points d’entrée en ligne de commande
------------------------------------

.. code-block:: bash

   spacr                                      # Qt application
   spacr-doctor                               # diagnose the installation
   spacr-run --list                           # list headless modules
   spacr-run --describe MODULE                # inspect a module contract
   spacr-run MODULE --settings settings.csv   # execute a module
   spacr-run validate --module MODULE \
       --settings settings.csv                # validate before running
   spacr-repro RUN_DIR                        # replay a recorded run

Définissez ``SPACR_LOG_LEVEL=DEBUG`` lors du dépannage. Les journaux rotatifs sont écrits dans ``~/.spacr/logs/spacr.log``.


Fonctionnalités
---------------

Les six modules les plus utilisés dans les criblages
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Mask** segmente avec Cellpose les cellules, noyaux, agents pathogènes et organites dans les images 2D ainsi que dans les données volumétriques ou les séries temporelles. La liste des modèles provient de l’installation de Cellpose au lieu d’être codée en dur, et le diamètre des objets est estimé à partir des images avant l’exécution. Les masques peuvent être corrigés manuellement dans la visionneuse de calques, ou envoyés dans napari pour y être modifiés puis réimportés.

**Measure** enregistre dans la base de données du projet les caractéristiques de morphologie, d’intensité, de texture et de colocalisation de chaque objet, avec les recadrages correspondants. Nouveauté de la version 1.5.0.0 : la correction d’éclairage estime le champ uniforme à partir de la plaque et corrige les images avant le calcul des caractéristiques d’intensité. Elle supprime ainsi le biais lié à la position des puits, visible sous forme d’effets de bord dans les cartes thermiques de la plaque. Avant l’exécution de Measure, une bannière de QC de segmentation décrit les masques en langage clair ; elle informe sans bloquer l’exécution. Un polygone tracé limite les mesures à une région d’intérêt.

**Annotate** affiche les recadrages dans une grille pilotée au clavier et enregistre directement les étiquettes dans SQLite. La boucle d’apprentissage actif est intégrée à l’écran : réentraînement sur les données déjà annotées, reclassement de la file selon l’incertitude, suivi de la courbe d’apprentissage et recommandation d’arrêt lorsque de nouvelles annotations ne modifient plus le modèle. La couverture est indiquée par classe, par puits et par plaque, et chaque cycle est enregistré.

**Classify** entraîne des CNN et des Transformer PyTorch sur les recadrages annotés, ainsi que des modèles classiques ou de boosting sur les tables de mesures. La précision par classe est désormais conservée à chaque epoch, et chaque checkpoint reçoit une fiche indiquant le jeu de données, l’équilibre des classes, la règle de partition et les métriques de l’ensemble de validation. Dans l’écran d’évaluation, chaque cellule de la matrice de confusion sert de requête : un clic ouvre les recadrages correspondants et sépare les erreurs à forte confiance des cas incertains.

**Map Barcodes** décode les codes-barres de ligne, de colonne et de gRNA dans les lectures FASTQ, attribue les identités des guides aux puits et les relie aux cellules imagées. Barcode QC indique le nombre de lectures par puits, le taux de collision et la fraction non attribuée, en explorant une plage autour du nombre de gRNA par puits attendu par l’utilisateur plutôt qu’un seuil fixe.

**Regression** estime les effets des guides, des gènes, des conditions et des témoins avec 17 familles de modèles, notamment les modèles mixtes, Logistic, Probit, Quantile, Beta, les GLM à variance quasi binomiale, Lasso, Ridge, Elastic Net, Hinge et Horseshoe. Le résultat est une liste de candidats classée et annotée, et non un simple ensemble de coefficients.

Nouveautés de la version 1.5.0.0
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Avant même qu’un criblage existe, le module Power / Design calcule le nombre de cellules et de puits nécessaires en tenant compte des erreurs de séquençage et de la perte des puits contenant trop peu de cellules imagées. Un concepteur d’expérience organise la plaque, les témoins et les réplicats, puis exporte le plan pour le pipeline. Ensuite, un tableau de bord de QC rassemble les contrôles de segmentation, de plaque, d’accord entre annotateurs et de fuite de données en un verdict unique ; ComBat est disponible aux côtés de ``center`` et ``zscore`` pour la correction des effets de lot.

Les résultats sont explorés directement plutôt qu’exportés puis réimportés. Graph Builder crée un graphique en faisant glisser des colonnes vers x, y, couleur, taille et facette. Les zones tracées dans un histogramme ou un nuage de points deviennent des filtres. Un explorateur de caractéristiques les classe selon leur capacité à séparer les classes. Les petits multiples, les ajustements dose-réponse, les cartes de contrôle et la détection robuste des valeurs aberrantes utilisent le même moteur d’axes. La sélection d’objets dans une vue se propage à toutes les autres ; l’ouverture de la sélection affiche les recadrages d’origine. Une visionneuse de calques superpose images, étiquettes, points et formes, avec des vues orthogonales, une grille de comparaison synchronisée et un arbre de filiation reliant cellule, noyau et agent pathogène.

Chaque exécution est maintenant identifiable par un ID, une graine aléatoire et une politique ``on_error``. Mask, Measure, Classify et l’export AnnData inscrivent leurs sorties dans un registre d’artefacts, ce qui permet de remonter d’un fichier de sortie aux paramètres qui l’ont produit. Un module ouvre la sortie réellement écrite par l’étape précédente, le graphe du pipeline signale les sorties obsolètes, la comparaison des exécutions affiche les différences de paramètres, de nombres d’objets et de listes de candidats, et chaque exécution depuis l’interface produit le script Python équivalent. Les mesures s’exportent au format ``.h5ad`` pour scanpy ; OME-Zarr et OMERO sont disponibles dans l’API Python. L’exportateur de méthodes et de résultats rédige ces deux sections du manuscrit à partir d’un résumé structuré de l’exécution : le modèle rédige le texte, mais chaque nombre provient du résumé, et tout brouillon contenant un nombre absent de celui-ci est rejeté. En cas de problème d’installation, ``spacr-doctor`` indique quelle installation de spaCR est réellement exécutée, si le GPU est utilisable, si Cellpose correspond à l’API appelée par spaCR et si la base de données et les paramètres du projet sont valides ; chaque contrôle en échec est accompagné d’une correction copiable.

Interface de bureau multilingue
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**spaCR → Préférences → Langue** retraduit l’application en cours d’exécution en anglais, suédois, allemand, espagnol, chinois simplifié, portugais, hindi, coréen, islandais ou français, sans redémarrage. Le choix est mémorisé et s’applique aux écrans ouverts par la suite.

La navigation, les préférences, les commandes AI et LIVE, les descriptions des modules et les messages de console produits par spaCR suivent la langue sélectionnée. Les sorties des processus, journaux, traces d’erreur, chemins, valeurs de base de données, annotations, réponses AI, mesures et résultats enregistrés ne sont jamais traduits : les résultats scientifiques restent ainsi dans leur forme anglaise canonique. Les infobulles de réglage qui n’ont pas encore été relues dans une langue restent en anglais afin d’éviter les explications bilingues. Le `guide de localisation <https://einarolafsson.github.io/spacr/localization.html>`_ décrit ce comportement, le remplacement par variable d’environnement et l’`aide contextuelle <https://einarolafsson.github.io/spacr/localization.html#contextual-help>`_ traduite avec l’interface.

Guide animé des paramètres
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

94 courtes animations montrent l’effet de 143 paramètres visuels sur une image. Survolez un paramètre puis cliquez sur **Animation** dans son infobulle pour lancer l’aperçu carré à côté du texte ; cliquez de nouveau pour le replier. Les animations ne démarrent qu’à la demande et peuvent être entièrement désactivées dans les Préférences. La `galerie <https://einarolafsson.github.io/spacr/setting_animations.html>`_ les présente toutes, et le `registre des animations de paramètres <https://einarolafsson.github.io/spacr/api/spacr/setting_animations/index.html>`_ indique le paramètre associé à chacune.

Référence des modules
~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 25 25 25

   * - Module
     - Fonction
     - État
     - Description
   * - **Expérience de bureau**
     -
     -
     -
   * - |api-qt-app|_
     - |doc-i18n|_
     - Stable
     - Retraduit instantanément les écrans ouverts ou créés à la demande dans les dix langues fournies.
   * - |api-qt-app|_
     - |doc-i18n-help|_
     - Stable
     - Localise les résumés de modules et l’interface d’aide des paramètres sans modifier les URL de l’API.
   * - |api-qt-ai|_
     - |api-qt-ai-console|_
     - Stable
     - Localise les commandes AI et LIVE sans modifier le contenu de l’utilisateur ni celui du modèle.
   * - |api-animations|_
     - |doc-animations|_
     - Stable
     - Lit depuis les infobulles 94 animations intégrées couvrant 143 paramètres visuels.
   * - |api-selection|_
     - |api-linked-views|_
     - Alpha
     - Partage une même sélection d’objets entre les vues tableau, plaque, plongement, nuage de points et graphique.
   * - |api-doctor|_
     - |api-doctor-checks|_
     - Alpha
     - Vérifie le GPU, l’API Cellpose, la base de données et les paramètres, avec une solution pour chaque contrôle en échec.
   * - **Analyse d’images**
     -
     -
     -
   * - |api-mask|_
     - |api-mask-2d|_
     - Stable
     - Segmente les cellules, noyaux, agents pathogènes et organites dans les images 2D.
   * - |api-mask|_
     - |api-mask-3d|_
     - Bêta
     - Segmente les images volumétriques et les séries temporelles 4D.
   * - |api-illumination|_
     - |api-flatfield|_
     - Alpha
     - Estime le champ uniforme à partir de la plaque et le corrige avant la mesure de l’intensité.
   * - |api-measure|_
     - |api-measure-2d|_
     - Stable
     - Mesure la morphologie, l’intensité, la texture et la colocalisation, puis enregistre les recadrages.
   * - |api-segqc|_
     - |api-segqc-verdict|_
     - Alpha
     - Décrit la qualité de la segmentation avant l’exécution de Measure, sans la bloquer.
   * - |api-timelapse|_
     - |api-tracking|_
     - Bêta
     - Suit les objets avec IoU, Trackpy, btrack, Trackastra ou ultrack et quantifie leur motilité.
   * - |api-layers|_
     - |api-layer-viewer|_
     - Alpha
     - Superpose les calques d’images, d’étiquettes, de points et de formes, avec vues orthogonales et grille de comparaison.
   * - |api-napari|_
     - |api-napari-curation|_
     - Alpha
     - Envoie un masque à napari pour correction, le récupère et consigne chaque modification.
   * - **AI et phénotypage**
     -
     -
     -
   * - |api-annotate|_
     - |api-annotation|_
     - Stable
     - Examine les recadrages dans une grille pilotée au clavier et enregistre les annotations dans SQLite.
   * - |api-active-learning|_
     - |api-al-loop|_
     - Alpha
     - Réentraîne le modèle dans Annotate, reclasse par incertitude et indique quand l’annotation peut s’arrêter.
   * - |api-classify|_
     - |api-classification|_
     - Stable
     - Entraîne et applique des CNN et des modèles transformer PyTorch.
   * - |api-classify|_
     - |api-model-cards|_
     - Alpha
     - Consigne pour chaque point de contrôle le jeu de données, l’équilibre des classes, la règle de partition et les métriques de validation.
   * - |api-confusion|_
     - |api-confusion-drill|_
     - Alpha
     - Ouvre les recadrages associés à une cellule de la matrice de confusion et sépare les erreurs certaines des cas incertains.
   * - |api-ml|_
     - |api-ml-models|_
     - Stable
     - Entraîne des modèles classiques et de boosting interprétables sur les tables de mesures.
   * - |api-classify|_
     - |api-activation|_
     - Bêta
     - Explique les prédictions avec Captum, SmoothGrad et TorchCAM.
   * - |api-umap|_
     - |api-embedding|_
     - Bêta
     - Explore interactivement les plongements d’images et propage les étiquettes de groupes.
   * - **Séquençage et analyse de criblage**
     -
     -
     -
   * - |api-sequencing|_
     - |api-barcodes|_
     - Stable
     - Associe les codes-barres de ligne, de colonne et de gRNA des lectures FASTQ et attribue les guides aux cellules imagées.
   * - |api-barcode-qc|_
     - |api-barcode-qc-sweep|_
     - Alpha
     - Rapporte les lectures par puits, le taux de collision et la fraction non attribuée selon le nombre de gRNA attendu par puits.
   * - |api-regression|_
     - |api-regression-models|_
     - Stable
     - Estime les effets des guides, gènes, conditions et témoins à l’aide de 17 familles de modèles.
   * - |api-power|_
     - |api-power-design|_
     - Alpha
     - Calcule le nombre de cellules et de puits requis pour un criblage en tenant compte des erreurs de séquençage et des puits perdus.
   * - |api-graph|_
     - |api-graph-builder|_
     - Alpha
     - Construit un graphique en faisant glisser des colonnes vers x, y, couleur, taille et facette.
   * - |api-artifacts|_
     - |api-provenance|_
     - Alpha
     - Consigne l’identifiant d’exécution, la graine et les paramètres à l’origine des sorties Mask, Measure, Classify et d’exportation.

.. |api-qt-app| replace:: **Application Qt**
.. _api-qt-app: https://einarolafsson.github.io/spacr/api/spacr/qt/app/index.html

.. |doc-i18n| replace:: **Localisation en dix langues**
.. _doc-i18n: https://einarolafsson.github.io/spacr/localization.html

.. |doc-i18n-help| replace:: **Aide contextuelle localisée**
.. _doc-i18n-help: https://einarolafsson.github.io/spacr/localization.html#contextual-help

.. |api-qt-ai| replace:: **Qt AI**
.. _api-qt-ai: https://einarolafsson.github.io/spacr/api/spacr/qt/ai/index.html

.. |api-qt-ai-console| replace:: **Console assistée par AI**
.. _api-qt-ai-console: https://einarolafsson.github.io/spacr/api/spacr/qt/ai/index.html

.. |api-animations| replace:: **Registre des animations de paramètres**
.. _api-animations: https://einarolafsson.github.io/spacr/api/spacr/setting_animations/index.html

.. |doc-animations| replace:: **Animations des paramètres visuels**
.. _doc-animations: https://einarolafsson.github.io/spacr/setting_animations.html

.. |api-selection| replace:: **Sélection**
.. _api-selection: https://einarolafsson.github.io/spacr/api/spacr/selection/index.html

.. |api-linked-views| replace:: **Sélection liée**
.. _api-linked-views: https://einarolafsson.github.io/spacr/api/spacr/qt/linked_selection/index.html

.. |api-doctor| replace:: **Doctor**
.. _api-doctor: https://einarolafsson.github.io/spacr/api/spacr/doctor/index.html

.. |api-doctor-checks| replace:: **Diagnostic de l’installation**
.. _api-doctor-checks: https://einarolafsson.github.io/spacr/api/spacr/doctor/index.html

.. |api-mask| replace:: **Mask**
.. _api-mask: https://einarolafsson.github.io/spacr/api/spacr/core/index.html

.. |api-mask-2d| replace:: **Génération de masques 2D**
.. _api-mask-2d: https://einarolafsson.github.io/spacr/api/spacr/core/index.html#spacr.core.preprocess_generate_masks

.. |api-mask-3d| replace:: **Génération de masques 3D et 4D**
.. _api-mask-3d: https://einarolafsson.github.io/spacr/api/spacr/core/index.html#spacr.core.preprocess_generate_masks

.. |api-illumination| replace:: **Éclairage**
.. _api-illumination: https://einarolafsson.github.io/spacr/api/spacr/illumination/index.html

.. |api-flatfield| replace:: **Correction de champ uniforme**
.. _api-flatfield: https://einarolafsson.github.io/spacr/api/spacr/illumination/index.html

.. |api-measure| replace:: **Measure**
.. _api-measure: https://einarolafsson.github.io/spacr/api/spacr/measure/index.html

.. |api-measure-2d| replace:: **Mesures des objets**
.. _api-measure-2d: https://einarolafsson.github.io/spacr/api/spacr/measure/index.html#spacr.measure.measure_crop

.. |api-segqc| replace:: **QC de segmentation**
.. _api-segqc: https://einarolafsson.github.io/spacr/api/spacr/seg_qc/index.html

.. |api-segqc-verdict| replace:: **Avis avant exécution**
.. _api-segqc-verdict: https://einarolafsson.github.io/spacr/api/spacr/seg_qc/index.html

.. |api-timelapse| replace:: **Timelapse**
.. _api-timelapse: https://einarolafsson.github.io/spacr/api/spacr/timelapse/index.html

.. |api-tracking| replace:: **Suivi des objets**
.. _api-tracking: https://einarolafsson.github.io/spacr/api/spacr/timelapse/index.html

.. |api-layers| replace:: **Calques**
.. _api-layers: https://einarolafsson.github.io/spacr/api/spacr/layers/index.html

.. |api-layer-viewer| replace:: **Visionneuse de calques**
.. _api-layer-viewer: https://einarolafsson.github.io/spacr/api/spacr/qt/layer_viewer/index.html

.. |api-napari| replace:: **Passerelle napari**
.. _api-napari: https://einarolafsson.github.io/spacr/api/spacr/napari_bridge/index.html

.. |api-napari-curation| replace:: **Correction des masques**
.. _api-napari-curation: https://einarolafsson.github.io/spacr/api/spacr/napari_bridge/index.html

.. |api-annotate| replace:: **Annotate**
.. _api-annotate: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/annotate/index.html

.. |api-annotation| replace:: **Annotation manuelle**
.. _api-annotation: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/annotate/index.html

.. |api-active-learning| replace:: **Apprentissage actif**
.. _api-active-learning: https://einarolafsson.github.io/spacr/api/spacr/active_learning/index.html

.. |api-al-loop| replace:: **Réentraîner et reclasser**
.. _api-al-loop: https://einarolafsson.github.io/spacr/api/spacr/active_learning/index.html

.. |api-classify| replace:: **Classify**
.. _api-classify: https://einarolafsson.github.io/spacr/api/spacr/deep_spacr/index.html

.. |api-classification| replace:: **Classification d’images**
.. _api-classification: https://einarolafsson.github.io/spacr/api/spacr/deep_spacr/index.html

.. |api-model-cards| replace:: **Fiches de modèles**
.. _api-model-cards: https://einarolafsson.github.io/spacr/api/spacr/deep_spacr/index.html

.. |api-activation| replace:: **Cartes d’activation**
.. _api-activation: https://einarolafsson.github.io/spacr/api/spacr/deep_spacr/index.html

.. |api-confusion| replace:: **Confusion**
.. _api-confusion: https://einarolafsson.github.io/spacr/api/spacr/confusion/index.html

.. |api-confusion-drill| replace:: **Exploration de la matrice de confusion**
.. _api-confusion-drill: https://einarolafsson.github.io/spacr/api/spacr/confusion/index.html

.. |api-ml| replace:: **Apprentissage automatique**
.. _api-ml: https://einarolafsson.github.io/spacr/api/spacr/ml/index.html

.. |api-ml-models| replace:: **Classification des mesures**
.. _api-ml-models: https://einarolafsson.github.io/spacr/api/spacr/ml/index.html

.. |api-umap| replace:: **Image UMAP**
.. _api-umap: https://einarolafsson.github.io/spacr/api/spacr/app_umap/index.html

.. |api-embedding| replace:: **Plongement interactif**
.. _api-embedding: https://einarolafsson.github.io/spacr/api/spacr/app_umap/index.html

.. |api-sequencing| replace:: **Séquençage**
.. _api-sequencing: https://einarolafsson.github.io/spacr/api/spacr/sequencing/index.html

.. |api-barcodes| replace:: **Association des codes-barres**
.. _api-barcodes: https://einarolafsson.github.io/spacr/api/spacr/sequencing/index.html

.. |api-barcode-qc| replace:: **QC des codes-barres**
.. _api-barcode-qc: https://einarolafsson.github.io/spacr/api/spacr/sequencing_qc/index.html

.. |api-barcode-qc-sweep| replace:: **Rapport sur les puits et les collisions**
.. _api-barcode-qc-sweep: https://einarolafsson.github.io/spacr/api/spacr/sequencing_qc/index.html

.. |api-regression| replace:: **Regression**
.. _api-regression: https://einarolafsson.github.io/spacr/api/spacr/ml/index.html

.. |api-regression-models| replace:: **Estimation des effets du criblage**
.. _api-regression-models: https://einarolafsson.github.io/spacr/api/spacr/ml/index.html

.. |api-power| replace:: **Power**
.. _api-power: https://einarolafsson.github.io/spacr/api/spacr/power_model/index.html

.. |api-power-design| replace:: **Puissance et plan expérimental**
.. _api-power-design: https://einarolafsson.github.io/spacr/api/spacr/power_simulate/index.html

.. |api-graph| replace:: **Graph**
.. _api-graph: https://einarolafsson.github.io/spacr/api/spacr/qt/widgets/graph_spec/index.html

.. |api-graph-builder| replace:: **Graph Builder**
.. _api-graph-builder: https://einarolafsson.github.io/spacr/api/spacr/qt/widgets/graph_builder/index.html

.. |api-artifacts| replace:: **Artefacts**
.. _api-artifacts: https://einarolafsson.github.io/spacr/api/spacr/artifacts/index.html

.. |api-provenance| replace:: **Provenance de l’exécution**
.. _api-provenance: https://einarolafsson.github.io/spacr/api/spacr/runctx/index.html


Données
-------

Jeux de données de référence
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- `Ensemble de données complet de microscopie: BioStudies S-BIAD2135 <https://doi.org/10.6019/S-BIAD2135>`_
- `Jeu de données de test : Hugging Face toxo_mito <https://huggingface.co/datasets/einarolafsson/toxo_mito>`_
- `Données de séquence: NCBI BioProject PRJNA1261935 <https://www.ncbi.nlm.nih.gov/bioproject/?term=PRJNA1261935>`_
- `Analyse de puissance: spaCRPower <https://github.com/maomlab/spaCRPower>`_


Contributions et assistance
---------------------------

Les rapports de bogues et les demandes de fonctionnalité précises sont les bienvenus dans les `tickets GitHub <https://github.com/EinarOlafsson/spacr/issues>`_. Lorsque vous signalez un échec, indiquez la version de spaCR, le système d’exploitation, la version de Python, les paramètres du module et l’extrait de journal pertinent. ``spacr-doctor`` rassemble automatiquement la plupart de ces informations.

Licence
~~~~~~~~~

Le code source de la branche de développement actuelle est disponible sous la `licence PolyForm Noncommercial 1.0.0 <https://github.com/EinarOlafsson/spacr/blob/main/LICENSE>`_. Toute utilisation commerciale nécessite une licence distincte du titulaire des droits d’auteur. Les versions publiées jusqu’à spaCR 1.4.9.9 inclus restent disponibles sous la licence MIT qui accompagnait ces versions.

Tutoriels
~~~~~~~~~

La `bibliothèque interactive de tutoriels spaCR <https://einarolafsson.github.io/spacr/tutorials/>`_ contient, en huit langues, des parcours commentés et sous-titrés sur l’installation et chaque flux de travail de l’application.

Citer spaCR
~~~~~~~~~~~~

Si spaCR contribue à votre recherche, citez :

Olafsson EB, *et al.* Un criblage CRISPR groupé fondé sur l’imagerie identifie EAF1 comme modulateur de la subversion d’ESCRT par *T. gondii*.

`préimpression bioRxiv <https://www.biorxiv.org/content/10.64898/2026.07.08.737057v1>`_ · `archive logicielle <https://doi.org/10.5281/zenodo.21343317>`_
