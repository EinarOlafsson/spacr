|Docs| |Tutorials| |PyPI| |Python| |Tests| |Qt| |Source| |Issues| |License| |DOI|

.. |Docs| image:: https://github.com/EinarOlafsson/spacr/actions/workflows/pages/pages-build-deployment/badge.svg
   :target: https://einarolafsson.github.io/spacr/
   :alt: Documentation
.. |Tutorials| image:: https://img.shields.io/badge/Tutorials-Interactive%20walkthrough-4A9EFF
   :target: https://einarolafsson.github.io/spacr/tutorials/
   :alt: Interactive tutorials
.. |PyPI| image:: https://img.shields.io/pypi/v/spacr
   :target: https://pypi.org/project/spacr/
   :alt: PyPI version
.. |Python| image:: https://img.shields.io/badge/Python-3.9%E2%80%933.14-3776AB?logo=python&logoColor=white
   :target: https://pypi.org/project/spacr/
   :alt: Python 3.9 through 3.14
.. |Tests| image:: https://github.com/EinarOlafsson/spacr/actions/workflows/tests.yml/badge.svg
   :target: https://github.com/EinarOlafsson/spacr/actions/workflows/tests.yml
   :alt: Test suite
.. |Qt| image:: https://img.shields.io/badge/GUI-Qt%20%28PySide6%29-41CD52
   :target: https://einarolafsson.github.io/spacr/
   :alt: Qt interface
.. |Source| image:: https://img.shields.io/badge/GitHub-Source-181717?logo=github
   :target: https://github.com/EinarOlafsson/spacr
   :alt: GitHub source
.. |Issues| image:: https://img.shields.io/github/issues/EinarOlafsson/spacr
   :target: https://github.com/EinarOlafsson/spacr/issues
   :alt: GitHub issues
.. |License| image:: https://img.shields.io/github/license/EinarOlafsson/spacr
   :target: https://github.com/EinarOlafsson/spacr/blob/main/LICENSE
   :alt: PolyForm Noncommercial license
.. |DOI| image:: https://img.shields.io/badge/DOI-10.5281%2Fzenodo.21343317-blue
   :target: https://doi.org/10.5281/zenodo.21343317
   :alt: Zenodo DOI
.. |Release| image:: https://img.shields.io/github/v/release/EinarOlafsson/spacr?label=Installers
   :target: https://github.com/EinarOlafsson/spacr/releases/latest
   :alt: Latest installers
.. |CondaRecipe| image:: https://img.shields.io/badge/conda--forge-recipe-44A833?logo=anaconda
   :target: https://github.com/EinarOlafsson/spacr/tree/main/conda-forge/recipe
   :alt: conda-forge recipe

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
   :alt: spaCR workflow and output organization
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

spaCR prend en charge Python **3.9 à 3.14** (sauf Python 3.14.1, ce qui exclut la torchvision). Python 3.12 a le plus grand choix de paquets scientifiques optionnels. Linux est recommandé pour les workflows CUDA; macOS et Windows sont également pris en charge.


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

L'installateur télécharge un Python privé 3.12 runtime, Qt, PyTorch, spaCR et les dépendances scientifiques pendant l'installation, de sorte que ni conda ni un Python existant n'est nécessaire. La compilation portable CPU est la valeur par défaut, ce qui empêche l'installation de tirer plusieurs gigaoctets de bibliothèques CUDA sans préavis. Windows offre une accélération NVIDIA en option, Linux accepte ``--torch-backend auto``, et la roue standard macOS PyTorch maintient l'accélération Apple MPS.

L'aide, le progrès et les erreurs d'installation suivent la langue du système d'exploitation dans les dix langues spaCR: anglais, suédois, allemand, espagnol, chinois simplifié, portugais, hindi, coréen, islandais et français.

Sur Linux, rendre l'exécutable d'installation téléchargé avant de l'ouvrir :

.. code-block:: bash

   chmod +x SpaCR-*-Linux-x86_64-Online.run
   ./SpaCR-*-Linux-x86_64-Online.run

À macOS, ouvrir le téléchargé ``.pkg``. Si Gatekeeper bloque l'installateur bêta actuel parce qu'il n'est pas notarié, ouvrez **Paramètres du système → Confidentialité et sécurité**, choisissez **Ouvrir de toute façon** pour spaCR, puis lancez le paquet à nouveau.

L'installateur valide spaCR, Qt, PyTorch et la cohérence de dépendance avant de remplacer une installation plus ancienne, de sorte qu'une mise à jour interrompue laisse l'environnement de travail précédent en place. Un journal de diagnostic est conservé comme ``install.log`` dans le répertoire d'installation privé spaCR.

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

La résolution des extras dépend de la version Python. Sur Python 3.13, les limites d'ultrack ``spacr[all]`` et la contrainte NumPy de TorchCAM limitent l'extra ``attribution``; le paquet de base et l'application Qt ne sont pas affectées. Sur Python 3.14, btrack est disponible via son extra. Le convertisseur CZI pylibCZIrw est optionnel et non testé; la lecture CZI basée sur czifile reste disponible.

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

**Masque** segmente les cellules, les noyaux, les pathogènes et les organites avec Cellpose, dans les images 2D et dans les données volumétriques ou chronologiques. La liste des modèles est lue à partir du Cellpose installé plutôt que codé dur, et un diamètre d'objet est estimé à partir des images avant le début de l'exécution.

**Mesure** Nouveau dans 1.5.0.0: correction de l'éclairage estime le champ plat de la plaque elle-même et la divise avant qu'une caractéristique d'intensité ne soit prise, ce qui élimine le biais de position que les cartes de chaleur de plaque montrent comme effets de bord. Une segmentation QC La bannière indique en langage clair à quoi ressemblent les masques avant que Mesure ne tourne; elle informe, elle ne bloque pas. Un polygone dessiné limite la mesure à une région d'intérêt.

**Annoter** affiche les vignettes sur une grille pilotée au clavier et écrit les étiquettes directement dans SQLite. Il ferme maintenant la boucle d'apprentissage actif : réentraîner un modèle sur les données étiquetées sans quitter le criblage, reclasser la file d'attente selon l'incertitude, examiner la courbe d'apprentissage et obtenir un critère d'arrêt lorsque de nouvelles étiquettes ne modifient plus le modèle. La couverture est indiquée par classe, par puits et par plaque, et chaque cycle est enregistré.

**Classifier** trains PyTorch CNN et transformateurs sur les vignettes annotées, et modèles classiques ou boostés sur les tables de mesure. La précision par classe est maintenant maintenue à chaque époque au lieu d'être jetée, et chaque point de contrôle obtient une carte modèle enregistrant son ensemble de données, son équilibre de classe, sa règle de fractionnement et les mesures maintenues.

**Codes barres de carte** décode la ligne, la colonne et gRNA code-barres de FASTQ lit, attribue les identités de guide aux puits, et les joint aux cellules imaged. QC les rapports se lit par puits, taux de collision et fraction non maculée, balayant autour du nombre de gRNA par puits vous dites que vous attendez plutôt qu'un seuil fixe.

**Les estimations de régression** indiquent les effets de référence, les gènes, les conditions et les contrôles à l'aide de 17 familles modèles, y compris des modèles mixtes, des modèles logistiques et probit, des modèles quantile, bêta, GLM avec variance quasi-binomiale, lasso, crête, filet élastique, charnière et fer à cheval.

Nouveautés de la version 1.5.0.0
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Avant l'apparition d'un criblage, le module Power / Design répond au nombre de cellules et de puits dont il a besoin, à prix avec erreur de séquençage et avec l'abandon qui vient de puits qui ont été représentés trop finement. Un concepteur d'expérience pose la plaque, ses commandes et ses répliques et exporte la disposition pour le pipeline. Ensuite, un tableau de bord QC recueille la segmentation, la plaque, l'accord annotateur et les vérifications de fuite dans un seul verdict, et ComBat est disponible à côté de ``center`` et ``zscore`` pour la correction par lots.

Les résultats sont explorés plutôt que exportés et réimportés. Un concepteur graphique trace une table en faisant glisser des colonnes sur x, y, couleur, taille et facette. Les portes tirées sur un histogramme ou un scatter deviennent des filtres. Un explorateur de fonction classe les caractéristiques en fonction de leur degré de séparation des classes. Les petits multiplex, les ajustements dose-réponse, les cartes de contrôle et la détection robuste des aberrations utilisent le même moteur d'axe.

Chaque exécution est maintenant identifiable. Chaque exécution porte un id, une graine et une politique ``on_error``; Masque, Mesure, Classer et l'export AnnData enregistre ce qu'ils ont écrit dans un registre d'artefacts, de sorte qu'un fichier de sortie renvoie aux paramètres qui l'ont produit. Un module s'ouvre sur ce que l'étape précédente a réellement écrit, les marques du graphique de pipeline qui sont des sorties discontinues, exécutent des comparaisons diffrent les paramètres, les nombres d'objets et les listes de succès de deux exécutions, et chaque exécution de GUI émet le script équivalent Python. Les mesures exportent vers ``.h5ad`` pour scanpy; OME-Zarr et OMERO sont disponibles par l'intermédiaire de Python API. L'exportateur de méthodes et de résultats rédige ces deux sections manuscrites d'un digest structuré de l'exécution : le modèle écrit la prose, mais chaque numéro provient du digest, et un projet contenant un numéro que le digest ne contient pas est rejeté. Lorsque quelque chose ne va pas avec l'installation, ``spacr-doctor`` signale ce qui est réellement lancé spaCR, si le GPU est utilisable, si le Cellpose correspond aux appels API spaCR, et si la base de données et les paramètres du projet sont sonores, avec une correction copiable sur chaque ligne qui n'est pas un passe.

Interface de bureau multilingue
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**spaCR → Préférences → Langue** retransforme l'application en anglais, suédois, allemand, espagnol, mandarin chinois, portugais, hindi, coréen, islandais ou français sans redémarrer. Le choix persiste et les criblages ouverts plus tard héritent de lui.

Navigation, Préférences, contrôles AI et LIVE, descriptions de modules et avis de consoles spaCR-autorisés suivent la langue sélectionnée. La sortie Worker, logs, tracebacks, chemins, valeurs de base de données, annotations, réponses AI, mesures et résultats enregistrés ne sont jamais traduits, donc la sortie scientifique reste canonique anglais. `guide de localisation <https://einarolafsson.github.io/spacr/localization.html>`_ documente le comportement, l'environnement et la `aide contextuelle <https://einarolafsson.github.io/spacr/localization.html#contextual-help>`_ qui est traduit avec elle.

Orientations de l'aménagement animé
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

94 animations courtes expliquent ce que 143 paramètres visuels font à une image. Déplacez un paramètre et cliquez sur **Animation** dans son tooltip pour lire le carré à côté du texte; cliquez-le à nouveau pour le plier. Les animations sont désactivées jusqu'à ce que vous le vouliez, et peuvent être désactivées dans Préférences. Le `galerie <https://einarolafsson.github.io/spacr/setting_animations.html>`_ les affiche tous, et les enregistrements `Configuration du registre d'animation <https://einarolafsson.github.io/spacr/api/spacr/setting_animations/index.html>`_ qui définissent chacun d'eux.

Référence des modules
~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 25 25 25

   * - Module
     - Feature
     - State
     - Description
   * - **Desktop experience**
     -
     -
     -
   * - |api-qt-app|_
     - |doc-i18n|_
     - Stable
     - Retranslates open and lazily created screens across ten bundled languages.
   * - |api-qt-app|_
     - |doc-i18n-help|_
     - Stable
     - Localizes module summaries and setting-help chrome while preserving exact API URLs.
   * - |api-qt-ai|_
     - |api-qt-ai-console|_
     - Stable
     - Localizes AI and LIVE controls without changing user or model content.
   * - |api-animations|_
     - |doc-animations|_
     - Stable
     - Plays 94 packaged animations for 143 visual settings from the setting tooltip.
   * - |api-selection|_
     - |api-linked-views|_
     - Alpha
     - Shares one object selection across the table, plate, embedding, scatter and graph views.
   * - |api-doctor|_
     - |api-doctor-checks|_
     - Alpha
     - Diagnoses the install — GPU, Cellpose API, database, settings — with a fix per failing check.
   * - **Image analysis**
     -
     -
     -
   * - |api-mask|_
     - |api-mask-2d|_
     - Stable
     - Segments cells, nuclei, pathogens and organelles in 2D images.
   * - |api-mask|_
     - |api-mask-3d|_
     - Beta
     - Segments volumetric images and 4D time series.
   * - |api-illumination|_
     - |api-flatfield|_
     - Alpha
     - Estimates the flat-field from the plate and divides it out before intensity is measured.
   * - |api-measure|_
     - |api-measure-2d|_
     - Stable
     - Measures morphology, intensity, texture and colocalization, and writes the crops.
   * - |api-segqc|_
     - |api-segqc-verdict|_
     - Alpha
     - States what the segmentation looks like before Measure runs, without blocking it.
   * - |api-timelapse|_
     - |api-tracking|_
     - Beta
     - Tracks objects with IoU, Trackpy, btrack, Trackastra or ultrack, and quantifies motility.
   * - |api-layers|_
     - |api-layer-viewer|_
     - Alpha
     - Stacks image, label, point and shape layers, with orthogonal views and a comparison grid.
   * - |api-napari|_
     - |api-napari-curation|_
     - Alpha
     - Hands a mask to napari for correction and takes it back, recording every edit.
   * - **AI and phenotyping**
     -
     -
     -
   * - |api-annotate|_
     - |api-annotation|_
     - Stable
     - Reviews crops on a keyboard-driven grid and saves annotations to SQLite.
   * - |api-active-learning|_
     - |api-al-loop|_
     - Alpha
     - Retrains inside Annotate, re-ranks by uncertainty, and says when labelling can stop.
   * - |api-classify|_
     - |api-classification|_
     - Stable
     - Trains and applies PyTorch CNN and transformer models.
   * - |api-classify|_
     - |api-model-cards|_
     - Alpha
     - Records dataset, class balance, split rule and held-out metrics beside each checkpoint.
   * - |api-confusion|_
     - |api-confusion-drill|_
     - Alpha
     - Opens the crops behind a confusion cell, confident errors listed apart from uncertain ones.
   * - |api-ml|_
     - |api-ml-models|_
     - Stable
     - Trains interpretable classical and boosted models on measurement tables.
   * - |api-classify|_
     - |api-activation|_
     - Beta
     - Explains predictions with Captum, SmoothGrad and TorchCAM.
   * - |api-umap|_
     - |api-embedding|_
     - Beta
     - Explores image embeddings interactively and propagates cluster labels.
   * - **Sequencing and screen analysis**
     -
     -
     -
   * - |api-sequencing|_
     - |api-barcodes|_
     - Stable
     - Maps row, column and gRNA barcodes from FASTQ reads and assigns guides to imaged cells.
   * - |api-barcode-qc|_
     - |api-barcode-qc-sweep|_
     - Alpha
     - Reports reads per well, collision rate and unmapped fraction against the expected gRNAs per well.
   * - |api-regression|_
     - |api-regression-models|_
     - Stable
     - Estimates guide, gene, condition and control effects with 17 model families.
   * - |api-power|_
     - |api-power-design|_
     - Alpha
     - Answers how many cells and wells a screen needs, with sequencing error and well dropout priced in.
   * - |api-graph|_
     - |api-graph-builder|_
     - Alpha
     - Builds a plot by dragging columns onto x, y, colour, size and facet.
   * - |api-artifacts|_
     - |api-provenance|_
     - Alpha
     - Records the run id, seed and settings behind mask, measure, classify and export outputs.

.. |api-qt-app| replace:: **Qt application**
.. _api-qt-app: https://einarolafsson.github.io/spacr/api/spacr/qt/app/index.html

.. |doc-i18n| replace:: **Ten-language localization**
.. _doc-i18n: https://einarolafsson.github.io/spacr/localization.html

.. |doc-i18n-help| replace:: **Localized contextual help**
.. _doc-i18n-help: https://einarolafsson.github.io/spacr/localization.html#contextual-help

.. |api-qt-ai| replace:: **Qt AI**
.. _api-qt-ai: https://einarolafsson.github.io/spacr/api/spacr/qt/ai/index.html

.. |api-qt-ai-console| replace:: **AI-assisted console**
.. _api-qt-ai-console: https://einarolafsson.github.io/spacr/api/spacr/qt/ai/index.html

.. |api-animations| replace:: **Setting animation registry**
.. _api-animations: https://einarolafsson.github.io/spacr/api/spacr/setting_animations/index.html

.. |doc-animations| replace:: **Visual setting animations**
.. _doc-animations: https://einarolafsson.github.io/spacr/setting_animations.html

.. |api-selection| replace:: **Selection**
.. _api-selection: https://einarolafsson.github.io/spacr/api/spacr/selection/index.html

.. |api-linked-views| replace:: **Linked selection**
.. _api-linked-views: https://einarolafsson.github.io/spacr/api/spacr/qt/linked_selection/index.html

.. |api-doctor| replace:: **Doctor**
.. _api-doctor: https://einarolafsson.github.io/spacr/api/spacr/doctor/index.html

.. |api-doctor-checks| replace:: **Installation diagnosis**
.. _api-doctor-checks: https://einarolafsson.github.io/spacr/api/spacr/doctor/index.html

.. |api-mask| replace:: **Mask**
.. _api-mask: https://einarolafsson.github.io/spacr/api/spacr/core/index.html

.. |api-mask-2d| replace:: **2D mask generation**
.. _api-mask-2d: https://einarolafsson.github.io/spacr/api/spacr/core/index.html#spacr.core.preprocess_generate_masks

.. |api-mask-3d| replace:: **3D and 4D mask generation**
.. _api-mask-3d: https://einarolafsson.github.io/spacr/api/spacr/core/index.html#spacr.core.preprocess_generate_masks

.. |api-illumination| replace:: **Illumination**
.. _api-illumination: https://einarolafsson.github.io/spacr/api/spacr/illumination/index.html

.. |api-flatfield| replace:: **Flat-field correction**
.. _api-flatfield: https://einarolafsson.github.io/spacr/api/spacr/illumination/index.html

.. |api-measure| replace:: **Measure**
.. _api-measure: https://einarolafsson.github.io/spacr/api/spacr/measure/index.html

.. |api-measure-2d| replace:: **Object measurements**
.. _api-measure-2d: https://einarolafsson.github.io/spacr/api/spacr/measure/index.html#spacr.measure.measure_crop

.. |api-segqc| replace:: **Segmentation QC**
.. _api-segqc: https://einarolafsson.github.io/spacr/api/spacr/seg_qc/index.html

.. |api-segqc-verdict| replace:: **Pre-run verdict**
.. _api-segqc-verdict: https://einarolafsson.github.io/spacr/api/spacr/seg_qc/index.html

.. |api-timelapse| replace:: **Timelapse**
.. _api-timelapse: https://einarolafsson.github.io/spacr/api/spacr/timelapse/index.html

.. |api-tracking| replace:: **Object tracking**
.. _api-tracking: https://einarolafsson.github.io/spacr/api/spacr/timelapse/index.html

.. |api-layers| replace:: **Layers**
.. _api-layers: https://einarolafsson.github.io/spacr/api/spacr/layers/index.html

.. |api-layer-viewer| replace:: **Layer viewer**
.. _api-layer-viewer: https://einarolafsson.github.io/spacr/api/spacr/qt/layer_viewer/index.html

.. |api-napari| replace:: **napari bridge**
.. _api-napari: https://einarolafsson.github.io/spacr/api/spacr/napari_bridge/index.html

.. |api-napari-curation| replace:: **Mask curation**
.. _api-napari-curation: https://einarolafsson.github.io/spacr/api/spacr/napari_bridge/index.html

.. |api-annotate| replace:: **Annotate**
.. _api-annotate: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/annotate/index.html

.. |api-annotation| replace:: **Manual annotation**
.. _api-annotation: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/annotate/index.html

.. |api-active-learning| replace:: **Active learning**
.. _api-active-learning: https://einarolafsson.github.io/spacr/api/spacr/active_learning/index.html

.. |api-al-loop| replace:: **Retrain and re-rank**
.. _api-al-loop: https://einarolafsson.github.io/spacr/api/spacr/active_learning/index.html

.. |api-classify| replace:: **Classify**
.. _api-classify: https://einarolafsson.github.io/spacr/api/spacr/deep_spacr/index.html

.. |api-classification| replace:: **Image classification**
.. _api-classification: https://einarolafsson.github.io/spacr/api/spacr/deep_spacr/index.html

.. |api-model-cards| replace:: **Model cards**
.. _api-model-cards: https://einarolafsson.github.io/spacr/api/spacr/deep_spacr/index.html

.. |api-activation| replace:: **Activation maps**
.. _api-activation: https://einarolafsson.github.io/spacr/api/spacr/deep_spacr/index.html

.. |api-confusion| replace:: **Confusion**
.. _api-confusion: https://einarolafsson.github.io/spacr/api/spacr/confusion/index.html

.. |api-confusion-drill| replace:: **Confusion drill-down**
.. _api-confusion-drill: https://einarolafsson.github.io/spacr/api/spacr/confusion/index.html

.. |api-ml| replace:: **Machine learning**
.. _api-ml: https://einarolafsson.github.io/spacr/api/spacr/ml/index.html

.. |api-ml-models| replace:: **Measurement classification**
.. _api-ml-models: https://einarolafsson.github.io/spacr/api/spacr/ml/index.html

.. |api-umap| replace:: **Image UMAP**
.. _api-umap: https://einarolafsson.github.io/spacr/api/spacr/app_umap/index.html

.. |api-embedding| replace:: **Interactive embedding**
.. _api-embedding: https://einarolafsson.github.io/spacr/api/spacr/app_umap/index.html

.. |api-sequencing| replace:: **Sequencing**
.. _api-sequencing: https://einarolafsson.github.io/spacr/api/spacr/sequencing/index.html

.. |api-barcodes| replace:: **Map barcodes**
.. _api-barcodes: https://einarolafsson.github.io/spacr/api/spacr/sequencing/index.html

.. |api-barcode-qc| replace:: **Barcode QC**
.. _api-barcode-qc: https://einarolafsson.github.io/spacr/api/spacr/sequencing_qc/index.html

.. |api-barcode-qc-sweep| replace:: **Well and collision report**
.. _api-barcode-qc-sweep: https://einarolafsson.github.io/spacr/api/spacr/sequencing_qc/index.html

.. |api-regression| replace:: **Regression**
.. _api-regression: https://einarolafsson.github.io/spacr/api/spacr/ml/index.html

.. |api-regression-models| replace:: **Screen effect estimation**
.. _api-regression-models: https://einarolafsson.github.io/spacr/api/spacr/ml/index.html

.. |api-power| replace:: **Power**
.. _api-power: https://einarolafsson.github.io/spacr/api/spacr/power_model/index.html

.. |api-power-design| replace:: **Power and design**
.. _api-power-design: https://einarolafsson.github.io/spacr/api/spacr/power_simulate/index.html

.. |api-graph| replace:: **Graph**
.. _api-graph: https://einarolafsson.github.io/spacr/api/spacr/qt/widgets/graph_spec/index.html

.. |api-graph-builder| replace:: **Graph Builder**
.. _api-graph-builder: https://einarolafsson.github.io/spacr/api/spacr/qt/widgets/graph_builder/index.html

.. |api-artifacts| replace:: **Artifacts**
.. _api-artifacts: https://einarolafsson.github.io/spacr/api/spacr/artifacts/index.html

.. |api-provenance| replace:: **Run provenance**
.. _api-provenance: https://einarolafsson.github.io/spacr/api/spacr/runctx/index.html


Données
-------

Jeux de données de référence
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- `Ensemble de données complet de microscopie: BioStudies S-BIAD2135 <https://doi.org/10.6019/S-BIAD2135>`_
- `Ensemble de données d'essai : Face de harnais toxo_mito <https://huggingface.co/datasets/einarolafsson/toxo_mito>`_
- `Données de séquence: NCBI BioProject PRJNA1261935 <https://www.ncbi.nlm.nih.gov/bioproject/?term=PRJNA1261935>`_
- `Analyse de puissance: spaCRPower <https://github.com/maomlab/spaCRPower>`_


Contributions et assistance
---------------------------

Les rapports de bogues et les requêtes de fonctionnalités ciblées sont les bienvenus par `Questions GitHub <https://github.com/EinarOlafsson/spacr/issues>`_. Lorsque vous signalez une défaillance, incluez la version spaCR, le système d'exploitation, la version Python, les paramètres du module et l'extrait de journal pertinent. ``spacr-doctor`` recueille la plupart de ces informations pour vous.

Licence
~~~~~~~~~

La branche de développement actuelle est disponible en source sous la rubrique `Licence non commerciale PolyForm 1.0.0 <https://github.com/EinarOlafsson/spacr/blob/main/LICENSE>`_. L'utilisation commerciale nécessite une licence distincte du titulaire du droit d'auteur. Les versions publiées par spaCR 1.4.9.9 restent disponibles sous la licence MIT qui accompagne ces versions.

Tutoriels
~~~~~~~~~

Le `bibliothèque interactive spaCR <https://einarolafsson.github.io/spacr/tutorials/>`_ contient des passages narratés et sous-titrés de l'installation et de chaque workflow d'application, en huit langues.

Citer spaCR
~~~~~~~~~~~~

Si spaCR contribue à votre recherche, citez :

Olafsson EB, *et al.* Un criblage à base d'images groupées CRISPR identifie EAF1 comme un modulateur *T. gondii* de la subversion ESCRT.

`préimpression bioRxiv <https://www.biorxiv.org/content/10.64898/2026.07.08.737057v1>`_ · `archive logicielle <https://doi.org/10.5281/zenodo.21343317>`_
