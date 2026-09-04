|Docs| |Tutorials| |PyPI| |Conda| |Python| |Tests| |Qt| |Source| |Issues| |License| |Preprint| |DOI|

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
.. |Tests| image:: https://github.com/EinarOlafsson/spacr/actions/workflows/tests.yml/badge.svg?branch=nightly
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
.. |Preprint| image:: https://img.shields.io/badge/bioRxiv-2026.07.08.737057-BF2636
   :target: https://www.biorxiv.org/content/10.64898/2026.07.08.737057v1
   :alt: DOI Zenodo
.. |DOI| image:: https://img.shields.io/badge/DOI-10.5281%2Fzenodo.21343316-blue
   :target: https://doi.org/10.5281/zenodo.21343316
   :alt: Derniers installateurs
.. |Release| image:: https://img.shields.io/github/v/release/EinarOlafsson/spacr?label=Installers
   :target: https://github.com/EinarOlafsson/spacr/releases/latest
   :alt: Version conda-forge
.. |Conda| image:: https://anaconda.org/conda-forge/spacr/badges/version.svg
   :target: https://anaconda.org/conda-forge/spacr
   :alt: spaCR

.. image:: ../../../spacr/resources/icons/logo_spacr_readme.png
   :alt: spaCR
   :width: 920

spaCR
=====

.. spacr-language-picker-begin

Langues: `🌐 Français ▾ <README.md>`_

.. spacr-language-picker-end

**Analyse spatiale des phénotypes de criblages CRISPR.**

spaCR segmente et mesure les cellules individuelles dans des images de microscopie à haut contenu, intègre les phénotypes par objet à l’abondance des guides dérivée du séquençage et estime quels gènes sont associés aux changements phénotypiques. À partir d’images de plaques et de lectures FASTQ, il produit des mesures par objet, des classificateurs entraînés, des estimations d’effet par guide et par gène, ainsi qu’une liste de résultats classée.

Les modules de segmentation, de mesure, d'annotation et de classification fonctionnent également sans bras de séquençage.

Images, masques, vignettes, mesures, annotations, prédictions, codes-barres et identificateurs de puits vivent dans un projet SQLite.

Fonctionne comme une application de bureau ou sans interface graphique sur un poste de travail, un serveur ou un cluster.

Support matériel
~~~~~~~~~~~~~~~~

.. spacr-hardware-begin

.. list-table::
   :header-rows: 1
   :widths: 32 18 18 22

   * - Hardware
     - Cellpose 4
     - Torch
     - UMAP / clustering
   * - NVIDIA (CUDA)
     - 🟢 GPU
     - 🟢 GPU
     - 🟢 GPU
   * - AMD on Linux (ROCm)
     - 🟣 GPU
     - 🟣 GPU
     - 🔴 CPU
   * - AMD in an Intel Mac (Metal)
     - 🟣 GPU
     - 🟣 GPU
     - 🔴 CPU
   * - Apple Silicon (Metal)
     - 🟣 GPU
     - 🟣 GPU
     - 🔴 CPU
   * - Intel Arc/Xe (XPU)
     - 🟣 GPU
     - 🟣 GPU
     - 🔴 CPU
   * - No GPU
     - 🟢 CPU
     - 🟢 CPU
     - 🟢 CPU

Soutien (stable) Soutien (bêta) CPU seulement

.. spacr-hardware-end


Installer spaCR
---------------

Application de bureau
~~~~~~~~~~~~~~~~~~~~~

Les installateurs regroupent leur propre Python. Conda n'est pas nécessaire.

.. spacr-installer-links-begin

|InstallerLinux| |InstallerMacOS| |InstallerWindows| |InstallerLegacy|

.. |InstallerWindows| image:: ../../../spacr/resources/icons/platforms/windows.png
   :width: 64
   :alt: Windows 10/11 : télécharger spaCR 1.5.0.4
   :target: https://github.com/EinarOlafsson/spacr/releases/download/v1.5.0.4/SpaCR-1.5.0.4-Windows-Online-Setup.exe
.. |InstallerMacOS| image:: ../../../spacr/resources/icons/platforms/macos.png
   :width: 64
   :alt: macOS 11+ (Intel et Apple Silicon) : télécharger spaCR 1.5.0.4
   :target: https://github.com/EinarOlafsson/spacr/releases/download/v1.5.0.4/SpaCR-1.5.0.4-macOS-Universal-Online.pkg
.. |InstallerLinux| image:: ../../../spacr/resources/icons/platforms/linux.png
   :width: 64
   :alt: Linux 64 bits : télécharger spaCR 1.5.0.4
   :target: https://github.com/EinarOlafsson/spacr/releases/download/v1.5.0.4/SpaCR-1.5.0.4-Linux-x86_64-Online.run
.. |InstallerLegacy| image:: ../../../spacr/resources/icons/platforms/legacy.png
   :width: 64
   :alt: Anciens programmes d’installation de spaCR
   :target: ../../source/installers.rst

.. spacr-installer-links-end

Les trois premières icônes téléchargent la version actuelle. L'icône spaCR ouvre l'archive complète de l'installateur. Les liens d'installation et les noms de fichiers en version sont mis à jour par le flux de travail de la version; les installateurs précédents restent dans la même archive de version.

Sous Linux, rendez le fichier téléchargé exécutable, puis exécutez-le :

.. code-block:: bash

   chmod +x SpaCR-*-Linux-x86_64-Online.run
   ./SpaCR-*-Linux-x86_64-Online.run

Sur macOS, ouvrez le ``.pkg``. La bêta actuelle n'est pas notariée; si Gatekeeper le bloque, choisissez **Paramètres du système → Confidentialité et sécurité → Ouvrez de toute façon**.

Consultez les instructions `guide d'installation <../../source/installer_guide.rst>`_ pour mettre à jour, désinstaller, déconnecter et dépanner.

Installation depuis PyPI
~~~~~~~~~~~~~~~~~~~~~~~~

Pour la version publiée sur PyPI, installez spaCR avec pip dans un environnement Conda. Python 3.12 offre le plus grand choix de paquets scientifiques facultatifs :

.. code-block:: bash

   conda create -n spacr python=3.12 -y
   conda activate spacr
   python -m pip install --upgrade pip
   python -m pip install "spacr[qt]"
   spacr

spaCR prend en charge Python **3.9 à 3.14**, à l'exception de Python 3.14.1, qui torchvision exclut. Linux est recommandé pour les flux de travail les plus lourds CUDA et ROCm; macOS et Windows sont également pris en charge, et les deux utilisent leur GPUs — macOS via Metal, qui couvre Apple Silicon et les cartes AMD dans Intel Macs, et Windows par CUDA ou DirectML.

Sur un serveur, un cluster ou un exécuteur CI, omettez Qt :

.. code-block:: bash

   python -m pip install spacr
   spacr-run --list

Les intégrations optionnelles sont installées séparément, par exemple ``spacr[zarr]``, ``spacr[omero]``,``spacr[napari]`` et ``spacr[czi,nd2,lif]``. Voir le `guide d'installation <../../source/installer_guide.rst>`_ pour les extras complets et la table de compatibilité Python-version.

Installation avec conda-forge
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Le paquet conda-forge officiel installe spaCR et les dépendances de son application de bureau dans l’environnement actif :

.. code-block:: bash

   conda create -n spacr python=3.12 -y
   conda activate spacr
   conda install conda-forge::spacr
   spacr

Installer à partir de la source
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Cloner le dépôt et l'installer en mode modifiable, de sorte que votre copie de travail *est* le paquet installé et les modifications prennent effet sans réinstaller::

    git clone https://github.com/EinarOlafsson/spacr.git
    cd spacr
    conda create -n spacr python=3.12 -y
    conda activate spacr
    pip install -e .
    spacr

La branche par défaut est ``nightly``. Pour une version spécifique::

    git clone --branch v1.5.0.5 https://github.com/EinarOlafsson/spacr.git

Pour tirer des changements plus tard, de l'intérieur du clone::

    git pull
    pip install -e .

La deuxième ligne n'est nécessaire que lorsque les dépendances ou les points d'entrée ont changé; le code Python est récupéré sans celui-ci. Si une commande exécute toujours un ancien code après avoir tiré, ``spacr-doctor`` signale que ``spacr`` est en fait sur votre chemin, ce qui est la cause habituelle.

Installer à partir de la source (lumière)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

clone complet: 427 Mo. clone de base: 76 Mo.

::

    curl -fsSL https://raw.githubusercontent.com/EinarOlafsson/spacr/nightly/packaging/install_from_source.sh -o install_spacr.sh
    sh install_spacr.sh --branch nightly

Passer ``docs/``, ``tests/``, les points de contrôle Cellpose, des chiffres archivés et les catalogues de traduction étendus. Le résultat est une commande normale.

Options : ``--dir``, ``--branch`` (par défaut ``main``), ``--with-tests``,``--with-docs``, [``--with-translations`` et ``--no-install``.

``packaging/source_install_excludes.txt`` liste tous les chemins échappés.


Points d’entrée en ligne de commande
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   spacr                                      # launch the Qt application
   spacr-doctor                               # diagnose the installation
   spacr-run --list                           # list headless modules
   spacr-run --describe MODULE                # inspect a module contract
   spacr-run MODULE --settings settings.csv   # execute a module
   spacr-run validate --module MODULE \
       --settings settings.csv                # validate before running
   spacr-repro RUN_DIR                        # replay a recorded run

Définissez ``SPACR_LOG_LEVEL=DEBUG`` lors du dépannage. Les journaux avec rotation sont écrits dans ``~/.spacr/logs/spacr.log``.

``spacr-run --list`` répertorie les modules dotés de points d’entrée en ligne de commande pour une exécution sans interface graphique. Les modules d’annotation, de curation, de comparaison et d’exploration disponibles uniquement dans l’interface graphique sont omis.


Flux de travail principal
-------------------------

Le flux de travail principal comprend six modules :

- **Mask** segmente les cellules, les noyaux, les agents pathogènes et les organites avec Cellpose.
- **Measure** enregistre dans SQLite les caractéristiques morphologiques, d’intensité, de texture, spatiales et de colocalisation, ainsi que les vignettes des objets.
- **Annotate** annote les vignettes dans une grille pilotée au clavier et prend en charge les files d’apprentissage actif.
- **Classify** entraîne des modèles fondés sur les images ou les mesures et enregistre, avec chaque checkpoint, les performances sur les données réservées.
- **Map Barcodes** associe les lectures FASTQ aux puits et aux gRNA, avec un contrôle qualité de l’abondance, des collisions et de la couverture.
- **Regression** estime les effets des guides, des gènes, des conditions et des contrôles avec des familles de modèles adaptées aux réponses continues, fractionnelles et de comptage.

Modules spaCR
-------------

.. spacr-workflow-begin

Cœur
^^^^

Core sequence from microscopy images through segmentation, measurements,
annotations, classification, barcode mapping and regression.

| |Module_mask|\ |Module_measure|\ |Module_annotate|\ |Module_classify_merged|\ |Module_map_barcodes|\ |Module_regression|

Données
^^^^^^^

Import images and tables into spaCR projects and execute reproducible
multi-plate workflows.

| |Module_foreign|\ |Module_run_compare|\ |Module_experiment_design|\ |Module_power|\ |Module_dose_response|\ |Module_qc_dashboard|

Outils
^^^^^^

Point these at a project: edit masks by hand, stitch tiles, read an
embedding, draw a gate, build a plot, check quality.

| |Module_make_masks|\ |Module_align|\ |Module_umap|\ |Module_gate_editor|\ |Module_graph_builder|

Essais
^^^^^^

Quantitative readouts for biological assays.

| |Module_analyze_plaques|\ |Module_recruitment|\ |Module_invasion|\ |Module_replication|

.. |Module_mask| image:: ../../../spacr/resources/icons/workflow/mask.png
   :width: 16.0%
   :alt: Ouvrir l’API de Mask
   :target: https://einarolafsson.github.io/spacr/api/spacr/core/index.html#spacr.core.preprocess_generate_masks
   :align: middle
.. |Module_measure| image:: ../../../spacr/resources/icons/workflow/measure.png
   :width: 16.0%
   :alt: Ouvrir l’API de Measure
   :target: https://einarolafsson.github.io/spacr/api/spacr/measure/index.html
   :align: middle
.. |Module_annotate| image:: ../../../spacr/resources/icons/workflow/annotate.png
   :width: 16.0%
   :alt: Ouvrir l’API de Annotate
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/annotate/index.html
   :align: middle
.. |Module_classify_merged| image:: ../../../spacr/resources/icons/workflow/classify_merged.png
   :width: 16.0%
   :alt: Ouvrir l’API de Classify
   :target: https://einarolafsson.github.io/spacr/api/spacr/classify/index.html
   :align: middle
.. |Module_map_barcodes| image:: ../../../spacr/resources/icons/workflow/map_barcodes.png
   :width: 16.0%
   :alt: Ouvrir l’API de Map Barcodes
   :target: https://einarolafsson.github.io/spacr/api/spacr/sequencing/index.html
   :align: middle
.. |Module_regression| image:: ../../../spacr/resources/icons/workflow/regression.png
   :width: 16.0%
   :alt: Ouvrir l’API de Regression
   :target: https://einarolafsson.github.io/spacr/api/spacr/ml/index.html
   :align: middle
.. |Module_foreign| image:: ../../../spacr/resources/icons/workflow/apps/foreign.png
   :width: 16.0%
   :alt: Ouvrir l’API de Import
   :target: https://einarolafsson.github.io/spacr/api/spacr/foreign/index.html
   :align: middle
.. |Module_run_compare| image:: ../../../spacr/resources/icons/workflow/apps/run_compare.png
   :width: 16.0%
   :alt: Ouvrir l’API de Run Compare
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/run_compare/index.html
   :align: middle
.. |Module_experiment_design| image:: ../../../spacr/resources/icons/workflow/apps/experiment_design.png
   :width: 16.0%
   :alt: Ouvrir l’API de Experiment Design
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/experiment_design/index.html
   :align: middle
.. |Module_power| image:: ../../../spacr/resources/icons/workflow/apps/power.png
   :width: 16.0%
   :alt: Ouvrir l’API de Power / Design
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/power/index.html
   :align: middle
.. |Module_dose_response| image:: ../../../spacr/resources/icons/workflow/apps/dose_response.png
   :width: 16.0%
   :alt: Ouvrir l’API de Dose–Response
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/dose_response/index.html
   :align: middle
.. |Module_qc_dashboard| image:: ../../../spacr/resources/icons/workflow/apps/qc_dashboard.png
   :width: 16.0%
   :alt: Ouvrir l’API de QC
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/qc_dashboard/index.html
   :align: middle
.. |Module_make_masks| image:: ../../../spacr/resources/icons/workflow/apps/make_masks.png
   :width: 16.0%
   :alt: Ouvrir l’API de Make Masks
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/make_masks/index.html
   :align: middle
.. |Module_align| image:: ../../../spacr/resources/icons/workflow/apps/align.png
   :width: 16.0%
   :alt: Ouvrir l’API de Align & Stitch
   :target: https://einarolafsson.github.io/spacr/api/spacr/align/index.html
   :align: middle
.. |Module_umap| image:: ../../../spacr/resources/icons/workflow/apps/umap.png
   :width: 16.0%
   :alt: Ouvrir l’API de Image UMAP
   :target: https://einarolafsson.github.io/spacr/api/spacr/core/index.html#spacr.core.generate_image_umap
   :align: middle
.. |Module_gate_editor| image:: ../../../spacr/resources/icons/workflow/apps/gate_editor.png
   :width: 16.0%
   :alt: Ouvrir l’API de Gate Editor
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/gate_editor/index.html
   :align: middle
.. |Module_graph_builder| image:: ../../../spacr/resources/icons/workflow/apps/graph_builder.png
   :width: 16.0%
   :alt: Ouvrir l’API de Graph Builder
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/graph_builder/index.html
   :align: middle
.. |Module_analyze_plaques| image:: ../../../spacr/resources/icons/workflow/apps/analyze_plaques.png
   :width: 16.0%
   :alt: Ouvrir l’API de Plaque Assay
   :target: https://einarolafsson.github.io/spacr/api/spacr/submodules/index.html#spacr.submodules.analyze_plaques
   :align: middle
.. |Module_recruitment| image:: ../../../spacr/resources/icons/workflow/apps/recruitment.png
   :width: 16.0%
   :alt: Ouvrir l’API de Recruitment
   :target: https://einarolafsson.github.io/spacr/api/spacr/submodules/index.html#spacr.submodules.analyze_recruitment
   :align: middle
.. |Module_invasion| image:: ../../../spacr/resources/icons/workflow/apps/invasion.png
   :width: 16.0%
   :alt: Ouvrir l’API de Invasion Assay
   :target: https://einarolafsson.github.io/spacr/api/spacr/submodules/index.html#spacr.submodules.analyze_invasion
   :align: middle
.. |Module_replication| image:: ../../../spacr/resources/icons/workflow/apps/replication.png
   :width: 16.0%
   :alt: Ouvrir l’API de Replication Assay
   :target: https://einarolafsson.github.io/spacr/api/spacr/submodules/index.html#spacr.submodules.analyze_replication
   :align: middle

.. spacr-workflow-end

Chaque module spaCR est livré, dans l'ordre, l'écran d'accueil les liste : les six modules de pipeline d'abord, puis tout le reste. Sélectionnez une tuile pour ouvrir la page API de ce module.


Make Masks
~~~~~~~~~~

Make Masks appears under **Tools** for manual correction of segmentation masks; its masthead opens the Cellpose workflows. Nine tools: **Brush**, **Erase**, **Erase object**, **Wand +**, **Wand −**, **Draw**, **Divide**, **Zoom** and **Recrop**. Draw makes one filled label from a closed outline, Divide separates a merged object along a drawn line, Recrop turns one object in a crowded field into its own field.

Voir le `guide des caractéristiques <../../source/features.rst>`_ pour chaque outil.

Autres ressources
~~~~~~~~~~~~~~~~~

- `Didacticiels interactifs <https://einarolafsson.github.io/spacr/tutorials/>`_ — 73 workflows guidés depuis l'installation jusqu'à l'enquête.
- `Python API démarrage rapide <../../source/python_api.rst>`_ — lancez et validez des pipelines à partir de scripts, de cahiers ou d'un cluster.
- `Guide des caractéristiques <../../source/features.rst>`_ — capacités, maturité et intégrations optionnelles.
- `Référence curée API <https://einarolafsson.github.io/spacr/api/index.html>`_ — points d'entrée pris en charge par tâche, avec la référence complète du module un niveau plus profond.
- `Guide de la langue et de la traduction <../../source/localization.rst>`_ — langages d'interface, aide contextuelle et politique d'output scientifique.

Langue et traduction
~~~~~~~~~~~~~~~~~~~~~~

L’interface prend en charge dix langues dans la navigation et les préférences. Les commandes AI et LIVE, les descriptions des modules et l’aide contextuelle révisée sont également traduites. Changez de langue sous **spaCR → Préférences → Langue** sans redémarrer. Les journaux, chemins, valeurs de base de données et mesures ne sont jamais traduits ; les résultats scientifiques restent en anglais canonique. Consultez la `politique d’aide contextuelle <../../source/localization.rst#contextual-help>`_.

Guide animé des paramètres
~~~~~~~~~~~~~~~~~~~~~~~~~~

Les paramètres accompagnés d’une explication visuelle proposent une commande **Animation** dans leur infobulle. Parcourez la `galerie des animations de paramètres <https://einarolafsson.github.io/spacr/setting_animations.html>`_ ou le `registre des animations de paramètres <https://einarolafsson.github.io/spacr/api/spacr/setting_animations/index.html>`_.

Données
-------

Jeux de données de référence
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

|DataBioStudies| |DataHuggingFace| |DataNCBI| |DataSpaCRPower| |DataBioRxiv|

.. |DataBioStudies| image:: ../../../spacr/resources/icons/databanks/biostudies_button.png
   :width: 72
   :alt: Ouvrir le jeu de microscopie dans BioStudies
   :target: https://doi.org/10.6019/S-BIAD2135
.. |DataHuggingFace| image:: ../../../spacr/resources/icons/databanks/huggingface_button.png
   :width: 72
   :alt: Ouvrir le jeu de test sur Hugging Face
   :target: https://huggingface.co/datasets/einarolafsson/toxo_mito
.. |DataNCBI| image:: ../../../spacr/resources/icons/databanks/ncbi_button.png
   :width: 72
   :alt: Ouvrir le jeu de séquençage dans NCBI
   :target: https://www.ncbi.nlm.nih.gov/bioproject/?term=PRJNA1261935
.. |DataSpaCRPower| image:: ../../../spacr/resources/icons/databanks/spacrpower_button.png
   :width: 72
   :alt: Ouvrir spaCRPower
   :target: https://github.com/maomlab/spaCRPower
.. |DataBioRxiv| image:: ../../../spacr/resources/icons/databanks/biorxiv_button.png
   :width: 72
   :alt: Ouvrir la prépublication bioRxiv
   :target: https://www.biorxiv.org/content/10.64898/2026.07.08.737057v1

Modèle zoo
~~~~~~~~~~

spaCR envoie un catalogue de modèles formés et les récupère sur demande. Ouvrez **Modèle Zoo** depuis l'écran d'accueil pour les parcourir et les installer, ou nommez une clé dans un fichier de paramètres -- ``pathogen_model: toxoplasma_pv_v1`` -- et le modèle est téléchargé et vérifié la première fois qu'il est nécessaire. Chaque entrée publiée porte un SHA-256; une entrée sans un est refusée plutôt que installée, parce qu'un point de contrôle tronqué ou substitué ne peut pas être dit du vrai.

.. spacr-model-zoo-begin

.. list-table::
   :header-rows: 1
   :widths: 24 34 42

   * - Model
     - Training data
     - Hold-out performance
   * - ``toxoplasma_pv_v1``
       (Cellpose-SAM (cpsam_v2))
     - anti-Toxoplasma-biotin and DsRed PV lumen; 115 images, 1 dataset
     - F1 0.867 against 0.713 for stock cpsam, at IoU 0.5
   * - ``toxoplasma_plaque_v1``
       (Cellpose-SAM (cpsam))
     - crystal violet plaque wells; 184 wells from 3 datasets, 95 in-house and 89 literature
     - F1 0.856 in-domain; 0.806 on literature (3-fold cross-validated, SD 0.020)
   * - ``toxoplasma_well_detector_v1``
       (YOLO11n)
     - whole-plate and multi-well crystal violet images; 562 images from 1 dataset, 190 of them with no well in them
     - mAP50 0.993, mAP50-95 0.886, precision and recall both 0.987

.. spacr-model-zoo-end

Chaque figure ci-dessus est mesurée sur des images que le modèle n'a jamais vues dans l'entraînement.

**La précision** est le nombre d'objets déclarés par un modèle qui sont réels; **recall** est celui des objets réels qu'il a trouvés. Ils échouent dans des directions opposées: une mauvaise précision invente des plaques, un mauvais souvenir les manque.

**F1** est les deux combinés et est cité parce que chacun d'eux est triviallement gamed - rapporter une plaque unique pour une précision presque parfaite, ou chaque blob sombre pour un rappel presque parfait. Ce que vous préféreriez perdre dépend de l'essai, et le comptage est généralement mieux servi par des surappels: le modèle de plaque a été accepté à la précision 0.858 avec le rappel 0.811 sur une ronde antérieure à 0.939 et 0.631.

**IoU**, intersection au-dessus de l'union, est combien un objet prédit et le vrai chevauchement, divisé par la zone qu'ils couvrent ensemble. C'est la règle que le reste sont lus contre, donc un score ne signifie rien sans son seuil: "F1 0.867 à IoU 0.5" compte une vacuole comme trouvé lorsque les deux contours sont d'accord sur la moitié de leur zone combinée.

**mAP50** et **mPA50-95** appartiennent au détecteur. Le premier demande si les puits ont été trouvés; le second le répète à travers dix seuils de 0,5 à 0,95, de sorte qu'il demande aussi à quel point chaque boîte est serrée. L'écart entre eux est le placement, et non la détection.

**Cross-validated**, with an **SD**, means the score is the mean of three runs on different splits and the SD is how far they moved apart. One split can be lucky: this model's literature figure is 0.834 on a single 19-well split and 0.806 across all three.

Les modèles sont hébergés sur le propre compte Hugging Face de leur auteur, ce qui signifie qu'on ne doit pas remettre l'accès d'écriture à quelqu'un d'autre. ``spacr.model_zoo`` ``publish_model`` exécute le téléchargement et imprime la ligne de catalogue à ajouter.


Diagnostic des performances
---------------------------

Générez un rapport matériel et joignez-le à un ticket relatif aux performances::

    python tools/spacr_hardware_report.py

Enregistre dans ``~/.spacr/reports`` et imprime le chemin. ``--quick`` saute les points de repère plus longs; ``--out PATH`` définit l'emplacement.

Ne lit pas de données de projet. Importations de temps, bibliothèques numériques, construction de fenêtres et d'animation. Reporte l'émulation processeur-architecture (un x86_64 Python construire sur Apple Silicon) et l'implémentation BLAS de NumPy.

Référence en ligne de commande
------------------------------

Chaque commande ci-dessous est installée par ``pip install spacr``. Tous acceptent ``--help``.

Lancement de l'application
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   spacr              # the desktop application
   spacr-tutorial     # the interactive tutorial library
   spacr-server       # no first-run setup screen, for unattended launches

``spacr-server`` saute le criblage de configuration modale, qui autrement bloquerait un travail sans surveillance.

``spacr-qt`` et ``spacr-nightly`` sont des alias de ``spacr``.

Quand spaCR ne démarrera pas
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   spacr-doctor       # diagnose the installation and say how to fix it
   safespacr          # the least spaCR that can still change a setting

``spacr-doctor`` affiche une ligne par vérification, avec une commande à exécuter pour chaque échec. Il indique également quel ``spacr`` est sur le chemin, qui est ce qu'une ancienne install shadows modifiable.

``safespacr`` lit chaque préférence comme son défaut et force la toile de fond, les animations, l'enregistrement de verbes et le préchargement. Utilisez-le quand une préférence sauvegardée casse le lancement. Il ne change rien de façon permanente.

Modules de exécution sans interface graphique
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Pas de Qt, pas d'affichage — pour les clusters, les serveurs et les CI.

.. code-block:: bash

   spacr-run --list                              # modules with a headless entry
   spacr-run --describe MODULE                   # what a module consumes and produces
   spacr-run validate --module MODULE \
       --settings settings.csv                   # check settings before spending the run
   spacr-run MODULE --settings settings.csv      # execute
   spacr-remote --help                           # submit and monitor SSH, Slurm or cloud jobs

``validate`` lit les mêmes paramètres que l'exécution et signale ce qui manque, contradictoire ou pointant vers rien.

``spacr-run --list`` ne montre que les modules avec un point d'entrée sans interface graphique; l'annotation, la curation et l'exploration sont interactifs et omis.

Inspecter une course après
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Chaque exécution est journalisée à ``~/.spacr/runs`` avec ses paramètres, entrées hashées, sorties, avertissements, versions et graines.

.. code-block:: bash

   spacr-repro RUN_DIR        # replay a recorded run from its journal
   spacr-workspace RUN_DIR    # what that run had open: databases, montages, views

Audit des données et installation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   spacr-db-audit DB      # SQLite health, integrity, locking, reader/writer probe
   spacr-leakage          # classifier train/test leakage audit
   spacr-plugins          # installed plugin registry and failure diagnostics

Environnement
~~~~~~~~~~~~~

.. code-block:: bash

   SPACR_LOG_LEVEL=DEBUG spacr      # verbose logging for one launch

Les journaux rotatifs sont écrits à ``~/.spacr/logs/spacr.log``. Joindre ce fichier à un rapport de bogue.


Contributions et assistance
---------------------------

Soumettez les rapports de bogues et les demandes de fonctionnalités bien délimitées via `GitHub Issues <https://github.com/EinarOlafsson/spacr/issues>`_. Lorsque vous signalez un échec, indiquez la version de spaCR, le système d’exploitation, la version de Python, les paramètres du module et l’extrait de journal pertinent. ``spacr-doctor`` collecte la plupart de ces informations ; joignez le rapport matériel lorsque vous signalez un problème de performances.

Licence
~~~~~~~~~

spaCR est libéré sous le `Licence BSD 3-Clause <https://github.com/EinarOlafsson/spacr/blob/main/LICENSE>`_.

Si spaCR a contribué à des travaux publiés, une citation est appréciée et n'est pas une condition de la licence — voir `Citing spaCR`_ ci-dessous.

Tutoriels
~~~~~~~~~

La `bibliothèque interactive de tutoriels spaCR <https://einarolafsson.github.io/spacr/tutorials/>`_ propose des parcours commentés et sous-titrés sur l’installation et chaque flux d’application : 73 leçons avec 50 voix dans huit langues.

Citer spaCR
~~~~~~~~~~~~

Si spaCR contribue à vos recherches, citez:

Olafsson EB, *et al.* Un criblage d'image groupée CRISPR identifie EAF1 comme un modulateur *T. gondii* de la subversion ESCRT.

`préimpression bioRxiv <https://www.biorxiv.org/content/10.64898/2026.07.08.737057v1>`_ · `archive logicielle <https://doi.org/10.5281/zenodo.21343316>`_

Remerciements
~~~~~~~~~~~~~~~

spaCR repose sur des logiciels scientifiques ouverts, notamment NumPy, pandas, scikit-image, scikit-learn, Cellpose, PyTorch et Qt. Consultez l’`attribution des modèles de traduction <../TRANSLATION_MODELS.md>`_ pour connaître les modèles utilisés dans la documentation multilingue et les catalogues de l’interface.
