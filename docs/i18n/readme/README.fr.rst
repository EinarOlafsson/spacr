|Docs| |Tutorials| |PyPI| |Conda| |Python| |Tests| |Qt| |Source| |Issues| |License| |DOI|

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
   :alt: Licence BSD 3-Clause
.. |DOI| image:: https://img.shields.io/badge/DOI-10.5281%2Fzenodo.21343316-blue
   :target: https://doi.org/10.5281/zenodo.21343316
   :alt: DOI Zenodo
.. |Release| image:: https://img.shields.io/github/v/release/EinarOlafsson/spacr?label=Installers
   :target: https://github.com/EinarOlafsson/spacr/releases/latest
   :alt: Derniers installateurs
.. |Conda| image:: https://anaconda.org/conda-forge/spacr/badges/version.svg
   :target: https://anaconda.org/conda-forge/spacr
   :alt: Version conda-forge

.. image:: ../../../spacr/resources/icons/logo_spacr_readme.png
   :alt: spaCR
   :width: 920

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

**Analyse spatiale des phénotypes de criblages CRISPR.**

spaCR segmente et mesure les cellules individuelles dans des images de microscopie à haut contenu, intègre les phénotypes par objet à l’abondance des guides dérivée du séquençage et estime quels gènes sont associés aux changements phénotypiques. À partir d’images de plaques et de lectures FASTQ, il produit des mesures par objet, des classificateurs entraînés, des estimations d’effet par guide et par gène, ainsi qu’une liste de résultats classée.

Pour les criblages CRISPR groupés fondés sur l’imagerie, spaCR fournit le flux de travail depuis la segmentation des images jusqu’à la hiérarchisation des résultats. Pour les études de microscopie à haut contenu sans criblage fondé sur le séquençage, les modules de segmentation, de mesure, d’annotation et de classification peuvent être utilisés indépendamment.

Les images, masques, recadrages, mesures, annotations, prédictions, codes-barres et identifiants de puits sont conservés dans un même projet SQLite, ce qui permet de relier chaque valeur d’un résultat à son objet d’origine.

Exécutez spaCR comme application de bureau ou sans interface graphique sur une station de travail, un serveur ou un cluster. Les deux modes utilisent les mêmes modules et CUDA est activé automatiquement lorsqu’un module le prend en charge.


Vue d’ensemble du flux de travail
---------------------------------

.. spacr-workflow-begin

|Workflow_mask|\ |Workflow_arrow|\ |Workflow_measure|\ |Workflow_arrow|\ |Workflow_annotate|\ |Workflow_arrow|\ |Workflow_classify_merged|\ |Workflow_arrow|\ |Workflow_map_barcodes|\ |Workflow_arrow|\ |Workflow_regression|

.. |Workflow_mask| image:: ../../../spacr/resources/icons/workflow/mask.png
   :width: 14.5%
   :alt: Ouvrir l’API de Mask
   :target: https://einarolafsson.github.io/spacr/api/spacr/core/index.html
   :align: middle
.. |Workflow_measure| image:: ../../../spacr/resources/icons/workflow/measure.png
   :width: 14.5%
   :alt: Ouvrir l’API de Measure
   :target: https://einarolafsson.github.io/spacr/api/spacr/measure/index.html
   :align: middle
.. |Workflow_annotate| image:: ../../../spacr/resources/icons/workflow/annotate.png
   :width: 14.5%
   :alt: Ouvrir l’API de Annotate
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/annotate/index.html
   :align: middle
.. |Workflow_classify_merged| image:: ../../../spacr/resources/icons/workflow/classify_merged.png
   :width: 14.5%
   :alt: Ouvrir l’API de Classify
   :target: https://einarolafsson.github.io/spacr/api/spacr/classify/index.html
   :align: middle
.. |Workflow_map_barcodes| image:: ../../../spacr/resources/icons/workflow/map_barcodes.png
   :width: 14.5%
   :alt: Ouvrir l’API de Map Barcodes
   :target: https://einarolafsson.github.io/spacr/api/spacr/sequencing/index.html
   :align: middle
.. |Workflow_regression| image:: ../../../spacr/resources/icons/workflow/regression.png
   :width: 14.5%
   :alt: Ouvrir l’API de Regression
   :target: https://einarolafsson.github.io/spacr/api/spacr/ml/index.html
   :align: middle
.. |Workflow_arrow| image:: ../../../spacr/resources/icons/workflow/arrow.png
   :width: 2.5%
   :align: middle

**Données**

|App_foreign|\ |App_run_compare|\ |App_experiment_design|\ |App_power|\ |App_dose_response|\ |App_qc_dashboard|

**Tools**

|App_make_masks|\ |App_align|\ |App_umap|\ |App_gate_editor|\ |App_graph_builder|

**Essais**

|App_analyze_plaques|\ |App_recruitment|\ |App_invasion|\ |App_replication|

.. |App_foreign| image:: ../../../spacr/resources/icons/workflow/apps/foreign.png
   :width: 16.583%
   :alt: Ouvrir l’API de Import
   :target: https://einarolafsson.github.io/spacr/api/spacr/foreign/index.html
   :align: left
.. |App_run_compare| image:: ../../../spacr/resources/icons/workflow/apps/run_compare.png
   :width: 16.583%
   :alt: Ouvrir l’API de Run Compare
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/run_compare/index.html
   :align: left
.. |App_experiment_design| image:: ../../../spacr/resources/icons/workflow/apps/experiment_design.png
   :width: 16.583%
   :alt: Ouvrir l’API de Experiment Design
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/experiment_design/index.html
   :align: left
.. |App_power| image:: ../../../spacr/resources/icons/workflow/apps/power.png
   :width: 16.583%
   :alt: Ouvrir l’API de Power / Design
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/power/index.html
   :align: left
.. |App_dose_response| image:: ../../../spacr/resources/icons/workflow/apps/dose_response.png
   :width: 16.583%
   :alt: Ouvrir l’API de Dose–Response
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/dose_response/index.html
   :align: left
.. |App_qc_dashboard| image:: ../../../spacr/resources/icons/workflow/apps/qc_dashboard.png
   :width: 16.583%
   :alt: Ouvrir l’API de QC
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/qc_dashboard/index.html
   :align: left
.. |App_make_masks| image:: ../../../spacr/resources/icons/workflow/apps/make_masks.png
   :width: 16.583%
   :alt: Ouvrir l’API de Make Masks
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/make_masks/index.html
   :align: left
.. |App_align| image:: ../../../spacr/resources/icons/workflow/apps/align.png
   :width: 16.583%
   :alt: Ouvrir l’API de Align & Stitch
   :target: https://einarolafsson.github.io/spacr/api/spacr/align/index.html
   :align: left
.. |App_umap| image:: ../../../spacr/resources/icons/workflow/apps/umap.png
   :width: 16.583%
   :alt: Ouvrir l’API de Image UMAP
   :target: https://einarolafsson.github.io/spacr/api/spacr/core/index.html
   :align: left
.. |App_gate_editor| image:: ../../../spacr/resources/icons/workflow/apps/gate_editor.png
   :width: 16.583%
   :alt: Ouvrir l’API de Gate Editor
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/gate_editor/index.html
   :align: left
.. |App_graph_builder| image:: ../../../spacr/resources/icons/workflow/apps/graph_builder.png
   :width: 16.583%
   :alt: Ouvrir l’API de Graph Builder
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/graph_builder/index.html
   :align: left
.. |App_analyze_plaques| image:: ../../../spacr/resources/icons/workflow/apps/analyze_plaques.png
   :width: 16.583%
   :alt: Ouvrir l’API de Plaque Assay
   :target: https://einarolafsson.github.io/spacr/api/spacr/submodules/index.html
   :align: left
.. |App_recruitment| image:: ../../../spacr/resources/icons/workflow/apps/recruitment.png
   :width: 16.583%
   :alt: Ouvrir l’API de Recruitment
   :target: https://einarolafsson.github.io/spacr/api/spacr/submodules/index.html
   :align: left
.. |App_invasion| image:: ../../../spacr/resources/icons/workflow/apps/invasion.png
   :width: 16.583%
   :alt: Ouvrir l’API de Invasion Assay
   :target: https://einarolafsson.github.io/spacr/api/spacr/submodules/index.html
   :align: left
.. |App_replication| image:: ../../../spacr/resources/icons/workflow/apps/replication.png
   :width: 16.583%
   :alt: Ouvrir l’API de Replication Assay
   :target: https://einarolafsson.github.io/spacr/api/spacr/submodules/index.html
   :align: left

.. spacr-workflow-end

Sélectionnez un module du flux de travail pour ouvrir sa page d’API. La grille contient toutes les autres applications, classées dans les mêmes catégories et dans le même ordre que sur l’écran d’accueil de spaCR.


Installer spaCR
---------------

Application de bureau
~~~~~~~~~~~~~~~~~~~~~

Les installateurs de bureau comprennent un environnement privé Python, donc conda et une installation existante Python ne sont pas nécessaires.

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

Installation avec conda-forge
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Le paquet conda-forge officiel installe spaCR et les dépendances de son application de bureau dans l’environnement actif :

.. code-block:: bash

   conda create -n spacr python=3.12 -y
   conda activate spacr
   conda install conda-forge::spacr
   spacr

Installation depuis PyPI
~~~~~~~~~~~~~~~~~~~~~~~~

Pour la version publiée sur PyPI, installez spaCR avec pip dans un environnement Conda. Python 3.12 offre le plus grand choix de paquets scientifiques facultatifs :

.. code-block:: bash

   conda create -n spacr python=3.12 -y
   conda activate spacr
   python -m pip install --upgrade pip
   python -m pip install "spacr[qt]"
   spacr

spaCR prend en charge Python **3.9 à 3.14**, à l’exception de Python 3.14.1, exclu par torchvision. Linux est recommandé pour les flux CUDA ; macOS et Windows sont également pris en charge.

Sur un serveur, un cluster ou un exécuteur CI, omettez Qt :

.. code-block:: bash

   python -m pip install spacr
   spacr-run --list

Les intégrations optionnelles sont installées séparément, par exemple ``spacr[zarr]``, ``spacr[omero]``,``spacr[napari]`` et ``spacr[czi,nd2,lif]``. Voir le `guide d'installation <../../source/installer_guide.rst>`_ pour les extras complets et la table de compatibilité Python-version.

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


Ce que vous pouvez faire
------------------------

Le flux de travail principal comprend six modules :

- **Mask** segmente les cellules, les noyaux, les agents pathogènes et les organites avec Cellpose.
- **Measure** enregistre dans SQLite les caractéristiques morphologiques, d’intensité, de texture, spatiales et de colocalisation, ainsi que les vignettes des objets.
- **Annotate** annote les vignettes dans une grille pilotée au clavier et prend en charge les files d’apprentissage actif.
- **Classify** entraîne des modèles fondés sur les images ou les mesures et enregistre, avec chaque checkpoint, les performances sur les données réservées.
- **Map Barcodes** associe les lectures FASTQ aux puits et aux gRNA, avec un contrôle qualité de l’abondance, des collisions et de la couverture.
- **Regression** estime les effets des guides, des gènes, des conditions et des contrôles avec des familles de modèles adaptées aux réponses continues, fractionnelles et de comptage.

Le même projet permet également de concevoir des plaques, d’estimer la puissance statistique, de corriger les effets de lot, d’examiner la qualité de la segmentation, d’explorer des graphiques et des extraits d’image liés, d’exporter AnnData, de reprendre un traitement interrompu et d’enregistrer les paramètres associés à chaque résultat.

Modules accessibles depuis leurs écrans hôtes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Vingt modules sont intégrés à des écrans hôtes associés au lieu d’être affichés sous forme de tuiles distinctes sur l’écran d’accueil. Chaque module s’ouvre depuis l’en-tête de son écran hôte et utilise le projet actif. Mask, Measure, Annotate, Classify, Map Barcodes, Regression, Image UMAP et Make Masks donnent accès à ces modules intégrés. Leur aide et leur documentation API restent disponibles, et les modules dotés de points d’entrée de pipeline peuvent toujours être exécutés sans interface graphique. Le `guide des fonctionnalités <../../source/features.rst>`_ répertorie chaque module intégré et son écran hôte.

Make Masks
~~~~~~~~~~

Make Masks apparaît sous **Data** et permet la correction manuelle des masques de segmentation. Son en-tête donne également accès aux flux de travail Cellpose. Le canevas comporte neuf outils : **Brush**, **Erase**, **Erase object**, **Wand +**, **Wand −**, **Draw**, **Divide**, **Zoom** et **Recrop**. Draw crée une étiquette remplie à partir d’un contour fermé tracé à main levée. Divide sépare un objet fusionné selon une ligne définie par l’utilisateur tout en préservant les étiquettes de tous les autres objets.

Recrop extrait un champ ne contenant qu’un objet à partir d’une image préparée qui en contient plusieurs. Une boîte englobante autour d’un objet enregistre les régions correspondantes de l’image et du masque dans un nouveau champ, programme ce champ après le champ actif et retire de la file de curation le champ initial contenant plusieurs objets. Recrop modifie le champ actif plutôt que les pixels des étiquettes.

L’exécution de Cellpose-SAM depuis Make Masks affiche deux résultats intermédiaires à côté du masque : la **carte de probabilité cellulaire** et le **champ de flux**. Le masque est défini par un seuil appliqué à la carte de probabilité, et les contrôles de cohérence du flux peuvent rejeter les objets dont les flux dérivés diffèrent du champ prédit. Examinez ces résultats pour distinguer une faible probabilité cellulaire d’un flux incohérent lors de l’évaluation d’un masque incorrect ou incomplet.

Objets et paramètres
~~~~~~~~~~~~~~~~~~~~

spaCR prend en charge les objets cellule, noyau et pathogène, un cytoplasme dérivé de leurs masques et entre zéro et vingt-six emplacements d’organites. Chaque emplacement d’organite possède son propre canal, diamètre, préréglage morphologique et mode de détection.

Le panneau des paramètres affiche les contrôles uniquement lorsqu’ils s’appliquent. Les emplacements d’organites au-delà du nombre configuré sont masqués, les objets sans canal attribué sont exclus de l’exécution et les contrôles propres à une morphologie ne sont affichés que pour la méthode sélectionnée. Les commutateurs **3D** et **Time** définissent la dimensionnalité : ``z_stack`` active les paramètres volumétriques, ``timelapse`` active les paramètres de suivi et les paramètres à quatre dimensions apparaissent lorsque les deux sont activés.

Choisissez la page suivante par ce que vous voulez faire:

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

Diagnostic des performances
---------------------------

Générez un rapport matériel et joignez-le à un ticket relatif aux performances::

    python tools/spacr_hardware_report.py

La commande affiche un rapport et enregistre une copie sous ``~/.spacr/reports`` ; la dernière ligne indique le chemin du fichier enregistré. ``--quick`` omet les mesures de performance les plus longues et ``--out PATH`` sélectionne un autre emplacement de sortie.

Le rapport n’ouvre aucun projet et ne lit aucune donnée de projet. Il enregistre les temps d’importation et des bibliothèques numériques, la mise à l’échelle de l’affichage, les préférences actives, la construction de la fenêtre principale et des écrans de modules, ainsi que les performances des animations. Le fichier de rapport est la seule sortie créée.

Le rapport identifie également l’émulation de l’architecture du processeur, par exemple une version x86_64 de Python sur Apple Silicon, et l’implémentation BLAS utilisée par NumPy. Ces deux facteurs peuvent affecter sensiblement les performances.

Contributions et assistance
---------------------------

Soumettez les rapports de bogues et les demandes de fonctionnalités bien délimitées via `GitHub Issues <https://github.com/EinarOlafsson/spacr/issues>`_. Lorsque vous signalez un échec, indiquez la version de spaCR, le système d’exploitation, la version de Python, les paramètres du module et l’extrait de journal pertinent. ``spacr-doctor`` collecte la plupart de ces informations ; joignez le rapport matériel lorsque vous signalez un problème de performances.

Licence
~~~~~~~~~

spaCR est un logiciel libre sous `BSD 3-Clause License <https://github.com/EinarOlafsson/spacr/blob/main/LICENSE>`_, la même licence que CellProfiler, napari et Cellpose. Il peut être utilisé à toutes fins, y compris commerciales. Les versions 1.5.0.0 à 1.5.0.4 étaient publiées sous la PolyForm Noncommercial License 1.0.0 et les versions jusqu’à 1.4.9.9 sous la licence MIT ; ces versions restent disponibles sous la licence qui les accompagnait.

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
