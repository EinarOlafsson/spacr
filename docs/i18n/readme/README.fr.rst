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


Installer spaCR
---------------

Application de bureau
~~~~~~~~~~~~~~~~~~~~~

Les installateurs de bureau comprennent un environnement privé Python, donc conda et une installation existante Python ne sont pas nécessaires.

.. spacr-installer-links-begin

|InstallerLinux| |InstallerMacOS| |InstallerWindows| |InstallerLegacy|

.. |InstallerWindows| image:: spacr/resources/icons/platforms/windows.png
   :width: 64
   :alt: Windows 10/11 : télécharger spaCR 1.5.0.4
   :target: https://github.com/EinarOlafsson/spacr/releases/download/v1.5.0.4/SpaCR-1.5.0.4-Windows-Online-Setup.exe
.. |InstallerMacOS| image:: spacr/resources/icons/platforms/macos.png
   :width: 64
   :alt: macOS 11+ (Intel et Apple Silicon) : télécharger spaCR 1.5.0.4
   :target: https://github.com/EinarOlafsson/spacr/releases/download/v1.5.0.4/SpaCR-1.5.0.4-macOS-Universal-Online.pkg
.. |InstallerLinux| image:: spacr/resources/icons/platforms/linux.png
   :width: 64
   :alt: Linux 64 bits : télécharger spaCR 1.5.0.4
   :target: https://github.com/EinarOlafsson/spacr/releases/download/v1.5.0.4/SpaCR-1.5.0.4-Linux-x86_64-Online.run
.. |InstallerLegacy| image:: spacr/resources/icons/platforms/legacy.png
   :width: 64
   :alt: Anciens programmes d’installation de spaCR
   :target: docs/source/installers.rst

.. spacr-installer-links-end

Les trois premières icônes téléchargent la version actuelle. L'icône spaCR ouvre l'archive complète de l'installateur. Les liens d'installation et les noms de fichiers en version sont mis à jour par le flux de travail de la version; les installateurs précédents restent dans la même archive de version.

Sur Linux, rendre l'exécutable du fichier téléchargé et l'exécuter :

.. code-block:: bash

   chmod +x SpaCR-*-Linux-x86_64-Online.run
   ./SpaCR-*-Linux-x86_64-Online.run

Sur macOS, ouvrez le ``.pkg``. La bêta actuelle n'est pas notariée; si Gatekeeper le bloque, choisissez **Paramètres du système → Confidentialité et sécurité → Ouvrez de toute façon**.

Consultez les instructions `guide d'installation <https://einarolafsson.github.io/spacr/installers.html>`_ pour mettre à jour, désinstaller, déconnecter et dépanner.

Installation avec Python
~~~~~~~~~~~~~~~~~~~~~~~~

Python 3.12 a le plus grand choix de paquets scientifiques optionnels:

.. code-block:: bash

   conda create -n spacr python=3.12 -y
   conda activate spacr
   python -m pip install --upgrade pip
   python -m pip install "spacr[qt]"
   spacr

spaCR prend en charge Python **3.9 à 3.14**, sauf Python 3.14.1, ce qui exclut torchvision. Linux est recommandé pour les workflows CUDA; macOS et Windows sont également pris en charge.

Pour un serveur, un cluster ou un coureur CI, omettre Qt:

.. code-block:: bash

   python -m pip install spacr
   spacr-run --list

Les intégrations optionnelles sont installées séparément, par exemple ``spacr[ome-zarr]``, ``spacr[omero]``,``spacr[napari]`` et ``spacr[czi,nd2,lif]``. Voir le `guide d'installation <https://einarolafsson.github.io/spacr/installers.html>`_ pour les extras complets et la table de compatibilité Python-version.

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

Définissez ``SPACR_LOG_LEVEL=DEBUG`` lors du dépannage. Les journaux rotatifs sont écrits dans ``~/.spacr/logs/spacr.log``. L'interface Tk classique reste disponible sous la forme ``spacr-legacy`` mais n'est plus développée.


Ce que vous pouvez faire
------------------------

La plupart des criblages suivent six modules:

- **Mask** segmente les cellules, les noyaux, les pathogènes et les organites avec Cellpose.
- **Measure** écrit la morphologie, l'intensité, la texture, les caractéristiques spatiales et de colocalisation, ainsi que les vignettes d'objets, à SQLite.
- **Annotate** les étiquettes se cultive dans une grille à clavier et prend en charge les files d'attente d'apprentissage actif.
- **Classify** entraîne des modèles basés sur l'image ou la mesure et enregistre les performances à chaque point de contrôle.
- **Map Barcodes** cartes FASTQ lit aux puits et gRNAs, avec abondance, collision et couverture QC.
- **Regression** guident les effets sur le gène, l'état et le contrôle avec des familles modèles adaptées aux réponses continues, fractionnelles et de dénombrement.

Le même projet peut également concevoir des plaques, estimer la puissance, corriger les effets des lots, inspecter la qualité de la segmentation, explorer les parcelles et les vignettes liées, exporter AnnData, reprendre les travaux interrompus et enregistrer les paramètres derrière chaque résultat.

Choisissez la page suivante par ce que vous voulez faire:

- `Didacticiels interactifs <https://einarolafsson.github.io/spacr/tutorials/>`_ — 73 workflows guidés depuis l'installation jusqu'à l'enquête.
- `Python API démarrage rapide <https://einarolafsson.github.io/spacr/python_api.html>`_ — lancez et validez des pipelines à partir de scripts, de cahiers ou d'un cluster.
- `Guide des caractéristiques <https://einarolafsson.github.io/spacr/features.html>`_ — capacités, maturité et intégrations optionnelles.
- `Référence curée API <https://einarolafsson.github.io/spacr/api/index.html>`_ — points d'entrée pris en charge par tâche, avec la référence complète du module un niveau plus profond.
- `Guide de localisation <https://einarolafsson.github.io/spacr/localization.html>`_ — langages d'interface, aide contextuelle et politique d'output scientifique.

Interface de bureau multilingue
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

La localisation en dix langues couvre la navigation, les préférences, les commandes AI et LIVE, les descriptions des modules et l’aide contextuelle révisée. Changez de langue sous **spaCR → Préférences → Langue** sans redémarrer. Les journaux, chemins, valeurs de base de données et mesures ne sont jamais traduits ; les résultats scientifiques restent en anglais canonique. Consultez la `politique d’aide contextuelle <https://einarolafsson.github.io/spacr/localization.html#contextual-help>`_.

Guide animé des paramètres
~~~~~~~~~~~~~~~~~~~~~~~~~~

Les paramètres accompagnés d’une explication visuelle proposent une commande **Animation** dans leur infobulle. Parcourez la `galerie des animations de paramètres <https://einarolafsson.github.io/spacr/setting_animations.html>`_ ou le `registre des animations de paramètres <https://einarolafsson.github.io/spacr/api/spacr/setting_animations/index.html>`_.

Données
-------

Jeux de données de référence
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- `Ensemble de données complet de microscopie: BioStudies S-BIAD2135 <https://doi.org/10.6019/S-BIAD2135>`_
- `Ensemble de données d'essai: Hugging Face toxo_mito <https://huggingface.co/datasets/einarolafsson/toxo_mito>`_
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

Le `bibliothèque interactive spaCR <https://einarolafsson.github.io/spacr/tutorials/>`_ contient des passages narratés et sous-titrés de l'installation et de chaque workflow d'application, en 73 leçons avec 50 voix dans huit langues.

Citer spaCR
~~~~~~~~~~~~

Si spaCR contribue à votre recherche, citez :

Olafsson EB, *et al.* Un criblage d'image groupée CRISPR identifie EAF1 comme un modulateur *T. gondii* de la subversion ESCRT.

`préimpression bioRxiv <https://www.biorxiv.org/content/10.64898/2026.07.08.737057v1>`_ · `archive logicielle <https://doi.org/10.5281/zenodo.21343317>`_
