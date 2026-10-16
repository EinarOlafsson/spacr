|Website| |Platforms| |Python| |Qt| |Test counts| |Coverage| |Issues| |Conda| |PyPI| |Docker| |PyPI Downloads| |Docs| |Tutorials| |Preprint| |DOI| |Cite| |License| |PyPI rank|

Site web: `einarolafsson.github.io/projects/spacr <https://einarolafsson.github.io/projects/spacr/>`_

.. |Website| image:: https://img.shields.io/badge/Website-spaCR-0A7BBB
   :target: https://einarolafsson.github.io/projects/spacr/
   :alt: Site web de spaCR

.. |Docs| image:: https://img.shields.io/github/actions/workflow/status/EinarOlafsson/spacr/pages%2Fpages-build-deployment?label=API&logo=readthedocs&logoColor=white
   :target: https://einarolafsson.github.io/spacr/
   :alt: Documentation de l’API

.. |Tutorials| image:: https://img.shields.io/badge/Tutorials-4A9EFF
   :target: https://einarolafsson.github.io/spacr/tutorials/
   :alt: Tutoriels interactifs

.. |PyPI| image:: https://img.shields.io/pypi/v/spacr?label=PyPI&logo=pypi&logoColor=white
   :target: https://pypi.org/project/spacr/
   :alt: Version PyPI

.. |Python| image:: https://img.shields.io/badge/Python-3.9%E2%80%933.14-3776AB?logo=python&logoColor=white
   :target: https://pypi.org/project/spacr/
   :alt: Python 3.9 à 3.14

.. |Coverage| image:: https://img.shields.io/codecov/c/github/EinarOlafsson/spacr/nightly?logo=codecov&logoColor=white
   :target: https://app.codecov.io/github/EinarOlafsson/spacr
   :alt: Couverture de tests Codecov

.. |Test counts| image:: https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2FEinarOlafsson%2Fspacr%2Fnightly%2Fdocs%2Fsource%2F_static%2Ftest-counts.json&cacheSeconds=300&logo=pytest&logoColor=white
   :target: https://github.com/EinarOlafsson/spacr/actions/workflows/tests.yml
   :alt: Suite de tests

.. |Qt| image:: https://img.shields.io/badge/GUI-Qt-41CD52?logo=qt&logoColor=white
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/index.html#module-spacr.qt
   :alt: Interface Qt

.. |Issues| image:: https://img.shields.io/github/issues/EinarOlafsson/spacr?logo=github
   :target: https://github.com/EinarOlafsson/spacr/issues
   :alt: Tickets GitHub

.. |License| image:: https://img.shields.io/badge/License-BSD--3-blue?logo=opensourceinitiative&logoColor=white
   :target: https://github.com/EinarOlafsson/spacr/blob/main/LICENSE
   :alt: Licence BSD 3-Clause

.. |Preprint| image:: https://img.shields.io/badge/bioRxiv-BF2636
   :target: https://www.biorxiv.org/content/10.64898/2026.07.08.737057v1
   :alt: Prépublication bioRxiv

.. |DOI| image:: https://img.shields.io/badge/DOI-blue?logo=zenodo&logoColor=white
   :target: https://doi.org/10.5281/zenodo.21343316
   :alt: DOI Zenodo

.. |Conda| image:: https://img.shields.io/conda/vn/conda-forge/spacr?label=Conda&logo=anaconda&logoColor=white
   :target: https://anaconda.org/conda-forge/spacr
   :alt: Version conda-forge

.. |Release date| image:: https://anaconda.org/conda-forge/spacr/badges/latest_release_date.svg
   :target: https://anaconda.org/conda-forge/spacr
   :alt: Date de la dernière version conda-forge

.. |Docker| image:: https://img.shields.io/badge/Docker-GHCR-2496ED?logo=docker&logoColor=white
   :target: https://github.com/EinarOlafsson/spacr/pkgs/container/spacr
   :alt: Images Docker sur GHCR

.. |PyPI Downloads| image:: https://img.shields.io/pepy/dt/spacr?label=downloads&logo=pypi&logoColor=white
   :target: https://pepy.tech/projects/spacr
   :alt: Téléchargements PyPI

.. |Platforms| image:: https://img.shields.io/badge/Linux%20OSX%20Win-lightgrey?logo=linux&logoColor=white
   :target: https://github.com/EinarOlafsson/spacr/blob/nightly/docs/source/installers.rst
   :alt: Linux, macOS et Windows

.. |Cite| image:: https://img.shields.io/badge/Cite-8A2BE2
   :target: https://github.com/EinarOlafsson/spacr/blob/main/CITATION.cff
   :alt: Citer spaCR

.. |PyPI rank| image:: https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fsql-clickhouse.clickhouse.com%2F%3Fuser%3Ddemo%26param_package_name%3Dspacr%26param_days%3D30%26query%3DWITH%2B%2528%2BSELECT%2Bsum%2528count%2529%2BFROM%2Bpypi.pypi_downloads_per_day%2BWHERE%2Bproject%2B%253D%2B%257Bpackage_name%253AString%257D%2BAND%2Bdate%2B%253E%253D%2BtoDate%2528now%2528%2527UTC%2527%2529%2529%2B-%2B%257Bdays%253AUInt16%257D%2BAND%2Bdate%2B%253C%2BtoDate%2528now%2528%2527UTC%2527%2529%2529%2B%2529%2BAS%2Bdownloads%2BSELECT%2Bdownloads%2BAS%2Bpackage_downloads%252C%2BcountIf%2528n%2B%253E%253D%2Bdownloads%2529%2BAS%2Brank%252C%2Bcount%2528%2529%2BAS%2Btotal_packages%252C%2B100.0%2B%252A%2Brank%2B%252F%2BnullIf%2528total_packages%252C%2B0%2529%2BAS%2Bpercentile%252C%2Bif%2528%2Btotal_packages%2B%253D%2B0%2BOR%2Bdownloads%2B%253D%2B0%252C%2B%2527no%2Bdata%2527%252C%2Bconcat%2528%2B%2527top%2B%2527%252C%2BtoString%2528ceil%25281000.0%2B%252A%2Brank%2B%252F%2BnullIf%2528total_packages%252C%2B0%2529%2529%2B%252F%2B10%2529%252C%2B%2527%2525%2527%2B%2529%2B%2529%2BAS%2Bmessage%2BFROM%2B%2528%2BSELECT%2Bproject%252C%2Bsum%2528count%2529%2BAS%2Bn%2BFROM%2Bpypi.pypi_downloads_per_day%2BWHERE%2Bdate%2B%253E%253D%2BtoDate%2528now%2528%2527UTC%2527%2529%2529%2B-%2B%257Bdays%253AUInt16%257D%2BAND%2Bdate%2B%253C%2BtoDate%2528now%2528%2527UTC%2527%2529%2529%2BGROUP%2BBY%2Bproject%2B%2529%2BFORMAT%2BJSON&query=%24.data%5B0%5D.message&label=Rank&color=brightgreen&logo=pypi&logoColor=white&cacheSeconds=86400
   :target: https://clickpy.clickhouse.com/dashboard/spacr
   :alt: Classement des téléchargements PyPI de spaCR sur les 30 derniers jours complets

.. raw:: html

   <div><a href="https://einarolafsson.github.io/spacr/_static/deck/"><picture><source media="(min-width: 1280px)" srcset="../../readme_assets/cover.desktop.svg"><img src="../../readme_assets/cover.svg" alt="spaCR"></picture></a></div>

.. raw:: html

   <div><a href="../../source/_static/deck/pages/57.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/fr/intro-1.desktop.svg"><img src="../../readme_images/fr/intro-1.svg" alt="← Précédent"></picture></a><a href="../../source/_static/deck/pages/02.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/fr/intro-2.desktop.svg"><img src="../../readme_images/fr/intro-2.svg" alt="Suivant →"></picture></a></div>

.. spacr-language-picker-begin

.. raw:: html

   <div><a href="README.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/fr/generated-language-picker.desktop.svg"><img src="../../readme_images/fr/generated-language-picker.svg" alt="Langues: Français"></picture></a></div>

.. spacr-language-picker-end

.. raw:: html

   <div><a href="../../readme_text/fr/spacr.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/fr/spacr.desktop.svg"><img src="../../readme_images/fr/spacr.svg" alt="spaCR. Analyse spatiale des phénotypes de criblages CRISPR. spaCR segmente et mesure les cellules individuelles dans des images de microscopie, intègre les phénotypes par objet à l’abondance des guides dérivée du séquençage et estime quels gènes sont associés aux changements phénotypiques. À partir d’images de plaques et de lectures FASTQ, il produit des mesures par objet, des classificateurs entraînés, des estimations d’effet par guide et par gène, ainsi qu’une liste de résultats classée. Les modules de segmentation, de mesure, d&#x27;annotation et de classification fonctionnent également sans bras de séquençage. Make Masks corrige les masques de segmentation et annote des rectangles indépendants avec des étiquettes de classe grâce à l’outil Box pour l’export YOLO. Les rectangles conservent leurs propres étiquettes et leur historique sans modifier les images ni les masques sources. Chaque outil est décrit dans le guide des fonctionnalités. Les images, masques et vignettes sont des fichiers sur disque. Les mesures, annotations et prédictions du modèle sont des tables de la base de données SQLite measurements.db du projet. Fonctionne comme une application de bureau ou sans interface graphique sur un poste de travail, un serveur ou un cluster."></picture></a></div>

.. raw:: html

   <div><a href="../../readme_text/fr/try-spacr.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/fr/try-spacr.desktop.svg"><img src="../../readme_images/fr/try-spacr.svg" alt="Essayer spaCR."></picture></a></div>

.. code-block:: bash

   conda create -n spacr python=3.12 -y
   conda activate spacr
   python -m pip install spacr
   spacr

.. raw:: html

   <div><a href="../../readme_text/fr/try-spacr-2.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/fr/try-spacr-2.desktop.svg"><img src="../../readme_images/fr/try-spacr-2.svg" alt="Utilisez Charger les données de test… dans Import, Make Masks, Annotate ou un écran d’essai pour télécharger des données d’exemple. Dans un terminal, utilisez spacr-download."></picture></a></div>

.. raw:: html

   <div><a href="../../readme_text/fr/hardware-support.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/fr/hardware-support.desktop.svg"><img src="../../readme_images/fr/hardware-support.svg" alt="Support matériel."></picture></a></div>

.. spacr-hardware-begin

.. raw:: html

   <div><a href="../../readme_text/fr/generated-hardware-1.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/fr/generated-hardware-1.desktop.svg"><img src="../../readme_images/fr/generated-hardware-1.svg" alt="NVIDIA (CUDA): Cellpose 4 GPU, Torch GPU, UMAP / clustering GPU. AMD on Linux (ROCm): Cellpose 4 GPU, Torch GPU, UMAP / clustering CPU. AMD in an Intel Mac (Metal): Cellpose 4 GPU, Torch GPU, UMAP / clustering CPU. Apple Silicon (Metal): Cellpose 4 GPU, Torch GPU, UMAP / clustering CPU. Intel Arc/Xe (XPU): Cellpose 4 GPU, Torch GPU, UMAP / clustering CPU. No GPU: Cellpose 4 CPU, Torch CPU, UMAP / clustering CPU. pris en charge (stable) implémenté (bêta) uniquement sur CPU."></picture></a></div>

.. raw:: html

   <div><a href="../../readme_text/fr/generated-hardware-2.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/fr/generated-hardware-2.desktop.svg"><img src="../../readme_images/fr/generated-hardware-2.svg" alt="Linux (x86-64, ARM64): True 3D, essayés dans l’ordre Vulkan (wgpu), OpenGL, CPU. Windows (x64, ARM64): True 3D, essayés dans l’ordre Vulkan / Direct3D 12 (wgpu), OpenGL, CPU. macOS (Intel, Apple silicon): True 3D, essayés dans l’ordre Metal (wgpu), OpenGL, CPU. Sans wgpu: True 3D, essayés dans l’ordre OpenGL, CPU (Python 3.9, autres processeurs, Linux avec glibc antérieure à 2.28 ou installé sans wgpu)."></picture></a></div>

.. spacr-hardware-end

.. raw:: html

   <div><a href="../../readme_text/fr/hardware-support-2.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/fr/hardware-support-2.desktop.svg"><img src="../../readme_images/fr/hardware-support-2.svg" alt="CPU execution is supported and can be slower. In one measurement on an Intel iMac (Core i5-10500, AMD Radeon Pro 5300 4 GB), Cellpose-SAM took 444.5 s on the CPU and 3.2 s on Metal for one 256×256 image, using torch 2.2.2, Python 3.12.11 and macOS 26.5.2. This is a historical example, not a speed guarantee. GPU acceleration requires a usable backend and compatible libraries. See the recorded hardware measurement."></picture></a></div>

.. raw:: html

   <div><a href="../../readme_text/fr/install-spacr.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/fr/install-spacr.desktop.svg"><img src="../../readme_images/fr/install-spacr.svg" alt="Installer spaCR. Choisissez une méthode d’installation : Application de bureau : un installateur pour Linux, macOS ou Windows avec son propre Python. pip : python -m pip install spacr dans un environnement Conda. conda-forge : conda install conda-forge::spacr dans un environnement Conda. Docker : les pipelines en ligne de commande dans un conteneur Linux, avec l’image CPU ou l’image pour GPU NVIDIA :"></picture></a></div>

.. code-block:: bash

   docker run --rm ghcr.io/einarolafsson/spacr:1.5.1.4 spacr-run --list
   docker run --rm --gpus all ghcr.io/einarolafsson/spacr:1.5.1.4-cuda12.4 spacr-run --list

.. raw:: html

   <div><a href="../../readme_text/fr/install-spacr-2.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/fr/install-spacr-2.desktop.svg"><img src="../../readme_images/fr/install-spacr-2.svg" alt="Le guide Docker décrit la configuration du GPU, le montage des données et les commandes des pipelines."></picture></a></div>

.. raw:: html

   <div><a href="../../readme_text/fr/desktop-application.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/fr/desktop-application.desktop.svg"><img src="../../readme_images/fr/desktop-application.svg" alt="Application de bureau. Les installateurs regroupent leur propre Python. Conda n&#x27;est pas nécessaire."></picture></a></div>

.. spacr-installer-links-begin

| |InstallerLinux|\ |InstallerMacOS|\ |InstallerWindows|

.. |InstallerWindows| raw:: html

   <a href="https://github.com/EinarOlafsson/spacr/releases/download/v1.5.1.4/spaCR-1.5.1.4-Windows-Online-Setup.exe"><img src="../../readme_assets/cells/3/platforms/windows.png" width="33.00%" align="middle" alt="Windows 10/11 : télécharger spaCR 1.5.1.4"></a>
.. |InstallerMacOS| raw:: html

   <a href="https://github.com/EinarOlafsson/spacr/releases/download/v1.5.1.4/spaCR-1.5.1.4-macOS-Universal-Online.pkg"><img src="../../readme_assets/cells/3/platforms/macos.png" width="33.00%" align="middle" alt="macOS 11+ (Intel et Apple Silicon) : télécharger spaCR 1.5.1.4"></a>
.. |InstallerLinux| raw:: html

   <a href="https://github.com/EinarOlafsson/spacr/releases/download/v1.5.1.4/spaCR-1.5.1.4-Linux-x86_64-Online.run"><img src="../../readme_assets/cells/3/platforms/linux.png" width="33.00%" align="middle" alt="Linux 64 bits : télécharger spaCR 1.5.1.4"></a>

.. spacr-installer-links-end

.. raw:: html

   <div><a href="../../readme_text/fr/desktop-application-2.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/fr/desktop-application-2.desktop.svg"><img src="../../readme_images/fr/desktop-application-2.svg" alt="Sous Linux, rendez le fichier téléchargé exécutable, puis exécutez-le :"></picture></a></div>

.. code-block:: bash

   chmod +x SpaCR-*-Linux-x86_64-Online.run
   ./SpaCR-*-Linux-x86_64-Online.run

.. raw:: html

   <div><a href="../../readme_text/fr/desktop-application-3.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/fr/desktop-application-3.desktop.svg"><img src="../../readme_images/fr/desktop-application-3.svg" alt="Sur macOS, ouvrez le .pkg. La bêta actuelle n&#x27;est pas notariée; si Gatekeeper le bloque, choisissez Paramètres du système → Confidentialité et sécurité → Ouvrez de toute façon. Consultez le guide d’installation pour les instructions de mise à jour, de désinstallation, d’utilisation hors ligne et de dépannage, et la configuration requise pour les recommandations concernant les stations de travail et les serveurs ainsi que les tableaux de compatibilité GPU."></picture></a></div>

.. raw:: html

   <div><a href="../../source/installer_guide.rst"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/fr/desktop-application-3-link-1.desktop.svg"><img src="../../readme_images/fr/desktop-application-3-link-1.svg" alt="guide d’installation"></picture></a><a href="../../source/system_requirements.rst"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/fr/desktop-application-3-link-2.desktop.svg"><img src="../../readme_images/fr/desktop-application-3-link-2.svg" alt="configuration requise"></picture></a></div>

.. raw:: html

   <div><a href="../../readme_text/fr/pypi-installation.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/fr/pypi-installation.desktop.svg"><img src="../../readme_images/fr/pypi-installation.svg" alt="Installation depuis PyPI. Pour la version publiée sur PyPI, installez spaCR avec pip dans un environnement Conda. Python 3.12 offre le plus grand choix de paquets scientifiques facultatifs :"></picture></a></div>

.. code-block:: bash

   conda create -n spacr python=3.12 -y
   conda activate spacr
   python -m pip install --upgrade pip
   python -m pip install spacr
   spacr

.. raw:: html

   <div><a href="../../readme_text/fr/pypi-installation-2.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/fr/pypi-installation-2.desktop.svg"><img src="../../readme_images/fr/pypi-installation-2.svg" alt="spaCR prend en charge Python 3.9 à 3.14, à l&#x27;exception de Python 3.14.1, que torchvision exclut. Linux est recommandé pour les flux de travail CUDA et ROCm les plus lourds; macOS et Windows sont également pris en charge, et tous deux utilisent leurs GPUs — macOS via Metal, qui couvre Apple Silicon et les cartes AMD des Mac Intel, et Windows via CUDA ou DirectML. L’installation standard inclut l’interface de bureau Qt. Sur un serveur, un cluster ou un exécuteur CI, lancez les pipelines en ligne de commande sans l’ouvrir :"></picture></a></div>

.. code-block:: bash

   python -m pip install spacr
   spacr-run --list

.. raw:: html

   <div><a href="../../readme_text/fr/pypi-installation-3.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/fr/pypi-installation-3.desktop.svg"><img src="../../readme_images/fr/pypi-installation-3.svg" alt="Les intégrations optionnelles sont installées séparément, par exemple spacr[zarr], spacr[omero],spacr[napari] et spacr[czi,nd2,lif]. Voir le guide d&#x27;installation pour les extras complets et la table de compatibilité Python-version."></picture></a></div>

.. raw:: html

   <div><a href="../../readme_text/fr/conda-forge-installation.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/fr/conda-forge-installation.desktop.svg"><img src="../../readme_images/fr/conda-forge-installation.svg" alt="Installation avec conda-forge. Le paquet conda-forge officiel installe spaCR et les dépendances de son application de bureau dans l’environnement actif :"></picture></a></div>

.. code-block:: bash

   conda create -n spacr python=3.12 -y
   conda activate spacr
   conda install conda-forge::spacr
   spacr

.. raw:: html

   <div><a href="../../readme_text/fr/install-from-source.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/fr/install-from-source.desktop.svg"><img src="../../readme_images/fr/install-from-source.svg" alt="Installation depuis les sources. Clonez le dépôt et installez-le en mode éditable : votre copie de travail est alors le paquet installé, et vos modifications prennent effet sans réinstallation:"></picture></a></div>

.. code-block:: bash

   git clone https://github.com/EinarOlafsson/spacr.git
   cd spacr
   conda create -n spacr python=3.12 -y
   conda activate spacr
   pip install -e .
   spacr

.. raw:: html

   <div><a href="../../readme_text/fr/install-from-source-2.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/fr/install-from-source-2.desktop.svg"><img src="../../readme_images/fr/install-from-source-2.svg" alt="Cette commande clone main, la branche par défaut, qui contient la dernière version publiée. Le développement se fait sur nightly ; ajoutez --branch nightly pour cloner cette branche à la place. Pour une version précise:"></picture></a></div>

.. code-block:: bash

   git clone --branch v1.5.0.5 https://github.com/EinarOlafsson/spacr.git

.. raw:: html

   <div><a href="../../readme_text/fr/install-from-source-3.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/fr/install-from-source-3.desktop.svg"><img src="../../readme_images/fr/install-from-source-3.svg" alt="Pour récupérer les modifications ultérieures, exécutez depuis le clone:"></picture></a></div>

.. code-block:: bash

   git pull
   pip install -e .

.. raw:: html

   <div><a href="../../readme_text/fr/install-from-source-4.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/fr/install-from-source-4.desktop.svg"><img src="../../readme_images/fr/install-from-source-4.svg" alt="Réinstallez lorsque les dépendances ou les points d’entrée changent. Les modifications du code Python prennent effet directement ; spacr-doctor identifie l’installation active."></picture></a></div>

.. raw:: html

   <div><a href="../../readme_text/fr/install-from-source-light.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/fr/install-from-source-light.desktop.svg"><img src="../../readme_images/fr/install-from-source-light.svg" alt="Installation depuis les sources (allégée). Les contributeurs ont besoin de l’historique des versions ; pour simplement exécuter spaCR, choisissez une option ci-dessous. Mesures de nightly à 05302fd5c le 2026-10-07, effectuées avec packaging/measure_clone_forms.sh:"></picture></a></div>

.. code-block:: bash

   # One commit instead of every version: 2048 MB downloaded, 140 s.
   # No history, so no git log, no git blame and no git bisect.
   # git pull still works, but stays shallow until git fetch --unshallow.
   git clone --depth 1 --branch nightly https://github.com/EinarOlafsson/spacr.git
   cd spacr && pip install -e .

   # Runtime files: 104 MB on disk, 6 s (Git: 38 MB; files: 66 MB).
   # No history, docs, tests, tools, features or example data.
   # --with-docs, --with-tests and --with-translations put those back;
   # --dir, --branch, --no-install and --help do the obvious things.
   # packaging/source_install_excludes.txt lists every skipped path.
   curl -fsSL https://raw.githubusercontent.com/EinarOlafsson/spacr/nightly/packaging/install_from_source.sh -o install_spacr.sh
   sh install_spacr.sh --branch nightly

.. raw:: html

   <div><a href="../../readme_text/fr/install-from-source-light-2.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/fr/install-from-source-light-2.desktop.svg"><img src="../../readme_images/fr/install-from-source-light-2.svg" alt="Le clone complet de nightly a téléchargé 9.25 GiB. Ajouter --filter=blob:none au clone superficiel ne réduit pas la copie de travail : son magasin d’objets Git pèse encore 2032 MB. Git ne signale pas les récupérations différées d’objets, aucun total de téléchargement n’a donc été mesuré. Les fichiers suivis de nightly occupent 4657 MB dans la copie de travail (mesurés le 2026-10-15), hors historique Git. La taille et la durée du téléchargement varient selon la branche."></picture></a></div>

.. raw:: html

   <div><a href="../../readme_text/fr/command-line-entry-points.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/fr/command-line-entry-points.desktop.svg"><img src="../../readme_images/fr/command-line-entry-points.svg" alt="Points d’entrée en ligne de commande."></picture></a></div>

.. code-block:: bash

   spacr                                      # launch the Qt application
   spacr-doctor                               # diagnose the installation
   spacr-run --list                           # list headless modules
   spacr-run --describe MODULE                # inspect a module contract
   spacr-run MODULE --settings settings.csv   # execute a module
   spacr-run validate --module MODULE \
       --settings settings.csv                # validate before running
   spacr-repro RUN_DIR                        # replay a recorded run
   spacr-download --list                      # what example data exists
   spacr-download measure annotate            # fetch example sets by name
   spacr-make-masks --folder DIR              # curate masks as a resumable queue
   spacr-make-masks --folder DIR --order easy --limit 50

.. raw:: html

   <div><a href="../../readme_text/fr/command-line-entry-points-2.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/fr/command-line-entry-points-2.desktop.svg"><img src="../../readme_images/fr/command-line-entry-points-2.svg" alt="spacr-run --list répertorie les modules dotés de points d’entrée en ligne de commande pour une exécution sans interface graphique. Les modules d’annotation, de curation, de comparaison et d’exploration disponibles uniquement dans l’interface graphique sont omis."></picture></a></div>

.. raw:: html

   <div><a href="../../readme_text/fr/core-workflow.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/fr/core-workflow.desktop.svg"><img src="../../readme_images/fr/core-workflow.svg" alt="Flux de travail principal. Le flux de travail principal comprend six modules : Mask segmente les cellules, les noyaux, les agents pathogènes et les organites avec Cellpose. Measure enregistre dans SQLite les caractéristiques morphologiques, d’intensité, de texture, spatiales et de colocalisation, ainsi que les vignettes des objets. Annotate annote les vignettes dans une grille pilotée au clavier et prend en charge les files d’apprentissage actif. Classify entraîne des modèles fondés sur les images ou les mesures et enregistre, avec chaque checkpoint, les performances sur les données réservées. Map Barcodes associe les lectures FASTQ aux puits et aux gRNA, avec un contrôle qualité de l’abondance, des collisions et de la couverture. Regression estime les effets des guides, des gènes, des conditions et des contrôles avec des familles de modèles adaptées aux réponses continues, fractionnelles et de comptage."></picture></a></div>

.. raw:: html

   <div><a href="../../readme_text/fr/spacr-modules.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/fr/spacr-modules.desktop.svg"><img src="../../readme_images/fr/spacr-modules.svg" alt="Modules spaCR."></picture></a></div>

.. spacr-workflow-begin

.. raw:: html

   <div><a href="../../readme_text/fr/generated-workflow-1.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/fr/generated-workflow-1.desktop.svg"><img src="../../readme_images/fr/generated-workflow-1.svg" alt="Cœur. Core sequence from microscopy images through segmentation, measurements, annotations, classification, barcode mapping and regression."></picture></a></div>

| |Module_mask|\ |Module_measure|\ |Module_annotate|\ |Module_classify_merged|\ |Module_map_barcodes|\ |Module_regression|

.. raw:: html

   <div><a href="../../readme_text/fr/generated-workflow-2.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/fr/generated-workflow-2.desktop.svg"><img src="../../readme_images/fr/generated-workflow-2.svg" alt="Données. Import images and tables into spaCR projects and execute reproducible multi-plate workflows."></picture></a></div>

| |Module_foreign|\ |Module_embeddings|\ |Module_run_compare|\ |Module_experiment_design|\ |Module_power|\ |Module_dose_response|\ |Module_qc_dashboard|

.. raw:: html

   <div><a href="../../readme_text/fr/generated-workflow-3.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/fr/generated-workflow-3.desktop.svg"><img src="../../readme_images/fr/generated-workflow-3.svg" alt="Outils. Point these at a project: edit masks by hand, stitch tiles, read an embedding, draw a gate, build a plot, check quality."></picture></a></div>

| |Module_make_masks|\ |Module_align|\ |Module_umap|\ |Module_gate_editor|\ |Module_graph_builder|

.. raw:: html

   <div><a href="../../readme_text/fr/generated-workflow-4.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/fr/generated-workflow-4.desktop.svg"><img src="../../readme_images/fr/generated-workflow-4.svg" alt="Organisme. Analyse d’images spécifique à l’organisme et résultats quantitatifs des essais."></picture></a></div>

| |Module_toxoplasma|

.. |Module_mask| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/core/index.html#spacr.core.preprocess_generate_masks"><img src="../../readme_assets/cells/6/workflow/mask.png" width="16.50%" align="middle" alt="Ouvrir l’API de Mask"></a>

.. |Module_measure| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/measure/index.html"><img src="../../readme_assets/cells/6/workflow/measure.png" width="16.50%" align="middle" alt="Ouvrir l’API de Measure"></a>

.. |Module_annotate| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/qt/screens/annotate/index.html"><img src="../../readme_assets/cells/6/workflow/annotate.png" width="16.50%" align="middle" alt="Ouvrir l’API de Annotate"></a>

.. |Module_classify_merged| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/classify/index.html"><img src="../../readme_assets/cells/6/workflow/classify_merged.png" width="16.50%" align="middle" alt="Ouvrir l’API de Classify"></a>

.. |Module_map_barcodes| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/sequencing/index.html"><img src="../../readme_assets/cells/6/workflow/map_barcodes.png" width="16.50%" align="middle" alt="Ouvrir l’API de Map Barcodes"></a>

.. |Module_regression| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/ml/index.html"><img src="../../readme_assets/cells/6/workflow/regression.png" width="16.50%" align="middle" alt="Ouvrir l’API de Regression"></a>

.. |Module_foreign| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/foreign/index.html"><img src="../../readme_assets/cells/7/workflow/apps/foreign.png" width="14.14%" align="middle" alt="Ouvrir l’API de Import"></a>

.. |Module_embeddings| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/qt/screens/embeddings/index.html"><img src="../../readme_assets/cells/7/workflow/apps/embeddings.png" width="14.14%" align="middle" alt="Ouvrir l’API de Embeddings"></a>

.. |Module_run_compare| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/qt/screens/run_compare/index.html"><img src="../../readme_assets/cells/7/workflow/apps/run_compare.png" width="14.14%" align="middle" alt="Ouvrir l’API de Run Compare"></a>

.. |Module_experiment_design| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/qt/screens/experiment_design/index.html"><img src="../../readme_assets/cells/7/workflow/apps/experiment_design.png" width="14.14%" align="middle" alt="Ouvrir l’API de Experiment Design"></a>

.. |Module_power| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/qt/screens/power/index.html"><img src="../../readme_assets/cells/7/workflow/apps/power.png" width="14.14%" align="middle" alt="Ouvrir l’API de Power / Design"></a>

.. |Module_dose_response| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/qt/screens/dose_response/index.html"><img src="../../readme_assets/cells/7/workflow/apps/dose_response.png" width="14.14%" align="middle" alt="Ouvrir l’API de Dose–Response"></a>

.. |Module_qc_dashboard| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/qt/screens/qc_dashboard/index.html"><img src="../../readme_assets/cells/7/workflow/apps/qc_dashboard.png" width="14.14%" align="middle" alt="Ouvrir l’API de QC"></a>

.. |Module_make_masks| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/qt/screens/make_masks/index.html"><img src="../../readme_assets/cells/5/workflow/apps/make_masks.png" width="19.80%" align="middle" alt="Ouvrir l’API de Make Masks"></a>

.. |Module_align| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/align/index.html"><img src="../../readme_assets/cells/5/workflow/apps/align.png" width="19.80%" align="middle" alt="Ouvrir l’API de Align &amp; Stitch"></a>

.. |Module_umap| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/core/index.html#spacr.core.generate_image_umap"><img src="../../readme_assets/cells/5/workflow/apps/umap.png" width="19.80%" align="middle" alt="Ouvrir l’API de Image UMAP"></a>

.. |Module_gate_editor| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/qt/screens/gate_editor/index.html"><img src="../../readme_assets/cells/5/workflow/apps/gate_editor.png" width="19.80%" align="middle" alt="Ouvrir l’API de Gate Editor"></a>

.. |Module_graph_builder| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/qt/screens/graph_builder/index.html"><img src="../../readme_assets/cells/5/workflow/apps/graph_builder.png" width="19.80%" align="middle" alt="Ouvrir l’API de Graph Builder"></a>

.. |Module_toxoplasma| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/qt/screens/organism_screen/index.html#spacr-qt-screens-organism-screen-toxoplasma"><img src="../../readme_assets/cells/1/workflow/apps/toxoplasma.png" width="99.00%" align="middle" alt="Ouvrir l’API de Toxoplasma"></a>

.. spacr-workflow-end

.. raw:: html

   <div><a href="../../readme_text/fr/spacr-modules-2.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/fr/spacr-modules-2.desktop.svg"><img src="../../readme_images/fr/spacr-modules-2.svg" alt="Tous les modules ayant une tuile sur l’écran d’accueil, dans l’ordre de cet écran : d’abord les six modules du pipeline, puis les autres. Sélectionnez une tuile pour ouvrir la page API du module."></picture></a></div>

.. raw:: html

   <div><a href="../../readme_text/fr/other-resources.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/fr/other-resources.desktop.svg"><img src="../../readme_images/fr/other-resources.svg" alt="Autres ressources."></picture></a></div>

.. raw:: html

   <div><a href="../../readme_text/fr/other-resources-2-1.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/fr/other-resources-2-1.desktop.svg"><img src="../../readme_images/fr/other-resources-2-1.svg" alt="Tutoriels interactifs — parcours guidés de l’installation à l’étude des résultats positifs du criblage."></picture></a></div>

.. raw:: html

   <div><a href="../../readme_text/fr/other-resources-2-2.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/fr/other-resources-2-2.desktop.svg"><img src="../../readme_images/fr/other-resources-2-2.svg" alt="Python API démarrage rapide — lancez et validez des pipelines à partir de scripts, de cahiers ou d&#x27;un cluster."></picture></a></div>

.. raw:: html

   <div><a href="../../readme_text/fr/other-resources-2-3.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/fr/other-resources-2-3.desktop.svg"><img src="../../readme_images/fr/other-resources-2-3.svg" alt="Guide des caractéristiques — capacités, maturité et intégrations optionnelles."></picture></a></div>

.. raw:: html

   <div><a href="../../readme_text/fr/other-resources-2-4.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/fr/other-resources-2-4.desktop.svg"><img src="../../readme_images/fr/other-resources-2-4.svg" alt="Référence curée API — points d&#x27;entrée pris en charge par tâche, avec la référence complète du module un niveau plus profond."></picture></a></div>

.. raw:: html

   <div><a href="../../readme_text/fr/other-resources-2-5.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/fr/other-resources-2-5.desktop.svg"><img src="../../readme_images/fr/other-resources-2-5.svg" alt="Guide de la langue et de la traduction — langages d&#x27;interface, aide contextuelle et politique d&#x27;output scientifique."></picture></a></div>

.. raw:: html

   <div><a href="../../readme_text/fr/language-translation.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/fr/language-translation.desktop.svg"><img src="../../readme_images/fr/language-translation.svg" alt="Langue et traduction. L’interface prend en charge dix langues dans la navigation et les préférences. Les commandes AI et LIVE, les descriptions des modules et l’aide contextuelle révisée sont également traduites. Changez de langue sous spaCR → Préférences → Langue sans redémarrer. Les journaux, chemins, valeurs de base de données et mesures ne sont jamais traduits ; les résultats scientifiques restent en anglais canonique. Consultez la politique d’aide contextuelle."></picture></a></div>

.. raw:: html

   <div><a href="../../readme_text/fr/language-translation-2.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/fr/language-translation-2.desktop.svg"><img src="../../readme_images/fr/language-translation-2.svg" alt="Les neuf catalogues non anglais sont rédigés par machine et revus techniquement plutôt que lus fin à fin par un locuteur natif. Le portée de l &#x27; examen enregistre quelles langues ont eu un passage humain et chaque terme laissé en anglais par décision."></picture></a></div>

.. raw:: html

   <div><a href="../../readme_text/fr/animated-setting-guidance.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/fr/animated-setting-guidance.desktop.svg"><img src="../../readme_images/fr/animated-setting-guidance.svg" alt="Guide animé des paramètres. Les paramètres accompagnés d’une explication visuelle proposent une commande Animation dans leur infobulle. Parcourez la galerie des animations de paramètres ou le registre des animations de paramètres."></picture></a></div>

.. raw:: html

   <div><a href="https://einarolafsson.github.io/spacr/setting_animations.html"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/fr/animated-setting-guidance-link-1.desktop.svg"><img src="../../readme_images/fr/animated-setting-guidance-link-1.svg" alt="galerie des animations de paramètres"></picture></a><a href="https://einarolafsson.github.io/spacr/api/spacr/setting_animations/index.html"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/fr/animated-setting-guidance-link-2.desktop.svg"><img src="../../readme_images/fr/animated-setting-guidance-link-2.svg" alt="registre des animations de paramètres"></picture></a></div>

.. raw:: html

   <div><a href="../../readme_text/fr/data.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/fr/data.desktop.svg"><img src="../../readme_images/fr/data.svg" alt="Données. Jeux de données de référence."></picture></a></div>

| |DataBioStudies|\ |DataHuggingFace|\ |DataNCBI|\ |DataSpaCRPower|\ |DataBioRxiv|

.. |DataBioStudies| raw:: html

   <a href="https://doi.org/10.6019/S-BIAD2135"><img src="../../readme_assets/cells/5/databanks/biostudies_button.png" width="19.80%" align="middle" alt="Ouvrir le jeu de microscopie dans BioStudies"></a>

.. |DataHuggingFace| raw:: html

   <a href="https://huggingface.co/datasets/einarolafsson/toxo_mito"><img src="../../readme_assets/cells/5/databanks/huggingface_button.png" width="19.80%" align="middle" alt="Ouvrir le jeu de test sur Hugging Face"></a>

.. |DataNCBI| raw:: html

   <a href="https://www.ncbi.nlm.nih.gov/bioproject/?term=PRJNA1261935"><img src="../../readme_assets/cells/5/databanks/ncbi_button.png" width="19.80%" align="middle" alt="Ouvrir le jeu de séquençage dans NCBI"></a>

.. |DataSpaCRPower| raw:: html

   <a href="https://github.com/maomlab/spaCRPower"><img src="../../readme_assets/cells/5/databanks/spacrpower_button.png" width="19.80%" align="middle" alt="Ouvrir spaCRPower"></a>

.. |DataBioRxiv| raw:: html

   <a href="https://www.biorxiv.org/content/10.64898/2026.07.08.737057v1"><img src="../../readme_assets/cells/5/databanks/biorxiv_button.png" width="19.80%" align="middle" alt="Ouvrir la prépublication bioRxiv"></a>

.. raw:: html

   <div><a href="../../readme_text/fr/model-zoo.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/fr/model-zoo.desktop.svg"><img src="../../readme_images/fr/model-zoo.svg" alt="Bibliothèque de modèles. spaCR envoie un catalogue de modèles formés et les récupère sur demande. Ouvrez Modèle Zoo depuis l&#x27;écran d&#x27;accueil pour les parcourir et les installer, ou nommez une clé dans un fichier de paramètres -- pathogen_model: toxoplasma_pv_v1 -- et le modèle est téléchargé et vérifié la première fois qu&#x27;il est nécessaire. Chaque entrée publiée porte un SHA-256; une entrée sans un est refusée plutôt que installée, parce qu&#x27;un point de contrôle tronqué ou substitué ne peut pas être dit du vrai."></picture></a></div>

.. spacr-model-zoo-begin

.. raw:: html

   <div><a href="../../models/README.md"><picture><source media="(min-width: 1280px)" srcset="../../models/cards/heading.svg"><img src="../../models/cards/heading.phone.svg" alt="Default models · held-out scores"></picture></a></div>
   <div><a href="../../models/toxoplasma_pv_v4.md"><picture><source media="(min-width: 1280px)" srcset="../../models/cards/toxoplasma_pv_v4.svg"><img src="../../models/cards/toxoplasma_pv_v4.phone.svg" alt="Score card of Parasite vacuoles v4 (toxoplasma_pv_v4, default): F1 0.761 on 209 held-out fields against 0.377 for stock cpsam_v2. F1 0.761 against stock cpsam_v2&#x27;s 0.377 on the 209 strict hold-out fields at IoU 0.5; AJI 0.582 against 0.040."></picture></a><a href="../../models/toxoplasma_plaque_v4.md"><picture><source media="(min-width: 1280px)" srcset="../../models/cards/toxoplasma_plaque_v4.svg"><img src="../../models/cards/toxoplasma_plaque_v4.phone.svg" alt="Score card of Plaques v4 (toxoplasma_plaque_v4, default): F1 0.765 on 329 held-out fields against 0.338 for stock cpsam_v2. F1 0.765 against stock cpsam_v2&#x27;s 0.338 on the 329 strict hold-out fields at IoU 0.5; AJI 0.662 against 0.261."></picture></a><a href="../../models/toxoplasma_well_detector_v3.md"><picture><source media="(min-width: 1280px)" srcset="../../models/cards/toxoplasma_well_detector_v3.svg"><img src="../../models/cards/toxoplasma_well_detector_v3.phone.svg" alt="Score card of Plaque wells v3 (toxoplasma_well_detector_v3, default): mAP50 0.994 on 73 held-out images. Stock was not evaluated in this run; see the named incumbent comparison on the model card."></picture></a><a href="../../models/toxoplasma_from_cellmask_v1.md"><picture><source media="(min-width: 1280px)" srcset="../../models/cards/toxoplasma_from_cellmask_v1.svg"><img src="../../models/cards/toxoplasma_from_cellmask_v1.phone.svg" alt="Score card of Vacuoles from cell mask v1 (toxoplasma_from_cellmask_v1, default): F1 0.606 on 463 held-out fields against 0.021 for stock cpsam_v2. F1 0.606 against 0.021 for stock cpsam_v2 on 463 well-grouped held-out fields, at IoU 0.5."></picture></a><a href="../../models/live_cell_v1.md"><picture><source media="(min-width: 1280px)" srcset="../../models/cards/live_cell_v1.svg"><img src="../../models/cards/live_cell_v1.phone.svg" alt="Score card of Live cells v1 (live_cell_v1, default): F1 0.694 on 2,199 held-out fields against 0.738 for stock cpsam_v2. On the datasets stock cpsam_v2 never trained on, F1 0.960 against 0.885 at IoU 0.5; over all 2,199 test fields, 0.694 against 0.738, because stock trained on LIVECell and YeaZ and wins on LIVECell."></picture></a><a href="../../models/nuclei_from_cellmask_v1.md"><picture><source media="(min-width: 1280px)" srcset="../../models/cards/nuclei_from_cellmask_v1.svg"><img src="../../models/cards/nuclei_from_cellmask_v1.phone.svg" alt="Score card of Nuclei from cell mask v1 (nuclei_from_cellmask_v1, default): F1 0.888 on 453 held-out fields against 0.201 for stock cpsam_v2. F1 0.888 against 0.201 for stock cpsam_v2 on 453 well-grouped held-out fields, at IoU 0.5."></picture></a><a href="../../models/cell_from_hoechst_v1.md"><picture><source media="(min-width: 1280px)" srcset="../../models/cards/cell_from_hoechst_v1.svg"><img src="../../models/cards/cell_from_hoechst_v1.phone.svg" alt="Score card of Cells from Hoechst v1 (cell_from_hoechst_v1, default): F1 0.870 on 451 held-out fields against 0.301 for stock cpsam_v2. F1 0.870 against stock cpsam_v2&#x27;s 0.301 on 451 held-out fields at IoU 0.5 -- a delta of 0.569."></picture></a><a href="../../models/toxoplasma_from_hoechst_v1.md"><picture><source media="(min-width: 1280px)" srcset="../../models/cards/toxoplasma_from_hoechst_v1.svg"><img src="../../models/cards/toxoplasma_from_hoechst_v1.phone.svg" alt="Score card of Vacuoles from Hoechst v1 (toxoplasma_from_hoechst_v1, default): F1 0.569 on 463 held-out fields against 0.002 for stock cpsam_v2. F1 0.569 against 0.002 for stock cpsam_v2 on 463 well-grouped held-out fields, at IoU 0.5."></picture></a></div>

.. raw:: html

   <div><a href="../../readme_text/fr/generated-model-zoo-1.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/fr/generated-model-zoo-1.desktop.svg"><img src="../../readme_images/fr/generated-model-zoo-1.svg" alt="Modèles précédents."></picture></a></div>

.. raw:: html

   <div><a href="../../models/toxoplasma_pv_v3.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/fr/generated-model-zoo-1-link-1.desktop.svg"><img src="../../readme_images/fr/generated-model-zoo-1-link-1.svg" alt="toxoplasma_pv_v3"></picture></a><a href="../../models/toxoplasma_pv_v2.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/fr/generated-model-zoo-1-link-2.desktop.svg"><img src="../../readme_images/fr/generated-model-zoo-1-link-2.svg" alt="toxoplasma_pv_v2"></picture></a><a href="../../models/toxoplasma_pv_v1.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/fr/generated-model-zoo-1-link-3.desktop.svg"><img src="../../readme_images/fr/generated-model-zoo-1-link-3.svg" alt="toxoplasma_pv_v1"></picture></a><a href="../../models/toxoplasma_plaque_v2.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/fr/generated-model-zoo-1-link-4.desktop.svg"><img src="../../readme_images/fr/generated-model-zoo-1-link-4.svg" alt="toxoplasma_plaque_v2"></picture></a><a href="../../models/toxoplasma_plaque_v1.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/fr/generated-model-zoo-1-link-5.desktop.svg"><img src="../../readme_images/fr/generated-model-zoo-1-link-5.svg" alt="toxoplasma_plaque_v1"></picture></a><a href="../../models/toxoplasma_well_detector_v2.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/fr/generated-model-zoo-1-link-6.desktop.svg"><img src="../../readme_images/fr/generated-model-zoo-1-link-6.svg" alt="toxoplasma_well_detector_v2"></picture></a><a href="../../models/toxoplasma_well_detector_v1.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/fr/generated-model-zoo-1-link-7.desktop.svg"><img src="../../readme_images/fr/generated-model-zoo-1-link-7.svg" alt="toxoplasma_well_detector_v1"></picture></a></div>

.. raw:: html

   <div><a href="../../readme_text/fr/generated-model-zoo-2.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/fr/generated-model-zoo-2.desktop.svg"><img src="../../readme_images/fr/generated-model-zoo-2.svg" alt="Alternatives."></picture></a></div>

.. raw:: html

   <div><a href="../../models/toxoplasma_pv_v5.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/fr/generated-model-zoo-2-link-1.desktop.svg"><img src="../../readme_images/fr/generated-model-zoo-2-link-1.svg" alt="toxoplasma_pv_v5"></picture></a><a href="../../models/toxoplasma_plaque_v3.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/fr/generated-model-zoo-2-link-2.desktop.svg"><img src="../../readme_images/fr/generated-model-zoo-2-link-2.svg" alt="toxoplasma_plaque_v3"></picture></a></div>

.. raw:: html

   <div><a href="../../readme_text/fr/generated-model-zoo-3.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/fr/generated-model-zoo-3.desktop.svg"><img src="../../readme_images/fr/generated-model-zoo-3.svg" alt="Générateur de masques"></picture></a></div>

.. spacr-model-zoo-end

.. raw:: html

   <div><a href="../../readme_text/fr/diagnosing-performance.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/fr/diagnosing-performance.desktop.svg"><img src="../../readme_images/fr/diagnosing-performance.svg" alt="Diagnostic des performances. Générez un rapport matériel et joignez-le à un ticket relatif aux performances:"></picture></a></div>

.. code-block:: bash

   python tools/spacr_hardware_report.py

.. raw:: html

   <div><a href="../../readme_text/fr/diagnosing-performance-2.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/fr/diagnosing-performance-2.desktop.svg"><img src="../../readme_images/fr/diagnosing-performance-2.svg" alt="Enregistre dans ~/.spacr/reports et imprime le chemin. --quick saute les points de repère plus longs; --out PATH définit l&#x27;emplacement. Ne lit aucune donnée du projet. Mesure le temps des imports, des bibliothèques numériques, de la construction des fenêtres et des animations, et signale l’émulation x86_64 sur Apple Silicon ainsi que l’implémentation BLAS utilisée par NumPy."></picture></a></div>

.. raw:: html

   <div><a href="../../readme_text/fr/command-line-reference.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/fr/command-line-reference.desktop.svg"><img src="../../readme_images/fr/command-line-reference.svg" alt="Référence en ligne de commande. Chaque commande ci-dessous est installée par pip install spacr. Tous acceptent --help."></picture></a></div>

.. raw:: html

   <div><a href="../../readme_text/fr/launching-the-application.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/fr/launching-the-application.desktop.svg"><img src="../../readme_images/fr/launching-the-application.svg" alt="Lancement de l&#x27;application."></picture></a></div>

.. code-block:: bash

   spacr              # the desktop application
   spacr-tutorial     # the interactive tutorial library
   spacr-server       # no first-run setup screen, for unattended launches

.. raw:: html

   <div><a href="../../readme_text/fr/when-spacr-will-not-start.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/fr/when-spacr-will-not-start.desktop.svg"><img src="../../readme_images/fr/when-spacr-will-not-start.svg" alt="Quand spaCR ne démarrera pas."></picture></a></div>

.. code-block:: bash

   spacr-doctor       # diagnose the installation and say how to fix it
   safespacr          # the least spaCR that can still change a setting

.. raw:: html

   <div><a href="../../readme_text/fr/when-spacr-will-not-start-2.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/fr/when-spacr-will-not-start-2.desktop.svg"><img src="../../readme_images/fr/when-spacr-will-not-start-2.svg" alt="spacr-doctor affiche une ligne par vérification, avec une commande à exécuter pour chaque échec. Il indique également quel spacr est sur le chemin. safespacr lit chaque préférence avec sa valeur par défaut et désactive l’arrière-plan, les animations, la journalisation détaillée et le préchargement. Utilisez-le lorsqu’une préférence enregistrée empêche le démarrage. Les préférences enregistrées en mode sans échec sont écrites normalement et sont conservées."></picture></a></div>

.. raw:: html

   <div><a href="../../readme_text/fr/running-modules-headlessly.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/fr/running-modules-headlessly.desktop.svg"><img src="../../readme_images/fr/running-modules-headlessly.svg" alt="Modules de exécution sans interface graphique. Pas de Qt, pas d&#x27;affichage — pour les clusters, les serveurs et les CI."></picture></a></div>

.. code-block:: bash

   spacr-run --list                              # modules with a headless entry
   spacr-run --describe MODULE                   # what a module consumes and produces
   spacr-run validate --module MODULE \
       --settings settings.csv                   # check settings before spending the run
   spacr-run MODULE --settings settings.csv      # execute
   spacr-remote --help                           # submit and monitor SSH, Slurm or cloud jobs

.. raw:: html

   <div><a href="../../readme_text/fr/running-modules-headlessly-2.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/fr/running-modules-headlessly-2.desktop.svg"><img src="../../readme_images/fr/running-modules-headlessly-2.svg" alt="validate lit les mêmes paramètres que l’exécution et liste chaque problème, par exemple un paramètre manquant ou un chemin inexistant, avec une correction."></picture></a></div>

.. raw:: html

   <div><a href="../../readme_text/fr/inspecting-a-run.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/fr/inspecting-a-run.desktop.svg"><img src="../../readme_images/fr/inspecting-a-run.svg" alt="Inspecter une exécution. Chaque exécution est journalisée à ~/.spacr/runs avec ses paramètres, entrées hashées, sorties, avertissements, versions et graines."></picture></a></div>

.. code-block:: bash

   spacr-repro RUN_DIR        # replay a recorded run from its journal
   spacr-workspace RUN_DIR    # what that run had open: databases, montages, views

.. raw:: html

   <div><a href="../../readme_text/fr/auditing-data-and-installation.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/fr/auditing-data-and-installation.desktop.svg"><img src="../../readme_images/fr/auditing-data-and-installation.svg" alt="Audit des données et installation."></picture></a></div>

.. code-block:: bash

   spacr-db-audit DB      # SQLite health, integrity, locking, reader/writer probe
   spacr-leakage          # classifier train/test leakage audit
   spacr-plugins          # installed plugin registry and failure diagnostics

.. raw:: html

   <div><a href="../../readme_text/fr/environment.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/fr/environment.desktop.svg"><img src="../../readme_images/fr/environment.svg" alt="Environnement."></picture></a></div>

.. code-block:: bash

   SPACR_LOG_LEVEL=DEBUG spacr      # verbose logging for one launch

.. raw:: html

   <div><a href="../../readme_text/fr/environment-2.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/fr/environment-2.desktop.svg"><img src="../../readme_images/fr/environment-2.svg" alt="Les journaux rotatifs sont écrits à ~/.spacr/logs/spacr.log. Joindre ce fichier à un rapport de bogue."></picture></a></div>

.. raw:: html

   <div><a href="../../readme_text/fr/contributing-and-support.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/fr/contributing-and-support.desktop.svg"><img src="../../readme_images/fr/contributing-and-support.svg" alt="Contributions et assistance. Soumettez les rapports de bogues et les demandes de fonctionnalités bien délimitées via GitHub Issues. Lorsque vous signalez un échec, indiquez la version de spaCR, le système d’exploitation, la version de Python, les paramètres du module et l’extrait de journal pertinent. spacr-doctor collecte la plupart de ces informations ; joignez le rapport matériel lorsque vous signalez un problème de performances."></picture></a></div>

.. raw:: html

   <div><a href="../../readme_text/fr/licensing.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/fr/licensing.desktop.svg"><img src="../../readme_images/fr/licensing.svg" alt="Licence. spaCR est libéré sous le Licence BSD 3-Clause. Si spaCR a contribué à des travaux publiés, une citation est appréciée et n&#x27;est pas une condition de la licence — voir Citer spaCR ci-dessous."></picture></a></div>

.. raw:: html

   <div><a href="../../readme_text/fr/tutorials.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/fr/tutorials.desktop.svg"><img src="../../readme_images/fr/tutorials.svg" alt="Tutoriels. La bibliothèque de tutoriels interactifs spaCR guide l’installation et l’utilisation des modules. Chaque leçon indique les narrations et les langues disponibles."></picture></a></div>

.. raw:: html

   <div><a href="../../readme_text/fr/citing-spacr.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/fr/citing-spacr.desktop.svg"><img src="../../readme_images/fr/citing-spacr.svg" alt="Citer spaCR. Si spaCR contribue à vos recherches, citez: Olafsson EB, et al. Un criblage d&#x27;image groupée CRISPR identifie EAF1 comme un modulateur T. gondii de la subversion ESCRT."></picture></a></div>

.. raw:: html

   <div><a href="https://www.biorxiv.org/content/10.64898/2026.07.08.737057v1"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/fr/citing-spacr-2-1.desktop.svg"><img src="../../readme_images/fr/citing-spacr-2-1.svg" alt="préimpression bioRxiv"></picture></a><a href="https://doi.org/10.5281/zenodo.21343316"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/fr/citing-spacr-2-2.desktop.svg"><img src="../../readme_images/fr/citing-spacr-2-2.svg" alt="archive logicielle"></picture></a></div>

.. raw:: html

   <div><a href="../../readme_text/fr/other-work-citing-spacr.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/fr/other-work-citing-spacr.desktop.svg"><img src="../../readme_images/fr/other-work-citing-spacr.svg" alt="Autres ouvrages citant spaCR."></picture></a></div>

.. spacr-citing-papers-begin

.. raw:: html

   <div><a href="../../readme_text/fr/other-work-citing-spacr-2-1.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/fr/other-work-citing-spacr-2-1.desktop.svg"><img src="../../readme_images/fr/other-work-citing-spacr-2-1.svg" alt="Adaptabilité métabolique et récupération des nutriments dans Toxoplasma gondii : aperçus de mutants déficients en voie d&#x27;ingestion."></picture></a></div>

.. raw:: html

   <div><a href="../../readme_text/fr/other-work-citing-spacr-2-2.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/fr/other-work-citing-spacr-2-2.desktop.svg"><img src="../../readme_images/fr/other-work-citing-spacr-2-2.svg" alt="L&#x27;IRE1α favorise le flux phagosomique de calcium pour améliorer l&#x27;activité fongicide des macrophages."></picture></a></div>

.. raw:: html

   <div><a href="../../readme_text/fr/other-work-citing-spacr-2-3.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/fr/other-work-citing-spacr-2-3.desktop.svg"><img src="../../readme_images/fr/other-work-citing-spacr-2-3.svg" alt="Toxoplasma GRA8 engage la protéine accessoire ESCRT de l&#x27;hôte ALG-2 et est nécessaire pour l&#x27;intégrité métabolique du parasite."></picture></a></div>

.. raw:: html

   <div><a href="../../readme_text/fr/other-work-citing-spacr-2-4.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/fr/other-work-citing-spacr-2-4.desktop.svg"><img src="../../readme_images/fr/other-work-citing-spacr-2-4.svg" alt="Hydrogen sulfide dynamically upregulates copper uptake and localization."></picture></a></div>

.. raw:: html

   <div><a href="../../readme_text/fr/other-work-citing-spacr-2-5.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/fr/other-work-citing-spacr-2-5.desktop.svg"><img src="../../readme_images/fr/other-work-citing-spacr-2-5.svg" alt="spaCR: Analyse du phénotype spatial des criblages CRISPR-Cas9 (version préimpression 1)."></picture></a></div>

.. spacr-citing-papers-end

.. raw:: html

   <div><a href="../../readme_text/fr/acknowledgments.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/fr/acknowledgments.desktop.svg"><img src="../../readme_images/fr/acknowledgments.svg" alt="Remerciements. spaCR repose sur des logiciels scientifiques ouverts, notamment NumPy, pandas, scikit-image, scikit-learn, Cellpose, PyTorch et Qt. Consultez l’attribution des modèles de traduction pour connaître les modèles utilisés dans la documentation multilingue et les catalogues de l’interface."></picture></a></div>
