|Website| |Platforms| |Python| |Qt| |Test counts| |Coverage| |Issues| |Conda| |PyPI| |Docker| |PyPI Downloads| |Docs| |Tutorials| |Preprint| |DOI| |Cite| |License| |PyPI rank|

Webbplats: `einarolafsson.github.io/projects/spacr <https://einarolafsson.github.io/projects/spacr/>`_

.. |Website| image:: https://img.shields.io/badge/Website-spaCR-0A7BBB
   :target: https://einarolafsson.github.io/projects/spacr/
   :alt: spaCR:s webbplats

.. |Docs| image:: https://img.shields.io/github/actions/workflow/status/EinarOlafsson/spacr/pages%2Fpages-build-deployment?label=API&logo=readthedocs&logoColor=white
   :target: https://einarolafsson.github.io/spacr/
   :alt: API-dokumentation

.. |Tutorials| image:: https://img.shields.io/badge/Tutorials-4A9EFF
   :target: https://einarolafsson.github.io/spacr/tutorials/
   :alt: Interaktiva handledningar

.. |PyPI| image:: https://img.shields.io/pypi/v/spacr?label=PyPI&logo=pypi&logoColor=white
   :target: https://pypi.org/project/spacr/
   :alt: PyPI-version

.. |Python| image:: https://img.shields.io/badge/Python-3.9%E2%80%933.14-3776AB?logo=python&logoColor=white
   :target: https://pypi.org/project/spacr/
   :alt: Python 3.9 till 3.14

.. |Coverage| image:: https://img.shields.io/codecov/c/github/EinarOlafsson/spacr/main?logo=codecov&logoColor=white
   :target: https://app.codecov.io/github/EinarOlafsson/spacr
   :alt: Testtäckning på Codecov

.. |Test counts| image:: https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2FEinarOlafsson%2Fspacr%2Fnightly%2Fdocs%2Fsource%2F_static%2Ftest-counts.json&cacheSeconds=300&logo=pytest&logoColor=white
   :target: https://github.com/EinarOlafsson/spacr/actions/workflows/tests.yml
   :alt: Testsvit

.. |Qt| image:: https://img.shields.io/badge/GUI-Qt-41CD52?logo=qt&logoColor=white
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/index.html#module-spacr.qt
   :alt: Qt-gränssnitt

.. |Issues| image:: https://img.shields.io/github/issues/EinarOlafsson/spacr?logo=github
   :target: https://github.com/EinarOlafsson/spacr/issues
   :alt: GitHub-ärenden

.. |License| image:: https://img.shields.io/badge/License-BSD_3--Clause-blue
   :target: https://github.com/EinarOlafsson/spacr/blob/nightly/LICENSE
   :alt: BSD 3-Clause

.. |Preprint| image:: https://img.shields.io/badge/bioRxiv-BF2636
   :target: https://www.biorxiv.org/content/10.64898/2026.07.08.737057v1
   :alt: bioRxiv-preprint

.. |DOI| image:: https://img.shields.io/badge/DOI-blue?logo=zenodo&logoColor=white
   :target: https://doi.org/10.5281/zenodo.21343316
   :alt: Zenodo-DOI

.. |Conda| image:: https://img.shields.io/conda/vn/conda-forge/spacr?label=Conda&logo=anaconda&logoColor=white
   :target: https://anaconda.org/conda-forge/spacr
   :alt: conda-forge-version

.. |Release date| image:: https://anaconda.org/conda-forge/spacr/badges/latest_release_date.svg
   :target: https://anaconda.org/conda-forge/spacr
   :alt: Datum för senaste conda-forge-versionen

.. |Docker| image:: https://img.shields.io/badge/Docker-GHCR-2496ED?logo=docker&logoColor=white
   :target: https://github.com/EinarOlafsson/spacr/pkgs/container/spacr
   :alt: Docker-avbildningar på GHCR

.. |PyPI Downloads| image:: https://img.shields.io/pepy/dt/spacr?label=downloads&logo=pypi&logoColor=white
   :target: https://pepy.tech/projects/spacr
   :alt: PyPI-nedladdningar

.. |Platforms| image:: https://img.shields.io/badge/Linux%20OSX%20Win-lightgrey?logo=linux&logoColor=white
   :target: https://github.com/EinarOlafsson/spacr/blob/nightly/docs/source/installers.rst
   :alt: Linux, macOS och Windows

.. |Cite| image:: https://img.shields.io/badge/Cite-8A2BE2
   :target: https://github.com/EinarOlafsson/spacr/blob/main/CITATION.cff
   :alt: Citera spaCR

.. |PyPI rank| image:: https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fsql-clickhouse.clickhouse.com%2F%3Fuser%3Ddemo%26param_package_name%3Dspacr%26param_days%3D30%26query%3DWITH%2B%2528%2BSELECT%2Bsum%2528count%2529%2BFROM%2Bpypi.pypi_downloads_per_day%2BWHERE%2Bproject%2B%253D%2B%257Bpackage_name%253AString%257D%2BAND%2Bdate%2B%253E%253D%2BtoDate%2528now%2528%2527UTC%2527%2529%2529%2B-%2B%257Bdays%253AUInt16%257D%2BAND%2Bdate%2B%253C%2BtoDate%2528now%2528%2527UTC%2527%2529%2529%2B%2529%2BAS%2Bdownloads%2BSELECT%2Bdownloads%2BAS%2Bpackage_downloads%252C%2BcountIf%2528n%2B%253E%253D%2Bdownloads%2529%2BAS%2Brank%252C%2Bcount%2528%2529%2BAS%2Btotal_packages%252C%2B100.0%2B%252A%2Brank%2B%252F%2BnullIf%2528total_packages%252C%2B0%2529%2BAS%2Bpercentile%252C%2Bif%2528%2Btotal_packages%2B%253D%2B0%2BOR%2Bdownloads%2B%253D%2B0%252C%2B%2527no%2Bdata%2527%252C%2Bconcat%2528%2B%2527top%2B%2527%252C%2BtoString%2528ceil%25281000.0%2B%252A%2Brank%2B%252F%2BnullIf%2528total_packages%252C%2B0%2529%2529%2B%252F%2B10%2529%252C%2B%2527%2525%2527%2B%2529%2B%2529%2BAS%2Bmessage%2BFROM%2B%2528%2BSELECT%2Bproject%252C%2Bsum%2528count%2529%2BAS%2Bn%2BFROM%2Bpypi.pypi_downloads_per_day%2BWHERE%2Bdate%2B%253E%253D%2BtoDate%2528now%2528%2527UTC%2527%2529%2529%2B-%2B%257Bdays%253AUInt16%257D%2BAND%2Bdate%2B%253C%2BtoDate%2528now%2528%2527UTC%2527%2529%2529%2BGROUP%2BBY%2Bproject%2B%2529%2BFORMAT%2BJSON&query=%24.data%5B0%5D.message&label=Rank&color=brightgreen&logo=pypi&logoColor=white&cacheSeconds=86400
   :target: https://clickpy.clickhouse.com/dashboard/spacr
   :alt: spaCR:s PyPI-nedladdningsranking de senaste 30 hela dygnen

.. raw:: html

   <div><a href="https://einarolafsson.github.io/spacr/_static/deck/"><picture><source media="(min-width: 1280px)" srcset="../../readme_assets/cover.desktop.svg"><img src="../../readme_assets/cover.svg" alt="spaCR"></picture></a></div>

.. raw:: html

   <div><a href="../../source/_static/deck/pages/57.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/sv/intro-1.desktop.svg"><img src="../../readme_images/sv/intro-1.svg" alt="← Föregående"></picture></a><a href="../../source/_static/deck/pages/02.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/sv/intro-2.desktop.svg"><img src="../../readme_images/sv/intro-2.svg" alt="Nästa →"></picture></a></div>

.. spacr-language-picker-begin

.. raw:: html

   <div><a href="README.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/sv/generated-language-picker.desktop.svg"><img src="../../readme_images/sv/generated-language-picker.svg" alt="Språk: Svenska"></picture></a></div>

.. spacr-language-picker-end

.. raw:: html

   <div><a href="../../readme_text/sv/spacr.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/sv/spacr.desktop.svg"><img src="../../readme_images/sv/spacr.svg" alt="spaCR. Rumslig fenotypanalys av CRISPR-screeningar. spaCR segmenterar och mäter enskilda celler i mikroskopibilder, integrerar fenotyper per objekt med sekvenseringshärledd guideförekomst och uppskattar vilka gener som är associerade med fenotypiska förändringar. Med plattbilder och FASTQ-läsningar som utgångspunkt producerar programmet mätningar per objekt, tränade klassificerare, effektskattningar per guide och gen samt en rangordnad träfflista. Modulerna för segmentering, mätning, annotering och klassificering kan också köras utan någon sekvenseringsdel. Make Masks korrigerar segmenteringsmasker och annoterar oberoende, klassmärkta rektanglar med verktyget Box för YOLO-export. Rutorna har egna etiketter och egen historik utan att ändra källbilder eller masker. Se funktionsguiden för varje verktyg. Bilder, masker och bildutsnitt är filer på disk. Mätningar, annoteringar och modellprediktioner är tabeller i projektets SQLite-databas measurements.db. Körs som ett skrivbordsprogram eller utan grafiskt gränssnitt på en arbetsstation, server eller kluster."></picture></a></div>

.. raw:: html

   <div><a href="../../readme_text/sv/try-spacr.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/sv/try-spacr.desktop.svg"><img src="../../readme_images/sv/try-spacr.svg" alt="Prova spaCR."></picture></a></div>

.. code-block:: bash

   conda create -n spacr python=3.12 -y
   conda activate spacr
   python -m pip install spacr
   spacr

.. raw:: html

   <div><a href="../../readme_text/sv/hardware-support.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/sv/hardware-support.desktop.svg"><img src="../../readme_images/sv/hardware-support.svg" alt="Hårdvarustöd."></picture></a></div>

.. spacr-hardware-begin

.. raw:: html

   <div><a href="../../readme_text/sv/generated-hardware-1.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/sv/generated-hardware-1.desktop.svg"><img src="../../readme_images/sv/generated-hardware-1.svg" alt="NVIDIA (CUDA): Cellpose 4 GPU, Torch GPU, UMAP / clustering GPU, 3D GPU. AMD on Linux (ROCm): Cellpose 4 GPU, Torch GPU, UMAP / clustering CPU, 3D GPU. AMD in an Intel Mac (Metal): Cellpose 4 GPU, Torch GPU, UMAP / clustering KMeans GPU; PCA anpassning CPU, projektion GPU / CPU; UMAP / t-SNE / DBSCAN CPU, 3D GPU. Apple Silicon (Metal): Cellpose 4 GPU, Torch GPU, UMAP / clustering KMeans GPU; PCA anpassning CPU, projektion GPU / CPU; UMAP / t-SNE / DBSCAN CPU, 3D GPU. Intel Arc/Xe (XPU): Cellpose 4 GPU, Torch GPU, UMAP / clustering CPU, 3D GPU. No GPU: Cellpose 4 CPU, Torch CPU, UMAP / clustering CPU, 3D CPU. stöds (stabilt) implementerat (beta) endast CPU-stöd."></picture></a></div>

.. spacr-hardware-end

.. raw:: html

   <div><a href="../../readme_text/sv/install-spacr.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/sv/install-spacr.desktop.svg"><img src="../../readme_images/sv/install-spacr.svg" alt="Installera spaCR. Välj en installationsväg: Skrivbordsprogram: ett installationsprogram för Linux, macOS eller Windows med egen Python. pip: python -m pip install spacr i en Conda-miljö. conda-forge: conda install conda-forge::spacr i en Conda-miljö. Docker: kommandoradens bearbetningsflöden i en Linux-container, från CPU-avbildningen eller avbildningen för NVIDIA-GPU:er:"></picture></a></div>

.. code-block:: bash

   docker run --rm ghcr.io/einarolafsson/spacr:1.5.1.6 spacr-run --list
   docker run --rm --gpus all ghcr.io/einarolafsson/spacr:1.5.1.6-cuda12.4 spacr-run --list

.. raw:: html

   <div><a href="../../readme_text/sv/install-spacr-2.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/sv/install-spacr-2.desktop.svg"><img src="../../readme_images/sv/install-spacr-2.svg" alt="Docker-guiden (nightly, engelska) beskriver GPU-inställning, montering av data och kommandon för bearbetningsflödena."></picture></a></div>

.. raw:: html

   <div><a href="../../readme_text/sv/desktop-application.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/sv/desktop-application.desktop.svg"><img src="../../readme_images/sv/desktop-application.svg" alt="Skrivbordsprogram. Installationsprogrammen innehåller en egen Python-miljö. Conda behövs inte."></picture></a></div>

.. spacr-installer-links-begin

| |InstallerLinux|\ |InstallerMacOS|\ |InstallerWindows|

.. |InstallerWindows| raw:: html

   <a href="https://github.com/EinarOlafsson/spacr/releases/download/v1.5.1.6/spaCR-1.5.1.6-Windows-Online-Setup.exe"><img src="../../readme_assets/cells/platforms/windows.png" width="14.14%" align="middle" alt="Hämta spaCR 1.5.1.6 för Windows 10/11"></a>
.. |InstallerMacOS| raw:: html

   <a href="https://github.com/EinarOlafsson/spacr/releases/download/v1.5.1.6/spaCR-1.5.1.6-macOS-Universal-Online.pkg"><img src="../../readme_assets/cells/platforms/macos.png" width="14.14%" align="middle" alt="Hämta spaCR 1.5.1.6 för macOS 11+ (Intel och Apple Silicon)"></a>
.. |InstallerLinux| raw:: html

   <a href="https://github.com/EinarOlafsson/spacr/releases/download/v1.5.1.6/spaCR-1.5.1.6-Linux-x86_64-Online.run"><img src="../../readme_assets/cells/platforms/linux.png" width="14.14%" align="middle" alt="Hämta spaCR 1.5.1.6 för 64-bitars Linux"></a>

.. spacr-installer-links-end

.. raw:: html

   <div><a href="../../readme_text/sv/desktop-application-2.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/sv/desktop-application-2.desktop.svg"><img src="../../readme_images/sv/desktop-application-2.svg" alt="Gör den hämtade filen körbar i Linux och kör den:"></picture></a></div>

.. code-block:: bash

   chmod +x SpaCR-*-Linux-x86_64-Online.run
   ./SpaCR-*-Linux-x86_64-Online.run

.. raw:: html

   <div><a href="../../readme_text/sv/desktop-application-3.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/sv/desktop-application-3.desktop.svg"><img src="../../readme_images/sv/desktop-application-3.svg" alt="På macOS, öppna .pkg. Den nuvarande betaversionen är inte notariserad. Om Gatekeeper blockerar den, välj Systeminställningar → Integritet &amp; Säkerhet → Öppna ändå. För installation, uppdatering, ominstallation, avinstallation, offlineanvändning och felsökning, se den aktuella installationsguiden på nightly (engelska). Se systemkraven för rekommendationer om arbetsstationer och servrar samt tabeller över GPU-kompatibilitet."></picture></a></div>

.. raw:: html

   <div><a href="https://github.com/EinarOlafsson/spacr/blob/nightly/docs/source/installer_guide.rst"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/sv/desktop-application-3-link-1.desktop.svg"><img src="../../readme_images/sv/desktop-application-3-link-1.svg" alt="aktuella installationsguiden på nightly (engelska)"></picture></a><a href="../../source/system_requirements.rst"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/sv/desktop-application-3-link-2.desktop.svg"><img src="../../readme_images/sv/desktop-application-3-link-2.svg" alt="systemkraven"></picture></a></div>

.. raw:: html

   <div><a href="../../readme_text/sv/pypi-installation.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/sv/pypi-installation.desktop.svg"><img src="../../readme_images/sv/pypi-installation.svg" alt="Installation från PyPI. För PyPI-utgåvan installerar du spaCR med pip i en Conda-miljö. Python 3.12 ger det största urvalet av valfria vetenskapliga paket:"></picture></a></div>

.. code-block:: bash

   conda create -n spacr python=3.12 -y
   conda activate spacr
   python -m pip install --upgrade pip
   python -m pip install spacr
   spacr

.. raw:: html

   <div><a href="../../readme_text/sv/pypi-installation-2.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/sv/pypi-installation-2.desktop.svg"><img src="../../readme_images/sv/pypi-installation-2.svg" alt="spaCR stöder Python 3.9 through 3.14, utom Python 3.14.1, som torchvision utesluter. Linux rekommenderas för de tyngsta CUDA- och ROCm-arbetsflödena; macOS och Windows stöds också, och båda använder sina GPU:er — macOS via Metal, som täcker Apple Silicon och AMD-korten i Intel-Mac-datorer, och Windows via CUDA eller DirectML. Standardinstallationen innehåller Qt-gränssnittet för skrivbordet. På en server, ett kluster eller en CI-körmiljö kan du köra kommandoradspipelines utan att öppna det:"></picture></a></div>

.. code-block:: bash

   python -m pip install spacr
   spacr-run --list

.. raw:: html

   <div><a href="../../readme_text/sv/pypi-installation-3.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/sv/pypi-installation-3.desktop.svg"><img src="../../readme_images/sv/pypi-installation-3.svg" alt="Valfria integrationer installeras separat, till exempel spacr[zarr], spacr[omero], spacr[napari] och spacr[czi,nd2,lif]. Den fullständiga listan över tillägg och tabellen över kompatibla Python-versioner finns i installationsguiden (nightly, engelska)."></picture></a></div>

.. raw:: html

   <div><a href="../../readme_text/sv/conda-forge-installation.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/sv/conda-forge-installation.desktop.svg"><img src="../../readme_images/sv/conda-forge-installation.svg" alt="Installation med conda-forge. Det officiella conda-forge-paketet installerar spaCR och dess skrivbordsberoenden i den aktiva miljön:"></picture></a></div>

.. code-block:: bash

   conda create -n spacr python=3.12 -y
   conda activate spacr
   conda install conda-forge::spacr
   spacr

.. raw:: html

   <div><a href="../../readme_text/sv/install-from-source.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/sv/install-from-source.desktop.svg"><img src="../../readme_images/sv/install-from-source.svg" alt="Installation från källkod. Klona kodförrådet och installera det i redigerbart läge, så att din arbetskopia är det installerade paketet och ändringar börjar gälla utan ominstallation:"></picture></a></div>

.. code-block:: bash

   git clone https://github.com/EinarOlafsson/spacr.git
   cd spacr
   conda create -n spacr python=3.12 -y
   conda activate spacr
   pip install -e .
   spacr

.. raw:: html

   <div><a href="../../readme_text/sv/install-from-source-2.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/sv/install-from-source-2.desktop.svg"><img src="../../readme_images/sv/install-from-source-2.svg" alt="Detta klonar main, standardgrenen, som innehåller den senaste utgåvan. Utvecklingen sker på nightly; lägg till --branch nightly för att klona den i stället. För en specifik utgåva:"></picture></a></div>

.. code-block:: bash

   git clone --branch v1.5.0.5 https://github.com/EinarOlafsson/spacr.git

.. raw:: html

   <div><a href="../../readme_text/sv/install-from-source-3.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/sv/install-from-source-3.desktop.svg"><img src="../../readme_images/sv/install-from-source-3.svg" alt="För att hämta senare ändringar kör du inifrån klonen:"></picture></a></div>

.. code-block:: bash

   git pull
   pip install -e .

.. raw:: html

   <div><a href="../../readme_text/sv/install-from-source-4.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/sv/install-from-source-4.desktop.svg"><img src="../../readme_images/sv/install-from-source-4.svg" alt="Installera om när beroenden eller kommandostarter ändras. Ändringar i Python-koden gäller direkt; spacr-doctor visar vilken installation som används."></picture></a></div>

.. raw:: html

   <div><a href="../../readme_text/sv/install-from-source-light.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/sv/install-from-source-light.desktop.svg"><img src="../../readme_images/sv/install-from-source-light.svg" alt="Installation från källkod (lättviktig). Bidragsgivare behöver versionshistoriken; välj ett alternativ nedan om du bara vill köra spaCR. Mätningarna gäller nightly vid 05302fd5c den 2026-10-07 och gjordes med packaging/measure_clone_forms.sh. Ytlig klon: 2048 MB hämtades på 140 s. Endast den senaste incheckningen behålls, så äldre historik är inte tillgänglig för git log, git blame eller git bisect. git pull fungerar men klonen förblir ytlig; git fetch --unshallow hämtar hela historiken."></picture></a></div>

.. code-block:: bash

   git clone --depth 1 --branch nightly https://github.com/EinarOlafsson/spacr.git
   cd spacr && pip install -e .

.. raw:: html

   <div><a href="../../readme_text/sv/install-from-source-light-2.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/sv/install-from-source-light-2.desktop.svg"><img src="../../readme_images/sv/install-from-source-light-2.svg" alt="Enbart körningsfiler: 104 MB på disk på 6 s (Git: 38 MB; filer: 66 MB). Historik, docs, tests, tools, features, exempeldata och utökade gränssnittsöversättningar utelämnas. --with-docs, --with-tests och --with-translations behåller motsvarande dokumentation, tester och översättningar. De kompakta gränssnittsöversättningarna finns kvar."></picture></a></div>

.. code-block:: bash

   curl -fsSL https://raw.githubusercontent.com/EinarOlafsson/spacr/nightly/packaging/install_from_source.sh -o install_spacr.sh
   sh install_spacr.sh --branch nightly

.. raw:: html

   <div><a href="../../readme_text/sv/install-from-source-light-3.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/sv/install-from-source-light-3.desktop.svg"><img src="../../readme_images/sv/install-from-source-light-3.svg" alt="--dir väljer klonens sökväg; --branch väljer gren eller tagg; --no-install hämtar utan att installera; --help visar alternativen. packaging/source_install_excludes.txt listar alla utelämnade sökvägar. Den fullständiga nightly-klonen laddade ner 9.25 GiB. Att lägga till --filter=blob:none till den grunda klonen minskar inte utcheckningen: dess Git-objektlager är fortfarande 2032 MB. Git rapporterar inte lata hämtningar av objekt, så ingen total nedladdning uppmättes. De versionshanterade filerna i nightly ger en utcheckning på 720 MB (uppmätt 2026-10-21), utan Git-historiken. Nedladdningens storlek och tid varierar med grenen."></picture></a></div>

.. raw:: html

   <div><a href="../../readme_text/sv/command-line-entry-points.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/sv/command-line-entry-points.desktop.svg"><img src="../../readme_images/sv/command-line-entry-points.svg" alt="Kommandoradskommandon."></picture></a></div>

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

   <div><a href="../../readme_text/sv/command-line-entry-points-2.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/sv/command-line-entry-points-2.desktop.svg"><img src="../../readme_images/sv/command-line-entry-points-2.svg" alt="spacr-run --list listar moduler med kommandoradsposter för körning utan grafiskt gränssnitt. GUI-bundna moduler för annotering, kurering, jämförelse och utforskning utelämnas."></picture></a></div>

.. raw:: html

   <div><a href="../../readme_text/sv/core-workflow.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/sv/core-workflow.desktop.svg"><img src="../../readme_images/sv/core-workflow.svg" alt="Kärnarbetsflöde. Det primära arbetsflödet består av sex moduler: Mask segmenterar celler, cellkärnor, patogener och organeller med Cellpose. Measure skriver morfologiska, intensitets-, textur-, rumsliga och kolokaliseringsmått samt objektutsnitt till SQLite. Annotate märker objektutsnitt i ett tangentbordsstyrt rutnät och stöder köer för aktiv inlärning. Classify tränar bild- eller mätningsbaserade modeller och sparar prestandan på undanhållna data med varje kontrollpunkt. Map Barcodes kopplar FASTQ-läsningar till brunnar och gRNA:er och rapporterar QC för förekomst, kollisioner och täckning. Regression skattar effekter för guider, gener, betingelser och kontroller med modellfamiljer för kontinuerliga data, andelar och antal."></picture></a></div>

.. raw:: html

   <div><a href="../../readme_text/sv/spacr-modules.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/sv/spacr-modules.desktop.svg"><img src="../../readme_images/sv/spacr-modules.svg" alt="spaCR-moduler."></picture></a></div>

.. spacr-workflow-begin

.. raw:: html

   <div><a href="../../readme_text/sv/generated-workflow-1.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/sv/generated-workflow-1.desktop.svg"><img src="../../readme_images/sv/generated-workflow-1.svg" alt="Kärna. Core sequence from microscopy images through segmentation, measurements, annotations, classification, barcode mapping and regression."></picture></a></div>

| |Module_mask|\ |Module_measure|\ |Module_annotate|\ |Module_classify_merged|\ |Module_map_barcodes|\ |Module_regression|

.. raw:: html

   <div><a href="../../readme_text/sv/generated-workflow-2.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/sv/generated-workflow-2.desktop.svg"><img src="../../readme_images/sv/generated-workflow-2.svg" alt="Data. Import images and tables into spaCR projects and execute reproducible multi-plate workflows."></picture></a></div>

| |Module_foreign|\ |Module_embeddings|\ |Module_run_compare|\ |Module_experiment_design|\ |Module_power|\ |Module_dose_response|\ |Module_qc_dashboard|

.. raw:: html

   <div><a href="../../readme_text/sv/generated-workflow-3.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/sv/generated-workflow-3.desktop.svg"><img src="../../readme_images/sv/generated-workflow-3.svg" alt="Verktyg. Point these at a project: edit masks by hand, stitch tiles, read an embedding, draw a gate, build a plot, check quality."></picture></a></div>

| |Module_make_masks|\ |Module_align|\ |Module_umap|\ |Module_gate_editor|\ |Module_graph_builder|

.. raw:: html

   <div><a href="../../readme_text/sv/generated-workflow-4.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/sv/generated-workflow-4.desktop.svg"><img src="../../readme_images/sv/generated-workflow-4.svg" alt="Organism. Organismspecifik bildanalys och kvantitativa resultat från biologiska analyser."></picture></a></div>

| |Module_toxoplasma|

.. |Module_mask| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/core/index.html#spacr.core.preprocess_generate_masks"><img src="../../readme_assets/cells/workflow/mask.png" width="14.14%" align="middle" alt="Öppna API-dokumentationen för Mask"></a>

.. |Module_measure| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/measure/index.html"><img src="../../readme_assets/cells/workflow/measure.png" width="14.14%" align="middle" alt="Öppna API-dokumentationen för Measure"></a>

.. |Module_annotate| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/qt/screens/annotate/index.html"><img src="../../readme_assets/cells/workflow/annotate.png" width="14.14%" align="middle" alt="Öppna API-dokumentationen för Annotate"></a>

.. |Module_classify_merged| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/classify/index.html"><img src="../../readme_assets/cells/workflow/classify_merged.png" width="14.14%" align="middle" alt="Öppna API-dokumentationen för Classify"></a>

.. |Module_map_barcodes| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/sequencing/index.html"><img src="../../readme_assets/cells/workflow/map_barcodes.png" width="14.14%" align="middle" alt="Öppna API-dokumentationen för Map Barcodes"></a>

.. |Module_regression| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/ml/index.html"><img src="../../readme_assets/cells/workflow/regression.png" width="14.14%" align="middle" alt="Öppna API-dokumentationen för Regression"></a>

.. |Module_foreign| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/foreign/index.html"><img src="../../readme_assets/cells/workflow/apps/foreign.png" width="14.14%" align="middle" alt="Öppna API-dokumentationen för Import"></a>

.. |Module_embeddings| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/qt/screens/embeddings/index.html"><img src="../../readme_assets/cells/workflow/apps/embeddings.png" width="14.14%" align="middle" alt="Öppna API-dokumentationen för Embeddings"></a>

.. |Module_run_compare| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/qt/screens/run_compare/index.html"><img src="../../readme_assets/cells/workflow/apps/run_compare.png" width="14.14%" align="middle" alt="Öppna API-dokumentationen för Run Compare"></a>

.. |Module_experiment_design| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/qt/screens/experiment_design/index.html"><img src="../../readme_assets/cells/workflow/apps/experiment_design.png" width="14.14%" align="middle" alt="Öppna API-dokumentationen för Experiment Design"></a>

.. |Module_power| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/qt/screens/power/index.html"><img src="../../readme_assets/cells/workflow/apps/power.png" width="14.14%" align="middle" alt="Öppna API-dokumentationen för Power / Design"></a>

.. |Module_dose_response| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/qt/screens/dose_response/index.html"><img src="../../readme_assets/cells/workflow/apps/dose_response.png" width="14.14%" align="middle" alt="Öppna API-dokumentationen för Dose–Response"></a>

.. |Module_qc_dashboard| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/qt/screens/qc_dashboard/index.html"><img src="../../readme_assets/cells/workflow/apps/qc_dashboard.png" width="14.14%" align="middle" alt="Öppna API-dokumentationen för QC"></a>

.. |Module_make_masks| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/qt/screens/make_masks/index.html"><img src="../../readme_assets/cells/workflow/apps/make_masks.png" width="14.14%" align="middle" alt="Öppna API-dokumentationen för Make Masks"></a>

.. |Module_align| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/align/index.html"><img src="../../readme_assets/cells/workflow/apps/align.png" width="14.14%" align="middle" alt="Öppna API-dokumentationen för Align &amp; Stitch"></a>

.. |Module_umap| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/core/index.html#spacr.core.generate_image_umap"><img src="../../readme_assets/cells/workflow/apps/umap.png" width="14.14%" align="middle" alt="Öppna API-dokumentationen för Image UMAP"></a>

.. |Module_gate_editor| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/qt/screens/gate_editor/index.html"><img src="../../readme_assets/cells/workflow/apps/gate_editor.png" width="14.14%" align="middle" alt="Öppna API-dokumentationen för Gate Editor"></a>

.. |Module_graph_builder| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/qt/screens/graph_builder/index.html"><img src="../../readme_assets/cells/workflow/apps/graph_builder.png" width="14.14%" align="middle" alt="Öppna API-dokumentationen för Graph Builder"></a>

.. |Module_toxoplasma| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/qt/screens/organism_screen/index.html#spacr-qt-screens-organism-screen-toxoplasma"><img src="../../readme_assets/cells/workflow/apps/toxoplasma.png" width="14.14%" align="middle" alt="Öppna API-dokumentationen för Toxoplasma"></a>

.. spacr-workflow-end

.. raw:: html

   <div><a href="../../readme_text/sv/spacr-modules-2.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/sv/spacr-modules-2.desktop.svg"><img src="../../readme_images/sv/spacr-modules-2.svg" alt="Alla moduler med en ruta på startskärmen, i startskärmens ordning: först de sex pipelinemodulerna, sedan resten. Välj en ruta för att öppna modulens API-sida."></picture></a></div>

.. raw:: html

   <div><a href="../../readme_text/sv/other-resources.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/sv/other-resources.desktop.svg"><img src="../../readme_images/sv/other-resources.svg" alt="Övriga resurser."></picture></a></div>

.. raw:: html

   <div><a href="../../readme_text/sv/other-resources-2-1.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/sv/other-resources-2-1.desktop.svg"><img src="../../readme_images/sv/other-resources-2-1.svg" alt="Interaktiva handledningar — guidade arbetsflöden från installation till undersökning av träffar."></picture></a></div>

.. raw:: html

   <div><a href="../../readme_text/sv/other-resources-2-2.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/sv/other-resources-2-2.desktop.svg"><img src="../../readme_images/sv/other-resources-2-2.svg" alt="Snabbstart för Python-API:t — kör och validera bearbetningsflöden från skript, anteckningsböcker eller ett kluster."></picture></a></div>

.. raw:: html

   <div><a href="../../readme_text/sv/other-resources-2-3.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/sv/other-resources-2-3.desktop.svg"><img src="../../readme_images/sv/other-resources-2-3.svg" alt="Funktionsguide — funktioner, mognadsgrad och valfria integrationer."></picture></a></div>

.. raw:: html

   <div><a href="../../readme_text/sv/other-resources-2-4.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/sv/other-resources-2-4.desktop.svg"><img src="../../readme_images/sv/other-resources-2-4.svg" alt="Utvald API-referens — stödda ingångspunkter ordnade efter uppgift, med den fullständiga modulreferensen en nivå djupare."></picture></a></div>

.. raw:: html

   <div><a href="../../readme_text/sv/other-resources-2-5.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/sv/other-resources-2-5.desktop.svg"><img src="../../readme_images/sv/other-resources-2-5.svg" alt="Språk- och översättningsguide — gränssnittsspråk, kontextuell hjälp och policy för vetenskapliga resultat."></picture></a></div>

.. raw:: html

   <div><a href="../../readme_text/sv/language-translation.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/sv/language-translation.desktop.svg"><img src="../../readme_images/sv/language-translation.svg" alt="Språk och översättning. Gränssnittet stöder tio språk i navigering och inställningar. AI- och LIVE-kontroller, modulbeskrivningar och granskad kontexthjälp översätts också. Byt språk under spaCR → Inställningar → Språk utan att starta om. Loggar, sökvägar, databasvärden och mätningar översätts aldrig; vetenskapliga utdata förblir på kanonisk engelska. Se policyn för kontexthjälp."></picture></a></div>

.. raw:: html

   <div><a href="../../readme_text/sv/animated-setting-guidance.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/sv/animated-setting-guidance.desktop.svg"><img src="../../readme_images/sv/animated-setting-guidance.svg" alt="Animerad hjälp för inställningar. Inställningar med en visuell förklaring har kontrollen Animation i verktygstipset. Bläddra i galleriet med inställningsanimationer eller registret över inställningsanimationer."></picture></a></div>

.. raw:: html

   <div><a href="https://einarolafsson.github.io/spacr/setting_animations.html"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/sv/animated-setting-guidance-link-1.desktop.svg"><img src="../../readme_images/sv/animated-setting-guidance-link-1.svg" alt="galleriet med inställningsanimationer"></picture></a><a href="https://einarolafsson.github.io/spacr/api/spacr/setting_animations/index.html"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/sv/animated-setting-guidance-link-2.desktop.svg"><img src="../../readme_images/sv/animated-setting-guidance-link-2.svg" alt="registret över inställningsanimationer"></picture></a></div>

.. raw:: html

   <div><a href="../../readme_text/sv/data.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/sv/data.desktop.svg"><img src="../../readme_images/sv/data.svg" alt="Data. Referensdatauppsättningar."></picture></a></div>

| |DataBioStudies|\ |DataHuggingFace|\ |DataNCBI|\ |DataSpaCRPower|\ |DataBioRxiv|

.. |DataBioStudies| raw:: html

   <a href="https://doi.org/10.6019/S-BIAD2135"><img src="../../readme_assets/cells/databanks/biostudies_button.png" width="14.14%" align="middle" alt="Öppna mikroskopidatamängden i BioStudies"></a>

.. |DataHuggingFace| raw:: html

   <a href="https://huggingface.co/datasets/einarolafsson/toxo_mito"><img src="../../readme_assets/cells/databanks/huggingface_button.png" width="14.14%" align="middle" alt="Öppna testdatamängden på Hugging Face"></a>

.. |DataNCBI| raw:: html

   <a href="https://www.ncbi.nlm.nih.gov/bioproject/?term=PRJNA1261935"><img src="../../readme_assets/cells/databanks/ncbi_button.png" width="14.14%" align="middle" alt="Öppna sekvenseringsdatamängden hos NCBI"></a>

.. |DataSpaCRPower| raw:: html

   <a href="https://github.com/maomlab/spaCRPower"><img src="../../readme_assets/cells/databanks/spacrpower_button.png" width="14.14%" align="middle" alt="Öppna spaCRPower"></a>

.. |DataBioRxiv| raw:: html

   <a href="https://www.biorxiv.org/content/10.64898/2026.07.08.737057v1"><img src="../../readme_assets/cells/databanks/biorxiv_button.png" width="14.14%" align="middle" alt="Öppna bioRxiv-förhandsversionen"></a>

.. raw:: html

   <div><a href="../../readme_text/sv/model-zoo.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/sv/model-zoo.desktop.svg"><img src="../../readme_images/sv/model-zoo.svg" alt="Modellbibliotek. spaCR innehåller en katalog över tränade modeller och hämtar dem vid behov. Öppna Model Zoo från startskärmen för att bläddra bland och installera modellerna, eller ange en modellnyckel i en inställningsfil, till exempel pathogen_model: toxoplasma_pv_v1. Modellen laddas då ned och verifieras med en kontrollsumma första gången den behövs. Varje publicerad post har en SHA-256-kontrollsumma. En post utan kontrollsumma nekas, eftersom en ofullständig eller utbytt modellfil annars inte kan skiljas från den riktiga."></picture></a></div>

.. spacr-model-zoo-begin

.. raw:: html

   <div><a href="../../models/README.md"><picture><source media="(min-width: 1280px)" srcset="../../models/cards/heading.svg"><img src="../../models/cards/heading.phone.svg" alt="Default models · held-out scores"></picture></a></div>
   <div><a href="../../models/toxoplasma_pv_v4.md"><picture><source media="(min-width: 1280px)" srcset="../../models/cards/toxoplasma_pv_v4.svg"><img src="../../models/cards/toxoplasma_pv_v4.phone.svg" alt="Score card of Parasite vacuoles v4 (toxoplasma_pv_v4, default): F1 0.761 on 209 held-out fields against 0.377 for stock cpsam_v2. F1 0.761 against stock cpsam_v2&#x27;s 0.377 on the 209 strict hold-out fields at IoU 0.5; AJI 0.582 against 0.040."></picture></a><a href="../../models/toxoplasma_plaque_v6.md"><picture><source media="(min-width: 1280px)" srcset="../../models/cards/toxoplasma_plaque_v6.svg"><img src="../../models/cards/toxoplasma_plaque_v6.phone.svg" alt="Score card of Plaques v6 (toxoplasma_plaque_v6, default): F1 0.768 on 160 held-out fields against 0.294 for stock cpsam_v2. F1 0.768 against stock cpsam_v2&#x27;s 0.294 on the 160 strict hold-out fields at IoU 0.5; AJI 0.752 against 0.249."></picture></a><a href="../../models/toxoplasma_well_detector_v3.md"><picture><source media="(min-width: 1280px)" srcset="../../models/cards/toxoplasma_well_detector_v3.svg"><img src="../../models/cards/toxoplasma_well_detector_v3.phone.svg" alt="Score card of Plaque wells v3 (toxoplasma_well_detector_v3, default): mAP50 0.994 on 73 held-out images. Stock was not evaluated in this run; see the named incumbent comparison on the model card."></picture></a><a href="../../models/toxoplasma_from_cellmask_v1.md"><picture><source media="(min-width: 1280px)" srcset="../../models/cards/toxoplasma_from_cellmask_v1.svg"><img src="../../models/cards/toxoplasma_from_cellmask_v1.phone.svg" alt="Score card of Vacuoles from cell mask v1 (toxoplasma_from_cellmask_v1, default): F1 0.606 on 463 held-out fields against 0.021 for stock cpsam_v2. F1 0.606 against 0.021 for stock cpsam_v2 on 463 well-grouped held-out fields, at IoU 0.5."></picture></a><a href="../../models/nuclei_from_cellmask_v1.md"><picture><source media="(min-width: 1280px)" srcset="../../models/cards/nuclei_from_cellmask_v1.svg"><img src="../../models/cards/nuclei_from_cellmask_v1.phone.svg" alt="Score card of Nuclei from cell mask v1 (nuclei_from_cellmask_v1, default): F1 0.888 on 453 held-out fields against 0.201 for stock cpsam_v2. F1 0.888 against 0.201 for stock cpsam_v2 on 453 well-grouped held-out fields, at IoU 0.5."></picture></a><a href="../../models/cell_from_hoechst_v1.md"><picture><source media="(min-width: 1280px)" srcset="../../models/cards/cell_from_hoechst_v1.svg"><img src="../../models/cards/cell_from_hoechst_v1.phone.svg" alt="Score card of Cells from Hoechst v1 (cell_from_hoechst_v1, default): F1 0.870 on 451 held-out fields against 0.301 for stock cpsam_v2. F1 0.870 against stock cpsam_v2&#x27;s 0.301 on 451 held-out fields at IoU 0.5 -- a delta of 0.569."></picture></a><a href="../../models/toxoplasma_from_hoechst_v1.md"><picture><source media="(min-width: 1280px)" srcset="../../models/cards/toxoplasma_from_hoechst_v1.svg"><img src="../../models/cards/toxoplasma_from_hoechst_v1.phone.svg" alt="Score card of Vacuoles from Hoechst v1 (toxoplasma_from_hoechst_v1, default): F1 0.569 on 463 held-out fields against 0.002 for stock cpsam_v2. F1 0.569 against 0.002 for stock cpsam_v2 on 463 well-grouped held-out fields, at IoU 0.5."></picture></a></div>

.. raw:: html

   <div><a href="../../readme_text/sv/generated-model-zoo-1.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/sv/generated-model-zoo-1.desktop.svg"><img src="../../readme_images/sv/generated-model-zoo-1.svg" alt="Äldre modeller."></picture></a></div>

.. raw:: html

   <div><a href="../../models/toxoplasma_pv_v3.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/sv/generated-model-zoo-1-link-1.desktop.svg"><img src="../../readme_images/sv/generated-model-zoo-1-link-1.svg" alt="toxoplasma_pv_v3"></picture></a><a href="../../models/toxoplasma_pv_v2.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/sv/generated-model-zoo-1-link-2.desktop.svg"><img src="../../readme_images/sv/generated-model-zoo-1-link-2.svg" alt="toxoplasma_pv_v2"></picture></a><a href="../../models/toxoplasma_pv_v1.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/sv/generated-model-zoo-1-link-3.desktop.svg"><img src="../../readme_images/sv/generated-model-zoo-1-link-3.svg" alt="toxoplasma_pv_v1"></picture></a><a href="../../models/toxoplasma_plaque_v5.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/sv/generated-model-zoo-1-link-4.desktop.svg"><img src="../../readme_images/sv/generated-model-zoo-1-link-4.svg" alt="toxoplasma_plaque_v5"></picture></a><a href="../../models/toxoplasma_plaque_v4.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/sv/generated-model-zoo-1-link-5.desktop.svg"><img src="../../readme_images/sv/generated-model-zoo-1-link-5.svg" alt="toxoplasma_plaque_v4"></picture></a><a href="../../models/toxoplasma_plaque_v2.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/sv/generated-model-zoo-1-link-6.desktop.svg"><img src="../../readme_images/sv/generated-model-zoo-1-link-6.svg" alt="toxoplasma_plaque_v2"></picture></a><a href="../../models/toxoplasma_plaque_v1.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/sv/generated-model-zoo-1-link-7.desktop.svg"><img src="../../readme_images/sv/generated-model-zoo-1-link-7.svg" alt="toxoplasma_plaque_v1"></picture></a><a href="../../models/toxoplasma_well_detector_v2.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/sv/generated-model-zoo-1-link-8.desktop.svg"><img src="../../readme_images/sv/generated-model-zoo-1-link-8.svg" alt="toxoplasma_well_detector_v2"></picture></a><a href="../../models/toxoplasma_well_detector_v1.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/sv/generated-model-zoo-1-link-9.desktop.svg"><img src="../../readme_images/sv/generated-model-zoo-1-link-9.svg" alt="toxoplasma_well_detector_v1"></picture></a></div>

.. raw:: html

   <div><a href="../../readme_text/sv/generated-model-zoo-2.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/sv/generated-model-zoo-2.desktop.svg"><img src="../../readme_images/sv/generated-model-zoo-2.svg" alt="Alternativ."></picture></a></div>

.. raw:: html

   <div><a href="../../models/toxoplasma_pv_v5.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/sv/generated-model-zoo-2-link-1.desktop.svg"><img src="../../readme_images/sv/generated-model-zoo-2-link-1.svg" alt="toxoplasma_pv_v5"></picture></a><a href="../../models/toxoplasma_plaque_v3.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/sv/generated-model-zoo-2-link-2.desktop.svg"><img src="../../readme_images/sv/generated-model-zoo-2-link-2.svg" alt="toxoplasma_plaque_v3"></picture></a><a href="../../models/live_cell_v1.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/sv/generated-model-zoo-2-link-3.desktop.svg"><img src="../../readme_images/sv/generated-model-zoo-2-link-3.svg" alt="live_cell_v1"></picture></a></div>

.. raw:: html

   <div><a href="../../readme_text/sv/generated-model-zoo-3.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/sv/generated-model-zoo-3.desktop.svg"><img src="../../readme_images/sv/generated-model-zoo-3.svg" alt="Maskgenerator"></picture></a></div>

.. spacr-model-zoo-end

.. raw:: html

   <div><a href="../../readme_text/sv/diagnosing-performance.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/sv/diagnosing-performance.desktop.svg"><img src="../../readme_images/sv/diagnosing-performance.svg" alt="Prestandadiagnostik. Skapa en maskinvarurapport och bifoga den till ett prestandarelaterat ärende:"></picture></a></div>

.. code-block:: bash

   python tools/spacr_hardware_report.py

.. raw:: html

   <div><a href="../../readme_text/sv/diagnosing-performance-2.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/sv/diagnosing-performance-2.desktop.svg"><img src="../../readme_images/sv/diagnosing-performance-2.svg" alt="Sparar i ~/.spacr/reports och skriver ut sökvägen. --quick hoppar över de längre prestandatesterna; --out PATH anger platsen. Läser inga projektdata. Mäter tiden för importer, numeriska bibliotek, fönsterkonstruktion och animering, och rapporterar x86_64-emulering på Apple Silicon samt NumPys BLAS."></picture></a></div>

.. raw:: html

   <div><a href="../../readme_text/sv/command-line-reference.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/sv/command-line-reference.desktop.svg"><img src="../../readme_images/sv/command-line-reference.svg" alt="Kommandoradsreferens. Varje kommando nedan installeras av pip install spacr. Alla av dem accepterar --help."></picture></a></div>

.. raw:: html

   <div><a href="../../readme_text/sv/launching-the-application.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/sv/launching-the-application.desktop.svg"><img src="../../readme_images/sv/launching-the-application.svg" alt="Starta programmet."></picture></a></div>

.. code-block:: bash

   spacr              # the desktop application
   spacr-nav
   spacr-navigator
   spacr-tutorial     # the interactive tutorial library
   spacr-server       # no first-run setup screen, for unattended launches

.. raw:: html

   <div><a href="../../readme_text/sv/when-spacr-will-not-start.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/sv/when-spacr-will-not-start.desktop.svg"><img src="../../readme_images/sv/when-spacr-will-not-start.svg" alt="När spaCR inte startar."></picture></a></div>

.. code-block:: bash

   spacr-doctor       # diagnose the installation and say how to fix it
   safespacr          # the least spaCR that can still change a setting

.. raw:: html

   <div><a href="../../readme_text/sv/when-spacr-will-not-start-2.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/sv/when-spacr-will-not-start-2.desktop.svg"><img src="../../readme_images/sv/when-spacr-will-not-start-2.svg" alt="spacr-doctor skriver ut en rad per kontroll, med ett kommando att köra för varje fel. Det rapporterar också vilken spacr som finns på sökvägen. safespacr läser varje inställning som dess standardvärde och stänger av bakgrunden, animationerna, utförlig loggning och förladdning. Använd det när en sparad inställning hindrar programmet från att starta. Inställningar som sparas i säkert läge skrivs som vanligt och finns kvar."></picture></a></div>

.. raw:: html

   <div><a href="../../readme_text/sv/running-modules-headlessly.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/sv/running-modules-headlessly.desktop.svg"><img src="../../readme_images/sv/running-modules-headlessly.svg" alt="Köra moduler utan grafiskt gränssnitt. Utan Qt eller bildskärm — för kluster, servrar och CI."></picture></a></div>

.. code-block:: bash

   spacr-run --list                              # modules with a headless entry
   spacr-run --describe MODULE                   # what a module consumes and produces
   spacr-run validate --module MODULE \
       --settings settings.csv                   # check settings before spending the run
   spacr-run MODULE --settings settings.csv      # execute
   spacr-remote --help                           # submit and monitor SSH, Slurm or cloud jobs

.. raw:: html

   <div><a href="../../readme_text/sv/running-modules-headlessly-2.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/sv/running-modules-headlessly-2.desktop.svg"><img src="../../readme_images/sv/running-modules-headlessly-2.svg" alt="validate läser samma inställningar som körningen skulle läsa och listar varje problem, till exempel en saknad inställning eller en sökväg som inte finns, tillsammans med en åtgärd."></picture></a></div>

.. raw:: html

   <div><a href="../../readme_text/sv/inspecting-a-run.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/sv/inspecting-a-run.desktop.svg"><img src="../../readme_images/sv/inspecting-a-run.svg" alt="Inspektera en körning. Varje körning journalförs i ~/.spacr/runs med inställningar, hashvärden för indata, utdata, varningar, versioner och slumpfrön."></picture></a></div>

.. code-block:: bash

   spacr-repro RUN_DIR        # replay a recorded run from its journal
   spacr-workspace RUN_DIR    # what that run had open: databases, montages, views

.. raw:: html

   <div><a href="../../readme_text/sv/auditing-data-and-installation.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/sv/auditing-data-and-installation.desktop.svg"><img src="../../readme_images/sv/auditing-data-and-installation.svg" alt="Granskning av data och installation."></picture></a></div>

.. code-block:: bash

   spacr-db-audit DB      # SQLite health, integrity, locking, reader/writer probe
   spacr-leakage          # classifier train/test leakage audit
   spacr-plugins          # installed plugin registry and failure diagnostics

.. raw:: html

   <div><a href="../../readme_text/sv/environment.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/sv/environment.desktop.svg"><img src="../../readme_images/sv/environment.svg" alt="Miljö."></picture></a></div>

.. code-block:: bash

   SPACR_LOG_LEVEL=DEBUG spacr      # verbose logging for one launch

.. raw:: html

   <div><a href="../../readme_text/sv/environment-2.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/sv/environment-2.desktop.svg"><img src="../../readme_images/sv/environment-2.svg" alt="Roterande loggar skrivs till ~/.spacr/logs/spacr.log. Bifoga filen till en felrapport."></picture></a></div>

.. raw:: html

   <div><a href="../../readme_text/sv/contributing-and-support.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/sv/contributing-and-support.desktop.svg"><img src="../../readme_images/sv/contributing-and-support.svg" alt="Bidrag och support. Skicka felrapporter och avgränsade funktionsförslag via GitHub-ärenden. Ange spaCR-version, operativsystem, Python-version, modulinställningar och relevant loggutdrag när du rapporterar ett fel. spacr-doctor samlar in det mesta av denna information; bifoga maskinvarurapporten vid prestandaproblem."></picture></a></div>

.. raw:: html

   <div><a href="../../readme_text/sv/licensing.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/sv/licensing.desktop.svg"><img src="../../readme_images/sv/licensing.svg" alt="Licens. Nytt originalmaterial i spaCR omfattas av BSD 3-Clause with additional terms. Kommersiell användning kräver ett separat skriftligt tillstånd; kontakta olafsson@med.umich.com. Tidigare BSD-licensrättigheter och tredjepartslicenser gäller fortfarande; se LICENSE för omfattning och meddelanden."></picture></a></div>

.. raw:: html

   <div><a href="https://github.com/EinarOlafsson/spacr/blob/nightly/LICENSE"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/sv/licensing-link-1.desktop.svg"><img src="../../readme_images/sv/licensing-link-1.svg" alt="BSD 3-Clause with additional terms"></picture></a><a href="mailto:olafsson@med.umich.com"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/sv/licensing-link-2.desktop.svg"><img src="../../readme_images/sv/licensing-link-2.svg" alt="olafsson@med.umich.com"></picture></a></div>

.. raw:: html

   <div><a href="../../readme_text/sv/licensing-2.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/sv/licensing-2.desktop.svg"><img src="../../readme_images/sv/licensing-2.svg" alt="Om spaCR bidrog till publicerade verk uppskattas en hänvisning och är inte ett villkor för licensen – se Citera spaCR nedan."></picture></a></div>

.. raw:: html

   <div><a href="../../readme_text/sv/tutorials.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/sv/tutorials.desktop.svg"><img src="../../readme_images/sv/tutorials.svg" alt="Handledningar. Det interaktiva biblioteket med spaCR-handledningar visar installation och användning av moduler steg för steg. Tillgänglig berättarröst och språk anges för varje lektion."></picture></a></div>

.. raw:: html

   <div><a href="../../readme_text/sv/citing-spacr.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/sv/citing-spacr.desktop.svg"><img src="../../readme_images/sv/citing-spacr.svg" alt="Citera spaCR. Om spaCR bidrar till din forskning, citera: Olafsson EB, et al. En poolad bildbaserad CRISPR screening identifierar EAF1 som en T. gondii modulator för ESCRT subversion."></picture></a></div>

.. raw:: html

   <div><a href="https://www.biorxiv.org/content/10.64898/2026.07.08.737057v1"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/sv/citing-spacr-2-1.desktop.svg"><img src="../../readme_images/sv/citing-spacr-2-1.svg" alt="BioRxiv preprint"></picture></a><a href="https://doi.org/10.5281/zenodo.21343316"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/sv/citing-spacr-2-2.desktop.svg"><img src="../../readme_images/sv/citing-spacr-2-2.svg" alt="programvaruarkiv"></picture></a></div>

.. raw:: html

   <div><a href="../../readme_text/sv/other-work-citing-spacr.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/sv/other-work-citing-spacr.desktop.svg"><img src="../../readme_images/sv/other-work-citing-spacr.svg" alt="Andra arbeten med hänvisning till spaCR."></picture></a></div>

.. spacr-citing-papers-begin

.. raw:: html

   <div><a href="../../readme_text/sv/other-work-citing-spacr-2-1.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/sv/other-work-citing-spacr-2-1.desktop.svg"><img src="../../readme_images/sv/other-work-citing-spacr-2-1.svg" alt="Metabol anpassningsförmåga och näringsavlagring i Toxoplasma gondii: insikter från intag väg-deficen mutanter."></picture></a></div>

.. raw:: html

   <div><a href="../../readme_text/sv/other-work-citing-spacr-2-2.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/sv/other-work-citing-spacr-2-2.desktop.svg"><img src="../../readme_images/sv/other-work-citing-spacr-2-2.svg" alt="IRE1α främjar fagsomalt kalciumflöde för att öka makrofag fungicid aktivitet."></picture></a></div>

.. raw:: html

   <div><a href="../../readme_text/sv/other-work-citing-spacr-2-3.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/sv/other-work-citing-spacr-2-3.desktop.svg"><img src="../../readme_images/sv/other-work-citing-spacr-2-3.svg" alt="Toxoplasma GRA8 engagerar värd ESCRT tillbehör protein ALG-2 och är nödvändig för parasit metabolisk integritet."></picture></a></div>

.. raw:: html

   <div><a href="../../readme_text/sv/other-work-citing-spacr-2-4.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/sv/other-work-citing-spacr-2-4.desktop.svg"><img src="../../readme_images/sv/other-work-citing-spacr-2-4.svg" alt="Hydrogen sulfide dynamically upregulates copper uptake and localization."></picture></a></div>

.. raw:: html

   <div><a href="../../readme_text/sv/other-work-citing-spacr-2-5.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/sv/other-work-citing-spacr-2-5.desktop.svg"><img src="../../readme_images/sv/other-work-citing-spacr-2-5.svg" alt="spaCR: Spatial fenotypanalys av CRISPR-Cas9-screeningar (preprint version 1)."></picture></a></div>

.. spacr-citing-papers-end

.. raw:: html

   <div><a href="../../readme_text/sv/acknowledgments.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/sv/acknowledgments.desktop.svg"><img src="../../readme_images/sv/acknowledgments.svg" alt="Tack. spaCR bygger på öppen vetenskaplig programvara, bland annat NumPy, pandas, scikit-image, scikit-learn, Cellpose, PyTorch och Qt."></picture></a></div>
