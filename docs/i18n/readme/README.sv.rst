|Website| |Platforms| |Python| |Qt| |Test counts| |Coverage| |Issues| |Conda| |PyPI| |Docker| |PyPI Downloads| |Docs| |Tutorials| |Preprint| |DOI| |Cite| |License| |PyPI rank|

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

.. |Coverage| image:: https://img.shields.io/codecov/c/github/EinarOlafsson/spacr/nightly?logo=codecov&logoColor=white
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

.. |License| image:: https://img.shields.io/badge/License-BSD--3-blue?logo=opensourceinitiative&logoColor=white
   :target: https://github.com/EinarOlafsson/spacr/blob/main/LICENSE
   :alt: BSD 3-Clause-licens

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

.. image:: ../../source/_static/deck/slides/slide_01.jpg
   :alt: spaCR
   :width: 920
   :target: https://einarolafsson.github.io/spacr/_static/deck/

| |intro-1|\ |intro-2|

.. |intro-1| image:: ../../readme_images/sv/intro-1.svg
   :alt: ← Föregående
   :width: 49.5%
   :target: ../../source/_static/deck/pages/57.md

.. |intro-2| image:: ../../readme_images/sv/intro-2.svg
   :alt: Nästa →
   :width: 49.5%
   :target: ../../source/_static/deck/pages/02.md

.. spacr-language-picker-begin

.. image:: ../../readme_images/sv/generated-language-picker.svg
   :alt: Språk: Svenska
   :width: 32.8%
   :target: README.md

.. spacr-language-picker-end

.. image:: ../../readme_images/sv/spacr.svg
   :alt: spaCR. Rumslig fenotypanalys av CRISPR-screeningar. spaCR segmenterar och mäter enskilda celler i mikroskopibilder, integrerar fenotyper per objekt med sekvenseringshärledd guideförekomst och uppskattar vilka gener som är associerade med fenotypiska förändringar. Med plattbilder och FASTQ-läsningar som utgångspunkt producerar programmet mätningar per objekt, tränade klassificerare, effektskattningar per guide och gen samt en rangordnad träfflista. Segmenterings-, mät-, annoterings- och klassificeringsmodulerna körs även utan en sekvensarm. Make Masks korrigerar segmenteringsmasker och annoterar oberoende, klassmärkta rektanglar med verktyget Box för YOLO-export. Rutorna har egna etiketter och egen historik utan att ändra källbilder eller masker. Se funktionsguiden för varje verktyg. Bilder, masker och bildutsnitt är filer på disk. Mätningar, annoteringar och modellprediktioner är tabeller i projektets SQLite-databas measurements.db. Körs som ett skrivbordsprogram eller utan grafiskt gränssnitt på en arbetsstation, server eller kluster.
   :width: 99.5%
   :target: ../../source/features.rst

.. image:: ../../readme_images/sv/try-spacr.svg
   :alt: Prova spaCR.
   :width: 99.5%

.. code-block:: bash

   conda create -n spacr python=3.12 -y
   conda activate spacr
   python -m pip install spacr
   spacr

.. image:: ../../readme_images/sv/try-spacr-2.svg
   :alt: Använd Ladda testdata… i Import, Make Masks, Annotate eller en analysskärm för att hämta exempeldata. Från en terminal använder du spacr-download.
   :width: 99.5%

.. image:: ../../readme_images/sv/hardware-support.svg
   :alt: Hårdvarustöd.
   :width: 99.5%

.. spacr-hardware-begin

.. image:: ../../readme_images/sv/generated-hardware-1.svg
   :alt: NVIDIA (CUDA): Cellpose 4 GPU, Torch GPU, UMAP / clustering GPU. AMD on Linux (ROCm): Cellpose 4 GPU, Torch GPU, UMAP / clustering CPU. AMD in an Intel Mac (Metal): Cellpose 4 GPU, Torch GPU, UMAP / clustering CPU. Apple Silicon (Metal): Cellpose 4 GPU, Torch GPU, UMAP / clustering CPU. Intel Arc/Xe (XPU): Cellpose 4 GPU, Torch GPU, UMAP / clustering CPU. No GPU: Cellpose 4 CPU, Torch CPU, UMAP / clustering CPU. stöds (stabilt) implementerat (beta) endast CPU-stöd.
   :width: 99.5%

.. spacr-hardware-end

.. image:: ../../readme_images/sv/install-spacr.svg
   :alt: Installera spaCR. Välj en installationsväg: Skrivbordsprogram: ett installationsprogram för Linux, macOS eller Windows med egen Python. pip: python -m pip install spacr i en Conda-miljö. conda-forge: conda install conda-forge::spacr i en Conda-miljö. Docker: kommandoradens bearbetningsflöden i en Linux-container, från CPU-avbildningen eller avbildningen för NVIDIA-GPU:er:
   :width: 99.5%

.. code-block:: bash

   docker run --rm ghcr.io/einarolafsson/spacr:1.5.1.4 spacr-run --list
   docker run --rm --gpus all ghcr.io/einarolafsson/spacr:1.5.1.4-cuda12.4 spacr-run --list

.. image:: ../../readme_images/sv/install-spacr-2.svg
   :alt: Docker-guiden beskriver GPU-inställning, montering av data och kommandon för bearbetningsflödena.
   :width: 99.5%
   :target: ../../source/installer_guide.rst#container-images

.. image:: ../../readme_images/sv/desktop-application.svg
   :alt: Skrivbordsprogram. Installatörerna buntar ihop sina egna Python. Conda krävs inte.
   :width: 99.5%

.. spacr-installer-links-begin

|InstallerLinux| |InstallerMacOS| |InstallerWindows|

.. |InstallerWindows| image:: ../../../spacr/resources/icons/platforms/windows.png
   :width: 64
   :alt: Hämta spaCR 1.5.1.4 för Windows 10/11
   :target: https://github.com/EinarOlafsson/spacr/releases/download/v1.5.1.4/spaCR-1.5.1.4-Windows-Online-Setup.exe
.. |InstallerMacOS| image:: ../../../spacr/resources/icons/platforms/macos.png
   :width: 64
   :alt: Hämta spaCR 1.5.1.4 för macOS 11+ (Intel och Apple Silicon)
   :target: https://github.com/EinarOlafsson/spacr/releases/download/v1.5.1.4/spaCR-1.5.1.4-macOS-Universal-Online.pkg
.. |InstallerLinux| image:: ../../../spacr/resources/icons/platforms/linux.png
   :width: 64
   :alt: Hämta spaCR 1.5.1.4 för 64-bitars Linux
   :target: https://github.com/EinarOlafsson/spacr/releases/download/v1.5.1.4/spaCR-1.5.1.4-Linux-x86_64-Online.run

.. spacr-installer-links-end

.. image:: ../../readme_images/sv/desktop-application-2.svg
   :alt: Gör den hämtade filen körbar i Linux och kör den:
   :width: 99.5%

.. code-block:: bash

   chmod +x SpaCR-*-Linux-x86_64-Online.run
   ./SpaCR-*-Linux-x86_64-Online.run

.. image:: ../../readme_images/sv/desktop-application-3.svg
   :alt: På macOS, öppna .pkg. Nuvarande beta notariseras inte. Om Gatekeeper blockerar den, välj Systeminställningar → Integritet & Säkerhet → Öppna ändå. Se installationsguiden för instruktioner om uppdatering, avinstallation, offlineanvändning och felsökning, och systemkraven för rekommendationer om arbetsstationer och servrar samt tabeller över GPU-kompatibilitet.
   :width: 99.5%

| |desktop-application-3-link-1|\ |desktop-application-3-link-2|

.. |desktop-application-3-link-1| image:: ../../readme_images/sv/desktop-application-3-link-1.svg
   :alt: installationsguiden
   :width: 49.5%
   :target: ../../source/installer_guide.rst

.. |desktop-application-3-link-2| image:: ../../readme_images/sv/desktop-application-3-link-2.svg
   :alt: systemkraven
   :width: 49.5%
   :target: ../../source/system_requirements.rst

.. image:: ../../readme_images/sv/pypi-installation.svg
   :alt: Installation från PyPI. För PyPI-utgåvan installerar du spaCR med pip i en Conda-miljö. Python 3.12 ger det största urvalet av valfria vetenskapliga paket:
   :width: 99.5%

.. code-block:: bash

   conda create -n spacr python=3.12 -y
   conda activate spacr
   python -m pip install --upgrade pip
   python -m pip install spacr
   spacr

.. image:: ../../readme_images/sv/pypi-installation-2.svg
   :alt: spaCR stöder Python 3.9 through 3.14, utom Python 3.14.1, som torchvision utesluter. Linux rekommenderas för de tyngsta CUDA- och ROCm-arbetsflödena; macOS och Windows stöds också, och båda använder sina GPU:er — macOS via Metal, som täcker Apple Silicon och AMD-korten i Intel-Mac-datorer, och Windows via CUDA eller DirectML. Standardinstallationen innehåller Qt-gränssnittet för skrivbordet. På en server, ett kluster eller en CI-körmiljö kan du köra kommandoradspipelines utan att öppna det:
   :width: 99.5%

.. code-block:: bash

   python -m pip install spacr
   spacr-run --list

.. image:: ../../readme_images/sv/pypi-installation-3.svg
   :alt: Optional integrations are installed separately, for example spacr[zarr], spacr[omero], spacr[napari] and spacr[czi,nd2,lif]. See the installationsguide for the complete extras and Python-version compatibility table.
   :width: 99.5%
   :target: ../../source/installer_guide.rst

.. image:: ../../readme_images/sv/conda-forge-installation.svg
   :alt: Installation med conda-forge. Det officiella conda-forge-paketet installerar spaCR och dess skrivbordsberoenden i den aktiva miljön:
   :width: 99.5%

.. code-block:: bash

   conda create -n spacr python=3.12 -y
   conda activate spacr
   conda install conda-forge::spacr
   spacr

.. image:: ../../readme_images/sv/install-from-source.svg
   :alt: Installation från källkod. Klona kodförrådet och installera det i redigerbart läge, så att din arbetskopia är det installerade paketet och ändringar börjar gälla utan ominstallation:
   :width: 99.5%

.. code-block:: bash

   git clone https://github.com/EinarOlafsson/spacr.git
   cd spacr
   conda create -n spacr python=3.12 -y
   conda activate spacr
   pip install -e .
   spacr

.. image:: ../../readme_images/sv/install-from-source-2.svg
   :alt: Detta klonar main, standardgrenen, som innehåller den senaste utgåvan. Utvecklingen sker på nightly; lägg till --branch nightly för att klona den i stället. För en specifik utgåva:
   :width: 99.5%

.. code-block:: bash

   git clone --branch v1.5.0.5 https://github.com/EinarOlafsson/spacr.git

.. image:: ../../readme_images/sv/install-from-source-3.svg
   :alt: För att hämta senare ändringar kör du inifrån klonen:
   :width: 99.5%

.. code-block:: bash

   git pull
   pip install -e .

.. image:: ../../readme_images/sv/install-from-source-4.svg
   :alt: Installera om när beroenden eller kommandostarter ändras. Ändringar i Python-koden gäller direkt; spacr-doctor visar vilken installation som används.
   :width: 99.5%

.. image:: ../../readme_images/sv/install-from-source-light.svg
   :alt: Installation från källkod (lättviktig). Bidragsgivare behöver versionshistoriken; välj ett alternativ nedan om du bara vill köra spaCR. Mätningarna gäller nightly vid 05302fd5c den 2026-10-07 och gjordes med packaging/measure_clone_forms.sh:
   :width: 99.5%

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

.. image:: ../../readme_images/sv/install-from-source-light-2.svg
   :alt: Den fullständiga nightly-klonen laddade ner 9.25 GiB. Att lägga till --filter=blob:none till den grunda klonen minskar inte utcheckningen: dess Git-objektlager är fortfarande 2032 MB. Git rapporterar inte lata hämtningar av objekt, så ingen total nedladdning uppmättes. De versionshanterade filerna i nightly ger en utcheckning på 3416 MB (uppmätt 2026-10-08), utan Git-historiken. Nedladdningens storlek och tid varierar med grenen.
   :width: 99.5%

.. image:: ../../readme_images/sv/command-line-entry-points.svg
   :alt: Kommandoradskommandon.
   :width: 99.5%

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

.. image:: ../../readme_images/sv/command-line-entry-points-2.svg
   :alt: spacr-run --list listar moduler med kommandoradsposter för körning utan grafiskt gränssnitt. GUI-bundna moduler för annotering, kurering, jämförelse och utforskning utelämnas.
   :width: 99.5%

.. image:: ../../readme_images/sv/core-workflow.svg
   :alt: Kärnarbetsflöde. Det primära arbetsflödet består av sex moduler: Mask segmenterar celler, cellkärnor, patogener och organeller med Cellpose. Measure skriver morfologiska, intensitets-, textur-, rumsliga och kolokaliseringsmått samt objektutsnitt till SQLite. Annotate märker objektutsnitt i ett tangentbordsstyrt rutnät och stöder köer för aktiv inlärning. Classify tränar bild- eller mätningsbaserade modeller och sparar prestandan på undanhållna data med varje kontrollpunkt. Map Barcodes kopplar FASTQ-läsningar till brunnar och gRNA:er och rapporterar QC för förekomst, kollisioner och täckning. Regression skattar effekter för guider, gener, betingelser och kontroller med modellfamiljer för kontinuerliga data, andelar och antal.
   :width: 99.5%

.. image:: ../../readme_images/sv/spacr-modules.svg
   :alt: spaCR-moduler.
   :width: 99.5%

.. spacr-workflow-begin

.. image:: ../../readme_images/sv/generated-workflow-1.svg
   :alt: Kärna. Core sequence from microscopy images through segmentation, measurements, annotations, classification, barcode mapping and regression.
   :width: 99.5%

| |Module_mask|\ |Module_measure|\ |Module_annotate|\ |Module_classify_merged|\ |Module_map_barcodes|\ |Module_regression|

.. image:: ../../readme_images/sv/generated-workflow-2.svg
   :alt: Data. Import images and tables into spaCR projects and execute reproducible multi-plate workflows.
   :width: 99.5%

| |Module_foreign|\ |Module_embeddings|\ |Module_run_compare|\ |Module_experiment_design|\ |Module_power|\ |Module_dose_response|
| |Module_qc_dashboard|

.. image:: ../../readme_images/sv/generated-workflow-3.svg
   :alt: Verktyg. Point these at a project: edit masks by hand, stitch tiles, read an embedding, draw a gate, build a plot, check quality.
   :width: 99.5%

| |Module_make_masks|\ |Module_align|\ |Module_umap|\ |Module_gate_editor|\ |Module_graph_builder|

.. image:: ../../readme_images/sv/generated-workflow-4.svg
   :alt: Organism. Organismspecifik bildanalys och kvantitativa resultat från biologiska analyser.
   :width: 99.5%

| |Module_toxoplasma|

.. |Module_mask| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/core/index.html#spacr.core.preprocess_generate_masks"><img src="../../../spacr/resources/icons/workflow/mask.png" width="16.0%" align="middle" alt="Öppna API-dokumentationen för Mask"></a>

.. |Module_measure| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/measure/index.html"><img src="../../../spacr/resources/icons/workflow/measure.png" width="16.0%" align="middle" alt="Öppna API-dokumentationen för Measure"></a>

.. |Module_annotate| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/qt/screens/annotate/index.html"><img src="../../../spacr/resources/icons/workflow/annotate.png" width="16.0%" align="middle" alt="Öppna API-dokumentationen för Annotate"></a>

.. |Module_classify_merged| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/classify/index.html"><img src="../../../spacr/resources/icons/workflow/classify_merged.png" width="16.0%" align="middle" alt="Öppna API-dokumentationen för Classify"></a>

.. |Module_map_barcodes| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/sequencing/index.html"><img src="../../../spacr/resources/icons/workflow/map_barcodes.png" width="16.0%" align="middle" alt="Öppna API-dokumentationen för Map Barcodes"></a>

.. |Module_regression| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/ml/index.html"><img src="../../../spacr/resources/icons/workflow/regression.png" width="16.0%" align="middle" alt="Öppna API-dokumentationen för Regression"></a>

.. |Module_foreign| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/foreign/index.html"><img src="../../../spacr/resources/icons/workflow/apps/foreign.png" width="16.0%" align="middle" alt="Öppna API-dokumentationen för Import"></a>

.. |Module_embeddings| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/qt/screens/embeddings/index.html"><img src="../../../spacr/resources/icons/workflow/apps/embeddings.png" width="16.0%" align="middle" alt="Öppna API-dokumentationen för Embeddings"></a>

.. |Module_run_compare| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/qt/screens/run_compare/index.html"><img src="../../../spacr/resources/icons/workflow/apps/run_compare.png" width="16.0%" align="middle" alt="Öppna API-dokumentationen för Run Compare"></a>

.. |Module_experiment_design| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/qt/screens/experiment_design/index.html"><img src="../../../spacr/resources/icons/workflow/apps/experiment_design.png" width="16.0%" align="middle" alt="Öppna API-dokumentationen för Experiment Design"></a>

.. |Module_power| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/qt/screens/power/index.html"><img src="../../../spacr/resources/icons/workflow/apps/power.png" width="16.0%" align="middle" alt="Öppna API-dokumentationen för Power / Design"></a>

.. |Module_dose_response| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/qt/screens/dose_response/index.html"><img src="../../../spacr/resources/icons/workflow/apps/dose_response.png" width="16.0%" align="middle" alt="Öppna API-dokumentationen för Dose–Response"></a>

.. |Module_qc_dashboard| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/qt/screens/qc_dashboard/index.html"><img src="../../../spacr/resources/icons/workflow/apps/qc_dashboard.png" width="16.0%" align="middle" alt="Öppna API-dokumentationen för QC"></a>

.. |Module_make_masks| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/qt/screens/make_masks/index.html"><img src="../../../spacr/resources/icons/workflow/apps/make_masks.png" width="16.0%" align="middle" alt="Öppna API-dokumentationen för Make Masks"></a>

.. |Module_align| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/align/index.html"><img src="../../../spacr/resources/icons/workflow/apps/align.png" width="16.0%" align="middle" alt="Öppna API-dokumentationen för Align &amp; Stitch"></a>

.. |Module_umap| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/core/index.html#spacr.core.generate_image_umap"><img src="../../../spacr/resources/icons/workflow/apps/umap.png" width="16.0%" align="middle" alt="Öppna API-dokumentationen för Image UMAP"></a>

.. |Module_gate_editor| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/qt/screens/gate_editor/index.html"><img src="../../../spacr/resources/icons/workflow/apps/gate_editor.png" width="16.0%" align="middle" alt="Öppna API-dokumentationen för Gate Editor"></a>

.. |Module_graph_builder| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/qt/screens/graph_builder/index.html"><img src="../../../spacr/resources/icons/workflow/apps/graph_builder.png" width="16.0%" align="middle" alt="Öppna API-dokumentationen för Graph Builder"></a>

.. |Module_toxoplasma| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/qt/screens/organism_screen/index.html#spacr-qt-screens-organism-screen-toxoplasma"><img src="../../../spacr/resources/icons/workflow/apps/toxoplasma.png" width="16.0%" align="middle" alt="Öppna API-dokumentationen för Toxoplasma"></a>

.. spacr-workflow-end

.. image:: ../../readme_images/sv/spacr-modules-2.svg
   :alt: Alla moduler med en ruta på startskärmen, i startskärmens ordning: först de sex pipelinemodulerna, sedan resten. Välj en ruta för att öppna modulens API-sida.
   :width: 99.5%

.. image:: ../../readme_images/sv/other-resources.svg
   :alt: Övriga resurser.
   :width: 99.5%

.. image:: ../../readme_images/sv/other-resources-2-1.svg
   :alt: Interaktiva handledningar — guidade arbetsflöden från installation till undersökning av träffar.
   :width: 99.5%
   :target: https://einarolafsson.github.io/spacr/tutorials/

.. image:: ../../readme_images/sv/other-resources-2-2.svg
   :alt: Snabbstart Python API – kör och validera arbetsflöden från skript, anteckningsböcker eller ett kluster.
   :width: 99.5%
   :target: ../../source/python_api.rst

.. image:: ../../readme_images/sv/other-resources-2-3.svg
   :alt: Handbok för funktioner – kapacitet, mognad och valfria integrationer.
   :width: 99.5%
   :target: ../../source/features.rst

.. image:: ../../readme_images/sv/other-resources-2-4.svg
   :alt: Kurerad API referens – understödda ingångspunkter för uppgift, med den fullständiga modulreferensen en nivå djupare.
   :width: 99.5%
   :target: https://einarolafsson.github.io/spacr/api/index.html

.. image:: ../../readme_images/sv/other-resources-2-5.svg
   :alt: Språk- och översättningsguide – gränssnittsspråk, kontextuell hjälp och policy för vetenskaplig output.
   :width: 99.5%
   :target: ../../source/localization.rst

.. image:: ../../readme_images/sv/language-translation.svg
   :alt: Språk och översättning. Gränssnittet stöder tio språk i navigering och inställningar. AI- och LIVE-kontroller, modulbeskrivningar och granskad kontexthjälp översätts också. Byt språk under spaCR → Inställningar → Språk utan att starta om. Loggar, sökvägar, databasvärden och mätningar översätts aldrig; vetenskapliga utdata förblir på kanonisk engelska. Se policyn för kontexthjälp.
   :width: 99.5%
   :target: ../../source/localization.rst#contextual-help

.. image:: ../../readme_images/sv/language-translation-2.svg
   :alt: De nio icke-engelska katalogerna är maskinskrivna och tekniskt recenserade snarare än lästa ända till slut av en infödd talare. De Översynens tillämpningsområde skivor som språken har haft ett mänskligt pass och varje term kvar på engelska av beslut.
   :width: 99.5%
   :target: ../REVIEW_SCOPE_2026-09-04.md

.. image:: ../../readme_images/sv/animated-setting-guidance.svg
   :alt: Animerad hjälp för inställningar. Inställningar med en visuell förklaring har kontrollen Animation i verktygstipset. Bläddra i galleriet med inställningsanimationer eller registret över inställningsanimationer.
   :width: 99.5%

| |animated-setting-guidance-link-1|\ |animated-setting-guidance-link-2|

.. |animated-setting-guidance-link-1| image:: ../../readme_images/sv/animated-setting-guidance-link-1.svg
   :alt: galleriet med inställningsanimationer
   :width: 49.5%
   :target: https://einarolafsson.github.io/spacr/setting_animations.html

.. |animated-setting-guidance-link-2| image:: ../../readme_images/sv/animated-setting-guidance-link-2.svg
   :alt: registret över inställningsanimationer
   :width: 49.5%
   :target: https://einarolafsson.github.io/spacr/api/spacr/setting_animations/index.html

.. image:: ../../readme_images/sv/data.svg
   :alt: Data. Referensdatauppsättningar.
   :width: 99.5%

|DataBioStudies| |DataHuggingFace| |DataNCBI| |DataSpaCRPower| |DataBioRxiv|

.. |DataBioStudies| image:: ../../../spacr/resources/icons/databanks/biostudies_button.png
   :width: 72
   :alt: Öppna mikroskopidatamängden i BioStudies
   :target: https://doi.org/10.6019/S-BIAD2135

.. |DataHuggingFace| image:: ../../../spacr/resources/icons/databanks/huggingface_button.png
   :width: 72
   :alt: Öppna testdatamängden på Hugging Face
   :target: https://huggingface.co/datasets/einarolafsson/toxo_mito

.. |DataNCBI| image:: ../../../spacr/resources/icons/databanks/ncbi_button.png
   :width: 72
   :alt: Öppna sekvenseringsdatamängden hos NCBI
   :target: https://www.ncbi.nlm.nih.gov/bioproject/?term=PRJNA1261935

.. |DataSpaCRPower| image:: ../../../spacr/resources/icons/databanks/spacrpower_button.png
   :width: 72
   :alt: Öppna spaCRPower
   :target: https://github.com/maomlab/spaCRPower

.. |DataBioRxiv| image:: ../../../spacr/resources/icons/databanks/biorxiv_button.png
   :width: 72
   :alt: Öppna bioRxiv-förhandsversionen
   :target: https://www.biorxiv.org/content/10.64898/2026.07.08.737057v1

.. image:: ../../readme_images/sv/model-zoo.svg
   :alt: Modellbibliotek. spaCR skickar en katalog med utbildade modeller och hämtar dem på begäran. Öppna Modellzoo från startskärmen för att bläddra och installera dem, eller namnge en nyckel i en inställningsfil -- pathogen_model: toxoplasma_pv_v1 -- och modellen laddas ner och kontrolleras första gången den behövs. Varje publicerad post innehåller en SHA-256; en post utan en nekas snarare än installeras, eftersom en trunkerad eller ersatt kontrollpunkt inte kan meddelas från den verkliga.
   :width: 99.5%

.. spacr-model-zoo-begin

| |zoo_heading|
| |zoo_card_toxoplasma_pv_v4|\ |zoo_card_toxoplasma_plaque_v4|
| |zoo_card_toxoplasma_well_detector_v3|\ |zoo_card_toxoplasma_from_cellmask_v1|
| |zoo_card_live_cell_v1|\ |zoo_card_nuclei_from_cellmask_v1|
| |zoo_card_cell_from_hoechst_v1|\ |zoo_card_toxoplasma_from_hoechst_v1|

.. |zoo_heading| image:: ../../models/cards/heading.svg
   :alt: Default models · held-out scores
   :target: ../../models/README.md
   :width: 99.5%

.. |zoo_card_toxoplasma_pv_v4| image:: ../../models/cards/toxoplasma_pv_v4.svg
   :alt: Score card of Parasite vacuoles v4 (toxoplasma_pv_v4, default): F1 0.854 on 11 held-out wells against 0.765 for stock cpsam_v2. F1 0.854 against stock cpsam_v2's 0.765 on the 11 anchor wells at IoU 0.5; AJI 0.776 against 0.505.
   :target: ../../models/toxoplasma_pv_v4.md
   :width: 49.5%

.. |zoo_card_toxoplasma_plaque_v4| image:: ../../models/cards/toxoplasma_plaque_v4.svg
   :alt: Score card of Plaques v4 (toxoplasma_plaque_v4, default): F1 0.815 on 263 held-out fields against 0.373 for stock cpsam_v2. F1 0.815 (toxoplasma_plaque_v3 0.844) on the 263 held-out fields at IoU 0.5; stock cpsam_v2 0.373 there with the Cellpose evaluation scorer.
   :target: ../../models/toxoplasma_plaque_v4.md
   :width: 49.5%

.. |zoo_card_toxoplasma_well_detector_v3| image:: ../../models/cards/toxoplasma_well_detector_v3.svg
   :alt: Score card of Plaque wells v3 (toxoplasma_well_detector_v3, default): mAP50 0.994 on 73 held-out images. Stock was not evaluated in this run; see the named incumbent comparison on the model card.
   :target: ../../models/toxoplasma_well_detector_v3.md
   :width: 49.5%

.. |zoo_card_toxoplasma_from_cellmask_v1| image:: ../../models/cards/toxoplasma_from_cellmask_v1.svg
   :alt: Score card of Vacuoles from cell mask v1 (toxoplasma_from_cellmask_v1, default): F1 0.606 on 463 held-out fields against 0.021 for stock cpsam_v2. F1 0.606 against 0.021 for stock cpsam_v2 on 463 well-grouped held-out fields, at IoU 0.5.
   :target: ../../models/toxoplasma_from_cellmask_v1.md
   :width: 49.5%

.. |zoo_card_live_cell_v1| image:: ../../models/cards/live_cell_v1.svg
   :alt: Score card of Live cells v1 (live_cell_v1, default): F1 0.694 on 2,199 held-out fields against 0.738 for stock cpsam_v2. On the datasets stock cpsam_v2 never trained on, F1 0.960 against 0.885 at IoU 0.5; over all 2,199 test fields, 0.694 against 0.738, because stock trained on LIVECell and YeaZ and wins on LIVECell.
   :target: ../../models/live_cell_v1.md
   :width: 49.5%

.. |zoo_card_nuclei_from_cellmask_v1| image:: ../../models/cards/nuclei_from_cellmask_v1.svg
   :alt: Score card of Nuclei from cell mask v1 (nuclei_from_cellmask_v1, default): F1 0.888 on 453 held-out fields against 0.201 for stock cpsam_v2. F1 0.888 against 0.201 for stock cpsam_v2 on 453 well-grouped held-out fields, at IoU 0.5.
   :target: ../../models/nuclei_from_cellmask_v1.md
   :width: 49.5%

.. |zoo_card_cell_from_hoechst_v1| image:: ../../models/cards/cell_from_hoechst_v1.svg
   :alt: Score card of Cells from Hoechst v1 (cell_from_hoechst_v1, default): F1 0.870 on 451 held-out fields against 0.301 for stock cpsam_v2. F1 0.870 against stock cpsam_v2's 0.301 on 451 held-out fields at IoU 0.5 -- a delta of 0.569.
   :target: ../../models/cell_from_hoechst_v1.md
   :width: 49.5%

.. |zoo_card_toxoplasma_from_hoechst_v1| image:: ../../models/cards/toxoplasma_from_hoechst_v1.svg
   :alt: Score card of Vacuoles from Hoechst v1 (toxoplasma_from_hoechst_v1, default): F1 0.569 on 463 held-out fields against 0.002 for stock cpsam_v2. F1 0.569 against 0.002 for stock cpsam_v2 on 463 well-grouped held-out fields, at IoU 0.5.
   :target: ../../models/toxoplasma_from_hoechst_v1.md
   :width: 49.5%

.. image:: ../../readme_images/sv/generated-model-zoo-1.svg
   :alt: Older models.
   :width: 99.5%

| |generated-model-zoo-1-link-1|\ |generated-model-zoo-1-link-2|\ |generated-model-zoo-1-link-3|
| |generated-model-zoo-1-link-4|\ |generated-model-zoo-1-link-5|\ |generated-model-zoo-1-link-6|
| |generated-model-zoo-1-link-7|

.. |generated-model-zoo-1-link-1| image:: ../../readme_images/sv/generated-model-zoo-1-link-1.svg
   :alt: toxoplasma_pv_v3
   :width: 32.8%
   :target: ../../models/toxoplasma_pv_v3.md

.. |generated-model-zoo-1-link-2| image:: ../../readme_images/sv/generated-model-zoo-1-link-2.svg
   :alt: toxoplasma_pv_v2
   :width: 32.8%
   :target: ../../models/toxoplasma_pv_v2.md

.. |generated-model-zoo-1-link-3| image:: ../../readme_images/sv/generated-model-zoo-1-link-3.svg
   :alt: toxoplasma_pv_v1
   :width: 32.8%
   :target: ../../models/toxoplasma_pv_v1.md

.. |generated-model-zoo-1-link-4| image:: ../../readme_images/sv/generated-model-zoo-1-link-4.svg
   :alt: toxoplasma_plaque_v2
   :width: 32.8%
   :target: ../../models/toxoplasma_plaque_v2.md

.. |generated-model-zoo-1-link-5| image:: ../../readme_images/sv/generated-model-zoo-1-link-5.svg
   :alt: toxoplasma_plaque_v1
   :width: 32.8%
   :target: ../../models/toxoplasma_plaque_v1.md

.. |generated-model-zoo-1-link-6| image:: ../../readme_images/sv/generated-model-zoo-1-link-6.svg
   :alt: toxoplasma_well_detector_v2
   :width: 32.8%
   :target: ../../models/toxoplasma_well_detector_v2.md

.. |generated-model-zoo-1-link-7| image:: ../../readme_images/sv/generated-model-zoo-1-link-7.svg
   :alt: toxoplasma_well_detector_v1
   :width: 32.8%
   :target: ../../models/toxoplasma_well_detector_v1.md

.. image:: ../../readme_images/sv/generated-model-zoo-2.svg
   :alt: Alternatives.
   :width: 99.5%

| |generated-model-zoo-2-link-1|\ |generated-model-zoo-2-link-2|

.. |generated-model-zoo-2-link-1| image:: ../../readme_images/sv/generated-model-zoo-2-link-1.svg
   :alt: toxoplasma_pv_v5
   :width: 32.8%
   :target: ../../models/toxoplasma_pv_v5.md

.. |generated-model-zoo-2-link-2| image:: ../../readme_images/sv/generated-model-zoo-2-link-2.svg
   :alt: toxoplasma_plaque_v3
   :width: 32.8%
   :target: ../../models/toxoplasma_plaque_v3.md

.. spacr-model-zoo-end

.. image:: ../../readme_images/sv/diagnosing-performance.svg
   :alt: Prestandadiagnostik. Skapa en maskinvarurapport och bifoga den till ett prestandarelaterat ärende:
   :width: 99.5%

.. code-block:: bash

   python tools/spacr_hardware_report.py

.. image:: ../../readme_images/sv/diagnosing-performance-2.svg
   :alt: Sparar till ~/.spacr/reports och skriver ut sökvägen. --quick hoppar över de längre riktmärkena; --out PATH anger platsen. Läser inga projektdata. Mäter tiden för importer, numeriska bibliotek, fönsterkonstruktion och animering, och rapporterar x86_64-emulering på Apple Silicon samt NumPys BLAS.
   :width: 99.5%

.. image:: ../../readme_images/sv/command-line-reference.svg
   :alt: Kommandoradsreferens. Varje kommando nedan installeras av pip install spacr. Alla av dem accepterar --help.
   :width: 99.5%

.. image:: ../../readme_images/sv/launching-the-application.svg
   :alt: Lansering av ansökan.
   :width: 99.5%

.. code-block:: bash

   spacr              # the desktop application
   spacr-tutorial     # the interactive tutorial library
   spacr-server       # no first-run setup screen, for unattended launches

.. image:: ../../readme_images/sv/when-spacr-will-not-start.svg
   :alt: När spaCR inte kommer att starta.
   :width: 99.5%

.. code-block:: bash

   spacr-doctor       # diagnose the installation and say how to fix it
   safespacr          # the least spaCR that can still change a setting

.. image:: ../../readme_images/sv/when-spacr-will-not-start-2.svg
   :alt: spacr-doctor skriver ut en rad per check, med ett kommando att köra för varje fel. Det rapporterar också vilken spacr som finns på sökvägen. safespacr läser varje inställning som dess standardvärde och stänger av bakgrunden, animationerna, utförlig loggning och förladdning. Använd det när en sparad inställning hindrar programmet från att starta. Inställningar som sparas i säkert läge skrivs som vanligt och finns kvar.
   :width: 99.5%

.. image:: ../../readme_images/sv/running-modules-headlessly.svg
   :alt: Drivmoduler utan grafiskt gränssnitt. Ingen Qt, ingen visning – för kluster, servrar och CI.
   :width: 99.5%

.. code-block:: bash

   spacr-run --list                              # modules with a headless entry
   spacr-run --describe MODULE                   # what a module consumes and produces
   spacr-run validate --module MODULE \
       --settings settings.csv                   # check settings before spending the run
   spacr-run MODULE --settings settings.csv      # execute
   spacr-remote --help                           # submit and monitor SSH, Slurm or cloud jobs

.. image:: ../../readme_images/sv/running-modules-headlessly-2.svg
   :alt: validate läser samma inställningar som körningen skulle läsa och listar varje problem, till exempel en saknad inställning eller en sökväg som inte finns, tillsammans med en åtgärd.
   :width: 99.5%

.. image:: ../../readme_images/sv/inspecting-a-run.svg
   :alt: Inspektera en körning. Varje körning är journaliserad till ~/.spacr/runs med dess inställningar, hashade ingångar, utgångar, varningar, versioner och frön.
   :width: 99.5%

.. code-block:: bash

   spacr-repro RUN_DIR        # replay a recorded run from its journal
   spacr-workspace RUN_DIR    # what that run had open: databases, montages, views

.. image:: ../../readme_images/sv/auditing-data-and-installation.svg
   :alt: Granskning av data och installation.
   :width: 99.5%

.. code-block:: bash

   spacr-db-audit DB      # SQLite health, integrity, locking, reader/writer probe
   spacr-leakage          # classifier train/test leakage audit
   spacr-plugins          # installed plugin registry and failure diagnostics

.. image:: ../../readme_images/sv/environment.svg
   :alt: Miljö.
   :width: 99.5%

.. code-block:: bash

   SPACR_LOG_LEVEL=DEBUG spacr      # verbose logging for one launch

.. image:: ../../readme_images/sv/environment-2.svg
   :alt: Roterande loggar skrivs till ~/.spacr/logs/spacr.log. Bifoga filen till en felrapport.
   :width: 99.5%

.. image:: ../../readme_images/sv/contributing-and-support.svg
   :alt: Bidrag och support. Skicka felrapporter och avgränsade funktionsförslag via GitHub-ärenden. Ange spaCR-version, operativsystem, Python-version, modulinställningar och relevant loggutdrag när du rapporterar ett fel. spacr-doctor samlar in det mesta av denna information; bifoga maskinvarurapporten vid prestandaproblem.
   :width: 99.5%
   :target: https://github.com/EinarOlafsson/spacr/issues

.. image:: ../../readme_images/sv/licensing.svg
   :alt: Licens. spaCR frisätts under BSD 3-Clause-licens. Om spaCR bidrog till publicerade verk uppskattas en hänvisning och är inte ett villkor för licensen – se Citera spaCR nedan.
   :width: 99.5%
   :target: https://github.com/EinarOlafsson/spacr/blob/main/LICENSE

.. image:: ../../readme_images/sv/tutorials.svg
   :alt: Handledningar. Det interaktiva biblioteket med spaCR-handledningar visar installation och användning av moduler steg för steg. Tillgänglig berättarröst och språk anges för varje lektion.
   :width: 99.5%
   :target: https://einarolafsson.github.io/spacr/tutorials/

.. image:: ../../readme_images/sv/citing-spacr.svg
   :alt: Citera spaCR. Om spaCR bidrar till din forskning, citera: Olafsson EB, et al. En poolad bildbaserad CRISPR screening identifierar EAF1 som en T. gondii modulator för ESCRT subversion.
   :width: 99.5%

| |citing-spacr-2-1|\ |citing-spacr-2-2|

.. |citing-spacr-2-1| image:: ../../readme_images/sv/citing-spacr-2-1.svg
   :alt: BioRxiv preprint
   :width: 49.5%
   :target: https://www.biorxiv.org/content/10.64898/2026.07.08.737057v1

.. |citing-spacr-2-2| image:: ../../readme_images/sv/citing-spacr-2-2.svg
   :alt: programvaruarkiv
   :width: 49.5%
   :target: https://doi.org/10.5281/zenodo.21343316

.. image:: ../../readme_images/sv/other-work-citing-spacr.svg
   :alt: Andra arbeten med hänvisning till spaCR.
   :width: 99.5%

.. spacr-citing-papers-begin

.. image:: ../../readme_images/sv/other-work-citing-spacr-2-1.svg
   :alt: Metabol anpassningsförmåga och näringsavlagring i Toxoplasma gondii: insikter från intag väg-deficen mutanter.
   :width: 99.5%
   :target: https://journals.asm.org/doi/full/10.1128/msphere.01011-24

.. image:: ../../readme_images/sv/other-work-citing-spacr-2-2.svg
   :alt: IRE1α främjar fagsomalt kalciumflöde för att öka makrofag fungicid aktivitet.
   :width: 99.5%
   :target: https://www.cell.com/cell-reports/fulltext/S2211-1247(25)00465-6

.. image:: ../../readme_images/sv/other-work-citing-spacr-2-3.svg
   :alt: Toxoplasma GRA8 engagerar värd ESCRT tillbehör protein ALG-2 och är nödvändig för parasit metabolisk integritet.
   :width: 99.5%
   :target: https://www.biorxiv.org/content/10.64898/2026.07.20.739547v1.abstract

.. image:: ../../readme_images/sv/other-work-citing-spacr-2-4.svg
   :alt: Hydrogen sulfide dynamically upregulates copper uptake and localization.
   :width: 99.5%
   :target: https://www.biorxiv.org/content/10.64898/2026.07.30.741779v1

.. image:: ../../readme_images/sv/other-work-citing-spacr-2-5.svg
   :alt: spaCR: Spatial fenotypanalys av CRISPR-Cas9-screeningar (preprint version 1).
   :width: 99.5%
   :target: https://www.researchsquare.com/article/rs-7368254/v1

.. spacr-citing-papers-end

.. image:: ../../readme_images/sv/acknowledgments.svg
   :alt: Tack. spaCR bygger på öppen vetenskaplig programvara, bland annat NumPy, pandas, scikit-image, scikit-learn, Cellpose, PyTorch och Qt. Se information om översättningsmodellerna för modellerna som användes till den flerspråkiga dokumentationen och gränssnittskatalogerna.
   :width: 99.5%
   :target: ../TRANSLATION_MODELS.md
