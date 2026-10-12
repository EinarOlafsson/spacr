|Website| |Platforms| |Python| |Qt| |Test counts| |Coverage| |Issues| |Conda| |PyPI| |Docker| |PyPI Downloads| |Docs| |Tutorials| |Preprint| |DOI| |Cite| |License| |PyPI rank|

.. |Website| image:: https://img.shields.io/badge/Website-spaCR-0A7BBB
   :target: https://einarolafsson.github.io/projects/spacr/
   :alt: spaCR-Website

.. |Docs| image:: https://img.shields.io/github/actions/workflow/status/EinarOlafsson/spacr/pages%2Fpages-build-deployment?label=API&logo=readthedocs&logoColor=white
   :target: https://einarolafsson.github.io/spacr/
   :alt: API-Dokumentation

.. |Tutorials| image:: https://img.shields.io/badge/Tutorials-4A9EFF
   :target: https://einarolafsson.github.io/spacr/tutorials/
   :alt: Interaktive Tutorials

.. |PyPI| image:: https://img.shields.io/pypi/v/spacr?label=PyPI&logo=pypi&logoColor=white
   :target: https://pypi.org/project/spacr/
   :alt: PyPI-Version

.. |Python| image:: https://img.shields.io/badge/Python-3.9%E2%80%933.14-3776AB?logo=python&logoColor=white
   :target: https://pypi.org/project/spacr/
   :alt: Python 3.9 bis 3.14

.. |Coverage| image:: https://img.shields.io/codecov/c/github/EinarOlafsson/spacr/nightly?logo=codecov&logoColor=white
   :target: https://app.codecov.io/github/EinarOlafsson/spacr
   :alt: Codecov-Testabdeckung

.. |Test counts| image:: https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2FEinarOlafsson%2Fspacr%2Fnightly%2Fdocs%2Fsource%2F_static%2Ftest-counts.json&cacheSeconds=300&logo=pytest&logoColor=white
   :target: https://github.com/EinarOlafsson/spacr/actions/workflows/tests.yml
   :alt: Testsuite

.. |Qt| image:: https://img.shields.io/badge/GUI-Qt-41CD52?logo=qt&logoColor=white
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/index.html#module-spacr.qt
   :alt: Qt-Oberfläche

.. |Issues| image:: https://img.shields.io/github/issues/EinarOlafsson/spacr?logo=github
   :target: https://github.com/EinarOlafsson/spacr/issues
   :alt: GitHub-Issues

.. |License| image:: https://img.shields.io/badge/License-BSD--3-blue?logo=opensourceinitiative&logoColor=white
   :target: https://github.com/EinarOlafsson/spacr/blob/main/LICENSE
   :alt: BSD-3-Clause-Lizenz

.. |Preprint| image:: https://img.shields.io/badge/bioRxiv-BF2636
   :target: https://www.biorxiv.org/content/10.64898/2026.07.08.737057v1
   :alt: bioRxiv-Preprint

.. |DOI| image:: https://img.shields.io/badge/DOI-blue?logo=zenodo&logoColor=white
   :target: https://doi.org/10.5281/zenodo.21343316
   :alt: Zenodo-DOI

.. |Conda| image:: https://img.shields.io/conda/vn/conda-forge/spacr?label=Conda&logo=anaconda&logoColor=white
   :target: https://anaconda.org/conda-forge/spacr
   :alt: conda-forge-Version

.. |Release date| image:: https://anaconda.org/conda-forge/spacr/badges/latest_release_date.svg
   :target: https://anaconda.org/conda-forge/spacr
   :alt: conda-forge-Datum der letzten Veröffentlichung

.. |Docker| image:: https://img.shields.io/badge/Docker-GHCR-2496ED?logo=docker&logoColor=white
   :target: https://github.com/EinarOlafsson/spacr/pkgs/container/spacr
   :alt: Docker-Images auf GHCR

.. |PyPI Downloads| image:: https://img.shields.io/pepy/dt/spacr?label=downloads&logo=pypi&logoColor=white
   :target: https://pepy.tech/projects/spacr
   :alt: PyPI-Downloads

.. |Platforms| image:: https://img.shields.io/badge/Linux%20OSX%20Win-lightgrey?logo=linux&logoColor=white
   :target: https://github.com/EinarOlafsson/spacr/blob/nightly/docs/source/installers.rst
   :alt: Linux, macOS und Windows

.. |Cite| image:: https://img.shields.io/badge/Cite-8A2BE2
   :target: https://github.com/EinarOlafsson/spacr/blob/main/CITATION.cff
   :alt: spaCR zitieren

.. |PyPI rank| image:: https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fsql-clickhouse.clickhouse.com%2F%3Fuser%3Ddemo%26param_package_name%3Dspacr%26param_days%3D30%26query%3DWITH%2B%2528%2BSELECT%2Bsum%2528count%2529%2BFROM%2Bpypi.pypi_downloads_per_day%2BWHERE%2Bproject%2B%253D%2B%257Bpackage_name%253AString%257D%2BAND%2Bdate%2B%253E%253D%2BtoDate%2528now%2528%2527UTC%2527%2529%2529%2B-%2B%257Bdays%253AUInt16%257D%2BAND%2Bdate%2B%253C%2BtoDate%2528now%2528%2527UTC%2527%2529%2529%2B%2529%2BAS%2Bdownloads%2BSELECT%2Bdownloads%2BAS%2Bpackage_downloads%252C%2BcountIf%2528n%2B%253E%253D%2Bdownloads%2529%2BAS%2Brank%252C%2Bcount%2528%2529%2BAS%2Btotal_packages%252C%2B100.0%2B%252A%2Brank%2B%252F%2BnullIf%2528total_packages%252C%2B0%2529%2BAS%2Bpercentile%252C%2Bif%2528%2Btotal_packages%2B%253D%2B0%2BOR%2Bdownloads%2B%253D%2B0%252C%2B%2527no%2Bdata%2527%252C%2Bconcat%2528%2B%2527top%2B%2527%252C%2BtoString%2528ceil%25281000.0%2B%252A%2Brank%2B%252F%2BnullIf%2528total_packages%252C%2B0%2529%2529%2B%252F%2B10%2529%252C%2B%2527%2525%2527%2B%2529%2B%2529%2BAS%2Bmessage%2BFROM%2B%2528%2BSELECT%2Bproject%252C%2Bsum%2528count%2529%2BAS%2Bn%2BFROM%2Bpypi.pypi_downloads_per_day%2BWHERE%2Bdate%2B%253E%253D%2BtoDate%2528now%2528%2527UTC%2527%2529%2529%2B-%2B%257Bdays%253AUInt16%257D%2BAND%2Bdate%2B%253C%2BtoDate%2528now%2528%2527UTC%2527%2529%2529%2BGROUP%2BBY%2Bproject%2B%2529%2BFORMAT%2BJSON&query=%24.data%5B0%5D.message&label=Rank&color=brightgreen&logo=pypi&logoColor=white&cacheSeconds=86400
   :target: https://clickpy.clickhouse.com/dashboard/spacr
   :alt: spaCR-Rang bei PyPI-Downloads der letzten 30 vollen Tage

.. image:: ../../source/_static/deck/slides/slide_01.jpg
   :alt: spaCR
   :width: 920
   :target: https://einarolafsson.github.io/spacr/_static/deck/

| |intro-1|\ |intro-2|

.. |intro-1| image:: ../../readme_images/de/intro-1.svg
   :alt: ← Zurück
   :width: 49.5%
   :target: ../../source/_static/deck/pages/57.md

.. |intro-2| image:: ../../readme_images/de/intro-2.svg
   :alt: Weiter →
   :width: 49.5%
   :target: ../../source/_static/deck/pages/02.md

.. spacr-language-picker-begin

.. image:: ../../readme_images/de/generated-language-picker.svg
   :alt: Sprachen: Deutsch
   :width: 32.8%
   :target: README.md

.. spacr-language-picker-end

.. image:: ../../readme_images/de/spacr.svg
   :alt: spaCR. Räumliche Phänotypanalyse von CRISPR-Screens. spaCR segmentiert und vermisst einzelne Zellen in Mikroskopiebildern, integriert Phänotypen einzelner Objekte mit sequenzierungsbasierten Guide-Häufigkeiten und schätzt, welche Gene mit phänotypischen Veränderungen assoziiert sind. Ausgehend von Plattenbildern und FASTQ-Reads erzeugt es Messungen pro Objekt, trainierte Klassifikatoren, Effektschätzungen pro Guide und Gen sowie eine Rangliste der Treffer. Die Segmentierungs-, Mess-, Anmerkungs- und Klassifizierungsmodule laufen auch ohne Sequenzierungsarm. Make Masks korrigiert Segmentierungsmasken und annotiert mit dem Werkzeug Box unabhängige Rechtecke mit Klassenlabels für den YOLO-Export. Die Rechtecke behalten ihre eigenen Labels und ihren Verlauf, ohne Quellbilder oder Masken zu ändern. Einzelheiten zu jedem Werkzeug stehen im Funktionsleitfaden. Bilder, Masken und Bildausschnitte sind Dateien auf der Festplatte. Messungen, Annotationen und Modellvorhersagen sind Tabellen in der SQLite-Datenbank measurements.db des Projekts. Läuft als Desktop-Anwendung oder ohne grafische Oberfläche auf einer Workstation, einem Server oder Cluster.
   :width: 100%
   :target: ../../source/features.rst

.. image:: ../../readme_images/de/try-spacr.svg
   :alt: spaCR ausprobieren.
   :width: 100%

.. code-block:: bash

   conda create -n spacr python=3.12 -y
   conda activate spacr
   python -m pip install spacr
   spacr

.. image:: ../../readme_images/de/try-spacr-2.svg
   :alt: Verwenden Sie Testdaten laden… in Import, Make Masks, Annotate oder einem Assay-Bildschirm, um Beispieldaten herunterzuladen. Im Terminal verwenden Sie spacr-download.
   :width: 100%

.. image:: ../../readme_images/de/hardware-support.svg
   :alt: Hardware-Unterstützung.
   :width: 100%

.. spacr-hardware-begin

.. image:: ../../readme_images/de/generated-hardware-1.svg
   :alt: NVIDIA (CUDA): Cellpose 4 GPU, Torch GPU, UMAP / clustering GPU. AMD on Linux (ROCm): Cellpose 4 GPU, Torch GPU, UMAP / clustering CPU. AMD in an Intel Mac (Metal): Cellpose 4 GPU, Torch GPU, UMAP / clustering CPU. Apple Silicon (Metal): Cellpose 4 GPU, Torch GPU, UMAP / clustering CPU. Intel Arc/Xe (XPU): Cellpose 4 GPU, Torch GPU, UMAP / clustering CPU. No GPU: Cellpose 4 CPU, Torch CPU, UMAP / clustering CPU. unterstützt (stabil) implementiert (Beta) nur CPU-Unterstützung.
   :width: 100%

.. spacr-hardware-end

.. image:: ../../readme_images/de/install-spacr.svg
   :alt: spaCR installieren. Wählen Sie einen Installationsweg: Desktopanwendung: ein Installationsprogramm für Linux, macOS oder Windows mit eigenem Python. pip: python -m pip install spacr in einer Conda-Umgebung. conda-forge: conda install conda-forge::spacr in einer Conda-Umgebung. Docker: die Kommandozeilen-Pipelines in einem Linux-Container, mit dem CPU-Image oder dem Image für NVIDIA-GPUs:
   :width: 100%

.. code-block:: bash

   docker run --rm ghcr.io/einarolafsson/spacr:1.5.1.4 spacr-run --list
   docker run --rm --gpus all ghcr.io/einarolafsson/spacr:1.5.1.4-cuda12.4 spacr-run --list

.. image:: ../../readme_images/de/install-spacr-2.svg
   :alt: Die Docker-Anleitung beschreibt die GPU-Einrichtung, das Einbinden von Daten und die Pipeline-Befehle.
   :width: 100%
   :target: ../../source/installer_guide.rst#container-images

.. image:: ../../readme_images/de/desktop-application.svg
   :alt: Desktopanwendung. Die Installateure bündeln ihre eigenen Python. Conda ist nicht erforderlich.
   :width: 100%

.. spacr-installer-links-begin

|InstallerLinux| |InstallerMacOS| |InstallerWindows|

.. |InstallerWindows| image:: ../../../spacr/resources/icons/platforms/windows.png
   :width: 64
   :alt: Windows 10/11: spaCR 1.5.1.4 herunterladen
   :target: https://github.com/EinarOlafsson/spacr/releases/download/v1.5.1.4/spaCR-1.5.1.4-Windows-Online-Setup.exe
.. |InstallerMacOS| image:: ../../../spacr/resources/icons/platforms/macos.png
   :width: 64
   :alt: macOS 11+ (Intel und Apple Silicon): spaCR 1.5.1.4 herunterladen
   :target: https://github.com/EinarOlafsson/spacr/releases/download/v1.5.1.4/spaCR-1.5.1.4-macOS-Universal-Online.pkg
.. |InstallerLinux| image:: ../../../spacr/resources/icons/platforms/linux.png
   :width: 64
   :alt: 64-Bit-Linux: spaCR 1.5.1.4 herunterladen
   :target: https://github.com/EinarOlafsson/spacr/releases/download/v1.5.1.4/spaCR-1.5.1.4-Linux-x86_64-Online.run

.. spacr-installer-links-end

.. image:: ../../readme_images/de/desktop-application-2.svg
   :alt: Machen Sie die heruntergeladene Datei unter Linux ausführbar und führen Sie sie aus:
   :width: 100%

.. code-block:: bash

   chmod +x SpaCR-*-Linux-x86_64-Online.run
   ./SpaCR-*-Linux-x86_64-Online.run

.. image:: ../../readme_images/de/desktop-application-3.svg
   :alt: Öffnen Sie auf macOS das .pkg. Die aktuelle Beta wird nicht beglaubigt; wenn Gatekeeper sie blockiert, wählen Sie Systemeinstellungen → Datenschutz & Sicherheit → Öffnen Sie trotzdem. Anweisungen zum Aktualisieren, Deinstallieren, Offlinebetrieb und zur Fehlerbehebung finden Sie in der Installationsanleitung. Die Systemanforderungen enthalten Empfehlungen für Arbeitsstationen und Server sowie Tabellen zur GPU-Kompatibilität.
   :width: 100%

| |desktop-application-3-link-1|\ |desktop-application-3-link-2|

.. |desktop-application-3-link-1| image:: ../../readme_images/de/desktop-application-3-link-1.svg
   :alt: Installationsanleitung
   :width: 49.5%
   :target: ../../source/installer_guide.rst

.. |desktop-application-3-link-2| image:: ../../readme_images/de/desktop-application-3-link-2.svg
   :alt: Systemanforderungen
   :width: 49.5%
   :target: ../../source/system_requirements.rst

.. image:: ../../readme_images/de/pypi-installation.svg
   :alt: Installation über PyPI. Installieren Sie die PyPI-Veröffentlichung von spaCR mit pip in einer Conda-Umgebung. Python 3.12 bietet die größte Auswahl an optionalen wissenschaftlichen Paketen:
   :width: 100%

.. code-block:: bash

   conda create -n spacr python=3.12 -y
   conda activate spacr
   python -m pip install --upgrade pip
   python -m pip install spacr
   spacr

.. image:: ../../readme_images/de/pypi-installation-2.svg
   :alt: spaCR unterstützt Python 3.9 through 3.14, außer Python 3.14.1, das von torchvision ausgeschlossen wird. Linux wird für die anspruchsvollsten CUDA- und ROCm-Workflows empfohlen; macOS und Windows werden ebenfalls unterstützt und nutzen beide ihre GPUs — macOS über Metal, das Apple Silicon und die AMD-Karten in Intel Macs abdeckt, und Windows über CUDA oder DirectML. Die Standardinstallation enthält die Qt-Desktopoberfläche. Auf einem Server, Cluster oder CI-Runner können Sie die Befehlszeilen-Pipelines ausführen, ohne sie zu öffnen:
   :width: 100%

.. code-block:: bash

   python -m pip install spacr
   spacr-run --list

.. image:: ../../readme_images/de/pypi-installation-3.svg
   :alt: Optional integrations are installed separately, for example spacr[zarr], spacr[omero], spacr[napari] and spacr[czi,nd2,lif]. See the Installationsanleitung for the complete extras and Python-version compatibility table.
   :width: 100%
   :target: ../../source/installer_guide.rst

.. image:: ../../readme_images/de/conda-forge-installation.svg
   :alt: Installation mit conda-forge. Das offizielle conda-forge-Paket installiert spaCR und seine Desktop-Abhängigkeiten in der aktiven Umgebung:
   :width: 100%

.. code-block:: bash

   conda create -n spacr python=3.12 -y
   conda activate spacr
   conda install conda-forge::spacr
   spacr

.. image:: ../../readme_images/de/install-from-source.svg
   :alt: Installation aus dem Quellcode. Klonen Sie das Repository und installieren Sie es im editierbaren Modus. Ihre Arbeitskopie ist dann das installierte Paket, und Änderungen werden ohne Neuinstallation wirksam:
   :width: 100%

.. code-block:: bash

   git clone https://github.com/EinarOlafsson/spacr.git
   cd spacr
   conda create -n spacr python=3.12 -y
   conda activate spacr
   pip install -e .
   spacr

.. image:: ../../readme_images/de/install-from-source-2.svg
   :alt: Dieser Befehl klont main, den Standard-Branch mit der neuesten Version. Entwickelt wird auf nightly; fügen Sie --branch nightly hinzu, um stattdessen diesen Branch zu klonen. Für eine bestimmte Version:
   :width: 100%

.. code-block:: bash

   git clone --branch v1.5.0.5 https://github.com/EinarOlafsson/spacr.git

.. image:: ../../readme_images/de/install-from-source-3.svg
   :alt: Um spätere Änderungen zu übernehmen, führen Sie im Klon aus:
   :width: 100%

.. code-block:: bash

   git pull
   pip install -e .

.. image:: ../../readme_images/de/install-from-source-4.svg
   :alt: Installieren Sie neu, wenn sich Abhängigkeiten oder Einstiegspunkte ändern. Änderungen am Python-Code gelten direkt; spacr-doctor zeigt die aktive Installation an.
   :width: 100%

.. image:: ../../readme_images/de/install-from-source-light.svg
   :alt: Installation aus dem Quellcode (schlank). Mitwirkende benötigen die Versionshistorie; um spaCR nur auszuführen, wählen Sie eine der folgenden Optionen. Messungen von nightly bei 05302fd5c am 2026-10-07 mit packaging/measure_clone_forms.sh:
   :width: 100%

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

.. image:: ../../readme_images/de/install-from-source-light-2.svg
   :alt: Der vollständige nightly-Klon lud 9.25 GiB herunter. --filter=blob:none zum flachen Klon hinzuzufügen verkleinert das ausgecheckte Arbeitsverzeichnis nicht: Sein Git-Objektspeicher umfasst weiterhin 2032 MB. Git meldet bei Bedarf nachgeladene Objekte nicht, daher wurde keine Gesamtgröße des Downloads gemessen. Die versionierten Dateien von nightly belegen im ausgecheckten Arbeitsverzeichnis 3416 MB (gemessen am 2026-10-08), ohne Git-Historie. Downloadgröße und Dauer hängen vom Branch ab.
   :width: 100%

.. image:: ../../readme_images/de/command-line-entry-points.svg
   :alt: Befehle für die Kommandozeile.
   :width: 100%

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

.. image:: ../../readme_images/de/command-line-entry-points-2.svg
   :alt: spacr-run --list listet Module mit Befehlszeileneinstiegspunkten für die Ausführung ohne grafische Oberfläche auf. Reine GUI-Module für Annotation, Kuratierung, Vergleich und Exploration werden nicht aufgeführt.
   :width: 100%

.. image:: ../../readme_images/de/core-workflow.svg
   :alt: Kern-Workflow. Der primäre Arbeitsablauf umfasst sechs Module: Mask segmentiert Zellen, Zellkerne, Pathogene und Organellen mit Cellpose. Measure schreibt Morphologie-, Intensitäts-, Textur-, räumliche und Kolokalisationsmerkmale sowie Objektausschnitte nach SQLite. Annotate beschriftet Objektausschnitte in einem tastaturgesteuerten Raster und unterstützt Active-Learning-Warteschlangen. Classify trainiert bild- oder messwertbasierte Modelle und speichert mit jedem Checkpoint die Leistung auf zurückgehaltenen Daten. Map Barcodes ordnet FASTQ-Reads Wells und gRNAs zu und liefert QC für Häufigkeit, Kollisionen und Abdeckung. Regression schätzt Guide-, Gen-, Bedingungs- und Kontrolleffekte mit Modellfamilien für kontinuierliche Werte, Anteile und Zähldaten.
   :width: 100%

.. image:: ../../readme_images/de/spacr-modules.svg
   :alt: spaCR-Module.
   :width: 100%

.. spacr-workflow-begin

.. image:: ../../readme_images/de/generated-workflow-1.svg
   :alt: Kern. Core sequence from microscopy images through segmentation, measurements, annotations, classification, barcode mapping and regression.
   :width: 100%

| |Module_mask|\ |Module_measure|\ |Module_annotate|\ |Module_classify_merged|\ |Module_map_barcodes|\ |Module_regression|

.. image:: ../../readme_images/de/generated-workflow-2.svg
   :alt: Daten. Import images and tables into spaCR projects and execute reproducible multi-plate workflows.
   :width: 100%

| |Module_foreign|\ |Module_embeddings|\ |Module_run_compare|\ |Module_experiment_design|\ |Module_power|\ |Module_dose_response|
| |Module_qc_dashboard|

.. image:: ../../readme_images/de/generated-workflow-3.svg
   :alt: Werkzeuge. Point these at a project: edit masks by hand, stitch tiles, read an embedding, draw a gate, build a plot, check quality.
   :width: 100%

| |Module_make_masks|\ |Module_align|\ |Module_umap|\ |Module_gate_editor|\ |Module_graph_builder|

.. image:: ../../readme_images/de/generated-workflow-4.svg
   :alt: Organismus. Organismusspezifische Bildanalyse und quantitative Assay-Ergebnisse.
   :width: 100%

| |Module_toxoplasma|

.. |Module_mask| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/core/index.html#spacr.core.preprocess_generate_masks"><img src="../../../spacr/resources/icons/workflow/mask.png" width="16.0%" align="middle" alt="API für Mask öffnen"></a>

.. |Module_measure| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/measure/index.html"><img src="../../../spacr/resources/icons/workflow/measure.png" width="16.0%" align="middle" alt="API für Measure öffnen"></a>

.. |Module_annotate| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/qt/screens/annotate/index.html"><img src="../../../spacr/resources/icons/workflow/annotate.png" width="16.0%" align="middle" alt="API für Annotate öffnen"></a>

.. |Module_classify_merged| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/classify/index.html"><img src="../../../spacr/resources/icons/workflow/classify_merged.png" width="16.0%" align="middle" alt="API für Classify öffnen"></a>

.. |Module_map_barcodes| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/sequencing/index.html"><img src="../../../spacr/resources/icons/workflow/map_barcodes.png" width="16.0%" align="middle" alt="API für Map Barcodes öffnen"></a>

.. |Module_regression| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/ml/index.html"><img src="../../../spacr/resources/icons/workflow/regression.png" width="16.0%" align="middle" alt="API für Regression öffnen"></a>

.. |Module_foreign| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/foreign/index.html"><img src="../../../spacr/resources/icons/workflow/apps/foreign.png" width="16.0%" align="middle" alt="API für Import öffnen"></a>

.. |Module_embeddings| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/qt/screens/embeddings/index.html"><img src="../../../spacr/resources/icons/workflow/apps/embeddings.png" width="16.0%" align="middle" alt="API für Embeddings öffnen"></a>

.. |Module_run_compare| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/qt/screens/run_compare/index.html"><img src="../../../spacr/resources/icons/workflow/apps/run_compare.png" width="16.0%" align="middle" alt="API für Run Compare öffnen"></a>

.. |Module_experiment_design| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/qt/screens/experiment_design/index.html"><img src="../../../spacr/resources/icons/workflow/apps/experiment_design.png" width="16.0%" align="middle" alt="API für Experiment Design öffnen"></a>

.. |Module_power| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/qt/screens/power/index.html"><img src="../../../spacr/resources/icons/workflow/apps/power.png" width="16.0%" align="middle" alt="API für Power / Design öffnen"></a>

.. |Module_dose_response| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/qt/screens/dose_response/index.html"><img src="../../../spacr/resources/icons/workflow/apps/dose_response.png" width="16.0%" align="middle" alt="API für Dose–Response öffnen"></a>

.. |Module_qc_dashboard| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/qt/screens/qc_dashboard/index.html"><img src="../../../spacr/resources/icons/workflow/apps/qc_dashboard.png" width="16.0%" align="middle" alt="API für QC öffnen"></a>

.. |Module_make_masks| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/qt/screens/make_masks/index.html"><img src="../../../spacr/resources/icons/workflow/apps/make_masks.png" width="16.0%" align="middle" alt="API für Make Masks öffnen"></a>

.. |Module_align| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/align/index.html"><img src="../../../spacr/resources/icons/workflow/apps/align.png" width="16.0%" align="middle" alt="API für Align &amp; Stitch öffnen"></a>

.. |Module_umap| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/core/index.html#spacr.core.generate_image_umap"><img src="../../../spacr/resources/icons/workflow/apps/umap.png" width="16.0%" align="middle" alt="API für Image UMAP öffnen"></a>

.. |Module_gate_editor| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/qt/screens/gate_editor/index.html"><img src="../../../spacr/resources/icons/workflow/apps/gate_editor.png" width="16.0%" align="middle" alt="API für Gate Editor öffnen"></a>

.. |Module_graph_builder| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/qt/screens/graph_builder/index.html"><img src="../../../spacr/resources/icons/workflow/apps/graph_builder.png" width="16.0%" align="middle" alt="API für Graph Builder öffnen"></a>

.. |Module_toxoplasma| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/qt/screens/organism_screen/index.html#spacr-qt-screens-organism-screen-toxoplasma"><img src="../../../spacr/resources/icons/workflow/apps/toxoplasma.png" width="16.0%" align="middle" alt="API für Toxoplasma öffnen"></a>

.. spacr-workflow-end

.. image:: ../../readme_images/de/spacr-modules-2.svg
   :alt: Alle Module mit einer Kachel auf der Startseite, in deren Reihenfolge: zuerst die sechs Pipeline-Module, dann die übrigen. Wählen Sie eine Kachel, um die API-Seite des Moduls zu öffnen.
   :width: 100%

.. image:: ../../readme_images/de/other-resources.svg
   :alt: Sonstige Mittel.
   :width: 100%

.. image:: ../../readme_images/de/other-resources-2-1.svg
   :alt: Interaktive Tutorials — geführte Arbeitsabläufe von der Installation bis zur Untersuchung von Treffern.
   :width: 100%
   :target: https://einarolafsson.github.io/spacr/tutorials/

.. image:: ../../readme_images/de/other-resources-2-2.svg
   :alt: Python API Schnellstart — Pipelines aus Skripten, Notebooks oder einem Cluster ausführen und validieren.
   :width: 100%
   :target: ../../source/python_api.rst

.. image:: ../../readme_images/de/other-resources-2-3.svg
   :alt: Funktionsleitfaden — Fähigkeiten, Reife und optionale Integrationen.
   :width: 100%
   :target: ../../source/features.rst

.. image:: ../../readme_images/de/other-resources-2-4.svg
   :alt: Kuratierte API Referenz — unterstützte Eingabepunkte nach Aufgabe, wobei das komplette Modul eine Ebene tiefer verweist.
   :width: 100%
   :target: https://einarolafsson.github.io/spacr/api/index.html

.. image:: ../../readme_images/de/other-resources-2-5.svg
   :alt: Sprach- und Übersetzungshandbuch — Schnittstellensprachen, kontextbezogene Hilfe und Politik der wissenschaftlichen Ergebnisse.
   :width: 100%
   :target: ../../source/localization.rst

.. image:: ../../readme_images/de/language-translation.svg
   :alt: Sprache und Übersetzung. Die Oberfläche unterstützt zehn Sprachen in der Navigation und den Einstellungen. AI- und LIVE-Steuerelemente, Modulbeschreibungen und geprüfte Kontexthilfe werden ebenfalls übersetzt. Ändern Sie die Sprache unter spaCR → Einstellungen → Sprache, ohne neu zu starten. Protokolle, Pfade, Datenbankwerte und Messungen werden nie übersetzt; wissenschaftliche Ausgaben bleiben im kanonischen Englisch. Siehe die Richtlinie zur Kontexthilfe.
   :width: 100%
   :target: ../../source/localization.rst#contextual-help

.. image:: ../../readme_images/de/language-translation-2.svg
   :alt: Die neun nicht-englischen Kataloge werden von einem Muttersprachler maschinengefertigt und technisch überarbeitet, anstatt zu Ende zu lesen. Die Überprüfungsspielraum Aufzeichnungen, welche Sprachen einen menschlichen Pass hatten und jeder Begriff auf Englisch durch Entscheidung übrig geblieben ist.
   :width: 100%
   :target: ../REVIEW_SCOPE_2026-09-04.md

.. image:: ../../readme_images/de/animated-setting-guidance.svg
   :alt: Animierte Einstellungshilfe. Einstellungen mit einer visuellen Erklärung bieten in ihrem Tooltip die Schaltfläche Animation. Durchsuchen Sie die Galerie der Einstellungsanimationen oder das Register der Einstellungsanimationen.
   :width: 100%

| |animated-setting-guidance-link-1|\ |animated-setting-guidance-link-2|

.. |animated-setting-guidance-link-1| image:: ../../readme_images/de/animated-setting-guidance-link-1.svg
   :alt: Galerie der Einstellungsanimationen
   :width: 49.5%
   :target: https://einarolafsson.github.io/spacr/setting_animations.html

.. |animated-setting-guidance-link-2| image:: ../../readme_images/de/animated-setting-guidance-link-2.svg
   :alt: Register der Einstellungsanimationen
   :width: 49.5%
   :target: https://einarolafsson.github.io/spacr/api/spacr/setting_animations/index.html

.. image:: ../../readme_images/de/data.svg
   :alt: Daten. Referenzdatensätze.
   :width: 100%

|DataBioStudies| |DataHuggingFace| |DataNCBI| |DataSpaCRPower| |DataBioRxiv|

.. |DataBioStudies| image:: ../../../spacr/resources/icons/databanks/biostudies_button.png
   :width: 72
   :alt: Mikroskopiedatensatz in BioStudies öffnen
   :target: https://doi.org/10.6019/S-BIAD2135

.. |DataHuggingFace| image:: ../../../spacr/resources/icons/databanks/huggingface_button.png
   :width: 72
   :alt: Testdatensatz auf Hugging Face öffnen
   :target: https://huggingface.co/datasets/einarolafsson/toxo_mito

.. |DataNCBI| image:: ../../../spacr/resources/icons/databanks/ncbi_button.png
   :width: 72
   :alt: Sequenzierungsdatensatz bei NCBI öffnen
   :target: https://www.ncbi.nlm.nih.gov/bioproject/?term=PRJNA1261935

.. |DataSpaCRPower| image:: ../../../spacr/resources/icons/databanks/spacrpower_button.png
   :width: 72
   :alt: spaCRPower öffnen
   :target: https://github.com/maomlab/spaCRPower

.. |DataBioRxiv| image:: ../../../spacr/resources/icons/databanks/biorxiv_button.png
   :width: 72
   :alt: bioRxiv-Preprint öffnen
   :target: https://www.biorxiv.org/content/10.64898/2026.07.08.737057v1

.. image:: ../../readme_images/de/model-zoo.svg
   :alt: Modellzoo. spaCR liefert einen Katalog von ausgebildeten Modellen und holt sie auf Anfrage ab. Öffnen Sie Model Zoo vom Home-Bildschirm, um sie zu durchsuchen und zu installieren, oder benennen Sie einen Schlüssel in einer Einstellungsdatei -- pathogen_model: toxoplasma_pv_v1 -- und das Modell wird heruntergeladen und Checksummen-verifiziert, wenn es zum ersten Mal benötigt wird. Jeder veröffentlichte Eintrag trägt eine SHA-256; ein Eintrag ohne einen wird abgelehnt, anstatt installiert, weil ein verkürzter oder ersetzter Checkpoint nicht vom realen angezeigt werden kann.
   :width: 100%

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

.. image:: ../../readme_images/de/generated-model-zoo-1.svg
   :alt: Older models.
   :width: 100%

| |generated-model-zoo-1-link-1|\ |generated-model-zoo-1-link-2|\ |generated-model-zoo-1-link-3|
| |generated-model-zoo-1-link-4|\ |generated-model-zoo-1-link-5|\ |generated-model-zoo-1-link-6|
| |generated-model-zoo-1-link-7|

.. |generated-model-zoo-1-link-1| image:: ../../readme_images/de/generated-model-zoo-1-link-1.svg
   :alt: toxoplasma_pv_v3
   :width: 32.8%
   :target: ../../models/toxoplasma_pv_v3.md

.. |generated-model-zoo-1-link-2| image:: ../../readme_images/de/generated-model-zoo-1-link-2.svg
   :alt: toxoplasma_pv_v2
   :width: 32.8%
   :target: ../../models/toxoplasma_pv_v2.md

.. |generated-model-zoo-1-link-3| image:: ../../readme_images/de/generated-model-zoo-1-link-3.svg
   :alt: toxoplasma_pv_v1
   :width: 32.8%
   :target: ../../models/toxoplasma_pv_v1.md

.. |generated-model-zoo-1-link-4| image:: ../../readme_images/de/generated-model-zoo-1-link-4.svg
   :alt: toxoplasma_plaque_v2
   :width: 32.8%
   :target: ../../models/toxoplasma_plaque_v2.md

.. |generated-model-zoo-1-link-5| image:: ../../readme_images/de/generated-model-zoo-1-link-5.svg
   :alt: toxoplasma_plaque_v1
   :width: 32.8%
   :target: ../../models/toxoplasma_plaque_v1.md

.. |generated-model-zoo-1-link-6| image:: ../../readme_images/de/generated-model-zoo-1-link-6.svg
   :alt: toxoplasma_well_detector_v2
   :width: 32.8%
   :target: ../../models/toxoplasma_well_detector_v2.md

.. |generated-model-zoo-1-link-7| image:: ../../readme_images/de/generated-model-zoo-1-link-7.svg
   :alt: toxoplasma_well_detector_v1
   :width: 32.8%
   :target: ../../models/toxoplasma_well_detector_v1.md

.. image:: ../../readme_images/de/generated-model-zoo-2.svg
   :alt: Alternatives.
   :width: 100%

| |generated-model-zoo-2-link-1|\ |generated-model-zoo-2-link-2|

.. |generated-model-zoo-2-link-1| image:: ../../readme_images/de/generated-model-zoo-2-link-1.svg
   :alt: toxoplasma_pv_v5
   :width: 32.8%
   :target: ../../models/toxoplasma_pv_v5.md

.. |generated-model-zoo-2-link-2| image:: ../../readme_images/de/generated-model-zoo-2-link-2.svg
   :alt: toxoplasma_plaque_v3
   :width: 32.8%
   :target: ../../models/toxoplasma_plaque_v3.md

.. spacr-model-zoo-end

.. image:: ../../readme_images/de/diagnosing-performance.svg
   :alt: Leistungsdiagnose. Erzeugen Sie einen Hardwarebericht und fügen Sie ihn einem leistungsbezogenen GitHub-Issue bei:
   :width: 100%

.. code-block:: bash

   python tools/spacr_hardware_report.py

.. image:: ../../readme_images/de/diagnosing-performance-2.svg
   :alt: Speichert auf ~/.spacr/reports und druckt den Pfad. --quick überspringt die längeren Benchmarks; --out PATH setzt den Speicherort. Liest keine Projektdaten. Misst die Dauer von Importen, numerischen Bibliotheksoperationen, Fensteraufbau und Animation und meldet x86_64-Emulation auf Apple Silicon sowie die BLAS-Implementierung von NumPy.
   :width: 100%

.. image:: ../../readme_images/de/command-line-reference.svg
   :alt: Befehlszeilenreferenz. Jeder Befehl unten wird von pip install spacr installiert. Alle von ihnen akzeptieren --help.
   :width: 100%

.. image:: ../../readme_images/de/launching-the-application.svg
   :alt: Start der Anwendung.
   :width: 100%

.. code-block:: bash

   spacr              # the desktop application
   spacr-tutorial     # the interactive tutorial library
   spacr-server       # no first-run setup screen, for unattended launches

.. image:: ../../readme_images/de/when-spacr-will-not-start.svg
   :alt: Wenn spaCR nicht startet.
   :width: 100%

.. code-block:: bash

   spacr-doctor       # diagnose the installation and say how to fix it
   safespacr          # the least spaCR that can still change a setting

.. image:: ../../readme_images/de/when-spacr-will-not-start-2.svg
   :alt: spacr-doctor gibt eine Zeile pro Check aus, mit einem Befehl, der für jeden Fehler ausgeführt werden soll. Es wird auch berichtet, welcher spacr auf dem Pfad ist. safespacr liest jede Einstellung mit ihrem Standardwert und schaltet Hintergrund, Animationen, ausführliche Protokollierung und Vorabladen ab. Verwenden Sie es, wenn eine gespeicherte Einstellung den Start verhindert. Im abgesicherten Modus gespeicherte Einstellungen werden normal geschrieben und bleiben erhalten.
   :width: 100%

.. image:: ../../readme_images/de/running-modules-headlessly.svg
   :alt: Laufende Module ohne grafische Oberfläche. Kein Qt, kein Display — für Cluster, Server und CI.
   :width: 100%

.. code-block:: bash

   spacr-run --list                              # modules with a headless entry
   spacr-run --describe MODULE                   # what a module consumes and produces
   spacr-run validate --module MODULE \
       --settings settings.csv                   # check settings before spending the run
   spacr-run MODULE --settings settings.csv      # execute
   spacr-remote --help                           # submit and monitor SSH, Slurm or cloud jobs

.. image:: ../../readme_images/de/running-modules-headlessly-2.svg
   :alt: validate liest dieselben Einstellungen wie der Lauf und listet jedes Problem auf, etwa eine fehlende Einstellung oder einen nicht vorhandenen Pfad, jeweils mit einer Abhilfe.
   :width: 100%

.. image:: ../../readme_images/de/inspecting-a-run.svg
   :alt: Einen Lauf untersuchen. Jeder Lauf wird mit seinen Einstellungen, Hash-Eingängen, Ausgängen, Warnungen, Versionen und Samen auf ~/.spacr/runs tagebucht.
   :width: 100%

.. code-block:: bash

   spacr-repro RUN_DIR        # replay a recorded run from its journal
   spacr-workspace RUN_DIR    # what that run had open: databases, montages, views

.. image:: ../../readme_images/de/auditing-data-and-installation.svg
   :alt: Prüfungsdaten und Installation.
   :width: 100%

.. code-block:: bash

   spacr-db-audit DB      # SQLite health, integrity, locking, reader/writer probe
   spacr-leakage          # classifier train/test leakage audit
   spacr-plugins          # installed plugin registry and failure diagnostics

.. image:: ../../readme_images/de/environment.svg
   :alt: Umgebung.
   :width: 100%

.. code-block:: bash

   SPACR_LOG_LEVEL=DEBUG spacr      # verbose logging for one launch

.. image:: ../../readme_images/de/environment-2.svg
   :alt: Drehprotokolle werden auf ~/.spacr/logs/spacr.log geschrieben. Fügen Sie diese Datei einem Fehlerbericht bei.
   :width: 100%

.. image:: ../../readme_images/de/contributing-and-support.svg
   :alt: Beiträge und Support. Übermitteln Sie Fehlerberichte und klar abgegrenzte Funktionswünsche über GitHub Issues. Geben Sie bei einer Fehlermeldung die spaCR-Version, das Betriebssystem, die Python-Version, die Moduleinstellungen und den relevanten Protokollauszug an. spacr-doctor erfasst den Großteil dieser Angaben; fügen Sie bei Leistungsproblemen den Hardwarebericht bei.
   :width: 100%
   :target: https://github.com/EinarOlafsson/spacr/issues

.. image:: ../../readme_images/de/licensing.svg
   :alt: Lizenz. spaCR is released under the BSD 3-Clause-Lizenz. Wenn spaCR zu veröffentlichten Arbeiten beigetragen hat, wird ein Zitat geschätzt und ist keine Bedingung der Lizenz — siehe spaCR zitieren unten.
   :width: 100%
   :target: https://github.com/EinarOlafsson/spacr/blob/main/LICENSE

.. image:: ../../readme_images/de/tutorials.svg
   :alt: Tutorials. Die interaktive spaCR-Tutorialbibliothek führt durch die Installation und die Verwendung der Module. Verfügbare Vertonungen und Sprachen werden für jede Lektion angegeben.
   :width: 100%
   :target: https://einarolafsson.github.io/spacr/tutorials/

.. image:: ../../readme_images/de/citing-spacr.svg
   :alt: spaCR zitieren. Wenn spaCR zu Ihrer Forschung beiträgt, zitieren Sie: Olafsson EB, et al. Ein gepoolter Bild-basierter CRISPR Screening identifiziert EAF1 als einen T. gondii Modulator der ESCRT-Subversion.
   :width: 100%

| |citing-spacr-2-1|\ |citing-spacr-2-2|

.. |citing-spacr-2-1| image:: ../../readme_images/de/citing-spacr-2-1.svg
   :alt: Vordruck bioRxiv
   :width: 49.5%
   :target: https://www.biorxiv.org/content/10.64898/2026.07.08.737057v1

.. |citing-spacr-2-2| image:: ../../readme_images/de/citing-spacr-2-2.svg
   :alt: Software-Archiv
   :width: 49.5%
   :target: https://doi.org/10.5281/zenodo.21343316

.. image:: ../../readme_images/de/other-work-citing-spacr.svg
   :alt: Sonstige Arbeiten unter Angabe von spaCR.
   :width: 100%

.. spacr-citing-papers-begin

.. image:: ../../readme_images/de/other-work-citing-spacr-2-1.svg
   :alt: Metabolische Anpassungsfähigkeit und Nährstoffverschrottung in Toxoplasma gondii: Erkenntnisse aus ingestionsweg-defizienten Mutanten.
   :width: 100%
   :target: https://journals.asm.org/doi/full/10.1128/msphere.01011-24

.. image:: ../../readme_images/de/other-work-citing-spacr-2-2.svg
   :alt: IRE1α fördert den Phagosomal-Calcium-Fluss zur Verbesserung der makrophagenfungiziden Aktivität.
   :width: 100%
   :target: https://www.cell.com/cell-reports/fulltext/S2211-1247(25)00465-6

.. image:: ../../readme_images/de/other-work-citing-spacr-2-3.svg
   :alt: Toxoplasma GRA8 greift das Wirts-ESCRT-Zubehörprotein ALG-2 auf und ist für die metabolische Integrität des Parasiten notwendig.
   :width: 100%
   :target: https://www.biorxiv.org/content/10.64898/2026.07.20.739547v1.abstract

.. image:: ../../readme_images/de/other-work-citing-spacr-2-4.svg
   :alt: Hydrogen sulfide dynamically upregulates copper uptake and localization.
   :width: 100%
   :target: https://www.biorxiv.org/content/10.64898/2026.07.30.741779v1

.. image:: ../../readme_images/de/other-work-citing-spacr-2-5.svg
   :alt: spaCR: Räumliche Phänotypanalyse von CRISPR-Cas9-Bildschirmen (Vordruckversion 1).
   :width: 100%
   :target: https://www.researchsquare.com/article/rs-7368254/v1

.. spacr-citing-papers-end

.. image:: ../../readme_images/de/acknowledgments.svg
   :alt: Danksagung. spaCR baut auf offener wissenschaftlicher Software auf, darunter NumPy, pandas, scikit-image, scikit-learn, Cellpose, PyTorch und Qt. Die für die mehrsprachige Dokumentation und die Oberflächenkataloge verwendeten Modelle sind in der Attribution der Übersetzungsmodelle aufgeführt.
   :width: 100%
   :target: ../TRANSLATION_MODELS.md
