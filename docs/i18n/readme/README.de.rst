|Website| |Platforms| |Python| |Qt| |Test counts| |Coverage| |Issues| |Conda| |PyPI| |Docker| |PyPI Downloads| |Docs| |Tutorials| |Preprint| |DOI| |Cite| |License| |PyPI rank|

Website: `einarolafsson.github.io/projects/spacr <https://einarolafsson.github.io/projects/spacr/>`_

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

.. |Coverage| image:: https://img.shields.io/codecov/c/github/EinarOlafsson/spacr/main?logo=codecov&logoColor=white
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

.. raw:: html

   <div><a href="https://einarolafsson.github.io/spacr/_static/deck/"><picture><source media="(min-width: 1280px)" srcset="../../readme_assets/cover.desktop.svg"><img src="../../readme_assets/cover.svg" alt="spaCR"></picture></a></div>

.. raw:: html

   <div><a href="../../source/_static/deck/pages/57.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/de/intro-1.desktop.svg"><img src="../../readme_images/de/intro-1.svg" alt="← Zurück"></picture></a><a href="../../source/_static/deck/pages/02.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/de/intro-2.desktop.svg"><img src="../../readme_images/de/intro-2.svg" alt="Weiter →"></picture></a></div>

.. spacr-language-picker-begin

.. raw:: html

   <div><a href="README.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/de/generated-language-picker.desktop.svg"><img src="../../readme_images/de/generated-language-picker.svg" alt="Sprachen: Deutsch"></picture></a></div>

.. spacr-language-picker-end

.. raw:: html

   <div><a href="../../readme_text/de/spacr.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/de/spacr.desktop.svg"><img src="../../readme_images/de/spacr.svg" alt="spaCR. Räumliche Phänotypanalyse von CRISPR-Screens. spaCR segmentiert und vermisst einzelne Zellen in Mikroskopiebildern, integriert Phänotypen einzelner Objekte mit sequenzierungsbasierten Guide-Häufigkeiten und schätzt, welche Gene mit phänotypischen Veränderungen assoziiert sind. Ausgehend von Plattenbildern und FASTQ-Reads erzeugt es Messungen pro Objekt, trainierte Klassifikatoren, Effektschätzungen pro Guide und Gen sowie eine Rangliste der Treffer. Die Module für Segmentierung, Messung, Annotation und Klassifizierung können auch ohne einen Sequenzierungsteil verwendet werden. Make Masks korrigiert Segmentierungsmasken und annotiert mit dem Werkzeug Box unabhängige Rechtecke mit Klassenlabels für den YOLO-Export. Die Rechtecke behalten ihre eigenen Labels und ihren Verlauf, ohne Quellbilder oder Masken zu ändern. Einzelheiten zu jedem Werkzeug stehen im Funktionsleitfaden. Bilder, Masken und Bildausschnitte sind Dateien auf der Festplatte. Messungen, Annotationen und Modellvorhersagen sind Tabellen in der SQLite-Datenbank measurements.db des Projekts. Läuft als Desktop-Anwendung oder ohne grafische Oberfläche auf einer Workstation, einem Server oder Cluster."></picture></a></div>

.. raw:: html

   <div><a href="../../readme_text/de/try-spacr.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/de/try-spacr.desktop.svg"><img src="../../readme_images/de/try-spacr.svg" alt="spaCR ausprobieren."></picture></a></div>

.. code-block:: bash

   conda create -n spacr python=3.12 -y
   conda activate spacr
   python -m pip install spacr
   spacr

.. raw:: html

   <div><a href="../../readme_text/de/hardware-support.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/de/hardware-support.desktop.svg"><img src="../../readme_images/de/hardware-support.svg" alt="Hardware-Unterstützung."></picture></a></div>

.. spacr-hardware-begin

.. raw:: html

   <div><a href="../../readme_text/de/generated-hardware-1.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/de/generated-hardware-1.desktop.svg"><img src="../../readme_images/de/generated-hardware-1.svg" alt="NVIDIA (CUDA): Cellpose 4 GPU, Torch GPU, UMAP / clustering GPU, 3D GPU. AMD on Linux (ROCm): Cellpose 4 GPU, Torch GPU, UMAP / clustering CPU, 3D GPU. AMD in an Intel Mac (Metal): Cellpose 4 GPU, Torch GPU, UMAP / clustering CPU, 3D GPU. Apple Silicon (Metal): Cellpose 4 GPU, Torch GPU, UMAP / clustering CPU, 3D GPU. Intel Arc/Xe (XPU): Cellpose 4 GPU, Torch GPU, UMAP / clustering CPU, 3D GPU. No GPU: Cellpose 4 CPU, Torch CPU, UMAP / clustering CPU, 3D CPU. unterstützt (stabil) implementiert (Beta) nur CPU-Unterstützung."></picture></a></div>

.. spacr-hardware-end

.. raw:: html

   <div><a href="../../readme_text/de/install-spacr.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/de/install-spacr.desktop.svg"><img src="../../readme_images/de/install-spacr.svg" alt="spaCR installieren. Wählen Sie einen Installationsweg: Desktopanwendung: ein Installationsprogramm für Linux, macOS oder Windows mit eigenem Python. pip: python -m pip install spacr in einer Conda-Umgebung. conda-forge: conda install conda-forge::spacr in einer Conda-Umgebung. Docker: die Kommandozeilen-Pipelines in einem Linux-Container, mit dem CPU-Image oder dem Image für NVIDIA-GPUs:"></picture></a></div>

.. code-block:: bash

   docker run --rm ghcr.io/einarolafsson/spacr:1.5.1.5 spacr-run --list
   docker run --rm --gpus all ghcr.io/einarolafsson/spacr:1.5.1.5-cuda12.4 spacr-run --list

.. raw:: html

   <div><a href="../../readme_text/de/install-spacr-2.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/de/install-spacr-2.desktop.svg"><img src="../../readme_images/de/install-spacr-2.svg" alt="Die Docker-Anleitung (nightly, Englisch) beschreibt die GPU-Einrichtung, das Einbinden von Daten und die Pipeline-Befehle."></picture></a></div>

.. raw:: html

   <div><a href="../../readme_text/de/desktop-application.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/de/desktop-application.desktop.svg"><img src="../../readme_images/de/desktop-application.svg" alt="Desktopanwendung. Die Installationsprogramme enthalten eine eigene Python-Umgebung. Conda ist nicht erforderlich."></picture></a></div>

.. spacr-installer-links-begin

| |InstallerLinux|\ |InstallerMacOS|\ |InstallerWindows|

.. |InstallerWindows| raw:: html

   <a href="https://github.com/EinarOlafsson/spacr/releases/download/v1.5.1.5/spaCR-1.5.1.5-Windows-Online-Setup.exe"><img src="../../readme_assets/cells/platforms/windows.png" width="14.14%" align="middle" alt="Windows 10/11: spaCR 1.5.1.5 herunterladen"></a>
.. |InstallerMacOS| raw:: html

   <a href="https://github.com/EinarOlafsson/spacr/releases/download/v1.5.1.5/spaCR-1.5.1.5-macOS-Universal-Online.pkg"><img src="../../readme_assets/cells/platforms/macos.png" width="14.14%" align="middle" alt="macOS 11+ (Intel und Apple Silicon): spaCR 1.5.1.5 herunterladen"></a>
.. |InstallerLinux| raw:: html

   <a href="https://github.com/EinarOlafsson/spacr/releases/download/v1.5.1.5/spaCR-1.5.1.5-Linux-x86_64-Online.run"><img src="../../readme_assets/cells/platforms/linux.png" width="14.14%" align="middle" alt="64-Bit-Linux: spaCR 1.5.1.5 herunterladen"></a>

.. spacr-installer-links-end

.. raw:: html

   <div><a href="../../readme_text/de/desktop-application-2.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/de/desktop-application-2.desktop.svg"><img src="../../readme_images/de/desktop-application-2.svg" alt="Machen Sie die heruntergeladene Datei unter Linux ausführbar und führen Sie sie aus:"></picture></a></div>

.. code-block:: bash

   chmod +x SpaCR-*-Linux-x86_64-Online.run
   ./SpaCR-*-Linux-x86_64-Online.run

.. raw:: html

   <div><a href="../../readme_text/de/desktop-application-3.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/de/desktop-application-3.desktop.svg"><img src="../../readme_images/de/desktop-application-3.svg" alt="Öffnen Sie unter macOS die Datei .pkg. Die aktuelle Beta ist nicht notarisiert. Wenn Gatekeeper sie blockiert, wählen Sie Systemeinstellungen → Datenschutz &amp; Sicherheit → Dennoch öffnen. Anleitungen zur Installation, Aktualisierung, Neuinstallation, Deinstallation, zum Offlinebetrieb und zur Fehlerbehebung finden Sie in der aktuellen Installationsanleitung auf nightly (Englisch). Die Systemanforderungen enthalten Empfehlungen für Arbeitsstationen und Server sowie Tabellen zur GPU-Kompatibilität."></picture></a></div>

.. raw:: html

   <div><a href="https://github.com/EinarOlafsson/spacr/blob/nightly/docs/source/installer_guide.rst"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/de/desktop-application-3-link-1.desktop.svg"><img src="../../readme_images/de/desktop-application-3-link-1.svg" alt="aktuellen Installationsanleitung auf nightly (Englisch)"></picture></a><a href="../../source/system_requirements.rst"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/de/desktop-application-3-link-2.desktop.svg"><img src="../../readme_images/de/desktop-application-3-link-2.svg" alt="Systemanforderungen"></picture></a></div>

.. raw:: html

   <div><a href="../../readme_text/de/pypi-installation.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/de/pypi-installation.desktop.svg"><img src="../../readme_images/de/pypi-installation.svg" alt="Installation über PyPI. Installieren Sie die PyPI-Veröffentlichung von spaCR mit pip in einer Conda-Umgebung. Python 3.12 bietet die größte Auswahl an optionalen wissenschaftlichen Paketen:"></picture></a></div>

.. code-block:: bash

   conda create -n spacr python=3.12 -y
   conda activate spacr
   python -m pip install --upgrade pip
   python -m pip install spacr
   spacr

.. raw:: html

   <div><a href="../../readme_text/de/pypi-installation-2.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/de/pypi-installation-2.desktop.svg"><img src="../../readme_images/de/pypi-installation-2.svg" alt="spaCR unterstützt Python 3.9 through 3.14, außer Python 3.14.1, das von torchvision ausgeschlossen wird. Linux wird für die anspruchsvollsten CUDA- und ROCm-Workflows empfohlen; macOS und Windows werden ebenfalls unterstützt und nutzen beide ihre GPUs — macOS über Metal, das Apple Silicon und die AMD-Karten in Intel Macs abdeckt, und Windows über CUDA oder DirectML. Die Standardinstallation enthält die Qt-Desktopoberfläche. Auf einem Server, Cluster oder CI-Runner können Sie die Befehlszeilen-Pipelines ausführen, ohne sie zu öffnen:"></picture></a></div>

.. code-block:: bash

   python -m pip install spacr
   spacr-run --list

.. raw:: html

   <div><a href="../../readme_text/de/pypi-installation-3.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/de/pypi-installation-3.desktop.svg"><img src="../../readme_images/de/pypi-installation-3.svg" alt="Optionale Integrationen werden separat installiert, zum Beispiel spacr[zarr], spacr[omero], spacr[napari] und spacr[czi,nd2,lif]. Die vollständige Liste der Extras und die Kompatibilitätstabelle für Python-Versionen finden Sie in der Installationsanleitung (nightly, Englisch)."></picture></a></div>

.. raw:: html

   <div><a href="../../readme_text/de/conda-forge-installation.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/de/conda-forge-installation.desktop.svg"><img src="../../readme_images/de/conda-forge-installation.svg" alt="Installation mit conda-forge. Das offizielle conda-forge-Paket installiert spaCR und seine Desktop-Abhängigkeiten in der aktiven Umgebung:"></picture></a></div>

.. code-block:: bash

   conda create -n spacr python=3.12 -y
   conda activate spacr
   conda install conda-forge::spacr
   spacr

.. raw:: html

   <div><a href="../../readme_text/de/install-from-source.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/de/install-from-source.desktop.svg"><img src="../../readme_images/de/install-from-source.svg" alt="Installation aus dem Quellcode. Klonen Sie das Repository und installieren Sie es im editierbaren Modus. Ihre Arbeitskopie ist dann das installierte Paket, und Änderungen werden ohne Neuinstallation wirksam:"></picture></a></div>

.. code-block:: bash

   git clone https://github.com/EinarOlafsson/spacr.git
   cd spacr
   conda create -n spacr python=3.12 -y
   conda activate spacr
   pip install -e .
   spacr

.. raw:: html

   <div><a href="../../readme_text/de/install-from-source-2.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/de/install-from-source-2.desktop.svg"><img src="../../readme_images/de/install-from-source-2.svg" alt="Dieser Befehl klont main, den Standard-Branch mit der neuesten Version. Entwickelt wird auf nightly; fügen Sie --branch nightly hinzu, um stattdessen diesen Branch zu klonen. Für eine bestimmte Version:"></picture></a></div>

.. code-block:: bash

   git clone --branch v1.5.0.5 https://github.com/EinarOlafsson/spacr.git

.. raw:: html

   <div><a href="../../readme_text/de/install-from-source-3.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/de/install-from-source-3.desktop.svg"><img src="../../readme_images/de/install-from-source-3.svg" alt="Um spätere Änderungen zu übernehmen, führen Sie im Klon aus:"></picture></a></div>

.. code-block:: bash

   git pull
   pip install -e .

.. raw:: html

   <div><a href="../../readme_text/de/install-from-source-4.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/de/install-from-source-4.desktop.svg"><img src="../../readme_images/de/install-from-source-4.svg" alt="Installieren Sie neu, wenn sich Abhängigkeiten oder Einstiegspunkte ändern. Änderungen am Python-Code gelten direkt; spacr-doctor zeigt die aktive Installation an."></picture></a></div>

.. raw:: html

   <div><a href="../../readme_text/de/install-from-source-light.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/de/install-from-source-light.desktop.svg"><img src="../../readme_images/de/install-from-source-light.svg" alt="Installation aus dem Quellcode (schlank). Mitwirkende benötigen die Versionshistorie; um spaCR nur auszuführen, wählen Sie eine der folgenden Optionen. Messungen von nightly bei 05302fd5c am 2026-10-07 mit packaging/measure_clone_forms.sh. Flacher Klon: 2048 MB in 140 s heruntergeladen. Nur der neueste Commit bleibt erhalten; ältere Historie steht git log, git blame und git bisect daher nicht zur Verfügung. git pull funktioniert, der Klon bleibt jedoch flach; git fetch --unshallow lädt die vollständige Historie."></picture></a></div>

.. code-block:: bash

   git clone --depth 1 --branch nightly https://github.com/EinarOlafsson/spacr.git
   cd spacr && pip install -e .

.. raw:: html

   <div><a href="../../readme_text/de/install-from-source-light-2.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/de/install-from-source-light-2.desktop.svg"><img src="../../readme_images/de/install-from-source-light-2.svg" alt="Nur Laufzeitdateien: 104 MB auf dem Datenträger in 6 s (Git: 38 MB; Dateien: 66 MB). Historie, docs, tests, tools, features, Beispieldaten und erweiterte Oberflächenübersetzungen werden ausgelassen. --with-docs, --with-tests und --with-translations behalten die entsprechende Dokumentation, Tests und Übersetzungen bei. Die kompakten Oberflächenübersetzungen bleiben verfügbar."></picture></a></div>

.. code-block:: bash

   curl -fsSL https://raw.githubusercontent.com/EinarOlafsson/spacr/nightly/packaging/install_from_source.sh -o install_spacr.sh
   sh install_spacr.sh --branch nightly

.. raw:: html

   <div><a href="../../readme_text/de/install-from-source-light-3.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/de/install-from-source-light-3.desktop.svg"><img src="../../readme_images/de/install-from-source-light-3.svg" alt="--dir wählt den Pfad des Klons; --branch wählt einen Branch oder Tag; --no-install lädt die Dateien ohne Installation; --help zeigt die Optionen. packaging/source_install_excludes.txt führt alle ausgelassenen Pfade auf. Der vollständige nightly-Klon lud 9.25 GiB herunter. --filter=blob:none zum flachen Klon hinzuzufügen verkleinert das ausgecheckte Arbeitsverzeichnis nicht: Sein Git-Objektspeicher umfasst weiterhin 2032 MB. Git meldet bei Bedarf nachgeladene Objekte nicht, daher wurde keine Gesamtgröße des Downloads gemessen. Die versionierten Dateien von nightly belegen im ausgecheckten Arbeitsverzeichnis 720 MB (gemessen am 2026-10-21), ohne Git-Historie. Downloadgröße und Dauer hängen vom Branch ab."></picture></a></div>

.. raw:: html

   <div><a href="../../readme_text/de/command-line-entry-points.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/de/command-line-entry-points.desktop.svg"><img src="../../readme_images/de/command-line-entry-points.svg" alt="Befehle für die Kommandozeile."></picture></a></div>

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

   <div><a href="../../readme_text/de/command-line-entry-points-2.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/de/command-line-entry-points-2.desktop.svg"><img src="../../readme_images/de/command-line-entry-points-2.svg" alt="spacr-run --list listet Module mit Befehlszeileneinstiegspunkten für die Ausführung ohne grafische Oberfläche auf. Reine GUI-Module für Annotation, Kuratierung, Vergleich und Exploration werden nicht aufgeführt."></picture></a></div>

.. raw:: html

   <div><a href="../../readme_text/de/core-workflow.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/de/core-workflow.desktop.svg"><img src="../../readme_images/de/core-workflow.svg" alt="Kern-Workflow. Der primäre Arbeitsablauf umfasst sechs Module: Mask segmentiert Zellen, Zellkerne, Pathogene und Organellen mit Cellpose. Measure schreibt Morphologie-, Intensitäts-, Textur-, räumliche und Kolokalisationsmerkmale sowie Objektausschnitte nach SQLite. Annotate beschriftet Objektausschnitte in einem tastaturgesteuerten Raster und unterstützt Active-Learning-Warteschlangen. Classify trainiert bild- oder messwertbasierte Modelle und speichert mit jedem Checkpoint die Leistung auf zurückgehaltenen Daten. Map Barcodes ordnet FASTQ-Reads Wells und gRNAs zu und liefert QC für Häufigkeit, Kollisionen und Abdeckung. Regression schätzt Guide-, Gen-, Bedingungs- und Kontrolleffekte mit Modellfamilien für kontinuierliche Werte, Anteile und Zähldaten."></picture></a></div>

.. raw:: html

   <div><a href="../../readme_text/de/spacr-modules.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/de/spacr-modules.desktop.svg"><img src="../../readme_images/de/spacr-modules.svg" alt="spaCR-Module."></picture></a></div>

.. spacr-workflow-begin

.. raw:: html

   <div><a href="../../readme_text/de/generated-workflow-1.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/de/generated-workflow-1.desktop.svg"><img src="../../readme_images/de/generated-workflow-1.svg" alt="Kern. Core sequence from microscopy images through segmentation, measurements, annotations, classification, barcode mapping and regression."></picture></a></div>

| |Module_mask|\ |Module_measure|\ |Module_annotate|\ |Module_classify_merged|\ |Module_map_barcodes|\ |Module_regression|

.. raw:: html

   <div><a href="../../readme_text/de/generated-workflow-2.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/de/generated-workflow-2.desktop.svg"><img src="../../readme_images/de/generated-workflow-2.svg" alt="Daten. Import images and tables into spaCR projects and execute reproducible multi-plate workflows."></picture></a></div>

| |Module_foreign|\ |Module_embeddings|\ |Module_run_compare|\ |Module_experiment_design|\ |Module_power|\ |Module_dose_response|\ |Module_qc_dashboard|

.. raw:: html

   <div><a href="../../readme_text/de/generated-workflow-3.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/de/generated-workflow-3.desktop.svg"><img src="../../readme_images/de/generated-workflow-3.svg" alt="Werkzeuge. Point these at a project: edit masks by hand, stitch tiles, read an embedding, draw a gate, build a plot, check quality."></picture></a></div>

| |Module_make_masks|\ |Module_align|\ |Module_umap|\ |Module_gate_editor|\ |Module_graph_builder|

.. raw:: html

   <div><a href="../../readme_text/de/generated-workflow-4.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/de/generated-workflow-4.desktop.svg"><img src="../../readme_images/de/generated-workflow-4.svg" alt="Organismus. Organismusspezifische Bildanalyse und quantitative Assay-Ergebnisse."></picture></a></div>

| |Module_toxoplasma|

.. |Module_mask| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/core/index.html#spacr.core.preprocess_generate_masks"><img src="../../readme_assets/cells/workflow/mask.png" width="14.14%" align="middle" alt="API für Mask öffnen"></a>

.. |Module_measure| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/measure/index.html"><img src="../../readme_assets/cells/workflow/measure.png" width="14.14%" align="middle" alt="API für Measure öffnen"></a>

.. |Module_annotate| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/qt/screens/annotate/index.html"><img src="../../readme_assets/cells/workflow/annotate.png" width="14.14%" align="middle" alt="API für Annotate öffnen"></a>

.. |Module_classify_merged| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/classify/index.html"><img src="../../readme_assets/cells/workflow/classify_merged.png" width="14.14%" align="middle" alt="API für Classify öffnen"></a>

.. |Module_map_barcodes| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/sequencing/index.html"><img src="../../readme_assets/cells/workflow/map_barcodes.png" width="14.14%" align="middle" alt="API für Map Barcodes öffnen"></a>

.. |Module_regression| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/ml/index.html"><img src="../../readme_assets/cells/workflow/regression.png" width="14.14%" align="middle" alt="API für Regression öffnen"></a>

.. |Module_foreign| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/foreign/index.html"><img src="../../readme_assets/cells/workflow/apps/foreign.png" width="14.14%" align="middle" alt="API für Import öffnen"></a>

.. |Module_embeddings| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/qt/screens/embeddings/index.html"><img src="../../readme_assets/cells/workflow/apps/embeddings.png" width="14.14%" align="middle" alt="API für Embeddings öffnen"></a>

.. |Module_run_compare| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/qt/screens/run_compare/index.html"><img src="../../readme_assets/cells/workflow/apps/run_compare.png" width="14.14%" align="middle" alt="API für Run Compare öffnen"></a>

.. |Module_experiment_design| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/qt/screens/experiment_design/index.html"><img src="../../readme_assets/cells/workflow/apps/experiment_design.png" width="14.14%" align="middle" alt="API für Experiment Design öffnen"></a>

.. |Module_power| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/qt/screens/power/index.html"><img src="../../readme_assets/cells/workflow/apps/power.png" width="14.14%" align="middle" alt="API für Power / Design öffnen"></a>

.. |Module_dose_response| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/qt/screens/dose_response/index.html"><img src="../../readme_assets/cells/workflow/apps/dose_response.png" width="14.14%" align="middle" alt="API für Dose–Response öffnen"></a>

.. |Module_qc_dashboard| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/qt/screens/qc_dashboard/index.html"><img src="../../readme_assets/cells/workflow/apps/qc_dashboard.png" width="14.14%" align="middle" alt="API für QC öffnen"></a>

.. |Module_make_masks| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/qt/screens/make_masks/index.html"><img src="../../readme_assets/cells/workflow/apps/make_masks.png" width="14.14%" align="middle" alt="API für Make Masks öffnen"></a>

.. |Module_align| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/align/index.html"><img src="../../readme_assets/cells/workflow/apps/align.png" width="14.14%" align="middle" alt="API für Align &amp; Stitch öffnen"></a>

.. |Module_umap| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/core/index.html#spacr.core.generate_image_umap"><img src="../../readme_assets/cells/workflow/apps/umap.png" width="14.14%" align="middle" alt="API für Image UMAP öffnen"></a>

.. |Module_gate_editor| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/qt/screens/gate_editor/index.html"><img src="../../readme_assets/cells/workflow/apps/gate_editor.png" width="14.14%" align="middle" alt="API für Gate Editor öffnen"></a>

.. |Module_graph_builder| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/qt/screens/graph_builder/index.html"><img src="../../readme_assets/cells/workflow/apps/graph_builder.png" width="14.14%" align="middle" alt="API für Graph Builder öffnen"></a>

.. |Module_toxoplasma| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/qt/screens/organism_screen/index.html#spacr-qt-screens-organism-screen-toxoplasma"><img src="../../readme_assets/cells/workflow/apps/toxoplasma.png" width="14.14%" align="middle" alt="API für Toxoplasma öffnen"></a>

.. spacr-workflow-end

.. raw:: html

   <div><a href="../../readme_text/de/spacr-modules-2.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/de/spacr-modules-2.desktop.svg"><img src="../../readme_images/de/spacr-modules-2.svg" alt="Alle Module mit einer Kachel auf der Startseite, in deren Reihenfolge: zuerst die sechs Pipeline-Module, dann die übrigen. Wählen Sie eine Kachel, um die API-Seite des Moduls zu öffnen."></picture></a></div>

.. raw:: html

   <div><a href="../../readme_text/de/other-resources.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/de/other-resources.desktop.svg"><img src="../../readme_images/de/other-resources.svg" alt="Weitere Ressourcen."></picture></a></div>

.. raw:: html

   <div><a href="../../readme_text/de/other-resources-2-1.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/de/other-resources-2-1.desktop.svg"><img src="../../readme_images/de/other-resources-2-1.svg" alt="Interaktive Tutorials — geführte Arbeitsabläufe von der Installation bis zur Untersuchung von Treffern."></picture></a></div>

.. raw:: html

   <div><a href="../../readme_text/de/other-resources-2-2.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/de/other-resources-2-2.desktop.svg"><img src="../../readme_images/de/other-resources-2-2.svg" alt="Python-API-Schnellstart — Pipelines aus Skripten, Notebooks oder auf einem Cluster ausführen und validieren."></picture></a></div>

.. raw:: html

   <div><a href="../../readme_text/de/other-resources-2-3.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/de/other-resources-2-3.desktop.svg"><img src="../../readme_images/de/other-resources-2-3.svg" alt="Funktionsleitfaden — Fähigkeiten, Reifegrad und optionale Integrationen."></picture></a></div>

.. raw:: html

   <div><a href="../../readme_text/de/other-resources-2-4.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/de/other-resources-2-4.desktop.svg"><img src="../../readme_images/de/other-resources-2-4.svg" alt="Kuratierte API-Referenz — unterstützte Einstiegspunkte nach Aufgabe; die vollständige Modulreferenz liegt eine Ebene tiefer."></picture></a></div>

.. raw:: html

   <div><a href="../../readme_text/de/other-resources-2-5.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/de/other-resources-2-5.desktop.svg"><img src="../../readme_images/de/other-resources-2-5.svg" alt="Sprach- und Übersetzungsleitfaden — Oberflächensprachen, kontextbezogene Hilfe und Richtlinien für wissenschaftliche Ausgaben."></picture></a></div>

.. raw:: html

   <div><a href="../../readme_text/de/language-translation.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/de/language-translation.desktop.svg"><img src="../../readme_images/de/language-translation.svg" alt="Sprache und Übersetzung. Die Oberfläche unterstützt zehn Sprachen in der Navigation und den Einstellungen. AI- und LIVE-Steuerelemente, Modulbeschreibungen und geprüfte Kontexthilfe werden ebenfalls übersetzt. Ändern Sie die Sprache unter spaCR → Einstellungen → Sprache, ohne neu zu starten. Protokolle, Pfade, Datenbankwerte und Messungen werden nie übersetzt; wissenschaftliche Ausgaben bleiben im kanonischen Englisch. Siehe die Richtlinie zur Kontexthilfe."></picture></a></div>

.. raw:: html

   <div><a href="../../readme_text/de/animated-setting-guidance.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/de/animated-setting-guidance.desktop.svg"><img src="../../readme_images/de/animated-setting-guidance.svg" alt="Animierte Einstellungshilfe. Einstellungen mit einer visuellen Erklärung bieten in ihrem Tooltip die Schaltfläche Animation. Durchsuchen Sie die Galerie der Einstellungsanimationen oder das Register der Einstellungsanimationen."></picture></a></div>

.. raw:: html

   <div><a href="https://einarolafsson.github.io/spacr/setting_animations.html"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/de/animated-setting-guidance-link-1.desktop.svg"><img src="../../readme_images/de/animated-setting-guidance-link-1.svg" alt="Galerie der Einstellungsanimationen"></picture></a><a href="https://einarolafsson.github.io/spacr/api/spacr/setting_animations/index.html"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/de/animated-setting-guidance-link-2.desktop.svg"><img src="../../readme_images/de/animated-setting-guidance-link-2.svg" alt="Register der Einstellungsanimationen"></picture></a></div>

.. raw:: html

   <div><a href="../../readme_text/de/data.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/de/data.desktop.svg"><img src="../../readme_images/de/data.svg" alt="Daten. Referenzdatensätze."></picture></a></div>

| |DataBioStudies|\ |DataHuggingFace|\ |DataNCBI|\ |DataSpaCRPower|\ |DataBioRxiv|

.. |DataBioStudies| raw:: html

   <a href="https://doi.org/10.6019/S-BIAD2135"><img src="../../readme_assets/cells/databanks/biostudies_button.png" width="14.14%" align="middle" alt="Mikroskopiedatensatz in BioStudies öffnen"></a>

.. |DataHuggingFace| raw:: html

   <a href="https://huggingface.co/datasets/einarolafsson/toxo_mito"><img src="../../readme_assets/cells/databanks/huggingface_button.png" width="14.14%" align="middle" alt="Testdatensatz auf Hugging Face öffnen"></a>

.. |DataNCBI| raw:: html

   <a href="https://www.ncbi.nlm.nih.gov/bioproject/?term=PRJNA1261935"><img src="../../readme_assets/cells/databanks/ncbi_button.png" width="14.14%" align="middle" alt="Sequenzierungsdatensatz bei NCBI öffnen"></a>

.. |DataSpaCRPower| raw:: html

   <a href="https://github.com/maomlab/spaCRPower"><img src="../../readme_assets/cells/databanks/spacrpower_button.png" width="14.14%" align="middle" alt="spaCRPower öffnen"></a>

.. |DataBioRxiv| raw:: html

   <a href="https://www.biorxiv.org/content/10.64898/2026.07.08.737057v1"><img src="../../readme_assets/cells/databanks/biorxiv_button.png" width="14.14%" align="middle" alt="bioRxiv-Preprint öffnen"></a>

.. raw:: html

   <div><a href="../../readme_text/de/model-zoo.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/de/model-zoo.desktop.svg"><img src="../../readme_images/de/model-zoo.svg" alt="Modellzoo. spaCR enthält einen Katalog trainierter Modelle und lädt sie bei Bedarf herunter. Öffnen Sie Model Zoo auf der Startseite, um Modelle zu durchsuchen und zu installieren, oder geben Sie einen Modellschlüssel in einer Einstellungsdatei an, etwa pathogen_model: toxoplasma_pv_v1. Das Modell wird bei der ersten Verwendung heruntergeladen und anhand seiner Prüfsumme verifiziert. Jeder veröffentlichte Eintrag enthält eine SHA-256-Prüfsumme. Ein Eintrag ohne Prüfsumme wird abgelehnt, da eine unvollständige oder ausgetauschte Checkpoint-Datei sonst nicht vom Original unterschieden werden kann."></picture></a></div>

.. spacr-model-zoo-begin

.. raw:: html

   <div><a href="../../models/README.md"><picture><source media="(min-width: 1280px)" srcset="../../models/cards/heading.svg"><img src="../../models/cards/heading.phone.svg" alt="Default models · held-out scores"></picture></a></div>
   <div><a href="../../models/toxoplasma_pv_v4.md"><picture><source media="(min-width: 1280px)" srcset="../../models/cards/toxoplasma_pv_v4.svg"><img src="../../models/cards/toxoplasma_pv_v4.phone.svg" alt="Score card of Parasite vacuoles v4 (toxoplasma_pv_v4, default): F1 0.761 on 209 held-out fields against 0.377 for stock cpsam_v2. F1 0.761 against stock cpsam_v2&#x27;s 0.377 on the 209 strict hold-out fields at IoU 0.5; AJI 0.582 against 0.040."></picture></a><a href="../../models/toxoplasma_plaque_v5.md"><picture><source media="(min-width: 1280px)" srcset="../../models/cards/toxoplasma_plaque_v5.svg"><img src="../../models/cards/toxoplasma_plaque_v5.phone.svg" alt="Score card of Plaques v5 (toxoplasma_plaque_v5, default): F1 0.816 on 153 held-out fields against 0.227 for stock cpsam_v2. F1 0.816 against stock cpsam_v2&#x27;s 0.227 on the 153 strict hold-out fields at IoU 0.5; AJI 0.737 against 0.178."></picture></a><a href="../../models/toxoplasma_well_detector_v3.md"><picture><source media="(min-width: 1280px)" srcset="../../models/cards/toxoplasma_well_detector_v3.svg"><img src="../../models/cards/toxoplasma_well_detector_v3.phone.svg" alt="Score card of Plaque wells v3 (toxoplasma_well_detector_v3, default): mAP50 0.994 on 73 held-out images. Stock was not evaluated in this run; see the named incumbent comparison on the model card."></picture></a><a href="../../models/toxoplasma_from_cellmask_v1.md"><picture><source media="(min-width: 1280px)" srcset="../../models/cards/toxoplasma_from_cellmask_v1.svg"><img src="../../models/cards/toxoplasma_from_cellmask_v1.phone.svg" alt="Score card of Vacuoles from cell mask v1 (toxoplasma_from_cellmask_v1, default): F1 0.606 on 463 held-out fields against 0.021 for stock cpsam_v2. F1 0.606 against 0.021 for stock cpsam_v2 on 463 well-grouped held-out fields, at IoU 0.5."></picture></a><a href="../../models/nuclei_from_cellmask_v1.md"><picture><source media="(min-width: 1280px)" srcset="../../models/cards/nuclei_from_cellmask_v1.svg"><img src="../../models/cards/nuclei_from_cellmask_v1.phone.svg" alt="Score card of Nuclei from cell mask v1 (nuclei_from_cellmask_v1, default): F1 0.888 on 453 held-out fields against 0.201 for stock cpsam_v2. F1 0.888 against 0.201 for stock cpsam_v2 on 453 well-grouped held-out fields, at IoU 0.5."></picture></a><a href="../../models/cell_from_hoechst_v1.md"><picture><source media="(min-width: 1280px)" srcset="../../models/cards/cell_from_hoechst_v1.svg"><img src="../../models/cards/cell_from_hoechst_v1.phone.svg" alt="Score card of Cells from Hoechst v1 (cell_from_hoechst_v1, default): F1 0.870 on 451 held-out fields against 0.301 for stock cpsam_v2. F1 0.870 against stock cpsam_v2&#x27;s 0.301 on 451 held-out fields at IoU 0.5 -- a delta of 0.569."></picture></a><a href="../../models/toxoplasma_from_hoechst_v1.md"><picture><source media="(min-width: 1280px)" srcset="../../models/cards/toxoplasma_from_hoechst_v1.svg"><img src="../../models/cards/toxoplasma_from_hoechst_v1.phone.svg" alt="Score card of Vacuoles from Hoechst v1 (toxoplasma_from_hoechst_v1, default): F1 0.569 on 463 held-out fields against 0.002 for stock cpsam_v2. F1 0.569 against 0.002 for stock cpsam_v2 on 463 well-grouped held-out fields, at IoU 0.5."></picture></a></div>

.. raw:: html

   <div><a href="../../readme_text/de/generated-model-zoo-1.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/de/generated-model-zoo-1.desktop.svg"><img src="../../readme_images/de/generated-model-zoo-1.svg" alt="Ältere Modelle."></picture></a></div>

.. raw:: html

   <div><a href="../../models/toxoplasma_pv_v3.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/de/generated-model-zoo-1-link-1.desktop.svg"><img src="../../readme_images/de/generated-model-zoo-1-link-1.svg" alt="toxoplasma_pv_v3"></picture></a><a href="../../models/toxoplasma_pv_v2.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/de/generated-model-zoo-1-link-2.desktop.svg"><img src="../../readme_images/de/generated-model-zoo-1-link-2.svg" alt="toxoplasma_pv_v2"></picture></a><a href="../../models/toxoplasma_pv_v1.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/de/generated-model-zoo-1-link-3.desktop.svg"><img src="../../readme_images/de/generated-model-zoo-1-link-3.svg" alt="toxoplasma_pv_v1"></picture></a><a href="../../models/toxoplasma_plaque_v4.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/de/generated-model-zoo-1-link-4.desktop.svg"><img src="../../readme_images/de/generated-model-zoo-1-link-4.svg" alt="toxoplasma_plaque_v4"></picture></a><a href="../../models/toxoplasma_plaque_v2.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/de/generated-model-zoo-1-link-5.desktop.svg"><img src="../../readme_images/de/generated-model-zoo-1-link-5.svg" alt="toxoplasma_plaque_v2"></picture></a><a href="../../models/toxoplasma_plaque_v1.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/de/generated-model-zoo-1-link-6.desktop.svg"><img src="../../readme_images/de/generated-model-zoo-1-link-6.svg" alt="toxoplasma_plaque_v1"></picture></a><a href="../../models/toxoplasma_well_detector_v2.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/de/generated-model-zoo-1-link-7.desktop.svg"><img src="../../readme_images/de/generated-model-zoo-1-link-7.svg" alt="toxoplasma_well_detector_v2"></picture></a><a href="../../models/toxoplasma_well_detector_v1.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/de/generated-model-zoo-1-link-8.desktop.svg"><img src="../../readme_images/de/generated-model-zoo-1-link-8.svg" alt="toxoplasma_well_detector_v1"></picture></a></div>

.. raw:: html

   <div><a href="../../readme_text/de/generated-model-zoo-2.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/de/generated-model-zoo-2.desktop.svg"><img src="../../readme_images/de/generated-model-zoo-2.svg" alt="Alternativen."></picture></a></div>

.. raw:: html

   <div><a href="../../models/toxoplasma_pv_v5.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/de/generated-model-zoo-2-link-1.desktop.svg"><img src="../../readme_images/de/generated-model-zoo-2-link-1.svg" alt="toxoplasma_pv_v5"></picture></a><a href="../../models/toxoplasma_plaque_v3.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/de/generated-model-zoo-2-link-2.desktop.svg"><img src="../../readme_images/de/generated-model-zoo-2-link-2.svg" alt="toxoplasma_plaque_v3"></picture></a><a href="../../models/live_cell_v1.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/de/generated-model-zoo-2-link-3.desktop.svg"><img src="../../readme_images/de/generated-model-zoo-2-link-3.svg" alt="live_cell_v1"></picture></a></div>

.. raw:: html

   <div><a href="../../readme_text/de/generated-model-zoo-3.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/de/generated-model-zoo-3.desktop.svg"><img src="../../readme_images/de/generated-model-zoo-3.svg" alt="Maskengenerator"></picture></a></div>

.. spacr-model-zoo-end

.. raw:: html

   <div><a href="../../readme_text/de/diagnosing-performance.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/de/diagnosing-performance.desktop.svg"><img src="../../readme_images/de/diagnosing-performance.svg" alt="Leistungsdiagnose. Erzeugen Sie einen Hardwarebericht und fügen Sie ihn einem leistungsbezogenen GitHub-Issue bei:"></picture></a></div>

.. code-block:: bash

   python tools/spacr_hardware_report.py

.. raw:: html

   <div><a href="../../readme_text/de/diagnosing-performance-2.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/de/diagnosing-performance-2.desktop.svg"><img src="../../readme_images/de/diagnosing-performance-2.svg" alt="Speichert unter ~/.spacr/reports und gibt den Pfad aus. --quick überspringt die längeren Leistungstests; --out PATH legt den Speicherort fest. Liest keine Projektdaten. Misst die Dauer von Importen, numerischen Bibliotheksoperationen, Fensteraufbau und Animation und meldet x86_64-Emulation auf Apple Silicon sowie die BLAS-Implementierung von NumPy."></picture></a></div>

.. raw:: html

   <div><a href="../../readme_text/de/command-line-reference.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/de/command-line-reference.desktop.svg"><img src="../../readme_images/de/command-line-reference.svg" alt="Befehlszeilenreferenz. Alle folgenden Befehle werden mit pip install spacr installiert und unterstützen --help."></picture></a></div>

.. raw:: html

   <div><a href="../../readme_text/de/launching-the-application.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/de/launching-the-application.desktop.svg"><img src="../../readme_images/de/launching-the-application.svg" alt="Start der Anwendung."></picture></a></div>

.. code-block:: bash

   spacr              # the desktop application
   spacr-tutorial     # the interactive tutorial library
   spacr-server       # no first-run setup screen, for unattended launches

.. raw:: html

   <div><a href="../../readme_text/de/when-spacr-will-not-start.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/de/when-spacr-will-not-start.desktop.svg"><img src="../../readme_images/de/when-spacr-will-not-start.svg" alt="Wenn spaCR nicht startet."></picture></a></div>

.. code-block:: bash

   spacr-doctor       # diagnose the installation and say how to fix it
   safespacr          # the least spaCR that can still change a setting

.. raw:: html

   <div><a href="../../readme_text/de/when-spacr-will-not-start-2.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/de/when-spacr-will-not-start-2.desktop.svg"><img src="../../readme_images/de/when-spacr-will-not-start-2.svg" alt="spacr-doctor gibt pro Prüfung eine Zeile aus, mit einem Befehl, der bei jedem Fehler auszuführen ist. Außerdem meldet es, welches spacr im Pfad liegt. safespacr liest jede Einstellung mit ihrem Standardwert und schaltet Hintergrund, Animationen, ausführliche Protokollierung und Vorabladen ab. Verwenden Sie es, wenn eine gespeicherte Einstellung den Start verhindert. Im abgesicherten Modus gespeicherte Einstellungen werden normal geschrieben und bleiben erhalten."></picture></a></div>

.. raw:: html

   <div><a href="../../readme_text/de/running-modules-headlessly.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/de/running-modules-headlessly.desktop.svg"><img src="../../readme_images/de/running-modules-headlessly.svg" alt="Module ohne grafische Oberfläche ausführen. Kein Qt, kein Display — für Cluster, Server und CI."></picture></a></div>

.. code-block:: bash

   spacr-run --list                              # modules with a headless entry
   spacr-run --describe MODULE                   # what a module consumes and produces
   spacr-run validate --module MODULE \
       --settings settings.csv                   # check settings before spending the run
   spacr-run MODULE --settings settings.csv      # execute
   spacr-remote --help                           # submit and monitor SSH, Slurm or cloud jobs

.. raw:: html

   <div><a href="../../readme_text/de/running-modules-headlessly-2.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/de/running-modules-headlessly-2.desktop.svg"><img src="../../readme_images/de/running-modules-headlessly-2.svg" alt="validate liest dieselben Einstellungen wie der Lauf und listet jedes Problem auf, etwa eine fehlende Einstellung oder einen nicht vorhandenen Pfad, jeweils mit einer Abhilfe."></picture></a></div>

.. raw:: html

   <div><a href="../../readme_text/de/inspecting-a-run.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/de/inspecting-a-run.desktop.svg"><img src="../../readme_images/de/inspecting-a-run.svg" alt="Einen Lauf untersuchen. Jeder Lauf wird unter ~/.spacr/runs mit Einstellungen, Hashwerten der Eingaben, Ausgaben, Warnungen, Versionen und Zufallsstartwerten protokolliert."></picture></a></div>

.. code-block:: bash

   spacr-repro RUN_DIR        # replay a recorded run from its journal
   spacr-workspace RUN_DIR    # what that run had open: databases, montages, views

.. raw:: html

   <div><a href="../../readme_text/de/auditing-data-and-installation.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/de/auditing-data-and-installation.desktop.svg"><img src="../../readme_images/de/auditing-data-and-installation.svg" alt="Daten und Installation prüfen."></picture></a></div>

.. code-block:: bash

   spacr-db-audit DB      # SQLite health, integrity, locking, reader/writer probe
   spacr-leakage          # classifier train/test leakage audit
   spacr-plugins          # installed plugin registry and failure diagnostics

.. raw:: html

   <div><a href="../../readme_text/de/environment.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/de/environment.desktop.svg"><img src="../../readme_images/de/environment.svg" alt="Umgebung."></picture></a></div>

.. code-block:: bash

   SPACR_LOG_LEVEL=DEBUG spacr      # verbose logging for one launch

.. raw:: html

   <div><a href="../../readme_text/de/environment-2.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/de/environment-2.desktop.svg"><img src="../../readme_images/de/environment-2.svg" alt="Rotierende Protokolle werden nach ~/.spacr/logs/spacr.log geschrieben. Fügen Sie diese Datei einem Fehlerbericht bei."></picture></a></div>

.. raw:: html

   <div><a href="../../readme_text/de/contributing-and-support.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/de/contributing-and-support.desktop.svg"><img src="../../readme_images/de/contributing-and-support.svg" alt="Beiträge und Support. Übermitteln Sie Fehlerberichte und klar abgegrenzte Funktionswünsche über GitHub Issues. Geben Sie bei einer Fehlermeldung die spaCR-Version, das Betriebssystem, die Python-Version, die Moduleinstellungen und den relevanten Protokollauszug an. spacr-doctor erfasst den Großteil dieser Angaben; fügen Sie bei Leistungsproblemen den Hardwarebericht bei."></picture></a></div>

.. raw:: html

   <div><a href="../../readme_text/de/licensing.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/de/licensing.desktop.svg"><img src="../../readme_images/de/licensing.svg" alt="Lizenz. spaCR wird unter der BSD3-Clause-Lizenz veröffentlicht. Wenn spaCR zu einer Veröffentlichung beigetragen hat, freuen wir uns über eine Quellenangabe. Sie ist keine Bedingung der Lizenz; siehe unten unter spaCR zitieren."></picture></a></div>

.. raw:: html

   <div><a href="../../readme_text/de/tutorials.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/de/tutorials.desktop.svg"><img src="../../readme_images/de/tutorials.svg" alt="Tutorials. Die interaktive spaCR-Tutorialbibliothek führt durch die Installation und die Verwendung der Module. Verfügbare Vertonungen und Sprachen werden für jede Lektion angegeben."></picture></a></div>

.. raw:: html

   <div><a href="../../readme_text/de/citing-spacr.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/de/citing-spacr.desktop.svg"><img src="../../readme_images/de/citing-spacr.svg" alt="spaCR zitieren. Wenn spaCR zu Ihrer Forschung beiträgt, zitieren Sie: Olafsson EB, et al. Ein gepoolter Bild-basierter CRISPR Screening identifiziert EAF1 als einen T. gondii Modulator der ESCRT-Subversion."></picture></a></div>

.. raw:: html

   <div><a href="https://www.biorxiv.org/content/10.64898/2026.07.08.737057v1"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/de/citing-spacr-2-1.desktop.svg"><img src="../../readme_images/de/citing-spacr-2-1.svg" alt="Vordruck bioRxiv"></picture></a><a href="https://doi.org/10.5281/zenodo.21343316"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/de/citing-spacr-2-2.desktop.svg"><img src="../../readme_images/de/citing-spacr-2-2.svg" alt="Software-Archiv"></picture></a></div>

.. raw:: html

   <div><a href="../../readme_text/de/other-work-citing-spacr.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/de/other-work-citing-spacr.desktop.svg"><img src="../../readme_images/de/other-work-citing-spacr.svg" alt="Weitere Arbeiten, die spaCR zitieren."></picture></a></div>

.. spacr-citing-papers-begin

.. raw:: html

   <div><a href="../../readme_text/de/other-work-citing-spacr-2-1.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/de/other-work-citing-spacr-2-1.desktop.svg"><img src="../../readme_images/de/other-work-citing-spacr-2-1.svg" alt="Metabolische Anpassungsfähigkeit und Nährstoffverschrottung in Toxoplasma gondii: Erkenntnisse aus ingestionsweg-defizienten Mutanten."></picture></a></div>

.. raw:: html

   <div><a href="../../readme_text/de/other-work-citing-spacr-2-2.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/de/other-work-citing-spacr-2-2.desktop.svg"><img src="../../readme_images/de/other-work-citing-spacr-2-2.svg" alt="IRE1α fördert den Phagosomal-Calcium-Fluss zur Verbesserung der makrophagenfungiziden Aktivität."></picture></a></div>

.. raw:: html

   <div><a href="../../readme_text/de/other-work-citing-spacr-2-3.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/de/other-work-citing-spacr-2-3.desktop.svg"><img src="../../readme_images/de/other-work-citing-spacr-2-3.svg" alt="Toxoplasma GRA8 greift das Wirts-ESCRT-Zubehörprotein ALG-2 auf und ist für die metabolische Integrität des Parasiten notwendig."></picture></a></div>

.. raw:: html

   <div><a href="../../readme_text/de/other-work-citing-spacr-2-4.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/de/other-work-citing-spacr-2-4.desktop.svg"><img src="../../readme_images/de/other-work-citing-spacr-2-4.svg" alt="Hydrogen sulfide dynamically upregulates copper uptake and localization."></picture></a></div>

.. raw:: html

   <div><a href="../../readme_text/de/other-work-citing-spacr-2-5.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/de/other-work-citing-spacr-2-5.desktop.svg"><img src="../../readme_images/de/other-work-citing-spacr-2-5.svg" alt="spaCR: Räumliche Phänotypanalyse von CRISPR-Cas9-Bildschirmen (Vordruckversion 1)."></picture></a></div>

.. spacr-citing-papers-end

.. raw:: html

   <div><a href="../../readme_text/de/acknowledgments.md"><picture><source media="(min-width: 1280px)" srcset="../../readme_images/de/acknowledgments.desktop.svg"><img src="../../readme_images/de/acknowledgments.svg" alt="Danksagung. spaCR baut auf offener wissenschaftlicher Software auf, darunter NumPy, pandas, scikit-image, scikit-learn, Cellpose, PyTorch und Qt."></picture></a></div>
