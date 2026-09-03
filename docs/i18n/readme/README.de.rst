|Docs| |Tutorials| |PyPI| |Conda| |Python| |Tests| |Qt| |Source| |Issues| |License| |Preprint| |DOI|

.. |Docs| image:: https://github.com/EinarOlafsson/spacr/actions/workflows/pages/pages-build-deployment/badge.svg
   :target: https://einarolafsson.github.io/spacr/
   :alt: Dokumentation
.. |Tutorials| image:: https://img.shields.io/badge/Tutorials-Interactive%20walkthrough-4A9EFF
   :target: https://einarolafsson.github.io/spacr/tutorials/
   :alt: Interaktive Tutorials
.. |PyPI| image:: https://img.shields.io/pypi/v/spacr
   :target: https://pypi.org/project/spacr/
   :alt: PyPI-Version
.. |Python| image:: https://img.shields.io/badge/Python-3.9%E2%80%933.14-3776AB?logo=python&logoColor=white
   :target: https://pypi.org/project/spacr/
   :alt: Python 3.9 bis 3.14
.. |Tests| image:: https://github.com/EinarOlafsson/spacr/actions/workflows/tests.yml/badge.svg?branch=nightly
   :target: https://github.com/EinarOlafsson/spacr/actions/workflows/tests.yml
   :alt: Testsuite
.. |Qt| image:: https://img.shields.io/badge/GUI-Qt%20%28PySide6%29-41CD52
   :target: https://einarolafsson.github.io/spacr/
   :alt: Qt-Oberfläche
.. |Source| image:: https://img.shields.io/badge/GitHub-Source-181717?logo=github
   :target: https://github.com/EinarOlafsson/spacr
   :alt: GitHub-Quellcode
.. |Issues| image:: https://img.shields.io/github/issues/EinarOlafsson/spacr
   :target: https://github.com/EinarOlafsson/spacr/issues
   :alt: GitHub-Issues
.. |License| image:: https://img.shields.io/github/license/EinarOlafsson/spacr
   :target: https://github.com/EinarOlafsson/spacr/blob/main/LICENSE
   :alt: PolyForm-Noncommercial-Lizenz
.. |Preprint| image:: https://img.shields.io/badge/bioRxiv-2026.07.08.737057-BF2636
   :target: https://www.biorxiv.org/content/10.64898/2026.07.08.737057v1
   :alt: bioRxiv preprint
.. |DOI| image:: https://img.shields.io/badge/DOI-10.5281%2Fzenodo.21343316-blue
   :target: https://doi.org/10.5281/zenodo.21343316
   :alt: Zenodo-DOI
.. |Release| image:: https://img.shields.io/github/v/release/EinarOlafsson/spacr?label=Installers
   :target: https://github.com/EinarOlafsson/spacr/releases/latest
   :alt: Neueste Installationsprogramme
.. |Conda| image:: https://anaconda.org/conda-forge/spacr/badges/version.svg
   :target: https://anaconda.org/conda-forge/spacr
   :alt: conda-forge-Version

.. image:: ../../../spacr/resources/icons/logo_spacr_readme.png
   :alt: spaCR
   :width: 920

spaCR
=====

.. spacr-language-picker-begin

Sprachen: `🌐 Deutsch ▾ <README.md>`_

.. spacr-language-picker-end

**Räumliche Phänotypanalyse von CRISPR-Screens.**

spaCR segmentiert und vermisst einzelne Zellen in High-Content-Mikroskopiebildern, integriert Phänotypen einzelner Objekte mit sequenzierungsbasierten Guide-Häufigkeiten und schätzt, welche Gene mit phänotypischen Veränderungen assoziiert sind. Ausgehend von Plattenbildern und FASTQ-Reads erzeugt es Messungen pro Objekt, trainierte Klassifikatoren, Effektschätzungen pro Guide und Gen sowie eine Rangliste der Treffer.

Die Segmentierungs-, Mess-, Anmerkungs- und Klassifizierungsmodule laufen auch ohne Sequenzierungsarm.

Bilder, Masken, Bildausschnitte, Messungen, Anmerkungen, Vorhersagen, Barcodes und Brunnenidentifikatoren leben in einem SQLite Projekt.

Läuft als Desktop-Anwendung oder ohne grafische Oberfläche auf einer Workstation, einem Server oder Cluster.

Hardware-Unterstützung
~~~~~~~~~~~~~~~~~~~~~~

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

🟢 supported (stable)   🟣 implemented (beta)   🔴 CPU support only

.. spacr-hardware-end


spaCR installieren
------------------

Desktopanwendung
~~~~~~~~~~~~~~~~~~~

Die Installateure bündeln ihre eigenen Python. Conda ist nicht erforderlich.

.. spacr-installer-links-begin

|InstallerLinux| |InstallerMacOS| |InstallerWindows| |InstallerLegacy|

.. |InstallerWindows| image:: ../../../spacr/resources/icons/platforms/windows.png
   :width: 64
   :alt: Windows 10/11: spaCR 1.5.0.4 herunterladen
   :target: https://github.com/EinarOlafsson/spacr/releases/download/v1.5.0.4/SpaCR-1.5.0.4-Windows-Online-Setup.exe
.. |InstallerMacOS| image:: ../../../spacr/resources/icons/platforms/macos.png
   :width: 64
   :alt: macOS 11+ (Intel und Apple Silicon): spaCR 1.5.0.4 herunterladen
   :target: https://github.com/EinarOlafsson/spacr/releases/download/v1.5.0.4/SpaCR-1.5.0.4-macOS-Universal-Online.pkg
.. |InstallerLinux| image:: ../../../spacr/resources/icons/platforms/linux.png
   :width: 64
   :alt: 64-Bit-Linux: spaCR 1.5.0.4 herunterladen
   :target: https://github.com/EinarOlafsson/spacr/releases/download/v1.5.0.4/SpaCR-1.5.0.4-Linux-x86_64-Online.run
.. |InstallerLegacy| image:: ../../../spacr/resources/icons/platforms/legacy.png
   :width: 64
   :alt: Ältere spaCR-Installationsprogramme
   :target: ../../source/installers.rst

.. spacr-installer-links-end

Die ersten drei Icons laden die aktuelle Version herunter. Das spaCR-Symbol öffnet das komplette Installationsarchiv. Installer-Links und versionierte Dateinamen werden durch den Release-Workflow aktualisiert; frühere Installationsdateien verbleiben im gleichen Release-Archiv.

Machen Sie die heruntergeladene Datei unter Linux ausführbar und führen Sie sie aus:

.. code-block:: bash

   chmod +x SpaCR-*-Linux-x86_64-Online.run
   ./SpaCR-*-Linux-x86_64-Online.run

Öffnen Sie auf macOS das ``.pkg``. Die aktuelle Beta wird nicht beglaubigt; wenn Gatekeeper sie blockiert, wählen Sie **Systemeinstellungen → Datenschutz & Sicherheit → Öffnen Sie trotzdem**.

Siehe die Anweisungen `Installationsanleitung <../../source/installer_guide.rst>`_ zur Aktualisierung, Deinstallation, Offline- und Fehlerbehebung.

Installation über PyPI
~~~~~~~~~~~~~~~~~~~~~~

Installieren Sie die PyPI-Veröffentlichung von spaCR mit pip in einer Conda-Umgebung. Python 3.12 bietet die größte Auswahl an optionalen wissenschaftlichen Paketen:

.. code-block:: bash

   conda create -n spacr python=3.12 -y
   conda activate spacr
   python -m pip install --upgrade pip
   python -m pip install "spacr[qt]"
   spacr

spaCR unterstützt Python **3,9 bis 3.14**, außer Python 3.14.1, was torchvision ausschließt. Linux wird für die schwersten CUDA- und ROCm-Workflows empfohlen; macOS und Windows werden ebenfalls unterstützt, und beide verwenden ihre GPUs- macOS über Metal, die Apple Silicon und die AMD-Karten in Intel Macs abdecken, sowie Windows durch CUDA oder DirectML.

Lassen Sie Qt auf einem Server, Cluster oder CI-Runner weg:

.. code-block:: bash

   python -m pip install spacr
   spacr-run --list

Optional integrations are installed separately, for example ``spacr[zarr]``, ``spacr[omero]``, ``spacr[napari]`` and ``spacr[czi,nd2,lif]``. See the `Installationsanleitung <../../source/installer_guide.rst>`_ for the complete extras and Python-version compatibility table.

Installation mit conda-forge
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Das offizielle conda-forge-Paket installiert spaCR und seine Desktop-Abhängigkeiten in der aktiven Umgebung:

.. code-block:: bash

   conda create -n spacr python=3.12 -y
   conda activate spacr
   conda install conda-forge::spacr
   spacr

Installieren aus der Quelle
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Klonen Sie das Projektarchiv und installieren Sie es im editierbaren Modus, so dass Ihre Arbeitskopie *ist* das installierte Paket und Bearbeitungen ohne Neuinstallation wirksam werden::

    git clone https://github.com/EinarOlafsson/spacr.git
    cd spacr
    conda create -n spacr python=3.12 -y
    conda activate spacr
    pip install -e .
    spacr

Der Standard-Zweig ist ``nightly``. Für eine bestimmte Version::

    git clone --branch v1.5.0.5 https://github.com/EinarOlafsson/spacr.git

Um spätere Änderungen zu ziehen, aus dem Inneren des Klons::

    git pull
    pip install -e .

Die zweite Zeile wird nur benötigt, wenn Abhängigkeiten oder Eingabepunkte geändert werden; Python-Code wird ohne sie abgeholt. Wenn ein Befehl nach dem Ziehen noch alten Code ausführt, meldet ``spacr-doctor``, welches ``spacr`` sich tatsächlich auf Ihrem Pfad befindet, was die übliche Ursache ist.

Installieren aus der Quelle (Licht)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Vollständiger Klon: 427 MB. Kernklon: 76 MB.

::

    curl -fsSL https://raw.githubusercontent.com/EinarOlafsson/spacr/nightly/packaging/install_from_source.sh -o install_spacr.sh
    sh install_spacr.sh --branch nightly

Skips ``docs/``, ``tests/``, Cellpose Checkpoints, archivierte Figuren und die erweiterten Übersetzungskataloge. Das Ergebnis ist eine normale Kasse.

Options: ``--dir``, ``--branch`` (default ``main``), ``--with-tests``, ``--with-docs``, ``--with-translations``, ``--no-install``.

``packaging/source_install_excludes.txt`` listet jeden übersprungenen Pfad auf.


Befehle für die Kommandozeile
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   spacr                                      # launch the Qt application
   spacr-doctor                               # diagnose the installation
   spacr-run --list                           # list headless modules
   spacr-run --describe MODULE                # inspect a module contract
   spacr-run MODULE --settings settings.csv   # execute a module
   spacr-run validate --module MODULE \
       --settings settings.csv                # validate before running
   spacr-repro RUN_DIR                        # replay a recorded run

Setzen Sie bei der Fehlerbehebung ``SPACR_LOG_LEVEL=DEBUG``. Rotierende Protokolle werden in ``~/.spacr/logs/spacr.log`` geschrieben.

``spacr-run --list`` listet Module mit Befehlszeileneinstiegspunkten für die Ausführung ohne grafische Oberfläche auf. Reine GUI-Module für Annotation, Kuratierung, Vergleich und Exploration werden nicht aufgeführt.


Kern-Workflow
-------------

Der primäre Arbeitsablauf umfasst sechs Module:

- **Mask** segmentiert Zellen, Zellkerne, Pathogene und Organellen mit Cellpose.
- **Measure** schreibt Morphologie-, Intensitäts-, Textur-, räumliche und Kolokalisationsmerkmale sowie Objektausschnitte nach SQLite.
- **Annotate** beschriftet Objektausschnitte in einem tastaturgesteuerten Raster und unterstützt Active-Learning-Warteschlangen.
- **Classify** trainiert bild- oder messwertbasierte Modelle und speichert mit jedem Checkpoint die Leistung auf zurückgehaltenen Daten.
- **Map Barcodes** ordnet FASTQ-Reads Wells und gRNAs zu und liefert QC für Häufigkeit, Kollisionen und Abdeckung.
- **Regression** schätzt Guide-, Gen-, Bedingungs- und Kontrolleffekte mit Modellfamilien für kontinuierliche Werte, Anteile und Zähldaten.

Dasselbe Projekt kann außerdem Versuchsplatten entwerfen, statistische Power schätzen, Batch-Effekte korrigieren, die Segmentierungsqualität prüfen, verknüpfte Diagramme und Bildausschnitte untersuchen, AnnData exportieren, unterbrochene Verarbeitung fortsetzen und die Einstellungen zu jedem Ergebnis protokollieren.

spaCR-Module
-------------

.. spacr-workflow-begin

| |Module_mask|\ |Module_measure|\ |Module_annotate|\ |Module_classify_merged|\ |Module_map_barcodes|\ |Module_regression|
| |Module_foreign|\ |Module_run_compare|\ |Module_experiment_design|\ |Module_power|\ |Module_dose_response|\ |Module_qc_dashboard|
| |Module_make_masks|\ |Module_align|\ |Module_umap|\ |Module_gate_editor|\ |Module_graph_builder|\ |Module_analyze_plaques|
| |Module_recruitment|\ |Module_invasion|\ |Module_replication|

.. |Module_mask| image:: ../../../spacr/resources/icons/workflow/mask.png
   :width: 16.0%
   :alt: API für Mask öffnen
   :target: https://einarolafsson.github.io/spacr/api/spacr/core/index.html#spacr.core.preprocess_generate_masks
   :align: middle
.. |Module_measure| image:: ../../../spacr/resources/icons/workflow/measure.png
   :width: 16.0%
   :alt: API für Measure öffnen
   :target: https://einarolafsson.github.io/spacr/api/spacr/measure/index.html
   :align: middle
.. |Module_annotate| image:: ../../../spacr/resources/icons/workflow/annotate.png
   :width: 16.0%
   :alt: API für Annotate öffnen
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/annotate/index.html
   :align: middle
.. |Module_classify_merged| image:: ../../../spacr/resources/icons/workflow/classify_merged.png
   :width: 16.0%
   :alt: API für Classify öffnen
   :target: https://einarolafsson.github.io/spacr/api/spacr/classify/index.html
   :align: middle
.. |Module_map_barcodes| image:: ../../../spacr/resources/icons/workflow/map_barcodes.png
   :width: 16.0%
   :alt: API für Map Barcodes öffnen
   :target: https://einarolafsson.github.io/spacr/api/spacr/sequencing/index.html
   :align: middle
.. |Module_regression| image:: ../../../spacr/resources/icons/workflow/regression.png
   :width: 16.0%
   :alt: API für Regression öffnen
   :target: https://einarolafsson.github.io/spacr/api/spacr/ml/index.html
   :align: middle
.. |Module_foreign| image:: ../../../spacr/resources/icons/workflow/apps/foreign.png
   :width: 16.0%
   :alt: API für Import öffnen
   :target: https://einarolafsson.github.io/spacr/api/spacr/foreign/index.html
   :align: middle
.. |Module_run_compare| image:: ../../../spacr/resources/icons/workflow/apps/run_compare.png
   :width: 16.0%
   :alt: API für Run Compare öffnen
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/run_compare/index.html
   :align: middle
.. |Module_experiment_design| image:: ../../../spacr/resources/icons/workflow/apps/experiment_design.png
   :width: 16.0%
   :alt: API für Experiment Design öffnen
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/experiment_design/index.html
   :align: middle
.. |Module_power| image:: ../../../spacr/resources/icons/workflow/apps/power.png
   :width: 16.0%
   :alt: API für Power / Design öffnen
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/power/index.html
   :align: middle
.. |Module_dose_response| image:: ../../../spacr/resources/icons/workflow/apps/dose_response.png
   :width: 16.0%
   :alt: API für Dose–Response öffnen
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/dose_response/index.html
   :align: middle
.. |Module_qc_dashboard| image:: ../../../spacr/resources/icons/workflow/apps/qc_dashboard.png
   :width: 16.0%
   :alt: API für QC öffnen
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/qc_dashboard/index.html
   :align: middle
.. |Module_make_masks| image:: ../../../spacr/resources/icons/workflow/apps/make_masks.png
   :width: 16.0%
   :alt: API für Make Masks öffnen
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/make_masks/index.html
   :align: middle
.. |Module_align| image:: ../../../spacr/resources/icons/workflow/apps/align.png
   :width: 16.0%
   :alt: API für Align & Stitch öffnen
   :target: https://einarolafsson.github.io/spacr/api/spacr/align/index.html
   :align: middle
.. |Module_umap| image:: ../../../spacr/resources/icons/workflow/apps/umap.png
   :width: 16.0%
   :alt: API für Image UMAP öffnen
   :target: https://einarolafsson.github.io/spacr/api/spacr/core/index.html#spacr.core.generate_image_umap
   :align: middle
.. |Module_gate_editor| image:: ../../../spacr/resources/icons/workflow/apps/gate_editor.png
   :width: 16.0%
   :alt: API für Gate Editor öffnen
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/gate_editor/index.html
   :align: middle
.. |Module_graph_builder| image:: ../../../spacr/resources/icons/workflow/apps/graph_builder.png
   :width: 16.0%
   :alt: API für Graph Builder öffnen
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/graph_builder/index.html
   :align: middle
.. |Module_analyze_plaques| image:: ../../../spacr/resources/icons/workflow/apps/analyze_plaques.png
   :width: 16.0%
   :alt: API für Plaque Assay öffnen
   :target: https://einarolafsson.github.io/spacr/api/spacr/submodules/index.html#spacr.submodules.analyze_plaques
   :align: middle
.. |Module_recruitment| image:: ../../../spacr/resources/icons/workflow/apps/recruitment.png
   :width: 16.0%
   :alt: API für Recruitment öffnen
   :target: https://einarolafsson.github.io/spacr/api/spacr/submodules/index.html#spacr.submodules.analyze_recruitment
   :align: middle
.. |Module_invasion| image:: ../../../spacr/resources/icons/workflow/apps/invasion.png
   :width: 16.0%
   :alt: API für Invasion Assay öffnen
   :target: https://einarolafsson.github.io/spacr/api/spacr/submodules/index.html#spacr.submodules.analyze_invasion
   :align: middle
.. |Module_replication| image:: ../../../spacr/resources/icons/workflow/apps/replication.png
   :width: 16.0%
   :alt: API für Replication Assay öffnen
   :target: https://einarolafsson.github.io/spacr/api/spacr/submodules/index.html#spacr.submodules.analyze_replication
   :align: middle

.. spacr-workflow-end

Jedes Modul spaCR Schiffe, in der Reihenfolge, die der Home-Bildschirm listet sie: die sechs Pipeline-Module zuerst, dann alles andere. Wählen Sie eine Kachel, um das Modul API Seite öffnen.


Make Masks
~~~~~~~~~~

Make Masks appears under **Tools** for manual correction of segmentation masks; its masthead opens the Cellpose workflows. Nine tools: **Brush**, **Erase**, **Erase object**, **Wand +**, **Wand −**, **Draw**, **Divide**, **Zoom** and **Recrop**. Draw makes one filled label from a closed outline, Divide separates a merged object along a drawn line, Recrop turns one object in a crowded field into its own field.

Cellpose-SAM läuft hier die Zell-Wahrscheinlichkeitskarte und das Flussfeld neben der Maske. Siehe `Feature-Führung <../../source/features.rst>`_ für jedes Werkzeug.

**Sonstige Ressourcen**

- `Interaktive Tutorials <https://einarolafsson.github.io/spacr/tutorials/>`_ — 73 geführte Workflows von der Installation bis zur Hit-Untersuchung.
- `Python API Schnellstart <../../source/python_api.rst>`_ — Pipelines aus Skripten, Notebooks oder einem Cluster ausführen und validieren.
- `Funktionsleitfaden <../../source/features.rst>`_ — Fähigkeiten, Reife und optionale Integrationen.
- `Kuratierte API Referenz <https://einarolafsson.github.io/spacr/api/index.html>`_ — unterstützte Eingabepunkte nach Aufgabe, wobei das komplette Modul eine Ebene tiefer verweist.
- `Sprach- und Übersetzungshandbuch <../../source/localization.rst>`_ — Schnittstellensprachen, kontextbezogene Hilfe und Politik der wissenschaftlichen Ergebnisse.

Sprache und Übersetzung
~~~~~~~~~~~~~~~~~~~~~~~

Die Oberfläche unterstützt zehn Sprachen in der Navigation und den Einstellungen. AI- und LIVE-Steuerelemente, Modulbeschreibungen und geprüfte Kontexthilfe werden ebenfalls übersetzt. Ändern Sie die Sprache unter **spaCR → Einstellungen → Sprache**, ohne neu zu starten. Protokolle, Pfade, Datenbankwerte und Messungen werden nie übersetzt; wissenschaftliche Ausgaben bleiben im kanonischen Englisch. Siehe die `Richtlinie zur Kontexthilfe <../../source/localization.rst#contextual-help>`_.

Animierte Einstellungshilfe
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Einstellungen mit einer visuellen Erklärung bieten in ihrem Tooltip die Schaltfläche **Animation**. Durchsuchen Sie die `Galerie der Einstellungsanimationen <https://einarolafsson.github.io/spacr/setting_animations.html>`_ oder das `Register der Einstellungsanimationen <https://einarolafsson.github.io/spacr/api/spacr/setting_animations/index.html>`_.

Daten
-----

Referenzdatensätze
~~~~~~~~~~~~~~~~~~

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

Modellzoo
~~~~~~~~~

spaCR liefert einen Katalog von ausgebildeten Modellen und holt sie auf Anfrage ab. Öffnen Sie **Model Zoo** vom Home-Bildschirm, um sie zu durchsuchen und zu installieren, oder benennen Sie einen Schlüssel in einer Einstellungsdatei -- ``pathogen_model: toxoplasma_pv_v1`` -- und das Modell wird heruntergeladen und Checksummen-verifiziert, wenn es zum ersten Mal benötigt wird. Jeder veröffentlichte Eintrag trägt eine SHA-256; ein Eintrag ohne einen wird abgelehnt, anstatt installiert, weil ein verkürzter oder ersetzter Checkpoint nicht vom realen angezeigt werden kann.

.. spacr-model-zoo-begin

.. list-table::
   :header-rows: 1
   :widths: 24 34 42

   * - Model
     - Training data
     - Hold-out against stock
   * - ``toxoplasma_pv_v1``
       (Cellpose-SAM (cpsam_v2))
     - anti-Toxoplasma-biotin and DsRed PV lumen; 115 images, 1 dataset
     - F1 0.867 against 0.713 for stock cpsam, at IoU 0.5
   * - ``toxoplasma_plaque_v1``
       (Cellpose-SAM (cpsam))
     - Toxoplasma plaque assays; 2 datasets, in-domain and literature; image count not recorded
     - F1 0.856 in-domain and 0.834 on the literature set; no stock cpsam baseline measured
   * - ``toxoplasma_well_detector_v1``
       (YOLO11n)
     - whole-plate and multi-well plaque-assay images; 1 dataset; image count not recorded
     - mAP50 0.993, mAP50-95 0.886; no stock model detects wells

.. spacr-model-zoo-end

Die obigen Zahlen sind diejenigen, die bei der Veröffentlichung gemessen werden, und die Grenzen sind mit ihnen angegeben: ein Modell ist nützlich für den Job, an dem es gemessen wurde, nicht für jeden Job. ``toxoplasma_well_detector_v1`` und ``toxoplasma_plaque_v1`` sind die beiden Hälften einer Pipeline -- der Detektor findet die Wells, der Segmenter findet die Plaques in ihnen, und der Brunnendurchmesser ist das, was Bereiche zwischen Mikroskopen vergleichbar macht.

Modelle werden auf dem eigenen Hugging Face-Konto ihres Autors gehostet, daher bedeutet der Beitrag nicht, Schreibzugriff auf das Konto eines anderen zu geben. ``spacr.model_zoo`` s ``publish_model`` führt den Upload aus und druckt die Katalogzeile zum Hinzufügen.


Leistungsdiagnose
----------------------

Erzeugen Sie einen Hardwarebericht und fügen Sie ihn einem leistungsbezogenen GitHub-Issue bei::

    python tools/spacr_hardware_report.py

Speichert auf ``~/.spacr/reports`` und druckt den Pfad. ``--quick`` überspringt die längeren Benchmarks; ``--out PATH`` setzt den Speicherort.

Reads no project data. Times imports, numeric libraries, window construction and animation. Reports processor-architecture emulation (an x86_64 Python build on Apple Silicon) and NumPy's BLAS implementation.

Befehlszeilenreferenz
----------------------

Jeder Befehl unten wird von ``pip install spacr`` installiert. Alle von ihnen akzeptieren ``--help``.

Start der Anwendung
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   spacr              # the desktop application
   spacr-tutorial     # the interactive tutorial library
   spacr-server       # no first-run setup screen, for unattended launches

``spacr-server`` überspringt den modalen Setup-Screening, der sonst einen unbeaufsichtigten Auftrag blockieren würde.

``spacr-qt`` und ``spacr-nightly`` sind Aliasnamen von ``spacr``.

Wenn spaCR nicht startet
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   spacr-doctor       # diagnose the installation and say how to fix it
   safespacr          # the least spaCR that can still change a setting

``spacr-doctor`` gibt eine Zeile pro Check aus, mit einem Befehl, der für jeden Fehler ausgeführt werden soll. Es wird auch berichtet, welcher ``spacr`` auf dem Pfad ist, was eine alte bearbeitbare Installationsschatten ist.

``safespacr`` liest jede Präferenz als Voreinstellung und zwingt die Kulisse, Animationen, das Protokollieren und das Vorladen. Verwenden Sie sie, wenn eine gespeicherte Präferenz den Start bricht. Es ändert nichts dauerhaft.

Laufende Module ohne grafische Oberfläche
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Kein Qt, kein Display — für Cluster, Server und CI.

.. code-block:: bash

   spacr-run --list                              # modules with a headless entry
   spacr-run --describe MODULE                   # what a module consumes and produces
   spacr-run validate --module MODULE \
       --settings settings.csv                   # check settings before spending the run
   spacr-run MODULE --settings settings.csv      # execute
   spacr-remote --help                           # submit and monitor SSH, Slurm or cloud jobs

``validate`` liest die gleichen Einstellungen, die der Lauf ausführen würde, und berichtet, was fehlt, widersprüchlich ist oder auf nichts hinweist.

``spacr-run --list`` zeigt nur Module mit einem kopflosen Einstiegspunkt; Anmerkung, Kuration und Exploration sind interaktiv und weggelassen.

Inspizieren eines Laufs danach
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Jeder Lauf wird mit seinen Einstellungen, Hash-Eingängen, Ausgängen, Warnungen, Versionen und Samen auf ``~/.spacr/runs`` tagebucht.

.. code-block:: bash

   spacr-repro RUN_DIR        # replay a recorded run from its journal
   spacr-workspace RUN_DIR    # what that run had open: databases, montages, views

Prüfungsdaten und Installation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   spacr-db-audit DB      # SQLite health, integrity, locking, reader/writer probe
   spacr-leakage          # classifier train/test leakage audit
   spacr-plugins          # installed plugin registry and failure diagnostics

Umgebung
~~~~~~~~~~~

.. code-block:: bash

   SPACR_LOG_LEVEL=DEBUG spacr      # verbose logging for one launch

Drehprotokolle werden auf ``~/.spacr/logs/spacr.log`` geschrieben. Fügen Sie diese Datei einem Fehlerbericht bei.


Beiträge und Support
------------------------

Übermitteln Sie Fehlerberichte und klar abgegrenzte Funktionswünsche über `GitHub Issues <https://github.com/EinarOlafsson/spacr/issues>`_. Geben Sie bei einer Fehlermeldung die spaCR-Version, das Betriebssystem, die Python-Version, die Moduleinstellungen und den relevanten Protokollauszug an. ``spacr-doctor`` erfasst den Großteil dieser Angaben; fügen Sie bei Leistungsproblemen den Hardwarebericht bei.

Lizenz
~~~~~~~~~

spaCR is released under the `BSD 3-Clause-Lizenz <https://github.com/EinarOlafsson/spacr/blob/main/LICENSE>`_.

Wenn spaCR zu veröffentlichten Arbeiten beigetragen hat, wird ein Zitat geschätzt und ist keine Bedingung der Lizenz — siehe `Citing spaCR`_ unten.

Tutorials
~~~~~~~~~

Die `interaktive spaCR-Tutorialsammlung <https://einarolafsson.github.io/spacr/tutorials/>`_ enthält vertonte und untertitelte Anleitungen zur Installation und zu jedem Anwendungsablauf: 73 Lektionen mit 50 Stimmen in acht Sprachen.

spaCR zitieren
~~~~~~~~~~~~~~

Wenn spaCR zu Ihrer Forschung beiträgt, zitieren Sie:

Olafsson EB, *et al.* Ein gepoolter Bild-basierter CRISPR Screening identifiziert EAF1 als einen *T. gondii* Modulator der ESCRT-Subversion.

`Vordruck bioRxiv <https://www.biorxiv.org/content/10.64898/2026.07.08.737057v1>`_ · `Software-Archiv <https://doi.org/10.5281/zenodo.21343316>`_

Danksagung
~~~~~~~~~~~~~~~

spaCR baut auf offener wissenschaftlicher Software auf, darunter NumPy, pandas, scikit-image, scikit-learn, Cellpose, PyTorch und Qt. Die für die mehrsprachige Dokumentation und die Oberflächenkataloge verwendeten Modelle sind in der `Attribution der Übersetzungsmodelle <../TRANSLATION_MODELS.md>`_ aufgeführt.
