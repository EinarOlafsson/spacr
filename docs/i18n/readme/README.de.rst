|Docs| |Tutorials| |PyPI| |Python| |Tests| |Qt| |Source| |Issues| |License| |DOI|

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
.. |DOI| image:: https://img.shields.io/badge/DOI-10.5281%2Fzenodo.21343317-blue
   :target: https://doi.org/10.5281/zenodo.21343317
   :alt: Zenodo-DOI
.. |Release| image:: https://img.shields.io/github/v/release/EinarOlafsson/spacr?label=Installers
   :target: https://github.com/EinarOlafsson/spacr/releases/latest
   :alt: Neueste Installationsprogramme
.. |CondaRecipe| image:: https://img.shields.io/badge/conda--forge-recipe-44A833?logo=anaconda
   :target: https://github.com/EinarOlafsson/spacr/tree/main/conda-forge/recipe
   :alt: conda-forge-Rezept

.. image:: ../../../spacr/resources/icons/logo_spacr_readme.png
   :alt: spaCR
   :width: 920

spaCR
=====

Sprachen: `English <../../../README.rst>`_ · `Svenska <README.sv.rst>`_ ·
`Deutsch <README.de.rst>`_ ·
`Español <README.es.rst>`_ ·
`简体中文 <README.zh_CN.rst>`_ ·
`Português <README.pt.rst>`_ ·
`हिन्दी <README.hi.rst>`_ ·
`한국어 <README.ko.rst>`_ ·
`Íslenska <README.is.rst>`_ ·
`Français <README.fr.rst>`_

**Räumliche Phänotypanalyse von CRISPR-Screens.**

spaCR segmentiert und vermisst einzelne Zellen in High-Content-Mikroskopiebildern, integriert Phänotypen einzelner Objekte mit sequenzierungsbasierten Guide-Häufigkeiten und schätzt, welche Gene mit phänotypischen Veränderungen assoziiert sind. Ausgehend von Plattenbildern und FASTQ-Reads erzeugt es Messungen pro Objekt, trainierte Klassifikatoren, Effektschätzungen pro Guide und Gen sowie eine Rangliste der Treffer.

Für bildbasierte gepoolte CRISPR-Screens stellt spaCR den Arbeitsablauf von der Bildsegmentierung bis zur Priorisierung von Treffern bereit. Bei High-Content-Mikroskopiestudien ohne sequenzierungsbasierte Screens können die Module für Segmentierung, Messung, Annotation und Klassifizierung unabhängig voneinander verwendet werden.

Bilder, Masken, Bildausschnitte, Messungen, Annotationen, Vorhersagen, Barcodes und Well-Kennungen liegen in einem einzigen SQLite-Projekt. Dadurch lässt sich ein Ergebniswert bis zu seinem Ursprungsobjekt zurückverfolgen.

Führen Sie spaCR als Desktopanwendung oder ohne grafische Oberfläche auf einer Workstation, einem Server oder Cluster aus. Beide Varianten verwenden dieselben Module; CUDA wird automatisch genutzt, wenn das jeweilige Modul es unterstützt.


Workflow auf einen Blick
------------------------

.. spacr-workflow-begin

|Workflow_mask|\ |Workflow_arrow|\ |Workflow_measure|\ |Workflow_arrow|\ |Workflow_annotate|\ |Workflow_arrow|\ |Workflow_classify_merged|\ |Workflow_arrow|\ |Workflow_map_barcodes|\ |Workflow_arrow|\ |Workflow_regression|

.. |Workflow_mask| image:: ../../../spacr/resources/icons/workflow/mask.png
   :width: 14.5%
   :alt: API für Mask öffnen
   :target: https://einarolafsson.github.io/spacr/api/spacr/core/index.html
   :align: middle
.. |Workflow_measure| image:: ../../../spacr/resources/icons/workflow/measure.png
   :width: 14.5%
   :alt: API für Measure öffnen
   :target: https://einarolafsson.github.io/spacr/api/spacr/measure/index.html
   :align: middle
.. |Workflow_annotate| image:: ../../../spacr/resources/icons/workflow/annotate.png
   :width: 14.5%
   :alt: API für Annotate öffnen
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/annotate/index.html
   :align: middle
.. |Workflow_classify_merged| image:: ../../../spacr/resources/icons/workflow/classify_merged.png
   :width: 14.5%
   :alt: API für Classify öffnen
   :target: https://einarolafsson.github.io/spacr/api/spacr/classify/index.html
   :align: middle
.. |Workflow_map_barcodes| image:: ../../../spacr/resources/icons/workflow/map_barcodes.png
   :width: 14.5%
   :alt: API für Map Barcodes öffnen
   :target: https://einarolafsson.github.io/spacr/api/spacr/sequencing/index.html
   :align: middle
.. |Workflow_regression| image:: ../../../spacr/resources/icons/workflow/regression.png
   :width: 14.5%
   :alt: API für Regression öffnen
   :target: https://einarolafsson.github.io/spacr/api/spacr/ml/index.html
   :align: middle
.. |Workflow_arrow| image:: ../../../spacr/resources/icons/workflow/arrow.png
   :width: 2.5%
   :align: middle

**Daten**

|App_align|\ |App_convert|\ |App_foreign|\ |App_external_masks|\ |App_queue|

|App_batch|\ |App_distributed_jobs|\ |App_db_browser|\ |App_make_masks|\ |App_data_manager|

|App_project_browser|

**Ergebnisse & Qualitätskontrolle**

|App_plate_view|\ |App_umap|\ |App_train_compare|\ |App_run_history|\ |App_report|

|App_run_compare|\ |App_investigate_hit|\ |App_control_chart|

**Erkunden**

|App_pipeline_graph|\ |App_profiler|\ |App_qc_dashboard|\ |App_lineage|\ |App_layer_viewer|

|App_graph_builder|\ |App_tabulate|\ |App_feature_dict|\ |App_trellis|\ |App_gate_editor|

|App_feature_explorer|\ |App_outliers|

**Assays**

|App_analyze_plaques|\ |App_recruitment|\ |App_invasion|\ |App_replication|

**Versuchsplanung**

|App_experiment_design|\ |App_power|\ |App_dose_response|

.. |App_align| image:: ../../../spacr/resources/icons/workflow/apps/align.png
   :width: 19.9%
   :alt: API für Align & Stitch öffnen
   :target: https://einarolafsson.github.io/spacr/api/spacr/align/index.html
   :align: middle
.. |App_convert| image:: ../../../spacr/resources/icons/workflow/apps/convert.png
   :width: 19.9%
   :alt: API für Format Converter öffnen
   :target: https://einarolafsson.github.io/spacr/api/spacr/convert/index.html
   :align: middle
.. |App_foreign| image:: ../../../spacr/resources/icons/workflow/apps/foreign.png
   :width: 19.9%
   :alt: API für Import Project öffnen
   :target: https://einarolafsson.github.io/spacr/api/spacr/foreign/index.html
   :align: middle
.. |App_external_masks| image:: ../../../spacr/resources/icons/workflow/apps/external_masks.png
   :width: 19.9%
   :alt: API für External Masks öffnen
   :target: https://einarolafsson.github.io/spacr/api/spacr/external_masks/index.html
   :align: middle
.. |App_queue| image:: ../../../spacr/resources/icons/workflow/apps/queue.png
   :width: 19.9%
   :alt: API für Plate Queue öffnen
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/plate_queue/index.html
   :align: middle
.. |App_batch| image:: ../../../spacr/resources/icons/workflow/apps/batch.png
   :width: 19.9%
   :alt: API für Batch Runner öffnen
   :target: https://einarolafsson.github.io/spacr/api/spacr/batch/index.html
   :align: middle
.. |App_distributed_jobs| image:: ../../../spacr/resources/icons/workflow/apps/distributed_jobs.png
   :width: 19.9%
   :alt: API für Distributed Jobs öffnen
   :target: https://einarolafsson.github.io/spacr/api/spacr/remote_execution/index.html
   :align: middle
.. |App_db_browser| image:: ../../../spacr/resources/icons/workflow/apps/db_browser.png
   :width: 19.9%
   :alt: API für Database Browser öffnen
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/db_browser/index.html
   :align: middle
.. |App_make_masks| image:: ../../../spacr/resources/icons/workflow/apps/make_masks.png
   :width: 19.9%
   :alt: API für Make Masks öffnen
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/make_masks/index.html
   :align: middle
.. |App_data_manager| image:: ../../../spacr/resources/icons/workflow/apps/data_manager.png
   :width: 19.9%
   :alt: API für Data Manager öffnen
   :target: https://einarolafsson.github.io/spacr/api/spacr/data_manager/index.html
   :align: middle
.. |App_project_browser| image:: ../../../spacr/resources/icons/workflow/apps/project_browser.png
   :width: 19.9%
   :alt: API für Project Browser öffnen
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/project_browser/index.html
   :align: middle
.. |App_plate_view| image:: ../../../spacr/resources/icons/workflow/apps/plate_view.png
   :width: 19.9%
   :alt: API für Plate Viewer öffnen
   :target: https://einarolafsson.github.io/spacr/api/spacr/plate_qc/index.html
   :align: middle
.. |App_umap| image:: ../../../spacr/resources/icons/workflow/apps/umap.png
   :width: 19.9%
   :alt: API für Image UMAP öffnen
   :target: https://einarolafsson.github.io/spacr/api/spacr/core/index.html
   :align: middle
.. |App_train_compare| image:: ../../../spacr/resources/icons/workflow/apps/train_compare.png
   :width: 19.9%
   :alt: API für Training Runs öffnen
   :target: https://einarolafsson.github.io/spacr/api/spacr/train_compare/index.html
   :align: middle
.. |App_run_history| image:: ../../../spacr/resources/icons/workflow/apps/run_history.png
   :width: 19.9%
   :alt: API für Run History öffnen
   :target: https://einarolafsson.github.io/spacr/api/spacr/run_journal/index.html
   :align: middle
.. |App_report| image:: ../../../spacr/resources/icons/workflow/apps/report.png
   :width: 19.9%
   :alt: API für Report öffnen
   :target: https://einarolafsson.github.io/spacr/api/spacr/report/index.html
   :align: middle
.. |App_run_compare| image:: ../../../spacr/resources/icons/workflow/apps/run_compare.png
   :width: 19.9%
   :alt: API für Run Compare öffnen
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/run_compare/index.html
   :align: middle
.. |App_investigate_hit| image:: ../../../spacr/resources/icons/workflow/apps/investigate_hit.png
   :width: 19.9%
   :alt: API für Investigate Hit öffnen
   :target: https://einarolafsson.github.io/spacr/api/spacr/hit_investigation/index.html
   :align: middle
.. |App_control_chart| image:: ../../../spacr/resources/icons/workflow/apps/control_chart.png
   :width: 19.9%
   :alt: API für Control Charts öffnen
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/control_chart/index.html
   :align: middle
.. |App_pipeline_graph| image:: ../../../spacr/resources/icons/workflow/apps/pipeline_graph.png
   :width: 19.9%
   :alt: API für Pipeline Graph öffnen
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/pipeline_graph/index.html
   :align: middle
.. |App_profiler| image:: ../../../spacr/resources/icons/workflow/apps/profiler.png
   :width: 19.9%
   :alt: API für Prediction Profiler öffnen
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/profiler/index.html
   :align: middle
.. |App_qc_dashboard| image:: ../../../spacr/resources/icons/workflow/apps/qc_dashboard.png
   :width: 19.9%
   :alt: API für QC Dashboard öffnen
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/qc_dashboard/index.html
   :align: middle
.. |App_lineage| image:: ../../../spacr/resources/icons/workflow/apps/lineage.png
   :width: 19.9%
   :alt: API für Lineage öffnen
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/lineage/index.html
   :align: middle
.. |App_layer_viewer| image:: ../../../spacr/resources/icons/workflow/apps/layer_viewer.png
   :width: 19.9%
   :alt: API für Layer Viewer öffnen
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/layer_viewer/index.html
   :align: middle
.. |App_graph_builder| image:: ../../../spacr/resources/icons/workflow/apps/graph_builder.png
   :width: 19.9%
   :alt: API für Graph Builder öffnen
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/graph_builder/index.html
   :align: middle
.. |App_tabulate| image:: ../../../spacr/resources/icons/workflow/apps/tabulate.png
   :width: 19.9%
   :alt: API für Tabulate öffnen
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/tabulate/index.html
   :align: middle
.. |App_feature_dict| image:: ../../../spacr/resources/icons/workflow/apps/feature_dict.png
   :width: 19.9%
   :alt: API für Feature Dictionary öffnen
   :target: https://einarolafsson.github.io/spacr/api/spacr/feature_dict/index.html
   :align: middle
.. |App_trellis| image:: ../../../spacr/resources/icons/workflow/apps/trellis.png
   :width: 19.9%
   :alt: API für Small Multiples öffnen
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/trellis/index.html
   :align: middle
.. |App_gate_editor| image:: ../../../spacr/resources/icons/workflow/apps/gate_editor.png
   :width: 19.9%
   :alt: API für Gate Editor öffnen
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/gate_editor/index.html
   :align: middle
.. |App_feature_explorer| image:: ../../../spacr/resources/icons/workflow/apps/feature_explorer.png
   :width: 19.9%
   :alt: API für Feature Explorer öffnen
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/feature_explorer/index.html
   :align: middle
.. |App_outliers| image:: ../../../spacr/resources/icons/workflow/apps/outliers.png
   :width: 19.9%
   :alt: API für Outliers öffnen
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/outliers/index.html
   :align: middle
.. |App_analyze_plaques| image:: ../../../spacr/resources/icons/workflow/apps/analyze_plaques.png
   :width: 19.9%
   :alt: API für Plaque Assay öffnen
   :target: https://einarolafsson.github.io/spacr/api/spacr/submodules/index.html
   :align: middle
.. |App_recruitment| image:: ../../../spacr/resources/icons/workflow/apps/recruitment.png
   :width: 19.9%
   :alt: API für Recruitment öffnen
   :target: https://einarolafsson.github.io/spacr/api/spacr/submodules/index.html
   :align: middle
.. |App_invasion| image:: ../../../spacr/resources/icons/workflow/apps/invasion.png
   :width: 19.9%
   :alt: API für Invasion Assay öffnen
   :target: https://einarolafsson.github.io/spacr/api/spacr/submodules/index.html
   :align: middle
.. |App_replication| image:: ../../../spacr/resources/icons/workflow/apps/replication.png
   :width: 19.9%
   :alt: API für Replication Assay öffnen
   :target: https://einarolafsson.github.io/spacr/api/spacr/submodules/index.html
   :align: middle
.. |App_experiment_design| image:: ../../../spacr/resources/icons/workflow/apps/experiment_design.png
   :width: 19.9%
   :alt: API für Experiment Design öffnen
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/experiment_design/index.html
   :align: middle
.. |App_power| image:: ../../../spacr/resources/icons/workflow/apps/power.png
   :width: 19.9%
   :alt: API für Power / Design öffnen
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/power/index.html
   :align: middle
.. |App_dose_response| image:: ../../../spacr/resources/icons/workflow/apps/dose_response.png
   :width: 19.9%
   :alt: API für Dose–Response öffnen
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/dose_response/index.html
   :align: middle

.. spacr-workflow-end

Wählen Sie ein Workflow-Modul aus, um dessen API-Seite zu öffnen. Das Raster enthält alle weiteren Anwendungen in denselben Kategorien und in derselben Reihenfolge wie auf der spaCR-Startseite.


spaCR installieren
------------------

Desktopanwendung
~~~~~~~~~~~~~~~~~~~

Die Desktop-Installationsdateien enthalten eine private Python-Umgebung, daher sind Conda und eine bestehende Python-Installation nicht erforderlich.

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

Python-Installation
~~~~~~~~~~~~~~~~~~~

Python 3.12 hat die größte Auswahl an optionalen wissenschaftlichen Paketen:

.. code-block:: bash

   conda create -n spacr python=3.12 -y
   conda activate spacr
   python -m pip install --upgrade pip
   python -m pip install "spacr[qt]"
   spacr

spaCR unterstützt Python **3.9 bis 3.14** mit Ausnahme von Python 3.14.1, das von torchvision ausgeschlossen wird. Für CUDA-Workflows wird Linux empfohlen; macOS und Windows werden ebenfalls unterstützt.

Lassen Sie Qt auf einem Server, Cluster oder CI-Runner weg:

.. code-block:: bash

   python -m pip install spacr
   spacr-run --list

Optional integrations are installed separately, for example ``spacr[zarr]``, ``spacr[omero]``, ``spacr[napari]`` and ``spacr[czi,nd2,lif]``. See the `Installationsanleitung <../../source/installer_guide.rst>`_ for the complete extras and Python-version compatibility table.

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


Was Sie tun können
------------------

Der primäre Arbeitsablauf umfasst sechs Module:

- **Mask** segmentiert Zellen, Zellkerne, Pathogene und Organellen mit Cellpose.
- **Measure** schreibt Morphologie-, Intensitäts-, Textur-, räumliche und Kolokalisationsmerkmale sowie Objektausschnitte nach SQLite.
- **Annotate** beschriftet Objektausschnitte in einem tastaturgesteuerten Raster und unterstützt Active-Learning-Warteschlangen.
- **Classify** trainiert bild- oder messwertbasierte Modelle und speichert mit jedem Checkpoint die Leistung auf zurückgehaltenen Daten.
- **Map Barcodes** ordnet FASTQ-Reads Wells und gRNAs zu und liefert QC für Häufigkeit, Kollisionen und Abdeckung.
- **Regression** schätzt Guide-, Gen-, Bedingungs- und Kontrolleffekte mit Modellfamilien für kontinuierliche Werte, Anteile und Zähldaten.

Dasselbe Projekt kann außerdem Versuchsplatten entwerfen, statistische Power schätzen, Batch-Effekte korrigieren, die Segmentierungsqualität prüfen, verknüpfte Diagramme und Bildausschnitte untersuchen, AnnData exportieren, unterbrochene Verarbeitung fortsetzen und die Einstellungen zu jedem Ergebnis protokollieren.

Über Host-Ansichten verfügbare Module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Zwanzig Module sind in thematisch zugehörige Host-Ansichten integriert, anstatt als separate Kacheln auf der Startseite angezeigt zu werden. Jedes Modul wird über die Kopfzeile seiner Host-Ansicht geöffnet und verwendet das aktive Projekt. Mask, Measure, Annotate, Classify, Map Barcodes, Regression, Image UMAP und Make Masks stellen diese integrierten Module bereit. Die Hilfe und API-Dokumentation bleiben verfügbar; Module mit Pipeline-Einstiegspunkten können weiterhin ohne grafische Oberfläche ausgeführt werden. Der `Funktionsleitfaden <../../source/features.rst>`_ listet jedes integrierte Modul und die zugehörige Host-Ansicht auf.

Make Masks
~~~~~~~~~~

Make Masks wird unter **Data** angezeigt und ermöglicht die manuelle Korrektur von Segmentierungsmasken. Über die Kopfzeile sind außerdem die Cellpose-Arbeitsabläufe zugänglich. Die Arbeitsfläche enthält neun Werkzeuge: **Brush**, **Erase**, **Erase object**, **Wand +**, **Wand −**, **Draw**, **Divide**, **Zoom** und **Recrop**. Draw erzeugt aus einer geschlossenen Freiformkontur eine ausgefüllte Objektbeschriftung. Divide trennt ein zusammengeführtes Objekt entlang einer benutzerdefinierten Linie und erhält alle anderen Objektbeschriftungen.

Recrop extrahiert aus einem vorbereiteten Bild mit mehreren Objekten ein Bildfeld mit einem einzelnen Objekt. Ein Begrenzungsrahmen um ein Objekt speichert die entsprechenden Bild- und Maskenbereiche als neues Bildfeld, plant dieses unmittelbar nach dem aktuellen Bildfeld ein und entfernt das ursprüngliche Bildfeld mit mehreren Objekten aus der Kuratierungswarteschlange. Recrop ändert das aktive Bildfeld, nicht die Beschriftungspixel.

Beim Ausführen von Cellpose-SAM aus Make Masks werden zwei Zwischenergebnisse neben der Maske angezeigt: die **Zellwahrscheinlichkeitskarte** und das **Flussfeld**. Die Maske wird durch einen Schwellenwert auf der Wahrscheinlichkeitskarte definiert; Flusskonsistenzprüfungen können Objekte verwerfen, deren abgeleitete Flüsse vom vorhergesagten Feld abweichen. Anhand dieser Ergebnisse lassen sich bei einer fehlerhaften oder unvollständigen Maske eine geringe Zellwahrscheinlichkeit und ein inkonsistenter Fluss unterscheiden.

Objekte und Einstellungen
~~~~~~~~~~~~~~~~~~~~~~~~~

spaCR unterstützt Zell-, Zellkern- und Pathogenobjekte, ein aus deren Masken abgeleitetes Zytoplasmaobjekt sowie zwischen null und sechsundzwanzig Organellenplätze. Jeder Organellenplatz besitzt einen unabhängigen Kanal, Durchmesser, eine Morphologie-Voreinstellung und eine Erkennungsmethode.

Das Einstellungsfeld zeigt Bedienelemente nur an, wenn sie anwendbar sind. Organellenplätze oberhalb der konfigurierten Anzahl werden ausgeblendet, Objekte ohne zugewiesenen Kanal werden vom Verarbeitungslauf ausgeschlossen und morphologiespezifische Bedienelemente werden nur für die ausgewählte Methode angezeigt. Die Schalter **3D** und **Time** legen die Dimensionalität fest: ``z_stack`` aktiviert volumetrische Einstellungen, ``timelapse`` aktiviert Tracking-Einstellungen und vierdimensionale Einstellungen werden angezeigt, wenn beide Schalter aktiviert sind.

Wählen Sie die nächste Seite nach dem, was Sie tun möchten:

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

Leistungsdiagnose
----------------------

Erzeugen Sie einen Hardwarebericht und fügen Sie ihn einem leistungsbezogenen GitHub-Issue bei::

    python tools/spacr_hardware_report.py

Der Befehl gibt einen Bericht aus und speichert eine Kopie unter ``~/.spacr/reports``; die letzte Zeile enthält den Pfad zur gespeicherten Datei. ``--quick`` lässt die längeren Leistungsmessungen aus, und ``--out PATH`` wählt einen anderen Ausgabeort.

Der Bericht öffnet kein Projekt und liest keine Projektdaten. Er erfasst die Laufzeiten von Importvorgängen und numerischen Bibliotheken, die Anzeigeskalierung, aktive Einstellungen, den Aufbau des Hauptfensters und der Modulansichten sowie die Animationsleistung. Die Berichtsdatei ist die einzige erzeugte Ausgabe.

Der Bericht erkennt außerdem die Emulation einer Prozessorarchitektur, beispielsweise eine x86_64-Version von Python auf Apple Silicon, und die von NumPy verwendete BLAS-Implementierung. Beide Faktoren können die Leistung erheblich beeinflussen.

Beiträge und Support
------------------------

Übermitteln Sie Fehlerberichte und klar abgegrenzte Funktionswünsche über `GitHub Issues <https://github.com/EinarOlafsson/spacr/issues>`_. Geben Sie bei einer Fehlermeldung die spaCR-Version, das Betriebssystem, die Python-Version, die Moduleinstellungen und den relevanten Protokollauszug an. ``spacr-doctor`` erfasst den Großteil dieser Angaben; fügen Sie bei Leistungsproblemen den Hardwarebericht bei.

Lizenz
~~~~~~~~~

Der aktuelle Entwicklungszweig ist unter der `PolyForm Noncommercial License 1.0.0 <https://github.com/EinarOlafsson/spacr/blob/main/LICENSE>`_ quelloffen einsehbar. Für die kommerzielle Nutzung ist eine separate Lizenz des Urheberrechtsinhabers erforderlich. Veröffentlichte Versionen bis einschließlich spaCR 1.4.9.9 bleiben unter der jeweils mitgelieferten MIT-Lizenz verfügbar.

Tutorials
~~~~~~~~~

Die `interaktive spaCR-Tutorialsammlung <https://einarolafsson.github.io/spacr/tutorials/>`_ enthält vertonte und untertitelte Anleitungen zur Installation und zu jedem Anwendungsablauf: 73 Lektionen mit 50 Stimmen in acht Sprachen.

spaCR zitieren
~~~~~~~~~~~~~~

Wenn spaCR zu Ihrer Forschung beiträgt, zitieren Sie:

Olafsson EB, *et al.* Ein gepoolter Bild-basierter CRISPR Screening identifiziert EAF1 als einen *T. gondii* Modulator der ESCRT-Subversion.

`Vordruck bioRxiv <https://www.biorxiv.org/content/10.64898/2026.07.08.737057v1>`_ · `Software-Archiv <https://doi.org/10.5281/zenodo.21343317>`_

Danksagung
~~~~~~~~~~~~~~~

spaCR baut auf offener wissenschaftlicher Software auf, darunter NumPy, pandas, scikit-image, scikit-learn, Cellpose, PyTorch und Qt. Die für die mehrsprachige Dokumentation und die Oberflächenkataloge verwendeten Modelle sind in der `Attribution der Übersetzungsmodelle <../TRANSLATION_MODELS.md>`_ aufgeführt.
