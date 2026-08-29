|Docs| |Tutorials| |PyPI| |Conda| |Python| |Tests| |Qt| |Source| |Issues| |License| |DOI|

.. |Docs| image:: https://github.com/EinarOlafsson/spacr/actions/workflows/pages/pages-build-deployment/badge.svg
   :target: https://einarolafsson.github.io/spacr/
   :alt: Dokumentation
.. |Tutorials| image:: https://img.shields.io/badge/Tutorials-Interactive%20walkthrough-4A9EFF
   :target: https://einarolafsson.github.io/spacr/tutorials/
   :alt: Interaktiva handledningar
.. |PyPI| image:: https://img.shields.io/pypi/v/spacr
   :target: https://pypi.org/project/spacr/
   :alt: PyPI-version
.. |Python| image:: https://img.shields.io/badge/Python-3.9%E2%80%933.14-3776AB?logo=python&logoColor=white
   :target: https://pypi.org/project/spacr/
   :alt: Python 3.9 till 3.14
.. |Tests| image:: https://github.com/EinarOlafsson/spacr/actions/workflows/tests.yml/badge.svg?branch=nightly
   :target: https://github.com/EinarOlafsson/spacr/actions/workflows/tests.yml
   :alt: Testsvit
.. |Qt| image:: https://img.shields.io/badge/GUI-Qt%20%28PySide6%29-41CD52
   :target: https://einarolafsson.github.io/spacr/
   :alt: Qt-gränssnitt
.. |Source| image:: https://img.shields.io/badge/GitHub-Source-181717?logo=github
   :target: https://github.com/EinarOlafsson/spacr
   :alt: Källkod på GitHub
.. |Issues| image:: https://img.shields.io/github/issues/EinarOlafsson/spacr
   :target: https://github.com/EinarOlafsson/spacr/issues
   :alt: GitHub-ärenden
.. |License| image:: https://img.shields.io/github/license/EinarOlafsson/spacr
   :target: https://github.com/EinarOlafsson/spacr/blob/main/LICENSE
   :alt: PolyForm Noncommercial-licens
.. |DOI| image:: https://img.shields.io/badge/DOI-10.5281%2Fzenodo.21343316-blue
   :target: https://doi.org/10.5281/zenodo.21343316
   :alt: Zenodo-DOI
.. |Release| image:: https://img.shields.io/github/v/release/EinarOlafsson/spacr?label=Installers
   :target: https://github.com/EinarOlafsson/spacr/releases/latest
   :alt: Senaste installationsprogrammen
.. |Conda| image:: https://anaconda.org/conda-forge/spacr/badges/version.svg
   :target: https://anaconda.org/conda-forge/spacr
   :alt: conda-forge-version

.. image:: ../../../spacr/resources/icons/logo_spacr_readme.png
   :alt: spaCR
   :width: 920

spaCR
=====

Språk: `English <../../../README.rst>`_ · `Svenska <README.sv.rst>`_ ·
`Deutsch <README.de.rst>`_ ·
`Español <README.es.rst>`_ ·
`简体中文 <README.zh_CN.rst>`_ ·
`Português <README.pt.rst>`_ ·
`हिन्दी <README.hi.rst>`_ ·
`한국어 <README.ko.rst>`_ ·
`Íslenska <README.is.rst>`_ ·
`Français <README.fr.rst>`_

**Rumslig fenotypanalys av CRISPR-screeningar.**

spaCR segmenterar och mäter enskilda celler i mikroskopibilder med högt innehåll, integrerar fenotyper per objekt med sekvenseringshärledd guideförekomst och uppskattar vilka gener som är associerade med fenotypiska förändringar. Med plattbilder och FASTQ-läsningar som utgångspunkt producerar programmet mätningar per objekt, tränade klassificerare, effektskattningar per guide och gen samt en rangordnad träfflista.

För bildbaserade poolade CRISPR-screeningar tillhandahåller spaCR arbetsflödet från bildsegmentering till prioritering av träffar. För studier med mikroskopi med högt innehåll utan sekvenseringsbaserade screeningar kan modulerna för segmentering, mätning, annotering och klassificering användas oberoende av varandra.

Bilder, masker, bildutsnitt, mätningar, annoteringar, prediktioner, streckkoder och brunnsidentifierare lagras i ett enda SQLite-projekt, så ett värde i ett resultat kan spåras tillbaka till objektet det kom från.

Kör spaCR som skrivbordsprogram eller utan grafiskt gränssnitt på en arbetsstation, server eller beräkningskluster. Båda sätten använder samma moduler, och CUDA används automatiskt när modulen stöder det.


Arbetsflödet i korthet
----------------------

.. spacr-workflow-begin

|Workflow_mask|\ |Workflow_arrow|\ |Workflow_measure|\ |Workflow_arrow|\ |Workflow_annotate|\ |Workflow_arrow|\ |Workflow_classify_merged|\ |Workflow_arrow|\ |Workflow_map_barcodes|\ |Workflow_arrow|\ |Workflow_regression|

.. |Workflow_mask| image:: ../../../spacr/resources/icons/workflow/mask.png
   :width: 14.5%
   :alt: Öppna API-dokumentationen för Mask
   :target: https://einarolafsson.github.io/spacr/api/spacr/core/index.html
   :align: middle
.. |Workflow_measure| image:: ../../../spacr/resources/icons/workflow/measure.png
   :width: 14.5%
   :alt: Öppna API-dokumentationen för Measure
   :target: https://einarolafsson.github.io/spacr/api/spacr/measure/index.html
   :align: middle
.. |Workflow_annotate| image:: ../../../spacr/resources/icons/workflow/annotate.png
   :width: 14.5%
   :alt: Öppna API-dokumentationen för Annotate
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/annotate/index.html
   :align: middle
.. |Workflow_classify_merged| image:: ../../../spacr/resources/icons/workflow/classify_merged.png
   :width: 14.5%
   :alt: Öppna API-dokumentationen för Classify
   :target: https://einarolafsson.github.io/spacr/api/spacr/classify/index.html
   :align: middle
.. |Workflow_map_barcodes| image:: ../../../spacr/resources/icons/workflow/map_barcodes.png
   :width: 14.5%
   :alt: Öppna API-dokumentationen för Map Barcodes
   :target: https://einarolafsson.github.io/spacr/api/spacr/sequencing/index.html
   :align: middle
.. |Workflow_regression| image:: ../../../spacr/resources/icons/workflow/regression.png
   :width: 14.5%
   :alt: Öppna API-dokumentationen för Regression
   :target: https://einarolafsson.github.io/spacr/api/spacr/ml/index.html
   :align: middle
.. |Workflow_arrow| image:: ../../../spacr/resources/icons/workflow/arrow.png
   :width: 2.5%
   :align: middle

**Data**

|App_align|\ |App_convert|\ |App_foreign|\ |App_external_masks|\ |App_queue|

|App_batch|\ |App_distributed_jobs|\ |App_db_browser|\ |App_make_masks|\ |App_data_manager|

|App_project_browser|

**Resultat och kvalitetskontroll**

|App_plate_view|\ |App_umap|\ |App_train_compare|\ |App_run_history|\ |App_report|

|App_run_compare|\ |App_investigate_hit|\ |App_control_chart|

**Utforska**

|App_pipeline_graph|\ |App_profiler|\ |App_qc_dashboard|\ |App_lineage|\ |App_layer_viewer|

|App_graph_builder|\ |App_tabulate|\ |App_feature_dict|\ |App_trellis|\ |App_gate_editor|

|App_feature_explorer|\ |App_outliers|

**Analyser**

|App_analyze_plaques|\ |App_recruitment|\ |App_invasion|\ |App_replication|

**Design**

|App_experiment_design|\ |App_power|\ |App_dose_response|

.. |App_align| image:: ../../../spacr/resources/icons/workflow/apps/align.png
   :width: 19.9%
   :alt: Öppna API-dokumentationen för Align & Stitch
   :target: https://einarolafsson.github.io/spacr/api/spacr/align/index.html
   :align: middle
.. |App_convert| image:: ../../../spacr/resources/icons/workflow/apps/convert.png
   :width: 19.9%
   :alt: Öppna API-dokumentationen för Format Converter
   :target: https://einarolafsson.github.io/spacr/api/spacr/convert/index.html
   :align: middle
.. |App_foreign| image:: ../../../spacr/resources/icons/workflow/apps/foreign.png
   :width: 19.9%
   :alt: Öppna API-dokumentationen för Import Project
   :target: https://einarolafsson.github.io/spacr/api/spacr/foreign/index.html
   :align: middle
.. |App_external_masks| image:: ../../../spacr/resources/icons/workflow/apps/external_masks.png
   :width: 19.9%
   :alt: Öppna API-dokumentationen för External Masks
   :target: https://einarolafsson.github.io/spacr/api/spacr/external_masks/index.html
   :align: middle
.. |App_queue| image:: ../../../spacr/resources/icons/workflow/apps/queue.png
   :width: 19.9%
   :alt: Öppna API-dokumentationen för Plate Queue
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/plate_queue/index.html
   :align: middle
.. |App_batch| image:: ../../../spacr/resources/icons/workflow/apps/batch.png
   :width: 19.9%
   :alt: Öppna API-dokumentationen för Batch Runner
   :target: https://einarolafsson.github.io/spacr/api/spacr/batch/index.html
   :align: middle
.. |App_distributed_jobs| image:: ../../../spacr/resources/icons/workflow/apps/distributed_jobs.png
   :width: 19.9%
   :alt: Öppna API-dokumentationen för Distributed Jobs
   :target: https://einarolafsson.github.io/spacr/api/spacr/remote_execution/index.html
   :align: middle
.. |App_db_browser| image:: ../../../spacr/resources/icons/workflow/apps/db_browser.png
   :width: 19.9%
   :alt: Öppna API-dokumentationen för Database Browser
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/db_browser/index.html
   :align: middle
.. |App_make_masks| image:: ../../../spacr/resources/icons/workflow/apps/make_masks.png
   :width: 19.9%
   :alt: Öppna API-dokumentationen för Make Masks
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/make_masks/index.html
   :align: middle
.. |App_data_manager| image:: ../../../spacr/resources/icons/workflow/apps/data_manager.png
   :width: 19.9%
   :alt: Öppna API-dokumentationen för Data Manager
   :target: https://einarolafsson.github.io/spacr/api/spacr/data_manager/index.html
   :align: middle
.. |App_project_browser| image:: ../../../spacr/resources/icons/workflow/apps/project_browser.png
   :width: 19.9%
   :alt: Öppna API-dokumentationen för Project Browser
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/project_browser/index.html
   :align: middle
.. |App_plate_view| image:: ../../../spacr/resources/icons/workflow/apps/plate_view.png
   :width: 19.9%
   :alt: Öppna API-dokumentationen för Plate Viewer
   :target: https://einarolafsson.github.io/spacr/api/spacr/plate_qc/index.html
   :align: middle
.. |App_umap| image:: ../../../spacr/resources/icons/workflow/apps/umap.png
   :width: 19.9%
   :alt: Öppna API-dokumentationen för Image UMAP
   :target: https://einarolafsson.github.io/spacr/api/spacr/core/index.html
   :align: middle
.. |App_train_compare| image:: ../../../spacr/resources/icons/workflow/apps/train_compare.png
   :width: 19.9%
   :alt: Öppna API-dokumentationen för Training Runs
   :target: https://einarolafsson.github.io/spacr/api/spacr/train_compare/index.html
   :align: middle
.. |App_run_history| image:: ../../../spacr/resources/icons/workflow/apps/run_history.png
   :width: 19.9%
   :alt: Öppna API-dokumentationen för Run History
   :target: https://einarolafsson.github.io/spacr/api/spacr/run_journal/index.html
   :align: middle
.. |App_report| image:: ../../../spacr/resources/icons/workflow/apps/report.png
   :width: 19.9%
   :alt: Öppna API-dokumentationen för Report
   :target: https://einarolafsson.github.io/spacr/api/spacr/report/index.html
   :align: middle
.. |App_run_compare| image:: ../../../spacr/resources/icons/workflow/apps/run_compare.png
   :width: 19.9%
   :alt: Öppna API-dokumentationen för Run Compare
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/run_compare/index.html
   :align: middle
.. |App_investigate_hit| image:: ../../../spacr/resources/icons/workflow/apps/investigate_hit.png
   :width: 19.9%
   :alt: Öppna API-dokumentationen för Investigate Hit
   :target: https://einarolafsson.github.io/spacr/api/spacr/hit_investigation/index.html
   :align: middle
.. |App_control_chart| image:: ../../../spacr/resources/icons/workflow/apps/control_chart.png
   :width: 19.9%
   :alt: Öppna API-dokumentationen för Control Charts
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/control_chart/index.html
   :align: middle
.. |App_pipeline_graph| image:: ../../../spacr/resources/icons/workflow/apps/pipeline_graph.png
   :width: 19.9%
   :alt: Öppna API-dokumentationen för Pipeline Graph
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/pipeline_graph/index.html
   :align: middle
.. |App_profiler| image:: ../../../spacr/resources/icons/workflow/apps/profiler.png
   :width: 19.9%
   :alt: Öppna API-dokumentationen för Prediction Profiler
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/profiler/index.html
   :align: middle
.. |App_qc_dashboard| image:: ../../../spacr/resources/icons/workflow/apps/qc_dashboard.png
   :width: 19.9%
   :alt: Öppna API-dokumentationen för QC Dashboard
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/qc_dashboard/index.html
   :align: middle
.. |App_lineage| image:: ../../../spacr/resources/icons/workflow/apps/lineage.png
   :width: 19.9%
   :alt: Öppna API-dokumentationen för Lineage
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/lineage/index.html
   :align: middle
.. |App_layer_viewer| image:: ../../../spacr/resources/icons/workflow/apps/layer_viewer.png
   :width: 19.9%
   :alt: Öppna API-dokumentationen för Layer Viewer
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/layer_viewer/index.html
   :align: middle
.. |App_graph_builder| image:: ../../../spacr/resources/icons/workflow/apps/graph_builder.png
   :width: 19.9%
   :alt: Öppna API-dokumentationen för Graph Builder
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/graph_builder/index.html
   :align: middle
.. |App_tabulate| image:: ../../../spacr/resources/icons/workflow/apps/tabulate.png
   :width: 19.9%
   :alt: Öppna API-dokumentationen för Tabulate
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/tabulate/index.html
   :align: middle
.. |App_feature_dict| image:: ../../../spacr/resources/icons/workflow/apps/feature_dict.png
   :width: 19.9%
   :alt: Öppna API-dokumentationen för Feature Dictionary
   :target: https://einarolafsson.github.io/spacr/api/spacr/feature_dict/index.html
   :align: middle
.. |App_trellis| image:: ../../../spacr/resources/icons/workflow/apps/trellis.png
   :width: 19.9%
   :alt: Öppna API-dokumentationen för Small Multiples
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/trellis/index.html
   :align: middle
.. |App_gate_editor| image:: ../../../spacr/resources/icons/workflow/apps/gate_editor.png
   :width: 19.9%
   :alt: Öppna API-dokumentationen för Gate Editor
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/gate_editor/index.html
   :align: middle
.. |App_feature_explorer| image:: ../../../spacr/resources/icons/workflow/apps/feature_explorer.png
   :width: 19.9%
   :alt: Öppna API-dokumentationen för Feature Explorer
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/feature_explorer/index.html
   :align: middle
.. |App_outliers| image:: ../../../spacr/resources/icons/workflow/apps/outliers.png
   :width: 19.9%
   :alt: Öppna API-dokumentationen för Outliers
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/outliers/index.html
   :align: middle
.. |App_analyze_plaques| image:: ../../../spacr/resources/icons/workflow/apps/analyze_plaques.png
   :width: 19.9%
   :alt: Öppna API-dokumentationen för Plaque Assay
   :target: https://einarolafsson.github.io/spacr/api/spacr/submodules/index.html
   :align: middle
.. |App_recruitment| image:: ../../../spacr/resources/icons/workflow/apps/recruitment.png
   :width: 19.9%
   :alt: Öppna API-dokumentationen för Recruitment
   :target: https://einarolafsson.github.io/spacr/api/spacr/submodules/index.html
   :align: middle
.. |App_invasion| image:: ../../../spacr/resources/icons/workflow/apps/invasion.png
   :width: 19.9%
   :alt: Öppna API-dokumentationen för Invasion Assay
   :target: https://einarolafsson.github.io/spacr/api/spacr/submodules/index.html
   :align: middle
.. |App_replication| image:: ../../../spacr/resources/icons/workflow/apps/replication.png
   :width: 19.9%
   :alt: Öppna API-dokumentationen för Replication Assay
   :target: https://einarolafsson.github.io/spacr/api/spacr/submodules/index.html
   :align: middle
.. |App_experiment_design| image:: ../../../spacr/resources/icons/workflow/apps/experiment_design.png
   :width: 19.9%
   :alt: Öppna API-dokumentationen för Experiment Design
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/experiment_design/index.html
   :align: middle
.. |App_power| image:: ../../../spacr/resources/icons/workflow/apps/power.png
   :width: 19.9%
   :alt: Öppna API-dokumentationen för Power / Design
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/power/index.html
   :align: middle
.. |App_dose_response| image:: ../../../spacr/resources/icons/workflow/apps/dose_response.png
   :width: 19.9%
   :alt: Öppna API-dokumentationen för Dose–Response
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/dose_response/index.html
   :align: middle

.. spacr-workflow-end

Välj en arbetsflödesmodul för att öppna dess API-sida. Rutnätet innehåller alla övriga program i samma kategorier och ordning som på spaCR:s startsida.


Installera spaCR
----------------

Skrivbordsprogram
~~~~~~~~~~~~~~~~~~~

Skrivbordsinstallatörerna inkluderar en privat miljö Python, så conda och en befintlig installation Python krävs inte.

.. spacr-installer-links-begin

|InstallerLinux| |InstallerMacOS| |InstallerWindows| |InstallerLegacy|

.. |InstallerWindows| image:: ../../../spacr/resources/icons/platforms/windows.png
   :width: 64
   :alt: Hämta spaCR 1.5.0.4 för Windows 10/11
   :target: https://github.com/EinarOlafsson/spacr/releases/download/v1.5.0.4/SpaCR-1.5.0.4-Windows-Online-Setup.exe
.. |InstallerMacOS| image:: ../../../spacr/resources/icons/platforms/macos.png
   :width: 64
   :alt: Hämta spaCR 1.5.0.4 för macOS 11+ (Intel och Apple Silicon)
   :target: https://github.com/EinarOlafsson/spacr/releases/download/v1.5.0.4/SpaCR-1.5.0.4-macOS-Universal-Online.pkg
.. |InstallerLinux| image:: ../../../spacr/resources/icons/platforms/linux.png
   :width: 64
   :alt: Hämta spaCR 1.5.0.4 för 64-bitars Linux
   :target: https://github.com/EinarOlafsson/spacr/releases/download/v1.5.0.4/SpaCR-1.5.0.4-Linux-x86_64-Online.run
.. |InstallerLegacy| image:: ../../../spacr/resources/icons/platforms/legacy.png
   :width: 64
   :alt: Äldre spaCR-installationsprogram
   :target: ../../source/installers.rst

.. spacr-installer-links-end

De tre första ikonerna laddar ner den aktuella utgåvan. Ikonen spaCR öppnar hela installationsprogrammets arkiv. Installerlänkar och versionerade filnamn uppdateras av utgivningsarbetsflödet; tidigare installatörer finns kvar i samma utgivningsarkiv.

Gör den hämtade filen körbar i Linux och kör den:

.. code-block:: bash

   chmod +x SpaCR-*-Linux-x86_64-Online.run
   ./SpaCR-*-Linux-x86_64-Online.run

På macOS, öppna ``.pkg``. Nuvarande beta notariseras inte. Om Gatekeeper blockerar den, välj **Systeminställningar → Integritet & Säkerhet → Öppna ändå**.

Se `installationsguide <../../source/installer_guide.rst>`_ för uppdatering, avinstallera, offline och felsökningsinstruktioner.

Installation med conda-forge
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Det officiella conda-forge-paketet installerar spaCR och dess skrivbordsberoenden i den aktiva miljön:

.. code-block:: bash

   conda create -n spacr python=3.12 -y
   conda activate spacr
   conda install conda-forge::spacr
   spacr

Installation från PyPI
~~~~~~~~~~~~~~~~~~~~~~

För PyPI-utgåvan installerar du spaCR med pip i en Conda-miljö. Python 3.12 ger det största urvalet av valfria vetenskapliga paket:

.. code-block:: bash

   conda create -n spacr python=3.12 -y
   conda activate spacr
   python -m pip install --upgrade pip
   python -m pip install "spacr[qt]"
   spacr

spaCR stöder Python **3.9 till 3.14**, med undantag för Python 3.14.1 som inte stöds av torchvision. Linux rekommenderas för CUDA-arbetsflöden; macOS och Windows stöds också.

Utelämna Qt på en server, ett beräkningskluster eller en CI-körare:

.. code-block:: bash

   python -m pip install spacr
   spacr-run --list

Optional integrations are installed separately, for example ``spacr[zarr]``, ``spacr[omero]``, ``spacr[napari]`` and ``spacr[czi,nd2,lif]``. See the `installationsguide <../../source/installer_guide.rst>`_ for the complete extras and Python-version compatibility table.

Kommandoradskommandon
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   spacr                                      # launch the Qt application
   spacr-doctor                               # diagnose the installation
   spacr-run --list                           # list headless modules
   spacr-run --describe MODULE                # inspect a module contract
   spacr-run MODULE --settings settings.csv   # execute a module
   spacr-run validate --module MODULE \
       --settings settings.csv                # validate before running
   spacr-repro RUN_DIR                        # replay a recorded run

Ange ``SPACR_LOG_LEVEL=DEBUG`` vid felsökning. Roterande loggar skrivs till ``~/.spacr/logs/spacr.log``.

``spacr-run --list`` listar moduler med kommandoradsposter för körning utan grafiskt gränssnitt. GUI-bundna moduler för annotering, kurering, jämförelse och utforskning utelämnas.


Det här kan du göra
-------------------

Det primära arbetsflödet består av sex moduler:

- **Mask** segmenterar celler, cellkärnor, patogener och organeller med Cellpose.
- **Measure** skriver morfologiska, intensitets-, textur-, rumsliga och kolokaliseringsmått samt objektutsnitt till SQLite.
- **Annotate** märker objektutsnitt i ett tangentbordsstyrt rutnät och stöder köer för aktiv inlärning.
- **Classify** tränar bild- eller mätningsbaserade modeller och sparar prestandan på undanhållna data med varje kontrollpunkt.
- **Map Barcodes** kopplar FASTQ-läsningar till brunnar och gRNA:er och rapporterar QC för förekomst, kollisioner och täckning.
- **Regression** skattar effekter för guider, gener, betingelser och kontroller med modellfamiljer för kontinuerliga data, andelar och antal.

Samma projekt kan även användas för att utforma plattor, uppskatta statistisk styrka, korrigera batcheffekter, granska segmenteringskvalitet, utforska sammankopplade diagram och bildutsnitt, exportera AnnData, återuppta avbruten bearbetning och registrera inställningarna bakom varje resultat.

Moduler som är tillgängliga från värdvyer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Tjugo moduler är integrerade i relaterade värdvyer i stället för att visas som separata paneler på startsidan. Varje modul öppnas från värdvyns rubrikrad och använder det aktiva projektet. Mask, Measure, Annotate, Classify, Map Barcodes, Regression, Image UMAP och Make Masks tillhandahåller dessa integrerade moduler. Hjälp- och API-dokumentationen är fortsatt tillgänglig, och moduler med bearbetningsflöden kan fortfarande köras utan grafiskt gränssnitt. `Funktionsguiden <../../source/features.rst>`_ anger varje integrerad modul och dess värd.

Make Masks
~~~~~~~~~~

Make Masks visas under **Data** och används för manuell korrigering av segmenteringsmasker. Rubrikraden ger även åtkomst till Cellpose-arbetsflödena. Arbetsytan har nio verktyg: **Brush**, **Erase**, **Erase object**, **Wand +**, **Wand −**, **Draw**, **Divide**, **Zoom** och **Recrop**. Draw skapar en fylld etikett från en sluten frihandskontur. Divide delar ett sammanvuxet objekt längs en användardefinierad linje och bevarar alla övriga objektetiketter.

Recrop extraherar ett bildfält med ett enda objekt från en förberedd bild som innehåller flera objekt. En avgränsningsruta runt ett objekt sparar motsvarande bild- och maskområden som ett nytt bildfält, schemalägger fältet direkt efter det aktuella och tar bort det ursprungliga fältet med flera objekt från kureringskön. Recrop ändrar det aktiva bildfältet i stället för etikettpixlarna.

När Cellpose-SAM körs från Make Masks visas två mellanresultat bredvid masken: **cell-sannolikhetskartan** och **flödesfältet**. Masken definieras genom ett tröskelvärde på sannolikhetskartan, och flödeskonsistenskontroller kan avvisa objekt vars härledda flöden avviker från det förutsagda fältet. Granska dessa resultat för att skilja låg cellsannolikhet från inkonsekvent flöde vid bedömning av en felaktig eller ofullständig mask.

Objekt och inställningar
~~~~~~~~~~~~~~~~~~~~~~~~

spaCR stöder cell-, kärn- och patogenobjekt, ett cytoplasmaobjekt som härleds från deras masker samt mellan noll och tjugosex organellplatser. Varje organellplats har en oberoende kanal, diameter, morfologiförinställning och detektionsmetod.

Inställningspanelen visar endast reglage när de är tillämpliga. Organellplatser över det konfigurerade antalet döljs, objekt utan tilldelad kanal utesluts från körningen och morfologispecifika reglage visas endast för den valda metoden. Reglagen **3D** och **Time** anger dimensionaliteten: ``z_stack`` aktiverar volyminställningar, ``timelapse`` aktiverar spårningsinställningar och fyrdimensionella inställningar visas när båda är aktiverade.

Välj nästa sida efter vad du vill göra:

- `Interaktiva handledningar <https://einarolafsson.github.io/spacr/tutorials/>`_ – 73 guidade arbetsflöden från installation genom träffundersökning.
- `Snabbstart Python API <../../source/python_api.rst>`_ – kör och validera arbetsflöden från skript, anteckningsböcker eller ett kluster.
- `Handbok för funktioner <../../source/features.rst>`_ – kapacitet, mognad och valfria integrationer.
- `Kurerad API referens <https://einarolafsson.github.io/spacr/api/index.html>`_ – understödda ingångspunkter för uppgift, med den fullständiga modulreferensen en nivå djupare.
- `Språk- och översättningsguide <../../source/localization.rst>`_ – gränssnittsspråk, kontextuell hjälp och policy för vetenskaplig output.

Språk och översättning
~~~~~~~~~~~~~~~~~~~~~~

Gränssnittet stöder tio språk i navigering och inställningar. AI- och LIVE-kontroller, modulbeskrivningar och granskad kontexthjälp översätts också. Byt språk under **spaCR → Inställningar → Språk** utan att starta om. Loggar, sökvägar, databasvärden och mätningar översätts aldrig; vetenskapliga utdata förblir på kanonisk engelska. Se `policyn för kontexthjälp <../../source/localization.rst#contextual-help>`_.

Animerad hjälp för inställningar
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Inställningar med en visuell förklaring har kontrollen **Animation** i verktygstipset. Bläddra i `galleriet med inställningsanimationer <https://einarolafsson.github.io/spacr/setting_animations.html>`_ eller `registret över inställningsanimationer <https://einarolafsson.github.io/spacr/api/spacr/setting_animations/index.html>`_.

Data
----

Referensdatauppsättningar
~~~~~~~~~~~~~~~~~~~~~~~~~

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

Prestandadiagnostik
----------------------

Skapa en maskinvarurapport och bifoga den till ett prestandarelaterat ärende::

    python tools/spacr_hardware_report.py

Kommandot skriver ut en rapport och sparar en kopia under ``~/.spacr/reports``; den sista raden anger sökvägen till den sparade filen. ``--quick`` utelämnar de längre prestandamätningarna och ``--out PATH`` väljer en annan plats för utdata.

Rapporten öppnar inget projekt och läser inga projektdata. Den registrerar tidsåtgång för import och numeriska bibliotek, bildskärmsskalning, aktiva inställningar, konstruktion av huvudfönstret och modulvyer samt animationsprestanda. Rapportfilen är den enda utdata som skapas.

Rapporten identifierar även emulering av processorarkitektur, exempelvis en x86_64-version av Python på Apple Silicon, och vilken BLAS-implementation NumPy använder. Båda kan påverka prestandan avsevärt.

Bidrag och support
------------------------

Skicka felrapporter och avgränsade funktionsförslag via `GitHub-ärenden <https://github.com/EinarOlafsson/spacr/issues>`_. Ange spaCR-version, operativsystem, Python-version, modulinställningar och relevant loggutdrag när du rapporterar ett fel. ``spacr-doctor`` samlar in det mesta av denna information; bifoga maskinvarurapporten vid prestandaproblem.

Licens
~~~~~~~~~

Den aktuella utvecklingsgrenen är källkodstillgänglig under `PolyForm Noncommercial License 1.0.0 <https://github.com/EinarOlafsson/spacr/blob/main/LICENSE>`_. Kommersiell användning kräver en separat licens från upphovsrättsinnehavaren. Utgivna versioner till och med spaCR 1.4.9.9 är fortsatt tillgängliga under den MIT-licens som medföljde dessa versioner.

Handledningar
~~~~~~~~~~~~~

Det `interaktiva biblioteket med spaCR-handledningar <https://einarolafsson.github.io/spacr/tutorials/>`_ innehåller berättade och textade genomgångar av installationen och varje programflöde: 73 lektioner med 50 röster på åtta språk.

Citera spaCR
~~~~~~~~~~~~

Om spaCR bidrar till din forskning, citera:

Olafsson EB, *et al.* En poolad bildbaserad CRISPR screening identifierar EAF1 som en *T. gondii* modulator för ESCRT subversion.

`BioRxiv preprint <https://www.biorxiv.org/content/10.64898/2026.07.08.737057v1>`_ · `programvaruarkiv <https://doi.org/10.5281/zenodo.21343316>`_

Tack
~~~~~~~~~~~~~~~

spaCR bygger på öppen vetenskaplig programvara, bland annat NumPy, pandas, scikit-image, scikit-learn, Cellpose, PyTorch och Qt. Se `information om översättningsmodellerna <../TRANSLATION_MODELS.md>`_ för modellerna som användes till den flerspråkiga dokumentationen och gränssnittskatalogerna.
