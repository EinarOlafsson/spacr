|Docs| |Tutorials| |PyPI| |Conda| |Python| |Tests| |Qt| |Source| |Issues| |License| |Preprint| |DOI|

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
.. |Preprint| image:: https://img.shields.io/badge/bioRxiv-2026.07.08.737057-BF2636
   :target: https://www.biorxiv.org/content/10.64898/2026.07.08.737057v1
   :alt: Zenodo-DOI
.. |DOI| image:: https://img.shields.io/badge/DOI-10.5281%2Fzenodo.21343316-blue
   :target: https://doi.org/10.5281/zenodo.21343316
   :alt: Senaste installationsprogrammen
.. |Release| image:: https://img.shields.io/github/v/release/EinarOlafsson/spacr?label=Installers
   :target: https://github.com/EinarOlafsson/spacr/releases/latest
   :alt: conda-forge-version
.. |Conda| image:: https://anaconda.org/conda-forge/spacr/badges/version.svg
   :target: https://anaconda.org/conda-forge/spacr
   :alt: spaCR

.. image:: ../../../spacr/resources/icons/logo_spacr_readme.png
   :alt: spaCR
   :width: 920

spaCR
=====

.. spacr-language-picker-begin

Språk: `🌐 Svenska ▾ <README.md>`_

.. spacr-language-picker-end

**Rumslig fenotypanalys av CRISPR-screeningar.**

spaCR segmenterar och mäter enskilda celler i mikroskopibilder med högt innehåll, integrerar fenotyper per objekt med sekvenseringshärledd guideförekomst och uppskattar vilka gener som är associerade med fenotypiska förändringar. Med plattbilder och FASTQ-läsningar som utgångspunkt producerar programmet mätningar per objekt, tränade klassificerare, effektskattningar per guide och gen samt en rangordnad träfflista.

Segmenterings-, mät-, annoterings- och klassificeringsmodulerna körs även utan en sekvensarm.

Bilder, masker, bildutsnitt, mätningar, annoteringar, förutsägelser, streckkoder och brunnsidentifierare ligger i ett och samma SQLite-projekt.

Körs som ett skrivbordsprogram eller utan grafiskt gränssnitt på en arbetsstation, server eller kluster.

Hårdvarustöd
~~~~~~~~~~~~~~~~

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

Stödda (stabila)  och genomförda (beta) - CPU stöd endast

.. spacr-hardware-end


Installera spaCR
----------------

Skrivbordsprogram
~~~~~~~~~~~~~~~~~~~

Installatörerna buntar ihop sina egna Python. Conda krävs inte.

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

Installation från PyPI
~~~~~~~~~~~~~~~~~~~~~~

För PyPI-utgåvan installerar du spaCR med pip i en Conda-miljö. Python 3.12 ger det största urvalet av valfria vetenskapliga paket:

.. code-block:: bash

   conda create -n spacr python=3.12 -y
   conda activate spacr
   python -m pip install --upgrade pip
   python -m pip install "spacr[qt]"
   spacr

spaCR stöder Python **3.9 through 3.14**, utom Python 3.14.1, som torchvision utesluter. Linux rekommenderas för de tyngsta CUDA- och ROCm-arbetsflödena; macOS och Windows stöds också, och båda använder sina GPU:er — macOS via Metal, som täcker Apple Silicon och AMD-korten i Intel-Mac-datorer, och Windows via CUDA eller DirectML.

Utelämna Qt på en server, ett beräkningskluster eller en CI-körare:

.. code-block:: bash

   python -m pip install spacr
   spacr-run --list

Optional integrations are installed separately, for example ``spacr[zarr]``, ``spacr[omero]``, ``spacr[napari]`` and ``spacr[czi,nd2,lif]``. See the `installationsguide <../../source/installer_guide.rst>`_ for the complete extras and Python-version compatibility table.

Installation med conda-forge
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Det officiella conda-forge-paketet installerar spaCR och dess skrivbordsberoenden i den aktiva miljön:

.. code-block:: bash

   conda create -n spacr python=3.12 -y
   conda activate spacr
   conda install conda-forge::spacr
   spacr

Installera från källkod
~~~~~~~~~~~~~~~~~~~~~~~

Klonera arkivet och installera det i redigerbart läge, så din arbetskopia *är* det installerade paketet och redigeringar träder i kraft utan att installera om::

    git clone https://github.com/EinarOlafsson/spacr.git
    cd spacr
    conda create -n spacr python=3.12 -y
    conda activate spacr
    pip install -e .
    spacr

Standardfilialen är ``nightly``. För en specifik utgåva::

    git clone --branch v1.5.0.5 https://github.com/EinarOlafsson/spacr.git

För att dra senare ändringar, inifrån klonen::

    git pull
    pip install -e .

Den andra raden behövs bara när beroenden eller ingångspunkter ändras; Python kod plockas upp utan den. Om ett kommando fortfarande kör gammal kod efter dragning, rapporterar ``spacr-doctor`` som ``spacr`` faktiskt är på din väg, vilket är den vanliga orsaken.

Installera från källa (ljus)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Full klon: 427 MB. Kärnklon: 76 MB.

::

    curl -fsSL https://raw.githubusercontent.com/EinarOlafsson/spacr/nightly/packaging/install_from_source.sh -o install_spacr.sh
    sh install_spacr.sh --branch nightly

Hoppar över ``docs/``, ``tests/`` och Cellpose kontrollpunkter, arkiverade siffror och utökade översättningskataloger. Resultatet är en normal checkout.

Options: ``--dir``, ``--branch`` (default ``main``), ``--with-tests``, ``--with-docs``, ``--with-translations``, ``--no-install``.

``packaging/source_install_excludes.txt`` listar varje överhoppad sökväg.


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


Kärnarbetsflöde
---------------

Det primära arbetsflödet består av sex moduler:

- **Mask** segmenterar celler, cellkärnor, patogener och organeller med Cellpose.
- **Measure** skriver morfologiska, intensitets-, textur-, rumsliga och kolokaliseringsmått samt objektutsnitt till SQLite.
- **Annotate** märker objektutsnitt i ett tangentbordsstyrt rutnät och stöder köer för aktiv inlärning.
- **Classify** tränar bild- eller mätningsbaserade modeller och sparar prestandan på undanhållna data med varje kontrollpunkt.
- **Map Barcodes** kopplar FASTQ-läsningar till brunnar och gRNA:er och rapporterar QC för förekomst, kollisioner och täckning.
- **Regression** skattar effekter för guider, gener, betingelser och kontroller med modellfamiljer för kontinuerliga data, andelar och antal.

spaCR-moduler
-------------

.. spacr-workflow-begin

Kärna
^^^^^

Core sequence from microscopy images through segmentation, measurements,
annotations, classification, barcode mapping and regression.

| |Module_mask|\ |Module_measure|\ |Module_annotate|\ |Module_classify_merged|\ |Module_map_barcodes|\ |Module_regression|

Data
^^^^

Import images and tables into spaCR projects and execute reproducible
multi-plate workflows.

| |Module_foreign|\ |Module_run_compare|\ |Module_experiment_design|\ |Module_power|\ |Module_dose_response|\ |Module_qc_dashboard|

Verktyg
^^^^^^^

Point these at a project: edit masks by hand, stitch tiles, read an
embedding, draw a gate, build a plot, check quality.

| |Module_make_masks|\ |Module_align|\ |Module_umap|\ |Module_gate_editor|\ |Module_graph_builder|

Analyser
^^^^^^^^

Quantitative readouts for biological assays.

| |Module_analyze_plaques|\ |Module_recruitment|\ |Module_invasion|\ |Module_replication|

.. |Module_mask| image:: ../../../spacr/resources/icons/workflow/mask.png
   :width: 16.0%
   :alt: Öppna API-dokumentationen för Mask
   :target: https://einarolafsson.github.io/spacr/api/spacr/core/index.html#spacr.core.preprocess_generate_masks
   :align: middle
.. |Module_measure| image:: ../../../spacr/resources/icons/workflow/measure.png
   :width: 16.0%
   :alt: Öppna API-dokumentationen för Measure
   :target: https://einarolafsson.github.io/spacr/api/spacr/measure/index.html
   :align: middle
.. |Module_annotate| image:: ../../../spacr/resources/icons/workflow/annotate.png
   :width: 16.0%
   :alt: Öppna API-dokumentationen för Annotate
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/annotate/index.html
   :align: middle
.. |Module_classify_merged| image:: ../../../spacr/resources/icons/workflow/classify_merged.png
   :width: 16.0%
   :alt: Öppna API-dokumentationen för Classify
   :target: https://einarolafsson.github.io/spacr/api/spacr/classify/index.html
   :align: middle
.. |Module_map_barcodes| image:: ../../../spacr/resources/icons/workflow/map_barcodes.png
   :width: 16.0%
   :alt: Öppna API-dokumentationen för Map Barcodes
   :target: https://einarolafsson.github.io/spacr/api/spacr/sequencing/index.html
   :align: middle
.. |Module_regression| image:: ../../../spacr/resources/icons/workflow/regression.png
   :width: 16.0%
   :alt: Öppna API-dokumentationen för Regression
   :target: https://einarolafsson.github.io/spacr/api/spacr/ml/index.html
   :align: middle
.. |Module_foreign| image:: ../../../spacr/resources/icons/workflow/apps/foreign.png
   :width: 16.0%
   :alt: Öppna API-dokumentationen för Import
   :target: https://einarolafsson.github.io/spacr/api/spacr/foreign/index.html
   :align: middle
.. |Module_run_compare| image:: ../../../spacr/resources/icons/workflow/apps/run_compare.png
   :width: 16.0%
   :alt: Öppna API-dokumentationen för Run Compare
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/run_compare/index.html
   :align: middle
.. |Module_experiment_design| image:: ../../../spacr/resources/icons/workflow/apps/experiment_design.png
   :width: 16.0%
   :alt: Öppna API-dokumentationen för Experiment Design
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/experiment_design/index.html
   :align: middle
.. |Module_power| image:: ../../../spacr/resources/icons/workflow/apps/power.png
   :width: 16.0%
   :alt: Öppna API-dokumentationen för Power / Design
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/power/index.html
   :align: middle
.. |Module_dose_response| image:: ../../../spacr/resources/icons/workflow/apps/dose_response.png
   :width: 16.0%
   :alt: Öppna API-dokumentationen för Dose–Response
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/dose_response/index.html
   :align: middle
.. |Module_qc_dashboard| image:: ../../../spacr/resources/icons/workflow/apps/qc_dashboard.png
   :width: 16.0%
   :alt: Öppna API-dokumentationen för QC
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/qc_dashboard/index.html
   :align: middle
.. |Module_make_masks| image:: ../../../spacr/resources/icons/workflow/apps/make_masks.png
   :width: 16.0%
   :alt: Öppna API-dokumentationen för Make Masks
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/make_masks/index.html
   :align: middle
.. |Module_align| image:: ../../../spacr/resources/icons/workflow/apps/align.png
   :width: 16.0%
   :alt: Öppna API-dokumentationen för Align & Stitch
   :target: https://einarolafsson.github.io/spacr/api/spacr/align/index.html
   :align: middle
.. |Module_umap| image:: ../../../spacr/resources/icons/workflow/apps/umap.png
   :width: 16.0%
   :alt: Öppna API-dokumentationen för Image UMAP
   :target: https://einarolafsson.github.io/spacr/api/spacr/core/index.html#spacr.core.generate_image_umap
   :align: middle
.. |Module_gate_editor| image:: ../../../spacr/resources/icons/workflow/apps/gate_editor.png
   :width: 16.0%
   :alt: Öppna API-dokumentationen för Gate Editor
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/gate_editor/index.html
   :align: middle
.. |Module_graph_builder| image:: ../../../spacr/resources/icons/workflow/apps/graph_builder.png
   :width: 16.0%
   :alt: Öppna API-dokumentationen för Graph Builder
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/graph_builder/index.html
   :align: middle
.. |Module_analyze_plaques| image:: ../../../spacr/resources/icons/workflow/apps/analyze_plaques.png
   :width: 16.0%
   :alt: Öppna API-dokumentationen för Plaque Assay
   :target: https://einarolafsson.github.io/spacr/api/spacr/submodules/index.html#spacr.submodules.analyze_plaques
   :align: middle
.. |Module_recruitment| image:: ../../../spacr/resources/icons/workflow/apps/recruitment.png
   :width: 16.0%
   :alt: Öppna API-dokumentationen för Recruitment
   :target: https://einarolafsson.github.io/spacr/api/spacr/submodules/index.html#spacr.submodules.analyze_recruitment
   :align: middle
.. |Module_invasion| image:: ../../../spacr/resources/icons/workflow/apps/invasion.png
   :width: 16.0%
   :alt: Öppna API-dokumentationen för Invasion Assay
   :target: https://einarolafsson.github.io/spacr/api/spacr/submodules/index.html#spacr.submodules.analyze_invasion
   :align: middle
.. |Module_replication| image:: ../../../spacr/resources/icons/workflow/apps/replication.png
   :width: 16.0%
   :alt: Öppna API-dokumentationen för Replication Assay
   :target: https://einarolafsson.github.io/spacr/api/spacr/submodules/index.html#spacr.submodules.analyze_replication
   :align: middle

.. spacr-workflow-end

Varje modul spaCR fartyg, i den ordning startskärmen listar dem: de sex rörledningsmodulerna först, sedan allt annat. Välj en bricka för att öppna modulens API sida.


Make Masks
~~~~~~~~~~

Make Masks appears under **Tools** for manual correction of segmentation masks; its masthead opens the Cellpose workflows. Nine tools: **Brush**, **Erase**, **Erase object**, **Wand +**, **Wand −**, **Draw**, **Divide**, **Zoom** and **Recrop**. Draw makes one filled label from a closed outline, Divide separates a merged object along a drawn line, Recrop turns one object in a crowded field into its own field.

Se `funktionsguide <../../source/features.rst>`_ för varje verktyg.

Övriga resurser
~~~~~~~~~~~~~~~

- `Interaktiva handledningar <https://einarolafsson.github.io/spacr/tutorials/>`_ – 73 guidade arbetsflöden från installation genom träffundersökning.
- `Snabbstart Python API <../../source/python_api.rst>`_ – kör och validera arbetsflöden från skript, anteckningsböcker eller ett kluster.
- `Handbok för funktioner <../../source/features.rst>`_ – kapacitet, mognad och valfria integrationer.
- `Kurerad API referens <https://einarolafsson.github.io/spacr/api/index.html>`_ – understödda ingångspunkter för uppgift, med den fullständiga modulreferensen en nivå djupare.
- `Språk- och översättningsguide <../../source/localization.rst>`_ – gränssnittsspråk, kontextuell hjälp och policy för vetenskaplig output.

Språk och översättning
~~~~~~~~~~~~~~~~~~~~~~

Gränssnittet stöder tio språk i navigering och inställningar. AI- och LIVE-kontroller, modulbeskrivningar och granskad kontexthjälp översätts också. Byt språk under **spaCR → Inställningar → Språk** utan att starta om. Loggar, sökvägar, databasvärden och mätningar översätts aldrig; vetenskapliga utdata förblir på kanonisk engelska. Se `policyn för kontexthjälp <docs/source/localization.rst#contextual-help>`_.

De nio icke-engelska katalogerna är maskinskrivna och tekniskt granskade i stället för att läsa ända till slutet av en infödd talare av varje språk. De `Översynens tillämpningsområde <docs/i18n/REVIEW_SCOPE_2026-09-04.md>`_ poster vilka språk har haft ett mänskligt pass, hur mycket av de corpus som täcker, och varje term kvar på engelska genom beslut.

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

Förlaga till djurpark
~~~~~~~~~~~~~~~~~~~~~

spaCR skickar en katalog med utbildade modeller och hämtar dem på begäran. Öppna **Modellzoo** från startskärmen för att bläddra och installera dem, eller namnge en nyckel i en inställningsfil -- ``pathogen_model: toxoplasma_pv_v1`` -- och modellen laddas ner och kontrolleras första gången den behövs. Varje publicerad post innehåller en SHA-256; en post utan en nekas snarare än installeras, eftersom en trunkerad eller ersatt kontrollpunkt inte kan meddelas från den verkliga.

.. spacr-model-zoo-begin

.. list-table::
   :header-rows: 1
   :widths: 24 34 42

   * - Model
     - Training data
     - Hold-out performance
   * - ``toxoplasma_pv_v1``
       (Cellpose-SAM (cpsam_v2))
     - anti-Toxoplasma-biotin and DsRed PV lumen; 115 images, 1 dataset
     - F1 0.867 against 0.713 for stock cpsam, at IoU 0.5
   * - ``toxoplasma_plaque_v1``
       (Cellpose-SAM (cpsam))
     - crystal violet plaque wells; 184 wells from 3 datasets, 95 in-house and 89 literature
     - F1 0.856 in-domain; 0.806 on literature (3-fold cross-validated, SD 0.020)
   * - ``toxoplasma_well_detector_v1``
       (YOLO11n)
     - whole-plate and multi-well crystal violet images; 562 images from 1 dataset, 190 of them with no well in them
     - mAP50 0.993, mAP50-95 0.886, precision and recall both 0.987

.. spacr-model-zoo-end

Varje figur ovan mäts på bilder modellen aldrig såg i träning.

**Precision** is how many of the objects a model reported are real; **recall** is how many of the real objects it found. They fail in opposite directions: poor precision invents plaques, poor recall misses them.

**F1** är de två kombinerade, och citeras eftersom var och en av dem är trivialt gamed - rapportera en omisskännlig plakett för nära perfekt precision, eller varje mörk blob för nära-perfect recall. Som du hellre skulle förlora beror på analysen, och räkning är vanligtvis bättre betjänas av over-calling: plaque-modellen accepterades med precision 0.858 med reclosure 0.811 under en tidigare runda på 0,939 och 0,631.

**IoU**, intersection over union, is how much a predicted object and the real one overlap, divided by the area they cover together. It is the ruler the rest are read against, so a score means nothing without its threshold: "F1 0.867 at IoU 0.5" counts a vacuole as found when the two outlines agree over half their combined area.

**mAP50** och **mAP50-95** tillhör detektorn. Den första frågar om brunnarna hittades; den andra upprepar det över tio tröskelvärden från 0,5 till 0,95, så den frågar också hur tätt varje låda dras. Klyftan mellan dem är placering, inte detektion.

**Cross-validerad**, med en **SD**, betyder att poängen är medelvärdet av tre körningar på olika splitar och SD är hur långt de flyttade isär. En split kan ha tur: denna modells litteraturfigur är 0,834 på en enda 19-håls split och 0,806 på alla tre.

Modeller är värd på sin författares eget Hugging Face konto, så bidragande betyder inte att ge skrivåtkomst till någon annans. ``spacr.model_zoo`` s ``publish_model`` utför uppladdningen och skriver ut katalograden att lägga till.


Prestandadiagnostik
----------------------

Skapa en maskinvarurapport och bifoga den till ett prestandarelaterat ärende::

    python tools/spacr_hardware_report.py

Sparar till ``~/.spacr/reports`` och skriver ut sökvägen. ``--quick`` hoppar över de längre riktmärkena; ``--out PATH`` anger platsen.

Läser inga projektdata. Tidsimport, numeriska bibliotek, fönsterkonstruktion och animering. Rapporter processor-arkitektur emulering (en x86_64 Python bygga på Apple Silicon) och NumPy s BLAS genomförande.

Kommandoradsreferens
----------------------

Varje kommando nedan installeras av ``pip install spacr``. Alla av dem accepterar ``--help``.

Lansering av ansökan
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   spacr              # the desktop application
   spacr-tutorial     # the interactive tutorial library
   spacr-server       # no first-run setup screen, for unattended launches

``spacr-server`` hoppar över skärmen för modal inställning, som annars skulle blockera ett oövervakat jobb.

``spacr-qt`` och ``spacr-nightly`` är alias till ``spacr``.

När spaCR inte kommer att starta
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   spacr-doctor       # diagnose the installation and say how to fix it
   safespacr          # the least spaCR that can still change a setting

``spacr-doctor`` skriver ut en rad per check, med ett kommando att köra för varje fel. Det rapporterar också som ``spacr`` är på sökvägen, vilket är vad en gammal redigerbar installera skuggor.

``safespacr`` reads every preference as its default and forces the backdrop, animations, verbose logging and preloading off. Use it when a saved preference breaks the launch. It changes nothing permanently.

Drivmoduler utan grafiskt gränssnitt
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Ingen Qt, ingen visning – för kluster, servrar och CI.

.. code-block:: bash

   spacr-run --list                              # modules with a headless entry
   spacr-run --describe MODULE                   # what a module consumes and produces
   spacr-run validate --module MODULE \
       --settings settings.csv                   # check settings before spending the run
   spacr-run MODULE --settings settings.csv      # execute
   spacr-remote --help                           # submit and monitor SSH, Slurm or cloud jobs

``validate`` läser samma inställningar som körningen skulle och rapporterar vad som saknas, motsägelsefullt eller pekar på ingenting.

``spacr-run --list`` visar endast moduler med en huvudlös ingångspunkt; annotering, kuration och prospektering är interaktiva och utelämnade.

Inspektera en körning efteråt
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Varje körning är journaliserad till ``~/.spacr/runs`` med dess inställningar, hashade ingångar, utgångar, varningar, versioner och frön.

.. code-block:: bash

   spacr-repro RUN_DIR        # replay a recorded run from its journal
   spacr-workspace RUN_DIR    # what that run had open: databases, montages, views

Granskning av data och installation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   spacr-db-audit DB      # SQLite health, integrity, locking, reader/writer probe
   spacr-leakage          # classifier train/test leakage audit
   spacr-plugins          # installed plugin registry and failure diagnostics

Miljö
~~~~~~~~~~~

.. code-block:: bash

   SPACR_LOG_LEVEL=DEBUG spacr      # verbose logging for one launch

Roterande loggar skrivs till ``~/.spacr/logs/spacr.log``. Bifoga filen till en felrapport.


Bidrag och support
------------------------

Skicka felrapporter och avgränsade funktionsförslag via `GitHub-ärenden <https://github.com/EinarOlafsson/spacr/issues>`_. Ange spaCR-version, operativsystem, Python-version, modulinställningar och relevant loggutdrag när du rapporterar ett fel. ``spacr-doctor`` samlar in det mesta av denna information; bifoga maskinvarurapporten vid prestandaproblem.

Licens
~~~~~~~~~

spaCR frisätts under `BSD 3-Clause-licens <https://github.com/EinarOlafsson/spacr/blob/main/LICENSE>`_.

Om spaCR bidrog till publicerade verk uppskattas en hänvisning och är inte ett villkor för licensen – se `Citing spaCR`_ nedan.

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

spaCR bygger på öppen vetenskaplig programvara, bland annat NumPy, pandas, scikit-image, scikit-learn, Cellpose, PyTorch och Qt. Se `information om översättningsmodellerna <docs/i18n/TRANSLATION_MODELS.md>`_ för modellerna som användes till den flerspråkiga dokumentationen och gränssnittskatalogerna.
