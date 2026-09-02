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
   :alt: BSD 3-Clause-licens
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

.. spacr-language-picker-begin

Språk: `🌐 Svenska ▾ <README.md>`_

.. spacr-language-picker-end

**Rumslig fenotypanalys av CRISPR-screeningar.**

spaCR segmenterar och mäter enskilda celler i mikroskopibilder med högt innehåll, integrerar fenotyper per objekt med sekvenseringshärledd guideförekomst och uppskattar vilka gener som är associerade med fenotypiska förändringar. Med plattbilder och FASTQ-läsningar som utgångspunkt producerar programmet mätningar per objekt, tränade klassificerare, effektskattningar per guide och gen samt en rangordnad träfflista.

Segmenterings-, mät-, annoterings- och klassificeringsmodulerna körs även utan en sekvensarm.

Bilder, masker, bildutsnitt, mätningar, kommentarer, förutsägelser, streckkoder och brunnsidentifierare lever i ett SQLite projekt.

Körs som ett skrivbordsprogram eller huvudlöst på en arbetsstation, server eller kluster.

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

🟢 supported (stable)   🟣 implemented (beta)   🔴 CPU support only

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

spaCR stöder Python **3.9 till 3.14**, utom Python 3.14.1, som torchvision utesluter. Linux rekommenderas för de tyngsta CUDA och ROCm arbetsflöden; macOS och Windows stöds också, och båda använder sina GPUs – macOS via metall, som täcker Apple Silicon och AMD-korten i Intel Macs, och Windows genom CUDA eller DirectML.

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

Samma projekt kan även användas för att utforma plattor, uppskatta statistisk styrka, korrigera batcheffekter, granska segmenteringskvalitet, utforska sammankopplade diagram och bildutsnitt, exportera AnnData, återuppta avbruten bearbetning och registrera inställningarna bakom varje resultat.

spaCR-moduler
-------------

.. spacr-workflow-begin

| |Module_mask|\ |Module_measure|\ |Module_annotate|\ |Module_classify_merged|\ |Module_map_barcodes|\ |Module_regression|
| |Module_foreign|\ |Module_run_compare|\ |Module_experiment_design|\ |Module_power|\ |Module_dose_response|\ |Module_qc_dashboard|
| |Module_make_masks|\ |Module_align|\ |Module_umap|\ |Module_gate_editor|\ |Module_graph_builder|\ |Module_analyze_plaques|
| |Module_recruitment|\ |Module_invasion|\ |Module_replication|

.. |Module_mask| image:: ../../../spacr/resources/icons/workflow/mask.png
   :width: 16.0%
   :alt: Öppna API-dokumentationen för Mask
   :target: https://einarolafsson.github.io/spacr/api/spacr/core/index.html
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
   :target: https://einarolafsson.github.io/spacr/api/spacr/core/index.html
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
   :target: https://einarolafsson.github.io/spacr/api/spacr/submodules/index.html
   :align: middle
.. |Module_recruitment| image:: ../../../spacr/resources/icons/workflow/apps/recruitment.png
   :width: 16.0%
   :alt: Öppna API-dokumentationen för Recruitment
   :target: https://einarolafsson.github.io/spacr/api/spacr/submodules/index.html
   :align: middle
.. |Module_invasion| image:: ../../../spacr/resources/icons/workflow/apps/invasion.png
   :width: 16.0%
   :alt: Öppna API-dokumentationen för Invasion Assay
   :target: https://einarolafsson.github.io/spacr/api/spacr/submodules/index.html
   :align: middle
.. |Module_replication| image:: ../../../spacr/resources/icons/workflow/apps/replication.png
   :width: 16.0%
   :alt: Öppna API-dokumentationen för Replication Assay
   :target: https://einarolafsson.github.io/spacr/api/spacr/submodules/index.html
   :align: middle

.. spacr-workflow-end

Varje modul spaCR fartyg, i den ordning startskärmen listar dem: de sex rörledningsmodulerna först, sedan allt annat. Välj en bricka för att öppna modulens API sida.


Make Masks
~~~~~~~~~~

Make Masks appears under **Tools** for manual correction of segmentation masks; its masthead opens the Cellpose workflows. Nine tools: **Brush**, **Erase**, **Erase object**, **Wand +**, **Wand −**, **Draw**, **Divide**, **Zoom** and **Recrop**. Draw makes one filled label from a closed outline, Divide separates a merged object along a drawn line, Recrop turns one object in a crowded field into its own field.

Cellpose- SAM körs här visar kartan över cellsannolikhet och flödesfältet bredvid masken. Se `funktionsguide <../../source/features.rst>`_ för varje verktyg.

**Other resources**

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

Förlaga till djurpark
~~~~~~~~~~~~~~~~~~~~~

spaCR levereras med en katalog med tränade modeller och hämtar dem vid behov. Öppna **Modellzoo** från startskärmen för att bläddra bland dem och installera dem, eller ange en nyckel i en inställningsfil -- ``pathogen_model: toxoplasma_pv_v1`` -- och modellen laddas ner och kontrollsummeverifieras första gången den behövs. Varje publicerad post innehåller en SHA-256; en post utan en sådan avvisas i stället för att installeras, eftersom en trunkerad eller utbytt kontrollpunkt inte kan skiljas från den äkta.

.. spacr-model-zoo-begin

.. list-table::
   :header-rows: 1
   :widths: 26 30 44

   * - Key
     - Trained on
     - Measured performance and limits
   * - ``toxoplasma_pv_v1``
       (cpsam_v2_toxo_r2)
     - Toxoplasma tachyzoite parasitophorous vacuoles stained with goat anti-Toxoplasma-biotin, and tachyzoites expressing DsRed in the PV lumen. 115 pairs (104 train / 11 test), 100 epochs, base cpsam_v2
     - F1 0.867 at IoU 0.5 against 0.713 for stock cpsam; AJI 0.808 against 0.426; accuracy falls sharply above IoU 0.8 -- suited to counting and area rather than precise morphometry
   * - ``toxoplasma_plaque_v1``
       (cpsam_plaque_r3)
     - Toxoplasma gondii plaque assays; round 3, evaluated in-domain (NAS) and against a literature generalisation set
     - F1 0.856 in-domain and 0.834 on the literature set, against 0.718 / 0.755 for round 1; round 3 trades precision (0.939 down to 0.858) for recall (0.631 up to 0.811) on the literature set, which is the right direction for a counting assay
   * - ``toxoplasma_well_detector_v1``
       (yolo_welldetect_v3.pt)
     - Whole-plate and multi-well Toxoplasma plaque-assay images; yolo11n base, 150 epochs, batch 16, imgsz 640
     - mAP50 0.993, mAP50-95 0.886, precision and recall both 0.987; locates WELLS, not plaques; it is the front half of a two-stage pipeline with toxoplasma_plaque_v1, and the well it finds also gives the diameter that makes areas comparable across microscopes

.. spacr-model-zoo-end

Siffrorna ovan är de som mäts vid publiceringen, och gränserna anges med dem: en modell är användbar för det arbete den mättes på, inte för varje jobb. ``toxoplasma_well_detector_v1`` och ``toxoplasma_plaque_v1`` är de två halvorna av en rörledning -- detektorn hittar brunnarna, segmentören hittar placken inuti dem, och brunnens diameter är det som gör områden jämförbara mellan mikroskop.

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

spaCR distribueras under `BSD 3-Clause-licensen <https://github.com/EinarOlafsson/spacr/blob/main/LICENSE>`_.

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

spaCR bygger på öppen vetenskaplig programvara, bland annat NumPy, pandas, scikit-image, scikit-learn, Cellpose, PyTorch och Qt. Se `information om översättningsmodellerna <../TRANSLATION_MODELS.md>`_ för modellerna som användes till den flerspråkiga dokumentationen och gränssnittskatalogerna.
