|Platforms| |Python| |Qt| |Tests| |Release| |Issues| |Source| |Conda| |PyPI| |Conda Downloads| |PyPI Downloads| |Docs| |Tutorials| |Preprint| |DOI| |Cite| |License| |PyPI rank|

.. |Docs| image:: https://img.shields.io/github/actions/workflow/status/EinarOlafsson/spacr/pages%2Fpages-build-deployment?label=API%20Documentation
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
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/index.html#module-spacr.qt
   :alt: Qt-Oberfläche
.. |Source| image:: https://img.shields.io/badge/GitHub-Source-181717?logo=github
   :target: https://github.com/EinarOlafsson/spacr
   :alt: GitHub-Quellcode
.. |Issues| image:: https://img.shields.io/github/issues/EinarOlafsson/spacr
   :target: https://github.com/EinarOlafsson/spacr/issues
   :alt: GitHub-Issues
.. |License| image:: https://img.shields.io/github/license/EinarOlafsson/spacr
   :target: https://github.com/EinarOlafsson/spacr/blob/main/LICENSE
   :alt: BSD-3-Clause-Lizenz
.. |Preprint| image:: https://img.shields.io/badge/bioRxiv-2026.07.08.737057-BF2636
   :target: https://www.biorxiv.org/content/10.64898/2026.07.08.737057v1
   :alt: bioRxiv-Preprint
.. |DOI| image:: https://img.shields.io/badge/DOI-10.5281%2Fzenodo.21343316-blue
   :target: https://doi.org/10.5281/zenodo.21343316
   :alt: Zenodo-DOI
.. |Release| image:: https://img.shields.io/github/v/release/EinarOlafsson/spacr?label=Installers
   :target: https://github.com/EinarOlafsson/spacr/releases/latest
   :alt: Neueste Installationsprogramme
.. |Conda| image:: https://anaconda.org/conda-forge/spacr/badges/version.svg
   :target: https://anaconda.org/conda-forge/spacr
   :alt: conda-forge-Version
.. |Conda Downloads| image:: https://anaconda.org/conda-forge/spacr/badges/downloads.svg
   :target: https://anaconda.org/conda-forge/spacr
   :alt: conda-forge-Downloads
.. |Release date| image:: https://anaconda.org/conda-forge/spacr/badges/latest_release_date.svg
   :target: https://anaconda.org/conda-forge/spacr
   :alt: conda-forge-Datum der letzten Veröffentlichung
.. |PyPI Downloads| image:: https://static.pepy.tech/personalized-badge/spacr?period=total&units=INTERNATIONAL_SYSTEM&left_color=GRAY&right_color=GREEN&left_text=downloads
   :target: https://pepy.tech/projects/spacr
   :alt: PyPI-Downloads
.. |Platforms| image:: https://img.shields.io/badge/Platforms-Linux%20%7C%20macOS%20%7C%20Windows-lightgrey
   :target: https://github.com/EinarOlafsson/spacr/blob/nightly/docs/source/installers.rst
   :alt: Linux, macOS und Windows
.. |Cite| image:: https://img.shields.io/badge/Cite-CITATION.cff-8A2BE2
   :target: https://github.com/EinarOlafsson/spacr/blob/main/CITATION.cff
   :alt: spaCR zitieren
.. |PyPI rank| image:: https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fsql-clickhouse.clickhouse.com%2F%3Fuser%3Ddemo%26param_package_name%3Dspacr%26param_days%3D30%26query%3DWITH%2B%2528%2BSELECT%2Bsum%2528count%2529%2BFROM%2Bpypi.pypi_downloads_per_day%2BWHERE%2Bproject%2B%253D%2B%257Bpackage_name%253AString%257D%2BAND%2Bdate%2B%253E%253D%2BtoDate%2528now%2528%2527UTC%2527%2529%2529%2B-%2B%257Bdays%253AUInt16%257D%2BAND%2Bdate%2B%253C%2BtoDate%2528now%2528%2527UTC%2527%2529%2529%2B%2529%2BAS%2Bdownloads%2BSELECT%2Bdownloads%2BAS%2Bpackage_downloads%252C%2BcountIf%2528n%2B%253E%253D%2Bdownloads%2529%2BAS%2Brank%252C%2Bcount%2528%2529%2BAS%2Btotal_packages%252C%2B100.0%2B%252A%2Brank%2B%252F%2BnullIf%2528total_packages%252C%2B0%2529%2BAS%2Bpercentile%252C%2Bif%2528%2Btotal_packages%2B%253D%2B0%2BOR%2Bdownloads%2B%253D%2B0%252C%2B%2527no%2Bdata%2527%252C%2Bconcat%2528%2B%2527top%2B%2527%252C%2BtoString%2528ceil%25281000.0%2B%252A%2Brank%2B%252F%2BnullIf%2528total_packages%252C%2B0%2529%2529%2B%252F%2B10%2529%252C%2B%2527%2525%2527%2B%2529%2B%2529%2BAS%2Bmessage%2BFROM%2B%2528%2BSELECT%2Bproject%252C%2Bsum%2528count%2529%2BAS%2Bn%2BFROM%2Bpypi.pypi_downloads_per_day%2BWHERE%2Bdate%2B%253E%253D%2BtoDate%2528now%2528%2527UTC%2527%2529%2529%2B-%2B%257Bdays%253AUInt16%257D%2BAND%2Bdate%2B%253C%2BtoDate%2528now%2528%2527UTC%2527%2529%2529%2BGROUP%2BBY%2Bproject%2B%2529%2BFORMAT%2BJSON&query=%24.data%5B0%5D.message&label=PyPI+rank+%2830d%29&color=brightgreen&cacheSeconds=86400
   :target: https://clickpy.clickhouse.com/dashboard/spacr
   :alt: spaCR-Rang bei PyPI-Downloads der letzten 30 vollen Tage

.. image:: ../../source/_static/deck/slides/slide_01.jpg
   :alt: spaCR
   :width: 920
   :target: https://einarolafsson.github.io/spacr/_static/deck/

`← Zurück <../../source/_static/deck/pages/51.md>`_   `Weiter → <../../source/_static/deck/pages/02.md>`_

spaCR
=====

.. spacr-language-picker-begin

Sprachen: `🌐 Deutsch ▾ <README.md>`_

.. spacr-language-picker-end

**Räumliche Phänotypanalyse von CRISPR-Screens.**

spaCR segmentiert und vermisst einzelne Zellen in High-Content-Mikroskopiebildern, integriert Phänotypen einzelner Objekte mit sequenzierungsbasierten Guide-Häufigkeiten und schätzt, welche Gene mit phänotypischen Veränderungen assoziiert sind. Ausgehend von Plattenbildern und FASTQ-Reads erzeugt es Messungen pro Objekt, trainierte Klassifikatoren, Effektschätzungen pro Guide und Gen sowie eine Rangliste der Treffer.

Die Segmentierungs-, Mess-, Anmerkungs- und Klassifizierungsmodule laufen auch ohne Sequenzierungsarm.

Bilder, Masken, Bildausschnitte, Messungen, Anmerkungen, Vorhersagen, Barcodes und Well-Identifikatoren liegen in einem einzigen SQLite-Projekt.

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
   :alt: Windows 10/11: spaCR 1.5.1.0 herunterladen
   :target: https://github.com/EinarOlafsson/spacr/releases/download/v1.5.1.0/spaCR-1.5.1.0-Windows-Online-Setup.exe
.. |InstallerMacOS| image:: ../../../spacr/resources/icons/platforms/macos.png
   :width: 64
   :alt: macOS 11+ (Intel und Apple Silicon): spaCR 1.5.1.0 herunterladen
   :target: https://github.com/EinarOlafsson/spacr/releases/download/v1.5.1.0/spaCR-1.5.1.0-macOS-Universal-Online.pkg
.. |InstallerLinux| image:: ../../../spacr/resources/icons/platforms/linux.png
   :width: 64
   :alt: 64-Bit-Linux: spaCR 1.5.1.0 herunterladen
   :target: https://github.com/EinarOlafsson/spacr/releases/download/v1.5.1.0/spaCR-1.5.1.0-Linux-x86_64-Online.run
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

spaCR unterstützt Python **3.9 through 3.14**, außer Python 3.14.1, das von torchvision ausgeschlossen wird. Linux wird für die anspruchsvollsten CUDA- und ROCm-Workflows empfohlen; macOS und Windows werden ebenfalls unterstützt und nutzen beide ihre GPUs — macOS über Metal, das Apple Silicon und die AMD-Karten in Intel Macs abdeckt, und Windows über CUDA oder DirectML.

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

Installation aus dem Quellcode
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Klonen Sie das Repository und installieren Sie es im editierbaren Modus. Ihre Arbeitskopie *ist* dann das installierte Paket, und Änderungen werden ohne Neuinstallation wirksam::

    git clone https://github.com/EinarOlafsson/spacr.git
    cd spacr
    conda create -n spacr python=3.12 -y
    conda activate spacr
    pip install -e .
    spacr

Dieser Befehl klont ``main``, den Standard-Branch mit der neuesten Version. Entwickelt wird auf ``nightly``; fügen Sie ``--branch nightly`` hinzu, um stattdessen diesen Branch zu klonen. Für eine bestimmte Version::

    git clone --branch v1.5.0.5 https://github.com/EinarOlafsson/spacr.git

Um spätere Änderungen zu übernehmen, führen Sie im Klon aus::

    git pull
    pip install -e .

Die zweite Zeile ist nur nötig, wenn sich Abhängigkeiten oder Einstiegspunkte geändert haben; Python-Code wird auch ohne sie übernommen. Führt ein Befehl nach dem Pull noch alten Code aus, zeigt ``spacr-doctor``, welches ``spacr`` tatsächlich im Suchpfad liegt – das ist die übliche Ursache.

Installation aus dem Quellcode (schlank)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Wer zu spaCR beiträgt, braucht den Verlauf; wer spaCR nur ausführen will, nimmt eine dieser Varianten, gemessen am 2026-09-15 mit ``packaging/measure_clone_forms.sh``::

    # One commit instead of every version: 540 MB downloaded, 69 s.
    # No history, so no git log, no git blame and no git bisect.
    # git pull still works, but stays shallow until git fetch --unshallow.
    git clone --depth 1 https://github.com/EinarOlafsson/spacr.git
    cd spacr && pip install -e .

    # Only the files spaCR runs from: 81 MB on disk, 39 s. No history
    # either, and no docs, tests, tools or example data.
    # --with-docs, --with-tests and --with-translations put those back;
    # --dir, --branch, --no-install and --help do the obvious things.
    # packaging/source_install_excludes.txt lists every skipped path.
    curl -fsSL https://raw.githubusercontent.com/EinarOlafsson/spacr/nightly/packaging/install_from_source.sh -o install_spacr.sh
    sh install_spacr.sh --branch nightly

Bei der Messung am 2026-09-15 lud der vollständige Klon 5,8 GB herunter. Beim flachen Klon verringerte ``--filter=blob:none`` die gemessene Downloadmenge nicht. Die versionierten Dateien von nightly ergeben ein Arbeitsverzeichnis von 1353 MB (gemessen am 2026-09-26), ohne Git-Verlauf. Downloadmenge und Dauer hängen vom Branch ab.


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
   spacr-download --list                      # what example data exists
   spacr-download measure annotate            # fetch example sets by name
   spacr-make-masks --folder DIR              # curate masks as a resumable queue
   spacr-make-masks --folder DIR --order easy --limit 50

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

spaCR-Module
-------------

.. spacr-workflow-begin

Kern
^^^^

Core sequence from microscopy images through segmentation, measurements,
annotations, classification, barcode mapping and regression.

| |Module_mask|\ |Module_measure|\ |Module_annotate|\ |Module_classify_merged|\ |Module_map_barcodes|\ |Module_regression|

Daten
^^^^^

Import images and tables into spaCR projects and execute reproducible
multi-plate workflows.

| |Module_foreign|\ |Module_embeddings|\ |Module_run_compare|\ |Module_experiment_design|\ |Module_power|\ |Module_dose_response|
| |Module_qc_dashboard|

Werkzeuge
^^^^^^^^^

Point these at a project: edit masks by hand, stitch tiles, read an
embedding, draw a gate, build a plot, check quality.

| |Module_make_masks|\ |Module_align|\ |Module_umap|\ |Module_gate_editor|\ |Module_graph_builder|

Assays
^^^^^^

Quantitative readouts for biological assays.

| |Module_toxoplasma|\ |Module_plasmodium|\ |Module_candida|

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
.. |Module_embeddings| image:: ../../../spacr/resources/icons/workflow/apps/embeddings.png
   :width: 16.0%
   :alt: API für Embeddings öffnen
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/embeddings/index.html
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
.. |Module_toxoplasma| image:: ../../../spacr/resources/icons/workflow/apps/toxoplasma.png
   :width: 16.0%
   :alt: API für Toxoplasma öffnen
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/organism_screen/index.html#spacr-qt-screens-organism-screen-toxoplasma
   :align: middle
.. |Module_plasmodium| image:: ../../../spacr/resources/icons/workflow/apps/plasmodium.png
   :width: 16.0%
   :alt: API für Plasmodium spp. öffnen
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/organism_screen/index.html#spacr-qt-screens-organism-screen-plasmodium
   :align: middle
.. |Module_candida| image:: ../../../spacr/resources/icons/workflow/apps/candida.png
   :width: 16.0%
   :alt: API für Candida spp. öffnen
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/organism_screen/index.html#spacr-qt-screens-organism-screen-candida
   :align: middle

.. spacr-workflow-end

Alle Module, die spaCR mitbringt, in der Reihenfolge des Startbildschirms: zuerst die sechs Pipeline-Module, dann alle übrigen. Wählen Sie eine Kachel, um die API-Seite des Moduls zu öffnen.

Einzelheiten zu jedem Werkzeug stehen im `Funktionsleitfaden <../../source/features.rst>`_.

Sonstige Mittel
~~~~~~~~~~~~~~~

- `Interaktive Tutorials <https://einarolafsson.github.io/spacr/tutorials/>`_ — 73 geführte Workflows von der Installation bis zur Hit-Untersuchung.
- `Python API Schnellstart <../../source/python_api.rst>`_ — Pipelines aus Skripten, Notebooks oder einem Cluster ausführen und validieren.
- `Funktionsleitfaden <../../source/features.rst>`_ — Fähigkeiten, Reife und optionale Integrationen.
- `Kuratierte API Referenz <https://einarolafsson.github.io/spacr/api/index.html>`_ — unterstützte Eingabepunkte nach Aufgabe, wobei das komplette Modul eine Ebene tiefer verweist.
- `Sprach- und Übersetzungshandbuch <../../source/localization.rst>`_ — Schnittstellensprachen, kontextbezogene Hilfe und Politik der wissenschaftlichen Ergebnisse.

Sprache und Übersetzung
~~~~~~~~~~~~~~~~~~~~~~~

Die Oberfläche unterstützt zehn Sprachen in der Navigation und den Einstellungen. AI- und LIVE-Steuerelemente, Modulbeschreibungen und geprüfte Kontexthilfe werden ebenfalls übersetzt. Ändern Sie die Sprache unter **spaCR → Einstellungen → Sprache**, ohne neu zu starten. Protokolle, Pfade, Datenbankwerte und Messungen werden nie übersetzt; wissenschaftliche Ausgaben bleiben im kanonischen Englisch. Siehe die `Richtlinie zur Kontexthilfe <../../source/localization.rst#contextual-help>`_.

Die neun nicht-englischen Kataloge werden von einem Muttersprachler jeder Sprache maschinengefertigt und technisch überarbeitet, anstatt zu Ende zu lesen. Die `Überprüfungsspielraum <../REVIEW_SCOPE_2026-09-04.md>`_ zeichnet auf, welche Sprachen einen menschlichen Pass hatten, wie viel von dem Corpus, der umfasst, und jeder Begriff, der auf Englisch durch Entscheidung übrig bleibt.

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
     - Hold-out performance
   * - ``toxoplasma_pv_v1``
       (Cellpose-SAM (cpsam_v2))
     - anti-Toxoplasma-biotin and DsRed PV lumen; 229 images from 2 datasets, 104 round-1 and 125 newly curated
     - F1 0.864 against 0.713 for stock cpsam on 11 held-out in-house wells, at IoU 0.5; literature hold-out pending
   * - ``toxoplasma_plaque_v1``
       (Cellpose-SAM (cpsam))
     - crystal violet plaque wells; 184 wells from 3 datasets, 95 in-house and 89 literature
     - F1 0.856 in-domain; 0.806 on literature (3-fold cross-validated, SD 0.020)
   * - ``toxoplasma_plaque_v2``
       (Cellpose-SAM (cpsam_v2))
     - 488 curated fields across four domains -- 298 wells cropped from published figures, 96 phone-camera wells, 67 PFA and 27 methanol-fixed whole-well microscope scans; 27,582 plaques
     - not scored against stock; on 81 held-out fields it ties round 3 on literature (0.819 vs 0.820) and beats it by 0.166 on phone-camera wells (0.415 vs 0.249)
   * - ``toxoplasma_well_detector_v1``
       (YOLO11n)
     - whole-plate and multi-well crystal violet images; 562 images from 1 dataset, 190 of them with no well in them
     - mAP50 0.993 on its own held-out split; on the test set shared with v2 it scores mAP50 0.8838, against v2's 0.9457
   * - ``toxoplasma_well_detector_v2``
       (YOLO26n (ultralytics 8.4.155))
     - plate images and literature figures; 1,070 train / 254 val / 129 test, split by PMC article so no paper is in two sets; training data at einarolafsson/toxoplasma-plaque-well-detector-dataset
     - mAP50 0.9457 and mAP50-95 0.8341 against v1's (yolo_welldetect_v3.pt) 0.8838 and 0.7630 on the SAME test set; stock YOLO has no plaque-well class, so v1 is the baseline
   * - ``toxoplasma_from_cellmask_v1``
       (Cellpose-SAM (cpsam_v2))
     - Toxoplasma PV masks predicted from the HOST CELL MASK channel alone; 2567 training and 463 held-out fields, split by well, hosts HFF/HeLa/THP1
     - F1 0.606 against 0.021 for stock cpsam_v2 on 463 well-grouped held-out fields, at IoU 0.5
   * - ``toxoplasma_pv_v2``
       (Cellpose-SAM (cpsam_v2))
     - anti-Toxoplasma-biotin and DsRed PV lumen; 556 curated images accumulated over five rounds
     - F1 0.817 +/- 0.036 by 5-fold cross-validation over 619 pairs; ~0.86 against 0.713 for stock on the 11 in-house held-out wells
   * - ``toxoplasma_pv_v3``
       (Cellpose-SAM (cpsam_v2))
     - the 556 curated PV fields of round 5, split 437 train / 108 validation / 11 test; training data at einarolafsson/toxoplasma-pv-segmentation-dataset
     - F1 0.860 against stock cpsam_v2's 0.765 on the 11 anchor wells at IoU 0.5; AJI 0.803 against 0.505
   * - ``live_cell_v1``
       (Cellpose-SAM (cpsam_v2))
     - 11,007 transmitted-light fields from 14 public datasets, split by acquisition 6,778 train / 2,030 validation / 2,199 test; training data at einarolafsson/live-cell-segmentation-dataset
     - on the datasets stock cpsam_v2 never trained on, F1 0.960 against 0.885 at IoU 0.5; over all 2,199 test fields, 0.694 against 0.738, because stock trained on LIVECell and YeaZ and wins on LIVECell
   * - ``nuclei_from_cellmask_v1``
       (Cellpose-SAM (cpsam_v2))
     - nuclei predicted from the HOST CELL MASK channel alone; 453 well-grouped held-out fields, hosts HFF/HeLa/THP1
     - F1 0.888 against 0.201 for stock cpsam_v2 on 453 well-grouped held-out fields, at IoU 0.5
   * - ``cell_from_hoechst_v1``
       (Cellpose-SAM (cpsam_v2))
     - the HOST CELL outline predicted from the Hoechst (nuclear) channel alone; 2,578 training fields and 451 held-out test fields, split by well so no well is on both sides
     - F1 0.870 against stock cpsam_v2's 0.301 on 451 held-out fields at IoU 0.5 -- a delta of 0.569
   * - ``toxoplasma_from_hoechst_v1``
       (Cellpose-SAM (cpsam_v2))
     - Toxoplasma PV masks predicted from the HOECHST channel alone; 2567 training and 463 held-out fields, split by well, hosts HFF/HeLa/THP1
     - F1 0.569 against 0.002 for stock cpsam_v2 on 463 well-grouped held-out fields, at IoU 0.5

.. spacr-model-zoo-end

Jede Abbildung oben wird auf Bildern gemessen, die das Modell im Training nie gesehen hat.

**Präzision** ist, wie viele der Objekte, von denen ein Modell berichtet wird, real sind; **Recall** ist wie viele Objekte es gefunden hat. Sie scheitern in entgegengesetzte Richtungen: schlechte Präzision erfindet Plaques, schlechte Erinnerung vermisst sie.

**F1** ist die Kombination der beiden, und wird zitiert, weil jeder einzelne trivial gespielt wird -- berichten Sie eine unverwechselbare Plaque für nahezu perfekte Präzision, oder jeder dunkle Blob für nahezu perfekten Rückruf. Was Sie lieber verlieren würden, hängt vom Assay ab, und Zählen wird in der Regel besser durch Überrufen bedient: Das Plaque-Modell wurde mit Präzision 0.858 mit Rückruf 0.811 in einer früheren Runde bei 0.939 und 0.631 akzeptiert.

**IoU** (Intersection over Union) teilt die Überlappungsfläche zwischen vorhergesagtem Objekt und Referenzobjekt durch ihre Vereinigungsfläche. Lesen Sie Kennwerte zusammen mit ihrem Schwellenwert: „F1 0.864 bei IoU 0.5“ zählt eine Vakuole als gefunden, wenn die Überlappung mindestens die Hälfte der Vereinigungsfläche erreicht.

**mAP50** und **mAPI50-95** gehören zum Detektor. Der erste fragt, ob die Wells gefunden wurden; der zweite wiederholt sie über zehn Schwellen von 0,5 bis 0,95, so dass er auch fragt, wie eng jede Box gezeichnet wird.

**Cross-validated**, mit einem **SD**, bedeutet, dass die Punktzahl das Mittel von drei Runs auf verschiedenen Splits ist und der SD ist, wie weit sie auseinander bewegt. Ein Split kann Glück haben: Die Literatur dieses Modells ist 0,834 auf einem einzigen 19-Well-Split und 0,806 auf allen drei.

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

Wenn spaCR zu veröffentlichten Arbeiten beigetragen hat, wird ein Zitat geschätzt und ist keine Bedingung der Lizenz — siehe `spaCR zitieren`_ unten.

Tutorials
~~~~~~~~~

Die `interaktive spaCR-Tutorialbibliothek <https://einarolafsson.github.io/spacr/tutorials/>`_ führt durch die Installation und die Verwendung der Module. Verfügbare Vertonungen und Sprachen werden für jede Lektion angegeben.

spaCR zitieren
~~~~~~~~~~~~~~

Wenn spaCR zu Ihrer Forschung beiträgt, zitieren Sie:

Olafsson EB, *et al.* Ein gepoolter Bild-basierter CRISPR Screening identifiziert EAF1 als einen *T. gondii* Modulator der ESCRT-Subversion.

`Vordruck bioRxiv <https://www.biorxiv.org/content/10.64898/2026.07.08.737057v1>`_ · `Software-Archiv <https://doi.org/10.5281/zenodo.21343316>`_

Weitere Arbeiten, die spaCR zitieren
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. spacr-citing-papers-begin

* `Metabolic adaptability and nutrient scavenging in Toxoplasma gondii: insights from ingestion pathway-deficient mutants. <https://journals.asm.org/doi/full/10.1128/msphere.01011-24>`_
* `IRE1α promotes phagosomal calcium flux to enhance macrophage fungicidal activity. <https://www.cell.com/cell-reports/fulltext/S2211-1247(25)00465-6>`_
* `Toxoplasma GRA8 engages the host ESCRT accessory protein ALG-2 and is necessary for parasite metabolic integrity. <https://www.biorxiv.org/content/10.64898/2026.07.20.739547v1.abstract>`_
* `spaCR: Spatial phenotype analysis of CRISPR-Cas9 screens (preprint version 1). <https://www.researchsquare.com/article/rs-7368254/v1>`_

.. spacr-citing-papers-end

Danksagung
~~~~~~~~~~~~~~~

spaCR baut auf offener wissenschaftlicher Software auf, darunter NumPy, pandas, scikit-image, scikit-learn, Cellpose, PyTorch und Qt. Die für die mehrsprachige Dokumentation und die Oberflächenkataloge verwendeten Modelle sind in der `Attribution der Übersetzungsmodelle <../TRANSLATION_MODELS.md>`_ aufgeführt.
