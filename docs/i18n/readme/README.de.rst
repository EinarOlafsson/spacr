|Docs| |Tutorials| |PyPI| |Python| |Tests| |Qt| |Source| |Issues| |License| |DOI|

.. |Docs| image:: https://github.com/EinarOlafsson/spacr/actions/workflows/pages/pages-build-deployment/badge.svg
   :target: https://einarolafsson.github.io/spacr/
   :alt: Documentation
.. |Tutorials| image:: https://img.shields.io/badge/Tutorials-Interactive%20walkthrough-4A9EFF
   :target: https://einarolafsson.github.io/spacr/tutorials/
   :alt: Interactive tutorials
.. |PyPI| image:: https://img.shields.io/pypi/v/spacr
   :target: https://pypi.org/project/spacr/
   :alt: PyPI version
.. |Python| image:: https://img.shields.io/badge/Python-3.9%E2%80%933.14-3776AB?logo=python&logoColor=white
   :target: https://pypi.org/project/spacr/
   :alt: Python 3.9 through 3.14
.. |Tests| image:: https://github.com/EinarOlafsson/spacr/actions/workflows/tests.yml/badge.svg
   :target: https://github.com/EinarOlafsson/spacr/actions/workflows/tests.yml
   :alt: Test suite
.. |Qt| image:: https://img.shields.io/badge/GUI-Qt%20%28PySide6%29-41CD52
   :target: https://einarolafsson.github.io/spacr/
   :alt: Qt interface
.. |Source| image:: https://img.shields.io/badge/GitHub-Source-181717?logo=github
   :target: https://github.com/EinarOlafsson/spacr
   :alt: GitHub source
.. |Issues| image:: https://img.shields.io/github/issues/EinarOlafsson/spacr
   :target: https://github.com/EinarOlafsson/spacr/issues
   :alt: GitHub issues
.. |License| image:: https://img.shields.io/github/license/EinarOlafsson/spacr
   :target: https://github.com/EinarOlafsson/spacr/blob/main/LICENSE
   :alt: PolyForm Noncommercial license
.. |DOI| image:: https://img.shields.io/badge/DOI-10.5281%2Fzenodo.21343317-blue
   :target: https://doi.org/10.5281/zenodo.21343317
   :alt: Zenodo DOI
.. |Release| image:: https://img.shields.io/github/v/release/EinarOlafsson/spacr?label=Installers
   :target: https://github.com/EinarOlafsson/spacr/releases/latest
   :alt: Latest installers
.. |CondaRecipe| image:: https://img.shields.io/badge/conda--forge-recipe-44A833?logo=anaconda
   :target: https://github.com/EinarOlafsson/spacr/tree/main/conda-forge/recipe
   :alt: conda-forge recipe

.. image:: https://raw.githubusercontent.com/EinarOlafsson/spacr/main/spacr/resources/icons/logo_spacr.png
   :alt: spaCR
   :align: center
   :width: 360

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

`Angaben zu den Übersetzungsmodellen <../TRANSLATION_MODELS.md>`_

**Räumliche Phänotypanalyse von CRISPR-Screens.**

spaCR segmentiert und vermisst einzelne Zellen in High-Content-Mikroskopiebildern, verknüpft jede Zelle mit der erhaltenen gRNA und berichtet, welche Gene den Phänotyp verändert haben. Plattenbilder und FASTQ-Reads dienen als Eingabe; ausgegeben werden Messungen pro Objekt, trainierte Klassifikatoren, Effektgrößen pro Guide und Gen sowie eine Rangliste der Treffer.

Für bildbasierte gepoolte CRISPR-Screens deckt dies den gesamten Arbeitsablauf ab. Bei High-Content-Mikroskopie ohne Screen können Segmentierung, Messung, Annotation und Klassifizierung eigenständig ausgeführt werden.

Bilder, Masken, Bildausschnitte, Messungen, Annotationen, Vorhersagen, Barcodes und Well-Kennungen liegen in einem einzigen SQLite-Projekt. Dadurch lässt sich ein Ergebniswert bis zu seinem Ursprungsobjekt zurückverfolgen.

Führen Sie spaCR als Desktopanwendung oder ohne grafische Oberfläche auf einer Workstation, einem Server oder Cluster aus. Beide Varianten verwenden dieselben Module; CUDA wird automatisch genutzt, wenn das jeweilige Modul es unterstützt.


Workflow auf einen Blick
------------------------

|Tutorials|

.. image:: https://raw.githubusercontent.com/EinarOlafsson/spacr/main/spacr/resources/icons/flow_chart_v3.png
   :alt: spaCR workflow and output organization
   :align: center

Mikroskopiebilder (TIFF, OME-TIFF, LIF, CZI, ND2) und Sequenzierungs-Reads (FASTQ) durchlaufen einander ergänzende Pipelines für Bildanalyse und Barcode-Zuordnung. Objekttabellen, Bildausschnitte, Annotationen, Vorhersagen, Guide-Identitäten, QC-Ergebnisse und Zusammenfassungen auf Well-Ebene werden anschließend gemeinsam analysiert.


Schnellstart
------------

.. code-block:: bash

   conda create -n spacr python=3.12 -y
   conda activate spacr
   python -m pip install --upgrade pip
   python -m pip install "spacr[qt]"
   spacr

spaCR unterstützt Python **3,9 bis 3.14** (außer Python 3.14.1, die Fackelvision nicht berücksichtigt). Python 3.12 hat die größte Auswahl an optionalen wissenschaftlichen Paketen. Linux wird für CUDA Workflows empfohlen; macOS und Windows werden ebenfalls unterstützt.


Installationsdetails
--------------------

|Release| |PyPI| |CondaRecipe|

**(beta) Leichte Desktop-Installer:**

.. spacr-installer-links-begin

* `Windows 10/11: SpaCR 1.5.0.4 herunterladen <https://github.com/EinarOlafsson/spacr/releases/download/v1.5.0.4/SpaCR-1.5.0.4-Windows-Online-Setup.exe>`_
* `macOS 11+ (Intel und Apple Silicium): SpaCR 1.5.0.4 herunterladen <https://github.com/EinarOlafsson/spacr/releases/download/v1.5.0.4/SpaCR-1.5.0.4-macOS-Universal-Online.pkg>`_
* `64-bit Linux: SpaCR 1.5.0.4 herunterladen <https://github.com/EinarOlafsson/spacr/releases/download/v1.5.0.4/SpaCR-1.5.0.4-Linux-x86_64-Online.run>`_

.. spacr-installer-links-end

Leichte Installationsprogramme — weder conda noch vorhandenes Python erforderlich
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Das Installationsprogramm lädt eine private Python 3.12 Laufzeit, Qt, PyTorch, spaCR und die wissenschaftlichen Abhängigkeiten während der Installation herunter, sodass weder Conda noch eine bestehende Python benötigt werden. Der portable CPU Build ist der Standard, der die Installation davon abhält, mehrere Gigabyte CUDA Bibliotheken unangekündigt zu ziehen. Windows bietet NVIDIA Beschleunigung als optionale Installationskomponente, Linux akzeptiert ``--torch-backend auto`` und das Standard-Rad macOS PyTorch hält die Beschleunigung von Apple MPS.

Installer-Hilfe, Fortschritt und Fehler folgen der Betriebssystem-Sprache in allen zehn spaCR Sprachen: Englisch, Schwedisch, Deutsch, Spanisch, Vereinfachtes Chinesisch, Portugiesisch, Hindi, Koreanisch, Isländisch und Französisch. Nicht unterstützte Lokalitäten fallen auf Englisch zurück.

Machen Sie auf Linux das heruntergeladene Installationsprogramm ausführbar, bevor Sie es öffnen:

.. code-block:: bash

   chmod +x SpaCR-*-Linux-x86_64-Online.run
   ./SpaCR-*-Linux-x86_64-Online.run

Öffnen Sie auf macOS den heruntergeladenen ``.pkg``. Wenn Gatekeeper den aktuellen Beta-Installer blockiert, weil er nicht beglaubigt ist, öffnen Sie **Systemeinstellungen → Datenschutz & Sicherheit**, wählen Sie **Open Anyway** für spaCR, dann führen Sie das Paket erneut aus.

Das Installationsprogramm validiert spaCR, Qt, PyTorch und Abhängigkeitskonsistenz, bevor eine ältere Installation ersetzt wird. Ein unterbrochenes Update lässt die vorherige Arbeitsumgebung in Kraft. Ein Diagnoseprotokoll wird als ``install.log`` im privaten spaCR Installationsverzeichnis aufbewahrt.

Desktopanwendung von PyPI
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   python -m pip install "spacr[qt]"
   spacr

Installation ohne grafische Oberfläche oder auf einem Server
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   python -m pip install spacr
   spacr-run --list

Neuester Entwicklungszweig
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   git clone https://github.com/EinarOlafsson/spacr.git
   cd spacr
   git switch nightly
   python -m pip install -e ".[qt]"

Conda-Umgebungen
~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   conda create -n spacr python=3.12 pip -y
   conda activate spacr
   python -m pip install "spacr[qt]"

Optionale Funktionen
~~~~~~~~~~~~~~~~~~~~~

Installieren Sie nur die Extras, die Ihr Workflow benötigt:

.. code-block:: bash

   python -m pip install "spacr[trackastra]"    # transformer tracking
   python -m pip install "spacr[ultrack]"       # global-optimization tracking
   python -m pip install "spacr[btrack]"        # btrack timelapse tracking
   python -m pip install "spacr[attribution]"   # TorchCAM methods
   python -m pip install "spacr[boosting]"      # LightGBM and CatBoost
   python -m pip install "spacr[zernike]"       # Zernike measurements
   python -m pip install "spacr[napari]"        # napari mask correction
   python -m pip install "spacr[czi,nd2,lif]"   # vendor file readers

Welche Extras auflösen, hängt von der Python Version ab. Auf Python 3.13, sind ultrack-Grenzwerte ``spacr[all]`` und TorchCAMs NumPy Beschränkung begrenzt das ``attribution`` Extra; das Core-Paket und die Qt Anwendung bleiben unberührt. Auf Python 3.14 ist btrack über sein Extra verfügbar. Der PylibCZIrw CZI-Konverter ist optional und unbetestet; czifile-basierte CZI-Lesedaten bleiben verfügbar.

Die alte Tk-Schnittstelle ist immer noch als ``spacr-legacy`` installiert, wird aber nicht mehr entwickelt.


Befehle für die Kommandozeile
-----------------------------

.. code-block:: bash

   spacr                                      # Qt application
   spacr-doctor                               # diagnose the installation
   spacr-run --list                           # list headless modules
   spacr-run --describe MODULE                # inspect a module contract
   spacr-run MODULE --settings settings.csv   # execute a module
   spacr-run validate --module MODULE \
       --settings settings.csv                # validate before running
   spacr-repro RUN_DIR                        # replay a recorded run

Setze ``SPACR_LOG_LEVEL=DEBUG`` beim Fehlerbehebung. Rotierende Protokolle werden auf ``~/.spacr/logs/spacr.log`` geschrieben.


Funktionen
----------

Die sechs Module, die in den meisten Screens verwendet werden
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Mask** segments cells, nuclei, pathogens and organelles with Cellpose, in 2D images and in volumetric or time-series data. The model list is read from the installed Cellpose rather than hard-coded, and an object diameter is estimated from the images before the run starts. Masks can be corrected by hand in the layer viewer, or sent to napari and back.

**Measure** schreibt pro-Objekt Morphologie, Intensität, Textur und Kolokalisierungsfunktionen in die Projektdatenbank zusammen mit den Bildausschnitte. Neu in 1.5.0.0: Die Beleuchtungskorrektur schätzt das Flachfeld von der Platte selbst ab und teilt es aus, bevor irgendeine Intensitätsfunktion genommen wird, was die Wellpositions-Bias, die Platten-Heatmaps als Kanteneffekte zeigen, entfernt. Eine Segmentierung QC Banner stellt in einfacher Sprache dar, wie die Masken vor dem Messen aussehen; es informiert, es blockiert nicht. Ein gezeichnetes Polygon beschränkt die Messung auf eine Region von Interesse.

**Annotate** zeigt die Ernten auf einem keyboardgesteuerten Raster und schreibt die Etiketten direkt auf SQLite. Es schließt nun die aktive Lernschleife: Retrain ein Modell auf dem, was Sie markiert haben, ohne den Screening zu verlassen, die Warteschlange durch Unsicherheit neu zu ranken, die Lernkurve zu beobachten und ein Stop Urteil zu erhalten, wenn weitere Etiketten aufhören, das Modell zu ändern.

**Classify** trainiert PyTorch CNNs und Transformatoren auf kommentierten Bildausschnitte und klassische oder aufgewertete Modelle auf Messtischen. Die Genauigkeit pro Klasse wird nun in jeder Epoche gehalten, anstatt verworfen zu werden, und jeder Checkpoint erhält eine Modellkarte, die den Datensatz, die Klassenbilanz, die geteilte Regel und die ausgehaltenen Metriken aufzeichnet. Im Auswertungsbildschirm ist eine Verwirrtheitsmatrix-Zelle eine Abfrage: Klicken Sie auf sie, um diese Bildausschnitte zu öffnen, wobei zuversichtlich falsche Vorhersagen außer unsicheren aufgeführt werden.

**Map Barcodes** entschlüsselt Zeilen, Spalten und gRNA Barcodes aus FASTQ liest, weist Identitäten zu Wells und verbindet sie mit Bildzellen. Barcode QC Berichte liest pro Wells, Kollisionsrate und unerschlossenen Bruch, fegt um die Anzahl der gRNAs pro Wells, die Sie erwarten, anstatt eine feste Schwelle.

**Regression** schätzt Guide, Gen, Zustand und Kontrolleffekte mit 17 Modellfamilien, einschließlich gemischter Modelle, Logistik und Probit, quantile, Beta, GLMs mit quasi-binomialer Varianz, Lasso, Grat, elastischem Netz, Scharnier und Hufeisen. Das Ergebnis ist eine Rangliste, kommentierte Hitliste statt eines Koeffizienten Dump.

Neu in 1.5.0.0
~~~~~~~~~~~~~~

Bevor ein Screening existiert, beantwortet das Power / Design Modul, wie viele Zellen und wie viele Wells es braucht, preislich mit Sequenzierungsfehler und mit dem Dropout, der von Wells kommt, die zu dünn abgebildet wurden. Ein Experiment Designer legt die Platte, ihre Steuerungen und ihre Repliken und exportiert das Layout für die Pipeline. Danach sammelt ein QC Dashboard die Segmentierung, Platte, Annotator-Agreement und Leckage-Prüfungen in einem Urteil, und ComBat ist neben ``center`` und ``zscore`` für Batchkorrektur verfügbar.

Die Ergebnisse werden nicht exportiert, sondern exportiert und wieder importiert. Ein Graph Builder zeichnet eine Tabelle auf, indem er Spalten auf x, y, Farbe, Größe und Facette zieht. Gates, die auf einem Histogramm oder einem Scatter gezeichnet werden, werden zu Filtern. Ein Feature Explorer reiht Funktionen, wie gut sie die Klassen trennen. Kleine Vielfache, Dosis-Antwort passt, Kontrollkarten und robuste Ausreißer Erkennung verwenden die gleiche Achse Motor. Wählen Objekte in einer Ansicht wählt sie in allen von ihnen, und Öffnen einer Auswahl bringt die Ernten, die diese Objekte kamen aus. Ein Ebenenbetrachter stapelt Bilder, Etiketten, Punkte und Formen, mit orthogonalen Ansichten, einem synchronisierten Vergleichsraster und einem Linienbaum von Zelle zu Kern zu Pathogen.

Runs are now identifiable. Each carries one run id, one seed and an ``on_error`` policy; Mask, Measure, Classify and the AnnData export register what they wrote in an artifact registry, so an output file leads back to the settings that produced it. A module opens on what the previous step actually wrote, the pipeline graph marks which outputs are stale, run comparison diffs the settings, object counts and hit lists of two runs, and every GUI run emits the equivalent Python script. Measurements export to ``.h5ad`` for scanpy; OME-Zarr and OMERO are available through the Python API. The methods-and-results exporter drafts those two manuscript sections from a structured digest of the run: the model writes the prose, but every number comes from the digest, and a draft containing a number the digest does not contain is rejected. Wenn etwas mit der Installation nicht stimmt, meldet ``spacr-doctor``, welches spaCR tatsächlich läuft, ob das GPU nutzbar ist, ob Cellpose mit den API spaCR Anrufen übereinstimmt und ob die Projektdatenbank und die Einstellungen gut sind, mit einem kopierbaren Fix auf jeder Zeile, die kein Pass ist.

Mehrsprachige Desktopoberfläche
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**spaCR → Einstellungen → Language** übersetzt die laufende Anwendung ohne Neustart ins Englische, Schwedische, Deutsche, Spanische, Mandarin-Chinesische, Portugiesische, Hindi, Koreanisch, Isländisch oder Französische. Die Wahl bleibt bestehen, und später geöffnete Screenings erben sie.

Navigation, Preferences, AI and LIVE controls, module descriptions and spaCR-authored console notices follow the selected language. Worker output, logs, tracebacks, paths, database values, annotations, AI responses, measurements and saved results are never translated, so scientific output remains canonical English. Setting tooltips not yet reviewed in a language stay in English rather than becoming a mixed-language explanation. The `Lokalisierungsleitfaden <https://einarolafsson.github.io/spacr/localization.html>`_ documents the behavior, the environment override, and the `Kontextuelle Hilfe <https://einarolafsson.github.io/spacr/localization.html#contextual-help>`_ that is translated with it.

Animierte Einstellungsführung
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

94 short animations explain what 143 visual settings do to an image. Hover a setting and click **Animation** in its tooltip to play the square beside the text; click it again to fold it away. Animations are off until asked for, and can be disabled in Preferences. The `Galerie <https://einarolafsson.github.io/spacr/setting_animations.html>`_ shows all of them, and the `Einstellung der Animationsregistrierung <https://einarolafsson.github.io/spacr/api/spacr/setting_animations/index.html>`_ records which setting each one belongs to.

Modulreferenz
~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 25 25 25

   * - Module
     - Feature
     - State
     - Description
   * - **Desktop experience**
     -
     -
     -
   * - |api-qt-app|_
     - |doc-i18n|_
     - Stable
     - Retranslates open and lazily created screens across ten bundled languages.
   * - |api-qt-app|_
     - |doc-i18n-help|_
     - Stable
     - Localizes module summaries and setting-help chrome while preserving exact API URLs.
   * - |api-qt-ai|_
     - |api-qt-ai-console|_
     - Stable
     - Localizes AI and LIVE controls without changing user or model content.
   * - |api-animations|_
     - |doc-animations|_
     - Stable
     - Plays 94 packaged animations for 143 visual settings from the setting tooltip.
   * - |api-selection|_
     - |api-linked-views|_
     - Alpha
     - Shares one object selection across the table, plate, embedding, scatter and graph views.
   * - |api-doctor|_
     - |api-doctor-checks|_
     - Alpha
     - Diagnoses the install — GPU, Cellpose API, database, settings — with a fix per failing check.
   * - **Image analysis**
     -
     -
     -
   * - |api-mask|_
     - |api-mask-2d|_
     - Stable
     - Segments cells, nuclei, pathogens and organelles in 2D images.
   * - |api-mask|_
     - |api-mask-3d|_
     - Beta
     - Segments volumetric images and 4D time series.
   * - |api-illumination|_
     - |api-flatfield|_
     - Alpha
     - Estimates the flat-field from the plate and divides it out before intensity is measured.
   * - |api-measure|_
     - |api-measure-2d|_
     - Stable
     - Measures morphology, intensity, texture and colocalization, and writes the crops.
   * - |api-segqc|_
     - |api-segqc-verdict|_
     - Alpha
     - States what the segmentation looks like before Measure runs, without blocking it.
   * - |api-timelapse|_
     - |api-tracking|_
     - Beta
     - Tracks objects with IoU, Trackpy, btrack, Trackastra or ultrack, and quantifies motility.
   * - |api-layers|_
     - |api-layer-viewer|_
     - Alpha
     - Stacks image, label, point and shape layers, with orthogonal views and a comparison grid.
   * - |api-napari|_
     - |api-napari-curation|_
     - Alpha
     - Hands a mask to napari for correction and takes it back, recording every edit.
   * - **AI and phenotyping**
     -
     -
     -
   * - |api-annotate|_
     - |api-annotation|_
     - Stable
     - Reviews crops on a keyboard-driven grid and saves annotations to SQLite.
   * - |api-active-learning|_
     - |api-al-loop|_
     - Alpha
     - Retrains inside Annotate, re-ranks by uncertainty, and says when labelling can stop.
   * - |api-classify|_
     - |api-classification|_
     - Stable
     - Trains and applies PyTorch CNN and transformer models.
   * - |api-classify|_
     - |api-model-cards|_
     - Alpha
     - Records dataset, class balance, split rule and held-out metrics beside each checkpoint.
   * - |api-confusion|_
     - |api-confusion-drill|_
     - Alpha
     - Opens the crops behind a confusion cell, confident errors listed apart from uncertain ones.
   * - |api-ml|_
     - |api-ml-models|_
     - Stable
     - Trains interpretable classical and boosted models on measurement tables.
   * - |api-classify|_
     - |api-activation|_
     - Beta
     - Explains predictions with Captum, SmoothGrad and TorchCAM.
   * - |api-umap|_
     - |api-embedding|_
     - Beta
     - Explores image embeddings interactively and propagates cluster labels.
   * - **Sequencing and screen analysis**
     -
     -
     -
   * - |api-sequencing|_
     - |api-barcodes|_
     - Stable
     - Maps row, column and gRNA barcodes from FASTQ reads and assigns guides to imaged cells.
   * - |api-barcode-qc|_
     - |api-barcode-qc-sweep|_
     - Alpha
     - Reports reads per well, collision rate and unmapped fraction against the expected gRNAs per well.
   * - |api-regression|_
     - |api-regression-models|_
     - Stable
     - Estimates guide, gene, condition and control effects with 17 model families.
   * - |api-power|_
     - |api-power-design|_
     - Alpha
     - Answers how many cells and wells a screen needs, with sequencing error and well dropout priced in.
   * - |api-graph|_
     - |api-graph-builder|_
     - Alpha
     - Builds a plot by dragging columns onto x, y, colour, size and facet.
   * - |api-artifacts|_
     - |api-provenance|_
     - Alpha
     - Records the run id, seed and settings behind mask, measure, classify and export outputs.

.. |api-qt-app| replace:: **Qt application**
.. _api-qt-app: https://einarolafsson.github.io/spacr/api/spacr/qt/app/index.html

.. |doc-i18n| replace:: **Ten-language localization**
.. _doc-i18n: https://einarolafsson.github.io/spacr/localization.html

.. |doc-i18n-help| replace:: **Localized contextual help**
.. _doc-i18n-help: https://einarolafsson.github.io/spacr/localization.html#contextual-help

.. |api-qt-ai| replace:: **Qt AI**
.. _api-qt-ai: https://einarolafsson.github.io/spacr/api/spacr/qt/ai/index.html

.. |api-qt-ai-console| replace:: **AI-assisted console**
.. _api-qt-ai-console: https://einarolafsson.github.io/spacr/api/spacr/qt/ai/index.html

.. |api-animations| replace:: **Setting animation registry**
.. _api-animations: https://einarolafsson.github.io/spacr/api/spacr/setting_animations/index.html

.. |doc-animations| replace:: **Visual setting animations**
.. _doc-animations: https://einarolafsson.github.io/spacr/setting_animations.html

.. |api-selection| replace:: **Selection**
.. _api-selection: https://einarolafsson.github.io/spacr/api/spacr/selection/index.html

.. |api-linked-views| replace:: **Linked selection**
.. _api-linked-views: https://einarolafsson.github.io/spacr/api/spacr/qt/linked_selection/index.html

.. |api-doctor| replace:: **Doctor**
.. _api-doctor: https://einarolafsson.github.io/spacr/api/spacr/doctor/index.html

.. |api-doctor-checks| replace:: **Installation diagnosis**
.. _api-doctor-checks: https://einarolafsson.github.io/spacr/api/spacr/doctor/index.html

.. |api-mask| replace:: **Mask**
.. _api-mask: https://einarolafsson.github.io/spacr/api/spacr/core/index.html

.. |api-mask-2d| replace:: **2D mask generation**
.. _api-mask-2d: https://einarolafsson.github.io/spacr/api/spacr/core/index.html#spacr.core.preprocess_generate_masks

.. |api-mask-3d| replace:: **3D and 4D mask generation**
.. _api-mask-3d: https://einarolafsson.github.io/spacr/api/spacr/core/index.html#spacr.core.preprocess_generate_masks

.. |api-illumination| replace:: **Illumination**
.. _api-illumination: https://einarolafsson.github.io/spacr/api/spacr/illumination/index.html

.. |api-flatfield| replace:: **Flat-field correction**
.. _api-flatfield: https://einarolafsson.github.io/spacr/api/spacr/illumination/index.html

.. |api-measure| replace:: **Measure**
.. _api-measure: https://einarolafsson.github.io/spacr/api/spacr/measure/index.html

.. |api-measure-2d| replace:: **Object measurements**
.. _api-measure-2d: https://einarolafsson.github.io/spacr/api/spacr/measure/index.html#spacr.measure.measure_crop

.. |api-segqc| replace:: **Segmentation QC**
.. _api-segqc: https://einarolafsson.github.io/spacr/api/spacr/seg_qc/index.html

.. |api-segqc-verdict| replace:: **Pre-run verdict**
.. _api-segqc-verdict: https://einarolafsson.github.io/spacr/api/spacr/seg_qc/index.html

.. |api-timelapse| replace:: **Timelapse**
.. _api-timelapse: https://einarolafsson.github.io/spacr/api/spacr/timelapse/index.html

.. |api-tracking| replace:: **Object tracking**
.. _api-tracking: https://einarolafsson.github.io/spacr/api/spacr/timelapse/index.html

.. |api-layers| replace:: **Layers**
.. _api-layers: https://einarolafsson.github.io/spacr/api/spacr/layers/index.html

.. |api-layer-viewer| replace:: **Layer viewer**
.. _api-layer-viewer: https://einarolafsson.github.io/spacr/api/spacr/qt/layer_viewer/index.html

.. |api-napari| replace:: **napari bridge**
.. _api-napari: https://einarolafsson.github.io/spacr/api/spacr/napari_bridge/index.html

.. |api-napari-curation| replace:: **Mask curation**
.. _api-napari-curation: https://einarolafsson.github.io/spacr/api/spacr/napari_bridge/index.html

.. |api-annotate| replace:: **Annotate**
.. _api-annotate: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/annotate/index.html

.. |api-annotation| replace:: **Manual annotation**
.. _api-annotation: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/annotate/index.html

.. |api-active-learning| replace:: **Active learning**
.. _api-active-learning: https://einarolafsson.github.io/spacr/api/spacr/active_learning/index.html

.. |api-al-loop| replace:: **Retrain and re-rank**
.. _api-al-loop: https://einarolafsson.github.io/spacr/api/spacr/active_learning/index.html

.. |api-classify| replace:: **Classify**
.. _api-classify: https://einarolafsson.github.io/spacr/api/spacr/deep_spacr/index.html

.. |api-classification| replace:: **Image classification**
.. _api-classification: https://einarolafsson.github.io/spacr/api/spacr/deep_spacr/index.html

.. |api-model-cards| replace:: **Model cards**
.. _api-model-cards: https://einarolafsson.github.io/spacr/api/spacr/deep_spacr/index.html

.. |api-activation| replace:: **Activation maps**
.. _api-activation: https://einarolafsson.github.io/spacr/api/spacr/deep_spacr/index.html

.. |api-confusion| replace:: **Confusion**
.. _api-confusion: https://einarolafsson.github.io/spacr/api/spacr/confusion/index.html

.. |api-confusion-drill| replace:: **Confusion drill-down**
.. _api-confusion-drill: https://einarolafsson.github.io/spacr/api/spacr/confusion/index.html

.. |api-ml| replace:: **Machine learning**
.. _api-ml: https://einarolafsson.github.io/spacr/api/spacr/ml/index.html

.. |api-ml-models| replace:: **Measurement classification**
.. _api-ml-models: https://einarolafsson.github.io/spacr/api/spacr/ml/index.html

.. |api-umap| replace:: **Image UMAP**
.. _api-umap: https://einarolafsson.github.io/spacr/api/spacr/app_umap/index.html

.. |api-embedding| replace:: **Interactive embedding**
.. _api-embedding: https://einarolafsson.github.io/spacr/api/spacr/app_umap/index.html

.. |api-sequencing| replace:: **Sequencing**
.. _api-sequencing: https://einarolafsson.github.io/spacr/api/spacr/sequencing/index.html

.. |api-barcodes| replace:: **Map barcodes**
.. _api-barcodes: https://einarolafsson.github.io/spacr/api/spacr/sequencing/index.html

.. |api-barcode-qc| replace:: **Barcode QC**
.. _api-barcode-qc: https://einarolafsson.github.io/spacr/api/spacr/sequencing_qc/index.html

.. |api-barcode-qc-sweep| replace:: **Well and collision report**
.. _api-barcode-qc-sweep: https://einarolafsson.github.io/spacr/api/spacr/sequencing_qc/index.html

.. |api-regression| replace:: **Regression**
.. _api-regression: https://einarolafsson.github.io/spacr/api/spacr/ml/index.html

.. |api-regression-models| replace:: **Screen effect estimation**
.. _api-regression-models: https://einarolafsson.github.io/spacr/api/spacr/ml/index.html

.. |api-power| replace:: **Power**
.. _api-power: https://einarolafsson.github.io/spacr/api/spacr/power_model/index.html

.. |api-power-design| replace:: **Power and design**
.. _api-power-design: https://einarolafsson.github.io/spacr/api/spacr/power_simulate/index.html

.. |api-graph| replace:: **Graph**
.. _api-graph: https://einarolafsson.github.io/spacr/api/spacr/qt/widgets/graph_spec/index.html

.. |api-graph-builder| replace:: **Graph Builder**
.. _api-graph-builder: https://einarolafsson.github.io/spacr/api/spacr/qt/widgets/graph_builder/index.html

.. |api-artifacts| replace:: **Artifacts**
.. _api-artifacts: https://einarolafsson.github.io/spacr/api/spacr/artifacts/index.html

.. |api-provenance| replace:: **Run provenance**
.. _api-provenance: https://einarolafsson.github.io/spacr/api/spacr/runctx/index.html


Daten
-----

Referenzdatensätze
~~~~~~~~~~~~~~~~~~

- `Vollständiger Mikroskopie-Datensatz: BioStudies S-BIAD2135 <https://doi.org/10.6019/S-BIAD2135>`_
- `Testdatensatz: Hugging Face toxo_mito <https://huggingface.co/datasets/einarolafsson/toxo_mito>`_
- `Sequenzierungsdaten: NCBI BioProject PRJNA1261935 <https://www.ncbi.nlm.nih.gov/bioproject/?term=PRJNA1261935>`_
- `Leistungsanalyse: spaCRPower <https://github.com/maomlab/spaCRPower>`_


Beiträge und Support
------------------------

Bug reports and focused feature requests are welcome through `GitHub Probleme <https://github.com/EinarOlafsson/spacr/issues>`_. When reporting a failure, include the spaCR version, operating system, Python version, module settings and the relevant log excerpt. ``spacr-doctor`` collects most of that for you.

Lizenz
~~~~~~~~~

The current development branch is source-available under the `PolyForm Nichtkommerzielle Lizenz 1.0.0 <https://github.com/EinarOlafsson/spacr/blob/main/LICENSE>`_. Commercial use requires a separate license from the copyright holder. Released versions through spaCR 1.4.9.9 remain available under the MIT License that accompanied those releases.

Tutorials
~~~~~~~~~

The `interaktive spaCR Tutorial-Bibliothek <https://einarolafsson.github.io/spacr/tutorials/>`_ contains narrated, captioned walkthroughs of installation and of each application workflow, in eight languages.

spaCR zitieren
~~~~~~~~~~~~~~

Wenn spaCR zu Ihrer Forschung beiträgt, zitieren Sie:

Olafsson EB, *et al.* Ein gepoolter Bild-basierter CRISPR Screening identifiziert EAF1 als einen *T. gondii* Modulator der ESCRT-Subversion.

`Vordruck bioRxiv <https://www.biorxiv.org/content/10.64898/2026.07.08.737057v1>`_ · `Software-Archiv <https://doi.org/10.5281/zenodo.21343317>`_
