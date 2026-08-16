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
.. |Tests| image:: https://github.com/EinarOlafsson/spacr/actions/workflows/tests.yml/badge.svg
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
   :alt: spaCR-Arbeitsablauf und Ausgabeorganisation
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

spaCR unterstützt Python **3.9 bis 3.14** (mit Ausnahme von Python 3.14.1, das von torchvision ausgeschlossen wird). Python 3.12 bietet die größte Auswahl an optionalen wissenschaftlichen Paketen. Für CUDA-Arbeitsabläufe wird Linux empfohlen; macOS und Windows werden ebenfalls unterstützt.


Installationsdetails
--------------------

|Release| |PyPI| |CondaRecipe|

**(beta) Leichte Desktop-Installer:**

.. spacr-installer-links-begin

|InstallerWindows| |InstallerMacOS| |InstallerLinux|

.. |InstallerWindows| image:: https://raw.githubusercontent.com/EinarOlafsson/spacr/main/spacr/resources/icons/platforms/windows.png
   :width: 64
   :alt: Windows 10/11: SpaCR 1.5.0.4 herunterladen
   :target: https://github.com/EinarOlafsson/spacr/releases/download/v1.5.0.4/SpaCR-1.5.0.4-Windows-Online-Setup.exe
.. |InstallerMacOS| image:: https://raw.githubusercontent.com/EinarOlafsson/spacr/main/spacr/resources/icons/platforms/macos.png
   :width: 64
   :alt: macOS 11+ (Intel und Apple Silicium): SpaCR 1.5.0.4 herunterladen
   :target: https://github.com/EinarOlafsson/spacr/releases/download/v1.5.0.4/SpaCR-1.5.0.4-macOS-Universal-Online.pkg
.. |InstallerLinux| image:: https://raw.githubusercontent.com/EinarOlafsson/spacr/main/spacr/resources/icons/platforms/linux.png
   :width: 64
   :alt: 64-bit Linux: SpaCR 1.5.0.4 herunterladen
   :target: https://github.com/EinarOlafsson/spacr/releases/download/v1.5.0.4/SpaCR-1.5.0.4-Linux-x86_64-Online.run

.. spacr-installer-links-end

Leichte Installationsprogramme — weder conda noch vorhandenes Python erforderlich
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Das Installationsprogramm lädt während der Installation eine private Python-3.12-Laufzeitumgebung, Qt, PyTorch, spaCR und die wissenschaftlichen Abhängigkeiten herunter; weder conda noch eine vorhandene Python-Installation sind erforderlich. Standardmäßig wird die portable CPU-Version installiert, damit nicht ohne Hinweis mehrere Gigabyte an CUDA-Bibliotheken heruntergeladen werden. Unter Windows ist NVIDIA-Beschleunigung eine optionale Installationskomponente, Linux akzeptiert ``--torch-backend auto``, und das reguläre PyTorch-Wheel für macOS behält die Apple-MPS-Beschleunigung bei.

Hilfe, Fortschrittsanzeigen und Fehlermeldungen des Installationsprogramms folgen der Sprache des Betriebssystems in allen zehn spaCR-Sprachen: Englisch, Schwedisch, Deutsch, Spanisch, vereinfachtes Chinesisch, Portugiesisch, Hindi, Koreanisch, Isländisch und Französisch. Bei nicht unterstützten Gebietsschemata wird Englisch verwendet.

Machen Sie auf Linux das heruntergeladene Installationsprogramm ausführbar, bevor Sie es öffnen:

.. code-block:: bash

   chmod +x spaCR-*-Linux-x86_64-Online.run
   ./spaCR-*-Linux-x86_64-Online.run

Öffnen Sie unter macOS die heruntergeladene ``.pkg``-Datei. Falls Gatekeeper das aktuelle Beta-Installationsprogramm blockiert, weil es nicht notarisiert ist, öffnen Sie **Systemeinstellungen → Datenschutz & Sicherheit**, wählen Sie für spaCR **Dennoch öffnen** und führen Sie das Paket erneut aus.

Das Installationsprogramm validiert spaCR, Qt, PyTorch und Abhängigkeitskonsistenz, bevor eine ältere Installation ersetzt wird. Ein unterbrochenes Update lässt die vorherige Arbeitsumgebung in Kraft. Ein Diagnoseprotokoll wird als ``install.log`` im privaten spaCR-Installationsverzeichnis aufbewahrt.

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

Welche Extras aufgelöst werden können, hängt von der Python-Version ab. Unter Python 3.13 schränkt ultrack ``spacr[all]`` ein, und die NumPy-Anforderung von TorchCAM begrenzt das Extra ``attribution``; das Kernpaket und die Qt-Anwendung sind davon nicht betroffen. Unter Python 3.14 ist btrack über sein Extra verfügbar. Der CZI-Konverter pylibCZIrw ist optional und nicht getestet; das Lesen von CZI-Dateien mit czifile bleibt verfügbar.

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

Setzen Sie zur Fehlerbehebung ``SPACR_LOG_LEVEL=DEBUG``. Rotierende Protokolle werden in ``~/.spacr/logs/spacr.log`` geschrieben.


Funktionen
----------

Die sechs Module, die in den meisten Screens verwendet werden
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Mask** segmentiert mit Cellpose Zellen, Zellkerne, Pathogene und Organellen in 2D-Bildern sowie in Volumen- oder Zeitreihendaten. Die Modellliste wird aus der installierten Cellpose-Version gelesen und nicht fest vorgegeben; außerdem wird vor dem Lauf ein Objektdurchmesser aus den Bildern geschätzt. Masken lassen sich im Ebenenbetrachter von Hand korrigieren oder zur Bearbeitung an napari übergeben und anschließend wieder einlesen.

**Measure** schreibt Morphologie-, Intensitäts-, Textur- und Kolokalisationsmerkmale pro Objekt zusammen mit den Bildausschnitten in die Projektdatenbank. Neu in 1.5.0.0: Die Beleuchtungskorrektur schätzt das Flatfield aus der Platte und korrigiert die Bilder vor der Berechnung von Intensitätsmerkmalen. Dadurch wird die von Kanteneffekten in Platten-Heatmaps erkennbare Positionsverzerrung der Wells entfernt. Ein Banner zur Segmentierungsqualität beschreibt vor dem Start von Measure in Klartext, wie die Masken aussehen; es informiert, blockiert den Lauf aber nicht. Ein gezeichnetes Polygon beschränkt die Messung auf einen relevanten Bildbereich.

**Annotate** zeigt Bildausschnitte in einem tastaturgesteuerten Raster und schreibt die Beschriftungen direkt in SQLite. Die aktive Lernschleife ist in den Bildschirm integriert: Trainieren Sie ein Modell mit den bereits beschrifteten Daten neu, sortieren Sie die Warteschlange nach Unsicherheit, beobachten Sie die Lernkurve und erhalten Sie eine Empfehlung zum Beenden, sobald weitere Beschriftungen das Modell nicht mehr verändern. Die Abdeckung wird nach Klasse, Well und Platte ausgewiesen, und jede Runde wird protokolliert.

**Classify** trainiert PyTorch-CNNs und Transformer mit annotierten Bildausschnitten sowie klassische oder Boosting-Modelle mit Messtabellen. Die Genauigkeit pro Klasse wird jetzt in jeder Epoche gespeichert, und jeder Checkpoint erhält eine Modellkarte mit Datensatz, Klassenverteilung, Aufteilungsregel und Holdout-Metriken. Im Auswertungsbildschirm dient jede Zelle der Konfusionsmatrix als Abfrage: Ein Klick öffnet die zugehörigen Bildausschnitte und trennt sichere Fehlvorhersagen von unsicheren Fällen.

**Map Barcodes** dekodiert Zeilen-, Spalten- und gRNA-Barcodes aus FASTQ-Reads, weist Wells Guide-Identitäten zu und verknüpft sie mit den abgebildeten Zellen. Barcode QC berichtet Reads pro Well, Kollisionsrate und nicht zugeordneten Anteil und untersucht dabei einen Bereich um die vom Benutzer erwartete Zahl von gRNAs pro Well statt eines festen Schwellenwerts.

**Regression** schätzt Guide-, Gen-, Bedingungs- und Kontrolleffekte mit 17 Modellfamilien, darunter gemischte Modelle, Logistic, Probit, Quantile, Beta, GLMs mit quasibinomialer Varianz, Lasso, Ridge, Elastic Net, Hinge und Horseshoe. Das Ergebnis ist eine sortierte, annotierte Trefferliste statt einer bloßen Sammlung von Koeffizienten.

Neu in 1.5.0.0
~~~~~~~~~~~~~~

Bevor ein Screen existiert, berechnet das Modul Power / Design, wie viele Zellen und Wells benötigt werden. Dabei werden Sequenzierungsfehler und Ausfälle von Wells mit zu wenigen abgebildeten Zellen berücksichtigt. Ein Versuchsplaner ordnet Platte, Kontrollen und Replikate an und exportiert das Layout für die Pipeline. Anschließend fasst ein QC-Dashboard die Prüfungen von Segmentierung, Platte, Übereinstimmung der Annotatoren und Datenleckage zu einem Urteil zusammen; ComBat steht neben ``center`` und ``zscore`` für die Batchkorrektur bereit.

Ergebnisse werden direkt untersucht, statt sie zu exportieren und erneut zu importieren. Graph Builder erstellt ein Diagramm, indem Spalten auf x, y, Farbe, Größe und Facette gezogen werden. Gates in Histogrammen oder Streudiagrammen werden zu Filtern. Ein Merkmalsexplorer ordnet Merkmale danach, wie gut sie Klassen trennen. Kleine Multiples, Dosis-Wirkungs-Anpassungen, Kontrollkarten und robuste Ausreißererkennung verwenden dieselbe Achsen-Engine. Die Auswahl von Objekten in einer Ansicht überträgt sich auf alle anderen Ansichten; beim Öffnen der Auswahl erscheinen die zugehörigen Bildausschnitte. Ein Ebenenbetrachter stapelt Bild-, Beschriftungs-, Punkt- und Formebenen und bietet orthogonale Ansichten, ein synchronisiertes Vergleichsraster sowie einen Abstammungsbaum von der Zelle über den Zellkern bis zum Pathogen.

Ausführungen sind jetzt eindeutig nachvollziehbar. Jede trägt eine Lauf-ID, einen Startwert und eine ``on_error``-Richtlinie; Mask, Measure, Classify und der AnnData-Export erfassen ihre Ausgaben in einem Artefaktregister, sodass sich eine Ausgabedatei bis zu den erzeugenden Einstellungen zurückverfolgen lässt. Ein Modul öffnet die tatsächliche Ausgabe des vorherigen Schritts, der Pipelinegraph kennzeichnet veraltete Ausgaben, der Laufvergleich zeigt Unterschiede bei Einstellungen, Objektzahlen und Trefferlisten, und jeder GUI-Lauf erzeugt das entsprechende Python-Skript. Messungen lassen sich als ``.h5ad`` für scanpy exportieren; OME-Zarr und OMERO stehen über die Python-API bereit. Der Methoden-und-Ergebnisse-Exporter entwirft diese beiden Manuskriptabschnitte aus einer strukturierten Laufzusammenfassung: Das Modell formuliert den Text, doch jede Zahl stammt aus der Zusammenfassung, und ein Entwurf mit einer dort nicht enthaltenen Zahl wird verworfen. Bei Installationsproblemen meldet ``spacr-doctor``, welche spaCR-Installation tatsächlich läuft, ob die GPU nutzbar ist, ob Cellpose zur verwendeten API passt und ob Projektdatenbank und Einstellungen gültig sind; für jede fehlgeschlagene Prüfung gibt es eine kopierbare Lösung.

Mehrsprachige Desktopoberfläche
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**spaCR → Einstellungen → Sprache** schaltet die laufende Anwendung ohne Neustart auf Englisch, Schwedisch, Deutsch, Spanisch, vereinfachtes Chinesisch, Portugiesisch, Hindi, Koreanisch, Isländisch oder Französisch um. Die Auswahl bleibt gespeichert und gilt auch für später geöffnete Ansichten.

Navigation, Einstellungen, AI- und LIVE-Bedienelemente, Modulbeschreibungen und von spaCR ausgegebene Konsolenhinweise folgen der gewählten Sprache. Worker-Ausgaben, Protokolle, Tracebacks, Pfade, Datenbankwerte, Annotationen, AI-Antworten, Messungen und gespeicherte Ergebnisse werden nie übersetzt; wissenschaftliche Ausgaben bleiben dadurch im kanonischen Englisch. Einstellungshilfen, die für eine Sprache noch nicht geprüft wurden, bleiben auf Englisch, statt eine gemischtsprachige Erklärung zu erzeugen. Der `Lokalisierungsleitfaden <https://einarolafsson.github.io/spacr/localization.html>`_ beschreibt dieses Verhalten, die Umgebungsvariable zur Sprachvorgabe und die ebenfalls übersetzte `Kontexthilfe <https://einarolafsson.github.io/spacr/localization.html#contextual-help>`_.

Animierte Einstellungshilfe
~~~~~~~~~~~~~~~~~~~~~~~~~~~

94 kurze Animationen zeigen, wie sich 143 visuelle Einstellungen auf ein Bild auswirken. Bewegen Sie den Zeiger über eine Einstellung und klicken Sie in der QuickInfo auf **Animation**, um die quadratische Vorschau neben dem Text abzuspielen; ein weiterer Klick klappt sie wieder ein. Animationen laufen nur auf Anforderung und können in den Einstellungen vollständig deaktiviert werden. Die `Galerie <https://einarolafsson.github.io/spacr/setting_animations.html>`_ zeigt alle Animationen, und das `Register der Einstellungsanimationen <https://einarolafsson.github.io/spacr/api/spacr/setting_animations/index.html>`_ ordnet jede Animation ihrer Einstellung zu.

Modulreferenz
~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 25 25 25

   * - Modul
     - Funktion
     - Status
     - Beschreibung
   * - **Desktop-Bedienung**
     -
     -
     -
   * - |api-qt-app|_
     - |doc-i18n|_
     - Stabil
     - Übersetzt geöffnete und bei Bedarf erstellte Ansichten sofort zwischen zehn mitgelieferten Sprachen.
   * - |api-qt-app|_
     - |doc-i18n-help|_
     - Stabil
     - Lokalisiert Modulzusammenfassungen und die Einstellungshilfe, ohne API-URLs zu verändern.
   * - |api-qt-ai|_
     - |api-qt-ai-console|_
     - Stabil
     - Lokalisiert AI- und LIVE-Bedienelemente, ohne Inhalte von Benutzern oder Modellen zu verändern.
   * - |api-animations|_
     - |doc-animations|_
     - Stabil
     - Spielt aus der Einstellungshilfe 94 mitgelieferte Animationen für 143 visuelle Einstellungen ab.
   * - |api-selection|_
     - |api-linked-views|_
     - Alpha
     - Teilt eine Objektauswahl zwischen Tabellen-, Platten-, Einbettungs-, Streu- und Diagrammansicht.
   * - |api-doctor|_
     - |api-doctor-checks|_
     - Alpha
     - Prüft GPU, Cellpose-API, Datenbank und Einstellungen und liefert für jede fehlgeschlagene Prüfung eine Lösung.
   * - **Bildanalyse**
     -
     -
     -
   * - |api-mask|_
     - |api-mask-2d|_
     - Stabil
     - Segmentiert Zellen, Zellkerne, Pathogene und Organellen in 2D-Bildern.
   * - |api-mask|_
     - |api-mask-3d|_
     - Beta
     - Segmentiert Volumenbilder und 4D-Zeitreihen.
   * - |api-illumination|_
     - |api-flatfield|_
     - Alpha
     - Schätzt das Flatfield aus der Platte und korrigiert es vor der Intensitätsmessung.
   * - |api-measure|_
     - |api-measure-2d|_
     - Stabil
     - Misst Morphologie, Intensität, Textur und Kolokalisation und speichert die Bildausschnitte.
   * - |api-segqc|_
     - |api-segqc-verdict|_
     - Alpha
     - Beschreibt vor dem Start von Measure die Segmentierungsqualität, ohne die Ausführung zu blockieren.
   * - |api-timelapse|_
     - |api-tracking|_
     - Beta
     - Verfolgt Objekte mit IoU, Trackpy, btrack, Trackastra oder ultrack und quantifiziert ihre Beweglichkeit.
   * - |api-layers|_
     - |api-layer-viewer|_
     - Alpha
     - Stapelt Bild-, Beschriftungs-, Punkt- und Formebenen mit orthogonalen Ansichten und Vergleichsraster.
   * - |api-napari|_
     - |api-napari-curation|_
     - Alpha
     - Übergibt eine Maske zur Korrektur an napari, übernimmt sie zurück und protokolliert jede Bearbeitung.
   * - **AI und Phänotypisierung**
     -
     -
     -
   * - |api-annotate|_
     - |api-annotation|_
     - Stabil
     - Prüft Bildausschnitte in einem tastaturgesteuerten Raster und speichert Annotationen in SQLite.
   * - |api-active-learning|_
     - |api-al-loop|_
     - Alpha
     - Trainiert innerhalb von Annotate neu, sortiert nach Unsicherheit und zeigt, wann die Beschriftung beendet werden kann.
   * - |api-classify|_
     - |api-classification|_
     - Stabil
     - Trainiert und verwendet CNN- und Transformer-Modelle mit PyTorch.
   * - |api-classify|_
     - |api-model-cards|_
     - Alpha
     - Dokumentiert bei jedem Checkpoint Datensatz, Klassenverteilung, Aufteilungsregel und Holdout-Metriken.
   * - |api-confusion|_
     - |api-confusion-drill|_
     - Alpha
     - Öffnet die Bildausschnitte hinter einer Konfusionsmatrixzelle und trennt sichere Fehler von unsicheren Fällen.
   * - |api-ml|_
     - |api-ml-models|_
     - Stabil
     - Trainiert interpretierbare klassische und Boosting-Modelle auf Messtabellen.
   * - |api-classify|_
     - |api-activation|_
     - Beta
     - Erklärt Vorhersagen mit Captum, SmoothGrad und TorchCAM.
   * - |api-umap|_
     - |api-embedding|_
     - Beta
     - Untersucht Bildeinbettungen interaktiv und überträgt Clusterbezeichnungen.
   * - **Sequenzierung und Screen-Analyse**
     -
     -
     -
   * - |api-sequencing|_
     - |api-barcodes|_
     - Stabil
     - Ordnet Zeilen-, Spalten- und gRNA-Barcodes aus FASTQ-Reads zu und weist abgebildeten Zellen Guides zu.
   * - |api-barcode-qc|_
     - |api-barcode-qc-sweep|_
     - Alpha
     - Berichtet Reads pro Well, Kollisionsrate und nicht zugeordneten Anteil im Verhältnis zu den erwarteten gRNAs pro Well.
   * - |api-regression|_
     - |api-regression-models|_
     - Stabil
     - Schätzt Guide-, Gen-, Bedingungs- und Kontrolleffekte mit 17 Modellfamilien.
   * - |api-power|_
     - |api-power-design|_
     - Alpha
     - Berechnet den Zell- und Well-Bedarf eines Screens unter Einbeziehung von Sequenzierungsfehlern und Well-Ausfällen.
   * - |api-graph|_
     - |api-graph-builder|_
     - Alpha
     - Erstellt ein Diagramm, indem Spalten auf x, y, Farbe, Größe und Facette gezogen werden.
   * - |api-artifacts|_
     - |api-provenance|_
     - Alpha
     - Erfasst Lauf-ID, Startwert und Einstellungen hinter den Ausgaben von Mask, Measure, Classify und Export.

.. |api-qt-app| replace:: **Qt-Anwendung**
.. _api-qt-app: https://einarolafsson.github.io/spacr/api/spacr/qt/app/index.html

.. |doc-i18n| replace:: **Lokalisierung in zehn Sprachen**
.. _doc-i18n: https://einarolafsson.github.io/spacr/localization.html

.. |doc-i18n-help| replace:: **Lokalisierte Kontexthilfe**
.. _doc-i18n-help: https://einarolafsson.github.io/spacr/localization.html#contextual-help

.. |api-qt-ai| replace:: **Qt AI**
.. _api-qt-ai: https://einarolafsson.github.io/spacr/api/spacr/qt/ai/index.html

.. |api-qt-ai-console| replace:: **AI-unterstützte Konsole**
.. _api-qt-ai-console: https://einarolafsson.github.io/spacr/api/spacr/qt/ai/index.html

.. |api-animations| replace:: **Register der Einstellungsanimationen**
.. _api-animations: https://einarolafsson.github.io/spacr/api/spacr/setting_animations/index.html

.. |doc-animations| replace:: **Animationen visueller Einstellungen**
.. _doc-animations: https://einarolafsson.github.io/spacr/setting_animations.html

.. |api-selection| replace:: **Auswahl**
.. _api-selection: https://einarolafsson.github.io/spacr/api/spacr/selection/index.html

.. |api-linked-views| replace:: **Verknüpfte Auswahl**
.. _api-linked-views: https://einarolafsson.github.io/spacr/api/spacr/qt/linked_selection/index.html

.. |api-doctor| replace:: **Doctor**
.. _api-doctor: https://einarolafsson.github.io/spacr/api/spacr/doctor/index.html

.. |api-doctor-checks| replace:: **Installationsdiagnose**
.. _api-doctor-checks: https://einarolafsson.github.io/spacr/api/spacr/doctor/index.html

.. |api-mask| replace:: **Mask**
.. _api-mask: https://einarolafsson.github.io/spacr/api/spacr/core/index.html

.. |api-mask-2d| replace:: **2D-Maskenerzeugung**
.. _api-mask-2d: https://einarolafsson.github.io/spacr/api/spacr/core/index.html#spacr.core.preprocess_generate_masks

.. |api-mask-3d| replace:: **3D- und 4D-Maskenerzeugung**
.. _api-mask-3d: https://einarolafsson.github.io/spacr/api/spacr/core/index.html#spacr.core.preprocess_generate_masks

.. |api-illumination| replace:: **Beleuchtung**
.. _api-illumination: https://einarolafsson.github.io/spacr/api/spacr/illumination/index.html

.. |api-flatfield| replace:: **Flatfield-Korrektur**
.. _api-flatfield: https://einarolafsson.github.io/spacr/api/spacr/illumination/index.html

.. |api-measure| replace:: **Measure**
.. _api-measure: https://einarolafsson.github.io/spacr/api/spacr/measure/index.html

.. |api-measure-2d| replace:: **Objektmessungen**
.. _api-measure-2d: https://einarolafsson.github.io/spacr/api/spacr/measure/index.html#spacr.measure.measure_crop

.. |api-segqc| replace:: **Segmentierungs-QC**
.. _api-segqc: https://einarolafsson.github.io/spacr/api/spacr/seg_qc/index.html

.. |api-segqc-verdict| replace:: **Prüfergebnis vor dem Lauf**
.. _api-segqc-verdict: https://einarolafsson.github.io/spacr/api/spacr/seg_qc/index.html

.. |api-timelapse| replace:: **Timelapse**
.. _api-timelapse: https://einarolafsson.github.io/spacr/api/spacr/timelapse/index.html

.. |api-tracking| replace:: **Objektverfolgung**
.. _api-tracking: https://einarolafsson.github.io/spacr/api/spacr/timelapse/index.html

.. |api-layers| replace:: **Ebenen**
.. _api-layers: https://einarolafsson.github.io/spacr/api/spacr/layers/index.html

.. |api-layer-viewer| replace:: **Ebenenbetrachter**
.. _api-layer-viewer: https://einarolafsson.github.io/spacr/api/spacr/qt/layer_viewer/index.html

.. |api-napari| replace:: **napari-Anbindung**
.. _api-napari: https://einarolafsson.github.io/spacr/api/spacr/napari_bridge/index.html

.. |api-napari-curation| replace:: **Maskenkorrektur**
.. _api-napari-curation: https://einarolafsson.github.io/spacr/api/spacr/napari_bridge/index.html

.. |api-annotate| replace:: **Annotate**
.. _api-annotate: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/annotate/index.html

.. |api-annotation| replace:: **Manuelle Annotation**
.. _api-annotation: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/annotate/index.html

.. |api-active-learning| replace:: **Aktives Lernen**
.. _api-active-learning: https://einarolafsson.github.io/spacr/api/spacr/active_learning/index.html

.. |api-al-loop| replace:: **Neu trainieren und ordnen**
.. _api-al-loop: https://einarolafsson.github.io/spacr/api/spacr/active_learning/index.html

.. |api-classify| replace:: **Classify**
.. _api-classify: https://einarolafsson.github.io/spacr/api/spacr/deep_spacr/index.html

.. |api-classification| replace:: **Bildklassifikation**
.. _api-classification: https://einarolafsson.github.io/spacr/api/spacr/deep_spacr/index.html

.. |api-model-cards| replace:: **Modellkarten**
.. _api-model-cards: https://einarolafsson.github.io/spacr/api/spacr/deep_spacr/index.html

.. |api-activation| replace:: **Aktivierungskarten**
.. _api-activation: https://einarolafsson.github.io/spacr/api/spacr/deep_spacr/index.html

.. |api-confusion| replace:: **Confusion**
.. _api-confusion: https://einarolafsson.github.io/spacr/api/spacr/confusion/index.html

.. |api-confusion-drill| replace:: **Detailansicht der Konfusionsmatrix**
.. _api-confusion-drill: https://einarolafsson.github.io/spacr/api/spacr/confusion/index.html

.. |api-ml| replace:: **Maschinelles Lernen**
.. _api-ml: https://einarolafsson.github.io/spacr/api/spacr/ml/index.html

.. |api-ml-models| replace:: **Klassifikation von Messungen**
.. _api-ml-models: https://einarolafsson.github.io/spacr/api/spacr/ml/index.html

.. |api-umap| replace:: **Image UMAP**
.. _api-umap: https://einarolafsson.github.io/spacr/api/spacr/app_umap/index.html

.. |api-embedding| replace:: **Interaktive Einbettung**
.. _api-embedding: https://einarolafsson.github.io/spacr/api/spacr/app_umap/index.html

.. |api-sequencing| replace:: **Sequenzierung**
.. _api-sequencing: https://einarolafsson.github.io/spacr/api/spacr/sequencing/index.html

.. |api-barcodes| replace:: **Barcodes zuordnen**
.. _api-barcodes: https://einarolafsson.github.io/spacr/api/spacr/sequencing/index.html

.. |api-barcode-qc| replace:: **Barcode-QC**
.. _api-barcode-qc: https://einarolafsson.github.io/spacr/api/spacr/sequencing_qc/index.html

.. |api-barcode-qc-sweep| replace:: **Well- und Kollisionsbericht**
.. _api-barcode-qc-sweep: https://einarolafsson.github.io/spacr/api/spacr/sequencing_qc/index.html

.. |api-regression| replace:: **Regression**
.. _api-regression: https://einarolafsson.github.io/spacr/api/spacr/ml/index.html

.. |api-regression-models| replace:: **Schätzung von Screen-Effekten**
.. _api-regression-models: https://einarolafsson.github.io/spacr/api/spacr/ml/index.html

.. |api-power| replace:: **Power**
.. _api-power: https://einarolafsson.github.io/spacr/api/spacr/power_model/index.html

.. |api-power-design| replace:: **Power und Versuchsplanung**
.. _api-power-design: https://einarolafsson.github.io/spacr/api/spacr/power_simulate/index.html

.. |api-graph| replace:: **Graph**
.. _api-graph: https://einarolafsson.github.io/spacr/api/spacr/qt/widgets/graph_spec/index.html

.. |api-graph-builder| replace:: **Graph Builder**
.. _api-graph-builder: https://einarolafsson.github.io/spacr/api/spacr/qt/widgets/graph_builder/index.html

.. |api-artifacts| replace:: **Artefakte**
.. _api-artifacts: https://einarolafsson.github.io/spacr/api/spacr/artifacts/index.html

.. |api-provenance| replace:: **Laufprovenienz**
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

Fehlerberichte und konkrete Funktionswünsche sind über `GitHub Issues <https://github.com/EinarOlafsson/spacr/issues>`_ willkommen. Geben Sie bei einem Fehler die spaCR-Version, das Betriebssystem, die Python-Version, die Moduleinstellungen und den relevanten Protokollauszug an. ``spacr-doctor`` sammelt den Großteil dieser Angaben automatisch.

Lizenz
~~~~~~~~~

Der Quellcode des aktuellen Entwicklungszweigs ist unter der `PolyForm Noncommercial License 1.0.0 <https://github.com/EinarOlafsson/spacr/blob/main/LICENSE>`_ verfügbar. Für kommerzielle Nutzung ist eine gesonderte Lizenz des Rechteinhabers erforderlich. Veröffentlichte Versionen bis einschließlich spaCR 1.4.9.9 bleiben unter der jeweils mitgelieferten MIT-Lizenz verfügbar.

Tutorials
~~~~~~~~~

Die `interaktive spaCR-Tutorialbibliothek <https://einarolafsson.github.io/spacr/tutorials/>`_ enthält in acht Sprachen vertonte und untertitelte Anleitungen zur Installation und zu jedem Arbeitsablauf der Anwendung.

spaCR zitieren
~~~~~~~~~~~~~~

Wenn spaCR zu Ihrer Forschung beiträgt, zitieren Sie:

Olafsson EB, *et al.* Ein gepoolter bildbasierter CRISPR-Screen identifiziert EAF1 als Modulator der ESCRT-Subversion durch *T. gondii*.

`Vordruck bioRxiv <https://www.biorxiv.org/content/10.64898/2026.07.08.737057v1>`_ · `Software-Archiv <https://doi.org/10.5281/zenodo.21343317>`_
