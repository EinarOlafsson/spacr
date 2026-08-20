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

.. image:: ../../../spacr/resources/icons/logo_spacr.png
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

**Räumliche Phänotypanalyse von CRISPR-Screens.**

spaCR segmentiert und vermisst einzelne Zellen in High-Content-Mikroskopiebildern, verknüpft jede Zelle mit der erhaltenen gRNA und berichtet, welche Gene den Phänotyp verändert haben. Plattenbilder und FASTQ-Reads dienen als Eingabe; ausgegeben werden Messungen pro Objekt, trainierte Klassifikatoren, Effektgrößen pro Guide und Gen sowie eine Rangliste der Treffer.

Für bildbasierte gepoolte CRISPR-Screens deckt dies den gesamten Arbeitsablauf ab. Bei High-Content-Mikroskopie ohne Screen können Segmentierung, Messung, Annotation und Klassifizierung eigenständig ausgeführt werden.

Bilder, Masken, Bildausschnitte, Messungen, Annotationen, Vorhersagen, Barcodes und Well-Kennungen liegen in einem einzigen SQLite-Projekt. Dadurch lässt sich ein Ergebniswert bis zu seinem Ursprungsobjekt zurückverfolgen.

Führen Sie spaCR als Desktopanwendung oder ohne grafische Oberfläche auf einer Workstation, einem Server oder Cluster aus. Beide Varianten verwenden dieselben Module; CUDA wird automatisch genutzt, wenn das jeweilige Modul es unterstützt.


Workflow auf einen Blick
------------------------

|WorkflowMask| |WorkflowArrow| |WorkflowMeasure| |WorkflowArrow| |WorkflowAnnotate| |WorkflowArrow| |WorkflowClassify| |WorkflowArrow| |WorkflowBarcodes| |WorkflowArrow| |WorkflowRegression|

.. |WorkflowMask| image:: ../../../spacr/resources/icons/workflow/mask.png
   :width: 96
   :alt: Mask API
   :target: https://einarolafsson.github.io/spacr/api/spacr/core/index.html
   :align: middle
.. |WorkflowMeasure| image:: ../../../spacr/resources/icons/workflow/measure.png
   :width: 96
   :alt: Measure API
   :target: https://einarolafsson.github.io/spacr/api/spacr/measure/index.html
   :align: middle
.. |WorkflowAnnotate| image:: ../../../spacr/resources/icons/workflow/annotate.png
   :width: 96
   :alt: Annotate API
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/annotate_engine/index.html
   :align: middle
.. |WorkflowClassify| image:: ../../../spacr/resources/icons/workflow/classify_merged.png
   :width: 96
   :alt: Classify API
   :target: https://einarolafsson.github.io/spacr/api/spacr/classify/index.html
   :align: middle
.. |WorkflowBarcodes| image:: ../../../spacr/resources/icons/workflow/map_barcodes.png
   :width: 96
   :alt: Map Barcodes API
   :target: https://einarolafsson.github.io/spacr/api/spacr/sequencing/index.html
   :align: middle
.. |WorkflowRegression| image:: ../../../spacr/resources/icons/workflow/regression.png
   :width: 96
   :alt: Regression API
   :target: https://einarolafsson.github.io/spacr/api/spacr/ml/index.html
   :align: middle
.. |WorkflowArrow| image:: ../../../spacr/resources/icons/workflow/arrow.png
   :width: 18
   :align: middle

.. image:: ../../../spacr/resources/icons/workflow_home_apps.png
   :alt: spaCR-Arbeitsablauf und Ausgabeorganisation
   :align: center

Der Hauptpfad ist Mask →Messen →Annotate →Klassifizieren → Karte Barcodes →Regression. Das Gitter darunter enthält jede andere Anwendung in den gleichen Kategorien und Reihenfolge, die auf dem spaCR Home-Bildschirm verwendet wird.


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

Machen Sie auf Linux die heruntergeladene Datei ausführbar und führen Sie sie aus:

.. code-block:: bash

   chmod +x SpaCR-*-Linux-x86_64-Online.run
   ./SpaCR-*-Linux-x86_64-Online.run

Öffnen Sie auf macOS das ``.pkg``. Die aktuelle Beta wird nicht beglaubigt; wenn Gatekeeper sie blockiert, wählen Sie **Systemeinstellungen → Datenschutz & Sicherheit → Öffnen Sie trotzdem**.

Siehe die Anweisungen `Installationsanleitung <../../source/installers.rst>`_ zur Aktualisierung, Deinstallation, Offline- und Fehlerbehebung.

Python-Installation
~~~~~~~~~~~~~~~~~~~

Python 3.12 hat die größte Auswahl an optionalen wissenschaftlichen Paketen:

.. code-block:: bash

   conda create -n spacr python=3.12 -y
   conda activate spacr
   python -m pip install --upgrade pip
   python -m pip install "spacr[qt]"
   spacr

spaCR unterstützt Python **3,9 bis 3.14**, außer Python 3.14.1, was torchvision ausschließt. Linux wird für CUDA Workflows empfohlen; macOS und Windows werden ebenfalls unterstützt.

Für einen Server, Cluster oder CI-Läufer Qt weglassen:

.. code-block:: bash

   python -m pip install spacr
   spacr-run --list

Optional integrations are installed separately, for example ``spacr[ome-zarr]``, ``spacr[omero]``, ``spacr[napari]`` and ``spacr[czi,nd2,lif]``. See the `Installationsanleitung <../../source/installers.rst>`_ for the complete extras and Python-version compatibility table.

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

Setzen Sie ``SPACR_LOG_LEVEL=DEBUG`` bei der Fehlerbehebung. Rotierende Protokolle werden auf ``~/.spacr/logs/spacr.log`` geschrieben. Die klassische Tk-Schnittstelle bleibt als ``spacr-legacy`` verfügbar, wird aber nicht mehr entwickelt.


Was Sie tun können
------------------

Die meisten Screenings folgen sechs Modulen:

- **Mask** Segmente von Zellen, Kernen, Krankheitserregern und Organellen mit Cellpose.
- **Measure** schreibt Morphologie, Intensität, Textur, räumliche und Kolokalisierungsmerkmale, zusammen mit Objekt-Bildausschnitte, zu SQLite.
- **Annotate** markiert Bildausschnitte in einem keyboardgesteuerten Grid und unterstützt aktive Lernschlangen.
- **Classify** trainiert Bild- oder messbasierte Modelle und zeichnet mit jedem Kontrollpunkt die ausgehaltene Leistung auf.
- **Map Barcodes** Karten FASTQ liest zu Wells und gRNAs, mit Fülle, Kollision und Abdeckung QC.
- **Regression** schätzt Leit-, Gen-, Zustands- und Kontrolleffekte mit Modellfamilien, die für kontinuierliche, fraktionelle und Zählungsreaktionen geeignet sind.

Das gleiche Projekt kann auch Platten entwerfen, Leistung schätzen, Batch-Effekte korrigieren, Segmentierungsqualität inspizieren, verknüpfte Parzellen und Bildausschnitte erkunden, AnnData exportieren, unterbrochene Arbeiten fortsetzen und die Einstellungen hinter jedem Ergebnis aufzeichnen.

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


Beiträge und Support
------------------------

Bug reports and focused feature requests are welcome through `GitHub Probleme <https://github.com/EinarOlafsson/spacr/issues>`_. When reporting a failure, include the spaCR version, operating system, Python version, module settings and the relevant log excerpt. ``spacr-doctor`` collects most of that for you.

Lizenz
~~~~~~~~~

The current development branch is source-available under the `PolyForm Nichtkommerzielle Lizenz 1.0.0 <https://github.com/EinarOlafsson/spacr/blob/main/LICENSE>`_. Commercial use requires a separate license from the copyright holder. Released versions through spaCR 1.4.9.9 remain available under the MIT License that accompanied those releases.

Tutorials
~~~~~~~~~

Der `interaktive spaCR Tutorial-Bibliothek <https://einarolafsson.github.io/spacr/tutorials/>`_ enthält erzählte, beschriftete Durchläufe der Installation und jedes Anwendungsworkflows in 73 Lektionen mit 50 Stimmen in acht Sprachen.

spaCR zitieren
~~~~~~~~~~~~~~

Wenn spaCR zu Ihrer Forschung beiträgt, zitieren Sie:

Olafsson EB, *et al.* Ein gepoolter Bild-basierter CRISPR Screening identifiziert EAF1 als einen *T. gondii* Modulator der ESCRT-Subversion.

`Vordruck bioRxiv <https://www.biorxiv.org/content/10.64898/2026.07.08.737057v1>`_ · `Software-Archiv <https://doi.org/10.5281/zenodo.21343317>`_

Danksagung
~~~~~~~~~~~~~~~

spaCR baut auf offener wissenschaftlicher Software auf, darunter NumPy, pandas, scikit-image, scikit-learn, Cellpose, PyTorch und Qt. Die für die mehrsprachige Dokumentation und die Oberflächenkataloge verwendeten Modelle sind in der `Attribution der Übersetzungsmodelle <../TRANSLATION_MODELS.md>`_ aufgeführt.
