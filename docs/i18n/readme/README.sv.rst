|Docs| |Tutorials| |PyPI| |Python| |Tests| |Qt| |Source| |Issues| |License| |DOI|

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
.. |Tests| image:: https://github.com/EinarOlafsson/spacr/actions/workflows/tests.yml/badge.svg
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
.. |DOI| image:: https://img.shields.io/badge/DOI-10.5281%2Fzenodo.21343317-blue
   :target: https://doi.org/10.5281/zenodo.21343317
   :alt: Zenodo-DOI
.. |Release| image:: https://img.shields.io/github/v/release/EinarOlafsson/spacr?label=Installers
   :target: https://github.com/EinarOlafsson/spacr/releases/latest
   :alt: Senaste installationsprogrammen
.. |CondaRecipe| image:: https://img.shields.io/badge/conda--forge-recipe-44A833?logo=anaconda
   :target: https://github.com/EinarOlafsson/spacr/tree/main/conda-forge/recipe
   :alt: conda-forge-recept

.. image:: https://raw.githubusercontent.com/EinarOlafsson/spacr/main/spacr/resources/icons/logo_spacr.png
   :alt: spaCR
   :align: center
   :width: 360

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

`Information om översättningsmodellerna <../TRANSLATION_MODELS.md>`_

**Rumslig fenotypanalys av CRISPR-screeningar.**

spaCR segmenterar och mäter enskilda celler i mikroskopibilder med högt innehåll, kopplar varje cell till den gRNA den fick och rapporterar vilka gener som förändrade fenotypen. Plattbilder och FASTQ-läsningar matas in; ut kommer mätningar per objekt, tränade klassificerare, effektstorlekar per guide och gen samt en rangordnad träfflista.

För bildbaserade poolade CRISPR-screeningar täcker detta hela arbetsflödet. Om du har mikroskopi med högt innehåll men ingen screening kan delarna för segmentering, mätning, annotering och klassificering köras fristående.

Bilder, masker, bildutsnitt, mätningar, annoteringar, prediktioner, streckkoder och brunnsidentifierare lagras i ett enda SQLite-projekt, så ett värde i ett resultat kan spåras tillbaka till objektet det kom från.

Kör spaCR som skrivbordsprogram eller utan grafiskt gränssnitt på en arbetsstation, server eller beräkningskluster. Båda sätten använder samma moduler, och CUDA används automatiskt när modulen stöder det.


Arbetsflödet i korthet
----------------------

|Tutorials|

.. image:: https://raw.githubusercontent.com/EinarOlafsson/spacr/main/spacr/resources/icons/flow_chart_v3.png
   :alt: spaCR:s arbetsflöde och struktur för utdata
   :align: center

Mikroskopibilder (TIFF, OME-TIFF, LIF, CZI, ND2) och sekvenseringsläsningar (FASTQ) matas in i kompletterande arbetsflöden för bildanalys och streckkodsmappning. Objekttabeller, bildutsnitt, annoteringar, prediktioner, guideidentiteter, QC-resultat och sammanfattningar per brunn analyseras sedan tillsammans.


Snabbstart
-----------

.. code-block:: bash

   conda create -n spacr python=3.12 -y
   conda activate spacr
   python -m pip install --upgrade pip
   python -m pip install "spacr[qt]"
   spacr

spaCR stöder Python **3.9 till 3.14** (utom Python 3.14.1, som torchvision utesluter). Python 3.12 har det största urvalet av valfria vetenskapliga paket. Linux rekommenderas för CUDA-arbetsflöden; macOS och Windows stöds också.


Installationsinformation
------------------------

|Release| |PyPI| |CondaRecipe|

**(beta) Lätta skrivbordsinstallationer:**

.. spacr-installer-links-begin

* `Windows 10/11: Hämta SpaCR 1.5.0.4 <https://github.com/EinarOlafsson/spacr/releases/download/v1.5.0.4/SpaCR-1.5.0.4-Windows-Online-Setup.exe>`_
* `macOS 11+ (Intel och Apple kisel): ladda ner SpaCR 1.5.0.4 <https://github.com/EinarOlafsson/spacr/releases/download/v1.5.0.4/SpaCR-1.5.0.4-macOS-Universal-Online.pkg>`_
* `64-bit Linux: Hämta SpaCR 1.5.0.4 <https://github.com/EinarOlafsson/spacr/releases/download/v1.5.0.4/SpaCR-1.5.0.4-Linux-x86_64-Online.run>`_

.. spacr-installer-links-end

Lätta installationsprogram — varken conda eller befintlig Python krävs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Installationsprogrammet hämtar en privat Python 3.12-miljö, Qt, PyTorch, spaCR och de vetenskapliga beroendena under installationen, så varken conda eller en befintlig Python-installation behövs. Den portabla CPU-versionen är standard och förhindrar att flera gigabyte CUDA-bibliotek hämtas utan förvarning. I Windows erbjuds NVIDIA-acceleration som en valfri komponent, Linux godtar ``--torch-backend auto`` och PyTorch-standardpaketet för macOS behåller Apple MPS-accelerationen.

Installationsprogrammets hjälp, förloppsmeddelanden och felmeddelanden följer operativsystemets språk på spaCR:s alla tio språk: engelska, svenska, tyska, spanska, förenklad kinesiska, portugisiska, hindi, koreanska, isländska och franska. Språk som inte stöds använder engelska som reservspråk.

På Linux, gör det nedladdade installationsprogrammet körbart innan det öppnas:

.. code-block:: bash

   chmod +x SpaCR-*-Linux-x86_64-Online.run
   ./SpaCR-*-Linux-x86_64-Online.run

På macOS öppnar du den hämtade ``.pkg``-filen. Om Gatekeeper blockerar det aktuella betainstallationsprogrammet för att det inte är notariserat, öppnar du **Systeminställningar → Integritet och säkerhet**, väljer **Öppna ändå** för spaCR och kör sedan paketet igen.

Installationsprogrammet validerar spaCR, Qt, PyTorch och beroendekonsistens innan en äldre installation byts ut, så en avbruten uppdatering lämnar den tidigare arbetsmiljön på plats. En diagnostisk logg behålls som ``install.log`` i den privata installationskatalogen spaCR.

Skrivbordsprogram från PyPI
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   python -m pip install "spacr[qt]"
   spacr

Installation utan grafiskt gränssnitt eller på server
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   python -m pip install spacr
   spacr-run --list

Senaste utvecklingsgrenen
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   git clone https://github.com/EinarOlafsson/spacr.git
   cd spacr
   git switch nightly
   python -m pip install -e ".[qt]"

Conda-miljöer
~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   conda create -n spacr python=3.12 pip -y
   conda activate spacr
   python -m pip install "spacr[qt]"

Valfria funktioner
~~~~~~~~~~~~~~~~~~~~~

Installera endast de tillägg som arbetsflödet behöver:

.. code-block:: bash

   python -m pip install "spacr[trackastra]"    # transformer tracking
   python -m pip install "spacr[ultrack]"       # global-optimization tracking
   python -m pip install "spacr[btrack]"        # btrack timelapse tracking
   python -m pip install "spacr[attribution]"   # TorchCAM methods
   python -m pip install "spacr[boosting]"      # LightGBM and CatBoost
   python -m pip install "spacr[zernike]"       # Zernike measurements
   python -m pip install "spacr[napari]"        # napari mask correction
   python -m pip install "spacr[czi,nd2,lif]"   # vendor file readers

Vilka tillägg som kan installeras beror på Python-versionen. I Python 3.13 begränsar ultrack ``spacr[all]``, och TorchCAM:s NumPy-krav begränsar tillägget ``attribution``; kärnpaketet och Qt-programmet påverkas inte. I Python 3.14 är btrack tillgängligt via sitt tillägg. CZI-konverteraren pylibCZIrw är valfri och oprövad; CZI-läsning baserad på czifile är fortfarande tillgänglig.

Det äldre Tk-gränssnittet är fortfarande installerat som ``spacr-legacy`` men är inte längre utvecklat.


Kommandoradskommandon
-------------------------

.. code-block:: bash

   spacr                                      # Qt application
   spacr-doctor                               # diagnose the installation
   spacr-run --list                           # list headless modules
   spacr-run --describe MODULE                # inspect a module contract
   spacr-run MODULE --settings settings.csv   # execute a module
   spacr-run validate --module MODULE \
       --settings settings.csv                # validate before running
   spacr-repro RUN_DIR                        # replay a recorded run

Ange ``SPACR_LOG_LEVEL=DEBUG`` vid felsökning. Roterande loggar skrivs till ``~/.spacr/logs/spacr.log``.


Funktioner
----------

De sex moduler som används i de flesta screeningar
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Mask** segmenterar celler, cellkärnor, patogener och organeller med Cellpose i 2D-bilder samt i volym- och tidsseriedata. Modellistan läses från den installerade Cellpose-versionen i stället för att vara hårdkodad, och objektens diameter uppskattas från bilderna innan körningen startar. Masker kan korrigeras manuellt i lagervisaren eller skickas till napari och tillbaka.

**Measure** skriver morfologi, intensitet, textur och samlokaliseringsmått för varje objekt till projektdatabasen tillsammans med bildutsnitten. Nytt i 1.5.0.0: belysningskorrigeringen uppskattar flatfältet från själva plattan och dividerar bort det innan intensitetsmåtten tas, vilket avlägsnar den brunnspositionsbias som syns som kanteffekter i plattans värmekarta. En segmenterings-QC beskriver maskerna på klarspråk innan Measure körs; den informerar men blockerar inte. En ritad polygon begränsar mätningen till ett intresseområde.

**Annotate** visar bildutsnitt i ett tangentbordsstyrt rutnät och skriver etiketterna direkt till SQLite. Hela den aktiva inlärningsloopen finns nu på samma skärm: träna om modellen med befintliga etiketter, ordna om kön efter osäkerhet, följ inlärningskurvan och få besked om när fler etiketter inte längre förändrar modellen. Täckningen rapporteras per klass, brunn och platta, och varje omgång registreras.

**Classify** tränar PyTorch-CNN:er och transformermodeller på annoterade bildutsnitt samt klassiska eller boostade modeller på mättabeller. Noggrannheten per klass sparas nu för varje epok, och varje kontrollpunkt får ett modellkort med datauppsättning, klassbalans, uppdelningsregel och mått för undanhållna data. I utvärderingsvyn fungerar en cell i förväxlingsmatrisen som en fråga: klicka för att öppna motsvarande bildutsnitt, där säkert felaktiga prediktioner visas separat från osäkra.

**Map Barcodes** avkodar rad-, kolumn- och gRNA-streckkoder från FASTQ-läsningar, tilldelar guideidentiteter till brunnar och kopplar dem till avbildade celler. Streckkods-QC rapporterar antalet läsningar per brunn, kollisionsfrekvens och omappad andel och söker kring det förväntade antalet gRNA per brunn som användaren anger, i stället för att använda ett fast tröskelvärde.

**Regression** skattar effekter av guider, gener, villkor och kontroller med 17 modellfamiljer, bland annat blandade modeller, logistisk regression, probit, kvantil- och betamodeller, GLM med kvasibinomial varians, lasso, ridge, elastic net, hinge och horseshoe. Resultatet är en rangordnad och annoterad träfflista, inte bara en samling koefficienter.

Nytt i 1.5.0.0
~~~~~~~~~~~~~~

Innan en screening finns beräknar modulen Power / Design hur många celler och brunnar som behövs, med hänsyn till sekvenseringsfel och bortfall från brunnar där för få celler avbildats. En experimentdesigner placerar ut plattan, kontrollerna och replikaten och exporterar layouten till arbetsflödet. Efteråt samlar en QC-panel kontroller av segmentering, platta, annotatörsöverensstämmelse och dataläckage i ett enda utlåtande; för batchkorrigering finns ComBat bredvid ``center`` och ``zscore``.

Resultaten utforskas direkt i stället för att exporteras och importeras på nytt. I Graph Builder ritas en tabell genom att kolumner dras till x, y, färg, storlek och fasett. Gates som ritas i ett histogram eller spridningsdiagram blir filter. Feature Explorer rangordnar funktioner efter hur väl de skiljer klasserna åt. Små multiplar, dos–responsanpassningar, kontrolldiagram och robust avvikelsedetektering använder samma axelmotor. Objekt som väljs i en vy väljs i alla, och när urvalet öppnas visas de bildutsnitt som objekten kommer från. Layer Viewer lägger bilder, etiketter, punkter och former i lager och erbjuder ortogonala vyer, ett synkroniserat jämförelserutnät och ett släktträd från cell till cellkärna och patogen.

Varje körning kan nu identifieras och spåras. Den har ett körnings-ID, ett slumpfrö och en ``on_error``-policy; Mask, Measure, Classify och AnnData-exporten registrerar sina utdata i ett artefaktregister, så att en utdatafil kan spåras tillbaka till de inställningar som skapade den. En modul öppnar det som det föregående steget faktiskt skrev, arbetsflödesgrafen markerar inaktuella utdata, körningsjämförelsen visar skillnader i inställningar, objektantal och träfflistor, och varje GUI-körning skapar motsvarande Python-skript. Mätningar exporteras som ``.h5ad`` för scanpy; OME-Zarr och OMERO är tillgängliga via Python-API:t. Exportören för metoder och resultat skriver utkast till dessa två manusavsnitt från en strukturerad sammanfattning av körningen: modellen skriver texten, men varje tal kommer från sammanfattningen, och utkast med tal som saknas där avvisas. Om installationen har problem rapporterar ``spacr-doctor`` vilken spaCR som faktiskt körs, om GPU:n kan användas, om Cellpose motsvarar det API som spaCR anropar samt om projektdatabasen och inställningarna är giltiga, med en kopierbar lösning för varje kontroll som inte godkänns.

Flerspråkigt skrivbordsgränssnitt
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**spaCR → Inställningar → Språk** översätter det aktiva programmet till engelska, svenska, tyska, spanska, mandarin-kinesiska, portugisiska, hindi, koreanska, isländska eller franska utan omstart. Valet sparas och moduler som öppnas senare använder det.

Navigering, Inställningar, AI- och LIVE-kontroller, modulbeskrivningar och konsolmeddelanden från spaCR följer det valda språket. Utdata från arbetsprocesser, loggar, spårutskrifter, sökvägar, databasvärden, annoteringar, AI-svar, mätningar och sparade resultat översätts aldrig, så vetenskapliga utdata behåller sin kanoniska engelska form. Inställningstips som ännu inte har granskats på ett språk förblir på engelska i stället för att bli en blandspråkig förklaring. `Lokaliseringsguiden <https://einarolafsson.github.io/spacr/localization.html>`_ beskriver beteendet, miljövariabeln som styr språket och den `sammanhangsberoende hjälpen <https://einarolafsson.github.io/spacr/localization.html#contextual-help>`_ som översätts tillsammans med gränssnittet.

Animerad hjälp för inställningar
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

94 korta animationer visar hur 143 visuella inställningar påverkar en bild. Håll pekaren över en inställning och klicka på **Animation** i hjälptexten för att spela upp den fyrkantiga förhandsvisningen bredvid texten; klicka igen för att fälla ihop den. Animationer spelas bara på begäran och kan stängas av helt i Inställningar. `Galleriet <https://einarolafsson.github.io/spacr/setting_animations.html>`_ visar dem alla och `registret över inställningsanimationer <https://einarolafsson.github.io/spacr/api/spacr/setting_animations/index.html>`_ anger vilken inställning varje animation hör till.

Modulreferens
~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 25 25 25

   * - Modul
     - Funktion
     - Status
     - Beskrivning
   * - **Skrivbordsupplevelse**
     -
     -
     -
   * - |api-qt-app|_
     - |doc-i18n|_
     - Stabil
     - Översätter öppna och behovsskapade vyer direkt mellan tio medföljande språk.
   * - |api-qt-app|_
     - |doc-i18n-help|_
     - Stabil
     - Lokaliserar modulsammanfattningar och inställningshjälpens gränssnitt utan att ändra API-adresser.
   * - |api-qt-ai|_
     - |api-qt-ai-console|_
     - Stabil
     - Lokaliserar AI- och LIVE-kontroller utan att ändra användarens eller modellens innehåll.
   * - |api-animations|_
     - |doc-animations|_
     - Stabil
     - Spelar 94 medföljande animationer för 143 visuella inställningar från inställningens hjälptext.
   * - |api-selection|_
     - |api-linked-views|_
     - Alfa
     - Delar ett objekturval mellan tabell-, platt-, inbäddnings-, spridnings- och grafvyer.
   * - |api-doctor|_
     - |api-doctor-checks|_
     - Alfa
     - Kontrollerar GPU, Cellpose-API, databas och inställningar och ger en lösning för varje misslyckad kontroll.
   * - **Bildanalys**
     -
     -
     -
   * - |api-mask|_
     - |api-mask-2d|_
     - Stabil
     - Segmenterar celler, cellkärnor, patogener och organeller i 2D-bilder.
   * - |api-mask|_
     - |api-mask-3d|_
     - Beta
     - Segmenterar volymbilder och 4D-tidsserier.
   * - |api-illumination|_
     - |api-flatfield|_
     - Alfa
     - Uppskattar flatfältet från plattan och korrigerar det innan intensiteten mäts.
   * - |api-measure|_
     - |api-measure-2d|_
     - Stabil
     - Mäter morfologi, intensitet, textur och kolokalisering och sparar bildutsnitten.
   * - |api-segqc|_
     - |api-segqc-verdict|_
     - Alfa
     - Beskriver segmenteringskvaliteten innan Measure körs utan att blockera körningen.
   * - |api-timelapse|_
     - |api-tracking|_
     - Beta
     - Spårar objekt med IoU, Trackpy, btrack, Trackastra eller ultrack och kvantifierar rörligheten.
   * - |api-layers|_
     - |api-layer-viewer|_
     - Alfa
     - Staplar bild-, etikett-, punkt- och formlager med ortogonala vyer och ett jämförelserutnät.
   * - |api-napari|_
     - |api-napari-curation|_
     - Alfa
     - Skickar en mask till napari för korrigering, tar tillbaka den och registrerar varje ändring.
   * - **AI och fenotypning**
     -
     -
     -
   * - |api-annotate|_
     - |api-annotation|_
     - Stabil
     - Granskar bildutsnitt i ett tangentbordsstyrt rutnät och sparar annoteringar i SQLite.
   * - |api-active-learning|_
     - |api-al-loop|_
     - Alfa
     - Tränar om modellen i Annotate, rangordnar efter osäkerhet och anger när märkningen kan avslutas.
   * - |api-classify|_
     - |api-classification|_
     - Stabil
     - Tränar och använder CNN- och transformermodeller i PyTorch.
   * - |api-classify|_
     - |api-model-cards|_
     - Alfa
     - Registrerar datamängd, klassbalans, uppdelningsregel och utvärderingsmått vid varje kontrollpunkt.
   * - |api-confusion|_
     - |api-confusion-drill|_
     - Alfa
     - Öppnar bildutsnitten bakom en cell i förväxlingsmatrisen och skiljer säkra fel från osäkra fall.
   * - |api-ml|_
     - |api-ml-models|_
     - Stabil
     - Tränar tolkbara klassiska modeller och boostningsmodeller på mättabeller.
   * - |api-classify|_
     - |api-activation|_
     - Beta
     - Förklarar prediktioner med Captum, SmoothGrad och TorchCAM.
   * - |api-umap|_
     - |api-embedding|_
     - Beta
     - Utforskar bildinbäddningar interaktivt och sprider klusteretiketter.
   * - **Sekvensering och screeninganalys**
     -
     -
     -
   * - |api-sequencing|_
     - |api-barcodes|_
     - Stabil
     - Mappar rad-, kolumn- och gRNA-streckkoder från FASTQ-läsningar och tilldelar guider till avbildade celler.
   * - |api-barcode-qc|_
     - |api-barcode-qc-sweep|_
     - Alfa
     - Rapporterar läsningar per brunn, kollisionsfrekvens och omappad andel utifrån förväntat antal gRNA per brunn.
   * - |api-regression|_
     - |api-regression-models|_
     - Stabil
     - Skattar guide-, gen-, betingelse- och kontrolleffekter med 17 modellfamiljer.
   * - |api-power|_
     - |api-power-design|_
     - Alfa
     - Beräknar hur många celler och brunnar en screening kräver med hänsyn till sekvenseringsfel och brunnsbortfall.
   * - |api-graph|_
     - |api-graph-builder|_
     - Alfa
     - Bygger ett diagram genom att dra kolumner till x, y, färg, storlek och facett.
   * - |api-artifacts|_
     - |api-provenance|_
     - Alfa
     - Registrerar körnings-ID, startvärde och inställningar bakom utdata från Mask, Measure, Classify och export.

.. |api-qt-app| replace:: **Qt-program**
.. _api-qt-app: https://einarolafsson.github.io/spacr/api/spacr/qt/app/index.html

.. |doc-i18n| replace:: **Lokalisering på tio språk**
.. _doc-i18n: https://einarolafsson.github.io/spacr/localization.html

.. |doc-i18n-help| replace:: **Lokaliserad sammanhangshjälp**
.. _doc-i18n-help: https://einarolafsson.github.io/spacr/localization.html#contextual-help

.. |api-qt-ai| replace:: **Qt AI**
.. _api-qt-ai: https://einarolafsson.github.io/spacr/api/spacr/qt/ai/index.html

.. |api-qt-ai-console| replace:: **AI-assisterad konsol**
.. _api-qt-ai-console: https://einarolafsson.github.io/spacr/api/spacr/qt/ai/index.html

.. |api-animations| replace:: **Register över inställningsanimationer**
.. _api-animations: https://einarolafsson.github.io/spacr/api/spacr/setting_animations/index.html

.. |doc-animations| replace:: **Animationer för visuella inställningar**
.. _doc-animations: https://einarolafsson.github.io/spacr/setting_animations.html

.. |api-selection| replace:: **Urval**
.. _api-selection: https://einarolafsson.github.io/spacr/api/spacr/selection/index.html

.. |api-linked-views| replace:: **Länkat urval**
.. _api-linked-views: https://einarolafsson.github.io/spacr/api/spacr/qt/linked_selection/index.html

.. |api-doctor| replace:: **Doctor**
.. _api-doctor: https://einarolafsson.github.io/spacr/api/spacr/doctor/index.html

.. |api-doctor-checks| replace:: **Installationsdiagnos**
.. _api-doctor-checks: https://einarolafsson.github.io/spacr/api/spacr/doctor/index.html

.. |api-mask| replace:: **Mask**
.. _api-mask: https://einarolafsson.github.io/spacr/api/spacr/core/index.html

.. |api-mask-2d| replace:: **Generering av 2D-masker**
.. _api-mask-2d: https://einarolafsson.github.io/spacr/api/spacr/core/index.html#spacr.core.preprocess_generate_masks

.. |api-mask-3d| replace:: **Generering av 3D- och 4D-masker**
.. _api-mask-3d: https://einarolafsson.github.io/spacr/api/spacr/core/index.html#spacr.core.preprocess_generate_masks

.. |api-illumination| replace:: **Belysning**
.. _api-illumination: https://einarolafsson.github.io/spacr/api/spacr/illumination/index.html

.. |api-flatfield| replace:: **Flatfältskorrigering**
.. _api-flatfield: https://einarolafsson.github.io/spacr/api/spacr/illumination/index.html

.. |api-measure| replace:: **Measure**
.. _api-measure: https://einarolafsson.github.io/spacr/api/spacr/measure/index.html

.. |api-measure-2d| replace:: **Objektmätningar**
.. _api-measure-2d: https://einarolafsson.github.io/spacr/api/spacr/measure/index.html#spacr.measure.measure_crop

.. |api-segqc| replace:: **Kvalitetskontroll av segmentering**
.. _api-segqc: https://einarolafsson.github.io/spacr/api/spacr/seg_qc/index.html

.. |api-segqc-verdict| replace:: **Bedömning före körning**
.. _api-segqc-verdict: https://einarolafsson.github.io/spacr/api/spacr/seg_qc/index.html

.. |api-timelapse| replace:: **Timelapse**
.. _api-timelapse: https://einarolafsson.github.io/spacr/api/spacr/timelapse/index.html

.. |api-tracking| replace:: **Objektspårning**
.. _api-tracking: https://einarolafsson.github.io/spacr/api/spacr/timelapse/index.html

.. |api-layers| replace:: **Lager**
.. _api-layers: https://einarolafsson.github.io/spacr/api/spacr/layers/index.html

.. |api-layer-viewer| replace:: **Lagervisare**
.. _api-layer-viewer: https://einarolafsson.github.io/spacr/api/spacr/qt/layer_viewer/index.html

.. |api-napari| replace:: **napari-brygga**
.. _api-napari: https://einarolafsson.github.io/spacr/api/spacr/napari_bridge/index.html

.. |api-napari-curation| replace:: **Maskkorrigering**
.. _api-napari-curation: https://einarolafsson.github.io/spacr/api/spacr/napari_bridge/index.html

.. |api-annotate| replace:: **Annotate**
.. _api-annotate: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/annotate/index.html

.. |api-annotation| replace:: **Manuell annotering**
.. _api-annotation: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/annotate/index.html

.. |api-active-learning| replace:: **Aktiv inlärning**
.. _api-active-learning: https://einarolafsson.github.io/spacr/api/spacr/active_learning/index.html

.. |api-al-loop| replace:: **Träna om och rangordna**
.. _api-al-loop: https://einarolafsson.github.io/spacr/api/spacr/active_learning/index.html

.. |api-classify| replace:: **Classify**
.. _api-classify: https://einarolafsson.github.io/spacr/api/spacr/deep_spacr/index.html

.. |api-classification| replace:: **Bildklassificering**
.. _api-classification: https://einarolafsson.github.io/spacr/api/spacr/deep_spacr/index.html

.. |api-model-cards| replace:: **Modellkort**
.. _api-model-cards: https://einarolafsson.github.io/spacr/api/spacr/deep_spacr/index.html

.. |api-activation| replace:: **Aktiveringskartor**
.. _api-activation: https://einarolafsson.github.io/spacr/api/spacr/deep_spacr/index.html

.. |api-confusion| replace:: **Confusion**
.. _api-confusion: https://einarolafsson.github.io/spacr/api/spacr/confusion/index.html

.. |api-confusion-drill| replace:: **Detaljgranskning av förväxlingsmatris**
.. _api-confusion-drill: https://einarolafsson.github.io/spacr/api/spacr/confusion/index.html

.. |api-ml| replace:: **Maskininlärning**
.. _api-ml: https://einarolafsson.github.io/spacr/api/spacr/ml/index.html

.. |api-ml-models| replace:: **Klassificering av mätningar**
.. _api-ml-models: https://einarolafsson.github.io/spacr/api/spacr/ml/index.html

.. |api-umap| replace:: **Image UMAP**
.. _api-umap: https://einarolafsson.github.io/spacr/api/spacr/app_umap/index.html

.. |api-embedding| replace:: **Interaktiv inbäddning**
.. _api-embedding: https://einarolafsson.github.io/spacr/api/spacr/app_umap/index.html

.. |api-sequencing| replace:: **Sekvensering**
.. _api-sequencing: https://einarolafsson.github.io/spacr/api/spacr/sequencing/index.html

.. |api-barcodes| replace:: **Mappa streckkoder**
.. _api-barcodes: https://einarolafsson.github.io/spacr/api/spacr/sequencing/index.html

.. |api-barcode-qc| replace:: **Kvalitetskontroll av streckkoder**
.. _api-barcode-qc: https://einarolafsson.github.io/spacr/api/spacr/sequencing_qc/index.html

.. |api-barcode-qc-sweep| replace:: **Brunns- och kollisionsrapport**
.. _api-barcode-qc-sweep: https://einarolafsson.github.io/spacr/api/spacr/sequencing_qc/index.html

.. |api-regression| replace:: **Regression**
.. _api-regression: https://einarolafsson.github.io/spacr/api/spacr/ml/index.html

.. |api-regression-models| replace:: **Skattning av screeningeffekter**
.. _api-regression-models: https://einarolafsson.github.io/spacr/api/spacr/ml/index.html

.. |api-power| replace:: **Power**
.. _api-power: https://einarolafsson.github.io/spacr/api/spacr/power_model/index.html

.. |api-power-design| replace:: **Statistisk styrka och design**
.. _api-power-design: https://einarolafsson.github.io/spacr/api/spacr/power_simulate/index.html

.. |api-graph| replace:: **Graph**
.. _api-graph: https://einarolafsson.github.io/spacr/api/spacr/qt/widgets/graph_spec/index.html

.. |api-graph-builder| replace:: **Graph Builder**
.. _api-graph-builder: https://einarolafsson.github.io/spacr/api/spacr/qt/widgets/graph_builder/index.html

.. |api-artifacts| replace:: **Artefakter**
.. _api-artifacts: https://einarolafsson.github.io/spacr/api/spacr/artifacts/index.html

.. |api-provenance| replace:: **Körningsproveniens**
.. _api-provenance: https://einarolafsson.github.io/spacr/api/spacr/runctx/index.html


Data
----

Referensdatauppsättningar
~~~~~~~~~~~~~~~~~~~~~~~~~

- `Fullständig mikroskopidatauppsättning: BioStudies S-BIAD2135 <https://doi.org/10.6019/S-BIAD2135>`_
- `Testdatauppsättning: Hugging Face toxo_mito <https://huggingface.co/datasets/einarolafsson/toxo_mito>`_
- `Sekvensdata: NCBI BioProject PRJNA1261935 <https://www.ncbi.nlm.nih.gov/bioproject/?term=PRJNA1261935>`_
- `Statistisk styrkeanalys: spaCRPower <https://github.com/maomlab/spaCRPower>`_


Bidrag och support
------------------------

Felrapporter och konkreta funktionsförslag är välkomna via `GitHub Issues <https://github.com/EinarOlafsson/spacr/issues>`_. Ange spaCR-version, operativsystem, Python-version, modulinställningar och relevant loggutdrag när du rapporterar ett fel. ``spacr-doctor`` samlar in det mesta av denna information automatiskt.

Licens
~~~~~~~~~

Källkoden för den aktuella utvecklingsgrenen är tillgänglig under `PolyForm Noncommercial License 1.0.0 <https://github.com/EinarOlafsson/spacr/blob/main/LICENSE>`_. Kommersiell användning kräver en separat licens från upphovsrättsinnehavaren. Publicerade versioner till och med spaCR 1.4.9.9 förblir tillgängliga under den MIT-licens som följde med respektive version.

Handledningar
~~~~~~~~~~~~~

Det `interaktiva biblioteket med spaCR-handledningar <https://einarolafsson.github.io/spacr/tutorials/>`_ innehåller berättarröst och undertexter för installationen och varje arbetsflöde i programmet, på åtta språk.

Citera spaCR
~~~~~~~~~~~~

Om spaCR bidrar till din forskning, citera:

Olafsson EB, *et al.* En poolad, bildbaserad CRISPR-screening identifierar EAF1 som en modulator av ESCRT-subversion i *T. gondii*.

`BioRxiv preprint <https://www.biorxiv.org/content/10.64898/2026.07.08.737057v1>`_ · `programvaruarkiv <https://doi.org/10.5281/zenodo.21343317>`_
