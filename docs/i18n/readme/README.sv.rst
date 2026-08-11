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
   :alt: spaCR workflow and output organization
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

spaCR stöder Python **3.9 till 3.14 ** (utom Python 3.14.1, som fackling utesluter). Python 3.12 har det bredaste valet av frivilliga vetenskapliga paket. Linux rekommenderas för CUDA arbetsflöden; macOS och Windows stöds också.


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

Installationsprogrammet laddar ner en privat Python 3.12 körtid, Qt, PyTorch, spaCR och de vetenskapliga beroendena under installationen, så varken conda eller en befintlig Python behövs. Den bärbara CPU bygget är standard, vilket hindrar installationen från att dra flera gigabyte av CUDA bibliotek oanmäld. Windows erbjuder NVIDIA acceleration som en valfri installationskomponent, Linux accepterar ``--torch-backend auto``, och standard macOS PyTorch hjulet håller Apple MPS acceleration.

Installer hjälp, framsteg och fel följer operativsystemet språk på alla tio spaCR språk: engelska, svenska, tyska, spanska, förenklad kinesiska, portugisiska, hindi, koreanska, isländska och franska. Ostött lokala faller tillbaka till engelska.

På Linux, gör det nedladdade installationsprogrammet körbart innan det öppnas:

.. code-block:: bash

   chmod +x SpaCR-*-Linux-x86_64-Online.run
   ./SpaCR-*-Linux-x86_64-Online.run

På macOS, öppna den nedladdade ``.pkg``. Om Gatekeeper blockerar den nuvarande beta-installer eftersom det inte är notarized, öppna **Systeminställningar → Sekretess och säkerhet**, välj **Open Anyway** för spaCR, sedan köra paketet igen.

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

Installera bara extras ditt arbetsflöde behöver:

.. code-block:: bash

   python -m pip install "spacr[trackastra]"    # transformer tracking
   python -m pip install "spacr[ultrack]"       # global-optimization tracking
   python -m pip install "spacr[btrack]"        # btrack timelapse tracking
   python -m pip install "spacr[attribution]"   # TorchCAM methods
   python -m pip install "spacr[boosting]"      # LightGBM and CatBoost
   python -m pip install "spacr[zernike]"       # Zernike measurements
   python -m pip install "spacr[napari]"        # napari mask correction
   python -m pip install "spacr[czi,nd2,lif]"   # vendor file readers

Vilka extras upplösning beror på Python versionen. På Python 3.13, ultrack limits ``spacr[all]`` och TorchCAMs NumPy begränsning begränsar ``attribution`` extra; kärnpaketet och Qt programmet påverkas inte. På Python 3.14 är btrack tillgängligt genom sitt extra. PylibCZIrw CZI-omvandlaren är valfri och oprövad; czifile-baserad CZI-avläsning finns fortfarande tillgänglig.

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

**Mask** segmentceller, nuclei, patogener och organeller med Cellpose, i 2D-avbildningar och i volymetrisk data eller tidsseriedata. Modelllistan läses från den installerade Cellpose istället för hårdkodad, och en objektdiameter uppskattas från bilderna innan körningen startar. Masker kan korrigeras för hand i lagervisningen, eller skickas till napari och tillbaka.

**Mäta** skriver per-objekt morfologi, intensitet, textur och colocalization funktioner till projektdatabasen, tillsammans med grödorna. Ny i 1.5.0.0: belysning korrigering uppskattar plana fältet från plattan själv och delar ut det innan någon intensitet funktion tas, som tar bort den välposition fördomar som plattan värmekartor visar som kanteffekter. En segmentering QC banner anger i klarspråk hur maskerna ser ut innan Mät körs; den informerar, det blockerar inte. En dragen polygon begränsar mätning till en region av intresse.

**Annotate** visar bildutsnitt på ett tangentbordsdrivet rutnät och skriver etiketter direkt till SQLite. Nu stänger den aktiva lärloopen: omskola en modell på vad du har märkt utan att lämna skärmen, ställ om kön genom osäkerhet, titta på inlärningskurvan och få ett stopp på domen när ytterligare etiketter slutar att ändra modellen. Täckning rapporteras per klass, per brunn och per platta, och varje runda registreras.

**Klassificering** tåg PyTorch CNN och transformatorer på kommenterade bildutsnitt, och klassiska eller boosted modeller på mättabeller. Per-klass noggrannhet hålls nu varje epok i stället för att kastas, och varje kontrollpunkt får ett modellkort registrera sin datauppsättning, klassbalans, delad regel och hållna-ut mätvärden. I utvärderingsskärmen, en förvirring-matris cell är en fråga: klicka på den för att öppna dessa bildutsnitt, med säkert felaktiga förutsägelser listas bortsett från osäkra.

**Karta Streckkoder** avkodar rad, kolumn och gRNA streckkoder från FASTQ läser, tilldelar guide identiteter till brunnar, och ansluter dem till avbildade celler. Streckkod QC rapporter läser per brunn, kollisionsfrekvens och oöverträffad fraktion, sveper runt antalet gRNAs per brunn du säger att du förväntar dig snarare än en fast tröskel.

**Regression** skattningar guide, gen, tillstånd och kontroll effekter med 17 modeller familjer, inklusive blandade modeller, logistiska och probit, quantile, beta, GLMs med kvasi-binomial varians, lasso, rås, elastiskt nät, gångjärn och hästsko. Resultatet är en rankad, kommenterad hit lista snarare än en koefficient dumpa.

Nytt i 1.5.0.0
~~~~~~~~~~~~~~

Innan en screening finns, svarar Power / Design modulen hur många celler och hur många brunnar den behöver, prissatt med sekvensering fel och med avhopp som kommer från brunnar som avbildades för tunt. En experiment designer lägger ut plattan, dess kontroller och dess replikat och exporterar layouten för rörledningen. Efteråt, en QC instrumentpanel samlar segmentering, platta, notator-avtal och läckage kontroller i en dom, och ComBat finns bredvid ``center`` och ``zscore`` för batch korrigering.

Resultaten utforskas snarare än exporteras och återimporteras. En grafbyggare ritar en tabell genom att dra kolumner till x, y, färg, storlek och fasett. Gates ritade på ett histogram eller en scatter blir filter. En funktionsutforskare rankas med hur väl de skiljer klasserna. Små multiplar, dosrespons passar, kontrolldiagram och robust avvikelsedetektering använder samma axelmotor. Väljer objekt i en vy väljer dem i alla, och öppna ett urval tar upp grödorna som objekten kom från. En lagertittar bilder, etiketter, punkter och former, med ortogonala vyer, en synkroniserad jämförelse rutnät, och en linjeträd från cell till kärna till patogen.

Körer är nu identifierbara. Varje kör-ID, ett frö och en ``on_error`` policy; Mask, Mät, Klassifiera och AnnData exportregister vad de skrev i ett artefaktregister, så en utdatafil leder tillbaka till inställningarna som producerade det. En modul öppnar på vad det föregående steget faktiskt skrev, rörledningsgraf markerar vilka utdata är gamla, kör jämförelse diffs inställningar, objekt räknas och träffar listor av två körningar, och varje GUI kör släpper motsvarande Python skript. Mätningar exporterar till ``.h5ad`` för skanning; OME-Zarr och OMERO är tillgängliga via Python API. Metoderna-och-resultat exportören drar dessa två manuskript avsnitt från en strukturerad smältning av körningen: modellen skriver prosen, men varje nummer kommer från smältningen, och ett utkast som innehåller ett antal smältningen inte innehåller avvisas. När något är fel med installationen, rapporterar ``spacr-doctor`` som spaCR faktiskt körs, om GPU är användbart, om Cellpose matchar samtalen API spaCR och om projektdatabasen och inställningarna är ljud, med en kopieringsbar fix på varje rad som inte är ett pass.

Flerspråkigt skrivbordsgränssnitt
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**spaCR → Inställningar → Språk** översätter det aktiva programmet till engelska, svenska, tyska, spanska, mandarin-kinesiska, portugisiska, hindi, koreanska, isländska eller franska utan omstart. Valet sparas och moduler som öppnas senare använder det.

Navigation, Preferences, AI and LIVE controls, module descriptions and spaCR-authored console notices follow the selected language. Worker output, logs, tracebacks, paths, database values, annotations, AI responses, measurements and saved results are never translated, so scientific output remains canonical English. Setting tooltips not yet reviewed in a language stay in English rather than becoming a mixed-language explanation. The `Lokaliseringsguide <https://einarolafsson.github.io/spacr/localization.html>`_ documents the behavior, the environment override, and the `Sammanhangsmässig hjälp <https://einarolafsson.github.io/spacr/localization.html#contextual-help>`_ that is translated with it.

Animerad inställningsvägledning
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

94 short animations explain what 143 visual settings do to an image. Hover a setting and click **Animation** in its tooltip to play the square beside the text; click it again to fold it away. Animations are off until asked for, and can be disabled in Preferences. The `galleri <https://einarolafsson.github.io/spacr/setting_animations.html>`_ shows all of them, and the `Ställa in animeringsregistret <https://einarolafsson.github.io/spacr/api/spacr/setting_animations/index.html>`_ records which setting each one belongs to.

Modulreferens
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


Data
----

Referensdatauppsättningar
~~~~~~~~~~~~~~~~~~~~~~~~~

- `Fullständiga mikroskopidatauppsättning: Biostudier S-BIAD2135 <https://doi.org/10.6019/S-BIAD2135>`_
- `Testdatauppsättning: Huggande ansikte toxo_mito <https://huggingface.co/datasets/einarolafsson/toxo_mito>`_
- `Sekvensdata: NCBI BioProject PRJNA1261935 <https://www.ncbi.nlm.nih.gov/bioproject/?term=PRJNA1261935>`_
- `Effektanalys: spaCRPower <https://github.com/maomlab/spaCRPower>`_


Bidrag och support
------------------------

Bug reports and focused feature requests are welcome through `GitHub Frågor <https://github.com/EinarOlafsson/spacr/issues>`_. When reporting a failure, include the spaCR version, operating system, Python version, module settings and the relevant log excerpt. ``spacr-doctor`` collects most of that for you.

Licens
~~~~~~~~~

The current development branch is source-available under the `PolyForm Icke-kommersiell licens 1.0.0 <https://github.com/EinarOlafsson/spacr/blob/main/LICENSE>`_. Commercial use requires a separate license from the copyright holder. Released versions through spaCR 1.4.9.9 remain available under the MIT License that accompanied those releases.

Handledningar
~~~~~~~~~~~~~

The `interaktivt spaCR handledningsbibliotek <https://einarolafsson.github.io/spacr/tutorials/>`_ contains narrated, captioned walkthroughs of installation and of each application workflow, in eight languages.

Citera spaCR
~~~~~~~~~~~~

Om spaCR bidrar till din forskning, citera:

Olafsson EB, *et al.* En poolad bildbaserad CRISPR screening identifierar EAF1 som en *T. gondii* modulator för ESCRT subversion.

`BioRxiv preprint <https://www.biorxiv.org/content/10.64898/2026.07.08.737057v1>`_ · `programvaruarkiv <https://doi.org/10.5281/zenodo.21343317>`_
