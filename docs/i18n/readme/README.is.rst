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

Tungumál: `English <../../../README.rst>`_ · `Svenska <README.sv.rst>`_ ·
`Deutsch <README.de.rst>`_ ·
`Español <README.es.rst>`_ ·
`简体中文 <README.zh_CN.rst>`_ ·
`Português <README.pt.rst>`_ ·
`हिन्दी <README.hi.rst>`_ ·
`한국어 <README.ko.rst>`_ ·
`Íslenska <README.is.rst>`_ ·
`Français <README.fr.rst>`_

`Upplýsingar um þýðingarlíkön <../TRANSLATION_MODELS.md>`_

**Rýmisbundin svipgerðargreining á CRISPR-skimunum.**

spaCR aðgreinir og mælir stakar frumur í afkastamiklum smásjármyndum, tengir hverja frumu við gRNA-ið sem hún fékk og greinir frá því hvaða gen breyttu svipgerðinni. Plötumyndir og FASTQ-raðir eru inntak; mælingar fyrir hvert viðfang, þjálfaðir flokkarar, áhrifastærðir fyrir hverja leiðarsameind og hvert gen og forgangsraðaður niðurstöðulisti eru úttak.

Fyrir myndgreindar samsettar CRISPR-skimanir nær þetta yfir allt verkflæðið. Ef þú ert með afkastamiklar smásjármyndir en enga skimun er hægt að keyra aðgreiningu, mælingar, merkingar og flokkun sjálfstætt.

Myndir, grímur, myndúrklippur, mælingar, merkingar, spár, strikamerki og brunnaauðkenni eru geymd í einu SQLite-verkefni, þannig að rekja má niðurstöðugildi aftur til viðfangsins sem það kom frá.

Keyrðu spaCR sem skjáborðsforrit eða án grafísks viðmóts á vinnustöð, þjóni eða reikniklasa. Báðar leiðir nota sömu einingar og CUDA er virkjað sjálfkrafa þegar einingin styður það.


Yfirlit yfir verkflæðið
-----------------------

|Tutorials|

.. image:: https://raw.githubusercontent.com/EinarOlafsson/spacr/main/spacr/resources/icons/flow_chart_v3.png
   :alt: spaCR workflow and output organization
   :align: center

Smásjármyndir (TIFF, OME-TIFF, LIF, CZI, ND2) og raðgreiningarlestur (FASTQ) fara í samverkandi ferli fyrir myndgreiningu og strikamerkjavörpun. Síðan eru viðfangstöflur, myndúrklippur, merkingar, spár, auðkenni leiðarsameinda, QC-niðurstöður og samantektir fyrir hvern brunn greind saman.


Flýtiræsing
-----------

.. code-block:: bash

   conda create -n spacr python=3.12 -y
   conda activate spacr
   python -m pip install --upgrade pip
   python -m pip install "spacr[qt]"
   spacr

spaCR styður Python **3.9 til 3.14** (aðlum Python 3.14.1, sem er ekki í hliðarvörun). Python 3.12 hefur breiðustu val af óþekkt vísindalegum pakka. Linux er mælt fyrir CUDA vinnuflokk; macOS og Windows eru einnig stuðning.


Upplýsingar um uppsetningu
--------------------------

|Release| |PyPI| |CondaRecipe|

**(beta) Lightweight stökkvöru uppsetur:**

.. spacr-installer-links-begin

* `Windows 10/11: Láttu niður SpaCR 1.5.0.4 <https://github.com/EinarOlafsson/spacr/releases/download/v1.5.0.4/SpaCR-1.5.0.4-Windows-Online-Setup.exe>`_
* `macOS 11+ (Intel og Apple silicon): hættu niður SpaCR 1.5.0.4 <https://github.com/EinarOlafsson/spacr/releases/download/v1.5.0.4/SpaCR-1.5.0.4-macOS-Universal-Online.pkg>`_
* `64-bit Linux: niður SpaCR 1.5.0.4 <https://github.com/EinarOlafsson/spacr/releases/download/v1.5.0.4/SpaCR-1.5.0.4-Linux-x86_64-Online.run>`_

.. spacr-installer-links-end

Létt uppsetningarforrit — hvorki conda né fyrirliggjandi Python þarf
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The installer downloads a private Python 3.12 runtime, Qt, PyTorch, spaCR and the scientific dependencies during installation, so neither conda nor an existing Python is needed. The portable CPU build is the default, which keeps the installation from pulling several gigabytes of CUDA libraries unannounced. Windows offers NVIDIA acceleration as an optional installer component, Linux accepts ``--torch-backend auto``, and the standard macOS PyTorch wheel keeps Apple MPS acceleration.

Hjálp, framtíð og mistök fylgja vinnslu-sjónalinu í öllum tíu spaCR tungumálum: Engelskt, Svíþjóðlegt, Þýskalandi, Spánskt, Einfalt Kínversku, Portúgalska, Indískt, Korean, Íslandi og Franska.

Á Linux, gera niðurstaða uppsetur verkandi áður en þú opnar það:

.. code-block:: bash

   chmod +x SpaCR-*-Linux-x86_64-Online.run
   ./SpaCR-*-Linux-x86_64-Online.run

On macOS, open the downloaded ``.pkg``. If Gatekeeper blocks the current beta installer because it is not notarized, open **System Settings → Privacy & Security**, choose **Open Anyway** for spaCR, then run the package again.

The installer validates spaCR, Qt, PyTorch and dependency consistency before replacing an older installation, so an interrupted update leaves the previous working environment in place. A diagnostic log is kept as ``install.log`` inside the private spaCR installation directory.

Skjáborðsforrit frá PyPI
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   python -m pip install "spacr[qt]"
   spacr

Uppsetning án grafísks viðmóts eða á þjóni
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   python -m pip install spacr
   spacr-run --list

Nýjasta þróunargrein
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   git clone https://github.com/EinarOlafsson/spacr.git
   cd spacr
   git switch nightly
   python -m pip install -e ".[qt]"

Conda-umhverfi
~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   conda create -n spacr python=3.12 pip -y
   conda activate spacr
   python -m pip install "spacr[qt]"

Valfrjálsir eiginleikar
~~~~~~~~~~~~~~~~~~~~~~~

Aðeins setja upp viðbótar sem þörf er á vinnuflu:

.. code-block:: bash

   python -m pip install "spacr[trackastra]"    # transformer tracking
   python -m pip install "spacr[ultrack]"       # global-optimization tracking
   python -m pip install "spacr[btrack]"        # btrack timelapse tracking
   python -m pip install "spacr[attribution]"   # TorchCAM methods
   python -m pip install "spacr[boosting]"      # LightGBM and CatBoost
   python -m pip install "spacr[zernike]"       # Zernike measurements
   python -m pip install "spacr[napari]"        # napari mask correction
   python -m pip install "spacr[czi,nd2,lif]"   # vendor file readers

Hvaða útgáfur leyfa er á Python útgáfu. á Python 3.13, ultrakum grændir ``spacr[all]`` og TorchCAM's NumPy takmarkaði hringja ``attribution`` extra; kjarna pakka og Qt tilboð eru ekki áhrif. á #Python 3.14, btrack er hægt að fá með því extra.

The öryggi Tk gríf er enn settur sem ``spacr-legacy`` en er ekki lengur þróað.


Skipanalínuskipanir
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

Set ``SPACR_LOG_LEVEL=DEBUG`` þegar er að leysa vandamál. Rotað logs eru skrifaðar í ``~/.spacr/logs/spacr.log``.


Eiginleikar
-----------

Einingarnar sex sem flestar skimanir nota
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Mask** segmentar celler, kjólur, patogens og organelles með Cellpose, í 2D myndum og í volymetric eða tímamótum. The lístúr er lesið frá uppsettu Cellpose en ekki hárkod, og hlutdiameter er ákvarðan frá myndum áður en ferlið byrjar. Masker geta verið korrigerðar höndlega í skammsýri, eða send til napari og bak.

**Messur** skrifa á hlutum morfóli, hærs, texta og lokalization eiginleika til verkefnið gögn, ásamt uppgötvun. Ný í 1.5.0.0: ljósskorpsur skulu fram á fátæktan frá plötu sjálfum og skiptir það út áður en hvaða hærssýni er tekið, sem fjarlægir vel-staðar bias sem plötur heittmappur sýna sem viðbót áhrif. A segmentación QC banner segir í ljóst tungumál hvað maskar líður út áður En Measure runn; það ber, það blokkir ekki.

**Annotate** sýnir gröfur á keyboard-driven grít og skrifa merki rétt til SQLite. Það fer nú upp að aktiv læringslúum: endurskoða mönnun á það sem þú hefur merkinn án þess að yfirgefa skrin, endurskera kvikinn með óvissu, horfa á læringskurva og fá stöðva dóm þegar fleiri merki hætta að breyta mönnuð.

**Classify** ferðir PyTorch CNNs og breytingar á notuð vöru, og klassísk eða boost móður á mæslum. Klasa nákvæmni er nú haldið hvert tímabundi en ekki verið fjarlægð, og hver skammtinn fær mönnunarkort sem skrá gögn sín, klasa jafnvægi, split regla og hold-out metrics. Í gildi skrin, blanda-matrix celler er spurning: klikkaðu það til að opna þá vöur, með örugglega rangt fyrirmyndum listuð frá óþekktum.

**Map Barcodes** dekoða rán, kolumn og gRNA barkódum frá FASTQ lesur, veita leiðbeiningar einkenni til bað, og tengir þá í myndu celler. Barkód QC skýringar lesa á bót, stöðu hratt og ómapped fraktion, svingja um fjölda gRNAs á brot þú segir að þú sért í stað þess að ákveðið takmörku.

**Regression** er að meta leiðbeining, gén, tilstand og stjórn áhrif með 17 móður fjölskyldur, þar á meðal blanda móðir, logistic og probit, kvantil, beta, GLMs með kvasi-binomial breyting, lasso, ridge, elasti net, hinga og hest. Resultat er rangt, notuð hit list en ekki samkeppnari dæmning.

Nýtt í 1.5.0.0
~~~~~~~~~~~~~~

Áður en skrin er til, Power / Design móddur svara hversu mörg celler og hversu margir bólum það þarf, verðlaun með sekkunar mistök og með dropp sem kemur frá bólunum sem voru myndin of þynni. Forsíðafræðingur leggur út plátan, stjórn hennar og endurspeglarnar og útgáfa á tengslum fyrir rúmið. Síðan QC tákn samlar segmentation, plátur, annotator-samningur og flokk athuga í einu ákvörðun, og ComBat er hægt að fá við ``center`` og ``zscore`` fyrir uppskrift.

Resultater eru að skoða en út og endur-tækta. Graph Builder plottar tól með því að draga kolumna á x, y, litur, stærð og faðir. Gæði drekka á histogram eða skatter verða filter. Einn gæði Explorer ránir eiginleika af hversu vel þeir skilja klasa. Smá fjölbreytingar, svör viðbrögð, stjórnun grafs og robust útlendis uppgötva notuð sama axis motor. Valgreining objekt í einu sýn valgar þá í öllum þeim, og opnar valkur kemur upp gróðurs sem objektum kom frá. A layer viewers stacks mynd, etiketter, punkts and shapes, with orthogonal views, a synchronized comparison grid, and a lineage from cell to nucleus to pathogen.

Hver hefur einn runn ID, einn semur og ``on_error`` lög; Mask, Measure, Classify og AnnData útgáfregist það sem þeir skrifað í listanum um vörum, þannig að útgáfur skila aftur til settingar sem framleiða það. A mól opnar á því sem fyrri stefnu er raunverulega skrifað, tækjum graf markar sem útgáfum eru stalla, Run samanburði diffs settingar, objekt tölur og hit lista af tveimur runnum, og hver GUI runn útgáfa samræmi Python skript. Mótningar útgáfu til ``.h5ad`` fyrir scanpy; OME-Zarr og OMERO eru til staðar í gegnum Python API. The metódus-results-exporter þessi tvö handskrift frá uppbyggingu móta: hlutinn tækni og hættir notkun fjöldi, en hvert runn er ekki skráður úr skrekinu. Þegar eitthvað er rangt með uppsetningu, ``spacr-doctor`` segir að spaCR er í raun að fara, hvort GPU er notuð, hvortCellpose matar við API spaCR hringjum og hvort verkefnið gögn og settingar eru hljóð, með kopíbarra staðfestingu á hverri rán sem er ekki pass.

Fjöltyngt skjáborðsviðmót
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**spaCR → Stillingar → Tungumál** þýðir virka forritið yfir á ensku, sænsku, þýsku, spænsku, mandarín-kínversku, portúgölsku, hindí, kóresku, íslensku eða frönsku án endurræsingar. Valið varðveitist og einingar sem eru opnaðar síðar nota það.

Leiðsögn, stillingar, gervigreindar- og LIVE-stýringar, einingarlýsingar og tilkynningar frá spaCR fylgja völdu tungumáli. Úttak vinnsluferla, annálar, rakningar, slóðir, gagnagrunnsgildi, merkingar, svör gervigreindar, mælingar og vistaðar niðurstöður eru aldrei þýdd; vísindalegt úttak helst því á viðurkenndu ensku formi. Verkfæraábendingar fyrir stillingar sem hafa ekki verið yfirfarnar á tungumálinu birtast á ensku fremur en í blandaðri þýðingu. `Staðfærsluleiðbeiningarnar <https://einarolafsson.github.io/spacr/localization.html>`_ lýsa hegðuninni, umhverfisbreytunni og þeirri `samhengisháðu hjálp <https://einarolafsson.github.io/spacr/localization.html#contextual-help>`_ sem er þýdd með viðmótinu.

Animated setting leiðbeiningar
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

94 stuttanir lýsa hvað 143 sýnilegar settingar gera á mynd. Hugsaðu settinginn og smellaðu **Animation** í tólstípi þess til að spila miðjan við texta; smellað aftur til að fylla það fjarlægð. Animaðir eru off þar til sem þú ert að spyrja, og hægt er að deaktivera í Forritum. `Gallerið <https://einarolafsson.github.io/spacr/setting_animations.html>`_ sýnir allar þeirra, og `Að setja upp áætlunarregistur <https://einarolafsson.github.io/spacr/api/spacr/setting_animations/index.html>`_ skrá sem setting hverni er til.

Tilvísun eininga
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


Gögn
----

Viðmiðunargagnasöfn
~~~~~~~~~~~~~~~~~~~

- `Full mikroskópur gögn: BioStudies S-BIAD2135 <https://doi.org/10.6019/S-BIAD2135>`_
- `Testið gögn: Hugging Face toxo_mito <https://huggingface.co/datasets/einarolafsson/toxo_mito>`_
- `Sækkun gögnum: NCBI BioProject PRJNA1261935 <https://www.ncbi.nlm.nih.gov/bioproject/?term=PRJNA1261935>`_
- `Power analysis: spaCRPower <https://github.com/maomlab/spaCRPower>`_


Framlög og aðstoð
------------------------

Bragar og hagsmyndaða greinar eru velkomnir í gegnum `GitHub Vörur <https://github.com/EinarOlafsson/spacr/issues>`_. Þegar við að ræða skammt, er það spaCR útgáfa, leitarhólksins, Python útskýrunar, módúlarsetningar og viðeigandi log útskýrt. ``spacr-doctor`` samlar mest af því fyrir þig.

Leyfi
~~~~~~~~~

Núverandi þróun rán er aðgangs-tilbúinn undir `PolyForm óþekktar leyfi 1.0.0 <https://github.com/EinarOlafsson/spacr/blob/main/LICENSE>`_. Kaupleg notkun krefst sérstakt leyfi frá höfundum. útgáfur sem eru útgáfað með spaCR 1.4.9.9 eru enn til staðar undir MIT leyfi sem fylgdi þessum útgáfum.

Kennsluefni
~~~~~~~~~~~

`Tólfáætlun spaCR kennslabúð <https://einarolafsson.github.io/spacr/tutorials/>`_ inniheldur uppsett og skýrðir ferðar á hönnunum og vinnufluðum hverra tilskipun í átta tungumálum.

Tilvísun í spaCR
~~~~~~~~~~~~~~~~

Ef spaCR hjálpar til rannsóknir þíns, heyrðu:

Olafsson EB, *et al.* A sameiginlegur myndbönd sem er bastir á CRISPR skrefinn skilur EAF1 sem *T. gondii* modulator ESCRT subversion.

`Bioregl fyrirframskrift <https://www.biorxiv.org/content/10.64898/2026.07.08.737057v1>`_ · `Programvarparkíf <https://doi.org/10.5281/zenodo.21343317>`_
