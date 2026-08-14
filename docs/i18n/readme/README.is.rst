|Docs| |Tutorials| |PyPI| |Python| |Tests| |Qt| |Source| |Issues| |License| |DOI|

.. |Docs| image:: https://github.com/EinarOlafsson/spacr/actions/workflows/pages/pages-build-deployment/badge.svg
   :target: https://einarolafsson.github.io/spacr/
   :alt: Skjöl
.. |Tutorials| image:: https://img.shields.io/badge/Tutorials-Interactive%20walkthrough-4A9EFF
   :target: https://einarolafsson.github.io/spacr/tutorials/
   :alt: Gagnvirkt kennsluefni
.. |PyPI| image:: https://img.shields.io/pypi/v/spacr
   :target: https://pypi.org/project/spacr/
   :alt: PyPI-útgáfa
.. |Python| image:: https://img.shields.io/badge/Python-3.9%E2%80%933.14-3776AB?logo=python&logoColor=white
   :target: https://pypi.org/project/spacr/
   :alt: Python 3.9 til 3.14
.. |Tests| image:: https://github.com/EinarOlafsson/spacr/actions/workflows/tests.yml/badge.svg
   :target: https://github.com/EinarOlafsson/spacr/actions/workflows/tests.yml
   :alt: Prófunarsafn
.. |Qt| image:: https://img.shields.io/badge/GUI-Qt%20%28PySide6%29-41CD52
   :target: https://einarolafsson.github.io/spacr/
   :alt: Qt-viðmót
.. |Source| image:: https://img.shields.io/badge/GitHub-Source-181717?logo=github
   :target: https://github.com/EinarOlafsson/spacr
   :alt: Frumkóði á GitHub
.. |Issues| image:: https://img.shields.io/github/issues/EinarOlafsson/spacr
   :target: https://github.com/EinarOlafsson/spacr/issues
   :alt: GitHub-mál
.. |License| image:: https://img.shields.io/github/license/EinarOlafsson/spacr
   :target: https://github.com/EinarOlafsson/spacr/blob/main/LICENSE
   :alt: PolyForm Noncommercial-leyfi
.. |DOI| image:: https://img.shields.io/badge/DOI-10.5281%2Fzenodo.21343317-blue
   :target: https://doi.org/10.5281/zenodo.21343317
   :alt: Zenodo DOI
.. |Release| image:: https://img.shields.io/github/v/release/EinarOlafsson/spacr?label=Installers
   :target: https://github.com/EinarOlafsson/spacr/releases/latest
   :alt: Nýjustu uppsetningarforrit
.. |CondaRecipe| image:: https://img.shields.io/badge/conda--forge-recipe-44A833?logo=anaconda
   :target: https://github.com/EinarOlafsson/spacr/tree/main/conda-forge/recipe
   :alt: conda-forge-uppskrift

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
   :alt: Verkflæði spaCR og skipulag úttaks
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

spaCR styður Python **3.9 til 3.14** (nema Python 3.14.1, sem torchvision styður ekki). Python 3.12 býður upp á fjölbreyttasta úrval valfrjálsra vísindapakka. Mælt er með Linux fyrir CUDA-verkflæði; macOS og Windows eru einnig studd.


Upplýsingar um uppsetningu
--------------------------

|Release| |PyPI| |CondaRecipe|

**(beta) Létt uppsetningarforrit fyrir skjáborð:**

.. spacr-installer-links-begin

* `Windows 10/11: hala niður SpaCR 1.5.0.4 <https://github.com/EinarOlafsson/spacr/releases/download/v1.5.0.4/SpaCR-1.5.0.4-Windows-Online-Setup.exe>`_
* `macOS 11+ (Intel og Apple silicon): hala niður SpaCR 1.5.0.4 <https://github.com/EinarOlafsson/spacr/releases/download/v1.5.0.4/SpaCR-1.5.0.4-macOS-Universal-Online.pkg>`_
* `64-bita Linux: hala niður SpaCR 1.5.0.4 <https://github.com/EinarOlafsson/spacr/releases/download/v1.5.0.4/SpaCR-1.5.0.4-Linux-x86_64-Online.run>`_

.. spacr-installer-links-end

Létt uppsetningarforrit — hvorki conda né uppsett Python nauðsynlegt
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Uppsetningarforritið sækir eigið Python 3.12-keyrsluumhverfi, Qt, PyTorch, spaCR og vísindalega hjálparpakka meðan á uppsetningu stendur, þannig að hvorki conda né uppsett Python er nauðsynlegt. Færanlega CPU-útgáfan er sjálfgefið val og kemur í veg fyrir að nokkur gígabæti af CUDA-söfnum séu sótt án fyrirvara. Í Windows er NVIDIA-hröðun valfrjáls hluti uppsetningarinnar, Linux tekur við ``--torch-backend auto`` og staðlaði PyTorch-pakkinn fyrir macOS styður áfram Apple MPS-hröðun.

Hjálpartexti, framvinda og villuboð uppsetningarforritsins fylgja tungumáli stýrikerfisins á öllum tíu tungumálum spaCR: ensku, sænsku, þýsku, spænsku, einfaldaðri kínversku, portúgölsku, hindí, kóresku, íslensku og frönsku. Tungumál sem ekki eru studd nota ensku.

Á Linux skaltu gera uppsetningarforritið sem var sótt keyranlegt áður en þú opnar það:

.. code-block:: bash

   chmod +x spaCR-*-Linux-x86_64-Online.run
   ./spaCR-*-Linux-x86_64-Online.run

Á macOS skaltu opna ``.pkg``-skrána sem var sótt. Ef Gatekeeper stöðvar núverandi beta-uppsetningarforrit vegna þess að það hefur ekki verið vottað af Apple skaltu opna **System Settings → Privacy & Security**, velja **Open Anyway** fyrir spaCR og keyra síðan pakkann aftur.

Uppsetningarforritið sannprófar spaCR, Qt, PyTorch og samræmi milli hjálparpakka áður en eldri uppsetningu er skipt út, svo rofin uppfærsla skilur fyrra virka umhverfið eftir óbreytt. Greiningarannáll er vistaður sem ``install.log`` í sérstakri uppsetningarmöppu spaCR.

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

Settu aðeins upp þá viðbótarpakka sem verkflæðið þitt þarfnast:

.. code-block:: bash

   python -m pip install "spacr[trackastra]"    # transformer tracking
   python -m pip install "spacr[ultrack]"       # global-optimization tracking
   python -m pip install "spacr[btrack]"        # btrack timelapse tracking
   python -m pip install "spacr[attribution]"   # TorchCAM methods
   python -m pip install "spacr[boosting]"      # LightGBM and CatBoost
   python -m pip install "spacr[zernike]"       # Zernike measurements
   python -m pip install "spacr[napari]"        # napari mask correction
   python -m pip install "spacr[czi,nd2,lif]"   # vendor file readers

Hvaða viðbótarpakka er hægt að setja upp fer eftir Python-útgáfunni. Í Python 3.13 takmarka háðakröfur ultrack ``spacr[all]`` og NumPy-útgáfukrafa TorchCAM takmarkar ``attribution``-viðbótina; þetta hefur ekki áhrif á kjarnapakkann eða Qt-forritið. Í Python 3.14 er btrack fáanlegt með viðkomandi viðbótarpakka. pylibCZIrw-breytirinn fyrir CZI er valfrjáls og óprófaður; enn er hægt að lesa CZI-skrár með czifile.

Eldra Tk-viðmótið er enn sett upp sem ``spacr-legacy`` en er ekki lengur í þróun.


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

Við bilanagreiningu skaltu stilla ``SPACR_LOG_LEVEL=DEBUG``. Annálaskrár skiptast sjálfkrafa og eru skrifaðar í ``~/.spacr/logs/spacr.log``.


Eiginleikar
-----------

Einingarnar sex sem flestar skimanir nota
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Mask** aðgreinir frumur, kjarna, sýkla og frumulíffæri með Cellpose, bæði í tvívíðum myndum og rúmmáls- eða tímaraðargögnum. Listinn yfir líkön er lesinn úr uppsettu Cellpose í stað þess að vera harðkóðaður, og þvermál viðfanga er metið út frá myndunum áður en keyrslan hefst. Hægt er að leiðrétta grímur handvirkt í lagaskoðaranum eða senda þær til napari og aftur til baka.

**Measure** vistar formfræðilega eiginleika, styrk, áferð og samstaðsetningu fyrir hvert viðfang í gagnagrunn verkefnisins, ásamt myndúrklippunum. Nýtt í 1.5.0.0: lýsingarleiðrétting metur flat-field út frá bakkanum sjálfum og leiðréttir myndirnar með því áður en nokkur styrkeiginleiki er mældur; þannig hverfur skekkja eftir staðsetningu brunna sem kemur fram sem jaðaráhrif í hitakorti bakkans. Borði fyrir gæðamat aðgreiningar lýsir á skýru máli hvernig grímurnar líta út áður en Measure keyrir; hann upplýsir en stöðvar ekki. Teiknaður marghyrningur takmarkar mælingar við áhugasvæði (ROI).

**Annotate** sýnir myndúrklippur í lyklaborðsstýrðu risti og skrifar merkingar beint í SQLite. Einingin lokar nú lykkju virks náms: hægt er að endurþjálfa líkan á því sem þegar hefur verið merkt án þess að yfirgefa skjáinn, endurraða biðröðinni eftir óvissu, fylgjast með námsferlinum og fá niðurstöðu um hvenær fleiri merkingar hætta að breyta líkaninu. Þekja er birt fyrir hvern flokk, brunn og bakka, og hver umferð er skráð.

**Classify** þjálfar PyTorch CNN- og transformer-líkön á merktum myndúrklippum og hefðbundin líkön eða eflingarlíkön á mælingatöflum. Nákvæmni hvers flokks er nú varðveitt fyrir hvert þjálfunartímabil í stað þess að vera fleygt, og hver varðpunktur fær líkanaspjald þar sem gagnasafn, jafnvægi flokka, skiptingarregla og mælikvarðar á fráteknu prófunarsafni eru skráð. Á matsskjánum virkar reitur í ruglingsfylki sem fyrirspurn: smelltu á hann til að opna samsvarandi myndúrklippur; öruggar rangar spár eru aðskildar frá óvissum spám.

**Map Barcodes** afkóðar strikamerki raða, dálka og gRNA úr FASTQ-lestrum, úthlutar brunnum auðkennum stýriraða og tengir þau við myndgreindar frumur. Gæðamat strikamerkja sýnir fjölda lestra á hvern brunn, árekstrartíðni og hlutfall óvarpaðra lestra, og kannar gildi í kringum þann vænta fjölda gRNA í hverjum brunni sem notandinn tilgreinir í stað þess að nota fastan þröskuld.

**Regression** metur áhrif stýriraða, gena, skilyrða og viðmiða með 17 líkanafjölskyldum, þar á meðal blönduðum líkönum, logistic- og probit-líkönum, quantile- og beta-líkönum, GLM-líkönum með quasi-binomial-dreifni, lasso, ridge, elastic net, hinge og horseshoe. Niðurstaðan er raðaður og skýrður listi yfir markverðar niðurstöður, ekki hráskrá yfir stuðla.

Nýtt í 1.5.0.0
~~~~~~~~~~~~~~

Áður en skimun verður til svarar einingin Power / Design því hve margar frumur og hve marga brunna þarf, með hliðsjón af raðgreiningarvillu og brottfalli sem stafar af brunnum sem voru myndaðir of gislega. Tilraunahönnuður raðar bakkanum, viðmiðum hans og endurtekningum og flytur út skipulagið fyrir vinnslulínuna. Að skimun lokinni safnar QC-stjórnborð prófunum á aðgreiningu, bakka, samræmi merkingaraðila og gagnaleka í eina niðurstöðu, og ComBat er tiltækt við hlið ``center`` og ``zscore`` fyrir lotuleiðréttingu.

Niðurstöður eru kannaðar í forritinu í stað þess að flytja þær út og inn aftur. Graph Builder teiknar töflu með því að draga dálka á x-ás, y-ás, lit, stærð og undirreit. Afmörkunarsvæði sem eru teiknuð á stuðlarit eða dreifirit verða að síum. Eiginleikaskoðari raðar eiginleikum eftir því hve vel þeir aðskilja flokkana. Smámyndafylki, aðhvarfslíkön fyrir skammtasvörun, stýririt og traust greining frávika nota sömu ásavél. Ef viðföng eru valin í einni sýn veljast þau í öllum sýnum, og þegar valið er opnað birtast myndúrklippurnar sem viðföngin komu úr. Lagaskoðari staflar myndum, merkimiðum, punktum og formum, með hornréttum sýnum, samstilltu samanburðarristi og ættarté frá frumu til kjarna og sýkils.

Keyrslur eru nú rekjanlegar. Hver keyrsla hefur auðkenni, slembifræ og ``on_error``-stefnu; Mask, Measure, Classify og AnnData-útflutningur skrá úttak sitt í afurðaskrá svo rekja megi úttaksskrá til stillinganna sem bjuggu hana til. Eining opnast með því úttaki sem fyrra skref skrifaði í raun, verkflæðisritið merkir úrelt úttak, samanburður keyrslna sýnir mun á stillingum, fjölda viðfanga og niðurstöðulistum og hver GUI-keyrsla býr til samsvarandi Python-skriftu. Mælingar má flytja út sem ``.h5ad`` fyrir scanpy; OME-Zarr og OMERO eru aðgengileg í gegnum Python-API. Útflytjandi aðferða og niðurstaðna semur þessa tvo handritskafla úr skipulagðri samantekt keyrslunnar: líkanið skrifar textann en sérhver tala kemur úr samantektinni, og drögum sem innihalda tölu sem þar er ekki að finna er hafnað. Ef uppsetningin er gölluð segir ``spacr-doctor`` hvaða spaCR-uppsetning er í raun í notkun, hvort GPU virkar, hvort Cellpose samsvarar API-köllunum og hvort gagnagrunnur og stillingar verkefnisins séu gild; jafnframt fylgir afritanleg lausn hverri misheppnaðri prófun.

Fjöltyngt skjáborðsviðmót
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**spaCR → Stillingar → Tungumál** þýðir virka forritið yfir á ensku, sænsku, þýsku, spænsku, mandarín-kínversku, portúgölsku, hindí, kóresku, íslensku eða frönsku án endurræsingar. Valið varðveitist og einingar sem eru opnaðar síðar nota það.

Leiðsögn, stillingar, gervigreindar- og LIVE-stýringar, einingarlýsingar og tilkynningar frá spaCR fylgja völdu tungumáli. Úttak vinnsluferla, annálar, rakningar, slóðir, gagnagrunnsgildi, merkingar, svör gervigreindar, mælingar og vistaðar niðurstöður eru aldrei þýdd; vísindalegt úttak helst því á viðurkenndu ensku formi. Verkfæraábendingar fyrir stillingar sem hafa ekki verið yfirfarnar á tungumálinu birtast á ensku fremur en í blandaðri þýðingu. `Staðfærsluleiðbeiningarnar <https://einarolafsson.github.io/spacr/localization.html>`_ lýsa hegðuninni, umhverfisbreytunni og þeirri `samhengisháðu hjálp <https://einarolafsson.github.io/spacr/localization.html#contextual-help>`_ sem er þýdd með viðmótinu.

Hreyfimyndaleiðbeiningar fyrir stillingar
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

94 stuttar hreyfimyndir sýna hvernig 143 sjónrænar stillingar hafa áhrif á mynd. Haltu bendlinum yfir stillingu og smelltu á **Hreyfimynd** í verkfæraábendingunni til að spila ferkantaða forskoðunina við hlið textans; smelltu aftur til að fella hana saman. Hreyfimyndir spilast aðeins þegar beðið er um þær og hægt er að slökkva alveg á þeim í Stillingum. `Galleríið <https://einarolafsson.github.io/spacr/setting_animations.html>`_ sýnir þær allar og `skrá yfir hreyfimyndir stillinga <https://einarolafsson.github.io/spacr/api/spacr/setting_animations/index.html>`_ tilgreinir hvaða stillingu hver mynd tilheyrir.

Tilvísun eininga
~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 25 25 25

   * - Eining
     - Eiginleiki
     - Staða
     - Lýsing
   * - **Skjáborðsupplifun**
     -
     -
     -
   * - |api-qt-app|_
     - |doc-i18n|_
     - Stöðugt
     - Endurþýðir opna skjái og skjái sem verða til eftir þörfum samstundis á tíu innbyggðum tungumálum.
   * - |api-qt-app|_
     - |doc-i18n-help|_
     - Stöðugt
     - Staðfærir samantektir eininga og viðmót stillingahjálpar án þess að breyta API-slóðum.
   * - |api-qt-ai|_
     - |api-qt-ai-console|_
     - Stöðugt
     - Staðfærir AI- og LIVE-stýringar án þess að breyta efni notenda eða líkana.
   * - |api-animations|_
     - |doc-animations|_
     - Stöðugt
     - Spilar 94 innbyggðar hreyfimyndir fyrir 143 sjónrænar stillingar úr verkfæraábendingunni.
   * - |api-selection|_
     - |api-linked-views|_
     - Alfa
     - Deilir einu vali viðfanga milli töflu-, bakka-, ívörpunar-, dreifi- og grafmynda.
   * - |api-doctor|_
     - |api-doctor-checks|_
     - Alfa
     - Prófar GPU, Cellpose-API, gagnagrunn og stillingar og gefur lausn fyrir hverja misheppnaða prófun.
   * - **Myndgreining**
     -
     -
     -
   * - |api-mask|_
     - |api-mask-2d|_
     - Stöðugt
     - Aðgreinir frumur, kjarna, sýkla og frumulíffæri í tvívíðum myndum.
   * - |api-mask|_
     - |api-mask-3d|_
     - Beta
     - Aðgreinir rúmmálsmyndir og fjórvíðar tímaraðir.
   * - |api-illumination|_
     - |api-flatfield|_
     - Alfa
     - Metur flata sviðið út frá bakkanum og leiðréttir það áður en styrkur er mældur.
   * - |api-measure|_
     - |api-measure-2d|_
     - Stöðugt
     - Mælir lögun, styrk, áferð og samstaðsetningu og vistar myndúrklippur.
   * - |api-segqc|_
     - |api-segqc-verdict|_
     - Alfa
     - Lýsir gæðum aðgreiningarinnar áður en Measure keyrir án þess að stöðva keyrsluna.
   * - |api-timelapse|_
     - |api-tracking|_
     - Beta
     - Rekur viðföng með IoU, Trackpy, btrack, Trackastra eða ultrack og magnmælir hreyfanleika.
   * - |api-layers|_
     - |api-layer-viewer|_
     - Alfa
     - Staflar mynd-, merkimiða-, punkta- og formalögum með hornréttum sýnum og samanburðarristi.
   * - |api-napari|_
     - |api-napari-curation|_
     - Alfa
     - Sendir grímu til napari til leiðréttingar, tekur hana aftur og skráir hverja breytingu.
   * - **AI og svipgerðargreining**
     -
     -
     -
   * - |api-annotate|_
     - |api-annotation|_
     - Stöðugt
     - Yfirfer myndúrklippur í lyklaborðsstýrðu risti og vistar merkingar í SQLite.
   * - |api-active-learning|_
     - |api-al-loop|_
     - Alfa
     - Endurþjálfar innan Annotate, endurraðar eftir óvissu og segir hvenær hægt er að hætta merkingu.
   * - |api-classify|_
     - |api-classification|_
     - Stöðugt
     - Þjálfar og beitir CNN- og transformer-líkönum í PyTorch.
   * - |api-classify|_
     - |api-model-cards|_
     - Alfa
     - Skráir gagnasafn, jafnvægi flokka, skiptingarreglu og prófunarmælikvarða við hvern varðpunkt.
   * - |api-confusion|_
     - |api-confusion-drill|_
     - Alfa
     - Opnar myndúrklippur að baki reit í ruglingsfylki og aðskilur öruggar villur frá óvissum tilvikum.
   * - |api-ml|_
     - |api-ml-models|_
     - Stöðugt
     - Þjálfar túlkanleg hefðbundin líkön og eflingarlíkön á mælingatöflum.
   * - |api-classify|_
     - |api-activation|_
     - Beta
     - Útskýrir spár með Captum, SmoothGrad og TorchCAM.
   * - |api-umap|_
     - |api-embedding|_
     - Beta
     - Skoðar myndívörpun gagnvirkt og dreifir klasamerkingum.
   * - **Raðgreining og skimunargreining**
     -
     -
     -
   * - |api-sequencing|_
     - |api-barcodes|_
     - Stöðugt
     - Varpar raða-, dálka- og gRNA-strikamerkjum úr FASTQ-lestrum og tengir leiðarsameindir við myndaðar frumur.
   * - |api-barcode-qc|_
     - |api-barcode-qc-sweep|_
     - Alfa
     - Skýrir frá lestrum á brunn, árekstrartíðni og óvörpuðu hlutfalli miðað við væntanleg gRNA á brunn.
   * - |api-regression|_
     - |api-regression-models|_
     - Stöðugt
     - Metur áhrif leiðarsameinda, gena, skilyrða og viðmiða með 17 líkanafjölskyldum.
   * - |api-power|_
     - |api-power-design|_
     - Alfa
     - Reiknar hve margar frumur og brunna skimun þarf með tilliti til raðgreiningarvillna og brottfalls brunna.
   * - |api-graph|_
     - |api-graph-builder|_
     - Alfa
     - Byggir graf með því að draga dálka á x, y, lit, stærð og flöt.
   * - |api-artifacts|_
     - |api-provenance|_
     - Alfa
     - Skráir keyrsluauðkenni, slembifræ og stillingar að baki úttaki Mask, Measure, Classify og útflutnings.

.. |api-qt-app| replace:: **Qt-forrit**
.. _api-qt-app: https://einarolafsson.github.io/spacr/api/spacr/qt/app/index.html

.. |doc-i18n| replace:: **Staðfærsla á tíu tungumálum**
.. _doc-i18n: https://einarolafsson.github.io/spacr/localization.html

.. |doc-i18n-help| replace:: **Staðfærð samhengishjálp**
.. _doc-i18n-help: https://einarolafsson.github.io/spacr/localization.html#contextual-help

.. |api-qt-ai| replace:: **Qt AI**
.. _api-qt-ai: https://einarolafsson.github.io/spacr/api/spacr/qt/ai/index.html

.. |api-qt-ai-console| replace:: **AI-studd stjórnstöð**
.. _api-qt-ai-console: https://einarolafsson.github.io/spacr/api/spacr/qt/ai/index.html

.. |api-animations| replace:: **Skrá yfir hreyfimyndir stillinga**
.. _api-animations: https://einarolafsson.github.io/spacr/api/spacr/setting_animations/index.html

.. |doc-animations| replace:: **Hreyfimyndir sjónrænna stillinga**
.. _doc-animations: https://einarolafsson.github.io/spacr/setting_animations.html

.. |api-selection| replace:: **Val**
.. _api-selection: https://einarolafsson.github.io/spacr/api/spacr/selection/index.html

.. |api-linked-views| replace:: **Tengt val**
.. _api-linked-views: https://einarolafsson.github.io/spacr/api/spacr/qt/linked_selection/index.html

.. |api-doctor| replace:: **Doctor**
.. _api-doctor: https://einarolafsson.github.io/spacr/api/spacr/doctor/index.html

.. |api-doctor-checks| replace:: **Greining uppsetningar**
.. _api-doctor-checks: https://einarolafsson.github.io/spacr/api/spacr/doctor/index.html

.. |api-mask| replace:: **Mask**
.. _api-mask: https://einarolafsson.github.io/spacr/api/spacr/core/index.html

.. |api-mask-2d| replace:: **Gerð tvívíðra gríma**
.. _api-mask-2d: https://einarolafsson.github.io/spacr/api/spacr/core/index.html#spacr.core.preprocess_generate_masks

.. |api-mask-3d| replace:: **Gerð þrí- og fjórvíðra gríma**
.. _api-mask-3d: https://einarolafsson.github.io/spacr/api/spacr/core/index.html#spacr.core.preprocess_generate_masks

.. |api-illumination| replace:: **Lýsing**
.. _api-illumination: https://einarolafsson.github.io/spacr/api/spacr/illumination/index.html

.. |api-flatfield| replace:: **Flatsviðsleiðrétting**
.. _api-flatfield: https://einarolafsson.github.io/spacr/api/spacr/illumination/index.html

.. |api-measure| replace:: **Measure**
.. _api-measure: https://einarolafsson.github.io/spacr/api/spacr/measure/index.html

.. |api-measure-2d| replace:: **Mælingar viðfanga**
.. _api-measure-2d: https://einarolafsson.github.io/spacr/api/spacr/measure/index.html#spacr.measure.measure_crop

.. |api-segqc| replace:: **Gæðamat aðgreiningar**
.. _api-segqc: https://einarolafsson.github.io/spacr/api/spacr/seg_qc/index.html

.. |api-segqc-verdict| replace:: **Mat fyrir keyrslu**
.. _api-segqc-verdict: https://einarolafsson.github.io/spacr/api/spacr/seg_qc/index.html

.. |api-timelapse| replace:: **Timelapse**
.. _api-timelapse: https://einarolafsson.github.io/spacr/api/spacr/timelapse/index.html

.. |api-tracking| replace:: **Rakning viðfanga**
.. _api-tracking: https://einarolafsson.github.io/spacr/api/spacr/timelapse/index.html

.. |api-layers| replace:: **Lög**
.. _api-layers: https://einarolafsson.github.io/spacr/api/spacr/layers/index.html

.. |api-layer-viewer| replace:: **Lagasjá**
.. _api-layer-viewer: https://einarolafsson.github.io/spacr/api/spacr/qt/layer_viewer/index.html

.. |api-napari| replace:: **napari-tenging**
.. _api-napari: https://einarolafsson.github.io/spacr/api/spacr/napari_bridge/index.html

.. |api-napari-curation| replace:: **Leiðrétting gríma**
.. _api-napari-curation: https://einarolafsson.github.io/spacr/api/spacr/napari_bridge/index.html

.. |api-annotate| replace:: **Annotate**
.. _api-annotate: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/annotate/index.html

.. |api-annotation| replace:: **Handvirk merking**
.. _api-annotation: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/annotate/index.html

.. |api-active-learning| replace:: **Virkt nám**
.. _api-active-learning: https://einarolafsson.github.io/spacr/api/spacr/active_learning/index.html

.. |api-al-loop| replace:: **Endurþjálfun og endurröðun**
.. _api-al-loop: https://einarolafsson.github.io/spacr/api/spacr/active_learning/index.html

.. |api-classify| replace:: **Classify**
.. _api-classify: https://einarolafsson.github.io/spacr/api/spacr/deep_spacr/index.html

.. |api-classification| replace:: **Flokkun mynda**
.. _api-classification: https://einarolafsson.github.io/spacr/api/spacr/deep_spacr/index.html

.. |api-model-cards| replace:: **Líkanaspjöld**
.. _api-model-cards: https://einarolafsson.github.io/spacr/api/spacr/deep_spacr/index.html

.. |api-activation| replace:: **Virkjunarkort**
.. _api-activation: https://einarolafsson.github.io/spacr/api/spacr/deep_spacr/index.html

.. |api-confusion| replace:: **Confusion**
.. _api-confusion: https://einarolafsson.github.io/spacr/api/spacr/confusion/index.html

.. |api-confusion-drill| replace:: **Ítarleg skoðun ruglingsfylkis**
.. _api-confusion-drill: https://einarolafsson.github.io/spacr/api/spacr/confusion/index.html

.. |api-ml| replace:: **Vélanám**
.. _api-ml: https://einarolafsson.github.io/spacr/api/spacr/ml/index.html

.. |api-ml-models| replace:: **Flokkun mælinga**
.. _api-ml-models: https://einarolafsson.github.io/spacr/api/spacr/ml/index.html

.. |api-umap| replace:: **Image UMAP**
.. _api-umap: https://einarolafsson.github.io/spacr/api/spacr/app_umap/index.html

.. |api-embedding| replace:: **Gagnvirk ívörpun**
.. _api-embedding: https://einarolafsson.github.io/spacr/api/spacr/app_umap/index.html

.. |api-sequencing| replace:: **Raðgreining**
.. _api-sequencing: https://einarolafsson.github.io/spacr/api/spacr/sequencing/index.html

.. |api-barcodes| replace:: **Vörpun strikamerkja**
.. _api-barcodes: https://einarolafsson.github.io/spacr/api/spacr/sequencing/index.html

.. |api-barcode-qc| replace:: **Gæðamat strikamerkja**
.. _api-barcode-qc: https://einarolafsson.github.io/spacr/api/spacr/sequencing_qc/index.html

.. |api-barcode-qc-sweep| replace:: **Skýrsla um brunna og árekstra**
.. _api-barcode-qc-sweep: https://einarolafsson.github.io/spacr/api/spacr/sequencing_qc/index.html

.. |api-regression| replace:: **Regression**
.. _api-regression: https://einarolafsson.github.io/spacr/api/spacr/ml/index.html

.. |api-regression-models| replace:: **Mat á áhrifum skimunar**
.. _api-regression-models: https://einarolafsson.github.io/spacr/api/spacr/ml/index.html

.. |api-power| replace:: **Power**
.. _api-power: https://einarolafsson.github.io/spacr/api/spacr/power_model/index.html

.. |api-power-design| replace:: **Tölfræðilegt afl og hönnun**
.. _api-power-design: https://einarolafsson.github.io/spacr/api/spacr/power_simulate/index.html

.. |api-graph| replace:: **Graph**
.. _api-graph: https://einarolafsson.github.io/spacr/api/spacr/qt/widgets/graph_spec/index.html

.. |api-graph-builder| replace:: **Graph Builder**
.. _api-graph-builder: https://einarolafsson.github.io/spacr/api/spacr/qt/widgets/graph_builder/index.html

.. |api-artifacts| replace:: **Afurðir**
.. _api-artifacts: https://einarolafsson.github.io/spacr/api/spacr/artifacts/index.html

.. |api-provenance| replace:: **Uppruni keyrslu**
.. _api-provenance: https://einarolafsson.github.io/spacr/api/spacr/runctx/index.html


Gögn
----

Viðmiðunargagnasöfn
~~~~~~~~~~~~~~~~~~~

- `Heildargagnasafn smásjármynda: BioStudies S-BIAD2135 <https://doi.org/10.6019/S-BIAD2135>`_
- `Prófunargagnasafn: Hugging Face toxo_mito <https://huggingface.co/datasets/einarolafsson/toxo_mito>`_
- `Raðgreiningargögn: NCBI BioProject PRJNA1261935 <https://www.ncbi.nlm.nih.gov/bioproject/?term=PRJNA1261935>`_
- `Aflgreining: spaCRPower <https://github.com/maomlab/spaCRPower>`_


Framlög og aðstoð
------------------------

Villutilkynningar og afmarkaðar tillögur að eiginleikum eru velkomnar á `GitHub Issues <https://github.com/EinarOlafsson/spacr/issues>`_. Þegar bilun er tilkynnt skal láta fylgja útgáfu spaCR, stýrikerfi, Python-útgáfu, stillingar einingarinnar og viðeigandi annálsbút. ``spacr-doctor`` safnar flestum þessum upplýsingum sjálfkrafa.

Leyfi
~~~~~~~~~

Frumkóði núverandi þróunargreinar er aðgengilegur samkvæmt `PolyForm Noncommercial License 1.0.0 <https://github.com/EinarOlafsson/spacr/blob/main/LICENSE>`_. Notkun í atvinnuskyni krefst sérstaks leyfis frá rétthafa. Útgefnar útgáfur til og með spaCR 1.4.9.9 eru áfram aðgengilegar samkvæmt MIT-leyfinu sem fylgdi þeim.

Kennsluefni
~~~~~~~~~~~

`Gagnvirka safnið af spaCR-kennsluefni <https://einarolafsson.github.io/spacr/tutorials/>`_ inniheldur talsettar og textaðar leiðbeiningar um uppsetningu og hvert verkflæði forritsins á átta tungumálum.

Tilvísun í spaCR
~~~~~~~~~~~~~~~~

Ef spaCR nýtist rannsóknunum þínum skaltu vitna í:

Olafsson EB, *o.fl.* Sameinuð myndgreiningarskimun með CRISPR greinir EAF1 sem mótara á yfirtöku ESCRT-kerfisins í *T. gondii*.

`bioRxiv-forprent <https://www.biorxiv.org/content/10.64898/2026.07.08.737057v1>`_ · `hugbúnaðarsafn <https://doi.org/10.5281/zenodo.21343317>`_
