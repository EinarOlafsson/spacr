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


Setja upp spaCR
---------------

Skjáborðsforrit
~~~~~~~~~~~~~~~~~~~

Stöðvarstöðvarnar innihalda persónulegt Python umhverfi, þannig að konda og núverandi Python uppsetningu er ekki nauðsynlegt.

.. spacr-installer-links-begin

|InstallerLinux| |InstallerMacOS| |InstallerWindows| |InstallerLegacy|

.. |InstallerWindows| image:: spacr/resources/icons/platforms/windows.png
   :width: 64
   :alt: Sækja spaCR 1.5.0.4 fyrir Windows 10/11
   :target: https://github.com/EinarOlafsson/spacr/releases/download/v1.5.0.4/SpaCR-1.5.0.4-Windows-Online-Setup.exe
.. |InstallerMacOS| image:: spacr/resources/icons/platforms/macos.png
   :width: 64
   :alt: Sækja spaCR 1.5.0.4 fyrir macOS 11+ (Intel og Apple Silicon)
   :target: https://github.com/EinarOlafsson/spacr/releases/download/v1.5.0.4/SpaCR-1.5.0.4-macOS-Universal-Online.pkg
.. |InstallerLinux| image:: spacr/resources/icons/platforms/linux.png
   :width: 64
   :alt: Sækja spaCR 1.5.0.4 fyrir 64-bita Linux
   :target: https://github.com/EinarOlafsson/spacr/releases/download/v1.5.0.4/SpaCR-1.5.0.4-Linux-x86_64-Online.run
.. |InstallerLegacy| image:: spacr/resources/icons/platforms/legacy.png
   :width: 64
   :alt: Eldri spaCR-uppsetningarforrit
   :target: docs/source/installers.rst

.. spacr-installer-links-end

Fyrstu þremur tákn leyfja núverandi útgáfu. spaCR táknin opnar fullkomið installer arkívu. Installer tengsl og verslun filnames eru uppfærdur af útgáfur vinnuflu; fyrri installerir eru enn í sama útgáfa arkíva.

Á Linux, gera niðurlaust faili verkandi og hlaða það:

.. code-block:: bash

   chmod +x SpaCR-*-Linux-x86_64-Online.run
   ./SpaCR-*-Linux-x86_64-Online.run

Á macOS, opna ``.pkg``. Núverandi beta er ekki notarið; ef Gatekeeper blokkir það, velja **System Settings → Privacy & Security → Open Anyway**.

Sjá `Installer leiðbeiningar <https://einarolafsson.github.io/spacr/installers.html>`_ til að uppgötva, deinstalla, offline og vandamálið.

Python-uppsetning
~~~~~~~~~~~~~~~~~~~

Python 3.12 hefur breiðustu val af ókeypis vísindalegar pakka:

.. code-block:: bash

   conda create -n spacr python=3.12 -y
   conda activate spacr
   python -m pip install --upgrade pip
   python -m pip install "spacr[qt]"
   spacr

spaCR styður Python **3.9 til 3.14**, nema Python 3.14.1, sem torchvision fjarlægir. Linux er mælt fyrir CUDA vinnuflu; macOS og Windows eru einnig styður.

Fyrir server, cluster eða CI runner, óttast Qt:

.. code-block:: bash

   python -m pip install spacr
   spacr-run --list

Opinlegri samsetningar eru settar sérstakt, t.d. ``spacr[ome-zarr]``, ``spacr[omero]``,``spacr[napari]`` og ``spacr[czi,nd2,lif]``. Sjá `Uppsetningu leiðbeiningar <https://einarolafsson.github.io/spacr/installers.html>`_ fyrir fullkomna útgáfur og Python-version samskipti tól.

Skipanalínuskipanir
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

Set ``SPACR_LOG_LEVEL=DEBUG`` þegar ákvarðanir. Rotating logs eru skrifað á ``~/.spacr/logs/spacr.log``. Klassískri Tk grindurinn er enn til staðar sem ``spacr-legacy`` en er ekki lengur þróað.


Það sem hægt er að gera
-----------------------

Meirihluti skrefna fylgja sex mólum:

- **Mask** segmentar celler, kjarna, patógen og organelles með Cellpose.
- **Measure** skrifa morfóliu, hjarta, textu, pláss og lokalization eiginleika, ásamt augum hlut, til SQLite.
- **Annotate** tákn grætur í keyboard-drived net og styður aktiv-learning coues.
- **Classify** veitir mynd eða stærðbasuð móður og skráð niðurstöður með hverjum athygli.
- **Map Barcodes** kartanum FASTQ lætur á bólum og gRNAs, með fullorðnum, stöðu og takmörku QC.
- **Regression** ákvarða leiðbeining, gein, tilstand og stjórnun áhrif með mönnun fjölskyldur sem passa á stöðugum, frakklandi og tækjum svörum.

Saminn verkefni getur einnig hönnuð plötur, uppgötvað styrk, rétt batch áhrif, athuga afgræðslu gæði, rannsaka tengda plöt og gróðurs, útvarpa AnnData, endurheimta ábreytt vinnu og skráð settingar bak hverra niðurstöðu.

Veldu næsta síðu með því sem þú vilt gera:

- `Samskiptaþjálfunar <https://einarolafsson.github.io/spacr/tutorials/>`_ — 73 leiðbeiningar vinnufluðum frá uppsetningu í gegnum hit rannsóknir.
- `Python API snemma byrjun <https://einarolafsson.github.io/spacr/python_api.html>`_ — hlaupa og staðfest pipelines frá skriptum, notebooks eða klúster.
- `Leikstjóri <https://einarolafsson.github.io/spacr/features.html>`_ — hæfileika, fullnægjandi og valfrjáls tengsl.
- `Heilluð API reference <https://einarolafsson.github.io/spacr/api/index.html>`_ — stuðlað innfangspunktur eftir verkefni, með fullkomna mótum tengslum einn hærra.
- `staðsetningu leiðbeiningar <https://einarolafsson.github.io/spacr/localization.html>`_ — samskipti tungumál, kontext hjálp og vísindaleg útleiðslu.

Fjöltyngt skjáborðsviðmót
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Staðfærsla á tíu tungumálum nær yfir leiðsögn, stillingar, AI- og LIVE-stýringar, lýsingar á einingum og yfirfarna samhengishjálp. Skiptu um tungumál undir **spaCR → Stillingar → Tungumál** án endurræsingar. Annálar, slóðir, gagnagrunnsgildi og mælingar eru aldrei þýdd; vísindaleg úttök haldast á viðurkenndri ensku. Sjá `stefnu um samhengishjálp <https://einarolafsson.github.io/spacr/localization.html#contextual-help>`_.

Hreyfimyndaleiðbeiningar fyrir stillingar
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Stillingar með sjónræna skýringu bjóða upp á **Animation**-stýringu í verkfæraábendingunni. Skoðaðu `myndasafn stillingahreyfimynda <https://einarolafsson.github.io/spacr/setting_animations.html>`_ eða `skrá stillingahreyfimynda <https://einarolafsson.github.io/spacr/api/spacr/setting_animations/index.html>`_.

Gögn
----

Viðmiðunargagnasöfn
~~~~~~~~~~~~~~~~~~~

- `Full mikroskópur gögn: BioStudies S-BIAD2135 <https://doi.org/10.6019/S-BIAD2135>`_
- `Testið gögn: Hugging Face toxo_mito <https://huggingface.co/datasets/einarolafsson/toxo_mito>`_
- `Sækkun gögnum: NCBI BioProject PRJNA1261935 <https://www.ncbi.nlm.nih.gov/bioproject/?term=PRJNA1261935>`_
- `Sjálfsögn: Spacepower <https://github.com/maomlab/spaCRPower>`_


Framlög og aðstoð
------------------------

Bragar og hagsmyndaða greinar eru velkomnir í gegnum `GitHub Vörur <https://github.com/EinarOlafsson/spacr/issues>`_. Þegar við að ræða skammt, er það spaCR útgáfa, leitarhólksins, Python útskýrunar, módúlarsetningar og viðeigandi log útskýrt. ``spacr-doctor`` samlar mest af því fyrir þig.

Leyfi
~~~~~~~~~

Núverandi þróun rán er aðgangs-tilbúinn undir `PolyForm óþekktar leyfi 1.0.0 <https://github.com/EinarOlafsson/spacr/blob/main/LICENSE>`_. Kaupleg notkun krefst sérstakt leyfi frá höfundum. útgáfur sem eru útgáfað með spaCR 1.4.9.9 eru enn til staðar undir MIT leyfi sem fylgdi þessum útgáfum.

Kennsluefni
~~~~~~~~~~~

The `Tólfáætlun spaCR kennslabúð <https://einarolafsson.github.io/spacr/tutorials/>`_ inniheldur sögulegt, skrifað vandræðum af uppsetningu og af hverri viðbrögð vinnuflokk, í 73 kennslum með 50 röðum á átta tungumálum.

Tilvísun í spaCR
~~~~~~~~~~~~~~~~

Ef spaCR hjálpar til rannsóknir þíns, heyrðu:

Olafsson EB, *et al.* A sameiginlegur myndbönd sem er bastir á CRISPR skrefinn skilur EAF1 sem *T. gondii* modulator ESCRT subversion.

`Bioregl fyrirframskrift <https://www.biorxiv.org/content/10.64898/2026.07.08.737057v1>`_ · `Programvarparkíf <https://doi.org/10.5281/zenodo.21343317>`_
