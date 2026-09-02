|Docs| |Tutorials| |PyPI| |Conda| |Python| |Tests| |Qt| |Source| |Issues| |License| |DOI|

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
.. |Tests| image:: https://github.com/EinarOlafsson/spacr/actions/workflows/tests.yml/badge.svg?branch=nightly
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
.. |DOI| image:: https://img.shields.io/badge/DOI-10.5281%2Fzenodo.21343316-blue
   :target: https://doi.org/10.5281/zenodo.21343316
   :alt: Zenodo DOI
.. |Release| image:: https://img.shields.io/github/v/release/EinarOlafsson/spacr?label=Installers
   :target: https://github.com/EinarOlafsson/spacr/releases/latest
   :alt: Nýjustu uppsetningarforrit
.. |Conda| image:: https://anaconda.org/conda-forge/spacr/badges/version.svg
   :target: https://anaconda.org/conda-forge/spacr
   :alt: conda-forge-útgáfa

.. image:: ../../../spacr/resources/icons/logo_spacr_readme.png
   :alt: spaCR
   :width: 920

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

**Rýmisbundin svipgerðargreining á CRISPR-skimunum.**

spaCR aðgreinir og mælir stakar frumur í afkastamiklum smásjármyndum, samþættir svipgerðir einstakra viðfanga við magn leiðarsameinda sem fæst úr raðgreiningu og metur hvaða gen tengjast svipgerðarbreytingum. Út frá plötumyndum og FASTQ-röðum býr það til mælingar fyrir hvert viðfang, þjálfaða flokkara, áhrifamat fyrir hverja leiðarsameind og hvert gen og forgangsraðaðan lista yfir niðurstöður.

Segmingu, mæling, notkun og flokksmiðju mótmælur virkar einnig án sekkunararms.

Myndir, maskar, vöru, mælingar, notkun, fyrirspurn, barkód og vel viðurkenningar eru í einum SQLite verkefni.

Það virkar eins og skrifstofu notkun eða heiðarlega á vinnustaði, serveri eða klústeri.

Hardware aðstoð
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


Setja upp spaCR
---------------

Skjáborðsforrit
~~~~~~~~~~~~~~~~~~~

Þessir uppbyggingar búnir eigin Python. Conda er ekki nauðsynlegt.

.. spacr-installer-links-begin

|InstallerLinux| |InstallerMacOS| |InstallerWindows| |InstallerLegacy|

.. |InstallerWindows| image:: ../../../spacr/resources/icons/platforms/windows.png
   :width: 64
   :alt: Sækja spaCR 1.5.0.4 fyrir Windows 10/11
   :target: https://github.com/EinarOlafsson/spacr/releases/download/v1.5.0.4/SpaCR-1.5.0.4-Windows-Online-Setup.exe
.. |InstallerMacOS| image:: ../../../spacr/resources/icons/platforms/macos.png
   :width: 64
   :alt: Sækja spaCR 1.5.0.4 fyrir macOS 11+ (Intel og Apple Silicon)
   :target: https://github.com/EinarOlafsson/spacr/releases/download/v1.5.0.4/SpaCR-1.5.0.4-macOS-Universal-Online.pkg
.. |InstallerLinux| image:: ../../../spacr/resources/icons/platforms/linux.png
   :width: 64
   :alt: Sækja spaCR 1.5.0.4 fyrir 64-bita Linux
   :target: https://github.com/EinarOlafsson/spacr/releases/download/v1.5.0.4/SpaCR-1.5.0.4-Linux-x86_64-Online.run
.. |InstallerLegacy| image:: ../../../spacr/resources/icons/platforms/legacy.png
   :width: 64
   :alt: Eldri spaCR-uppsetningarforrit
   :target: ../../source/installers.rst

.. spacr-installer-links-end

Fyrstu þremur tákn leyfja núverandi útgáfu. spaCR táknin opnar fullkomið installer arkívu. Installer tengsl og verslun filnames eru uppfærdur af útgáfur vinnuflu; fyrri installerir eru enn í sama útgáfa arkíva.

Í Linux skaltu gera skrána sem var sótt keyranlega og keyra hana:

.. code-block:: bash

   chmod +x SpaCR-*-Linux-x86_64-Online.run
   ./SpaCR-*-Linux-x86_64-Online.run

Á macOS, opna ``.pkg``. Núverandi beta er ekki notarið; ef Gatekeeper blokkir það, velja **System Settings → Privacy & Security → Open Anyway**.

Sjá `Installer leiðbeiningar <../../source/installer_guide.rst>`_ til að uppgötva, deinstalla, offline og vandamálið.

Uppsetning frá PyPI
~~~~~~~~~~~~~~~~~~~

Fyrir útgáfuna á PyPI skaltu setja spaCR upp með pip inni í Conda-umhverfi. Python 3.12 býður upp á mesta úrvalið af valfrjálsum vísindapökkum:

.. code-block:: bash

   conda create -n spacr python=3.12 -y
   conda activate spacr
   python -m pip install --upgrade pip
   python -m pip install "spacr[qt]"
   spacr

spaCR styður Python **3.9 til 3.14**, nema Python 3.14.1, sem torchvision útskýrir. Linux er mælt fyrir þyngsta CUDA og ROCm vinnuflöðum; macOS og Windows eru einnig styður, og bæði nota GPUs — macOS með Metal, sem dregur Apple Silicon og AMD kort í Intel Macs, og Windows með CUDA eða DirectML.

Slepptu Qt á þjóni, reikniklasa eða CI-keyrsluumhverfi:

.. code-block:: bash

   python -m pip install spacr
   spacr-run --list

Opinlegri samsetningar eru settar sérstakt, t.d. ``spacr[zarr]``, ``spacr[omero]``,``spacr[napari]`` og ``spacr[czi,nd2,lif]``. Sjá `Uppsetningu leiðbeiningar <../../source/installer_guide.rst>`_ fyrir fullkomna útgáfur og Python-version samskipti tól.

Uppsetning með conda-forge
~~~~~~~~~~~~~~~~~~~~~~~~~~

Opinberi conda-forge-pakkinn setur spaCR og nauðsynlegar einingar skjáborðsforritsins upp í virka umhverfinu:

.. code-block:: bash

   conda create -n spacr python=3.12 -y
   conda activate spacr
   conda install conda-forge::spacr
   spacr

Uppsetur frá kjarninu
~~~~~~~~~~~~~~~~~~~~~

Klónaðu upphafinn og setja upp það í breyttan hátt, þannig að vinnumópi þína *is* byggð pakka og breytingar munu virka án endursetningu::

    git clone https://github.com/EinarOlafsson/spacr.git
    cd spacr
    conda create -n spacr python=3.12 -y
    conda activate spacr
    pip install -e .
    spacr

Skammslan er ``nightly``. Fyrir ákveðinn útgáfur::

    git clone --branch v1.5.0.5 https://github.com/EinarOlafsson/spacr.git

Til að draga eftirfarandi breytingar, frá innri klóna::

    git pull
    pip install -e .

2. línu er aðeins nauðsynlegt þegar afhengingar eða innfangspunktur breytist; Python kóða er taka upp án þess. ef lögun er enn að hlaupa gamla kóða eftir að taka, ``spacr-doctor`` segir að ``spacr`` er í raun á leiðinni, sem er venjulega ástæða.

Að setja upp úr ljósið (Light)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Fullt klón: 427 MB. Kjarnklón: 76 MB.

::

    curl -fsSL https://raw.githubusercontent.com/EinarOlafsson/spacr/nightly/packaging/install_from_source.sh -o install_spacr.sh
    sh install_spacr.sh --branch nightly

Skips ``docs/``, ``tests/`` og Cellpose athygli, skráðir tölur og útbreiddar þýðingar.

Options: ``--dir``, ``--branch`` (default ``main``), ``--with-tests``, ``--with-docs``, ``--with-translations``, ``--no-install``.

``packaging/source_install_excludes.txt`` listar hvert skipað leið.


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

Stilltu ``SPACR_LOG_LEVEL=DEBUG`` við bilanagreiningu. Annálaskrár með skráaveltu eru skrifaðar í ``~/.spacr/logs/spacr.log``.

``spacr-run --list`` listar einingar sem hafa skipanalínuinngang til keyrslu án grafísks viðmóts. Einingum fyrir merkingu, gagnayfirferð, samanburð og könnun sem eingöngu eru í GUI er sleppt.


Kjarnaverkflæði
---------------

Aðalvinnuflæðið samanstendur af sex einingum:

- **Mask** hlutgreinir frumur, frumukjarna, sýkla og frumulíffæri með Cellpose.
- **Measure** skrifar lögunar-, styrkleika-, áferðar-, rúm- og samstaðsetningareiginleika ásamt myndúrklippum viðfanga í SQLite.
- **Annotate** merkir myndúrklippur í lyklaborðsstýrðu hnitaneti og styður biðraðir virks náms.
- **Classify** þjálfar líkön byggð á myndum eða mælingum og skráir frammistöðu á fráteknum gögnum með hverjum varðpunkti.
- **Map Barcodes** varpar FASTQ-lestrum á brunna og gRNA og veitir gæðamat fyrir magn, árekstra og þekju.
- **Regression** metur áhrif leiðarsameinda, gena, skilyrða og viðmiða með líkanafjölskyldum sem henta samfelldum gildum, hlutföllum og talningum.

Sama verkefni má einnig nota til að hanna plötur, meta tölfræðilegan styrk, leiðrétta lotuáhrif, kanna gæði hlutunar, skoða tengd gröf og myndúrklippur, flytja út AnnData, halda áfram vinnslu sem var stöðvuð og skrá stillingarnar sem liggja að baki hverri niðurstöðu.

spaCR-einingar
--------------

.. spacr-workflow-begin

|Module_mask|\ |Module_measure|\ |Module_annotate|\ |Module_classify_merged|\ |Module_map_barcodes|\ |Module_regression|

|Module_foreign|\ |Module_run_compare|\ |Module_experiment_design|\ |Module_power|\ |Module_dose_response|\ |Module_qc_dashboard|

|Module_make_masks|\ |Module_align|\ |Module_umap|\ |Module_gate_editor|\ |Module_graph_builder|\ |Module_analyze_plaques|

|Module_recruitment|\ |Module_invasion|\ |Module_replication|

.. |Module_mask| image:: ../../../spacr/resources/icons/workflow/mask.png
   :width: 16.0%
   :alt: Opna API-skjölin fyrir Mask
   :target: https://einarolafsson.github.io/spacr/api/spacr/core/index.html
   :align: middle
.. |Module_measure| image:: ../../../spacr/resources/icons/workflow/measure.png
   :width: 16.0%
   :alt: Opna API-skjölin fyrir Measure
   :target: https://einarolafsson.github.io/spacr/api/spacr/measure/index.html
   :align: middle
.. |Module_annotate| image:: ../../../spacr/resources/icons/workflow/annotate.png
   :width: 16.0%
   :alt: Opna API-skjölin fyrir Annotate
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/annotate/index.html
   :align: middle
.. |Module_classify_merged| image:: ../../../spacr/resources/icons/workflow/classify_merged.png
   :width: 16.0%
   :alt: Opna API-skjölin fyrir Classify
   :target: https://einarolafsson.github.io/spacr/api/spacr/classify/index.html
   :align: middle
.. |Module_map_barcodes| image:: ../../../spacr/resources/icons/workflow/map_barcodes.png
   :width: 16.0%
   :alt: Opna API-skjölin fyrir Map Barcodes
   :target: https://einarolafsson.github.io/spacr/api/spacr/sequencing/index.html
   :align: middle
.. |Module_regression| image:: ../../../spacr/resources/icons/workflow/regression.png
   :width: 16.0%
   :alt: Opna API-skjölin fyrir Regression
   :target: https://einarolafsson.github.io/spacr/api/spacr/ml/index.html
   :align: middle
.. |Module_foreign| image:: ../../../spacr/resources/icons/workflow/apps/foreign.png
   :width: 16.0%
   :alt: Opna API-skjölin fyrir Import
   :target: https://einarolafsson.github.io/spacr/api/spacr/foreign/index.html
   :align: middle
.. |Module_run_compare| image:: ../../../spacr/resources/icons/workflow/apps/run_compare.png
   :width: 16.0%
   :alt: Opna API-skjölin fyrir Run Compare
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/run_compare/index.html
   :align: middle
.. |Module_experiment_design| image:: ../../../spacr/resources/icons/workflow/apps/experiment_design.png
   :width: 16.0%
   :alt: Opna API-skjölin fyrir Experiment Design
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/experiment_design/index.html
   :align: middle
.. |Module_power| image:: ../../../spacr/resources/icons/workflow/apps/power.png
   :width: 16.0%
   :alt: Opna API-skjölin fyrir Power / Design
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/power/index.html
   :align: middle
.. |Module_dose_response| image:: ../../../spacr/resources/icons/workflow/apps/dose_response.png
   :width: 16.0%
   :alt: Opna API-skjölin fyrir Dose–Response
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/dose_response/index.html
   :align: middle
.. |Module_qc_dashboard| image:: ../../../spacr/resources/icons/workflow/apps/qc_dashboard.png
   :width: 16.0%
   :alt: Opna API-skjölin fyrir QC
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/qc_dashboard/index.html
   :align: middle
.. |Module_make_masks| image:: ../../../spacr/resources/icons/workflow/apps/make_masks.png
   :width: 16.0%
   :alt: Opna API-skjölin fyrir Make Masks
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/make_masks/index.html
   :align: middle
.. |Module_align| image:: ../../../spacr/resources/icons/workflow/apps/align.png
   :width: 16.0%
   :alt: Opna API-skjölin fyrir Align & Stitch
   :target: https://einarolafsson.github.io/spacr/api/spacr/align/index.html
   :align: middle
.. |Module_umap| image:: ../../../spacr/resources/icons/workflow/apps/umap.png
   :width: 16.0%
   :alt: Opna API-skjölin fyrir Image UMAP
   :target: https://einarolafsson.github.io/spacr/api/spacr/core/index.html
   :align: middle
.. |Module_gate_editor| image:: ../../../spacr/resources/icons/workflow/apps/gate_editor.png
   :width: 16.0%
   :alt: Opna API-skjölin fyrir Gate Editor
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/gate_editor/index.html
   :align: middle
.. |Module_graph_builder| image:: ../../../spacr/resources/icons/workflow/apps/graph_builder.png
   :width: 16.0%
   :alt: Opna API-skjölin fyrir Graph Builder
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/graph_builder/index.html
   :align: middle
.. |Module_analyze_plaques| image:: ../../../spacr/resources/icons/workflow/apps/analyze_plaques.png
   :width: 16.0%
   :alt: Opna API-skjölin fyrir Plaque Assay
   :target: https://einarolafsson.github.io/spacr/api/spacr/submodules/index.html
   :align: middle
.. |Module_recruitment| image:: ../../../spacr/resources/icons/workflow/apps/recruitment.png
   :width: 16.0%
   :alt: Opna API-skjölin fyrir Recruitment
   :target: https://einarolafsson.github.io/spacr/api/spacr/submodules/index.html
   :align: middle
.. |Module_invasion| image:: ../../../spacr/resources/icons/workflow/apps/invasion.png
   :width: 16.0%
   :alt: Opna API-skjölin fyrir Invasion Assay
   :target: https://einarolafsson.github.io/spacr/api/spacr/submodules/index.html
   :align: middle
.. |Module_replication| image:: ../../../spacr/resources/icons/workflow/apps/replication.png
   :width: 16.0%
   :alt: Opna API-skjölin fyrir Replication Assay
   :target: https://einarolafsson.github.io/spacr/api/spacr/submodules/index.html
   :align: middle

.. spacr-workflow-end

Hver mólur spaCR skipar, í orði heimaskæran listar þá: sjö pipeline mólus fyrst, þá allt annað. Veldu skál til að opna API síðu þessara mólusa.


Make Masks
~~~~~~~~~~

Make Masks birtast undir **Tools** fyrir höndilega korrigeringu af sviði maskar; másthead hans opnar Cellpose vinnuflokk. Nín tól: **Brush**, **Erase**,**Erasa objekt**, #**Wand +**, [**Wan −**, "**Draw**, '**Divide**,'**Zoom** og '**Recrop**.

Cellpose-SAM fer hér að sýna möguleika kartan og flutningsfólkið við maskinn. Sjá `Leikstjóri <../../source/features.rst>`_ fyrir hvert tól.

**Andrar auðlindir**

- `Samskiptaþjálfunar <https://einarolafsson.github.io/spacr/tutorials/>`_ — 73 leiðbeiningar vinnufluðum frá uppsetningu í gegnum hit rannsóknir.
- `Python API snemma byrjun <../../source/python_api.rst>`_ — hlaupa og staðfest pipelines frá skriptum, notebooks eða klúster.
- `Leikstjóri <../../source/features.rst>`_ — hæfileika, fullnægjandi og valfrjáls tengsl.
- `Heilluð API reference <https://einarolafsson.github.io/spacr/api/index.html>`_ — stuðlað innfangspunktur eftir verkefni, með fullkomna mótum tengslum einn hærra.
- `Sjálf tungumál og þýðingu leiðbeining <../../source/localization.rst>`_ — samskipti tungumál, kontext hjálp og vísindaleg útleiðslu.

Tungumál og þýðingar
~~~~~~~~~~~~~~~~~~~~~~

Viðmótið styður tíu tungumál í leiðsögn og stillingum. AI- og LIVE-stýringar, lýsingar á einingum og yfirfarin samhengishjálp eru einnig þýdd. Skiptu um tungumál undir **spaCR → Stillingar → Tungumál** án endurræsingar. Annálar, slóðir, gagnagrunnsgildi og mælingar eru aldrei þýdd; vísindaleg úttök haldast á viðurkenndri ensku. Sjá `stefnu um samhengishjálp <../../source/localization.rst#contextual-help>`_.

Hreyfimyndaleiðbeiningar fyrir stillingar
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Stillingar með sjónræna skýringu bjóða upp á **Animation**-stýringu í verkfæraábendingunni. Skoðaðu `myndasafn stillingahreyfimynda <https://einarolafsson.github.io/spacr/setting_animations.html>`_ eða `skrá stillingahreyfimynda <https://einarolafsson.github.io/spacr/api/spacr/setting_animations/index.html>`_.

Gögn
----

Viðmiðunargagnasöfn
~~~~~~~~~~~~~~~~~~~

|DataBioStudies| |DataHuggingFace| |DataNCBI| |DataSpaCRPower| |DataBioRxiv|

.. |DataBioStudies| image:: ../../../spacr/resources/icons/databanks/biostudies_button.png
   :width: 72
   :alt: Opna smásjárgagnasafnið í BioStudies
   :target: https://doi.org/10.6019/S-BIAD2135
.. |DataHuggingFace| image:: ../../../spacr/resources/icons/databanks/huggingface_button.png
   :width: 72
   :alt: Opna prófunargagnasafnið á Hugging Face
   :target: https://huggingface.co/datasets/einarolafsson/toxo_mito
.. |DataNCBI| image:: ../../../spacr/resources/icons/databanks/ncbi_button.png
   :width: 72
   :alt: Opna raðgreiningargagnasafnið hjá NCBI
   :target: https://www.ncbi.nlm.nih.gov/bioproject/?term=PRJNA1261935
.. |DataSpaCRPower| image:: ../../../spacr/resources/icons/databanks/spacrpower_button.png
   :width: 72
   :alt: Opna spaCRPower
   :target: https://github.com/maomlab/spaCRPower
.. |DataBioRxiv| image:: ../../../spacr/resources/icons/databanks/biorxiv_button.png
   :width: 72
   :alt: Opna bioRxiv-forprentið
   :target: https://www.biorxiv.org/content/10.64898/2026.07.08.737057v1

Góðursvæði
~~~~~~~~~~

spaCR fylgir safn af þjálfuðum líkönum og sækir þau eftir þörfum. Opnaðu **Líkanasafn** af heimaskjánum til að skoða þau og setja upp, eða tilgreindu lykil í stillingaskrá -- ``pathogen_model: toxoplasma_pv_v1`` -- og líkanið er sótt og gátsumma þess staðfest í fyrsta sinn sem þess er þörf. Hver birt færsla ber SHA-256; færslu án hennar er hafnað fremur en sett upp, því ekki er hægt að greina stytt eða útskipt líkan frá því rétta.

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

Númerur ovan eru þeir sem mætt eru á útgáfu, og takmarkanir eru tilkynnt með þeim: myndavél er gagnlegt fyrir vinnu sem það var mætt á, ekki fyrir hverri vinnu. ``toxoplasma_well_detector_v1`` og ``toxoplasma_plaque_v1`` eru tvö hálfa af einum tækjum - uppgötvuninn finnur bólkurnar, seggjandi finnur plakkum inni í þeim, og vel þægindi er það sem gerir svæðum samanburðar milli mikroskópum.

Modelli eru veitt á eigin Hugging Face reikningum rithöfundar síns, þannig að að taka þátt þýðir ekki að veita skrifu aðgang að einhverjum öðrum. ``spacr.model_zoo`` ``publish_model`` gerir upplifun og trúa á listanum eftir að bæta.


Greining á afköstum
----------------------

Búðu til vélbúnaðarskýrslu og hengdu hana við mál um afköst::

    python tools/spacr_hardware_report.py

Spara til ``~/.spacr/reports`` og trúa leiðinni. ``--quick`` skiptir lengri skilyrði; ``--out PATH`` setur staðsetningu.

Lesa engin verkefni gögnum. Tíms innfang, fjölbreytna bókasafn, vinstri byggingu og uppgötvun. Rannsóknir um meðferð-arquitectur emulans (a x86_64 Python bygging á Apple Silicon) og BLAS framkvæmd NumPy.

Orðlinna reference
----------------------

Öll beint hér að neðan er sett með ``pip install spacr``. Allir samþykkir ``--help``.

Að byrja við umsókn
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   spacr              # the desktop application
   spacr-tutorial     # the interactive tutorial library
   spacr-server       # no first-run setup screen, for unattended launches

``spacr-server`` skípa modal setup skján, sem annars myndi blokkja óþekkt vinnu.

``spacr-qt`` og ``spacr-nightly`` eru alias af ``spacr``.

Þegar spaCR mun ekki byrja
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   spacr-doctor       # diagnose the installation and say how to fix it
   safespacr          # the least spaCR that can still change a setting

``spacr-doctor`` drukkar eitt línu á athygli, með komandi til að kjósa fyrir hvert mismunandi. Það segir einnig hvaða ``spacr`` er á leiðinni, sem er það sem gamla redigable uppsetningu skugga.

``safespacr`` lætur hvert forrit eins og uppáhaldsins og þykir bakgrunni, tegundum, verbose logging og hlaða út. Nottu það þegar sparaður forrit breytir upphafið. Það breytist ekkert stöðugt.

Að hlaupa modúlum án heiðar
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Engin Qt, engin sýning — fyrir klúster, þjónusta og CI.

.. code-block:: bash

   spacr-run --list                              # modules with a headless entry
   spacr-run --describe MODULE                   # what a module consumes and produces
   spacr-run validate --module MODULE \
       --settings settings.csv                   # check settings before spending the run
   spacr-run MODULE --settings settings.csv      # execute
   spacr-remote --help                           # submit and monitor SSH, Slurm or cloud jobs

``validate`` lætur sömu settun sem fer myndi og segir hvað er saknað, óþekkt eða sýnir ekkert.

``spacr-run --list`` sýnir aðeins mólur með heiðarlegt innfangspunkt; notkun, lækning og rannsóknir eru samskipt og yfirgefið.

Spurning á leiðinni síðar
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Hver rán er skráður á ``~/.spacr/runs`` með settum sínum, hashed inntölum, úttökum, varningar, útgáfur og frönum.

.. code-block:: bash

   spacr-repro RUN_DIR        # replay a recorded run from its journal
   spacr-workspace RUN_DIR    # what that run had open: databases, montages, views

Ákvarðanir gögnum og uppsetningu
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   spacr-db-audit DB      # SQLite health, integrity, locking, reader/writer probe
   spacr-leakage          # classifier train/test leakage audit
   spacr-plugins          # installed plugin registry and failure diagnostics

Umhverfi
~~~~~~~~~~~

.. code-block:: bash

   SPACR_LOG_LEVEL=DEBUG spacr      # verbose logging for one launch

Rotating logs eru skrifað í ``~/.spacr/logs/spacr.log``. Sættu þessar skál á bug-report.


Framlög og aðstoð
------------------------

Sendu villutilkynningar og afmarkaðar óskir um eiginleika í gegnum `GitHub-mál <https://github.com/EinarOlafsson/spacr/issues>`_. Þegar bilun er tilkynnt skal tilgreina útgáfu spaCR, stýrikerfi, útgáfu Python, stillingar einingarinnar og viðeigandi hluta úr annálnum. ``spacr-doctor`` safnar flestum þessara upplýsinga; láttu vélbúnaðarskýrsluna fylgja þegar tilkynnt er um afkastavandamál.

Leyfi
~~~~~~~~~

spaCR er frelsað undir `BSD 3-Klausur leyfi <https://github.com/EinarOlafsson/spacr/blob/main/LICENSE>`_.

Ef spaCR hjálpaði að útgáfa verk, er nefndur verðmæt og er ekki skilyrði fyrir leyfi — sjá `Citing spaCR`_ hér neðan.

Kennsluefni
~~~~~~~~~~~

`Gagnvirka spaCR-kennslusafnið <https://einarolafsson.github.io/spacr/tutorials/>`_ inniheldur talsettar og textaðar leiðbeiningar um uppsetningu og hvert verkflæði: 73 kennslustundir með 50 röddum á átta tungumálum.

Tilvísun í spaCR
~~~~~~~~~~~~~~~~

Ef spaCR nýtist við rannsóknina skaltu vitna í:

Olafsson EB, *et al.* A sameiginlegur myndbönd sem er bastir á CRISPR skrefinn skilur EAF1 sem *T. gondii* modulator ESCRT subversion.

`Bioregl fyrirframskrift <https://www.biorxiv.org/content/10.64898/2026.07.08.737057v1>`_ · `Programvarparkíf <https://doi.org/10.5281/zenodo.21343316>`_

Þakkir
~~~~~~~~~~~~~~~

spaCR byggir á opnum vísindahugbúnaði, meðal annars NumPy, pandas, scikit-image, scikit-learn, Cellpose, PyTorch og Qt. Sjá `upplýsingar um þýðingarlíkön <../TRANSLATION_MODELS.md>`_ fyrir líkönin sem voru notuð við gerð fjöltyngdra skjala og viðmótsskráa.
