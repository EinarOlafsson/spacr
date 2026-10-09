|Platforms| |Python| |Qt| |Tests| |Release| |Issues| |Source| |Conda| |PyPI| |Conda Downloads| |PyPI Downloads| |Docs| |Tutorials| |Preprint| |DOI| |Cite| |License| |PyPI rank|

.. |Docs| image:: https://img.shields.io/github/actions/workflow/status/EinarOlafsson/spacr/pages%2Fpages-build-deployment?label=API%20Documentation
   :target: https://einarolafsson.github.io/spacr/
   :alt: API-skjöl
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
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/index.html#module-spacr.qt
   :alt: Qt-viðmót
.. |Source| image:: https://img.shields.io/badge/GitHub-Source-181717?logo=github
   :target: https://github.com/EinarOlafsson/spacr
   :alt: Frumkóði á GitHub
.. |Issues| image:: https://img.shields.io/github/issues/EinarOlafsson/spacr
   :target: https://github.com/EinarOlafsson/spacr/issues
   :alt: GitHub-mál
.. |License| image:: https://img.shields.io/github/license/EinarOlafsson/spacr
   :target: https://github.com/EinarOlafsson/spacr/blob/main/LICENSE
   :alt: BSD 3-Clause-leyfi
.. |Preprint| image:: https://img.shields.io/badge/bioRxiv-2026.07.08.737057-BF2636
   :target: https://www.biorxiv.org/content/10.64898/2026.07.08.737057v1
   :alt: bioRxiv-forprentun
.. |DOI| image:: https://img.shields.io/badge/DOI-10.5281%2Fzenodo.21343316-blue
   :target: https://doi.org/10.5281/zenodo.21343316
   :alt: Zenodo DOI
.. |Release| image:: https://img.shields.io/github/v/release/EinarOlafsson/spacr?label=Installers
   :target: https://github.com/EinarOlafsson/spacr/releases/latest
   :alt: Nýjustu uppsetningarforrit
.. |Conda| image:: https://anaconda.org/conda-forge/spacr/badges/version.svg
   :target: https://anaconda.org/conda-forge/spacr
   :alt: conda-forge-útgáfa
.. |Conda Downloads| image:: https://anaconda.org/conda-forge/spacr/badges/downloads.svg
   :target: https://anaconda.org/conda-forge/spacr
   :alt: conda-forge-niðurhal
.. |Release date| image:: https://anaconda.org/conda-forge/spacr/badges/latest_release_date.svg
   :target: https://anaconda.org/conda-forge/spacr
   :alt: Dagsetning nýjustu útgáfu á conda-forge
.. |PyPI Downloads| image:: https://static.pepy.tech/personalized-badge/spacr?period=total&units=INTERNATIONAL_SYSTEM&left_color=GRAY&right_color=GREEN&left_text=downloads
   :target: https://pepy.tech/projects/spacr
   :alt: PyPI-niðurhal
.. |Platforms| image:: https://img.shields.io/badge/Platforms-Linux%20%7C%20macOS%20%7C%20Windows-lightgrey
   :target: https://github.com/EinarOlafsson/spacr/blob/nightly/docs/source/installers.rst
   :alt: Linux, macOS og Windows
.. |Cite| image:: https://img.shields.io/badge/Cite-CITATION.cff-8A2BE2
   :target: https://github.com/EinarOlafsson/spacr/blob/main/CITATION.cff
   :alt: Vitna í spaCR
.. |PyPI rank| image:: https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fsql-clickhouse.clickhouse.com%2F%3Fuser%3Ddemo%26param_package_name%3Dspacr%26param_days%3D30%26query%3DWITH%2B%2528%2BSELECT%2Bsum%2528count%2529%2BFROM%2Bpypi.pypi_downloads_per_day%2BWHERE%2Bproject%2B%253D%2B%257Bpackage_name%253AString%257D%2BAND%2Bdate%2B%253E%253D%2BtoDate%2528now%2528%2527UTC%2527%2529%2529%2B-%2B%257Bdays%253AUInt16%257D%2BAND%2Bdate%2B%253C%2BtoDate%2528now%2528%2527UTC%2527%2529%2529%2B%2529%2BAS%2Bdownloads%2BSELECT%2Bdownloads%2BAS%2Bpackage_downloads%252C%2BcountIf%2528n%2B%253E%253D%2Bdownloads%2529%2BAS%2Brank%252C%2Bcount%2528%2529%2BAS%2Btotal_packages%252C%2B100.0%2B%252A%2Brank%2B%252F%2BnullIf%2528total_packages%252C%2B0%2529%2BAS%2Bpercentile%252C%2Bif%2528%2Btotal_packages%2B%253D%2B0%2BOR%2Bdownloads%2B%253D%2B0%252C%2B%2527no%2Bdata%2527%252C%2Bconcat%2528%2B%2527top%2B%2527%252C%2BtoString%2528ceil%25281000.0%2B%252A%2Brank%2B%252F%2BnullIf%2528total_packages%252C%2B0%2529%2529%2B%252F%2B10%2529%252C%2B%2527%2525%2527%2B%2529%2B%2529%2BAS%2Bmessage%2BFROM%2B%2528%2BSELECT%2Bproject%252C%2Bsum%2528count%2529%2BAS%2Bn%2BFROM%2Bpypi.pypi_downloads_per_day%2BWHERE%2Bdate%2B%253E%253D%2BtoDate%2528now%2528%2527UTC%2527%2529%2529%2B-%2B%257Bdays%253AUInt16%257D%2BAND%2Bdate%2B%253C%2BtoDate%2528now%2528%2527UTC%2527%2529%2529%2BGROUP%2BBY%2Bproject%2B%2529%2BFORMAT%2BJSON&query=%24.data%5B0%5D.message&label=PyPI+rank+%2830d%29&color=brightgreen&cacheSeconds=86400
   :target: https://clickpy.clickhouse.com/dashboard/spacr
   :alt: Röðun spaCR eftir PyPI-niðurhali síðustu 30 heilu dagana

.. image:: ../../source/_static/deck/slides/slide_01.jpg
   :alt: spaCR
   :width: 920
   :target: https://einarolafsson.github.io/spacr/_static/deck/

`← Til baka <../../source/_static/deck/pages/57.md>`_   `Næsta → <../../source/_static/deck/pages/02.md>`_

spaCR
=====

.. spacr-language-picker-begin

Tungumál: `🌐 Íslenska ▾ <README.md>`_

.. spacr-language-picker-end

**Rýmisbundin svipgerðargreining á CRISPR-skimunum.**

spaCR aðgreinir og mælir stakar frumur í smásjármyndum, samþættir svipgerðir einstakra viðfanga við magn leiðarsameinda sem fæst úr raðgreiningu og metur hvaða gen tengjast svipgerðarbreytingum. Út frá plötumyndum og FASTQ-röðum býr það til mælingar fyrir hvert viðfang, þjálfaða flokkara, áhrifamat fyrir hverja leiðarsameind og hvert gen og forgangsraðaðan lista yfir niðurstöður.

Segmingu, mæling, notkun og flokksmiðju mótmælur virkar einnig án sekkunararms.

Make Masks leiðréttir skiptingargrímur og merkir sjálfstæða rétthyrninga með flokkamerkjum með **Box** verkfærinu fyrir YOLO-útflutning. Rammarnir halda eigin merkjum og sögu án þess að breyta upprunamyndum eða grímum.

Hvert verkfæri er útskýrt í `eiginleikahandbókinni <../../source/features.rst>`_.

Myndir, grímur, myndúrklippur, mælingar, merkingar, spár, strikamerki og auðkenni brunna eru geymd í einu SQLite-verkefni.

Keyrist sem skjáborðsforrit eða án grafísks viðmóts á vinnustöð, þjóni eða reikniklasa.

Prófa spaCR
~~~~~~~~~~~

.. code-block:: bash

   conda create -n spacr python=3.12 -y
   conda activate spacr
   python -m pip install spacr
   spacr

Notaðu **Hlaða prófunargögnum…** í Import, Make Masks, Annotate eða á greiningarskjá til að sækja sýnigögn. Notaðu ``spacr-download`` í skipanalínu.

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

stuðlað (stabil)  framkvæmd (beta) 🔴 CPU stuðning aðeins

.. spacr-hardware-end


Setja upp spaCR
~~~~~~~~~~~~~~~

Skjáborðsforrit
-------------------

Þessir uppbyggingar búnir eigin Python. Conda er ekki nauðsynlegt.

.. spacr-installer-links-begin

|InstallerLinux| |InstallerMacOS| |InstallerWindows| |InstallerLegacy|

.. |InstallerWindows| image:: ../../../spacr/resources/icons/platforms/windows.png
   :width: 64
   :alt: Sækja spaCR 1.5.1.3 fyrir Windows 10/11
   :target: https://github.com/EinarOlafsson/spacr/releases/download/v1.5.1.3/spaCR-1.5.1.3-Windows-Online-Setup.exe
.. |InstallerMacOS| image:: ../../../spacr/resources/icons/platforms/macos.png
   :width: 64
   :alt: Sækja spaCR 1.5.1.3 fyrir macOS 11+ (Intel og Apple Silicon)
   :target: https://github.com/EinarOlafsson/spacr/releases/download/v1.5.1.3/spaCR-1.5.1.3-macOS-Universal-Online.pkg
.. |InstallerLinux| image:: ../../../spacr/resources/icons/platforms/linux.png
   :width: 64
   :alt: Sækja spaCR 1.5.1.3 fyrir 64-bita Linux
   :target: https://github.com/EinarOlafsson/spacr/releases/download/v1.5.1.3/spaCR-1.5.1.3-Linux-x86_64-Online.run
.. |InstallerLegacy| image:: ../../../spacr/resources/icons/platforms/legacy.png
   :width: 64
   :alt: Eldri spaCR-uppsetningarforrit
   :target: ../../source/installers.rst

.. spacr-installer-links-end

Í Linux skaltu gera skrána sem var sótt keyranlega og keyra hana:

.. code-block:: bash

   chmod +x SpaCR-*-Linux-x86_64-Online.run
   ./SpaCR-*-Linux-x86_64-Online.run

Á macOS, opna ``.pkg``. Núverandi beta er ekki notarið; ef Gatekeeper blokkir það, velja **System Settings → Privacy & Security → Open Anyway**.

Sjá `uppsetningarleiðbeiningarnar <../../source/installer_guide.rst>`_ fyrir leiðbeiningar um uppfærslu, fjarlægingu, notkun án nettengingar og bilanagreiningu. Í `kerfiskröfunum <../../source/system_requirements.rst>`_ eru ráðleggingar um vinnustöðvar og netþjóna og töflur um GPU-samhæfi.

Uppsetning frá PyPI
-------------------

Fyrir útgáfuna á PyPI skaltu setja spaCR upp með pip inni í Conda-umhverfi. Python 3.12 býður upp á mesta úrvalið af valfrjálsum vísindapökkum:

.. code-block:: bash

   conda create -n spacr python=3.12 -y
   conda activate spacr
   python -m pip install --upgrade pip
   python -m pip install spacr
   spacr

spaCR styður Python **3.9 til 3.14**, nema Python 3.14.1, sem torchvision útilokar. Mælt er með Linux fyrir þyngstu CUDA- og ROCm-verkflæðin; macOS og Windows eru einnig studd og nýta bæði GPU sín — macOS í gegnum Metal, sem nær yfir Apple Silicon og AMD-kortin í Intel-Mac-tölvum, og Windows í gegnum CUDA eða DirectML.

Hefðbundin uppsetning inniheldur Qt-skjáborðsviðmótið. Á þjóni, reikniklasa eða CI-keyrsluumhverfi má keyra skipanalínuverkflæðin án þess að opna það:

.. code-block:: bash

   python -m pip install spacr
   spacr-run --list

Opinlegri samsetningar eru settar sérstakt, t.d. ``spacr[zarr]``, ``spacr[omero]``,``spacr[napari]`` og ``spacr[czi,nd2,lif]``. Sjá `Uppsetningu leiðbeiningar <../../source/installer_guide.rst>`_ fyrir fullkomna útgáfur og Python-version samskipti tól.

Uppsetning með conda-forge
--------------------------

Opinberi conda-forge-pakkinn setur spaCR og nauðsynlegar einingar skjáborðsforritsins upp í virka umhverfinu:

.. code-block:: bash

   conda create -n spacr python=3.12 -y
   conda activate spacr
   conda install conda-forge::spacr
   spacr

Uppsetning með Docker
---------------------

Keyrðu skipanalínuvinnslur spaCR í gámi með útgefnum `Docker-myndum á GHCR <https://github.com/EinarOlafsson/spacr/pkgs/container/spacr>`_. Settu upp `Docker <https://docs.docker.com/get-started/get-docker/>`_ og birtu síðan lista yfir tiltækar vinnslur með þessari útgefnu CPU-mynd:

.. code-block:: bash

   docker run --rm ghcr.io/einarolafsson/spacr:1.5.1.0 spacr-run --list

Samsvarandi mynd fyrir NVIDIA-skjákort er ``ghcr.io/einarolafsson/spacr:1.5.1.0-cuda12.4``. Báðar myndirnar eru ætlaðar fyrir Linux-gáma á x86-64. Sjá `uppsetningarleiðbeiningar fyrir Docker <../../source/installer_guide.rst#container-images>`_ um forkröfur fyrir skjákort, tengingu gagna- og líkanamappa, stillingaskrár og heildarskipanir fyrir vinnslurnar.

Uppsetning frá frumkóða
-----------------------

Klónaðu kóðasafnið og settu það upp í breytanlegum ham, svo að vinnueintakið þitt *sé* uppsetti pakkinn og breytingar taki gildi án enduruppsetningar::

    git clone https://github.com/EinarOlafsson/spacr.git
    cd spacr
    conda create -n spacr python=3.12 -y
    conda activate spacr
    pip install -e .
    spacr

Þetta klónar ``main``, sjálfgefnu greinina, sem geymir nýjustu útgáfuna. Þróunin fer fram á ``nightly``; bættu ``--branch nightly`` við til að klóna hana í staðinn. Fyrir tiltekna útgáfu::

    git clone --branch v1.5.0.5 https://github.com/EinarOlafsson/spacr.git

Til að sækja síðari breytingar skaltu keyra inni í klóninu::

    git pull
    pip install -e .

Settu upp aftur þegar háðir pakkar eða inngangspunktar breytast. Breytingar á Python-kóða taka gildi beint; ``spacr-doctor`` sýnir hvaða uppsetning er virk.

Uppsetning frá frumkóða (létt)
------------------------------

Þátttakendur þurfa útgáfusöguna; veldu einn af kostunum hér fyrir neðan ef þú vilt aðeins keyra spaCR. Mælingar á ``nightly`` við ``05302fd5c`` þann 2026-10-07 með ``packaging/measure_clone_forms.sh``::

    # One commit instead of every version: 2048 MB downloaded, 140 s.
    # No history, so no git log, no git blame and no git bisect.
    # git pull still works, but stays shallow until git fetch --unshallow.
    git clone --depth 1 --branch nightly https://github.com/EinarOlafsson/spacr.git
    cd spacr && pip install -e .

    # Runtime files: 104 MB on disk, 6 s (Git: 38 MB; files: 66 MB).
    # No history, docs, tests, tools, features or example data.
    # --with-docs, --with-tests and --with-translations put those back;
    # --dir, --branch, --no-install and --help do the obvious things.
    # packaging/source_install_excludes.txt lists every skipped path.
    curl -fsSL https://raw.githubusercontent.com/EinarOlafsson/spacr/nightly/packaging/install_from_source.sh -o install_spacr.sh
    sh install_spacr.sh --branch nightly

Full klónun nightly sótti 9.25 GiB. Að bæta ``--filter=blob:none`` við grunna klónið minnkar ekki vinnueintakið: Git-hlutasafnið tekur enn 2032 MB. Hljóðlausar niðurhalssóknir eftir þörfum koma í veg fyrir fullkomna mælingu á heildarniðurhali. Útgáfustýrðu skrárnar í nightly taka 3416 MB í vinnueintakinu (mælt 2026-10-08), án Git-sögunnar. Stærð og tími niðurhals eru mismunandi eftir grein.


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
   spacr-download --list                      # what example data exists
   spacr-download measure annotate            # fetch example sets by name
   spacr-make-masks --folder DIR              # curate masks as a resumable queue
   spacr-make-masks --folder DIR --order easy --limit 50

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

spaCR-einingar
--------------

.. spacr-workflow-begin

Kjarni
^^^^^^

Core sequence from microscopy images through segmentation, measurements,
annotations, classification, barcode mapping and regression.

| |Module_mask|\ |Module_measure|\ |Module_annotate|\ |Module_classify_merged|\ |Module_map_barcodes|\ |Module_regression|

Gögn
^^^^

Import images and tables into spaCR projects and execute reproducible
multi-plate workflows.

| |Module_foreign|\ |Module_embeddings|\ |Module_run_compare|\ |Module_experiment_design|\ |Module_power|\ |Module_dose_response|
| |Module_qc_dashboard|

Verkfæri
^^^^^^^^

Point these at a project: edit masks by hand, stitch tiles, read an
embedding, draw a gate, build a plot, check quality.

| |Module_make_masks|\ |Module_align|\ |Module_umap|\ |Module_gate_editor|\ |Module_graph_builder|

Lífvera
^^^^^^^

Myndgreining fyrir tilteknar lífverur og megindlegar niðurstöður líffræðilegra prófana.

| |Module_toxoplasma|

.. |Module_mask| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/core/index.html#spacr.core.preprocess_generate_masks"><img src="../../../spacr/resources/icons/workflow/mask.png" width="16.0%" align="middle" alt="Opna API-skjölin fyrir Mask"></a>

.. |Module_measure| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/measure/index.html"><img src="../../../spacr/resources/icons/workflow/measure.png" width="16.0%" align="middle" alt="Opna API-skjölin fyrir Measure"></a>

.. |Module_annotate| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/qt/screens/annotate/index.html"><img src="../../../spacr/resources/icons/workflow/annotate.png" width="16.0%" align="middle" alt="Opna API-skjölin fyrir Annotate"></a>

.. |Module_classify_merged| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/classify/index.html"><img src="../../../spacr/resources/icons/workflow/classify_merged.png" width="16.0%" align="middle" alt="Opna API-skjölin fyrir Classify"></a>

.. |Module_map_barcodes| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/sequencing/index.html"><img src="../../../spacr/resources/icons/workflow/map_barcodes.png" width="16.0%" align="middle" alt="Opna API-skjölin fyrir Map Barcodes"></a>

.. |Module_regression| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/ml/index.html"><img src="../../../spacr/resources/icons/workflow/regression.png" width="16.0%" align="middle" alt="Opna API-skjölin fyrir Regression"></a>

.. |Module_foreign| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/foreign/index.html"><img src="../../../spacr/resources/icons/workflow/apps/foreign.png" width="16.0%" align="middle" alt="Opna API-skjölin fyrir Import"></a>

.. |Module_embeddings| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/qt/screens/embeddings/index.html"><img src="../../../spacr/resources/icons/workflow/apps/embeddings.png" width="16.0%" align="middle" alt="Opna API-skjölin fyrir Embeddings"></a>

.. |Module_run_compare| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/qt/screens/run_compare/index.html"><img src="../../../spacr/resources/icons/workflow/apps/run_compare.png" width="16.0%" align="middle" alt="Opna API-skjölin fyrir Run Compare"></a>

.. |Module_experiment_design| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/qt/screens/experiment_design/index.html"><img src="../../../spacr/resources/icons/workflow/apps/experiment_design.png" width="16.0%" align="middle" alt="Opna API-skjölin fyrir Experiment Design"></a>

.. |Module_power| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/qt/screens/power/index.html"><img src="../../../spacr/resources/icons/workflow/apps/power.png" width="16.0%" align="middle" alt="Opna API-skjölin fyrir Power / Design"></a>

.. |Module_dose_response| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/qt/screens/dose_response/index.html"><img src="../../../spacr/resources/icons/workflow/apps/dose_response.png" width="16.0%" align="middle" alt="Opna API-skjölin fyrir Dose–Response"></a>

.. |Module_qc_dashboard| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/qt/screens/qc_dashboard/index.html"><img src="../../../spacr/resources/icons/workflow/apps/qc_dashboard.png" width="16.0%" align="middle" alt="Opna API-skjölin fyrir QC"></a>

.. |Module_make_masks| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/qt/screens/make_masks/index.html"><img src="../../../spacr/resources/icons/workflow/apps/make_masks.png" width="16.0%" align="middle" alt="Opna API-skjölin fyrir Make Masks"></a>

.. |Module_align| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/align/index.html"><img src="../../../spacr/resources/icons/workflow/apps/align.png" width="16.0%" align="middle" alt="Opna API-skjölin fyrir Align &amp; Stitch"></a>

.. |Module_umap| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/core/index.html#spacr.core.generate_image_umap"><img src="../../../spacr/resources/icons/workflow/apps/umap.png" width="16.0%" align="middle" alt="Opna API-skjölin fyrir Image UMAP"></a>

.. |Module_gate_editor| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/qt/screens/gate_editor/index.html"><img src="../../../spacr/resources/icons/workflow/apps/gate_editor.png" width="16.0%" align="middle" alt="Opna API-skjölin fyrir Gate Editor"></a>

.. |Module_graph_builder| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/qt/screens/graph_builder/index.html"><img src="../../../spacr/resources/icons/workflow/apps/graph_builder.png" width="16.0%" align="middle" alt="Opna API-skjölin fyrir Graph Builder"></a>

.. |Module_toxoplasma| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/qt/screens/organism_screen/index.html#spacr-qt-screens-organism-screen-toxoplasma"><img src="../../../spacr/resources/icons/workflow/apps/toxoplasma.png" width="16.0%" align="middle" alt="Opna API-skjölin fyrir Toxoplasma"></a>

.. spacr-workflow-end

Allar einingar sem eiga reit á upphafsskjánum, í sömu röð og þar: fyrst sex einingar vinnslukeðjunnar, síðan hinar. Veldu reit til að opna API-síðu einingarinnar.

Öll aðrar auðlindir
~~~~~~~~~~~~~~~~~~~

- `Gagnvirkt kennsluefni <https://einarolafsson.github.io/spacr/tutorials/>`_ — leiðsögn um verkferla frá uppsetningu til rannsóknar á niðurstöðum skimunar.
- `Python API snemma byrjun <../../source/python_api.rst>`_ — hlaupa og staðfest pipelines frá skriptum, notebooks eða klúster.
- `Leikstjóri <../../source/features.rst>`_ — hæfileika, fullnægjandi og valfrjáls tengsl.
- `Heilluð API reference <https://einarolafsson.github.io/spacr/api/index.html>`_ — stuðlað innfangspunktur eftir verkefni, með fullkomna mótum tengslum einn hærra.
- `Sjálf tungumál og þýðingu leiðbeining <../../source/localization.rst>`_ — samskipti tungumál, kontext hjálp og vísindaleg útleiðslu.

Tungumál og þýðingar
~~~~~~~~~~~~~~~~~~~~~~

Viðmótið styður tíu tungumál í leiðsögn og stillingum. AI- og LIVE-stýringar, lýsingar á einingum og yfirfarin samhengishjálp eru einnig þýdd. Skiptu um tungumál undir **spaCR → Stillingar → Tungumál** án endurræsingar. Annálar, slóðir, gagnagrunnsgildi og mælingar eru aldrei þýdd; vísindaleg úttök haldast á viðurkenndri ensku. Sjá `stefnu um samhengishjálp <../../source/localization.rst#contextual-help>`_.

Nín ekki Engleska sögu eru stutt og tæknilegt endurskoðað í stað þess að lesa end til end af heimilum tala. `Sjáðu skammt <../REVIEW_SCOPE_2026-09-04.md>`_ skrá sem tungumálir hafa haft mannleg útgang og hvert orð eftir á Englesku eftir ákvörðun.

Hreyfimyndaleiðbeiningar fyrir stillingar
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Stillingar með sjónræna skýringu bjóða upp á **Animation**-stýringu í verkfæraábendingunni. Skoðaðu `myndasafn stillingahreyfimynda <https://einarolafsson.github.io/spacr/setting_animations.html>`_ eða `skrá stillingahreyfimynda <https://einarolafsson.github.io/spacr/api/spacr/setting_animations/index.html>`_.

Gögn
~~~~

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

Líkanasafn
~~~~~~~~~~

spaCR skipar listan af þjálfað mönnunum og snúa þeim á eftirspurn. Opna **Mönnun Zoo** frá heimaskjólum til að skoða og setja upp þá, eða nefna key í setningarfilenni - ``pathogen_model: toxoplasma_pv_v1`` - og mönnin er hlaðið niður og checksum-verified fyrsta sinn sem það er nauðsynlegt.

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
   * - ``toxoplasma_pv_v4``
       (Cellpose-SAM (cpsam_v2))
     - round 6's 556 curated fields plus 80 hand-curated fields of a new plate (Anu revision, Replication09182026 plate 1); 502 train / 123 validation / 11 test; training data at einarolafsson/toxoplasma-pv-segmentation-dataset-r7
     - F1 0.854 against stock cpsam_v2's 0.765 on the 11 anchor wells at IoU 0.5; AJI 0.776 against 0.505
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

Hvert dæmi yfir er metið á myndum sem myndavél hefur aldrei séð í æfingu.

**Trekkur** er hversu margir af hlutum mönnun er raunverulegur; **reikla** er hve mörg af raunverulegum hlutum það fann.

**F1** er tvö sameiginlegt, og er kvótað vegna þess að hver einn er trivially gamed - tala um einn ómeðlilegt plakk fyrir næstum fullkomna nákvæmni, eða hvert myrkur blob fyrir næstu fullkomnu endurskoðun. Það sem þú myndi helst missa af því að mæla, og fjölda er yfirleitt betra með yfirskoðun: plakkamálið var samþykkt á nákvóm 0.858 með endurskoða 0.811 yfir fyrri runda á 0.939 og 0.631.

**IoU** (intersection over union) er flatarmál skörunar milli spáðs hlutar og viðmiðunarhlutar, deilt með flatarmáli sammengis þeirra. Lesið gildi ásamt viðmiðunarmörkum: „F1 0.864 við IoU 0.5“ telur frymisbólu fundna þegar skörunin nær að minnsta kosti helmingi flatarmáls sammengisins.

**mAP50** og **map50-95** eru með uppgötvuna. fyrri spyr hvort bólkurnar voru fundið; annar endurtekur það yfir tíu þremur frá 0.5 til 0.95, þannig að það spyr einnig hversu þreyttur hver boksi er þreytur.

**Cross-validated**, með **SD**, þýðir að skólan er miðjan þremur rún á mismunandi rúnum og SD er hversu langt þeir flytja út.

Líkön eru hýst á eigin Hugging Face-reikningi hvers höfundar; ``spacr.model_zoo.publish_model`` hleður upp líkani og prentar færsluna sem bæta á í líkanaskrána.


Greining á afköstum
----------------------

Búðu til vélbúnaðarskýrslu og hengdu hana við mál um afköst::

    python tools/spacr_hardware_report.py

Spara til ``~/.spacr/reports`` og trúa leiðinni. ``--quick`` skiptir lengri skilyrði; ``--out PATH`` setur staðsetningu.

Les engin verkefnisgögn. Mælir tímann sem innflutningur, töluleg söfn, gerð glugga og hreyfimyndir taka og greinir frá x86_64-hermun á Apple Silicon og BLAS-útfærslu NumPy.

Orðlinna reference
----------------------

Öll beint hér að neðan er sett með ``pip install spacr``. Allir samþykkir ``--help``.

Að byrja við umsókn
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   spacr              # the desktop application
   spacr-tutorial     # the interactive tutorial library
   spacr-server       # no first-run setup screen, for unattended launches

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
~~~~~~~~~~~~~~~~~~~~~~~~

Sendu villutilkynningar og afmarkaðar óskir um eiginleika í gegnum `GitHub-mál <https://github.com/EinarOlafsson/spacr/issues>`_. Þegar bilun er tilkynnt skal tilgreina útgáfu spaCR, stýrikerfi, útgáfu Python, stillingar einingarinnar og viðeigandi hluta úr annálnum. ``spacr-doctor`` safnar flestum þessara upplýsinga; láttu vélbúnaðarskýrsluna fylgja þegar tilkynnt er um afkastavandamál.

Leyfi
~~~~~~~~~

spaCR er frelsað undir `BSD 3-Klausur leyfi <https://github.com/EinarOlafsson/spacr/blob/main/LICENSE>`_.

Ef spaCR hjálpaði að útgáfa verk, er nefndur verðmæt og er ekki skilyrði fyrir leyfi — sjá `Tilvísun í spaCR`_ hér neðan.

Kennsluefni
~~~~~~~~~~~

`Gagnvirka spaCR-kennslusafnið <https://einarolafsson.github.io/spacr/tutorials/>`_ leiðir þig í gegnum uppsetningu og notkun eininganna. Hver kennslustund tilgreinir hvaða upplestur og tungumál eru í boði.

Tilvísun í spaCR
~~~~~~~~~~~~~~~~

Ef spaCR nýtist við rannsóknina skaltu vitna í:

Olafsson EB, *et al.* A sameiginlegur myndbönd sem er bastir á CRISPR skrefinn skilur EAF1 sem *T. gondii* modulator ESCRT subversion.

`Bioregl fyrirframskrift <https://www.biorxiv.org/content/10.64898/2026.07.08.737057v1>`_ · `Programvarparkíf <https://doi.org/10.5281/zenodo.21343316>`_

Aðrir verk sem heitir spaCR
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. spacr-citing-papers-begin

* `Metabolic adaptability og næringssvopning í Toxoplasma gondii: innsýn frá innihaldskortum mutantum. <https://journals.asm.org/doi/full/10.1128/msphere.01011-24>`_
* `IRE1α stuðlar að fagosomal kalsiumflokk til að bæta makrofága svítilverkefni. <https://www.cell.com/cell-reports/fulltext/S2211-1247(25)00465-6>`_
* `Toxoplasma GRA8 er nauðsynlegur fyrir aðstoðproteín ALG-2 og er mikilvægt fyrir efnahagsleikum parasita. <https://www.biorxiv.org/content/10.64898/2026.07.20.739547v1.abstract>`_
* `spaCR: Spatial fenotypeanalyse af CRISPR-Cas9 skrefum (preprint útgáfa 1). <https://www.researchsquare.com/article/rs-7368254/v1>`_

.. spacr-citing-papers-end

Þakkir
~~~~~~~~~~~~~~~

spaCR byggir á opnum vísindahugbúnaði, meðal annars NumPy, pandas, scikit-image, scikit-learn, Cellpose, PyTorch og Qt. Sjá `upplýsingar um þýðingarlíkön <../TRANSLATION_MODELS.md>`_ fyrir líkönin sem voru notuð við gerð fjöltyngdra skjala og viðmótsskráa.
