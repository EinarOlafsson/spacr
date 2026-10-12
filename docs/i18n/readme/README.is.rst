|Website| |Platforms| |Python| |Qt| |Test counts| |Coverage| |Issues| |Conda| |PyPI| |Docker| |PyPI Downloads| |Docs| |Tutorials| |Preprint| |DOI| |Cite| |License| |PyPI rank|

.. |Website| image:: https://img.shields.io/badge/Website-spaCR-0A7BBB
   :target: https://einarolafsson.github.io/projects/spacr/
   :alt: Vefsíða spaCR

.. |Docs| image:: https://img.shields.io/github/actions/workflow/status/EinarOlafsson/spacr/pages%2Fpages-build-deployment?label=API&logo=readthedocs&logoColor=white
   :target: https://einarolafsson.github.io/spacr/
   :alt: API-skjöl

.. |Tutorials| image:: https://img.shields.io/badge/Tutorials-4A9EFF
   :target: https://einarolafsson.github.io/spacr/tutorials/
   :alt: Gagnvirkt kennsluefni

.. |PyPI| image:: https://img.shields.io/pypi/v/spacr?label=PyPI&logo=pypi&logoColor=white
   :target: https://pypi.org/project/spacr/
   :alt: PyPI-útgáfa

.. |Python| image:: https://img.shields.io/badge/Python-3.9%E2%80%933.14-3776AB?logo=python&logoColor=white
   :target: https://pypi.org/project/spacr/
   :alt: Python 3.9 til 3.14

.. |Coverage| image:: https://img.shields.io/codecov/c/github/EinarOlafsson/spacr/nightly?logo=codecov&logoColor=white
   :target: https://app.codecov.io/github/EinarOlafsson/spacr
   :alt: Prófunarþekja á Codecov

.. |Test counts| image:: https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2FEinarOlafsson%2Fspacr%2Fnightly%2Fdocs%2Fsource%2F_static%2Ftest-counts.json&cacheSeconds=300&logo=pytest&logoColor=white
   :target: https://github.com/EinarOlafsson/spacr/actions/workflows/tests.yml
   :alt: Prófunarsafn

.. |Qt| image:: https://img.shields.io/badge/GUI-Qt-41CD52?logo=qt&logoColor=white
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/index.html#module-spacr.qt
   :alt: Qt-viðmót

.. |Issues| image:: https://img.shields.io/github/issues/EinarOlafsson/spacr?logo=github
   :target: https://github.com/EinarOlafsson/spacr/issues
   :alt: GitHub-mál

.. |License| image:: https://img.shields.io/badge/License-BSD--3-blue?logo=opensourceinitiative&logoColor=white
   :target: https://github.com/EinarOlafsson/spacr/blob/main/LICENSE
   :alt: BSD 3-Clause-leyfi

.. |Preprint| image:: https://img.shields.io/badge/bioRxiv-BF2636
   :target: https://www.biorxiv.org/content/10.64898/2026.07.08.737057v1
   :alt: bioRxiv-forprentun

.. |DOI| image:: https://img.shields.io/badge/DOI-blue?logo=zenodo&logoColor=white
   :target: https://doi.org/10.5281/zenodo.21343316
   :alt: Zenodo DOI

.. |Conda| image:: https://img.shields.io/conda/vn/conda-forge/spacr?label=Conda&logo=anaconda&logoColor=white
   :target: https://anaconda.org/conda-forge/spacr
   :alt: conda-forge-útgáfa

.. |Release date| image:: https://anaconda.org/conda-forge/spacr/badges/latest_release_date.svg
   :target: https://anaconda.org/conda-forge/spacr
   :alt: Dagsetning nýjustu útgáfu á conda-forge

.. |Docker| image:: https://img.shields.io/badge/Docker-GHCR-2496ED?logo=docker&logoColor=white
   :target: https://github.com/EinarOlafsson/spacr/pkgs/container/spacr
   :alt: Docker-myndir á GHCR

.. |PyPI Downloads| image:: https://img.shields.io/pepy/dt/spacr?label=downloads&logo=pypi&logoColor=white
   :target: https://pepy.tech/projects/spacr
   :alt: PyPI-niðurhal

.. |Platforms| image:: https://img.shields.io/badge/Linux%20OSX%20Win-lightgrey?logo=linux&logoColor=white
   :target: https://github.com/EinarOlafsson/spacr/blob/nightly/docs/source/installers.rst
   :alt: Linux, macOS og Windows

.. |Cite| image:: https://img.shields.io/badge/Cite-8A2BE2
   :target: https://github.com/EinarOlafsson/spacr/blob/main/CITATION.cff
   :alt: Vitna í spaCR

.. |PyPI rank| image:: https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fsql-clickhouse.clickhouse.com%2F%3Fuser%3Ddemo%26param_package_name%3Dspacr%26param_days%3D30%26query%3DWITH%2B%2528%2BSELECT%2Bsum%2528count%2529%2BFROM%2Bpypi.pypi_downloads_per_day%2BWHERE%2Bproject%2B%253D%2B%257Bpackage_name%253AString%257D%2BAND%2Bdate%2B%253E%253D%2BtoDate%2528now%2528%2527UTC%2527%2529%2529%2B-%2B%257Bdays%253AUInt16%257D%2BAND%2Bdate%2B%253C%2BtoDate%2528now%2528%2527UTC%2527%2529%2529%2B%2529%2BAS%2Bdownloads%2BSELECT%2Bdownloads%2BAS%2Bpackage_downloads%252C%2BcountIf%2528n%2B%253E%253D%2Bdownloads%2529%2BAS%2Brank%252C%2Bcount%2528%2529%2BAS%2Btotal_packages%252C%2B100.0%2B%252A%2Brank%2B%252F%2BnullIf%2528total_packages%252C%2B0%2529%2BAS%2Bpercentile%252C%2Bif%2528%2Btotal_packages%2B%253D%2B0%2BOR%2Bdownloads%2B%253D%2B0%252C%2B%2527no%2Bdata%2527%252C%2Bconcat%2528%2B%2527top%2B%2527%252C%2BtoString%2528ceil%25281000.0%2B%252A%2Brank%2B%252F%2BnullIf%2528total_packages%252C%2B0%2529%2529%2B%252F%2B10%2529%252C%2B%2527%2525%2527%2B%2529%2B%2529%2BAS%2Bmessage%2BFROM%2B%2528%2BSELECT%2Bproject%252C%2Bsum%2528count%2529%2BAS%2Bn%2BFROM%2Bpypi.pypi_downloads_per_day%2BWHERE%2Bdate%2B%253E%253D%2BtoDate%2528now%2528%2527UTC%2527%2529%2529%2B-%2B%257Bdays%253AUInt16%257D%2BAND%2Bdate%2B%253C%2BtoDate%2528now%2528%2527UTC%2527%2529%2529%2BGROUP%2BBY%2Bproject%2B%2529%2BFORMAT%2BJSON&query=%24.data%5B0%5D.message&label=Rank&color=brightgreen&logo=pypi&logoColor=white&cacheSeconds=86400
   :target: https://clickpy.clickhouse.com/dashboard/spacr
   :alt: Röðun spaCR eftir PyPI-niðurhali síðustu 30 heilu dagana

.. image:: ../../source/_static/deck/slides/slide_01.jpg
   :alt: spaCR
   :width: 920
   :target: https://einarolafsson.github.io/spacr/_static/deck/

| |intro-1|\ |intro-2|

.. |intro-1| image:: ../../readme_images/is/intro-1.svg
   :alt: ← Til baka
   :width: 49.5%
   :target: ../../source/_static/deck/pages/57.md

.. |intro-2| image:: ../../readme_images/is/intro-2.svg
   :alt: Næsta →
   :width: 49.5%
   :target: ../../source/_static/deck/pages/02.md

.. spacr-language-picker-begin

.. image:: ../../readme_images/is/generated-language-picker.svg
   :alt: Tungumál: Íslenska
   :width: 32.8%
   :target: README.md

.. spacr-language-picker-end

.. image:: ../../readme_images/is/spacr.svg
   :alt: spaCR. Rýmisbundin svipgerðargreining á CRISPR-skimunum. spaCR aðgreinir og mælir stakar frumur í smásjármyndum, samþættir svipgerðir einstakra viðfanga við magn leiðarsameinda sem fæst úr raðgreiningu og metur hvaða gen tengjast svipgerðarbreytingum. Út frá plötumyndum og FASTQ-röðum býr það til mælingar fyrir hvert viðfang, þjálfaða flokkara, áhrifamat fyrir hverja leiðarsameind og hvert gen og forgangsraðaðan lista yfir niðurstöður. Segmingu, mæling, notkun og flokksmiðju mótmælur virkar einnig án sekkunararms. Make Masks leiðréttir skiptingargrímur og merkir sjálfstæða rétthyrninga með flokkamerkjum með Box verkfærinu fyrir YOLO-útflutning. Rammarnir halda eigin merkjum og sögu án þess að breyta upprunamyndum eða grímum. Hvert verkfæri er útskýrt í eiginleikahandbókinni. Myndir, grímur og myndúrklippur eru skrár á diski. Mælingar, merkingar og spár líkana eru töflur í SQLite-gagnagrunni verkefnisins, measurements.db. Keyrist sem skjáborðsforrit eða án grafísks viðmóts á vinnustöð, þjóni eða reikniklasa.
   :width: 100%
   :target: ../../source/features.rst

.. image:: ../../readme_images/is/try-spacr.svg
   :alt: Prófa spaCR.
   :width: 100%

.. code-block:: bash

   conda create -n spacr python=3.12 -y
   conda activate spacr
   python -m pip install spacr
   spacr

.. image:: ../../readme_images/is/try-spacr-2.svg
   :alt: Notaðu Hlaða prófunargögnum… í Import, Make Masks, Annotate eða á greiningarskjá til að sækja sýnigögn. Notaðu spacr-download í skipanalínu.
   :width: 100%

.. image:: ../../readme_images/is/hardware-support.svg
   :alt: Hardware aðstoð.
   :width: 100%

.. spacr-hardware-begin

.. image:: ../../readme_images/is/generated-hardware-1.svg
   :alt: NVIDIA (CUDA): Cellpose 4 GPU, Torch GPU, UMAP / clustering GPU. AMD on Linux (ROCm): Cellpose 4 GPU, Torch GPU, UMAP / clustering CPU. AMD in an Intel Mac (Metal): Cellpose 4 GPU, Torch GPU, UMAP / clustering CPU. Apple Silicon (Metal): Cellpose 4 GPU, Torch GPU, UMAP / clustering CPU. Intel Arc/Xe (XPU): Cellpose 4 GPU, Torch GPU, UMAP / clustering CPU. No GPU: Cellpose 4 CPU, Torch CPU, UMAP / clustering CPU. stutt (stöðugt) útfært (beta) aðeins CPU-stuðningur.
   :width: 100%

.. spacr-hardware-end

.. image:: ../../readme_images/is/install-spacr.svg
   :alt: Setja upp spaCR. Veldu uppsetningarleið: Skjáborðsforrit: uppsetningarforrit fyrir Linux, macOS eða Windows með eigið Python. pip: python -m pip install spacr í Conda-umhverfi. conda-forge: conda install conda-forge::spacr í Conda-umhverfi. Docker: skipanalínuvinnslurnar í Linux-gámi, úr CPU-myndinni eða myndinni fyrir NVIDIA-skjákort:
   :width: 100%

.. code-block:: bash

   docker run --rm ghcr.io/einarolafsson/spacr:1.5.1.4 spacr-run --list
   docker run --rm --gpus all ghcr.io/einarolafsson/spacr:1.5.1.4-cuda12.4 spacr-run --list

.. image:: ../../readme_images/is/install-spacr-2.svg
   :alt: Docker-leiðbeiningarnar fjalla um uppsetningu skjákorts, tengingu gagnamappa og skipanir fyrir vinnslurnar.
   :width: 100%
   :target: ../../source/installer_guide.rst#container-images

.. image:: ../../readme_images/is/desktop-application.svg
   :alt: Skjáborðsforrit. Þessir uppbyggingar búnir eigin Python. Conda er ekki nauðsynlegt.
   :width: 100%

.. spacr-installer-links-begin

|InstallerLinux| |InstallerMacOS| |InstallerWindows|

.. |InstallerWindows| image:: ../../../spacr/resources/icons/platforms/windows.png
   :width: 64
   :alt: Sækja spaCR 1.5.1.4 fyrir Windows 10/11
   :target: https://github.com/EinarOlafsson/spacr/releases/download/v1.5.1.4/spaCR-1.5.1.4-Windows-Online-Setup.exe
.. |InstallerMacOS| image:: ../../../spacr/resources/icons/platforms/macos.png
   :width: 64
   :alt: Sækja spaCR 1.5.1.4 fyrir macOS 11+ (Intel og Apple Silicon)
   :target: https://github.com/EinarOlafsson/spacr/releases/download/v1.5.1.4/spaCR-1.5.1.4-macOS-Universal-Online.pkg
.. |InstallerLinux| image:: ../../../spacr/resources/icons/platforms/linux.png
   :width: 64
   :alt: Sækja spaCR 1.5.1.4 fyrir 64-bita Linux
   :target: https://github.com/EinarOlafsson/spacr/releases/download/v1.5.1.4/spaCR-1.5.1.4-Linux-x86_64-Online.run

.. spacr-installer-links-end

.. image:: ../../readme_images/is/desktop-application-2.svg
   :alt: Í Linux skaltu gera skrána sem var sótt keyranlega og keyra hana:
   :width: 100%

.. code-block:: bash

   chmod +x SpaCR-*-Linux-x86_64-Online.run
   ./SpaCR-*-Linux-x86_64-Online.run

.. image:: ../../readme_images/is/desktop-application-3.svg
   :alt: Á macOS, opna .pkg. Núverandi beta er ekki notarið; ef Gatekeeper blokkir það, velja System Settings → Privacy & Security → Open Anyway. Sjá uppsetningarleiðbeiningarnar fyrir leiðbeiningar um uppfærslu, fjarlægingu, notkun án nettengingar og bilanagreiningu. Í kerfiskröfunum eru ráðleggingar um vinnustöðvar og netþjóna og töflur um GPU-samhæfi.
   :width: 100%

| |desktop-application-3-link-1|\ |desktop-application-3-link-2|

.. |desktop-application-3-link-1| image:: ../../readme_images/is/desktop-application-3-link-1.svg
   :alt: uppsetningarleiðbeiningarnar
   :width: 49.5%
   :target: ../../source/installer_guide.rst

.. |desktop-application-3-link-2| image:: ../../readme_images/is/desktop-application-3-link-2.svg
   :alt: kerfiskröfunum
   :width: 49.5%
   :target: ../../source/system_requirements.rst

.. image:: ../../readme_images/is/pypi-installation.svg
   :alt: Uppsetning frá PyPI. Fyrir útgáfuna á PyPI skaltu setja spaCR upp með pip inni í Conda-umhverfi. Python 3.12 býður upp á mesta úrvalið af valfrjálsum vísindapökkum:
   :width: 100%

.. code-block:: bash

   conda create -n spacr python=3.12 -y
   conda activate spacr
   python -m pip install --upgrade pip
   python -m pip install spacr
   spacr

.. image:: ../../readme_images/is/pypi-installation-2.svg
   :alt: spaCR styður Python 3.9 til 3.14, nema Python 3.14.1, sem torchvision útilokar. Mælt er með Linux fyrir þyngstu CUDA- og ROCm-verkflæðin; macOS og Windows eru einnig studd og nýta bæði GPU sín — macOS í gegnum Metal, sem nær yfir Apple Silicon og AMD-kortin í Intel-Mac-tölvum, og Windows í gegnum CUDA eða DirectML. Hefðbundin uppsetning inniheldur Qt-skjáborðsviðmótið. Á þjóni, reikniklasa eða CI-keyrsluumhverfi má keyra skipanalínuverkflæðin án þess að opna það:
   :width: 100%

.. code-block:: bash

   python -m pip install spacr
   spacr-run --list

.. image:: ../../readme_images/is/pypi-installation-3.svg
   :alt: Opinlegri samsetningar eru settar sérstakt, t.d. spacr[zarr], spacr[omero],spacr[napari] og spacr[czi,nd2,lif]. Sjá Uppsetningu leiðbeiningar fyrir fullkomna útgáfur og Python-version samskipti tól.
   :width: 100%
   :target: ../../source/installer_guide.rst

.. image:: ../../readme_images/is/conda-forge-installation.svg
   :alt: Uppsetning með conda-forge. Opinberi conda-forge-pakkinn setur spaCR og nauðsynlegar einingar skjáborðsforritsins upp í virka umhverfinu:
   :width: 100%

.. code-block:: bash

   conda create -n spacr python=3.12 -y
   conda activate spacr
   conda install conda-forge::spacr
   spacr

.. image:: ../../readme_images/is/install-from-source.svg
   :alt: Uppsetning frá frumkóða. Klónaðu kóðasafnið og settu það upp í breytanlegum ham, svo að vinnueintakið þitt sé uppsetti pakkinn og breytingar taki gildi án enduruppsetningar:
   :width: 100%

.. code-block:: bash

   git clone https://github.com/EinarOlafsson/spacr.git
   cd spacr
   conda create -n spacr python=3.12 -y
   conda activate spacr
   pip install -e .
   spacr

.. image:: ../../readme_images/is/install-from-source-2.svg
   :alt: Þetta klónar main, sjálfgefnu greinina, sem geymir nýjustu útgáfuna. Þróunin fer fram á nightly; bættu --branch nightly við til að klóna hana í staðinn. Fyrir tiltekna útgáfu:
   :width: 100%

.. code-block:: bash

   git clone --branch v1.5.0.5 https://github.com/EinarOlafsson/spacr.git

.. image:: ../../readme_images/is/install-from-source-3.svg
   :alt: Til að sækja síðari breytingar skaltu keyra inni í klóninu:
   :width: 100%

.. code-block:: bash

   git pull
   pip install -e .

.. image:: ../../readme_images/is/install-from-source-4.svg
   :alt: Settu upp aftur þegar háðir pakkar eða inngangspunktar breytast. Breytingar á Python-kóða taka gildi beint; spacr-doctor sýnir hvaða uppsetning er virk.
   :width: 100%

.. image:: ../../readme_images/is/install-from-source-light.svg
   :alt: Uppsetning frá frumkóða (létt). Þátttakendur þurfa útgáfusöguna; veldu einn af kostunum hér fyrir neðan ef þú vilt aðeins keyra spaCR. Mælingar á nightly við 05302fd5c þann 2026-10-07 með packaging/measure_clone_forms.sh:
   :width: 100%

.. code-block:: bash

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

.. image:: ../../readme_images/is/install-from-source-light-2.svg
   :alt: Full klónun nightly sótti 9.25 GiB. Að bæta --filter=blob:none við grunna klónið minnkar ekki vinnueintakið: Git-hlutasafnið tekur enn 2032 MB. Git tilkynnir ekki um hluti sem eru sóttir síðar eftir þörfum, svo heildarniðurhal var ekki mælt. Útgáfustýrðu skrárnar í nightly taka 3416 MB í vinnueintakinu (mælt 2026-10-08), án Git-sögunnar. Stærð og tími niðurhals eru mismunandi eftir grein.
   :width: 100%

.. image:: ../../readme_images/is/command-line-entry-points.svg
   :alt: Skipanalínuskipanir.
   :width: 100%

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

.. image:: ../../readme_images/is/command-line-entry-points-2.svg
   :alt: spacr-run --list listar einingar sem hafa skipanalínuinngang til keyrslu án grafísks viðmóts. Einingum fyrir merkingu, gagnayfirferð, samanburð og könnun sem eingöngu eru í GUI er sleppt.
   :width: 100%

.. image:: ../../readme_images/is/core-workflow.svg
   :alt: Kjarnaverkflæði. Aðalvinnuflæðið samanstendur af sex einingum: Mask hlutgreinir frumur, frumukjarna, sýkla og frumulíffæri með Cellpose. Measure skrifar lögunar-, styrkleika-, áferðar-, rúm- og samstaðsetningareiginleika ásamt myndúrklippum viðfanga í SQLite. Annotate merkir myndúrklippur í lyklaborðsstýrðu hnitaneti og styður biðraðir virks náms. Classify þjálfar líkön byggð á myndum eða mælingum og skráir frammistöðu á fráteknum gögnum með hverjum varðpunkti. Map Barcodes varpar FASTQ-lestrum á brunna og gRNA og veitir gæðamat fyrir magn, árekstra og þekju. Regression metur áhrif leiðarsameinda, gena, skilyrða og viðmiða með líkanafjölskyldum sem henta samfelldum gildum, hlutföllum og talningum.
   :width: 100%

.. image:: ../../readme_images/is/spacr-modules.svg
   :alt: spaCR-einingar.
   :width: 100%

.. spacr-workflow-begin

.. image:: ../../readme_images/is/generated-workflow-1.svg
   :alt: Kjarni. Core sequence from microscopy images through segmentation, measurements, annotations, classification, barcode mapping and regression.
   :width: 100%

| |Module_mask|\ |Module_measure|\ |Module_annotate|\ |Module_classify_merged|\ |Module_map_barcodes|\ |Module_regression|

.. image:: ../../readme_images/is/generated-workflow-2.svg
   :alt: Gögn. Import images and tables into spaCR projects and execute reproducible multi-plate workflows.
   :width: 100%

| |Module_foreign|\ |Module_embeddings|\ |Module_run_compare|\ |Module_experiment_design|\ |Module_power|\ |Module_dose_response|
| |Module_qc_dashboard|

.. image:: ../../readme_images/is/generated-workflow-3.svg
   :alt: Verkfæri. Point these at a project: edit masks by hand, stitch tiles, read an embedding, draw a gate, build a plot, check quality.
   :width: 100%

| |Module_make_masks|\ |Module_align|\ |Module_umap|\ |Module_gate_editor|\ |Module_graph_builder|

.. image:: ../../readme_images/is/generated-workflow-4.svg
   :alt: Lífvera. Myndgreining fyrir tilteknar lífverur og megindlegar niðurstöður líffræðilegra prófana.
   :width: 100%

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

.. image:: ../../readme_images/is/spacr-modules-2.svg
   :alt: Allar einingar sem eiga reit á upphafsskjánum, í sömu röð og þar: fyrst sex einingar vinnslukeðjunnar, síðan hinar. Veldu reit til að opna API-síðu einingarinnar.
   :width: 100%

.. image:: ../../readme_images/is/other-resources.svg
   :alt: Öll aðrar auðlindir.
   :width: 100%

.. image:: ../../readme_images/is/other-resources-2-1.svg
   :alt: Gagnvirkt kennsluefni — leiðsögn um verkferla frá uppsetningu til rannsóknar á niðurstöðum skimunar.
   :width: 100%
   :target: https://einarolafsson.github.io/spacr/tutorials/

.. image:: ../../readme_images/is/other-resources-2-2.svg
   :alt: Python API snemma byrjun — hlaupa og staðfest pipelines frá skriptum, notebooks eða klúster.
   :width: 100%
   :target: ../../source/python_api.rst

.. image:: ../../readme_images/is/other-resources-2-3.svg
   :alt: Leikstjóri — hæfileika, fullnægjandi og valfrjáls tengsl.
   :width: 100%
   :target: ../../source/features.rst

.. image:: ../../readme_images/is/other-resources-2-4.svg
   :alt: Heilluð API reference — stuðlað innfangspunktur eftir verkefni, með fullkomna mótum tengslum einn hærra.
   :width: 100%
   :target: https://einarolafsson.github.io/spacr/api/index.html

.. image:: ../../readme_images/is/other-resources-2-5.svg
   :alt: Sjálf tungumál og þýðingu leiðbeining — samskipti tungumál, kontext hjálp og vísindaleg útleiðslu.
   :width: 100%
   :target: ../../source/localization.rst

.. image:: ../../readme_images/is/language-translation.svg
   :alt: Tungumál og þýðingar. Viðmótið styður tíu tungumál í leiðsögn og stillingum. AI- og LIVE-stýringar, lýsingar á einingum og yfirfarin samhengishjálp eru einnig þýdd. Skiptu um tungumál undir spaCR → Stillingar → Tungumál án endurræsingar. Annálar, slóðir, gagnagrunnsgildi og mælingar eru aldrei þýdd; vísindaleg úttök haldast á viðurkenndri ensku. Sjá stefnu um samhengishjálp.
   :width: 100%
   :target: ../../source/localization.rst#contextual-help

.. image:: ../../readme_images/is/language-translation-2.svg
   :alt: Nín ekki Engleska sögu eru stutt og tæknilegt endurskoðað í stað þess að lesa end til end af heimilum tala. Sjáðu skammt skrá sem tungumálir hafa haft mannleg útgang og hvert orð eftir á Englesku eftir ákvörðun.
   :width: 100%
   :target: ../REVIEW_SCOPE_2026-09-04.md

.. image:: ../../readme_images/is/animated-setting-guidance.svg
   :alt: Hreyfimyndaleiðbeiningar fyrir stillingar. Stillingar með sjónræna skýringu bjóða upp á Animation-stýringu í verkfæraábendingunni. Skoðaðu myndasafn stillingahreyfimynda eða skrá stillingahreyfimynda.
   :width: 100%

| |animated-setting-guidance-link-1|\ |animated-setting-guidance-link-2|

.. |animated-setting-guidance-link-1| image:: ../../readme_images/is/animated-setting-guidance-link-1.svg
   :alt: myndasafn stillingahreyfimynda
   :width: 49.5%
   :target: https://einarolafsson.github.io/spacr/setting_animations.html

.. |animated-setting-guidance-link-2| image:: ../../readme_images/is/animated-setting-guidance-link-2.svg
   :alt: skrá stillingahreyfimynda
   :width: 49.5%
   :target: https://einarolafsson.github.io/spacr/api/spacr/setting_animations/index.html

.. image:: ../../readme_images/is/data.svg
   :alt: Gögn. Viðmiðunargagnasöfn.
   :width: 100%

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

.. image:: ../../readme_images/is/model-zoo.svg
   :alt: Líkanasafn. spaCR skipar listan af þjálfað mönnunum og snúa þeim á eftirspurn. Opna Mönnun Zoo frá heimaskjólum til að skoða og setja upp þá, eða nefna key í setningarfilenni - pathogen_model: toxoplasma_pv_v1 - og mönnin er hlaðið niður og checksum-verified fyrsta sinn sem það er nauðsynlegt.
   :width: 100%

.. spacr-model-zoo-begin

| |zoo_heading|
| |zoo_card_toxoplasma_pv_v4|\ |zoo_card_toxoplasma_plaque_v4|
| |zoo_card_toxoplasma_well_detector_v3|\ |zoo_card_toxoplasma_from_cellmask_v1|
| |zoo_card_live_cell_v1|\ |zoo_card_nuclei_from_cellmask_v1|
| |zoo_card_cell_from_hoechst_v1|\ |zoo_card_toxoplasma_from_hoechst_v1|

.. |zoo_heading| image:: ../../models/cards/heading.svg
   :alt: Default models · held-out scores
   :target: ../../models/README.md
   :width: 99.5%

.. |zoo_card_toxoplasma_pv_v4| image:: ../../models/cards/toxoplasma_pv_v4.svg
   :alt: Score card of Parasite vacuoles v4 (toxoplasma_pv_v4, default): F1 0.761 on 209 held-out fields against 0.377 for stock cpsam_v2. F1 0.761 against stock cpsam_v2's 0.377 on the 209 strict hold-out fields at IoU 0.5; AJI 0.582 against 0.040.
   :target: ../../models/toxoplasma_pv_v4.md
   :width: 49.5%

.. |zoo_card_toxoplasma_plaque_v4| image:: ../../models/cards/toxoplasma_plaque_v4.svg
   :alt: Score card of Plaques v4 (toxoplasma_plaque_v4, default): F1 0.765 on 329 held-out fields against 0.338 for stock cpsam_v2. F1 0.765 against stock cpsam_v2's 0.338 on the 329 strict hold-out fields at IoU 0.5; AJI 0.662 against 0.261.
   :target: ../../models/toxoplasma_plaque_v4.md
   :width: 49.5%

.. |zoo_card_toxoplasma_well_detector_v3| image:: ../../models/cards/toxoplasma_well_detector_v3.svg
   :alt: Score card of Plaque wells v3 (toxoplasma_well_detector_v3, default): mAP50 0.994 on 73 held-out images. Stock was not evaluated in this run; see the named incumbent comparison on the model card.
   :target: ../../models/toxoplasma_well_detector_v3.md
   :width: 49.5%

.. |zoo_card_toxoplasma_from_cellmask_v1| image:: ../../models/cards/toxoplasma_from_cellmask_v1.svg
   :alt: Score card of Vacuoles from cell mask v1 (toxoplasma_from_cellmask_v1, default): F1 0.606 on 463 held-out fields against 0.021 for stock cpsam_v2. F1 0.606 against 0.021 for stock cpsam_v2 on 463 well-grouped held-out fields, at IoU 0.5.
   :target: ../../models/toxoplasma_from_cellmask_v1.md
   :width: 49.5%

.. |zoo_card_live_cell_v1| image:: ../../models/cards/live_cell_v1.svg
   :alt: Score card of Live cells v1 (live_cell_v1, default): F1 0.694 on 2,199 held-out fields against 0.738 for stock cpsam_v2. On the datasets stock cpsam_v2 never trained on, F1 0.960 against 0.885 at IoU 0.5; over all 2,199 test fields, 0.694 against 0.738, because stock trained on LIVECell and YeaZ and wins on LIVECell.
   :target: ../../models/live_cell_v1.md
   :width: 49.5%

.. |zoo_card_nuclei_from_cellmask_v1| image:: ../../models/cards/nuclei_from_cellmask_v1.svg
   :alt: Score card of Nuclei from cell mask v1 (nuclei_from_cellmask_v1, default): F1 0.888 on 453 held-out fields against 0.201 for stock cpsam_v2. F1 0.888 against 0.201 for stock cpsam_v2 on 453 well-grouped held-out fields, at IoU 0.5.
   :target: ../../models/nuclei_from_cellmask_v1.md
   :width: 49.5%

.. |zoo_card_cell_from_hoechst_v1| image:: ../../models/cards/cell_from_hoechst_v1.svg
   :alt: Score card of Cells from Hoechst v1 (cell_from_hoechst_v1, default): F1 0.870 on 451 held-out fields against 0.301 for stock cpsam_v2. F1 0.870 against stock cpsam_v2's 0.301 on 451 held-out fields at IoU 0.5 -- a delta of 0.569.
   :target: ../../models/cell_from_hoechst_v1.md
   :width: 49.5%

.. |zoo_card_toxoplasma_from_hoechst_v1| image:: ../../models/cards/toxoplasma_from_hoechst_v1.svg
   :alt: Score card of Vacuoles from Hoechst v1 (toxoplasma_from_hoechst_v1, default): F1 0.569 on 463 held-out fields against 0.002 for stock cpsam_v2. F1 0.569 against 0.002 for stock cpsam_v2 on 463 well-grouped held-out fields, at IoU 0.5.
   :target: ../../models/toxoplasma_from_hoechst_v1.md
   :width: 49.5%

.. image:: ../../readme_images/is/generated-model-zoo-1.svg
   :alt: Older models.
   :width: 100%

| |generated-model-zoo-1-link-1|\ |generated-model-zoo-1-link-2|\ |generated-model-zoo-1-link-3|
| |generated-model-zoo-1-link-4|\ |generated-model-zoo-1-link-5|\ |generated-model-zoo-1-link-6|
| |generated-model-zoo-1-link-7|

.. |generated-model-zoo-1-link-1| image:: ../../readme_images/is/generated-model-zoo-1-link-1.svg
   :alt: toxoplasma_pv_v3
   :width: 32.8%
   :target: ../../models/toxoplasma_pv_v3.md

.. |generated-model-zoo-1-link-2| image:: ../../readme_images/is/generated-model-zoo-1-link-2.svg
   :alt: toxoplasma_pv_v2
   :width: 32.8%
   :target: ../../models/toxoplasma_pv_v2.md

.. |generated-model-zoo-1-link-3| image:: ../../readme_images/is/generated-model-zoo-1-link-3.svg
   :alt: toxoplasma_pv_v1
   :width: 32.8%
   :target: ../../models/toxoplasma_pv_v1.md

.. |generated-model-zoo-1-link-4| image:: ../../readme_images/is/generated-model-zoo-1-link-4.svg
   :alt: toxoplasma_plaque_v2
   :width: 32.8%
   :target: ../../models/toxoplasma_plaque_v2.md

.. |generated-model-zoo-1-link-5| image:: ../../readme_images/is/generated-model-zoo-1-link-5.svg
   :alt: toxoplasma_plaque_v1
   :width: 32.8%
   :target: ../../models/toxoplasma_plaque_v1.md

.. |generated-model-zoo-1-link-6| image:: ../../readme_images/is/generated-model-zoo-1-link-6.svg
   :alt: toxoplasma_well_detector_v2
   :width: 32.8%
   :target: ../../models/toxoplasma_well_detector_v2.md

.. |generated-model-zoo-1-link-7| image:: ../../readme_images/is/generated-model-zoo-1-link-7.svg
   :alt: toxoplasma_well_detector_v1
   :width: 32.8%
   :target: ../../models/toxoplasma_well_detector_v1.md

.. image:: ../../readme_images/is/generated-model-zoo-2.svg
   :alt: Alternatives.
   :width: 100%

| |generated-model-zoo-2-link-1|\ |generated-model-zoo-2-link-2|

.. |generated-model-zoo-2-link-1| image:: ../../readme_images/is/generated-model-zoo-2-link-1.svg
   :alt: toxoplasma_pv_v5
   :width: 32.8%
   :target: ../../models/toxoplasma_pv_v5.md

.. |generated-model-zoo-2-link-2| image:: ../../readme_images/is/generated-model-zoo-2-link-2.svg
   :alt: toxoplasma_plaque_v3
   :width: 32.8%
   :target: ../../models/toxoplasma_plaque_v3.md

.. spacr-model-zoo-end

.. image:: ../../readme_images/is/diagnosing-performance.svg
   :alt: Greining á afköstum. Búðu til vélbúnaðarskýrslu og hengdu hana við mál um afköst:
   :width: 100%

.. code-block:: bash

   python tools/spacr_hardware_report.py

.. image:: ../../readme_images/is/diagnosing-performance-2.svg
   :alt: Spara til ~/.spacr/reports og trúa leiðinni. --quick skiptir lengri skilyrði; --out PATH setur staðsetningu. Les engin verkefnisgögn. Mælir tímann sem innflutningur, töluleg söfn, gerð glugga og hreyfimyndir taka og greinir frá x86_64-hermun á Apple Silicon og BLAS-útfærslu NumPy.
   :width: 100%

.. image:: ../../readme_images/is/command-line-reference.svg
   :alt: Orðlinna reference. Öll beint hér að neðan er sett með pip install spacr. Allir samþykkir --help.
   :width: 100%

.. image:: ../../readme_images/is/launching-the-application.svg
   :alt: Að byrja við umsókn.
   :width: 100%

.. code-block:: bash

   spacr              # the desktop application
   spacr-tutorial     # the interactive tutorial library
   spacr-server       # no first-run setup screen, for unattended launches

.. image:: ../../readme_images/is/when-spacr-will-not-start.svg
   :alt: Þegar spaCR mun ekki byrja.
   :width: 100%

.. code-block:: bash

   spacr-doctor       # diagnose the installation and say how to fix it
   safespacr          # the least spaCR that can still change a setting

.. image:: ../../readme_images/is/when-spacr-will-not-start-2.svg
   :alt: spacr-doctor drukkar eitt línu á athygli, með komandi til að kjósa fyrir hvert mismunandi. Það segir einnig hvaða spacr er á slóðinni. safespacr les hverja stillingu sem sjálfgefið gildi hennar og slekkur á bakgrunni, hreyfimyndum, ítarlegri skráningu og forhleðslu. Notaðu það þegar vistuð stilling kemur í veg fyrir að forritið ræsist. Stillingar sem vistaðar eru í öruggum ham eru skrifaðar eins og venjulega og haldast.
   :width: 100%

.. image:: ../../readme_images/is/running-modules-headlessly.svg
   :alt: Að hlaupa modúlum án heiðar. Engin Qt, engin sýning — fyrir klúster, þjónusta og CI.
   :width: 100%

.. code-block:: bash

   spacr-run --list                              # modules with a headless entry
   spacr-run --describe MODULE                   # what a module consumes and produces
   spacr-run validate --module MODULE \
       --settings settings.csv                   # check settings before spending the run
   spacr-run MODULE --settings settings.csv      # execute
   spacr-remote --help                           # submit and monitor SSH, Slurm or cloud jobs

.. image:: ../../readme_images/is/running-modules-headlessly-2.svg
   :alt: validate les sömu stillingar og keyrslan myndi lesa og telur upp hvert vandamál, til dæmis stillingu sem vantar eða slóð sem er ekki til, ásamt lagfæringu.
   :width: 100%

.. image:: ../../readme_images/is/inspecting-a-run.svg
   :alt: Skoða keyrslu. Hver rán er skráður á ~/.spacr/runs með settum sínum, hashed inntölum, úttökum, varningar, útgáfur og frönum.
   :width: 100%

.. code-block:: bash

   spacr-repro RUN_DIR        # replay a recorded run from its journal
   spacr-workspace RUN_DIR    # what that run had open: databases, montages, views

.. image:: ../../readme_images/is/auditing-data-and-installation.svg
   :alt: Ákvarðanir gögnum og uppsetningu.
   :width: 100%

.. code-block:: bash

   spacr-db-audit DB      # SQLite health, integrity, locking, reader/writer probe
   spacr-leakage          # classifier train/test leakage audit
   spacr-plugins          # installed plugin registry and failure diagnostics

.. image:: ../../readme_images/is/environment.svg
   :alt: Umhverfi.
   :width: 100%

.. code-block:: bash

   SPACR_LOG_LEVEL=DEBUG spacr      # verbose logging for one launch

.. image:: ../../readme_images/is/environment-2.svg
   :alt: Rotating logs eru skrifað í ~/.spacr/logs/spacr.log. Sættu þessar skál á bug-report.
   :width: 100%

.. image:: ../../readme_images/is/contributing-and-support.svg
   :alt: Framlög og aðstoð. Sendu villutilkynningar og afmarkaðar óskir um eiginleika í gegnum GitHub-mál. Þegar bilun er tilkynnt skal tilgreina útgáfu spaCR, stýrikerfi, útgáfu Python, stillingar einingarinnar og viðeigandi hluta úr annálnum. spacr-doctor safnar flestum þessara upplýsinga; láttu vélbúnaðarskýrsluna fylgja þegar tilkynnt er um afkastavandamál.
   :width: 100%
   :target: https://github.com/EinarOlafsson/spacr/issues

.. image:: ../../readme_images/is/licensing.svg
   :alt: Leyfi. spaCR er frelsað undir BSD 3-Klausur leyfi. Ef spaCR hjálpaði að útgáfa verk, er nefndur verðmæt og er ekki skilyrði fyrir leyfi — sjá Tilvísun í spaCR hér neðan.
   :width: 100%
   :target: https://github.com/EinarOlafsson/spacr/blob/main/LICENSE

.. image:: ../../readme_images/is/tutorials.svg
   :alt: Kennsluefni. Gagnvirka spaCR-kennslusafnið leiðir þig í gegnum uppsetningu og notkun eininganna. Hver kennslustund tilgreinir hvaða upplestur og tungumál eru í boði.
   :width: 100%
   :target: https://einarolafsson.github.io/spacr/tutorials/

.. image:: ../../readme_images/is/citing-spacr.svg
   :alt: Tilvísun í spaCR. Ef spaCR nýtist við rannsóknina skaltu vitna í: Olafsson EB, et al. A sameiginlegur myndbönd sem er bastir á CRISPR skrefinn skilur EAF1 sem T. gondii modulator ESCRT subversion.
   :width: 100%

| |citing-spacr-2-1|\ |citing-spacr-2-2|

.. |citing-spacr-2-1| image:: ../../readme_images/is/citing-spacr-2-1.svg
   :alt: Bioregl fyrirframskrift
   :width: 49.5%
   :target: https://www.biorxiv.org/content/10.64898/2026.07.08.737057v1

.. |citing-spacr-2-2| image:: ../../readme_images/is/citing-spacr-2-2.svg
   :alt: Programvarparkíf
   :width: 49.5%
   :target: https://doi.org/10.5281/zenodo.21343316

.. image:: ../../readme_images/is/other-work-citing-spacr.svg
   :alt: Aðrir verk sem heitir spaCR.
   :width: 100%

.. spacr-citing-papers-begin

.. image:: ../../readme_images/is/other-work-citing-spacr-2-1.svg
   :alt: Metabolic adaptability og næringssvopning í Toxoplasma gondii: innsýn frá innihaldskortum mutantum.
   :width: 100%
   :target: https://journals.asm.org/doi/full/10.1128/msphere.01011-24

.. image:: ../../readme_images/is/other-work-citing-spacr-2-2.svg
   :alt: IRE1α stuðlar að fagosomal kalsiumflokk til að bæta makrofága svítilverkefni.
   :width: 100%
   :target: https://www.cell.com/cell-reports/fulltext/S2211-1247(25)00465-6

.. image:: ../../readme_images/is/other-work-citing-spacr-2-3.svg
   :alt: Toxoplasma GRA8 er nauðsynlegur fyrir aðstoðproteín ALG-2 og er mikilvægt fyrir efnahagsleikum parasita.
   :width: 100%
   :target: https://www.biorxiv.org/content/10.64898/2026.07.20.739547v1.abstract

.. image:: ../../readme_images/is/other-work-citing-spacr-2-4.svg
   :alt: Hydrogen sulfide dynamically upregulates copper uptake and localization.
   :width: 100%
   :target: https://www.biorxiv.org/content/10.64898/2026.07.30.741779v1

.. image:: ../../readme_images/is/other-work-citing-spacr-2-5.svg
   :alt: spaCR: Spatial fenotypeanalyse af CRISPR-Cas9 skrefum (preprint útgáfa 1).
   :width: 100%
   :target: https://www.researchsquare.com/article/rs-7368254/v1

.. spacr-citing-papers-end

.. image:: ../../readme_images/is/acknowledgments.svg
   :alt: Þakkir. spaCR byggir á opnum vísindahugbúnaði, meðal annars NumPy, pandas, scikit-image, scikit-learn, Cellpose, PyTorch og Qt. Sjá upplýsingar um þýðingarlíkön fyrir líkönin sem voru notuð við gerð fjöltyngdra skjala og viðmótsskráa.
   :width: 100%
   :target: ../TRANSLATION_MODELS.md
