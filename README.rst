|Website| |Platforms| |Python| |Qt| |Test counts| |Coverage| |Issues| |Conda| |PyPI| |Docker| |PyPI Downloads| |Docs| |Tutorials| |Preprint| |DOI| |Cite| |License| |PyPI rank|

Website: `einarolafsson.github.io/projects/spacr <https://einarolafsson.github.io/projects/spacr/>`_

.. |Website| image:: https://img.shields.io/badge/Website-spaCR-0A7BBB
   :target: https://einarolafsson.github.io/projects/spacr/
   :alt: spaCR website

.. |Docs| image:: https://img.shields.io/github/actions/workflow/status/EinarOlafsson/spacr/pages%2Fpages-build-deployment?label=API&logo=readthedocs&logoColor=white
   :target: https://einarolafsson.github.io/spacr/
   :alt: API Documentation

.. |Tutorials| image:: https://img.shields.io/badge/Tutorials-4A9EFF
   :target: https://einarolafsson.github.io/spacr/tutorials/
   :alt: Interactive tutorials

.. |PyPI| image:: https://img.shields.io/pypi/v/spacr?label=PyPI&logo=pypi&logoColor=white
   :target: https://pypi.org/project/spacr/
   :alt: PyPI version

.. |Python| image:: https://img.shields.io/badge/Python-3.9%E2%80%933.14-3776AB?logo=python&logoColor=white
   :target: https://pypi.org/project/spacr/
   :alt: Python 3.9 through 3.14

.. |Coverage| image:: https://img.shields.io/codecov/c/github/EinarOlafsson/spacr/main?logo=codecov&logoColor=white
   :target: https://app.codecov.io/github/EinarOlafsson/spacr
   :alt: Codecov coverage

.. |Test counts| image:: https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2FEinarOlafsson%2Fspacr%2Fnightly%2Fdocs%2Fsource%2F_static%2Ftest-counts.json&cacheSeconds=300&logo=pytest&logoColor=white
   :target: https://github.com/EinarOlafsson/spacr/actions/workflows/tests.yml
   :alt: tests passed/total

.. |Qt| image:: https://img.shields.io/badge/GUI-Qt-41CD52?logo=qt&logoColor=white
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/index.html#module-spacr.qt
   :alt: Qt interface

.. |Issues| image:: https://img.shields.io/github/issues/EinarOlafsson/spacr?logo=github
   :target: https://github.com/EinarOlafsson/spacr/issues
   :alt: GitHub issues

.. |License| image:: https://img.shields.io/badge/License-BSD_3--Clause-blue
   :target: https://github.com/EinarOlafsson/spacr/blob/nightly/LICENSE
   :alt: BSD 3-Clause

.. |Preprint| image:: https://img.shields.io/badge/bioRxiv-BF2636
   :target: https://www.biorxiv.org/content/10.64898/2026.07.08.737057v1
   :alt: bioRxiv preprint

.. |DOI| image:: https://img.shields.io/badge/DOI-blue?logo=zenodo&logoColor=white
   :target: https://doi.org/10.5281/zenodo.21343316
   :alt: Zenodo DOI

.. |Conda| image:: https://img.shields.io/conda/vn/conda-forge/spacr?label=Conda&logo=anaconda&logoColor=white
   :target: https://anaconda.org/conda-forge/spacr
   :alt: conda-forge version

.. |Release date| image:: https://anaconda.org/conda-forge/spacr/badges/latest_release_date.svg
   :target: https://anaconda.org/conda-forge/spacr
   :alt: conda-forge latest_release_date

.. |Docker| image:: https://img.shields.io/badge/Docker-GHCR-2496ED?logo=docker&logoColor=white
   :target: https://github.com/EinarOlafsson/spacr/pkgs/container/spacr
   :alt: Docker images on GHCR

.. |PyPI Downloads| image:: https://img.shields.io/pepy/dt/spacr?label=downloads&logo=pypi&logoColor=white
   :target: https://pepy.tech/projects/spacr
   :alt: PyPI Downloads

.. |Platforms| image:: https://img.shields.io/badge/Linux%20OSX%20Win-lightgrey?logo=linux&logoColor=white
   :target: https://github.com/EinarOlafsson/spacr/blob/nightly/docs/source/installers.rst
   :alt: Linux, macOS, and Windows

.. |Cite| image:: https://img.shields.io/badge/Cite-8A2BE2
   :target: https://github.com/EinarOlafsson/spacr/blob/main/CITATION.cff
   :alt: Cite spaCR

.. |PyPI rank| image:: https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fsql-clickhouse.clickhouse.com%2F%3Fuser%3Ddemo%26param_package_name%3Dspacr%26param_days%3D30%26query%3DWITH%2B%2528%2BSELECT%2Bsum%2528count%2529%2BFROM%2Bpypi.pypi_downloads_per_day%2BWHERE%2Bproject%2B%253D%2B%257Bpackage_name%253AString%257D%2BAND%2Bdate%2B%253E%253D%2BtoDate%2528now%2528%2527UTC%2527%2529%2529%2B-%2B%257Bdays%253AUInt16%257D%2BAND%2Bdate%2B%253C%2BtoDate%2528now%2528%2527UTC%2527%2529%2529%2B%2529%2BAS%2Bdownloads%2BSELECT%2Bdownloads%2BAS%2Bpackage_downloads%252C%2BcountIf%2528n%2B%253E%253D%2Bdownloads%2529%2BAS%2Brank%252C%2Bcount%2528%2529%2BAS%2Btotal_packages%252C%2B100.0%2B%252A%2Brank%2B%252F%2BnullIf%2528total_packages%252C%2B0%2529%2BAS%2Bpercentile%252C%2Bif%2528%2Btotal_packages%2B%253D%2B0%2BOR%2Bdownloads%2B%253D%2B0%252C%2B%2527no%2Bdata%2527%252C%2Bconcat%2528%2B%2527top%2B%2527%252C%2BtoString%2528ceil%25281000.0%2B%252A%2Brank%2B%252F%2BnullIf%2528total_packages%252C%2B0%2529%2529%2B%252F%2B10%2529%252C%2B%2527%2525%2527%2B%2529%2B%2529%2BAS%2Bmessage%2BFROM%2B%2528%2BSELECT%2Bproject%252C%2Bsum%2528count%2529%2BAS%2Bn%2BFROM%2Bpypi.pypi_downloads_per_day%2BWHERE%2Bdate%2B%253E%253D%2BtoDate%2528now%2528%2527UTC%2527%2529%2529%2B-%2B%257Bdays%253AUInt16%257D%2BAND%2Bdate%2B%253C%2BtoDate%2528now%2528%2527UTC%2527%2529%2529%2BGROUP%2BBY%2Bproject%2B%2529%2BFORMAT%2BJSON&query=%24.data%5B0%5D.message&label=Rank&color=brightgreen&logo=pypi&logoColor=white&cacheSeconds=86400
   :target: https://clickpy.clickhouse.com/dashboard/spacr
   :alt: spaCR PyPI download ranking over the previous 30 complete days

.. raw:: html

   <div><a href="https://einarolafsson.github.io/spacr/_static/deck/"><picture><source media="(min-width: 1280px)" srcset="docs/readme_assets/cover.desktop.svg"><img src="docs/readme_assets/cover.svg" alt="spaCR"></picture></a></div>

.. raw:: html

   <div><a href="docs/source/_static/deck/pages/57.md"><picture><source media="(min-width: 1280px)" srcset="docs/readme_images/en/intro-1.desktop.svg"><img src="docs/readme_images/en/intro-1.svg" alt="← Back"></picture></a><a href="docs/source/_static/deck/pages/02.md"><picture><source media="(min-width: 1280px)" srcset="docs/readme_images/en/intro-2.desktop.svg"><img src="docs/readme_images/en/intro-2.svg" alt="Next →"></picture></a></div>

.. spacr-language-picker-begin

.. raw:: html

   <div><a href="docs/i18n/readme/README.md"><picture><source media="(min-width: 1280px)" srcset="docs/readme_images/en/generated-language-picker.desktop.svg"><img src="docs/readme_images/en/generated-language-picker.svg" alt="Languages: English"></picture></a></div>

.. spacr-language-picker-end

.. raw:: html

   <div><a href="docs/readme_text/en/spacr.md"><picture><source media="(min-width: 1280px)" srcset="docs/readme_images/en/spacr.desktop.svg"><img src="docs/readme_images/en/spacr.svg" alt="spaCR. Spatial phenotype analysis of CRISPR screens. spaCR segments and measures single cells in microscopy images, integrates per-object phenotypes with sequencing-derived guide abundance, and estimates which genes are associated with phenotypic changes. Starting from plate images and FASTQ reads, it produces per-object measurements, trained classifiers, per-guide and per-gene effect estimates, and a ranked hit list. The segmentation, measurement, annotation and classification modules also run without a sequencing arm. Make Masks corrects segmentation masks and annotates independent, class-labelled rectangles with the Box tool for YOLO export. Boxes keep their own labels and history without changing source images or masks. See the feature guide for each tool. Images, masks and crops are files on disk. Measurements, annotations and model predictions are tables in the project&#x27;s measurements.db SQLite database. Runs as a desktop application or headlessly on a workstation, server or cluster."></picture></a></div>

.. raw:: html

   <div><a href="docs/readme_text/en/try-spacr.md"><picture><source media="(min-width: 1280px)" srcset="docs/readme_images/en/try-spacr.desktop.svg"><img src="docs/readme_images/en/try-spacr.svg" alt="Try spaCR."></picture></a></div>

.. code-block:: bash

   conda create -n spacr python=3.12 -y
   conda activate spacr
   python -m pip install spacr
   spacr

.. raw:: html

   <div><a href="docs/readme_text/en/hardware-support.md"><picture><source media="(min-width: 1280px)" srcset="docs/readme_images/en/hardware-support.desktop.svg"><img src="docs/readme_images/en/hardware-support.svg" alt="Hardware support."></picture></a></div>

.. spacr-hardware-begin

.. raw:: html

   <div><a href="docs/readme_text/en/generated-hardware-1.md"><picture><source media="(min-width: 1280px)" srcset="docs/readme_images/en/generated-hardware-1.desktop.svg"><img src="docs/readme_images/en/generated-hardware-1.svg" alt="NVIDIA (CUDA): Cellpose 4 GPU, Torch GPU, UMAP / clustering GPU, 3D GPU. AMD on Linux (ROCm): Cellpose 4 GPU, Torch GPU, UMAP / clustering CPU, 3D GPU. AMD in an Intel Mac (Metal): Cellpose 4 GPU, Torch GPU, UMAP / clustering KMeans GPU; PCA fitting CPU, projection GPU / CPU; UMAP / t-SNE / DBSCAN CPU, 3D GPU. Apple Silicon (Metal): Cellpose 4 GPU, Torch GPU, UMAP / clustering KMeans GPU; PCA fitting CPU, projection GPU / CPU; UMAP / t-SNE / DBSCAN CPU, 3D GPU. Intel Arc/Xe (XPU): Cellpose 4 GPU, Torch GPU, UMAP / clustering CPU, 3D GPU. No GPU: Cellpose 4 CPU, Torch CPU, UMAP / clustering CPU, 3D CPU. supported (stable) implemented (beta) CPU support only."></picture></a></div>

.. spacr-hardware-end

.. raw:: html

   <div><a href="docs/readme_text/en/install-spacr.md"><picture><source media="(min-width: 1280px)" srcset="docs/readme_images/en/install-spacr.desktop.svg"><img src="docs/readme_images/en/install-spacr.svg" alt="Install spaCR. Choose an install route: Desktop application: a Linux, macOS or Windows installer with its own Python. pip: python -m pip install spacr in a Conda environment. conda-forge: conda install conda-forge::spacr in a Conda environment. Docker: the command-line pipelines in a Linux container, from the CPU image or the NVIDIA GPU image:"></picture></a></div>

.. code-block:: bash

   docker run --rm ghcr.io/einarolafsson/spacr:1.5.1.6 spacr-run --list
   docker run --rm --gpus all ghcr.io/einarolafsson/spacr:1.5.1.6-cuda12.4 spacr-run --list

.. raw:: html

   <div><a href="docs/readme_text/en/install-spacr-2.md"><picture><source media="(min-width: 1280px)" srcset="docs/readme_images/en/install-spacr-2.desktop.svg"><img src="docs/readme_images/en/install-spacr-2.svg" alt="The Docker guide (nightly, English) covers GPU setup, data mounts and pipeline commands."></picture></a></div>

.. raw:: html

   <div><a href="docs/readme_text/en/desktop-application.md"><picture><source media="(min-width: 1280px)" srcset="docs/readme_images/en/desktop-application.desktop.svg"><img src="docs/readme_images/en/desktop-application.svg" alt="Desktop application. The installers bundle their own Python. Conda is not required."></picture></a></div>

.. spacr-installer-links-begin

| |InstallerLinux|\ |InstallerMacOS|\ |InstallerWindows|

.. |InstallerWindows| raw:: html

   <a href="https://github.com/EinarOlafsson/spacr/releases/download/v1.5.1.6/spaCR-1.5.1.6-Windows-Online-Setup.exe"><img src="docs/readme_assets/cells/platforms/windows.png" width="14.14%" align="middle" alt="Download spaCR 1.5.1.6 for Windows 10/11"></a>
.. |InstallerMacOS| raw:: html

   <a href="https://github.com/EinarOlafsson/spacr/releases/download/v1.5.1.6/spaCR-1.5.1.6-macOS-Universal-Online.pkg"><img src="docs/readme_assets/cells/platforms/macos.png" width="14.14%" align="middle" alt="Download spaCR 1.5.1.6 for macOS 11+ (Intel and Apple silicon)"></a>
.. |InstallerLinux| raw:: html

   <a href="https://github.com/EinarOlafsson/spacr/releases/download/v1.5.1.6/spaCR-1.5.1.6-Linux-x86_64-Online.run"><img src="docs/readme_assets/cells/platforms/linux.png" width="14.14%" align="middle" alt="Download spaCR 1.5.1.6 for 64-bit Linux"></a>

.. spacr-installer-links-end

.. raw:: html

   <div><a href="docs/readme_text/en/desktop-application-2.md"><picture><source media="(min-width: 1280px)" srcset="docs/readme_images/en/desktop-application-2.desktop.svg"><img src="docs/readme_images/en/desktop-application-2.svg" alt="On Linux, make the downloaded file executable and run it:"></picture></a></div>

.. code-block:: bash

   chmod +x SpaCR-*-Linux-x86_64-Online.run
   ./SpaCR-*-Linux-x86_64-Online.run

.. raw:: html

   <div><a href="docs/readme_text/en/desktop-application-3.md"><picture><source media="(min-width: 1280px)" srcset="docs/readme_images/en/desktop-application-3.desktop.svg"><img src="docs/readme_images/en/desktop-application-3.svg" alt="On macOS, open the .pkg. The current beta is not notarized; if Gatekeeper blocks it, choose System Settings → Privacy &amp; Security → Open Anyway. For installation, updates, reinstallation, uninstallation, offline use and troubleshooting, see the current nightly installer guide (English). See system requirements for workstation and server recommendations and GPU compatibility tables."></picture></a></div>

.. raw:: html

   <div><a href="https://github.com/EinarOlafsson/spacr/blob/nightly/docs/source/installer_guide.rst"><picture><source media="(min-width: 1280px)" srcset="docs/readme_images/en/desktop-application-3-link-1.desktop.svg"><img src="docs/readme_images/en/desktop-application-3-link-1.svg" alt="current nightly installer guide (English)"></picture></a><a href="docs/source/system_requirements.rst"><picture><source media="(min-width: 1280px)" srcset="docs/readme_images/en/desktop-application-3-link-2.desktop.svg"><img src="docs/readme_images/en/desktop-application-3-link-2.svg" alt="system requirements"></picture></a></div>

.. raw:: html

   <div><a href="docs/readme_text/en/pypi-installation.md"><picture><source media="(min-width: 1280px)" srcset="docs/readme_images/en/pypi-installation.desktop.svg"><img src="docs/readme_images/en/pypi-installation.svg" alt="PyPI installation. For the PyPI release, install spaCR with pip inside a Conda environment. Python 3.12 has the widest choice of optional scientific packages:"></picture></a></div>

.. code-block:: bash

   conda create -n spacr python=3.12 -y
   conda activate spacr
   python -m pip install --upgrade pip
   python -m pip install spacr
   spacr

.. raw:: html

   <div><a href="docs/readme_text/en/pypi-installation-2.md"><picture><source media="(min-width: 1280px)" srcset="docs/readme_images/en/pypi-installation-2.desktop.svg"><img src="docs/readme_images/en/pypi-installation-2.svg" alt="spaCR supports Python 3.9 through 3.14, except Python 3.14.1, which torchvision excludes. Linux is recommended for the heaviest CUDA and ROCm workflows; macOS and Windows are also supported, and both use their GPUs — macOS through Metal, which covers Apple Silicon and the AMD cards in Intel Macs, and Windows through CUDA or DirectML. The standard installation includes the Qt desktop interface. For a server, cluster or CI runner, run the command-line pipelines without opening it:"></picture></a></div>

.. code-block:: bash

   python -m pip install spacr
   spacr-run --list

.. raw:: html

   <div><a href="docs/readme_text/en/pypi-installation-3.md"><picture><source media="(min-width: 1280px)" srcset="docs/readme_images/en/pypi-installation-3.desktop.svg"><img src="docs/readme_images/en/pypi-installation-3.svg" alt="Optional integrations are installed separately, for example spacr[zarr], spacr[omero], spacr[napari] and spacr[czi,nd2,lif]. See the installation guide (nightly, English) for the complete extras and Python-version compatibility table."></picture></a></div>

.. raw:: html

   <div><a href="docs/readme_text/en/conda-forge-installation.md"><picture><source media="(min-width: 1280px)" srcset="docs/readme_images/en/conda-forge-installation.desktop.svg"><img src="docs/readme_images/en/conda-forge-installation.svg" alt="Conda-forge installation. The official conda-forge package installs spaCR and its desktop dependencies into the active environment:"></picture></a></div>

.. code-block:: bash

   conda create -n spacr python=3.12 -y
   conda activate spacr
   conda install conda-forge::spacr
   spacr

.. raw:: html

   <div><a href="docs/readme_text/en/install-from-source.md"><picture><source media="(min-width: 1280px)" srcset="docs/readme_images/en/install-from-source.desktop.svg"><img src="docs/readme_images/en/install-from-source.svg" alt="Install from source. Clone the repository and install it in editable mode, so your working copy is the installed package and edits take effect without reinstalling:"></picture></a></div>

.. code-block:: bash

   git clone https://github.com/EinarOlafsson/spacr.git
   cd spacr
   conda create -n spacr python=3.12 -y
   conda activate spacr
   pip install -e .
   spacr

.. raw:: html

   <div><a href="docs/readme_text/en/install-from-source-2.md"><picture><source media="(min-width: 1280px)" srcset="docs/readme_images/en/install-from-source-2.desktop.svg"><img src="docs/readme_images/en/install-from-source-2.svg" alt="This clones main, the default branch, which carries the latest release. Development happens on nightly; add --branch nightly to clone it instead. For a specific release:"></picture></a></div>

.. code-block:: bash

   git clone --branch v1.5.0.5 https://github.com/EinarOlafsson/spacr.git

.. raw:: html

   <div><a href="docs/readme_text/en/install-from-source-3.md"><picture><source media="(min-width: 1280px)" srcset="docs/readme_images/en/install-from-source-3.desktop.svg"><img src="docs/readme_images/en/install-from-source-3.svg" alt="To pull later changes, from inside the clone:"></picture></a></div>

.. code-block:: bash

   git pull
   pip install -e .

.. raw:: html

   <div><a href="docs/readme_text/en/install-from-source-4.md"><picture><source media="(min-width: 1280px)" srcset="docs/readme_images/en/install-from-source-4.desktop.svg"><img src="docs/readme_images/en/install-from-source-4.svg" alt="Reinstall when dependencies or entry points change. Python edits take effect directly; spacr-doctor identifies the active installation."></picture></a></div>

.. raw:: html

   <div><a href="docs/readme_text/en/install-from-source-light.md"><picture><source media="(min-width: 1280px)" srcset="docs/readme_images/en/install-from-source-light.desktop.svg"><img src="docs/readme_images/en/install-from-source-light.svg" alt="Install from source (light). Contributors need history; to run spaCR, choose below. Measurements: nightly at 05302fd5c on 2026-10-07, using packaging/measure_clone_forms.sh. Shallow clone: 2048 MB downloaded in 140 s. Only the latest commit is kept, so older history is unavailable to git log, git blame and git bisect. git pull works but stays shallow; git fetch --unshallow retrieves the full history."></picture></a></div>

.. code-block:: bash

   git clone --depth 1 --branch nightly https://github.com/EinarOlafsson/spacr.git
   cd spacr && pip install -e .

.. raw:: html

   <div><a href="docs/readme_text/en/install-from-source-light-2.md"><picture><source media="(min-width: 1280px)" srcset="docs/readme_images/en/install-from-source-light-2.desktop.svg"><img src="docs/readme_images/en/install-from-source-light-2.svg" alt="Runtime-only checkout: 104 MB on disk in 6 s (Git: 38 MB; files: 66 MB). It omits history, docs, tests, tools, features, example data and extended interface translations. --with-docs, --with-tests and --with-translations retain the corresponding documentation, tests and translations. The compact interface translations remain available."></picture></a></div>

.. code-block:: bash

   curl -fsSL https://raw.githubusercontent.com/EinarOlafsson/spacr/nightly/packaging/install_from_source.sh -o install_spacr.sh
   sh install_spacr.sh --branch nightly

.. raw:: html

   <div><a href="docs/readme_text/en/install-from-source-light-3.md"><picture><source media="(min-width: 1280px)" srcset="docs/readme_images/en/install-from-source-light-3.desktop.svg"><img src="docs/readme_images/en/install-from-source-light-3.svg" alt="--dir chooses the checkout path; --branch selects a branch or tag; --no-install fetches without installing; --help lists the options. packaging/source_install_excludes.txt lists every skipped path. The full nightly clone downloaded 9.25 GiB. Adding --filter=blob:none to the shallow clone does not reduce the checkout: its object store is still 2032 MB. Git does not report lazy blob fetches, so no download total was measured. The nightly tracked tree is a 720 MB checkout (measured 2026-10-21), excluding Git history. Download sizes and times vary with the branch."></picture></a></div>

.. raw:: html

   <div><a href="docs/readme_text/en/command-line-entry-points.md"><picture><source media="(min-width: 1280px)" srcset="docs/readme_images/en/command-line-entry-points.desktop.svg"><img src="docs/readme_images/en/command-line-entry-points.svg" alt="Command-line entry points."></picture></a></div>

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

.. raw:: html

   <div><a href="docs/readme_text/en/command-line-entry-points-2.md"><picture><source media="(min-width: 1280px)" srcset="docs/readme_images/en/command-line-entry-points-2.desktop.svg"><img src="docs/readme_images/en/command-line-entry-points-2.svg" alt="spacr-run --list lists modules with headless command-line entry points. GUI-only annotation, curation, comparison and exploration modules are omitted."></picture></a></div>

.. raw:: html

   <div><a href="docs/readme_text/en/core-workflow.md"><picture><source media="(min-width: 1280px)" srcset="docs/readme_images/en/core-workflow.desktop.svg"><img src="docs/readme_images/en/core-workflow.svg" alt="Core workflow. The primary workflow comprises six modules: Mask segments cells, nuclei, pathogens and organelles with Cellpose. Measure writes morphology, intensity, texture, spatial and colocalization features, together with object crops, to SQLite. Annotate labels crops in a keyboard-driven grid and supports active-learning queues. Classify trains image or measurement-based models and records held-out performance with each checkpoint. Map Barcodes maps FASTQ reads to wells and gRNAs, with abundance, collision and coverage QC. Regression estimates guide, gene, condition and control effects with model families suited to continuous, fractional and count responses."></picture></a></div>

.. raw:: html

   <div><a href="docs/readme_text/en/spacr-modules.md"><picture><source media="(min-width: 1280px)" srcset="docs/readme_images/en/spacr-modules.desktop.svg"><img src="docs/readme_images/en/spacr-modules.svg" alt="spaCR modules."></picture></a></div>

.. spacr-workflow-begin

.. raw:: html

   <div><a href="docs/readme_text/en/generated-workflow-1.md"><picture><source media="(min-width: 1280px)" srcset="docs/readme_images/en/generated-workflow-1.desktop.svg"><img src="docs/readme_images/en/generated-workflow-1.svg" alt="Core. Core sequence from microscopy images through segmentation, measurements, annotations, classification, barcode mapping and regression."></picture></a></div>

| |Module_mask|\ |Module_measure|\ |Module_annotate|\ |Module_classify_merged|\ |Module_map_barcodes|\ |Module_regression|

.. raw:: html

   <div><a href="docs/readme_text/en/generated-workflow-2.md"><picture><source media="(min-width: 1280px)" srcset="docs/readme_images/en/generated-workflow-2.desktop.svg"><img src="docs/readme_images/en/generated-workflow-2.svg" alt="Data. Import images and tables into spaCR projects and execute reproducible multi-plate workflows."></picture></a></div>

| |Module_foreign|\ |Module_embeddings|\ |Module_run_compare|\ |Module_experiment_design|\ |Module_power|\ |Module_dose_response|\ |Module_qc_dashboard|

.. raw:: html

   <div><a href="docs/readme_text/en/generated-workflow-3.md"><picture><source media="(min-width: 1280px)" srcset="docs/readme_images/en/generated-workflow-3.desktop.svg"><img src="docs/readme_images/en/generated-workflow-3.svg" alt="Tools. Point these at a project: edit masks by hand, stitch tiles, read an embedding, draw a gate, build a plot, check quality."></picture></a></div>

| |Module_make_masks|\ |Module_align|\ |Module_umap|\ |Module_gate_editor|\ |Module_graph_builder|

.. raw:: html

   <div><a href="docs/readme_text/en/generated-workflow-4.md"><picture><source media="(min-width: 1280px)" srcset="docs/readme_images/en/generated-workflow-4.desktop.svg"><img src="docs/readme_images/en/generated-workflow-4.svg" alt="Organism. Organism-specific image analysis and quantitative assay readouts."></picture></a></div>

| |Module_toxoplasma|

.. |Module_mask| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/core/index.html#spacr.core.preprocess_generate_masks"><img src="docs/readme_assets/cells/workflow/mask.png" width="14.14%" align="middle" alt="Open the Mask API"></a>

.. |Module_measure| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/measure/index.html"><img src="docs/readme_assets/cells/workflow/measure.png" width="14.14%" align="middle" alt="Open the Measure API"></a>

.. |Module_annotate| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/qt/screens/annotate/index.html"><img src="docs/readme_assets/cells/workflow/annotate.png" width="14.14%" align="middle" alt="Open the Annotate API"></a>

.. |Module_classify_merged| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/classify/index.html"><img src="docs/readme_assets/cells/workflow/classify_merged.png" width="14.14%" align="middle" alt="Open the Classify API"></a>

.. |Module_map_barcodes| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/sequencing/index.html"><img src="docs/readme_assets/cells/workflow/map_barcodes.png" width="14.14%" align="middle" alt="Open the Map Barcodes API"></a>

.. |Module_regression| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/ml/index.html"><img src="docs/readme_assets/cells/workflow/regression.png" width="14.14%" align="middle" alt="Open the Regression API"></a>

.. |Module_foreign| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/foreign/index.html"><img src="docs/readme_assets/cells/workflow/apps/foreign.png" width="14.14%" align="middle" alt="Open the Import API"></a>

.. |Module_embeddings| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/qt/screens/embeddings/index.html"><img src="docs/readme_assets/cells/workflow/apps/embeddings.png" width="14.14%" align="middle" alt="Open the Embeddings API"></a>

.. |Module_run_compare| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/qt/screens/run_compare/index.html"><img src="docs/readme_assets/cells/workflow/apps/run_compare.png" width="14.14%" align="middle" alt="Open the Run Compare API"></a>

.. |Module_experiment_design| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/qt/screens/experiment_design/index.html"><img src="docs/readme_assets/cells/workflow/apps/experiment_design.png" width="14.14%" align="middle" alt="Open the Experiment Design API"></a>

.. |Module_power| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/qt/screens/power/index.html"><img src="docs/readme_assets/cells/workflow/apps/power.png" width="14.14%" align="middle" alt="Open the Power / Design API"></a>

.. |Module_dose_response| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/qt/screens/dose_response/index.html"><img src="docs/readme_assets/cells/workflow/apps/dose_response.png" width="14.14%" align="middle" alt="Open the Dose–Response API"></a>

.. |Module_qc_dashboard| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/qt/screens/qc_dashboard/index.html"><img src="docs/readme_assets/cells/workflow/apps/qc_dashboard.png" width="14.14%" align="middle" alt="Open the QC API"></a>

.. |Module_make_masks| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/qt/screens/make_masks/index.html"><img src="docs/readme_assets/cells/workflow/apps/make_masks.png" width="14.14%" align="middle" alt="Open the Make Masks API"></a>

.. |Module_align| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/align/index.html"><img src="docs/readme_assets/cells/workflow/apps/align.png" width="14.14%" align="middle" alt="Open the Align &amp; Stitch API"></a>

.. |Module_umap| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/core/index.html#spacr.core.generate_image_umap"><img src="docs/readme_assets/cells/workflow/apps/umap.png" width="14.14%" align="middle" alt="Open the Image UMAP API"></a>

.. |Module_gate_editor| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/qt/screens/gate_editor/index.html"><img src="docs/readme_assets/cells/workflow/apps/gate_editor.png" width="14.14%" align="middle" alt="Open the Gate Editor API"></a>

.. |Module_graph_builder| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/qt/screens/graph_builder/index.html"><img src="docs/readme_assets/cells/workflow/apps/graph_builder.png" width="14.14%" align="middle" alt="Open the Graph Builder API"></a>

.. |Module_toxoplasma| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/qt/screens/organism_screen/index.html#spacr-qt-screens-organism-screen-toxoplasma"><img src="docs/readme_assets/cells/workflow/apps/toxoplasma.png" width="14.14%" align="middle" alt="Open the Toxoplasma API"></a>

.. spacr-workflow-end

.. raw:: html

   <div><a href="docs/readme_text/en/spacr-modules-2.md"><picture><source media="(min-width: 1280px)" srcset="docs/readme_images/en/spacr-modules-2.desktop.svg"><img src="docs/readme_images/en/spacr-modules-2.svg" alt="Every module with a Home tile, in Home&#x27;s order: the six pipeline modules first, then the rest. Select a tile to open that module&#x27;s API page."></picture></a></div>

.. raw:: html

   <div><a href="docs/readme_text/en/other-resources.md"><picture><source media="(min-width: 1280px)" srcset="docs/readme_images/en/other-resources.desktop.svg"><img src="docs/readme_images/en/other-resources.svg" alt="Other resources."></picture></a></div>

.. raw:: html

   <div><a href="docs/readme_text/en/other-resources-2-1.md"><picture><source media="(min-width: 1280px)" srcset="docs/readme_images/en/other-resources-2-1.desktop.svg"><img src="docs/readme_images/en/other-resources-2-1.svg" alt="Interactive tutorials — guided workflows from installation through hit investigation."></picture></a></div>

.. raw:: html

   <div><a href="docs/readme_text/en/other-resources-2-2.md"><picture><source media="(min-width: 1280px)" srcset="docs/readme_images/en/other-resources-2-2.desktop.svg"><img src="docs/readme_images/en/other-resources-2-2.svg" alt="Python API quickstart — run and validate pipelines from scripts, notebooks or a cluster."></picture></a></div>

.. raw:: html

   <div><a href="docs/readme_text/en/other-resources-2-3.md"><picture><source media="(min-width: 1280px)" srcset="docs/readme_images/en/other-resources-2-3.desktop.svg"><img src="docs/readme_images/en/other-resources-2-3.svg" alt="Feature guide — capabilities, maturity and optional integrations."></picture></a></div>

.. raw:: html

   <div><a href="docs/readme_text/en/other-resources-2-4.md"><picture><source media="(min-width: 1280px)" srcset="docs/readme_images/en/other-resources-2-4.desktop.svg"><img src="docs/readme_images/en/other-resources-2-4.svg" alt="Curated API reference — supported entry points by task, with the complete module reference one level deeper."></picture></a></div>

.. raw:: html

   <div><a href="docs/readme_text/en/other-resources-2-5.md"><picture><source media="(min-width: 1280px)" srcset="docs/readme_images/en/other-resources-2-5.desktop.svg"><img src="docs/readme_images/en/other-resources-2-5.svg" alt="Language &amp; translation guide — interface languages, contextual help and scientific-output policy."></picture></a></div>

.. raw:: html

   <div><a href="docs/readme_text/en/language-translation.md"><picture><source media="(min-width: 1280px)" srcset="docs/readme_images/en/language-translation.desktop.svg"><img src="docs/readme_images/en/language-translation.svg" alt="Language &amp; translation. The interface supports ten languages across navigation and Preferences. AI and LIVE controls, module descriptions and reviewed contextual help are also translated. Change the language under spaCR → Preferences → Language without restarting. Logs, paths, database values and measurements are never translated; scientific output remains canonical English. See the contextual-help policy."></picture></a></div>

.. raw:: html

   <div><a href="docs/readme_text/en/animated-setting-guidance.md"><picture><source media="(min-width: 1280px)" srcset="docs/readme_images/en/animated-setting-guidance.desktop.svg"><img src="docs/readme_images/en/animated-setting-guidance.svg" alt="Animated setting guidance. Settings with a visual explanation offer an Animation control in their tooltip. Browse the setting animation gallery or the Setting animation registry."></picture></a></div>

.. raw:: html

   <div><a href="https://einarolafsson.github.io/spacr/setting_animations.html"><picture><source media="(min-width: 1280px)" srcset="docs/readme_images/en/animated-setting-guidance-link-1.desktop.svg"><img src="docs/readme_images/en/animated-setting-guidance-link-1.svg" alt="setting animation gallery"></picture></a><a href="https://einarolafsson.github.io/spacr/api/spacr/setting_animations/index.html"><picture><source media="(min-width: 1280px)" srcset="docs/readme_images/en/animated-setting-guidance-link-2.desktop.svg"><img src="docs/readme_images/en/animated-setting-guidance-link-2.svg" alt="Setting animation registry"></picture></a></div>

.. raw:: html

   <div><a href="docs/readme_text/en/data.md"><picture><source media="(min-width: 1280px)" srcset="docs/readme_images/en/data.desktop.svg"><img src="docs/readme_images/en/data.svg" alt="Data. Reference datasets."></picture></a></div>

| |DataBioStudies|\ |DataHuggingFace|\ |DataNCBI|\ |DataSpaCRPower|\ |DataBioRxiv|

.. |DataBioStudies| raw:: html

   <a href="https://doi.org/10.6019/S-BIAD2135"><img src="docs/readme_assets/cells/databanks/biostudies_button.png" width="14.14%" align="middle" alt="Open the BioStudies microscopy dataset"></a>

.. |DataHuggingFace| raw:: html

   <a href="https://huggingface.co/datasets/einarolafsson/toxo_mito"><img src="docs/readme_assets/cells/databanks/huggingface_button.png" width="14.14%" align="middle" alt="Open the Hugging Face testing dataset"></a>

.. |DataNCBI| raw:: html

   <a href="https://www.ncbi.nlm.nih.gov/bioproject/?term=PRJNA1261935"><img src="docs/readme_assets/cells/databanks/ncbi_button.png" width="14.14%" align="middle" alt="Open the NCBI sequencing dataset"></a>

.. |DataSpaCRPower| raw:: html

   <a href="https://github.com/maomlab/spaCRPower"><img src="docs/readme_assets/cells/databanks/spacrpower_button.png" width="14.14%" align="middle" alt="Open spaCRPower"></a>

.. |DataBioRxiv| raw:: html

   <a href="https://www.biorxiv.org/content/10.64898/2026.07.08.737057v1"><img src="docs/readme_assets/cells/databanks/biorxiv_button.png" width="14.14%" align="middle" alt="Open the bioRxiv preprint"></a>

.. raw:: html

   <div><a href="docs/readme_text/en/model-zoo.md"><picture><source media="(min-width: 1280px)" srcset="docs/readme_images/en/model-zoo.desktop.svg"><img src="docs/readme_images/en/model-zoo.svg" alt="Model zoo. spaCR ships a catalogue of trained models and fetches them on demand. Open Model Zoo from the home screen to browse and install them, or name a key in a settings file -- pathogen_model: toxoplasma_pv_v1 -- and the model is downloaded and checksum-verified the first time it is needed. Every published entry carries a SHA-256; an entry without one is refused rather than installed, because a truncated or substituted checkpoint cannot be told from the real one."></picture></a></div>

.. spacr-model-zoo-begin

.. raw:: html

   <div><a href="docs/models/README.md"><picture><source media="(min-width: 1280px)" srcset="docs/models/cards/heading.svg"><img src="docs/models/cards/heading.phone.svg" alt="Default models · held-out scores"></picture></a></div>
   <div><a href="docs/models/toxoplasma_pv_v4.md"><picture><source media="(min-width: 1280px)" srcset="docs/models/cards/toxoplasma_pv_v4.svg"><img src="docs/models/cards/toxoplasma_pv_v4.phone.svg" alt="Score card of Parasite vacuoles v4 (toxoplasma_pv_v4, default): F1 0.761 on 209 held-out fields against 0.377 for stock cpsam_v2. F1 0.761 against stock cpsam_v2&#x27;s 0.377 on the 209 strict hold-out fields at IoU 0.5; AJI 0.582 against 0.040."></picture></a><a href="docs/models/toxoplasma_plaque_v6.md"><picture><source media="(min-width: 1280px)" srcset="docs/models/cards/toxoplasma_plaque_v6.svg"><img src="docs/models/cards/toxoplasma_plaque_v6.phone.svg" alt="Score card of Plaques v6 (toxoplasma_plaque_v6, default): F1 0.768 on 160 held-out fields against 0.294 for stock cpsam_v2. F1 0.768 against stock cpsam_v2&#x27;s 0.294 on the 160 strict hold-out fields at IoU 0.5; AJI 0.752 against 0.249."></picture></a><a href="docs/models/toxoplasma_well_detector_v3.md"><picture><source media="(min-width: 1280px)" srcset="docs/models/cards/toxoplasma_well_detector_v3.svg"><img src="docs/models/cards/toxoplasma_well_detector_v3.phone.svg" alt="Score card of Plaque wells v3 (toxoplasma_well_detector_v3, default): mAP50 0.994 on 73 held-out images. Stock was not evaluated in this run; see the named incumbent comparison on the model card."></picture></a><a href="docs/models/toxoplasma_from_cellmask_v1.md"><picture><source media="(min-width: 1280px)" srcset="docs/models/cards/toxoplasma_from_cellmask_v1.svg"><img src="docs/models/cards/toxoplasma_from_cellmask_v1.phone.svg" alt="Score card of Vacuoles from cell mask v1 (toxoplasma_from_cellmask_v1, default): F1 0.606 on 463 held-out fields against 0.021 for stock cpsam_v2. F1 0.606 against 0.021 for stock cpsam_v2 on 463 well-grouped held-out fields, at IoU 0.5."></picture></a><a href="docs/models/nuclei_from_cellmask_v1.md"><picture><source media="(min-width: 1280px)" srcset="docs/models/cards/nuclei_from_cellmask_v1.svg"><img src="docs/models/cards/nuclei_from_cellmask_v1.phone.svg" alt="Score card of Nuclei from cell mask v1 (nuclei_from_cellmask_v1, default): F1 0.888 on 453 held-out fields against 0.201 for stock cpsam_v2. F1 0.888 against 0.201 for stock cpsam_v2 on 453 well-grouped held-out fields, at IoU 0.5."></picture></a><a href="docs/models/cell_from_hoechst_v1.md"><picture><source media="(min-width: 1280px)" srcset="docs/models/cards/cell_from_hoechst_v1.svg"><img src="docs/models/cards/cell_from_hoechst_v1.phone.svg" alt="Score card of Cells from Hoechst v1 (cell_from_hoechst_v1, default): F1 0.870 on 451 held-out fields against 0.301 for stock cpsam_v2. F1 0.870 against stock cpsam_v2&#x27;s 0.301 on 451 held-out fields at IoU 0.5 -- a delta of 0.569."></picture></a><a href="docs/models/toxoplasma_from_hoechst_v1.md"><picture><source media="(min-width: 1280px)" srcset="docs/models/cards/toxoplasma_from_hoechst_v1.svg"><img src="docs/models/cards/toxoplasma_from_hoechst_v1.phone.svg" alt="Score card of Vacuoles from Hoechst v1 (toxoplasma_from_hoechst_v1, default): F1 0.569 on 463 held-out fields against 0.002 for stock cpsam_v2. F1 0.569 against 0.002 for stock cpsam_v2 on 463 well-grouped held-out fields, at IoU 0.5."></picture></a></div>

.. raw:: html

   <div><a href="docs/readme_text/en/generated-model-zoo-1.md"><picture><source media="(min-width: 1280px)" srcset="docs/readme_images/en/generated-model-zoo-1.desktop.svg"><img src="docs/readme_images/en/generated-model-zoo-1.svg" alt="Older models."></picture></a></div>

.. raw:: html

   <div><a href="docs/models/toxoplasma_pv_v3.md"><picture><source media="(min-width: 1280px)" srcset="docs/readme_images/en/generated-model-zoo-1-link-1.desktop.svg"><img src="docs/readme_images/en/generated-model-zoo-1-link-1.svg" alt="toxoplasma_pv_v3"></picture></a><a href="docs/models/toxoplasma_pv_v2.md"><picture><source media="(min-width: 1280px)" srcset="docs/readme_images/en/generated-model-zoo-1-link-2.desktop.svg"><img src="docs/readme_images/en/generated-model-zoo-1-link-2.svg" alt="toxoplasma_pv_v2"></picture></a><a href="docs/models/toxoplasma_pv_v1.md"><picture><source media="(min-width: 1280px)" srcset="docs/readme_images/en/generated-model-zoo-1-link-3.desktop.svg"><img src="docs/readme_images/en/generated-model-zoo-1-link-3.svg" alt="toxoplasma_pv_v1"></picture></a><a href="docs/models/toxoplasma_plaque_v5.md"><picture><source media="(min-width: 1280px)" srcset="docs/readme_images/en/generated-model-zoo-1-link-4.desktop.svg"><img src="docs/readme_images/en/generated-model-zoo-1-link-4.svg" alt="toxoplasma_plaque_v5"></picture></a><a href="docs/models/toxoplasma_plaque_v4.md"><picture><source media="(min-width: 1280px)" srcset="docs/readme_images/en/generated-model-zoo-1-link-5.desktop.svg"><img src="docs/readme_images/en/generated-model-zoo-1-link-5.svg" alt="toxoplasma_plaque_v4"></picture></a><a href="docs/models/toxoplasma_plaque_v2.md"><picture><source media="(min-width: 1280px)" srcset="docs/readme_images/en/generated-model-zoo-1-link-6.desktop.svg"><img src="docs/readme_images/en/generated-model-zoo-1-link-6.svg" alt="toxoplasma_plaque_v2"></picture></a><a href="docs/models/toxoplasma_plaque_v1.md"><picture><source media="(min-width: 1280px)" srcset="docs/readme_images/en/generated-model-zoo-1-link-7.desktop.svg"><img src="docs/readme_images/en/generated-model-zoo-1-link-7.svg" alt="toxoplasma_plaque_v1"></picture></a><a href="docs/models/toxoplasma_well_detector_v2.md"><picture><source media="(min-width: 1280px)" srcset="docs/readme_images/en/generated-model-zoo-1-link-8.desktop.svg"><img src="docs/readme_images/en/generated-model-zoo-1-link-8.svg" alt="toxoplasma_well_detector_v2"></picture></a><a href="docs/models/toxoplasma_well_detector_v1.md"><picture><source media="(min-width: 1280px)" srcset="docs/readme_images/en/generated-model-zoo-1-link-9.desktop.svg"><img src="docs/readme_images/en/generated-model-zoo-1-link-9.svg" alt="toxoplasma_well_detector_v1"></picture></a></div>

.. raw:: html

   <div><a href="docs/readme_text/en/generated-model-zoo-2.md"><picture><source media="(min-width: 1280px)" srcset="docs/readme_images/en/generated-model-zoo-2.desktop.svg"><img src="docs/readme_images/en/generated-model-zoo-2.svg" alt="Alternatives."></picture></a></div>

.. raw:: html

   <div><a href="docs/models/toxoplasma_pv_v5.md"><picture><source media="(min-width: 1280px)" srcset="docs/readme_images/en/generated-model-zoo-2-link-1.desktop.svg"><img src="docs/readme_images/en/generated-model-zoo-2-link-1.svg" alt="toxoplasma_pv_v5"></picture></a><a href="docs/models/toxoplasma_plaque_v3.md"><picture><source media="(min-width: 1280px)" srcset="docs/readme_images/en/generated-model-zoo-2-link-2.desktop.svg"><img src="docs/readme_images/en/generated-model-zoo-2-link-2.svg" alt="toxoplasma_plaque_v3"></picture></a><a href="docs/models/live_cell_v1.md"><picture><source media="(min-width: 1280px)" srcset="docs/readme_images/en/generated-model-zoo-2-link-3.desktop.svg"><img src="docs/readme_images/en/generated-model-zoo-2-link-3.svg" alt="live_cell_v1"></picture></a></div>

.. raw:: html

   <div><a href="docs/readme_text/en/generated-model-zoo-3.md"><picture><source media="(min-width: 1280px)" srcset="docs/readme_images/en/generated-model-zoo-3.desktop.svg"><img src="docs/readme_images/en/generated-model-zoo-3.svg" alt="Mask generator"></picture></a></div>

.. spacr-model-zoo-end

.. raw:: html

   <div><a href="docs/readme_text/en/diagnosing-performance.md"><picture><source media="(min-width: 1280px)" srcset="docs/readme_images/en/diagnosing-performance.desktop.svg"><img src="docs/readme_images/en/diagnosing-performance.svg" alt="Diagnosing performance. Generate a hardware report and attach it to a performance-related issue:"></picture></a></div>

.. code-block:: bash

   python tools/spacr_hardware_report.py

.. raw:: html

   <div><a href="docs/readme_text/en/diagnosing-performance-2.md"><picture><source media="(min-width: 1280px)" srcset="docs/readme_images/en/diagnosing-performance-2.desktop.svg"><img src="docs/readme_images/en/diagnosing-performance-2.svg" alt="Saves to ~/.spacr/reports and prints the path. --quick skips the longer benchmarks; --out PATH sets the location. Reads no project data. Times imports, numeric libraries, window construction and animation, and reports x86_64 emulation on Apple Silicon and NumPy&#x27;s BLAS."></picture></a></div>

.. raw:: html

   <div><a href="docs/readme_text/en/command-line-reference.md"><picture><source media="(min-width: 1280px)" srcset="docs/readme_images/en/command-line-reference.desktop.svg"><img src="docs/readme_images/en/command-line-reference.svg" alt="Command-line reference. Every command below is installed by pip install spacr. All of them accept --help."></picture></a></div>

.. raw:: html

   <div><a href="docs/readme_text/en/launching-the-application.md"><picture><source media="(min-width: 1280px)" srcset="docs/readme_images/en/launching-the-application.desktop.svg"><img src="docs/readme_images/en/launching-the-application.svg" alt="Launching the application."></picture></a></div>

.. code-block:: bash

   spacr              # the desktop application
   spacr-nav
   spacr-navigator
   spacr-tutorial     # the interactive tutorial library
   spacr-server       # no first-run setup screen, for unattended launches

.. raw:: html

   <div><a href="docs/readme_text/en/when-spacr-will-not-start.md"><picture><source media="(min-width: 1280px)" srcset="docs/readme_images/en/when-spacr-will-not-start.desktop.svg"><img src="docs/readme_images/en/when-spacr-will-not-start.svg" alt="When spaCR will not start."></picture></a></div>

.. code-block:: bash

   spacr-doctor       # diagnose the installation and say how to fix it
   safespacr          # the least spaCR that can still change a setting

.. raw:: html

   <div><a href="docs/readme_text/en/when-spacr-will-not-start-2.md"><picture><source media="(min-width: 1280px)" srcset="docs/readme_images/en/when-spacr-will-not-start-2.desktop.svg"><img src="docs/readme_images/en/when-spacr-will-not-start-2.svg" alt="spacr-doctor prints one line per check, with a command to run for each failure. It also reports which spacr is on the path. safespacr reads every preference as its default and forces the backdrop, animations, verbose logging and preloading off. Use it when a saved preference breaks the launch. Preferences saved in safe mode are written normally and persist."></picture></a></div>

.. raw:: html

   <div><a href="docs/readme_text/en/running-modules-headlessly.md"><picture><source media="(min-width: 1280px)" srcset="docs/readme_images/en/running-modules-headlessly.desktop.svg"><img src="docs/readme_images/en/running-modules-headlessly.svg" alt="Running modules headlessly. No Qt, no display — for clusters, servers and CI."></picture></a></div>

.. code-block:: bash

   spacr-run --list                              # modules with a headless entry
   spacr-run --describe MODULE                   # what a module consumes and produces
   spacr-run validate --module MODULE \
       --settings settings.csv                   # check settings before spending the run
   spacr-run MODULE --settings settings.csv      # execute
   spacr-remote --help                           # submit and monitor SSH, Slurm or cloud jobs

.. raw:: html

   <div><a href="docs/readme_text/en/running-modules-headlessly-2.md"><picture><source media="(min-width: 1280px)" srcset="docs/readme_images/en/running-modules-headlessly-2.desktop.svg"><img src="docs/readme_images/en/running-modules-headlessly-2.svg" alt="validate reads the same settings the run would and lists each problem, such as a missing setting or a path that does not exist, with a fix."></picture></a></div>

.. raw:: html

   <div><a href="docs/readme_text/en/inspecting-a-run.md"><picture><source media="(min-width: 1280px)" srcset="docs/readme_images/en/inspecting-a-run.desktop.svg"><img src="docs/readme_images/en/inspecting-a-run.svg" alt="Inspecting a run. Every run is journalled to ~/.spacr/runs with its settings, hashed inputs, outputs, warnings, versions and seeds."></picture></a></div>

.. code-block:: bash

   spacr-repro RUN_DIR        # replay a recorded run from its journal
   spacr-workspace RUN_DIR    # what that run had open: databases, montages, views

.. raw:: html

   <div><a href="docs/readme_text/en/auditing-data-and-installation.md"><picture><source media="(min-width: 1280px)" srcset="docs/readme_images/en/auditing-data-and-installation.desktop.svg"><img src="docs/readme_images/en/auditing-data-and-installation.svg" alt="Auditing data and installation."></picture></a></div>

.. code-block:: bash

   spacr-db-audit DB      # SQLite health, integrity, locking, reader/writer probe
   spacr-leakage          # classifier train/test leakage audit
   spacr-plugins          # installed plugin registry and failure diagnostics

.. raw:: html

   <div><a href="docs/readme_text/en/environment.md"><picture><source media="(min-width: 1280px)" srcset="docs/readme_images/en/environment.desktop.svg"><img src="docs/readme_images/en/environment.svg" alt="Environment."></picture></a></div>

.. code-block:: bash

   SPACR_LOG_LEVEL=DEBUG spacr      # verbose logging for one launch

.. raw:: html

   <div><a href="docs/readme_text/en/environment-2.md"><picture><source media="(min-width: 1280px)" srcset="docs/readme_images/en/environment-2.desktop.svg"><img src="docs/readme_images/en/environment-2.svg" alt="Rotating logs are written to ~/.spacr/logs/spacr.log. Attach that file to a bug report."></picture></a></div>

.. raw:: html

   <div><a href="docs/readme_text/en/contributing-and-support.md"><picture><source media="(min-width: 1280px)" srcset="docs/readme_images/en/contributing-and-support.desktop.svg"><img src="docs/readme_images/en/contributing-and-support.svg" alt="Contributing and support. Submit bug reports and focused feature requests through GitHub Issues. When reporting a failure, include the spaCR version, operating system, Python version, module settings and the relevant log excerpt. spacr-doctor collects most of this information; include the hardware report when reporting performance problems."></picture></a></div>

.. raw:: html

   <div><a href="docs/readme_text/en/licensing.md"><picture><source media="(min-width: 1280px)" srcset="docs/readme_images/en/licensing.desktop.svg"><img src="docs/readme_images/en/licensing.svg" alt="Licensing. New original spaCR material is licensed under the BSD 3-Clause with additional terms. Commercial use requires separate written permission; contact olafsson@med.umich.com. Earlier BSD grants and third-party licenses remain in effect; see LICENSE for scope and notices."></picture></a></div>

.. raw:: html

   <div><a href="https://github.com/EinarOlafsson/spacr/blob/nightly/LICENSE"><picture><source media="(min-width: 1280px)" srcset="docs/readme_images/en/licensing-link-1.desktop.svg"><img src="docs/readme_images/en/licensing-link-1.svg" alt="BSD 3-Clause with additional terms"></picture></a><a href="mailto:olafsson@med.umich.com"><picture><source media="(min-width: 1280px)" srcset="docs/readme_images/en/licensing-link-2.desktop.svg"><img src="docs/readme_images/en/licensing-link-2.svg" alt="olafsson@med.umich.com"></picture></a></div>

.. raw:: html

   <div><a href="docs/readme_text/en/licensing-2.md"><picture><source media="(min-width: 1280px)" srcset="docs/readme_images/en/licensing-2.desktop.svg"><img src="docs/readme_images/en/licensing-2.svg" alt="If spaCR contributed to published work, a citation is appreciated and is not a condition of the licence — see Citing spaCR below."></picture></a></div>

.. raw:: html

   <div><a href="docs/readme_text/en/tutorials.md"><picture><source media="(min-width: 1280px)" srcset="docs/readme_images/en/tutorials.desktop.svg"><img src="docs/readme_images/en/tutorials.svg" alt="Tutorials. The interactive spaCR tutorial library provides installation and module walkthroughs. Available narration and languages are listed for each lesson."></picture></a></div>

.. raw:: html

   <div><a href="docs/readme_text/en/citing-spacr.md"><picture><source media="(min-width: 1280px)" srcset="docs/readme_images/en/citing-spacr.desktop.svg"><img src="docs/readme_images/en/citing-spacr.svg" alt="Citing spaCR. If spaCR contributes to your research, cite: Olafsson EB, et al. A pooled image-based CRISPR screen identifies EAF1 as a T. gondii modulator of ESCRT subversion."></picture></a></div>

.. raw:: html

   <div><a href="https://www.biorxiv.org/content/10.64898/2026.07.08.737057v1"><picture><source media="(min-width: 1280px)" srcset="docs/readme_images/en/citing-spacr-2-1.desktop.svg"><img src="docs/readme_images/en/citing-spacr-2-1.svg" alt="bioRxiv preprint"></picture></a><a href="https://doi.org/10.5281/zenodo.21343316"><picture><source media="(min-width: 1280px)" srcset="docs/readme_images/en/citing-spacr-2-2.desktop.svg"><img src="docs/readme_images/en/citing-spacr-2-2.svg" alt="software archive"></picture></a></div>

.. raw:: html

   <div><a href="docs/readme_text/en/other-work-citing-spacr.md"><picture><source media="(min-width: 1280px)" srcset="docs/readme_images/en/other-work-citing-spacr.desktop.svg"><img src="docs/readme_images/en/other-work-citing-spacr.svg" alt="Other work citing spaCR."></picture></a></div>

.. spacr-citing-papers-begin

.. raw:: html

   <div><a href="docs/readme_text/en/other-work-citing-spacr-2-1.md"><picture><source media="(min-width: 1280px)" srcset="docs/readme_images/en/other-work-citing-spacr-2-1.desktop.svg"><img src="docs/readme_images/en/other-work-citing-spacr-2-1.svg" alt="Metabolic adaptability and nutrient scavenging in Toxoplasma gondii: insights from ingestion pathway-deficient mutants."></picture></a></div>

.. raw:: html

   <div><a href="docs/readme_text/en/other-work-citing-spacr-2-2.md"><picture><source media="(min-width: 1280px)" srcset="docs/readme_images/en/other-work-citing-spacr-2-2.desktop.svg"><img src="docs/readme_images/en/other-work-citing-spacr-2-2.svg" alt="IRE1α promotes phagosomal calcium flux to enhance macrophage fungicidal activity."></picture></a></div>

.. raw:: html

   <div><a href="docs/readme_text/en/other-work-citing-spacr-2-3.md"><picture><source media="(min-width: 1280px)" srcset="docs/readme_images/en/other-work-citing-spacr-2-3.desktop.svg"><img src="docs/readme_images/en/other-work-citing-spacr-2-3.svg" alt="Toxoplasma GRA8 engages the host ESCRT accessory protein ALG-2 and is necessary for parasite metabolic integrity."></picture></a></div>

.. raw:: html

   <div><a href="docs/readme_text/en/other-work-citing-spacr-2-4.md"><picture><source media="(min-width: 1280px)" srcset="docs/readme_images/en/other-work-citing-spacr-2-4.desktop.svg"><img src="docs/readme_images/en/other-work-citing-spacr-2-4.svg" alt="Hydrogen sulfide dynamically upregulates copper uptake and localization."></picture></a></div>

.. raw:: html

   <div><a href="docs/readme_text/en/other-work-citing-spacr-2-5.md"><picture><source media="(min-width: 1280px)" srcset="docs/readme_images/en/other-work-citing-spacr-2-5.desktop.svg"><img src="docs/readme_images/en/other-work-citing-spacr-2-5.svg" alt="spaCR: Spatial phenotype analysis of CRISPR-Cas9 screens (preprint version 1)."></picture></a></div>

.. spacr-citing-papers-end

.. raw:: html

   <div><a href="docs/readme_text/en/acknowledgments.md"><picture><source media="(min-width: 1280px)" srcset="docs/readme_images/en/acknowledgments.desktop.svg"><img src="docs/readme_images/en/acknowledgments.svg" alt="Acknowledgments. spaCR builds on open scientific software including NumPy, pandas, scikit-image, scikit-learn, Cellpose, PyTorch and Qt."></picture></a></div>
