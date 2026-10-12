|Website| |Platforms| |Python| |Qt| |Test counts| |Coverage| |Issues| |Conda| |PyPI| |Docker| |PyPI Downloads| |Docs| |Tutorials| |Preprint| |DOI| |Cite| |License| |PyPI rank|

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
.. |Coverage| image:: https://img.shields.io/codecov/c/github/EinarOlafsson/spacr/nightly?logo=codecov&logoColor=white
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
.. |License| image:: https://img.shields.io/badge/License-BSD--3-blue?logo=opensourceinitiative&logoColor=white
   :target: https://github.com/EinarOlafsson/spacr/blob/main/LICENSE
   :alt: BSD 3-Clause license
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

.. image:: docs/source/_static/deck/slides/slide_01.jpg
   :alt: spaCR
   :width: 920
   :target: https://einarolafsson.github.io/spacr/_static/deck/

`← Back <docs/source/_static/deck/pages/57.md>`_   `Next → <docs/source/_static/deck/pages/02.md>`_

spaCR
=====

.. spacr-language-picker-begin

Languages: `🌐 English ▾ <docs/i18n/readme/README.md>`_

.. spacr-language-picker-end

**Spatial phenotype analysis of CRISPR screens.**

spaCR segments and measures single cells in microscopy images,
integrates per-object phenotypes with sequencing-derived guide abundance, and
estimates which genes are associated with phenotypic changes. Starting from
plate images and FASTQ reads, it produces per-object measurements, trained
classifiers, per-guide and per-gene effect estimates, and a ranked hit list.

The segmentation, measurement, annotation and classification modules also
run without a sequencing arm.

Make Masks corrects segmentation masks and annotates independent,
class-labelled rectangles with the **Box** tool for YOLO export. Boxes keep
their own labels and history without changing source images or masks.

See the `feature guide <docs/source/features.rst>`_ for each tool.

Images, masks and crops are files on disk. Measurements, annotations and
model predictions are tables in the project's ``measurements.db`` SQLite
database.

Runs as a desktop application or headlessly on a workstation, server or
cluster.

Try spaCR
~~~~~~~~~

.. code-block:: bash

   conda create -n spacr python=3.12 -y
   conda activate spacr
   python -m pip install spacr
   spacr

Use **Load test data…** in Import, Make Masks, Annotate or an assay screen
to download example data. From a terminal, use ``spacr-download``.

Hardware support
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


Install spaCR
~~~~~~~~~~~~~

Choose an install route:

- **Desktop application**: a Linux, macOS or Windows installer with its own Python.
- **pip**: ``python -m pip install spacr`` in a Conda environment.
- **conda-forge**: ``conda install conda-forge::spacr`` in a Conda environment.
- **Docker**: the command-line pipelines in a Linux container, from the CPU image or the NVIDIA GPU image:

.. code-block:: bash

   docker run --rm ghcr.io/einarolafsson/spacr:1.5.1.4 spacr-run --list
   docker run --rm --gpus all ghcr.io/einarolafsson/spacr:1.5.1.4-cuda12.4 spacr-run --list

The `Docker guide <docs/source/installer_guide.rst#container-images>`_ covers
GPU setup, data mounts and pipeline commands.

Desktop application
-------------------

The installers bundle their own Python. Conda is not required.

.. spacr-installer-links-begin

|InstallerLinux| |InstallerMacOS| |InstallerWindows|

.. |InstallerWindows| image:: spacr/resources/icons/platforms/windows.png
   :width: 64
   :alt: Download spaCR 1.5.1.4 for Windows 10/11
   :target: https://github.com/EinarOlafsson/spacr/releases/download/v1.5.1.4/spaCR-1.5.1.4-Windows-Online-Setup.exe
.. |InstallerMacOS| image:: spacr/resources/icons/platforms/macos.png
   :width: 64
   :alt: Download spaCR 1.5.1.4 for macOS 11+ (Intel and Apple silicon)
   :target: https://github.com/EinarOlafsson/spacr/releases/download/v1.5.1.4/spaCR-1.5.1.4-macOS-Universal-Online.pkg
.. |InstallerLinux| image:: spacr/resources/icons/platforms/linux.png
   :width: 64
   :alt: Download spaCR 1.5.1.4 for 64-bit Linux
   :target: https://github.com/EinarOlafsson/spacr/releases/download/v1.5.1.4/spaCR-1.5.1.4-Linux-x86_64-Online.run

.. spacr-installer-links-end

On Linux, make the downloaded file executable and run it:

.. code-block:: bash

   chmod +x SpaCR-*-Linux-x86_64-Online.run
   ./SpaCR-*-Linux-x86_64-Online.run

On macOS, open the ``.pkg``. The current beta is not notarized; if Gatekeeper
blocks it, choose **System Settings → Privacy & Security → Open Anyway**.

See the `installer guide <docs/source/installer_guide.rst>`_ for update, uninstall,
offline and troubleshooting instructions, and `system requirements
<docs/source/system_requirements.rst>`_ for workstation and server recommendations
and the GPU compatibility tables.

PyPI installation
-----------------

For the PyPI release, install spaCR with pip inside a Conda
environment. Python 3.12 has the widest choice of optional scientific packages:

.. code-block:: bash

   conda create -n spacr python=3.12 -y
   conda activate spacr
   python -m pip install --upgrade pip
   python -m pip install spacr
   spacr

spaCR supports Python **3.9 through 3.14**, except Python 3.14.1, which
torchvision excludes. Linux is recommended for the heaviest CUDA and ROCm
workflows; macOS and Windows are also supported, and both use their GPUs —
macOS through Metal, which covers Apple Silicon and the AMD cards in Intel
Macs, and Windows through CUDA or DirectML.

The standard installation includes the Qt desktop interface. For a server,
cluster or CI runner, run the command-line pipelines without opening it:

.. code-block:: bash

   python -m pip install spacr
   spacr-run --list

Optional integrations are installed separately, for example
``spacr[zarr]``, ``spacr[omero]``, ``spacr[napari]`` and
``spacr[czi,nd2,lif]``. See the `installation guide
<docs/source/installer_guide.rst>`_ for the complete extras and Python-version
compatibility table.

Conda-forge installation
------------------------

The official conda-forge package installs spaCR and its desktop dependencies
into the active environment:

.. code-block:: bash

   conda create -n spacr python=3.12 -y
   conda activate spacr
   conda install conda-forge::spacr
   spacr

Install from source
-------------------

Clone the repository and install it in editable mode, so your working copy
*is* the installed package and edits take effect without reinstalling::

    git clone https://github.com/EinarOlafsson/spacr.git
    cd spacr
    conda create -n spacr python=3.12 -y
    conda activate spacr
    pip install -e .
    spacr

This clones ``main``, the default branch, which carries the latest release.
Development happens on ``nightly``; add ``--branch nightly`` to clone it
instead. For a specific release::

    git clone --branch v1.5.0.5 https://github.com/EinarOlafsson/spacr.git

To pull later changes, from inside the clone::

    git pull
    pip install -e .

Reinstall when dependencies or entry points change. Python edits take effect
directly; ``spacr-doctor`` identifies the active installation.

Install from source (light)
---------------------------

Contributors need history; to run spaCR, choose below. Measurements:
``nightly`` at ``05302fd5c`` on 2026-10-07, using
``packaging/measure_clone_forms.sh``::

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

The full nightly clone downloaded 9.25 GiB. Adding ``--filter=blob:none``
to the shallow clone does not reduce the checkout: its object store is still
2032 MB. Git does not report lazy blob fetches, so no download total was
measured. The nightly tracked tree is a 3416 MB checkout (measured 2026-10-08),
excluding Git history. Download sizes and times vary with the branch.


Command-line entry points
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

``spacr-run --list`` lists modules with headless command-line entry points.
GUI-only annotation, curation, comparison and exploration modules are omitted.


Core workflow
-------------

The primary workflow comprises six modules:

- **Mask** segments cells, nuclei, pathogens and organelles with Cellpose.
- **Measure** writes morphology, intensity, texture, spatial and
  colocalization features, together with object crops, to SQLite.
- **Annotate** labels crops in a keyboard-driven grid and supports
  active-learning queues.
- **Classify** trains image or measurement-based models and records held-out
  performance with each checkpoint.
- **Map Barcodes** maps FASTQ reads to wells and gRNAs, with abundance,
  collision and coverage QC.
- **Regression** estimates guide, gene, condition and control effects with
  model families suited to continuous, fractional and count responses.

spaCR modules
-------------

.. spacr-workflow-begin

Core
^^^^

Core sequence from microscopy images through segmentation, measurements,
annotations, classification, barcode mapping and regression.

| |Module_mask|\ |Module_measure|\ |Module_annotate|\ |Module_classify_merged|\ |Module_map_barcodes|\ |Module_regression|

Data
^^^^

Import images and tables into spaCR projects and execute reproducible
multi-plate workflows.

| |Module_foreign|\ |Module_embeddings|\ |Module_run_compare|\ |Module_experiment_design|\ |Module_power|\ |Module_dose_response|
| |Module_qc_dashboard|

Tools
^^^^^

Point these at a project: edit masks by hand, stitch tiles, read an
embedding, draw a gate, build a plot, check quality.

| |Module_make_masks|\ |Module_align|\ |Module_umap|\ |Module_gate_editor|\ |Module_graph_builder|

Organism
^^^^^^^^

Organism-specific image analysis and quantitative assay readouts.

| |Module_toxoplasma|

.. |Module_mask| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/core/index.html#spacr.core.preprocess_generate_masks"><img src="spacr/resources/icons/workflow/mask.png" width="16.0%" align="middle" alt="Open the Mask API"></a>

.. |Module_measure| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/measure/index.html"><img src="spacr/resources/icons/workflow/measure.png" width="16.0%" align="middle" alt="Open the Measure API"></a>

.. |Module_annotate| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/qt/screens/annotate/index.html"><img src="spacr/resources/icons/workflow/annotate.png" width="16.0%" align="middle" alt="Open the Annotate API"></a>

.. |Module_classify_merged| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/classify/index.html"><img src="spacr/resources/icons/workflow/classify_merged.png" width="16.0%" align="middle" alt="Open the Classify API"></a>

.. |Module_map_barcodes| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/sequencing/index.html"><img src="spacr/resources/icons/workflow/map_barcodes.png" width="16.0%" align="middle" alt="Open the Map Barcodes API"></a>

.. |Module_regression| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/ml/index.html"><img src="spacr/resources/icons/workflow/regression.png" width="16.0%" align="middle" alt="Open the Regression API"></a>

.. |Module_foreign| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/foreign/index.html"><img src="spacr/resources/icons/workflow/apps/foreign.png" width="16.0%" align="middle" alt="Open the Import API"></a>

.. |Module_embeddings| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/qt/screens/embeddings/index.html"><img src="spacr/resources/icons/workflow/apps/embeddings.png" width="16.0%" align="middle" alt="Open the Embeddings API"></a>

.. |Module_run_compare| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/qt/screens/run_compare/index.html"><img src="spacr/resources/icons/workflow/apps/run_compare.png" width="16.0%" align="middle" alt="Open the Run Compare API"></a>

.. |Module_experiment_design| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/qt/screens/experiment_design/index.html"><img src="spacr/resources/icons/workflow/apps/experiment_design.png" width="16.0%" align="middle" alt="Open the Experiment Design API"></a>

.. |Module_power| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/qt/screens/power/index.html"><img src="spacr/resources/icons/workflow/apps/power.png" width="16.0%" align="middle" alt="Open the Power / Design API"></a>

.. |Module_dose_response| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/qt/screens/dose_response/index.html"><img src="spacr/resources/icons/workflow/apps/dose_response.png" width="16.0%" align="middle" alt="Open the Dose–Response API"></a>

.. |Module_qc_dashboard| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/qt/screens/qc_dashboard/index.html"><img src="spacr/resources/icons/workflow/apps/qc_dashboard.png" width="16.0%" align="middle" alt="Open the QC API"></a>

.. |Module_make_masks| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/qt/screens/make_masks/index.html"><img src="spacr/resources/icons/workflow/apps/make_masks.png" width="16.0%" align="middle" alt="Open the Make Masks API"></a>

.. |Module_align| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/align/index.html"><img src="spacr/resources/icons/workflow/apps/align.png" width="16.0%" align="middle" alt="Open the Align &amp; Stitch API"></a>

.. |Module_umap| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/core/index.html#spacr.core.generate_image_umap"><img src="spacr/resources/icons/workflow/apps/umap.png" width="16.0%" align="middle" alt="Open the Image UMAP API"></a>

.. |Module_gate_editor| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/qt/screens/gate_editor/index.html"><img src="spacr/resources/icons/workflow/apps/gate_editor.png" width="16.0%" align="middle" alt="Open the Gate Editor API"></a>

.. |Module_graph_builder| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/qt/screens/graph_builder/index.html"><img src="spacr/resources/icons/workflow/apps/graph_builder.png" width="16.0%" align="middle" alt="Open the Graph Builder API"></a>

.. |Module_toxoplasma| raw:: html

   <a href="https://einarolafsson.github.io/spacr/api/spacr/qt/screens/organism_screen/index.html#spacr-qt-screens-organism-screen-toxoplasma"><img src="spacr/resources/icons/workflow/apps/toxoplasma.png" width="16.0%" align="middle" alt="Open the Toxoplasma API"></a>

.. spacr-workflow-end

Every module with a Home tile, in Home's order: the six pipeline modules
first, then the rest. Select a tile to open that
module's API page.

Other resources
~~~~~~~~~~~~~~~

- `Interactive tutorials <https://einarolafsson.github.io/spacr/tutorials/>`_
  — guided workflows from installation through hit investigation.
- `Python API quickstart <docs/source/python_api.rst>`_ — run and validate
  pipelines from scripts, notebooks or a cluster.
- `Feature guide <docs/source/features.rst>`_ — capabilities, maturity and
  optional integrations.
- `Curated API reference
  <https://einarolafsson.github.io/spacr/api/index.html>`_ — supported entry
  points by task, with the complete module reference one level deeper.
- `Language & translation guide <docs/source/localization.rst>`_ — interface
  languages, contextual help and scientific-output policy.

Language & translation
~~~~~~~~~~~~~~~~~~~~~~

The interface supports ten languages across navigation and Preferences. AI and
LIVE controls, module descriptions and reviewed contextual help are also
translated. Change the language under **spaCR → Preferences → Language**
without restarting. Logs, paths, database values and measurements are never
translated; scientific output remains canonical English. See the
`contextual-help policy <docs/source/localization.rst#contextual-help>`_.

The nine non-English catalogs are machine-drafted and technically reviewed
rather than read end to end by a native speaker. The
`review scope <docs/i18n/REVIEW_SCOPE_2026-09-04.md>`_ records which languages
have had a human pass and every term left in English by decision.

Animated setting guidance
~~~~~~~~~~~~~~~~~~~~~~~~~

Settings with a visual explanation offer an **Animation** control in their
tooltip. Browse the `setting animation gallery
<https://einarolafsson.github.io/spacr/setting_animations.html>`_ or the
`Setting animation registry
<https://einarolafsson.github.io/spacr/api/spacr/setting_animations/index.html>`_.

Data
~~~~

Reference datasets
~~~~~~~~~~~~~~~~~~

|DataBioStudies| |DataHuggingFace| |DataNCBI| |DataSpaCRPower| |DataBioRxiv|

.. |DataBioStudies| image:: spacr/resources/icons/databanks/biostudies_button.png
   :width: 72
   :alt: Open the BioStudies microscopy dataset
   :target: https://doi.org/10.6019/S-BIAD2135
.. |DataHuggingFace| image:: spacr/resources/icons/databanks/huggingface_button.png
   :width: 72
   :alt: Open the Hugging Face testing dataset
   :target: https://huggingface.co/datasets/einarolafsson/toxo_mito
.. |DataNCBI| image:: spacr/resources/icons/databanks/ncbi_button.png
   :width: 72
   :alt: Open the NCBI sequencing dataset
   :target: https://www.ncbi.nlm.nih.gov/bioproject/?term=PRJNA1261935
.. |DataSpaCRPower| image:: spacr/resources/icons/databanks/spacrpower_button.png
   :width: 72
   :alt: Open spaCRPower
   :target: https://github.com/maomlab/spaCRPower
.. |DataBioRxiv| image:: spacr/resources/icons/databanks/biorxiv_button.png
   :width: 72
   :alt: Open the bioRxiv preprint
   :target: https://www.biorxiv.org/content/10.64898/2026.07.08.737057v1

Model zoo
~~~~~~~~~

spaCR ships a catalogue of trained models and fetches them on demand. Open
**Model Zoo** from the home screen to browse and install them, or name a key
in a settings file -- ``pathogen_model: toxoplasma_pv_v1`` -- and the model is
downloaded and checksum-verified the first time it is needed. Every published
entry carries a SHA-256; an entry without one is refused rather than installed,
because a truncated or substituted checkpoint cannot be told from the real one.

.. spacr-model-zoo-begin

.. list-table::
   :widths: 50 50

   * - |zoo_card_toxoplasma_pv_v4|

       `toxoplasma_pv_v4 <docs/models/toxoplasma_pv_v4.md>`__
     - |zoo_card_toxoplasma_plaque_v4|

       `toxoplasma_plaque_v4 <docs/models/toxoplasma_plaque_v4.md>`__
   * - |zoo_card_toxoplasma_well_detector_v3|

       `toxoplasma_well_detector_v3 <docs/models/toxoplasma_well_detector_v3.md>`__
     - |zoo_card_toxoplasma_from_cellmask_v1|

       `toxoplasma_from_cellmask_v1 <docs/models/toxoplasma_from_cellmask_v1.md>`__
   * - |zoo_card_live_cell_v1|

       `live_cell_v1 <docs/models/live_cell_v1.md>`__
     - |zoo_card_nuclei_from_cellmask_v1|

       `nuclei_from_cellmask_v1 <docs/models/nuclei_from_cellmask_v1.md>`__
   * - |zoo_card_cell_from_hoechst_v1|

       `cell_from_hoechst_v1 <docs/models/cell_from_hoechst_v1.md>`__
     - |zoo_card_toxoplasma_from_hoechst_v1|

       `toxoplasma_from_hoechst_v1 <docs/models/toxoplasma_from_hoechst_v1.md>`__

.. |zoo_card_toxoplasma_pv_v4| image:: docs/models/cards/toxoplasma_pv_v4.svg
   :alt: Score card of Toxoplasma PV v4 (round 7) (default): F1 at IoU 0.5 0.854 against 0.765 for stock cpsam_v2. F1 0.854 against stock cpsam_v2's 0.765 on the 11 anchor wells at IoU 0.5; AJI 0.776 against 0.505.
   :target: docs/models/toxoplasma_pv_v4.md
   :width: 380

.. |zoo_card_toxoplasma_plaque_v4| image:: docs/models/cards/toxoplasma_plaque_v4.svg
   :alt: Score card of Toxoplasma Plaque v4 (round 6, cpsam_v2) (default): F1 at IoU 0.5 0.815 against 0.373 for stock cpsam_v2. F1 0.815 (toxoplasma_plaque_v3 0.844) on the 263 held-out fields at IoU 0.5; stock cpsam_v2 0.373 there with the Cellpose evaluation scorer.
   :target: docs/models/toxoplasma_plaque_v4.md
   :width: 380

.. |zoo_card_toxoplasma_well_detector_v3| image:: docs/models/cards/toxoplasma_well_detector_v3.svg
   :alt: Score card of Toxoplasma Well Detector v3 (YOLO11 Gel Doc) (default): Box mAP at IoU 0.5 0.994. Stock was not evaluated in this run; see the named incumbent comparison on the model card.
   :target: docs/models/toxoplasma_well_detector_v3.md
   :width: 380

.. |zoo_card_toxoplasma_from_cellmask_v1| image:: docs/models/cards/toxoplasma_from_cellmask_v1.svg
   :alt: Score card of Toxoplasma from Cell Mask (cross-channel) (default): F1 at IoU 0.5 0.606 against 0.021 for stock cpsam_v2. F1 0.606 against 0.021 for stock cpsam_v2 on 463 well-grouped held-out fields, at IoU 0.5.
   :target: docs/models/toxoplasma_from_cellmask_v1.md
   :width: 380

.. |zoo_card_live_cell_v1| image:: docs/models/cards/live_cell_v1.svg
   :alt: Score card of Live cell v1 (phase, brightfield, DIC) (default): F1 at IoU 0.5, all test fields 0.694 against 0.738 for stock cpsam_v2. On the datasets stock cpsam_v2 never trained on, F1 0.960 against 0.885 at IoU 0.5; over all 2,199 test fields, 0.694 against 0.738, because stock trained on LIVECell and YeaZ and wins on LIVECell.
   :target: docs/models/live_cell_v1.md
   :width: 380

.. |zoo_card_nuclei_from_cellmask_v1| image:: docs/models/cards/nuclei_from_cellmask_v1.svg
   :alt: Score card of Cross-channel nuclei-from-cellmask (default): F1 at IoU 0.5 0.888 against 0.201 for stock cpsam_v2. F1 0.888 against 0.201 for stock cpsam_v2 on 453 well-grouped held-out fields, at IoU 0.5.
   :target: docs/models/nuclei_from_cellmask_v1.md
   :width: 380

.. |zoo_card_cell_from_hoechst_v1| image:: docs/models/cards/cell_from_hoechst_v1.svg
   :alt: Score card of Cross-channel cell-from-hoechst (default): F1 at IoU 0.5 0.870 against 0.301 for stock cpsam_v2. F1 0.870 against stock cpsam_v2's 0.301 on 451 held-out fields at IoU 0.5 -- a delta of 0.569.
   :target: docs/models/cell_from_hoechst_v1.md
   :width: 380

.. |zoo_card_toxoplasma_from_hoechst_v1| image:: docs/models/cards/toxoplasma_from_hoechst_v1.svg
   :alt: Score card of Toxoplasma from Hoechst (cross-channel) (default): F1 at IoU 0.5 0.569 against 0.002 for stock cpsam_v2. F1 0.569 against 0.002 for stock cpsam_v2 on 463 well-grouped held-out fields, at IoU 0.5.
   :target: docs/models/toxoplasma_from_hoechst_v1.md
   :width: 380

Older models: `toxoplasma_pv_v3 <docs/models/toxoplasma_pv_v3.md>`__, `toxoplasma_pv_v2 <docs/models/toxoplasma_pv_v2.md>`__, `toxoplasma_pv_v1 <docs/models/toxoplasma_pv_v1.md>`__, `toxoplasma_plaque_v2 <docs/models/toxoplasma_plaque_v2.md>`__, `toxoplasma_plaque_v1 <docs/models/toxoplasma_plaque_v1.md>`__, `toxoplasma_well_detector_v2 <docs/models/toxoplasma_well_detector_v2.md>`__, `toxoplasma_well_detector_v1 <docs/models/toxoplasma_well_detector_v1.md>`__

Alternatives: `toxoplasma_pv_v5 <docs/models/toxoplasma_pv_v5.md>`__, `toxoplasma_plaque_v3 <docs/models/toxoplasma_plaque_v3.md>`__

.. spacr-model-zoo-end


Diagnosing performance
----------------------

Generate a hardware report and attach it to a performance-related issue::

    python tools/spacr_hardware_report.py

Saves to ``~/.spacr/reports`` and prints the path. ``--quick`` skips the
longer benchmarks; ``--out PATH`` sets the location.

Reads no project data. Times imports, numeric libraries, window
construction and animation, and reports x86_64 emulation on Apple Silicon
and NumPy's BLAS.

Command-line reference
----------------------

Every command below is installed by ``pip install spacr``. All of them accept
``--help``.

Launching the application
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   spacr              # the desktop application
   spacr-tutorial     # the interactive tutorial library
   spacr-server       # no first-run setup screen, for unattended launches

When spaCR will not start
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   spacr-doctor       # diagnose the installation and say how to fix it
   safespacr          # the least spaCR that can still change a setting

``spacr-doctor`` prints one line per check, with a command to run for each
failure. It also reports which ``spacr`` is on the path.

``safespacr`` reads every preference as its default and forces the backdrop,
animations, verbose logging and preloading off. Use it when a saved
preference breaks the launch. Preferences saved in safe mode are written
normally and persist.

Running modules headlessly
~~~~~~~~~~~~~~~~~~~~~~~~~~

No Qt, no display — for clusters, servers and CI.

.. code-block:: bash

   spacr-run --list                              # modules with a headless entry
   spacr-run --describe MODULE                   # what a module consumes and produces
   spacr-run validate --module MODULE \
       --settings settings.csv                   # check settings before spending the run
   spacr-run MODULE --settings settings.csv      # execute
   spacr-remote --help                           # submit and monitor SSH, Slurm or cloud jobs

``validate`` reads the same settings the run would and lists each problem,
such as a missing setting or a path that does not exist, with a fix.

Inspecting a run
~~~~~~~~~~~~~~~~

Every run is journalled to ``~/.spacr/runs`` with its settings, hashed
inputs, outputs, warnings, versions and seeds.

.. code-block:: bash

   spacr-repro RUN_DIR        # replay a recorded run from its journal
   spacr-workspace RUN_DIR    # what that run had open: databases, montages, views

Auditing data and installation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   spacr-db-audit DB      # SQLite health, integrity, locking, reader/writer probe
   spacr-leakage          # classifier train/test leakage audit
   spacr-plugins          # installed plugin registry and failure diagnostics

Environment
~~~~~~~~~~~

.. code-block:: bash

   SPACR_LOG_LEVEL=DEBUG spacr      # verbose logging for one launch

Rotating logs are written to ``~/.spacr/logs/spacr.log``. Attach that file
to a bug report.


Contributing and support
~~~~~~~~~~~~~~~~~~~~~~~~

Submit bug reports and focused feature requests through
`GitHub Issues <https://github.com/EinarOlafsson/spacr/issues>`_.
When reporting a failure, include the spaCR version, operating system, Python
version, module settings and the relevant log excerpt. ``spacr-doctor``
collects most of this information; include the hardware report when reporting
performance problems.

Licensing
~~~~~~~~~

spaCR is released under the `BSD 3-Clause License
<https://github.com/EinarOlafsson/spacr/blob/main/LICENSE>`_.

If spaCR contributed to published work, a citation is appreciated and is not
a condition of the licence — see `Citing spaCR`_ below.

Tutorials
~~~~~~~~~

The `interactive spaCR tutorial library
<https://einarolafsson.github.io/spacr/tutorials/>`_ provides installation and
module walkthroughs. Available narration and languages are listed for each lesson.

Citing spaCR
~~~~~~~~~~~~

If spaCR contributes to your research, cite:

Olafsson EB, *et al.* A pooled image-based CRISPR screen identifies
EAF1 as a *T. gondii* modulator of ESCRT subversion.

`bioRxiv preprint <https://www.biorxiv.org/content/10.64898/2026.07.08.737057v1>`_ ·
`software archive <https://doi.org/10.5281/zenodo.21343316>`_

Other work citing spaCR
~~~~~~~~~~~~~~~~~~~~~~~

.. spacr-citing-papers-begin

* `Metabolic adaptability and nutrient scavenging in Toxoplasma gondii: insights from ingestion pathway-deficient mutants. <https://journals.asm.org/doi/full/10.1128/msphere.01011-24>`_
* `IRE1α promotes phagosomal calcium flux to enhance macrophage fungicidal activity. <https://www.cell.com/cell-reports/fulltext/S2211-1247(25)00465-6>`_
* `Toxoplasma GRA8 engages the host ESCRT accessory protein ALG-2 and is necessary for parasite metabolic integrity. <https://www.biorxiv.org/content/10.64898/2026.07.20.739547v1.abstract>`_
* `Hydrogen sulfide dynamically upregulates copper uptake and localization. <https://www.biorxiv.org/content/10.64898/2026.07.30.741779v1>`_
* `spaCR: Spatial phenotype analysis of CRISPR-Cas9 screens (preprint version 1). <https://www.researchsquare.com/article/rs-7368254/v1>`_

.. spacr-citing-papers-end

Acknowledgments
~~~~~~~~~~~~~~~

spaCR builds on open scientific software including NumPy, pandas,
scikit-image, scikit-learn, Cellpose, PyTorch and Qt. See the
`translation model attribution <docs/i18n/TRANSLATION_MODELS.md>`_ for the
models used to prepare the multilingual documentation and interface catalogs.
