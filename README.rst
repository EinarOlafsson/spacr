|Docs| |PyPI| |Python| |Tests| |Qt| |License| |DOI|

.. |Docs| image:: https://github.com/EinarOlafsson/spacr/actions/workflows/pages/pages-build-deployment/badge.svg
   :target: https://einarolafsson.github.io/spacr/
   :alt: Documentation
.. |PyPI| image:: https://img.shields.io/pypi/v/spacr
   :target: https://pypi.org/project/spacr/
   :alt: PyPI version
.. |Python| image:: https://img.shields.io/pypi/pyversions/spacr
   :target: https://pypi.org/project/spacr/
   :alt: Supported Python versions
.. |Tests| image:: https://github.com/EinarOlafsson/spacr/actions/workflows/tests.yml/badge.svg
   :target: https://github.com/EinarOlafsson/spacr/actions/workflows/tests.yml
   :alt: Test suite
.. |Qt| image:: https://img.shields.io/badge/GUI-Qt%20%28PySide6%29-41CD52
   :target: https://einarolafsson.github.io/spacr/
   :alt: Qt interface
.. |License| image:: https://img.shields.io/github/license/EinarOlafsson/spacr
   :target: https://github.com/EinarOlafsson/spacr/blob/main/LICENSE
   :alt: MIT license
.. |DOI| image:: https://img.shields.io/badge/DOI-10.5281%2Fzenodo.21343317-blue
   :target: https://doi.org/10.5281/zenodo.21343317
   :alt: Zenodo DOI

.. image:: https://raw.githubusercontent.com/EinarOlafsson/spacr/main/spacr/resources/icons/logo_spacr.png
   :alt: spaCR
   :align: center
   :width: 360

spaCR
=====

**Spatial phenotype analysis of CRISPR screens.**

spaCR is an end-to-end platform for image-based pooled CRISPR screens,
high-content microscopy, and single-cell phenotype discovery. It connects
raw microscopy images and sequencing reads to segmentation, measurements,
annotation, predictive models, screen scores, quality control, and
publication-ready results.

Every object stays traceable. Images, masks, measurements, annotations,
model predictions, barcodes, and experimental identifiers are linked through
a SQLite-backed project rather than scattered across unrelated files.

`Documentation <https://einarolafsson.github.io/spacr/>`_ ·
`PyPI <https://pypi.org/project/spacr/>`_ ·
`Source <https://github.com/EinarOlafsson/spacr>`_ ·
`Issues <https://github.com/EinarOlafsson/spacr/issues>`_


Why spaCR?
----------

- **One connected workflow — Stable.** Move from microscope output and FASTQ
  files to object-level phenotypes and gene-level screen results.
- **Biology-aware data model — Stable.** Keep plate, well, field, object,
  crop, annotation, prediction, and barcode identities linked.
- **Desktop and headless operation — Stable.** Use the PySide6 application
  interactively or run the same modules on a workstation, server, or cluster.
- **Live visual feedback — Stable/Beta.** Preview masks, tracks, timelapse
  frames, training metrics, activation maps, and image embeddings where the
  corresponding module supports them.
- **Reproducible execution — Stable.** Validate settings, record manifests,
  preserve run journals, rotate logs, resume supported jobs, and export
  reports with settings and package versions.
- **CPU and GPU execution — Stable.** Run general analysis on CPU and use
  CUDA automatically for supported segmentation and deep-learning workloads.


Features
--------

Core
~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 24 10 46

   * - Module
     - Feature
     - State
     - Description
   * - `Core <https://einarolafsson.github.io/spacr/api/spacr/core/index.html>`_
     - `Image pipeline <https://einarolafsson.github.io/spacr/api/spacr/core/index.html>`_
     - Stable
     - Runs the connected image-to-object processing workflow.
   * - `Qt application <https://einarolafsson.github.io/spacr/api/spacr/qt/app/index.html>`_
     - `Desktop interface <https://einarolafsson.github.io/spacr/api/spacr/qt/app/index.html>`_
     - Stable
     - Provides the modern module-based desktop application.
   * - `Batch <https://einarolafsson.github.io/spacr/api/spacr/batch/index.html>`_
     - `Headless execution <https://einarolafsson.github.io/spacr/api/spacr/batch/index.html>`_
     - Stable
     - Runs validated modules from scripts, servers, and clusters.
   * - `Logging <https://einarolafsson.github.io/spacr/api/spacr/logging_util/index.html>`_
     - `Reproducible runs <https://einarolafsson.github.io/spacr/api/spacr/logging_util/index.html>`_
     - Stable
     - Records settings, manifests, journals, progress, and rotating logs.

Data and I/O
~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 24 10 46

   * - Module
     - Feature
     - State
     - Description
   * - `I/O <https://einarolafsson.github.io/spacr/api/spacr/io/index.html>`_
     - `Image import <https://einarolafsson.github.io/spacr/api/spacr/io/index.html>`_
     - Stable
     - Reads microscopy images and maintains linked project data.
   * - `I/O <https://einarolafsson.github.io/spacr/api/spacr/io/index.html>`_
     - `Format converter <https://einarolafsson.github.io/spacr/api/spacr/io/index.html>`_
     - Alpha
     - Converts ND2, CZI, LIF, and OME-TIFF acquisitions.
   * - `I/O <https://einarolafsson.github.io/spacr/api/spacr/io/index.html>`_
     - `Project import <https://einarolafsson.github.io/spacr/api/spacr/io/index.html>`_
     - Alpha
     - Imports external images, masks, tables, and databases.
   * - `Object data <https://einarolafsson.github.io/spacr/api/spacr/object/index.html>`_
     - `Object schema <https://einarolafsson.github.io/spacr/api/spacr/object/index.html>`_
     - Stable
     - Links images, masks, crops, measurements, and identifiers.
   * - `I/O <https://einarolafsson.github.io/spacr/api/spacr/io/index.html>`_
     - `Database browser <https://einarolafsson.github.io/spacr/api/spacr/io/index.html>`_
     - Alpha
     - Filters, inspects, and exports project SQLite tables.
   * - `Stitching <https://einarolafsson.github.io/spacr/api/spacr/spacrops/index.html>`_
     - `Align and stitch <https://einarolafsson.github.io/spacr/api/spacr/spacrops/index.html>`_
     - Alpha
     - Registers tile layouts and writes large stitched canvases.

Segmentation and masks
~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 24 10 46

   * - Module
     - Feature
     - State
     - Description
   * - `Mask <https://einarolafsson.github.io/spacr/api/spacr/core/index.html>`_
     - `2D mask generation <https://einarolafsson.github.io/spacr/api/spacr/core/index.html#spacr.core.preprocess_generate_masks>`_
     - Stable
     - Generates cell, nucleus, pathogen, and organelle masks.
   * - `Mask <https://einarolafsson.github.io/spacr/api/spacr/core/index.html>`_
     - `3D mask generation <https://einarolafsson.github.io/spacr/api/spacr/core/index.html#spacr.core.preprocess_generate_masks>`_
     - Beta
     - Segments volumetric image stacks.
   * - `Mask <https://einarolafsson.github.io/spacr/api/spacr/core/index.html>`_
     - `4D mask generation <https://einarolafsson.github.io/spacr/api/spacr/core/index.html#spacr.core.preprocess_generate_masks>`_
     - Beta
     - Segments volumetric time series.
   * - `Mask <https://einarolafsson.github.io/spacr/api/spacr/core/index.html>`_
     - `Live preview <https://einarolafsson.github.io/spacr/api/spacr/core/index.html>`_
     - Stable
     - Previews segmentation settings before a full run.
   * - `Cellpose <https://einarolafsson.github.io/spacr/api/spacr/spacr_cellpose/index.html>`_
     - `Make masks <https://einarolafsson.github.io/spacr/api/spacr/spacr_cellpose/index.html>`_
     - Beta
     - Creates and edits mask-training datasets.
   * - `Cellpose <https://einarolafsson.github.io/spacr/api/spacr/spacr_cellpose/index.html>`_
     - `Train Cellpose <https://einarolafsson.github.io/spacr/api/spacr/spacr_cellpose/index.html>`_
     - Beta
     - Trains custom segmentation checkpoints.
   * - `Cellpose <https://einarolafsson.github.io/spacr/api/spacr/spacr_cellpose/index.html>`_
     - `Model comparison <https://einarolafsson.github.io/spacr/api/spacr/spacr_cellpose/index.html>`_
     - Alpha
     - Compares masks, counts, and agreement between models.
   * - `Cellpose <https://einarolafsson.github.io/spacr/api/spacr/spacr_cellpose/index.html>`_
     - `Model zoo <https://einarolafsson.github.io/spacr/api/spacr/spacr_cellpose/index.html>`_
     - Alpha
     - Finds, verifies, downloads, and benchmarks checkpoints.

Tracking and timelapse
~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 24 10 46

   * - Module
     - Feature
     - State
     - Description
   * - `Timelapse <https://einarolafsson.github.io/spacr/api/spacr/timelapse/index.html>`_
     - `Object tracking <https://einarolafsson.github.io/spacr/api/spacr/timelapse/index.html>`_
     - Beta
     - Links objects with IoU, Trackpy, btrack, Trackastra, or ultrack.
   * - `Timelapse <https://einarolafsson.github.io/spacr/api/spacr/timelapse/index.html>`_
     - `Track preview <https://einarolafsson.github.io/spacr/api/spacr/timelapse/index.html>`_
     - Beta
     - Shows tracks on demand before full processing.
   * - `Timelapse <https://einarolafsson.github.io/spacr/api/spacr/timelapse/index.html>`_
     - `Motility assay <https://einarolafsson.github.io/spacr/api/spacr/timelapse/index.html#spacr.timelapse.automated_motility_assay>`_
     - Beta
     - Summarizes displacement, velocity, persistence, and infection.

Measurements
~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 24 10 46

   * - Module
     - Feature
     - State
     - Description
   * - `Measure <https://einarolafsson.github.io/spacr/api/spacr/measure/index.html>`_
     - `2D measurements <https://einarolafsson.github.io/spacr/api/spacr/measure/index.html#spacr.measure.measure_crop>`_
     - Stable
     - Measures morphology, intensity, texture, and colocalization.
   * - `Measure <https://einarolafsson.github.io/spacr/api/spacr/measure/index.html>`_
     - `3D measurements <https://einarolafsson.github.io/spacr/api/spacr/measure/index.html#spacr.measure.measure_crop>`_
     - Beta
     - Measures objects throughout volumetric image stacks.
   * - `Measure <https://einarolafsson.github.io/spacr/api/spacr/measure/index.html>`_
     - `4D measurements <https://einarolafsson.github.io/spacr/api/spacr/measure/index.html#spacr.measure.measure_crop>`_
     - Beta
     - Measures volumetric objects over time.
   * - `Measure <https://einarolafsson.github.io/spacr/api/spacr/measure/index.html>`_
     - `Object crops <https://einarolafsson.github.io/spacr/api/spacr/measure/index.html#spacr.measure.measure_crop>`_
     - Stable
     - Writes database-linked single-object images and arrays.

Annotation
~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 24 10 46

   * - Module
     - Feature
     - State
     - Description
   * - `Annotate <https://einarolafsson.github.io/spacr/api/spacr/qt/screens/annotate/index.html>`_
     - `Manual annotation <https://einarolafsson.github.io/spacr/api/spacr/qt/screens/annotate/index.html>`_
     - Stable
     - Reviews crops and saves labels directly to the database.

AI and machine learning
~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 24 10 46

   * - Module
     - Feature
     - State
     - Description
   * - `Computer vision <https://einarolafsson.github.io/spacr/api/spacr/deep_spacr/index.html>`_
     - `Image classification <https://einarolafsson.github.io/spacr/api/spacr/deep_spacr/index.html>`_
     - Stable
     - Trains and applies PyTorch CNN and transformer models.
   * - `Computer vision <https://einarolafsson.github.io/spacr/api/spacr/deep_spacr/index.html>`_
     - `Live training metrics <https://einarolafsson.github.io/spacr/api/spacr/deep_spacr/index.html>`_
     - Beta
     - Streams loss, accuracy, and training images to TensorBoard.
   * - `Machine learning <https://einarolafsson.github.io/spacr/api/spacr/ml/index.html>`_
     - `Measurement classification <https://einarolafsson.github.io/spacr/api/spacr/ml/index.html>`_
     - Stable
     - Trains interpretable classical and boosted models.
   * - `Image UMAP <https://einarolafsson.github.io/spacr/api/spacr/app_umap/index.html>`_
     - `Interactive embedding <https://einarolafsson.github.io/spacr/api/spacr/app_umap/index.html>`_
     - Beta
     - Inspects points, draws clusters, and writes labels to SQLite.
   * - `Computer vision <https://einarolafsson.github.io/spacr/api/spacr/deep_spacr/index.html>`_
     - `Activation maps <https://einarolafsson.github.io/spacr/api/spacr/deep_spacr/index.html>`_
     - Beta
     - Explains predictions with Captum, SmoothGrad, and TorchCAM.
   * - `Computer vision <https://einarolafsson.github.io/spacr/api/spacr/deep_spacr/index.html>`_
     - `Training-run comparison <https://einarolafsson.github.io/spacr/api/spacr/deep_spacr/index.html>`_
     - Alpha
     - Compares metrics and settings across model runs.
   * - `Hyperparameters <https://einarolafsson.github.io/spacr/api/spacr/hyperparam/index.html>`_
     - `Hyperparameter search <https://einarolafsson.github.io/spacr/api/spacr/hyperparam/index.html>`_
     - Beta
     - Searches supported embedding and model settings.
   * - `Qt AI <https://einarolafsson.github.io/spacr/api/spacr/qt/ai/index.html>`_
     - `AI-assisted console <https://einarolafsson.github.io/spacr/api/spacr/qt/ai/index.html>`_
     - Beta
     - Connects supported modules to local AI command-line tools.

Sequencing and screen analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 24 10 46

   * - Module
     - Feature
     - State
     - Description
   * - `Sequencing <https://einarolafsson.github.io/spacr/api/spacr/sequencing/index.html>`_
     - `Map barcodes <https://einarolafsson.github.io/spacr/api/spacr/sequencing/index.html>`_
     - Stable
     - Maps row, column, and gRNA barcodes from FASTQ reads.
   * - `Statistics <https://einarolafsson.github.io/spacr/api/spacr/sp_stats/index.html>`_
     - `Regression <https://einarolafsson.github.io/spacr/api/spacr/sp_stats/index.html>`_
     - Stable
     - Estimates guide, gene, condition, and control effects.
   * - `Simulation <https://einarolafsson.github.io/spacr/api/spacr/sim/index.html>`_
     - `Screen simulation <https://einarolafsson.github.io/spacr/api/spacr/sim/index.html>`_
     - Stable
     - Explores pooled-screen designs over parameter grids.

Visualization, QC, and reporting
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 24 10 46

   * - Module
     - Feature
     - State
     - Description
   * - `Plotting <https://einarolafsson.github.io/spacr/api/spacr/plot/index.html>`_
     - `Plate viewer <https://einarolafsson.github.io/spacr/api/spacr/plot/index.html>`_
     - Alpha
     - Displays measurement heatmaps and spatial plate effects.
   * - `Analysis <https://einarolafsson.github.io/spacr/api/spacr/submodules/index.html>`_
     - `Annotator agreement <https://einarolafsson.github.io/spacr/api/spacr/submodules/index.html>`_
     - Alpha
     - Quantifies agreement and exposes conflicting annotations.
   * - `Report <https://einarolafsson.github.io/spacr/api/spacr/report/index.html>`_
     - `Analysis report <https://einarolafsson.github.io/spacr/api/spacr/report/index.html>`_
     - Alpha
     - Builds HTML or PDF reports with QC and provenance.

Biological assays
~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 24 10 46

   * - Module
     - Feature
     - State
     - Description
   * - `Assays <https://einarolafsson.github.io/spacr/api/spacr/submodules/index.html>`_
     - `Plaque assay <https://einarolafsson.github.io/spacr/api/spacr/submodules/index.html#spacr.submodules.analyze_plaques>`_
     - Beta
     - Quantifies plaque number, area, intensity, and morphology.
   * - `Assays <https://einarolafsson.github.io/spacr/api/spacr/submodules/index.html>`_
     - `Recruitment <https://einarolafsson.github.io/spacr/api/spacr/submodules/index.html#spacr.submodules.analyze_recruitment>`_
     - Stable
     - Compares marker recruitment between conditions.
   * - `Toxoplasma analysis <https://einarolafsson.github.io/spacr/api/spacr/toxo/index.html>`_
     - `Invasion assay <https://einarolafsson.github.io/spacr/api/spacr/toxo/index.html>`_
     - Alpha
     - Distinguishes attached and invaded parasites.
   * - `Assays <https://einarolafsson.github.io/spacr/api/spacr/submodules/index.html>`_
     - `Replication assay <https://einarolafsson.github.io/spacr/api/spacr/submodules/index.html#spacr.submodules.analyze_endodyogeny>`_
     - Beta
     - Summarizes parasites per vacuole and replication rates.

Automation and specialist tools
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 24 10 46

   * - Module
     - Feature
     - State
     - Description
   * - `Batch <https://einarolafsson.github.io/spacr/api/spacr/batch/index.html>`_
     - `Plate queue <https://einarolafsson.github.io/spacr/api/spacr/batch/index.html>`_
     - Alpha
     - Chains several plates through a shared workflow.
   * - `Batch <https://einarolafsson.github.io/spacr/api/spacr/batch/index.html>`_
     - `Batch runner <https://einarolafsson.github.io/spacr/api/spacr/batch/index.html>`_
     - Alpha
     - Queues modules and reports dependency-aware failures.
   * - `Computer vision <https://einarolafsson.github.io/spacr/api/spacr/deep_spacr/index.html>`_
     - `Training-only pipeline <https://einarolafsson.github.io/spacr/api/spacr/deep_spacr/index.html>`_
     - Stable
     - Trains an existing image dataset without rebuilding crops.
   * - `Cellpose <https://einarolafsson.github.io/spacr/api/spacr/spacr_cellpose/index.html>`_
     - `Model sweep <https://einarolafsson.github.io/spacr/api/spacr/spacr_cellpose/index.html>`_
     - Stable
     - Compares available Cellpose models on the same images.

Tutorials
---------

Tutorials are coming soon.


Workflow
--------

.. image:: https://raw.githubusercontent.com/EinarOlafsson/spacr/main/spacr/resources/icons/flow_chart_v3.png
   :alt: spaCR workflow and output organization
   :align: center

Microscopy images (TIFF, OME-TIFF, LIF, CZI, and ND2) and sequencing
reads (FASTQ) enter complementary image-analysis and barcode-mapping
pipelines. Object tables, crops, annotations, predictions, guide identities,
QC results, and well-level summaries can then be analyzed together.


Quick start
-----------

The recommended installation uses an isolated conda environment with Python
3.12 and the Qt desktop extra:

.. code-block:: bash

   conda create -n spacr python=3.12 -y
   conda activate spacr
   python -m pip install --upgrade pip
   python -m pip install "spacr[qt]"
   spacr

spaCR supports Python **3.9 through 3.14** (except Python 3.14.1, which is
excluded by torchvision). Python 3.12 is recommended for
the broadest combination of optional scientific packages. Linux is
recommended for CUDA workflows; macOS and Windows are also supported.


Installation options
--------------------

Lightweight installers — no conda or existing Python required
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The recommended desktop downloads are the small online installers stored in
``spacr/application``. These links are rewritten automatically whenever a
new version is built:

.. spacr-installer-links-begin

* `Windows 10/11: download SpaCR 1.4.9.8 <https://github.com/EinarOlafsson/spacr/raw/spacr-codex/spacr/application/SpaCR-1.4.9.8-Windows-Online-Setup.exe>`_
* `macOS 11+ (Intel and Apple silicon): download SpaCR 1.4.9.8 <https://github.com/EinarOlafsson/spacr/raw/spacr-codex/spacr/application/SpaCR-1.4.9.8-macOS-Universal-Online.pkg>`_
* `64-bit Linux: download SpaCR 1.4.9.8 <https://github.com/EinarOlafsson/spacr/raw/spacr-codex/spacr/application/SpaCR-1.4.9.8-Linux-x86_64-Online.run>`_

.. spacr-installer-links-end

No conda installation and no existing Python installation are required. The
installer downloads a private Python 3.12 runtime, Qt, PyTorch, spaCR, and the
scientific dependencies during installation. PyTorch automatically selects a
compatible GPU backend when one is detected and otherwise installs its CPU
build. The installer download therefore stays small while the installed
application is complete and isolated from system Python.

On Linux, make the downloaded installer executable before opening it:

.. code-block:: bash

   chmod +x SpaCR-*-Linux-x86_64-Online.run
   ./SpaCR-*-Linux-x86_64-Online.run

The installer validates spaCR, Qt, PyTorch, and dependency consistency before
replacing an older installation, so an interrupted update leaves the previous
working environment in place.

Desktop application from PyPI
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   python -m pip install "spacr[qt]"
   spacr

Headless or server installation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   python -m pip install spacr
   spacr-run --list

Latest development branch
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   git clone https://github.com/EinarOlafsson/spacr.git
   cd spacr
   git switch spacr-nightly
   python -m pip install -e ".[qt]"

Conda environments
~~~~~~~~~~~~~~~~~~

The native conda-forge recipe is ready in
`conda-forge/recipe <https://github.com/EinarOlafsson/spacr/tree/spacr-codex/conda-forge/recipe>`_.
Conda-forge requires a one-time reviewed onboarding pull request before the
package name becomes available. After that review, every PyPI release is
detected, tested, and published by the conda-forge update bot:

.. code-block:: bash

   conda install -c conda-forge spacr

The short one-time maintainer procedure is documented in
`conda-forge/README.md <https://github.com/EinarOlafsson/spacr/blob/spacr-codex/conda-forge/README.md>`_.

Optional capabilities
~~~~~~~~~~~~~~~~~~~~~

Install only the extras needed by your workflow:

.. code-block:: bash

   python -m pip install "spacr[trackastra]"    # transformer tracking
   python -m pip install "spacr[ultrack]"       # global-optimization tracking
   python -m pip install "spacr[attribution]"   # TorchCAM methods
   python -m pip install "spacr[boosting]"      # LightGBM and CatBoost
   python -m pip install "spacr[zernike]"       # Zernike measurements
   python -m pip install "spacr[btrack]"        # btrack timelapse tracking
   python -m pip install "spacr[czi,nd2,lif]"   # vendor file readers

Optional dependency availability varies by Python version. In particular,
ultrack currently limits ``spacr[all]`` on Python 3.13, and TorchCAM's NumPy
constraint limits the ``attribution`` extra there. The core package and Qt
application remain supported. On Python 3.14, btrack is supported through its
optional extra. The high-performance pylibCZIrw CZI converter remains optional
and outside the tested profile. Other timelapse backends and czifile-based CZI
reading remain available.

The legacy Tk interface remains available as ``spacr-legacy`` but is no
longer under active development.


Command-line entry points
-------------------------

.. code-block:: bash

   spacr                                      # Qt application
   spacr-run --list                           # list headless modules
   spacr-run --describe MODULE                # inspect a module contract
   spacr-run MODULE --settings settings.csv   # execute a module
   spacr-run validate --module MODULE \
       --settings settings.csv                # validate before running
   spacr-repro --help                         # reproducibility tools

Set ``SPACR_LOG_LEVEL=DEBUG`` when troubleshooting. Rotating logs are stored
under ``~/.spacr/logs/spacr.log``.


Project data model
------------------

A typical project contains:

- normalized channel stacks and object masks;
- merged image/mask arrays;
- ``measurements/measurements.db`` with object-linked tables;
- per-object PNG crops and dataset splits;
- annotations and model predictions;
- barcode mappings and screen-level summaries;
- settings snapshots, manifests, QC scorecards, and run reports.

This layout lets desktop modules, headless jobs, and external analysis code
work on the same source of truth.


Data
----

- `Full microscopy dataset: BioStudies S-BIAD2135 <https://doi.org/10.6019/S-BIAD2135>`_
- `Testing dataset: Hugging Face toxo_mito <https://huggingface.co/datasets/einarolafsson/toxo_mito>`_
- `Sequencing data: NCBI BioProject PRJNA1261935 <https://www.ncbi.nlm.nih.gov/bioproject/?term=PRJNA1261935>`_
- `Power analysis: spaCRPower <https://github.com/maomlab/spaCRPower>`_


Citing spaCR
------------

If spaCR contributes to your research, cite:

Olafsson EB, *et al.* A pooled image-based CRISPR screen identifies
EAF1 as a *T. gondii* modulator of ESCRT subversion.

`bioRxiv preprint <https://www.biorxiv.org/content/10.64898/2026.07.08.737057v1>`_ ·
`software archive <https://doi.org/10.5281/zenodo.21343317>`_


Contributing and support
------------------------

Bug reports and focused feature requests are welcome through
`GitHub Issues <https://github.com/EinarOlafsson/spacr/issues>`_.
When reporting a failure, include the spaCR version, operating system,
Python version, module settings, and the relevant log excerpt.

spaCR is released under the
`MIT License <https://github.com/EinarOlafsson/spacr/blob/main/LICENSE>`_.
