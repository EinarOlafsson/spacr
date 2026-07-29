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


Feature maturity
----------------

The maturity label written beside every feature below is the same status used
by the spaCR home screen and settings panels.

- **Stable** — established and suitable for routine work.
- **Beta** — functional and in regular use, but validate it on representative
  data before production analysis.
- **Alpha** — available for evaluation; expect workflow and interface changes.

The established 2D pipeline is the most mature. New 3D and 4D settings remain
**Beta**, even when they extend a module whose conventional 2D workflow is
**Stable**.


Main features
-------------

These modules form the primary image-to-screen workflow.

**Mask — Stable.**
  Ingest microscopy images, organize channels, generate Cellpose masks for
  cells, nuclei, pathogens, and organelles, inspect a live preview, and write
  merged arrays plus segmentation-QC scorecards. 3D and 4D mask paths are
  **Beta**.

**Timelapse — Beta.**
  Segment time series and link objects across frames with IoU, Trackpy,
  btrack, Trackastra, or optional ultrack backends, with an on-demand track
  preview.

**Motility Assay — Beta.**
  Calculate per-track displacement, velocity, persistence, and infection QC,
  then summarize motility by well and condition.

**Measure — Stable.**
  Capture object morphology, intensity, texture, colocalization, radial,
  neighborhood, and optional Zernike features; generate linked single-object
  PNG crops and write measurements to SQLite. New 3D/4D measurement paths are
  **Beta**.

**Annotate — Stable.**
  Review single-object images in a responsive grid, assign or revise labels,
  filter records, use AI-assisted console tools, and save annotations directly
  to the measurements database.

**Classify (CV) — Stable.**
  Build image datasets, train and evaluate PyTorch CNN or transformer models,
  watch live loss and accuracy in TensorBoard, apply models to new objects,
  and merge predictions back into the database.

**Classify (ML) — Stable.**
  Train classical models—including logistic regression, random forests,
  XGBoost, and optional LightGBM/CatBoost—on measurement features with feature
  selection, cross-validation, interpretation, and screen-level scoring.

**Map Barcodes — Stable.**
  Decode row, column, and gRNA barcodes from FASTQ reads, use bundled barcode
  references or custom libraries, inspect regex parsing, run sequencing QC,
  and join guide identities to imaging data.

**Regression — Stable.**
  Model screen scores, compare conditions and controls, aggregate guide-level
  effects, calculate gene-level summaries, and produce statistical figures.


Secondary features
------------------

Segmentation models
~~~~~~~~~~~~~~~~~~~

**Make Masks — Beta.**
  Create and edit training masks, then fine-tune Cellpose models against
  dataset-specific annotations.

**Train Cellpose — Beta.**
  Train custom Cellpose checkpoints with configurable optimization,
  augmentation, validation, and model output.

**Cellpose Masks — Beta.**
  Run focused Cellpose mask generation outside the complete preprocessing
  pipeline.

**Model Compare — Alpha.**
  Apply two Cellpose models to the same fields and compare masks, object
  counts, and adjusted Rand index differences side by side.

**Model Zoo — Alpha.**
  Browse available Cellpose and classifier checkpoints, verify checksums,
  download models, and benchmark them on representative fields.

Results, interpretation, and quality control
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Plate Viewer — Alpha.**
  Render any measurement as a plate heatmap and inspect spatial bias and edge
  effects.

**Annotator Agreement — Alpha.**
  Calculate Cohen's or Fleiss' kappa between annotation columns and review
  disagreements.

**Image UMAP — Beta.**
  Create UMAP or t-SNE embeddings with image glyphs, click points to inspect
  cells, draw around clusters, assign manual labels, or propagate
  automatically generated cluster labels to the database.

**Activation — Beta.**
  Generate model-attribution maps with Captum, SmoothGrad, and optional
  TorchCAM methods to inspect which image regions drive predictions.

**Training Runs — Alpha.**
  Overlay loss and accuracy curves from multiple training runs and compare
  the settings that changed between them.

**Report — Alpha.**
  Build a self-contained HTML or PDF report containing QC verdicts, figures,
  statistics, settings, environment information, and package versions.

Specialized biological assays
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Plaque Assay — Beta.**
  Detect plaques and quantify their number, area, intensity, and morphology
  across conditions.

**Recruitment — Stable.**
  Quantify recruitment of a marker to cells, pathogens, or another measured
  compartment and compare recruitment across conditions.

**Invasion Assay — Alpha.**
  Analyze two-color outside/inside staining, distinguish attached from invaded
  parasites, and calculate invasion efficiency per well.

**Replication Assay — Beta.**
  Count parasites per vacuole and summarize endodyogeny and replication rates
  by condition.


Other tools
-----------

Data preparation and automation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Align & Stitch — Alpha.**
  Register arbitrary tile layouts, solve offsets globally, flag fallback
  placements, and write large stitched canvases incrementally.

**Format Converter — Alpha.**
  Convert ND2, CZI, LIF, and OME-TIFF acquisitions into spaCR-compatible
  Yokogawa-style TIFFs after previewing the filename mapping; preserve a map
  back to source metadata.

**Import Project — Alpha.**
  Convert third-party images, masks, and CSV/TSV/SQLite measurements into a
  validated spaCR project with an explicit, reviewable column map.

**Plate Queue — Alpha.**
  Chain several plates through the same processing sequence.

**Batch Runner — Alpha.**
  Queue different modules, plates, and settings for unattended execution with
  validation and dependency-aware failure reporting.

**Database Browser — Alpha.**
  Browse, filter, inspect, and export ``measurements.db`` without using the
  SQLite command-line interface.

Headless and specialist commands
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Training-only pipeline — Stable.**
  Run ``spacr-run train_only`` to train and evaluate an existing image dataset
  without rebuilding crops.

**Cellpose model sweep — Stable.**
  Run ``spacr-run cellpose_all`` to compare every available Cellpose model on
  the same images.

**Screen simulation — Stable.**
  Run ``spacr-run simulation`` to explore pooled-screen designs over parameter
  grids and estimate expected performance.

**Reproducibility tools — Stable.**
  Use settings validation, run journals, resumable stages, structured logs,
  manifest checks, and ``spacr-repro`` to inspect or replay recorded runs.

**AI-assisted console — Beta.**
  Open the collapsible console in supported modules and use locally installed
  Claude, Codex, or Gemini command-line providers without storing API keys in
  spaCR.

**Hyperparameter search — Beta.**
  Search supported UMAP, computer-vision, and machine-learning settings and
  compare trial outcomes from the application.


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

spaCR supports Python **3.9 through 3.13**. Python 3.12 is recommended for
the broadest combination of optional scientific packages. Linux is
recommended for CUDA workflows; macOS and Windows are also supported.


Installation options
--------------------

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

spaCR is not yet published on conda-forge. Conda can manage the Python
environment while pip installs spaCR, as shown above. A native conda-forge
recipe must be accepted before this command becomes available:

.. code-block:: bash

   conda install -c conda-forge spacr

Optional capabilities
~~~~~~~~~~~~~~~~~~~~~

Install only the extras needed by your workflow:

.. code-block:: bash

   python -m pip install "spacr[trackastra]"    # transformer tracking
   python -m pip install "spacr[ultrack]"       # global-optimization tracking
   python -m pip install "spacr[attribution]"   # TorchCAM methods
   python -m pip install "spacr[boosting]"      # LightGBM and CatBoost
   python -m pip install "spacr[zernike]"       # Zernike measurements
   python -m pip install "spacr[czi,nd2,lif]"   # vendor file readers

Optional dependency availability varies by Python version. In particular,
ultrack currently limits ``spacr[all]`` on Python 3.13, and TorchCAM's NumPy
constraint limits the ``attribution`` extra there. The core package and Qt
application remain supported.

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
