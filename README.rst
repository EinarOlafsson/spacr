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

**Spatial phenotype analysis of CRISPR screens.**

spaCR is an end-to-end platform for image-based pooled CRISPR screens,
high-content microscopy, and single-cell phenotype discovery. It connects
raw microscopy images and sequencing reads to segmentation, measurements,
annotation, predictive models, screen scores, quality control, and
publication-ready results.

Every object stays traceable. Images, masks, measurements, annotations,
model predictions, barcodes, and experimental identifiers are linked through
a SQLite-backed project rather than scattered across unrelated files.

- **Image analysis.** Segment 2D, 3D, and 4D microscopy; measure
  morphology, intensity, texture, and spatial relationships; then track,
  crop, annotate, and classify individual objects.
- **FASTQ analysis.** Decode row, column, and gRNA barcodes, assign guide
  identities, assess read quality, and connect sequencing results to imaged
  cells.
- **One connected workflow.** Microscope images + FASTQ files →
  genotype–phenotype associations.
- **Desktop and headless operation.** Use the PySide6 application
  interactively or run the same modules on a workstation, server, or cluster.
- **Ten interface languages.** Switch at runtime between English, Swedish,
  German, Spanish, Mandarin Chinese, Portuguese, Hindi, Korean, Icelandic,
  and French from Preferences. Navigation, AI and LIVE controls, spaCR-authored
  console notices, module descriptions, and setting-help chrome follow the
  selected language while scientific output remains canonical English. See
  the `localization guide <https://einarolafsson.github.io/spacr/localization.html>`_.
- **Live visual feedback.** Preview masks, tracks, timelapse
  frames, training metrics, activation maps, and image embeddings where the
  corresponding module supports them.
- **Animated setting guidance.** Purple help dots open 94 short biological
  animations for visual settings, while the teal dots below retain direct API
  links. See the `setting animation gallery
  <https://einarolafsson.github.io/spacr/setting_animations.html>`_.
- **Reproducible execution.** Validate settings, record manifests,
  preserve run journals, rotate logs, resume supported jobs, and export
  reports with settings and package versions.
- **CPU and GPU execution.** Run general analysis on CPU and use
  CUDA automatically for supported segmentation and deep-learning workloads.


Workflow at a glance
~~~~~~~~~~~~~~~~~~~~

|Tutorials|

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


Installation details
--------------------

Lightweight installers — no conda or existing Python required
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

No conda installation and no existing Python installation are required. The
installer downloads a private Python 3.12 runtime, Qt, PyTorch, spaCR, and the
scientific dependencies during installation. The portable CPU build is the
default so installation does not unexpectedly download several gigabytes of
CUDA libraries. Windows offers NVIDIA acceleration as an optional installer
component, Linux accepts ``--torch-backend auto``, and the standard macOS
PyTorch wheel retains Apple MPS acceleration. The installer download therefore
stays small while the installed application is complete and isolated from
system Python.

On Linux, make the downloaded installer executable before opening it:

.. code-block:: bash

   chmod +x SpaCR-*-Linux-x86_64-Online.run
   ./SpaCR-*-Linux-x86_64-Online.run

On macOS, open the downloaded ``.pkg``. If Gatekeeper blocks the current beta
installer because it is not notarized, open **System Settings → Privacy &
Security** and choose **Open Anyway** for spaCR, then run the package again.

The installer validates spaCR, Qt, PyTorch, and dependency consistency before
replacing an older installation, so an interrupted update leaves the previous
working environment in place. A complete diagnostic log is retained as
``install.log`` inside the private spaCR installation directory.

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
   git switch nightly
   python -m pip install -e ".[qt]"

Conda environments
~~~~~~~~~~~~~~~~~~

Conda users can install the released PyPI package inside an isolated
environment:

.. code-block:: bash

   conda create -n spacr python=3.12 pip -y
   conda activate spacr
   python -m pip install "spacr[qt]"

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


Installation options
--------------------

|Release| |PyPI| |CondaRecipe|

**(beta) Lightweight desktop installers:**

.. spacr-installer-links-begin

* `Windows 10/11: download SpaCR 1.4.9.9 <https://github.com/EinarOlafsson/spacr/releases/download/v1.4.9.9/SpaCR-1.4.9.9-Windows-Online-Setup.exe>`_
* `macOS 11+ (Intel and Apple silicon): download SpaCR 1.4.9.9 <https://github.com/EinarOlafsson/spacr/releases/download/v1.4.9.9/SpaCR-1.4.9.9-macOS-Universal-Online.pkg>`_
* `64-bit Linux: download SpaCR 1.4.9.9 <https://github.com/EinarOlafsson/spacr/releases/download/v1.4.9.9/SpaCR-1.4.9.9-Linux-x86_64-Online.run>`_

.. spacr-installer-links-end

Install the Qt application into an existing Python environment with:

.. code-block:: bash

   python -m pip install "spacr[qt]"
   spacr

Detailed desktop, headless, development, and conda instructions appear in
`Installation details`_.


Features
--------

Internationalized desktop interface
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Choose **spaCR → Preferences → Language** to retranslate the open application
without restarting it. The selection persists for later launches, and module
screens opened afterward inherit it automatically.

- **Application chrome:** navigation, Preferences, common actions, AI and LIVE
  controls, provider setup, chat status text, and spaCR-authored run notices.
- **Contextual help:** reviewed descriptions for all built-in modules plus
  localized setting names, type hints, generic explanations, and API-link
  captions. Documentation URLs remain stable across languages.
- **Output safety:** worker stdout, logs, tracebacks, paths, filenames,
  database values, annotations, user messages, AI responses, measurements,
  reports, and saved results are never translated. Unreviewed scientific
  tooltip prose also remains English instead of becoming a misleading
  mixed-language explanation.

The full behavior, environment override, and translation-contribution format
are documented in the
`localization guide <https://einarolafsson.github.io/spacr/localization.html>`_.

Animated setting guidance
~~~~~~~~~~~~~~~~~~~~~~~~~

Visual settings can carry a purple animation dot above their teal API dot.
Clicking it opens a square GIF immediately above the setting; clicking outside
or pressing Escape closes it. The 94 deterministic diagrams map to 143 exact
setting keys and share one biological grammar: rounded fibroblasts, motile
immune cells, nuclei with unequal nucleoli, paired *Toxoplasma* tachyzoites in
a tightly wrapped outline-only vacuole, and curved Golgi cisternae. The cell,
nucleus, parasite-vacuole, and Golgi artwork is rendered from the reviewed SVG
templates at high resolution, and template hashes are recorded in the asset
manifest for reproducibility. The
`animation gallery <https://einarolafsson.github.io/spacr/setting_animations.html>`_
and `setting-animation registry API
<https://einarolafsson.github.io/spacr/api/spacr/setting_animations/index.html>`_
document every mapping.

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
   * - |feature-api-003|_
     - |feature-docs-076|_
     - Stable
     - Retranslates open and lazily created screens across ten bundled languages.
   * - |feature-api-003|_
     - |feature-docs-077|_
     - Stable
     - Localizes module summaries and setting-help chrome while preserving exact API URLs.
   * - |feature-api-050|_
     - |feature-api-051|_
     - Stable
     - Localizes AI/LIVE controls, chat chrome, and spaCR notices without changing user or model content.
   * - |feature-api-078|_
     - |feature-docs-079|_
     - Stable
     - Opens 94 packaged visual-setting animations from purple help dots while preserving teal API links.
   * - **Image analysis**
     -
     -
     -
   * - |feature-api-018|_
     - |feature-api-019|_
     - Stable
     - Segments cells, nuclei, pathogens, and organelles in 2D images.
   * - |feature-api-018|_
     - |feature-api-020|_
     - Beta
     - Segments volumetric images and 4D time series.
   * - |feature-api-032|_
     - |feature-api-033|_
     - Stable
     - Measures morphology, intensity, texture, and colocalization.
   * - |feature-api-032|_
     - |feature-api-034|_
     - Beta
     - Measures objects through 3D volumes and 4D experiments.
   * - |feature-api-028|_
     - |feature-api-029|_
     - Beta
     - Tracks objects with IoU, Trackpy, btrack, Trackastra, or ultrack.
   * - |feature-api-028|_
     - |feature-api-031|_
     - Beta
     - Quantifies motility, displacement, velocity, and persistence.
   * - |feature-api-009|_
     - |feature-api-011|_
     - Alpha
     - Converts ND2, CZI, LIF, TIFF, and OME-TIFF acquisitions.
   * - |feature-api-013|_
     - |feature-api-014|_
     - Stable
     - Links images, masks, crops, measurements, and object identities.
   * - **AI and phenotyping**
     -
     -
     -
   * - |feature-api-037|_
     - |feature-api-038|_
     - Stable
     - Reviews crops and saves annotations directly to SQLite.
   * - |feature-api-039|_
     - |feature-api-040|_
     - Stable
     - Trains and applies PyTorch CNN and transformer models.
   * - |feature-api-039|_
     - |feature-api-041|_
     - Beta
     - Streams loss, accuracy, and training images to TensorBoard.
   * - |feature-api-039|_
     - |feature-api-046|_
     - Beta
     - Explains predictions with Captum, SmoothGrad, and TorchCAM.
   * - |feature-api-044|_
     - |feature-api-045|_
     - Beta
     - Explores images interactively and propagates cluster labels.
   * - |feature-api-042|_
     - |feature-api-043|_
     - Stable
     - Trains interpretable classical and boosted measurement models.
   * - |feature-api-048|_
     - |feature-api-049|_
     - Beta
     - Searches embedding and model hyperparameters.
   * - |feature-api-023|_
     - |feature-api-025|_
     - Beta
     - Trains custom Cellpose segmentation checkpoints.
   * - **Sequencing and screen analysis**
     -
     -
     -
   * - |feature-api-052|_
     - |feature-api-053|_
     - Stable
     - Maps row, column, and gRNA barcodes from FASTQ reads.
   * - |feature-api-052|_
     - |feature-api-074|_
     - Stable
     - Connects guide identities to imaged single-cell phenotypes.
   * - |feature-api-001|_
     - |feature-api-075|_
     - Stable
     - Joins imaging phenotypes and sequencing identities in one workflow.
   * - |feature-api-054|_
     - |feature-api-055|_
     - Stable
     - Estimates guide, gene, condition, and control effects.
   * - |feature-api-064|_
     - |feature-api-065|_
     - Beta
     - Quantifies plaque number, area, intensity, and morphology.
   * - |feature-api-067|_
     - |feature-api-068|_
     - Alpha
     - Distinguishes attached and invaded parasites.
   * - |feature-api-062|_
     - |feature-api-063|_
     - Alpha
     - Builds reports with quality control and run provenance.
   * - |feature-api-005|_
     - |feature-api-071|_
     - Alpha
     - Runs validated modules in dependency-aware plate queues.

.. |feature-api-001| replace:: **Core**
.. _feature-api-001: https://einarolafsson.github.io/spacr/api/spacr/core/index.html

.. |feature-api-002| replace:: **Image pipeline**
.. _feature-api-002: https://einarolafsson.github.io/spacr/api/spacr/core/index.html

.. |feature-api-003| replace:: **Qt application**
.. _feature-api-003: https://einarolafsson.github.io/spacr/api/spacr/qt/app/index.html

.. |feature-api-004| replace:: **Desktop interface**
.. _feature-api-004: https://einarolafsson.github.io/spacr/api/spacr/qt/app/index.html

.. |feature-api-005| replace:: **Batch**
.. _feature-api-005: https://einarolafsson.github.io/spacr/api/spacr/batch/index.html

.. |feature-api-006| replace:: **Headless execution**
.. _feature-api-006: https://einarolafsson.github.io/spacr/api/spacr/batch/index.html

.. |feature-api-007| replace:: **Logging**
.. _feature-api-007: https://einarolafsson.github.io/spacr/api/spacr/logging_util/index.html

.. |feature-api-008| replace:: **Reproducible runs**
.. _feature-api-008: https://einarolafsson.github.io/spacr/api/spacr/logging_util/index.html

.. |feature-api-009| replace:: **I/O**
.. _feature-api-009: https://einarolafsson.github.io/spacr/api/spacr/io/index.html

.. |feature-api-010| replace:: **Image import**
.. _feature-api-010: https://einarolafsson.github.io/spacr/api/spacr/io/index.html

.. |feature-api-011| replace:: **Format converter**
.. _feature-api-011: https://einarolafsson.github.io/spacr/api/spacr/io/index.html

.. |feature-api-012| replace:: **Project import**
.. _feature-api-012: https://einarolafsson.github.io/spacr/api/spacr/io/index.html

.. |feature-api-013| replace:: **Object data**
.. _feature-api-013: https://einarolafsson.github.io/spacr/api/spacr/object/index.html

.. |feature-api-014| replace:: **Object schema**
.. _feature-api-014: https://einarolafsson.github.io/spacr/api/spacr/object/index.html

.. |feature-api-015| replace:: **Database browser**
.. _feature-api-015: https://einarolafsson.github.io/spacr/api/spacr/io/index.html

.. |feature-api-016| replace:: **Stitching**
.. _feature-api-016: https://einarolafsson.github.io/spacr/api/spacr/spacrops/index.html

.. |feature-api-017| replace:: **Align and stitch**
.. _feature-api-017: https://einarolafsson.github.io/spacr/api/spacr/spacrops/index.html

.. |feature-api-018| replace:: **Mask**
.. _feature-api-018: https://einarolafsson.github.io/spacr/api/spacr/core/index.html

.. |feature-api-019| replace:: **2D mask generation**
.. _feature-api-019: https://einarolafsson.github.io/spacr/api/spacr/core/index.html#spacr.core.preprocess_generate_masks

.. |feature-api-020| replace:: **3D and 4D mask generation**
.. _feature-api-020: https://einarolafsson.github.io/spacr/api/spacr/core/index.html#spacr.core.preprocess_generate_masks

.. |feature-api-021| replace:: **4D mask generation**
.. _feature-api-021: https://einarolafsson.github.io/spacr/api/spacr/core/index.html#spacr.core.preprocess_generate_masks

.. |feature-api-022| replace:: **Live preview**
.. _feature-api-022: https://einarolafsson.github.io/spacr/api/spacr/core/index.html

.. |feature-api-023| replace:: **Cellpose**
.. _feature-api-023: https://einarolafsson.github.io/spacr/api/spacr/spacr_cellpose/index.html

.. |feature-api-024| replace:: **Make masks**
.. _feature-api-024: https://einarolafsson.github.io/spacr/api/spacr/spacr_cellpose/index.html

.. |feature-api-025| replace:: **Train Cellpose**
.. _feature-api-025: https://einarolafsson.github.io/spacr/api/spacr/spacr_cellpose/index.html

.. |feature-api-026| replace:: **Model comparison**
.. _feature-api-026: https://einarolafsson.github.io/spacr/api/spacr/spacr_cellpose/index.html

.. |feature-api-027| replace:: **Model zoo**
.. _feature-api-027: https://einarolafsson.github.io/spacr/api/spacr/spacr_cellpose/index.html

.. |feature-api-028| replace:: **Timelapse**
.. _feature-api-028: https://einarolafsson.github.io/spacr/api/spacr/timelapse/index.html

.. |feature-api-029| replace:: **Object tracking**
.. _feature-api-029: https://einarolafsson.github.io/spacr/api/spacr/timelapse/index.html

.. |feature-api-030| replace:: **Track preview**
.. _feature-api-030: https://einarolafsson.github.io/spacr/api/spacr/timelapse/index.html

.. |feature-api-031| replace:: **Motility assay**
.. _feature-api-031: https://einarolafsson.github.io/spacr/api/spacr/timelapse/index.html#spacr.timelapse.automated_motility_assay

.. |feature-api-032| replace:: **Measure**
.. _feature-api-032: https://einarolafsson.github.io/spacr/api/spacr/measure/index.html

.. |feature-api-033| replace:: **2D measurements**
.. _feature-api-033: https://einarolafsson.github.io/spacr/api/spacr/measure/index.html#spacr.measure.measure_crop

.. |feature-api-034| replace:: **3D and 4D measurements**
.. _feature-api-034: https://einarolafsson.github.io/spacr/api/spacr/measure/index.html#spacr.measure.measure_crop

.. |feature-api-035| replace:: **4D measurements**
.. _feature-api-035: https://einarolafsson.github.io/spacr/api/spacr/measure/index.html#spacr.measure.measure_crop

.. |feature-api-036| replace:: **Object crops**
.. _feature-api-036: https://einarolafsson.github.io/spacr/api/spacr/measure/index.html#spacr.measure.measure_crop

.. |feature-api-037| replace:: **Annotate**
.. _feature-api-037: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/annotate/index.html

.. |feature-api-038| replace:: **Manual annotation**
.. _feature-api-038: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/annotate/index.html

.. |feature-api-039| replace:: **Computer vision**
.. _feature-api-039: https://einarolafsson.github.io/spacr/api/spacr/deep_spacr/index.html

.. |feature-api-040| replace:: **Image classification**
.. _feature-api-040: https://einarolafsson.github.io/spacr/api/spacr/deep_spacr/index.html

.. |feature-api-041| replace:: **Live training metrics**
.. _feature-api-041: https://einarolafsson.github.io/spacr/api/spacr/deep_spacr/index.html

.. |feature-api-042| replace:: **Machine learning**
.. _feature-api-042: https://einarolafsson.github.io/spacr/api/spacr/ml/index.html

.. |feature-api-043| replace:: **Measurement classification**
.. _feature-api-043: https://einarolafsson.github.io/spacr/api/spacr/ml/index.html

.. |feature-api-044| replace:: **Image UMAP**
.. _feature-api-044: https://einarolafsson.github.io/spacr/api/spacr/app_umap/index.html

.. |feature-api-045| replace:: **Interactive embedding**
.. _feature-api-045: https://einarolafsson.github.io/spacr/api/spacr/app_umap/index.html

.. |feature-api-046| replace:: **Activation maps**
.. _feature-api-046: https://einarolafsson.github.io/spacr/api/spacr/deep_spacr/index.html

.. |feature-api-047| replace:: **Training-run comparison**
.. _feature-api-047: https://einarolafsson.github.io/spacr/api/spacr/deep_spacr/index.html

.. |feature-api-048| replace:: **Hyperparameters**
.. _feature-api-048: https://einarolafsson.github.io/spacr/api/spacr/hyperparam/index.html

.. |feature-api-049| replace:: **Hyperparameter search**
.. _feature-api-049: https://einarolafsson.github.io/spacr/api/spacr/hyperparam/index.html

.. |feature-api-050| replace:: **Qt AI**
.. _feature-api-050: https://einarolafsson.github.io/spacr/api/spacr/qt/ai/index.html

.. |feature-api-051| replace:: **AI-assisted console**
.. _feature-api-051: https://einarolafsson.github.io/spacr/api/spacr/qt/ai/index.html

.. |feature-api-052| replace:: **Sequencing**
.. _feature-api-052: https://einarolafsson.github.io/spacr/api/spacr/sequencing/index.html

.. |feature-api-053| replace:: **Map barcodes**
.. _feature-api-053: https://einarolafsson.github.io/spacr/api/spacr/sequencing/index.html

.. |feature-api-054| replace:: **Statistics**
.. _feature-api-054: https://einarolafsson.github.io/spacr/api/spacr/sp_stats/index.html

.. |feature-api-055| replace:: **Regression**
.. _feature-api-055: https://einarolafsson.github.io/spacr/api/spacr/sp_stats/index.html

.. |feature-api-056| replace:: **Simulation**
.. _feature-api-056: https://einarolafsson.github.io/spacr/api/spacr/sim/index.html

.. |feature-api-057| replace:: **Screen simulation**
.. _feature-api-057: https://einarolafsson.github.io/spacr/api/spacr/sim/index.html

.. |feature-api-058| replace:: **Plotting**
.. _feature-api-058: https://einarolafsson.github.io/spacr/api/spacr/plot/index.html

.. |feature-api-059| replace:: **Plate viewer**
.. _feature-api-059: https://einarolafsson.github.io/spacr/api/spacr/plot/index.html

.. |feature-api-060| replace:: **Analysis**
.. _feature-api-060: https://einarolafsson.github.io/spacr/api/spacr/submodules/index.html

.. |feature-api-061| replace:: **Annotator agreement**
.. _feature-api-061: https://einarolafsson.github.io/spacr/api/spacr/submodules/index.html

.. |feature-api-062| replace:: **Report**
.. _feature-api-062: https://einarolafsson.github.io/spacr/api/spacr/report/index.html

.. |feature-api-063| replace:: **Analysis report**
.. _feature-api-063: https://einarolafsson.github.io/spacr/api/spacr/report/index.html

.. |feature-api-064| replace:: **Assays**
.. _feature-api-064: https://einarolafsson.github.io/spacr/api/spacr/submodules/index.html

.. |feature-api-065| replace:: **Plaque assay**
.. _feature-api-065: https://einarolafsson.github.io/spacr/api/spacr/submodules/index.html#spacr.submodules.analyze_plaques

.. |feature-api-066| replace:: **Recruitment**
.. _feature-api-066: https://einarolafsson.github.io/spacr/api/spacr/submodules/index.html#spacr.submodules.analyze_recruitment

.. |feature-api-067| replace:: **Toxoplasma analysis**
.. _feature-api-067: https://einarolafsson.github.io/spacr/api/spacr/toxo/index.html

.. |feature-api-068| replace:: **Invasion assay**
.. _feature-api-068: https://einarolafsson.github.io/spacr/api/spacr/toxo/index.html

.. |feature-api-069| replace:: **Replication assay**
.. _feature-api-069: https://einarolafsson.github.io/spacr/api/spacr/submodules/index.html#spacr.submodules.analyze_endodyogeny

.. |feature-api-070| replace:: **Plate queue**
.. _feature-api-070: https://einarolafsson.github.io/spacr/api/spacr/batch/index.html

.. |feature-api-071| replace:: **Batch runner**
.. _feature-api-071: https://einarolafsson.github.io/spacr/api/spacr/batch/index.html

.. |feature-api-072| replace:: **Training-only pipeline**
.. _feature-api-072: https://einarolafsson.github.io/spacr/api/spacr/deep_spacr/index.html

.. |feature-api-073| replace:: **Model sweep**
.. _feature-api-073: https://einarolafsson.github.io/spacr/api/spacr/spacr_cellpose/index.html

.. |feature-api-074| replace:: **gRNA assignment**
.. _feature-api-074: https://einarolafsson.github.io/spacr/api/spacr/sequencing/index.html

.. |feature-api-075| replace:: **Genotype–phenotype linking**
.. _feature-api-075: https://einarolafsson.github.io/spacr/api/spacr/core/index.html

.. |feature-docs-076| replace:: **Ten-language localization**
.. _feature-docs-076: https://einarolafsson.github.io/spacr/localization.html

.. |feature-docs-077| replace:: **Localized contextual help**
.. _feature-docs-077: https://einarolafsson.github.io/spacr/localization.html#contextual-help

.. |feature-api-078| replace:: **Setting animation registry**
.. _feature-api-078: https://einarolafsson.github.io/spacr/api/spacr/setting_animations/index.html

.. |feature-docs-079| replace:: **Visual setting animations**
.. _feature-docs-079: https://einarolafsson.github.io/spacr/setting_animations.html


Data
----

Reference datasets
~~~~~~~~~~~~~~~~~~

- `Full microscopy dataset: BioStudies S-BIAD2135 <https://doi.org/10.6019/S-BIAD2135>`_
- `Testing dataset: Hugging Face toxo_mito <https://huggingface.co/datasets/einarolafsson/toxo_mito>`_
- `Sequencing data: NCBI BioProject PRJNA1261935 <https://www.ncbi.nlm.nih.gov/bioproject/?term=PRJNA1261935>`_
- `Power analysis: spaCRPower <https://github.com/maomlab/spaCRPower>`_


Contributing and support
------------------------

Bug reports and focused feature requests are welcome through
`GitHub Issues <https://github.com/EinarOlafsson/spacr/issues>`_.
When reporting a failure, include the spaCR version, operating system,
Python version, module settings, and the relevant log excerpt.

Licensing
~~~~~~~~~

The current development branch is source-available under the
`PolyForm Noncommercial License 1.0.0
<https://github.com/EinarOlafsson/spacr/blob/main/LICENSE>`_. Commercial use
requires a separate license from the copyright holder. Released versions
through spaCR 1.4.9.9 remain available under the MIT License that accompanied
those releases.

Tutorials
~~~~~~~~~

Use the `interactive spaCR tutorial library
<https://einarolafsson.github.io/spacr/tutorials/>`_ for narrated,
captioned walkthroughs of installation and application workflows. Its
language-first player is configured for all 54 Kokoro voices across English,
Spanish, French, Hindi, Italian, Brazilian Portuguese, Japanese, and Mandarin
Chinese.

Citing spaCR
~~~~~~~~~~~~

If spaCR contributes to your research, cite:

Olafsson EB, *et al.* A pooled image-based CRISPR screen identifies
EAF1 as a *T. gondii* modulator of ESCRT subversion.

`bioRxiv preprint <https://www.biorxiv.org/content/10.64898/2026.07.08.737057v1>`_ ·
`software archive <https://doi.org/10.5281/zenodo.21343317>`_
