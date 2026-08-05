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

spaCR segments and measures single cells in high-content microscopy images,
links each cell to the gRNA it received, and reports which genes changed the
phenotype. Plate images and FASTQ reads go in; per-object measurements,
trained classifiers, per-guide and per-gene effect sizes, and a ranked hit
list come out.

If you run image-based pooled CRISPR screens, that is the whole path. If you
have high-content microscopy and no screen, the segmentation, measurement,
annotation and classification half runs on its own.

Images, masks, crops, measurements, annotations, predictions, barcodes and
well identifiers live in one SQLite project, so a number in a result can be
traced back to the object it came from.

Run spaCR as a desktop application or headlessly on a workstation, server or
cluster. Both drive the same modules, and CUDA is used automatically where a
module supports it.


Workflow at a glance
--------------------

|Tutorials|

.. image:: https://raw.githubusercontent.com/EinarOlafsson/spacr/main/spacr/resources/icons/flow_chart_v3.png
   :alt: spaCR workflow and output organization
   :align: center

Microscopy images (TIFF, OME-TIFF, LIF, CZI, ND2) and sequencing reads
(FASTQ) enter complementary image-analysis and barcode-mapping pipelines.
Object tables, crops, annotations, predictions, guide identities, QC results
and well-level summaries are then analyzed together.


Quick start
-----------

.. code-block:: bash

   conda create -n spacr python=3.12 -y
   conda activate spacr
   python -m pip install --upgrade pip
   python -m pip install "spacr[qt]"
   spacr

spaCR supports Python **3.9 through 3.14** (except Python 3.14.1, which
torchvision excludes). Python 3.12 has the widest choice of optional
scientific packages. Linux is recommended for CUDA workflows; macOS and
Windows are also supported.


Installation details
--------------------

|Release| |PyPI| |CondaRecipe|

**(beta) Lightweight desktop installers:**

.. spacr-installer-links-begin

* `Windows 10/11: download SpaCR 1.5.0.0 <https://github.com/EinarOlafsson/spacr/releases/download/v1.5.0.0/SpaCR-1.5.0.0-Windows-Online-Setup.exe>`_
* `macOS 11+ (Intel and Apple silicon): download SpaCR 1.5.0.0 <https://github.com/EinarOlafsson/spacr/releases/download/v1.5.0.0/SpaCR-1.5.0.0-macOS-Universal-Online.pkg>`_
* `64-bit Linux: download SpaCR 1.5.0.0 <https://github.com/EinarOlafsson/spacr/releases/download/v1.5.0.0/SpaCR-1.5.0.0-Linux-x86_64-Online.run>`_

.. spacr-installer-links-end

Lightweight installers — no conda or existing Python required
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The installer downloads a private Python 3.12 runtime, Qt, PyTorch, spaCR and
the scientific dependencies during installation, so neither conda nor an
existing Python is needed. The portable CPU build is the default, which keeps
the installation from pulling several gigabytes of CUDA libraries
unannounced. Windows offers NVIDIA acceleration as an optional installer
component, Linux accepts ``--torch-backend auto``, and the standard macOS
PyTorch wheel keeps Apple MPS acceleration.

On Linux, make the downloaded installer executable before opening it:

.. code-block:: bash

   chmod +x SpaCR-*-Linux-x86_64-Online.run
   ./SpaCR-*-Linux-x86_64-Online.run

On macOS, open the downloaded ``.pkg``. If Gatekeeper blocks the current beta
installer because it is not notarized, open **System Settings → Privacy &
Security**, choose **Open Anyway** for spaCR, then run the package again.

The installer validates spaCR, Qt, PyTorch and dependency consistency before
replacing an older installation, so an interrupted update leaves the previous
working environment in place. A diagnostic log is kept as ``install.log``
inside the private spaCR installation directory.

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

.. code-block:: bash

   conda create -n spacr python=3.12 pip -y
   conda activate spacr
   python -m pip install "spacr[qt]"

Optional capabilities
~~~~~~~~~~~~~~~~~~~~~

Install only the extras your workflow needs:

.. code-block:: bash

   python -m pip install "spacr[trackastra]"    # transformer tracking
   python -m pip install "spacr[ultrack]"       # global-optimization tracking
   python -m pip install "spacr[btrack]"        # btrack timelapse tracking
   python -m pip install "spacr[attribution]"   # TorchCAM methods
   python -m pip install "spacr[boosting]"      # LightGBM and CatBoost
   python -m pip install "spacr[zernike]"       # Zernike measurements
   python -m pip install "spacr[napari]"        # napari mask correction
   python -m pip install "spacr[czi,nd2,lif]"   # vendor file readers

Which extras resolve depends on the Python version. On Python 3.13, ultrack
limits ``spacr[all]`` and TorchCAM's NumPy constraint limits the
``attribution`` extra; the core package and the Qt application are unaffected.
On Python 3.14, btrack is available through its extra. The pylibCZIrw CZI
converter is optional and untested; czifile-based CZI reading remains
available.

The legacy Tk interface is still installed as ``spacr-legacy`` but is no
longer developed.


Command-line entry points
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

Set ``SPACR_LOG_LEVEL=DEBUG`` when troubleshooting. Rotating logs are written
to ``~/.spacr/logs/spacr.log``.


Features
--------

The six modules most screens use
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Mask** segments cells, nuclei, pathogens and organelles with Cellpose, in 2D
images and in volumetric or time-series data. The model list is read from the
installed Cellpose rather than hard-coded, and an object diameter is estimated
from the images before the run starts. Masks can be corrected by hand in the
layer viewer, or sent to napari and back.

**Measure** writes per-object morphology, intensity, texture and colocalization
features to the project database, together with the crops. New in 1.5.0.0:
illumination correction estimates the flat-field from the plate itself and
divides it out before any intensity feature is taken, which removes the
well-position bias that plate heatmaps show as edge effects. A segmentation QC
banner states in plain language what the masks look like before Measure runs;
it informs, it does not block. A drawn polygon restricts measurement to a
region of interest.

**Annotate** shows crops on a keyboard-driven grid and writes labels straight
to SQLite. It now closes the active-learning loop: retrain a model on what you
have labelled without leaving the screen, re-rank the queue by uncertainty,
watch the learning curve, and get a stopping verdict when further labels stop
changing the model. Coverage is reported per class, per well and per plate,
and every round is recorded.

**Classify** trains PyTorch CNNs and transformers on annotated crops, and
classical or boosted models on measurement tables. Per-class accuracy is now
kept every epoch instead of being discarded, and each checkpoint gets a model
card recording its dataset, class balance, split rule and held-out metrics.
In the evaluation screen, a confusion-matrix cell is a query: click it to open
those crops, with confidently wrong predictions listed apart from uncertain
ones.

**Map Barcodes** decodes row, column and gRNA barcodes from FASTQ reads,
assigns guide identities to wells, and joins them to imaged cells. Barcode QC
reports reads per well, collision rate and unmapped fraction, sweeping around
the number of gRNAs per well you say you expect rather than a fixed threshold.

**Regression** estimates guide, gene, condition and control effects using 17
model families, including mixed models, logistic and probit, quantile, beta,
GLMs with quasi-binomial variance, lasso, ridge, elastic net, hinge and
horseshoe. The result is a ranked, annotated hit list rather than a
coefficient dump.

New in 1.5.0.0
~~~~~~~~~~~~~~

Before a screen exists, the Power / Design module answers how many cells and
how many wells it needs, priced with sequencing error and with the dropout
that comes from wells that were imaged too thinly. An experiment designer lays
out the plate, its controls and its replicates and exports the layout for the
pipeline. Afterwards, a QC dashboard collects the segmentation, plate,
annotator-agreement and leakage checks into one verdict, and ComBat is
available beside ``center`` and ``zscore`` for batch correction.

Results are explored rather than exported and re-imported. A Graph Builder
plots a table by dragging columns onto x, y, colour, size and facet. Gates
drawn on a histogram or a scatter become filters. A feature explorer ranks
features by how well they separate the classes. Small multiples, dose-response
fits, control charts and robust outlier detection use the same axis engine.
Selecting objects in one view selects them in all of them, and opening a
selection brings up the crops those objects came from. A layer viewer stacks
images, labels, points and shapes, with orthogonal views, a synchronised
comparison grid, and a lineage tree from cell to nucleus to pathogen.

Runs are now identifiable. Each carries one run id, one seed and an
``on_error`` policy; Mask, Measure, Classify and the AnnData export register
what they wrote in an artifact registry, so an output file leads back to the
settings that produced it. A module opens on what the previous step actually
wrote, the pipeline graph marks which outputs are stale, run comparison diffs
the settings, object counts and hit lists of two runs, and every GUI run emits
the equivalent Python script. Measurements export to ``.h5ad`` for scanpy;
OME-Zarr and OMERO are available through the Python API. The methods-and-results
exporter drafts those two manuscript sections from a structured digest of the
run: the model writes the prose, but every number comes from the digest, and a
draft containing a number the digest does not contain is rejected. When
something is wrong with the installation, ``spacr-doctor`` reports which spaCR
is actually running, whether the GPU is usable, whether Cellpose matches the
API spaCR calls, and whether the project database and settings are sound, with
a copyable fix on every line that is not a pass.

Internationalized desktop interface
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**spaCR → Preferences → Language** retranslates the running application into
English, Swedish, German, Spanish, Mandarin Chinese, Portuguese, Hindi,
Korean, Icelandic or French without a restart. The choice persists, and
screens opened later inherit it.

Navigation, Preferences, AI and LIVE controls, module descriptions and
spaCR-authored console notices follow the selected language. Worker output,
logs, tracebacks, paths, database values, annotations, AI responses,
measurements and saved results are never translated, so
scientific output remains canonical English. Setting tooltips not yet
reviewed in a language stay in English rather than becoming a
mixed-language explanation.
The `localization guide
<https://einarolafsson.github.io/spacr/localization.html>`_ documents the
behavior, the environment override, and the
`contextual help <https://einarolafsson.github.io/spacr/localization.html#contextual-help>`_
that is translated with it.

Animated setting guidance
~~~~~~~~~~~~~~~~~~~~~~~~~

94 short animations explain what 143 visual settings do to an image. Hover a
setting and click **Animation** in its tooltip to play the square beside the
text; click it again to fold it away. Animations are off until asked for, and
can be disabled in Preferences. The `gallery
<https://einarolafsson.github.io/spacr/setting_animations.html>`_ shows all of
them, and the `Setting animation registry
<https://einarolafsson.github.io/spacr/api/spacr/setting_animations/index.html>`_
records which setting each one belongs to.

Module reference
~~~~~~~~~~~~~~~~

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
   * - |api-qt-app|_
     - |doc-i18n|_
     - Stable
     - Retranslates open and lazily created screens across ten bundled languages.
   * - |api-qt-app|_
     - |doc-i18n-help|_
     - Stable
     - Localizes module summaries and setting-help chrome while preserving exact API URLs.
   * - |api-qt-ai|_
     - |api-qt-ai-console|_
     - Stable
     - Localizes AI and LIVE controls without changing user or model content.
   * - |api-animations|_
     - |doc-animations|_
     - Stable
     - Plays 94 packaged animations for 143 visual settings from the setting tooltip.
   * - |api-selection|_
     - |api-linked-views|_
     - Alpha
     - Shares one object selection across the table, plate, embedding, scatter and graph views.
   * - |api-doctor|_
     - |api-doctor-checks|_
     - Alpha
     - Diagnoses the install — GPU, Cellpose API, database, settings — with a fix per failing check.
   * - **Image analysis**
     -
     -
     -
   * - |api-mask|_
     - |api-mask-2d|_
     - Stable
     - Segments cells, nuclei, pathogens and organelles in 2D images.
   * - |api-mask|_
     - |api-mask-3d|_
     - Beta
     - Segments volumetric images and 4D time series.
   * - |api-illumination|_
     - |api-flatfield|_
     - Alpha
     - Estimates the flat-field from the plate and divides it out before intensity is measured.
   * - |api-measure|_
     - |api-measure-2d|_
     - Stable
     - Measures morphology, intensity, texture and colocalization, and writes the crops.
   * - |api-segqc|_
     - |api-segqc-verdict|_
     - Alpha
     - States what the segmentation looks like before Measure runs, without blocking it.
   * - |api-timelapse|_
     - |api-tracking|_
     - Beta
     - Tracks objects with IoU, Trackpy, btrack, Trackastra or ultrack, and quantifies motility.
   * - |api-layers|_
     - |api-layer-viewer|_
     - Alpha
     - Stacks image, label, point and shape layers, with orthogonal views and a comparison grid.
   * - |api-napari|_
     - |api-napari-curation|_
     - Alpha
     - Hands a mask to napari for correction and takes it back, recording every edit.
   * - **AI and phenotyping**
     -
     -
     -
   * - |api-annotate|_
     - |api-annotation|_
     - Stable
     - Reviews crops on a keyboard-driven grid and saves annotations to SQLite.
   * - |api-active-learning|_
     - |api-al-loop|_
     - Alpha
     - Retrains inside Annotate, re-ranks by uncertainty, and says when labelling can stop.
   * - |api-classify|_
     - |api-classification|_
     - Stable
     - Trains and applies PyTorch CNN and transformer models.
   * - |api-classify|_
     - |api-model-cards|_
     - Alpha
     - Records dataset, class balance, split rule and held-out metrics beside each checkpoint.
   * - |api-confusion|_
     - |api-confusion-drill|_
     - Alpha
     - Opens the crops behind a confusion cell, confident errors listed apart from uncertain ones.
   * - |api-ml|_
     - |api-ml-models|_
     - Stable
     - Trains interpretable classical and boosted models on measurement tables.
   * - |api-classify|_
     - |api-activation|_
     - Beta
     - Explains predictions with Captum, SmoothGrad and TorchCAM.
   * - |api-umap|_
     - |api-embedding|_
     - Beta
     - Explores image embeddings interactively and propagates cluster labels.
   * - **Sequencing and screen analysis**
     -
     -
     -
   * - |api-sequencing|_
     - |api-barcodes|_
     - Stable
     - Maps row, column and gRNA barcodes from FASTQ reads and assigns guides to imaged cells.
   * - |api-barcode-qc|_
     - |api-barcode-qc-sweep|_
     - Alpha
     - Reports reads per well, collision rate and unmapped fraction against the expected gRNAs per well.
   * - |api-regression|_
     - |api-regression-models|_
     - Stable
     - Estimates guide, gene, condition and control effects with 17 model families.
   * - |api-power|_
     - |api-power-design|_
     - Alpha
     - Answers how many cells and wells a screen needs, with sequencing error and well dropout priced in.
   * - |api-graph|_
     - |api-graph-builder|_
     - Alpha
     - Builds a plot by dragging columns onto x, y, colour, size and facet.
   * - |api-artifacts|_
     - |api-provenance|_
     - Alpha
     - Records the run id, seed and settings behind mask, measure, classify and export outputs.

.. |api-qt-app| replace:: **Qt application**
.. _api-qt-app: https://einarolafsson.github.io/spacr/api/spacr/qt/app/index.html

.. |doc-i18n| replace:: **Ten-language localization**
.. _doc-i18n: https://einarolafsson.github.io/spacr/localization.html

.. |doc-i18n-help| replace:: **Localized contextual help**
.. _doc-i18n-help: https://einarolafsson.github.io/spacr/localization.html#contextual-help

.. |api-qt-ai| replace:: **Qt AI**
.. _api-qt-ai: https://einarolafsson.github.io/spacr/api/spacr/qt/ai/index.html

.. |api-qt-ai-console| replace:: **AI-assisted console**
.. _api-qt-ai-console: https://einarolafsson.github.io/spacr/api/spacr/qt/ai/index.html

.. |api-animations| replace:: **Setting animation registry**
.. _api-animations: https://einarolafsson.github.io/spacr/api/spacr/setting_animations/index.html

.. |doc-animations| replace:: **Visual setting animations**
.. _doc-animations: https://einarolafsson.github.io/spacr/setting_animations.html

.. |api-selection| replace:: **Selection**
.. _api-selection: https://einarolafsson.github.io/spacr/api/spacr/selection/index.html

.. |api-linked-views| replace:: **Linked selection**
.. _api-linked-views: https://einarolafsson.github.io/spacr/api/spacr/qt/linked_selection/index.html

.. |api-doctor| replace:: **Doctor**
.. _api-doctor: https://einarolafsson.github.io/spacr/api/spacr/doctor/index.html

.. |api-doctor-checks| replace:: **Installation diagnosis**
.. _api-doctor-checks: https://einarolafsson.github.io/spacr/api/spacr/doctor/index.html

.. |api-mask| replace:: **Mask**
.. _api-mask: https://einarolafsson.github.io/spacr/api/spacr/core/index.html

.. |api-mask-2d| replace:: **2D mask generation**
.. _api-mask-2d: https://einarolafsson.github.io/spacr/api/spacr/core/index.html#spacr.core.preprocess_generate_masks

.. |api-mask-3d| replace:: **3D and 4D mask generation**
.. _api-mask-3d: https://einarolafsson.github.io/spacr/api/spacr/core/index.html#spacr.core.preprocess_generate_masks

.. |api-illumination| replace:: **Illumination**
.. _api-illumination: https://einarolafsson.github.io/spacr/api/spacr/illumination/index.html

.. |api-flatfield| replace:: **Flat-field correction**
.. _api-flatfield: https://einarolafsson.github.io/spacr/api/spacr/illumination/index.html

.. |api-measure| replace:: **Measure**
.. _api-measure: https://einarolafsson.github.io/spacr/api/spacr/measure/index.html

.. |api-measure-2d| replace:: **Object measurements**
.. _api-measure-2d: https://einarolafsson.github.io/spacr/api/spacr/measure/index.html#spacr.measure.measure_crop

.. |api-segqc| replace:: **Segmentation QC**
.. _api-segqc: https://einarolafsson.github.io/spacr/api/spacr/seg_qc/index.html

.. |api-segqc-verdict| replace:: **Pre-run verdict**
.. _api-segqc-verdict: https://einarolafsson.github.io/spacr/api/spacr/seg_qc/index.html

.. |api-timelapse| replace:: **Timelapse**
.. _api-timelapse: https://einarolafsson.github.io/spacr/api/spacr/timelapse/index.html

.. |api-tracking| replace:: **Object tracking**
.. _api-tracking: https://einarolafsson.github.io/spacr/api/spacr/timelapse/index.html

.. |api-layers| replace:: **Layers**
.. _api-layers: https://einarolafsson.github.io/spacr/api/spacr/layers/index.html

.. |api-layer-viewer| replace:: **Layer viewer**
.. _api-layer-viewer: https://einarolafsson.github.io/spacr/api/spacr/qt/layer_viewer/index.html

.. |api-napari| replace:: **napari bridge**
.. _api-napari: https://einarolafsson.github.io/spacr/api/spacr/napari_bridge/index.html

.. |api-napari-curation| replace:: **Mask curation**
.. _api-napari-curation: https://einarolafsson.github.io/spacr/api/spacr/napari_bridge/index.html

.. |api-annotate| replace:: **Annotate**
.. _api-annotate: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/annotate/index.html

.. |api-annotation| replace:: **Manual annotation**
.. _api-annotation: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/annotate/index.html

.. |api-active-learning| replace:: **Active learning**
.. _api-active-learning: https://einarolafsson.github.io/spacr/api/spacr/active_learning/index.html

.. |api-al-loop| replace:: **Retrain and re-rank**
.. _api-al-loop: https://einarolafsson.github.io/spacr/api/spacr/active_learning/index.html

.. |api-classify| replace:: **Classify**
.. _api-classify: https://einarolafsson.github.io/spacr/api/spacr/deep_spacr/index.html

.. |api-classification| replace:: **Image classification**
.. _api-classification: https://einarolafsson.github.io/spacr/api/spacr/deep_spacr/index.html

.. |api-model-cards| replace:: **Model cards**
.. _api-model-cards: https://einarolafsson.github.io/spacr/api/spacr/deep_spacr/index.html

.. |api-activation| replace:: **Activation maps**
.. _api-activation: https://einarolafsson.github.io/spacr/api/spacr/deep_spacr/index.html

.. |api-confusion| replace:: **Confusion**
.. _api-confusion: https://einarolafsson.github.io/spacr/api/spacr/confusion/index.html

.. |api-confusion-drill| replace:: **Confusion drill-down**
.. _api-confusion-drill: https://einarolafsson.github.io/spacr/api/spacr/confusion/index.html

.. |api-ml| replace:: **Machine learning**
.. _api-ml: https://einarolafsson.github.io/spacr/api/spacr/ml/index.html

.. |api-ml-models| replace:: **Measurement classification**
.. _api-ml-models: https://einarolafsson.github.io/spacr/api/spacr/ml/index.html

.. |api-umap| replace:: **Image UMAP**
.. _api-umap: https://einarolafsson.github.io/spacr/api/spacr/app_umap/index.html

.. |api-embedding| replace:: **Interactive embedding**
.. _api-embedding: https://einarolafsson.github.io/spacr/api/spacr/app_umap/index.html

.. |api-sequencing| replace:: **Sequencing**
.. _api-sequencing: https://einarolafsson.github.io/spacr/api/spacr/sequencing/index.html

.. |api-barcodes| replace:: **Map barcodes**
.. _api-barcodes: https://einarolafsson.github.io/spacr/api/spacr/sequencing/index.html

.. |api-barcode-qc| replace:: **Barcode QC**
.. _api-barcode-qc: https://einarolafsson.github.io/spacr/api/spacr/sequencing_qc/index.html

.. |api-barcode-qc-sweep| replace:: **Well and collision report**
.. _api-barcode-qc-sweep: https://einarolafsson.github.io/spacr/api/spacr/sequencing_qc/index.html

.. |api-regression| replace:: **Regression**
.. _api-regression: https://einarolafsson.github.io/spacr/api/spacr/ml/index.html

.. |api-regression-models| replace:: **Screen effect estimation**
.. _api-regression-models: https://einarolafsson.github.io/spacr/api/spacr/ml/index.html

.. |api-power| replace:: **Power**
.. _api-power: https://einarolafsson.github.io/spacr/api/spacr/power_model/index.html

.. |api-power-design| replace:: **Power and design**
.. _api-power-design: https://einarolafsson.github.io/spacr/api/spacr/power_simulate/index.html

.. |api-graph| replace:: **Graph**
.. _api-graph: https://einarolafsson.github.io/spacr/api/spacr/qt/widgets/graph_spec/index.html

.. |api-graph-builder| replace:: **Graph Builder**
.. _api-graph-builder: https://einarolafsson.github.io/spacr/api/spacr/qt/widgets/graph_builder/index.html

.. |api-artifacts| replace:: **Artifacts**
.. _api-artifacts: https://einarolafsson.github.io/spacr/api/spacr/artifacts/index.html

.. |api-provenance| replace:: **Run provenance**
.. _api-provenance: https://einarolafsson.github.io/spacr/api/spacr/runctx/index.html


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
When reporting a failure, include the spaCR version, operating system, Python
version, module settings and the relevant log excerpt. ``spacr-doctor``
collects most of that for you.

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

The `interactive spaCR tutorial library
<https://einarolafsson.github.io/spacr/tutorials/>`_ contains narrated,
captioned walkthroughs of installation and of each application workflow, in
eight languages.

Citing spaCR
~~~~~~~~~~~~

If spaCR contributes to your research, cite:

Olafsson EB, *et al.* A pooled image-based CRISPR screen identifies
EAF1 as a *T. gondii* modulator of ESCRT subversion.

`bioRxiv preprint <https://www.biorxiv.org/content/10.64898/2026.07.08.737057v1>`_ ·
`software archive <https://doi.org/10.5281/zenodo.21343317>`_
