Welcome to spaCR
================

.. image:: _static/logo_spacr.png
   :align: center
   :alt: spaCR Logo
   :width: 200px

**spaCR** — *Spatial phenotype analysis of CRISPR screens.*

A Python toolkit for quantifying and visualising phenotypic changes in
high-throughput microscopy screens. Ships with a modern PySide6 GUI
(``spacr``), a headless pipeline (:mod:`spacr.core`), and a
plate-to-classification workflow that runs on top of PyTorch,
Cellpose, scikit-image, and scipy.

It is built for cell biologists running pooled or arrayed CRISPR screens
who need per-cell measurements out of plate images. The GUI route needs no
programming; the same steps are available as a scripted pipeline when a
screen outgrows one desktop.

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: 🚀 Get started
      :link: https://github.com/EinarOlafsson/spacr#quickstart
      :link-type: url

      Install spaCR from PyPI and launch the Qt GUI in two commands.

   .. grid-item-card:: 🎓 Interactive tutorials
      :link: tutorials/
      :link-type: url

      The lesson library — 40 narrated, step-by-step lessons covering the
      whole pipeline.

   .. grid-item-card:: 📖 API reference
      :link: api/index
      :link-type: doc

      Every public function, method, and class — grouped by module.

   .. grid-item-card:: 🎬 Video tutorials
      :link: https://github.com/EinarOlafsson/spacr#narrated-video-tutorials
      :link-type: url

      Narrated walkthroughs of each pipeline module.

   .. grid-item-card:: 🐛 Report an issue
      :link: https://github.com/EinarOlafsson/spacr/issues/new
      :link-type: url

      File a bug, request a feature, or ask a question.


Pipeline overview
-----------------

.. image:: https://github.com/EinarOlafsson/spacr/raw/main/spacr/resources/icons/flow_chart_v3.png
   :alt: spaCR workflow
   :align: center

The GUI ships 63 apps, grouped into seven categories — *Core*, *Data*,
*Segmentation models*, *Results & QC*, *Explore*, *Toxoplasma*, and *Design*.
The nine Core apps form the main pipeline; these five are the path most
screens take, each with its own :doc:`API reference <api/index>` module:

+---------------------+-----------------------------------------------------+
| **Mask**            | Cellpose segmentation of cells, nuclei, pathogens.  |
|                     | :func:`spacr.core.preprocess_generate_masks`        |
+---------------------+-----------------------------------------------------+
| **Measure**         | Per-object feature extraction into a SQLite DB.     |
|                     | :func:`spacr.measure.measure_crop`                  |
+---------------------+-----------------------------------------------------+
| **Annotate**        | Grid-based manual labelling of single-cell crops.   |
|                     | :mod:`spacr.app_annotate`                           |
+---------------------+-----------------------------------------------------+
| **Classify**        | CNN / XGBoost training from annotations.            |
|                     | :mod:`spacr.deep_spacr`, :mod:`spacr.ml`            |
+---------------------+-----------------------------------------------------+
| **Map Barcodes**    | Map FASTQ reads to row/column/gRNA barcodes.        |
|                     | :mod:`spacr.sequencing`                             |
+---------------------+-----------------------------------------------------+


Key modules by category
-----------------------

**Core pipelines**
   :mod:`spacr.core` · :mod:`spacr.io` · :mod:`spacr.measure` ·
   :mod:`spacr.object` · :mod:`spacr.utils`

**Machine learning + classification**
   :mod:`spacr.ml` · :mod:`spacr.deep_spacr` · :mod:`spacr.predictions` ·
   :mod:`spacr.spacr_cellpose`

**Analysis**
   :mod:`spacr.plot` · :mod:`spacr.sp_stats` · :mod:`spacr.submodules` ·
   :mod:`spacr.toxo` · :mod:`spacr.timelapse` · :mod:`spacr.sim`

**Sequencing**
   :mod:`spacr.sequencing`

**Modern Qt GUI**
   ``spacr.qt`` — launched via the ``spacr`` or ``spacr-qt`` CLI.

**Classic Tk GUI**
   :mod:`spacr.gui` · :mod:`spacr.gui_core` · :mod:`spacr.gui_utils`


Installation
------------

The GUI lives behind the ``qt`` extra, so the desktop install must ask for
it — plain ``pip install spacr`` gives you the pipelines but no PySide6, and
``spacr`` will tell you so rather than launch.

.. code-block:: bash

   python -m pip install "spacr[qt]"
   spacr                    # launch the Qt GUI

Headless (cluster, server, CI) — no Qt, no display:

.. code-block:: bash

   python -m pip install spacr
   spacr-run --list         # list the headless pipeline modules


Learn spaCR
-----------

The `interactive tutorial library <tutorials/>`_ contains 69 narrated,
step-by-step lessons covering every module, with 50 voices across eight
languages. It is also reachable from the GUI:
**Help → Tutorial (web)**, or the spaCR logo on the classic Tk start screen.


Contents
--------

.. toctree::
   :maxdepth: 2

   localization
   setting_animations
   checkpoint_resume
   reproducibility_manifests
   run_history
   batch_correction
   classifier_evaluation
   guide_permutation
   model_explanation
   umap_multiobjective
   distributed_execution
   plugin_sdk
   leakage_audit
   threading_cancellation_audit
   database_concurrency_audit
   api/index
