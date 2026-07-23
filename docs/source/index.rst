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

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: 🚀 Get started
      :link: https://github.com/EinarOlafsson/spacr#quickstart
      :link-type: url

      Install spaCR from PyPI and launch the Qt GUI in two commands.

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

spaCR is organised around five pipeline apps, each with its own
:doc:`API reference <api/index>` module:

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
   :mod:`spacr.ml` · :mod:`spacr.deep_spacr` · :mod:`spacr.spacr_cellpose`

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

.. code-block:: bash

   pip install spacr
   spacr                    # launch the Qt GUI


Contents
--------

.. toctree::
   :maxdepth: 2

   api/index
