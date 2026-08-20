Welcome to spaCR
================

.. image:: _static/logo_spacr.png
   :align: center
   :alt: spaCR Logo
   :width: 200px

**spaCR** — *Spatial phenotype analysis of CRISPR screens.*

.. note::

   You are reading the |docs-channel| documentation for spaCR
   |spacr-version|. The public site follows released ``main``; nightly
   builds validate upcoming changes without replacing the stable site.

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
      :link: https://github.com/EinarOlafsson/spacr#install-spacr
      :link-type: url

      Install spaCR from PyPI and launch the Qt GUI in two commands.

   .. grid-item-card:: 🎓 Interactive tutorials
      :link: tutorials/
      :link-type: url

      The lesson library — |lesson-count| narrated, step-by-step lessons covering the
      whole pipeline.

   .. grid-item-card:: 📖 API reference
      :link: api/index
      :link-type: doc

      Supported workflow entry points, with the complete module reference
      available for contributors.

   .. grid-item-card:: 🎬 Video tutorials
      :link: tutorials/
      :link-type: url

      Narrated walkthroughs of each pipeline module.

   .. grid-item-card:: 🐛 Report an issue
      :link: https://github.com/EinarOlafsson/spacr/issues/new
      :link-type: url

      File a bug, request a feature, or ask a question.


Applications and workflow
-------------------------

.. image:: ../../spacr/resources/icons/workflow_home_apps.png
   :alt: The six-step spaCR workflow followed by all 58 applications grouped as they appear on the home screen
   :align: center

Most screens follow **Mask → Measure → Annotate → Classify → Map Barcodes →
Regression**. The remaining tiles are grouped exactly as they appear on the
home screen: Data, Segmentation models, Results & QC, Explore, Toxoplasma and
Design. Start with the :doc:`feature guide <features>` for capabilities or the
:doc:`curated API reference <api/index>` for supported Python entry points.


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

The `interactive tutorial library <tutorials/>`_ contains |lesson-count| narrated,
step-by-step lessons covering every module, with 50 voices across eight
languages. It is also reachable from the GUI:
**Help → Tutorial (web)**, or the spaCR logo on the classic Tk start screen.


Contents
--------

.. toctree::
   :maxdepth: 2

   installers
   features
   python_api
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
