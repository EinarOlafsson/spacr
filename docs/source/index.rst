Welcome to spaCR
================

.. image:: _static/logo_spacr_docs.png
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
who need per-cell measurements from plate images. The GUI route needs no
programming; the same processing steps are available through the Python API
for scripted and reproducible workflows.

The GUI groups its applications into six categories: *Core* for the
segment-measure-classify pipeline, *Data* for import, inspection and hand
correction of masks, *Results & QC* for regression and its diagnostics,
*Explore* for the interactive figures, *Assays* for the parasite-specific
readouts, and *Design* for planning a screen before it runs.

Not every screen is a tile, which is why no count of them is printed here.
Work that only makes sense inside another step opens from that step's
masthead instead — **Timelapse** from Mask, **Illumination** and the
**Motility Assay** from Measure, **Classifier Evaluation** and **Explain CV
Model** from Classify, **Annotator Agreement** from Annotate, and the
**Cellpose Workbench**, **Model Compare**, **Model Zoo** and **Curate** from
Make Masks, among them — so a screen with no tile below is one step further
in rather than gone. Home lists whatever the running build offers.

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

      Supported workflow entry points and the complete module reference.

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

Every tile links to the API page used by that application's in-product help.

.. include:: _generated/workflow_grid.rst


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
**Help → Tutorial (web)**.


Contents
--------

.. toctree::
   :maxdepth: 2

   installer_guide
   installers
   features
   python_api
   Language <localization>
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
