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

spaCR is an end-to-end toolkit for image-based pooled CRISPR screens
and high-content microscopy. It connects segmentation, object
measurements, single-cell crops, annotation, phenotype classification,
barcode mapping, and statistical analysis in one reproducible workflow.
Microscopy-derived objects and measurements are kept in SQLite so that
images, labels, model predictions, and sequencing results remain linked.

`Documentation <https://einarolafsson.github.io/spacr/>`_ ·
`Tutorials <https://einarolafsson.github.io/spacr/tutorial/>`_ ·
`PyPI <https://pypi.org/project/spacr/>`_ ·
`Issues <https://github.com/EinarOlafsson/spacr/issues>`_


Highlights
----------

- **Segmentation and quality control** — Cellpose-based masks for cells,
  nuclei, pathogens, and organelles, with interactive previews.
- **Measurements and crops** — object morphology, intensity, texture,
  colocalization, radial-distribution features, and linked single-object
  PNG crops.
- **Annotation** — a Qt grid interface for reviewing images and writing
  labels directly to the measurements database.
- **Model training and interpretation** — classical ML and PyTorch
  computer-vision workflows, live TensorBoard metrics, and optional
  class-activation maps.
- **Timelapse analysis** — object tracking, motility measurements, and
  preview controls for time-resolved experiments.
- **Exploratory analysis** — interactive image UMAPs, manual cluster
  selection, cluster propagation, and database annotation.
- **Pooled-screen linkage** — FASTQ barcode mapping for row, column, and
  gRNA libraries, followed by regression and gene-level summaries.
- **Reproducibility** — a headless CLI, settings validation, manifests,
  run reports, structured logging, and explicit data-shape contracts.

The established 2D workflow is the most mature. Settings and workflows
marked **Beta** or **Alpha** in the application—including newer 3D and
4D paths—should be validated on representative data before production
use.


Workflow
--------

.. image:: https://raw.githubusercontent.com/EinarOlafsson/spacr/main/spacr/resources/icons/flow_chart_v3.png
   :alt: spaCR workflow and output organization
   :align: center

Microscopy images (TIFF, LIF, CZI, and ND2) and sequencing reads (FASTQ)
enter complementary image-analysis and barcode-mapping pipelines. The
resulting object tables, crops, predictions, and well-level summaries
can then be analyzed together.


Quick start
-----------

The recommended installation uses an isolated environment and the Qt
extra:

.. code-block:: bash

   conda create -n spacr python=3.12 -y
   conda activate spacr
   python -m pip install --upgrade pip
   python -m pip install "spacr[qt]"
   spacr

spaCR supports Python 3.10 through 3.13. Linux is the recommended
platform for GPU workflows; macOS and Windows are also supported.


Installation options
--------------------

**Desktop application from PyPI**

.. code-block:: bash

   python -m pip install "spacr[qt]"
   spacr

**Headless or server installation**

.. code-block:: bash

   python -m pip install spacr
   spacr-run --list

**Latest development branch**

.. code-block:: bash

   git clone https://github.com/EinarOlafsson/spacr.git
   cd spacr
   git switch spacr-nightly
   python -m pip install -e ".[qt]"

**Conda**

spaCR is not yet published as a conda-forge package. Conda can still
manage the environment while pip installs spaCR, as shown in the quick
start. A native conda-forge recipe is planned; until it is accepted,
``conda install -c conda-forge spacr`` will not work.

Some workflows have optional dependencies. Install only the extras you
need:

.. code-block:: bash

   python -m pip install "spacr[trackastra]"     # Trackastra tracking
   python -m pip install "spacr[ultrack]"        # ultrack tracking
   python -m pip install "spacr[umap]"           # UMAP workflows
   python -m pip install "spacr[attribution]"    # TorchCAM attribution
   python -m pip install "spacr[tutorial]"       # tutorial environment

The legacy Tk interface remains available as ``spacr-legacy`` but is no
longer under active development.


Command-line entry points
-------------------------

.. code-block:: bash

   spacr                                      # Qt application
   spacr-run --list                           # list headless modules
   spacr-run MODULE --settings settings.csv   # execute a module
   spacr-run validate --module MODULE \
       --settings settings.csv                # validate before running
   spacr-repro --help                         # reproducibility tools

Set ``SPACR_LOG_LEVEL=DEBUG`` when troubleshooting. Logs rotate under
``~/.spacr/logs/spacr.log``.


Example notebooks
-----------------

- `Generate masks <https://github.com/EinarOlafsson/spacr/blob/main/Notebooks/1_spacr_generate_masks.ipynb>`_
- `Capture single-cell images and measurements <https://github.com/EinarOlafsson/spacr/blob/main/Notebooks/2_spacr_generate_mesurments_crop_images.ipynb>`_
- `Machine-learning object classification <https://github.com/EinarOlafsson/spacr/blob/main/Notebooks/3a_spacr_machine_learning.ipynb>`_
- `Computer-vision object classification <https://github.com/EinarOlafsson/spacr/blob/main/Notebooks/3b_spacr_computer_vision.ipynb>`_
- `Map sequencing barcodes <https://github.com/EinarOlafsson/spacr/blob/main/Notebooks/4_spacr_map_barecodes.ipynb>`_
- `Finetune Cellpose models <https://github.com/EinarOlafsson/spacr/blob/main/Notebooks/5_spacr_train_cellpose.ipynb>`_


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
