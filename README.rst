|Docs| |PyPI version| |Python version| |Licence: MIT| |repo size| |Tests| |Qt| |Tutorial| |DOI|

.. |Docs| image:: https://github.com/EinarOlafsson/spacr/actions/workflows/pages/pages-build-deployment/badge.svg
   :target: https://einarolafsson.github.io/spacr/index.html
.. |PyPI version| image:: https://badge.fury.io/py/spacr.svg
   :target: https://pypi.org/project/spacr/
.. |Python version| image:: https://img.shields.io/pypi/pyversions/spacr
   :target: https://pypistats.org/packages/spacr
.. |Licence: MIT| image:: https://img.shields.io/github/license/EinarOlafsson/spacr
   :target: https://github.com/EinarOlafsson/spacr/blob/main/LICENSE
.. |repo size| image:: https://img.shields.io/github/repo-size/EinarOlafsson/spacr
   :target: https://github.com/EinarOlafsson/spacr/
.. |Tests| image:: https://github.com/EinarOlafsson/spacr/actions/workflows/tests.yml/badge.svg?branch=nightly
   :target: https://github.com/EinarOlafsson/spacr/actions/workflows/tests.yml
.. |Qt| image:: https://img.shields.io/badge/GUI-Qt%20(PySide6)-4A9EFF
   :target: https://einarolafsson.github.io/spacr/index.html
.. |Tutorial| image:: https://img.shields.io/badge/Tutorial-Click%20Here-brightgreen
   :target: https://einarolafsson.github.io/spacr/tutorial/
.. |DOI| image:: https://img.shields.io/badge/DOI-10.5281%2Fzenodo.21343317-blue.svg
   :target: https://doi.org/10.5281/zenodo.21343317

.. _docs: https://einarolafsson.github.io/spacr/index.html


spaCR
=====

**Spatial phenotype analysis of CRISPR-Cas9 screens.**

The spatial organization of organelles and proteins within cells
constitutes a key level of functional regulation. In the context of
infectious disease, the spatial relationships between host cell
structures and intracellular pathogens are critical to understanding
host clearance mechanisms and how pathogens evade them. **spaCR** is a
Python toolkit for generating single-cell image data for deep-learning
sub-cellular / cellular phenotypic classification from pooled genetic
CRISPR-Cas9 screens. It provides a flexible toolset to extract
single-cell images and measurements from high-content cell painting
experiments, train deep-learning models to classify cellular
phenotypes, simulate CRISPR-Cas9 imaging screens, and analyze
pooled-screen data end to end.

📖 **Full documentation:** https://einarolafsson.github.io/spacr/

Features
--------

- **Generate Masks** — Cellpose masks for cells, nuclei, pathogens and organelles.
- **Object Measurements** — scikit-image regionprops, intensity percentiles, Shannon entropy, Pearson's and Manders' correlations, homogeneity, radial distribution. Saved to a SQL database in object-level tables.
- **Crop Images** — Save cropped objects (cells, nuclei, pathogen, cytoplasm) as PNGs alongside their DB rows.
- **Train CNNs or Transformers** — PyTorch training loops for single-object classification.
- **Manual Annotation** — Grid-based single-cell annotation UI that writes labels straight to the measurements DB.
- **Finetune Cellpose Models** — Refine pretrained Cellpose weights against your own hand-drawn masks.
- **Timelapse Data Support** — Track objects across timepoints; every module respects the ``T`` filename dimension.
- **Simulations** — Generate synthetic phenotype screens with configurable perturbation effects.
- **Sequencing** — Map FASTQ reads to row / column / gRNA barcodes for pooled-screen genotype-phenotype linking.
- **Analysis Suite** — UMAP, ML/DL classification, regression, recruitment, activation, plaque, Ca²⁺ oscillation.

.. image:: https://github.com/EinarOlafsson/spacr/raw/main/spacr/resources/icons/flow_chart_v3.png
   :alt: spaCR workflow
   :align: center

**Overview and data organization of spaCR.**

**a.** Schematic workflow of the spaCR pipeline for pooled image-based CRISPR screens. Microscopy images (TIFF, LIF, CZI, NDI) and sequencing reads (FASTQ) are used as inputs (black). The main modules (teal) are: (1) Mask — generates object masks for cells, nuclei, pathogens, and cytoplasm; (2) Measure — extracts object-level features and crops object images, storing quantitative data in an SQL database; (3) Classify — applies ML (e.g., XGBoost) or DL (e.g., PyTorch) models to classify objects, summarising results as well-level classification scores; (4) Map Barcodes — extracts and maps row, column, and gRNA barcodes from sequencing data to corresponding wells; (5) Regression — estimates gRNA effect sizes and gene scores via multiple linear regression using well-level summary statistics.
**b.** Downstream submodules available for extended analyses at each stage.
**c.** Output folder structure for each module, including locations for raw and processed images, masks, object-level measurements, datasets, and results.
**d.** List of all spaCR package modules.


Quickstart
----------

.. code-block:: bash

   pip install spacr
   spacr                        # launches the Qt GUI


Installation
------------

**Linux is the recommended platform.** Windows users are encouraged to
switch to Linux — it's free, open-source, and simply works better with
the scientific Python + GPU stack.

**macOS prerequisites** (before ``pip install``):

.. code-block:: bash

   brew install libomp hdf5 cmake openssl

**Linux prerequisites** (only if you also want the classic Tk GUI):

.. code-block:: bash

   sudo apt-get install python3-tk

**Install** (PyPI):

.. code-block:: bash

   pip install spacr

**Install** (from source, latest development branch):

.. code-block:: bash

   git clone https://github.com/EinarOlafsson/spacr.git
   cd spacr && pip install -e '.[qt]'

**Launch**:

.. code-block:: bash

   spacr           # Qt GUI (default)
   spacr-qt        # explicit alias for the Qt GUI
   spacr-legacy    # classic Tk GUI


Example Notebooks
-----------------

The following Jupyter notebooks illustrate common workflows:

- `Generate masks <https://github.com/EinarOlafsson/spacr/blob/main/Notebooks/1_spacr_generate_masks.ipynb>`_ — generate cell, nuclei and pathogen segmentation masks from microscopy images using Cellpose.
- `Capture single-cell images and measurements <https://github.com/EinarOlafsson/spacr/blob/main/Notebooks/2_spacr_generate_mesurments_crop_images.ipynb>`_ — extract object-level measurements and crop single-cell images for downstream analysis.
- `Machine-learning object classification <https://github.com/EinarOlafsson/spacr/blob/main/Notebooks/3a_spacr_machine_learning.ipynb>`_ — train traditional ML models (e.g., XGBoost) to classify cell phenotypes.
- `Computer-vision object classification <https://github.com/EinarOlafsson/spacr/blob/main/Notebooks/3b_spacr_computer_vision.ipynb>`_ — train and evaluate deep-learning models (PyTorch CNNs/Transformers) on cropped object images.
- `Map sequencing barcodes <https://github.com/EinarOlafsson/spacr/blob/main/Notebooks/4_spacr_map_barecodes.ipynb>`_ — map sequencing reads to row, column, and gRNA barcodes for genotype-phenotype mapping.
- `Finetune Cellpose models <https://github.com/EinarOlafsson/spacr/blob/main/Notebooks/5_spacr_train_cellpose.ipynb>`_ — refine Cellpose models with your own annotated training data.


Interactive Tutorial
--------------------

Click below to explore the step-by-step GUI and Notebook tutorials for spaCR:

|Tutorial|


Debugging & logs
----------------

Every subsystem funnels through the same rotating log at
``~/.spacr/logs/spacr.log`` (5 MB × 3 backups). Crank the level via
env var:

.. code-block:: bash

   SPACR_LOG_LEVEL=DEBUG spacr

Or, interactively:

.. code-block:: python

   from spacr.logging_util import setup_logging, enable_debug
   setup_logging()              # once at program start
   enable_debug()               # all spacr.* loggers → DEBUG


spaCRPower
----------

Power analysis of pooled-perturbation spaCR screens.

`spaCRPower <https://github.com/maomlab/spaCRPower>`_


Data Availability
-----------------

- **Full microscopy image dataset:** `EMBL-EBI BioStudies S-BIAD2135 <https://doi.org/10.6019/S-BIAD2135>`_
- **Testing dataset:** `Hugging Face toxo_mito <https://huggingface.co/datasets/einarolafsson/toxo_mito>`_
- **Sequencing data:** `NCBI BioProject PRJNA1261935 <https://www.ncbi.nlm.nih.gov/bioproject/?term=PRJNA1261935>`_


License
-------

spaCR is distributed under the terms of the MIT License. See the
`LICENSE <https://github.com/EinarOlafsson/spacr/blob/main/LICENSE>`_
file for details.


How to Cite
-----------

If you use spaCR in your research, please cite:

Olafsson EB, *et al.* A pooled image-based CRISPR screen identifies
EAF1 as a *T. gondii* modulator of ESCRT subversion. *Manuscript under
consideration.*

- `Preprint (bioRxiv) — A pooled image-based CRISPR screen identifies EAF1 as a T. gondii modulator of ESCRT subversion <https://www.biorxiv.org/content/10.64898/2026.07.08.737057v1>`_


Papers Using spaCR
------------------

Selected publications that have used or cited spaCR:

- Olafsson EB, *et al.* *SpaCR: Spatial phenotype analysis of CRISPR-Cas9 screens.* Manuscript in preparation.
- `IRE1α promotes phagosomal calcium flux to enhance macrophage fungicidal activity <https://doi.org/10.1016/j.celrep.2025.115694>`_
- `Metabolic adaptability and nutrient scavenging in Toxoplasma gondii: insights from ingestion pathway-deficient mutants <https://doi.org/10.1128/msphere.01011-24>`_
