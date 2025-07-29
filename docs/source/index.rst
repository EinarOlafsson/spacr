Welcome to SpaCr
================

.. |Docs| image:: https://github.com/EinarOlafsson/spacr/actions/workflows/pages/pages-build-deployment/badge.svg
   :target: https://einarolafsson.github.io/spacr/index.html
.. |PyPI version| image:: https://badge.fury.io/py/spacr.svg
   :target: https://badge.fury.io/py/spacr
.. |Python version| image:: https://img.shields.io/pypi/pyversions/spacr
   :target: https://pypistats.org/packages/spacr
.. |Licence: MIT| image:: https://img.shields.io/github/license/EinarOlafsson/spacr
   :target: https://github.com/EinarOlafsson/spacr/blob/main/LICENSE
.. |repo size| image:: https://img.shields.io/github/repo-size/EinarOlafsson/spacr
   :target: https://github.com/EinarOlafsson/spacr/
.. |Tutorial| image:: https://img.shields.io/badge/Tutorial-Click%20Here-brightgreen
   :target: https://einarolafsson.github.io/spacr/tutorial/

|Docs| |PyPI version| |Python version| |Licence: MIT| |repo size| |Tutorial|

**spaCR** (Spatial Phenotype Analysis of CRISPR Screens) is a Python toolkit for analyzing pooled CRISPR-Cas9 imaging screens. It integrates high-content imaging data with sequencing-based mutant identification to enable genotype-to-phenotype mapping at the single-cell level.

spaCR supports:

- **Segmentation** of microscopy images using models like Cellpose.
- **Single-cell feature extraction** and image cropping.
- **Classification** of phenotypes using classical and deep learning models.
- **Barcode decoding** from sequencing reads and well-level mutant quantification.
- **Statistical analysis**, including regression models to link genotypes to phenotypes.
- **Interactive visualization** of results including Grad-CAMs and phenotype maps.
- **GUI tools** for mask curation, annotation, and exploratory analysis.

Example Notebooks
=================

The following example Jupyter notebooks illustrate common workflows using spaCR.

- `Generate masks <https://github.com/EinarOlafsson/spacr/blob/main/Notebooks/1_spacr_generate_masks.ipynb>`_  
  *Generate cell, nuclei, and pathogen segmentation masks from microscopy images using Cellpose.*

- `Capture single cell images and measurements <https://github.com/EinarOlafsson/spacr/blob/main/Notebooks/2_spacr_generate_mesurments_crop_images.ipynb>`_  
  *Extract object-level measurements and crop single-cell images for downstream analysis.*

- `Machine learning based object classification <https://github.com/EinarOlafsson/spacr/blob/main/Notebooks/3a_spacr_machine_learning.ipynb>`_  
  *Train traditional machine learning models (e.g., XGBoost) to classify cell phenotypes based on extracted features.*

- `Computer vision based object classification <https://github.com/EinarOlafsson/spacr/blob/main/Notebooks/3b_spacr_computer_vision.ipynb>`_  
  *Train and evaluate deep learning models (PyTorch CNNs/Transformers) on cropped object images.*

- `Map sequencing barcodes <https://github.com/EinarOlafsson/spacr/blob/main/Notebooks/4_spacr_map_barecodes.ipynb>`_  
  *Map sequencing reads to row, column, and gRNA barcodes for CRISPR screen genotype-phenotype mapping.*

- `Finetune cellpose models <https://github.com/EinarOlafsson/spacr/blob/main/Notebooks/5_spacr_train_cellpose.ipynb>`_  
  *Finetune Cellpose models using your own annotated training data for improved segmentation accuracy.*

API Reference by Category
=========================

.. toctree::
   :caption: Core Modules
   :maxdepth: 1

   Core Logic <api/spacr/core/index>
   IO Utilities <api/spacr/io/index>
   General Utilities <api/spacr/utils/index>
   Settings <api/spacr/settings/index>
   Statistics <api/spacr/sp_stats/index>

.. toctree::
   :caption: Image Analysis
   :maxdepth: 1

   Measurement <api/spacr/measure/index>
   Plotting <api/spacr/plot/index>
   Cellpose Integration <api/spacr/spacr_cellpose/index>

.. toctree::
   :caption: Classification
   :maxdepth: 1

   Classical ML <api/spacr/ml/index>
   Deep Learning <api/spacr/deep_spacr/index>

.. toctree::
   :caption: GUI Components
   :maxdepth: 1

   GUI Main App <api/spacr/gui/index>
   GUI Core <api/spacr/gui_core/index>
   GUI Elements <api/spacr/gui_elements/index>
   GUI Utilities <api/spacr/gui_utils/index>

.. toctree::
   :caption: Sequencing & Submodules
   :maxdepth: 1

   Sequencing <api/spacr/sequencing/index>
   Toxoplasma Tools <api/spacr/toxo/index>
   Submodules <api/spacr/submodules/index>

GitHub Repository
=================

Visit the source code on GitHub: https://github.com/EinarOlafsson/spacr

.. toctree::
   :hidden:

   api/index
