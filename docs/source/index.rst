Welcome to SpaCr
================

**spaCR** (Spatial Phenotype Analysis of CRISPR Screens) is a Python toolkit for analyzing pooled CRISPR-Cas9 imaging screens. It integrates high-content imaging data with sequencing-based mutant identification to enable genotype-to-phenotype mapping at the single-cell level.

spaCR provides a modular and extensible framework that supports:

- **Segmentation** of microscopy images using models like Cellpose.
- **Single-cell feature extraction** and image cropping.
- **Classification** of phenotypes using classical and deep learning models.
- **Barcode decoding** from sequencing reads and well-level mutant quantification.
- **Statistical analysis**, including regression models to link genotypes to phenotypes.
- **Interactive visualization** of results including Grad-CAMs and phenotype maps.
- **GUI tools** for mask curation, annotation, and exploratory analysis.

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
   
