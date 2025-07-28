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

**Core Modules**
- :doc:`Core Logic <api/spacr/core/index>`
- :doc:`IO Utilities <api/spacr/io/index>`
- :doc:`General Utilities <api/spacr/utils/index>`
- :doc:`Settings <api/spacr/settings/index>`
- :doc:`Statistics <api/spacr/sp_stats/index>`

**Image Analysis**
- :doc:`Measurement <api/spacr/measure/index>`
- :doc:`Plotting <api/spacr/plot/index>`
- :doc:`Cellpose Integration <api/spacr/spacr_cellpose/index>`

**Classification**
- :doc:`Classical ML <api/spacr/ml/index>`
- :doc:`Deep Learning <api/spacr/deep_spacr/index>`

**GUI Components**
- :doc:`GUI Main App <api/spacr/gui/index>`
- :doc:`GUI Core <api/spacr/gui_core/index>`
- :doc:`GUI Elements <api/spacr/gui_elements/index>`
- :doc:`GUI Utilities <api/spacr/gui_utils/index>`

**Sequencing & Submodules**
- :doc:`Sequencing <api/spacr/sequencing/index>`
- :doc:`Toxoplasma Tools <api/spacr/toxo/index>`
- :doc:`Submodules <api/spacr/submodules/index>`

GitHub Repository
=================

Visit the source code on GitHub: https://github.com/EinarOlafsson/spacr
