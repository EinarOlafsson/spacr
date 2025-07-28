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

This API reference documents all public classes, functions, and modules included in the  package. For usage examples, please refer to the tutorials and GUI walkthroughs in the documentation.

.. toctree::
   :maxdepth: 1

   api/index
