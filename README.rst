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


.. _docs: https://einarolafsson.github.io/spacr/index.html

Badges
------
|Docs| |PyPI version| |Python version| |Licence: MIT| |repo size| |Tutorial|


.. |logo| image:: https://raw.githubusercontent.com/EinarOlafsson/spacr/main/spacr/resources/icons/logo_spacr.png
   :height: 100

+------------------------------------------------+----------------------------------------------------+
| |logo|                                         | **spaCR**                                          |
|                                                | Spatial phenotype analysis of CRISPR-              |
|                                                | Cas9 screens (spaCR).                              |
+------------------------------------------------+----------------------------------------------------+

The spatial organization of organelles and proteins within cells constitutes a key level of functional regulation. In the context of infectious disease, the spatial relationships between host cell structures and intracellular pathogens are critical to understanding host clearance mechanisms and how pathogens evade them. SpaCr is a Python-based software package for generating single-cell image data for deep-learning sub-cellular/cellular phenotypic classification from pooled genetic CRISPR-Cas9 screens. SpaCr provides a flexible toolset to extract single-cell images and measurements from high-content cell painting experiments, train deep-learning models to classify cellular/subcellular phenotypes, simulate, and analyze pooled CRISPR-Cas9 imaging screens.

Features
--------

-  **Generate Masks:** Generate cellpose masks of cell, nuclei, and pathogen objects.
-  **Object Measurements:** Measurements for each object including scikit-image regionprops, intensity percentiles, shannon-entropy, Pearson’s and Manders’ correlations, homogeneity, and radial distribution. Measurements are saved to a SQL database in object-level tables.
-  **Crop Images:** Save objects (cells, nuclei, pathogen, cytoplasm) as images. Object image paths are saved in a SQL database.
-  **Train CNNs or Transformers:** Train Torch models to classify single object images.
-  **Manual Annotation (nightly):** Supports manual annotation of single-cell images and segmentation to refine training datasets for training CNNs/Transformers or cellpose, respectively.
-  **Finetune Cellpose Models (nightly):** Adjust pre-existing Cellpose models to your specific dataset for improved performance.
-  **Timelapse Data Support (nightly):** Track objects in timelapse image data.
-  **Simulations (nightly):** Simulate spatial phenotype screens.
-  **Sequencing:** Map FASTQ reads to barcode and gRNA barcode metadata.
-  **Misc (nightly):** Analyze Ca oscillation, recruitment, infection rate, plaque size/count.

.. image:: https://github.com/EinarOlafsson/spacr/raw/main/spacr/resources/icons/flow_chart_v3.png
   :alt: SpaCr workflow
   :align: center

**Overview and data organization of spaCR.**

**a.** Schematic workflow of the spaCR pipeline for pooled image-based CRISPR screens. Microscopy images (TIFF, LIF, CZI, NDI) and sequencing reads (FASTQ) are used as inputs (black). The main modules (teal) are: (1) Mask: generates object masks for cells, nuclei, pathogens, and cytoplasm; (2) Measure: extracts object-level features and crops object images, storing quantitative data in an SQL database; (3) Classify—applies machine learning (ML, e.g., XGBoost) or deep learning (DL, e.g., PyTorch) models to classify objects, summarizing results as well-level classification scores; (4) Map Barcodes: extracts and maps row, column, and gRNA barcodes from sequencing data to corresponding wells; (5) Regression: estimates gRNA effect sizes and gene scores via multiple linear regression using well-level summary statistics.
**b.** Output folder structure for each module, including locations for raw and processed images, masks, object-level measurements, datasets, and results.
**c.** List of all spaCR package modules.

Installation
------------

**Linux recommended.**  
If using Windows, switch to Linux—it's free, open-source, and better.

**macOS prerequisites (before install):**

::

   brew install libomp hdf5 cmake openssl

**Linux GUI requirement:**  
SpaCr GUI requires Tkinter.  **b.** Downstream submodules available for extended analyses at each stage.

**Install stable version (main):**

::

   pip install spacr

**Install nightly version:**

::

   pip install spacr-nightly

**Launch GUI (stable):**

::

   spacr

**Launch GUI (nightly):**

::

   spacrn



Example Notebooks
-----------------

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

Interactive Tutorial (under construction)
-----------------------------------------

Click below to explore the step-by-step GUI and Notebook tutorials for spaCR:

|Tutorial|

spaCRPower
----------

Power analasys of pooled perterbation spaCR screens.

`spaCRPower <https://github.com/maomlab/spaCRPower>`_

Data Availability
-----------------

- **Full microscopy image dataset:**  
  `EMBL-EBI BioStudies S-BIAD2135 <https://doi.org/10.6019/S-BIAD2135>`_

- **Testing dataset:**  
  `Hugging Face toxo_mito <https://huggingface.co/datasets/einarolafsson/toxo_mito>`_

- **Sequencing data:**  
  `NCBI BioProject PRJNA1261935 <https://www.ncbi.nlm.nih.gov/bioproject/?term=PRJNA1261935>`_

License
-------
spaCR is distributed under the terms of the MIT License.
See the `LICENSE <https://github.com/EinarOlafsson/spacr/blob/main/LICENSE>`_ file for details.

How to Cite
-----------
If you use spaCR in your research, please cite:  
Olafsson EB, et al. SpaCr: Spatial phenotype analysis of CRISPR-Cas9 screens. *Manuscript in preparation*.

Papers Using spaCR
-------------------
Below are selected publications that have used or cited spaCR:

- Olafsson EB, et al. *SpaCr: Spatial phenotype analysis of CRISPR-Cas9 screens.* Manuscript in preparation.
- `IRE1α promotes phagosomal calcium flux to enhance macrophage fungicidal activity  <https://doi.org/10.1016/j.celrep.2025.115694>`_
- `Metabolic adaptability and nutrient scavenging in Toxoplasma gondii: insights from ingestion pathway-deficient mutants  <https://doi.org/10.1128/msphere.01011-24>`_