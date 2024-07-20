.. |Documentation Status| image:: https://readthedocs.org/projects/spacr/badge/?version=latest
   :target: https://spacr.readthedocs.io/en/latest/?badge=latest
.. |PyPI version| image:: https://badge.fury.io/py/spacr.svg
   :target: https://badge.fury.io/py/spacr
.. |Python version| image:: https://img.shields.io/pypi/pyversions/spacr
   :target: https://pypistats.org/packages/spacr
.. |Licence: GPL v3| image:: https://img.shields.io/github/license/EinarOlafsson/spacr
   :target: https://github.com/EinarOlafsson/spacr/blob/master/LICENSE
.. |repo size| image:: https://img.shields.io/github/repo-size/EinarOlafsson/spacr
   :target: https://github.com/EinarOlafsson/spacr/

|Documentation Status| |PyPI version| |Python version| |Licence: GPL v3| |repo size|

SpaCr
=====

Spatial phenotype analysis of CRISPR-Cas9 screens (SpaCr). The spatial organization of organelles and proteins within cells constitutes a key level of functional regulation. In the context of infectious disease, the spatial relationships between host cell structures and intracellular pathogens are critical to understanding host clearance mechanisms and how pathogens evade them. SpaCr is a Python-based software package for generating single-cell image data for deep-learning sub-cellular/cellular phenotypic classification from pooled genetic CRISPR-Cas9 screens. SpaCr provides a flexible toolset to extract single-cell images and measurements from high-content cell painting experiments, train deep-learning models to classify cellular/subcellular phenotypes, simulate, and analyze pooled CRISPR-Cas9 imaging screens.

Features
--------

-  **Generate Masks:** Generate cellpose masks of cell, nuclei, and pathogen objects.

-  **Object Measurements:** Measurements for each object including scikit-image-regionprops, intensity percentiles, shannon-entropy, pearsons and manders correlations, homogeneity, and radial distribution. Measurements are saved to a SQL database in object-level tables.

-  **Crop Images:** Save objects (cells, nuclei, pathogen, cytoplasm) as images. Object image paths are saved in a SQL database.

-  **Train CNNs or Transformers:** Train Torch models to classify single object images.

-  **Manual Annotation:** Supports manual annotation of single-cell images and segmentation to refine training datasets for training CNNs/Transformers or cellpose, respectively.

-  **Finetune Cellpose Models:** Adjust pre-existing Cellpose models to your specific dataset for improved performance.

-  **Timelapse Data Support:** Track objects in timelapse image data.

-  **Simulations:** Simulate spatial phenotype screens.

-  **Sequencing:** Map FASTQ reads to barcode and gRNA barcode metadata.

-  **Misc:** Analyze Ca oscillation, recruitment, infection rate, plaque size/count.

Installation
------------

If using Windows, switch to Linux—it's free, open-source, and better.

Before installing SpaCr on OSX ensure OpenMP is installed::

   brew install libomp

SpaCr GUI requires Tkinter. On Linux, ensure Tkinter is installed. (Tkinter is included with the standard Python installation on macOS and Windows)::

   sudo apt-get install python3-tk

Install SpaCr with pip::

   pip install spacr

Run SpaCr GUI::

   spacr
