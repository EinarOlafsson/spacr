[![PyPI version](https://badge.fury.io/py/spacr.svg)](https://badge.fury.io/py/spacr)
[![Python version](https://img.shields.io/pypi/pyversions/spacr)](https://pypistats.org/packages/spacr)
[![Licence: GPL v3](https://img.shields.io/github/license/EinarOlafsson/spacr)](https://github.com/EinarOlafsson/spacr/blob/master/LICENSE)
[![repo size](https://img.shields.io/github/repo-size/EinarOlafsson/spacr)](https://github.com/EinarOlafsson/spacr/)

# SpaCr
<table>
<tr>
<td>
  
Spatial phenotype analysis of CRISPR-Cas9 screens (SpaCr). The spatial organization of organelles and proteins within cells constitutes a key level of functional regulation. In the context of infectious disease, the spatial relationships between host cell structures and intracellular pathogens are critical to understand host clearance mechanisms and how pathogens evade them. Spacr is a Python-based software package for generating single cell image data for deep-learning sub-cellular/cellular phenotypic classification from pooled genetic CRISPR-Cas9 screens. Spacr provides a flexible toolset to extract single cell images and measurements from high content cell painting experiments, train deep-learning models to classify cellular/ subcellular phenotypes, simulate and analyze pooled CRISPR-Cas9 imaging screens.

</td>
<td>

<img src="spacr/logo_spacr.png" alt="SPACR Logo" title="SPACR Logo" width="600"/>

</td>
</tr>
</table>

## Features

- **Generate Masks:** Generate cellpose masks of cell, nuclei and pathogen objects.

- **Object Measurements:** Measurements for each object including scikit-image-regionprops, intensity percentiles, shannon-entropy, pearsons and manders correlations, homogenicity and radial distribution. Measurements are saved to a sql database in object level tables.

- **Crop Images:** Objects (e.g. cells) can be saved as PNGs from the object area or bounding box area of each object. Object paths are saved in an sql database that can be annotated and used to train CNNs/Transformer models for classefication tasks.

- **Train CNNs or Transformers:** Train Torch Convolutional Neural Networks (CNNs) or Transformers to classify single object images. Train Torch models with IRM/ERM, checkpointing.

- **Manual Annotation:** Supports manual annotation of single cell images and segmentation to refine training datasets for training CNNs/Transformers or cellpose, respectively.

- **Finetune Cellpose Models:** Adjust pre-existing Cellpose models to your specific dataset for improved performance.

- **Timelapse Data Support:** Track objects in timelapse image data.

- **Simulations:** Simulate spatial phenotype screens.

- **Sequencing:** Map FASTQ reads to barecode and gRNA barecode metadata.

- **Misc:** Analyze Ca oscillation, recruitment, infection rate, plaque size/count.

## Installation

Requires Tkinter for graphical user interface features.

### Ubuntu

Before installing spacr, ensure Tkinter is installed:

(Tkinter is included with the standard Python installation on macOS, and Windows)

On Linux:

```
sudo apt-get install python3-tk
```

install spacr with pip

```
pip install spacr
```

Run spacr GUI:

```
gui
```