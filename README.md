[![PyPI version](https://badge.fury.io/py/spacr.svg)](https://badge.fury.io/py/spacr)
[![Python version](https://img.shields.io/pypi/pyversions/spacr)](https://pypistats.org/packages/spacr)
[![Licence: GPL v3](https://img.shields.io/github/license/EinarOlafsson/spacr)](https://github.com/EinarOlafsson/spacr/blob/master/LICENSE)
[![repo size](https://img.shields.io/github/repo-size/EinarOlafsson/spacr)](https://github.com/EinarOlafsson/spacr/)

# SpaCr
<table>
<tr>
<td>
  
Spatial phenotype analysis of crisp screens (SpaCr). A collection of functions for generating cellpose masks -> single object images and measurements -> annotation and classification of single object images. Spacr uses batch normalization to facilitate accurate segmentation of objects with low foreground representation.

</td>
<td>

<img src="spacr/logo_spacr.png" alt="SPACR Logo" title="SPACR Logo" width="600"/>

</td>
</tr>
</table>

## Features

- **Generate Masks:** Generate cellpose masks for cells, nuclei and pathogen images.

- **Object Measurements:** Measurements for each object including scikit-image-regionprops, intensity quantiles, shannon-entropy, pearsons and manders correlation, homogenicity and radial distribution. Measurements are saved to a sql database in object level tables.

- **Crop Images:** Objects (e.g. cells) can be saved as PNGs from the object area or bounding box area of each object. Object paths are saved in an sql database that can be annotated and used to train CNNs/Transformer models for classefication tasks.

- **Train CNNs or Transformers:** Train Torch Convolutional Neural Networks (CNNs) or Transformers to classify single object images. Train Torch models with IRM/ERM, checkpointing, 

- **Manual Annotation:** Supports manual annotation of single cell images and segmentation to refine training datasets for training CNNs/Transformers or cellpose, respectively.

- **Finetune Cellpose Models:** Adjust pre-existing Cellpose models to your specific dataset for improved performance.

- **Timelapse Data Support:** Includes support for analyzing timelapse data.

- **Simulations:** Simulate spatial phenotype screens.

## Installation

spacr requires Tkinter for its graphical user interface features.

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

To run spacr GUIs after installing spacr:

To generate masks:
```
gui_mask
```
To generate measurements and cropped images:
```
gui_measure
```
To curate masks for finetuning cellpose models:
```
gui_make_masks
```
To annotate paths to images in sql database created in gui_measure:
```
gui_annotate
```
Train torch CNNs/Transformers to classify single object images.
```
gui_classify
```
Simulate spatial phenotype screens.
```
gui_sim
```