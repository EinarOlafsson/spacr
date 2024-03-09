# SpaCr
Spatial phenotype analysis of crisp screens (SpaCr). A collection of functions for generating measurement and classification data from microscopy images from high content imaging screens.

## Features

- **Generate Masks:** From cells' nuclei and pathogen images.
- **Collect Measurements & Crop Images:** Automate the collection of measurements and crop single cell images for further analysis.
- **Train CNNs or Transformers:** Utilize PyTorch to train Convolutional Neural Networks (CNNs) or Transformers for classifying single cell images.
- **Manual Annotation:** Supports manual annotation of single cell images and segmentation to refine training datasets for training CNNs/Transformers or cellpose, respectively.
- **Finetune Cellpose Models:** Adjust pre-existing Cellpose models to your specific dataset for improved performance.
- **Timelapse Data Support:** Includes support for analyzing timelapse data.
- **Simulations:** Simulate spatial phenotype screens.

## Installation

SpaCr requires Tkinter for its graphical user interface features.

### Ubuntu

Before installing SpaCr, ensure Tkinter is installed:

(Tkinter is included with the standard Python installation on macOS, and Windows)

On Linux:

```bash
sudo apt-get install python3-tk