# SpaCr
Spatial phenotype analysis of crisp screens (SpaCr). A collection of functions for generating measurement and classification data from microscopy images from high content imaging screens.

## Features

- **Generate Masks:** Generate cellpose masks from up to 3 object classes (e.g. cells, nuclei, pathogen).
- **Measurements:** Collect object measurement data and save to sql database.
- **Crop object images:** Crop object (e.g. single cell) PNG images for downstream training/classefication with CNNs/Transformers.
- **Train CNNs/Transformers:** Train PyTorch Convolutional Neural Networks (CNNs) or Transformers to classify cropped PNGs.
- **Manual Annotation:** Manually annotate cropped PNGs and manually curate segmentation masks to generate training datasets for CNNs/Transformers or Cellpose, respectively.
- **Finetune Cellpose Models:** Fine-tune pre-existing Cellpose models to your specific dataset for improved performance.
- **Models:** Use our fine-tuned models for segmentation of Toxoplasma gondii parasitopherous vacuole and plaques.
- **Timelapse Data Support:** Includes support for analyzing timelapse data.
- **Simulations:** Simulate pooled spatial phenotype screens to determine optimal paramiters in future screens. 

## Installation

SpaCr requires Tkinter for its graphical user interface features.

### Ubuntu

Before installing SpaCr, ensure Tkinter is installed:

Microsoft Visual C++ 14.0 or greater required.

(Tkinter is included with the standard Python installation on macOS, and Windows)

On Linux:

```bash
sudo apt-get install python3-tk