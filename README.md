# SpaCr

SpaCr is a Python package designed for morphological processing, specifically tailored for cellular analysis. It offers a comprehensive toolset for generating masks from cells' nuclei and pathogens, collecting measurements, cropping single cell images, and much more.

## Features

- **Generate Masks:** From cells' nuclei and pathogen images, enabling detailed morphological analysis.
- **Collect Measurements & Crop Images:** Automate the collection of measurements and crop single cell images for further analysis.
- **Train CNNs or Transformers:** Utilize PyTorch to train Convolutional Neural Networks (CNNs) or Transformers for classifying single cell images.
- **Manual Annotation:** Supports manual annotation of single cell images to refine training datasets or for detailed study.
- **Finetune Cellpose Models:** Adjust pre-existing Cellpose models to your specific dataset for improved performance.
- **Timelapse Data Support:** Includes support for analyzing timelapse data.

## Installation

SpaCr requires Tkinter for its graphical user interface features.

### Ubuntu

Before installing SpaCr, ensure Tkinter is installed:

(Tkinter is included with the standard Python installation on macOS, and Windows)

On Linux:

```bash
sudo apt-get install python3-tk