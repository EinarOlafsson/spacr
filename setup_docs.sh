#!/bin/bash

# Remove the existing docs folder
rm -rf docs

# Create the documentation structure with Sphinx quickstart
sphinx-quickstart docs -q -p spacr -a "Einar Birnir Olafsson" --ext-autodoc --ext-viewcode --release 0.0.70 --makefile --no-batchfile

# Navigate to the docs directory
cd docs

# Create a requirements file for the documentation
cat <<EOT >> requirements.txt
sphinx
sphinx_rtd_theme
EOT

# Create the conf.py file
cat <<EOT > source/conf.py
import os
import sys
sys.path.insert(0, os.path.abspath('../../spacr'))

project = 'spacr'
author = 'Einar Birnir Olafsson'
release = '0.0.70'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx_rtd_theme'
]

autodoc_mock_imports = ["torch", "cv2", "pandas", "shap", "skimage", "scipy", "matplotlib", "numpy", "tifffile", "fastremap", "natsort", "numba"]

templates_path = ['_templates']
exclude_patterns = []

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
EOT

# Create the index.rst file
cat <<EOT > source/index.rst
.. spacr documentation master file, created by
   sphinx-quickstart on Thu Oct 14 12:34:56 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root \`toctree\` directive.

Welcome to spacr's documentation!
=================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   modules

Indices and tables
==================

* :ref:\`genindex\`
* :ref:\`modindex\`
* :ref:\`search\`
EOT

# Generate the API documentation
sphinx-apidoc -o source ../spacr

# Build the HTML documentation
make html

# Create the .readthedocs.yaml file
cat <<EOT > ../.readthedocs.yaml
version: 2

sphinx:
  configuration: docs/source/conf.py

python:
  version: 3.9
  install:
    - requirements: docs/requirements.txt
EOT

# Navigate back to the root directory
cd ..

# Add the new documentation to git, commit and push
git add .
git commit -m "Add Sphinx documentation"
git push origin main

echo "Setup complete. Push the changes to your repository and configure Read the Docs."
