#!/bin/bash

# Install necessary Sphinx extensions
pip install sphinx sphinx_rtd_theme sphinx-autodoc-typehints

# Remove existing docs directory if it exists
rm -rf docs

# Create the docs directory and source subdirectory
mkdir -p docs/source
cd docs

# Create docs/requirements.txt before initializing Sphinx
cat <<EOL > requirements.txt
sphinx
sphinx_rtd_theme
sphinx-autodoc-typehints
EOL

# Create the .readthedocs.yaml configuration file
cat <<EOL > ../.readthedocs.yaml
version: 2

build:
  os: ubuntu-22.04
  tools:
    python: "3.9"

sphinx:
  configuration: docs/source/conf.py

python:
  install:
    - requirements: docs/requirements.txt
    - method: pip
      path: .
EOL

# Get the version of spacr from the version.py file
VERSION=$(python -c "from spacr.version import version; print(version)")

# Run sphinx-quickstart with necessary configurations in the source directory
sphinx-quickstart source -q -p "spacr" -a "Einar Olafsson" -v "$VERSION" --ext-autodoc --ext-viewcode --ext-todo

# Update conf.py with additional configurations
cat <<EOT >> source/conf.py

# Add any Sphinx extension module names here, as strings. They can be extensions
# coming with Sphinx (named 'sphinx.ext.*') or your custom ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.todo',
    'sphinx_autodoc_typehints'
]

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
EOT

# Create an index.rst file with basic content
cat <<EOT > source/index.rst
.. spacr documentation master file, created by
   sphinx-quickstart on $(date).

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

# Run sphinx-apidoc to generate the reStructuredText files
sphinx-apidoc -o source/ ../spacr

# Build the HTML documentation
make -C source html

# Commit the changes to git
cd ..
git add docs .readthedocs.yaml
git commit -m "Set up Sphinx documentation with Read the Docs"
git push origin main

echo "Setup complete. Push the changes to your repository and configure Read the Docs."
echo "spacr version: $VERSION"

# Instructions to build the documentation on Read the Docs
# chmod +x setup_docs.sh make the script executable.
# Run the script with ./setup_docs.sh
# 1. Go to https://readthedocs.org/ and log in with GitHub account.
# 2. Click on 'Import a Project'.
# 3. Select your repository and click 'Next'.
