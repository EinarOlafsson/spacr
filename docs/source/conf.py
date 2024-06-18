import os
import sys
sys.path.insert(0, os.path.abspath('../../spacr'))

# -- Project information -----------------------------------------------------
project = 'spacr'
author = 'Einar Birnir Olafsson'
release = '0.0.70'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx_rtd_theme'
]

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
