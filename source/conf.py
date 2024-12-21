# Configuration file for the Sphinx documentation builder.

import os
import sys
import datetime

# The full version, including alpha/beta/rc tags.
from spacr.version import version as release

# -- Path setup --------------------------------------------------------------

# If your module (spacr) is one level above your docs folder, do:
sys.path.insert(0, os.path.abspath('../../'))

# -- Project information -----------------------------------------------------

project = 'spacr'
author = 'Einar Birnir Olafsson'
copyright = f"{datetime.date.today().year}, {author}"

# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.todo',
    'sphinx_autodoc_typehints'
]

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------

html_theme = 'sphinx_rtd_theme'

# Enable todo directives if you want them
todo_include_todos = True
