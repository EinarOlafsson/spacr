# docs/source/conf.py

import os
import sys
import sphinx_rtd_theme

# -- Path setup --------------------------------------------------------------
# so Sphinx can import your package and pick up static files
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'spacr')))

# -- Project information -----------------------------------------------------
project = 'spacr'
author  = 'Einar Birnir Olafsson'
try:
    from importlib.metadata import version as _ver
except ImportError:
    from importlib_metadata import version as _ver
release = _ver('spacr')

# -- Extensions --------------------------------------------------------------
extensions = [
    'sphinx_rtd_theme',    # register the RTD theme
    'sphinx.ext.autodoc',  # for automodule
    'sphinx.ext.napoleon', # for Google/NumPy docstrings
    'sphinx.ext.viewcode', # “view source” links
]

# avoid trying to import torch when autodoc runs
autodoc_mock_imports = ['torch']

# -- HTML theme --------------------------------------------------------------
html_theme = 'sphinx_rtd_theme'
# no need for html_theme_path if sphinx_rtd_theme is in extensions

html_theme_options = {
    'logo_only': True,
    'collapse_navigation': False,
    'navigation_depth': 4,
    'style_nav_header_background': '#005f73',
}

# -- Static files & CSS -----------------------------------------------------
templates_path   = ['_templates']
html_static_path = ['_static']
html_logo        = '_static/logo_spacr.png'
# html_favicon     = '_static/favicon.ico'  # if you have one

html_css_files   = ['custom.css']
