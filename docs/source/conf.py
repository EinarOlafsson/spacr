import os
import sys

# 1) Add your docs folder so Sphinx can load _static, index.rst, etc.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# -- Project information -----------------------------------------------------
project = 'spacr'
author  = 'Einar Birnir Olafsson'
try:
    from importlib.metadata import version as _ver
except ImportError:
    from importlib_metadata import version as _ver
release = _ver('spacr')

# -- General configuration ---------------------------------------------------
extensions = [
    'autoapi.extension',     # <<<<< AutoAPI
    'sphinx.ext.viewcode',   # view-source links
]

# AutoAPI settings
autoapi_type            = 'python'
autoapi_dirs            = [os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                         '..','..','spacr'))]
autoapi_root            = 'api'
autoapi_add_toctree_entry = True
autoapi_options         = [
    'members',
    'undoc-members',
    'show-inheritance',
]
# You can ignore tests or other unwanted files:
autoapi_ignore          = ['*/tests/*']

# -- Options for HTML output -------------------------------------------------
html_theme           = 'sphinx_rtd_theme'
html_theme_options   = {
    'collapse_navigation': False,
    'navigation_depth':    4,
}
html_static_path     = ['_static']
html_logo            = '_static/logo_spacr.png'
