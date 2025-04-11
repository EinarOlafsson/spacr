# docs/source/conf.py

import os
import sys
import sphinx_rtd_theme

# 1) Make docs/ importable so AutoAPI and static paths work
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
    'autoapi.extension',    # parse your code via AST
    'sphinx.ext.viewcode',  # add “view source” links
]

# AutoAPI settings
autoapi_type               = 'python'
autoapi_dirs               = [os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'spacr')
)]
autoapi_root               = 'api'
autoapi_add_toctree_entry  = True
autoapi_options            = [
    'members',
    'undoc-members',
    'show-inheritance',
]
autoapi_ignore             = ['*/tests/*']

# -- Options for HTML output -------------------------------------------------
html_theme      = 'sphinx_rtd_theme'
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

html_theme_options = {
    'logo_only': True,
    'collapse_navigation': False,
    'navigation_depth': 4,
    'style_nav_header_background': '#005f73',
    'display_version': True,
}

# static assets
html_static_path = ['_static']
html_logo        = '_static/logo_spacr.png'
html_favicon     = '_static/favicon.ico'  # if you have a favicon

# load custom CSS
html_css_files = [
    'custom.css',
]


# -- (Optional) Custom CSS ---------------------------------------------------
# Create docs/source/_static/custom.css with styles like:
#
#   body {
#     font-size: 1.1em;
#     line-height: 1.6;
#   }
#   .wy-nav-side {
#     background-color: #f7f7f7;
#   }
#   .highlight {
#     background: #fafafa;
#     border: 1px solid #e0e0e0;
#     padding: 0.5em;
#     border-radius: 4px;
#   }
