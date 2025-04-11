# docs/source/conf.py

import os
import sys
import sphinx_rtd_theme

# -------------------------------------------------------------------
# Tell Sphinx to use the RTD theme and where to find it
# -------------------------------------------------------------------
html_theme      = 'sphinx_rtd_theme'
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

# -------------------------------------------------------------------
# Make docs/ importable so AutoAPI and static assets work
# -------------------------------------------------------------------
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# -------------------------------------------------------------------
# Project information
# -------------------------------------------------------------------
project = 'spacr'
author  = 'Einar Birnir Olafsson'
try:
    from importlib.metadata import version as _ver
except ImportError:
    from importlib_metadata import version as _ver
release = _ver('spacr')

# -------------------------------------------------------------------
# Sphinx extensions
# -------------------------------------------------------------------
extensions = [
    'autoapi.extension',    # parse your code via AST (no imports)
    'sphinx.ext.viewcode',  # add “view source” links
]

# -------------------------------------------------------------------
# AutoAPI configuration
# -------------------------------------------------------------------
autoapi_type              = 'python'
autoapi_dirs              = [
    os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'spacr'))
]
autoapi_root              = 'api'
autoapi_add_toctree_entry = True
autoapi_options           = [
    'members',
    'undoc-members',
    'show-inheritance',
]
autoapi_ignore            = ['*/tests/*']

# -------------------------------------------------------------------
# HTML output options
# -------------------------------------------------------------------
# Theme customizations
html_theme_options = {
    'logo_only': True,
    'collapse_navigation': False,
    'navigation_depth': 4,
    'style_nav_header_background': '#005f73',
    # 'display_version': True,  # RTD theme no longer supports this option
}

# Paths for custom static files (logo, CSS, etc.)
templates_path   = ['_templates']
html_static_path = ['_static']
html_logo        = '_static/logo_spacr.png'
# html_favicon     = '_static/favicon.ico'  # uncomment if you have one

# Custom CSS file
html_css_files   = ['custom.css']
