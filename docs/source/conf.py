import os
import sys
import sphinx_rtd_theme

# ------------------------------------------------------------------------------
# Make sure Python sees your package
# (Adjust as needed for your actual package structure)
# ------------------------------------------------------------------------------
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# ------------------------------------------------------------------------------
# Project information
# ------------------------------------------------------------------------------
project = 'spacr'
author  = 'Einar Birnir Olafsson'
try:
    from importlib.metadata import version as _ver
except ImportError:
    from importlib_metadata import version as _ver
release = _ver('spacr')  # e.g., "0.4.60"

# ------------------------------------------------------------------------------
# Sphinx Extensions
# ------------------------------------------------------------------------------
extensions = [
    'autoapi.extension',    # parse code via AST
    'sphinx.ext.viewcode',  # add “view source” links
    # 'sphinx.ext.autodoc', # optional if you also want autodoc usage
]

# ------------------------------------------------------------------------------
# AutoAPI Configuration
# ------------------------------------------------------------------------------
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

# ------------------------------------------------------------------------------
# Theme
# ------------------------------------------------------------------------------
html_theme = 'sphinx_rtd_theme'
# Modern Sphinx doesn’t require manually setting html_theme_path for RTD theme,
# but if you prefer to specify it:
# html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

html_theme_options = {
    'logo_only': False,
    'collapse_navigation': False,
    'navigation_depth': 4,
    'style_nav_header_background': '#005f73',
}

# ------------------------------------------------------------------------------
# Remove references to custom CSS and logo
# ------------------------------------------------------------------------------
templates_path   = []
html_static_path = []
# No html_logo or custom CSS
# html_logo        = '_static/logo_spacr.png'
# html_css_files   = ['custom.css']
