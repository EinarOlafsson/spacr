import os, sys
import sphinx_rtd_theme

# -- Path setup --------------------------------------------------------------
# (not used for imports, but needed for viewcode linking)
sys.path.insert(0, os.path.abspath(os.path.join(__file__, '..', '..', 'spacr')))

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
    'sphinx.ext.napoleon',     # for Google/NumPy style docstrings
    'sphinx.ext.viewcode',     # link to source
    'autoapi.extension',       # parse your code via AST
]

# suppress “Missing matching underline for section title overline” errors
suppress_warnings = ['misc.section']

# -- AutoAPI settings --------------------------------------------------------
autoapi_type               = 'python'
autoapi_dirs               = [os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'spacr')
)]
autoapi_root               = 'api'
autoapi_add_toctree_entry  = True
autoapi_options = [
    'members',
    'undoc-members',
    'show-inheritance',
    'special-members',
]
autoapi_python_class_content = 'both'  # Only show class-level docs, skip module variables
autoapi_keep_files = True

autoapi_ignore = [
    '*/tests/*',
    '*/spacr/__main__.py',
    '*/spacr/__init__.py',
    '*/spacr/app_classify.py',
    '*/spacr/app_mask.py',
    '*/spacr/app_measure.py',
    '*/spacr/app_sequencing.py',
    '*/spacr/version.py',
    '*/spacr/logger.py',
    '*/spacr/logger.py',
    '*/spacr/timelapse.py',
]

# -- Options for HTML output -------------------------------------------------
html_theme      = 'sphinx_rtd_theme'
#html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
html_theme_options = {
    'logo_only': True,
    'collapse_navigation': False,
    'navigation_depth': 4,
    'style_nav_header_background': '#005f73',
}

templates_path   = ['_templates']
html_static_path = ['_static']
html_logo        = '_static/logo_spacr.png'
html_css_files   = ['custom.css']
