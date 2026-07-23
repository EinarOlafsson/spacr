"""Sphinx configuration for the spaCR API reference.

Theme: furo — auto-switches between light and dark modes based on
the reader's OS preference, has a clean sidebar for autoapi's per-
module tree, and renders reST docstrings without visual noise.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(
    os.path.join(__file__, '..', '..', 'spacr')
))

# -- Project metadata ------------------------------------------------------
project   = 'spaCR'
author    = 'Einar Birnir Olafsson'
copyright = f'2025-2026, {author}'

try:
    from importlib.metadata import version as _ver
except ImportError:
    from importlib_metadata import version as _ver
try:
    release = _ver('spacr')
except Exception:
    # Fall back to reading spacr/version.py when the editable install
    # didn't lay down .egg-info metadata.
    sys.path.insert(0, os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', '..')
    ))
    try:
        import spacr as _spacr_pkg
        release = getattr(_spacr_pkg, '__version__', '') or 'dev'
    except Exception:
        release = 'dev'
version = release

# -- General configuration -------------------------------------------------
extensions = [
    'sphinx.ext.napoleon',     # Google / NumPy / reST field lists
    'sphinx.ext.viewcode',     # [source] links on every symbol
    'sphinx.ext.intersphinx',  # cross-refs into stdlib / numpy / torch
    'sphinx_design',           # grid / card / tab directives on landing
    'autoapi.extension',       # AST-walk based auto reference
]

suppress_warnings = ['misc.section']
default_role = 'py:obj'

intersphinx_mapping = {
    'python':      ('https://docs.python.org/3',                       None),
    'numpy':       ('https://numpy.org/doc/stable',                    None),
    'pandas':      ('https://pandas.pydata.org/docs',                  None),
    'scipy':       ('https://docs.scipy.org/doc/scipy',                None),
    'sklearn':     ('https://scikit-learn.org/stable',                 None),
    'torch':       ('https://pytorch.org/docs/stable',                 None),
    'matplotlib':  ('https://matplotlib.org/stable',                   None),
}

# Napoleon (Google / NumPy → reST bridge) — spaCR uses reST field
# lists natively but napoleon stays on so any legacy Args/Returns
# still render cleanly.
napoleon_google_docstring = True
napoleon_numpy_docstring  = True
napoleon_include_init_with_doc = False
napoleon_use_rtype = False

# -- AutoAPI ---------------------------------------------------------------
autoapi_type              = 'python'
autoapi_dirs              = [os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'spacr')
)]
autoapi_root              = 'api'
autoapi_add_toctree_entry = True
autoapi_options           = [
    'members',
    'undoc-members',
    'show-inheritance',
    'show-module-summary',
    'special-members',
    'imported-members',
]
autoapi_ignore            = ['*/tests/*', '*/qt/tutorial/*']
autoapi_python_class_content = 'both'   # class docstring + __init__ docstring
autoapi_member_order         = 'groupwise'   # attrs → methods, alphabetical inside

# -- HTML output — furo ----------------------------------------------------
html_theme      = 'furo'
html_title      = 'spaCR'
html_logo       = '_static/logo_spacr.png'
html_favicon    = '_static/logo_spacr.png'
templates_path  = ['_templates']
html_static_path = ['_static']
html_css_files  = ['custom.css']

html_theme_options = {
    # Auto-switching light/dark, with a manual toggle in the top bar
    'light_css_variables': {
        'color-brand-primary':  '#4A9EFF',
        'color-brand-content':  '#4A9EFF',
        'color-admonition-background': '#eef4ff',
        'font-stack':           'Inter, ui-sans-serif, system-ui, -apple-system, sans-serif',
        'font-stack--monospace': 'JetBrains Mono, Consolas, monospace',
    },
    'dark_css_variables': {
        'color-brand-primary':  '#82b8ff',
        'color-brand-content':  '#82b8ff',
        'color-background-primary':   '#0d0d0d',
        'color-background-secondary': '#141414',
        'color-background-hover':     '#1c1c1c',
        'color-background-border':    '#262626',
        'color-foreground-primary':   '#e5e5e5',
        'color-foreground-secondary': '#c4c4c4',
        'color-admonition-background': '#141a24',
    },
    'sidebar_hide_name': False,
    'navigation_with_keys': True,
    'source_repository':  'https://github.com/EinarOlafsson/spacr/',
    'source_branch':      'main',
    'source_directory':   'spacr/',
    'footer_icons': [
        {
            'name': 'GitHub',
            'url':  'https://github.com/EinarOlafsson/spacr',
            'html': '',
            'class': 'fa-brands fa-solid fa-github fa-2x',
        },
    ],
}
