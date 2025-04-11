# docs/source/conf.py

import os, sys, re

# 1) Make sure Sphinx can import your package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'spacr')))

# 2) Make sure Sphinx can import deps_list.py
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 3) Pull in the raw dependency lists
from deps_list import dependencies, extra_gui

def strip_version_specifiers(deps):
    cleaned = []
    for dep in deps:
        idxs = [dep.find(c) for c in ('<', '>') if c in dep]
        if idxs:
            dep = dep[:min(idxs)]
        cleaned.append(dep.strip())
    return cleaned

# 4) Build and normalize mock list
base_names = strip_version_specifiers(dependencies + extra_gui)
mods = [name.replace('-', '_') for name in base_names]
overrides = {
    'scikit_posthocs': 'scikit_posthocs',
    'scikit_learn': 'sklearn',
    'scikit_image': 'skimage',
    'opencv_python_headless': 'cv2',
    'biopython': 'Bio',
    'pillow': 'PIL',
    'huggingface_hub': 'huggingface_hub',
}
for pkg, mod in overrides.items():
    if pkg in mods:
        mods[mods.index(pkg)] = mod

# -- Project information -----------------------------------------------------
project   = 'spacr'
author    = 'Einar Birnir Olafsson'
try:
    from importlib.metadata import version as _ver
except ImportError:
    from importlib_metadata import version as _ver
release   = _ver('spacr')

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.autosummary',
    'sphinx.ext.viewcode',
]

autosummary_generate         = True
autosummary_imported_members = False

autodoc_mock_imports = mods

templates_path   = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
html_theme           = 'sphinx_rtd_theme'
html_theme_options   = {
    'collapse_navigation': False,
    'navigation_depth':    4,
    'style_nav_header_background': '#2980B9',
}
html_logo        = '_static/logo_spacr.png'
html_static_path = ['_static']
