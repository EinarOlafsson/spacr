# docs/source/conf.py

import os
import sys
import re

# 1) Tell Sphinx where to find your spacr package
#    docs/source → ../../spacr
sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', '..', 'spacr')
    )
)

# 2) Tell Sphinx where to find deps_list.py
#    docs/source → ../ (which is docs/)
sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..')
    )
)

# 3) Import your raw dependency lists (pure data, no side‑effects)
from deps_list import dependencies, extra_gui

def strip_version_specifiers(deps):
    """
    Strip off everything from the first '<' or '>' (inclusive) in each string.
    """
    cleaned = []
    for dep in deps:
        # find earliest '<' or '>' and cut there
        idxs = [dep.find(c) for c in ('<', '>') if c in dep]
        if idxs:
            dep = dep[:min(idxs)]
        cleaned.append(dep.strip())
    return cleaned

# 4) Build the list of module names to mock
base_names = strip_version_specifiers(dependencies + extra_gui)
mods = [name.replace('-', '_') for name in base_names]

# 5) Special‑case imports whose name differs from the PyPI package
overrides = {
    'scikit_image': 'skimage',
    'opencv_python_headless': 'cv2',
    'biopython': 'Bio',
    # add more overrides here if needed
}
for pkg, mod in overrides.items():
    if pkg in mods:
        mods[mods.index(pkg)] = mod

# -- Project information -----------------------------------------------------
# You can also pull VERSION from importlib.metadata if you like
project = 'spacr'
author  = 'Einar Birnir Olafsson'
try:
    from importlib.metadata import version as _ver
except ImportError:
    from importlib_metadata import version as _ver
release = _ver('spacr')

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.autosummary',   # generate summary tables + stub pages
    'sphinx.ext.viewcode',      # add “view source” links
 ]

# Automatically generate the stub .rst files for autosummary directives
autosummary_generate = True
autosummary_imported_members = False

# This tells Sphinx to mock these modules instead of trying to import them
autodoc_mock_imports = mods

# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
