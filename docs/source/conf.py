# docs/source/conf.py
import os
import sys
import re

# 0) Let Sphinx import your package (installed via --no-deps)
sys.path.insert(0, os.path.abspath('../../spacr'))

# 1) Pull in your raw lists (no side-effects)
from docs.deps_list import dependencies, extra_gui

# 2) Strip off version pins
def strip_version_specifiers(deps):
    cleaned = []
    for dep in deps:
        # cut at first '<' or '>' if present
        idxs = [dep.find(c) for c in ('<', '>') if c in dep]
        if idxs:
            dep = dep[:min(idxs)]
        cleaned.append(dep.strip())
    return cleaned

base_names = strip_version_specifiers(dependencies + extra_gui)

# 3) Normalize hyphens to underscores, plus special overrides
mods = [name.replace('-', '_') for name in base_names]

# special cases where import name ≠ package name
overrides = {
    'scikit_image': 'skimage',
    'opencv_python_headless': 'cv2',
    'biopython': 'Bio',
    'huggingface_hub': 'huggingface_hub',  # unchanged, but you get the idea
}
for pkg, mod in overrides.items():
    if pkg in mods:
        mods[mods.index(pkg)] = mod

# -- Project information -----------------------------------------------------
# You can also grab VERSION from importlib.metadata if you installed your package:
try:
    from importlib.metadata import version as _ver
except ImportError:
    from importlib_metadata import version as _ver

project = 'spacr'
author  = 'Einar Birnir Olafsson'
release = _ver('spacr')  # e.g. "0.4.60"

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
]

# Tell Sphinx to mock these imports so it never tries to load them
autodoc_mock_imports = mods

# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
