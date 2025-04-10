import os
import sys
# make sure Sphinx can import your package
sys.path.insert(0, os.path.abspath('../../spacr'))

# -- Project information -----------------------------------------------------
project = 'spacr'
copyright = '2025, Einar Birnir Olafsson'
author = 'Einar Birnir Olafsson'
release = '0.4.60'

# -- General configuration ---------------------------------------------------
# Add any Sphinx extension module names here, as strings.
extensions = [
    'sphinx.ext.autodoc',    # core autodoc functionality
    'sphinx.ext.napoleon',   # for Google- and NumPy‑style docstrings
]

# Mock out heavy/compiled/3rd‑party modules so autodoc won't try to import them
autodoc_mock_imports = [
    'tables',
    'h5py',
    'wandb',
    'huggingface_hub',
    'openai',
    'tqdm',
    'ipywidgets',
    'ipykernel',
    'screeninfo',
    'brokenaxes',
    'gdown',
    'rapidfuzz',
    # add any other modules your code imports but you don't need to execute
]

templates_path = ['_templates']
exclude_patterns = []

# If your source files aren’t UTF‑8, change this:
# source_encoding = 'utf-8-sig'

# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# -- Napoleon settings (optional) --------------------------------------------
# If you use Google or NumPy style docstrings and want to tweak Napoleon:
# napoleon_google_docstring = True
# napoleon_numpy_docstring  = True
# napoleon_include_init_with_doc = False
