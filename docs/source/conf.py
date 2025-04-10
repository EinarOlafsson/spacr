# -- Project information -----------------------------------------------------
project = 'spacr'
author  = 'Einar Birnir Olafsson'
release = '0.4.60'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',    # core autodoc
    'sphinx.ext.napoleon',   # Google/NumPy docstrings
]

# Mock out heavy 3rd‑party modules so autodoc won't try to import them
autodoc_mock_imports = [
    'tables', 'h5py', 'wandb', 'huggingface_hub', 'openai',
    'tqdm', 'ipywidgets', 'ipykernel', 'screeninfo',
    'brokenaxes', 'gdown', 'rapidfuzz',
]

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
