import sphinx_rtd_theme

# Configuration file for the Sphinx documentation builder.

# -- Project information -----------------------------------------------------
project = 'spacr'
author = 'Einar Birnir Olafsson'
release = '0.0.70'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
]

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_rtd_theme'
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
html_static_path = ['_static']

# Specify the master document
master_doc = 'index'

