# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'spacr'
copyright = '2025, Einar Olafsson'
author = 'Einar Olafsson'

version = '0.4.60'
release = '0.4.60'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.todo',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']

# -- Options for todo extension ----------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/todo.html#configuration

todo_include_todos = True

# -- Additional Sphinx Extensions --------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.todo',
    'sphinx_autodoc_typehints'
]

# -- Mock out big imports if you're having import failures --------------------
#   Remove items from this list if you want the real docstrings from them.
autodoc_mock_imports = [
    "numpy",
    "scipy",
    "seaborn",
    "pandas",   # if you use it
    "sklearn",  # if you use it
    # add more if needed
]

# -- Theme --------------------------------------------------------------------
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
