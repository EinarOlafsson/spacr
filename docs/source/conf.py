import sphinx_rtd_theme, os, re

# Configuration file for the Sphinx documentation builder.

# -- Project information -----------------------------------------------------
project = 'spacr'
author = 'Einar Birnir Olafsson'
release = '0.0.70'
copyright = '2024, Einar Birnir Olafsson'

# The full version, including alpha/beta/rc tags
release = re.sub('^v', '', os.popen('git describe --tags').read().strip())

# The short X.Y version.
version = release

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
html_theme_path = [
    "_themes",
]
html_theme_options = {
    'canonical_url': '',
    'analytics_id': 'UA-XXXXXXX-1',  #  Provided by Google in your dashboard
    'logo_only': False,
    'display_version': True,
    'prev_next_buttons_location': 'top',
    'style_external_links': False,
    'style_nav_header_background': 'black',
    # Toc options
    'collapse_navigation': True,
    'sticky_navigation': True,
    'navigation_depth': 4,
    'includehidden': True,
    'titles_only': False
}
#html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
html_static_path = ['_static']

# Specify the master document
master_doc = 'index'

