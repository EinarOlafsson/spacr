import os
import sys
sys.path.insert(0, os.path.abspath('../..'))

project = 'spacr'
author = 'Einar Birnir Olafsson'
release = '0.0.70'
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode'
]
html_theme = 'sphinx_rtd_theme'