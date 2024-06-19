import os
import sys
sys.path.insert(0, os.path.abspath('../../spacr'))

project = 'spacr'
author = 'Einar Birnir Olafsson'
release = '0.0.70'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx_rtd_theme'
]

autodoc_mock_imports = ["torch", "cv2", "pandas", "shap", "skimage", "scipy", "matplotlib", "numpy", "tifffile", "fastremap", "natsort", "numba"]

templates_path = ['_templates']
exclude_patterns = []

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
