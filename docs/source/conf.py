# docs/source/conf.py
import sys, os
sys.path.insert(0, os.path.abspath('../../spacr'))

project = 'spacr'
author  = 'Einar Birnir Olafsson'
release = '0.4.60'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
]

autodoc_mock_imports = [
    'tables','h5py','wandb','huggingface_hub','openai',
    'tqdm','ipywidgets','ipykernel','screeninfo',
    'brokenaxes','gdown','rapidfuzz',
]

html_theme = 'sphinx_rtd_theme'
