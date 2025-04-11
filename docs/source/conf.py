import os, sys, types

# —— stub out the spacr package so its __init__.py never runs ——————
srcdir = os.path.abspath(os.path.join(__file__, '..', '..', 'spacr'))
sys.path.insert(0, srcdir)
_spacr_pkg = types.ModuleType('spacr')
_spacr_pkg.__path__ = [srcdir]
sys.modules['spacr'] = _spacr_pkg

# -- Project information -----------------------------------------------------
project = 'spacr'
author  = 'Einar Birnir Olafsson'
try:
    from importlib.metadata import version as _ver
except ImportError:
    from importlib_metadata import version as _ver
release = _ver('spacr')

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx_rtd_theme',
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
]

# mock out heavy deps so autodoc never actually loads them
autodoc_mock_imports = [
    'torch',
    'torchvision',
    'monai',
    'itk',
    'train_tools',
    'zarr',
]

# -- HTML output options -----------------------------------------------------
html_theme = 'sphinx_rtd_theme'
html_theme_options = {
    'logo_only': True,
    'collapse_navigation': False,
    'navigation_depth': 4,
    'style_nav_header_background': '#005f73',
}

templates_path   = ['_templates']
html_static_path = ['_static']
html_logo        = '_static/logo_spacr.png'
html_css_files   = ['custom.css']
