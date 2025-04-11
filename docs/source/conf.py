import os, sys, types, importlib.machinery, importlib.util

# — locate your real sp‍acr package root —
HERE     = os.path.dirname(__file__)
PKG_ROOT = os.path.abspath(os.path.join(HERE, '..', '..', 'spacr'))

# — stub torch so any `import torch` just gives an empty module —
sys.modules['torch'] = types.ModuleType('torch')

# — stub the top‑level `spacr` package so its __init__.py never runs —
pkg = types.ModuleType('spacr')
pkg.__path__ = [PKG_ROOT]
sys.modules['spacr'] = pkg

# — manually load only core.py as sp‍acr.core —
core_path = os.path.join(PKG_ROOT, 'core.py')
loader    = importlib.machinery.SourceFileLoader('spacr.core', core_path)
spec      = importlib.util.spec_from_loader(loader.name, loader)
mod       = importlib.util.module_from_spec(spec)
sys.modules['spacr.core'] = mod
loader.exec_module(mod)

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
# no need for autodoc_mock_imports now

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
