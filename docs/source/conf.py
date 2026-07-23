import os, sys
import sphinx_rtd_theme

sys.path.insert(0, os.path.abspath(os.path.join(__file__, '..', '..', 'spacr')))

project = 'spacr'
author  = 'Einar Birnir Olafsson'
try:
    from importlib.metadata import version as _ver
except ImportError:
    from importlib_metadata import version as _ver
try:
    release = _ver('spacr')
except Exception:
    # importlib.metadata sometimes returns PackageNotFoundError on
    # editable installs where the .egg-info wasn't laid down (some
    # pip / setuptools combos in CI). Fall back to importing spacr
    # directly — release only affects the HTML footer, so a soft
    # fallback is fine.
    sys.path.insert(0, os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', '..')
    ))
    try:
        import spacr as _spacr_pkg
        release = getattr(_spacr_pkg, '__version__', '') or 'dev'
    except Exception:
        release = 'dev'

extensions = [
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'autoapi.extension',
]

suppress_warnings = ['misc.section']

autoapi_type              = 'python'
autoapi_dirs              = [os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'spacr')
)]
autoapi_root              = 'api'
autoapi_add_toctree_entry = True
autoapi_options           = ['members', 'undoc-members', 'show-inheritance']
autoapi_ignore            = ['*/tests/*']

html_theme       = 'sphinx_rtd_theme'
templates_path   = ['_templates']
html_static_path = ['_static']
html_logo        = '_static/logo_spacr.png'
html_css_files   = ['custom.css']
