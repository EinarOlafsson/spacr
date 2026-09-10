#!/usr/bin/env bash
set -euo pipefail

# ——— 0) Your custom landing‑page blurb —————————————————————————————
DESCRIPTION="SpaCr (Spatial phenotype analysis of CRISPR screens) is a Python toolkit for quantifying and visualizing phenotypic changes in high‑throughput imaging assays."

# ——— 1) Wipe only generated artifacts ——————————————————————————
rm -rf docs/_build docs/api

# ——— 2) Create the Sphinx source tree ——————————————————————————
mkdir -p docs/source/_static
#touch docs/.nojekyll   # prevent Jekyll from stripping _static
rm -rf docs/source/api/


# ——— 3) Write docs/source/conf.py —————————————————————————————
cat > docs/source/conf.py << 'EOF'
import os, sys
import sphinx_rtd_theme

# -- Path setup --------------------------------------------------------------
# (not used for imports, but needed for viewcode linking)
sys.path.insert(0, os.path.abspath(os.path.join(__file__, '..', '..', 'spacr')))

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
    'sphinx.ext.napoleon',     # for Google/NumPy style docstrings
    'sphinx.ext.viewcode',     # link to source
    'autoapi.extension',       # parse your code via AST
]

# suppress “Missing matching underline for section title overline” errors
suppress_warnings = ['misc.section']

# -- AutoAPI settings --------------------------------------------------------
autoapi_type               = 'python'
autoapi_dirs               = [os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'spacr')
)]
autoapi_root               = 'api'
autoapi_add_toctree_entry  = True
autoapi_options            = [
    'members',
    'undoc-members',
    'show-inheritance',
]
autoapi_ignore             = ['*/tests/*']

# -- Options for HTML output -------------------------------------------------
html_theme      = 'sphinx_rtd_theme'
#html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
html_theme_options = {
    'logo_only': True,
    'collapse_navigation': False,
    'navigation_depth': 4,
    'style_nav_header_background': '#005f73',
}

templates_path   = ['_templates']
html_static_path = ['_static']
html_logo        = '_static/logo_spacr_docs.png'
html_css_files   = ['custom.css']
EOF

# ——— 4) Write docs/source/index.rst —————————————————————————————
cat > docs/source/index.rst << EOF
Welcome to SpaCr
================

.. image:: _static/logo_spacr_docs.png
   :align: center
   :alt: SpaCr Logo

$DESCRIPTION

.. toctree::
   :maxdepth: 1

   api/index
EOF

# ——— 5) Copy your logo into _static — adjust path if needed —————————
LOGO_SRC="spacr/resources/icons/logo_spacr.png"
if [[ -f "$LOGO_SRC" ]]; then
  cp "$LOGO_SRC" docs/source/_static/logo_spacr.png
else
  echo "⚠️  Warning: logo not found at $LOGO_SRC"
fi

# ——— 6) Write a tiny custom.css —————————————————————————————
cat > docs/source/_static/custom.css << 'EOF'
/* custom.css */
body {
  font-size: 1.1em;
  line-height: 1.6;
}
.wy-nav-side {
  background-color: #f7f7f7;
}
.highlight {
  background: #fafafa;
  border: 1px solid #e0e0e0;
  padding: 0.5em;
  border-radius: 4px;
}
EOF

# ——— 7) Install Sphinx, RTD theme & AutoAPI if missing —————————————————
if ! command -v sphinx-build &>/dev/null; then
  echo "Installing Sphinx, RTD theme, and sphinx-autoapi…"
  pip install --upgrade pip
  pip install sphinx sphinx_rtd_theme sphinx-autoapi
fi

# ——— 8) Build HTML into _build —————————————————————————
echo "🛠  Building HTML into _build …"
sphinx-build -E -b html docs/source docs/_build/html

# ——— 9) Copy built HTML back into docs/ —————————————————————————
echo "📂  Copying built HTML into docs/ …"
cp -r docs/_build/html/* docs/
git add docs
git commit -m "📚 Rebuild docs (autoapi)"
git push origin main

echo "✅ Done! Your docs are now live at https://einarolafsson.github.io/spacr"
