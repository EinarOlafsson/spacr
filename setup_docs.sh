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
autoapi_options = [
    'members',
    'undoc-members',
    'show-inheritance',
    'special-members',
]
autoapi_python_class_content = 'both'  # Only show class-level docs, skip module variables
autoapi_keep_files = True

autoapi_ignore = [
    '*/tests/*',
    '*/spacr/__main__.py',
    '*/spacr/__init__.py',
    '*/spacr/app_classify.py',
    '*/spacr/app_mask.py',
    '*/spacr/app_measure.py',
    '*/spacr/app_sequencing.py',
    '*/spacr/version.py',
    '*/spacr/logger.py',
    '*/spacr/logger.py',
    '*/spacr/timelapse.py',
]

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
html_logo        = '_static/logo_spacr.png'
html_css_files   = ['custom.css']
EOF

# ——— 4) Write docs/source/index.rst —————————————————————————————
cat > docs/source/index.rst << 'EOF'
Welcome to SpaCr
================

.. |Docs| image:: https://github.com/EinarOlafsson/spacr/actions/workflows/pages/pages-build-deployment/badge.svg
   :target: https://einarolafsson.github.io/spacr/index.html
.. |PyPI version| image:: https://badge.fury.io/py/spacr.svg
   :target: https://badge.fury.io/py/spacr
.. |Python version| image:: https://img.shields.io/pypi/pyversions/spacr
   :target: https://pypistats.org/packages/spacr
.. |Licence: MIT| image:: https://img.shields.io/github/license/EinarOlafsson/spacr
   :target: https://github.com/EinarOlafsson/spacr/blob/main/LICENSE
.. |repo size| image:: https://img.shields.io/github/repo-size/EinarOlafsson/spacr
   :target: https://github.com/EinarOlafsson/spacr/
.. |Tutorial| image:: https://img.shields.io/badge/Tutorial-Click%20Here-brightgreen
   :target: https://einarolafsson.github.io/spacr/tutorial/

|Docs| |PyPI version| |Python version| |Licence: MIT| |repo size| |Tutorial|

**spaCR** (Spatial Phenotype Analysis of CRISPR Screens) is a Python toolkit for analyzing pooled CRISPR-Cas9 imaging screens. It integrates high-content imaging data with sequencing-based mutant identification to enable genotype-to-phenotype mapping at the single-cell level.

spaCR supports:

- **Segmentation** of microscopy images using models like Cellpose.
- **Single-cell feature extraction** and image cropping.
- **Classification** of phenotypes using classical and deep learning models.
- **Barcode decoding** from sequencing reads and well-level mutant quantification.
- **Statistical analysis**, including regression models to link genotypes to phenotypes.
- **Interactive visualization** of results including Grad-CAMs and phenotype maps.
- **GUI tools** for mask curation, annotation, and exploratory analysis.

Example Notebooks
=================

The following example Jupyter notebooks illustrate common workflows using spaCR.

- `Generate masks <https://github.com/EinarOlafsson/spacr/blob/main/Notebooks/1_spacr_generate_masks.ipynb>`_  
  *Generate cell, nuclei, and pathogen segmentation masks from microscopy images using Cellpose.*

- `Capture single cell images and measurements <https://github.com/EinarOlafsson/spacr/blob/main/Notebooks/2_spacr_generate_mesurments_crop_images.ipynb>`_  
  *Extract object-level measurements and crop single-cell images for downstream analysis.*

- `Machine learning based object classification <https://github.com/EinarOlafsson/spacr/blob/main/Notebooks/3a_spacr_machine_learning.ipynb>`_  
  *Train traditional machine learning models (e.g., XGBoost) to classify cell phenotypes based on extracted features.*

- `Computer vision based object classification <https://github.com/EinarOlafsson/spacr/blob/main/Notebooks/3b_spacr_computer_vision.ipynb>`_  
  *Train and evaluate deep learning models (PyTorch CNNs/Transformers) on cropped object images.*

- `Map sequencing barcodes <https://github.com/EinarOlafsson/spacr/blob/main/Notebooks/4_spacr_map_barecodes.ipynb>`_  
  *Map sequencing reads to row, column, and gRNA barcodes for CRISPR screen genotype-phenotype mapping.*

- `Finetune cellpose models <https://github.com/EinarOlafsson/spacr/blob/main/Notebooks/5_spacr_train_cellpose.ipynb>`_  
  *Finetune Cellpose models using your own annotated training data for improved segmentation accuracy.*

API Reference by Category
=========================

.. toctree::
   :caption: Core Modules
   :maxdepth: 1

   Core Logic <api/spacr/core/index>
   IO Utilities <api/spacr/io/index>
   General Utilities <api/spacr/utils/index>
   Settings <api/spacr/settings/index>
   Statistics <api/spacr/sp_stats/index>

.. toctree::
   :caption: Image Analysis
   :maxdepth: 1

   Measurement <api/spacr/measure/index>
   Plotting <api/spacr/plot/index>
   Cellpose Integration <api/spacr/spacr_cellpose/index>

.. toctree::
   :caption: Classification
   :maxdepth: 1

   Classical ML <api/spacr/ml/index>
   Deep Learning <api/spacr/deep_spacr/index>

.. toctree::
   :caption: GUI Components
   :maxdepth: 1

   GUI Main App <api/spacr/gui/index>
   GUI Core <api/spacr/gui_core/index>
   GUI Elements <api/spacr/gui_elements/index>
   GUI Utilities <api/spacr/gui_utils/index>

.. toctree::
   :caption: Sequencing & Submodules
   :maxdepth: 1

   Sequencing <api/spacr/sequencing/index>
   Toxoplasma Tools <api/spacr/toxo/index>
   Submodules <api/spacr/submodules/index>

GitHub Repository
=================

Visit the source code on GitHub: https://github.com/EinarOlafsson/spacr

.. toctree::
   :hidden:

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
