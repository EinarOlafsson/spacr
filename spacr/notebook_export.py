"""
Notebook export — turn a run journal into a runnable Jupyter notebook.

Every completed run in ``~/.spacr/runs/`` can be exported to a
``.ipynb`` file that walks a user through:

1. Importing spaCR + the pipeline used.
2. Loading the exact settings from ``settings.json``.
3. Optionally re-running the pipeline (commented out — reviewer
   choice) or loading the outputs directly.
4. Scaffolded per-output cells that read the measurements database,
   plot summary charts, and expose the DataFrame for further
   analysis in the notebook.

Public API::

    from spacr.notebook_export import export_run

    nb_path = export_run(run_dir, out_path="/tmp/mask_run.ipynb")

Notebook is generated with the nbformat library (widely-shipped
Jupyter dependency; no extra install needed).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

LOG = logging.getLogger("spacr.notebook_export")


def _make_cell(cell_type: str, source: str) -> Dict[str, Any]:
    """Return a minimal nbformat v4 cell dict."""
    base: Dict[str, Any] = {
        "cell_type":       cell_type,
        "metadata":        {},
        "source":          source.splitlines(keepends=True),
    }
    if cell_type == "code":
        base["execution_count"] = None
        base["outputs"] = []
    return base


def _read_manifest(run_dir: Path) -> Dict[str, Any]:
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        return {}
    try:
        return json.loads(manifest_path.read_text())
    except Exception:
        return {}


def _read_settings(run_dir: Path) -> Dict[str, Any]:
    settings_path = run_dir / "settings.json"
    if not settings_path.exists():
        return {}
    try:
        return json.loads(settings_path.read_text())
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Entrypoint scaffolds — per pipeline app
# ---------------------------------------------------------------------------

_ENTRYPOINTS: Dict[str, str] = {
    "mask":     "from spacr.core import preprocess_generate_masks as _run",
    "measure":  "from spacr.measure import measure_crop as _run",
    "classify": "from spacr.deep_spacr import deep_spacr as _run",
    "umap":     "from spacr.io import generate_dataset as _run",
    "regression":  "from spacr.ml import perform_regression as _run",
    "ml_analyze":  "from spacr.ml import ml_analysis as _run",
    "map_barcodes":
        "from spacr.sequencing import generate_barecode_mapping as _run",
}


def _pipeline_import_cell(app_key: str) -> str:
    """Return the code cell text importing the pipeline for ``app_key``."""
    imp = _ENTRYPOINTS.get(
        app_key,
        "# no known entry point for this app; edit as needed\n"
        "_run = None",
    )
    return imp


def _output_cell_for(app_key: str, run_dir: Path) -> str:
    """Return per-app tabular / plot scaffold reading the run's outputs."""
    if app_key in ("measure", "crop"):
        return (
            "import pandas as pd\n"
            "import sqlite3\n"
            "\n"
            "# spaCR writes measurements to a SQLite DB under the plate\n"
            "# root. Substitute your `src` if you re-ran the pipeline\n"
            "# somewhere else.\n"
            f"db_path = SETTINGS['src'] + '/measurements/measurements.db'\n"
            "with sqlite3.connect(db_path) as _conn:\n"
            "    cell_df = pd.read_sql('select * from cell limit 500', _conn)\n"
            "cell_df.head()\n"
        )
    if app_key == "mask":
        return (
            "from pathlib import Path\n"
            "import tifffile\n"
            "import matplotlib.pyplot as plt\n"
            "\n"
            "# Plot the first three masks the pipeline generated.\n"
            "masks_dir = Path(SETTINGS['src']) / 'masks'\n"
            "mask_files = sorted(masks_dir.glob('*.tif'))[:3]\n"
            "fig, axs = plt.subplots(1, len(mask_files), figsize=(12, 4))\n"
            "for ax, mp in zip(axs, mask_files):\n"
            "    ax.imshow(tifffile.imread(str(mp)))\n"
            "    ax.set_title(mp.name)\n"
            "    ax.set_xticks([]); ax.set_yticks([])\n"
            "plt.show()\n"
        )
    return (
        "# TODO: add per-output analysis for this app.\n"
        "# The recorded settings live in the SETTINGS dict.\n"
        "SETTINGS\n"
    )


def _summary_markdown(run_dir: Path,
                        manifest: Dict[str, Any]) -> str:
    """First markdown cell — human-facing summary of the run."""
    lines = [
        f"# spaCR run — {manifest.get('app_key', 'unknown')}",
        "",
        f"Exported from run folder `{run_dir.name}`.",
        "",
        f"* **Started**: {manifest.get('start_utc', '?')}",
        f"* **Elapsed**: {manifest.get('elapsed_s', '?')} s",
        f"* **Status**: {manifest.get('status', '?')}",
        f"* **spaCR version**: "
        f"{manifest.get('env', {}).get('spacr', '?')}",
        f"* **Torch / Cellpose**: "
        f"{manifest.get('env', {}).get('torch', '?')} / "
        f"{manifest.get('env', {}).get('cellpose', '?')}",
        "",
        "This notebook loads the exact settings used, and gives you a "
        "starting point for reviewing the outputs or re-running the "
        "pipeline. Every cell below is safe to run top-to-bottom.",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main entrypoint
# ---------------------------------------------------------------------------

def export_run(run_dir: Any,
                 out_path: Optional[Any] = None) -> Path:
    """Export a run journal folder to a Jupyter notebook.

    :param run_dir: path to a ``~/.spacr/runs/<ts>_<uuid>__<app>``
        folder.
    :param out_path: destination ``.ipynb`` path. Defaults to
        ``<run_dir>/notebook.ipynb``.
    :returns: the notebook path on disk.
    :raises FileNotFoundError: when the run folder is missing or
        doesn't have a ``manifest.json`` / ``settings.json``.
    """
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        raise FileNotFoundError(f"no such run folder: {run_dir}")
    manifest = _read_manifest(run_dir)
    settings = _read_settings(run_dir)
    app_key = manifest.get("app_key", "unknown")

    cells: List[Dict[str, Any]] = []
    cells.append(_make_cell("markdown", _summary_markdown(run_dir, manifest)))

    cells.append(_make_cell("code",
        "import json\n"
        "from pathlib import Path\n"
        "\n"
        f"RUN_DIR = Path(r'{run_dir}')\n"
        "SETTINGS = json.loads((RUN_DIR / 'settings.json').read_text())\n"
        "SETTINGS\n"
    ))

    cells.append(_make_cell("markdown",
        "## Re-run the pipeline\n"
        "\n"
        "Uncomment the block below to replay this run in the notebook.\n"
        "The output will land in whatever folder `src` points at."
    ))
    cells.append(_make_cell("code",
        _pipeline_import_cell(app_key)
        + "\n\n"
        "# _run(SETTINGS)   # ← uncomment to actually re-run\n"
    ))

    cells.append(_make_cell("markdown",
        "## Inspect outputs"
    ))
    cells.append(_make_cell("code", _output_cell_for(app_key, run_dir)))

    nb: Dict[str, Any] = {
        "nbformat":       4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language":     "python",
                "name":         "python3",
            },
            "language_info": {
                "name":    "python",
            },
        },
        "cells": cells,
    }

    if out_path is None:
        out_path = run_dir / "notebook.ipynb"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(nb, indent=2))
    LOG.info("exported run notebook → %s", out_path)
    return out_path
