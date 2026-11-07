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

import ast
import json
import logging
import os
import pprint
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

LOG = logging.getLogger("spacr.notebook_export")


def _make_cell(cell_type: str, source: str) -> Dict[str, Any]:
    """Return a minimal nbformat v4 cell dict."""
    base: Dict[str, Any] = {
        "id":              uuid.uuid4().hex[:8],
        "cell_type":       cell_type,
        "metadata":        {},
        "source":          source.splitlines(keepends=True),
    }
    if cell_type == "code":
        base["execution_count"] = None
        base["outputs"] = []
    return base


def _read_manifest(run_dir: Path) -> Dict[str, Any]:
    """Read a run's JSON manifest when it is available and valid.

    :param run_dir: Run-journal directory that may contain ``manifest.json``.
    :returns: Decoded manifest, or an empty dictionary when it is missing,
        unreadable, or malformed.
    """
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        return {}
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _read_settings(run_dir: Path) -> Dict[str, Any]:
    """The run's recorded settings, or ``{}`` when it recorded none.

    A settings.json that is *present but unparseable* raises. This call is
    what ``export_run`` uses to "validate that the recorded settings exist and
    parse" before writing the notebook, and while it swallowed the parse error
    that validation was a no-op: the notebook was exported with
    ``json.loads((RUN_DIR / 'settings.json').read_text())`` as its first code
    cell, so the failure moved from the export — where it can be reported —
    into the user's notebook, where it is a traceback on cell 1 of a file they
    were told had been produced successfully.

    :raises ValueError: (``json.JSONDecodeError``) when the file will not
        parse. ``OSError`` propagates for the same reason.
    """
    settings_path = run_dir / "settings.json"
    if not settings_path.exists():
        return {}
    return json.loads(settings_path.read_text(encoding="utf-8"))



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
    for name in ("manifest.json", "settings.json"):
        required = run_dir / name
        if not required.exists():
            raise FileNotFoundError(required)
    manifest = _read_manifest(run_dir)
    _read_settings(run_dir)
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

    nb = _notebook_document(cells)

    if out_path is None:
        out_path = run_dir / "notebook.ipynb"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(nb, indent=2), encoding="utf-8")
    LOG.info("exported run notebook → %s", out_path)
    return out_path


#: The folder, under a project, that a GUI run's notebook is written to.
#: ``spacr.utils.save_settings`` writes each run's settings CSV there too.
LAUNCH_NOTEBOOK_FOLDER = "settings"


def _notebook_document(cells: List[Dict[str, Any]]) -> Dict[str, Any]:
    """An nbformat v4 notebook holding ``cells``."""
    return {
        "nbformat":       4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language":     "python",
                "name":         "python3",
            },
            "language_info": {"name": "python"},
        },
        "cells": cells,
    }


def _settings_literal(settings: Mapping[str, Any]) -> Optional[str]:
    """``settings`` as Python source that evaluates back to an equal dict.

    :returns: the source, or ``None`` when a value has no literal form (an
        object, a NaN), since a notebook built from it would run with
        different settings.
    """
    text = pprint.pformat(dict(settings), indent=1, width=88, sort_dicts=False)
    try:
        if ast.literal_eval(text) != dict(settings):
            return None
    except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
        return None
    return text


def _launch_notebook_cells(app_key: str, settings: Mapping[str, Any],
                           entry_module: str, entry_func: str,
                           settings_text: str) -> List[Dict[str, Any]]:
    """The cells of the notebook a GUI run writes when it starts.

    The run cell calls the function the GUI's Run button calls, after the
    same type coercion :func:`spacr.qt.bridge.resolve_pipeline_entry`
    applies, so running the notebook is running the GUI run again.
    """
    from .macro import _version

    started = time.strftime("%Y-%m-%d %H:%M:%S")
    summary = "\n".join([
        f"# spaCR {app_key} run",
        "",
        f"Written by the spaCR GUI when Run was pressed, {started}, with "
        f"spaCR {_version()}.",
        "",
        f"Running every cell repeats that run: `{entry_module}.{entry_func}` "
        "is called with the settings below, exactly as the GUI called it. "
        "Edit a value to run the analysis again with that change.",
    ])
    run = "\n".join([
        "from spacr.validate import coerce_expected_types",
        f"from {entry_module} import {entry_func}",
        "",
        f"settings = coerce_expected_types(dict(SETTINGS), {app_key!r})",
        f"result = {entry_func}(settings)",
    ])
    return [
        _make_cell("markdown", summary),
        _make_cell("markdown", "## Settings"),
        _make_cell("code", f"SETTINGS = {settings_text}"),
        _make_cell("markdown", "## Run"),
        _make_cell("code", run),
    ]


def _write_launch_notebook(app_key: str, settings: Mapping[str, Any],
                           folder: Any = None) -> Optional[Path]:
    """Write the notebook that repeats a GUI run, into its project.

    :param app_key: the module the run is for.
    :param settings: the settings the run is started with.
    :param folder: where to write; by default the ``settings`` folder of
        the run's project.
    :returns: the notebook path, or ``None`` when the module has no
        importable entry point, the run has no project folder on disk, or a
        setting has no literal form.
    """
    from .macro import _project_root, entry_for

    entry_module, entry_func = entry_for(app_key)
    if not entry_module or not entry_func:
        return None
    settings_text = _settings_literal(settings)
    if settings_text is None:
        LOG.info("no notebook for this %s run: a setting has no literal form",
                 app_key)
        return None
    if folder is None:
        root = _project_root(app_key, settings)
        if not root or not os.path.isdir(root):
            return None
        folder = os.path.join(root, LAUNCH_NOTEBOOK_FOLDER)
    cells = _launch_notebook_cells(app_key, settings, entry_module,
                                   entry_func, settings_text)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    path = Path(folder) / f"{app_key}_run_{stamp}.ipynb"
    number = 1
    while path.exists():
        number += 1
        path = Path(folder) / f"{app_key}_run_{stamp}_{number}.ipynb"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_notebook_document(cells), indent=1),
                    encoding="utf-8")
    LOG.info("wrote the notebook of this %s run: %s", app_key, path)
    return path

