"""Showing a PNG must not cost the segmentation stack.

Reported 2026-08-19: "in the annotation app images load almost
instintaniously while in the regression cell montage it takes way longer".

MEASURED, not guessed. The first montage build took 6.7 SECONDS and pulled in
4,336 modules -- torch, torchvision, cv2, sklearn, statsmodels, skimage -- to
draw nine crops that were already on disk. The second took 78 ms, which is
the shape of a one-time import cost rather than of slow work.

Two one-line imports did it. `cell_montage.load_montage_objects` asked
`spacr.utils` for `correct_metadata_column_names` and `spacr.io` for
`crop_rows_from_png_list`; both of those modules import torch on their line 3,
and neither function needs it. They now live in `spacr.schema` and
`spacr.png_list`, and both old modules re-export them so no caller changed.

These tests pin the IMPORT GRAPH rather than a timing, because a timing test
on a shared machine is a flake generator and the cost here is deterministic:
either the heavy module is imported or it is not.
"""
import ast
import pathlib

import pytest

#: Modules whose import pulls in the numerical/segmentation stack. Each one
#: costs seconds; `spacr/utils.py` and `spacr/io.py` both import torch,
#: torchvision and cv2 at module scope.
HEAVY = ("spacr.utils", "spacr.io", "spacr.ml", "spacr.submodules")


def _module_level_imports(path):
    """Every module imported at import time, ignoring imports inside defs."""
    tree = ast.parse(pathlib.Path(path).read_text(encoding="utf-8"))
    out = set()
    for node in tree.body:                      # BODY, not walk: top level only
        if isinstance(node, ast.Import):
            out.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            out.add(("." * node.level) + node.module)
    return out


def test_the_light_modules_are_light():
    """`spacr.png_list` and `spacr.schema` must not reach for the heavy ones."""
    for module in ("spacr/png_list.py", "spacr/schema.py"):
        imported = _module_level_imports(module)
        assert not {"torch", "torchvision", "cv2", "cellpose"} & imported, (
            f"{module} imports the segmentation stack at module scope")


def test_the_helpers_are_where_the_montage_can_reach_them_cheaply():
    from spacr.png_list import crop_rows_from_png_list  # noqa: F401
    from spacr.schema import correct_metadata_column_names  # noqa: F401


def test_the_old_homes_still_export_them():
    """Every caller that imported them from `utils`/`io` is unchanged."""
    import spacr.io
    import spacr.png_list
    import spacr.schema
    import spacr.utils

    assert spacr.io.crop_rows_from_png_list is (
        spacr.png_list.crop_rows_from_png_list)
    assert spacr.io._merged_field_paths is spacr.png_list._merged_field_paths
    # utils re-exports by delegation rather than by identity, so compare
    # behaviour: importing utils is what this whole file exists to avoid.
    assert callable(spacr.utils.correct_metadata_column_names)


def test_the_montage_module_does_not_ask_the_heavy_modules_at_import_time():
    imported = _module_level_imports("spacr/cell_montage.py")
    assert not {"torch", "cv2", ".utils", ".io"} & imported


def test_loading_a_montage_never_imports_torch(tmp_path, qtbot):
    """THE WHOLE POINT, driven through the real widget.

    Run in a subprocess so the measurement is honest: by the time the suite
    reaches this file, some other test has certainly imported torch already,
    and asserting on `sys.modules` in-process would pass whatever happened.
    """
    import subprocess
    import sys
    import textwrap

    script = textwrap.dedent(
        """
        import os, sys
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        sys.path.insert(0, "tests/qt"); sys.path.insert(0, "tests")
        from PySide6.QtWidgets import QApplication
        app = QApplication.instance() or QApplication([])
        import pandas as pd, tempfile, pathlib
        import test_cells_behind_the_dot_tab as T
        tmp = pathlib.Path(tempfile.mkdtemp())
        root, db, csv = T._screen(tmp, with_png=True)
        view = T.CellMontageView(
            frame_provider=lambda: pd.read_csv(csv),
            results_provider=lambda: csv,
            database_provider=lambda: T._rows(db), threaded=False)
        view.set_coefficient(T.GENE_KEY)
        view.build()
        assert view.plans(), "the montage drew nothing, so this proves nothing"
        print("TORCH" if "torch" in sys.modules else "NO_TORCH")
        """)
    done = subprocess.run([sys.executable, "-c", script],
                          capture_output=True, text=True, timeout=600)
    assert "NO_TORCH" in done.stdout, (
        f"the montage imported torch to show a PNG.\n"
        f"stdout: {done.stdout[-800:]}\nstderr: {done.stderr[-800:]}")
