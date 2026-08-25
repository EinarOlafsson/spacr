"""
Regression tests guarding the refactors landed on the spacr-claude branch.

Each test names the change it defends. If a future commit re-introduces
the fixed problem, one of these tests should fail loudly.

Eight tests here defended the Tkinter interface: that gui_elements did not
`import pyautogui` at module load, that spacr.gui imported without a
DISPLAY, and the palette, spacing, font, divider and button-hover shapes of
set_dark_style and the spacr* widgets. That interface is gone -- no
legacy_tk package, no gui.py/gui_core.py/gui_elements.py/gui_utils.py, no
MainApp and no set_dark_style anywhere in the tree -- so all eight were
guarding files that no longer exist. Qt widget behaviour is covered under
tests/qt/. The two module-name tables below kept their live rows and lost
only the ones naming deleted modules: a table asserting the shape of the
tree is exactly what catches the NEXT accidental deletion, so it is
narrowed rather than dropped.
"""
from __future__ import annotations

import ast
import inspect
import os
import time
from pathlib import Path

import pytest

PKG_ROOT = Path(__file__).resolve().parent.parent / "spacr"


# ---------------------------------------------------------------------------
# fix(measure): broken `spacr.build.lib.spacr` import
# ---------------------------------------------------------------------------

def test_measure_has_no_broken_build_import():
    src = (PKG_ROOT / "measure.py").read_text()
    assert "spacr.build" not in src, (
        "measure.py should not import from the sdist build tree (spacr.build.*)"
    )


def test_measure_exposes_the_entry_points_its_dispatchers_name_by_string():
    """``import spacr.measure`` alone is already covered by
    tests/test_smoke.py::test_module_imports[measure]. What that does NOT
    cover is the failure mode where the module imports fine but a public
    name has moved: both dispatch tables below name ``measure_crop`` as a
    *string*, so a rename breaks ``spacr-run measure`` and every dry_run at
    call time, not at import time.
    """
    import importlib
    import spacr.measure as m
    from spacr.validate import APP_FUNCTIONS

    # 1. spacr.validate's app -> function map.
    dotted = APP_FUNCTIONS["measure"]
    assert dotted == "spacr.measure.measure_crop"
    mod_name, _, attr = dotted.rpartition(".")
    assert callable(getattr(importlib.import_module(mod_name), attr))

    # 2. The CLI module registry's "module:function" entry point.
    from spacr.cli import MODULES
    entry = MODULES["measure"].entry
    assert entry == "spacr.measure:measure_crop"
    mod_name, _, attr = entry.partition(":")
    assert callable(getattr(importlib.import_module(mod_name), attr))

    # 3. The rest of the public surface other spacr modules import by name.
    for name in ("measure_crop", "_measure_crop_core", "get_components",
                 "resolve_n_jobs", "resolve_measurement_spacing",
                 "crop_objects_from_array"):
        assert callable(getattr(m, name)), f"spacr.measure.{name} is missing"

    # Contrast: the same lookup for a name that is NOT part of the surface
    # fails — so the loop above is checking presence, not just truthiness.
    with pytest.raises(AttributeError):
        getattr(m, "measure_crop_v1")


def test_measure_exposes_settings_binding():
    import spacr.measure as m
    assert hasattr(m, "settings"), (
        "spacr.measure should expose the settings module it imports"
    )


# ---------------------------------------------------------------------------
# refactor(various): dead _v1 / _v2 / _old and shadowed duplicate removal
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "mod_name,gone",
    [
        ("sim", "classifier_v2"),
        ("toxo", "custom_volcano_plot_v1"),
        ("submodules", "_plot_proportion_stacked_bars_v1"),
        ("submodules", "analyze_endodyogeny_v1"),
        ("plot", "plot_image_mask_overlay_old"),
        ("plot", "plot_proportion_stacked_bars_v1"),
        ("utils", "_get_cellpose_channels_v1"),
        ("utils", "_get_cellpose_channels_v2"),
        ("utils", "_split_data_v1"),
        ("utils", "choose_model_v2"),
        ("utils", "_merge_cells_based_on_parasite_overlap_v2"),
    ],
)
def test_dead_variants_are_gone(mod_name, gone):
    """No module here may grow back the superseded name beside its keeper.

    The table lost its app_annotate, gui_core, gui_utils and gui_elements
    rows: those modules are gone with the Tkinter interface, so
    `import spacr.gui_utils` no longer raises AttributeError-adjacent
    trouble, it raises ImportError, and a row asserting that a deleted
    module lacks an attribute tests nothing. The rows that remain name
    live modules, which is what makes this table worth keeping -- it is
    the check that catches a revert dragging a dead variant back in.

    The import is unguarded now. The old `except ...DisplayConnection...:
    pytest.skip` existed for the gui_* rows, which pulled in pyautogui and
    opened the X display at import time; nothing left in this table touches
    a display, so an import that fails here is a real breakage and must be
    read as one.
    """
    import importlib
    mod = importlib.import_module(f"spacr.{mod_name}")
    assert not hasattr(mod, gone), f"{mod_name}.{gone} should have been removed"


@pytest.mark.parametrize(
    "mod_name,name",
    [
        ("object", "_segment_single_image"),
        ("object", "_segment_spots"),
        ("object", "_segment_network"),
        ("utils", "suggest_training_changes"),
        ("timelapse", "_track_by_iou"),
        ("plot", "volcano_plot"),
    ],
)
def test_no_duplicate_top_level_defs(mod_name, name):
    """The earlier of every shadowed pair was removed — exactly one def now."""
    src = (PKG_ROOT / f"{mod_name}.py").read_text()
    tree = ast.parse(src)
    defs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name]
    assert len(defs) == 1, f"expected 1 def of {name} in {mod_name}.py, found {len(defs)}"


@pytest.mark.parametrize(
    "mod_name,name",
    [
        ("sim", "classifier"),
        ("toxo", "custom_volcano_plot"),
        ("plot", "plot_image_mask_overlay"),
        ("plot", "plot_proportion_stacked_bars"),
    ],
)
def test_kept_sibling_survives(mod_name, name):
    """The non-versioned sibling of every dropped _v1 must still be callable.

    Half this table was Tk: initiate_annotation_app, toggle_settings,
    attach_dependency_listeners and hide_all_settings were the survivors of
    a _v1/_v2 cull inside modules that have since been deleted outright with
    the Tkinter interface, so there is no sibling left to survive. Those four
    rows are gone; the four that remain name live modules, and this stays the
    pair of test_dead_variants_are_gone -- deleting the loser of a shadowed
    pair must not take the winner with it.

    The import is unguarded now, for the same reason as its sibling test:
    only the gui_* rows ever needed a display.
    """
    import importlib
    mod = importlib.import_module(f"spacr.{mod_name}")
    obj = getattr(mod, name, None)
    assert obj is not None, f"{mod_name}.{name} unexpectedly gone"
    assert callable(obj), f"{mod_name}.{name} is not callable"


# ---------------------------------------------------------------------------
# fix(submodules): correct `is 0` -> `== 0` comparison
# ---------------------------------------------------------------------------

def test_no_is_int_literal_comparisons():
    """`x is <int>` is a CPython-implementation quirk and a SyntaxWarning."""
    for py in PKG_ROOT.glob("*.py"):
        tree = ast.parse(py.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Compare):
                for op, comp in zip(node.ops, node.comparators):
                    if isinstance(op, (ast.Is, ast.IsNot)):
                        if (
                            isinstance(comp, ast.Constant)
                            and isinstance(comp.value, int)
                            and not isinstance(comp.value, bool)  # `is True/False` is fine here
                        ):
                            pytest.fail(
                                f"{py.name}:{node.lineno} `is`/`is not` with int literal "
                                f"({comp.value}) reintroduced"
                            )


# ---------------------------------------------------------------------------
# fix(utils): non-deprecated scipy import
# ---------------------------------------------------------------------------

def test_no_deprecated_scipy_ndimage_filters_import():
    src = (PKG_ROOT / "utils.py").read_text()
    assert "scipy.ndimage.filters" not in src, (
        "scipy.ndimage.filters is deprecated; import from scipy.ndimage instead"
    )


# ---------------------------------------------------------------------------
# refactor(various): no bare `except:` clauses remain
# ---------------------------------------------------------------------------

def test_no_bare_except_clauses():
    offenders = []
    for py in PKG_ROOT.glob("*.py"):
        tree = ast.parse(py.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler) and node.type is None:
                offenders.append(f"{py.name}:{node.lineno}")
    assert not offenders, "bare except: reintroduced at " + ", ".join(offenders)


# ---------------------------------------------------------------------------
# refactor(various): no mutable-literal default args remain
# ---------------------------------------------------------------------------

def test_no_mutable_default_args():
    offenders = []
    for py in PKG_ROOT.glob("*.py"):
        tree = ast.parse(py.read_text())
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for d in node.args.defaults + [x for x in node.args.kw_defaults if x is not None]:
                    if isinstance(d, (ast.List, ast.Dict, ast.Set)):
                        offenders.append(f"{py.name}:{node.lineno} def {node.name}")
    assert not offenders, "mutable default arg reintroduced at:\n  " + "\n  ".join(offenders)


def test_mutable_default_fix_behaviour_settings_dict():
    """`settings=None` sentinel must yield a fresh dict on each call."""
    from spacr.core import generate_image_umap
    sig = inspect.signature(generate_image_umap)
    assert sig.parameters["settings"].default is None, (
        "expected sentinel default; refactor may have regressed"
    )


# ---------------------------------------------------------------------------
# refactor(__init__): sim / object / spacrops exposed via lazy loader
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["sim", "object", "spacrops"])
def test_newly_exposed_submodules_reachable_via_getattr(name):
    import spacr
    assert name in spacr._SUBMODULES, f"{name} not in _SUBMODULES"
    assert getattr(spacr, name) is not None


# ---------------------------------------------------------------------------
# perf(__init__): download_models() no longer called at import time
# ---------------------------------------------------------------------------

def test_init_does_not_call_download_models_at_import():
    """The bottom-of-file eager `download_models()` line must stay gone."""
    src = (PKG_ROOT / "__init__.py").read_text()
    # It's fine for __init__ to reference download_models inside __getattr__.
    # It is NOT fine for the module to call it at top level.
    tree = ast.parse(src)
    for node in tree.body:  # top-level only
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            fn = node.value.func
            if isinstance(fn, ast.Name) and fn.id == "download_models":
                pytest.fail(
                    "download_models() called at top level in spacr/__init__.py — "
                    "would defeat the lazy loader again"
                )


def test_import_spacr_is_fast():
    """`python -c 'import spacr'` well under the pre-deferral baseline
    (~7 s). Threshold at 4 s to absorb normal CI/system variance."""
    import subprocess, sys
    t = time.perf_counter()
    proc = subprocess.run(
        [sys.executable, "-c", "import spacr"],
        env={**os.environ, "PYTHONPATH": str(PKG_ROOT.parent)},
        capture_output=True, timeout=30,
    )
    elapsed = time.perf_counter() - t
    assert proc.returncode == 0, f"import spacr failed: {proc.stderr.decode()[:400]}"
    assert elapsed < 4.0, f"import spacr took {elapsed:.2f}s (regressed from ~0.9s baseline)"

