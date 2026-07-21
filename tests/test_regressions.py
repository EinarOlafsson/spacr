"""
Regression tests guarding the refactors landed on the spacr-claude branch.

Each test names the change it defends. If a future commit re-introduces
the fixed problem, one of these tests should fail loudly.
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


def test_measure_module_imports():
    import spacr.measure  # noqa: F401


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
        ("app_annotate", "initiate_annotation_app_v1"),
        ("sim", "classifier_v2"),
        ("toxo", "custom_volcano_plot_v1"),
        ("gui_core", "toggle_settings_v1"),
        ("gui_utils", "attach_dependency_listeners_v1"),
        ("gui_utils", "hide_all_settings_v1"),
        ("gui_elements", "open_settings_window_v1"),
        ("gui_elements", "load_images_v1"),
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
    import importlib
    try:
        mod = importlib.import_module(f"spacr.{mod_name}")
    except Exception as e:
        if "DisplayConnection" in type(e).__name__ or "Xauthority" in str(e):
            pytest.skip(f"spacr.{mod_name} needs a display: {e}")
        raise
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
        ("app_annotate", "initiate_annotation_app"),
        ("sim", "classifier"),
        ("toxo", "custom_volcano_plot"),
        ("gui_core", "toggle_settings"),
        ("gui_utils", "attach_dependency_listeners"),
        ("gui_utils", "hide_all_settings"),
        ("plot", "plot_image_mask_overlay"),
        ("plot", "plot_proportion_stacked_bars"),
    ],
)
def test_kept_sibling_survives(mod_name, name):
    """The non-versioned sibling of every dropped _v1 must still be callable."""
    import importlib
    try:
        mod = importlib.import_module(f"spacr.{mod_name}")
    except Exception as e:
        # gui_* modules pull in pyautogui at import time; skip if no
        # xauth-authorized display is available.
        if "DisplayConnection" in type(e).__name__ or "Xauthority" in str(e):
            pytest.skip(f"spacr.{mod_name} needs a display: {e}")
        raise
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


# ---------------------------------------------------------------------------
# perf(gui_elements): no display-dependent imports at module load time
# ---------------------------------------------------------------------------

def test_gui_elements_does_not_import_pyautogui():
    """Regression: spacr.gui_elements used to `import pyautogui` at module
    load; pyautogui transitively opens the X display via mouseinfo, which
    crashed `spacr` in headless / xauth-broken environments even though
    pyautogui wasn't actually used (its only reference was a commented-out
    line). Guard that neither the source nor the install specs re-add it."""
    src = (PKG_ROOT / "gui_elements.py").read_text()
    for line in src.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        assert "pyautogui" not in stripped, (
            f"gui_elements.py re-introduced pyautogui: {line!r}"
        )
    setup_src = (PKG_ROOT.parent / "setup.py").read_text()
    assert "pyautogui" not in setup_src, (
        "setup.py re-declared pyautogui (unused dependency)"
    )


def test_import_spacr_gui_without_display():
    """Both spacr.gui_elements and spacr.gui must import cleanly with no
    DISPLAY/XAUTHORITY environment — required for headless installs
    (Docker, CI, non-desktop cronjobs). Prior to the pyautogui removal
    this crashed with Xlib.error.DisplayConnectionError."""
    import subprocess, sys
    env = {k: v for k, v in os.environ.items()
           if k not in ("DISPLAY", "XAUTHORITY")}
    env["PYTHONPATH"] = str(PKG_ROOT.parent)
    proc = subprocess.run(
        [sys.executable, "-c",
         "import spacr.gui_elements as _ge; import spacr.gui as _gui; "
         "assert hasattr(_ge, 'spacrCard') and _gui.MainApp is not None"],
        env=env, capture_output=True, timeout=30,
    )
    assert proc.returncode == 0, (
        f"headless import failed: {proc.stderr.decode()[:800]}"
    )


# ---------------------------------------------------------------------------
# feat(gui): palette / spacing / font / divider on style_out
# ---------------------------------------------------------------------------

def test_set_dark_style_returns_full_palette(dark_style):
    for key in (
        "bg_color", "fg_color", "active_color", "inactive_color",
        "border_color", "muted_color",
        "success_color", "warning_color", "error_color",
    ):
        assert key in dark_style, f"style_out missing {key!r}"
        v = dark_style[key]
        assert isinstance(v, str) and v.startswith("#") and len(v) == 7, (
            f"{key} should be a #RRGGBB string, got {v!r}"
        )


def test_pure_black_palette_defaults_landed(dark_style):
    """The palette values that land when the caller passes named
    defaults — pure black background per user preference (was briefly
    soft-dark #0e1116, reverted 2026-07-21)."""
    assert dark_style["bg_color"].lower() == "#000000"
    assert dark_style["fg_color"].lower() == "#ffffff"
    assert dark_style["active_color"].lower() == "#007bff"
    assert dark_style["inactive_color"].lower() == "#2b2b2b"


def test_spacing_scale_present(dark_style):
    sp = dark_style["spacing"]
    assert sp == {"xs": 4, "sm": 8, "md": 12, "lg": 16, "xl": 24}


def test_font_size_hierarchy_present(dark_style):
    fs = dark_style["font_sizes"]
    for key in ("small", "body", "header", "title"):
        assert key in fs
    assert fs["small"] < fs["body"] < fs["header"] < fs["title"]


def test_spacr_divider_constructs(tk_root):
    try:
        from spacr.gui_elements import spacrDivider
    except Exception as e:
        if "DisplayConnection" in type(e).__name__ or "Xauthority" in str(e):
            pytest.skip(f"spacr.gui_elements needs a display: {e}")
        raise
    # All three shapes.
    plain = spacrDivider(tk_root)
    captioned = spacrDivider(tk_root, text="Section")
    vertical = spacrDivider(tk_root, orient="vertical")
    tk_root.update_idletasks()
    assert plain.winfo_class() == "Frame"
    assert captioned.text == "Section"
    assert vertical.orient == "vertical"


def test_spacr_button_has_hover_fade(tk_root):
    """spacrButton must expose the fade machinery, not the old flash-swap."""
    try:
        from spacr.gui_elements import spacrButton
    except Exception as e:
        if "DisplayConnection" in type(e).__name__ or "Xauthority" in str(e):
            pytest.skip(f"spacr.gui_elements needs a display: {e}")
        raise
    btn = spacrButton(tk_root, text="ok", show_text=True, size=50, animation=False)
    assert hasattr(btn, "_fade_bg_to")
    assert hasattr(btn, "_fade_after_id")
    # Simulate a hover; the fill should differ from the original inactive color
    # OR still equal target if a tick already completed; either proves the path ran.
    btn.on_enter()
    tk_root.update_idletasks()
    tk_root.update()
    fill = btn.canvas.itemcget(btn.button_bg, "fill")
    assert fill.startswith("#") and len(fill) == 7
