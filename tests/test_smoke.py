"""
Smoke tests — cheap tests that catch the widest possible breakage.

If any of these fail, something fundamental about the package is broken;
they are the first line of defense before more specific tests run.
"""
from __future__ import annotations

import ast
import importlib
import os
from pathlib import Path

import pytest


PKG_ROOT = Path(__file__).resolve().parent.parent / "spacr"

# Every top-level .py in the spacr/ directory — resolved at import time so
# adding a new module automatically shows up here.
ALL_MODULES = sorted(
    p.stem
    for p in PKG_ROOT.glob("*.py")
    if p.stem not in {"__init__", "__main__"}
)

# The set exposed via __init__.py's __getattr__ lazy loader.
EXPECTED_LAZY = {
    "core", "io", "utils", "settings", "plot", "measure", "sequencing",
    "timelapse", "deep_spacr", "gui_utils", "gui_elements", "gui_core",
    "gui", "app_annotate", "app_make_masks", "app_mask", "app_measure",
    "app_classify", "app_sequencing", "app_umap", "submodules", "ml",
    "toxo", "spacr_cellpose", "spacrops", "sp_stats", "sim", "object",
    "logger", "version",
}


# ---------------------------------------------------------------------------
# 1. Every source file parses AND every module imports.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mod_name", ALL_MODULES)
def test_source_parses(mod_name):
    """Every spacr/*.py must parse without SyntaxError."""
    src = (PKG_ROOT / f"{mod_name}.py").read_text()
    ast.parse(src)


@pytest.mark.parametrize("mod_name", ALL_MODULES)
def test_module_imports(mod_name):
    """Every submodule imports cleanly under `import spacr.<mod>`."""
    try:
        importlib.import_module(f"spacr.{mod_name}")
    except Exception as e:
        # Same skip logic as test_module_dir_is_iterable — gui modules
        # need an accessible X display at import time.
        if "DisplayConnection" in type(e).__name__ or "Xauthority" in str(e):
            pytest.skip(f"spacr.{mod_name} needs a display: {e}")
        raise


# ---------------------------------------------------------------------------
# 2. The __init__ lazy loader agrees with the file layout.
# ---------------------------------------------------------------------------

def test_lazy_loader_matches_files():
    """Anything present in spacr/*.py should be reachable via `getattr(spacr, name)`."""
    import spacr
    listed = set(spacr._SUBMODULES)
    on_disk = set(ALL_MODULES)
    missing = on_disk - listed
    extra = listed - on_disk
    assert not missing, f"file present but not in _SUBMODULES: {missing}"
    assert not extra, f"in _SUBMODULES but no file: {extra}"


def test_lazy_loader_returns_modules():
    """Every listed submodule can actually be fetched via attribute access."""
    import spacr
    for name in EXPECTED_LAZY:
        try:
            mod = getattr(spacr, name)
        except Exception as e:
            # gui_* modules pull in pyautogui at import time; skip in
            # display-less subprocess runs.
            if "DisplayConnection" in type(e).__name__ or "Xauthority" in str(e):
                continue
            raise
        assert mod is not None, f"getattr(spacr, {name!r}) returned None"


def test_getattr_raises_for_unknown_name():
    import spacr
    with pytest.raises(AttributeError):
        _ = spacr.definitely_not_a_real_module


# ---------------------------------------------------------------------------
# 3. Version / metadata sanity.
# ---------------------------------------------------------------------------

def test_version_attribute_is_string():
    import spacr
    assert isinstance(spacr.__version__, str)
    assert spacr.__version__  # non-empty


def test_setup_version_matches_expected():
    """The setup.py VERSION should be a valid X.Y.Z semver string."""
    setup_src = (PKG_ROOT.parent / "setup.py").read_text()
    for line in setup_src.splitlines():
        if line.strip().startswith("VERSION"):
            _, _, val = line.partition("=")
            v = val.strip().strip("'\"")
            parts = v.split(".")
            assert len(parts) == 3 and all(p.isdigit() for p in parts), \
                f"unexpected VERSION string in setup.py: {val!r}"
            return
    pytest.fail("VERSION not found in setup.py")


# ---------------------------------------------------------------------------
# 4. No SyntaxWarnings across the package.
# ---------------------------------------------------------------------------

def test_no_syntax_warnings_across_package():
    """Loading every submodule should not emit any SyntaxWarnings."""
    import warnings
    import spacr

    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        for name in EXPECTED_LAZY:
            try:
                getattr(spacr, name)
            except Exception as e:
                if "DisplayConnection" in type(e).__name__ or "Xauthority" in str(e):
                    continue
                raise
        syn = [w for w in recorded if issubclass(w.category, SyntaxWarning)]
    assert not syn, "SyntaxWarnings: " + "\n".join(f"  {w.filename}:{w.lineno} {w.message}" for w in syn)


# ---------------------------------------------------------------------------
# 5. Every top-level function/class in the package can be introspected.
#    (Catches import-time NameErrors that only surface on attribute access.)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mod_name", ALL_MODULES)
def test_module_dir_is_iterable(mod_name):
    try:
        mod = importlib.import_module(f"spacr.{mod_name}")
    except Exception as e:
        # gui_utils / gui_core / gui_elements transitively import pyautogui,
        # which opens the X display at import time. In subprocess pytest
        # runs (e.g. under coverage) the xauth cookie may not be visible.
        # Treat that specific failure as a skip so the smoke suite still
        # runs elsewhere.
        if "DisplayConnection" in type(e).__name__ or "Xauthority" in str(e):
            pytest.skip(f"spacr.{mod_name} needs a display: {e}")
        raise
    names = dir(mod)
    assert isinstance(names, list)
    assert len(names) > 0
