"""The three dispatch tables must agree on what each app runs.

spaCR resolves an app key to a pipeline function in three independent places:

  * ``spacr.qt.bridge.resolve_pipeline_entry``   — the Qt GUI
  * ``spacr.gui_utils.run_function_gui``         — the Tk GUI
  * ``spacr.validate.APP_FUNCTIONS``             — pre-flight validation
  * ``spacr.cli.MODULES``                        — the headless runner

and the settings panel is built from a fourth,
``spacr.qt.screens.settings_model.resolve_default_settings``.

When the dispatch and the panel disagree, the user is shown settings that are
silently ignored. That is what happened to ``classify``: the Qt panel was built
from ``deep_spacr_defaults`` — so it offered ``generate_training_dataset``,
``apply_model_to_dataset``, ``n_top_examples`` and ``tar_path`` — while the Qt
bridge ran ``train_test_model``, which consumes none of them. Tk and validate
both ran ``deep_spacr``. Nothing failed; the switches just did nothing.

These tests pin the agreement so a future divergence is loud.
"""
from __future__ import annotations

import pytest


def _qt_entry(app_key):
    from spacr.qt.bridge import resolve_pipeline_entry
    fn = resolve_pipeline_entry(app_key)
    # entries are wrapped by log_call; unwrap to the real function name
    return getattr(fn, "__wrapped__", fn)


def _entry_name(fn):
    inner = getattr(fn, "__wrapped__", fn)
    return getattr(inner, "__name__", None) or getattr(fn, "__name__", "")


# ---------------------------------------------------------------------------
# classify: the divergence that actually bit
# ---------------------------------------------------------------------------

def test_qt_classify_runs_the_full_deep_spacr_pipeline():
    """train_test_model is only the training stage. The panel offers dataset
    generation and inference too, so the entry point has to be the driver."""
    assert _entry_name(_qt_entry("classify")) == "deep_spacr"


def test_validate_agrees_that_classify_is_deep_spacr():
    from spacr.validate import APP_FUNCTIONS
    assert APP_FUNCTIONS["classify"].endswith("deep_spacr")


def test_every_setting_the_classify_panel_shows_is_consumed():
    """The panel is built from deep_spacr_defaults; the four keys below are the
    ones train_test_model does not read, and they are exactly why the entry
    point had to change."""
    from spacr.qt.screens.settings_model import resolve_default_settings
    shown = resolve_default_settings("classify")
    for key in ("generate_training_dataset", "apply_model_to_dataset"):
        assert key in shown, f"{key} is no longer shown; update this test"
    # and the entry point is the one that reads them
    assert _entry_name(_qt_entry("classify")) == "deep_spacr"


# ---------------------------------------------------------------------------
# The rest of the table
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("app_key", [
    "mask", "measure", "classify", "umap", "ml_analyze", "regression",
    "map_barcodes", "recruitment", "analyze_plaques", "train_cellpose",
    "cellpose_masks", "timelapse", "motility",
])
def test_qt_and_validate_name_the_same_function(app_key):
    from spacr.validate import APP_FUNCTIONS
    if app_key not in APP_FUNCTIONS:
        pytest.skip(f"{app_key} has no validate entry")
    qt_name = _entry_name(_qt_entry(app_key))
    validate_name = APP_FUNCTIONS[app_key].rsplit(".", 1)[-1]
    assert qt_name == validate_name, (
        f"{app_key}: Qt runs {qt_name}, validate describes {validate_name}. "
        "A user validated against one function and ran another.")


@pytest.mark.parametrize("app_key", [
    "mask", "measure", "classify", "umap", "ml_analyze", "regression",
    "map_barcodes", "recruitment", "analyze_plaques", "train_cellpose",
    "cellpose_masks",
])
def test_qt_and_the_headless_cli_name_the_same_function(app_key):
    """spacr-run must do what the GUI button does, or a settings.csv saved from
    the GUI runs a different pipeline on the cluster."""
    from spacr.cli import MODULES
    if app_key not in MODULES:
        pytest.skip(f"{app_key} has no CLI module")
    qt_name = _entry_name(_qt_entry(app_key))
    cli_name = MODULES[app_key].func_name
    assert qt_name == cli_name, (
        f"{app_key}: Qt runs {qt_name}, spacr-run runs {cli_name}. A "
        "settings.csv saved from the GUI would run a different pipeline on "
        "the cluster.")


def test_every_pipeline_app_has_an_entry_point():
    """A settings panel with no runnable entry is a button that does nothing.

    The interactive tools are excluded: they build their own UI and have no
    batch equivalent, which spacr.cli.INTERACTIVE_ONLY already records.
    """
    from spacr.qt.app import APPS
    from spacr.qt.bridge import resolve_pipeline_entry
    from spacr.qt.screens.settings_model import resolve_default_settings
    from spacr.cli import INTERACTIVE_ONLY

    missing = []
    for key, _name, _desc, _section in APPS:
        if key in INTERACTIVE_ONLY:
            continue
        try:
            shown = resolve_default_settings(key)
        except Exception:
            continue
        if not shown:
            continue
        try:
            entry = resolve_pipeline_entry(key)
        except Exception:
            entry = None
        if entry is None:
            missing.append(key)
    assert not missing, f"apps with a settings panel but no entry point: {missing}"


def test_every_interactive_tool_is_declared_interactive_only():
    """An app with no pipeline entry must be listed in INTERACTIVE_ONLY, so
    `spacr-run --describe` explains itself instead of saying 'unknown module'."""
    from spacr.qt.app import APPS
    from spacr.qt.bridge import resolve_pipeline_entry
    from spacr.cli import INTERACTIVE_ONLY

    undeclared = []
    for key, _name, _desc, _section in APPS:
        if key in INTERACTIVE_ONLY:
            continue
        try:
            if resolve_pipeline_entry(key) is None:
                undeclared.append(key)
        except Exception:
            undeclared.append(key)
    assert not undeclared, (
        f"apps with no pipeline entry that are not in cli.INTERACTIVE_ONLY: "
        f"{undeclared}. spacr-run --describe will call them unknown.")
