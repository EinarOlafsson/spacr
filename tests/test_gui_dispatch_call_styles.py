"""The dispatcher must call each pipeline with the signature it actually has.

This file was written against the Tk dispatcher, where ``run_function_gui``
chose between three hard-coded call shapes with an integer ``imports`` flag
and ``function_gui_wrapper`` swallowed the resulting TypeError into a queue
nobody was necessarily watching. Two settings types raised on every run and
nobody noticed:

  * ``regression`` was dispatched with ``imports=2``, i.e.
    ``perform_regression(src=..., settings=...)``, but ``ml.perform_regression``
    takes one positional ``settings``.
  * ``convert`` was dispatched with ``imports=1``, i.e.
    ``process_non_tif_non_2D_images(settings=...)``, but ``io.process_non_tif_
    non_2D_images`` takes a bare ``folder`` path.

and an unrecognised ``imports`` value fell off the end of the if/elif with no
else, so the worker was never called and the wrapper returned as though the
module had completed successfully.

That dispatcher is gone with the Tk interface, and its successor states the
same thing declaratively: every :class:`spacr.cli.Module` carries a
``call_style``, and :func:`spacr.cli._call_entry` is the one place that turns
it into a call. So the tests below bind each dispatched callable to its
declared call style there instead, which is the same defect asked of the code
that still runs it.

One test did not survive the move. The Tk wrapper replaced ``plt.show`` for
the duration of a run and had to put it back even when the worker raised;
nothing in the Qt or headless path does that, so there is no longer anything
to assert about it.
"""
from __future__ import annotations

import inspect
import queue

import pytest

import matplotlib
matplotlib.use("Agg")


# ---------------------------------------------------------------------------
# _call_entry call styles
# ---------------------------------------------------------------------------

def _module(**overrides):
    """A throwaway :class:`spacr.cli.Module` with the given call style."""
    from spacr.cli import Module

    fields = dict(key="probe", summary="probe", entry="spacr.core:probe",
                  defaults=None, validate_key="")
    fields.update(overrides)
    return Module(**fields)


def test_settings_call_style_passes_the_whole_dict():
    from spacr.cli import _call_entry

    seen = {}

    def worker(settings):
        seen["settings"] = settings
        return "ran"

    module = _module(call_style="settings")
    assert _call_entry(module, worker, {"src": "/x"}) == "ran"
    assert seen["settings"] == {"src": "/x"}


def test_folder_call_style_passes_a_bare_path():
    """The style a plugin declares: one positional path, no settings kwarg."""
    from spacr.cli import _call_entry

    seen = {}

    def worker(folder):
        seen["folder"] = folder
        return "ran"

    module = _module(call_style="folder")
    assert _call_entry(module, worker, {"src": "/images"}) == "ran"
    assert seen["folder"] == "/images"


@pytest.mark.parametrize("src", [None, "", "   ", 7, ["/a", "/b"]])
def test_folder_call_style_refuses_an_src_that_is_not_one_folder(src):
    """The worker must not run: `func(None)` and `func(['/a','/b'])` are the
    calls that used to reach a pipeline and fail somewhere unrelated."""
    from spacr.cli import SettingsError, _call_entry

    calls = []
    module = _module(call_style="folder")
    with pytest.raises(SettingsError, match="needs a single folder"):
        _call_entry(module, lambda folder: calls.append(folder),
                    {"src": src})
    assert calls == [], "worker must not run without a usable folder"


# ---------------------------------------------------------------------------
# The dispatch table must agree with the real signatures
# ---------------------------------------------------------------------------

def _all_module_keys():
    from spacr.cli import MODULES
    return sorted(MODULES)


def _accepts(func, *args, **kwargs):
    """True if ``func(*args, **kwargs)`` would bind without a TypeError."""
    try:
        inspect.signature(func).bind(*args, **kwargs)
        return True
    except TypeError:
        return False


def test_the_module_table_is_not_empty():
    """A table that failed to build would make every test below vacuous."""
    keys = _all_module_keys()
    assert len(keys) >= 20, f"only found {len(keys)} headless modules"
    for expected in ("mask", "measure", "classify", "regression", "convert"):
        assert expected in keys


@pytest.mark.parametrize("key", _all_module_keys())
def test_every_dispatched_worker_binds_to_its_call_style(key):
    """The whole table, not just the two that were broken."""
    from spacr.cli import MODULES, import_entry

    module = MODULES[key]
    func = import_entry(module)

    if module.call_style == "settings":
        assert _accepts(func, {}), (
            f"{key}: dispatched as function(settings) but "
            f"{module.func_name}{inspect.signature(func)} does not accept it")
    elif module.call_style == "folder":
        assert _accepts(func, "/x"), (
            f"{key}: dispatched as function(path) but "
            f"{module.func_name}{inspect.signature(func)} does not accept it")
    else:
        pytest.fail(f"{key}: unknown call style {module.call_style!r}")


def test_regression_is_dispatched_with_a_single_settings_argument():
    from spacr.cli import MODULES, import_entry
    from spacr.ml import perform_regression

    module = MODULES["regression"]
    assert module.call_style == "settings", (
        "perform_regression(settings) takes one positional")
    assert import_entry(module) is perform_regression


def test_convert_is_dispatched_with_the_complete_settings_dict():
    """The bug was a bare folder here; the entry now reads the whole dict, so
    the keys the convert panel offers reach the pipeline."""
    from spacr.cli import MODULES, import_entry

    module = MODULES["convert"]
    assert module.call_style == "settings"
    assert _accepts(import_entry(module), {})


def test_unknown_settings_type_resolves_to_nothing():
    """An unknown key must not fall through to a module that runs."""
    from spacr.cli import _unknown_module_message, resolve_module

    assert resolve_module("no_such_module") is None
    assert "no_such_module" in _unknown_module_message("no_such_module")


# ---------------------------------------------------------------------------
# Packaging: no console script may point at a module that does not exist
# ---------------------------------------------------------------------------

def test_every_console_script_target_module_exists():
    """`sim=spacr.app_sim:gui_sim` pointed at a file deleted long ago, so the
    installed `sim` command died with ImportError."""
    import ast
    import importlib.util
    from pathlib import Path

    setup_py = Path(__file__).resolve().parents[1] / "setup.py"
    tree = ast.parse(setup_py.read_text())

    scripts = []
    for node in ast.walk(tree):
        if isinstance(node, ast.keyword) and node.arg == "entry_points":
            for k, v in zip(node.value.keys, node.value.values):
                if getattr(k, "value", None) == "console_scripts":
                    scripts = [e.value for e in v.elts]
    assert scripts, "could not parse console_scripts out of setup.py"

    missing = []
    for entry in scripts:
        module = entry.split("=", 1)[1].split(":", 1)[0]
        if importlib.util.find_spec(module) is None:
            missing.append(entry)
    assert not missing, f"console scripts pointing at missing modules: {missing}"


# ---------------------------------------------------------------------------
# validate.run_preflight trailer
# ---------------------------------------------------------------------------

def test_run_preflight_default_trailer_names_the_dry_run_flag():
    from spacr.validate import run_preflight
    lines = []
    run_preflight({"src": "/nope"}, "mask", printer=lines.append)
    assert "dry_run=False" in lines[-1]


def test_run_preflight_accepts_a_caller_specific_trailer():
    """spacr-run --dry-run must not tell a CLI user to set dry_run=False."""
    from spacr.validate import run_preflight
    lines = []
    run_preflight({"src": "/nope"}, "mask", printer=lines.append,
                  trailer="--dry-run: stopping here. Drop the flag to run.")
    assert lines[-1] == "--dry-run: stopping here. Drop the flag to run."


def test_run_preflight_empty_trailer_prints_nothing_extra():
    from spacr.validate import run_preflight
    with_trailer, without = [], []
    run_preflight({"src": "/nope"}, "mask", printer=with_trailer.append)
    run_preflight({"src": "/nope"}, "mask", printer=without.append, trailer="")
    # the default adds a blank line plus the trailer
    assert len(with_trailer) == len(without) + 2


def test_run_preflight_still_returns_the_problems_it_found():
    from spacr.validate import run_preflight
    problems = run_preflight({"src": "/definitely/not/here"}, "mask",
                             printer=lambda _: None, trailer="")
    assert problems, "a missing src must be reported as a problem"
