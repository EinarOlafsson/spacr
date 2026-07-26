"""The Tk dispatcher must call each worker with the signature it actually has.

Two settings types raised TypeError on every single run and nobody noticed,
because ``function_gui_wrapper`` catches the exception and posts it to a queue
the user is not necessarily watching:

  * ``regression`` was dispatched with ``imports=2``, i.e.
    ``perform_regression(src=..., settings=...)``, but ``ml.perform_regression``
    takes one positional ``settings``.
  * ``convert`` was dispatched with ``imports=1``, i.e.
    ``process_non_tif_non_2D_images(settings=...)``, but ``io.process_non_tif_
    non_2D_images`` takes a bare ``folder`` path.

And an unrecognised ``imports`` value fell off the end of the if/elif with no
else, so the worker was never called and the wrapper returned as though the
module had completed successfully.

These tests bind each dispatched callable to its real signature, so a future
change to either side breaks here instead of in front of a user.
"""
from __future__ import annotations

import inspect
import queue

import pytest

import matplotlib
matplotlib.use("Agg")


# ---------------------------------------------------------------------------
# function_gui_wrapper call styles
# ---------------------------------------------------------------------------

def _wrapper():
    from spacr.gui_utils import function_gui_wrapper
    return function_gui_wrapper


def test_call_style_1_passes_settings_as_a_keyword():
    seen = {}

    def worker(settings=None):
        seen["settings"] = settings

    _wrapper()(worker, {"src": "/x"}, queue.Queue(), queue.Queue(), 1)
    assert seen["settings"] == {"src": "/x"}


def test_call_style_2_passes_src_and_settings():
    seen = {}

    def worker(src=None, settings=None):
        seen["src"], seen["settings"] = src, settings

    _wrapper()(worker, {"src": "/x"}, queue.Queue(), queue.Queue(), 2)
    assert seen == {"src": "/x", "settings": {"src": "/x"}}


def test_call_style_3_passes_a_bare_folder():
    """The style `convert` needs: one positional path, no settings kwarg."""
    seen = {}

    def worker(folder):
        seen["folder"] = folder

    _wrapper()(worker, {"src": "/images"}, queue.Queue(), queue.Queue(), 3)
    assert seen["folder"] == "/images"


def test_unknown_call_style_reports_instead_of_silently_running_nothing():
    """The bug: no else branch, so the worker was skipped and the run 'passed'."""
    calls = []
    q = queue.Queue()

    def worker(settings=None):
        calls.append(settings)

    _wrapper()(worker, {"src": "/x"}, q, queue.Queue(), 99)

    assert calls == [], "worker must not run under an unknown call style"
    msg = q.get_nowait()
    assert "unknown call style" in msg
    assert "99" in msg


def test_wrapper_restores_plt_show_even_when_the_worker_raises():
    import matplotlib.pyplot as plt

    original = plt.show

    def boom(settings=None):
        raise RuntimeError("worker exploded")

    _wrapper()(boom, {}, queue.Queue(), queue.Queue(), 1)
    assert plt.show is original


# ---------------------------------------------------------------------------
# The dispatch table must agree with the real signatures
# ---------------------------------------------------------------------------

def _dispatch(settings_type):
    """Return (function, imports) for a settings_type without running anything.

    ``run_function_gui`` does its imports and dispatch inline, then calls the
    wrapper, so we patch the wrapper and capture what it was handed.
    """
    import spacr.gui_utils as GU

    captured = {}

    def fake_wrapper(function=None, settings=None, q=None, fig_queue=None, imports=1):
        captured["function"] = function
        captured["imports"] = imports

    real_wrapper = GU.function_gui_wrapper
    real_stdout = GU.process_stdout_stderr
    GU.function_gui_wrapper = fake_wrapper
    GU.process_stdout_stderr = lambda q: None
    try:
        class _Flag:
            value = 0
        GU.run_function_gui(settings_type, {"src": "/x"}, queue.Queue(),
                            queue.Queue(), _Flag())
    finally:
        GU.function_gui_wrapper = real_wrapper
        GU.process_stdout_stderr = real_stdout
    return captured["function"], captured["imports"]


def _accepts(func, **kwargs):
    """True if ``func(**kwargs)`` would bind without a TypeError."""
    try:
        inspect.signature(func).bind(**kwargs)
        return True
    except TypeError:
        return False


def _accepts_positional(func, n):
    try:
        inspect.signature(func).bind(*range(n))
        return True
    except TypeError:
        return False


@pytest.mark.parametrize("settings_type", [
    "mask", "measure", "classify", "train_cellpose", "ml_analyze",
    "cellpose_masks", "cellpose_all", "map_barcodes", "regression",
    "recruitment", "umap", "analyze_plaques", "convert",
])
def test_every_dispatched_worker_binds_to_its_call_style(settings_type):
    """The whole table, not just the two that were broken."""
    func, imports = _dispatch(settings_type)

    if imports == 1:
        assert _accepts(func, settings={}), (
            f"{settings_type}: dispatched as function(settings=...) but "
            f"{func.__name__}{inspect.signature(func)} does not accept it")
    elif imports == 2:
        assert _accepts(func, src="/x", settings={}), (
            f"{settings_type}: dispatched as function(src=..., settings=...) "
            f"but {func.__name__}{inspect.signature(func)} does not accept it")
    elif imports == 3:
        assert _accepts_positional(func, 1), (
            f"{settings_type}: dispatched as function(path) but "
            f"{func.__name__}{inspect.signature(func)} does not accept it")
    else:
        pytest.fail(f"{settings_type}: unknown call style {imports}")


def test_regression_is_dispatched_with_a_single_settings_argument():
    from spacr.ml import perform_regression
    func, imports = _dispatch("regression")
    assert func is perform_regression
    assert imports == 1, "perform_regression(settings) takes one positional"


def test_convert_is_dispatched_with_a_bare_folder():
    from spacr.io import process_non_tif_non_2D_images
    func, imports = _dispatch("convert")
    assert func is process_non_tif_non_2D_images
    assert imports == 3, "process_non_tif_non_2D_images(folder) takes a path"


def test_unknown_settings_type_raises():
    with pytest.raises(ValueError, match="Invalid settings type"):
        _dispatch("no_such_module")


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
