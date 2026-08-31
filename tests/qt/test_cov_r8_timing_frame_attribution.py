"""Which frame the import timer blames, when the environment lies.

THE BUG. The timer decided a stack frame belonged to spaCR by asking
whether its path contained the segment "/spacr/". This project's own
conda environment is named `spacr`, so every file in it -- the standard
library, site-packages, pytest itself -- lives under
`.../envs/spacr/lib/python3.12/...` and matched.

The consequence is not a missing attribution but a WRONG one: "3 s of
torch, asked by lib/python3.12/site-packages/_pytest/python.py:167".
That is worse than saying nothing, because it names an innocent file
with a plausible-looking line number, and the whole point of the feature
is to tell a maintainer which of their own modules is costing the
startup. Anyone whose home directory, checkout or virtualenv contains
the word had the same wrong answer.

These are written against paths rather than against a live stack, so
they hold on a machine whose environment is NOT called spacr -- which is
where this defect is invisible.
"""
from __future__ import annotations

import os
import sysconfig

import pytest

from spacr.qt.timing import _the_spacr_frame


def _reset_the_cache(monkeypatch):
    """The roots are resolved once and cached; tests must not inherit one."""
    import spacr.qt.timing as timing

    monkeypatch.setattr(timing, "_SPACR_ROOT", None)
    monkeypatch.setattr(timing, "_LIBRARY_DIRS", ())


def test_a_file_in_the_installed_package_is_named_relatively(monkeypatch):
    _reset_the_cache(monkeypatch)
    import spacr

    root = os.path.dirname(os.path.abspath(spacr.__file__))
    assert _the_spacr_frame(os.path.join(root, "qt", "app.py")) == "qt/app.py"


def test_an_environment_merely_named_spacr_is_not_the_package(monkeypatch):
    """THE REPORTED DEFECT, stated directly.

    A stdlib file inside an environment called `spacr` contains the
    segment and must still not be attributed.
    """
    _reset_the_cache(monkeypatch)
    stdlib = sysconfig.get_paths()["purelib"]
    intruder = os.path.join(stdlib, "_pytest", "python.py")
    if "/spacr/" not in intruder:
        pytest.skip("this interpreter's environment is not named spacr, "
                    "which is exactly the machine the bug hides on")
    assert _the_spacr_frame(intruder) == "", (
        "a site-packages file was attributed to spaCR because the "
        "environment happens to be called spacr")


def test_a_stdlib_path_with_the_segment_is_refused_whatever_the_env(
        monkeypatch):
    """The same claim, forced, so it also runs on other machines.

    The library directories are injected, so the assertion does not
    depend on what this particular environment is called.
    """
    import spacr.qt.timing as timing

    monkeypatch.setattr(timing, "_SPACR_ROOT", "/real/pkg/spacr/")
    monkeypatch.setattr(timing, "_LIBRARY_DIRS",
                        ("/opt/envs/spacr/lib/python3.12/",))
    assert timing._the_spacr_frame(
        "/opt/envs/spacr/lib/python3.12/site-packages/_pytest/python.py") == ""
    assert timing._the_spacr_frame("/real/pkg/spacr/qt/app.py") == "qt/app.py"


def test_a_source_tree_at_an_unusual_path_is_still_recognised(monkeypatch):
    """The second way to qualify, and why it is kept.

    A checkout that is not the installed package -- a worktree, a test's
    synthetic frame -- still carries the segment and is not inside the
    interpreter's libraries, so it is named.
    """
    import spacr.qt.timing as timing

    monkeypatch.setattr(timing, "_SPACR_ROOT", "/real/pkg/spacr/")
    monkeypatch.setattr(timing, "_LIBRARY_DIRS",
                        ("/opt/envs/spacr/lib/python3.12/",))
    assert timing._the_spacr_frame(
        "/nowhere/spacr/qt/asked_here.py") == "qt/asked_here.py"


def test_a_path_with_no_segment_at_all_is_not_named(monkeypatch):
    _reset_the_cache(monkeypatch)
    assert _the_spacr_frame("/nowhere/elsewhere/asked_here.py") == ""


def test_the_timer_never_blames_itself(monkeypatch):
    """`timing.py` is the observer; naming it would say nothing."""
    _reset_the_cache(monkeypatch)
    import spacr

    root = os.path.dirname(os.path.abspath(spacr.__file__))
    assert _the_spacr_frame(os.path.join(root, "qt", "timing.py")) == ""


def test_an_empty_path_is_not_named(monkeypatch):
    """`co_filename` can be empty for a synthesised code object."""
    _reset_the_cache(monkeypatch)
    assert _the_spacr_frame("") == ""


def test_a_package_that_cannot_be_located_does_not_raise(monkeypatch):
    """This runs inside find_spec, on every import the process makes.

    Raising here would break importing itself, so a failure to resolve
    the root has to degrade to the segment test rather than propagate.
    """
    import builtins

    import spacr.qt.timing as timing

    _reset_the_cache(monkeypatch)
    real_import = builtins.__import__

    def refuse(name, *args, **kwargs):
        if name == "spacr":
            raise ImportError("no spacr here")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", refuse)
    assert timing._the_spacr_frame("/nowhere/spacr/qt/app.py") == "qt/app.py"
