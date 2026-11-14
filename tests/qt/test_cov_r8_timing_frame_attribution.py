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
