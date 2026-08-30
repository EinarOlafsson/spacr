"""Two refusals in settings validation, and two in the pyqtgraph scene check.

Both pairs are messages a user reads when something is wrong, and neither had
a test -- the usual shape for validation code, where the valid case is
exercised by every other test in the suite and the complaints by none.
"""
from __future__ import annotations

import builtins
import os

import pytest


# ---------------------------------------------------------------------------
# validate._check_regression_output_src
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("raw", [3, 3.5, ["/data/out"], {"path": "/data/out"},
                                 object()])
def test_a_src_that_is_not_a_path_string_is_refused(raw):
    """Line 573. The value is quoted back, which is what makes it findable.

    ``src`` reaches here from a settings file, a notebook cell or a panel, and
    a non-string is what a user gets by typing a bare number into a text field
    or by pasting a list. The complaint names the value so they can see WHICH
    setting they are looking at, and it is an ERROR rather than a warning
    because there is no sensible way to proceed.
    """
    from spacr.validate import _check_regression_output_src

    problems = _check_regression_output_src(raw)

    assert len(problems) == 1
    assert problems[0].setting == "src"
    assert "not a path string" in problems[0].message
    assert repr(raw) in problems[0].message


@pytest.mark.parametrize("placeholder", ["path", "/path/to/src"])
def test_the_shipped_placeholder_src_is_refused(placeholder):
    """Line 581. The default that always fails is refused before it runs.

    These two strings are what the shipped settings carry, and a run started
    on them writes to a folder called "path" beside the working directory --
    which is the failure mode instruction 181 is named after. Catching it in
    validation is the difference between a clear message and a mystery folder.
    """
    from spacr.validate import _check_regression_output_src

    problems = _check_regression_output_src(placeholder)

    assert len(problems) == 1
    assert "placeholder" in problems[0].message


@pytest.mark.parametrize("raw", [None, "", "   "])
def test_an_unset_src_is_not_a_problem(raw):
    """The early return: blank means "write beside the first count table"."""
    from spacr.validate import _check_regression_output_src

    assert _check_regression_output_src(raw) == []


def test_a_real_directory_is_not_a_problem(tmp_path):
    """The valid case, so the three refusals above are visibly the exceptions."""
    from spacr.validate import _check_regression_output_src

    assert _check_regression_output_src(str(tmp_path)) == []


# ---------------------------------------------------------------------------
# figures.scene.pyqtgraph_ready
# ---------------------------------------------------------------------------

def test_a_missing_pyqtgraph_is_reported_rather_than_raised(monkeypatch):
    """Lines 188-189: (False, reason) instead of an ImportError.

    ``pyqtgraph_ready`` is asked BEFORE a scene is built, precisely so the
    caller can choose the matplotlib path instead. Raising would defeat the
    question: the answer "no, and here is why" is the entire product of this
    function, and the reason string is what the user sees when a figure
    silently comes back in the other style.
    """
    from spacr.figures import scene

    real_import = builtins.__import__

    def refusing(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "pyqtgraph":
            raise ImportError("no module named pyqtgraph")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", refusing)

    ready, reason = scene.pyqtgraph_ready()

    assert ready is False
    assert "pyqtgraph is unavailable here" in reason
    assert "no module named pyqtgraph" in reason


def test_a_qapplication_that_will_not_start_is_reported(monkeypatch):
    """Lines 196-197: the second refusal, with its own distinct reason.

    The two failures are kept apart on purpose. "pyqtgraph is not installed"
    and "no display could be opened" call for completely different actions
    from the user, and one message covering both would tell them to install a
    package they already have.
    """
    from spacr.figures import scene

    monkeypatch.setattr(scene, "_APPLICATION", None, raising=False)

    class _RefusingQApplication:
        @staticmethod
        def instance():
            return None

        def __init__(self, *_args):
            raise RuntimeError("no display could be opened")

    import types
    real_import = builtins.__import__

    def patched(name, globals=None, locals=None, fromlist=(), level=0):
        # The function only verifies that pyqtgraph can be imported here; a
        # sentinel keeps this test on the QApplication branch without making
        # pyqtgraph's own Qt-binding import depend on our deliberately tiny
        # QtWidgets shim.
        if name == "pyqtgraph":
            return types.ModuleType("pyqtgraph")
        if name == "PySide6.QtWidgets" or (
                name == "PySide6" and "QtWidgets" in (fromlist or ())):
            module = types.ModuleType("PySide6.QtWidgets")
            module.QApplication = _RefusingQApplication
            if name == "PySide6":
                shim = types.ModuleType("PySide6")
                shim.QtWidgets = module
                return shim
            return module
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", patched)

    ready, reason = scene.pyqtgraph_ready()

    assert ready is False
    assert "no QApplication could be started" in reason
    assert "pyqtgraph is unavailable" not in reason
