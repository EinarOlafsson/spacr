"""`python -m spacr.qt` is a documented way in, so it is exercised as one.

The package docstring advertises two launchers -- the `spacr-qt` console
script and `python -m spacr.qt` -- and only the first has a test. The second
is five lines that nothing imports, which is exactly the shape of entry point
that rots: a rename of `run`, or a stray top-level statement, is invisible
until a user types the command.

Driven through `runpy` with the module's real `__main__` name so the guard
at the bottom runs, and with `spacr.qt.run` replaced by a recorder -- the one
substitution that has to be made, because the real call opens a window and
enters an event loop.
"""

import runpy
import sys

import pytest

import spacr.qt


def _launch(monkeypatch, argv, *, exit_code=0):
    """Run `python -m spacr.qt` with `argv`, returning what `run` received."""
    seen = []

    def recorder(args=None):
        seen.append(args)
        return exit_code

    monkeypatch.setattr(spacr.qt, "run", recorder)
    monkeypatch.setattr(sys, "argv", ["spacr.qt", *argv])
    with pytest.raises(SystemExit) as left:
        runpy.run_module("spacr.qt.__main__", run_name="__main__")
    return seen, left.value.code


def test_the_module_launcher_hands_the_arguments_to_run(monkeypatch):
    """Everything after the module name reaches `run`, and nothing else does.

    `sys.argv[0]` is the launcher, not a user argument; passing the whole of
    `sys.argv` would make `run` see a spurious first setting.
    """
    seen, code = _launch(monkeypatch, ["--version", "measure"])
    assert seen == [["--version", "measure"]]
    assert code == 0


def test_a_bare_launch_passes_an_empty_argument_list(monkeypatch):
    """With no arguments `run` gets `[]`, not `None` and not the module name."""
    seen, code = _launch(monkeypatch, [])
    assert seen == [[]]
    assert code == 0


def test_the_exit_status_is_the_one_run_returned(monkeypatch):
    """A non-zero return becomes the process's exit status.

    A launcher that always exits 0 makes a failed start look like a clean
    one to any shell or CI step that checks.
    """
    _seen, code = _launch(monkeypatch, ["--broken"], exit_code=3)
    assert code == 3


def test_importing_the_launcher_does_not_launch_anything(monkeypatch):
    """Imported rather than run as `__main__`, it starts no GUI.

    The guard is what keeps `import spacr.qt.__main__` -- which any tool that
    walks the package does -- from opening a window.
    """
    def refuse():
        raise AssertionError("importing the launcher started the GUI")

    monkeypatch.setattr(spacr.qt, "run", refuse)
    monkeypatch.delitem(sys.modules, "spacr.qt.__main__", raising=False)
    namespace = runpy.run_module("spacr.qt.__main__", run_name="not_main")
    assert namespace["run"] is refuse
