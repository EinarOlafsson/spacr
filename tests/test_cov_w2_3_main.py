"""``python -m spacr`` dispatch: which screen each subcommand opens on.

Every window subcommand opens the one Qt application and lands on a named
tab, so a script still saying ``python -m spacr mask`` reaches the Mask
screen rather than an import error. The dispatch table and the parser's
choices are two lists that have to agree, and this drives the path taken
when they do not.
"""
from __future__ import annotations

import argparse
import runpy
import sys

import pytest

from spacr import __main__ as M


@pytest.fixture
def opened(monkeypatch):
    """Record the argv handed to the Qt launcher instead of opening it."""
    import spacr.qt as qt

    seen = []

    def fake_run(argv=None):
        seen.append(list(argv or []))
        return 0

    monkeypatch.setattr(qt, "run", fake_run)
    return seen


@pytest.mark.parametrize("command,key", [
    ("mask", "mask"),
    ("measure", "measure"),
    ("classify", "classify_merged"),
    ("annotate", "annotate"),
    ("sequencing", "map_barcodes"),
    ("umap", "umap"),
    ("make-masks", "make_masks"),
])
def test_a_window_subcommand_opens_the_application_on_its_tab(
        command, key, opened):
    """The subcommand's tab key is the launcher's first positional."""
    assert M.main([command]) == 0
    assert opened == [[key]]


def test_the_bare_gui_subcommand_opens_on_home(opened):
    """``gui`` names no tab, so the launcher gets no positional at all."""
    assert M.main([]) == 0
    assert M.main(["gui"]) == 0
    assert opened == [[], []]


def test_the_version_subcommand_prints_the_version_and_stops(capsys):
    """No window is opened for a question about the installed version."""
    from spacr.version import version_str

    assert M.main(["version"]) == 0
    assert capsys.readouterr().out.strip() == version_str


def test_a_choice_the_dispatch_table_does_not_know_is_an_argument_error(
        monkeypatch, capsys):
    """A command accepted by the parser but not dispatched fails loudly.

    The parser's ``choices`` list and the dispatch below it are two lists
    that can drift apart. When they do, the user must be told rather than
    handed a silent success.
    """
    def parser_with_an_undispatched_choice():
        parser = argparse.ArgumentParser(prog="spacr")
        parser.add_argument("command", nargs="?", default="gui",
                            choices=["gui", "brand_new_screen"])
        return parser

    monkeypatch.setattr(M, "build_parser", parser_with_an_undispatched_choice)
    with pytest.raises(SystemExit) as excinfo:
        M.main(["brand_new_screen"])
    assert excinfo.value.code == 2
    assert "Unknown command: brand_new_screen" in capsys.readouterr().err


def test_running_the_module_exits_with_the_code_main_returned(
        monkeypatch, capsys):
    """``python -m spacr version`` exits 0 after printing the version."""
    from spacr.version import version_str

    monkeypatch.setattr(sys, "argv", ["spacr", "version"])
    with pytest.raises(SystemExit) as excinfo:
        runpy.run_module("spacr.__main__", run_name="__main__")
    assert excinfo.value.code == 0
    assert capsys.readouterr().out.strip() == version_str
