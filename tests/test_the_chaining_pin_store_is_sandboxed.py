"""A test must never read or write the user's real chaining pins.

Chaining pins are the paths a user typed into a module (Mask's `src`, say),
kept in XDG state storage so they survive a restart. Until 2026-09-15 no
sandbox covered that store: the real `~/.local/state/spacr/chaining/pins.json`
on the maintainer's machine pinned Mask to a deleted pytest directory, the
app opened Mask there, and a tutorial recording showed the path in its Source
field. See `_isolated_chaining_pin_store` in tests/conftest.py, and the
matching XDG_STATE_HOME isolation in tools/tutorials/capture_refresh.py.
"""
from __future__ import annotations

import ast
import os
from pathlib import Path

from spacr import chaining

REPO = Path(__file__).resolve().parents[1]


def _real_pin_file() -> str:
    """The file `state_path()` resolves to with no override in force."""
    xdg = os.environ.get("XDG_STATE_HOME", "").strip()
    root = (os.path.join(os.path.expanduser(xdg), "spacr") if xdg else
            os.path.join(os.path.expanduser("~"), ".local", "state", "spacr"))
    return os.path.abspath(os.path.join(root, "chaining", "pins.json"))


def test_a_test_does_not_resolve_the_real_pin_file():
    assert os.environ.get(chaining.PIN_STATE_ENV), (
        "no pin sandbox is in force for this test")
    assert chaining.state_path() != _real_pin_file()
    assert chaining.pin_store().path != _real_pin_file()


def test_a_test_starts_with_no_pins():
    # The user's real file (or an earlier test's) would show up here as a
    # remembered Mask source this test never chose.
    store = chaining.pin_store()
    assert store.pins("mask") == {}
    assert store.pinned("mask", "src") is None


def test_a_pin_written_in_a_test_lands_in_that_tests_sandbox(tmp_path):
    chosen = str(tmp_path / "plate1")
    store = chaining.pin_store()
    store.pin("mask", "src", chosen)
    assert os.path.isfile(store.path)
    assert os.path.abspath(store.path) != _real_pin_file()
    assert chaining.PinStore(store.path).pinned("mask", "src") == chosen


def test_the_tutorial_capture_isolates_state_storage_too():
    # XDG_CONFIG_HOME moves QSettings only; the pins follow XDG_STATE_HOME.
    source = (REPO / "tools" / "tutorials" / "capture_refresh.py").read_text(
        encoding="utf-8")
    environment_keys = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Dict):
            keys = {k.value for k in node.keys
                    if isinstance(k, ast.Constant) and isinstance(k.value, str)}
            if "XDG_CONFIG_HOME" in keys:
                environment_keys |= keys
    assert "XDG_CONFIG_HOME" in environment_keys
    assert "XDG_STATE_HOME" in environment_keys
    assert "os.environ.pop('SPACR_CHAINING_PINS', None)" in source
