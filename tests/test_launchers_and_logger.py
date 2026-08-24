"""
Tests for the tiny launcher / logger surface of spacr:

  * spacr.__main__: build_parser, `spacr version` dispatch, the Qt tab each
    window subcommand opens on, and error handling for unknown commands.
  * spacr.logger: configure_logger idempotence, log_function_call
    decorator, _safe_repr truncation.
  * The installed console scripts: every declared command names a target
    that imports and is callable.

The launcher section changed subject with the interface. It used to import
the seven ``spacr.app_*`` modules and assert each exported a callable
``start_*_app``; those modules opened Tk windows and are gone, and the
commands they backed are gone from ``console_scripts`` with them. The
question the section asks is unchanged — does every command spaCR installs
actually start something — so it now asks it of the entry-point table
itself, which is the only place the answer still lives.
"""
from __future__ import annotations

import io
import logging
import sys
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# 1. spacr.__main__: CLI parser + dispatch
# ---------------------------------------------------------------------------

def test_build_parser_default_command():
    from spacr.__main__ import build_parser
    parser = build_parser()
    args = parser.parse_args([])
    assert args.command == "gui"


@pytest.mark.parametrize("cmd", [
    "gui", "mask", "measure", "classify", "annotate",
    "sequencing", "umap", "make-masks", "version",
])
def test_build_parser_accepts_all_known_commands(cmd):
    from spacr.__main__ import build_parser
    args = build_parser().parse_args([cmd])
    assert args.command == cmd


def test_build_parser_rejects_unknown_command():
    from spacr.__main__ import build_parser
    with pytest.raises(SystemExit):
        build_parser().parse_args(["unknown_command"])


def test_main_version_command_prints_and_exits_zero(capsys):
    from spacr.__main__ import main
    rc = main(["version"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "spacr version" in out.lower() or "python" in out.lower()


#: subcommand -> the tab key handed to ``spacr.qt.run``; ``gui`` names none
#: and lands on Home.
_COMMAND_TABS = [
    ("gui", []),
    ("mask", ["mask"]),
    ("measure", ["measure"]),
    ("classify", ["classify_merged"]),
    ("annotate", ["annotate"]),
    ("sequencing", ["map_barcodes"]),
    ("umap", ["umap"]),
    ("make-masks", ["make_masks"]),
]


@pytest.mark.parametrize("cmd,tab_argv", _COMMAND_TABS)
def test_main_dispatches_each_command_to_its_tab(cmd, tab_argv):
    """`spacr <cmd>` opens the Qt application on the screen that command names.

    Each of these used to start its own Tk window. They are tabs now, so the
    dispatch is asserted on the argv `main` hands to `spacr.qt.run`.
    """
    from spacr.__main__ import main

    with patch("spacr.qt.run", return_value=0) as spy:
        rc = main([cmd])
    assert rc == 0
    spy.assert_called_once()
    passed = spy.call_args.args[0] if spy.call_args.args else []
    assert list(passed) == tab_argv


@pytest.mark.parametrize("tab_argv", [a for _c, a in _COMMAND_TABS if a])
def test_each_command_tab_exists_in_the_app_registry(tab_argv):
    """A tab key the registry does not know silently opens Home instead."""
    from spacr.qt.app import APPS

    assert tab_argv[0] in {entry[0] for entry in APPS}


# ---------------------------------------------------------------------------
# 2. spacr.logger: configure_logger + log_function_call decorator
# ---------------------------------------------------------------------------

def test_configure_logger_creates_logger_at_requested_level(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    from spacr import logger as mod
    lg = mod.configure_logger(name="spacr.test.a", level=logging.DEBUG)
    assert isinstance(lg, logging.Logger)
    assert lg.level == logging.DEBUG
    assert lg.handlers, "logger should have at least one handler"


def test_configure_logger_is_idempotent(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    from spacr import logger as mod
    a = mod.configure_logger(name="spacr.test.b")
    handler_count_before = len(a.handlers)
    b = mod.configure_logger(name="spacr.test.b")
    assert b is a
    assert len(b.handlers) == handler_count_before, (
        "configure_logger should not stack handlers on repeated calls"
    )


def test_configure_logger_stream_handler_optional(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    from spacr import logger as mod
    lg = mod.configure_logger(name="spacr.test.c", stream=True)
    has_stream = any(
        isinstance(h, logging.StreamHandler) and not hasattr(h, "baseFilename")
        for h in lg.handlers
    )
    assert has_stream


def test_safe_repr_truncates_long_values():
    from spacr.logger import _safe_repr
    s = _safe_repr("x" * 500, max_length=50)
    assert len(s) <= 50
    assert s.endswith("...")


def test_safe_repr_handles_unreprable_objects():
    from spacr.logger import _safe_repr

    class Boom:
        def __repr__(self):
            raise RuntimeError("nope")

    s = _safe_repr(Boom())
    assert "unreprable" in s


def test_log_function_call_wraps_and_returns_result(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    from spacr.logger import log_function_call

    @log_function_call
    def add(a, b):
        return a + b

    assert add(2, 3) == 5


def test_log_function_call_reraises_exceptions(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    from spacr.logger import log_function_call

    @log_function_call
    def kaboom():
        raise ValueError("expected")

    with pytest.raises(ValueError, match="expected"):
        kaboom()


# ---------------------------------------------------------------------------
# 3. Every console script names something that imports and is callable
# ---------------------------------------------------------------------------

#: The commands that used to open a Tk window. Each is a tab in `spacr` now,
#: so an install that still declares one would put a command on the user's
#: PATH that raises ModuleNotFoundError the moment it is typed.
_DELETED_TK_COMMANDS = frozenset(
    {"mask", "measure", "make_masks", "annotate", "classify"})


def _declared_console_scripts():
    """``{command: 'module:attr'}`` parsed out of setup.py's source.

    Read from the file rather than from installed metadata because an
    editable install keeps whatever entry points it was built with, so a
    command deleted from setup.py today still answers `importlib.metadata`
    until someone reinstalls.
    """
    import ast
    import pathlib

    setup_py = pathlib.Path(__file__).resolve().parents[1] / "setup.py"
    tree = ast.parse(setup_py.read_text(encoding="utf-8"))
    scripts = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        for key, value in zip(node.keys, node.values):
            if getattr(key, "value", None) != "console_scripts":
                continue
            for element in value.elts:
                command, target = element.value.split("=", 1)
                scripts[command.strip()] = target.strip()
    return scripts


def test_setup_py_declares_console_scripts():
    """Every assertion below is vacuous if the parse found nothing."""
    scripts = _declared_console_scripts()
    assert len(scripts) >= 10, f"only parsed {len(scripts)} console scripts"
    assert scripts.get("spacr") == "spacr.qt:run"


def test_no_console_script_survives_from_the_deleted_tk_launchers():
    """The five bare-word Tk commands went with the windows they opened."""
    declared = set(_declared_console_scripts())
    assert not (declared & _DELETED_TK_COMMANDS), (
        f"setup.py still installs {sorted(declared & _DELETED_TK_COMMANDS)}, "
        "whose launcher modules no longer exist")


def test_the_headless_commands_are_still_installed():
    """The GUI going Qt-only must not take the no-display commands with it."""
    declared = set(_declared_console_scripts())
    for command in ("spacr", "spacr-run", "spacr-repro", "spacr-workspace",
                    "spacr-doctor", "spacr-remote"):
        assert command in declared, f"{command} is no longer installed"


@pytest.mark.parametrize("command", sorted(_declared_console_scripts()))
def test_every_console_script_names_a_callable_that_exists(command):
    """A command whose target attribute is gone dies with AttributeError the
    first time it is typed, and nothing else in the suite types them.

    Only the attribute half is asserted here. Whether the target MODULE still
    exists is the subject of
    ``test_gui_dispatch_call_styles.test_every_console_script_target_module_exists``,
    so a deleted module is reported once, by name, in one place rather than
    once per command that pointed at it.
    """
    import importlib.util

    target = _declared_console_scripts()[command]
    module_name, _, attr = target.partition(":")
    if importlib.util.find_spec(module_name) is None:
        pytest.skip(f"{module_name} is missing; that is the other test's subject")
    module = importlib.import_module(module_name)
    entry = getattr(module, attr, None)
    assert callable(entry), f"{command} -> {target} is not callable"


def test_make_masks_is_reachable_as_a_qt_screen():
    """What ``app_make_masks.initiate_make_mask_app`` used to be asked here.

    The Tk mask editor is `MakeMasksScreen`, opened from the registry key
    `make_masks`, so the hand-correction tool is still reachable rather than
    having been dropped along with its launcher.
    """
    from spacr.qt.app import APPS
    from spacr.qt.screens.make_masks import MakeMasksScreen

    assert "make_masks" in {entry[0] for entry in APPS}
    assert isinstance(MakeMasksScreen, type)
