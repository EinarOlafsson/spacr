"""
Tests for the tiny launcher / logger surface of spacr:

  * spacr.__main__: build_parser, `spacr version` dispatch, and error
    handling for unknown commands.
  * spacr.logger: configure_logger idempotence, log_function_call
    decorator, _safe_repr truncation.
  * The seven `app_*` modules: every start_*_app entry point exists,
    is callable, and gets exercised by build_parser dispatch.
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


@pytest.mark.parametrize("cmd,func_path", [
    ("gui", "spacr.gui.gui_app"),
    ("mask", "spacr.app_mask.start_mask_app"),
    ("measure", "spacr.app_measure.start_measure_app"),
    ("classify", "spacr.app_classify.start_classify_app"),
    ("annotate", "spacr.app_annotate.start_annotate_app"),
    ("sequencing", "spacr.app_sequencing.start_seq_app"),
    ("umap", "spacr.app_umap.start_umap_app"),
    ("make-masks", "spacr.app_make_masks.start_make_mask_app"),
])
def test_main_dispatches_each_command_to_its_entry_point(cmd, func_path):
    """`spacr <cmd>` must import and invoke the matching start_*_app."""
    from spacr.__main__ import main
    with patch(func_path) as spy:
        rc = main([cmd])
    assert rc == 0
    spy.assert_called_once()


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
# 3. Every launcher module exports its start_*_app function
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mod_name,fn", [
    ("app_mask",       "start_mask_app"),
    ("app_measure",    "start_measure_app"),
    ("app_classify",   "start_classify_app"),
    ("app_annotate",   "start_annotate_app"),
    ("app_sequencing", "start_seq_app"),
    ("app_umap",       "start_umap_app"),
    ("app_make_masks", "start_make_mask_app"),
])
def test_launcher_start_function_is_callable(mod_name, fn):
    import importlib
    mod = importlib.import_module(f"spacr.{mod_name}")
    entry = getattr(mod, fn, None)
    assert callable(entry), f"spacr.{mod_name}.{fn} must be callable"


def test_app_make_masks_initiate_helper_exists():
    """The parent-frame initiator hook AnnotateApp / MainApp uses."""
    import spacr.app_make_masks as m
    assert callable(getattr(m, "initiate_make_mask_app", None))
