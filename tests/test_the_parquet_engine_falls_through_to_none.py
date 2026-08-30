"""Which Parquet engine is available, and what happens when neither is.

``_columnar_engine`` tries pyarrow then fastparquet and returns None for
neither. The order is a preference, and the None is what lets ``stage`` fall
back to a format every install can read -- so a machine with no columnar engine
still hands frames off, it just does so more slowly.

Returning a name that cannot be imported would fail inside ``to_parquet``,
deep in pandas, with a message about the engine rather than about the install.
"""
from __future__ import annotations

import builtins
import importlib.util

import pytest


def test_the_answer_is_exactly_what_this_machine_can_import():
    """The ordinary answer here, cross-checked against the install.

    The contract is not "some string" -- it is a name pandas can pass to
    ``to_parquet``. So the assertion is an equality against what is actually
    importable on this machine, which catches a typo'd name and a stale
    preference order alike, and holds on a machine with neither engine.
    """
    import importlib.util

    from spacr.frame_handoff import _columnar_engine

    engine = _columnar_engine()
    available = [name for name in ("pyarrow", "fastparquet")
                 if importlib.util.find_spec(name) is not None]

    assert engine == (available[0] if available else None)
    if engine is not None:
        module = __import__(engine)
        assert module.__name__ == engine         # the name really is importable


def test_the_first_engine_wins_when_both_are_present(monkeypatch):
    """The order is a preference, not an accident.

    pyarrow is tried first, and a machine with both must not get a different
    answer from one run to the next -- a handoff written by one engine and
    read by the other is where dtype differences show up.
    """
    from spacr import frame_handoff

    real_import = builtins.__import__

    def both_present(name, *args, **kwargs):
        if name in ("pyarrow", "fastparquet"):
            return object()
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", both_present)

    assert frame_handoff._columnar_engine() == "pyarrow"


def test_the_second_engine_is_tried_when_the_first_is_absent(monkeypatch):
    """The ``continue``: one missing engine does not end the search."""
    from spacr import frame_handoff

    real_import = builtins.__import__

    def only_fastparquet(name, *args, **kwargs):
        if name == "pyarrow":
            raise ImportError("no pyarrow here")
        if name == "fastparquet":
            return object()
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", only_fastparquet)

    assert frame_handoff._columnar_engine() == "fastparquet"


def test_neither_engine_reports_none(monkeypatch):
    """The loop running out, which is what a minimal install produces.

    None is the signal to use a format every install can read. Returning a
    name that cannot be imported would fail inside pandas' to_parquet, with a
    message about the engine rather than about the install.
    """
    from spacr import frame_handoff

    real_import = builtins.__import__

    def neither(name, *args, **kwargs):
        if name in ("pyarrow", "fastparquet"):
            raise ImportError(f"no {name} here")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", neither)

    assert frame_handoff._columnar_engine() is None


def test_a_frame_is_staged_even_with_no_columnar_engine(tmp_path, monkeypatch):
    """The fallback that None exists for: the handoff still happens.

    A machine without pyarrow must still be able to pass a frame between
    stages -- more slowly, not not at all.
    """
    import pandas as pd

    from spacr import frame_handoff

    monkeypatch.setattr(frame_handoff, "_columnar_engine", lambda: None)

    written = frame_handoff.stage(pd.DataFrame({"a": [1, 2], "b": ["x", "y"]}),
                                  tmp_path, "handoff", report=lambda *_a: None)

    assert written
    import os
    assert os.path.isfile(written)
