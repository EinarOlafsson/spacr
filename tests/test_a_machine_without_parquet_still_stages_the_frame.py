"""Staging a frame on a machine where no Parquet engine is installed.

The durable copy is written columnar when pandas can, because Parquet both
writes and reads several times faster than CSV at a fraction of the size. An
engine is not guaranteed to be there, though, and a handoff that raised on
that would take the whole merge down over a file format. It falls back to CSV
and SAYS SO in the line it reports -- a run that suddenly takes three minutes
to write its measurements should say why in its own log rather than leave the
user timing it.
"""
from __future__ import annotations

import builtins
import os

import pandas as pd
import pytest

from spacr import frame_handoff


@pytest.fixture
def no_columnar_engine(monkeypatch):
    """Neither Parquet engine importable, as on a bare conda environment."""
    real_import = builtins.__import__

    def _import(name, *args, **kwargs):
        if name in ("pyarrow", "fastparquet"):
            raise ImportError(f"No module named {name!r}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _import)


def _frame():
    return pd.DataFrame({"objectID": [1, 2, 3],
                         "cell_area": [10.5, 11.0, 9.25]})


def test_neither_engine_installed_means_no_columnar_format(
        no_columnar_engine):
    assert frame_handoff._columnar_engine() is None


def test_an_installed_engine_is_named_rather_than_assumed():
    """The control: this environment does have one, and it is reported."""
    engine = frame_handoff._columnar_engine()
    assert engine in (None, "pyarrow", "fastparquet")
    if engine is not None:
        __import__(engine)          # the name must be importable, not a guess


def test_the_frame_is_staged_as_csv_when_no_engine_is_installed(
        tmp_path, no_columnar_engine):
    """The artefact still exists, still readable, just not columnar."""
    lines = []
    frame = _frame()

    path = frame_handoff.stage(frame, tmp_path, "merged", report=lines.append)

    assert path.endswith(".csv")
    assert os.path.isfile(path)
    pd.testing.assert_frame_equal(pd.read_csv(path), frame)
    frame_handoff.release(path)


def test_the_fallback_to_csv_is_reported_not_silent(tmp_path,
                                                    no_columnar_engine):
    """A run that got slower has to say what changed."""
    lines = []

    path = frame_handoff.stage(_frame(), tmp_path, "merged",
                               report=lines.append)

    assert len(lines) == 1
    assert "no Parquet engine installed, so CSV" in lines[0]
    frame_handoff.release(path)


def test_the_in_memory_offer_stands_whichever_format_was_written(
        tmp_path, no_columnar_engine):
    """The whole point of staging survives the missing engine.

    A reader that would have parsed the CSV gets the object, so falling back
    on format must not also fall back to a parse.
    """
    frame = _frame()

    path = frame_handoff.stage(frame, tmp_path, "merged", report=None)
    try:
        assert frame_handoff.held(path) is frame
        assert "3 rows x 2 columns" in frame_handoff.describe(path)
    finally:
        frame_handoff.release(path)
