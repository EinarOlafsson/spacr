"""The scorecard as a model-zoo surface, and the sentence owed when it is absent.

370 asks for FOUR SURFACES FROM ONE SOURCE -- Hugging Face, the tooltip, the
API page and the Zoo screen -- and gives the reason: "if the tooltip and the
API page can disagree, they eventually will", citing 366's finding that six
README tiles pointed at three different API pages. The CSV is the source, the
entry's `metrics` is loaded from it, and every surface renders the entry. So
these tests pin the entry, not the renderings.

TWO TRAPS 370 NAMES, both tested here because both have been paid for once
somewhere else:

* THE ZOO IMPORTS WITHOUT TORCH, deliberately. Browsing models and reading a
  scorecard must work on a machine with neither torch nor cellpose; only
  re-running an evaluation may need them.
* A MISSING SCORECARD MUST SAY SO rather than show blanks. A model with no
  numbers did not score zero, and a table of empty cells reads as the second.
"""
from __future__ import annotations

import builtins
import sys

import pytest

from spacr.model_zoo import ModelEntry
from spacr.scorecard import (NO_SCORECARD, read_scorecard_csv,
                             scorecard_is_present)

CSV = (
    "metric,finetuned,vanilla,delta,n_fields,n_objects,holdout,holdout_version\n"
    "n_truth,120,120,,4,120,toxo-pv-round3,2026-09-10\n"
    "f1,0.86,0.32,0.54,4,120,toxo-pv-round3,2026-09-10\n"
    "dice,0.98,0.92,0.06,4,120,toxo-pv-round3,2026-09-10\n"
)


def test_the_csv_round_trips_into_metrics():
    metrics = read_scorecard_csv(CSV)
    assert metrics["f1"]["finetuned"] == pytest.approx(0.86)
    assert metrics["f1"]["vanilla"] == pytest.approx(0.32)
    assert metrics["holdout"] == "toxo-pv-round3"


def test_a_count_is_not_a_float():
    """"on 12517.0 objects" reads as a rounding, and a count is not one."""
    metrics = read_scorecard_csv(CSV)
    assert metrics["n_objects"] == 120
    assert isinstance(metrics["n_objects"], int)


def test_a_scored_model_says_what_it_scored_against():
    """A SCORECARD WITHOUT ITS SET IS A NUMBER WITHOUT A UNIT. Two people
    quoting an F1 have said nothing to each other unless they scored the
    same masks."""
    entry = ModelEntry(key="m", name="M", metrics=read_scorecard_csv(CSV))
    assert entry.scorecard_known
    assert entry.scorecard_holdout == "toxo-pv-round3 @ 2026-09-10"
    assert any("hold-out set" in line for line in entry.scorecard_lines())


def test_an_unscored_model_says_so_rather_than_showing_blanks():
    entry = ModelEntry(key="m", name="M")
    assert not entry.scorecard_known
    assert entry.scorecard_lines() == [NO_SCORECARD]
    assert entry.scorecard_holdout == ""


def test_a_scorecard_of_only_counts_is_not_a_scorecard():
    """Rows exist but nothing was measured -- `n_truth` alone is bookkeeping,
    not accuracy, and reporting it as a scorecard would be worse than
    reporting none."""
    counts_only = ("metric,finetuned,vanilla,delta\n"
                   "n_truth,,,\n")
    assert not scorecard_is_present(read_scorecard_csv(counts_only))


def test_reading_a_scorecard_imports_neither_torch_nor_cellpose():
    """The Zoo imports without torch and a test asserts it; this is the same
    promise for the scorecard that now hangs off every entry.

    THE MODULES ARE PUT BACK, AND THAT IS NOT TIDINESS. Popping `torch` out
    of `sys.modules` drops the last reference to it, and the garbage
    collector then runs a C extension's teardown while the process is still
    using it. Measured: this file followed by
    `tests/test_test_suite_hygiene.py` segfaulted inside `ast.parse`, with
    the fault frame reading "Garbage-collecting" -- in a test that has
    nothing to do with torch, several minutes after this one passed.

    Restoring the same module OBJECTS means nothing is collected and no
    teardown runs. The pop still does its job: it is what stops an
    already-imported torch from hiding a fresh import from the guard below.
    """
    poisoned = [m for m in list(sys.modules)
                if m.split(".")[0] in ("torch", "cellpose")]
    saved = {name: sys.modules.pop(name) for name in poisoned}
    real = builtins.__import__

    def guard(name, *args, **kwargs):
        if name.split(".")[0] in ("torch", "cellpose"):
            raise AssertionError(f"reading a scorecard imported {name}")
        return real(name, *args, **kwargs)

    builtins.__import__ = guard
    try:
        entry = ModelEntry(key="m", name="M", metrics=read_scorecard_csv(CSV))
        lines = entry.scorecard_lines()
        blank = ModelEntry(key="b", name="B").scorecard_lines()
        arrived = [m for m in sys.modules
                   if m.split(".")[0] in ("torch", "cellpose")]
    finally:
        builtins.__import__ = real
        sys.modules.update(saved)

    # THE GUARD IS NOT THE ASSERTION, and a test whose only failure mode is
    # someone else's exception is one that passes when the code under it
    # stops running at all. So: the work actually happened, and neither
    # package arrived while it did.
    assert lines and any("F1" in line for line in lines)
    assert all(isinstance(line, str) for line in blank)
    assert not arrived


def test_the_lines_lead_with_what_decides_a_choice():
    """The tooltip has the least room of the four surfaces, so it leads with
    the handful of numbers that decide a choice."""
    entry = ModelEntry(key="m", name="M", metrics=read_scorecard_csv(CSV))
    lines = entry.scorecard_lines()
    assert "F1" in lines[0]
    assert any("vs stock" in line for line in lines)
