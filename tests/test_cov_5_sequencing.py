"""The barcode mismatch budget: the cache, and the run-wide announcement.

A read whose barcode carries one sequencing error is still that guide's
read, and throwing it away costs real library coverage. Two pieces of that
budget are pinned here: the per-sequence cache, which is what keeps a
non-zero budget from turning every read into a linear scan of the whole
reference, and the line that tells the user a budget is in force -- silence
would leave an ambiguous read looking like a mapping failure.
"""
from __future__ import annotations

import pandas as pd

import spacr.sequencing as sequencing
from spacr.sequencing import _map_within


class _CountingBudget:
    """A mismatch budget that records how often it is compared.

    ``_map_within`` compares the running error count against the budget once
    per reference candidate it scans, so the comparison count IS the number
    of scans -- which is what the cache is there to hold down.
    """

    def __init__(self, value):
        self.value = int(value)
        self.comparisons = 0

    def __lt__(self, other):          # reflected from ``wrong > budget``
        self.comparisons += 1
        return self.value < other

    def __ge__(self, other):          # reflected from ``wrong <= budget``
        self.comparisons += 1
        return self.value >= other


def test_a_repeated_near_miss_is_scanned_only_once():
    """The second copy of an inexact sequence costs no further scanning.

    Without the cache every occurrence of a barcode that is not an exact hit
    re-scans every reference of its length -- on a lane of millions of reads
    that is the difference between a usable budget and an unusable one. The
    budget object counts the comparisons the scan makes, so ten copies of a
    sequence costing exactly what one copy costs is the cache doing its job.
    """
    reference = {"AAAA": "guide_1", "CCCC": "guide_2", "GGGG": "guide_3"}

    once = _CountingBudget(1)
    assert _map_within(reference, ["AAAT"], once) == ["guide_1"]
    assert once.comparisons > 0

    ten = _CountingBudget(1)
    assert _map_within(reference, ["AAAT"] * 10, ten) == ["guide_1"] * 10
    assert ten.comparisons == once.comparisons


def test_an_ambiguous_near_miss_is_unassigned_both_times():
    """Two references within budget give NA, and the cache keeps it NA.

    Handing the read to whichever reference was found first would put one
    guide's counts on another; the cached answer has to carry the same
    refusal as the computed one.
    """
    reference = {"AAAA": "guide_1", "AAAC": "guide_2"}
    out = _map_within(reference, ["AAAG", "AAAG"], 1)
    assert len(out) == 2
    assert all(value is pd.NA for value in out)


def test_a_budget_over_zero_says_so_before_any_read_is_mapped(
        tmp_path, monkeypatch, capsys):
    """A run with a non-zero budget announces it and arms the workers.

    The budget reaches the worker processes through a module global set
    before the pool is forked, so a run that failed to set it would map
    every read at zero and give no sign of it. An empty source folder is
    enough: the announcement and the global are both settled before any
    sample is looked at.
    """
    monkeypatch.setattr(sequencing, "BARCODE_MISMATCHES", 0, raising=False)
    src = tmp_path / "fastq"
    src.mkdir()

    sequencing.generate_barecode_mapping({
        "src": str(src), "mode": "single", "single_direction": "R1",
        "barcode_mismatches": 2, "save_h5": False, "n_jobs": 1,
    })

    assert sequencing.BARCODE_MISMATCHES == 2
    printed = capsys.readouterr().out
    assert "up to 2 mismatched" in printed
    assert "unassigned" in printed
