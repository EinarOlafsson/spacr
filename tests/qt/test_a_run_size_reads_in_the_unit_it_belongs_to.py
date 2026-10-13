"""What the run-history table prints in its size and duration columns.

Both formatters answer over the whole range a real run produces, and both
have to answer at all for a value that is not a number -- a run recorded
before the field existed carries ``None``, and a table cell reading ``None``
is worse than one reading a dash.

The size column is binary units all the way up. A screen's merged arrays run
to hundreds of gigabytes and a four-plate run's output folder past a
terabyte, so the top unit is not decoration: a number printed in GiB at that
size is five digits nobody reads.
"""
from __future__ import annotations

import math
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

pytestmark = pytest.mark.qt

from spacr.qt.screens import run_history as RH                   # noqa: E402


KIB = 1024.0


@pytest.mark.parametrize("value,expected", [
    (0, "0 B"),
    (512, "512 B"),
    (2 * KIB, "2.0 KiB"),
    (3.5 * KIB ** 2, "3.5 MiB"),
    (12 * KIB ** 3, "12.0 GiB"),
])
def test_each_size_reads_in_its_own_unit(value, expected):
    assert RH._bytes(value) == expected


def test_a_run_bigger_than_a_terabyte_reads_in_tebibytes():
    """The top unit: there is nothing above it to promote to."""
    assert RH._bytes(2.5 * KIB ** 4) == "2.5 TiB"
    assert RH._bytes(4096 * KIB ** 4) == "4096.0 TiB"


def test_the_unit_changes_exactly_at_the_boundary():
    assert RH._bytes(KIB - 1) == "1023 B"
    assert RH._bytes(KIB) == "1.0 KiB"
    assert RH._bytes(KIB ** 4 - 1) == "1024.0 GiB"
    assert RH._bytes(KIB ** 4) == "1.0 TiB"


def test_a_size_that_is_not_a_number_reads_as_a_dash():
    assert RH._bytes(None) == "—"
    assert RH._bytes("not a size") == "—"


def test_a_duration_that_is_not_a_number_reads_as_a_dash():
    assert RH._seconds(None) == "—"
    assert RH._seconds("a while") == "—"


@pytest.mark.parametrize("value,expected", [
    (0.5, "0.5s"),
    (59.9, "59.9s"),
    (60, "1m 00s"),
    (605, "10m 05s"),
    (3600, "1h 00m"),
    (7 * 3600 + 9 * 60, "7h 09m"),
])
def test_each_duration_reads_in_its_own_unit(value, expected):
    assert RH._seconds(value) == expected


def test_a_size_that_is_not_finite_still_answers():
    """NaN reaches here from an empty aggregate rather than from a bug."""
    answer = RH._bytes(float("nan"))
    assert answer.endswith("TiB")
    assert math.isnan(float(answer.split()[0]))
