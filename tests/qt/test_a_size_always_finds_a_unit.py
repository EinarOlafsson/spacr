"""``_readable_size`` always returns from inside its loop.

Instruction 288. The function had a trailing ``return`` after the loop,
marked ``# pragma: no cover - loop``. The reason was right: "GB" is the
last unit and the loop's condition carries ``or unit == "GB"``, so the
final iteration returns whatever the size is and the loop cannot fall out
of the bottom.

That rests on two facts about one line each, and either could change
without anyone noticing the trailing path came back to life -- so both
are pinned here.
"""
from __future__ import annotations

import inspect
import random

import pytest

from spacr.qt.widgets import sweep_runs
from spacr.qt.widgets.sweep_runs import _readable_size


@pytest.mark.parametrize("total,expected", [
    (0, "0 B"),
    (1, "1 B"),
    (1023, "1023 B"),
    (1024, "1.0 KB"),
    (10 * 1024, "10 KB"),
    (1024 ** 2, "1.0 MB"),
    (1024 ** 3, "1.0 GB"),
])
def test_the_ordinary_sizes(total, expected):
    assert _readable_size(total) == expected


def test_a_size_past_the_last_unit_stays_in_that_unit():
    """THE CASE the trailing return looked like it was for.

    It is not: the loop's own final iteration handles it, and says GB
    rather than falling through to a second copy of the same format.
    """
    huge = _readable_size(2 ** 63 - 1)
    assert huge.endswith(" GB")
    assert huge == "8589934592 GB"


def test_a_negative_total_is_floored_at_zero():
    """`max(0, int(total))` -- so a negative cannot walk the units
    backwards or produce a negative size."""
    assert _readable_size(-5) == "0 B"
    assert _readable_size(-(10 ** 9)) == "0 B"


def test_gb_is_still_the_last_unit():
    """PREMISE 1. A unit added after GB would leave the loop able to
    finish without returning."""
    source = inspect.getsource(_readable_size)
    assert '("B", "KB", "MB", "GB")' in source


def test_the_last_unit_short_circuits_the_size_check():
    """PREMISE 2. Without `or unit == "GB"` a size of 1024 GB or more
    would fail the condition on the final pass and fall out of the loop."""
    source = inspect.getsource(_readable_size)
    assert 'or unit == "GB"' in source


def test_no_value_falls_out_of_the_loop():
    """The sweep that settled it, kept smaller.

    Twenty thousand ran when the trailing return was removed; two
    thousand here, so a change to the unit table fails a test rather than
    reaching the assertion that replaced it.
    """
    random.seed(4)
    for _ in range(2000):
        total = random.randint(0, 10 ** random.randint(0, 19))
        answer = _readable_size(total)
        assert isinstance(answer, str) and answer
        assert answer.rsplit(" ", 1)[1] in ("B", "KB", "MB", "GB")


def test_what_replaced_the_return_is_not_a_silent_fallthrough():
    """If the premises ever break, the function must fail loudly rather
    than format a size in a unit it never chose."""
    source = inspect.getsource(sweep_runs._readable_size)
    assert "raise AssertionError" in source
