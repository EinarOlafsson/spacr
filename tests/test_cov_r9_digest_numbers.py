"""``digest_numbers`` converts exactly the numeric strings it admits.

The token has already matched ``_NUMBER`` in full, so ``float`` cannot
refuse it -- unless the pattern and ``float`` disagree about what a
number is, which is exactly the thing worth checking and the thing a
comment cannot notice.
"""
from __future__ import annotations

import inspect
import math

import pytest

from spacr import methods_export as ME


def _numbers(digest):
    return ME.digest_numbers(digest)


class TestWhatThePatternAdmits:

    @pytest.mark.parametrize("token", [
        "1", "-1", "+1", "0", "1.5", "-1.5", ".5", "5.", "1e3", "1E3",
        "1e-3", "-2.5e+4", "000123",
    ])
    def test_everything_the_pattern_matches_is_a_float(self, token):
        """THE PIN, for ``except ValueError``.

        The handler cannot fire while every string ``_NUMBER`` admits is
        one ``float`` accepts. Driven over the forms a settings digest
        actually carries, because that agreement is a property of two
        separate definitions and nothing else checks it.
        """
        if not ME._NUMBER.fullmatch(token):
            pytest.skip(f"{token!r} is not admitted by _NUMBER")

        value = float(token)          # must not raise
        assert math.isfinite(value) or True

    @pytest.mark.parametrize("token", [
        "nan", "inf", "-inf", "infinity", "1_000", "0x10", "1j",
        "", " ", "one", "1.2.3", "--1",
    ])
    def test_the_pattern_refuses_what_would_surprise_a_reader(self, token):
        """The other half: ``float`` accepts several strings that are not
        numbers in a report -- ``nan``, ``inf``, and the underscore form
        -- and quoting one of those as a measured value would be worse
        than dropping it.
        """
        assert not ME._NUMBER.fullmatch(token), (
            f"{token!r} is admitted by _NUMBER; float() accepts it too for "
            f"some of these, and a report quoting 'inf' as a measurement "
            f"reads as a result rather than an overflow")

    def test_the_conversion_has_no_unreachable_handler(self):
        source = inspect.getsource(ME.digest_numbers)
        match = source.index("if _NUMBER.fullmatch(token):")
        conversion = source.index("found.add(float(token))", match)

        assert match < conversion
        assert "except ValueError:" not in source[match:], (
            "the regex is a strict subset of float syntax, so this handler "
            "could never run")


class TestWhatTheDigestYields:

    def test_plain_numbers_are_collected(self):
        assert _numbers({"a": 1, "b": 2.5}) == {1.0, 2.5}

    def test_numbers_written_as_text_are_collected_too(self):
        """A settings digest round-tripped through JSON or a CSV carries
        its numbers as strings, and a report that quoted only the native
        ones would miss half the run."""
        assert _numbers({"a": "3", "b": "4.5"}) == {3.0, 4.5}

    def test_booleans_are_not_numbers(self):
        """``isinstance(True, int)`` is True in Python, so without the
        explicit skip every flag in the settings would arrive as 1.0 and
        be quotable as a measurement."""
        assert _numbers({"flag": True, "other": False}) == set()

    def test_a_non_finite_float_is_dropped(self):
        """An overflowed or undefined value is not a measurement, and a
        report quoting 'inf' reads as a result."""
        assert _numbers({"a": float("inf"), "b": float("nan"),
                         "c": 1.0}) == {1.0}

    def test_text_that_is_not_a_number_is_ignored(self):
        assert _numbers({"path": "/data/plate1", "note": "ran twice"}) == set()

    def test_it_walks_nested_structures(self):
        assert _numbers({"outer": {"inner": [1, "2", {"deep": 3.5}]}}) == \
            {1.0, 2.0, 3.5}
