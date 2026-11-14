"""Deciding which numbers in generated prose are supported by the digest.

The methods exporter writes prose and then checks every number in it
against the run's own digest. A number that is not supported is a
number the text invented, and an invented number in a methods section
is the worst kind of error this package can produce -- it reads as a
measurement.

Two `except ValueError` arms guard the float conversions. One cannot
fire; the other can, and its own contract is worth asserting.
"""
from __future__ import annotations

import random

import pytest

from spacr.methods_export import (_NUMBER, _supported, digest_numbers,
                                  extract_numbers)


@pytest.mark.parametrize("token, expected", [
    pytest.param("0.05", True, id="a-number-the-digest-holds"),
    pytest.param("0.99", False, id="a-number-the-digest-does-not-hold"),
    pytest.param("0.043", True, id="a-rounding-of-a-held-number-is-the-same-measurement"),
    pytest.param("not a number", False, id="text-is-not-a-number"),
    pytest.param("", False, id="empty-is-not-a-number"),
    pytest.param("1.2.3", False, id="a-version-string-is-not-a-number"),
    pytest.param("nan", False, id="nan-parses-but-is-not-a-measurement"),
    pytest.param("inf", False, id="inf-parses-but-is-not-a-measurement"),
    pytest.param("-inf", False, id="minus-inf-parses-but-is-not-a-measurement"),
])
def test_a_token_is_supported_only_when_the_digest_holds_that_measurement(token, expected):
    """`_supported` is a predicate over a string: an unparseable or non-finite
    token must give False, not a ValueError, because the check runs over
    generated prose and one odd token must not stop the whole methods section
    being verified."""
    assert _supported(token, {0.05, 0.0432198}) is expected


class TestExtractingNumbersFromProse:
    """`extract_numbers` returns the number TOKENS as written.

    Strings, not floats, and deliberately: the caller reports which
    tokens in the prose were unsupported, and "0.050" and "0.05" are
    different things to a reader even though they are one float.
    """

    def test_the_numbers_in_a_sentence_are_found_as_written(self):
        found = extract_numbers("We kept 1234 cells at a q of 0.05.")
        assert "1234" in found
        assert "0.05" in found

    def test_a_sentence_with_no_numbers_yields_none(self):
        assert extract_numbers("We kept every cell.") == []

    def test_scientific_notation_is_read(self):
        assert "1e-05" in extract_numbers("a q of 1e-05")

    def test_a_named_string_is_not_mistaken_for_a_number(self):
        """The `strings` argument names values that are not measurements."""
        found = extract_numbers("plate1 held 12 wells", ("plate1",))
        assert "12" in found


class TestReadingTheDigestsOwnNumbers:
    """`digest_numbers` walks the run digest and collects every figure.

    These are the values the prose is allowed to quote, so anything it
    misses becomes an "unsupported" number in a methods section that was
    in fact measured.
    """

    def test_numbers_nested_anywhere_are_found(self):
        digest = {"a": 1234, "b": {"c": 0.05}, "d": ["7", {"e": 2.5}]}
        found = digest_numbers(digest)
        assert {1234.0, 0.05, 7.0, 2.5} <= found

    def test_a_string_that_is_not_a_number_is_skipped(self):
        assert digest_numbers({"note": "no numbers here"}) == set()

    def test_an_empty_digest_yields_nothing(self):
        assert digest_numbers({}) == set()


class TestTheNumericConversion:
    """The numeric regex is a sufficient precondition for ``float``.

    A token only reaches `float()` after `_NUMBER.fullmatch` has
    accepted it, and that pattern --

        [-+]?\\d+(?:\\.\\d+)?(?:[eE][-+]?\\d+)?

    -- is a strict subset of what `float()` parses. Every string it
    matches has digits, at most one dot between digits, and an optional
    signed integer exponent.

    Pinned by search rather than argument alone.
    """

    def test_nothing_the_pattern_accepts_is_rejected_by_float(self):
        rng = random.Random(20260831)
        chars = "0123456789.eE+-"
        tried = 0
        for _ in range(200000):
            token = "".join(rng.choice(chars)
                            for _ in range(rng.randint(1, 7)))
            if not _NUMBER.fullmatch(token):
                continue
            tried += 1
            try:
                float(token)
            except ValueError:                       # pragma: no cover
                pytest.fail(
                    f"{token!r} matches _NUMBER but float() rejects it; the "
                    "ValueError guard in extract_numbers is now reachable")
        assert tried > 1000, "the search accepted too few tokens to mean much"

