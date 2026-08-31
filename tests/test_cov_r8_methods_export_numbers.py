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


class TestWhatCountsAsSupported:

    def test_a_number_the_digest_holds_is_supported(self):
        assert _supported("0.05", {0.05}) is True

    def test_a_number_the_digest_does_not_hold_is_not(self):
        assert _supported("0.99", {0.05}) is False

    def test_a_rounding_of_a_held_number_is_supported(self):
        """A measurement quoted to fewer decimals is the same measurement."""
        assert _supported("0.043", {0.0432198}) is True

    def test_a_token_that_is_not_a_number_at_all_is_not_supported(self):
        """THE REACHABLE GUARD.

        `_supported` is a predicate over a string, and a caller that
        hands it something unparseable must get False rather than a
        ValueError -- the check runs over generated prose, and one odd
        token must not stop the whole methods section being verified.
        """
        assert _supported("not a number", {0.05}) is False
        assert _supported("", {0.05}) is False
        assert _supported("1.2.3", {0.05}) is False

    def test_a_non_finite_token_is_not_supported(self):
        """`nan` and `inf` parse but are not measurements.

        Accepting them would let "inf" pass as a supported figure.
        """
        assert _supported("nan", {0.05}) is False
        assert _supported("inf", {0.05}) is False
        assert _supported("-inf", {0.05}) is False


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


class TestTheConversionGuardThatCannotFire:
    """`except ValueError` inside `digest_numbers` is unreachable.

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

    def test_the_pattern_is_still_the_one_that_guards_the_conversion(self):
        import inspect

        source = inspect.getsource(digest_numbers)
        assert "_NUMBER.fullmatch(token)" in source, (
            "the conversion is no longer guarded by the pattern, so its "
            "ValueError arm may now be reachable")
