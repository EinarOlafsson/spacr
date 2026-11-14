"""Table-driven tests of how ``spacr.methods_export`` reads numbers out of a digest.

Two units: the ``_NUMBER`` pattern that decides whether a string leaf is a
number, and :func:`digest_numbers`, which walks a settings digest and
collects the numbers a methods section may quote.  The pattern must admit
only what ``float`` parses (so the conversion cannot fail) and must refuse
spellings ``float`` accepts but a report must not quote (``nan``, ``inf``,
``1_000``).
"""
from __future__ import annotations

from spacr import methods_export as ME


def test_the_number_pattern_admits_plain_decimals_only(subtests):
    """Every admitted token parses with float to the expected value; the rest are refused."""
    admitted = [
        ("1", 1.0), ("-1", -1.0), ("+1", 1.0), ("0", 0.0), ("1.5", 1.5),
        ("-1.5", -1.5), ("1e3", 1000.0), ("1E3", 1000.0), ("1e-3", 0.001),
        ("-2.5e+4", -25000.0), ("000123", 123.0), ("0.35", 0.35),
        ("1e-6", 1e-6), ("+2", 2.0), ("1E5", 100000.0),
    ]
    for token, value in admitted:
        with subtests.test(msg=f"admits {token!r}"):
            assert ME._NUMBER.fullmatch(token)
            assert float(token) == value

    refused = [
        "nan", "inf", "-inf", "infinity", "1_000", "0x10", "1j", "", " ",
        "one", "1.2.3", "--1", ".5", "5.", "3.", "otsu", "1/2", "3px",
        "nan cells", "1e", "e3",
    ]
    for token in refused:
        with subtests.test(msg=f"refuses {token!r}"):
            assert ME._NUMBER.fullmatch(token) is None


def test_digest_numbers_collects_finite_numbers_and_numeric_text(subtests):
    """Numbers and fully numeric strings are collected; bools, non-finite and prose are not."""
    cases = [
        ("plain numbers", {"a": 1, "b": 2.5}, {1.0, 2.5}),
        ("numbers written as text", {"a": "3", "b": "4.5"}, {3.0, 4.5}),
        ("surrounding whitespace is stripped", {"a": " 7 "}, {7.0}),
        ("mixed native and text", {"threshold": "0.35", "count": 7},
         {0.35, 7.0}),
        ("booleans are not numbers", {"flag": True, "other": False}, set()),
        ("non-finite floats are dropped",
         {"a": float("inf"), "b": float("nan"), "c": 1.0}, {1.0}),
        ("non-numeric text is ignored",
         {"path": "/data/plate1", "note": "ran twice"}, set()),
        ("digits inside prose are not harvested",
         {"method": "otsu", "note": "about 3 cells"}, set()),
        ("nested dicts and lists are walked",
         {"outer": {"inner": [1, "2", {"deep": 3.5}]}}, {1.0, 2.0, 3.5}),
        ("empty digest", {}, set()),
    ]
    for case_id, digest, expected in cases:
        with subtests.test(msg=case_id):
            found = ME.digest_numbers(digest)
            assert found == expected
            assert all(type(value) is float for value in found)
