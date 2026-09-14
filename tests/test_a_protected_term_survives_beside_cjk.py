"""A protected term must stay visible when the script beside it has no spaces.

THE ROOT CAUSE, hit three separate ways on 2026-09-13 before anyone named it:
Python's ``\\w`` is Unicode-aware, so CJK and Hangul characters ARE word
characters. Every "word boundary" idiom -- ``\\b``, ``(?<!\\w)``, ``(?!\\w)`` --
therefore stops firing beside them, and a protected term next to Chinese or
Korean text becomes INVISIBLE to the checker. The checker then reports a
correct, human-reviewed translation as having dropped the term.

That is a false negative, which is the expensive kind: the next step after a
false rejection is "repairing" a translation that was never broken. Two of the
three instances were found only because someone hand-wrote a record and could
not understand why it failed.

    Mask는              a trailing Hangul particle defeated (?!\\w)
    是Run History        a leading CJK character defeated (?<!\\w)
    Apple Silicon에서     the same, in a record that was already correct

The boundary actually wanted is "not part of a LATIN word": ``Masking`` must
not match ``Mask`` and ``CUDA_PATH`` must not match ``CUDA``, while a Hangul
particle is a perfectly good place for a term to end.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

TOOLS = str(Path(__file__).resolve().parent.parent / "tools")


def _protect_re():
    sys.path.insert(0, TOOLS)
    try:
        import build_i18n_catalogs
        return build_i18n_catalogs._PRODUCT_PROTECT_RE
    finally:
        sys.path.remove(TOOLS)


#: (text, the term it contains). Every one of these was invisible to the
#: checker before the ASCII-boundary fix.
BESIDE_CJK = [
    ("是Run History 界面所搜索的对象", "Run History"),
    ("Format Converter는 체크포인트를 다시 엽니다", "Format Converter"),
    ("각 TIFF를 다시 엽니다", "TIFF"),
    ("地图Plate Queue的设置", "Plate Queue"),
    ("CUDA에서 실행됩니다", "CUDA"),
]

#: Latin neighbours that must still suppress the match, because the term is
#: genuinely part of a longer identifier or word here.
INSIDE_A_LATIN_WORD = ["Masking", "CUDA_PATH", "TIFFs_out", "spaCRish"]


@pytest.mark.parametrize("text,term", BESIDE_CJK)
def test_a_term_beside_cjk_or_hangul_is_still_seen(text, term):
    found = _protect_re().search(text)
    assert found is not None and found.group(0).startswith(term), (
        f"{term!r} is invisible inside {text!r}. A checker that cannot see it "
        f"reports a CORRECT translation as having dropped the term, and the "
        f"next person repairs a translation that was never broken.")


@pytest.mark.parametrize("text", INSIDE_A_LATIN_WORD)
def test_a_term_inside_a_latin_word_is_not_matched(text):
    assert _protect_re().search(text) is None, (
        f"{text!r} matched, so the boundary is too loose: protecting a "
        f"substring of an ordinary word or identifier would freeze text that "
        f"has to be translatable")


def test_the_boundary_is_ascii_rather_than_unicode_aware():
    """The property, stated once, so a future edit cannot quietly undo it."""
    pattern = _protect_re().pattern
    assert r"(?<!\w)" not in pattern and r"(?!\w)" not in pattern, (
        r"the term boundary is back to \w, which is Unicode-aware and "
        r"therefore treats CJK and Hangul as word characters -- see this "
        r"module's docstring for the three bugs that caused")
