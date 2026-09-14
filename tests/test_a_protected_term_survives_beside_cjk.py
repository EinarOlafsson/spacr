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


# ---------------------------------------------------------------------------
# The other half: what a TARGET may name that the source only implied.
# ---------------------------------------------------------------------------

def _syntax_preserved():
    sys.path.insert(0, TOOLS)
    try:
        import build_i18n_catalogs
        return build_i18n_catalogs._syntax_preserved
    finally:
        sys.path.remove(TOOLS)


def test_chinese_may_say_guide_rna_where_the_english_says_guides():
    """引导 alone is "guidance/lead" and is ambiguous in a CRISPR screen.

    All 57 rows doing this on 2026-09-14 followed a source saying guide,
    guides, gRNA or sgRNA -- no exceptions -- so the allowance is triggered
    by the source rather than granted to RNA generally.
    """
    assert _syntax_preserved()(
        "uses a sparse prior when most guides have small effects",
        "使用稀疏先验，当大多数引导 RNA具有小效应时",
    ), ("the Chinese expansion of 'guides' to '引导 RNA' is domain-correct and "
        "must not be rejected as an invented product name")


def test_a_target_may_canonicalise_an_acronym_the_source_lower_cased():
    assert _syntax_preserved()(
        "Writes a FOLDER: the figure as pdf and png",
        "写一个文件夹：图像为 PDF 和 png",
    ), "the token is the source's own; only its case changed"


def test_an_invented_product_name_is_still_refused():
    """The hazard the rule exists for, unchanged.

    RNA is licensed only by a source that mentions a guide, and nothing
    licenses naming a segmentation library the English never mentioned.
    """
    assert not _syntax_preserved()(
        "segments the objects in the channel",
        "使用 Cellpose 分割通道中的对象",
    ), "a product the source never named must still be refused"
    assert not _syntax_preserved()(
        "counts the rows in the table",
        "统计表中的 RNA 行数",
    ), "RNA without a guide in the source must still be refused"


#: Probes contributed by the work session on 2026-09-14, kept because the
#: `guidebook` one is what separates an allowance from a hole and the original
#: tests did not have it.
IMPLIED_TERM_CASES = [
    ("uses a sparse prior when most guides have small effects",
     "使用稀疏先验，当大多数引导 RNA具有小效应时", True,
     "RNA licensed by 'guides' -- the real case"),
    ("counts the rows in the table", "统计表中的 RNA 行数", False,
     "RNA with no guide anywhere in the source"),
    ("segments the objects in the channel", "使用 Cellpose 分割通道中的对象", False,
     "a product the English never named"),
    ("open the guidebook for an explanation", "打开 RNA 手册以获得解释", False,
     "'guidebook' must not license RNA -- the substring trap"),
    ("most guides have small effects", "大多数引导 DNA具有小效应", False,
     "DNA is not the licensed token"),
    ("each sgRNA is counted once", "每条 sgRNA 引导 RNA计数一次", True,
     "sgRNA licenses RNA"),
]


@pytest.mark.parametrize("source,target,allowed,why", IMPLIED_TERM_CASES)
def test_the_implied_term_allowance_is_an_allowance_not_a_hole(
        source, target, allowed, why):
    got = _syntax_preserved()(source, target)
    assert got is allowed, (
        f"{why}: expected {'accepted' if allowed else 'refused'}, got "
        f"{'accepted' if got else 'refused'}. _IMPLIED_BY_THE_SOURCE licenses "
        f"an acronym only where the source earns it; widening that to 'RNA "
        f"anywhere' would let a model invent a domain term.")
