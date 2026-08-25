"""Regex detection degrades instead of failing the import dialog.

Detection runs while the user is looking at a folder they have just dropped
in. Anything that raises here is a dialog that cannot open, so the two guards
that matter are: a pattern that will not compile is treated as naming no
groups, and a failure inside the inference step leaves the built-in fallback
exactly as it was.
"""
from __future__ import annotations

import pytest

from spacr.qt.regex_detect import _group_names, auto_detect_regex

#: Names from a microscope with no built-in pattern; inference reads their
#: shared shape and names the fields the import actually uses.
_UNKNOWN_SCOPE = [
    "plate1_A01_f01_c1.tif",
    "plate1_A01_f02_c1.tif",
    "plate1_B02_f01_c2.tif",
    "plate2_C03_f03_c3.tif",
]


def test_a_pattern_that_will_not_compile_names_no_groups():
    """A broken regex reports no group names rather than raising.

    The group names decide whether a proposal is a metadata regex at all, and
    they are read from patterns that can come from inference or from the
    user's own editing. An unbalanced group must answer "none" so the caller
    can reject the proposal, not abort detection.
    """
    assert _group_names("(?P<plateID>") == ()
    assert _group_names("(?P<plateID>[A-Z]+)_") == ("plateID",)


def test_an_unknown_microscope_gets_a_regex_that_names_its_fields():
    """Inference beats the built-ins when it captures fields the import reads.

    This is the case the fallback exists for, and it has to be established
    before its absence can mean anything.
    """
    pattern, label, hits = auto_detect_regex(_UNKNOWN_SCOPE)
    assert label == "inferred"
    assert hits == len(_UNKNOWN_SCOPE)
    assert "(?P<wellID>" in pattern


def test_a_broken_inference_step_leaves_the_fallback_untouched(monkeypatch):
    """When inference raises, detection still returns a usable answer.

    Inference is an improvement on the built-in fallback, never a
    requirement: an exception from it must not turn a folder the user can
    still import by hand into a dialog that cannot be opened.
    """
    import spacr.regex_infer as regex_infer

    def _explode(filenames):
        raise RuntimeError("alignment failed")

    monkeypatch.setattr(regex_infer, "propose", _explode)

    pattern, label, hits = auto_detect_regex(_UNKNOWN_SCOPE)

    assert label != "inferred"
    assert pattern is None or isinstance(pattern, str)
    assert hits >= 0
