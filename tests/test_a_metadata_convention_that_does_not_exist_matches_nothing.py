"""Item 288: the filename-convention table's answers to a key it lacks.

``spacr.regex_infer`` keeps one table of microscope filename conventions.
Three answers the ordinary matching tests do not ask for:

* matching a filename against a key the table does not have -- or a custom
  pattern that does not compile -- is "no match", not an exception, so a
  sweep over many conventions can carry on;
* the pattern for a missing key -- with one extension or any of several --
  raises ``KeyError`` naming the key, for the caller to turn into a
  sentence;
* an image format of ``None`` means ``tif``.
"""
from __future__ import annotations

import pytest

from spacr import regex_infer as ri


def test_an_unknown_key_matches_nothing():
    assert ri._metadata_match("A01_s1_w1.tif", "no_such_scope") is None


def test_a_custom_pattern_that_does_not_compile_matches_nothing():
    assert ri._metadata_match("A01.tif", "custom", custom_regex="(?P<") is None


def test_a_pattern_is_refused_for_an_unknown_key_by_name():
    with pytest.raises(KeyError) as excinfo:
        ri._metadata_pattern("no_such_scope")
    assert excinfo.value.args == ("no_such_scope",)


def test_no_image_format_means_tif():
    assert ri._metadata_pattern("cellvoyager", None) == \
        ri._metadata_pattern("cellvoyager", "tif")
    assert "tif" in ri._metadata_pattern("cellvoyager", None)


def test_the_any_extension_pattern_refuses_an_unknown_key_too():
    with pytest.raises(KeyError):
        ri._metadata_pattern_any_extension("no_such_scope", ("tif", "png"))
    pattern = ri._metadata_pattern_any_extension("cellvoyager", ("tif", "png"))
    assert "(?:tif|png)" in pattern
