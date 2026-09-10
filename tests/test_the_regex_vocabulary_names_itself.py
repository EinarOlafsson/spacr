"""The four metadata layouts spaCR reads, and what a fifth one gets told.

``_get_regex`` maps a ``metadata_type`` setting to the regular expression that
parses filenames from that microscope. It is reached straight from a settings
file, so an unrecognised value is a typo a user makes, not an internal error --
and the message they get is the whole of their ability to fix it.
"""
from __future__ import annotations

import re

import pytest


@pytest.mark.parametrize("metadata_type", ["cellvoyager", "cq1", "auto"])
def test_each_known_layout_returns_a_usable_expression(metadata_type):
    """The three built-in vocabularies compile and name their groups."""
    from spacr.utils import _get_regex

    pattern = re.compile(_get_regex(metadata_type, "tif"))

    assert "wellID" in pattern.groupindex
    assert "chanID" in pattern.groupindex


def test_a_custom_layout_wraps_the_users_own_expression():
    """The fourth, which is how a microscope spaCR has never seen is read."""
    from spacr.utils import _get_regex

    regex = _get_regex("custom", "tif",
                       custom_regex=r"(?P<wellID>[A-H]\d\d)")

    assert "wellID" in regex
    assert regex.endswith(".tif")


def test_a_missing_image_format_defaults_to_tif():
    """The guard above the vocabulary, so the tests here are reached cleanly."""
    from spacr.utils import _get_regex

    assert _get_regex("cq1", None).endswith(".tif")


def test_an_unknown_layout_is_refused_by_name():
    """The else that used to be missing.

    Falling through left ``regex`` unbound, so an unrecognised metadata_type
    raised "cannot access local variable 'regex'" from inside this function --
    an error naming an implementation detail rather than the setting the user
    got wrong. The message now lists the four accepted values, because this is
    read from a settings file and the list IS the documentation at the moment
    it is needed.
    """
    from spacr.utils import _get_regex

    with pytest.raises(ValueError) as excinfo:
        _get_regex("nikon", "tif")

    message = str(excinfo.value)
    assert "nikon" in message
    for known in ("cellvoyager", "cq1", "auto", "custom"):
        assert known in message


@pytest.mark.parametrize("metadata_type", ["", None, "CellVoyager", 3])
def test_every_other_unrecognised_value_is_refused_the_same_way(metadata_type):
    """Including the near-misses: case matters, and so does the type.

    'CellVoyager' is the spelling a user copies out of the vendor's own
    documentation, and it must fail with the list rather than with an
    UnboundLocalError.
    """
    from spacr.utils import _get_regex

    with pytest.raises(ValueError):
        _get_regex(metadata_type, "tif")
