"""Folder names built from a feature selection, including the mixed one.

The docstring promises the result is "safe on every filesystem", and that is
the whole job: a user may reasonably filter on ``mean intensity``, and a column
fragment can carry anything -- a slash, a colon, a space. The mixed case, where
a selection holds both channel numbers and free text, is the one line that had
never run, and it is the one where both rules have to be applied in the same
string.
"""
from __future__ import annotations

import re

import pytest


def test_no_selection_is_named_all_features():
    """The documented default."""
    from spacr.utils import feature_folder_name

    assert feature_folder_name(None) == "all_features"


def test_one_channel_is_named_for_that_channel():
    """A single int, which is the commonest selection."""
    from spacr.utils import feature_folder_name

    assert feature_folder_name(1) == "channel_1"


def test_several_channels_share_one_prefix():
    """Line 9754: the all-int list, named ``channels_`` rather than repeating."""
    from spacr.utils import feature_folder_name

    assert feature_folder_name([1, 2]) == "channels_1_2"


def test_a_named_selection_keeps_its_name():
    """A non-list, non-int selection goes through the slugifier alone."""
    from spacr.utils import feature_folder_name

    assert feature_folder_name("morphology") == "morphology"


def test_a_mixture_of_channels_and_names_joins_both_rules():
    """Line 9755, the mixed list -- the only line where both rules meet.

    An int becomes ``channel_<n>`` and everything else is slugified, in the
    order given. Getting this wrong does not raise: it produces a folder name
    that is merely different, so two distinct selections could share a folder
    and one run's features would overwrite the other's.
    """
    from spacr.utils import feature_folder_name

    name = feature_folder_name([1, "morphology"])

    assert name == "channel_1_morphology"


def test_free_text_in_a_mixture_is_slugified():
    """The same line, with text a filesystem could not take verbatim.

    A user filtering on "mean intensity" or "area/perimeter" is the case the
    docstring's promise is about.
    """
    from spacr.utils import feature_folder_name

    name = feature_folder_name([2, "mean intensity"])

    assert name.startswith("channel_2_")
    assert re.fullmatch(r"[0-9A-Za-z_]+", name), name
    assert " " not in name and "/" not in name


def test_two_different_mixtures_do_not_share_a_folder():
    """The property the name exists for, stated as a test.

    Two selections that differ must not name the same folder -- that is what
    turns a naming bug into one run's features overwriting another's.
    """
    from spacr.utils import feature_folder_name

    assert feature_folder_name([1, "morphology"]) != \
        feature_folder_name([2, "morphology"])
    assert feature_folder_name([1, "morphology"]) != \
        feature_folder_name([1, "intensity"])


def test_a_channel_number_in_a_mixture_keeps_its_digits():
    """An int member is used as written, not slugified.

    ``re.sub`` on ``str(1)`` would answer "1" as well, so this branch looks
    redundant -- and it is not, because the difference shows on the FOLDER
    the run writes into and a naming change is a run that cannot find its own
    features. Two channels in a mixture must still read as two channels.
    """
    from spacr.utils import feature_folder_name

    assert feature_folder_name([1, 2]) == "channels_1_2"
    assert feature_folder_name([1, "morphology"]) == "channel_1_morphology"


def test_every_channel_in_a_multi_channel_mixture_appears():
    """The name is what tells two runs apart, so nothing may be dropped."""
    from spacr.utils import feature_folder_name

    name = feature_folder_name([0, 1, 2, "morphology"])

    for channel in ("0", "1", "2"):
        assert channel in name, f"channel {channel} vanished from {name!r}"
    assert "morphology" in name


def test_a_filter_that_slugifies_to_nothing_still_names_a_folder():
    """``'x'`` is the fallback, and it has to exist.

    A filter of punctuation alone -- a user pasting "---" into the box --
    would otherwise produce an empty component, and the folder name would
    collapse into a doubled separator or into the name of a different
    selection entirely.
    """
    from spacr.utils import feature_folder_name

    name = feature_folder_name([1, "---"])

    assert name and not name.endswith("_")
    assert re.fullmatch(r"[0-9A-Za-z_]+", name), name
    assert name != feature_folder_name([1])
