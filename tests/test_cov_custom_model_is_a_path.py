"""`custom_model` names a Cellpose checkpoint, so --set must accept a path.

`expected_types` declares it `bool`. Everything that reads it treats it as a
path: settings.py's own description calls it "(str) - Path to a saved Cellpose
model", spacr_cellpose.py runs `os.path.exists()` on it, and validate.py
validates it as a path. The consequence was that

    --set custom_model=/models/my_cellpose.pth

came back as "cannot be read as bool", so a custom model could not be chosen
from the command line at all -- the same shape as the `masks` bug already
recorded in cli.py's `_APP_TYPE_OVERRIDES`.
"""

import pytest

from spacr.cli import SettingsError, coerce_value
from spacr.settings import expected_types


def test_a_checkpoint_path_is_accepted():
    """The bug, stated directly: a path must survive --set."""
    value = coerce_value("custom_model", "/models/my_cellpose.pth", None,
                         expected_types, "mask")
    assert value == "/models/my_cellpose.pth"


def test_none_still_means_no_custom_model():
    """`None` is the value every reader tests for; it must stay reachable."""
    assert coerce_value("custom_model", "none", None,
                        expected_types, "mask") is None


@pytest.mark.parametrize("text, wanted", [("False", False), ("True", True)])
def test_a_bool_is_still_read_as_a_bool(text, wanted):
    """settings.py still seeds this key with False in two places.

    Widening to str alone would have turned `--set custom_model=False` into
    the *string* "False", which `os.path.exists` then reports as a missing
    model rather than as "no custom model".
    """
    got = coerce_value("custom_model", text, None, expected_types, "mask")
    assert got is wanted


def test_the_two_override_tables_agree_about_it():
    """cli and validate mirror each other deliberately; a fix must land in both.

    A value the validator accepts has to be a value --set can write.
    """
    from spacr.cli import _TYPE_OVERRIDES
    from spacr.validate import _EXPECTED_TYPE_OVERRIDES

    assert (_TYPE_OVERRIDES["custom_model"]
            == _EXPECTED_TYPE_OVERRIDES["custom_model"])
