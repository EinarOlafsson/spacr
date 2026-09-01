"""`custom_model` names a Cellpose checkpoint, so --set must accept a path.

Everything that reads it treats it as a path: settings.py's own description
calls it "(str) - Path to a saved Cellpose model", spacr_cellpose.py runs
`os.path.exists()` on it, and validate.py validates it as a path. The old
boolean selector has been retired. The path must still survive

    --set custom_model=/models/my_cellpose.pth

without allowing ``True`` or ``False`` to silently become checkpoint names.
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


@pytest.mark.parametrize("text", ["False", "True"])
def test_the_retired_bool_is_rejected_instead_of_becoming_a_path(text):
    with pytest.raises(SettingsError, match="checkpoint path, not a boolean"):
        coerce_value("custom_model", text, None, expected_types, "mask")


def test_neither_override_table_reintroduces_the_retired_bool():
    from spacr.cli import _TYPE_OVERRIDES
    from spacr.validate import _EXPECTED_TYPE_OVERRIDES

    assert "custom_model" not in _TYPE_OVERRIDES
    assert "custom_model" not in _EXPECTED_TYPE_OVERRIDES
    assert expected_types["custom_model"] == (str, type(None))
