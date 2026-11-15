"""`custom_model` names a Cellpose checkpoint, so --set must accept a path.

Everything that reads it treats it as a path: settings.py's own description
calls it "(str) - Path to a saved Cellpose model", spacr_cellpose.py runs
`os.path.exists()` on it, and validate.py validates it as a path. The old
boolean selector has been retired. The path must still survive

    --set custom_model=/models/my_cellpose.pth

without allowing ``True`` or ``False`` to silently become checkpoint names.
"""


from spacr.settings import expected_types


def test_neither_override_table_reintroduces_the_retired_bool():
    from spacr.cli import _TYPE_OVERRIDES
    from spacr.validate import _EXPECTED_TYPE_OVERRIDES

    assert "custom_model" not in _TYPE_OVERRIDES
    assert "custom_model" not in _EXPECTED_TYPE_OVERRIDES
    assert expected_types["custom_model"] == (str, type(None))
