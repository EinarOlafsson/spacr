"""Regression tests for deprecation warnings that gate CI."""


def test_pandas_and_sklearn_deprecations_are_ci_errors(pytestconfig):
    rules = set(pytestconfig.getini("filterwarnings"))

    required = {
        r"error::FutureWarning:pandas\..*",
        r"error::DeprecationWarning:pandas\..*",
        r"error::FutureWarning:sklearn\..*",
        r"error::DeprecationWarning:sklearn\..*",
        r"error:Downcasting object dtype arrays.*:FutureWarning",
        r"error:Setting an item of incompatible dtype is deprecated.*:FutureWarning",
    }

    assert required <= rules

    # SettingWithCopyWarning is registered by tests/conftest.py rather than by
    # pytest.ini, and only when pandas still HAS the class -- pandas 3 removed
    # it, and an ini line naming a missing class kills collection outright. So
    # the rule is conditional, and so is the assertion.
    try:
        from pandas.errors import SettingWithCopyWarning  # noqa: F401
    except ImportError:
        assert not any("SettingWithCopyWarning" in r for r in rules), (
            "pandas no longer defines SettingWithCopyWarning, so no filter "
            "may name it -- pytest resolves the class at startup and the "
            "whole suite fails to collect")
    else:
        assert any("SettingWithCopyWarning" in r for r in rules), (
            "pandas still defines SettingWithCopyWarning, so writing through "
            "a slice must still be an error rather than a warning nobody sees")
