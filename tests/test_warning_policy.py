"""Regression tests for deprecation warnings that gate CI."""


def test_pandas_and_sklearn_deprecations_are_ci_errors(pytestconfig):
    rules = set(pytestconfig.getini("filterwarnings"))

    required = {
        r"error::FutureWarning:pandas\..*",
        r"error::DeprecationWarning:pandas\..*",
        r"error::FutureWarning:sklearn\..*",
        r"error::DeprecationWarning:sklearn\..*",
        r"error::pandas.errors.SettingWithCopyWarning",
        r"error:Downcasting object dtype arrays.*:FutureWarning",
        r"error:Setting an item of incompatible dtype is deprecated.*:FutureWarning",
    }

    assert required <= rules
