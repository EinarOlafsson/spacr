"""With nonparametric inference, the chosen regression_type is not read.

Verified on the maintainer's four-plate TSG101 screen: `regression_type='ols'`
and `regression_type='mixed'` under `inference='nonparametric'` produced
byte-identical results -- 1,612 rows, all 24 columns equal. That is why "i
ran a mixed model and an ols model and even if the ols model is marked as
loaded i think i still see the mixed results" was a correct observation, not
a display bug: they ARE the same numbers.

The run summary already said so afterwards. This says it BEFORE the fit, so a
user does not queue a second identical run to find out.
"""
import numpy as np
import pandas as pd
import pytest

from tests.test_regression_entry_points import APP_KEY, _write_screen


def _run(tmp_path, capsys, **overrides):
    from spacr.cli import MODULES, resolve_settings
    from spacr.ml import perform_regression

    score_csv, count_csv, _ = _write_screen(tmp_path)
    settings_csv = tmp_path / "regression.csv"
    pd.DataFrame(
        [("score_data", repr([score_csv])), ("count_data", repr([count_csv])),
         ("toxo", "False"), ("metadata_files", "[]")],
        columns=["Key", "Value"]).to_csv(settings_csv, index=False)
    settings = resolve_settings(MODULES[APP_KEY], str(settings_csv))
    settings["min_cell_count"] = None
    settings.update(overrides)
    np.random.seed(0)
    out = perform_regression(settings)
    return out, capsys.readouterr().out


def test_it_says_the_type_is_not_read_before_it_runs(tmp_path, capsys):
    _out, printed = _run(tmp_path, capsys, regression_type="mixed")

    assert "is not read" in printed
    assert "'mixed'" in printed
    assert "inference='parametric'" in printed, (
        "name the way to actually fit the model that was chosen")


def test_the_parametric_path_says_nothing_of_the_kind(tmp_path, capsys):
    _out, printed = _run(tmp_path, capsys, regression_type="ols",
                         inference="parametric")

    assert "is not read" not in printed
