"""Backend detection for uniquely numbered regression result folders."""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from spacr.qt.widgets.regression_results import backend_of  # noqa: E402


@pytest.mark.parametrize(
    ("folder", "expected"),
    [
        ("lasso_3", "lasso"),
        ("elasticnet_12", "elasticnet"),
        ("group_lasso_2", "group_lasso"),
    ],
)
def test_numbered_penalized_runs_keep_their_backend(folder, expected):
    """A collision suffix must not make penalized p-values look inferential."""
    path = f"/analysis/results/{folder}/results.csv"
    assert backend_of(path) == expected


def test_numbered_parametric_run_is_not_misclassified():
    assert backend_of("/analysis/results/ols_4/results.csv") is None
