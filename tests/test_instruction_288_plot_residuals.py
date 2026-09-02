"""Direct owners for the last current-source branches in :mod:`spacr.plot`."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from spacr import plot as P


def test_feature_importance_uses_the_supplied_title():
    frame = pd.DataFrame({
        "feature": ["area", "intensity"],
        "importance": [0.25, 0.75],
    })

    figure = P.plot_feature_importance(frame, title="Permutation importance")

    assert figure.axes[0].get_title() == "Permutation importance"
    plt.close(figure)


def test_clustered_proportion_failure_is_reported_per_bin(monkeypatch, capsys):
    frame = pd.DataFrame({
        "condition": ["control", "control", "treated", "treated"],
        "class": ["low", "high", "low", "high"],
        "well": ["A01", "A02", "B01", "B02"],
    })

    def refuse_fit(*_args, **_kwargs):
        raise RuntimeError("injected singular fit")

    monkeypatch.setattr(P.sm, "GLM", refuse_fit)
    result = P.proportion_mixed_model(
        frame, "condition", "class", "well")

    assert len(result) == 2
    assert np.isnan(result["statistic"]).all()
    assert np.isnan(result["p_value"]).all()
    output = capsys.readouterr().out
    assert output.count("injected singular fit") == 2
    assert "mixed model for bin 'low' did not fit" in output
    assert "mixed model for bin 'high' did not fit" in output
