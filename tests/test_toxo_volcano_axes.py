"""The toxo volcano: its y-limit forms and its hit list (``spacr.toxo``).

``_normalize_y_lims`` accepts ``None`` (fit to the finite data), ``[low, high]``
or a broken ``[[low1, high1], [low2, high2]]`` axis and refuses anything else;
``custom_volcano_plot`` returns the called features in table order whether it
reads its tables from disk or memory, draws one or two panels, or writes a file.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from spacr import toxo as T  # noqa: E402

GENES = [str(220000 + i) for i in range(8)]
HITS = ["220000_1", "220001_1", "220004_1", "220005_1"]


@pytest.fixture(autouse=True)
def _close():
    yield
    plt.close("all")


def _regression():
    return pd.DataFrame({
        "feature": [f"{g}_1" for g in GENES] + ["Intercept"],
        "coefficient": [0.6, -0.6, 0.01, 0.02, 0.5, -0.5, 0.0, 0.03, 0.2],
        "p_value": [1e-6, 1e-6, 0.9, 0.8, 1e-4, 1e-4, 0.7, 0.6, 1e-9],
    })


def _metadata():
    return pd.DataFrame({"gene_nr": GENES,
                         "tagm_location": ["cytosol", "nucleus"] * 4})


def test_y_limits_are_normalised_into_one_or_two_panels(subtests):
    """Every accepted y_lims form gives its exact triple; bad forms raise."""
    cases = [
        ("auto-fit to the finite maximum", None, np.array([1.0, 2.0, np.inf]),
         (False, [0.0, 2.1], None)),
        ("auto-fit never below one", None, np.array([0.2, 0.5]),
         (False, [0.0, 1.0], None)),
        ("no finite values", None, np.array([np.inf, np.inf]),
         (False, [0.0, 1.0], None)),
        ("explicit single panel", [0, 10], np.array([1.0]),
         (False, [0, 10], None)),
        ("open upper end", (0, None), np.array([1.0]),
         (False, [0, None], None)),
        ("broken axis", [[0, 5], [10, 20]], np.array([1.0]),
         (True, [0, 5], [10, 20])),
    ]
    for case_id, y_lims, neg_log_p, expected in cases:
        with subtests.test(msg=case_id):
            broken, lower, upper = T._normalize_y_lims(y_lims, neg_log_p)
            assert type(broken) is bool
            assert (broken, lower, upper) == expected

    for case_id, y_lims in [("three values", [1, 2, 3]),
                            ("text values", ["a", "b"]),
                            ("not a sequence", 5),
                            ("mixed number and pair", [1, [2, 3]])]:
        with subtests.test(msg=case_id):
            with pytest.raises(ValueError, match="y_lims must be None"):
                T._normalize_y_lims(y_lims, np.array([1.0]))


def test_a_volcano_returns_its_called_features_however_it_is_drawn(
        subtests, tmp_path):
    """Disk or memory input, one or two panels: the same exact hit list."""
    data_path = tmp_path / "regression.csv"
    _regression().to_csv(data_path, index=False)
    meta_path = tmp_path / "meta.csv"
    _metadata().to_csv(meta_path, index=False)

    with subtests.test(msg="tables read from disk, no figure"):
        plt.close("all")
        assert T.custom_volcano_plot(str(data_path), _metadata(),
                                     draw=False) == HITS
        assert plt.get_fignums() == []

    with subtests.test(msg="a coefficient threshold narrows the calls"):
        assert T.custom_volcano_plot(_regression(), str(meta_path),
                                     threshold=0.55, draw=False) == HITS[:2]

    with subtests.test(msg="single panel written to a file"):
        out = tmp_path / "v.pdf"
        hits = T.custom_volcano_plot(_regression(), str(meta_path),
                                     point_size=50, figsize=6,
                                     save_path=str(out),
                                     highlight_location="cytosol")
        assert hits == HITS
        assert out.is_file() and out.stat().st_size > 0
        (axes,) = plt.gcf().axes
        assert axes.get_xlim() == (-0.5, 0.5)
        assert axes.get_ylim() == pytest.approx((0.0, 6 * 1.05))
        assert sorted(t.get_text() for t in axes.texts
                      if t.get_text() in HITS) == HITS

    with subtests.test(msg="broken axis splits the labels across two panels"):
        plt.close("all")
        hits = T.custom_volcano_plot(_regression(), _metadata(), figsize=6,
                                     threshold=0.1,
                                     y_lims=[[0, 5], [5, 12]])
        assert hits == HITS
        upper, lower = plt.gcf().axes
        assert lower.get_ylim() == (0.0, 5.0)
        assert upper.get_ylim() == (5.0, 12.0)
        assert sorted(t.get_text() for t in upper.texts
                      if t.get_text() in HITS) == HITS[:2]
        assert sorted(t.get_text() for t in lower.texts
                      if t.get_text() in HITS) == HITS[2:]
