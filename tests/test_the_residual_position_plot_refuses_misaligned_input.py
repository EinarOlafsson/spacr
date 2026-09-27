"""Item 288: the residual-by-position plot's refusals and its silent verdict.

``spacr.permutation_qc.plot_residual_by_position`` draws the permuted
residuals against plate position. It refuses inputs that do not line up --
a figure of misaligned residuals is a picture of the wrong wells -- and
writes a verdict's remedy under the panels only when there is one. Also:
numpy integers in the report are written to JSON as plain ints.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

pytest.importorskip("matplotlib")

from spacr import permutation_qc as pq  # noqa: E402


def _footer(fig):
    return next(t.get_text() for t in fig.texts
                if t.get_text().startswith("Removed before"))


def test_residuals_and_blocks_of_different_lengths_are_refused():
    with pytest.raises(ValueError) as excinfo:
        pq.plot_residual_by_position([0.1, 0.2, 0.3], ["b1", "b1"],
                                     {"rowID": ["r1", "r2", "r3"]})
    assert "same length" in str(excinfo.value)


def test_a_position_column_of_the_wrong_length_is_refused_by_name():
    with pytest.raises(ValueError) as excinfo:
        pq.plot_residual_by_position([0.1, 0.2], ["b1", "b1"],
                                     {"columnID": ["c1"]})
    assert "'columnID'" in str(excinfo.value)


def test_a_verdict_without_a_remedy_adds_no_remedy_line():
    fig = pq.plot_residual_by_position(
        [0.1, -0.2, 0.3, -0.1], ["b1", "b1", "b2", "b2"],
        {"rowID": ["r1", "r2", "r1", "r2"]},
        verdict={"ok": False, "remedy": ""})
    assert _footer(fig) == ("Removed before residualisation: block only "
                            "(guide_nuisance_columns is empty)")


def test_numpy_integers_are_written_as_plain_ints():
    plain = pq._plain({"n": np.int64(7), "values": [np.int32(1), 2.5]})
    assert plain == {"n": 7, "values": [1, 2.5]}
    assert type(plain["n"]) is int
    assert json.loads(json.dumps(plain)) == plain
