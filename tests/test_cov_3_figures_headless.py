"""A bundle of nothing is not written.

`render_bundle` writes a whole folder -- figure, data, statistics, settings --
for a run that has no screen. An empty table draws no groups, and a bundle
written for it would put a folder in the gallery holding a blank picture and
an empty data file, which reads to a user as a result rather than as an
absence.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("PySide6")

from spacr.figures import headless                              # noqa: E402


def _spec(frame):
    from spacr.qt.widgets.grouped_plot import PlotSpec

    return PlotSpec(frame=frame, value="fraction", group="gene", unit="well",
                    title="fraction by gene", x_label="gene",
                    y_label="fraction")


def _populated():
    rng = np.random.default_rng(11)
    return pd.DataFrame({
        "gene": ["nc"] * 12 + ["GRA14"] * 12,
        "fraction": np.r_[rng.normal(0.2, 0.05, 12),
                          rng.normal(0.6, 0.05, 12)],
    })


def test_an_empty_table_writes_no_bundle_folder(tmp_path, qapp):
    """Nothing to draw means nothing on disk: no folder, not an empty one."""
    empty = pd.DataFrame({"gene": [], "fraction": []})

    assert headless.render_bundle(_spec(empty), str(tmp_path), "nothing") is None
    assert os.listdir(str(tmp_path)) == [], (
        "a refused bundle still left files behind")


def test_a_populated_table_still_writes_its_bundle(tmp_path, qapp):
    """The contrast that makes the refusal meaningful -- the same call on a
    table with rows produces the folder."""
    folder = headless.render_bundle(_spec(_populated()), str(tmp_path), "real")

    assert folder, "a drawable table produced no bundle"
    assert os.path.isdir(folder)
    assert os.listdir(folder), "the bundle folder is empty"
