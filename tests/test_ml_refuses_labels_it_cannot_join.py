"""Classify (ML) refuses labels it cannot join, and strata it cannot read.

Annotation labels reach the measurements only through ``png_list.prcfo``.
A png_list that holds the label column but not ``prcfo`` -- a table copied
in by hand, or written by an older tool -- must stop the run with a sentence
naming both columns, before a join that would silently drop every label.
The annotation sampler's strata are columns too, and a stratum that names no
column is refused rather than sampled as one stratum.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from spacr import ml
from tests.test_cov_ml_ml_scores import _make_src, _ml_settings


def test_a_png_list_without_prcfo_is_refused_by_name(tmp_path):
    rng = np.random.default_rng(0)
    src = _make_src(tmp_path, "plate", rng, with_centroids=False)
    database = Path(src) / "measurements" / "measurements.db"
    with sqlite3.connect(database) as connection:
        connection.execute("ALTER TABLE png_list ADD COLUMN label TEXT")
        connection.execute("UPDATE png_list SET label = 'a'")
        connection.execute("ALTER TABLE png_list DROP COLUMN prcfo")

    settings = _ml_settings(str(src), dataset_mode="annotation", annotation_column="label")
    with pytest.raises(ValueError) as caught:
        ml.generate_ml_scores(settings)

    message = str(caught.value)
    assert message.startswith(
        "png_list needs the columns 'prcfo' and 'label' to train on annotations; it has: ")
    assert "label" in message.split("it has: ", 1)[1]


def test_strata_name_columns_and_quantile_bins_or_are_refused():
    frame = pd.DataFrame({"well": ["a", "a", "b", "b"], "area": [1.0, 2.0, 3.0, 4.0]})

    keys = ml._annotate_strata(frame, "well, area:2")
    assert [key.name for key in keys] == ["well", "area"]
    assert keys[1].tolist() == ["0", "1", "0", "1"]

    with pytest.raises(ValueError, match="annotate_sample_strata names 'volume', not a column"):
        ml._annotate_strata(frame, ["well", "volume"])
