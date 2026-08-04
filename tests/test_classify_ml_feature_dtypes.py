"""Classify (ML) on a real measurements.db: the object-dtype feature crash.

A user's Classify run died at ``utils.filter_dataframe_features`` with

    Declared model feature 'cell_channel_0_mode_intensity' must be numeric,
    got object. Convert or exclude it before fitting.

Nothing was wrong with the measurement. spaCR writes an honest NaN as SQL
``NULL``, and whole measurements are legitimately NaN for a whole database --
``skew_intensity`` and ``kurtosis_intensity`` are NaN for every uniform
object, and ``mode_intensity`` was NaN for every object in every database
written before the SciPy shim in ``measure._extended_regionprops_table``.
``pandas.read_sql`` builds its frame from the rows it gets and never asks
SQLite what the column was declared, so a column that is NULL in every row
arrives as a column of ``None`` typed ``object``. The strict model boundary
then refused it.

These tests build the database with spaCR's own writers -- no hand-rolled
SQL, no hand-set dtypes -- so the reproduction is the pipeline's, not the
test's.
"""
from __future__ import annotations

import os
import sqlite3

import numpy as np
import pandas as pd
import pytest

from spacr import schema
from tests.test_measure_hooks import _project, _settings


# ---------------------------------------------------------------------------
# The real database
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def measured_project(tmp_path_factory):
    """One field measured by ``_measure_crop_core`` into a real project."""
    from spacr.measure import _measure_crop_core

    tmp_path = tmp_path_factory.mktemp("measured")
    project, merged, name = _project(tmp_path)
    _measure_crop_core(0, [], name, _settings(merged))
    return str(project)


def _db(project: str) -> str:
    return os.path.join(project, "measurements", "measurements.db")


def test_the_writer_declares_the_column_numeric_and_writes_null(
        measured_project):
    """The writer is not at fault: REAL affinity, NULL values.

    This is the evidence for fixing the reader rather than the writer. There
    is no text in the database where a number belongs -- coercing on read
    would be repairing damage that was never done.
    """
    con = sqlite3.connect(_db(measured_project))
    try:
        declared = {row[1]: (row[2] or "").upper()
                    for row in con.execute("PRAGMA table_info(cell)")}
        assert declared["cell_channel_0_skew_intensity"] == "REAL"
        types = {row[0] for row in con.execute(
            "SELECT DISTINCT typeof(cell_channel_0_skew_intensity) FROM cell")}
        assert types == {"null"}
    finally:
        con.close()


def test_pandas_reads_that_numeric_column_back_as_object(measured_project):
    """The reproduction, in the reader. Guard rail for the fix below."""
    con = sqlite3.connect(_db(measured_project))
    try:
        frame = pd.read_sql("SELECT * FROM cell", con)
    finally:
        con.close()

    series = frame["cell_channel_0_skew_intensity"]
    assert series.dtype == object
    assert series.isna().all()


def test_the_strict_boundary_still_refuses_it_unrepaired(measured_project):
    """``model_feature_columns`` alone has no data to repair with, so it
    still refuses -- but it now says which columns and why."""
    con = sqlite3.connect(_db(measured_project))
    try:
        frame = pd.read_sql("SELECT * FROM cell", con)
    finally:
        con.close()

    with pytest.raises(schema.ModelFeatureSchemaError) as excinfo:
        schema.model_feature_columns(frame)

    message = str(excinfo.value)
    # Every offender at once, not one per run.
    assert "cell_channel_0_skew_intensity (object)" in message
    assert "cell_channel_0_kurtosis_intensity (object)" in message
    assert "cell_channel_1_skew_intensity (object)" in message
    assert "every value is missing" in message
    assert "Exclude" in message


# ---------------------------------------------------------------------------
# The path that crashed
# ---------------------------------------------------------------------------

def _merged_frame(project: str) -> pd.DataFrame:
    """The frame ``ml_analysis`` is handed, via the real read/merge."""
    from spacr.io import _read_and_merge_data

    frame, _ = _read_and_merge_data(
        [_db(project)], ["cell", "nucleus", "pathogen", "cytoplasm"],
        verbose=False, nuclei_limit=10, pathogen_limit=10)
    return frame


def test_the_merged_frame_still_carries_the_object_column(measured_project):
    frame = _merged_frame(measured_project)
    offenders = [c for c in frame.columns
                 if "skew_intensity" in str(c) and frame[c].dtype == object]
    assert offenders, "the reproduction stopped reproducing"


def test_filter_dataframe_features_completes_on_the_real_database(
        measured_project):
    """The reported crash, gone: the run completes."""
    from spacr.utils import filter_dataframe_features

    frame = _merged_frame(measured_project)

    filtered, features = filter_dataframe_features(
        frame, channel_of_interest=0,
        remove_low_variance_features=False,
        remove_highly_correlated_features=False)

    assert features, "no features survived the filter"
    # Whatever survives is fittable: numeric, and no NaN columns.
    assert all(pd.api.types.is_numeric_dtype(filtered[c]) for c in features)
    # The all-NULL measurement was repaired to float NaN and then dropped by
    # the frame's own NaN filter, rather than stopping the run.
    assert not any("skew_intensity" in str(c) for c in features)


def test_excluding_a_column_removes_it_from_the_fit_not_just_the_settings(
        measured_project):
    """Exclude has to change the feature matrix, not only the dict."""
    from spacr.utils import filter_dataframe_features

    frame = _merged_frame(measured_project)
    _, baseline = filter_dataframe_features(
        frame, channel_of_interest=0,
        remove_low_variance_features=False,
        remove_highly_correlated_features=False)
    assert len(baseline) >= 3

    dropped = list(baseline[:3])
    filtered, features = filter_dataframe_features(
        frame, channel_of_interest=0, exclude=dropped,
        remove_low_variance_features=False,
        remove_highly_correlated_features=False)

    for name in dropped:
        assert name not in features
        assert name not in filtered.columns
    assert set(features) == set(baseline) - set(dropped)


def test_a_single_excluded_column_still_works(measured_project):
    """The old one-name-at-a-time contract is unchanged by the list one."""
    from spacr.utils import filter_dataframe_features

    frame = _merged_frame(measured_project)
    _, baseline = filter_dataframe_features(
        frame, channel_of_interest=0,
        remove_low_variance_features=False,
        remove_highly_correlated_features=False)

    _, features = filter_dataframe_features(
        frame, channel_of_interest=0, exclude=baseline[0],
        remove_low_variance_features=False,
        remove_highly_correlated_features=False)

    assert baseline[0] not in features
    assert set(features) == set(baseline) - {baseline[0]}


def test_unreadable_text_names_every_offender_and_never_becomes_nan(
        measured_project):
    """Text that is not a number stops the run -- naming all of it.

    The important half is the second: an unparseable token must never be
    coerced to NaN and fitted on.
    """
    from spacr.utils import filter_dataframe_features

    frame = _merged_frame(measured_project).copy()
    _, baseline = filter_dataframe_features(
        frame, channel_of_interest=0,
        remove_low_variance_features=False,
        remove_highly_correlated_features=False)
    bad = list(baseline[:2])
    for name in bad:
        frame[name] = "n/a"

    with pytest.raises(schema.ModelFeatureSchemaError) as excinfo:
        filter_dataframe_features(
            frame, channel_of_interest=0,
            remove_low_variance_features=False,
            remove_highly_correlated_features=False)

    message = str(excinfo.value)
    for name in bad:
        assert f"{name} (object)" in message
    assert "'n/a'" in message
    assert "Exclude" in message
    assert "any number of columns" in message


def test_excluding_the_bad_columns_lets_the_same_run_finish(measured_project):
    """The advice the error gives actually works, and it takes several names."""
    from spacr.utils import filter_dataframe_features

    frame = _merged_frame(measured_project).copy()
    _, baseline = filter_dataframe_features(
        frame, channel_of_interest=0,
        remove_low_variance_features=False,
        remove_highly_correlated_features=False)
    bad = list(baseline[:2])
    for name in bad:
        frame[name] = "n/a"

    _, features = filter_dataframe_features(
        frame, channel_of_interest=0, exclude=bad,
        remove_low_variance_features=False,
        remove_highly_correlated_features=False)

    assert not set(bad) & set(features)
    assert features


def test_numeric_text_is_recovered_loudly_on_the_real_frame(measured_project):
    """'12.0' is recoverable, and recovering it says so out loud."""
    from spacr.utils import filter_dataframe_features

    frame = _merged_frame(measured_project).copy()
    _, baseline = filter_dataframe_features(
        frame, channel_of_interest=0,
        remove_low_variance_features=False,
        remove_highly_correlated_features=False)
    victim = baseline[0]
    values = frame[victim].astype(float)
    frame[victim] = values.map(lambda v: f"{v}")

    with pytest.warns(UserWarning, match="stored as text"):
        filtered, features = filter_dataframe_features(
            frame, channel_of_interest=0,
            remove_low_variance_features=False,
            remove_highly_correlated_features=False)

    assert victim in features
    assert np.allclose(filtered[victim].to_numpy(dtype=float),
                       values.to_numpy(dtype=float))
