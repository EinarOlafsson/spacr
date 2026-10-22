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

def _channel_exclusion_frame():
    columns = [
        "pathogen_rad_dist_channel_0_bin_0",
        "cell_channel_2_mean_intensity",
        "nucleus_channel_3_mean_intensity",
        "cell_channel_1_channel_3_pearsons",
        "cell_channel_1_mean_intensity",
        "cell_channel_01_mean_intensity",
        "cell_channel_10_mean_intensity",
        "cell_channel_0suffix_mean_intensity",
        "cell_area",
    ]
    return pd.DataFrame({name: [1.0, 2.0, 3.0] for name in columns})


@pytest.mark.parametrize("exclude, dropped", [
    (["channel_0", "channel_2", "channel_3"], [
        "pathogen_rad_dist_channel_0_bin_0",
        "cell_channel_2_mean_intensity",
        "nucleus_channel_3_mean_intensity",
        "cell_channel_1_channel_3_pearsons",
    ]),
    ("channel_0", ["pathogen_rad_dist_channel_0_bin_0"]),
])
def test_channel_group_exclusions_resolve_before_numeric_validation(exclude, dropped):
    from spacr.utils import filter_dataframe_features

    frame = _channel_exclusion_frame()
    frame["pathogen_rad_dist_channel_0_bin_0"] = "unreadable"
    filtered, features = filter_dataframe_features(
        frame, channel_of_interest=None, exclude=exclude,
        remove_low_variance_features=False,
        remove_highly_correlated_features=False)
    expected = [name for name in frame.columns if name not in dropped]
    assert features == expected
    pd.testing.assert_frame_equal(filtered, frame[expected])


def test_exact_column_exclusions_take_priority_over_channel_groups():
    from spacr.utils import filter_dataframe_features

    frame = _channel_exclusion_frame()
    frame["channel_0"] = [7.0, 8.0, 9.0]
    filtered, features = filter_dataframe_features(
        frame, channel_of_interest=None, exclude="channel_0",
        remove_low_variance_features=False,
        remove_highly_correlated_features=False)
    assert features == list(_channel_exclusion_frame().columns)
    assert "pathogen_rad_dist_channel_0_bin_0" in filtered
    assert "channel_0" not in filtered


def test_exact_feature_names_and_channel_groups_can_be_excluded_together():
    from spacr.utils import filter_dataframe_features

    frame = _channel_exclusion_frame()
    filtered, features = filter_dataframe_features(
        frame, channel_of_interest=None,
        exclude=["channel_0", "cell_area", "cell_channel_10_mean_intensity"],
        remove_low_variance_features=False,
        remove_highly_correlated_features=False)
    assert features == [name for name in frame.columns if name not in {
        "pathogen_rad_dist_channel_0_bin_0", "cell_area", "cell_channel_10_mean_intensity"}]
    assert list(filtered.columns) == features


@pytest.mark.parametrize("unmatched", ["channel_9", "cell_area_typo", "channel_0.*", "channel_*"])
def test_unmatched_selectors_warn_and_leave_the_feature_matrix_unchanged(unmatched, capsys):
    from spacr.utils import filter_dataframe_features

    frame = _channel_exclusion_frame()
    with pytest.warns(UserWarning, match="matched no columns") as caught:
        filtered, features = filter_dataframe_features(
            frame, channel_of_interest=None, exclude=unmatched,
            remove_low_variance_features=False,
            remove_highly_correlated_features=False)
    assert unmatched in str(caught[0].message)
    assert features == list(frame.columns)
    pd.testing.assert_frame_equal(filtered, frame)
    assert f"{unmatched!r}: 0 matching column(s)." in capsys.readouterr().out


@pytest.mark.parametrize("invalid", ["[", "(", "*"])
def test_invalid_exclusion_regexes_name_the_pattern_and_keep_the_cause(invalid):
    import re
    from spacr.utils import filter_dataframe_features

    with pytest.raises(ValueError, match="Invalid feature-exclusion regular expression") as caught:
        filter_dataframe_features(
            _channel_exclusion_frame(), channel_of_interest=None,
            exclude=invalid,
            remove_low_variance_features=False,
            remove_highly_correlated_features=False)
    assert repr(invalid) in str(caught.value)
    assert isinstance(caught.value.__cause__, re.error)


def test_existing_feature_names_with_regex_metacharacters_are_literal():
    from spacr.utils import filter_dataframe_features

    frame = _channel_exclusion_frame()
    literal = "cell_channel_0.mean_intensity"
    other = "cell_channel_0Xmean_intensity"
    frame[literal] = [1.0, 2.0, 3.0]
    frame[other] = [4.0, 5.0, 6.0]
    filtered, features = filter_dataframe_features(
        frame, channel_of_interest=None, exclude=literal,
        remove_low_variance_features=False,
        remove_highly_correlated_features=False)
    assert features == [name for name in frame.columns if name != literal]
    assert other in filtered


@pytest.mark.parametrize("verbose", [False, True])
def test_overlapping_regex_exclusions_log_hit_counts_and_optional_names(verbose, capsys):
    from spacr.utils import filter_dataframe_features

    frame = _channel_exclusion_frame()
    literal = "pathogen_rad_dist_channel_0_bin_0"
    frame[literal] = "unreadable"
    filtered, features = filter_dataframe_features(
        frame, channel_of_interest=None, exclude=[".*channel_0.*", literal],
        remove_low_variance_features=False,
        remove_highly_correlated_features=False, verbose=verbose)
    dropped = {literal, "cell_channel_01_mean_intensity", "cell_channel_0suffix_mean_intensity"}
    assert features == [name for name in frame.columns if name not in dropped]
    pd.testing.assert_frame_equal(filtered, frame[features])
    output = capsys.readouterr().out
    assert "'.*channel_0.*': 3 matching column(s)." in output
    assert f"{literal!r}: 1 matching column(s)." in output
    assert ("cell_channel_01_mean_intensity" in output) is verbose


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
