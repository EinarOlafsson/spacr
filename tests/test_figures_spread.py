"""Statistical spread helpers state what a bar's whiskers mean."""

from __future__ import annotations

import numpy as np
import pytest

from spacr.figures.spread import (
    SPREAD_NONE,
    SPREAD_SD,
    SPREAD_SEM,
    SPREAD_VAR,
    spread_label,
    spread_of,
    summarise,
)


def test_fewer_than_two_finite_observations_have_no_measurable_spread():
    """One finite value cannot estimate sample dispersion."""
    assert np.isnan(spread_of([np.nan, 4.0, np.inf], SPREAD_SD))


@pytest.mark.parametrize(
    ("kind", "expected"),
    [
        (SPREAD_SD, 1.0),
        (SPREAD_VAR, 1.0),
        (SPREAD_SEM, 1.0 / np.sqrt(3.0)),
    ],
)
def test_spread_statistics_use_the_sample_definition(kind, expected):
    """SD, variance, and SEM agree on one documented sample."""
    assert spread_of([1.0, 2.0, 3.0], kind) == pytest.approx(expected)


def test_asking_for_no_whisker_returns_nan():
    """No spread is represented as absent, never as a zero-width whisker."""
    assert np.isnan(spread_of([1.0, 2.0, 3.0], SPREAD_NONE))


def test_an_unknown_spread_is_refused_with_the_available_values():
    """A misspelled statistic cannot silently become a different one."""
    with pytest.raises(ValueError) as excinfo:
        spread_of([1.0, 2.0], "standard-eror")

    message = str(excinfo.value)
    assert "standard-eror" in message
    assert all(kind in message for kind in (SPREAD_NONE, SPREAD_SD,
                                            SPREAD_SEM, SPREAD_VAR))


def test_no_spread_has_no_axis_label():
    assert spread_label(SPREAD_NONE, unit="micrometre") == ""


@pytest.mark.parametrize(
    ("kind", "plain", "with_unit"),
    [
        (SPREAD_SD, "mean ± SD", "mean ± SD (px)"),
        (SPREAD_SEM, "mean ± SEM", "mean ± SEM (px)"),
        (SPREAD_VAR, "mean ± variance", "mean ± variance (px²)"),
    ],
)
def test_each_axis_label_names_the_statistic_and_its_units(
    kind,
    plain,
    with_unit,
):
    """Variance alone squares the supplied measurement unit."""
    assert spread_label(kind) == plain
    assert spread_label(kind, unit="px") == with_unit


def test_an_unknown_spread_has_no_invented_axis_label():
    with pytest.raises(ValueError, match="mystery"):
        spread_label("mystery")


def test_group_summary_drops_nonfinite_groups_and_preserves_real_counts():
    """An unmeasured group is omitted instead of becoming a zero-valued bar."""
    result = summarise(
        {
            7: [1.0, 3.0, np.nan],
            "unmeasured": [np.nan, np.inf],
        },
        SPREAD_SD,
    )

    assert set(result) == {"7"}
    assert result["7"] == {
        "mean": 2.0,
        "spread": pytest.approx(np.sqrt(2.0)),
        "n": 2.0,
    }


def test_an_empty_group_mapping_has_an_empty_summary():
    assert summarise({}, SPREAD_SEM) == {}
