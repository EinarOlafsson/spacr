"""Outliers are removed BEFORE annotation (instruction 210).

BEFORE, NOT AFTER, and the order is the substance of the request. Annotation
methods that normalise have their denominator set by which objects are
present, so removing a segmentation artefact after the fractions are
computed leaves its reads redistributed across the guides; removing it first
means it never contributed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spacr.outlier_filter import (CRITERIA, DEFAULT_MADS, apply, column_for,
                                  describe, outliers)


@pytest.fixture
def cells():
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "cell_area": list(rng.normal(500, 50, 200)) + [9000.0, 12000.0],
        "nucleus_area": list(rng.normal(200, 20, 200)) + [210.0, 205.0],
        "cell_channel_1_mean_intensity":
            list(rng.normal(1000, 100, 200)) + [1010.0, 990.0],
    })


class TestAllFourIndependently:

    def test_there_are_four(self):
        assert len(CRITERIA) == 4

    def test_they_are_the_four_asked_for(self):
        assert [c for c, _ in CRITERIA] == [
            "cell_area", "nucleus_area", "cell_intensity",
            "nucleus_intensity"]

    def test_one_filter_does_not_touch_another(self, cells):
        """The outliers are in cell_area only."""
        _, report = apply(cells, {"nucleus_area_outlier_mads": 5})
        assert report[0]["removed"] == 0

    def test_the_cell_area_filter_finds_them(self, cells):
        out, report = apply(cells, {"cell_area_outlier_mads": 5})
        assert report[0]["removed"] == 2
        assert len(out) == len(cells) - 2

    def test_two_filters_at_once_report_separately(self, cells):
        """"412 objects removed" does not tell a user whether their area cut
        or their intensity cut was the loose one."""
        _, report = apply(cells, {"cell_area_outlier_mads": 5,
                                  "nucleus_area_outlier_mads": 5})
        assert len(report) == 2
        assert {r["criterion"] for r in report} == {"cell_area",
                                                    "nucleus_area"}


class TestOffByDefault:
    """"This changes which cells exist, and a filter that silently drops
    objects is a filter that will be forgotten and then blamed on the
    annotation"."""

    def test_no_settings_removes_nothing(self, cells):
        out, report = apply(cells, {})
        assert len(out) == len(cells) and report == []

    def test_none_is_off(self, cells):
        out, report = apply(cells, {"cell_area_outlier_mads": None})
        assert len(out) == len(cells) and report == []

    def test_the_spacr_default_is_none(self):
        from spacr.settings import get_perform_regression_default_settings

        got = get_perform_regression_default_settings({})
        for criterion, _ in CRITERIA:
            assert got[f"{criterion}_outlier_mads"] is None

    def test_every_criterion_has_a_setting_and_a_tooltip(self):
        from spacr.settings import expected_types, tooltips

        for criterion, _ in CRITERIA:
            key = f"{criterion}_outlier_mads"
            assert key in expected_types
            assert key in tooltips


class TestTheCountIsReported:

    def test_the_report_names_the_column_it_read(self, cells):
        _, report = apply(cells, {"cell_area_outlier_mads": 5})
        assert report[0]["column"] == "cell_area"

    def test_a_missing_column_says_so_rather_than_nothing(self, cells):
        """A filter switched on that found no column removed nothing, and a
        run that says nothing about it looks exactly like one where the
        filter worked and found nothing."""
        _, report = apply(cells, {"nucleus_intensity_outlier_mads": 5})
        assert "no nucleus channel intensity column" in report[0]["note"]

    def test_the_sentence_says_the_order(self, cells):
        _, report = apply(cells, {"cell_area_outlier_mads": 5})
        text = describe(report)
        assert "computed on what survived" in text

    def test_nothing_on_means_nothing_said(self, cells):
        assert describe(apply(cells, {})[1]) == ""


class TestTheCutIsAMad:
    """"a standard-deviation cut on skewed data removes real cells from the
    long tail" -- the exact population a screen is usually looking for."""

    def test_a_mad_of_zero_does_not_empty_the_table(self):
        """Over half the values identical happens on a quantised column, and
        a rule flagging every non-modal value there would empty it."""
        flat = pd.Series([1.0] * 50 + [2.0, 3.0])
        assert not outliers(flat).any()

    def test_too_few_values_flags_nothing(self):
        assert not outliers(pd.Series([1.0, 2.0])).any()

    def test_a_symmetric_sample_loses_almost_nothing(self):
        rng = np.random.default_rng(3)
        normal = pd.Series(rng.normal(0, 1, 5000))
        assert outliers(normal, mads=DEFAULT_MADS).sum() <= 5

    def test_a_long_tail_is_kept_where_an_sd_cut_would_not(self):
        rng = np.random.default_rng(4)
        skewed = pd.Series(rng.lognormal(0, 1, 5000))
        kept = (~outliers(skewed, mads=DEFAULT_MADS)).sum()
        assert kept > 4700

    def test_non_finite_values_are_never_outliers(self):
        values = pd.Series([1.0, 2.0, 3.0, np.nan, np.inf] * 10)
        assert not outliers(values)[3::5].any()


class TestTheOrderMatters:
    """The test the instruction asks for: "removing an object changes the
    fractions of the guides in its well"."""

    def test_removing_an_object_changes_the_wells_fractions(self):
        frame = pd.DataFrame({
            "prc": ["w1"] * 4,
            "grna": ["g1", "g1", "g2", "g2"],
            "cell_area": [500.0, 510.0, 495.0, 90000.0],
        })

        def fractions(table):
            counts = table.groupby("grna").size()
            return (counts / counts.sum()).to_dict()

        before = fractions(frame)
        after = fractions(apply(frame, {"cell_area_outlier_mads": 5})[0])
        assert before != after, (
            "if the fractions did not move, the order would not matter and "
            "this instruction would be about nothing")
        assert after["g1"] > before["g1"], (
            "the artefact's share went back to the guides that earned it")

    def test_filtering_after_would_leave_the_reads_redistributed(self):
        """The counterfactual, stated as a test so the reason survives."""
        frame = pd.DataFrame({
            "prc": ["w1"] * 4,
            "grna": ["g1", "g1", "g2", "g2"],
            "cell_area": [500.0, 510.0, 495.0, 90000.0],
        })
        counts = frame.groupby("grna").size()
        formed_first = (counts / counts.sum()).to_dict()

        kept = apply(frame, {"cell_area_outlier_mads": 5})[0]
        counts = kept.groupby("grna").size()
        removed_first = (counts / counts.sum()).to_dict()

        assert formed_first["g2"] == 0.5
        assert removed_first["g2"] < 0.5


class TestTheColumnResolution:

    def test_an_intensity_column_carrying_its_channel_is_found(self):
        frame = pd.DataFrame({"cell_channel_3_mean_intensity": [1.0]})
        assert column_for(frame, "cell_intensity") == \
            "cell_channel_3_mean_intensity"

    def test_a_table_without_it_gives_none(self):
        assert column_for(pd.DataFrame({"x": [1]}), "cell_area") is None
