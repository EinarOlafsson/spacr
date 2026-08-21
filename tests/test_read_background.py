"""What a guide's reads look like in wells it is not in.

Proposed 2026-08-21: measure the spurious rate in the control columns of
known composition, and use it to set the fraction threshold instead of
guessing at it with a constant.

THE IDEA IS RIGHT AND THE ANSWER IS THE PER-GUIDE FACTOR. On the screen this
was written against, the median guide's background was 0.013% and the 99th
percentile 0.15% -- a haze any small threshold removes -- while one guide sat
at 9.2% in every single control well. No single threshold serves both: low
enough to catch the outlier deletes real biology, and where it is now admits
the outlier anyway.
"""
from __future__ import annotations

import numpy as np
import pytest

from spacr import read_background as module


@pytest.fixture
def controls():
    """Four control wells: one intended guide, a diffuse haze, and ONE guide
    that turns up everywhere it cannot be."""
    rng = np.random.default_rng(0)
    fractions, intended = {}, {}
    for index in range(4):
        well = f"w{index}"
        here = {"real": 0.70, "everywhere": 0.09}
        for tail in range(40):
            here[f"t{tail}"] = float(rng.uniform(0.0001, 0.002))
        total = sum(here.values())
        fractions[well] = {g: v / total for g, v in here.items()}
        intended[well] = ["real"]
    return fractions, intended


class TestMeasuringTheBackground:

    def test_the_intended_guide_is_not_background(self, controls):
        fractions, intended = controls
        got = module.background_from_controls(fractions, intended)
        assert "real" not in got["background"]

    def test_a_well_with_no_stated_contents_is_skipped(self, controls):
        """Assuming it empty would turn a real guide into background."""
        fractions, intended = controls
        intended = dict(intended)
        intended.pop("w0")
        got = module.background_from_controls(fractions, intended)
        assert got["control_wells"] == 3

    def test_the_spurious_mass_is_reported_per_well_not_only_averaged(
            self, controls):
        """Its SPREAD says whether these wells describe one phenomenon: tight
        is a sequencing property, wide means some wells were contaminated."""
        fractions, intended = controls
        got = module.background_from_controls(fractions, intended)
        assert len(got["spurious_mass_per_well"]) == 4
        assert got["spurious_mass_min"] <= got["spurious_mass_median"]
        assert got["spurious_mass_median"] <= got["spurious_mass_max"]

    def test_the_median_is_used_so_one_bad_well_cannot_set_it(self):
        fractions = {f"w{i}": {"real": 0.9, "odd": 0.1} for i in range(4)}
        fractions["w0"]["odd"] = 0.9        # one contaminated well
        intended = {w: ["real"] for w in fractions}

        by_median = module.background_from_controls(fractions, intended)
        by_mean = module.background_from_controls(fractions, intended,
                                                  statistic="mean")

        assert by_median["background"]["odd"] < by_mean["background"]["odd"]


class TestTheThresholdComesFromTheData:

    def test_it_is_far_below_a_two_percent_constant_for_the_haze(self,
                                                                controls):
        fractions, intended = controls
        got = module.background_from_controls(fractions, intended)
        suggestion = module.suggest_threshold(got, quantile=0.99)
        # The diffuse background is trace-level; 2% is orders of magnitude
        # above it and removes real biology for no gain.
        assert suggestion["threshold"] < 0.02

    def test_it_says_how_many_guides_it_cannot_serve(self, controls):
        """So the number cannot be read as a complete answer."""
        fractions, intended = controls
        got = module.background_from_controls(fractions, intended)
        suggestion = module.suggest_threshold(got)
        assert suggestion["guides_needing_their_own"] >= 1

    def test_an_empty_measurement_gives_no_number(self):
        assert not np.isfinite(
            module.suggest_threshold({"background": {}})["threshold"])


class TestTheOutlierIsFound:

    def test_a_guide_present_in_every_control_well_is_flagged(self, controls):
        fractions, intended = controls
        got = module.background_from_controls(fractions, intended)

        flagged = module.suspicious(got)

        assert [row["guide"] for row in flagged] == ["everywhere"]
        assert flagged[0]["in_wells"] == flagged[0]["of_wells"]

    def test_a_single_contaminated_well_is_not_flagged(self):
        """Appearing everywhere is the stronger signal, more than the level:
        one high well is a contaminated well, not a systematic effect."""
        fractions = {f"w{i}": {"real": 0.99, "once": 0.01} for i in range(8)}
        fractions["w0"] = {"real": 0.5, "once": 0.5}
        intended = {w: ["real"] for w in fractions}

        got = module.background_from_controls(fractions, intended)
        assert module.suspicious(got, everywhere=1.01) == []

    def test_it_refuses_to_call_it_an_artefact(self, controls):
        """The reads cannot separate a sequencing artefact from a guide
        genuinely over-represented in the library, and the correction is
        opposite in the two cases."""
        fractions, intended = controls
        got = module.background_from_controls(fractions, intended)
        verdict = module.suspicious(got)[0]["verdict"]
        assert "cannot say which" in verdict
        assert "imaging" in verdict


class TestSubtracting:

    def test_it_removes_a_guides_own_background_and_not_another_s(self):
        before = {"a": 0.50, "b": 0.30, "c": 0.20}
        after = module.subtract_background(
            before, {"b": 0.10}, renormalise=False)
        assert after["a"] == pytest.approx(0.50)
        assert after["b"] == pytest.approx(0.20)
        assert after["c"] == pytest.approx(0.20)

    def test_it_cannot_go_below_zero(self):
        after = module.subtract_background({"a": 0.01}, {"a": 0.5},
                                           renormalise=False)
        assert after["a"] == 0.0

    def test_renormalising_moves_the_shares_not_the_total(self):
        before = {"a": 0.6, "b": 0.4}
        after = module.subtract_background(before, {"b": 0.2})
        assert sum(after.values()) == pytest.approx(sum(before.values()))
        assert after["a"] > before["a"]      # a gained what b lost

    def test_the_scale_exists_because_controls_overstate_it(self):
        """Index hopping scales with the source's abundance, and a control
        well has one guide at seventy per cent where an ordinary well's
        largest is five -- so the control columns are an upper bound."""
        before = {"a": 0.5, "b": 0.5}
        full = module.subtract_background(before, {"b": 0.1}, scale=1.0,
                                          renormalise=False)
        half = module.subtract_background(before, {"b": 0.1}, scale=0.5,
                                          renormalise=False)
        assert half["b"] > full["b"]


class TestNothingIsBakedIn:

    def test_no_guide_name_or_measured_level_is_a_constant(self):
        """Read the SYNTAX, not the text. The module docstring quotes the
        request that prompted it, guide names and all, and a grep over the
        source cannot tell a quoted question from a baked-in coefficient --
        it failed on exactly that."""
        import ast
        import inspect

        tree = ast.parse(inspect.getsource(module))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            for constant in ast.walk(node.value):
                if isinstance(constant, ast.Constant) and isinstance(
                        constant.value, str):
                    assert "TGGT1" not in constant.value
                    assert "233460" not in constant.value

    def test_the_background_is_always_an_argument(self):
        import inspect

        signature = inspect.signature(module.subtract_background)
        assert (signature.parameters["background"].default
                is inspect.Parameter.empty)
