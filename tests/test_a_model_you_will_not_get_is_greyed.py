"""The model settings go grey when the run will not fit a model.

Asked on 2026-08-20 -- "if i use a mixed model an nonparametric, is that still
regression or multiple linear regression" -- and the honest answer is that you
do not get both.

`inference='nonparametric'` selects `analysis_mode='guide_permutation'`, which
NEVER CALLS `regression_model`. It is a per-guide marginal association test
with Freedman-Lane permutations blocked within plate, and it fits no
simultaneous model at all -- its own module says so: "It does not claim to
estimate a simultaneous conditional coefficient for every guide when the
number or correlation structure of guides makes that model unidentified."

`regression_type` is read, stored, saved into the settings CSV, and then not
used. The run summary said so AFTERWARDS; 106's rule is to say it at the
point of choosing, and a user picking `mixed` here is choosing a model they
will not get.
"""
from __future__ import annotations

import pytest

from spacr.settings import get_setting_dependencies

#: Settings that only a fitted model reads.
MODEL_KEYS = ("regression_type", "regression_backend", "cov_type",
              "model_plate_position", "random_row_column_effects")


@pytest.fixture(scope="module")
def rules():
    return get_setting_dependencies()


def _settings(inference, mode="regression", regression_type="ols"):
    """A settings dict for the rules to read.

    `regression_type='ols'` and not 'mixed', because several of these keys
    carry a SECOND rule from the family table -- `cov_type` is not read by a
    mixed model, so under 'mixed' it is greyed for a reason that has nothing
    to do with inference. Combining the two is correct behaviour; a fixture
    that triggers both cannot tell which one fired.
    """
    return {"inference": inference, "analysis_mode": mode,
            "regression_type": regression_type}


class TestTheModelSettingsGoQuietUnderPermutation:

    @pytest.mark.parametrize("key", MODEL_KEYS)
    def test_greyed_when_the_run_will_permute(self, rules, key):
        assert not rules[key]["predicate"](_settings("nonparametric"), None)

    @pytest.mark.parametrize("key", MODEL_KEYS)
    def test_live_when_the_run_will_fit(self, rules, key):
        assert rules[key]["predicate"](_settings("parametric"), None)

    @pytest.mark.parametrize("key", MODEL_KEYS)
    def test_live_under_auto_because_auto_may_fit(self, rules, key):
        """THE MISTAKE THIS GUARDS. The first version negated
        `permutation_active`, which answers True under 'auto' on purpose so
        the PERMUTATION controls stay live while the resolution is unknown.
        Negating it greyed the MODEL controls under 'auto' too -- and 'auto'
        may well resolve to regression, in which case those are exactly the
        settings the run reads.
        """
        assert rules[key]["predicate"](_settings("auto"), None)

    def test_a_mixed_model_is_greyed_too_which_is_the_reported_case(
            self, rules):
        """The question that prompted this: mixed AND nonparametric."""
        assert not rules["regression_type"]["predicate"](
            _settings("nonparametric", regression_type="mixed"), None)

    def test_the_reason_says_no_model_is_fitted(self, rules):
        said = rules["regression_type"]["reason"](
            _settings("nonparametric"), None)

        assert "fits no model" in said
        assert "each guide on its own" in said, "say what it does instead"
        assert "kept and saved" in said, "and that nothing was thrown away"

    def test_the_rule_declares_what_it_reads(self, rules):
        for key in MODEL_KEYS:
            sources = rules[key]["sources"]
            assert "inference" in sources
            assert "analysis_mode" in sources


class TestThePermutationSettingsAreTheMirror:
    """The two families must never be live or dead at the same time under a
    decided inference -- one of them is always the one that runs."""

    @pytest.mark.parametrize("inference,model_live", [
        ("parametric", True), ("nonparametric", False)])
    def test_exactly_one_family_is_live(self, rules, inference, model_live):
        model = rules["regression_type"]["predicate"](
            _settings(inference), None)
        permutation = rules["guide_permutations"]["predicate"](
            _settings(inference), None)

        assert model is model_live
        assert permutation is not model_live

    def test_auto_leaves_both_live(self, rules):
        """Greying a control the run may use is the worse of the two errors."""
        assert rules["regression_type"]["predicate"](_settings("auto"), None)
        assert rules["guide_permutations"]["predicate"](
            _settings("auto"), None)
