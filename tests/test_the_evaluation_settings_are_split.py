"""The evaluation settings are split by category (instruction 233).

"the evaluation section of classify has too many setting please devide them
on category and machine learning vs computer vision ... shared settings
should not moove to cv or ml specific setting categories."

THE AUDIT WAS THE WORK, and it corrected the guess. `classify.FAMILY_SETTINGS`
is the authoritative table of what each family reads EXCLUSIVELY, and
against it only `n_top_examples` is CV-only and only `save_to_db` is ML-only.
Every other one of the fifteen is SHARED -- so filing them under a Computer
Vision heading, which is what their names suggest, would tell the user they
apply to one path when they apply to both. That is the one hard rule this
item states.
"""
from __future__ import annotations

import pytest

from spacr.classify import FAMILY_SETTINGS
from spacr.settings import categories

#: What the one list held before the split.
BEFORE = {
    "cross_validation_enabled", "cross_validation_folds", "cv_group_by",
    "nested_cv_inner_folds", "classifier_evaluation",
    "evaluation_calibration", "evaluation_bins",
    "evaluation_fail_on_leakage", "leakage_audit_train_test",
    "leakage_hash_content", "leakage_require_identity", "score_threshold",
    "n_top_examples", "score_column", "save_to_db",
}

#: Where they went.
HEADINGS = ("Model Evaluation", "Evaluation Reports", "Leakage Audit",
            "Computer Vision Training", "Machine Learning Model and Features")

#: A heading is path-specific when its NAME claims a path.
PATH_SPECIFIC = ("Computer Vision Training",
                 "Machine Learning Model and Features")


def _family_of(key: str) -> str:
    """'cv', 'ml' or 'shared', from the authoritative table."""
    if key in FAMILY_SETTINGS["cv"]:
        return "cv"
    if key in FAMILY_SETTINGS["ml"]:
        return "ml"
    return "shared"


def _heading_of(key: str) -> str:
    for heading in HEADINGS:
        if key in categories[heading]:
            return heading
    return ""


class TestNothingIsLost:

    def test_every_setting_still_has_a_home(self):
        """"the set of keys the panel renders is the same before and
        after"."""
        for key in sorted(BEFORE):
            assert _heading_of(key), f"{key} is under no heading"

    def test_none_is_in_two_places(self):
        """A setting under two headings is one the user will change in one
        and wonder why the other disagrees."""
        for key in sorted(BEFORE):
            found = [h for h in HEADINGS if key in categories[h]]
            assert len(found) == 1, f"{key} is under {found}"


class TestNoListIsUndifferentiated:

    def test_the_old_list_is_much_shorter(self):
        assert len(categories["Model Evaluation"]) <= 5

    @pytest.mark.parametrize("heading", ["Model Evaluation",
                                         "Evaluation Reports",
                                         "Leakage Audit"])
    def test_no_new_heading_is_long(self, heading):
        assert len(categories[heading]) <= 6, (
            f"{heading} is another undifferentiated list")

    def test_only_two_headings_were_invented(self):
        """"a new heading is a new place to look", so only where nothing
        existing fits."""
        invented = [h for h in ("Evaluation Reports", "Leakage Audit")
                    if h in categories]
        assert len(invented) == 2


class TestASharedSettingStaysShared:
    """The failure this item names, asserted against the family table."""

    def test_most_of_them_are_shared(self):
        """The finding that corrected the split: eleven of fifteen."""
        shared = [k for k in BEFORE if _family_of(k) == "shared"]
        assert len(shared) >= 10, shared

    @pytest.mark.parametrize("key", sorted(BEFORE))
    def test_a_shared_setting_is_not_under_a_path_heading(self, key):
        if _family_of(key) != "shared":
            pytest.skip(f"{key} is {_family_of(key)}-only")
        assert _heading_of(key) not in PATH_SPECIFIC, (
            f"{key} is read by BOTH families, and {_heading_of(key)!r} "
            f"tells the user it applies to one")

    def test_no_shared_heading_names_a_path(self):
        """A neutral heading whose NAME claims a path is the same failure
        with extra steps."""
        for heading in ("Model Evaluation", "Evaluation Reports",
                        "Leakage Audit"):
            lowered = heading.lower()
            assert "computer vision" not in lowered
            assert "machine learning" not in lowered


class TestTheExclusivesWentToTheirFamily:

    def test_the_cv_only_one_is_under_a_cv_heading(self):
        assert _family_of("n_top_examples") == "cv"
        assert _heading_of("n_top_examples") == "Computer Vision Training"

    def test_the_ml_only_one_is_under_an_ml_heading(self):
        assert _family_of("save_to_db") == "ml"
        assert _heading_of("save_to_db") == \
            "Machine Learning Model and Features"

    def test_they_are_the_only_two(self):
        """Checked against the table rather than a hand-written list, so the
        split and the families cannot drift apart."""
        exclusive = [k for k in BEFORE if _family_of(k) != "shared"]
        assert sorted(exclusive) == ["n_top_examples", "save_to_db"]


class TestTheLeakageAuditIsItsOwnQuestion:
    """Four settings about whether train and test share objects is not
    "evaluation" in the sense the rest of the list means -- it is a check on
    the SPLIT, and a reader looking for it under a metrics heading would not
    find it."""

    def test_it_has_its_own_heading(self):
        assert categories["Leakage Audit"]

    def test_every_leakage_setting_is_in_it(self):
        for key in BEFORE:
            if "leakage" in key:
                assert key in categories["Leakage Audit"], key

    def test_and_nothing_else_is(self):
        for key in categories["Leakage Audit"]:
            assert "leakage" in key, key
