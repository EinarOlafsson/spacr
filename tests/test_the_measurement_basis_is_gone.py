"""`dataset_mode='measurement'` is removed, and old settings still load.

Instruction 229: "remove the measurement option from dataset mode and remove
the metadata type by and measurement rules".

REMOVED, NOT HIDDEN -- a setting behind a gate is still a setting somebody
will find. But a removal that turns every settings CSV naming it into a file
that RAISES is not a removal, it is a break, so the retired value is
migrated and the reason is recorded beside the mapping.
"""
from __future__ import annotations

import pytest


class TestTheOptionIsGone:

    def test_it_is_not_in_the_bases(self):
        from spacr.training_basis import TRAINING_BASES

        assert "measurement" not in TRAINING_BASES
        assert TRAINING_BASES == ("metadata", "annotation")

    def test_the_spec_does_not_offer_it(self):
        import spacr.settings_spec as spec

        source = open(spec.__file__).read()
        after = source.split("'dataset_mode':")[1][:120]
        assert "measurement" not in after, after

    def test_the_tk_gui_does_not_offer_it_either(self):
        """Tk accommodates new code: it is updated to fit, not left broken."""
        import spacr.gui_elements as tk

        source = open(tk.__file__).read()
        assert "values=['annotation','metadata','measurement']" not in source

    def test_the_rules_setting_is_gone(self):
        from spacr.settings import descriptions, tooltips

        for table in (descriptions, tooltips):
            assert "measurement_rules" not in table

    def test_the_metadata_type_by_setting_is_gone(self):
        from spacr.settings import descriptions, tooltips

        for table in (descriptions, tooltips):
            assert "metadata_type_by" not in table

    def test_the_code_path_is_gone(self):
        import spacr.ml as ml

        assert not hasattr(ml, "_labels_from_measurements")


class TestAnOldSettingsFileStillLoads:

    def test_the_retired_basis_is_migrated_not_refused(self):
        from spacr.training_basis import resolve_basis

        assert resolve_basis({"dataset_mode": "measurement"}) == "annotation"

    def test_and_it_is_recorded_why(self):
        from spacr.training_basis import RETIRED_BASES

        assert RETIRED_BASES["measurement"] == "annotation"

    def test_normalize_pins_the_migrated_value(self):
        from spacr.training_basis import normalize_settings

        out = normalize_settings({"dataset_mode": "measurement"})
        assert out["dataset_mode"] == "annotation"

    def test_a_basis_that_never_existed_still_raises(self):
        """Migration is for what WAS valid, not a blanket fallback."""
        from spacr.training_basis import TrainingBasisError, resolve_basis

        with pytest.raises(TrainingBasisError):
            resolve_basis({"dataset_mode": "nonsense"})


class TestTheColumnComesFromTheClasses:
    """`metadata_type_by` named the column a class is defined by, which is
    the Classes editor's own column field."""

    def test_it_is_read_off_classes(self):
        from spacr.io import _class_column

        assert _class_column(
            {"classes": {"pc": {"column": "rowID", "value": "r2"}}}) == "rowID"

    def test_an_old_key_still_wins(self):
        """A CSV written before the removal runs unchanged."""
        from spacr.io import _class_column

        assert _class_column({"metadata_type_by": "condition"}) == "condition"
        assert _class_column({
            "metadata_type_by": "condition",
            "classes": {"pc": {"column": "rowID", "value": "r2"}},
        }) == "condition"

    def test_the_default_is_what_it_always_was(self):
        from spacr.io import _class_column

        assert _class_column({}) == "columnID"


class TestTheSampleRename:

    def test_the_label_says_what_it_bounds(self):
        from spacr.object_roles import setting_label

        assert setting_label("sample") == "Sample size limit"

    def test_the_key_is_untouched(self):
        """Every settings CSV in existence uses `sample`."""
        from spacr.settings import tooltips

        assert "sample" in tooltips
