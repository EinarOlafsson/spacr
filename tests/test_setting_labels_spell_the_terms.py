"""A settings label spells the terms of art the way the tool does.

Asked 2026-08-21: "whenever you write grna write gRNA".

`capitalize()` lower-cases everything after the first character, so `grna`
rendered as `Grna` and `exclude_grnas` as `Exclude grnas` -- the spelling the
tool uses nowhere else in its own documentation.

FIXED IN THE ONE FUNCTION THAT BUILDS EVERY LABEL, not in the four settings
that happen to contain the word today. The fifth would have arrived spelled
wrong.
"""
from __future__ import annotations

import pytest

from spacr.object_roles import setting_label


class TestTheTermsKeepTheirCase:

    @pytest.mark.parametrize("key,expected", [
        ("grna", "gRNA"),
        ("grna_csv", "gRNA CSV"),
        ("exclude_grnas", "Exclude gRNAs"),
        ("count_grna_column", "Count gRNA column"),
    ])
    def test_grna_is_spelled_grna(self, key, expected):
        assert setting_label(key) == expected

    @pytest.mark.parametrize("key,expected", [
        ("png_size", "PNG size"),
        ("umap_canvas_width", "UMAP canvas width"),
    ])
    def test_the_other_acronyms_too(self, key, expected):
        assert setting_label(key) == expected

    def test_every_setting_with_grna_in_it_is_covered(self):
        """The general statement: no label may render the term lower-case,
        whichever settings exist."""
        import spacr.settings as settings

        for key in settings.expected_types:
            if "grna" not in str(key).lower():
                continue
            label = setting_label(key)
            assert "grna" not in label, f"{key} -> {label}"
            assert "Grna" not in label, f"{key} -> {label}"


class TestTheIdentifierKeys:
    """The five columns instruction 213 standardises are camelCase, so the
    underscore split left them whole and `capitalize()` rendered `Plateid`."""

    @pytest.mark.parametrize("key,expected", [
        ("plateID", "Plate ID"),
        ("rowID", "Row ID"),
        ("columnID", "Column ID"),
        ("fieldID", "Field ID"),
        ("objectID", "Object ID"),
    ])
    def test_they_read_as_words(self, key, expected):
        assert setting_label(key) == expected

    def test_a_word_merely_ending_in_id_is_left_alone(self):
        """`identity` must not become `ident ID y`."""
        assert setting_label("identity") == "Identity"
        assert setting_label("valid") == "Valid"


class TestOrdinaryLabelsAreUnchanged:
    """The fix must not have become a general rewriter."""

    @pytest.mark.parametrize("key,expected", [
        ("cell_diameter", "Cell diameter"),
        ("exclude_rows", "Exclude rows"),
        ("batch_correction", "Batch correction"),
        ("", ""),
    ])
    def test_they_are_what_they_were(self, key, expected):
        assert setting_label(key) == expected

    def test_no_label_in_the_whole_settings_set_becomes_empty(self):
        import spacr.settings as settings

        for key in settings.expected_types:
            assert setting_label(key), key


class TestItSitsUnderTheNegativeControl:
    """"exclude grnas should be under negative controll in controls and
    filters"."""

    def test_it_follows_negative_control_in_the_category(self):
        import spacr.settings as settings

        keys = settings.categories["Plate Layout & Controls"]
        assert keys[keys.index("negative_control") + 1] == "exclude_grnas"
