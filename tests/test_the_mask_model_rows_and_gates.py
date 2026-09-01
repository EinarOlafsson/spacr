"""Model rows carry the zoo button, and two controls grey out when dead.

Asked for 2026-09-01: the same zoo button on cell/nucleus/organelle model
names as on the pathogen one; the retired pre-SAM spellings gone from the LIVE
view; organelle_model_name live only under cellpose; custom_regex live only
where it is read.
"""
from __future__ import annotations

import pytest

from spacr import settings as S


def test_custom_regex_is_live_for_custom_AND_auto():
    """'auto' is included, and it is the non-obvious half.

    metadata_type's own description says 'auto' renames using custom_regex
    "when supplied, otherwise automatic detection". Greying the field there
    would remove a documented behaviour while looking like a tidy-up -- the
    request said "unless metadata type is custom", and taking that literally
    would have been the bug.
    """
    rules = S.get_setting_dependencies()
    predicate = rules["custom_regex"]["predicate"]

    assert predicate({"metadata_type": "custom"}, None) is True
    assert predicate({"metadata_type": "auto"}, None) is True
    assert predicate({"metadata_type": "cellvoyager"}, None) is False
    assert predicate({"metadata_type": "cq1"}, None) is False


def test_the_greyed_reason_names_the_value_that_caused_it():
    """A greyed control with no reason is a control that looks broken."""
    rules = S.get_setting_dependencies()
    reason = rules["custom_regex"]["reason"]({"metadata_type": "cq1"}, None)
    assert "cq1" in reason
    assert "kept and saved" in reason, (
        "a user must be told their value survives being greyed")


def test_organelle_model_name_is_live_only_under_cellpose():
    """Every other organelle_method is a threshold or a filter and loads no
    checkpoint, so the field would change nothing."""
    rules = S.get_setting_dependencies()
    predicate = rules["organelle_model_name"]["predicate"]

    assert predicate({"organelle_method": "cellpose"}, None) is True
    for method in ("otsu", "adaptive", "log", "dog", "ridge", "hysteresis"):
        assert predicate({"organelle_method": method}, None) is False, method


def test_the_live_menu_drops_the_pre_sam_spellings():
    """All four resolve to cpsam, so offering them is four labels for one
    model. The settings menu keeps them -- a saved file naming cyto2 must
    still show the user their own value -- and the live view does not."""
    saved = S.cellpose_model_menu()
    live = S.cellpose_live_model_menu()

    assert "cpsam" in live
    for legacy in ("cyto", "cyto2", "cyto3", "nuclei"):
        if legacy in saved:
            assert legacy not in live, f"{legacy} is still offered live"


def test_a_downloaded_zoo_model_survives_into_the_live_menu(monkeypatch,
                                                            tmp_path):
    """Those ARE different models, unlike the aliases, so they stay."""
    checkpoint = tmp_path / "cpsam_v2_toxo_r2"
    checkpoint.write_bytes(b"weights")
    monkeypatch.setattr(S, "downloaded_zoo_models",
                        lambda: (str(checkpoint),))
    assert str(checkpoint) in S.cellpose_live_model_menu()


def test_every_object_model_row_gets_the_zoo_button():
    """Cell, nucleus, pathogen and organelle all take a Cellpose checkpoint,
    so all four offer the same way of finding one."""
    pytest.importorskip("PySide6")
    from spacr.qt.screens.app_screen import AppScreen

    for key in ("cell_model_name", "nucleus_model_name",
                "pathogen_model_name", "pathogen_model",
                "organelle_model_name"):
        assert key in AppScreen._MODEL_ZOO_KEYS, key


def test_every_organelle_gets_the_zoo_button_not_only_the_first():
    """Reported 2026-09-01: the button appeared on organelle 1 and no other.

    With more than one organelle the settings are GENERATED per organelle --
    organelleb_model_name, organellec_model_name, ... -- so a fixed tuple of
    literal names covers the first and silently withholds the button from
    every organelle after it. That is the failure a literal list always
    eventually has when the names are generated.
    """
    pytest.importorskip("PySide6")
    from spacr.qt.screens.app_screen import AppScreen

    assert AppScreen._takes_a_cellpose_checkpoint("organelle_model_name")
    for suffix in "bcdefgh":
        key = f"organelle{suffix}_model_name"
        assert AppScreen._takes_a_cellpose_checkpoint(key), key


def test_a_classifier_field_does_not_get_the_cellpose_button():
    """custom_model_path in Classify holds a torch classifier, so offering
    cpsam checkpoints there offers something that screen cannot load."""
    pytest.importorskip("PySide6")
    from spacr.qt.screens.app_screen import AppScreen

    assert not AppScreen._takes_a_cellpose_checkpoint("custom_model_path")


def test_pathogen_model_is_no_longer_offered_but_is_still_read():
    """One control for one value.

    `pathogen_model` and `pathogen_model_name` named the same thing, and two
    controls for one value is how a user sets one and wonders why the other
    wins. The old one is retired from the panel and still READ, so a settings
    CSV written before this keeps segmenting with the model it names rather
    than silently falling back to cpsam.
    """
    pytest.importorskip("PySide6")
    from spacr.qt.screens.settings_model import _APP_HIDDEN_KEYS

    assert "pathogen_model" in _APP_HIDDEN_KEYS.get("mask", set())

    import inspect

    from spacr import object as spacr_object
    source = inspect.getsource(spacr_object.generate_cellpose_masks_sam)
    assert "settings['pathogen_model']" in source, (
        "the legacy value must still be honoured for old settings files")
