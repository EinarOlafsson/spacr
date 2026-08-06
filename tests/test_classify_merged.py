"""The merged Classify module, and the two it does NOT replace.

Classify (CV) trains a Torch model on object crops; Classify (ML) fits a
gradient-boosted model on measured features. They answer the same question
and shared six setting names out of 78 and 37, two of which disagreed on
their default.

The merged module is a front end over both: one settings dict, one
``classifier_family`` switch, one call. Both originals stay, deliberately --
a merged screen that replaced them would strand every saved settings CSV and
every notebook importing their entry points.

The load-bearing property is that the merged module REIMPLEMENTS NOTHING. It
calls ``deep_spacr`` or ``generate_ml_scores`` unchanged, so a run through it
and a run through the module it dispatches to are the same run rather than
two implementations that have to be kept in step.
"""
from __future__ import annotations

import pytest

from spacr import classify


# ---------------------------------------------------------------------------
# The family switch
# ---------------------------------------------------------------------------

def test_the_default_family_is_cv():
    """The merged module's own defaults are the CV ones, so a dict with no
    family is most likely a Classify (CV) CSV opened in the merged screen."""
    assert classify.resolve_family({}) == "cv"
    assert classify.resolve_family({"classifier_family": None}) == "cv"
    assert classify.resolve_family({"classifier_family": ""}) == "cv"


@pytest.mark.parametrize("family", classify.CLASSIFIER_FAMILIES)
def test_every_family_resolves_to_itself(family):
    assert classify.resolve_family({"classifier_family": family}) == family
    assert classify.resolve_family(
        {"classifier_family": family.upper()}) == family


def test_an_unknown_family_is_refused_not_guessed():
    """Guessing would train a different kind of model than was asked for and
    report success."""
    with pytest.raises(classify.ClassifierFamilyError, match="not one of"):
        classify.resolve_family({"classifier_family": "torch"})


def test_the_families_do_not_claim_each_others_settings():
    cv = set(classify.FAMILY_SETTINGS["cv"])
    ml = set(classify.FAMILY_SETTINGS["ml"])
    assert cv.isdisjoint(ml), sorted(cv & ml)


@pytest.mark.parametrize("family", classify.CLASSIFIER_FAMILIES)
def test_inapplicable_is_exactly_the_other_family(family):
    mine = set(classify.FAMILY_SETTINGS[family])
    theirs = set(classify.inapplicable_settings(family))
    assert mine.isdisjoint(theirs)
    everything = set().union(*classify.FAMILY_SETTINGS.values())
    assert mine | theirs == everything


# ---------------------------------------------------------------------------
# Dispatch -- it must call the real pipelines, not copies of them
# ---------------------------------------------------------------------------

def test_cv_dispatches_to_deep_spacr(monkeypatch):
    seen = {}
    import spacr.deep_spacr as ds
    def _fake(settings):
        seen["settings"] = settings
        return "cv-ran"
    monkeypatch.setattr(ds, "deep_spacr", _fake)
    assert classify.classify({"classifier_family": "cv"}) == "cv-ran"
    assert seen["settings"]["dataset_mode"] == "metadata"


def test_ml_dispatches_to_generate_ml_scores(monkeypatch):
    seen = {}
    import spacr.ml as ml
    def _fake(settings):
        seen["settings"] = settings
        return "ml-ran"
    monkeypatch.setattr(ml, "generate_ml_scores", _fake)
    assert classify.classify({"classifier_family": "ml"}) == "ml-ran"
    assert seen["settings"]["dataset_mode"] == "metadata"


def test_the_ml_pipeline_still_gets_the_names_it_reads(monkeypatch):
    """normalize_settings renames model_type_ml -> model_type for the shared
    vocabulary. generate_ml_scores reads the OLD name, so the merged entry
    point hands it back rather than editing a working pipeline to suit a
    settings panel."""
    seen = {}
    import spacr.ml as ml
    monkeypatch.setattr(ml, "generate_ml_scores",
                        lambda s: seen.setdefault("settings", s))
    classify.classify({
        "classifier_family": "ml", "model_type_ml": "xgboost",
        "test_size": 0.25, "cross_validation": False,
    })
    s = seen["settings"]
    assert s["model_type_ml"] == "xgboost"
    assert s["test_size"] == 0.25
    assert s["cross_validation"] is False
    # ...and the shared names are present too, so either can be read.
    assert s["model_type"] == "xgboost"
    assert s["test_split"] == 0.25


def test_the_training_basis_survives_dispatch(monkeypatch):
    seen = {}
    import spacr.deep_spacr as ds
    monkeypatch.setattr(ds, "deep_spacr", lambda s: seen.setdefault("s", s))
    classify.classify({"annotation_column": "test"})
    # The implicit old rule still resolves, through the merged entry point.
    assert seen["s"]["dataset_mode"] == "annotation"


# ---------------------------------------------------------------------------
# Registration -- five seams, each a place a module ends up unreachable
# ---------------------------------------------------------------------------

def test_the_merged_module_is_registered_everywhere_it_has_to_be():
    """Four features have spent weeks built, tested and unreachable because
    one of these was missed."""
    from spacr.qt.app import APPS
    from spacr.qt import iconset
    from spacr.qt.screens import app_screen

    keys = {row[0] for row in APPS}
    assert {"classify_merged", "classify", "ml_analyze"} <= keys, (
        "the merged module must be registered AND both originals kept")

    row = next(r for r in APPS if r[0] == "classify_merged")
    assert row[1] == "Classify"
    assert row[3], "no Home section, so it falls into the fallback band"
    glyph = iconset._NAME_TO_GLYPH.get("classify_merged")
    assert glyph, "no icon"
    # Not the same glyph as another module: two identical icons in the
    # sidebar is a worse affordance than a missing one.
    others = [k for k, v in iconset._NAME_TO_GLYPH.items()
              if v == glyph and k != "classify_merged"]
    assert not others, f"icon collides with {others}"
    assert app_screen.APP_TITLES.get("classify_merged") == "Classify"
    assert app_screen.APP_INTROS.get("classify_merged")


def test_the_merged_defaults_are_the_union_of_both():
    from spacr.settings import (
        deep_spacr_defaults, set_default_analyze_screen, set_default_classify,
    )

    merged = set_default_classify(settings={})
    cv = set(deep_spacr_defaults(settings={}))
    ml = set(set_default_analyze_screen(settings={}))
    assert cv <= set(merged), sorted(cv - set(merged))
    assert ml <= set(merged), sorted(ml - set(merged))
    assert merged["classifier_family"] == "cv"


def test_the_retired_keys_are_gone_from_every_classify_module():
    """annotated_classes and custom_measurement were collected by the form
    and read by nothing. Carrying a dead key into a new module is how it
    survives another five years."""
    from spacr.settings import (
        deep_spacr_defaults, set_default_analyze_screen, set_default_classify,
    )

    for factory in (deep_spacr_defaults, set_default_analyze_screen,
                    set_default_classify):
        keys = set(factory(settings={}))
        assert "annotated_classes" not in keys, factory.__name__
        assert "custom_measurement" not in keys, factory.__name__
