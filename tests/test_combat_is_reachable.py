"""ComBat is reachable from the GUI, and still refuses when it should.

``spacr.batch_correction`` has had a working empirical-Bayes ComBat for a
while. Nothing could select it: ``combat`` was absent from every
``batch_correction`` dropdown, and the two settings it needs —
``batch_covariate_column`` and ``batch_combat_mean_only`` — did not exist
as settings at all, so even a hand-edited CSV could not supply them.

The refusal is the interesting half. ComBat estimates the batch effect
from the residuals after the declared covariates, so anything *not*
declared is treated as noise and removed along with the plate effect. A
ComBat run with no covariate does not fail; it quietly deletes the
contrast the screen was measuring. ``batch_covariate_column`` left blank
therefore raises rather than running, and these tests hold that line as
firmly as they hold the wiring.

One thing deliberately NOT done: ``correction_kwargs`` did not grow the
two keys. Its output is ``**``-splatted into several different
signatures, and a key that only some of them accept is a ``TypeError`` in
the others.
"""
from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# It can be selected
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("app_key", ["umap", "ml_analyze", "regression"])
def test_combat_is_offered_by_every_module_that_corrects_batches(app_key):
    pytest.importorskip("PySide6")
    from spacr.qt.screens.settings_model import _APP_COMBO_OPTIONS

    options = _APP_COMBO_OPTIONS[app_key]["batch_correction"]
    assert "combat" in options, (
        f"{app_key} does not offer combat, so the implementation in "
        "spacr.batch_correction cannot be reached from the GUI")
    assert options[0] == "none", "the default must stay 'do nothing'"


@pytest.mark.parametrize("app_key", ["umap", "ml_analyze", "regression"])
def test_combats_two_settings_are_on_the_page_beside_it(app_key):
    pytest.importorskip("PySide6")
    from spacr.qt.screens.settings_model import categories_for_app
    from spacr.settings import categories

    group = categories_for_app(app_key, categories)["Plate & Batch Correction"]
    for key in ("batch_covariate_column", "batch_combat_mean_only"):
        assert key in group, (
            f"{key} is not in {app_key}'s batch-correction group, so combat "
            "can be selected but not configured")


@pytest.mark.parametrize("app_key", ["umap", "ml_analyze", "regression"])
def test_the_two_settings_have_defaults(app_key):
    pytest.importorskip("PySide6")
    from spacr.qt.screens.settings_model import resolve_default_settings

    defaults = resolve_default_settings(app_key)
    assert defaults["batch_covariate_column"] is None, (
        "blank by default: guessing a covariate is how combat deletes the "
        "contrast it was supposed to preserve")
    assert defaults["batch_combat_mean_only"] is False


def test_both_settings_carry_a_real_tooltip():
    from spacr.settings import tooltips

    for key in ("batch_covariate_column", "batch_combat_mean_only"):
        text = tooltips[key]
        assert text.startswith("("), f"{key} does not declare its type"
        assert len(text.split()) >= 25, f"{key}'s tooltip is a stub"
        assert "combat" in text.lower()
        assert "spacr.batch_correction.correct_batch_effects" in text


def test_the_method_tooltip_says_combat_needs_the_covariate():
    """The dropdown's own help has to warn before the run, not after."""
    from spacr.settings import tooltips

    text = tooltips["batch_correction"]
    assert "combat" in text
    assert "batch_covariate_column" in text


# ---------------------------------------------------------------------------
# The kwargs reach the implementation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dotted", [
    "spacr.utils.preprocess_data",
    "spacr.ml.ml_analysis",
])
def test_the_pipeline_entry_points_accept_the_two_kwargs(dotted):
    import importlib

    module_name, _, func_name = dotted.rpartition(".")
    func = getattr(importlib.import_module(module_name), func_name)
    params = inspect.signature(func).parameters
    for key in ("batch_covariate_column", "batch_combat_mean_only"):
        assert key in params, f"{dotted} does not accept {key}"


def test_correction_kwargs_did_not_grow_the_two_keys():
    """It is splatted into signatures that never grew them."""
    from spacr.batch_correction import correction_kwargs

    produced = correction_kwargs({"batch_correction": "combat"})
    for key in ("batch_covariate_column", "batch_combat_mean_only"):
        assert key not in produced, (
            f"correction_kwargs now emits {key}; every caller that splats it "
            "into a signature without that parameter is a TypeError")


def test_preprocess_data_forwards_the_covariate(monkeypatch):
    """Named, not positional, and not dropped on the way through."""
    from spacr import utils

    seen = {}

    def fake(features, metadata, **kwargs):
        seen.update(kwargs)
        report = type("R", (), {
            "method": "combat", "batches": ["p1"],
            "centroid_spread_before": 1.0, "centroid_spread_after": 0.1,
            "warnings": [],
        })()
        return features, report

    import spacr.batch_correction as bc
    monkeypatch.setattr(bc, "correct_from_metadata", fake)

    frame = pd.DataFrame({
        "plateID": ["p1"] * 6,
        "condition": ["a", "b"] * 3,
        "feature_one": np.linspace(0.0, 1.0, 6),
        "feature_two": np.linspace(1.0, 2.0, 6),
    })
    utils.preprocess_data(
        frame, None, False, False, None,
        batch_correction="combat",
        batch_covariate_column="condition",
        batch_combat_mean_only=True,
    )
    assert seen["batch_correction"] == "combat"
    assert seen["batch_covariate_column"] == "condition"
    assert seen["batch_combat_mean_only"] is True


# ---------------------------------------------------------------------------
# The refusal still works
# ---------------------------------------------------------------------------

def _frame(n=12):
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "f1": rng.normal(size=n),
        "f2": rng.normal(size=n),
    })


def _batches(n=12):
    return pd.Series(["p1"] * (n // 2) + ["p2"] * (n // 2))


def test_combat_with_no_covariate_refuses_instead_of_running():
    """The behaviour the wiring must not have loosened."""
    from spacr.batch_correction import correct_batch_effects

    with pytest.raises(ValueError) as raised:
        correct_batch_effects(_frame(), _batches(), method="combat")
    assert "covariate" in str(raised.value).lower()


def test_a_blank_covariate_setting_still_refuses():
    """Blank in the GUI is 'unanswered', not 'there is none'."""
    from spacr.batch_correction import correct_from_metadata

    features = _frame()
    metadata = pd.DataFrame({"plateID": _batches(),
                             "condition": ["a", "b"] * 6})
    for blank in (None, "", "   "):
        with pytest.raises(ValueError):
            correct_from_metadata(features, metadata,
                                  batch_correction="combat",
                                  batch_covariate_column=blank)


def test_combat_runs_once_the_covariate_is_declared():
    from spacr.batch_correction import correct_from_metadata

    features = _frame()
    metadata = pd.DataFrame({"plateID": _batches(),
                             "condition": ["a", "b"] * 6})
    corrected, report = correct_from_metadata(
        features, metadata,
        batch_correction="combat",
        batch_covariate_column="condition",
    )
    assert report.method == "combat"
    assert corrected.shape == features.shape
    assert not corrected.isna().any().any()


def test_mean_only_reaches_the_implementation():
    """Not merely accepted — it has to change the answer.

    ``combat_mean_only`` leaves each batch's scale alone. If the flag were
    swallowed on the way through, both runs would return the same numbers
    and the setting would be decoration.
    """
    from spacr.batch_correction import correct_from_metadata

    rng = np.random.default_rng(1)
    n = 24
    condition = ["a", "b"] * (n // 2)
    plate = ["p1"] * (n // 2) + ["p2"] * (n // 2)
    # Second plate: shifted AND stretched, so mean-only and full ComBat
    # cannot agree.
    scale = np.where(np.array(plate) == "p2", 4.0, 1.0)
    shift = np.where(np.array(plate) == "p2", 5.0, 0.0)
    features = pd.DataFrame({
        "f1": rng.normal(size=n) * scale + shift,
        "f2": rng.normal(size=n) * scale + shift,
    })
    metadata = pd.DataFrame({"plateID": plate, "condition": condition})

    both, _ = correct_from_metadata(
        features, metadata, batch_correction="combat",
        batch_covariate_column="condition", batch_combat_mean_only=False)
    mean_only, _ = correct_from_metadata(
        features, metadata, batch_correction="combat",
        batch_covariate_column="condition", batch_combat_mean_only=True)

    assert not np.allclose(both.to_numpy(), mean_only.to_numpy()), (
        "batch_combat_mean_only did not change the correction, so it is "
        "being swallowed somewhere between the GUI and _combat")
