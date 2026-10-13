"""Classify's Essentials view is enough to pick and train a model.

Asked for on 2026-09-28: "for classify the essentialls category only includes
the classifier category. this is not enough. essentialls should include
enough to pick and train a computer vision mode (resnet or Maxvit, etc) or a
tabular based algorithm".

The Essentials view has to carry both families' training settings, and the
family switch greys the half that does not apply.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from spacr.qt.screens.settings_model import essential_keys  # noqa: E402
from spacr.qt.settings_search import (  # noqa: E402
    ESSENTIALS,
    forget_disclosure,
    install,
)

SHARED = ("classifier_family", "src", "dataset_mode", "classes",
          "test_split", "learning_rate")
#: Shown or hidden by ``dataset_mode``, the class definition's own switch.
BY_THE_MODE = ("metadata_item_1_name", "metadata_item_1_value")
IMAGE = ("model_type", "epochs", "batch_size", "val_split",
         "train_channels", "image_size", "mixed_precision",
         "generate_training_dataset", "train")
TABULAR = ("model_type_ml", "n_estimators", "reg_alpha", "reg_lambda")


@pytest.fixture
def classify_screen(qtbot):
    forget_disclosure()
    from spacr.qt.screens.app_screen import AppScreen
    screen = AppScreen("classify_merged")
    qtbot.addWidget(screen)
    yield screen
    forget_disclosure()


def test_essentials_name_both_families_training_settings():
    keys = set(essential_keys("classify_merged"))
    missing = [k for k in (*SHARED, *BY_THE_MODE, *IMAGE, *TABULAR)
               if k not in keys]
    assert not missing, missing


def test_essentials_leave_the_advanced_settings_in_their_categories():
    keys = set(essential_keys("classify_merged"))
    for advanced in ("focal_gamma", "tta_enabled", "batch_correction",
                     "leakage_hash_content", "prune_features"):
        assert advanced not in keys


def test_the_essentials_view_shows_them(classify_screen):
    bar = install(classify_screen)
    assert bar.level() == ESSENTIALS
    shown = set(bar.visible_keys())
    missing = [k for k in (*SHARED, *IMAGE, *TABULAR) if k not in shown]
    assert not missing, missing
    assert len(shown) < len(bar.indexed_keys())


@pytest.mark.parametrize("family, live, greyed", [
    ("cv", IMAGE, TABULAR),
    ("ml", TABULAR, IMAGE),
])
def test_only_the_chosen_family_is_live(classify_screen, family, live,
                                        greyed):
    install(classify_screen)
    panel = classify_screen._settings_model
    control = panel._widgets.get("classifier_family")
    control.setCurrentIndex(control.findData(family))
    panel._on_classifier_family_changed()
    built = dict(panel._built_controls())
    for key in live:
        if key in built:
            assert built[key].isEnabled(), (family, key)
    for key in greyed:
        if key in built:
            assert not built[key].isEnabled(), (family, key)
    for key in SHARED[1:]:
        if key in built:
            assert built[key].isEnabled(), (family, key)
    assert set(live) & set(built) and set(greyed) & set(built)
